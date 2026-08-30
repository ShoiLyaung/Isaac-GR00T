#!/usr/bin/env python3
"""Bounded offline supervisor for causal C5 Codec-only training.

This successor preserves the historical C5 optimization schedule while binding
training to one explicitly supplied causal fixture manifest.  It never resumes
the historical C5 optimizer or rewrites any preparation/training result.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _load_sibling(name: str, filename: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


history = _load_sibling("c5_historical_training_supervisor", "launch_c5_codec_only.py")
preparation = _load_sibling("c5_causal_preparation", "prepare_c5_causal_codec_only.py")
base = history.base

TRAINING_PLAN_SCHEMA_VERSION = "g1-all-lerobot-c5-causal-codec-only-training-plan-v1"
TRAINING_SUMMARY_SCHEMA_VERSION = "g1-all-lerobot-c5-causal-codec-only-training-summary-v1"
TRAINING_PHASE = "F2_CAUSAL_CODEC_EVIDENCE"
RETAINED_CHECKPOINT = history.RETAINED_CHECKPOINT
CHECKPOINT_HASHES = dict(history.CHECKPOINT_HASHES)
EXPECTED_PARAMETER_NAMES = history.EXPECTED_PARAMETER_NAMES
EXPECTED_PARAMETER_NAMES_SHA256 = history.EXPECTED_PARAMETER_NAMES_SHA256
EXPECTED_TRAINABLE_NUMEL = history.EXPECTED_TRAINABLE_NUMEL
EXPECTED_NATIVE_TENSOR_COUNT = 1_031
CODEC_PARAMETER_PREFIX = "action_head.codec_conditioning."
EXPECTED_INITIAL_CODEC_STATE_SHA256 = (
    "fbbf68b8ff2bec8bd037f62ffa41cab3ca2825f872bcd09745febe96a703e674"
)
FIXTURE_ROOT = preparation.FIXTURE_ROOT
SOURCE_IDS = history.SOURCE_IDS
NATIVE_PREPARATION = history.NATIVE_PREPARATION
NATIVE_MANIFEST_SHA256 = history.NATIVE_MANIFEST_SHA256
NATIVE_CLOSURE_SHA256 = history.NATIVE_CLOSURE_SHA256
NATIVE_ADAPTER_SHA256 = dict(history.NATIVE_ADAPTER_SHA256)
MODES = history.MODES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _model_codec_contract() -> dict[str, Any]:
    return history._model_codec_contract()


def _adapter_codec_contract(fixture_manifest_sha256: str) -> dict[str, Any]:
    return {
        **_model_codec_contract(),
        "fixture_root": str(FIXTURE_ROOT),
        "fixture_manifest_sha256": _require_sha256(
            fixture_manifest_sha256, "fixture_manifest_sha256"
        ),
    }


def validate_retained_native_index(checkpoint_root: Path) -> dict[str, Any]:
    """Prove before allocation that retained native20k contains no Codec tensors."""

    index = base.load_json(checkpoint_root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("retained native checkpoint index is unavailable")
    names = sorted(weight_map)
    codec_names = [name for name in names if name.startswith(CODEC_PARAMETER_PREFIX)]
    if len(names) != EXPECTED_NATIVE_TENSOR_COUNT or codec_names:
        raise RuntimeError(
            "retained native checkpoint must contain exactly 1031 native keys and zero Codec keys"
        )
    return {
        "native_tensor_count": len(names),
        "codec_tensor_count": 0,
        "parameter_names_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "complete": True,
    }


def validate_materials(preparation_root: Path, fixture_manifest_sha256: str) -> dict[str, Any]:
    """Close the causal fixture, native data, and retained-native identities."""

    manifest_sha256 = _require_sha256(fixture_manifest_sha256, "fixture_manifest_sha256")
    summary_path = preparation_root / "PREPARATION_SUMMARY.json"
    manifest_path = preparation_root / "DATASET_MANIFEST.json"
    summary = preparation.load_json(summary_path)
    manifest = preparation.load_json(manifest_path)
    if not (
        summary.get("schema_version") == preparation.PREPARATION_SUMMARY_SCHEMA_VERSION
        and summary.get("status") == "PASS"
        and summary.get("source_count") == 5
        and summary.get("total_episodes") == preparation.EXPECTED_EPISODES
        and summary.get("total_frames") == preparation.EXPECTED_TOTAL_FRAMES
        and summary.get("total_h40_anchors") == preparation.EXPECTED_H40_ANCHORS
        and summary.get("input_fixture_manifest_sha256") == manifest_sha256
    ):
        raise RuntimeError("causal C5 preparation summary is not the accepted exact closure")
    if summary.get("dataset_manifest_sha256") != preparation.sha256_file(manifest_path):
        raise RuntimeError("causal C5 preparation dataset manifest identity drift")
    plan = summary.get("plan")
    if not isinstance(plan, Mapping) or not (
        plan.get("schema_version") == preparation.PREPARATION_PLAN_SCHEMA_VERSION
        and plan.get("native_manifest_sha256") == NATIVE_MANIFEST_SHA256
        and plan.get("native_cpu_mixture_closure_sha256") == NATIVE_CLOSURE_SHA256
        and plan.get("native_adapter_sha256") == NATIVE_ADAPTER_SHA256
        and plan.get("fixture_root") == str(FIXTURE_ROOT)
        and plan.get("fixture_manifest_sha256") == manifest_sha256
        and plan.get("covered_episode_uid_digest") == preparation.EXPECTED_EPISODE_UID_DIGEST
    ):
        raise RuntimeError("causal C5 preparation provenance closure drift")

    datasets = manifest.get("datasets")
    if (
        manifest.get("schema_version") != "gr00t-finetune-dataset-manifest-v1"
        or not isinstance(datasets, list)
        or len(datasets) != len(SOURCE_IDS)
    ):
        raise RuntimeError("causal C5 dataset manifest must contain exactly five sources")

    source_ids = []
    mix_ratios = []
    adapter_hashes = summary.get("adapter_config_sha256")
    if not isinstance(adapter_hashes, Mapping):
        raise RuntimeError("causal C5 adapter hash closure is unavailable")
    expected_evidence = _adapter_codec_contract(manifest_sha256)
    for expected_source_id, entry in zip(SOURCE_IDS, datasets):
        dataset_path = Path(entry["dataset_path"]).resolve(strict=True)
        adapter_path = Path(entry["adapter_config_path"]).resolve(strict=True)
        adapter = preparation.load_json(adapter_path)
        source_id = adapter.get("source_id")
        source_ids.append(source_id)
        mix_ratios.append(float(entry["mix_ratio"]))
        if source_id != expected_source_id or adapter.get("codec_evidence") != expected_evidence:
            raise RuntimeError(f"causal C5 Codec adapter contract drift: {adapter_path}")
        if adapter_hashes.get(source_id) != preparation.sha256_file(adapter_path):
            raise RuntimeError(f"causal C5 adapter identity drift: {adapter_path}")
        native_adapter_path = (
            NATIVE_PREPARATION / "sources" / expected_source_id / "ADAPTER_CONFIG.json"
        )
        native_adapter = preparation.load_json(native_adapter_path)
        if preparation.sha256_file(native_adapter_path) != NATIVE_ADAPTER_SHA256[source_id]:
            raise RuntimeError(f"native adapter identity drift: {native_adapter_path}")
        if adapter != {**native_adapter, "codec_evidence": expected_evidence}:
            raise RuntimeError(
                f"derived causal C5 adapter is not the exact native overlay: {adapter_path}"
            )
        expected_dataset_path = (
            NATIVE_PREPARATION / "sources" / expected_source_id / "LOADER_VIEW"
        ).resolve(strict=True)
        if dataset_path != expected_dataset_path:
            raise RuntimeError(f"causal C5 loader-view path drift: {dataset_path}")
        try:
            adapter_path.relative_to(preparation_root)
        except ValueError as exc:
            raise RuntimeError(
                "derived causal C5 adapters must stay inside preparation root"
            ) from exc
        for name in ("data", "videos"):
            payload = dataset_path / name
            if not payload.is_symlink():
                raise RuntimeError(f"loader-view payload is not a read-only symlink: {payload}")
            expected_target = (Path(native_adapter["canonical_root"]) / name).resolve(strict=True)
            if payload.resolve(strict=True) != expected_target:
                raise RuntimeError(f"loader-view payload target drift: {payload}")
    if tuple(source_ids) != SOURCE_IDS or abs(sum(mix_ratios) - 1.0) > 1e-9:
        raise RuntimeError("causal C5 source order or mix ratio drift")

    fixture_manifest = FIXTURE_ROOT / "fixture_manifest.json"
    if preparation.sha256_file(fixture_manifest) != manifest_sha256:
        raise RuntimeError("causal fixture manifest identity drift")
    preparation.load_json(fixture_manifest)
    fixture_identity = preparation.validate_causal_fixture_manifest(FIXTURE_ROOT, manifest_sha256)
    if (
        plan.get("codec_contract_sha256") != fixture_identity["codec_contract_sha256"]
        or plan.get("codec_contract") != fixture_identity["codec_contract"]
        or plan.get("input_snapshot_sha256") != fixture_identity["input_snapshot_sha256"]
        or plan.get("budget_sha256") != fixture_identity["budget_sha256"]
        or plan.get("producer_identity") != fixture_identity["producer_identity"]
        or plan.get("producer_attempt_plan_sha256")
        != fixture_identity["producer_attempt_plan_sha256"]
        or plan.get("execution_plan_sha256") != fixture_identity["execution_plan_sha256"]
        or plan.get("causal_prefix_equivalence_proof")
        != fixture_identity["causal_prefix_equivalence_proof"]
    ):
        raise RuntimeError("preparation and live causal fixture/proof identities differ")

    for name, expected in CHECKPOINT_HASHES.items():
        path = RETAINED_CHECKPOINT / name
        if base.sha256_file(path) != expected:
            raise RuntimeError(f"retained checkpoint identity drift: {path}")
    retained_index_identity = validate_retained_native_index(RETAINED_CHECKPOINT)
    if not base.COSMOS_SNAPSHOT.is_file():
        raise RuntimeError(f"offline Cosmos snapshot is absent: {base.COSMOS_SNAPSHOT}")
    return {
        "preparation_root": str(preparation_root),
        "preparation_summary_sha256": preparation.sha256_file(summary_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": preparation.sha256_file(manifest_path),
        "adapter_config_sha256": dict(adapter_hashes),
        "fixture_root": str(FIXTURE_ROOT),
        **fixture_identity,
        "retained_checkpoint": str(RETAINED_CHECKPOINT),
        "retained_checkpoint_hashes": CHECKPOINT_HASHES,
        "retained_native_index": retained_index_identity,
        "source_ids": source_ids,
        "mix_ratios": mix_ratios,
    }


def build_train_command(
    *, mode: Any, dataset_manifest: Path, train_output: Path, master_port: int
) -> list[str]:
    return history.build_train_command(
        mode=mode,
        dataset_manifest=dataset_manifest,
        train_output=train_output,
        master_port=master_port,
    )


def _codec_posttrain_evidence(
    train_output: Path, expected_steps: int, source_ids: list[str]
) -> dict[str, Any]:
    evidence = history._codec_posttrain_evidence(train_output, expected_steps, source_ids)
    codec_ranks = evidence.get("codec_ranks") or []
    initial_state_complete = len(codec_ranks) == 2 and all(
        rank.get("initial_state_sha256") == EXPECTED_INITIAL_CODEC_STATE_SHA256
        and rank.get("initial_gate") == 0.0
        for rank in codec_ranks
    )
    for rank in codec_ranks:
        rank["frozen_initial_state_match"] = (
            rank.get("initial_state_sha256") == EXPECTED_INITIAL_CODEC_STATE_SHA256
            and rank.get("initial_gate") == 0.0
        )
    evidence["expected_initial_codec_state_sha256"] = EXPECTED_INITIAL_CODEC_STATE_SHA256
    evidence["initial_codec_state_closure_complete"] = initial_state_complete
    evidence["codec_state_closure_complete"] = bool(
        evidence.get("codec_state_closure_complete") and initial_state_complete
    )
    evidence["complete"] = bool(evidence.get("complete") and initial_state_complete)
    return evidence


def _tensor_payload(tensor: Any) -> bytes:
    import torch

    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _update_tensor_digest(digest: Any, name: str, tensor: Any, payload: bytes) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)


def _native_tensor_closure(checkpoint_root: Path) -> dict[str, Any]:
    """Require every retained native tensor to be value- and byte-identical."""

    from safetensors import safe_open
    import torch

    native_index = base.load_json(RETAINED_CHECKPOINT / "model.safetensors.index.json")
    target_index = base.load_json(checkpoint_root / "model.safetensors.index.json")
    native_map = native_index.get("weight_map")
    target_map = target_index.get("weight_map")
    if not isinstance(native_map, dict) or not isinstance(target_map, dict):
        raise RuntimeError("native/target checkpoint index is unavailable")
    native_names = set(native_map)
    codec_names = set(EXPECTED_PARAMETER_NAMES)
    target_names = set(target_map)
    key_closure_complete = (
        len(native_names) == EXPECTED_NATIVE_TENSOR_COUNT
        and target_names == native_names | codec_names
        and native_names.isdisjoint(codec_names)
    )

    native_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    value_mismatches = []
    byte_mismatches = []
    with ExitStack() as stack:
        native_handles = {
            shard: stack.enter_context(
                safe_open(RETAINED_CHECKPOINT / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(native_map.values()))
        }
        target_handles = {
            shard: stack.enter_context(
                safe_open(checkpoint_root / shard, framework="pt", device="cpu")
            )
            for shard in sorted({target_map[name] for name in native_names if name in target_map})
        }
        for name in sorted(native_names):
            if name not in target_map:
                value_mismatches.append(name)
                byte_mismatches.append(name)
                continue
            native_tensor = native_handles[native_map[name]].get_tensor(name)
            target_tensor = target_handles[target_map[name]].get_tensor(name)
            native_payload = _tensor_payload(native_tensor)
            target_payload = _tensor_payload(target_tensor)
            _update_tensor_digest(native_digest, name, native_tensor, native_payload)
            _update_tensor_digest(target_digest, name, target_tensor, target_payload)
            if not (
                native_tensor.dtype == target_tensor.dtype
                and native_tensor.shape == target_tensor.shape
                and torch.equal(native_tensor, target_tensor)
            ):
                value_mismatches.append(name)
            if native_payload != target_payload:
                byte_mismatches.append(name)
    native_state_sha256 = native_digest.hexdigest()
    target_native_state_sha256 = target_digest.hexdigest()
    complete = (
        key_closure_complete
        and not value_mismatches
        and not byte_mismatches
        and native_state_sha256 == target_native_state_sha256
    )
    return {
        "complete": complete,
        "native_tensor_count": len(native_names),
        "expected_native_tensor_count": EXPECTED_NATIVE_TENSOR_COUNT,
        "codec_tensor_count": len(codec_names),
        "key_closure_complete": key_closure_complete,
        "torch_equal_complete": not value_mismatches,
        "byte_equal_complete": not byte_mismatches,
        "value_mismatch_names": value_mismatches,
        "byte_mismatch_names": byte_mismatches,
        "retained_native_state_sha256": native_state_sha256,
        "target_native_state_sha256": target_native_state_sha256,
    }


def _saved_codec_checkpoint(
    train_output: Path, expected_state_sha256: str | None
) -> dict[str, Any]:
    result = history._saved_codec_checkpoint(train_output, expected_state_sha256)
    native_closure = _native_tensor_closure(train_output)
    result["native_tensor_closure"] = native_closure
    result["complete"] = bool(result.get("complete") and native_closure["complete"])
    return result


def _throughput_projection(
    mode_name: str, wall_seconds: float, max_steps: int
) -> tuple[float | None, bool | None]:
    return history._throughput_projection(mode_name, wall_seconds, max_steps)


def _source_coverage_complete(mode_name: str, source_sample_totals: dict[str, int]) -> bool:
    return history._source_coverage_complete(mode_name, source_sample_totals)


def run(args: argparse.Namespace) -> int:
    mode = MODES[args.mode]
    fixture_manifest_sha256 = _require_sha256(
        args.fixture_manifest_sha256, "fixture_manifest_sha256"
    )
    preparation_root = args.preparation_root.resolve(strict=True)
    source = base.validate_source(args.expected_source_commit)
    materials = validate_materials(preparation_root, fixture_manifest_sha256)
    train_output = args.run_dir / "train"
    train_command = build_train_command(
        mode=mode,
        dataset_manifest=Path(materials["dataset_manifest"]),
        train_output=train_output,
        master_port=args.master_port,
    )
    executed_command = base.isolation_command(train_command, args.run_dir)
    plan = {
        "schema_version": TRAINING_PLAN_SCHEMA_VERSION,
        "phase": TRAINING_PHASE,
        "mode": args.mode,
        "run_dir": str(args.run_dir),
        "source": source,
        "materials": materials,
        "training": {
            "max_steps": mode.max_steps,
            "fresh_optimizer": True,
            "retained_checkpoint_weights_only": True,
            "global_batch_size": 32,
            "seed": 42,
            "action_horizon": 40,
            "state_dropout_prob": 0.2,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.0 if mode.max_steps == 2 else 0.05,
            "codec_trainability": "codec_only",
            "expected_trainable_parameter_count": len(EXPECTED_PARAMETER_NAMES),
            "expected_trainable_numel": EXPECTED_TRAINABLE_NUMEL,
            "expected_initial_codec_state_sha256": EXPECTED_INITIAL_CODEC_STATE_SHA256,
            "expected_native_tensor_count": EXPECTED_NATIVE_TENSOR_COUNT,
            "native_tensor_value_and_byte_closure_required": True,
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": False,
            "tune_diffusion_model": False,
            "tune_vlln": False,
            "timeout_seconds": mode.timeout_seconds,
            "storage_limit_bytes": mode.storage_limit_bytes,
            "automatic_retries": 0,
        },
        "official_command": train_command,
        "executed_command": executed_command,
        "environment": base.child_environment(args.run_dir, source["source_commit"]),
        "network": "bwrap network namespace unshared; every model/data input is local",
        "gpu_gate": {
            "physical_indices": [0, 1],
            "minimum_free_mib": args.minimum_free_mib,
            "existing_compute_processes_allowed": False,
        },
    }
    if args.action == "plan":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.run_dir.exists():
        raise FileExistsError(f"attempt directory already exists: {args.run_dir}")
    if not args.run_dir.parent.is_dir():
        raise FileNotFoundError(f"attempt parent does not exist: {args.run_dir.parent}")
    free_bytes = os.statvfs(args.run_dir.parent)
    if free_bytes.f_bavail * free_bytes.f_frsize < mode.storage_limit_bytes:
        raise RuntimeError("insufficient free disk for bounded causal C5 attempt")

    gate_snapshot = base.gpu_snapshot()
    base.gate_gpus(gate_snapshot, args.minimum_free_mib)
    by_index = {gpu["index"]: gpu for gpu in gate_snapshot["gpus"]}
    run_id = args.run_dir.name
    with ExitStack() as leases:
        for index in (0, 1):
            leases.enter_context(base.GpuLease(by_index[index], args.lease_root, run_id))
        gate_snapshot = base.gpu_snapshot()
        base.gate_gpus(gate_snapshot, args.minimum_free_mib)

        args.run_dir.mkdir()
        for name in ("home", "cache", "matplotlib", "uv-cache"):
            (args.run_dir / name).mkdir()
        plan["gpu_at_gate"] = gate_snapshot
        plan["created_at"] = base.now()
        base.atomic_json(args.run_dir / "ATTEMPT.json", plan)

        start = time.monotonic()
        stop_reason = None
        cleanup = "not_needed"
        exit_code = None
        peak_used = {"0": 0, "1": 0}
        sample_count = 0
        owned = {}
        last_storage_check = 0.0
        process = None
        with (
            (args.run_dir / "training.stdout.log").open("xb") as stdout,
            (args.run_dir / "training.stderr.log").open("xb") as stderr,
            (args.run_dir / "RESOURCE_SAMPLES.jsonl").open("x", encoding="utf-8") as samples,
        ):
            try:
                process = subprocess.Popen(
                    executed_command,
                    cwd=SOURCE_ROOT,
                    env=plan["environment"],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                root_identity = base.proc_identity(process.pid)
                if root_identity is None:
                    raise RuntimeError("owned root process identity vanished at spawn")
                owned[process.pid] = root_identity
                while process.poll() is None:
                    owned.update(base.discover_owned(owned))
                    snapshot = base.gpu_snapshot()
                    for app in snapshot["compute_apps"]:
                        if app["pid"] not in owned and base.app_matches_attempt(
                            app["pid"], args.run_dir, source["source_commit"]
                        ):
                            identity = base.proc_identity(app["pid"])
                            if identity is not None:
                                owned[app["pid"]] = identity
                    unknown = [app for app in snapshot["compute_apps"] if app["pid"] not in owned]
                    if unknown:
                        owned.update(base.discover_owned(owned))
                        unknown = [
                            app for app in snapshot["compute_apps"] if app["pid"] not in owned
                        ]
                    if unknown:
                        stop_reason = "unrelated GPU compute process appeared: " + json.dumps(
                            unknown, sort_keys=True
                        )
                    for gpu in snapshot["gpus"]:
                        key = str(gpu["index"])
                        if key in peak_used:
                            peak_used[key] = max(peak_used[key], gpu["used_mib"])
                            if gpu["used_mib"] >= 47000:
                                stop_reason = f"GPU{key} used-memory stop line reached"
                    samples.write(json.dumps(snapshot, sort_keys=True) + "\n")
                    samples.flush()
                    sample_count += 1
                    elapsed = time.monotonic() - start
                    if elapsed >= mode.timeout_seconds:
                        stop_reason = "hard wall exceeded"
                    if elapsed - last_storage_check >= 30:
                        last_storage_check = elapsed
                        if base.directory_size(args.run_dir) > mode.storage_limit_bytes:
                            stop_reason = "attempt storage limit exceeded"
                    if stop_reason:
                        cleanup = base.terminate_owned(owned)
                        break
                    time.sleep(2)
                exit_code = process.wait()
                owned.update(base.discover_owned(owned))
                if any(base.identity_matches(identity) for identity in owned.values()):
                    cleanup = base.terminate_owned(owned)
            except BaseException as exc:
                stop_reason = stop_reason or f"{type(exc).__name__}: {exc}"
                if owned:
                    cleanup = base.terminate_owned(owned)
                if process is not None:
                    exit_code = process.poll()

        wall_seconds = time.monotonic() - start
        try:
            posttrain = _codec_posttrain_evidence(
                train_output, mode.max_steps, materials["source_ids"]
            )
        except BaseException as exc:
            posttrain = {
                "complete": False,
                "codec_ranks": [],
                "source_sample_totals": {source_id: 0 for source_id in materials["source_ids"]},
                "error": f"{type(exc).__name__}: {exc}",
            }
        expected_state = None
        if posttrain["codec_ranks"]:
            expected_state = posttrain["codec_ranks"][0]["final_state_sha256"]
        checkpoint_root = train_output / f"checkpoint-{mode.max_steps}"
        try:
            saved_checkpoint = _saved_codec_checkpoint(checkpoint_root, expected_state)
            saved_checkpoint["checkpoint_root"] = str(checkpoint_root)
        except BaseException as exc:
            saved_checkpoint = {
                "complete": False,
                "checkpoint_root": str(checkpoint_root),
                "error": f"{type(exc).__name__}: {exc}",
            }
        optimizer_state_required = args.mode in {"pilot", "formal"}
        optimizer_state_complete = None
        if optimizer_state_required:
            optimizer_root = checkpoint_root / f"global_step{mode.max_steps}"
            optimizer_state_complete = (
                len(list(optimizer_root.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt")))
                == 2
            )
        projected_20k_with_margin, throughput_gate = _throughput_projection(
            args.mode, wall_seconds, mode.max_steps
        )
        attempt_bytes = base.directory_size(args.run_dir)
        owned_alive = [identity for identity in owned.values() if base.identity_matches(identity)]
        gpu_after_cleanup = base.gpu_snapshot()
        cleanup_complete = not owned_alive and not gpu_after_cleanup["compute_apps"]
        technical_valid = (
            exit_code == 0
            and stop_reason is None
            and posttrain["complete"]
            and saved_checkpoint["complete"]
            and (not optimizer_state_required or optimizer_state_complete is True)
            and throughput_gate is not False
            and attempt_bytes <= mode.storage_limit_bytes
            and cleanup_complete
            and _source_coverage_complete(args.mode, posttrain["source_sample_totals"])
        )
        summary = {
            "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
            "phase": TRAINING_PHASE,
            "mode": args.mode,
            "run_id": run_id,
            "finished_at": base.now(),
            "exit_code": exit_code,
            "stop_reason": stop_reason,
            "cleanup": cleanup,
            "wall_seconds": wall_seconds,
            "peak_total_used_mib": peak_used,
            "resource_sample_count": sample_count,
            "attempt_bytes": attempt_bytes,
            "storage_limit_bytes": mode.storage_limit_bytes,
            "owned_processes": sorted(owned.values(), key=lambda item: item["pid"]),
            "owned_processes_alive_after_cleanup": owned_alive,
            "gpu_after_cleanup": gpu_after_cleanup,
            "cleanup_complete": cleanup_complete,
            "posttrain_evidence": posttrain,
            "saved_codec_checkpoint": saved_checkpoint,
            "optimizer_state_required": optimizer_state_required,
            "optimizer_state_complete": optimizer_state_complete,
            "projected_20k_seconds_with_25_percent_margin": projected_20k_with_margin,
            "throughput_gate": throughput_gate,
            "technical_valid": technical_valid,
            "scientific_interpretation": (
                "causal F2 training execution evidence only; evaluator/native success is not inferred"
            ),
            "source_commit": source["source_commit"],
            "dataset_manifest_sha256": materials["dataset_manifest_sha256"],
            "fixture_manifest_sha256": fixture_manifest_sha256,
            "codec_contract_sha256": materials["codec_contract_sha256"],
            "model_output": str(train_output),
        }
        base.atomic_json(args.run_dir / "SUPERVISOR_SUMMARY.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if technical_valid else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--mode", required=True, choices=tuple(MODES))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--fixture-manifest-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--lease-root", type=Path, default=Path("/tmp/research-control-gpu-leases"))
    parser.add_argument("--minimum-free-mib", type=int, default=40000)
    parser.add_argument("--master-port", type=int, default=29573)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.run_dir.is_absolute() or not args.preparation_root.is_absolute():
        parser.error("run-dir and preparation-root must be absolute")
    try:
        _require_sha256(args.fixture_manifest_sha256, "fixture_manifest_sha256")
    except ValueError as exc:
        parser.error(str(exc))
    if args.minimum_free_mib < 0:
        parser.error("minimum-free-mib must be non-negative")
    if not 1024 <= args.master_port <= 65535:
        parser.error("master-port must be between 1024 and 65535")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
