#!/usr/bin/env python3
"""Bounded offline supervisor for C5 strict Codec-only F2 training."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[2]
BASE_SUPERVISOR_PATH = SOURCE_ROOT / "examples/G1_ALL_LEROBOT_H40/launch_joint_training.py"
SPEC = importlib.util.spec_from_file_location("c5_base_training_supervisor", BASE_SUPERVISOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load base training supervisor: {BASE_SUPERVISOR_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


RETAINED_CHECKPOINT = Path(
    "/home/sl/icra27-runs/research-control/groot-n17-all-lerobot-20k-r1/experiments/"
    "groot-n17-all-lerobot-20k-r1/runs/"
    "train-20k-attempt-1-20260827T192825p0800.artifacts/attempt/train/checkpoint-20000"
)
CHECKPOINT_HASHES = {
    "model-00001-of-00002.safetensors": (
        "5524cf9933dd6ec40687637444b22db6dc76921dfe0e1843c7140e8701763cf7"
    ),
    "model-00002-of-00002.safetensors": (
        "51784715d68c870fbbcdb8b5790c4d4514b86f91722506e3d04d6df1b594739d"
    ),
    "model.safetensors.index.json": (
        "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
    ),
    "config.json": "b9f983cdc721cbd97d93c78127619dfbfef626d8f037f95f0d01654b6822a155",
    "processor_config.json": ("24c2a1dfd82d1988e46c46be9ddef496b348865507a21e6278f970af403ee282"),
    "statistics.json": "3f382750ac8603f12dda748ef94495b6e2a8bce26b346d5325182a5af498148e",
    "embodiment_id.json": "6f01100d50c52602f0c7c68304eea48f79a1f8c2426a960c04824c0ed4557869",
}
FIXTURE_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-full-r2")
FIXTURE_MANIFEST_SHA256 = "c05736e1e33745ee0580d45aeebf651891a6533701297f23f41a1bd5090c3a88"
NATIVE_PREPARATION = Path(
    "/home/sl/icra27-runs/groot-n17-all-lerobot-20k-r1/prep-20260827T171352+0800"
)
NATIVE_MANIFEST_SHA256 = "57769ef4469e3f094240299ae4f2c5935f7c9c452a06ff2349122f509776bdbe"
NATIVE_CLOSURE_SHA256 = "3fe4d1625920032d5242bcfec298552314f2b9cac3917db4cdf734d2cfd51974"
EXPECTED_TRAINABLE_NUMEL = 9_571_649
EXPECTED_PARAMETER_SUFFIXES = (
    "core.mv_stem.weight",
    "core.mv_stem.bias",
    "core.rpe_stem.weight",
    "core.rpe_stem.bias",
    "core.i_frame_stem.weight",
    "core.i_frame_stem.bias",
    "core.metadata_stem.weight",
    "core.metadata_stem.bias",
    "core.norm.weight",
    "core.norm.bias",
    "core.mlp.0.weight",
    "core.mlp.0.bias",
    "core.mlp.2.weight",
    "core.mlp.2.bias",
    "projector.projection.weight",
    "projector.projection.bias",
    "conditioner.gate",
    "conditioner.query_norm.weight",
    "conditioner.query_norm.bias",
    "conditioner.context_norm.weight",
    "conditioner.context_norm.bias",
    "conditioner.attention.in_proj_weight",
    "conditioner.attention.in_proj_bias",
    "conditioner.attention.out_proj.weight",
    "conditioner.attention.out_proj.bias",
    "conditioner.output_norm.weight",
    "conditioner.output_norm.bias",
)
EXPECTED_PARAMETER_NAMES = tuple(
    f"action_head.codec_conditioning.{suffix}" for suffix in EXPECTED_PARAMETER_SUFFIXES
)
EXPECTED_PARAMETER_NAMES_SHA256 = hashlib.sha256(
    "\n".join(EXPECTED_PARAMETER_NAMES).encode("utf-8")
).hexdigest()
SOURCE_IDS = (
    "no_torque_100",
    "aug1p2mm_220",
    "aug5p4mm_153",
    "plain_300",
    "with_torque_33",
)
NATIVE_ADAPTER_SHA256 = {
    "no_torque_100": "8ecf02755019e8b81e3fc018b2ffe6a747916e7596eabc5f232dcc5f1da4a9ee",
    "aug1p2mm_220": "d30a214850d25b5075c0d127872f5d45300a87d37c9d8095a169ce3e55dd4140",
    "aug5p4mm_153": "625bc4ec6b4ee1117b90bf58da53f751f450e24dd7f768c3253acf0628acdcdd",
    "plain_300": "1c0a13d7667f0d1dd4a83b5094f8667f1d781ac643f8c2aa38b665f000f820ef",
    "with_torque_33": "63267173407ffa0c9077ec7f70d0cfefe9e90adb3a154052bf11ef33d63150c7",
}


MODES = {
    "profile": base.Mode(
        max_steps=2,
        save_steps=2,
        save_total_limit=1,
        save_only_model=True,
        timeout_seconds=3300,
        storage_limit_bytes=24 * 1024**3,
    ),
    "pilot": base.Mode(
        max_steps=500,
        save_steps=500,
        save_total_limit=1,
        save_only_model=False,
        timeout_seconds=3300,
        storage_limit_bytes=48 * 1024**3,
    ),
    "formal": base.Mode(
        max_steps=20000,
        save_steps=1000,
        save_total_limit=2,
        save_only_model=False,
        timeout_seconds=28200,
        storage_limit_bytes=128 * 1024**3,
    ),
}


def _model_codec_contract() -> dict[str, Any]:
    return {
        "enabled": True,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "d_codec": 64,
        "conditioner_heads": 8,
    }


def _adapter_codec_contract() -> dict[str, Any]:
    return {
        **_model_codec_contract(),
        "fixture_root": str(FIXTURE_ROOT),
        "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
    }


def validate_materials(preparation_root: Path) -> dict[str, Any]:
    summary_path = preparation_root / "PREPARATION_SUMMARY.json"
    manifest_path = preparation_root / "DATASET_MANIFEST.json"
    summary = base.load_json(summary_path)
    manifest = base.load_json(manifest_path)
    if not (
        summary.get("schema_version") == "g1-all-lerobot-c5-codec-only-preparation-summary-v1"
        and summary.get("status") == "PASS"
        and summary.get("source_count") == 5
        and summary.get("total_episodes") == 806
        and summary.get("total_h40_anchors") == 265906
    ):
        raise RuntimeError("C5 preparation summary is not the accepted exact closure")
    if summary.get("dataset_manifest_sha256") != base.sha256_file(manifest_path):
        raise RuntimeError("C5 preparation dataset manifest identity drift")
    plan = summary.get("plan", {})
    if not (
        plan.get("native_manifest_sha256") == NATIVE_MANIFEST_SHA256
        and plan.get("native_cpu_mixture_closure_sha256") == NATIVE_CLOSURE_SHA256
        and plan.get("native_adapter_sha256") == NATIVE_ADAPTER_SHA256
        and plan.get("fixture_manifest_sha256") == FIXTURE_MANIFEST_SHA256
    ):
        raise RuntimeError("C5 preparation provenance closure drift")
    datasets = manifest.get("datasets")
    if manifest.get("schema_version") != "gr00t-finetune-dataset-manifest-v1" or not isinstance(
        datasets, list
    ):
        raise RuntimeError("C5 dataset manifest schema is invalid")
    if len(datasets) != 5:
        raise RuntimeError("C5 dataset manifest must contain exactly five sources")

    source_ids = []
    mix_ratios = []
    adapter_hashes = summary.get("adapter_config_sha256", {})
    for expected_source_id, entry in zip(SOURCE_IDS, datasets):
        dataset_path = Path(entry["dataset_path"]).resolve(strict=True)
        adapter_path = Path(entry["adapter_config_path"]).resolve(strict=True)
        adapter = base.load_json(adapter_path)
        source_id = adapter.get("source_id")
        source_ids.append(source_id)
        mix_ratios.append(float(entry["mix_ratio"]))
        if adapter.get("codec_evidence") != _adapter_codec_contract():
            raise RuntimeError(f"C5 Codec adapter contract drift: {adapter_path}")
        if adapter_hashes.get(source_id) != base.sha256_file(adapter_path):
            raise RuntimeError(f"C5 adapter identity drift: {adapter_path}")
        native_adapter_path = (
            NATIVE_PREPARATION / "sources" / expected_source_id / "ADAPTER_CONFIG.json"
        )
        native_adapter = base.load_json(native_adapter_path)
        if base.sha256_file(native_adapter_path) != NATIVE_ADAPTER_SHA256[expected_source_id]:
            raise RuntimeError(f"native adapter identity drift: {native_adapter_path}")
        if adapter != {**native_adapter, "codec_evidence": _adapter_codec_contract()}:
            raise RuntimeError(
                f"derived C5 adapter is not the exact native overlay: {adapter_path}"
            )
        expected_dataset_path = (
            NATIVE_PREPARATION / "sources" / expected_source_id / "LOADER_VIEW"
        ).resolve(strict=True)
        if dataset_path != expected_dataset_path:
            raise RuntimeError(f"C5 loader-view path drift: {dataset_path}")
        try:
            adapter_path.relative_to(preparation_root)
        except ValueError as exc:
            raise RuntimeError("derived C5 adapters must stay inside preparation root") from exc
        for name in ("data", "videos"):
            payload = dataset_path / name
            if not payload.is_symlink():
                raise RuntimeError(
                    f"loader-view payload is not a read-only symlink: {dataset_path}"
                )
            expected_target = (Path(native_adapter["canonical_root"]) / name).resolve(strict=True)
            if payload.resolve(strict=True) != expected_target:
                raise RuntimeError(f"loader-view payload target drift: {payload}")
    if tuple(source_ids) != SOURCE_IDS or abs(sum(mix_ratios) - 1.0) > 1e-9:
        raise RuntimeError("C5 source order or mix ratio drift")

    fixture_manifest = FIXTURE_ROOT / "fixture_manifest.json"
    if base.sha256_file(fixture_manifest) != FIXTURE_MANIFEST_SHA256:
        raise RuntimeError("full-R2 fixture manifest identity drift")
    fixture = base.load_json(fixture_manifest)
    if not (
        fixture.get("complete") is True
        and fixture.get("covered_episode_count") == 806
        and len(fixture.get("objects", [])) == 806
    ):
        raise RuntimeError("full-R2 fixture exact coverage is absent")
    for name, expected in CHECKPOINT_HASHES.items():
        path = RETAINED_CHECKPOINT / name
        if base.sha256_file(path) != expected:
            raise RuntimeError(f"retained checkpoint identity drift: {path}")
    if not base.COSMOS_SNAPSHOT.is_file():
        raise RuntimeError(f"offline Cosmos snapshot is absent: {base.COSMOS_SNAPSHOT}")
    return {
        "preparation_root": str(preparation_root),
        "preparation_summary_sha256": base.sha256_file(summary_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": base.sha256_file(manifest_path),
        "adapter_config_sha256": adapter_hashes,
        "fixture_root": str(FIXTURE_ROOT),
        "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "retained_checkpoint": str(RETAINED_CHECKPOINT),
        "retained_checkpoint_hashes": CHECKPOINT_HASHES,
        "source_ids": source_ids,
        "mix_ratios": mix_ratios,
    }


def build_train_command(
    *, mode: Any, dataset_manifest: Path, train_output: Path, master_port: int
) -> list[str]:
    warmup_ratio = 0.0 if mode.max_steps == 2 else 0.05
    return [
        "uv",
        "run",
        "torchrun",
        "--nproc_per_node=2",
        f"--master_port={master_port}",
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        str(RETAINED_CHECKPOINT),
        "--dataset-manifest-path",
        str(dataset_manifest),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(base.MODALITY_CONFIG),
        "--dataset-adapter",
        "g1_all_lerobot_h40_v1",
        "--num-gpus",
        "2",
        "--output-dir",
        str(train_output),
        "--max-steps",
        str(mode.max_steps),
        "--seed",
        "42",
        "--global-batch-size",
        "32",
        "--gradient-accumulation-steps",
        "1",
        "--dataloader-num-workers",
        "0",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "1e-05",
        "--warmup-ratio",
        str(warmup_ratio),
        "--save-steps",
        str(mode.save_steps),
        "--save-total-limit",
        str(mode.save_total_limit),
        "--action-horizon",
        "40",
        "--state-dropout-prob",
        "0.2",
        "--no-tune-llm",
        "--no-tune-visual",
        "--no-tune-projector",
        "--no-tune-diffusion-model",
        "--no-tune-vlln",
        "--codec-trainability",
        "codec_only",
        "--no-use-wandb",
        "--save-only-model" if mode.save_only_model else "--no-save-only-model",
    ]


def _codec_posttrain_evidence(
    train_output: Path, expected_steps: int, source_ids: list[str]
) -> dict[str, Any]:
    evidence = base._posttrain_evidence(train_output, expected_steps, source_ids)
    codec_ranks = []
    for path in sorted((train_output / "distributed_evidence").glob("rank_*_posttrain.json")):
        value = base.load_json(path)
        optimizer = value.get("optimizer_trainability") or {}
        trainability = (value.get("training_monitor") or {}).get("trainability") or {}
        codec_ranks.append(
            {
                "rank_file": str(path),
                "optimizer_complete": (
                    optimizer.get("mode") == "codec_only"
                    and optimizer.get("parameter_count") == len(EXPECTED_PARAMETER_NAMES)
                    and optimizer.get("trainable_numel") == EXPECTED_TRAINABLE_NUMEL
                    and optimizer.get("parameter_names") == list(EXPECTED_PARAMETER_NAMES)
                    and optimizer.get("parameter_names_sha256") == EXPECTED_PARAMETER_NAMES_SHA256
                ),
                "gradient_complete": trainability.get("complete") is True,
                "initial_state_sha256": trainability.get("initial_state_sha256"),
                "final_state_sha256": trainability.get("final_state_sha256"),
                "initial_gate": trainability.get("initial_gate"),
                "final_gate": trainability.get("final_gate"),
                "first_gradient_steps": trainability.get("first_gradient_steps"),
            }
        )
    expected_non_gate = set(EXPECTED_PARAMETER_NAMES) - {
        "action_head.codec_conditioning.conditioner.gate"
    }
    gradient_capture_count = min(expected_steps, 3)
    distributed_gradients_complete = len(codec_ranks) == 2 and all(
        len(rank["first_gradient_steps"] or []) == gradient_capture_count for rank in codec_ranks
    )
    if distributed_gradients_complete:
        gradient_steps = [rank["first_gradient_steps"] for rank in codec_ranks]
        for step_index in range(gradient_capture_count):
            step_ranks = [steps[step_index] for steps in gradient_steps]
            gate_entries = [rank["gate"] for rank in step_ranks]
            non_gate_present = {
                name for rank in step_ranks for name in rank["non_gate_present_names"]
            }
            if (
                not any(gate["present"] for gate in gate_entries)
                or any(gate["present"] and not gate["finite"] for gate in gate_entries)
                or non_gate_present != expected_non_gate
                or any(rank["non_gate_nonfinite_names"] for rank in step_ranks)
            ):
                distributed_gradients_complete = False
                break
        if distributed_gradients_complete:
            first_gate = [rank["gate"] for rank in (steps[0] for steps in gradient_steps)]
            later_non_gate_nonzero = {
                name
                for steps in gradient_steps
                for step in steps[1:]
                for name in step["non_gate_nonzero_names"]
            }
            distributed_gradients_complete = any(
                gate["present"] and gate["nonzero"] for gate in first_gate
            ) and bool(later_non_gate_nonzero)
    state_closure = (
        len(codec_ranks) == 2
        and all(rank["optimizer_complete"] and rank["gradient_complete"] for rank in codec_ranks)
        and distributed_gradients_complete
        and len({rank["initial_state_sha256"] for rank in codec_ranks}) == 1
        and len({rank["final_state_sha256"] for rank in codec_ranks}) == 1
        and all(rank["initial_gate"] == 0.0 for rank in codec_ranks)
        and all(
            isinstance(rank["final_gate"], (int, float))
            and math.isfinite(rank["final_gate"])
            and rank["final_gate"] != 0.0
            for rank in codec_ranks
        )
    )
    evidence["codec_ranks"] = codec_ranks
    evidence["distributed_gradients_complete"] = distributed_gradients_complete
    evidence["codec_state_closure_complete"] = state_closure
    evidence["complete"] = evidence["complete"] and state_closure
    return evidence


def _saved_codec_checkpoint(
    train_output: Path, expected_state_sha256: str | None
) -> dict[str, Any]:
    config = base.load_json(train_output / "config.json")
    processor = base.load_json(train_output / "processor_config.json")
    index = base.load_json(train_output / "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    config_complete = (
        config.get("codec_evidence") == _model_codec_contract()
        and config.get("codec_trainability") == "codec_only"
        and all(
            config.get(name) in (False, 0)
            for name in (
                "tune_llm",
                "tune_visual",
                "tune_projector",
                "tune_diffusion_model",
                "tune_vlln",
                "tune_top_llm_layers",
            )
        )
    )
    processor_complete = (
        processor.get("processor_kwargs", {}).get("codec_evidence") == _model_codec_contract()
    )
    support_files = {
        name: base.sha256_file(train_output / name)
        for name in ("statistics.json", "embodiment_id.json")
    }
    retained_statistics = base.load_json(RETAINED_CHECKPOINT / "statistics.json")
    retained_embodiment_mapping = base.load_json(RETAINED_CHECKPOINT / "embodiment_id.json")
    support_semantics_complete = (
        base.load_json(train_output / "statistics.json") == retained_statistics
        and base.load_json(train_output / "embodiment_id.json") == retained_embodiment_mapping
    )
    offline_values = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    }
    previous_environment = {name: os.environ.get(name) for name in offline_values}
    os.environ.update(offline_values)
    try:
        import gr00t.model  # noqa: F401 - registers the local Auto classes
        from safetensors import safe_open
        import torch
        from transformers import AutoConfig, AutoProcessor

        loaded_config = AutoConfig.from_pretrained(
            train_output,
            trust_remote_code=True,
            local_files_only=True,
        )
        loaded_processor = AutoProcessor.from_pretrained(
            train_output,
            trust_remote_code=True,
            local_files_only=True,
            transformers_loading_kwargs={
                "trust_remote_code": True,
                "local_files_only": True,
            },
        )
    finally:
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    standalone_reload_complete = (
        getattr(loaded_config, "codec_evidence", None) == _model_codec_contract()
        and getattr(loaded_config, "codec_trainability", None) == "codec_only"
        and getattr(loaded_processor, "codec_evidence", None) == _model_codec_contract()
        and getattr(loaded_processor, "embodiment_id_mapping", None) == retained_embodiment_mapping
    )
    mapped_names = [name for name in EXPECTED_PARAMETER_NAMES if name in weight_map]
    tensors = {}
    for shard in sorted({weight_map[name] for name in mapped_names}):
        names = [name for name in mapped_names if weight_map[name] == shard]
        with safe_open(train_output / shard, framework="pt", device="cpu") as stream:
            for name in names:
                tensors[name] = stream.get_tensor(name)
    digest = hashlib.sha256()
    numel = 0
    finite = True
    for name in EXPECTED_PARAMETER_NAMES:
        tensor = tensors.get(name)
        if tensor is None:
            continue
        numel += tensor.numel()
        finite = finite and bool(torch.isfinite(tensor).all().item())
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
    state_sha256 = digest.hexdigest()
    gate = tensors.get("action_head.codec_conditioning.conditioner.gate")
    gate_value = float(gate.float().item()) if gate is not None and gate.numel() == 1 else None
    complete = (
        config_complete
        and processor_complete
        and support_semantics_complete
        and standalone_reload_complete
        and tuple(mapped_names) == EXPECTED_PARAMETER_NAMES
        and len(tensors) == len(EXPECTED_PARAMETER_NAMES)
        and numel == EXPECTED_TRAINABLE_NUMEL
        and finite
        and gate_value is not None
        and math.isfinite(gate_value)
        and gate_value != 0.0
        and state_sha256 == expected_state_sha256
    )
    return {
        "config_complete": config_complete,
        "processor_complete": processor_complete,
        "support_file_sha256": support_files,
        "support_semantics_complete": support_semantics_complete,
        "standalone_reload_complete": standalone_reload_complete,
        "loaded_config_class": f"{type(loaded_config).__module__}.{type(loaded_config).__qualname__}",
        "loaded_processor_class": (
            f"{type(loaded_processor).__module__}.{type(loaded_processor).__qualname__}"
        ),
        "parameter_count": len(tensors),
        "trainable_numel": numel,
        "finite": finite,
        "gate": gate_value,
        "state_sha256": state_sha256,
        "matches_posttrain_state": state_sha256 == expected_state_sha256,
        "complete": complete,
    }


def run(args: argparse.Namespace) -> int:
    mode = MODES[args.mode]
    preparation_root = args.preparation_root.resolve(strict=True)
    source = base.validate_source(args.expected_source_commit)
    materials = validate_materials(preparation_root)
    train_output = args.run_dir / "train"
    port = args.master_port
    train_command = build_train_command(
        mode=mode,
        dataset_manifest=Path(materials["dataset_manifest"]),
        train_output=train_output,
        master_port=port,
    )
    executed_command = base.isolation_command(train_command, args.run_dir)
    plan = {
        "schema_version": "g1-all-lerobot-c5-codec-only-training-plan-v1",
        "phase": "F2_REAL_CODEC_EVIDENCE",
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
        raise RuntimeError("insufficient free disk for bounded C5 attempt")

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
        optimizer_state_complete = True
        if args.mode in {"pilot", "formal"}:
            optimizer_root = (
                train_output / f"checkpoint-{mode.max_steps}/global_step{mode.max_steps}"
            )
            optimizer_state_complete = (
                len(list(optimizer_root.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt")))
                == 2
            )
        projected_20k_with_margin = (
            wall_seconds / mode.max_steps * 20000 * 1.25 if args.mode == "pilot" else None
        )
        throughput_gate = (
            projected_20k_with_margin is None
            or projected_20k_with_margin <= MODES["formal"].timeout_seconds
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
            and optimizer_state_complete
            and throughput_gate
            and attempt_bytes <= mode.storage_limit_bytes
            and cleanup_complete
            and (
                args.mode != "formal"
                or all(value > 0 for value in posttrain["source_sample_totals"].values())
            )
        )
        summary = {
            "schema_version": "g1-all-lerobot-c5-codec-only-training-summary-v1",
            "phase": "F2_REAL_CODEC_EVIDENCE",
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
            "optimizer_state_complete": optimizer_state_complete,
            "projected_20k_seconds_with_25_percent_margin": projected_20k_with_margin,
            "throughput_gate": throughput_gate,
            "technical_valid": technical_valid,
            "scientific_interpretation": (
                "F2 training execution evidence only; no evaluator/native success is inferred"
            ),
            "source_commit": source["source_commit"],
            "dataset_manifest_sha256": materials["dataset_manifest_sha256"],
            "model_output": str(train_output),
        }
        base.atomic_json(args.run_dir / "SUPERVISOR_SUMMARY.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if technical_valid else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--mode", required=True, choices=tuple(MODES))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--lease-root", type=Path, default=Path("/tmp/research-control-gpu-leases"))
    parser.add_argument("--minimum-free-mib", type=int, default=40000)
    parser.add_argument("--master-port", type=int, default=29573)
    args = parser.parse_args(argv)
    if not args.run_dir.is_absolute() or not args.preparation_root.is_absolute():
        parser.error("run-dir and preparation-root must be absolute")
    if args.minimum_free_mib < 0:
        parser.error("minimum-free-mib must be non-negative")
    if not 1024 <= args.master_port <= 65535:
        parser.error("master-port must be between 1024 and 65535")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
