#!/usr/bin/env python3
"""Materialize an immutable C5 overlay for the causal full fixture.

The native five-source loader views and retained training cohort remain read-only.
The caller must provide the already-committed causal fixture manifest SHA-256;
this entry point never discovers or rewrites that identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import sys
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gr00t.data.codec_evidence.codec_contract import (  # noqa: E402
    CODEC_CONTRACT_SCHEMA_VERSION,
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    validate_codec_contract,
)
from gr00t.data.codec_evidence.warehouse import (  # noqa: E402
    ATTEMPT_SCHEMA_VERSION,
    canonical_json_sha256,
    validate_full_fixture_manifest,
)


PREPARATION_PLAN_SCHEMA_VERSION = "g1-all-lerobot-c5-causal-codec-only-preparation-plan-v1"
PREPARATION_SUMMARY_SCHEMA_VERSION = "g1-all-lerobot-c5-causal-codec-only-preparation-summary-v1"
NATIVE_PREPARATION = Path(
    "/home/sl/icra27-runs/groot-n17-all-lerobot-20k-r1/prep-20260827T171352+0800"
)
NATIVE_MANIFEST_SHA256 = "57769ef4469e3f094240299ae4f2c5935f7c9c452a06ff2349122f509776bdbe"
NATIVE_CLOSURE_SHA256 = "3fe4d1625920032d5242bcfec298552314f2b9cac3917db4cdf734d2cfd51974"
FIXTURE_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r2")
EXPECTED_FIXTURE_ID = "g1-all-lerobot-causal-full-r2"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PREFIX_FACTORY_PATH = SOURCE_ROOT / "gr00t/data/codec_evidence/prefix_materializer.py"
PREFIX_FACTORY_SHA256 = "88c29c9f288651d05b691ac5b5648168742c44a4eaf113ee2a5b400f09ad2f0a"
PRODUCTION_MODULE_PATH = SOURCE_ROOT / "gr00t/data/codec_evidence/production.py"
PRODUCTION_MODULE_SHA256 = "526a1439e5fe240290d144b3b601de7e7e232fc0edc83eea2ff912d8ded5c04a"
EXPECTED_EPISODES = 806
EXPECTED_TOTAL_FRAMES = 297_340
EXPECTED_H40_ANCHORS = 265_906
EXPECTED_EPISODE_UID_DIGEST = "c930141211807aaa2e370954b6543828a5abe75e5d83f8a636095eb95ea40bc8"
EXPECTED_INPUT_SNAPSHOT_SHA256 = "db9e78c8d7c6ada793f20ab3c41e1fb9e84550d2886646ea8b050f6c3f1de483"
EXPECTED_BUDGET_SHA256 = "5f24fb31ea766941b3861247478b97b38b62de42ae9c34b64fe56d5dc762644b"
EXPECTED_INPUT_FREEZE_FILE_SHA256 = (
    "f2e950071eb6187be7747658ae46598b8a3f741c37b710a6e2288c041ea7b9cd"
)
EXPECTED_EPISODE_FREEZE_CORE_SHA256 = (
    "8d05f252b6ea388cb27a3155db8359ce7ea759436dfd9e9294c865fc47ea594a"
)
CAUSAL_PROOF_PATH = Path(
    "/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-contract-r1/"
    "CAUSAL_PREFIX_EQUIVALENCE.json"
)
CAUSAL_PROOF_SHA256 = "44a0e88cd1577dee942e84fc9420c111b86754957e966d39d155d87446c903c9"
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PRODUCER_FIELDS = {
    "schema_version",
    "attempt_id",
    "resume_chain",
    "attempt_plan_sha256",
    "execution_plan_sha256",
    "git_head",
    "toolchain_identity_sha256",
    "implementation_identity_sha256",
    "input_freeze_file_sha256",
    "input_snapshot_sha256",
    "episode_freeze_core_sha256",
    "expected_episode_lengths_sha256",
    "expected_derivations_sha256",
    "automatic_retries",
}
_ATTEMPT_STARTED_FIELDS = {
    "schema_version",
    "attempt_id",
    "mode",
    "resume_from_attempt_id",
    "plan_sha256",
    "plan",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _causal_builder_module():
    path = Path(__file__).resolve().with_name("build_c3_full_fixture.py")
    spec = importlib.util.spec_from_file_location("c5_causal_fixture_authority", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load causal fixture authority: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_causal_full_r1()
    return module


def validate_causal_proof() -> dict[str, Any]:
    """Reuse the fixture producer's proof/ancestry validator verbatim."""

    builder = _causal_builder_module()
    identity = builder._validate_causal_prefix_proof(SOURCE_ROOT)
    if not isinstance(identity, dict) or not (
        identity.get("path") == str(CAUSAL_PROOF_PATH)
        and identity.get("sha256") == CAUSAL_PROOF_SHA256
    ):
        raise RuntimeError("causal proof identity differs from the frozen successor contract")
    return identity


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"JSON input must be a regular non-symlink file: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one JSON file without replacing any prior path."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def codec_evidence_config(fixture_manifest_sha256: str) -> dict[str, Any]:
    manifest_sha256 = _require_sha256(fixture_manifest_sha256, "fixture_manifest_sha256")
    return {
        "enabled": True,
        "fixture_root": str(FIXTURE_ROOT),
        "fixture_manifest_sha256": manifest_sha256,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "d_codec": 64,
        "conditioner_heads": 8,
    }


def validate_producer_and_proof(fixture_root: Path, fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the committed manifest to its exact attempt plan and causal proof."""

    if not (
        fixture.get("input_snapshot_sha256") == EXPECTED_INPUT_SNAPSHOT_SHA256
        and fixture.get("budget_sha256") == EXPECTED_BUDGET_SHA256
    ):
        raise RuntimeError("causal fixture input snapshot/budget identity drifted")
    producer = fixture.get("producer_identity")
    if not isinstance(producer, Mapping) or set(producer) != _PRODUCER_FIELDS:
        raise RuntimeError("causal fixture producer identity fields are not exact")
    attempt_id = producer.get("attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise RuntimeError("causal fixture producer attempt ID is unsafe")
    started_path = fixture_root / "attempts" / attempt_id / "ATTEMPT_STARTED.json"
    started = load_json(started_path)
    if not (
        set(started) == _ATTEMPT_STARTED_FIELDS
        and started.get("schema_version") == ATTEMPT_SCHEMA_VERSION
        and started.get("attempt_id") == attempt_id
        and isinstance(started.get("plan"), dict)
        and canonical_json_sha256(started["plan"]) == started.get("plan_sha256")
        and started.get("plan_sha256") == producer.get("attempt_plan_sha256")
    ):
        raise RuntimeError("causal fixture producer attempt plan closure drifted")
    plan = started["plan"]
    detached_plan = dict(plan)
    embedded_plan_sha256 = detached_plan.pop("plan_sha256", None)
    if embedded_plan_sha256 != canonical_json_sha256(detached_plan):
        raise RuntimeError("causal fixture execution plan digest is internally inconsistent")
    proof_identity = validate_causal_proof()
    lengths = sorted(
        (
            {"episode_uid": item["episode_uid"], "num_frames": item["num_frames"]}
            for item in fixture["objects"]
        ),
        key=lambda item: item["episode_uid"],
    )
    derivations = sorted(
        (
            {
                "episode_uid": item["episode_uid"],
                "derivation_sha256": item["derivation_sha256"],
            }
            for item in fixture["objects"]
        ),
        key=lambda item: item["episode_uid"],
    )
    expected_producer = {
        "schema_version": "g1-all-lerobot-c3-producer-identity-v1",
        "attempt_id": attempt_id,
        "attempt_plan_sha256": started["plan_sha256"],
        "execution_plan_sha256": embedded_plan_sha256,
        "git_head": plan.get("toolchain", {}).get("git", {}).get("head"),
        "toolchain_identity_sha256": plan.get("toolchain_identity_sha256"),
        "implementation_identity_sha256": plan.get("implementation_identity_sha256"),
        "input_freeze_file_sha256": EXPECTED_INPUT_FREEZE_FILE_SHA256,
        "input_snapshot_sha256": EXPECTED_INPUT_SNAPSHOT_SHA256,
        "episode_freeze_core_sha256": EXPECTED_EPISODE_FREEZE_CORE_SHA256,
        "expected_episode_lengths_sha256": canonical_json_sha256(lengths),
        "expected_derivations_sha256": canonical_json_sha256(derivations),
        "automatic_retries": 0,
    }
    if any(producer.get(key) != value for key, value in expected_producer.items()):
        raise RuntimeError("causal fixture producer identity differs from its execution plan")
    resume_chain = producer.get("resume_chain")
    if not isinstance(resume_chain, list) or any(
        not isinstance(item, str) or _ATTEMPT_ID_RE.fullmatch(item) is None for item in resume_chain
    ):
        raise RuntimeError("causal fixture producer resume chain is malformed")
    if not (
        plan.get("fixture_id") == EXPECTED_FIXTURE_ID
        and plan.get("output_root") == str(FIXTURE_ROOT)
        and plan.get("input_snapshot_sha256") == EXPECTED_INPUT_SNAPSHOT_SHA256
        and plan.get("budget_sha256") == EXPECTED_BUDGET_SHA256
        and plan.get("episode_freeze_core_sha256") == EXPECTED_EPISODE_FREEZE_CORE_SHA256
        and plan.get("expected_episode_count") == EXPECTED_EPISODES
        and plan.get("expected_total_frames") == EXPECTED_TOTAL_FRAMES
        and plan.get("expected_episode_uid_digest") == EXPECTED_EPISODE_UID_DIGEST
        and plan.get("expected_episode_lengths_sha256")
        == producer.get("expected_episode_lengths_sha256")
        and plan.get("codec_contract") == fixture.get("codec_contract")
        and plan.get("codec_contract_sha256") == fixture.get("codec_contract_sha256")
        and plan.get("causal_prefix_equivalence_proof") == proof_identity
    ):
        raise RuntimeError("causal fixture execution plan/proof binding drifted")
    return {
        "input_snapshot_sha256": EXPECTED_INPUT_SNAPSHOT_SHA256,
        "budget_sha256": EXPECTED_BUDGET_SHA256,
        "producer_identity": dict(producer),
        "producer_attempt_plan_sha256": started["plan_sha256"],
        "execution_plan_sha256": embedded_plan_sha256,
        "causal_prefix_equivalence_proof": proof_identity,
    }


def validate_causal_fixture_manifest(
    fixture_root: Path, fixture_manifest_sha256: str
) -> dict[str, Any]:
    """Validate authoritative CAS plus exact v2 causal and producer closure."""

    manifest_sha256 = _require_sha256(fixture_manifest_sha256, "fixture_manifest_sha256")
    try:
        fixture = validate_full_fixture_manifest(
            fixture_root,
            fixture_manifest_sha256=manifest_sha256,
            verify_objects=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("authoritative causal full-fixture validation failed") from exc
    if not (
        fixture.get("schema_version") == "codec-evidence-full-fixture-manifest-v1"
        and fixture.get("fixture_id") == EXPECTED_FIXTURE_ID
        and fixture.get("complete") is True
        and fixture.get("coverage_kind") == "FULL_EPISODE_SET"
        and fixture.get("expected_episode_count") == EXPECTED_EPISODES
        and fixture.get("covered_episode_count") == EXPECTED_EPISODES
        and fixture.get("camera_id") == "cam_left_high"
        and fixture.get("source_kind") == "NATIVE_CANONICAL_DECODED_DERIVED"
    ):
        raise RuntimeError("causal fixture top-level closure is not exact")

    try:
        contract = validate_codec_contract(fixture.get("codec_contract"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("causal fixture global codec contract is invalid") from exc
    if not (
        contract["schema_version"] == CODEC_CONTRACT_SCHEMA_VERSION
        and CODEC_CONTRACT_SCHEMA_VERSION == "codec-evidence-global-contract-v2"
        and contract["fixture_mode"] == PRODUCTION_FIXTURE_MODE
        and contract["rc_lookahead"] == 0
        and contract["mbtree"] == 0
        and contract["gop_size"] == 30
        and contract["keyint"] == 30
        and contract["min_keyint"] == 16
        and contract["bframes"] == 0
        and contract["reference_frames"] == 1
    ):
        raise RuntimeError("causal fixture encoder contract drifted")
    contract_sha256 = codec_contract_sha256(contract)
    if fixture.get("codec_contract_sha256") != contract_sha256:
        raise RuntimeError("causal fixture codec contract SHA-256 mismatch")

    covered_uids = fixture.get("covered_episode_uids")
    objects = fixture.get("objects")
    if (
        not isinstance(covered_uids, list)
        or len(covered_uids) != EXPECTED_EPISODES
        or any(
            not isinstance(uid, str) or _SHA256_RE.fullmatch(uid) is None for uid in covered_uids
        )
        or len(set(covered_uids)) != EXPECTED_EPISODES
        or not isinstance(objects, list)
        or len(objects) != EXPECTED_EPISODES
    ):
        raise RuntimeError("causal fixture 806-UID/object closure is malformed")
    observed_uid_digest = _canonical_json_sha256(sorted(covered_uids))
    if (
        fixture.get("covered_episode_uid_digest") != observed_uid_digest
        or observed_uid_digest != EXPECTED_EPISODE_UID_DIGEST
    ):
        raise RuntimeError("causal fixture episode UID digest drifted")
    object_uids = [item.get("episode_uid") for item in objects if isinstance(item, Mapping)]
    if len(object_uids) != EXPECTED_EPISODES or set(object_uids) != set(covered_uids):
        raise RuntimeError("causal fixture objects differ from the covered UID set")
    if any(type(item.get("num_frames")) is not int for item in objects):
        raise RuntimeError("causal fixture object frame counts are unavailable")
    if sum(item["num_frames"] for item in objects) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError("causal fixture total frame closure drifted")
    producer_identity = validate_producer_and_proof(fixture_root, fixture)
    return {
        "fixture_manifest_sha256": manifest_sha256,
        "codec_contract": contract,
        "codec_contract_sha256": contract_sha256,
        "covered_episode_uid_digest": observed_uid_digest,
        "covered_episode_count": EXPECTED_EPISODES,
        "total_frames": EXPECTED_TOTAL_FRAMES,
        **producer_identity,
    }


def validate_inputs(
    fixture_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_sha256 = _require_sha256(fixture_manifest_sha256, "fixture_manifest_sha256")
    manifest_path = NATIVE_PREPARATION / "DATASET_MANIFEST.json"
    closure_path = NATIVE_PREPARATION / "CPU_MIXTURE_CLOSURE.json"
    fixture_path = FIXTURE_ROOT / "fixture_manifest.json"
    expected_hashes = {
        manifest_path: NATIVE_MANIFEST_SHA256,
        closure_path: NATIVE_CLOSURE_SHA256,
        fixture_path: manifest_sha256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"input identity drift: {path}: {actual} != {expected}")

    native_manifest = load_json(manifest_path)
    datasets = native_manifest.get("datasets")
    if (
        native_manifest.get("schema_version") != "gr00t-finetune-dataset-manifest-v1"
        or not isinstance(datasets, list)
        or len(datasets) != len(SOURCE_IDS)
    ):
        raise RuntimeError("native dataset manifest must contain exactly five sources")
    observed_ids = []
    for entry in datasets:
        adapter_path = Path(entry["adapter_config_path"])
        adapter = load_json(adapter_path)
        if "codec_evidence" in adapter:
            raise RuntimeError("native adapter unexpectedly already contains codec_evidence")
        source_id = adapter.get("source_id")
        observed_ids.append(source_id)
        if NATIVE_ADAPTER_SHA256.get(source_id) != sha256_file(adapter_path):
            raise RuntimeError(f"native adapter identity drift: {adapter_path}")
        view = Path(entry["dataset_path"])
        for name in ("data", "videos"):
            payload = view / name
            if not payload.is_symlink():
                raise RuntimeError(f"native loader-view payload must remain a symlink: {payload}")
            expected_target = (Path(adapter["canonical_root"]) / name).resolve(strict=True)
            if payload.resolve(strict=True) != expected_target:
                raise RuntimeError(f"native loader-view payload target drift: {payload}")
    if tuple(observed_ids) != SOURCE_IDS:
        raise RuntimeError(f"native source order drift: {observed_ids}")
    if abs(sum(float(entry["mix_ratio"]) for entry in datasets) - 1.0) > 1e-9:
        raise RuntimeError("native mix ratios no longer sum to one")

    closure = load_json(closure_path)
    if closure.get("status") != "PASS" or closure.get("total_episodes") != EXPECTED_EPISODES:
        raise RuntimeError("native CPU mixture closure is not the accepted 806-episode closure")
    load_json(fixture_path)
    fixture_identity = validate_causal_fixture_manifest(FIXTURE_ROOT, manifest_sha256)
    return native_manifest, fixture_identity


def resolved_plan(output_root: Path, fixture_manifest_sha256: str) -> dict[str, Any]:
    native_manifest, fixture_identity = validate_inputs(fixture_manifest_sha256)
    if sha256_file(PREFIX_FACTORY_PATH) != PREFIX_FACTORY_SHA256:
        raise RuntimeError("causal prefix factory identity drifted")
    if sha256_file(PRODUCTION_MODULE_PATH) != PRODUCTION_MODULE_SHA256:
        raise RuntimeError("causal production module identity drifted")
    evidence = codec_evidence_config(fixture_manifest_sha256)
    datasets = []
    for source_id, entry in zip(SOURCE_IDS, native_manifest["datasets"]):
        datasets.append(
            {
                "source_id": source_id,
                "dataset_path": str(Path(entry["dataset_path"]).resolve(strict=True)),
                "mix_ratio": float(entry["mix_ratio"]),
                "native_adapter_config_path": str(
                    Path(entry["adapter_config_path"]).resolve(strict=True)
                ),
                "derived_adapter_config_path": str(
                    (output_root / "sources" / source_id / "ADAPTER_CONFIG.json").resolve()
                ),
            }
        )
    return {
        "schema_version": PREPARATION_PLAN_SCHEMA_VERSION,
        "output_root": str(output_root.resolve()),
        "output_must_not_exist": True,
        "native_preparation": str(NATIVE_PREPARATION),
        "native_manifest_sha256": NATIVE_MANIFEST_SHA256,
        "native_cpu_mixture_closure_sha256": NATIVE_CLOSURE_SHA256,
        "native_adapter_sha256": NATIVE_ADAPTER_SHA256,
        "fixture_root": str(FIXTURE_ROOT),
        "online_prefix_factory": {
            "path": str(PREFIX_FACTORY_PATH.resolve(strict=True)),
            "sha256": PREFIX_FACTORY_SHA256,
            "production_module_path": str(PRODUCTION_MODULE_PATH.resolve(strict=True)),
            "production_module_sha256": PRODUCTION_MODULE_SHA256,
            "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
            "execution_location": "HOST_POLICY_SERVER_EXACT_TRAINING_TOOLCHAIN",
        },
        **fixture_identity,
        "codec_evidence": evidence,
        "datasets": datasets,
    }


def materialize(output_root: Path, fixture_manifest_sha256: str) -> dict[str, Any]:
    plan = resolved_plan(output_root, fixture_manifest_sha256)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"causal C5 preparation root already exists: {output_root}")
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"causal C5 preparation parent is absent: {output_root.parent}")
    output_root.mkdir()
    (output_root / "sources").mkdir()

    manifest_entries = []
    adapter_hashes = {}
    evidence = codec_evidence_config(fixture_manifest_sha256)
    for entry in plan["datasets"]:
        source_root = output_root / "sources" / entry["source_id"]
        source_root.mkdir()
        native_adapter = load_json(Path(entry["native_adapter_config_path"]))
        derived_adapter = {**native_adapter, "codec_evidence": evidence}
        adapter_path = source_root / "ADAPTER_CONFIG.json"
        write_json_new(adapter_path, derived_adapter)
        adapter_hashes[entry["source_id"]] = sha256_file(adapter_path)
        manifest_entries.append(
            {
                "dataset_path": entry["dataset_path"],
                "mix_ratio": entry["mix_ratio"],
                "adapter_config_path": str(adapter_path.resolve()),
            }
        )

    manifest = {
        "schema_version": "gr00t-finetune-dataset-manifest-v1",
        "datasets": manifest_entries,
    }
    manifest_path = output_root / "DATASET_MANIFEST.json"
    write_json_new(manifest_path, manifest)
    summary = {
        "schema_version": PREPARATION_SUMMARY_SCHEMA_VERSION,
        "status": "PASS",
        "plan": plan,
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "input_fixture_manifest_sha256": fixture_manifest_sha256,
        "adapter_config_sha256": adapter_hashes,
        "source_count": len(manifest_entries),
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_TOTAL_FRAMES,
        "total_h40_anchors": EXPECTED_H40_ANCHORS,
        "payload_policy": (
            "native loader views and causal full fixture are immutable read-only references"
        ),
    }
    write_json_new(output_root / "PREPARATION_SUMMARY.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--fixture-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    manifest_sha256 = _require_sha256(args.fixture_manifest_sha256, "fixture_manifest_sha256")
    result = (
        resolved_plan(output_root, manifest_sha256)
        if args.dry_run
        else materialize(output_root, manifest_sha256)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
