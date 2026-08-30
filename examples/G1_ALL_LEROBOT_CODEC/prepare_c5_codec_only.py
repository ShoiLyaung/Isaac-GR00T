#!/usr/bin/env python3
"""Materialize the tiny, immutable C5 adapter/manifest overlay.

The native five-source loader views and the full-R2 fixture remain read-only.
Only five derived adapter JSON files plus one training manifest are created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any


NATIVE_PREPARATION = Path(
    "/home/sl/icra27-runs/groot-n17-all-lerobot-20k-r1/prep-20260827T171352+0800"
)
NATIVE_MANIFEST_SHA256 = "57769ef4469e3f094240299ae4f2c5935f7c9c452a06ff2349122f509776bdbe"
NATIVE_CLOSURE_SHA256 = "3fe4d1625920032d5242bcfec298552314f2b9cac3917db4cdf734d2cfd51974"
FIXTURE_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-full-r2")
FIXTURE_MANIFEST_SHA256 = "c05736e1e33745ee0580d45aeebf651891a6533701297f23f41a1bd5090c3a88"
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
CODEC_EVIDENCE = {
    "enabled": True,
    "fixture_root": str(FIXTURE_ROOT),
    "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
    "modalities": ["mv", "rpe", "i_frame", "metadata"],
    "gate_init": 0.0,
    "schema_version": "codec-evidence-episode-bundle-v1",
    "camera_id": "cam_left_high",
    "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
    "d_codec": 64,
    "conditioner_heads": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = NATIVE_PREPARATION / "DATASET_MANIFEST.json"
    closure_path = NATIVE_PREPARATION / "CPU_MIXTURE_CLOSURE.json"
    fixture_path = FIXTURE_ROOT / "fixture_manifest.json"
    expected_hashes = {
        manifest_path: NATIVE_MANIFEST_SHA256,
        closure_path: NATIVE_CLOSURE_SHA256,
        fixture_path: FIXTURE_MANIFEST_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"input identity drift: {path}: {actual} != {expected}")

    native_manifest = load_json(manifest_path)
    if native_manifest.get("schema_version") != "gr00t-finetune-dataset-manifest-v1":
        raise RuntimeError("unsupported native dataset manifest")
    datasets = native_manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != len(SOURCE_IDS):
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
    if closure.get("status") != "PASS" or closure.get("total_episodes") != 806:
        raise RuntimeError("native CPU mixture closure is not the accepted 806-episode closure")
    fixture = load_json(fixture_path)
    if not (
        fixture.get("complete") is True
        and fixture.get("coverage_kind") == "FULL_EPISODE_SET"
        and fixture.get("covered_episode_count") == 806
        and fixture.get("expected_episode_count") == 806
        and fixture.get("camera_id") == "cam_left_high"
        and fixture.get("source_kind") == "NATIVE_CANONICAL_DECODED_DERIVED"
        and len(fixture.get("objects", [])) == 806
    ):
        raise RuntimeError("full-R2 fixture manifest is not the accepted exact coverage")
    return native_manifest, fixture


def resolved_plan(output_root: Path) -> dict[str, Any]:
    native_manifest, fixture = validate_inputs()
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
        "schema_version": "g1-all-lerobot-c5-codec-only-preparation-plan-v1",
        "output_root": str(output_root.resolve()),
        "output_must_not_exist": True,
        "native_preparation": str(NATIVE_PREPARATION),
        "native_manifest_sha256": NATIVE_MANIFEST_SHA256,
        "native_cpu_mixture_closure_sha256": NATIVE_CLOSURE_SHA256,
        "native_adapter_sha256": NATIVE_ADAPTER_SHA256,
        "fixture_root": str(FIXTURE_ROOT),
        "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "fixture_episode_uid_digest": fixture["covered_episode_uid_digest"],
        "codec_evidence": CODEC_EVIDENCE,
        "datasets": datasets,
    }


def materialize(output_root: Path) -> dict[str, Any]:
    plan = resolved_plan(output_root)
    if output_root.exists():
        raise FileExistsError(f"C5 preparation root already exists: {output_root}")
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"C5 preparation parent is absent: {output_root.parent}")
    output_root.mkdir()
    (output_root / "sources").mkdir()

    manifest_entries = []
    adapter_hashes = {}
    for entry in plan["datasets"]:
        source_root = output_root / "sources" / entry["source_id"]
        source_root.mkdir()
        native_adapter = load_json(Path(entry["native_adapter_config_path"]))
        derived_adapter = {**native_adapter, "codec_evidence": dict(CODEC_EVIDENCE)}
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
        "schema_version": "g1-all-lerobot-c5-codec-only-preparation-summary-v1",
        "status": "PASS",
        "plan": plan,
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "adapter_config_sha256": adapter_hashes,
        "source_count": len(manifest_entries),
        "total_episodes": 806,
        "total_h40_anchors": 265906,
        "payload_policy": "native loader views and full-R2 fixture are read-only references",
    }
    write_json_new(output_root / "PREPARATION_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    result = resolved_plan(output_root) if args.dry_run else materialize(output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
