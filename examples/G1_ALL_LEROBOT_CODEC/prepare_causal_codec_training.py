#!/usr/bin/env python3
"""Create tiny adapter overlays for the active causal Codec training path."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "g1-all-lerobot-main-causal-codec-preparation-v1"
DATASET_MANIFEST_SCHEMA_VERSION = "gr00t-finetune-dataset-manifest-v1"
ADAPTER_KIND = "g1_all_lerobot_h40_v1"
SOURCE_KIND = "NATIVE_CANONICAL_DECODED_DERIVED"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def resolve_plan(
    baseline_manifest: Path,
    output_root: Path,
    fixture_root: Path,
    fixture_manifest_sha256: str,
) -> dict[str, Any]:
    baseline_manifest = baseline_manifest.resolve(strict=True)
    fixture_root = fixture_root.resolve(strict=True)
    fixture_manifest = fixture_root / "fixture_manifest.json"
    if not fixture_manifest.is_file() or fixture_manifest.is_symlink():
        raise ValueError("fixture_manifest.json must be a regular file")
    if _sha256(fixture_manifest) != fixture_manifest_sha256:
        raise ValueError("fixture_manifest.json SHA-256 mismatch")
    manifest = _load_json(baseline_manifest)
    if set(manifest) != {"schema_version", "datasets"}:
        raise ValueError("Baseline manifest fields differ from the training contract")
    if manifest["schema_version"] != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Baseline dataset manifest schema mismatch")
    entries = manifest["datasets"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Baseline dataset manifest has no datasets")

    resolved_entries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "dataset_path",
            "mix_ratio",
            "adapter_config_path",
        }:
            raise ValueError(f"Baseline dataset entry {index} is malformed")
        dataset_path = Path(entry["dataset_path"]).resolve(strict=True)
        adapter_path = Path(entry["adapter_config_path"]).resolve(strict=True)
        adapter = _load_json(adapter_path)
        if adapter.get("adapter_kind") != ADAPTER_KIND:
            raise ValueError(f"Baseline dataset entry {index} is not {ADAPTER_KIND}")
        source_id = adapter.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Baseline dataset entry {index} has no source_id")
        resolved_entries.append(
            {
                "source_id": source_id,
                "dataset_path": str(dataset_path),
                "mix_ratio": entry["mix_ratio"],
                "baseline_adapter": str(adapter_path),
                "output_adapter": str((output_root / f"{source_id}.adapter.json").absolute()),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_manifest": str(baseline_manifest),
        "fixture_root": str(fixture_root),
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "output_root": str(output_root.absolute()),
        "datasets": resolved_entries,
    }


def materialize(plan: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(plan["output_root"])
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    codec_block = {
        "enabled": True,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": SOURCE_KIND,
        "d_codec": 64,
        "conditioner_heads": 8,
        "fixture_root": plan["fixture_root"],
        "fixture_manifest_sha256": plan["fixture_manifest_sha256"],
    }
    entries = []
    for entry in plan["datasets"]:
        adapter = _load_json(Path(entry["baseline_adapter"]))
        if "codec_evidence" in adapter:
            raise ValueError("Baseline adapter already contains codec_evidence")
        adapter["codec_evidence"] = codec_block
        adapter_path = Path(entry["output_adapter"])
        _write_new(adapter_path, adapter)
        entries.append(
            {
                "dataset_path": entry["dataset_path"],
                "mix_ratio": entry["mix_ratio"],
                "adapter_config_path": str(adapter_path),
            }
        )
    output_manifest = output_root / "DATASET_MANIFEST.json"
    _write_new(
        output_manifest,
        {"schema_version": DATASET_MANIFEST_SCHEMA_VERSION, "datasets": entries},
    )
    result = {**plan, "dataset_manifest": str(output_manifest), "status": "READY"}
    _write_new(output_root / "PREPARATION_SUMMARY.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--fixture-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = resolve_plan(
        args.baseline_manifest,
        args.output_root,
        args.fixture_root,
        args.fixture_manifest_sha256,
    )
    result = plan if args.dry_run else materialize(plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
