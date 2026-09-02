from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/G1_ALL_LEROBOT_CODEC/prepare_causal_codec_training.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_active_codec", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def _inputs(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    fixture_manifest = _write(fixture / "fixture_manifest.json", {"complete": True})
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    adapter = _write(
        tmp_path / "baseline.adapter.json",
        {
            "adapter_kind": prepare.ADAPTER_KIND,
            "source_id": "source-a",
        },
    )
    baseline = _write(
        tmp_path / "DATASET_MANIFEST.json",
        {
            "schema_version": prepare.DATASET_MANIFEST_SCHEMA_VERSION,
            "datasets": [
                {
                    "dataset_path": str(dataset),
                    "mix_ratio": 1.0,
                    "adapter_config_path": str(adapter),
                }
            ],
        },
    )
    return baseline, fixture, prepare._sha256(fixture_manifest)


def test_prepare_active_codec_writes_only_small_overlays(tmp_path):
    baseline, fixture, digest = _inputs(tmp_path)
    output = tmp_path / "output"
    plan = prepare.resolve_plan(baseline, output, fixture, digest)
    assert not output.exists()

    result = prepare.materialize(plan)
    assert result["status"] == "READY"
    manifest = json.loads((output / "DATASET_MANIFEST.json").read_text())
    adapter_path = Path(manifest["datasets"][0]["adapter_config_path"])
    adapter = json.loads(adapter_path.read_text())
    assert adapter["codec_evidence"]["schema_version"] == ("codec-evidence-episode-bundle-v1")
    assert adapter["codec_evidence"]["fixture_root"] == str(fixture.resolve())


def test_prepare_active_codec_rejects_hash_drift_and_overwrite(tmp_path):
    baseline, fixture, digest = _inputs(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare.resolve_plan(baseline, tmp_path / "output", fixture, "f" * 64)

    plan = prepare.resolve_plan(baseline, tmp_path / "output", fixture, digest)
    prepare.materialize(plan)
    with pytest.raises(FileExistsError):
        prepare.materialize(plan)
