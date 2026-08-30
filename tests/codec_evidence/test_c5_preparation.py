from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "examples/G1_ALL_LEROBOT_CODEC/prepare_c5_codec_only.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_c5_codec_only", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
c5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c5)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_c5_preparation_creates_only_codec_adapter_overlay(tmp_path, monkeypatch):
    native_root = tmp_path / "native-preparation"
    fixture_root = tmp_path / "fixture"
    entries = []
    for source_id in c5.SOURCE_IDS:
        source = native_root / "sources" / source_id
        view = source / "LOADER_VIEW"
        view.mkdir(parents=True)
        canonical = tmp_path / "canonical" / source_id
        (canonical / "data").mkdir(parents=True)
        (canonical / "videos").mkdir()
        (view / "data").symlink_to(canonical / "data")
        (view / "videos").symlink_to(canonical / "videos")
        adapter = source / "ADAPTER_CONFIG.json"
        _write_json(
            adapter,
            {
                "source_id": source_id,
                "baseline_id": "native",
                "canonical_root": str(canonical),
            },
        )
        entries.append(
            {
                "dataset_path": str(view),
                "mix_ratio": 0.2,
                "adapter_config_path": str(adapter),
            }
        )
    native_manifest = native_root / "DATASET_MANIFEST.json"
    native_closure = native_root / "CPU_MIXTURE_CLOSURE.json"
    fixture_manifest = fixture_root / "fixture_manifest.json"
    _write_json(
        native_manifest,
        {"schema_version": "gr00t-finetune-dataset-manifest-v1", "datasets": entries},
    )
    _write_json(native_closure, {"status": "PASS", "total_episodes": 806})
    _write_json(
        fixture_manifest,
        {
            "complete": True,
            "coverage_kind": "FULL_EPISODE_SET",
            "covered_episode_count": 806,
            "expected_episode_count": 806,
            "camera_id": "cam_left_high",
            "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
            "covered_episode_uid_digest": "a" * 64,
            "objects": [{}] * 806,
        },
    )
    monkeypatch.setattr(c5, "NATIVE_PREPARATION", native_root)
    monkeypatch.setattr(c5, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(c5, "NATIVE_MANIFEST_SHA256", c5.sha256_file(native_manifest))
    monkeypatch.setattr(c5, "NATIVE_CLOSURE_SHA256", c5.sha256_file(native_closure))
    monkeypatch.setattr(c5, "FIXTURE_MANIFEST_SHA256", c5.sha256_file(fixture_manifest))
    monkeypatch.setattr(
        c5,
        "NATIVE_ADAPTER_SHA256",
        {
            source_id: c5.sha256_file(native_root / "sources" / source_id / "ADAPTER_CONFIG.json")
            for source_id in c5.SOURCE_IDS
        },
    )
    monkeypatch.setitem(c5.CODEC_EVIDENCE, "fixture_root", str(fixture_root))
    monkeypatch.setitem(
        c5.CODEC_EVIDENCE,
        "fixture_manifest_sha256",
        c5.sha256_file(fixture_manifest),
    )

    output = tmp_path / "c5-preparation"
    summary = c5.materialize(output)

    assert summary["status"] == "PASS"
    manifest = c5.load_json(output / "DATASET_MANIFEST.json")
    assert [Path(entry["dataset_path"]) for entry in manifest["datasets"]] == [
        Path(entry["dataset_path"]).resolve() for entry in entries
    ]
    for source_id, entry in zip(c5.SOURCE_IDS, manifest["datasets"]):
        adapter = c5.load_json(Path(entry["adapter_config_path"]))
        assert adapter["source_id"] == source_id
        assert adapter["codec_evidence"] == c5.CODEC_EVIDENCE
        assert list((output / "sources" / source_id).iterdir()) == [
            output / "sources" / source_id / "ADAPTER_CONFIG.json"
        ]
    with pytest.raises(FileExistsError):
        c5.materialize(output)

    retargeted = native_root / "sources" / c5.SOURCE_IDS[0] / "LOADER_VIEW" / "data"
    retargeted.unlink()
    wrong_target = tmp_path / "wrong-data"
    wrong_target.mkdir()
    retargeted.symlink_to(wrong_target)
    with pytest.raises(RuntimeError, match="payload target drift"):
        c5.validate_inputs()
