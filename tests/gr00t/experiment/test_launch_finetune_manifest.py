from __future__ import annotations

import json
from pathlib import Path

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.launch_finetune import DATASET_MANIFEST_SCHEMA_VERSION, resolve_dataset_specs
import pytest


def _config(**overrides) -> FinetuneConfig:
    values = {
        "base_model_path": "/checkpoint",
        "embodiment_tag": "NEW_EMBODIMENT",
    }
    values.update(overrides)
    return FinetuneConfig(**values)


def _write_manifest(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema_version": DATASET_MANIFEST_SCHEMA_VERSION, "datasets": entries}) + "\n"
    )
    return path


def _entry(tmp_path: Path, name: str, weight: float) -> dict:
    dataset = tmp_path / name
    dataset.mkdir()
    adapter = tmp_path / f"{name}.adapter.json"
    adapter.write_text("{}\n")
    return {
        "dataset_path": str(dataset),
        "mix_ratio": weight,
        "adapter_config_path": str(adapter),
    }


def test_valid_multi_dataset_manifest_resolves_in_order(tmp_path: Path):
    entries = [_entry(tmp_path, "one", 0.25), _entry(tmp_path, "two", 0.75)]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    specs = resolve_dataset_specs(
        _config(
            dataset_manifest_path=str(manifest),
            dataset_adapter="g1_all_lerobot_h40_v1",
        ),
        "new_embodiment",
    )
    assert [spec["dataset_paths"] for spec in specs] == [
        [str((tmp_path / "one").resolve())],
        [str((tmp_path / "two").resolve())],
    ]
    assert [spec["mix_ratio"] for spec in specs] == [0.25, 0.75]
    assert [spec["adapter_config_path"] for spec in specs] == [
        str((tmp_path / "one.adapter.json").resolve()),
        str((tmp_path / "two.adapter.json").resolve()),
    ]
    assert all(spec["embodiment_tag"] == "new_embodiment" for spec in specs)
    assert all(spec["dataset_adapter"] == "g1_all_lerobot_h40_v1" for spec in specs)


def test_dataset_source_requires_exactly_one_mode(tmp_path: Path):
    dataset = tmp_path / "legacy"
    dataset.mkdir()
    manifest = _write_manifest(tmp_path / "manifest.json", [_entry(tmp_path, "one", 1.0)])
    with pytest.raises(ValueError, match="Exactly one"):
        resolve_dataset_specs(_config(), "new_embodiment")
    with pytest.raises(ValueError, match="Exactly one"):
        resolve_dataset_specs(
            _config(dataset_path=str(dataset), dataset_manifest_path=str(manifest)),
            "new_embodiment",
        )
    legacy = resolve_dataset_specs(
        _config(dataset_path=str(dataset), dataset_adapter="adapter"), "new_embodiment"
    )
    assert legacy[0]["dataset_paths"] == [str(dataset)]
    assert legacy[0]["mix_ratio"] == 1.0


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf"), True])
def test_manifest_rejects_non_positive_or_non_finite_weight(tmp_path: Path, weight):
    manifest = _write_manifest(tmp_path / "manifest.json", [_entry(tmp_path, "one", weight)])
    with pytest.raises(ValueError, match="mix_ratio"):
        resolve_dataset_specs(_config(dataset_manifest_path=str(manifest)), "new_embodiment")


def test_manifest_rejects_duplicate_resolved_paths(tmp_path: Path):
    first = _entry(tmp_path, "one", 0.5)
    second_adapter = tmp_path / "two.adapter.json"
    second_adapter.write_text("{}\n")
    second = {
        "dataset_path": str(tmp_path / "one" / ".." / "one"),
        "mix_ratio": 0.5,
        "adapter_config_path": str(second_adapter),
    }
    manifest = _write_manifest(tmp_path / "manifest.json", [first, second])
    with pytest.raises(ValueError, match="Duplicate dataset root"):
        resolve_dataset_specs(_config(dataset_manifest_path=str(manifest)), "new_embodiment")


def test_manifest_schema_is_fail_closed(tmp_path: Path):
    entry = _entry(tmp_path, "one", 1.0)
    bad_values = [
        {"schema_version": "wrong", "datasets": [entry]},
        {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "datasets": [entry],
            "extra": True,
        },
        {"schema_version": DATASET_MANIFEST_SCHEMA_VERSION, "datasets": []},
        {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "datasets": [{**entry, "extra": True}],
        },
    ]
    for index, value in enumerate(bad_values):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(json.dumps(value) + "\n")
        with pytest.raises(ValueError):
            resolve_dataset_specs(_config(dataset_manifest_path=str(path)), "new_embodiment")
