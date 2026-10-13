from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import gr00t.experimental.mikasa_history_hard_order_runtime as runtime
import numpy as np
import pytest


def _synthetic_data() -> runtime.CachedOrderData:
    return runtime.CachedOrderData(
        arrays={
            "temporal_grid": np.zeros((48, 130, 2, 8, 8, 9), np.float32),
            "current_rgb_grid": np.zeros((48, 2, 8, 8, 3), np.float32),
            "current_proprio": np.zeros((48, 7), np.float32),
        },
        labels=np.tile(np.asarray([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]), (8, 1)),
        groups=np.repeat(np.arange(8), 6),
        splits=("train",) * 36 + ("dev",) * 12,
        sample_ids=tuple(f"sample-{index}" for index in range(48)),
    )


def test_frozen_evidence_and_receipt_discovery_bind_all_48_rows() -> None:
    evidence = runtime.validate_frozen_evidence(include_target=False)
    assert evidence["target_manifest_opened"] is False
    assert evidence["fresh_data_opened"] is False
    sources = runtime.HistoryHardSources.discover()
    assert len(sources.rows) == 48
    assert [row.row_id for row in sources.rows] == list(range(48))
    assert len({row.model.sha256 for row in sources.rows}) == 48
    assert all(row.model.path.suffix == ".npz" for row in sources.rows)


def test_target_blind_cache_source_does_not_open_target_manifest(monkeypatch) -> None:
    calls = []
    original = runtime._bound_json

    def guarded(path, *args, **kwargs):
        calls.append(Path(path))
        if Path(path) == runtime.TARGET_MANIFEST:
            raise AssertionError("target manifest opened during target-blind validation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runtime, "_bound_json", guarded)
    runtime.validate_frozen_evidence(include_target=False)
    assert runtime.TARGET_MANIFEST not in calls


def test_material_tamper_and_symlink_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "material.bin"
    path.write_bytes(b"frozen")
    path.chmod(0o444)
    identity = runtime.MaterialIdentity(path, 6, runtime.sha256_file(path))
    identity.verify(root=tmp_path)
    with pytest.raises(runtime.RuntimeContractError, match="identity"):
        replace(identity, sha256="0" * 64).verify(root=tmp_path)
    link = tmp_path / "link.bin"
    link.symlink_to(path)
    with pytest.raises(runtime.RuntimeContractError, match="regular"):
        runtime.MaterialIdentity(link, 6, identity.sha256).verify(root=tmp_path)


def test_cache_arrays_have_no_label_or_identity_tensor() -> None:
    data = _synthetic_data()
    runtime.validate_cache_arrays(data.arrays)
    assert set(data.arrays) == set(runtime.CACHE_FIELDS)
    encoded = json.dumps(sorted(data.arrays))
    for forbidden in ("label", "path", "pair", "group", "split", "row", "filename", "target"):
        assert forbidden not in encoded


def test_train_dev_batches_and_fresh_groups_fail_closed() -> None:
    data = _synthetic_data()
    batch, labels = data.batch_for_groups((0, 1, 2, 3))
    assert batch.temporal_grid.shape == (24, 130, 2, 8, 8, 9)
    assert labels.shape == (24, 2)
    dev, _ = data.batch_for_groups((6, 7))
    assert dev.batch_size == 12
    with pytest.raises(PermissionError, match="fresh"):
        data.batch_for_groups((8,))


def test_group_balanced_schedule_is_deterministic_and_balanced() -> None:
    first = runtime.GroupBalancedSchedule(230923)
    second = runtime.GroupBalancedSchedule(230923)
    batches = [first.batch(step) for step in range(15)]
    assert batches == [second.batch(step) for step in range(15)]
    assert len(set(batches)) == 15
    counts = {group: sum(group in batch for batch in batches) for group in runtime.TRAIN_GROUPS}
    assert set(counts.values()) == {10}
    assert all(len(batch) == 4 and len(set(batch)) == 4 for batch in batches)


def test_write_once_rejects_second_attempt(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    identity = runtime.write_once(path, b"{}\n")
    assert identity["mode"] == "0444"
    with pytest.raises(FileExistsError):
        runtime.write_once(path, b"changed\n")
