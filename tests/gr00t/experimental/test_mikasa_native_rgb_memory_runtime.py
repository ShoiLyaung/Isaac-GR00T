from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gr00t.experimental.mikasa_native_rgb_memory_runtime as runtime
from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    DATA_SCHEMA_SHA256,
    MODEL_INPUT_KEYS,
    PRIVILEGED_TARGET_KEYS,
    SHARD_SCHEMA,
    FeatureCacheCorpus,
    GroupArrays,
    GroupBalancedSchedule,
    ShardedMemoryCorpus,
    TargetStats,
    _block_mean_16,
    build_feature_cache,
    group_equal_mean,
    physical_metrics,
    seed_equal_mean,
    write_new_json,
)
import numpy as np
import pytest
import torch


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_manifest_group(root: Path, split: str, group_id: int) -> Path:
    directory = root / split / f"group_{group_id:03d}"
    directory.mkdir(parents=True)
    model = b"model"
    target = b"target"
    (directory / "model_inputs.npz").write_bytes(model)
    (directory / "privileged_targets.npz").write_bytes(target)
    manifest = {
        "schema": SHARD_SCHEMA,
        "version": 1,
        "frozen_cohort_sha256": runtime.COHORT_SHA256,
        "data_schema_sha256": DATA_SCHEMA_SHA256,
        "group_id": group_id,
        "split": split,
        "ordered_pair_rows": [list(pair) for pair in runtime.ORDERED_SHAPE_IDS],
        "model_inputs": {
            "name": "model_inputs.npz",
            "bytes": len(model),
            "sha256": _sha(model),
        },
        "privileged_targets": {
            "name": "privileged_targets.npz",
            "bytes": len(target),
            "sha256": _sha(target),
        },
        "model_input_keys": sorted(MODEL_INPUT_KEYS),
        "privileged_target_keys": sorted(PRIVILEGED_TARGET_KEYS),
        "camera_order": ["base_camera", "hand_camera"],
        "cue_elapsed_inclusive": [1, 130],
        "action_onset_elapsed": 131,
        "post_capture_elapsed": 132,
        "sixpack": {"gates": {"all_exact": True}},
        "sentinel_repeat": (
            {
                "performed": True,
                "all_exact": True,
                "sample_hash_parity": {"01": True},
                "repeat_sixpack_gates": {"all_exact": True},
            }
            if group_id == runtime.EXPECTED_SENTINEL_GROUPS[split]
            else None
        ),
    }
    path = directory / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_first_loader_resolves_exact_regular_materials(tmp_path: Path) -> None:
    for group_id in runtime.EXPECTED_SPLIT_GROUPS["train"]:
        _write_manifest_group(tmp_path, "train", group_id)
    corpus = ShardedMemoryCorpus.discover(tmp_path, "train")
    assert corpus.group_ids == runtime.EXPECTED_SPLIT_GROUPS["train"]
    shard = corpus.shards[7]
    assert shard.model_input.path.name == "model_inputs.npz"
    assert shard.privileged_target.path.name == "privileged_targets.npz"
    shard.model_input.verify()
    shard.privileged_target.verify()


def test_model_only_discovery_never_opens_privileged_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for group_id in runtime.EXPECTED_SPLIT_GROUPS["train"]:
        _write_manifest_group(tmp_path, "train", group_id)
    opened_targets = []
    original_open = Path.open

    def audited_open(self, *args, **kwargs):
        if self.name == "privileged_targets.npz":
            opened_targets.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    corpus = ShardedMemoryCorpus.discover(
        tmp_path,
        "train",
        verify_privileged_targets=False,
    )
    assert corpus.group_ids == runtime.EXPECTED_SPLIT_GROUPS["train"]
    assert opened_targets == []


def test_real_manifest_cache_build_never_opens_privileged_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for group_id in runtime.EXPECTED_SPLIT_GROUPS["train"]:
        _write_manifest_group(tmp_path / "corpus", "train", group_id)
    scalar_rgb = np.zeros((), dtype=np.uint8)
    model = {
        "cue_base_rgb": np.broadcast_to(scalar_rgb, (6, 130, 128, 128, 3)),
        "cue_hand_rgb": np.broadcast_to(scalar_rgb, (6, 130, 128, 128, 3)),
        "cue_proprio": np.zeros((6, 130, 7), dtype=np.float32),
        "cue_elapsed_steps": np.tile(np.arange(1, 131, dtype=np.int64), (6, 1)),
        "cue_valid_mask": np.ones((6, 130), dtype=np.bool_),
        "post_base_rgb": np.broadcast_to(scalar_rgb, (6, 128, 128, 3)),
        "post_hand_rgb": np.broadcast_to(scalar_rgb, (6, 128, 128, 3)),
        "post_proprio": np.zeros((6, 7), dtype=np.float32),
        "post_elapsed_steps": np.full(6, 132, dtype=np.int64),
        "post_valid_mask": np.ones(6, dtype=np.bool_),
    }
    features = {
        "temporal_grid": np.zeros((6, 130, 2, 8, 8, 9), dtype=np.float32),
        "post_rgb_grid": np.zeros((6, 2, 8, 8, 3), dtype=np.float32),
        "post_proprio": np.zeros((6, 7), dtype=np.float32),
    }
    privileged_opens = []

    def synthetic_npz(material, expected_keys):
        material.verify()
        if expected_keys == PRIVILEGED_TARGET_KEYS:
            privileged_opens.append(material.path)
            raise AssertionError("privileged target was opened")
        assert expected_keys == MODEL_INPUT_KEYS
        return model

    monkeypatch.setattr(runtime, "_load_npz_exact", synthetic_npz)
    monkeypatch.setattr(runtime, "build_native_rgb_features", lambda value: features)
    corpus = ShardedMemoryCorpus.discover(
        tmp_path / "corpus",
        "train",
        verify_privileged_targets=False,
    )
    records = build_feature_cache(corpus, tmp_path / "cache")
    assert len(records) == 24
    assert privileged_opens == []


def test_loader_rejects_failed_sixpack_and_symlink_split(tmp_path: Path) -> None:
    for group_id in runtime.EXPECTED_SPLIT_GROUPS["train"]:
        _write_manifest_group(tmp_path, "train", group_id)
    manifest_path = tmp_path / "train/group_003/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sixpack"]["gates"]["all_exact"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sixpack gates"):
        ShardedMemoryCorpus.discover(tmp_path, "train")

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    (linked_root / "train").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        ShardedMemoryCorpus.discover(linked_root, "train")


def test_manifest_loader_rejects_path_escape_and_extra_files(tmp_path: Path) -> None:
    manifest_path = _write_manifest_group(tmp_path, "train", 7)
    manifest = json.loads(manifest_path.read_text())
    manifest["model_inputs"]["name"] = "../model_inputs.npz"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="basename"):
        ShardedMemoryCorpus.discover(tmp_path, "train")

    other = tmp_path / "other"
    _write_manifest_group(other, "train", 7)
    (other / "train/group_007/leak.npy").write_bytes(b"leak")
    with pytest.raises(ValueError, match="file set"):
        ShardedMemoryCorpus.discover(other, "train")


def test_confirmation_is_rejected_before_any_root_access(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def forbidden_resolve(self):
        calls.append(self)
        raise AssertionError("confirmation root was touched")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    with pytest.raises(PermissionError, match="sealed"):
        ShardedMemoryCorpus.discover(tmp_path / "confirmation-secret", "confirmation")
    assert calls == []
    with pytest.raises(PermissionError, match="forbidden"):
        FeatureCacheCorpus.discover(tmp_path, "confirmation")


def test_block_mean_is_exact_and_target_blind() -> None:
    value = np.zeros((2, 128, 128, 3), dtype=np.uint8)
    value[0, :16, :16, 0] = 255
    value[1, :, :, :] = 128
    grid = _block_mean_16(value)
    assert grid.shape == (2, 8, 8, 3)
    assert grid.dtype == np.float32
    assert grid[0, 0, 0, 0] == 1.0
    assert np.count_nonzero(grid[0]) == 1
    np.testing.assert_allclose(grid[1], np.float32(128 / 255), rtol=0, atol=1e-7)


def test_feature_cache_builder_never_requests_targets(tmp_path: Path, monkeypatch) -> None:
    temporal = np.zeros((6, 130, 2, 8, 8, 9), dtype=np.float32)
    post = np.zeros((6, 2, 8, 8, 3), dtype=np.float32)
    features = {
        "temporal_grid": temporal,
        "post_rgb_grid": post,
        "post_proprio": np.zeros((6, 7), dtype=np.float32),
    }

    class FakeCorpus:
        split = "train"
        group_ids = (0,)
        _by_group = {0: SimpleNamespace(model_input=SimpleNamespace(sha256="a" * 64))}

        def load_model_group(self, group_id):
            assert group_id == 0
            return {key: object() for key in MODEL_INPUT_KEYS}

        def load_group(self, group_id):
            raise AssertionError(f"privileged target opened for group {group_id}")

    monkeypatch.setattr(runtime, "build_native_rgb_features", lambda model: features)
    records = build_feature_cache(FakeCorpus(), tmp_path / "cache")
    assert len(records) == 1
    manifest = json.loads(
        (tmp_path / "cache/train/group_000/MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["targets_opened"] is False
    assert manifest["feature_keys"] == sorted(runtime.FEATURE_KEYS)


def test_group_balanced_schedule_has_four_complete_sixpacks() -> None:
    schedule = GroupBalancedSchedule(tuple(range(24)), seed=230923)
    first_epoch = [schedule.batch(step) for step in range(6)]
    assert all(len(batch) == 4 and len(set(batch)) == 4 for batch in first_epoch)
    assert sorted(group for batch in first_epoch for group in batch) == list(range(24))
    assert schedule.batch(0) == GroupBalancedSchedule(tuple(range(24)), seed=230923).batch(0)
    assert schedule.batch(6) != schedule.batch(0)


def test_feature_cache_must_bind_the_exact_source_model_shards(tmp_path: Path) -> None:
    records = {
        group_id: runtime.MaterialIdentity(tmp_path / f"{group_id}.npz", 1, "a" * 64)
        for group_id in runtime.EXPECTED_SPLIT_GROUPS["train"]
    }
    source = {group_id: "b" * 64 for group_id in records}
    cache = FeatureCacheCorpus(tmp_path, "train", records, source)
    corpus = SimpleNamespace(
        split="train",
        group_ids=runtime.EXPECTED_SPLIT_GROUPS["train"],
        _by_group={
            group_id: SimpleNamespace(model_input=SimpleNamespace(sha256="b" * 64))
            for group_id in records
        },
    )
    cache.validate_source(corpus)
    corpus._by_group[7].model_input.sha256 = "c" * 64
    with pytest.raises(ValueError, match="source identity"):
        cache.validate_source(corpus)


class _StatsCorpus:
    def __init__(self, split: str):
        self.split = split

    def iter_target_groups(self):
        base = np.arange(6 * 48, dtype=np.float32).reshape(6, 2, 12, 2)
        yield GroupArrays(0, {}, {"ordered_checkpoints": base})
        yield GroupArrays(1, {}, {"ordered_checkpoints": base + 1.0})


def test_target_stats_are_train_only_and_round_trip() -> None:
    with pytest.raises(ValueError, match="train split"):
        TargetStats.fit(_StatsCorpus("dev"))
    stats = TargetStats.fit(_StatsCorpus("train"))
    value = torch.arange(96, dtype=torch.float32).reshape(2, 48)
    torch.testing.assert_close(stats.denormalize(stats.normalize(value)), value)
    restored = TargetStats.from_json(stats.as_json())
    np.testing.assert_array_equal(restored.mean, stats.mean)
    np.testing.assert_array_equal(restored.std, stats.std)


def test_physical_metrics_and_group_seed_equal_aggregation() -> None:
    stats = TargetStats(np.zeros(48, np.float32), np.ones(48, np.float32))
    target = torch.zeros(2, 48)
    prediction = target.clone()
    prediction[1] = 1.0
    candidates = torch.stack(
        (target, target + 1.0, target + 2.0, target + 3.0, target + 4.0, target + 5.0),
        dim=1,
    )
    metrics = physical_metrics(prediction, target, stats, candidates, torch.tensor([0, 0]))
    assert metrics["first_mae_m"].tolist() == [0.0, 1.0]
    assert metrics["complete_mae_m"].tolist() == [0.0, 1.0]
    assert metrics["order_correct"].tolist() == [1.0, 0.0]

    records = [
        {"group_id": 1, "metric": 1.0},
        {"group_id": 1, "metric": 3.0},
        {"group_id": 2, "metric": 8.0},
    ]
    assert group_equal_mean(records, "metric") == 5.0
    assert seed_equal_mean({2: records, 1: [{"group_id": 3, "metric": 1.0}]}, "metric") == 3.0


def test_write_once_result_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    identity = write_new_json(path, {"status": "passed"})
    assert identity["sha256"] == runtime.sha256_file(path)
    with pytest.raises(FileExistsError):
        write_new_json(path, {"status": "changed"})
