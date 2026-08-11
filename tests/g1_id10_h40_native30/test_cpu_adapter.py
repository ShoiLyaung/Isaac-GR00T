from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTIVE_ACTION_DIM,
    ADAPTER_KIND,
    BASELINE_ID,
    CONFIG_SCHEMA_VERSION,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    G1Id10H40Native30EpisodeLoader,
    HORIZON,
    LANGUAGE,
    TEMPORAL_OPTION,
    apply_active_action_mask,
    build_metadata_loader_view,
    compute_h40_relative_statistics,
    h40_native30_offsets,
    load_adapter_config,
)


STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _stats(dim: int) -> dict[str, list[float]]:
    return {
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "min": [-1.0] * dim,
        "max": [1.0] * dim,
        "q01": [-1.0] * dim,
        "q99": [1.0] * dim,
    }


def _make_canonical(root: Path, length: int = 45) -> Path:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/chunk-000/observation.images.cam_left_high").mkdir(parents=True)
    states, actions = [], []
    for index in range(length):
        arm = np.linspace(-0.2, 0.2, 14, dtype=np.float32) + index * 0.001
        hand = np.linspace(0.1, 0.9, 12, dtype=np.float32)
        states.append(np.concatenate([arm, hand]).astype(np.float32))
        actions.append(np.concatenate([arm + 0.05, hand]).astype(np.float32))
    pd.DataFrame(
        {
            "observation.state": states,
            "action": actions,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.zeros(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            "annotation.human.action.task_description": ["insertion_shaft_main_300ep"] * length,
            "annotation.human.validity": np.ones(length, dtype=np.int64),
        }
    ).to_parquet(root / "data/chunk-000/episode_000000.parquet", index=False)
    _write_json(
        root / "meta/info.json",
        {
            "chunks_size": 1000,
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [26]},
                "action": {"dtype": "float32", "shape": [26]},
                "observation.images.cam_left_high": {"dtype": "video", "shape": [480, 640, 3]},
            },
        },
    )
    _write_json(
        root / "meta/modality.json",
        {
            "state": {
                "left_arm": {"start": 0, "end": 7, "original_key": "observation.state"},
                "right_arm": {"start": 7, "end": 14, "original_key": "observation.state"},
                "left_hand": {"start": 14, "end": 20, "original_key": "observation.state"},
                "right_hand": {"start": 20, "end": 26, "original_key": "observation.state"},
            },
            "action": {
                "left_arm": {"start": 0, "end": 7, "original_key": "action"},
                "right_arm": {"start": 7, "end": 14, "original_key": "action"},
                "left_hand": {"start": 14, "end": 20, "original_key": "action"},
                "right_hand": {"start": 20, "end": 26, "original_key": "action"},
            },
            "video": {"cam_left_high": {"original_key": "observation.images.cam_left_high"}},
            "annotation": {
                "human.action.task_description": {
                    "original_key": "annotation.human.action.task_description"
                }
            },
        },
    )
    _write_json(root / "meta/stats.json", {"observation.state": _stats(26), "action": _stats(26)})
    _write_jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "length": length, "tasks": ["slug"]}])
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "slug"}])
    return root


def _adapter_config(path: Path, stats_path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "adapter_kind": ADAPTER_KIND,
            "baseline_id": BASELINE_ID,
            "embodiment_tag": EMBODIMENT_TAG,
            "embodiment_id": EMBODIMENT_ID,
            "temporal_option": TEMPORAL_OPTION,
            "source_index_formula": "n+k",
            "row_grid_rate_hz": 30,
            "true_30hz_hardware_clock_proven": False,
            "true_50hz_capture": False,
            "horizon": HORIZON,
            "active_action_dims": ACTIVE_ACTION_DIM,
            "hand_scope": "PROJECT_INSPIRE6_EXTENSION_NO_DIRECT_OFFICIAL_MATCH",
            "language": LANGUAGE,
            "runtime_verified_fixed_state": False,
            "fixed_state": {"left_leg": [0.0] * 6, "right_leg": [0.0] * 6, "waist": [0.0] * 3},
            "fixed_action": {"waist": [0.0] * 3, "base_height_command": [0.0], "navigate_command": [0.0] * 3},
            "derived_relative_stats_path": str(stats_path),
        },
    )
    return path


def _modality(repo: Path):
    config = repo / "examples/G1_ID10_H40_NATIVE30/g1_id10_h40_native30_config.py"
    spec = importlib.util.spec_from_file_location("test_h40_native30", config)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    return MODALITY_CONFIGS[EMBODIMENT_TAG]


def test_native_h40_contract_and_stats(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical")
    view = tmp_path / "view"
    manifest = build_metadata_loader_view(canonical, view, enforce_frozen_identity=False)
    assert manifest["baseline_id"] == BASELINE_ID
    assert manifest["row_grid_rate_hz"] == 30
    assert manifest["true_30hz_hardware_clock_proven"] is False
    stats_path = tmp_path / "derived/h40_relative.json"
    stats = compute_h40_relative_statistics(canonical, stats_path)
    assert stats["anchors"] == 6
    assert stats["source_index_formula"] == "n+k"
    for side in ("left_arm", "right_arm"):
        for name in STAT_NAMES:
            assert np.asarray(stats["relative_action"][side][name]).shape == (40, 7)
    config_path = _adapter_config(tmp_path / "adapter.json", stats_path)
    assert load_adapter_config(config_path)["horizon"] == 40
    modality = _modality(Path(__file__).parents[2])
    assert modality["video"].delta_indices == [0]
    assert modality["state"].delta_indices == [0]
    assert modality["action"].delta_indices == list(range(40))
    loader = G1Id10H40Native30EpisodeLoader(
        view, modality, config_path, require_runtime_verified_fixed_state=False
    )
    frame = loader._load_parquet_data(0)
    assert len(frame) == 45
    assert frame["language.annotation.human.task_description"].unique().tolist() == [LANGUAGE]
    loaded_stats = loader.get_dataset_statistics()
    assert np.asarray(loaded_stats["relative_action"]["left_arm"]["mean"]).shape == (40, 7)


def test_mask_is_exactly_40_by_26():
    item = {"action_mask": torch.ones((40, 132), dtype=torch.float32)}
    result = apply_active_action_mask(item)
    assert int(result["action_mask"].sum().item()) == 1040
    assert torch.all(result["action_mask"][:, :26] == 1)
    assert torch.all(result["action_mask"][:, 26:] == 0)
    with pytest.raises(ValueError, match=r"\[40,132\]"):
        apply_active_action_mask({"action_mask": torch.ones((50, 132))})


def test_h50_config_is_rejected_as_h40(tmp_path: Path):
    stats = tmp_path / "stats.json"
    _write_json(stats, {})
    config = _adapter_config(tmp_path / "adapter.json", stats)
    value = json.loads(config.read_text())
    value["horizon"] = 50
    _write_json(tmp_path / "wrong.json", value)
    with pytest.raises(ValueError, match="horizon"):
        load_adapter_config(tmp_path / "wrong.json")


def test_new_embodiment_mapping_is_ten():
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
        EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    )

    assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX["new_embodiment"] == 10


def test_factory_h40_adapter_never_writes_canonical_stats():
    from gr00t.data.dataset.factory import DatasetFactory

    spec = SimpleNamespace(
        dataset_paths=["loader-view"],
        embodiment_tag=EMBODIMENT_TAG,
        mix_ratio=1.0,
        dataset_adapter=ADAPTER_KIND,
        adapter_config_path="adapter.json",
    )
    data = SimpleNamespace(
        datasets=[spec],
        mode="single_turn",
        modality_configs={EMBODIMENT_TAG: {}},
        video_backend="torchcodec",
        shard_size=128,
        episode_sampling_rate=0.1,
        seed=42,
        allow_padding=False,
        num_shards_per_epoch=1,
        override_pretraining_statistics=True,
    )
    config = SimpleNamespace(data=data, training=SimpleNamespace(eval_strategy="no"))
    fake_dataset = MagicMock()
    fake_dataset.__len__.return_value = 1
    mixture = object()
    with (
        patch("gr00t.data.dataset.factory.generate_stats") as generate_stats,
        patch("gr00t.data.dataset.factory.generate_rel_stats") as generate_rel_stats,
        patch(
            "gr00t.data.dataset.g1_id10_h40_native30_adapter.G1Id10H40Native30SingleStepDataset",
            return_value=fake_dataset,
        ) as adapter_dataset,
        patch("gr00t.data.dataset.factory.ShardedMixtureDataset", return_value=mixture),
    ):
        result, evaluation = DatasetFactory(config).build(MagicMock())
    assert result is mixture and evaluation is None
    generate_stats.assert_not_called()
    generate_rel_stats.assert_not_called()
    adapter_dataset.assert_called_once()
