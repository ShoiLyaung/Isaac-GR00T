from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.dataset.g1_id25_h50_adapter import (
    ACTION_GROUPS,
    ACTIVE_ACTION_DIM,
    ADAPTER_KIND,
    BASELINE_ID,
    CONFIG_SCHEMA_VERSION,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    G1Id25H50EpisodeLoader,
    HORIZON,
    LANGUAGE,
    STATE_GROUPS,
    TEMPORAL_OPTION,
    apply_active_action_mask,
    build_metadata_loader_view,
    compute_h50_relative_statistics,
    h50_zoh_offsets,
    inspire_rad_to_u,
    inspire_u_to_rad,
    load_adapter_config,
    resolve_decoded_action_26d,
)


STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")


def _stats(dim: int) -> dict[str, list[float]]:
    return {
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "min": [-1.0] * dim,
        "max": [1.0] * dim,
        "q01": [-1.0] * dim,
        "q99": [1.0] * dim,
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_canonical(root: Path, length: int = 35) -> Path:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/chunk-000/observation.images.cam_left_high").mkdir(parents=True)
    state = []
    action = []
    for i in range(length):
        arm = np.linspace(-0.2, 0.2, 14, dtype=np.float32) + i * 0.001
        hand = np.linspace(0.1, 0.9, 12, dtype=np.float32)
        state.append(np.concatenate([arm, hand]).astype(np.float32))
        action.append(np.concatenate([arm + 0.05, hand]).astype(np.float32))
    frame = pd.DataFrame(
        {
            "observation.state": state,
            "action": action,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.zeros(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            "annotation.human.action.task_description": ["insertion_shaft_main_300ep"]
            * length,
            "annotation.human.validity": np.ones(length, dtype=np.int64),
        }
    )
    frame.to_parquet(root / "data/chunk-000/episode_000000.parquet", index=False)
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
                "observation.images.cam_left_high": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                },
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
            "video": {
                "cam_left_high": {
                    "original_key": "observation.images.cam_left_high",
                }
            },
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


def _write_adapter_config(path: Path, relative_stats: Path, runtime_verified=False) -> Path:
    _write_json(
        path,
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "adapter_kind": ADAPTER_KIND,
            "baseline_id": BASELINE_ID,
            "embodiment_tag": EMBODIMENT_TAG,
            "embodiment_id": EMBODIMENT_ID,
            "temporal_option": TEMPORAL_OPTION,
            "source_index_formula": "n+floor(3*k/5)",
            "source_rate_hz": 30,
            "target_rate_hz": 50,
            "true_50hz_capture": False,
            "horizon": HORIZON,
            "active_action_dims": ACTIVE_ACTION_DIM,
            "hand_scope": "PROJECT_INSPIRE6_EXTENSION_NO_DIRECT_OFFICIAL_MATCH",
            "language": LANGUAGE,
            "runtime_verified_fixed_state": runtime_verified,
            "fixed_state": {
                "left_leg": [0.0] * 6,
                "right_leg": [0.0] * 6,
                "waist": [0.0] * 3,
            },
            "fixed_action": {
                "waist": [0.0] * 3,
                "base_height_command": [0.0],
                "navigate_command": [0.0] * 3,
            },
            "derived_relative_stats_path": str(relative_stats),
        },
    )
    return path


def _load_modality_config(repo_root: Path):
    path = repo_root / "examples/G1_ID25_H50/g1_id25_h50_config.py"
    spec = importlib.util.spec_from_file_location("test_g1_id25_h50_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    return MODALITY_CONFIGS[EMBODIMENT_TAG]


def test_frozen_zoh_offsets_are_exact_and_repeated():
    offsets = h50_zoh_offsets()
    assert len(offsets) == 50
    assert offsets[:10] == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
    assert offsets[-1] == 29
    assert offsets == [(3 * k) // 5 for k in range(50)]
    assert len(set(offsets)) == 30


def test_loader_view_stats_and_episode_contract(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical")
    view = tmp_path / "view"
    manifest = build_metadata_loader_view(canonical, view)
    assert manifest["true_50hz_capture"] is False
    assert (view / "data").is_symlink()
    assert (view / "videos").is_symlink()
    assert "episode_chunk" in json.loads((view / "meta/info.json").read_text())["data_path"]
    assert json.loads((view / "meta/tasks.jsonl").read_text())["task"] == LANGUAGE

    derived = tmp_path / "derived/relative_stats.json"
    result = compute_h50_relative_statistics(canonical, derived)
    assert result["anchors"] == 6
    assert result["true_50hz_capture"] is False
    for side in ("left_arm", "right_arm"):
        for name in STAT_NAMES:
            assert np.asarray(result["relative_action"][side][name]).shape == (50, 7)

    config_path = _write_adapter_config(tmp_path / "adapter.json", derived)
    reloaded = load_adapter_config(config_path)
    assert reloaded["temporal_option"] == TEMPORAL_OPTION
    modality = _load_modality_config(Path(__file__).parents[2])
    assert modality["action"].delta_indices == h50_zoh_offsets()
    loader = G1Id25H50EpisodeLoader(
        view,
        modality,
        config_path,
        require_runtime_verified_fixed_state=False,
    )
    episode = loader._load_parquet_data(0)
    assert len(episode) == 35
    assert list(episode["state.left_arm"].iloc[0]) == pytest.approx(
        list(pd.read_parquet(canonical / "data/chunk-000/episode_000000.parquet")["observation.state"].iloc[0][:7])
    )
    assert episode["language.annotation.human.task_description"].unique().tolist() == [LANGUAGE]
    assert {column.removeprefix("state.") for column in episode if column.startswith("state.")} == set(STATE_GROUPS)
    assert {column.removeprefix("action.") for column in episode if column.startswith("action.")} == set(ACTION_GROUPS)
    statistics = loader.get_dataset_statistics()
    assert np.asarray(statistics["relative_action"]["left_arm"]["mean"]).shape == (50, 7)
    assert np.asarray(statistics["state"]["left_hand"]["mean"]).shape == (6,)


def test_loader_view_is_non_overwriting(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical")
    view = tmp_path / "view"
    before = (canonical / "meta/info.json").read_bytes()
    build_metadata_loader_view(canonical, view)
    assert (canonical / "meta/info.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        build_metadata_loader_view(canonical, view)
    with pytest.raises(ValueError):
        build_metadata_loader_view(canonical, canonical / "nested-view")


def test_invalid_row_and_unverified_runtime_are_rejected(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical")
    view = tmp_path / "view"
    build_metadata_loader_view(canonical, view)
    derived = tmp_path / "relative.json"
    compute_h50_relative_statistics(canonical, derived)
    config_path = _write_adapter_config(tmp_path / "adapter.json", derived)
    modality = _load_modality_config(Path(__file__).parents[2])
    with pytest.raises(RuntimeError, match="runtime-verified"):
        G1Id25H50EpisodeLoader(view, modality, config_path)

    parquet = canonical / "data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(parquet)
    frame.loc[3, "annotation.human.validity"] = 0
    frame.to_parquet(parquet, index=False)
    loader = G1Id25H50EpisodeLoader(
        view, modality, config_path, require_runtime_verified_fixed_state=False
    )
    with pytest.raises(ValueError, match="Invalid demonstration"):
        loader._load_parquet_data(0)


def test_action_mask_decode_lock_and_inspire_roundtrip():
    item = {"action_mask": torch.ones((50, 132), dtype=torch.float32)}
    masked = apply_active_action_mask(item)["action_mask"]
    assert torch.all(masked[:, :26] == 1)
    assert torch.all(masked[:, 26:] == 0)

    decoded = {
        "left_arm": np.ones((50, 7), dtype=np.float32),
        "right_arm": np.full((50, 7), 2.0, dtype=np.float32),
        "left_hand": np.full((50, 6), -0.1, dtype=np.float32),
        "right_hand": np.full((50, 6), 1.1, dtype=np.float32),
        "waist": np.full((50, 3), 99.0, dtype=np.float32),
        "base_height_command": np.full((50, 1), 99.0, dtype=np.float32),
        "navigate_command": np.full((50, 3), 99.0, dtype=np.float32),
    }
    active, clamp_count = resolve_decoded_action_26d(decoded)
    assert active.shape == (50, 26)
    assert clamp_count == 600
    assert np.all(active[:, 14:20] == 0)
    assert np.all(active[:, 20:26] == 1)
    assert not np.any(active == 99.0)

    u = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    assert inspire_rad_to_u(inspire_u_to_rad(u)) == pytest.approx(u, abs=2e-7)


def test_finetune_config_exposes_bounded_adapter_and_horizon_fields():
    cfg = FinetuneConfig(
        base_model_path="checkpoint",
        dataset_path="view",
        embodiment_tag="UNITREE_G1",
        dataset_adapter=ADAPTER_KIND,
        adapter_config_path="adapter.json",
        action_horizon=50,
    )
    assert cfg.dataset_adapter == ADAPTER_KIND
    assert cfg.adapter_config_path == "adapter.json"
    assert cfg.action_horizon == 50
