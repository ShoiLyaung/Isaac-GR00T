from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    CANONICAL_INFO_SHA256,
    CANONICAL_MODALITY_SHA256,
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
    scatter_inspire_u_to_sim_joint_targets,
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
    manifest = build_metadata_loader_view(canonical, view, enforce_frozen_identity=False)
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
    build_metadata_loader_view(canonical, view, enforce_frozen_identity=False)
    assert (canonical / "meta/info.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        build_metadata_loader_view(canonical, view, enforce_frozen_identity=False)
    with pytest.raises(ValueError):
        build_metadata_loader_view(
            canonical, canonical / "nested-view", enforce_frozen_identity=False
        )


def test_invalid_row_and_unverified_runtime_are_rejected(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical")
    view = tmp_path / "view"
    build_metadata_loader_view(canonical, view, enforce_frozen_identity=False)
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

    left = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    right = np.linspace(1.0, 0.0, 6, dtype=np.float32)
    full = scatter_inspire_u_to_sim_joint_targets(np.zeros(53, dtype=np.float32), left, right)
    assert full[[31, 32, 30, 29, 43, 33]] == pytest.approx(inspire_u_to_rad(left))
    assert full[[36, 37, 35, 34, 48, 38]] == pytest.approx(inspire_u_to_rad(right))
    inactive = np.ones(53, dtype=bool)
    inactive[[31, 32, 30, 29, 43, 33, 36, 37, 35, 34, 48, 38]] = False
    assert np.all(full[inactive] == 0.0)


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


def test_checkpoint_processor_h40_can_be_overridden_to_h50(tmp_path: Path, monkeypatch):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    checkpoint = tmp_path / "processor-checkpoint"
    _write_json(
        checkpoint / "processor_config.json",
        {
            "processor_kwargs": {
                "modality_configs": {},
                "max_action_horizon": 40,
            }
        },
    )
    _write_json(checkpoint / "statistics.json", {})
    _write_json(checkpoint / "embodiment_id.json", {})

    captured = {}

    def capture_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(Gr00tN1d7Processor, "__init__", capture_init)
    Gr00tN1d7Processor.from_pretrained(
        checkpoint,
        max_action_horizon=50,
        transformers_loading_kwargs={"local_files_only": True},
    )
    assert captured["max_action_horizon"] == 50
    assert captured["model_name"] == "nvidia/Cosmos-Reason2-2B"


def test_factory_adapter_branch_never_calls_canonical_stats_writers():
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
            "gr00t.data.dataset.g1_id25_h50_adapter.G1Id25H50SingleStepDataset",
            return_value=fake_dataset,
        ) as adapter_dataset,
        patch("gr00t.data.dataset.factory.ShardedMixtureDataset", return_value=mixture),
    ):
        result, eval_result = DatasetFactory(config).build(MagicMock())

    assert result is mixture
    assert eval_result is None
    generate_stats.assert_not_called()
    generate_rel_stats.assert_not_called()
    adapter_dataset.assert_called_once()


def test_frozen_canonical_metadata_identity_is_current():
    root = Path("/home/sl/datasets/insertion_shaft_main_300ep_groot/meta")
    assert hashlib.sha256((root / "info.json").read_bytes()).hexdigest() == CANONICAL_INFO_SHA256
    assert (
        hashlib.sha256((root / "modality.json").read_bytes()).hexdigest()
        == CANONICAL_MODALITY_SHA256
    )


def test_adapter_config_rejects_unknown_fields(tmp_path: Path):
    relative = tmp_path / "relative.json"
    config = _write_adapter_config(tmp_path / "adapter.json", relative)
    payload = json.loads(config.read_text())
    payload["unknown"] = "forbidden"
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_adapter_config(config)


@pytest.mark.parametrize("defect", ["float64", "nan", "hand_range"])
def test_loader_rejects_dtype_nonfinite_and_hand_range(tmp_path: Path, defect: str):
    canonical = _make_canonical(tmp_path / "canonical")
    derived = tmp_path / "relative.json"
    compute_h50_relative_statistics(canonical, derived)
    config = _write_adapter_config(tmp_path / "adapter.json", derived)
    view = tmp_path / "view"
    build_metadata_loader_view(canonical, view, enforce_frozen_identity=False)
    parquet = canonical / "data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(parquet)
    rows = [np.array(row, copy=True) for row in frame["observation.state"]]
    if defect == "float64":
        rows = [row.astype(np.float64) for row in rows]
    elif defect == "nan":
        rows[0][0] = np.nan
    else:
        rows[0][14] = 1.1
    frame["observation.state"] = rows
    frame.to_parquet(parquet, index=False)
    modality = _load_modality_config(Path(__file__).parents[2])
    loader = G1Id25H50EpisodeLoader(
        view, modality, config, require_runtime_verified_fixed_state=False
    )
    error = TypeError if defect == "float64" else ValueError
    with pytest.raises(error):
        loader._load_parquet_data(0)
