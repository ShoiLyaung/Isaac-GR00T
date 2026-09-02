from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.data.dataset.g1_all_lerobot_h40_adapter import (
    ADAPTER_KIND,
    BASELINE_ID,
    CONFIG_SCHEMA_VERSION,
    VIEW_SCHEMA_VERSION,
    G1AllLerobotH40EpisodeLoader,
    build_metadata_loader_view,
    load_adapter_config,
)
from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTIVE_ACTION_DIM,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    HORIZON,
    TEMPORAL_OPTION,
    _sha256,
    compute_h40_relative_statistics,
)
import numpy as np
import pandas as pd
import pytest


TRAINING_LANGUAGE = "insert the shaft into the housing"
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


def _make_canonical(
    root: Path,
    *,
    raw_label: str,
    cameras: tuple[str, ...] = ("cam_left_high",),
    length: int = 45,
) -> Path:
    (root / "data/chunk-000").mkdir(parents=True)
    for camera in cameras:
        (root / f"videos/chunk-000/observation.images.{camera}").mkdir(parents=True)
    states, actions = [], []
    for index in range(length):
        arms = np.linspace(-0.2, 0.2, 14, dtype=np.float32) + index * 0.001
        hands = np.linspace(0.1, 0.9, 12, dtype=np.float32)
        states.append(np.concatenate([arms, hands]).astype(np.float32))
        actions.append(np.concatenate([arms + 0.05, hands]).astype(np.float32))
    pd.DataFrame(
        {
            "observation.state": states,
            "action": actions,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.zeros(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            "annotation.human.action.task_description": [raw_label] * length,
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
                **{
                    f"observation.images.{camera}": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    }
                    for camera in cameras
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
                camera: {"original_key": f"observation.images.{camera}"} for camera in cameras
            },
            "annotation": {
                "human.action.task_description": {
                    "original_key": "annotation.human.action.task_description"
                }
            },
        },
    )
    _write_json(root / "meta/stats.json", {"observation.state": _stats(26), "action": _stats(26)})
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": 0, "length": length, "tasks": [raw_label]}],
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": raw_label}])
    return root


def _modality(repo: Path):
    config = repo / "examples/G1_ID10_H40_NATIVE30/g1_id10_h40_native30_config.py"
    spec = importlib.util.spec_from_file_location("test_all_lerobot_h40", config)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    return MODALITY_CONFIGS[EMBODIMENT_TAG]


def _adapter_config(
    path: Path, *, canonical: Path, raw_label: str, stats_path: Path, source_id: str
) -> Path:
    _write_json(
        path,
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "adapter_kind": ADAPTER_KIND,
            "baseline_id": BASELINE_ID,
            "source_id": source_id,
            "canonical_root": str(canonical.resolve()),
            "canonical_info_sha256": _sha256(canonical / "meta/info.json"),
            "canonical_modality_sha256": _sha256(canonical / "meta/modality.json"),
            "raw_task_labels": [raw_label],
            "training_language": TRAINING_LANGUAGE,
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
            "runtime_verified_fixed_state": False,
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
            "derived_relative_stats_path": str(stats_path.resolve()),
        },
    )
    return path


@pytest.mark.parametrize(
    ("raw_label", "cameras"),
    [
        ("insert the shaft into the housing", ("cam_left_high",)),
        ("100_notau", ("cam_left_high",)),
        (
            "insertion_with_torque",
            ("cam_left_high", "cam_left_wrist", "cam_right_wrist"),
        ),
    ],
)
def test_loader_view_preserves_source_label_and_normalizes_language(
    tmp_path: Path, raw_label: str, cameras: tuple[str, ...]
):
    canonical = _make_canonical(tmp_path / "canonical", raw_label=raw_label, cameras=cameras)
    protected = {
        path: path.read_bytes()
        for path in [
            canonical / "meta/info.json",
            canonical / "meta/modality.json",
            canonical / "meta/tasks.jsonl",
            canonical / "data/chunk-000/episode_000000.parquet",
        ]
    }
    source_id = "source"
    view = tmp_path / "view"
    manifest = build_metadata_loader_view(
        canonical,
        view,
        source_id=source_id,
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=[raw_label],
        training_language=TRAINING_LANGUAGE,
    )
    assert manifest["schema_version"] == VIEW_SCHEMA_VERSION
    assert manifest["raw_task_labels"] == [raw_label]
    assert manifest["training_language"] == TRAINING_LANGUAGE
    assert set(json.loads((view / "meta/modality.json").read_text())["video"]) == {"ego_view"}
    assert json.loads((view / "meta/tasks.jsonl").read_text())["task"] == TRAINING_LANGUAGE
    assert (view / "data").is_symlink() and (view / "data").resolve() == (
        canonical / "data"
    ).resolve()
    assert (view / "videos").is_symlink() and (view / "videos").resolve() == (
        canonical / "videos"
    ).resolve()
    assert all(path.read_bytes() == content for path, content in protected.items())

    stats_path = tmp_path / "relative.json"
    stats = compute_h40_relative_statistics(canonical, stats_path)
    assert stats["anchors"] == 6
    config = _adapter_config(
        tmp_path / "adapter.json",
        canonical=canonical,
        raw_label=raw_label,
        stats_path=stats_path,
        source_id=source_id,
    )
    assert load_adapter_config(config)["raw_task_labels"] == [raw_label]
    loader = G1AllLerobotH40EpisodeLoader(
        view,
        _modality(Path(__file__).parents[2]),
        config,
        require_runtime_verified_fixed_state=False,
    )
    frame = loader._load_parquet_data(0)
    assert frame["language.annotation.human.task_description"].unique().tolist() == [
        TRAINING_LANGUAGE
    ]
    for side in ("left_arm", "right_arm"):
        for name in STAT_NAMES:
            assert np.asarray(
                loader.get_dataset_statistics()["relative_action"][side][name]
            ).shape == (
                40,
                7,
            )


def test_loader_fails_closed_on_raw_label_mismatch(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="100_notau")
    view = tmp_path / "view"
    build_metadata_loader_view(
        canonical,
        view,
        source_id="source",
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=["100_notau"],
        training_language=TRAINING_LANGUAGE,
    )
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    config = _adapter_config(
        tmp_path / "adapter.json",
        canonical=canonical,
        raw_label="100_notau",
        stats_path=stats_path,
        source_id="source",
    )
    parquet = canonical / "data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(parquet)
    frame["annotation.human.action.task_description"] = "wrong"
    frame.to_parquet(parquet, index=False)
    loader = G1AllLerobotH40EpisodeLoader(
        view,
        _modality(Path(__file__).parents[2]),
        config,
        require_runtime_verified_fixed_state=False,
    )
    with pytest.raises(ValueError, match="Raw task label mismatch"):
        loader._load_parquet_data(0)


def test_loader_view_is_non_overwriting_and_external_only(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    kwargs = {
        "source_id": "source",
        "expected_info_sha256": _sha256(canonical / "meta/info.json"),
        "expected_modality_sha256": _sha256(canonical / "meta/modality.json"),
        "raw_task_labels": ["task"],
        "training_language": TRAINING_LANGUAGE,
    }
    with pytest.raises(ValueError, match="must not be inside"):
        build_metadata_loader_view(canonical, canonical / "view", **kwargs)
    view = tmp_path / "view"
    build_metadata_loader_view(canonical, view, **kwargs)
    with pytest.raises(FileExistsError):
        build_metadata_loader_view(canonical, view, **kwargs)


def test_factory_adapter_never_writes_canonical_stats():
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
        patch("gr00t.data.dataset.factory._resolve_all_lerobot_codec_config", return_value=None),
        patch(
            "gr00t.data.dataset.g1_all_lerobot_h40_adapter.G1AllLerobotH40SingleStepDataset",
            return_value=fake_dataset,
        ) as adapter_dataset,
        patch("gr00t.data.dataset.factory.ShardedMixtureDataset", return_value=mixture),
    ):
        result, evaluation = DatasetFactory(config).build(MagicMock())
    assert result is mixture and evaluation is None
    generate_stats.assert_not_called()
    generate_rel_stats.assert_not_called()
    adapter_dataset.assert_called_once()
