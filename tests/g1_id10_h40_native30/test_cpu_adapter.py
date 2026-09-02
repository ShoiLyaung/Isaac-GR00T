from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gr00t.data.dataset.g1_id10_h40_native30_adapter as h40_adapter
from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTIVE_ACTION_DIM,
    ADAPTER_KIND,
    BASELINE_ID,
    CONFIG_SCHEMA_VERSION,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    HORIZON,
    LANGUAGE,
    TEMPORAL_OPTION,
    G1Id10H40ManifestEpisodeCollection,
    G1Id10H40Native30EpisodeLoader,
    G1Id10H40Native30SingleStepDataset,
    apply_active_action_mask,
    build_metadata_loader_view,
    compute_h40_relative_statistics,
    compute_h40_train_statistics,
    load_adapter_config,
    resolve_decoded_action_26d,
)
import numpy as np
import pandas as pd
import pytest
import torch


STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
FIXTURE_EXPECTED_COHORTS = {
    "fixture-main": {
        "dataset_name": "insert the shaft into the housing",
        "expected_episodes": 2,
        "split_counts": {"train": 1, "val": 1, "test": 0},
    }
}


def _use_fixture_cohort_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h40_adapter, "EXPECTED_COHORTS", FIXTURE_EXPECTED_COHORTS)


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
            "timestamp": np.arange(length, dtype=np.float64) / 30.0,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.zeros(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            "annotation.human.action.task_description": ["insertion_shaft_main_300ep"] * length,
            "annotation.human.validity": np.ones(length, dtype=np.int64),
        }
    ).to_parquet(root / "data/chunk-000/episode_000000.parquet", index=False)
    (root / "videos/chunk-000/observation.images.cam_left_high/episode_000000.mp4").write_bytes(
        b"fixture-video-0"
    )
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
    _write_jsonl(
        root / "meta/episodes.jsonl", [{"episode_index": 0, "length": length, "tasks": ["slug"]}]
    )
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
            "fixed_action": {
                "waist": [0.0] * 3,
                "base_height_command": [0.0],
                "navigate_command": [0.0] * 3,
            },
            "derived_relative_stats_path": str(stats_path),
        },
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fixture_episode_uid(
    dataset_name: str, episode_index: int, parquet_sha256: str, video_sha256: str
) -> str:
    value = f"icra27-episode-v1\0{dataset_name}\0{episode_index}\0{parquet_sha256}\0{video_sha256}"
    return hashlib.sha256(value.encode()).hexdigest()


def _fixture_family_id(uid: str) -> str:
    return hashlib.sha256(f"icra27-family-v1\n{uid}\n".encode()).hexdigest()


def _fixture_rank_key(catalog_sha256: str, cohort_id: str, family_id: str) -> str:
    value = f"ICRA27-DUAL-VLA-SPLIT-R1\0{catalog_sha256}\0{cohort_id}\0{family_id}\0{42}"
    return hashlib.sha256(value.encode()).hexdigest()


def _make_manifest_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "groot_datasets"
    dataset_name = "insert the shaft into the housing"
    canonical = _make_canonical(dataset_root / "lerobot" / dataset_name)
    first = pd.read_parquet(canonical / "data/chunk-000/episode_000000.parquet")
    second = first.copy()
    second["episode_index"] = 1
    second["action"] = [
        np.asarray(row, dtype=np.float32) + np.float32(0.01) for row in second["action"]
    ]
    second.to_parquet(canonical / "data/chunk-000/episode_000001.parquet", index=False)
    (
        canonical / "videos/chunk-000/observation.images.cam_left_high/episode_000001.mp4"
    ).write_bytes(b"fixture-video-1")
    _write_jsonl(
        canonical / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "length": len(first), "tasks": ["slug"]},
            {"episode_index": 1, "length": len(second), "tasks": ["slug"]},
        ],
    )
    catalog = dataset_root / "catalog/current.sqlite3"
    catalog.parent.mkdir(parents=True)
    logical_records = {}
    with sqlite3.connect(catalog) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE files (
              relative_path TEXT PRIMARY KEY,
              entry_type TEXT NOT NULL,
              size INTEGER NOT NULL,
              sha256 TEXT NOT NULL
            );
            CREATE TABLE lerobot_episodes (
              relative_path TEXT PRIMARY KEY,
              dataset_name TEXT NOT NULL,
              rows INTEGER NOT NULL,
              state_sha256 TEXT NOT NULL,
              action_sha256 TEXT NOT NULL,
              state_action_sha256 TEXT NOT NULL,
              state_action_round1e5_sha256 TEXT NOT NULL,
              state_dim INTEGER NOT NULL,
              action_dim INTEGER NOT NULL,
              finite_ok INTEGER NOT NULL,
              frame_index_ok INTEGER NOT NULL,
              episode_index_ok INTEGER NOT NULL,
              timestamp_30hz_ok INTEGER NOT NULL,
              validity_ok INTEGER NOT NULL,
              task_index_zero_ok INTEGER NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [("run_id", "fixture-index-r1"), ("root", str(dataset_root))],
        )
        for episode_index, frame in ((0, first), (1, second)):
            parquet_path = canonical / f"data/chunk-000/episode_{episode_index:06d}.parquet"
            video_path = (
                canonical
                / "videos/chunk-000/observation.images.cam_left_high"
                / f"episode_{episode_index:06d}.mp4"
            )
            parquet_relative = str(parquet_path.relative_to(dataset_root))
            video_relative = str(video_path.relative_to(dataset_root))
            state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
            action = np.stack(frame["action"].to_numpy()).astype(np.float32)
            record = {
                "dataset_name": dataset_name,
                "episode_index": episode_index,
                "rows": len(frame),
                "parquet_relative_path": parquet_relative,
                "parquet_sha256": _sha256(parquet_path),
                "parquet_bytes": parquet_path.stat().st_size,
                "cam_left_high_relative_path": video_relative,
                "cam_left_high_sha256": _sha256(video_path),
                "cam_left_high_bytes": video_path.stat().st_size,
                "state_sha256": _fingerprint(state),
                "action_sha256": _fingerprint(action),
                "state_action_sha256": _fingerprint(state, action),
                "state_action_round1e5_sha256": _fingerprint(
                    np.round(state, 5), np.round(action, 5)
                ),
            }
            logical_records[episode_index] = record
            connection.executemany(
                "INSERT INTO files VALUES (?,?,?,?)",
                [
                    (
                        parquet_relative,
                        "file",
                        record["parquet_bytes"],
                        record["parquet_sha256"],
                    ),
                    (
                        video_relative,
                        "file",
                        record["cam_left_high_bytes"],
                        record["cam_left_high_sha256"],
                    ),
                ],
            )
            connection.execute(
                "INSERT INTO lerobot_episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    parquet_relative.removeprefix("lerobot/"),
                    dataset_name,
                    len(frame),
                    record["state_sha256"],
                    record["action_sha256"],
                    record["state_action_sha256"],
                    record["state_action_round1e5_sha256"],
                    26,
                    26,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ),
            )
    catalog_identity = {
        "run_id": "fixture-index-r1",
        "database_path": str(catalog),
        "database_sha256": _sha256(catalog),
    }
    cohort_id = "fixture-main"
    cohort_record = {
        "cohort_id": cohort_id,
        "dataset_name": dataset_name,
        "expected_episodes": 2,
        "dataset_relative_path": f"lerobot/{dataset_name}",
    }
    payload_manifest = tmp_path / "payload_manifest.json"
    payload_episodes = []
    for episode_index in (1, 0):
        record = {**logical_records[episode_index], "cohort_id": cohort_id}
        record["episode_uid"] = _fixture_episode_uid(
            dataset_name,
            episode_index,
            record["parquet_sha256"],
            record["cam_left_high_sha256"],
        )
        payload_episodes.append(record)
    _write_json(
        payload_manifest,
        {
            "schema_version": "icra27-dual-vla-payload-manifest-v1",
            "contract_id": "ICRA27-DUAL-VLA-DATASET-R1",
            "dataset_root": str(dataset_root),
            "catalog": catalog_identity,
            "selected_camera_key": "observation.images.cam_left_high",
            "contract_instruction": LANGUAGE,
            "cohorts": [cohort_record],
            "episodes": payload_episodes,
        },
    )
    assignments = []
    for payload_row in payload_episodes:
        family_id = _fixture_family_id(payload_row["episode_uid"])
        assignments.append(
            {
                "episode_uid": payload_row["episode_uid"],
                "cohort_id": cohort_id,
                "dataset_name": dataset_name,
                "episode_index": payload_row["episode_index"],
                "rows": payload_row["rows"],
                "family_id": family_id,
                "family_evidence": ["STATE_ACTION_EXACT_OR_ROUND1E5_COMPONENT"],
                "session_id": "UNKNOWN:fixture-main",
                "session_id_source": "NO_RECORDER_SESSION_ID_AVAILABLE",
                "eligible": True,
                "exclusion_reason": None,
                "split": None,
                "rank_key": _fixture_rank_key(
                    catalog_identity["database_sha256"], cohort_id, family_id
                ),
            }
        )
    by_rank = sorted(assignments, key=lambda row: row["rank_key"])
    by_rank[0]["split"] = "train"
    by_rank[1]["split"] = "val"
    train_assignment = by_rank[0]
    train_digest = hashlib.sha256((train_assignment["episode_uid"] + "\n").encode()).hexdigest()
    split_manifest = tmp_path / "split_manifest.json"
    _write_json(
        split_manifest,
        {
            "schema_version": "icra27-dual-vla-split-manifest-v1",
            "split_manifest_id": "ICRA27-DUAL-VLA-SPLIT-R1",
            "contract_id": "ICRA27-DUAL-VLA-DATASET-R1",
            "payload_manifest_path": str(payload_manifest),
            "payload_manifest_sha256": _sha256(payload_manifest),
            "dataset_root": str(dataset_root),
            "catalog": catalog_identity,
            "selected_camera_key": "observation.images.cam_left_high",
            "contract_instruction": LANGUAGE,
            "split_algorithm_id": "FAMILY_HASH_HAMILTON_V1",
            "split_seed": 42,
            "cohorts": [cohort_record],
            "assignments": assignments,
            "target_counts_by_cohort": {cohort_id: {"train": 1, "val": 1, "test": 0}},
            "actual_counts_global": {"train": 1, "val": 1, "test": 0},
            "actual_counts_by_cohort": {cohort_id: {"train": 1, "val": 1, "test": 0}},
            "validation_summary": {
                "eligible_episode_count": 2,
                "cross_split_family_intersection": 0,
                "cross_split_exact_trajectory_intersection": 0,
                "cross_split_rounded_trajectory_intersection": 0,
                "unresolved_parent_count": 0,
                "physical_payload_mutations": 0,
            },
            "train_episode_uid_digest": train_digest,
        },
    )
    train_stats = tmp_path / "train_stats.json"
    result = compute_h40_train_statistics(split_manifest, train_stats)
    assert result["fit_split"] == "train"
    assert result["episodes"] == 1
    return dataset_root, split_manifest, train_stats


def _resign_fixture_bundle(
    payload_path: Path, split_path: Path, payload: dict, split: dict
) -> None:
    assignments = []
    for payload_row in payload["episodes"]:
        family_id = _fixture_family_id(payload_row["episode_uid"])
        assignments.append(
            {
                "episode_uid": payload_row["episode_uid"],
                "cohort_id": payload_row["cohort_id"],
                "dataset_name": payload_row["dataset_name"],
                "episode_index": payload_row["episode_index"],
                "rows": payload_row["rows"],
                "family_id": family_id,
                "family_evidence": ["STATE_ACTION_EXACT_OR_ROUND1E5_COMPONENT"],
                "session_id": f"UNKNOWN:{payload_row['cohort_id']}",
                "session_id_source": "NO_RECORDER_SESSION_ID_AVAILABLE",
                "eligible": True,
                "exclusion_reason": None,
                "split": None,
                "rank_key": _fixture_rank_key(
                    split["catalog"]["database_sha256"],
                    payload_row["cohort_id"],
                    family_id,
                ),
            }
        )
    for index, row in enumerate(sorted(assignments, key=lambda item: item["rank_key"])):
        row["split"] = "train" if index == 0 else "val"
    split["assignments"] = assignments
    actual = {
        label: sum(row["split"] == label for row in assignments)
        for label in ("train", "val", "test")
    }
    split["actual_counts_global"] = actual
    split["actual_counts_by_cohort"] = {"fixture-main": actual}
    split["validation_summary"]["eligible_episode_count"] = len(assignments)
    train_uids = sorted(row["episode_uid"] for row in assignments if row["split"] == "train")
    split["train_episode_uid_digest"] = hashlib.sha256(
        ("\n".join(train_uids) + "\n").encode()
    ).hexdigest()
    _write_json(payload_path, payload)
    split["payload_manifest_sha256"] = _sha256(payload_path)
    _write_json(split_path, split)


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


def test_h40_finite_action_helper_is_horizon_specific_and_fail_closed():
    decoded = {
        "left_arm": np.ones((40, 7), dtype=np.float32),
        "right_arm": np.full((40, 7), 2.0, dtype=np.float32),
        "left_hand": np.full((40, 6), -0.1, dtype=np.float32),
        "right_hand": np.full((40, 6), 1.1, dtype=np.float32),
        "waist": np.full((40, 3), 99.0, dtype=np.float32),
        "base_height_command": np.full((40, 1), 99.0, dtype=np.float32),
        "navigate_command": np.full((40, 3), 99.0, dtype=np.float32),
    }
    active, clamp_count = resolve_decoded_action_26d(decoded)
    assert active.shape == (40, 26)
    assert np.isfinite(active).all()
    assert clamp_count == 480
    assert np.all(active[:, 14:20] == 0)
    assert np.all(active[:, 20:26] == 1)
    assert not np.any(active == 99.0)

    with pytest.raises(ValueError, match=r"finite shape \(40, 7\)"):
        resolve_decoded_action_26d({**decoded, "left_arm": np.ones((50, 7))})
    with pytest.raises(ValueError, match="outside"):
        resolve_decoded_action_26d(decoded, clamp_hand=False)
    with pytest.raises(ValueError, match=r"finite shape \(40, 6\)"):
        bad = np.zeros((40, 6), dtype=np.float32)
        bad[0, 0] = np.nan
        resolve_decoded_action_26d({**decoded, "left_hand": bad})
    with pytest.raises(KeyError, match="navigate_command"):
        resolve_decoded_action_26d(
            {key: value for key, value in decoded.items() if key != "navigate_command"}
        )


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
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX

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


def test_manifest_filters_real_episode_access_and_reuses_train_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fixture_cohort_contract(monkeypatch)
    dataset_root, split_manifest, train_stats = _make_manifest_fixture(tmp_path)
    config_path = _adapter_config(tmp_path / "adapter.json", train_stats)
    config = json.loads(config_path.read_text())
    config.update(
        {
            "split_manifest_path": str(split_manifest),
            "split": "train",
            "runtime_verified_fixed_state": True,
        }
    )
    _write_json(config_path, config)
    modality = _modality(Path(__file__).parents[2])
    canonical = dataset_root / "lerobot/insert the shaft into the housing"
    loader = G1Id10H40Native30EpisodeLoader(
        canonical, modality, config_path, require_runtime_verified_fixed_state=True
    )
    assert len(loader) == 1
    expected_train_assignment = next(
        row
        for row in json.loads(split_manifest.read_text())["assignments"]
        if row["split"] == "train"
    )
    expected_train_uid = expected_train_assignment["episode_uid"]
    expected_train_index = expected_train_assignment["episode_index"]
    assert [row["episode_index"] for row in loader.episodes_metadata] == [expected_train_index]
    assert loader.selected_episode_uids == [expected_train_uid]
    assert len(loader._load_parquet_data(expected_train_index)) == 45
    statistics = loader.get_dataset_statistics()
    assert np.asarray(statistics["state"]["left_arm"]["mean"]).shape == (7,)
    assert np.asarray(statistics["relative_action"]["left_arm"]["mean"]).shape == (40, 7)

    collection = G1Id10H40ManifestEpisodeCollection(dataset_root / "lerobot", modality, config_path)
    assert len(collection) == 1
    assert collection.episodes_metadata[0]["source_episode_index"] == expected_train_index
    dataset = G1Id10H40Native30SingleStepDataset(
        dataset_path=dataset_root / "lerobot",
        embodiment_tag=EMBODIMENT_TAG,
        modality_configs=modality,
        shard_size=128,
        episode_sampling_rate=1.0,
        seed=42,
        allow_padding=False,
        adapter_config_path=config_path,
    )
    assert len(dataset.episode_loader) == 1
    assert (
        dataset.episode_loader.episodes_metadata[0]["source_episode_index"] == expected_train_index
    )
    assert {episode_index for shard in dataset.sharded_episodes for episode_index, _ in shard} == {
        0
    }

    val_config = json.loads(config_path.read_text())
    val_config["split"] = "val"
    val_path = tmp_path / "adapter-val.json"
    _write_json(val_path, val_config)
    val_loader = G1Id10H40Native30EpisodeLoader(
        canonical, modality, val_path, require_runtime_verified_fixed_state=True
    )
    expected_val_index = next(
        row["episode_index"]
        for row in json.loads(split_manifest.read_text())["assignments"]
        if row["split"] == "val"
    )
    assert [row["episode_index"] for row in val_loader.episodes_metadata] == [expected_val_index]
    assert val_loader.train_statistics["fit_split"] == "train"


def test_manifest_rejects_non_train_statistics_and_catalog_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fixture_cohort_contract(monkeypatch)
    dataset_root, split_manifest, train_stats = _make_manifest_fixture(tmp_path)
    config_path = _adapter_config(tmp_path / "adapter.json", train_stats)
    config = json.loads(config_path.read_text())
    config.update({"split_manifest_path": str(split_manifest), "split": "val"})
    bad_stats = json.loads(train_stats.read_text())
    bad_stats["fit_split"] = "val"
    _write_json(train_stats, bad_stats)
    _write_json(config_path, config)
    modality = _modality(Path(__file__).parents[2])
    canonical = dataset_root / "lerobot/insert the shaft into the housing"
    with pytest.raises(ValueError, match="fit on train only"):
        G1Id10H40Native30EpisodeLoader(
            canonical, modality, config_path, require_runtime_verified_fixed_state=False
        )

    # A post-rsync/re-index catalog identity change invalidates both the old
    # manifest and the statistics bound to it.
    (dataset_root / "catalog/current.sqlite3").write_bytes(b"changed-catalog")
    with pytest.raises(ValueError, match="stale for the current catalog"):
        G1Id10H40Native30EpisodeLoader(
            canonical, modality, config_path, require_runtime_verified_fixed_state=False
        )


def test_manifest_rejects_weak_or_leaking_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fixture_cohort_contract(monkeypatch)
    _, split_manifest, _ = _make_manifest_fixture(tmp_path / "family-leak")
    value = json.loads(split_manifest.read_text())
    value["assignments"][1]["family_id"] = value["assignments"][0]["family_id"]
    _write_json(split_manifest, value)
    with pytest.raises(ValueError, match="family ID does not derive"):
        compute_h40_train_statistics(split_manifest, tmp_path / "family-leak-stats.json")

    _, split_manifest, _ = _make_manifest_fixture(tmp_path / "payload-mismatch")
    value = json.loads(split_manifest.read_text())
    value["assignments"][0]["episode_index"] = 0
    _write_json(split_manifest, value)
    with pytest.raises(ValueError, match="episode_index differs from payload"):
        compute_h40_train_statistics(split_manifest, tmp_path / "payload-mismatch-stats.json")

    _, split_manifest, _ = _make_manifest_fixture(tmp_path / "weak-contract")
    value = json.loads(split_manifest.read_text())
    value.pop("payload_manifest_sha256")
    _write_json(split_manifest, value)
    with pytest.raises(ValueError, match="payload-manifest SHA"):
        compute_h40_train_statistics(split_manifest, tmp_path / "weak-contract-stats.json")

    _, split_manifest, _ = _make_manifest_fixture(tmp_path / "forged-catalog-binding")
    payload_path = split_manifest.with_name("payload_manifest.json")
    payload = json.loads(payload_path.read_text())
    forged = payload["episodes"][0]
    forged["parquet_sha256"] = "f" * 64
    forged["episode_uid"] = _fixture_episode_uid(
        forged["dataset_name"],
        forged["episode_index"],
        forged["parquet_sha256"],
        forged["cam_left_high_sha256"],
    )
    _write_json(payload_path, payload)
    value = json.loads(split_manifest.read_text())
    value["payload_manifest_sha256"] = _sha256(payload_path)
    assignment = value["assignments"][0]
    assignment["episode_uid"] = forged["episode_uid"]
    assignment["family_id"] = _fixture_family_id(forged["episode_uid"])
    assignment["rank_key"] = _fixture_rank_key(
        value["catalog"]["database_sha256"],
        assignment["cohort_id"],
        assignment["family_id"],
    )
    value["train_episode_uid_digest"] = hashlib.sha256(
        (forged["episode_uid"] + "\n").encode()
    ).hexdigest()
    _write_json(split_manifest, value)
    with pytest.raises(ValueError, match="differs from current catalog"):
        compute_h40_train_statistics(split_manifest, tmp_path / "forged-catalog-binding-stats.json")


def test_manifest_rejects_resigned_episode_index_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fixture_cohort_contract(monkeypatch)
    _, split_path, _ = _make_manifest_fixture(tmp_path)
    payload_path = split_path.with_name("payload_manifest.json")
    payload = json.loads(payload_path.read_text())
    split = json.loads(split_path.read_text())
    first, second = payload["episodes"]
    first["episode_index"], second["episode_index"] = (
        second["episode_index"],
        first["episode_index"],
    )
    for row in payload["episodes"]:
        row["episode_uid"] = _fixture_episode_uid(
            row["dataset_name"],
            row["episode_index"],
            row["parquet_sha256"],
            row["cam_left_high_sha256"],
        )
    _resign_fixture_bundle(payload_path, split_path, payload, split)
    with pytest.raises(ValueError, match="path does not derive"):
        compute_h40_train_statistics(split_path, tmp_path / "swap-stats.json")


def test_manifest_rejects_resigned_fixed_cohort_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fixture_cohort_contract(monkeypatch)
    _, split_path, _ = _make_manifest_fixture(tmp_path)
    payload_path = split_path.with_name("payload_manifest.json")
    payload = json.loads(payload_path.read_text())
    split = json.loads(split_path.read_text())
    payload["episodes"].pop()
    _resign_fixture_bundle(payload_path, split_path, payload, split)
    with pytest.raises(ValueError, match="completely cover a fixed R1 cohort"):
        compute_h40_train_statistics(split_path, tmp_path / "subset-stats.json")


def test_manifest_rejects_resigned_split_label_shuffle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _use_fixture_cohort_contract(monkeypatch)
    _, split_path, _ = _make_manifest_fixture(tmp_path)
    split = json.loads(split_path.read_text())
    first, second = split["assignments"]
    first["split"], second["split"] = second["split"], first["split"]
    train_uid = next(row["episode_uid"] for row in split["assignments"] if row["split"] == "train")
    split["train_episode_uid_digest"] = hashlib.sha256((train_uid + "\n").encode()).hexdigest()
    _write_json(split_path, split)
    with pytest.raises(ValueError, match="does not derive from the frozen algorithm"):
        compute_h40_train_statistics(split_path, tmp_path / "shuffle-stats.json")
