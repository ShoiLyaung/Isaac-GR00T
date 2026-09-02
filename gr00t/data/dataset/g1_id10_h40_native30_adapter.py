"""CPU-side adapter for the ID10/H40 native-represented 30 Hz baseline.

The canonical timestamps are a generated 30 Hz row grid.  This module never
claims that the original hardware action clock was independently measured at
30 Hz, and it never relabels the training targets as 50 Hz.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.dataset.g1_id10_h50_adapter import (
    ACTION_GROUPS,
    ACTION_SLICES,
    ACTIVE_ACTION_DIM,
    CANONICAL_INFO_SHA256,
    CANONICAL_MODALITY_SHA256,
    FIXED_ACTION_DIMS,
    FIXED_STATE_DIMS,
    LANGUAGE,
    PADDED_DIM,
    STATE_GROUPS,
    STATE_SLICES,
    G1Id10H50EpisodeLoader,
    _assert_external_output,
    _constant_stats,
    _format_episode_path,
    _json_dump_new,
    _json_load,
    _sha256,
    _slice_stats,
    _stack_float32_rows,
    inspire_rad_to_u,
    inspire_u_to_rad,
    scatter_inspire_u_to_sim_joint_targets,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import ModalityConfig


ADAPTER_KIND = "g1_id10_h40_native30_v1"
BASELINE_ID = "ID10_INSPIRE6_TASK_BASELINE_H40_NATIVE30"
EMBODIMENT_TAG = "new_embodiment"
EMBODIMENT_ID = 10
TEMPORAL_OPTION = "CANONICAL_NATIVE_REPRESENTED_30HZ_ROW_GRID_V1"
VIEW_SCHEMA_VERSION = "g1-id10-h40-native30-loader-view-v1"
CONFIG_SCHEMA_VERSION = "g1-id10-h40-native30-adapter-config-v1"
STATS_SCHEMA_VERSION = "g1-id10-h40-native30-relative-stats-v1"
TRAIN_STATS_SCHEMA_VERSION = "g1-id10-h40-native30-train-statistics-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "icra27-dual-vla-split-manifest-v1"
PAYLOAD_MANIFEST_SCHEMA_VERSION = "icra27-dual-vla-payload-manifest-v1"
DATASET_CONTRACT_ID = "ICRA27-DUAL-VLA-DATASET-R1"
SPLIT_MANIFEST_ID = "ICRA27-DUAL-VLA-SPLIT-R1"
SPLIT_ALGORITHM_ID = "FAMILY_HASH_HAMILTON_V1"
SPLIT_SEED = 42
EXPECTED_COHORTS: dict[str, dict[str, Any]] = {
    "main_300_20260602": {
        "dataset_name": "insert the shaft into the housing",
        "expected_episodes": 300,
        "split_counts": {"train": 240, "val": 30, "test": 30},
    },
    "aug_1p2mm_220_20260625_26": {
        "dataset_name": "aug1.2mm_insert the shaft into the housing",
        "expected_episodes": 220,
        "split_counts": {"train": 176, "val": 22, "test": 22},
    },
    "aug_5p4mm_153_20260624": {
        "dataset_name": "aug5.4mm_insert the shaft into the housing",
        "expected_episodes": 153,
        "split_counts": {"train": 123, "val": 15, "test": 15},
    },
}
HORIZON = 40
SOURCE_SPAN = 40
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")


def h40_native30_offsets() -> list[int]:
    offsets = list(range(HORIZON))
    assert offsets[0] == 0 and offsets[-1] == 39
    return offsets


def build_metadata_loader_view(
    canonical_root: str | Path,
    view_root: str | Path,
    *,
    enforce_frozen_identity: bool = True,
) -> dict[str, Any]:
    """Create a non-overwriting H40 metadata view over canonical payloads."""

    canonical_root = Path(canonical_root).resolve()
    view_root = Path(view_root)
    _assert_external_output(canonical_root, view_root)
    if view_root.exists():
        raise FileExistsError(view_root)
    canonical_meta = canonical_root / "meta"
    required = [
        canonical_meta / "info.json",
        canonical_meta / "modality.json",
        canonical_meta / "episodes.jsonl",
        canonical_meta / "tasks.jsonl",
        canonical_meta / "stats.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Canonical metadata missing: {missing}")
    if enforce_frozen_identity:
        if _sha256(canonical_meta / "info.json") != CANONICAL_INFO_SHA256:
            raise ValueError("Canonical info.json identity mismatch")
        if _sha256(canonical_meta / "modality.json") != CANONICAL_MODALITY_SHA256:
            raise ValueError("Canonical modality.json identity mismatch")

    view_meta = view_root / "meta"
    view_meta.mkdir(parents=True)
    for path in sorted(canonical_meta.iterdir()):
        if path.name in {"info.json", "modality.json", "tasks.jsonl", "relative_stats.json"}:
            continue
        if path.is_file():
            shutil.copy2(path, view_meta / path.name)

    info = _json_load(canonical_meta / "info.json")
    info["data_path"] = info["data_path"].replace("{chunk_index", "{episode_chunk")
    if info.get("video_path"):
        info["video_path"] = info["video_path"].replace("{chunk_index", "{episode_chunk")
    _json_dump_new(view_meta / "info.json", info)

    modality = _json_load(canonical_meta / "modality.json")
    head = deepcopy(modality["video"]["cam_left_high"])
    head["original_key"] = "observation.images.cam_left_high"
    modality["video"] = {"ego_view": head}
    modality["state"] = {key: deepcopy(modality["state"][key]) for key in STATE_SLICES}
    modality["action"] = {key: deepcopy(modality["action"][key]) for key in ACTION_SLICES}
    modality["annotation"] = {"human.task_description": {"original_key": "task_index"}}
    _json_dump_new(view_meta / "modality.json", modality)
    with (view_meta / "tasks.jsonl").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"task_index": 0, "task": LANGUAGE}, sort_keys=True) + "\n")

    for name in ("data", "videos"):
        target = canonical_root / name
        if not target.exists():
            raise FileNotFoundError(target)
        (view_root / name).symlink_to(target, target_is_directory=True)

    manifest = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "temporal_option": TEMPORAL_OPTION,
        "row_grid_rate_hz": 30,
        "true_30hz_hardware_clock_proven": False,
        "true_50hz_capture": False,
        "horizon": HORIZON,
        "source_index_formula": "n+k",
        "canonical_root": str(canonical_root),
        "canonical_info_sha256": _sha256(canonical_meta / "info.json"),
        "canonical_modality_sha256": _sha256(canonical_meta / "modality.json"),
        "camera_mapping": {
            "model_key": "ego_view",
            "canonical_key": "cam_left_high",
            "original_key": "observation.images.cam_left_high",
        },
        "language": LANGUAGE,
        "payload_policy": "read-only symlink to canonical data/videos",
    }
    _json_dump_new(view_meta / "g1_id10_h40_native30_view_manifest.json", manifest)
    return manifest


def load_adapter_config(path: str | Path) -> dict[str, Any]:
    cfg = _json_load(Path(path))
    required = {
        "schema_version",
        "adapter_kind",
        "baseline_id",
        "embodiment_tag",
        "embodiment_id",
        "temporal_option",
        "source_index_formula",
        "row_grid_rate_hz",
        "true_30hz_hardware_clock_proven",
        "true_50hz_capture",
        "horizon",
        "active_action_dims",
        "hand_scope",
        "language",
        "runtime_verified_fixed_state",
        "fixed_state",
        "fixed_action",
        "derived_relative_stats_path",
    }
    optional = {"split_manifest_path", "split", "codec_evidence"}
    keys = set(cfg)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise ValueError(
            f"Adapter config schema mismatch: missing={sorted(required - keys)}, "
            f"unknown={sorted(keys - required - optional)}"
        )
    present_split = keys & {"split_manifest_path", "split"}
    if present_split and present_split != {"split_manifest_path", "split"}:
        raise ValueError("split_manifest_path and split must be supplied together")
    if present_split:
        if cfg["split"] not in {"train", "val", "test"}:
            raise ValueError("Adapter config split must be train, val, or test")
        if not str(cfg["split_manifest_path"]).strip():
            raise ValueError("split_manifest_path must be non-empty")
    codec_block = cfg.get("codec_evidence")
    if codec_block is not None:
        if not isinstance(codec_block, dict):
            raise TypeError("codec_evidence must be an object")
        codec_enabled = codec_block.get("enabled", False)
        if not isinstance(codec_enabled, bool):
            raise TypeError("codec_evidence.enabled must be boolean")
        if codec_enabled:
            from gr00t.data.codec_evidence.config import parse_codec_evidence_config

            parse_codec_evidence_config(codec_block)
            if not present_split:
                raise ValueError("Enabled codec evidence requires split manifest mode")
    exact = {
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
    }
    for key, expected in exact.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Adapter config {key}={cfg.get(key)!r}, expected {expected!r}")
    for key, dim in FIXED_STATE_DIMS.items():
        values = np.asarray(cfg.get("fixed_state", {}).get(key), dtype=np.float32)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"fixed_state.{key} must be finite shape {(dim,)}")
    for key, dim in FIXED_ACTION_DIMS.items():
        values = np.asarray(cfg.get("fixed_action", {}).get(key), dtype=np.float32)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"fixed_action.{key} must be finite shape {(dim,)}")
    return cfg


def _train_uid_digest(assignments: list[dict[str, Any]]) -> str:
    values = sorted(
        row["episode_uid"]
        for row in assignments
        if row.get("eligible") and row.get("split") == "train"
    )
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _episode_uid_from_catalog(row: dict[str, Any]) -> str:
    value = (
        f"icra27-episode-v1\0{row['dataset_name']}\0{int(row['episode_index'])}\0"
        f"{row['parquet_sha256']}\0{row['cam_left_high_sha256']}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _catalog_family_ids(payload_episodes: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for field in ("state_action_sha256", "state_action_round1e5_sha256"):
        seen: dict[tuple[int, str], str] = {}
        for row in payload_episodes:
            identity = (int(row["rows"]), row[field])
            previous = seen.setdefault(identity, row["cohort_id"])
            if previous != row["cohort_id"]:
                raise ValueError("Current catalog contradicts independent cohort identity")
    for row in payload_episodes:
        by_cohort[row["cohort_id"]].append(row)
    for cohort_rows in by_cohort.values():
        parent = list(range(len(cohort_rows)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for field in ("state_action_sha256", "state_action_round1e5_sha256"):
            seen: dict[tuple[int, str], int] = {}
            for index, row in enumerate(cohort_rows):
                identity = (int(row["rows"]), row[field])
                previous = seen.setdefault(identity, index)
                union(previous, index)
        components: dict[int, list[str]] = defaultdict(list)
        for index, row in enumerate(cohort_rows):
            components[find(index)].append(row["episode_uid"])
        for member_uids in components.values():
            members = sorted(member_uids)
            family_id = hashlib.sha256(
                ("icra27-family-v1\n" + "\n".join(members) + "\n").encode("utf-8")
            ).hexdigest()
            for uid in members:
                result[uid] = family_id
    return result


def _rank_key(catalog_sha256: str, cohort_id: str, family_id: str) -> str:
    value = (
        f"{SPLIT_MANIFEST_ID}\0{catalog_sha256}\0{cohort_id}\0"
        f"{family_id}\0{SPLIT_SEED}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expected_split_by_uid(
    payload_episodes: list[dict[str, Any]],
    family_ids: dict[str, str],
    catalog_sha256: str,
) -> dict[str, str]:
    """Reproduce the frozen split algorithm instead of trusting split labels."""

    result: dict[str, str] = {}
    for cohort_id, expected in EXPECTED_COHORTS.items():
        families: dict[str, list[str]] = defaultdict(list)
        for row in payload_episodes:
            if row["cohort_id"] == cohort_id:
                families[family_ids[row["episode_uid"]]].append(row["episode_uid"])
        family_rows = [
            {
                "family_id": family_id,
                "members": sorted(members),
                "size": len(members),
                "rank_key": _rank_key(catalog_sha256, cohort_id, family_id),
            }
            for family_id, members in families.items()
        ]
        targets = expected["split_counts"]
        assigned: list[tuple[dict[str, Any], str]] = []
        if all(row["size"] == 1 for row in family_rows):
            family_rows.sort(key=lambda row: row["rank_key"])
            train_end = targets["train"]
            val_end = train_end + targets["val"]
            for index, family in enumerate(family_rows):
                split = "train" if index < train_end else "val" if index < val_end else "test"
                assigned.append((family, split))
        else:
            counts = {label: 0 for label in ("train", "val", "test")}
            labels = ("train", "val", "test")
            for family in sorted(
                family_rows, key=lambda row: (-row["size"], row["rank_key"])
            ):
                candidates = []
                for label_index, label in enumerate(labels):
                    trial = dict(counts)
                    trial[label] += family["size"]
                    score = (
                        sum(abs(trial[name] - targets[name]) for name in labels),
                        max(abs(trial[name] - targets[name]) for name in labels),
                        abs(trial["val"] - trial["test"]),
                        label_index,
                    )
                    candidates.append((score, label))
                split = min(candidates)[1]
                counts[split] += family["size"]
                assigned.append((family, split))
        for family, split in assigned:
            for uid in family["members"]:
                result[uid] = split
    return result


def _bind_payload_to_catalog(
    catalog_path: Path,
    *,
    dataset_root: Path,
    catalog_identity: dict[str, Any],
    payload_episodes: list[dict[str, Any]],
) -> None:
    """Recompute every frozen episode identity from the current SQLite catalog."""

    with sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Current dataset catalog quick_check failed")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("run_id") != catalog_identity.get("run_id"):
            raise ValueError("Split manifest catalog run ID differs from SQLite")
        if metadata.get("root") != str(dataset_root):
            raise ValueError("Split manifest dataset root differs from SQLite")
        required_episode_columns = {
            "relative_path",
            "dataset_name",
            "rows",
            "state_sha256",
            "action_sha256",
            "state_action_sha256",
            "state_action_round1e5_sha256",
            "state_dim",
            "action_dim",
            "finite_ok",
            "frame_index_ok",
            "episode_index_ok",
            "timestamp_30hz_ok",
            "validity_ok",
            "task_index_zero_ok",
        }
        actual_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(lerobot_episodes)")
        }
        if not required_episode_columns.issubset(actual_columns):
            raise ValueError("Current catalog lacks required episode audit columns")

        info_cache: dict[str, dict[str, Any]] = {}
        for payload_row in payload_episodes:
            dataset_name = payload_row["dataset_name"]
            episode_index = int(payload_row["episode_index"])
            info = info_cache.setdefault(
                dataset_name,
                _json_load(dataset_root / "lerobot" / dataset_name / "meta/info.json"),
            )
            expected_parquet = "lerobot/" + dataset_name + "/" + _format_episode_path(
                info["data_path"], episode_index, int(info["chunks_size"])
            )
            expected_video = "lerobot/" + dataset_name + "/" + _format_episode_path(
                info["video_path"],
                episode_index,
                int(info["chunks_size"]),
                video_key="observation.images.cam_left_high",
            )
            if payload_row["parquet_relative_path"] != expected_parquet:
                raise ValueError("Payload Parquet path does not derive from dataset/episode")
            if payload_row["cam_left_high_relative_path"] != expected_video:
                raise ValueError("Payload video path does not derive from dataset/episode")
            for path_field, sha_field, bytes_field in (
                ("parquet_relative_path", "parquet_sha256", "parquet_bytes"),
                (
                    "cam_left_high_relative_path",
                    "cam_left_high_sha256",
                    "cam_left_high_bytes",
                ),
            ):
                file_row = connection.execute(
                    "SELECT sha256,size FROM files "
                    "WHERE relative_path=? AND entry_type='file'",
                    (payload_row[path_field],),
                ).fetchone()
                if file_row is None:
                    raise ValueError("Payload manifest references a file absent from catalog")
                if file_row != (payload_row[sha_field], int(payload_row[bytes_field])):
                    raise ValueError("Payload file hash/size differs from current catalog")

            parquet_relative = payload_row["parquet_relative_path"]
            if not parquet_relative.startswith("lerobot/"):
                raise ValueError("Payload Parquet path is outside the LeRobot partition")
            logical = connection.execute(
                """
                SELECT dataset_name,rows,state_sha256,action_sha256,
                       state_action_sha256,state_action_round1e5_sha256,
                       state_dim,action_dim,finite_ok,frame_index_ok,
                       episode_index_ok,timestamp_30hz_ok,validity_ok,
                       task_index_zero_ok
                FROM lerobot_episodes WHERE relative_path=?
                """,
                (parquet_relative.removeprefix("lerobot/"),),
            ).fetchone()
            if logical is None:
                raise ValueError("Payload episode is absent from the current logical catalog")
            expected_logical = (
                payload_row["dataset_name"],
                int(payload_row["rows"]),
                payload_row["state_sha256"],
                payload_row["action_sha256"],
                payload_row["state_action_sha256"],
                payload_row["state_action_round1e5_sha256"],
                26,
                26,
                1,
                1,
                1,
                1,
                1,
                1,
            )
            if logical != expected_logical:
                raise ValueError("Payload episode identity/audit differs from current catalog")
            if payload_row["episode_uid"] != _episode_uid_from_catalog(payload_row):
                raise ValueError("Payload episode UID does not derive from current catalog")


def _load_split_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    manifest = _json_load(path)
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Dataset split-manifest schema mismatch")
    if manifest.get("contract_id") != DATASET_CONTRACT_ID:
        raise ValueError("Dataset contract ID mismatch")
    if manifest.get("split_manifest_id") != SPLIT_MANIFEST_ID:
        raise ValueError("Dataset split-manifest ID mismatch")
    if manifest.get("split_algorithm_id") != SPLIT_ALGORITHM_ID:
        raise ValueError("Dataset split algorithm mismatch")
    if manifest.get("split_seed") != SPLIT_SEED:
        raise ValueError("Dataset split seed mismatch")
    if manifest.get("selected_camera_key") != "observation.images.cam_left_high":
        raise ValueError("Split manifest must select cam_left_high")
    if manifest.get("contract_instruction") != LANGUAGE:
        raise ValueError("Split manifest instruction differs from the frozen literal")

    expected_payload_sha256 = manifest.get("payload_manifest_sha256")
    if not isinstance(expected_payload_sha256, str) or len(expected_payload_sha256) != 64:
        raise ValueError("Split manifest has no valid payload-manifest SHA")
    declared_payload_path = Path(str(manifest.get("payload_manifest_path", ""))).resolve()
    sibling_payload_path = path.with_name("payload_manifest.json")
    payload_path = (
        sibling_payload_path
        if sibling_payload_path.is_file()
        and _sha256(sibling_payload_path) == expected_payload_sha256
        else declared_payload_path
    )
    if not payload_path.is_file() or _sha256(payload_path) != expected_payload_sha256:
        raise ValueError("Split manifest payload binding is missing or has changed")
    payload = _json_load(payload_path)
    if payload.get("schema_version") != PAYLOAD_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Dataset payload-manifest schema mismatch")
    if payload.get("contract_id") != DATASET_CONTRACT_ID:
        raise ValueError("Payload dataset contract ID mismatch")
    if payload.get("selected_camera_key") != manifest["selected_camera_key"]:
        raise ValueError("Payload and split manifest camera contracts differ")
    if payload.get("contract_instruction") != manifest["contract_instruction"]:
        raise ValueError("Payload and split manifest instructions differ")
    if payload.get("catalog") != manifest.get("catalog"):
        raise ValueError("Payload and split manifest catalog identities differ")
    if Path(str(payload.get("dataset_root", ""))).resolve() != Path(
        str(manifest.get("dataset_root", ""))
    ).resolve():
        raise ValueError("Payload and split manifest dataset roots differ")

    dataset_root = Path(manifest["dataset_root"]).resolve()
    stale_marker = dataset_root / "catalog/STALE_AFTER_SYNC.json"
    if stale_marker.exists():
        raise ValueError("Dataset catalog, manifest, and statistics are stale after rsync")
    catalog_path = Path(manifest["catalog"]["database_path"]).resolve()
    expected_catalog = dataset_root / "catalog/current.sqlite3"
    if catalog_path != expected_catalog or not catalog_path.is_file():
        raise ValueError("Split manifest catalog path is not the current dataset catalog")
    if _sha256(catalog_path) != manifest["catalog"]["database_sha256"]:
        raise ValueError("Split manifest is stale for the current catalog identity")

    payload_episodes = payload.get("episodes")
    if not isinstance(payload_episodes, list) or not payload_episodes:
        raise ValueError("Payload manifest has no episodes")
    if payload.get("cohorts") != manifest.get("cohorts"):
        raise ValueError("Payload and split manifest cohort records differ")
    cohort_records = manifest.get("cohorts")
    if not isinstance(cohort_records, list):
        raise ValueError("Split manifest has no cohort records")
    cohort_by_id = {
        row.get("cohort_id"): row for row in cohort_records if isinstance(row, dict)
    }
    if set(cohort_by_id) != set(EXPECTED_COHORTS):
        raise ValueError("Split manifest does not contain the fixed R1 cohorts")
    for cohort_id, expected in EXPECTED_COHORTS.items():
        record = cohort_by_id[cohort_id]
        if (
            record.get("dataset_name") != expected["dataset_name"]
            or record.get("expected_episodes") != expected["expected_episodes"]
        ):
            raise ValueError("Split manifest cohort identity/count mismatch")
    payload_by_uid: dict[str, dict[str, Any]] = {}
    payload_sources: set[tuple[str, int]] = set()
    required_payload_fields = {
        "episode_uid",
        "cohort_id",
        "dataset_name",
        "episode_index",
        "rows",
        "parquet_relative_path",
        "parquet_sha256",
        "parquet_bytes",
        "cam_left_high_relative_path",
        "cam_left_high_sha256",
        "cam_left_high_bytes",
        "state_sha256",
        "action_sha256",
        "state_action_sha256",
        "state_action_round1e5_sha256",
    }
    for row in payload_episodes:
        if not isinstance(row, dict) or not required_payload_fields.issubset(row):
            raise ValueError("Payload episode record is incomplete")
        uid = row["episode_uid"]
        source = (row["dataset_name"], int(row["episode_index"]))
        if not isinstance(uid, str) or not uid or uid in payload_by_uid:
            raise ValueError("Payload episode UID is empty or duplicated")
        if source in payload_sources:
            raise ValueError("Payload dataset/episode source is duplicated")
        payload_by_uid[uid] = row
        payload_sources.add(source)
    for cohort_id, expected in EXPECTED_COHORTS.items():
        cohort_rows = [row for row in payload_episodes if row["cohort_id"] == cohort_id]
        if len(cohort_rows) != expected["expected_episodes"]:
            raise ValueError("Payload does not completely cover a fixed R1 cohort")
        if {row["dataset_name"] for row in cohort_rows} != {expected["dataset_name"]}:
            raise ValueError("Payload dataset-to-cohort mapping mismatch")
        if {int(row["episode_index"]) for row in cohort_rows} != set(
            range(expected["expected_episodes"])
        ):
            raise ValueError("Payload cohort episode-index coverage is incomplete")
    _bind_payload_to_catalog(
        catalog_path,
        dataset_root=dataset_root,
        catalog_identity=manifest["catalog"],
        payload_episodes=payload_episodes,
    )
    catalog_family_ids = _catalog_family_ids(payload_episodes)
    expected_split_by_uid = _expected_split_by_uid(
        payload_episodes,
        catalog_family_ids,
        manifest["catalog"]["database_sha256"],
    )

    assignments = manifest.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Split manifest has no assignments")
    if len(assignments) != len(payload_by_uid):
        raise ValueError("Split assignments do not cover the payload exactly once")
    assignment_by_uid: dict[str, dict[str, Any]] = {}
    assignment_sources: set[tuple[str, int]] = set()
    family_splits: dict[str, set[str]] = defaultdict(set)
    actual_global = {label: 0 for label in ("train", "val", "test")}
    actual_by_cohort: dict[str, dict[str, int]] = defaultdict(
        lambda: {label: 0 for label in ("train", "val", "test")}
    )
    required_assignment_fields = {
        "episode_uid",
        "cohort_id",
        "dataset_name",
        "episode_index",
        "rows",
        "family_id",
        "family_evidence",
        "session_id",
        "session_id_source",
        "eligible",
        "exclusion_reason",
        "split",
        "rank_key",
    }
    for row in assignments:
        if not isinstance(row, dict) or not required_assignment_fields.issubset(row):
            raise ValueError("Split assignment record is incomplete")
        uid = row["episode_uid"]
        split = row["split"]
        source = (row["dataset_name"], int(row["episode_index"]))
        if uid in assignment_by_uid or source in assignment_sources:
            raise ValueError("Split assignment UID or source is duplicated")
        if uid not in payload_by_uid:
            raise ValueError("Split assignment UID is absent from the payload manifest")
        payload_row = payload_by_uid[uid]
        for field in ("cohort_id", "dataset_name", "episode_index", "rows"):
            if row[field] != payload_row[field]:
                raise ValueError(f"Split assignment {field} differs from payload")
        if row["eligible"] is not True or row["exclusion_reason"] is not None:
            raise ValueError("R1 split assignments must all be eligible")
        if split not in actual_global:
            raise ValueError("Split assignment has an invalid split")
        if split != expected_split_by_uid[uid]:
            raise ValueError("Split label does not derive from the frozen algorithm")
        if not isinstance(row["family_id"], str) or not row["family_id"]:
            raise ValueError("Split assignment has no family identity")
        if row["family_id"] != catalog_family_ids[uid]:
            raise ValueError("Split family ID does not derive from the current catalog")
        if row["rank_key"] != _rank_key(
            manifest["catalog"]["database_sha256"],
            row["cohort_id"],
            row["family_id"],
        ):
            raise ValueError("Split rank key does not derive from the current catalog")
        if not isinstance(row["family_evidence"], list) or not row["family_evidence"]:
            raise ValueError("Split assignment has no family evidence")
        family_splits[row["family_id"]].add(split)
        actual_global[split] += 1
        actual_by_cohort[row["cohort_id"]][split] += 1
        assignment_by_uid[uid] = row
        assignment_sources.add(source)
    if set(assignment_by_uid) != set(payload_by_uid):
        raise ValueError("Split assignment UID set differs from the payload")
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise ValueError("Split manifest leaks a family across splits")
    if manifest.get("actual_counts_global") != actual_global:
        raise ValueError("Split manifest global counts are inconsistent")
    if manifest.get("actual_counts_by_cohort") != dict(actual_by_cohort):
        raise ValueError("Split manifest cohort counts are inconsistent")
    expected_split_counts = {
        cohort_id: value["split_counts"]
        for cohort_id, value in EXPECTED_COHORTS.items()
    }
    if manifest.get("target_counts_by_cohort") != expected_split_counts:
        raise ValueError("Split target counts do not match the fixed R1 cohort contract")
    if dict(actual_by_cohort) != expected_split_counts:
        raise ValueError("Split counts do not match the fixed R1 cohort contract")
    summary = manifest.get("validation_summary")
    required_zero_summary = {
        "cross_split_family_intersection",
        "cross_split_exact_trajectory_intersection",
        "cross_split_rounded_trajectory_intersection",
        "unresolved_parent_count",
        "physical_payload_mutations",
    }
    if not isinstance(summary, dict) or any(
        summary.get(key) != 0 for key in required_zero_summary
    ):
        raise ValueError("Split manifest validation summary is incomplete or non-zero")
    if summary.get("eligible_episode_count") != len(assignments):
        raise ValueError("Split manifest eligible episode count is inconsistent")
    if _train_uid_digest(assignments) != manifest.get("train_episode_uid_digest"):
        raise ValueError("Split manifest train UID digest mismatch")
    return manifest, _sha256(path)


def _validate_train_statistics(
    stats: dict[str, Any],
    *,
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
) -> dict[str, Any]:
    if stats.get("schema_version") != TRAIN_STATS_SCHEMA_VERSION:
        raise ValueError("Train-only statistics schema mismatch")
    if stats.get("fit_split") != "train":
        raise ValueError("Normalization and relative statistics must be fit on train only")
    if stats.get("split_manifest_sha256") != split_manifest_sha256:
        raise ValueError("Train statistics were fit for a different split manifest")
    if stats.get("train_episode_uid_digest") != split_manifest["train_episode_uid_digest"]:
        raise ValueError("Train statistics use a different train episode set")
    if stats.get("catalog_database_sha256") != split_manifest["catalog"]["database_sha256"]:
        raise ValueError("Train statistics are stale for the current catalog")
    for field in ("observation.state", "action"):
        source = stats.get("normalization", {}).get(field, {})
        for name in STAT_NAMES:
            values = np.asarray(source.get(name), dtype=np.float64)
            if values.shape != (ACTIVE_ACTION_DIM,) or not np.isfinite(values).all():
                raise ValueError(f"normalization.{field}.{name} must be finite [26]")
    _validate_relative_stats(stats)
    return stats


def _validate_relative_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if stats.get("schema_version") not in {
        STATS_SCHEMA_VERSION,
        TRAIN_STATS_SCHEMA_VERSION,
    }:
        raise ValueError("H40 relative-statistics schema mismatch")
    if stats.get("temporal_option") != TEMPORAL_OPTION:
        raise ValueError("H40 relative statistics use a different row-grid contract")
    if stats.get("true_30hz_hardware_clock_proven") is not False:
        raise ValueError("H40 statistics overclaim the hardware clock")
    result: dict[str, Any] = {}
    for key in ("left_arm", "right_arm"):
        result[key] = {}
        for name in STAT_NAMES:
            values = np.asarray(stats.get("relative_action", {}).get(key, {}).get(name), dtype=np.float64)
            if values.shape != (HORIZON, 7) or not np.isfinite(values).all():
                raise ValueError(f"relative_action.{key}.{name} must be finite [40,7]")
            result[key][name] = values.tolist()
    return result


class G1Id10H40Native30EpisodeLoader(G1Id10H50EpisodeLoader):
    """H40/native-row-grid loader; H50 temporal evidence is never accepted."""

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        adapter_config_path: str | Path,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        require_runtime_verified_fixed_state: bool = True,
    ) -> None:
        self.adapter_config_path = Path(adapter_config_path)
        self.adapter_config = load_adapter_config(self.adapter_config_path)
        self._manifest_mode = "split_manifest_path" in self.adapter_config
        self.split_manifest: dict[str, Any] | None = None
        self.split_manifest_sha256: str | None = None
        self.train_statistics: dict[str, Any] | None = None
        if self._manifest_mode:
            self.split_manifest, self.split_manifest_sha256 = _load_split_manifest(
                self.adapter_config["split_manifest_path"]
            )
        if require_runtime_verified_fixed_state and not self.adapter_config.get(
            "runtime_verified_fixed_state", False
        ):
            raise RuntimeError(
                "Training requires runtime-verified fixed leg/waist state; CPU fixtures may opt out"
            )
        LeRobotEpisodeLoader.__init__(
            self,
            dataset_path,
            modality_configs,
            video_backend,
            video_backend_kwargs,
        )
        if self._manifest_mode:
            assert self.split_manifest is not None
            split = self.adapter_config["split"]
            selected = {
                int(row["episode_index"]): row
                for row in self.split_manifest["assignments"]
                if row.get("eligible")
                and row.get("split") == split
                and row.get("dataset_name") == self.dataset_path.name
            }
            if not selected:
                raise ValueError(
                    f"Split {split!r} selects no episodes for {self.dataset_path.name!r}"
                )
            metadata_by_index = {
                int(row["episode_index"]): row for row in self.episodes_metadata
            }
            missing = sorted(set(selected) - set(metadata_by_index))
            if missing:
                raise ValueError(f"Split manifest references missing episodes: {missing[:10]}")
            self.episodes_metadata = [metadata_by_index[index] for index in sorted(selected)]
            self.episode_lengths = self.get_episode_lengths()
            self.selected_episode_uids = [selected[index]["episode_uid"] for index in sorted(selected)]
            self.train_statistics = _validate_train_statistics(
                _json_load(Path(self.adapter_config["derived_relative_stats_path"])),
                split_manifest=self.split_manifest,
                split_manifest_sha256=self.split_manifest_sha256,
            )
        else:
            manifest = _json_load(
                self.dataset_path / "meta/g1_id10_h40_native30_view_manifest.json"
            )
            if manifest.get("schema_version") != VIEW_SCHEMA_VERSION:
                raise ValueError("Dataset is not an ID10/H40/native30 loader view")
            if manifest.get("true_30hz_hardware_clock_proven") is not False:
                raise ValueError("Loader view overclaims a high-precision hardware clock")

    def _load_metadata(self) -> None:
        LeRobotEpisodeLoader._load_metadata(self)
        if not self._manifest_mode:
            return
        self.data_path_pattern = self.data_path_pattern.replace(
            "{chunk_index", "{episode_chunk"
        )
        if self.video_path_pattern:
            self.video_path_pattern = self.video_path_pattern.replace(
                "{chunk_index", "{episode_chunk"
            )
        for section in ("state", "action", "video"):
            if section not in self.modality_meta:
                raise KeyError(f"Canonical modality.json lacks {section}")
        head = deepcopy(self.modality_meta["video"]["cam_left_high"])
        head["original_key"] = "observation.images.cam_left_high"
        self.modality_meta["video"] = {"ego_view": head}
        self.modality_meta["state"] = {
            key: deepcopy(self.modality_meta["state"][key]) for key in STATE_SLICES
        }
        self.modality_meta["action"] = {
            key: deepcopy(self.modality_meta["action"][key]) for key in ACTION_SLICES
        }
        self.modality_meta["annotation"] = {
            "human.task_description": {"original_key": "task_index"}
        }
        self.tasks_map = {0: LANGUAGE}

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        if not self._manifest_mode:
            return G1Id10H50EpisodeLoader._load_parquet_data(self, episode_index)
        filename = _format_episode_path(
            self.data_path_pattern, episode_index, self.chunk_size
        )
        raw = pd.read_parquet(self.dataset_path / filename)
        required = {
            "observation.state",
            "action",
            "frame_index",
            "episode_index",
            "task_index",
            "annotation.human.action.task_description",
            "annotation.human.validity",
        }
        missing = required.difference(raw.columns)
        if missing:
            raise KeyError(f"Canonical parquet columns missing: {sorted(missing)}")
        states = _stack_float32_rows(raw["observation.state"], field="observation.state")
        actions = _stack_float32_rows(raw["action"], field="action")
        if states.shape != (len(raw), ACTIVE_ACTION_DIM):
            raise ValueError(f"Expected canonical state [L,26], got {states.shape}")
        if actions.shape != (len(raw), ACTIVE_ACTION_DIM):
            raise ValueError(f"Expected canonical action [L,26], got {actions.shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite canonical state/action")
        if not np.array_equal(raw["frame_index"].to_numpy(), np.arange(len(raw))):
            raise ValueError("frame_index must be zero-based and contiguous within episode")
        if not np.all(raw["episode_index"].to_numpy() == episode_index):
            raise ValueError("episode_index mismatch")
        if not np.all(raw["task_index"].to_numpy() == 0):
            raise ValueError("Only frozen task_index=0 is accepted")
        observed_language = raw["annotation.human.action.task_description"]
        if observed_language.isna().any() or not all(
            isinstance(value, str) and value.strip() for value in observed_language
        ):
            raise ValueError("Raw language provenance must be a non-empty string")
        if not np.all(raw["annotation.human.validity"].to_numpy() == 1):
            raise ValueError("Invalid demonstration row encountered")
        for slc in (slice(14, 20), slice(20, 26)):
            values = np.concatenate((states[:, slc], actions[:, slc]), axis=0)
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError("Inspire hand command u must be within [0,1]")

        result = pd.DataFrame(index=raw.index)
        result["language.annotation.human.task_description"] = [LANGUAGE] * len(raw)
        for key, dim in FIXED_STATE_DIMS.items():
            value = np.asarray(self.adapter_config["fixed_state"][key], dtype=np.float32)
            result[f"state.{key}"] = [value.copy() for _ in range(len(raw))]
        for key, (start, end) in STATE_SLICES.items():
            result[f"state.{key}"] = [row.copy() for row in states[:, start:end]]
        for key, (start, end) in ACTION_SLICES.items():
            result[f"action.{key}"] = [row.copy() for row in actions[:, start:end]]
        for key, dim in FIXED_ACTION_DIMS.items():
            value = np.asarray(self.adapter_config["fixed_action"][key], dtype=np.float32)
            result[f"action.{key}"] = [value.copy() for _ in range(len(raw))]
        return result

    def get_dataset_statistics(self) -> dict[str, Any]:
        if self._manifest_mode:
            assert self.train_statistics is not None
            canonical_state = self.train_statistics["normalization"]["observation.state"]
            canonical_action = self.train_statistics["normalization"]["action"]
        else:
            canonical_state = self.stats["observation.state"]
            canonical_action = self.stats["action"]
        result: dict[str, Any] = {"state": {}, "action": {}, "relative_action": {}}
        for key in FIXED_STATE_DIMS:
            result["state"][key] = _constant_stats(
                np.asarray(self.adapter_config["fixed_state"][key], dtype=np.float32)
            )
        for key, (start, end) in STATE_SLICES.items():
            result["state"][key] = _slice_stats(canonical_state, start, end)
        for key, (start, end) in ACTION_SLICES.items():
            result["action"][key] = _slice_stats(canonical_action, start, end)
        for key in FIXED_ACTION_DIMS:
            result["action"][key] = _constant_stats(
                np.asarray(self.adapter_config["fixed_action"][key], dtype=np.float32)
            )
        relative_source = (
            self.train_statistics
            if self._manifest_mode
            else _json_load(Path(self.adapter_config["derived_relative_stats_path"]))
        )
        assert relative_source is not None
        relative = _validate_relative_stats(relative_source)
        result["relative_action"] = relative
        return result


class G1Id10H40ManifestEpisodeCollection:
    """One logical split spanning the three immutable LeRobot cohort roots."""

    def __init__(
        self,
        collection_root: str | Path,
        modality_configs: dict[str, ModalityConfig],
        adapter_config_path: str | Path,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        require_runtime_verified_fixed_state: bool = True,
        dataset_generation_guard: Any | None = None,
    ) -> None:
        self.dataset_generation_guard = dataset_generation_guard
        if self.dataset_generation_guard is not None:
            self.dataset_generation_guard.ensure_current_process()
        self.collection_root = Path(collection_root).resolve()
        self.adapter_config = load_adapter_config(adapter_config_path)
        if "split_manifest_path" not in self.adapter_config:
            raise ValueError("Manifest collection requires split_manifest_path")
        manifest, _ = _load_split_manifest(self.adapter_config["split_manifest_path"])
        expected_collection = Path(manifest["dataset_root"]).resolve() / "lerobot"
        if self.collection_root != expected_collection:
            raise ValueError(
                f"Manifest collection root must be {expected_collection}, got {self.collection_root}"
            )
        selected_names = {
            row["dataset_name"]
            for row in manifest["assignments"]
            if row.get("eligible") and row.get("split") == self.adapter_config["split"]
        }
        self.loaders: list[G1Id10H40Native30EpisodeLoader] = []
        self.mapping: list[tuple[int, int]] = []
        self.episodes_metadata: list[dict[str, Any]] = []
        self.episode_lengths: list[int] = []
        for cohort in manifest["cohorts"]:
            dataset_name = cohort["dataset_name"]
            if dataset_name not in selected_names:
                continue
            dataset_root = Path(manifest["dataset_root"]) / cohort["dataset_relative_path"]
            loader = G1Id10H40Native30EpisodeLoader(
                dataset_root,
                modality_configs,
                adapter_config_path,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
                require_runtime_verified_fixed_state=require_runtime_verified_fixed_state,
            )
            loader_index = len(self.loaders)
            self.loaders.append(loader)
            for local_index, metadata in enumerate(loader.episodes_metadata):
                self.mapping.append((loader_index, local_index))
                self.episode_lengths.append(loader.get_episode_length(local_index))
                self.episodes_metadata.append(
                    {
                        **metadata,
                        "dataset_name": dataset_name,
                        "source_episode_index": int(metadata["episode_index"]),
                    }
                )
        if not self.mapping:
            raise ValueError("Manifest split selected no episodes")

    def __len__(self) -> int:
        return len(self.mapping)

    def get_episode_length(self, index: int) -> int:
        return self.episode_lengths[index]

    def __getitem__(self, index: int) -> pd.DataFrame:
        if self.dataset_generation_guard is not None:
            self.dataset_generation_guard.ensure_current_process()
        loader_index, local_index = self.mapping[index]
        return self.loaders[loader_index][local_index]

    def _load_parquet_data(self, index: int) -> pd.DataFrame:
        if self.dataset_generation_guard is not None:
            self.dataset_generation_guard.ensure_current_process()
        loader_index, local_index = self.mapping[index]
        loader = self.loaders[loader_index]
        episode_index = int(loader.episodes_metadata[local_index]["episode_index"])
        return loader._load_parquet_data(episode_index)

    def get_dataset_statistics(self) -> dict[str, Any]:
        first = self.loaders[0].get_dataset_statistics()
        for loader in self.loaders[1:]:
            if loader.get_dataset_statistics() != first:
                raise ValueError("Manifest cohort loaders do not share identical train statistics")
        return first

    def get_initial_actions(self) -> list[Any]:
        return []


class G1Id10H40Native30SingleStepDataset(ShardedSingleStepDataset):
    def __init__(self, *args, adapter_config_path: str | Path, **kwargs) -> None:
        adapter_config = load_adapter_config(adapter_config_path)
        if "split_manifest_path" not in adapter_config:
            # This branch is intentionally unchanged: the original superclass
            # sharding and loader replacement preserve the old 300-episode path.
            super().__init__(*args, **kwargs)
            self.episode_loader = G1Id10H40Native30EpisodeLoader(
                dataset_path=self.dataset_path,
                modality_configs=self.modality_configs,
                adapter_config_path=adapter_config_path,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
                require_runtime_verified_fixed_state=True,
            )
            return

        codec_enabled = bool(adapter_config.get("codec_evidence", {}).get("enabled", False))
        self.dataset_generation_guard = None
        if codec_enabled:
            # Import and acquire only for the explicit enabled path.  Missing
            # codec config, enabled=false, and the legacy no-manifest adapter
            # retain their historical zero-generation-guard behavior.
            from gr00t.data.dataset.dataset_generation_guard import DatasetGenerationGuard

            self.dataset_generation_guard = DatasetGenerationGuard.acquire_for_manifest(
                adapter_config["split_manifest_path"],
                adapter_config["derived_relative_stats_path"],
                blocking=False,
            )

        try:
            dataset_path = kwargs.pop("dataset_path", args[0] if args else None)
            if dataset_path is None:
                raise TypeError("dataset_path is required")
            embodiment_tag = kwargs.pop("embodiment_tag")
            modality_configs = kwargs.pop("modality_configs")
            video_backend = kwargs.pop("video_backend", "torchcodec")
            video_backend_kwargs = kwargs.pop("video_backend_kwargs", None)
            shard_size = kwargs.pop("shard_size", 2**10)
            episode_sampling_rate = kwargs.pop("episode_sampling_rate", 0.1)
            seed = kwargs.pop("seed", 42)
            allow_padding = kwargs.pop("allow_padding", False)
            if kwargs or len(args) > 1:
                raise TypeError(f"Unexpected manifest dataset arguments: {sorted(kwargs)}")
            self.dataset_path = Path(dataset_path)
            self.embodiment_tag = embodiment_tag
            self.modality_configs = modality_configs
            self.video_backend = video_backend
            self.video_backend_kwargs = video_backend_kwargs
            self.shard_size = shard_size
            self.episode_sampling_rate = episode_sampling_rate
            self.seed = seed
            self.allow_padding = allow_padding
            self.processor = None
            self.rng = np.random.default_rng(seed)
            action_delta_indices = modality_configs["action"].delta_indices
            self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            self.episode_loader = G1Id10H40ManifestEpisodeCollection(
                collection_root=self.dataset_path,
                modality_configs=modality_configs,
                adapter_config_path=adapter_config_path,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
                require_runtime_verified_fixed_state=True,
                dataset_generation_guard=self.dataset_generation_guard,
            )
            self.shard_dataset()
        except BaseException:
            if self.dataset_generation_guard is not None:
                self.dataset_generation_guard.close()
            raise

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        item = super().get_datapoint(episode_data, step_index)
        return apply_active_action_mask(item)


def apply_active_action_mask(item: dict[str, Any]) -> dict[str, Any]:
    if "action_mask" not in item:
        raise KeyError("Processed training item has no action_mask")
    mask = item["action_mask"]
    if tuple(mask.shape) != (HORIZON, PADDED_DIM):
        raise ValueError(f"Expected action_mask [40,132], got {tuple(mask.shape)}")
    mask = mask.clone() if hasattr(mask, "clone") else np.array(mask, copy=True)
    mask[:, ACTIVE_ACTION_DIM:] = 0
    if not np.all(np.asarray(mask[:, :ACTIVE_ACTION_DIM]) == 1):
        raise ValueError("Active arm+hand dimensions must remain supervised")
    result = dict(item)
    result["action_mask"] = mask
    return result


def resolve_decoded_action_26d(
    decoded: dict[str, np.ndarray], *, clamp_hand: bool = True
) -> tuple[np.ndarray, int]:
    """Resolve one finite H40 decoded chunk to the active 26-D boundary.

    ``StateActionProcessor.unapply_action`` has already converted the relative
    arm outputs to absolute joint targets.  This helper therefore validates and
    concatenates the decoded groups without adding the anchor state a second
    time.  Inactive full-body groups are shape-checked and then excluded from
    the returned simulator-facing chunk.
    """

    required_dims = {
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 6,
        "right_hand": 6,
        **FIXED_ACTION_DIMS,
    }
    arrays: dict[str, np.ndarray] = {}
    for key, dim in required_dims.items():
        if key not in decoded:
            raise KeyError(f"Decoded action missing {key}")
        value = np.asarray(decoded[key], dtype=np.float32)
        if value.shape != (HORIZON, dim) or not np.isfinite(value).all():
            raise ValueError(f"Decoded {key} must be finite shape {(HORIZON, dim)}")
        arrays[key] = value

    clamp_count = 0
    hands = []
    for key in ("left_hand", "right_hand"):
        value = arrays[key]
        outside = (value < 0.0) | (value > 1.0)
        outside_count = int(outside.sum())
        clamp_count += outside_count
        if outside_count and not clamp_hand:
            raise ValueError("Decoded Inspire command outside [0,1]")
        hands.append(np.clip(value, 0.0, 1.0) if clamp_hand else value)

    active = np.concatenate(
        [arrays["left_arm"], arrays["right_arm"], hands[0], hands[1]], axis=-1
    )
    if active.shape != (HORIZON, ACTIVE_ACTION_DIM) or not np.isfinite(active).all():
        raise AssertionError("Resolved H40 active action must be finite [40,26]")
    return active, clamp_count


def _stats_for_trajectories(values: np.ndarray) -> dict[str, list[list[float]]]:
    if values.ndim != 3 or values.shape[1:] != (HORIZON, 7):
        raise ValueError(f"Expected [N,40,7], got {values.shape}")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _stats_for_rows(values: np.ndarray) -> dict[str, list[float]]:
    if values.ndim != 2 or values.shape[1] != ACTIVE_ACTION_DIM:
        raise ValueError(f"Expected [N,26], got {values.shape}")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_h40_train_statistics(
    split_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Fit absolute normalization and H40 relative statistics on train only."""

    split_manifest_path = Path(split_manifest_path).resolve()
    split_manifest, split_manifest_sha256 = _load_split_manifest(split_manifest_path)
    dataset_root = Path(split_manifest["dataset_root"]).resolve()
    output_path = Path(output_path)
    _assert_external_output(dataset_root, output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    train_assignments = [
        row
        for row in split_manifest["assignments"]
        if row.get("eligible") and row.get("split") == "train"
    ]
    if not train_assignments:
        raise ValueError("Split manifest has no eligible train episodes")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    relative = {"left_arm": [], "right_arm": []}
    anchor_count = 0
    offsets = np.asarray(h40_native30_offsets(), dtype=np.int64)
    info_cache: dict[str, dict[str, Any]] = {}
    for assignment in train_assignments:
        dataset_name = assignment["dataset_name"]
        dataset_path = dataset_root / "lerobot" / dataset_name
        info = info_cache.setdefault(dataset_name, _json_load(dataset_path / "meta/info.json"))
        episode_index = int(assignment["episode_index"])
        filename = _format_episode_path(
            info["data_path"], episode_index, int(info["chunks_size"])
        )
        frame = pd.read_parquet(
            dataset_path / filename,
            columns=["observation.state", "action", "annotation.human.validity"],
        )
        if not np.all(frame["annotation.human.validity"].to_numpy() == 1):
            raise ValueError(f"Invalid row in {dataset_name} episode {episode_index}")
        state = _stack_float32_rows(frame["observation.state"], field="observation.state")
        action = _stack_float32_rows(frame["action"], field="action")
        if state.shape != action.shape or state.shape[1] != ACTIVE_ACTION_DIM:
            raise ValueError(f"{dataset_name} episode {episode_index} state/action mismatch")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{dataset_name} episode {episode_index} is non-finite")
        hands = np.concatenate((state[:, 14:26], action[:, 14:26]), axis=0)
        if np.any(hands < 0.0) or np.any(hands > 1.0):
            raise ValueError(f"{dataset_name} episode {episode_index} hand u outside [0,1]")
        states.append(state)
        actions.append(action)
        n_anchors = len(frame) - SOURCE_SPAN + 1
        if n_anchors <= 0:
            continue
        target_indices = np.arange(n_anchors)[:, None] + offsets[None, :]
        for key, (start, end) in {"left_arm": (0, 7), "right_arm": (7, 14)}.items():
            absolute = action[target_indices, start:end]
            reference = state[:n_anchors, None, start:end]
            relative[key].append((absolute - reference).astype(np.float32))
        anchor_count += n_anchors
    if anchor_count == 0:
        raise ValueError("No legal H40 train anchors")

    result = {
        "schema_version": TRAIN_STATS_SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "temporal_option": TEMPORAL_OPTION,
        "source_index_formula": "n+k",
        "row_grid_rate_hz": 30,
        "true_30hz_hardware_clock_proven": False,
        "true_50hz_capture": False,
        "horizon": HORIZON,
        "fit_split": "train",
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_manifest_sha256,
        "train_episode_uid_digest": split_manifest["train_episode_uid_digest"],
        "catalog_database_sha256": split_manifest["catalog"]["database_sha256"],
        "episodes": len(train_assignments),
        "anchors": anchor_count,
        "normalization": {
            "observation.state": _stats_for_rows(np.concatenate(states, axis=0)),
            "action": _stats_for_rows(np.concatenate(actions, axis=0)),
        },
        "relative_action": {},
    }
    for key in ("left_arm", "right_arm"):
        result["relative_action"][key] = _stats_for_trajectories(
            np.concatenate(relative[key], axis=0)
        )
    _json_dump_new(output_path, result)
    return result


def compute_h40_relative_statistics(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Compute native-row-grid H40 arm-relative stats without input writes."""

    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path)
    _assert_external_output(dataset_root, output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    info_path = dataset_root / "meta/info.json"
    episodes_path = dataset_root / "meta/episodes.jsonl"
    info = _json_load(info_path)
    with episodes_path.open("r", encoding="utf-8") as stream:
        episodes = [json.loads(line) for line in stream]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    offsets = np.asarray(h40_native30_offsets(), dtype=np.int64)
    collected = {"left_arm": [], "right_arm": []}
    anchor_count = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        filename = _format_episode_path(info["data_path"], episode_index, int(info["chunks_size"]))
        frame = pd.read_parquet(
            dataset_root / filename,
            columns=["observation.state", "action", "annotation.human.validity"],
        )
        if not np.all(frame["annotation.human.validity"].to_numpy() == 1):
            raise ValueError(f"Invalid row in episode {episode_index}")
        state = _stack_float32_rows(frame["observation.state"], field="observation.state")
        action = _stack_float32_rows(frame["action"], field="action")
        if state.shape != action.shape or state.shape[1] != ACTIVE_ACTION_DIM:
            raise ValueError(f"Episode {episode_index} state/action shape mismatch")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Episode {episode_index} contains non-finite state/action")
        hands = np.concatenate((state[:, 14:26], action[:, 14:26]), axis=0)
        if np.any(hands < 0.0) or np.any(hands > 1.0):
            raise ValueError(f"Episode {episode_index} Inspire command is outside [0,1]")
        n_anchors = len(frame) - SOURCE_SPAN + 1
        if n_anchors <= 0:
            continue
        target_indices = np.arange(n_anchors)[:, None] + offsets[None, :]
        for key, (start, end) in {"left_arm": (0, 7), "right_arm": (7, 14)}.items():
            absolute = action[target_indices, start:end]
            reference = state[:n_anchors, None, start:end]
            collected[key].append((absolute - reference).astype(np.float32))
        anchor_count += n_anchors
    if anchor_count == 0:
        raise ValueError("No legal H40 anchors")

    result = {
        "schema_version": STATS_SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "temporal_option": TEMPORAL_OPTION,
        "source_index_formula": "n+k",
        "row_grid_rate_hz": 30,
        "true_30hz_hardware_clock_proven": False,
        "true_50hz_capture": False,
        "horizon": HORIZON,
        "episodes": len(episodes),
        "anchors": anchor_count,
        "canonical_info_sha256": _sha256(info_path),
        "relative_action": {},
    }
    for key in ("left_arm", "right_arm"):
        result["relative_action"][key] = _stats_for_trajectories(
            np.concatenate(collected[key], axis=0)
        )
    _json_dump_new(output_path, result)
    return result


__all__ = [
    "ACTION_GROUPS",
    "ACTIVE_ACTION_DIM",
    "ADAPTER_KIND",
    "BASELINE_ID",
    "CONFIG_SCHEMA_VERSION",
    "EMBODIMENT_ID",
    "EMBODIMENT_TAG",
    "G1Id10H40Native30EpisodeLoader",
    "G1Id10H40ManifestEpisodeCollection",
    "G1Id10H40Native30SingleStepDataset",
    "HORIZON",
    "LANGUAGE",
    "PADDED_DIM",
    "STATE_GROUPS",
    "STATS_SCHEMA_VERSION",
    "TRAIN_STATS_SCHEMA_VERSION",
    "TEMPORAL_OPTION",
    "apply_active_action_mask",
    "build_metadata_loader_view",
    "compute_h40_relative_statistics",
    "compute_h40_train_statistics",
    "h40_native30_offsets",
    "inspire_rad_to_u",
    "inspire_u_to_rad",
    "load_adapter_config",
    "resolve_decoded_action_26d",
    "scatter_inspire_u_to_sim_joint_targets",
]
