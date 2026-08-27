"""Read-only H40 adapter for a declared mixture of compatible 26-D G1 datasets."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTIVE_ACTION_DIM,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    HORIZON,
    TEMPORAL_OPTION,
    _validate_relative_stats,
    apply_active_action_mask,
)
from gr00t.data.dataset.g1_id10_h50_adapter import (
    ACTION_SLICES,
    FIXED_ACTION_DIMS,
    FIXED_STATE_DIMS,
    STATE_SLICES,
    _assert_external_output,
    _constant_stats,
    _format_episode_path,
    _json_dump_new,
    _sha256,
    _slice_stats,
    _stack_float32_rows,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import ModalityConfig


ADAPTER_KIND = "g1_all_lerobot_h40_v1"
BASELINE_ID = "G1_ALL_LEROBOT_26D_H40"
CONFIG_SCHEMA_VERSION = "g1-all-lerobot-h40-adapter-config-v1"
VIEW_SCHEMA_VERSION = "g1-all-lerobot-h40-loader-view-v1"
# Kept local so absent/disabled adapters never import the optional Codec package.
CODEC_EPISODE_UID_SCHEMA_VERSION = "codec-evidence-episode-identity-v1"
CODEC_CAMERA_ID = "cam_left_high"
CODEC_VIDEO_KEY = "observation.images.cam_left_high"
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _read_tasks(path: Path) -> list[str]:
    tasks: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"task_index", "task"}:
                raise ValueError(f"Unexpected tasks row in {path}")
            task_index = row["task_index"]
            task = row["task"]
            if isinstance(task_index, bool) or not isinstance(task_index, int):
                raise ValueError(f"Invalid task_index in {path}")
            if not isinstance(task, str) or not task.strip() or task_index in tasks:
                raise ValueError(f"Invalid or duplicate task in {path}")
            tasks[task_index] = task
    if sorted(tasks) != list(range(len(tasks))):
        raise ValueError(f"Task indices must be contiguous from zero in {path}")
    return [tasks[index] for index in range(len(tasks))]


def build_metadata_loader_view(
    canonical_root: str | Path,
    view_root: str | Path,
    *,
    source_id: str,
    expected_info_sha256: str,
    expected_modality_sha256: str,
    raw_task_labels: list[str],
    training_language: str,
) -> dict[str, Any]:
    """Create a non-overwriting metadata overlay with read-only payload links."""

    canonical_root = Path(canonical_root).resolve()
    view_root = Path(view_root)
    _assert_external_output(canonical_root, view_root)
    if view_root.exists():
        raise FileExistsError(view_root)
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise ValueError(f"Invalid source_id: {source_id!r}")
    if not isinstance(training_language, str) or not training_language.strip():
        raise ValueError("training_language must be non-empty")
    if not raw_task_labels or any(
        not isinstance(item, str) or not item for item in raw_task_labels
    ):
        raise ValueError("raw_task_labels must contain non-empty strings")

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
    info_sha256 = _sha256(canonical_meta / "info.json")
    modality_sha256 = _sha256(canonical_meta / "modality.json")
    if info_sha256 != expected_info_sha256:
        raise ValueError(f"Canonical info.json identity mismatch for {source_id}")
    if modality_sha256 != expected_modality_sha256:
        raise ValueError(f"Canonical modality.json identity mismatch for {source_id}")
    if _read_tasks(canonical_meta / "tasks.jsonl") != raw_task_labels:
        raise ValueError(f"Canonical task labels mismatch for {source_id}")

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
    try:
        head = deepcopy(modality["video"]["cam_left_high"])
    except KeyError as exc:
        raise ValueError(f"{source_id} has no cam_left_high modality") from exc
    head["original_key"] = "observation.images.cam_left_high"
    modality["video"] = {"ego_view": head}
    modality["state"] = {key: deepcopy(modality["state"][key]) for key in STATE_SLICES}
    modality["action"] = {key: deepcopy(modality["action"][key]) for key in ACTION_SLICES}
    modality["annotation"] = {"human.task_description": {"original_key": "task_index"}}
    _json_dump_new(view_meta / "modality.json", modality)
    with (view_meta / "tasks.jsonl").open("x", encoding="utf-8") as stream:
        for task_index, _ in enumerate(raw_task_labels):
            stream.write(
                json.dumps({"task_index": task_index, "task": training_language}, sort_keys=True)
                + "\n"
            )

    for name in ("data", "videos"):
        target = canonical_root / name
        if not target.is_dir():
            raise FileNotFoundError(target)
        (view_root / name).symlink_to(target, target_is_directory=True)

    manifest = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "source_id": source_id,
        "canonical_root": str(canonical_root),
        "canonical_info_sha256": info_sha256,
        "canonical_modality_sha256": modality_sha256,
        "canonical_tasks_sha256": _sha256(canonical_meta / "tasks.jsonl"),
        "raw_task_labels": raw_task_labels,
        "training_language": training_language,
        "language_policy": "explicit_manifest_normalization",
        "camera_mapping": {
            "model_key": "ego_view",
            "canonical_key": "cam_left_high",
            "original_key": "observation.images.cam_left_high",
        },
        "horizon": HORIZON,
        "source_index_formula": "n+k",
        "row_grid_rate_hz": 30,
        "payload_policy": "read-only symlink to canonical data/videos",
    }
    _json_dump_new(view_meta / "g1_all_lerobot_h40_view_manifest.json", manifest)
    return manifest


def load_adapter_config(path: str | Path) -> dict[str, Any]:
    cfg = _json_load(Path(path))
    required = {
        "schema_version",
        "adapter_kind",
        "baseline_id",
        "source_id",
        "canonical_root",
        "canonical_info_sha256",
        "canonical_modality_sha256",
        "raw_task_labels",
        "training_language",
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
        "runtime_verified_fixed_state",
        "fixed_state",
        "fixed_action",
        "derived_relative_stats_path",
    }
    optional = {"codec_evidence"}
    if not isinstance(cfg, dict) or not required <= set(cfg) <= required | optional:
        actual = set(cfg) if isinstance(cfg, dict) else set()
        mismatch = (required - actual) | (actual - required - optional)
        raise ValueError(f"Adapter config schema mismatch: {sorted(mismatch)}")
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
    }
    for key, expected in exact.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Adapter config {key}={cfg.get(key)!r}, expected {expected!r}")
    if not SOURCE_ID_RE.fullmatch(cfg["source_id"]):
        raise ValueError("Adapter source_id is invalid")
    if not isinstance(cfg["training_language"], str) or not cfg["training_language"].strip():
        raise ValueError("Adapter training_language must be non-empty")
    if not isinstance(cfg["raw_task_labels"], list) or not cfg["raw_task_labels"]:
        raise ValueError("Adapter raw_task_labels must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in cfg["raw_task_labels"]):
        raise ValueError("Adapter raw_task_labels must contain strings")
    canonical_root = Path(cfg["canonical_root"])
    if not canonical_root.is_absolute():
        raise ValueError("Adapter canonical_root must be absolute")
    for key in ("canonical_info_sha256", "canonical_modality_sha256"):
        value = cfg[key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Adapter {key} must be a lowercase SHA256")
    for key, dim in FIXED_STATE_DIMS.items():
        values = np.asarray(cfg.get("fixed_state", {}).get(key), dtype=np.float32)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"fixed_state.{key} must be finite shape {(dim,)}")
    for key, dim in FIXED_ACTION_DIMS.items():
        values = np.asarray(cfg.get("fixed_action", {}).get(key), dtype=np.float32)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"fixed_action.{key} must be finite shape {(dim,)}")
    if "codec_evidence" in cfg:
        parse_adapter_codec_evidence_config(cfg)
    return cfg


def parse_adapter_codec_evidence_config(adapter_config: dict[str, Any]):
    """Parse the optional block without resolving or reading a fixture root."""

    value = adapter_config.get("codec_evidence")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("codec_evidence must be an object")
    enabled = value.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("codec_evidence.enabled must be boolean")
    if not enabled:
        return None

    from gr00t.data.codec_evidence.config import parse_codec_evidence_config

    parsed = parse_codec_evidence_config(value)
    if parsed.camera_id != CODEC_CAMERA_ID:
        raise ValueError(f"G1 all-LeRobot codec evidence requires {CODEC_CAMERA_ID}")
    return parsed


def _payload_file_identity(canonical_root: Path, relative_path: str) -> dict[str, Any]:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Canonical payload path is not relative and contained: {relative_path}")
    resolved = (canonical_root / relative).resolve(strict=True)
    if canonical_root != resolved and canonical_root not in resolved.parents:
        raise ValueError(f"Canonical payload escapes its dataset root: {relative_path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "relative_path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def build_codec_episode_identity(
    *,
    source_id: str,
    canonical_root: str | Path,
    canonical_info_sha256: str,
    canonical_modality_sha256: str,
    data_path_pattern: str,
    video_path_pattern: str,
    chunk_size: int,
    episode_index: int,
    episode_length: int,
) -> dict[str, Any]:
    """Bind one stable UID to source, dataset and exact Parquet/head-video bytes."""

    from gr00t.data.codec_evidence.identity import episode_identity_with_uid

    if not SOURCE_ID_RE.fullmatch(source_id):
        raise ValueError("Codec episode identity source_id is invalid")
    for name, value in (
        ("canonical_info_sha256", canonical_info_sha256),
        ("canonical_modality_sha256", canonical_modality_sha256),
    ):
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"Codec episode identity {name} must be a lowercase SHA256")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Codec episode identity chunk_size must be positive")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("Codec episode identity episode_index must be non-negative")
    if (
        isinstance(episode_length, bool)
        or not isinstance(episode_length, int)
        or episode_length < 1
    ):
        raise ValueError("Codec episode identity episode_length must be positive")
    canonical = Path(canonical_root).resolve(strict=True)
    if not canonical.is_dir():
        raise ValueError("Codec canonical_root must be a directory")
    data_relative = _format_episode_path(data_path_pattern, episode_index, chunk_size)
    video_relative = _format_episode_path(
        video_path_pattern,
        episode_index,
        chunk_size,
        video_key=CODEC_VIDEO_KEY,
    )
    payload = {
        "schema_version": CODEC_EPISODE_UID_SCHEMA_VERSION,
        "source_id": source_id,
        "canonical_dataset_identity": {
            "info_sha256": canonical_info_sha256,
            "modality_sha256": canonical_modality_sha256,
        },
        "episode_index": episode_index,
        "episode_length": episode_length,
        "camera_id": CODEC_CAMERA_ID,
        "parquet": _payload_file_identity(canonical, data_relative),
        "video": _payload_file_identity(canonical, video_relative),
    }
    return episode_identity_with_uid(payload)


class G1AllLerobotH40EpisodeLoader(LeRobotEpisodeLoader):
    """Load one declared source while preserving its raw-label provenance."""

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
        if (
            require_runtime_verified_fixed_state
            and not self.adapter_config["runtime_verified_fixed_state"]
        ):
            raise RuntimeError("Training requires runtime-verified fixed leg/waist state")
        super().__init__(dataset_path, modality_configs, video_backend, video_backend_kwargs)
        manifest = _json_load(self.dataset_path / "meta/g1_all_lerobot_h40_view_manifest.json")
        matched = {
            "schema_version": VIEW_SCHEMA_VERSION,
            "baseline_id": BASELINE_ID,
            "source_id": self.adapter_config["source_id"],
            "canonical_root": str(Path(self.adapter_config["canonical_root"]).resolve()),
            "canonical_info_sha256": self.adapter_config["canonical_info_sha256"],
            "canonical_modality_sha256": self.adapter_config["canonical_modality_sha256"],
            "raw_task_labels": self.adapter_config["raw_task_labels"],
            "training_language": self.adapter_config["training_language"],
        }
        for key, expected in matched.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Loader view {key} does not match adapter config")
        if "codec_evidence" in self.adapter_config:
            codec_config = parse_adapter_codec_evidence_config(self.adapter_config)
            if codec_config is not None:
                self._validate_codec_loader_view_transport(manifest)
                self._initialize_codec_episode_identities()

    def _validate_codec_loader_view_transport(self, manifest: dict[str, Any]) -> None:
        """Bind the loader's actual view paths and metadata to the UID source."""

        canonical_root = Path(self.adapter_config["canonical_root"]).resolve(strict=True)
        for name in ("data", "videos"):
            view_payload = self.dataset_path / name
            canonical_payload = (canonical_root / name).resolve(strict=True)
            if not view_payload.is_symlink():
                raise ValueError(f"Enabled Codec loader view {name} must be a symlink")
            if view_payload.resolve(strict=True) != canonical_payload:
                raise ValueError(
                    f"Enabled Codec loader view {name} target differs from canonical payload"
                )

        for name in ("episodes.jsonl", "stats.json"):
            canonical_meta = canonical_root / "meta" / name
            view_meta = self.dataset_path / "meta" / name
            if not view_meta.is_file() or view_meta.is_symlink():
                raise ValueError(f"Enabled Codec loader view {name} must be a regular file")
            if _sha256(view_meta) != _sha256(canonical_meta):
                raise ValueError(
                    f"Enabled Codec loader view {name} differs from canonical metadata"
                )

        expected_tasks = [self.adapter_config["training_language"]] * len(
            self.adapter_config["raw_task_labels"]
        )
        if _read_tasks(self.dataset_path / "meta/tasks.jsonl") != expected_tasks:
            raise ValueError("Enabled Codec loader view tasks.jsonl differs from normalization")

        expected_manifest = {
            "schema_version": VIEW_SCHEMA_VERSION,
            "baseline_id": BASELINE_ID,
            "source_id": self.adapter_config["source_id"],
            "canonical_root": str(canonical_root),
            "canonical_info_sha256": self.adapter_config["canonical_info_sha256"],
            "canonical_modality_sha256": self.adapter_config["canonical_modality_sha256"],
            "canonical_tasks_sha256": _sha256(canonical_root / "meta/tasks.jsonl"),
            "raw_task_labels": self.adapter_config["raw_task_labels"],
            "training_language": self.adapter_config["training_language"],
            "language_policy": "explicit_manifest_normalization",
            "camera_mapping": {
                "model_key": "ego_view",
                "canonical_key": CODEC_CAMERA_ID,
                "original_key": CODEC_VIDEO_KEY,
            },
            "horizon": HORIZON,
            "source_index_formula": "n+k",
            "row_grid_rate_hz": 30,
            "payload_policy": "read-only symlink to canonical data/videos",
        }
        if manifest != expected_manifest:
            raise ValueError("Enabled Codec loader view manifest differs from its contract")

        expected_info = _json_load(canonical_root / "meta/info.json")
        expected_info["data_path"] = expected_info["data_path"].replace(
            "{chunk_index", "{episode_chunk"
        )
        if expected_info.get("video_path"):
            expected_info["video_path"] = expected_info["video_path"].replace(
                "{chunk_index", "{episode_chunk"
            )
        if self.info_meta != expected_info:
            raise ValueError("Enabled Codec loader view info.json differs from canonical mapping")
        if type(self.fps) is not int or self.fps != 30:
            raise ValueError("Enabled Codec loader view requires exact integer 30 fps")

        expected_modality = _json_load(canonical_root / "meta/modality.json")
        try:
            head = deepcopy(expected_modality["video"][CODEC_CAMERA_ID])
        except KeyError as exc:
            raise ValueError("Canonical dataset has no cam_left_high modality") from exc
        head["original_key"] = CODEC_VIDEO_KEY
        expected_modality["video"] = {"ego_view": head}
        expected_modality["state"] = {
            key: deepcopy(expected_modality["state"][key]) for key in STATE_SLICES
        }
        expected_modality["action"] = {
            key: deepcopy(expected_modality["action"][key]) for key in ACTION_SLICES
        }
        expected_modality["annotation"] = {"human.task_description": {"original_key": "task_index"}}
        if self.modality_meta != expected_modality:
            raise ValueError(
                "Enabled Codec loader view modality.json differs from canonical mapping"
            )

    def _initialize_codec_episode_identities(self) -> None:
        """Hash exact payload identities only for an explicitly enabled Codec leaf."""

        canonical_root = Path(self.adapter_config["canonical_root"]).resolve(strict=True)
        current_metadata = {
            "canonical_info_sha256": _sha256(canonical_root / "meta/info.json"),
            "canonical_modality_sha256": _sha256(canonical_root / "meta/modality.json"),
        }
        for key, actual in current_metadata.items():
            if actual != self.adapter_config[key]:
                raise ValueError(f"Codec canonical metadata identity changed: {key}")
        if not self.video_path_pattern:
            raise ValueError("Enabled Codec evidence requires a canonical video path pattern")

        identities = []
        seen_episode_indices: set[int] = set()
        for episode in self.episodes_metadata:
            episode_index = episode.get("episode_index")
            episode_length = episode.get("length")
            if (
                isinstance(episode_index, bool)
                or not isinstance(episode_index, int)
                or episode_index in seen_episode_indices
            ):
                raise ValueError("Codec episode indices must be unique integers")
            seen_episode_indices.add(episode_index)
            identities.append(
                build_codec_episode_identity(
                    source_id=self.adapter_config["source_id"],
                    canonical_root=canonical_root,
                    canonical_info_sha256=self.adapter_config["canonical_info_sha256"],
                    canonical_modality_sha256=self.adapter_config["canonical_modality_sha256"],
                    data_path_pattern=self.data_path_pattern,
                    video_path_pattern=self.video_path_pattern,
                    chunk_size=self.chunk_size,
                    episode_index=episode_index,
                    episode_length=episode_length,
                )
            )
        episode_uids = tuple(identity["episode_uid"] for identity in identities)
        if len(episode_uids) != len(set(episode_uids)):
            raise ValueError("Codec episode UIDs must be unique within one source")
        self.codec_episode_identities = tuple(identities)
        self.selected_episode_uids = episode_uids

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        filename = _format_episode_path(self.data_path_pattern, episode_index, self.chunk_size)
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
        expected_shape = (len(raw), ACTIVE_ACTION_DIM)
        if states.shape != expected_shape or actions.shape != expected_shape:
            raise ValueError(f"Expected canonical state/action {expected_shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite canonical state/action")
        if not np.array_equal(raw["frame_index"].to_numpy(), np.arange(len(raw))):
            raise ValueError("frame_index must be zero-based and contiguous within episode")
        if not np.all(raw["episode_index"].to_numpy() == episode_index):
            raise ValueError("episode_index mismatch")
        task_indices = raw["task_index"].to_numpy()
        labels = self.adapter_config["raw_task_labels"]
        if np.any(task_indices < 0) or np.any(task_indices >= len(labels)):
            raise ValueError("task_index is outside the declared raw task labels")
        expected_labels = np.asarray(labels, dtype=object)[task_indices]
        if not np.array_equal(
            raw["annotation.human.action.task_description"].to_numpy(), expected_labels
        ):
            raise ValueError("Raw task label mismatch")
        if not np.all(raw["annotation.human.validity"].to_numpy() == 1):
            raise ValueError("Invalid demonstration row encountered")
        hands = np.concatenate((states[:, 14:26], actions[:, 14:26]), axis=0)
        if np.any(hands < 0.0) or np.any(hands > 1.0):
            raise ValueError("Inspire hand command u must be within [0,1]")

        result = pd.DataFrame(index=raw.index)
        result["language.annotation.human.task_description"] = [
            self.adapter_config["training_language"]
        ] * len(raw)
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
        canonical_state = self.stats["observation.state"]
        canonical_action = self.stats["action"]
        result: dict[str, Any] = {"state": {}, "action": {}, "relative_action": {}}
        for key, dim in FIXED_STATE_DIMS.items():
            result["state"][key] = _constant_stats(
                np.asarray(self.adapter_config["fixed_state"][key], dtype=np.float32)
            )
        for key, (start, end) in STATE_SLICES.items():
            result["state"][key] = _slice_stats(canonical_state, start, end)
        for key, (start, end) in ACTION_SLICES.items():
            result["action"][key] = _slice_stats(canonical_action, start, end)
        for key, dim in FIXED_ACTION_DIMS.items():
            result["action"][key] = _constant_stats(
                np.asarray(self.adapter_config["fixed_action"][key], dtype=np.float32)
            )
        relative_raw = _json_load(Path(self.adapter_config["derived_relative_stats_path"]))
        if (
            relative_raw.get("canonical_info_sha256")
            != self.adapter_config["canonical_info_sha256"]
        ):
            raise ValueError("Relative statistics source identity mismatch")
        result["relative_action"] = _validate_relative_stats(relative_raw)
        return result


class G1AllLerobotH40SingleStepDataset(ShardedSingleStepDataset):
    def __init__(self, *args, adapter_config_path: str | Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.episode_loader = G1AllLerobotH40EpisodeLoader(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            adapter_config_path=adapter_config_path,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
            require_runtime_verified_fixed_state=True,
        )

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        return apply_active_action_mask(super().get_datapoint(episode_data, step_index))


__all__ = [
    "ADAPTER_KIND",
    "BASELINE_ID",
    "CODEC_CAMERA_ID",
    "CODEC_EPISODE_UID_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "G1AllLerobotH40EpisodeLoader",
    "G1AllLerobotH40SingleStepDataset",
    "VIEW_SCHEMA_VERSION",
    "build_codec_episode_identity",
    "build_metadata_loader_view",
    "load_adapter_config",
    "parse_adapter_codec_evidence_config",
]
