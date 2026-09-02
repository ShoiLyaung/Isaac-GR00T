"""Read-only H40 adapter for a declared mixture of compatible 26-D G1 datasets."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTION_SLICES,
    ACTIVE_ACTION_DIM,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    FIXED_ACTION_DIMS,
    FIXED_STATE_DIMS,
    HORIZON,
    STATE_SLICES,
    TEMPORAL_OPTION,
    _assert_external_output,
    _constant_stats,
    _format_episode_path,
    _json_dump_new,
    _json_load,
    _sha256,
    _slice_stats,
    _stack_float32_rows,
    _validate_relative_stats,
    apply_active_action_mask,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import ModalityConfig


ADAPTER_KIND = "g1_all_lerobot_h40_v1"
BASELINE_ID = "G1_ALL_LEROBOT_26D_H40"
CONFIG_SCHEMA_VERSION = "g1-all-lerobot-h40-adapter-config-v1"
VIEW_SCHEMA_VERSION = "g1-all-lerobot-h40-loader-view-v1"
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


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
    allowed = {
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
    if not isinstance(cfg, dict) or set(cfg) != allowed:
        actual = set(cfg) if isinstance(cfg, dict) else set()
        raise ValueError(f"Adapter config schema mismatch: {sorted(actual ^ allowed)}")
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
    return cfg


class G1AllLerobotH40EpisodeLoader(LeRobotEpisodeLoader):
    """Load one declared source while preserving its raw-label provenance."""

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        adapter_config_path: str | Path,
        decoder_kwargs: dict[str, Any] | None = None,
        require_runtime_verified_fixed_state: bool = True,
    ) -> None:
        self.adapter_config_path = Path(adapter_config_path)
        self.adapter_config = load_adapter_config(self.adapter_config_path)
        if (
            require_runtime_verified_fixed_state
            and not self.adapter_config["runtime_verified_fixed_state"]
        ):
            raise RuntimeError("Training requires runtime-verified fixed leg/waist state")
        super().__init__(dataset_path, modality_configs, decoder_kwargs)
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
            require_runtime_verified_fixed_state=True,
        )

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        return apply_active_action_mask(super().get_datapoint(episode_data, step_index))


__all__ = [
    "ADAPTER_KIND",
    "BASELINE_ID",
    "CONFIG_SCHEMA_VERSION",
    "G1AllLerobotH40EpisodeLoader",
    "G1AllLerobotH40SingleStepDataset",
    "VIEW_SCHEMA_VERSION",
    "build_metadata_loader_view",
    "load_adapter_config",
]
