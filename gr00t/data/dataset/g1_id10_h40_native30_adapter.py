"""CPU-side adapter for the ID10/H40 native-represented 30 Hz baseline.

The canonical timestamps are a generated 30 Hz row grid.  This module never
claims that the original hardware action clock was independently measured at
30 Hz, and it never relabels the training targets as 50 Hz.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
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
    RAW_LANGUAGE_SLUG,
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
from gr00t.data.types import EmbodimentTag, ModalityConfig


ADAPTER_KIND = "g1_id10_h40_native30_v1"
BASELINE_ID = "ID10_INSPIRE6_TASK_BASELINE_H40_NATIVE30"
EMBODIMENT_TAG = "new_embodiment"
EMBODIMENT_ID = 10
TEMPORAL_OPTION = "CANONICAL_NATIVE_REPRESENTED_30HZ_ROW_GRID_V1"
VIEW_SCHEMA_VERSION = "g1-id10-h40-native30-loader-view-v1"
CONFIG_SCHEMA_VERSION = "g1-id10-h40-native30-adapter-config-v1"
STATS_SCHEMA_VERSION = "g1-id10-h40-native30-relative-stats-v1"
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
    allowed = {
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
    if set(cfg) != allowed:
        raise ValueError(f"Adapter config schema mismatch: {sorted(set(cfg) ^ allowed)}")
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


def _validate_relative_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if stats.get("schema_version") != STATS_SCHEMA_VERSION:
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
        manifest = _json_load(
            self.dataset_path / "meta/g1_id10_h40_native30_view_manifest.json"
        )
        if manifest.get("schema_version") != VIEW_SCHEMA_VERSION:
            raise ValueError("Dataset is not an ID10/H40/native30 loader view")
        if manifest.get("true_30hz_hardware_clock_proven") is not False:
            raise ValueError("Loader view overclaims a high-precision hardware clock")

    def get_dataset_statistics(self) -> dict[str, Any]:
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
        relative = _validate_relative_stats(
            _json_load(Path(self.adapter_config["derived_relative_stats_path"]))
        )
        result["relative_action"] = relative
        return result


class G1Id10H40Native30SingleStepDataset(ShardedSingleStepDataset):
    def __init__(self, *args, adapter_config_path: str | Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.episode_loader = G1Id10H40Native30EpisodeLoader(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            adapter_config_path=adapter_config_path,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
            require_runtime_verified_fixed_state=True,
        )

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
    "G1Id10H40Native30SingleStepDataset",
    "HORIZON",
    "LANGUAGE",
    "PADDED_DIM",
    "STATE_GROUPS",
    "STATS_SCHEMA_VERSION",
    "TEMPORAL_OPTION",
    "apply_active_action_mask",
    "build_metadata_loader_view",
    "compute_h40_relative_statistics",
    "h40_native30_offsets",
    "inspire_rad_to_u",
    "inspire_u_to_rad",
    "load_adapter_config",
    "resolve_decoded_action_26d",
    "scatter_inspire_u_to_sim_joint_targets",
]
