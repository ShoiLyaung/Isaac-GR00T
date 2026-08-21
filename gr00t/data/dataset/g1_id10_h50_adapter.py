# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side adapter for the Human-selected ICRA27 G1 ID10/H50 candidate.

The candidate is not a frozen semantic contract until independent review.  The
adapter intentionally describes its temporal view as synthetic: canonical
30 Hz rows are repeated onto a nominal 50 Hz action-target grid with causal
zero-order hold.  It never claims that the source demonstrations were captured
at 50 Hz.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.types import EmbodimentTag, ModalityConfig


ADAPTER_KIND = "g1_id10_h50_synthetic_zoh_v1"
BASELINE_ID = "ID10_INSPIRE6_TASK_BASELINE"
EMBODIMENT_TAG = "new_embodiment"
EMBODIMENT_ID = 10
TEMPORAL_OPTION = "SYNTHETIC_30HZ_TO_50HZ_CAUSAL_ZOH_V1"
LANGUAGE = "Insert the shaft into the manipulator housing."
RAW_LANGUAGE_SLUG = "insertion_shaft_main_300ep"
VIEW_SCHEMA_VERSION = "g1-id10-h50-loader-view-v1"
CONFIG_SCHEMA_VERSION = "g1-id10-h50-adapter-config-v1"
STATS_SCHEMA_VERSION = "g1-id10-h50-relative-stats-v1"
CANONICAL_INFO_SHA256 = "74bd5f7758fa000939fe6bc461e325ebc8ff276e6c03f232c6d1179f76cf53fb"
CANONICAL_MODALITY_SHA256 = "d20c33542b42e3f114af54e9f3e7432d3528e38a64b4216a6c44d33bcdcba67d"

STATE_SLICES = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 20),
    "right_hand": (20, 26),
}
ACTION_SLICES = dict(STATE_SLICES)
FIXED_STATE_DIMS = {"left_leg": 6, "right_leg": 6, "waist": 3}
FIXED_ACTION_DIMS = {"waist": 3, "base_height_command": 1, "navigate_command": 3}
STATE_GROUPS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
]
ACTION_GROUPS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
]
ACTIVE_ACTION_DIM = 26
FULL_ACTION_DIM = 33
PADDED_DIM = 132
HORIZON = 50
SOURCE_SPAN = 30
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")

RIGHT_INSPIRE_SIM_INDICES = (36, 37, 35, 34, 48, 38)
LEFT_INSPIRE_SIM_INDICES = (31, 32, 30, 29, 43, 33)


def h50_zoh_offsets() -> list[int]:
    """Return the exact frozen k -> floor(3*k/5) source offsets."""

    offsets = [(3 * k) // 5 for k in range(HORIZON)]
    assert len(offsets) == HORIZON
    assert offsets[-1] == SOURCE_SPAN - 1
    return offsets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_dump_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _format_episode_path(pattern: str, episode_index: int, chunk_size: int, **kwargs) -> str:
    chunk_index = episode_index // chunk_size
    return pattern.format(
        chunk_index=chunk_index,
        episode_chunk=chunk_index,
        episode_index=episode_index,
        **kwargs,
    )


def _assert_external_output(source_root: Path, output_path: Path) -> None:
    source = source_root.resolve()
    output = output_path.resolve()
    if output == source or source in output.parents:
        raise ValueError(f"Derived output must not be inside canonical input: {output}")


def build_metadata_loader_view(
    canonical_root: str | Path,
    view_root: str | Path,
    *,
    enforce_frozen_identity: bool = True,
) -> dict[str, Any]:
    """Create a non-overwriting metadata view with read-only data/video symlinks.

    The view fixes path-template compatibility, exposes exactly one real head
    camera as ``ego_view``, and maps task index 0 to the frozen language literal.
    It does not copy or mutate parquet/video payloads.
    """

    canonical_root = Path(canonical_root).resolve()
    view_root = Path(view_root)
    _assert_external_output(canonical_root, view_root)
    if view_root.exists():
        raise FileExistsError(f"Loader view already exists: {view_root}")

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
    for section in ("state", "action", "video"):
        if section not in modality:
            raise KeyError(f"Canonical modality.json lacks {section}")
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
        "source_rate_hz": 30,
        "target_rate_hz": 50,
        "true_50hz_capture": False,
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
    _json_dump_new(view_meta / "g1_id10_h50_view_manifest.json", manifest)
    return manifest


def load_adapter_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = _json_load(path)
    allowed_keys = {
        "schema_version",
        "adapter_kind",
        "baseline_id",
        "embodiment_tag",
        "embodiment_id",
        "temporal_option",
        "source_index_formula",
        "source_rate_hz",
        "target_rate_hz",
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
    if set(cfg) != allowed_keys:
        raise ValueError(f"Adapter config schema mismatch: {sorted(set(cfg) ^ allowed_keys)}")
    required_exact = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "adapter_kind": ADAPTER_KIND,
        "baseline_id": BASELINE_ID,
        "embodiment_tag": EMBODIMENT_TAG,
        "embodiment_id": EMBODIMENT_ID,
        "temporal_option": TEMPORAL_OPTION,
        "language": LANGUAGE,
        "horizon": HORIZON,
        "source_rate_hz": 30,
        "target_rate_hz": 50,
        "true_50hz_capture": False,
    }
    for key, expected in required_exact.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Adapter config {key}={cfg.get(key)!r}, expected {expected!r}")
    if cfg.get("source_index_formula") != "n+floor(3*k/5)":
        raise ValueError("Frozen source-index formula mismatch")
    if cfg.get("active_action_dims") != ACTIVE_ACTION_DIM:
        raise ValueError("Active action dimension must be 26")
    if cfg.get("hand_scope") != "PROJECT_INSPIRE6_EXTENSION_NO_DIRECT_OFFICIAL_MATCH":
        raise ValueError("Inspire-6 project-extension identity is required")

    fixed_state = cfg.get("fixed_state", {})
    fixed_action = cfg.get("fixed_action", {})
    for key, dim in FIXED_STATE_DIMS.items():
        values = np.asarray(fixed_state.get(key), dtype=np.float32)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"fixed_state.{key} must be finite shape {(dim,)}")
    for key, dim in FIXED_ACTION_DIMS.items():
        values = np.asarray(fixed_action.get(key), dtype=np.float32)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"fixed_action.{key} must be finite shape {(dim,)}")
    return cfg


def _constant_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    zeros = np.zeros_like(values)
    return {
        "mean": values.tolist(),
        "std": zeros.tolist(),
        "min": values.tolist(),
        "max": values.tolist(),
        "q01": values.tolist(),
        "q99": values.tolist(),
    }


def _slice_stats(stats: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    result = {}
    for name in STAT_NAMES:
        values = np.asarray(stats[name], dtype=np.float64)
        if values.ndim != 1 or values.shape[0] < end:
            raise ValueError(f"Invalid canonical statistic {name} shape {values.shape}")
        result[name] = values[start:end].tolist()
    return result


def _stack_float32_rows(series: pd.Series, *, field: str) -> np.ndarray:
    rows = []
    for value in series:
        row = np.asarray(value)
        if row.dtype != np.float32:
            raise TypeError(f"{field} must materialize as float32, got {row.dtype}")
        rows.append(row)
    return np.vstack(rows)


def _validate_relative_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if stats.get("schema_version") != STATS_SCHEMA_VERSION:
        raise ValueError("Derived relative-statistics schema mismatch")
    if stats.get("temporal_option") != TEMPORAL_OPTION:
        raise ValueError("Derived relative statistics use a different temporal option")
    if stats.get("true_50hz_capture") is not False:
        raise ValueError("Derived statistics must retain synthetic lineage")
    result = {}
    for key in ("left_arm", "right_arm"):
        group = stats.get("relative_action", {}).get(key, {})
        result[key] = {}
        for name in STAT_NAMES:
            values = np.asarray(group.get(name), dtype=np.float64)
            if values.shape != (HORIZON, 7) or not np.isfinite(values).all():
                raise ValueError(f"relative_action.{key}.{name} must be finite [50,7]")
            result[key][name] = values.tolist()
    return result


class G1Id10H50EpisodeLoader(LeRobotEpisodeLoader):
    """Episode loader that exposes the frozen full-body project surface."""

    raw_language_slug = RAW_LANGUAGE_SLUG
    language = LANGUAGE

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
        super().__init__(dataset_path, modality_configs, video_backend, video_backend_kwargs)
        view_manifest = _json_load(
            self.dataset_path / "meta" / "g1_id10_h50_view_manifest.json"
        )
        if view_manifest.get("schema_version") != VIEW_SCHEMA_VERSION:
            raise ValueError("Dataset is not a G1 ID10/H50 loader view")
        if view_manifest.get("true_50hz_capture") is not False:
            raise ValueError("Loader view lost synthetic temporal lineage")

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
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
        if set(raw["annotation.human.action.task_description"].tolist()) != {
            self.raw_language_slug
        }:
            raise ValueError("Raw language slug mismatch")
        if not np.all(raw["annotation.human.validity"].to_numpy() == 1):
            raise ValueError("Invalid demonstration row encountered")
        for slc in (slice(14, 20), slice(20, 26)):
            values = np.concatenate((states[:, slc], actions[:, slc]), axis=0)
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError("Inspire hand command u must be within [0,1]")

        result = pd.DataFrame(index=raw.index)
        result["language.annotation.human.task_description"] = [self.language] * len(raw)
        for key, dim in FIXED_STATE_DIMS.items():
            value = np.asarray(self.adapter_config["fixed_state"][key], dtype=np.float32)
            assert value.shape == (dim,)
            result[f"state.{key}"] = [value.copy() for _ in range(len(raw))]
        for key, (start, end) in STATE_SLICES.items():
            result[f"state.{key}"] = [row.copy() for row in states[:, start:end]]
        for key, (start, end) in ACTION_SLICES.items():
            result[f"action.{key}"] = [row.copy() for row in actions[:, start:end]]
        for key, dim in FIXED_ACTION_DIMS.items():
            value = np.asarray(self.adapter_config["fixed_action"][key], dtype=np.float32)
            assert value.shape == (dim,)
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

        relative_path = Path(self.adapter_config["derived_relative_stats_path"])
        relative = _validate_relative_stats(_json_load(relative_path))
        result["relative_action"] = relative
        return result


class G1Id10H50SingleStepDataset(ShardedSingleStepDataset):
    """Production dataset wrapper that enforces the active 26-D loss mask."""

    def __init__(self, *args, adapter_config_path: str | Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.episode_loader = G1Id10H50EpisodeLoader(
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
    """Zero the loss mask for inactive waist/base/nav and padded dimensions."""

    if "action_mask" not in item:
        raise KeyError("Processed training item has no action_mask")
    mask = item["action_mask"]
    if tuple(mask.shape) != (HORIZON, PADDED_DIM):
        raise ValueError(f"Expected action_mask [50,132], got {tuple(mask.shape)}")
    mask = mask.clone() if hasattr(mask, "clone") else np.array(mask, copy=True)
    mask[:, ACTIVE_ACTION_DIM:] = 0
    if not np.all(np.asarray(mask[:, :ACTIVE_ACTION_DIM]) == 1):
        raise ValueError("Active arm+hand dimensions must remain supervised")
    item = dict(item)
    item["action_mask"] = mask
    return item


def resolve_decoded_action_26d(
    decoded: dict[str, np.ndarray], *, clamp_hand: bool = True
) -> tuple[np.ndarray, int]:
    """Resolve processor-decoded absolute actions to the 26-D simulator boundary.

    Relative arm conversion is already performed exactly once by
    ``StateActionProcessor.unapply_action``.  This function must therefore not
    add the current arm state again.  Inactive full-body outputs are validated
    for shape but deliberately ignored.
    """

    required_dims = {
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 6,
        "right_hand": 6,
        **FIXED_ACTION_DIMS,
    }
    arrays = {}
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
        clamp_count += int(outside.sum())
        if clamp_count and not clamp_hand:
            raise ValueError("Decoded Inspire command outside [0,1]")
        hands.append(np.clip(value, 0.0, 1.0) if clamp_hand else value)
    active = np.concatenate(
        [arrays["left_arm"], arrays["right_arm"], hands[0], hands[1]], axis=-1
    )
    assert active.shape == (HORIZON, ACTIVE_ACTION_DIM)
    return active, clamp_count


def inspire_u_to_rad(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    if u.shape[-1] != 6 or not np.isfinite(u).all():
        raise ValueError("Inspire u must be finite [...,6]")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("Inspire u must be within [0,1]")
    result = np.empty_like(u)
    result[..., :4] = 1.7 * (1.0 - u[..., :4])
    result[..., 4] = 0.5 * (1.0 - u[..., 4])
    result[..., 5] = 1.3 - 1.4 * u[..., 5]
    return result


def inspire_rad_to_u(rad: np.ndarray) -> np.ndarray:
    rad = np.asarray(rad, dtype=np.float32)
    if rad.shape[-1] != 6 or not np.isfinite(rad).all():
        raise ValueError("Inspire radians must be finite [...,6]")
    result = np.empty_like(rad)
    result[..., :4] = (1.7 - rad[..., :4]) / 1.7
    result[..., 4] = (0.5 - rad[..., 4]) / 0.5
    result[..., 5] = (1.3 - rad[..., 5]) / 1.4
    return result


def scatter_inspire_u_to_sim_joint_targets(
    full_joint_targets: np.ndarray,
    left_u: np.ndarray,
    right_u: np.ndarray,
) -> np.ndarray:
    """Apply the frozen Inspire-6 order exactly once at simulator ingress."""

    targets = np.asarray(full_joint_targets, dtype=np.float32)
    if targets.ndim != 1 or targets.shape[0] <= max(*LEFT_INSPIRE_SIM_INDICES, *RIGHT_INSPIRE_SIM_INDICES):
        raise ValueError("Full simulator joint target vector is too short")
    result = targets.copy()
    left_rad = inspire_u_to_rad(left_u)
    right_rad = inspire_u_to_rad(right_u)
    if left_rad.shape != (6,) or right_rad.shape != (6,):
        raise ValueError("Simulator ingress expects one six-dimensional command per hand")
    result[list(LEFT_INSPIRE_SIM_INDICES)] = left_rad
    result[list(RIGHT_INSPIRE_SIM_INDICES)] = right_rad
    return result


def _stats_for_trajectories(values: np.ndarray) -> dict[str, list[list[float]]]:
    if values.ndim != 3 or values.shape[1:] != (HORIZON, 7):
        raise ValueError(f"Expected [N,50,7], got {values.shape}")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_h50_relative_statistics(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Compute H50 arm-relative stats without writing the input dataset."""

    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path)
    _assert_external_output(dataset_root, output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    info = _json_load(info_path)
    with episodes_path.open("r", encoding="utf-8") as stream:
        episodes = [json.loads(line) for line in stream]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    offsets = np.asarray(h50_zoh_offsets(), dtype=np.int64)
    collected = {"left_arm": [], "right_arm": []}
    anchor_count = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        filename = _format_episode_path(
            info["data_path"], episode_index, int(info["chunks_size"])
        )
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
        hand_values = np.concatenate((state[:, 14:26], action[:, 14:26]), axis=0)
        if np.any(hand_values < 0.0) or np.any(hand_values > 1.0):
            raise ValueError(f"Episode {episode_index} Inspire command is outside [0,1]")
        n_anchors = len(frame) - SOURCE_SPAN + 1
        if n_anchors <= 0:
            continue
        target_indices = np.arange(n_anchors)[:, None] + offsets[None, :]
        for key, (start, end) in {
            "left_arm": (0, 7),
            "right_arm": (7, 14),
        }.items():
            absolute = action[target_indices, start:end]
            reference = state[:n_anchors, None, start:end]
            collected[key].append((absolute - reference).astype(np.float32))
        anchor_count += n_anchors

    if anchor_count == 0:
        raise ValueError("No legal H50 anchors")
    result = {
        "schema_version": STATS_SCHEMA_VERSION,
        "baseline_id": BASELINE_ID,
        "temporal_option": TEMPORAL_OPTION,
        "source_index_formula": "n+floor(3*k/5)",
        "source_rate_hz": 30,
        "target_rate_hz": 50,
        "true_50hz_capture": False,
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


def write_h50_temporal_provenance(
    dataset_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write exact per-anchor/tick synthetic-ZOH lineage as a derived parquet."""

    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path)
    _assert_external_output(dataset_root, output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    info = _json_load(info_path)
    with episodes_path.open("r", encoding="utf-8") as stream:
        episodes = [json.loads(line) for line in stream]

    schema = pa.schema(
        [
            ("episode_index", pa.int32()),
            ("anchor_row", pa.int32()),
            ("target_tick", pa.int16()),
            ("source_offset", pa.int16()),
            ("source_row", pa.int32()),
            ("repeated_from_previous_tick", pa.bool_()),
        ],
        metadata={
            b"schema_version": b"g1-id10-h50-temporal-provenance-v1",
            b"baseline_id": BASELINE_ID.encode(),
            b"temporal_option": TEMPORAL_OPTION.encode(),
            b"source_index_formula": b"n+floor(3*k/5)",
            b"source_rate_hz": b"30",
            b"target_rate_hz": b"50",
            b"true_50hz_capture": b"false",
            b"canonical_info_sha256": _sha256(info_path).encode(),
        },
    )
    offsets = np.asarray(h50_zoh_offsets(), dtype=np.int16)
    repeated = np.zeros(HORIZON, dtype=bool)
    repeated[1:] = offsets[1:] == offsets[:-1]
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    anchors_total = 0
    rows_total = 0
    try:
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            n_anchors = max(episode_length - SOURCE_SPAN + 1, 0)
            if n_anchors == 0:
                continue
            anchors = np.repeat(np.arange(n_anchors, dtype=np.int32), HORIZON)
            tiled_offsets = np.tile(offsets, n_anchors)
            table = pa.table(
                {
                    "episode_index": np.full(n_anchors * HORIZON, episode_index, np.int32),
                    "anchor_row": anchors,
                    "target_tick": np.tile(np.arange(HORIZON, dtype=np.int16), n_anchors),
                    "source_offset": tiled_offsets,
                    "source_row": anchors + tiled_offsets.astype(np.int32),
                    "repeated_from_previous_tick": np.tile(repeated, n_anchors),
                },
                schema=schema,
            )
            writer.write_table(table)
            anchors_total += n_anchors
            rows_total += n_anchors * HORIZON
    finally:
        writer.close()

    return {
        "schema_version": "g1-id10-h50-temporal-provenance-v1",
        "episodes": len(episodes),
        "anchors": anchors_total,
        "rows": rows_total,
        "horizon": HORIZON,
        "true_50hz_capture": False,
        "output_sha256": _sha256(output_path),
    }
