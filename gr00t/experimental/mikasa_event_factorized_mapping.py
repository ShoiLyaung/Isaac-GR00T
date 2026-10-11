"""Factorized shape and checkpoint retrieval over sealed event trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
import time
from typing import Any, Final

import numpy as np

from gr00t.experimental.mikasa_event_trajectory import (
    CALIBRATION_GROUPS,
    CAMERAS,
    CONDITIONS,
    FIT_GROUPS,
    TOKENS_PER_SLOT,
    EventCache,
    canonicalize_positions,
    gates,
    group_permutations,
    lexicographic_neighbor,
)
from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    MaterialIdentity,
    ShardedMemoryCorpus,
    write_new_bytes,
)
from gr00t.experimental.mikasa_segment_slot_train_calibration import fit_target_mean


SHAPE_DIM: Final = 2 * 2 * 65 * 3
CHECKPOINT_DIM: Final = 2 * 2 * 65 * 4 + 7
SCHEMA: Final = "g1-mikasa-event-factorized-mapping-diagnosis-r1/mapping-v1"


def transform_raw(
    arrays: Mapping[str, np.ndarray], condition: str, group_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    raw = arrays["raw"].copy()
    valid = arrays["valid"].copy()
    permutation = None
    if condition == "endpoint_repeat":
        for row in range(6):
            for slot in range(2):
                for camera in range(CAMERAS):
                    indices = np.flatnonzero(valid[row, slot, camera])
                    if len(indices):
                        raw[row, slot, camera, indices] = raw[row, slot, camera, indices[-1]]
    elif condition.startswith("shuffle_"):
        base_seed = int(condition.split("_")[1])
        permutation = group_permutations(base_seed, group_id)
        for row in range(6):
            for slot in range(2):
                for camera in range(CAMERAS):
                    valid_indices = np.flatnonzero(valid[row, slot, camera])
                    order = permutation[slot, camera][
                        np.isin(permutation[slot, camera], valid_indices)
                    ]
                    raw[row, slot, camera, valid_indices] = raw[row, slot, camera, order]
    elif condition != "ordered":
        raise ValueError("unknown factorized condition")
    return raw, valid, permutation


def factorized_views(
    arrays: Mapping[str, np.ndarray], condition: str, group_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    raw, valid, permutation = transform_raw(arrays, condition, group_id)
    canonical = canonicalize_positions(raw, valid)
    mask = valid[..., None].astype(np.float32)
    shape = np.concatenate((canonical * mask, mask), axis=-1).reshape(6, -1)
    checkpoint = np.concatenate((raw * mask, mask), axis=-1).reshape(6, -1)
    checkpoint = np.concatenate((checkpoint, arrays["post_proprio"]), axis=-1)
    if shape.shape != (6, SHAPE_DIM) or checkpoint.shape != (6, CHECKPOINT_DIM):
        raise RuntimeError("factorized descriptor dimensions changed")
    if not np.isfinite(shape).all() or not np.isfinite(checkpoint).all():
        raise ValueError("factorized descriptors contain nonfinite values")
    return shape.astype(np.float32), checkpoint.astype(np.float32), permutation


def _npz_bytes(value: Mapping[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **value)
    return stream.getvalue()


def _freeze_view(
    fit: np.ndarray,
    cal: np.ndarray,
    fit_ids: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = fit.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized_fit = (fit - mean) / std
    normalized_cal = (cal - mean) / std
    groups, rows, distances = [], [], []
    for query in normalized_cal:
        distance, group, row = lexicographic_neighbor(query, normalized_fit, fit_ids)
        groups.append(group)
        rows.append(row)
        distances.append(distance)
    return {
        "mean": mean,
        "std": std,
        "neighbor_group": np.asarray(groups, np.int64),
        "neighbor_row": np.asarray(rows, np.int64),
        "distance": np.asarray(distances, np.float32),
    }


def freeze_factorized_mapping(
    cache: EventCache, output_root: Path, deadline: float
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    cached = {group: cache.load_group(group) for group in cache.materials}
    materials = {}
    for condition in CONDITIONS:
        shape_fit, checkpoint_fit, fit_ids = [], [], []
        shape_cal, checkpoint_cal, cal_ids = [], [], []
        permutations = {}
        for group_id in FIT_GROUPS + CALIBRATION_GROUPS:
            shape, checkpoint, permutation = factorized_views(cached[group_id], condition, group_id)
            if permutation is not None:
                permutations[str(group_id)] = permutation
            if group_id in FIT_GROUPS:
                shape_fit.append(shape)
                checkpoint_fit.append(checkpoint)
                fit_ids.extend((group_id, row) for row in range(6))
            else:
                shape_cal.append(shape)
                checkpoint_cal.append(checkpoint)
                cal_ids.extend((group_id, row) for row in range(6))
        shape_mapping = _freeze_view(np.concatenate(shape_fit), np.concatenate(shape_cal), fit_ids)
        checkpoint_mapping = _freeze_view(
            np.concatenate(checkpoint_fit), np.concatenate(checkpoint_cal), fit_ids
        )
        payload = {
            "cal_group": np.asarray([value[0] for value in cal_ids], np.int64),
            "cal_row": np.asarray([value[1] for value in cal_ids], np.int64),
        }
        for prefix, mapping in (
            ("shape", shape_mapping),
            ("checkpoint", checkpoint_mapping),
        ):
            for name, value in mapping.items():
                payload[f"{prefix}_{name}"] = value
        for group_id, value in permutations.items():
            payload[f"permutation_{group_id}"] = value
        path = output_root / f"{condition}.npz"
        materials[condition] = write_new_bytes(path, _npz_bytes(payload))
    if time.monotonic() > deadline:
        raise TimeoutError("factorized mapping exceeded deadline")
    return {"materials": materials, "all_descriptors_distances_finite": True}


class FactorizedMapping:
    def __init__(self, root: Path, materials: Mapping[str, MaterialIdentity]):
        self.root = root.resolve()
        self.materials = dict(materials)
        if set(self.materials) != set(CONDITIONS):
            raise ValueError("factorized mapping condition set changed")

    @classmethod
    def discover(cls, root: Path, result: Mapping[str, Any]) -> "FactorizedMapping":
        resolved = root.resolve()
        if (
            root.is_symlink()
            or result.get("status") != "passed"
            or result.get("mode") != "factorized_event_mapping"
            or result.get("mapping_root") != str(resolved)
            or result.get("schema") != SCHEMA
            or result.get("targets_opened") is not False
            or result.get("all_descriptors_distances_finite") is not True
        ):
            raise ValueError("factorized mapping receipt changed")
        materials = {}
        for condition, spec in result.get("materials", {}).items():
            material = MaterialIdentity(Path(spec["path"]), spec["bytes"], spec["sha256"])
            if material.path.resolve() != resolved / f"{condition}.npz":
                raise ValueError("factorized mapping material escapes root")
            material.verify()
            validate_mapping(condition, material.path)
            materials[condition] = material
        return cls(root, materials)


def validate_mapping(condition: str, path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"cal_group", "cal_row"}
        for prefix in ("shape", "checkpoint"):
            expected |= {
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_neighbor_group",
                f"{prefix}_neighbor_row",
                f"{prefix}_distance",
            }
        if condition.startswith("shuffle_"):
            expected |= {f"permutation_{group}" for group in FIT_GROUPS + CALIBRATION_GROUPS}
        if set(archive.files) != expected:
            raise ValueError("factorized mapping keys changed")
        for prefix, dimension in (("shape", SHAPE_DIM), ("checkpoint", CHECKPOINT_DIM)):
            mean, std = archive[f"{prefix}_mean"], archive[f"{prefix}_std"]
            group = archive[f"{prefix}_neighbor_group"]
            row = archive[f"{prefix}_neighbor_row"]
            distance = archive[f"{prefix}_distance"]
            if (
                mean.shape != (dimension,)
                or std.shape != (dimension,)
                or mean.dtype != np.float32
                or std.dtype != np.float32
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or not (std > 0).all()
                or group.shape != (36,)
                or group.dtype != np.int64
                or row.shape != (36,)
                or row.dtype != np.int64
                or distance.shape != (36,)
                or distance.dtype != np.float32
                or not np.isfinite(distance).all()
                or np.any(distance < 0)
                or not set(group.tolist()).issubset(set(FIT_GROUPS))
                or np.any((row < 0) | (row > 5))
            ):
                raise ValueError(f"factorized {prefix} mapping contract changed")
        if not np.array_equal(
            archive["cal_group"], np.repeat(CALIBRATION_GROUPS, 6)
        ) or not np.array_equal(archive["cal_row"], np.tile(np.arange(6), 6)):
            raise ValueError("factorized calibration row order changed")
        if condition.startswith("shuffle_"):
            base_seed = int(condition.split("_")[1])
            for group in FIT_GROUPS + CALIBRATION_GROUPS:
                value = archive[f"permutation_{group}"]
                if value.dtype != np.int64 or value.shape != (2, 2, TOKENS_PER_SLOT):
                    raise ValueError("factorized permutation shape/dtype changed")
                if not np.array_equal(value, group_permutations(base_seed, group)):
                    raise ValueError("factorized permutation content changed")


def score_factorized(mapping: FactorizedMapping, corpus: ShardedMemoryCorpus) -> dict[str, Any]:
    fit_mean = fit_target_mean(corpus).reshape(2, 12, 2)
    fit = {group: corpus.load_target_group(group) for group in FIT_GROUPS}
    cal = {group: corpus.load_target_group(group) for group in CALIBRATION_GROUPS}
    evidence = {}
    baseline_records = []
    for group in CALIBRATION_GROUPS:
        target = cal[group].target["ordered_checkpoints"]
        error = np.abs(target - fit_mean[None])
        for row in range(6):
            baseline_records.append(
                {
                    "group_id": group,
                    "first_mae_m": float(error[row, 0].mean()),
                    "complete_mae_m": float(error[row].mean()),
                }
            )
    for condition, material in mapping.materials.items():
        with np.load(material.path, allow_pickle=False) as archive:
            records = []
            for index in range(36):
                group = int(archive["cal_group"][index])
                row = int(archive["cal_row"][index])
                shape_group = int(archive["shape_neighbor_group"][index])
                shape_row = int(archive["shape_neighbor_row"][index])
                checkpoint_group = int(archive["checkpoint_neighbor_group"][index])
                checkpoint_row = int(archive["checkpoint_neighbor_row"][index])
                shape_pred = fit[shape_group].target["ordered_shape_ids"][shape_row]
                checkpoint_pred = fit[checkpoint_group].target["ordered_checkpoints"][
                    checkpoint_row
                ]
                target = cal[group].target
                shape_target = target["ordered_shape_ids"][row]
                checkpoint_target = target["ordered_checkpoints"][row]
                error = np.abs(checkpoint_pred - checkpoint_target)
                records.append(
                    {
                        "group_id": group,
                        "first_shape_accuracy": float(shape_pred[0] == shape_target[0]),
                        "second_shape_accuracy": float(shape_pred[1] == shape_target[1]),
                        "ordered_pair_accuracy": float(np.array_equal(shape_pred, shape_target)),
                        "first_mae_m": float(error[0].mean()),
                        "complete_mae_m": float(error.mean()),
                    }
                )
        metrics = (
            "first_shape_accuracy",
            "second_shape_accuracy",
            "ordered_pair_accuracy",
            "first_mae_m",
            "complete_mae_m",
        )
        evidence[condition] = {
            "records": records,
            "group_equal": {
                metric: float(np.mean([record[metric] for record in records])) for metric in metrics
            },
        }
    result = {
        "conditions": evidence,
        "fit_target_mean_control": {
            metric: float(np.mean([record[metric] for record in baseline_records]))
            for metric in ("first_mae_m", "complete_mae_m")
        },
        "baseline_records": baseline_records,
        "technical": {
            "mapping_descriptors_distances_finite": True,
            "metrics_finite": True,
        },
    }
    if not all(
        np.isfinite(value)
        for condition in evidence.values()
        for value in condition["group_equal"].values()
    ):
        raise ValueError("factorized score contains nonfinite metrics")
    return result


def unchanged_gates(evidence: Mapping[str, Any]) -> dict[str, bool]:
    return gates(evidence)
