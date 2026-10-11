"""Target-blind event trajectories, frozen 1-NN mappings, and scoring gates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any, Final

import numpy as np

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    EXPECTED_SPLIT_GROUPS,
    MaterialIdentity,
    ShardedMemoryCorpus,
    group_equal_mean,
    write_new_bytes,
)
from gr00t.experimental.mikasa_segment_slot_train_calibration import (
    CALIBRATION_GROUPS,
    FIT_GROUPS,
    fit_target_mean,
)


SLOT_BOUNDARIES: Final = ((0, 66), (66, 130))
TOKENS_PER_SLOT: Final = 65
TOP_PIXELS: Final = 64
CAMERAS: Final = 2
CONDITIONS: Final = (
    "ordered",
    "endpoint_repeat",
    "shuffle_230923",
    "shuffle_230924",
    "shuffle_230925",
    "shuffle_230926",
)
SHUFFLE_BASE_SEEDS: Final = (230923, 230924, 230925, 230926)
DESCRIPTOR_DIM: Final = 2 * 2 * 65 * 6 + 7
EPSILON: Final = 1e-6


def extraction_contract() -> dict[str, Any]:
    return {
        "schema": "g1-mikasa-event-trajectory-sufficiency-r1/extraction-v1",
        "top_pixels": TOP_PIXELS,
        "slot_boundaries": [list(value) for value in SLOT_BOUNDARIES],
        "tokens_per_slot": TOKENS_PER_SLOT,
        "descriptor_dim": DESCRIPTOR_DIM,
        "epsilon": EPSILON,
        "cameras": CAMERAS,
    }


def stable_top64_token(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, bool]:
    if previous.shape != (128, 128, 3) or current.shape != previous.shape:
        raise ValueError("event frames must be [128,128,3]")
    energy = np.mean(
        np.abs(current.astype(np.float32) - previous.astype(np.float32)), axis=-1
    ) / np.float32(255.0)
    flat = energy.reshape(-1)
    indices = np.argsort(-flat, kind="stable")[:TOP_PIXELS]
    selected = flat[indices]
    total = np.sum(selected, dtype=np.float32)
    if not total > 0:
        return np.asarray((0.5, 0.5, 0.0), dtype=np.float32), False
    y, x = np.divmod(indices, 128)
    normalized_x = (x.astype(np.float32) + 0.5) / 128.0
    normalized_y = (y.astype(np.float32) + 0.5) / 128.0
    centroid_x = np.sum(normalized_x * selected, dtype=np.float32) / total
    centroid_y = np.sum(normalized_y * selected, dtype=np.float32) / total
    return np.asarray((centroid_x, centroid_y, np.log1p(total)), dtype=np.float32), True


def canonicalize_positions(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if raw.shape[-2:] != (TOKENS_PER_SLOT, 3) or valid.shape != raw.shape[:-1]:
        raise ValueError("event token canonicalization shape changed")
    result = np.zeros((*raw.shape[:-1], 2), dtype=np.float32)
    for index in np.ndindex(raw.shape[:-2]):
        mask = valid[index]
        if not np.any(mask):
            continue
        positions = raw[index][mask, :2].astype(np.float32)
        centered = positions - positions.mean(axis=0, dtype=np.float32)
        rms = np.sqrt(np.mean(np.sum(centered**2, axis=-1), dtype=np.float32))
        if rms <= EPSILON:
            continue
        result[index][mask] = centered / np.float32(rms + EPSILON)
    return result


def extract_row_camera(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rgb.shape != (130, 128, 128, 3) or rgb.dtype != np.uint8:
        raise ValueError("cue RGB must be uint8 [130,128,128,3]")
    raw = np.zeros((2, TOKENS_PER_SLOT, 3), dtype=np.float32)
    valid = np.zeros((2, TOKENS_PER_SLOT), dtype=np.bool_)
    for slot, (start, stop) in enumerate(SLOT_BOUNDARIES):
        transitions = stop - start - 1
        for token in range(transitions):
            raw[slot, token], valid[slot, token] = stable_top64_token(
                rgb[start + token], rgb[start + token + 1]
            )
        if transitions < TOKENS_PER_SLOT:
            raw[slot, transitions:] = raw[slot, transitions - 1]
            valid[slot, transitions:] = False
    canonical = canonicalize_positions(raw, valid)
    return raw, canonical, valid


def extract_group(model: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    raw = np.zeros((6, 2, CAMERAS, TOKENS_PER_SLOT, 3), dtype=np.float32)
    valid = np.zeros((6, 2, CAMERAS, TOKENS_PER_SLOT), dtype=np.bool_)
    for row in range(6):
        for camera, key in enumerate(("cue_base_rgb", "cue_hand_rgb")):
            camera_raw, _, camera_valid = extract_row_camera(model[key][row])
            raw[row, :, camera] = camera_raw
            valid[row, :, camera] = camera_valid
    canonical = canonicalize_positions(raw, valid)
    post = np.ascontiguousarray(model["post_proprio"], dtype=np.float32)
    return {"raw": raw, "canonical": canonical, "valid": valid, "post_proprio": post}


def descriptor(raw: np.ndarray, valid: np.ndarray, post_proprio: np.ndarray) -> np.ndarray:
    canonical = canonicalize_positions(raw, valid)
    mask = valid[..., None].astype(np.float32)
    fields = np.concatenate((raw * mask, canonical * mask, mask), axis=-1)
    value = np.concatenate((fields.reshape(fields.shape[0], -1), post_proprio), axis=-1)
    if value.shape[1] != DESCRIPTOR_DIM or not np.isfinite(value).all():
        raise ValueError("event descriptor contract changed")
    return np.ascontiguousarray(value, dtype=np.float32)


def shuffle_seed(base_seed: int, group_id: int, slot: int, camera: int) -> int:
    payload = f"{base_seed}|{group_id}|{slot}|{camera}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def group_permutations(base_seed: int, group_id: int) -> np.ndarray:
    result = np.zeros((2, CAMERAS, TOKENS_PER_SLOT), dtype=np.int64)
    for slot in range(2):
        for camera in range(CAMERAS):
            result[slot, camera] = np.random.Generator(
                np.random.PCG64(shuffle_seed(base_seed, group_id, slot, camera))
            ).permutation(TOKENS_PER_SLOT)
    return result


def condition_view(
    arrays: Mapping[str, np.ndarray], condition: str, group_id: int
) -> tuple[np.ndarray, np.ndarray | None]:
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
        raise ValueError("unknown event trajectory condition")
    return descriptor(raw, valid, arrays["post_proprio"]), permutation


def _npz_bytes(value: Mapping[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **value)
    return stream.getvalue()


def lexicographic_neighbor(
    query: np.ndarray,
    fit: np.ndarray,
    fit_ids: list[tuple[int, int]],
) -> tuple[float, int, int]:
    mse = np.mean((fit - query) ** 2, axis=1)
    candidates = [
        (float(mse[index]), fit_ids[index][0], fit_ids[index][1])
        for index in range(len(fit_ids))
        if np.isfinite(mse[index])
    ]
    if not candidates:
        raise ValueError("no finite cross-group neighbor")
    return min(candidates)


class EventCache:
    def __init__(self, root: Path, materials: Mapping[int, MaterialIdentity]):
        self.root = root.resolve()
        self.materials = dict(sorted(materials.items()))
        if tuple(self.materials) != EXPECTED_SPLIT_GROUPS["train"]:
            raise ValueError("event cache train groups changed")

    @classmethod
    def discover(
        cls, root: Path, result: Mapping[str, Any], corpus: ShardedMemoryCorpus
    ) -> "EventCache":
        resolved = root.resolve()
        if {entry.name for entry in resolved.iterdir()} != {"train", "RESULT.json"}:
            raise ValueError("event cache root entry set changed")
        if root.is_symlink() or result.get("cache_root") != str(resolved):
            raise ValueError("event cache root receipt changed")
        if (
            result.get("status") != "passed"
            or result.get("mode") != "event_trajectory_cache"
            or result.get("targets_opened") is not False
            or result.get("dev_opened") is not False
            or result.get("confirmation_opened") is not False
            or result.get("extraction_contract") != extraction_contract()
        ):
            raise ValueError("event cache boundary receipt changed")
        materials = {}
        records = result.get("group_records")
        if not isinstance(records, list):
            raise ValueError("event cache group receipts missing")
        for record in records:
            group_id = record.get("group_id")
            spec = record.get("event")
            if not isinstance(group_id, int) or not isinstance(spec, dict):
                raise ValueError("event cache group receipt changed")
            material = MaterialIdentity(Path(spec["path"]), int(spec["bytes"]), str(spec["sha256"]))
            expected = resolved / "train" / f"group_{group_id:03d}" / "events.npz"
            if material.path.resolve() != expected:
                raise ValueError("event cache material escapes root")
            material.verify()
            for source_name in ("source_manifest", "source_model"):
                source = record.get(source_name)
                if (
                    not isinstance(source, dict)
                    or set(source) != {"path", "bytes", "sha256"}
                    or not isinstance(source["path"], str)
                    or not isinstance(source["bytes"], int)
                    or not isinstance(source["sha256"], str)
                ):
                    raise ValueError(f"event cache {source_name} receipt changed")
            shard = corpus._by_group[group_id]
            for source_name, source_material in (
                ("source_manifest", shard.manifest),
                ("source_model", shard.model_input),
            ):
                if record[source_name] != {
                    "path": str(source_material.path.resolve()),
                    "bytes": source_material.bytes,
                    "sha256": source_material.sha256,
                }:
                    raise ValueError(f"event cache {source_name} source binding changed")
            materials[group_id] = material
        return cls(root, materials)

    def load_group(self, group_id: int) -> dict[str, np.ndarray]:
        material = self.materials[group_id]
        material.verify()
        with np.load(material.path, allow_pickle=False) as archive:
            expected = {"raw", "canonical", "valid", "post_proprio"}
            if set(archive.files) != expected:
                raise ValueError("event cache archive keys changed")
            arrays = {key: np.ascontiguousarray(archive[key]) for key in expected}
        expected_specs = {
            "raw": ((6, 2, 2, 65, 3), np.float32),
            "canonical": ((6, 2, 2, 65, 2), np.float32),
            "valid": ((6, 2, 2, 65), np.bool_),
            "post_proprio": ((6, 7), np.float32),
        }
        for key, (shape, dtype) in expected_specs.items():
            value = arrays[key]
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"event cache {key} shape/dtype changed")
            if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"event cache {key} contains nonfinite values")
        if not np.all(arrays["canonical"][~arrays["valid"]] == 0):
            raise ValueError("invalid event tokens must have zero canonical position")
        recomputed = canonicalize_positions(arrays["raw"], arrays["valid"])
        if not np.allclose(arrays["canonical"], recomputed, rtol=0, atol=1e-6):
            raise ValueError("event canonical position invariant changed")
        if arrays["valid"][:, 1, :, 63:].any() or not np.array_equal(
            arrays["raw"][:, 1, :, 63:],
            np.broadcast_to(arrays["raw"][:, 1, :, 62:63], (6, 2, 2, 3)),
        ):
            raise ValueError("slot1 63-to-65 padding contract changed")
        return arrays


def build_cache(corpus: ShardedMemoryCorpus, output_root: Path, deadline: float) -> dict[str, Any]:
    if output_root.exists() or corpus.split != "train":
        raise FileExistsError(output_root)
    records = []
    for group_id in corpus.group_ids:
        if time.monotonic() > deadline:
            raise TimeoutError("event cache exceeded deadline")
        arrays = extract_group(corpus.load_model_group(group_id))
        path = output_root / "train" / f"group_{group_id:03d}" / "events.npz"
        identity = write_new_bytes(path, _npz_bytes(arrays))
        records.append(
            {
                "group_id": group_id,
                "source_manifest": {
                    "path": str(corpus._by_group[group_id].manifest.path.resolve()),
                    "bytes": corpus._by_group[group_id].manifest.bytes,
                    "sha256": corpus._by_group[group_id].manifest.sha256,
                },
                "source_model": {
                    "path": str(corpus._by_group[group_id].model_input.path.resolve()),
                    "bytes": corpus._by_group[group_id].model_input.bytes,
                    "sha256": corpus._by_group[group_id].model_input.sha256,
                },
                "event": identity,
            }
        )
    return {"records": records}


def freeze_mapping(cache: EventCache, output_root: Path, deadline: float) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    arrays = {group_id: cache.load_group(group_id) for group_id in cache.materials}
    materials = {}
    all_descriptors_distances_finite = True
    for condition in CONDITIONS:
        fit_desc, fit_ids, permutations = [], [], {}
        cal_desc, cal_ids = [], []
        for group_id in FIT_GROUPS + CALIBRATION_GROUPS:
            view, permutation = condition_view(arrays[group_id], condition, group_id)
            if permutation is not None:
                permutations[str(group_id)] = permutation
            target_list, id_list = (
                (fit_desc, fit_ids) if group_id in FIT_GROUPS else (cal_desc, cal_ids)
            )
            target_list.append(view)
            id_list.extend((group_id, row) for row in range(6))
        fit = np.concatenate(fit_desc)
        cal = np.concatenate(cal_desc)
        mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = fit.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        normalized_fit = (fit - mean) / std
        normalized_cal = (cal - mean) / std
        all_descriptors_distances_finite &= bool(
            np.isfinite(fit).all()
            and np.isfinite(cal).all()
            and np.isfinite(normalized_fit).all()
            and np.isfinite(normalized_cal).all()
            and np.isfinite(mean).all()
            and np.isfinite(std).all()
            and (std > 0).all()
        )
        neighbor_group, neighbor_row, distances = [], [], []
        for query in normalized_cal:
            distance, group_id, row = lexicographic_neighbor(query, normalized_fit, fit_ids)
            neighbor_group.append(group_id)
            neighbor_row.append(row)
            distances.append(distance)
        payload = {
            "mean": mean,
            "std": std,
            "neighbor_group": np.asarray(neighbor_group, np.int64),
            "neighbor_row": np.asarray(neighbor_row, np.int64),
            "distance": np.asarray(distances, np.float32),
            "cal_group": np.asarray([value[0] for value in cal_ids], np.int64),
            "cal_row": np.asarray([value[1] for value in cal_ids], np.int64),
        }
        for group_id, permutation in permutations.items():
            payload[f"permutation_{group_id}"] = permutation
        path = output_root / f"{condition}.npz"
        materials[condition] = write_new_bytes(path, _npz_bytes(payload))
        all_descriptors_distances_finite &= bool(np.isfinite(payload["distance"]).all())
    if time.monotonic() > deadline:
        raise TimeoutError("event mapping exceeded deadline")
    if not all_descriptors_distances_finite:
        raise ValueError("event descriptors or mapping distances are nonfinite")
    return {
        "materials": materials,
        "all_descriptors_distances_finite": True,
    }


def validate_mapping_material(condition: str, path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        base_keys = {
            "mean",
            "std",
            "neighbor_group",
            "neighbor_row",
            "distance",
            "cal_group",
            "cal_row",
        }
        expected_permutations = (
            {f"permutation_{group_id}" for group_id in EXPECTED_SPLIT_GROUPS["train"]}
            if condition.startswith("shuffle_")
            else set()
        )
        if set(archive.files) != base_keys | expected_permutations:
            raise ValueError(f"mapping archive keys changed for {condition}")
        mean, std = archive["mean"], archive["std"]
        neighbor_group = archive["neighbor_group"]
        neighbor_row = archive["neighbor_row"]
        distance = archive["distance"]
        cal_group, cal_row = archive["cal_group"], archive["cal_row"]
        if (
            mean.shape != (DESCRIPTOR_DIM,)
            or std.shape != (DESCRIPTOR_DIM,)
            or mean.dtype != np.float32
            or std.dtype != np.float32
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or not (std > 0).all()
        ):
            raise ValueError("mapping normalization contract changed")
        for value, label in (
            (neighbor_group, "neighbor_group"),
            (neighbor_row, "neighbor_row"),
            (cal_group, "cal_group"),
            (cal_row, "cal_row"),
        ):
            if value.shape != (36,) or value.dtype != np.int64:
                raise ValueError(f"mapping {label} shape/dtype changed")
        if distance.shape != (36,) or distance.dtype != np.float32:
            raise ValueError("mapping distance shape/dtype changed")
        if not np.isfinite(distance).all() or np.any(distance < 0):
            raise ValueError("mapping distances must be finite and nonnegative")
        if not set(neighbor_group.tolist()).issubset(set(FIT_GROUPS)) or np.any(
            (neighbor_row < 0) | (neighbor_row > 5)
        ):
            raise ValueError("mapping neighbor indices changed")
        if not np.array_equal(cal_group, np.repeat(CALIBRATION_GROUPS, 6)) or not np.array_equal(
            cal_row, np.tile(np.arange(6), 6)
        ):
            raise ValueError("mapping calibration row order changed")
        if condition.startswith("shuffle_"):
            base_seed = int(condition.split("_")[1])
            for group_id in EXPECTED_SPLIT_GROUPS["train"]:
                value = archive[f"permutation_{group_id}"]
                if value.shape != (2, 2, 65) or value.dtype != np.int64:
                    raise ValueError("mapping permutation shape/dtype changed")
                if not np.array_equal(value, group_permutations(base_seed, group_id)):
                    raise ValueError("mapping permutation content changed")


class FrozenMapping:
    def __init__(self, root: Path, materials: Mapping[str, MaterialIdentity]):
        self.root = root.resolve()
        self.materials = dict(materials)
        if set(self.materials) != set(CONDITIONS):
            raise ValueError("mapping condition set changed")

    @classmethod
    def discover(
        cls, root: Path, result: Mapping[str, Any], corpus: ShardedMemoryCorpus
    ) -> "FrozenMapping":
        resolved = root.resolve()
        if {entry.name for entry in resolved.iterdir()} != {
            "RESULT.json",
            *[f"{condition}.npz" for condition in CONDITIONS],
        }:
            raise ValueError("mapping root entry set changed")
        if (
            root.is_symlink()
            or result.get("status") != "passed"
            or result.get("mode") != "event_trajectory_mapping"
            or result.get("mapping_root") != str(resolved)
            or result.get("targets_opened") is not False
            or result.get("dev_opened") is not False
            or result.get("confirmation_opened") is not False
            or result.get("all_descriptors_distances_finite") is not True
        ):
            raise ValueError("mapping receipt root/boundary changed")
        cache_spec = result.get("cache")
        if not isinstance(cache_spec, dict):
            raise ValueError("mapping cache provenance missing")
        cache_result_path = Path(cache_spec["path"])
        cache_raw = cache_result_path.read_bytes()
        if (
            cache_result_path.is_symlink()
            or len(cache_raw) != cache_spec["bytes"]
            or hashlib.sha256(cache_raw).hexdigest() != cache_spec["sha256"]
        ):
            raise ValueError("mapping cache provenance identity changed")
        cache_result = json.loads(cache_raw)
        EventCache.discover(Path(cache_result["cache_root"]), cache_result, corpus)
        materials = {}
        specs = result.get("materials")
        if not isinstance(specs, dict) or set(specs) != set(CONDITIONS):
            raise ValueError("mapping material receipt set changed")
        for condition, spec in specs.items():
            material = MaterialIdentity(Path(spec["path"]), spec["bytes"], spec["sha256"])
            if material.path.resolve() != resolved / f"{condition}.npz":
                raise ValueError("mapping material escapes root")
            material.verify()
            validate_mapping_material(condition, material.path)
            materials[condition] = material
        return cls(root, materials)


def score_mapping(
    mapping: FrozenMapping,
    corpus: ShardedMemoryCorpus,
) -> dict[str, Any]:
    # Target-open order is explicit: all fit labels, then calibration labels once.
    fit_mean = fit_target_mean(corpus).reshape(2, 12, 2)
    fit_targets = {group_id: corpus.load_target_group(group_id) for group_id in FIT_GROUPS}
    cal_targets = {group_id: corpus.load_target_group(group_id) for group_id in CALIBRATION_GROUPS}
    evidence = {}
    baseline_records = []
    for group_id in CALIBRATION_GROUPS:
        target = cal_targets[group_id].target["ordered_checkpoints"]
        error = np.abs(target - fit_mean[None])
        for row in range(6):
            baseline_records.append(
                {
                    "group_id": group_id,
                    "first_mae_m": float(error[row, 0].mean()),
                    "complete_mae_m": float(error[row].mean()),
                }
            )
    for condition in CONDITIONS:
        with np.load(mapping.materials[condition].path, allow_pickle=False) as mapping_values:
            records = []
            for index in range(36):
                group_id = int(mapping_values["cal_group"][index])
                row = int(mapping_values["cal_row"][index])
                fit_group = int(mapping_values["neighbor_group"][index])
                fit_row = int(mapping_values["neighbor_row"][index])
                prediction = fit_targets[fit_group].target
                target = cal_targets[group_id].target
                shape_pred = prediction["ordered_shape_ids"][fit_row]
                shape_target = target["ordered_shape_ids"][row]
                checkpoint_pred = prediction["ordered_checkpoints"][fit_row]
                checkpoint_target = target["ordered_checkpoints"][row]
                error = np.abs(checkpoint_pred - checkpoint_target)
                records.append(
                    {
                        "group_id": group_id,
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
            "group_equal": {metric: group_equal_mean(records, metric) for metric in metrics},
        }
    all_metrics_finite = all(
        np.isfinite(float(value))
        for condition in evidence.values()
        for record in condition["records"]
        for key, value in record.items()
        if key != "group_id"
    )
    if not all_metrics_finite:
        raise ValueError("event scoring metrics are nonfinite")
    return {
        "conditions": evidence,
        "fit_target_mean_control": {
            metric: group_equal_mean(baseline_records, metric)
            for metric in ("first_mae_m", "complete_mae_m")
        },
        "baseline_records": baseline_records,
        "technical": {
            "mapping_descriptors_distances_finite": True,
            "metrics_finite": True,
        },
    }


def gates(evidence: Mapping[str, Any]) -> dict[str, bool]:
    ordered = evidence["conditions"]["ordered"]
    aggregate = ordered["group_equal"]
    control_mean = evidence["fit_target_mean_control"]
    result: dict[str, Any] = {
        "all_descriptors_distances_metrics_finite": (
            evidence.get("technical")
            == {
                "mapping_descriptors_distances_finite": True,
                "metrics_finite": True,
            }
        ),
        "first_shape_accuracy_at_least_90pct": aggregate["first_shape_accuracy"] >= 0.90,
        "second_shape_accuracy_at_least_90pct": aggregate["second_shape_accuracy"] >= 0.90,
        "ordered_pair_accuracy_at_least_80pct": aggregate["ordered_pair_accuracy"] >= 0.80,
        "first_mae_at_most_0p035": aggregate["first_mae_m"] <= 0.035,
        "complete_mae_at_most_0p035": aggregate["complete_mae_m"] <= 0.035,
        "first_mae_20pct_better_than_mean": aggregate["first_mae_m"]
        <= 0.8 * control_mean["first_mae_m"],
        "complete_mae_20pct_better_than_mean": aggregate["complete_mae_m"]
        <= 0.8 * control_mean["complete_mae_m"],
    }
    thresholds = {
        "first_shape_accuracy": (">=", 0.90),
        "second_shape_accuracy": (">=", 0.90),
        "ordered_pair_accuracy": (">=", 0.80),
        "first_mae_m": ("<=", 0.035),
        "complete_mae_m": ("<=", 0.035),
    }
    mean_control_groups = {
        metric: _group_values(evidence["baseline_records"], metric)
        for metric in ("first_mae_m", "complete_mae_m")
    }
    for metric, (direction, threshold) in thresholds.items():
        groups = _group_values(ordered["records"], metric)
        result[f"ordered_{metric}_passes_5of6"] = (
            sum(
                (
                    value >= threshold
                    if direction == ">="
                    else value <= threshold and value <= 0.8 * mean_control_groups[metric][group]
                )
                for group, value in groups.items()
            )
            >= 5
        )
    for condition in CONDITIONS[1:]:
        control = evidence["conditions"][condition]
        for metric, margin in (
            ("first_shape_accuracy", 0.10),
            ("second_shape_accuracy", 0.10),
            ("ordered_pair_accuracy", 0.20),
        ):
            result[f"ordered_{metric}_beats_{condition}"] = (
                aggregate[metric] >= control["group_equal"][metric] + margin
            )
        for metric in ("first_mae_m", "complete_mae_m"):
            result[f"ordered_{metric}_beats_{condition}"] = (
                aggregate[metric] <= 0.9 * control["group_equal"][metric]
            )
        for metric in thresholds:
            ordered_groups = _group_values(ordered["records"], metric)
            control_groups = _group_values(control["records"], metric)
            higher = "accuracy" in metric
            result[f"ordered_{metric}_direction_5of6_vs_{condition}"] = (
                sum(
                    ordered_groups[group] > control_groups[group]
                    if higher
                    else ordered_groups[group] < control_groups[group]
                    for group in CALIBRATION_GROUPS
                )
                >= 5
            )
    return {name: bool(value) for name, value in result.items()}


def _group_values(records: list[Mapping[str, Any]], metric: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for record in records:
        values[int(record["group_id"])].append(float(record[metric]))
    return {group: float(np.mean(group_values)) for group, group_values in values.items()}
