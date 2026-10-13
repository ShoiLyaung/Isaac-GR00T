"""Pure fixed-spatial-pyramid contracts for the pretrained vision cue audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any, Final

import numpy as np
import torch


EXPERIMENT_ID: Final = "g1-mikasa-history-hard-pretrained-vision-spatial-pyramid-r1"
CAMERAS: Final = ("base_camera", "hand_camera")
CUE_FRAMES: Final = 130
SLOT_FRAMES: Final = 65
TOKENS_PER_IMAGE: Final = 64
GRID_SIDE: Final = 8
PYRAMID_CELLS: Final = 4
CELL_SIDE: Final = 4
CELL_ORDER: Final = ("top_left", "top_right", "bottom_left", "bottom_right")
PYRAMID_CELL_INDICES: Final = tuple(
    tuple(row * GRID_SIDE + column for row in rows for column in columns)
    for rows, columns in (
        (range(0, CELL_SIDE), range(0, CELL_SIDE)),
        (range(0, CELL_SIDE), range(CELL_SIDE, GRID_SIDE)),
        (range(CELL_SIDE, GRID_SIDE), range(0, CELL_SIDE)),
        (range(CELL_SIDE, GRID_SIDE), range(CELL_SIDE, GRID_SIDE)),
    )
)
GLOBAL_MEAN_ATOL: Final = 1e-6
GLOBAL_MEAN_RTOL: Final = 1e-6
EMBED_DIM: Final = 2048
ROWS: Final = 48
ROWS_PER_GROUP: Final = 6
TRAIN_GROUPS: Final = (0, 1, 2, 3, 4, 5)
DEV_GROUPS: Final = (6, 7)
TRAIN_ROWS: Final = tuple(range(36))
DEV_ROWS: Final = tuple(range(36, 48))
PAIR_CODES: Final = ("01", "02", "10", "12", "20", "21")
SHUFFLE_SEEDS: Final = (230941, 230942, 230943, 230944)
PRIMARY_CONDITION: Final = "ordered_current_relative"
GATED_CONTROLS: Final = ("current_repeat", "endpoint_repeat", "sample_shuffle")
DESCRIPTIVE_CONDITION: Final = "raw_ordered"
ALL_CONDITIONS: Final = (PRIMARY_CONDITION, *GATED_CONTROLS, DESCRIPTIVE_CONDITION)


class CueAuditContractError(RuntimeError):
    """A frozen descriptor, label, mapping, or metric contract changed."""


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def validate(self) -> None:
        if (
            self.mean.shape != (len(CAMERAS), PYRAMID_CELLS, EMBED_DIM)
            or self.scale.shape != (len(CAMERAS), PYRAMID_CELLS, EMBED_DIM)
            or self.mean.dtype != np.float32
            or self.scale.dtype != np.float32
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or not bool((self.scale > 0).all())
        ):
            raise CueAuditContractError("invalid train-only descriptor standardizer")


def pool_public_image_features(
    image_features: Sequence[torch.Tensor], *, expected_images: int
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Pool each row-major 8x8 token grid into four ordered 4x4 cells."""

    if len(image_features) != expected_images:
        raise CueAuditContractError(
            f"public get_image_features returned {len(image_features)} images, expected {expected_images}"
        )
    pooled = []
    parity = []
    for index, value in enumerate(image_features):
        if tuple(value.shape) != (TOKENS_PER_IMAGE, EMBED_DIM):
            raise CueAuditContractError(
                f"image {index} merged-token shape changed: {tuple(value.shape)}"
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise CueAuditContractError(
                f"image {index} merged tokens are not finite floating point"
            )
        value_fp32 = value.float()
        cells = torch.stack(
            [value_fp32[list(indices)].mean(dim=0) for indices in PYRAMID_CELL_INDICES]
        )
        global_mean = value.float().mean(dim=0)
        pyramid_mean = cells.mean(dim=0)
        max_abs = float((pyramid_mean - global_mean).abs().max())
        if not torch.allclose(
            pyramid_mean,
            global_mean,
            rtol=GLOBAL_MEAN_RTOL,
            atol=GLOBAL_MEAN_ATOL,
        ):
            raise CueAuditContractError(
                f"image {index} pyramid/global mean parity failed: {max_abs}"
            )
        pooled.append(cells)
        parity.append({"max_abs": max_abs})
    result = torch.stack(pooled)
    if result.shape != (expected_images, PYRAMID_CELLS, EMBED_DIM) or not bool(
        torch.isfinite(result).all()
    ):
        raise CueAuditContractError("spatial-pyramid public vision descriptors are invalid")
    return result, parity


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def encode_public_merged_descriptors(
    backbone: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    frames: int,
    hash_tokens: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Call the frozen public HF boundary and return ``[time,camera,4,2048]``.

    Deliberately accepts no language, state, action, target, or action-head object.
    """

    if frames < 1:
        raise ValueError("frames must be positive")
    expected_images = frames * len(CAMERAS)
    expected_grid = torch.tensor([[1, 16, 16]] * expected_images, dtype=image_grid_thw.dtype)
    if image_grid_thw.device.type != "cpu":
        expected_grid = expected_grid.to(image_grid_thw.device)
    if image_grid_thw.shape != expected_grid.shape or not torch.equal(
        image_grid_thw, expected_grid
    ):
        raise CueAuditContractError("image_grid_thw must be ordered [time,camera] [1,16,16]")
    owner = getattr(backbone, "model", None)
    method = getattr(owner, "get_image_features", None)
    if owner is None or not callable(method):
        raise CueAuditContractError("backbone.model.get_image_features public boundary is missing")
    image_features, deepstack = method(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    pooled, parity = pool_public_image_features(image_features, expected_images=expected_images)
    descriptor = pooled.reshape(frames, len(CAMERAS), PYRAMID_CELLS, EMBED_DIM)
    receipt = {
        "owner_type": f"{type(owner).__module__}.{type(owner).__qualname__}",
        "method_module": getattr(method, "__module__", None),
        "method_qualname": getattr(method, "__qualname__", None),
        "images": expected_images,
        "merged_tokens_per_image": TOKENS_PER_IMAGE,
        "embedding_dim": EMBED_DIM,
        "grid_side": GRID_SIDE,
        "pyramid_cells": PYRAMID_CELLS,
        "cell_side": CELL_SIDE,
        "cell_order": list(CELL_ORDER),
        "global_mean_atol": GLOBAL_MEAN_ATOL,
        "global_mean_rtol": GLOBAL_MEAN_RTOL,
        "global_mean_max_abs": max(value["max_abs"] for value in parity),
        "global_mean_parity": True,
        "deepstack_outputs": len(deepstack),
        "merged_token_sha256": (
            [_tensor_sha256(value) for value in image_features] if hash_tokens else None
        ),
    }
    return descriptor, receipt


def validate_descriptor_cache(
    cue: np.ndarray,
    current: np.ndarray,
    sample_ids: Sequence[str],
) -> None:
    if cue.shape != (ROWS, CUE_FRAMES, len(CAMERAS), PYRAMID_CELLS, EMBED_DIM):
        raise CueAuditContractError(f"cue cache shape changed: {cue.shape}")
    if current.shape != (ROWS, len(CAMERAS), PYRAMID_CELLS, EMBED_DIM):
        raise CueAuditContractError(f"current cache shape changed: {current.shape}")
    for name, value in (("cue", cue), ("current", current)):
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise CueAuditContractError(f"{name} cache must be finite float32")
    if len(sample_ids) != ROWS or len(set(sample_ids)) != ROWS:
        raise CueAuditContractError("cache must contain 48 unique sample IDs")
    if any(not isinstance(value, str) or len(value) != 64 for value in sample_ids):
        raise CueAuditContractError("cache sample ID contract changed")


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or bool((norm <= 0).any()):
        raise CueAuditContractError("descriptor has zero or nonfinite L2 norm")
    result = array / norm.astype(np.float32)
    if not np.isfinite(result).all():
        raise CueAuditContractError("L2-normalized descriptor is nonfinite")
    return np.ascontiguousarray(result, dtype=np.float32)


def _split_slots(value: np.ndarray) -> np.ndarray:
    if value.ndim != 5 or value.shape[1:] != (
        CUE_FRAMES,
        len(CAMERAS),
        PYRAMID_CELLS,
        EMBED_DIM,
    ):
        raise CueAuditContractError("full cue representation must be [rows,130,2,4,2048]")
    return np.ascontiguousarray(
        np.stack((value[:, :SLOT_FRAMES], value[:, SLOT_FRAMES:]), axis=1),
        dtype=np.float32,
    )


def _permutation(sample_id: str, slot: int, seed: int) -> np.ndarray:
    payload = f"g1-vlm-cue-shuffle-v1\0{seed}\0{sample_id}\0{slot}".encode()
    generator_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(generator_seed).permutation(SLOT_FRAMES)


def condition_slots(
    cue: np.ndarray,
    current: np.ndarray,
    sample_ids: Sequence[str],
    condition: str,
    *,
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """Construct target-independent ``[row,slot,65,camera,cell,dim]`` evidence."""

    if cue.ndim != 5 or cue.shape[1:] != (
        CUE_FRAMES,
        len(CAMERAS),
        PYRAMID_CELLS,
        EMBED_DIM,
    ):
        raise CueAuditContractError("cue descriptors have invalid shape")
    if current.shape != (cue.shape[0], len(CAMERAS), PYRAMID_CELLS, EMBED_DIM):
        raise CueAuditContractError("current descriptors have invalid shape")
    if len(sample_ids) != cue.shape[0]:
        raise CueAuditContractError("sample IDs do not align with descriptors")
    cue_norm = _l2_normalize(cue)
    current_norm = _l2_normalize(current)
    relative = cue_norm - current_norm[:, None]

    if condition == PRIMARY_CONDITION:
        result = _split_slots(relative)
    elif condition == "current_repeat":
        result = np.zeros(
            (
                cue.shape[0],
                2,
                SLOT_FRAMES,
                len(CAMERAS),
                PYRAMID_CELLS,
                EMBED_DIM,
            ),
            dtype=np.float32,
        )
    elif condition == "endpoint_repeat":
        first = np.repeat(relative[:, 64:65], SLOT_FRAMES, axis=1)
        second = np.repeat(relative[:, 129:130], SLOT_FRAMES, axis=1)
        result = np.ascontiguousarray(np.stack((first, second), axis=1), dtype=np.float32)
    elif condition == "sample_shuffle":
        if shuffle_seed not in SHUFFLE_SEEDS:
            raise CueAuditContractError("sample_shuffle requires one frozen seed")
        result = _split_slots(relative).copy()
        for row, sample_id in enumerate(sample_ids):
            for slot in range(2):
                result[row, slot] = result[row, slot, _permutation(sample_id, slot, shuffle_seed)]
    elif condition == DESCRIPTIVE_CONDITION:
        result = _split_slots(cue_norm)
    else:
        raise CueAuditContractError(f"unknown descriptor condition: {condition}")
    if not np.isfinite(result).all():
        raise CueAuditContractError(f"condition {condition} produced nonfinite values")
    return result


def fit_standardizer(slots: np.ndarray, train_rows: Sequence[int]) -> Standardizer:
    indices = np.asarray(tuple(int(value) for value in train_rows), dtype=np.int64)
    rows = tuple(indices.tolist())
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(row not in TRAIN_ROWS for row in rows)
        or any(
            sum(row // ROWS_PER_GROUP == group for row in rows) not in (0, ROWS_PER_GROUP)
            for group in TRAIN_GROUPS
        )
    ):
        raise CueAuditContractError(
            "standardization rows must be complete train-only geometry groups"
        )
    train = slots[indices].astype(np.float64)
    # Cameras and pyramid cells remain distinct representation dimensions;
    # fit only across candidate rows, slots, and time.
    axes = (0, 1, 2)
    mean64 = train.mean(axis=axes)
    scale64 = train.std(axis=axes)
    scale64 = np.where(scale64 > 1e-12, scale64, 1.0)
    standardizer = Standardizer(
        mean=np.ascontiguousarray(mean64, dtype=np.float32),
        scale=np.ascontiguousarray(scale64, dtype=np.float32),
    )
    standardizer.validate()
    return standardizer


def apply_standardizer(slots: np.ndarray, standardizer: Standardizer) -> np.ndarray:
    standardizer.validate()
    result = (slots.astype(np.float32) - standardizer.mean) / standardizer.scale
    if not np.isfinite(result).all():
        raise CueAuditContractError("standardized descriptors are nonfinite")
    return np.ascontiguousarray(result, dtype=np.float32)


def build_neighbor_mapping(
    slots: np.ndarray,
    sample_ids: Sequence[str],
    groups: np.ndarray,
    *,
    query_rows: Sequence[int],
    candidate_rows: Sequence[int],
) -> list[dict[str, Any]]:
    """Freeze descriptor-only nearest neighbors; labels are intentionally absent."""

    if slots.ndim != 6 or slots.shape[1:] != (
        2,
        SLOT_FRAMES,
        len(CAMERAS),
        PYRAMID_CELLS,
        EMBED_DIM,
    ):
        raise CueAuditContractError("retrieval slots must be [rows,2,65,2,4,2048]")
    if groups.shape != (slots.shape[0],) or len(sample_ids) != slots.shape[0]:
        raise CueAuditContractError("retrieval identities do not align")
    queries = tuple(int(value) for value in query_rows)
    candidates = tuple(int(value) for value in candidate_rows)
    if len(set(queries)) != len(queries) or len(set(candidates)) != len(candidates):
        raise CueAuditContractError("query/candidate rows must be unique")
    records = []
    for query_row in queries:
        for query_slot in range(2):
            choices = []
            query = slots[query_row, query_slot].astype(np.float64)
            for candidate_row in candidates:
                if int(groups[candidate_row]) == int(groups[query_row]):
                    continue
                for candidate_slot in range(2):
                    delta = query - slots[candidate_row, candidate_slot].astype(np.float64)
                    distance = float(np.mean(np.square(delta), dtype=np.float64))
                    if not np.isfinite(distance):
                        raise CueAuditContractError("retrieval distance is nonfinite")
                    choices.append(
                        (
                            distance,
                            str(sample_ids[candidate_row]),
                            candidate_slot,
                            candidate_row,
                        )
                    )
            if not choices:
                raise CueAuditContractError("query has no allowed other-geometry candidate")
            distance, candidate_id, candidate_slot, candidate_row = min(choices)
            records.append(
                {
                    "query_row": query_row,
                    "query_slot": query_slot,
                    "query_sample_id": str(sample_ids[query_row]),
                    "candidate_row": candidate_row,
                    "candidate_slot": candidate_slot,
                    "candidate_sample_id": candidate_id,
                    "distance": distance,
                }
            )
    return records


def score_neighbor_mapping(
    mapping: Sequence[Mapping[str, Any]], labels: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    """Reveal candidate/query labels only after descriptor mapping is frozen."""

    if labels.ndim != 2 or labels.shape[1] != 2 or labels.dtype != np.int64:
        raise CueAuditContractError("ordered labels must be int64 [rows,2]")
    by_row: dict[int, dict[int, int]] = {}
    records = []
    for record in mapping:
        query_row = int(record["query_row"])
        query_slot = int(record["query_slot"])
        candidate_row = int(record["candidate_row"])
        candidate_slot = int(record["candidate_slot"])
        prediction = int(labels[candidate_row, candidate_slot])
        target = int(labels[query_row, query_slot])
        by_row.setdefault(query_row, {})[query_slot] = prediction
        records.append(
            {
                **dict(record),
                "prediction": prediction,
                "target": target,
                "correct": prediction == target,
                "group_id": int(groups[query_row]),
            }
        )
    if any(set(value) != {0, 1} for value in by_row.values()):
        raise CueAuditContractError("mapping did not produce exactly two slots per query row")
    query_rows = tuple(sorted(by_row))
    group_ids = tuple(sorted({int(groups[row]) for row in query_rows}))
    by_group = {}
    for group in group_ids:
        rows = [row for row in query_rows if int(groups[row]) == group]
        slot0 = [float(by_row[row][0] == int(labels[row, 0])) for row in rows]
        slot1 = [float(by_row[row][1] == int(labels[row, 1])) for row in rows]
        pair = [float(slot0[index] == 1.0 and slot1[index] == 1.0) for index in range(len(rows))]
        by_group[str(group)] = {
            "rows": len(rows),
            "pair_accuracy": float(np.mean(pair)),
            "slot0_accuracy": float(np.mean(slot0)),
            "slot1_accuracy": float(np.mean(slot1)),
        }
    aggregate = {
        key: float(np.mean([value[key] for value in by_group.values()]))
        for key in ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    }
    return {"records": records, "by_group": by_group, "group_equal": aggregate}


def mean_score_sections(sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(sections) != len(SHUFFLE_SEEDS):
        raise CueAuditContractError("shuffle aggregation requires four frozen seeds")
    groups = tuple(sections[0]["by_group"])
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    if any(tuple(section["by_group"]) != groups for section in sections):
        raise CueAuditContractError("shuffle sections use different group sets")
    by_group = {
        group: {
            "rows": sections[0]["by_group"][group]["rows"],
            **{
                metric: float(np.mean([section["by_group"][group][metric] for section in sections]))
                for metric in metrics
            },
        }
        for group in groups
    }
    return {
        "seeds": list(SHUFFLE_SEEDS),
        "sections": list(sections),
        "by_group": by_group,
        "group_equal": {
            metric: float(np.mean([by_group[group][metric] for group in groups]))
            for metric in metrics
        },
    }


def audit_gates(evidence: Mapping[str, Any], *, expected_groups: int) -> dict[str, bool]:
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    primary = evidence[PRIMARY_CONDITION]
    controls = {name: evidence[name] for name in GATED_CONTROLS}
    gates: dict[str, bool] = {
        "primary_pair_at_least_83_33": primary["group_equal"]["pair_accuracy"] >= 5.0 / 6.0,
        "primary_slot0_at_least_91_67": primary["group_equal"]["slot0_accuracy"] >= 11.0 / 12.0,
        "primary_slot1_at_least_91_67": primary["group_equal"]["slot1_accuracy"] >= 11.0 / 12.0,
    }
    required_direction = 5 if expected_groups == 6 else expected_groups
    for name, control in controls.items():
        gates[f"primary_vs_{name}_pair_plus_50"] = (
            primary["group_equal"]["pair_accuracy"] - control["group_equal"]["pair_accuracy"]
            >= 0.50
        )
        for slot in ("slot0_accuracy", "slot1_accuracy"):
            gates[f"primary_vs_{name}_{slot}_plus_25"] = (
                primary["group_equal"][slot] - control["group_equal"][slot] >= 0.25
            )
        directions = sum(
            primary["by_group"][group]["pair_accuracy"]
            > control["by_group"][group]["pair_accuracy"]
            for group in primary["by_group"]
        )
        gates[f"primary_vs_{name}_pair_direction"] = directions >= required_direction
    all_values = [
        section["group_equal"][metric] for section in evidence.values() for metric in metrics
    ]
    gates["all_metrics_finite"] = all(np.isfinite(float(value)) for value in all_values)
    gates["group_count_exact"] = len(primary["by_group"]) == expected_groups
    return {key: bool(value) for key, value in gates.items()}
