"""Frozen fit18/cal6 training calibration for the unchanged segment-slot model."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
import hashlib
from pathlib import Path
import random
import statistics
import time
from typing import Any, Final

import numpy as np
import torch

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    BATCH_SIZE,
    ROWS_PER_GROUP,
    FeatureCacheCorpus,
    GroupArrays,
    ShardedMemoryCorpus,
    TargetStats,
    canonical_json_bytes,
    concatenate_feature_groups,
    group_equal_mean,
    serialize_checkpoint,
    write_new_bytes,
    write_new_json,
)
from gr00t.experimental.mikasa_segment_slot_bottleneck import (
    MikasaSegmentSlotModel,
    SegmentSlotBatch,
)
from gr00t.experimental.mikasa_segment_slot_experiment import (
    BETAS,
    FINAL_LEARNING_RATE as _FINAL_LEARNING_RATE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    TRAIN_SEEDS,
    TRAIN_STEPS,
    WARMUP_STEPS as _WARMUP_STEPS,
    WEIGHT_DECAY,
    _checkpoint_payload,
    _lr_multiplier,
    _module_has_finite_nonzero_gradient,
    _targets_for_groups,
    evaluate_group,
    load_checkpoint,
    segment_slot_loss,
)


SPLIT_SALT: Final = "g1-mikasa-segment-slot-train-calibration-r1/split-v1"
FIT_GROUPS: Final = (0, 2, 3, 4, 6, 7, 10, 11, 14, 18, 19, 20, 24, 26, 31, 33, 34, 35)
CALIBRATION_GROUPS: Final = (1, 5, 16, 22, 23, 28)
SPLIT_SHA256: Final = "c9fd3d603d02a73182e5649c6a9a9477af4f6758d398d9126acdc844b33246d2"
CHECKPOINT_THRESHOLD_M: Final = 0.035
ARTIFACT_LIMIT_BYTES: Final = 1024 * 1024 * 1024
FINAL_LEARNING_RATE: Final = _FINAL_LEARNING_RATE
WARMUP_STEPS: Final = _WARMUP_STEPS


def selection_digest(group_id: int) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}|{group_id}".encode("utf-8")).hexdigest()


def validate_frozen_split(train_groups: Sequence[int]) -> dict[str, Any]:
    observed = tuple(int(group_id) for group_id in train_groups)
    if len(observed) != 24 or len(set(observed)) != 24:
        raise ValueError("frozen training cohort must contain 24 unique groups")
    remaining = [group_id for group_id in observed if group_id != 0]
    selected = tuple(
        sorted(
            sorted(remaining, key=selection_digest, reverse=True)[:6],
        )
    )
    fit = tuple(group_id for group_id in observed if group_id not in selected)
    if selected != CALIBRATION_GROUPS or fit != FIT_GROUPS:
        raise ValueError("salted fit/calibration split changed")
    return {
        "schema": "g1-mikasa-segment-slot-train-calibration-r1/split-v1",
        "salt": SPLIT_SALT,
        "fit_groups": list(FIT_GROUPS),
        "calibration_groups": list(CALIBRATION_GROUPS),
        "forced_fit_group": 0,
        "selection_rule": "six_highest_sha256(salt|group_id)_excluding_forced_group_0",
        "canonical_split_sha256": SPLIT_SHA256,
        "selection_digests": {
            str(group_id): selection_digest(group_id) for group_id in observed if group_id != 0
        },
    }


class FitGroupBalancedSchedule:
    """Nine-step cycles; every fit geometry appears exactly twice per cycle."""

    def __init__(self, *, seed: int):
        self.seed = int(seed)
        self.group_ids = FIT_GROUPS

    def batch(self, step: int) -> tuple[int, ...]:
        if step < 0:
            raise ValueError("step must be nonnegative")
        cycle, offset = divmod(step, 9)
        ordered = list(self.group_ids)
        random.Random(self.seed + cycle).shuffle(ordered)
        start = (offset * 4) % len(ordered)
        selected = tuple(ordered[(start + index) % len(ordered)] for index in range(4))
        if len(set(selected)) != 4 or any(group_id not in FIT_GROUPS for group_id in selected):
            raise RuntimeError("fit-only group-balanced schedule changed")
        return selected

    def cycle_counts(self, cycle: int) -> Counter[int]:
        return Counter(
            group_id for step in range(cycle * 9, (cycle + 1) * 9) for group_id in self.batch(step)
        )


class FitTargetCorpus:
    """Train-only target-statistics view exposing only the 18 fit groups."""

    split = "train"

    def __init__(self, corpus: ShardedMemoryCorpus):
        if corpus.split != "train":
            raise ValueError("fit target view requires train corpus")
        self.corpus = corpus

    def iter_target_groups(self) -> Iterator[GroupArrays]:
        for group_id in FIT_GROUPS:
            yield self.corpus.load_target_group(group_id)


class FitOnlyData:
    """Optimizer-facing data view that never loads a calibration group."""

    def __init__(self, corpus: ShardedMemoryCorpus, cache: FeatureCacheCorpus):
        if corpus.split != "train" or cache.split != "train":
            raise ValueError("fit-only data requires train corpus/cache")
        validate_frozen_split(corpus.group_ids)
        cache.validate_source(corpus)
        self.group_ids = FIT_GROUPS
        self.features = {group_id: cache.load_group(group_id) for group_id in FIT_GROUPS}
        self.targets = {group_id: corpus.load_target_group(group_id) for group_id in FIT_GROUPS}

    def batch(self, group_ids: Sequence[int]):
        selected = tuple(int(group_id) for group_id in group_ids)
        if len(selected) != 4 or len(set(selected)) != 4:
            raise ValueError("fit optimizer batch must contain four distinct groups")
        if any(group_id not in FIT_GROUPS for group_id in selected):
            raise PermissionError("optimizer attempted to read a calibration group")
        feature = concatenate_feature_groups([self.features[group_id] for group_id in selected])
        if feature.arrays["temporal_grid"].shape[0] != BATCH_SIZE:
            raise RuntimeError("fit optimizer batch must contain 24 rows")
        return feature, [self.targets[group_id] for group_id in selected]


def fit_target_mean(corpus: ShardedMemoryCorpus) -> np.ndarray:
    values = [
        group.target["ordered_checkpoints"].reshape(ROWS_PER_GROUP, -1).astype(np.float64)
        for group in FitTargetCorpus(corpus).iter_target_groups()
    ]
    mean = np.concatenate(values, axis=0).mean(axis=0)
    if mean.shape != (48,) or not np.isfinite(mean).all():
        raise ValueError("fit target mean control is invalid")
    return mean.astype(np.float32)


def train_fit_seed(
    data: FitOnlyData,
    stats: TargetStats,
    *,
    seed: int,
    device: torch.device,
    deadline: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if seed not in TRAIN_SEEDS:
        raise ValueError("unfrozen fit seed")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MikasaSegmentSlotModel(seed=seed).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    schedule = FitGroupBalancedSchedule(seed=seed)
    curve = []
    gradient_seen = {
        "camera_encoder": False,
        "camera_fusion": False,
        "shape_head": False,
        "checkpoint_head": False,
    }
    schedule_digest = hashlib.sha256()
    optimizer_groups_seen: set[int] = set()
    started = time.monotonic()
    model.train()
    for step in range(TRAIN_STEPS):
        if time.monotonic() > deadline:
            raise TimeoutError("fit training exceeded the frozen wall deadline")
        group_ids = schedule.batch(step)
        if any(group_id in CALIBRATION_GROUPS for group_id in group_ids):
            raise PermissionError("calibration group entered optimizer schedule")
        optimizer_groups_seen.update(group_ids)
        schedule_digest.update((",".join(str(value) for value in group_ids) + "\n").encode())
        feature, target_groups = data.batch(group_ids)
        temporal = torch.from_numpy(feature.arrays["temporal_grid"]).to(
            device=device, dtype=torch.float32
        )
        shape_target, checkpoint_target = _targets_for_groups(target_groups, device)
        output = model(SegmentSlotBatch(temporal))
        total, checkpoint_mse, shape_ce = segment_slot_loss(
            output,
            shape_target,
            checkpoint_target,
            stats,
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("fit training loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        for name in gradient_seen:
            gradient_seen[name] |= _module_has_finite_nonzero_gradient(getattr(model, name))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("fit gradient norm is nonfinite")
        multiplier = _lr_multiplier(step, TRAIN_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * multiplier
        optimizer.step()
        curve.append(
            {
                "step": step + 1,
                "total_loss": float(total.detach().cpu()),
                "checkpoint_mse": float(checkpoint_mse.detach().cpu()),
                "shape_ce": float(shape_ce.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        )
    if not all(gradient_seen.values()):
        raise RuntimeError(f"fit training gradients missing: {gradient_seen}")
    if optimizer_groups_seen != set(FIT_GROUPS):
        raise RuntimeError("fit optimizer did not cover the exact 18 groups")
    checkpoint = write_new_bytes(
        checkpoint_path,
        serialize_checkpoint(_checkpoint_payload(model, stats, seed=seed, steps=TRAIN_STEPS)),
    )
    return {
        "seed": seed,
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "optimizer_groups_seen": sorted(optimizer_groups_seen),
        "calibration_groups_seen_by_optimizer": [],
        "schedule_sha256": schedule_digest.hexdigest(),
        "curve": curve,
        "gradient_seen": gradient_seen,
        "checkpoint": checkpoint,
        "wall_seconds": time.monotonic() - started,
    }


def _baseline_records(corpus: ShardedMemoryCorpus, target_mean: np.ndarray) -> list[dict[str, Any]]:
    records = []
    prediction = target_mean.reshape(2, 12, 2)
    for group_id in CALIBRATION_GROUPS:
        target = corpus.load_target_group(group_id).target["ordered_checkpoints"]
        error = np.abs(target - prediction[None])
        for row in range(ROWS_PER_GROUP):
            records.append(
                {
                    "group_id": group_id,
                    "row": row,
                    "first_mae_m": float(error[row, 0].mean()),
                    "complete_mae_m": float(error[row].mean()),
                }
            )
    return records


def evaluate_calibration(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    checkpoints: Mapping[int, Path],
    target_mean: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if corpus.split != "train" or cache.split != "train":
        raise ValueError("calibration must remain inside the train split")
    validate_frozen_split(corpus.group_ids)
    cache.validate_source(corpus)
    if set(checkpoints) != set(TRAIN_SEEDS):
        raise ValueError("calibration checkpoint seed set changed")
    by_seed = {}
    for seed in TRAIN_SEEDS:
        model, stats = load_checkpoint(
            checkpoints[seed],
            seed=seed,
            expected_steps=TRAIN_STEPS,
            device=device,
        )
        records = []
        for group_id in CALIBRATION_GROUPS:
            records.extend(
                evaluate_group(
                    model,
                    stats,
                    cache.load_group(group_id),
                    corpus.load_target_group(group_id),
                    device=device,
                )
            )
        by_seed[seed] = records
    metrics = (
        "first_shape_accuracy",
        "second_shape_accuracy",
        "ordered_pair_accuracy",
        "first_mae_m",
        "complete_mae_m",
        "swap_shape_exact",
        "swap_checkpoint_exact",
        "swap_checkpoint_max_abs",
    )
    per_seed = {
        str(seed): {metric: group_equal_mean(by_seed[seed], metric) for metric in metrics}
        for seed in TRAIN_SEEDS
    }
    seed_equal = {
        metric: float(np.mean([per_seed[str(seed)][metric] for seed in TRAIN_SEEDS]))
        for metric in metrics
    }
    baseline_records = _baseline_records(corpus, target_mean)
    baseline = {
        "first_mae_m": group_equal_mean(baseline_records, "first_mae_m"),
        "complete_mae_m": group_equal_mean(baseline_records, "complete_mae_m"),
    }
    return {
        "records": by_seed,
        "aggregates": {"by_seed": per_seed, "seed_equal": seed_equal},
        "train_target_mean_control": {
            "target_mean": target_mean.tolist(),
            "records": baseline_records,
            "group_equal": baseline,
        },
    }


def _seed_records(by_seed: Mapping[Any, Any], seed: int) -> Sequence[Mapping[str, Any]]:
    return by_seed[seed] if seed in by_seed else by_seed[str(seed)]


def _group_means(records: Sequence[Mapping[str, Any]], metric: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for record in records:
        values[int(record["group_id"])].append(float(record[metric]))
    return {group_id: float(np.mean(group_values)) for group_id, group_values in values.items()}


def evaluate_calibration_gates(evidence: Mapping[str, Any]) -> dict[str, bool]:
    aggregate = evidence["aggregates"]["seed_equal"]
    baseline = evidence["train_target_mean_control"]["group_equal"]
    gates: dict[str, Any] = {
        "first_shape_accuracy_at_least_90pct": aggregate["first_shape_accuracy"] >= 0.90,
        "second_shape_accuracy_at_least_90pct": aggregate["second_shape_accuracy"] >= 0.90,
        "ordered_pair_accuracy_at_least_80pct": aggregate["ordered_pair_accuracy"] >= 0.80,
        "first_mae_at_most_0p035m": aggregate["first_mae_m"] <= CHECKPOINT_THRESHOLD_M,
        "complete_mae_at_most_0p035m": aggregate["complete_mae_m"] <= CHECKPOINT_THRESHOLD_M,
        "first_mae_improves_20pct_over_train_mean": (
            aggregate["first_mae_m"] <= 0.80 * baseline["first_mae_m"]
        ),
        "complete_mae_improves_20pct_over_train_mean": (
            aggregate["complete_mae_m"] <= 0.80 * baseline["complete_mae_m"]
        ),
        "swap_shape_exact_at_least_80pct": aggregate["swap_shape_exact"] >= 0.80,
        "all_outputs_and_checkpoints_finite": all(
            bool(record["finite"])
            for seed in TRAIN_SEEDS
            for record in _seed_records(evidence["records"], seed)
        ),
    }
    baseline_group = {
        metric: _group_means(evidence["train_target_mean_control"]["records"], metric)
        for metric in ("first_mae_m", "complete_mae_m")
    }
    task_metrics = {
        "first_shape_accuracy": (">=", 0.90),
        "second_shape_accuracy": (">=", 0.90),
        "ordered_pair_accuracy": (">=", 0.80),
        "first_mae_m": ("<=", CHECKPOINT_THRESHOLD_M),
        "complete_mae_m": ("<=", CHECKPOINT_THRESHOLD_M),
    }
    for metric, (direction, threshold) in task_metrics.items():
        passing_groups = 0
        all_seed_medians = True
        for group_id in CALIBRATION_GROUPS:
            values = [
                _group_means(_seed_records(evidence["records"], seed), metric)[group_id]
                for seed in TRAIN_SEEDS
            ]
            group_value = float(np.mean(values))
            passed = group_value >= threshold if direction == ">=" else group_value <= threshold
            if metric in baseline_group:
                passed &= group_value <= 0.80 * baseline_group[metric][group_id]
            passing_groups += bool(passed)
        for seed in TRAIN_SEEDS:
            values = list(_group_means(_seed_records(evidence["records"], seed), metric).values())
            median = statistics.median(values)
            passed = median >= threshold if direction == ">=" else median <= threshold
            if metric in baseline_group:
                baseline_median = statistics.median(baseline_group[metric].values())
                passed &= median <= 0.80 * baseline_median
            all_seed_medians &= passed
        gates[f"{metric}_passes_5of6_groups_and_all_seed_medians"] = (
            passing_groups >= 5 and all_seed_medians
        )
    return {name: bool(value) for name, value in gates.items()}


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def run_train_calibration(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    output_root: Path,
    *,
    device: torch.device,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    split = validate_frozen_split(corpus.group_ids)
    data = FitOnlyData(corpus, cache)
    stats = TargetStats.fit(FitTargetCorpus(corpus))
    target_mean = fit_target_mean(corpus)
    if not np.array_equal(stats.mean, target_mean):
        raise RuntimeError("fit normalization mean and train-target control differ")
    deadline = time.monotonic() + 60 * 60
    training = []
    checkpoints = {}
    for seed in TRAIN_SEEDS:
        path = output_root / "checkpoints" / f"seed_{seed}" / "checkpoint_final.pt"
        training.append(
            train_fit_seed(
                data,
                stats,
                seed=seed,
                device=device,
                deadline=deadline,
                checkpoint_path=path,
            )
        )
        checkpoints[seed] = path
    if any(run["calibration_groups_seen_by_optimizer"] for run in training):
        raise PermissionError("optimizer consumed calibration groups")
    calibration_opened_after_optimizer_steps = len(TRAIN_SEEDS) * TRAIN_STEPS
    evidence = evaluate_calibration(corpus, cache, checkpoints, target_mean, device=device)
    gates = evaluate_calibration_gates(evidence)
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "segment_slot_train_calibration",
        "split": split,
        "seeds": list(TRAIN_SEEDS),
        "steps_per_seed": TRAIN_STEPS,
        "training": training,
        "calibration_opened_after_optimizer_steps": calibration_opened_after_optimizer_steps,
        **evidence,
        "gates": gates,
        "provenance": dict(provenance),
        "original_dev_opened": False,
        "confirmation_opened": False,
        "event_token_mamba_design_allowed": all(gates.values()),
    }
    if time.monotonic() > deadline:
        raise TimeoutError("train-calibration run exceeded its frozen wall deadline")
    projected = _directory_bytes(output_root) + len(canonical_json_bytes(result))
    if projected > ARTIFACT_LIMIT_BYTES:
        raise RuntimeError("train-calibration artifact exceeds 1 GiB")
    write_new_json(output_root / "RESULT.json", result)
    return result
