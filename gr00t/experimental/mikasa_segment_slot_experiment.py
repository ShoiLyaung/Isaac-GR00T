"""Train and evaluate the MIKASA segment-slot bottleneck diagnosis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import statistics
import time
from typing import Any, Final

import numpy as np
import torch
from torch.nn import functional as F

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    BATCH_SIZE,
    ROWS_PER_GROUP,
    FeatureCacheCorpus,
    FeatureGroup,
    GroupArrays,
    GroupBalancedSchedule,
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
    CHECKPOINT_DIM_PER_SLOT,
    SHAPE_CLASSES,
    SLOTS,
    MikasaSegmentSlotModel,
    SegmentSlotBatch,
    SegmentSlotOutput,
    trainable_parameter_count,
)


TRAIN_SEEDS: Final = (230923, 230924, 230925)
OVERFIT_STEPS: Final = 500
TRAIN_STEPS: Final = 1500
LEARNING_RATE: Final = 3e-4
FINAL_LEARNING_RATE: Final = 3e-5
WARMUP_STEPS: Final = 100
WEIGHT_DECAY: Final = 1e-4
BETAS: Final = (0.9, 0.95)
GRADIENT_CLIP: Final = 1.0
DENSE_FIRST_MAE_M: Final = 0.06892572091547428
DENSE_COMPLETE_MAE_M: Final = 0.06577257129053275
OVERFIT_ARTIFACT_LIMIT: Final = 512 * 1024 * 1024
MATCHED_ARTIFACT_LIMIT: Final = 1024 * 1024 * 1024


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _enforce_result_budget(output_root: Path, result: Mapping[str, Any], limit: int) -> None:
    projected = _directory_bytes(output_root) + len(canonical_json_bytes(result))
    if projected > limit:
        raise RuntimeError(f"segment-slot artifact would exceed {limit} bytes")


class LoadedSegmentSlotData:
    """Bounded train view over the frozen feature cache and label shards."""

    def __init__(self, corpus: ShardedMemoryCorpus, cache: FeatureCacheCorpus):
        if corpus.split != "train" or cache.split != "train":
            raise ValueError("segment-slot training data must use train split")
        cache.validate_source(corpus)
        self.features = {group_id: cache.load_group(group_id) for group_id in corpus.group_ids}
        self.targets = {
            group_id: corpus.load_target_group(group_id) for group_id in corpus.group_ids
        }
        self.group_ids = corpus.group_ids

    def batch(self, group_ids: Sequence[int]) -> tuple[FeatureGroup, list[GroupArrays]]:
        if len(group_ids) not in (1, 4):
            raise ValueError("segment-slot batch must contain one overfit or four matched groups")
        features = concatenate_feature_groups([self.features[group_id] for group_id in group_ids])
        targets = [self.targets[group_id] for group_id in group_ids]
        expected_rows = ROWS_PER_GROUP * len(group_ids)
        if features.arrays["temporal_grid"].shape[0] != expected_rows:
            raise RuntimeError("segment-slot group-balanced row count changed")
        if len(group_ids) == 4 and expected_rows != BATCH_SIZE:
            raise RuntimeError("segment-slot matched batch must contain 24 rows")
        return features, targets


def _targets_for_groups(
    groups: Sequence[GroupArrays], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    shapes = np.concatenate([group.target["ordered_shape_ids"] for group in groups], axis=0)
    checkpoints = np.concatenate([group.target["ordered_checkpoints"] for group in groups], axis=0)
    return (
        torch.from_numpy(np.ascontiguousarray(shapes, dtype=np.int64)).to(device=device),
        torch.from_numpy(np.ascontiguousarray(checkpoints, dtype=np.float32)).to(device=device),
    )


def _lr_multiplier(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = (step + 1 - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return (
        FINAL_LEARNING_RATE / LEARNING_RATE + (1.0 - FINAL_LEARNING_RATE / LEARNING_RATE) * cosine
    )


def segment_slot_loss(
    output: SegmentSlotOutput,
    shape_target: torch.Tensor,
    checkpoint_target_physical: torch.Tensor,
    stats: TargetStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if shape_target.shape != (output.shape_logits.shape[0], SLOTS):
        raise ValueError("shape target must have shape [batch,2]")
    if checkpoint_target_physical.shape != (
        output.shape_logits.shape[0],
        SLOTS,
        12,
        2,
    ):
        raise ValueError("checkpoint target must have shape [batch,2,12,2]")
    normalized_target = stats.normalize(
        checkpoint_target_physical.reshape(output.shape_logits.shape[0], -1)
    ).reshape(-1, SLOTS, CHECKPOINT_DIM_PER_SLOT)
    checkpoint_mse = torch.square(output.normalized_checkpoints - normalized_target).mean()
    shape_ce = F.cross_entropy(
        output.shape_logits.reshape(-1, SHAPE_CLASSES),
        shape_target.reshape(-1),
        reduction="mean",
    )
    return checkpoint_mse + shape_ce, checkpoint_mse, shape_ce


def _module_has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    return (
        bool(gradients)
        and all(
            gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        and any(
            bool(torch.count_nonzero(gradient)) for gradient in gradients if gradient is not None
        )
    )


def _checkpoint_payload(
    model: MikasaSegmentSlotModel,
    stats: TargetStats,
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    return {
        "schema": "g1-mikasa-segment-slot-bottleneck-diagnosis-r1/checkpoint-v1",
        "seed": seed,
        "steps": steps,
        "dtype": "float32",
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "target_stats": stats.as_json(),
        "trainable_parameters": trainable_parameter_count(model),
    }


def load_checkpoint(
    path: Path, *, seed: int, expected_steps: int, device: torch.device
) -> tuple[MikasaSegmentSlotModel, TargetStats]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != "g1-mikasa-segment-slot-bottleneck-diagnosis-r1/checkpoint-v1":
        raise ValueError("segment-slot checkpoint schema changed")
    if (
        value.get("seed") != seed
        or value.get("steps") != expected_steps
        or value.get("dtype") != "float32"
    ):
        raise ValueError("segment-slot checkpoint identity changed")
    model = MikasaSegmentSlotModel(seed=seed).to(device=device, dtype=torch.float32)
    model.load_state_dict(value["model_state_dict"], strict=True)
    if trainable_parameter_count(model) != value.get("trainable_parameters"):
        raise ValueError("segment-slot checkpoint parameter identity changed")
    return model.eval(), TargetStats.from_json(value["target_stats"])


def train_one(
    data: LoadedSegmentSlotData,
    stats: TargetStats,
    *,
    seed: int,
    steps: int,
    device: torch.device,
    deadline: float,
    checkpoint_path: Path,
    overfit_group: int | None = None,
) -> dict[str, Any]:
    if seed not in TRAIN_SEEDS or steps not in (OVERFIT_STEPS, TRAIN_STEPS):
        raise ValueError("unfrozen segment-slot seed or step count")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MikasaSegmentSlotModel(seed=seed).to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("segment-slot model is not entirely FP32")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    schedule = GroupBalancedSchedule(data.group_ids, seed=seed)
    curve: list[dict[str, float | int]] = []
    gradient_seen = {
        "camera_encoder": False,
        "camera_fusion": False,
        "shape_head": False,
        "checkpoint_head": False,
    }
    started = time.monotonic()
    model.train()
    for step in range(steps):
        if time.monotonic() > deadline:
            raise TimeoutError("segment-slot training exceeded its frozen wall deadline")
        group_ids = (overfit_group,) if overfit_group is not None else schedule.batch(step)
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
            raise FloatingPointError("segment-slot loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        for name in gradient_seen:
            gradient_seen[name] |= _module_has_finite_nonzero_gradient(getattr(model, name))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("segment-slot gradient norm is nonfinite")
        multiplier = _lr_multiplier(step, steps)
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
        raise RuntimeError(f"segment-slot gradients missing: {gradient_seen}")
    checkpoint = write_new_bytes(
        checkpoint_path,
        serialize_checkpoint(_checkpoint_payload(model, stats, seed=seed, steps=steps)),
    )
    return {
        "seed": seed,
        "steps": steps,
        "batch_size": ROWS_PER_GROUP if overfit_group is not None else BATCH_SIZE,
        "curve": curve,
        "gradient_seen": gradient_seen,
        "checkpoint": checkpoint,
        "wall_seconds": time.monotonic() - started,
    }


@torch.inference_mode()
def evaluate_group(
    model: MikasaSegmentSlotModel,
    stats: TargetStats,
    feature: FeatureGroup,
    target: GroupArrays,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    temporal = torch.from_numpy(feature.arrays["temporal_grid"]).to(
        device=device, dtype=torch.float32
    )
    shape_target = torch.from_numpy(
        np.ascontiguousarray(target.target["ordered_shape_ids"], dtype=np.int64)
    ).to(device=device)
    checkpoint_target = torch.from_numpy(
        np.ascontiguousarray(target.target["ordered_checkpoints"], dtype=np.float32)
    ).to(device=device)
    slot_maps = model.pool_segments(temporal)
    output = model.forward_slot_maps(slot_maps)
    swapped = model.forward_slot_maps(slot_maps.flip(1))
    prediction_shape = output.shape_logits.argmax(dim=-1)
    swapped_shape = swapped.shape_logits.argmax(dim=-1)
    physical = stats.denormalize(output.normalized_checkpoints.reshape(ROWS_PER_GROUP, -1)).reshape(
        ROWS_PER_GROUP, SLOTS, 12, 2
    )
    error = (physical - checkpoint_target).abs()
    swap_checkpoint_error = (
        swapped.normalized_checkpoints - output.normalized_checkpoints.flip(1)
    ).abs()
    finite = (
        torch.isfinite(output.shape_logits).all(dim=(1, 2))
        & torch.isfinite(output.normalized_checkpoints).all(dim=(1, 2))
        & torch.isfinite(physical).all(dim=(1, 2, 3))
        & torch.isfinite(swapped.shape_logits).all(dim=(1, 2))
        & torch.isfinite(swapped.normalized_checkpoints).all(dim=(1, 2))
    )
    records = []
    for row in range(ROWS_PER_GROUP):
        records.append(
            {
                "group_id": feature.group_id,
                "row": row,
                "first_shape_accuracy": float(prediction_shape[row, 0] == shape_target[row, 0]),
                "second_shape_accuracy": float(prediction_shape[row, 1] == shape_target[row, 1]),
                "ordered_pair_accuracy": float(
                    bool(torch.equal(prediction_shape[row], shape_target[row]))
                ),
                "first_mae_m": float(error[row, 0].mean().cpu()),
                "second_mae_m": float(error[row, 1].mean().cpu()),
                "complete_mae_m": float(error[row].mean().cpu()),
                "swap_shape_exact": float(
                    bool(torch.equal(swapped_shape[row], prediction_shape[row].flip(0)))
                ),
                "swap_checkpoint_exact": float(
                    bool(
                        torch.equal(
                            swapped.normalized_checkpoints[row],
                            output.normalized_checkpoints[row].flip(0),
                        )
                    )
                ),
                "swap_checkpoint_max_abs": float(swap_checkpoint_error[row].max().cpu()),
                "finite": bool(finite[row].cpu()),
            }
        )
    return records


def evaluate_checkpoint_set(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    checkpoints: Mapping[int, Path],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if corpus.split != "dev" or cache.split != "dev":
        raise ValueError("segment-slot evaluation is dev-only")
    cache.validate_source(corpus)
    if set(checkpoints) != set(TRAIN_SEEDS):
        raise ValueError("segment-slot checkpoint seed set changed")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in TRAIN_SEEDS:
        model, stats = load_checkpoint(
            checkpoints[seed],
            seed=seed,
            expected_steps=TRAIN_STEPS,
            device=device,
        )
        records = []
        for group_id in corpus.group_ids:
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
        "second_mae_m",
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
    return {"records": by_seed, "aggregates": {"by_seed": per_seed, "seed_equal": seed_equal}}


def _seed_records(by_seed: Mapping[Any, Any], seed: int) -> Sequence[Mapping[str, Any]]:
    return by_seed[seed] if seed in by_seed else by_seed[str(seed)]


def _group_means(records: Sequence[Mapping[str, Any]], metric: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for record in records:
        values[int(record["group_id"])].append(float(record[metric]))
    return {group_id: float(np.mean(group_values)) for group_id, group_values in values.items()}


def evaluate_dev_gates(
    evidence: Mapping[str, Any], dense_result: Mapping[str, Any]
) -> dict[str, bool]:
    aggregate = evidence["aggregates"]["seed_equal"]
    dense_aggregate = dense_result["aggregates"]["mamba_full"]["seed_equal"]
    gates: dict[str, Any] = {
        "first_shape_accuracy_at_least_90pct": aggregate["first_shape_accuracy"] >= 0.90,
        "second_shape_accuracy_at_least_90pct": aggregate["second_shape_accuracy"] >= 0.90,
        "ordered_pair_accuracy_at_least_80pct": aggregate["ordered_pair_accuracy"] >= 0.80,
        "first_mae_improves_10pct_over_dense": (
            aggregate["first_mae_m"] <= 0.90 * dense_aggregate["first_mae_m"]
        ),
        "complete_mae_improves_10pct_over_dense": (
            aggregate["complete_mae_m"] <= 0.90 * dense_aggregate["complete_mae_m"]
        ),
        "swap_shape_exact_at_least_80pct": aggregate["swap_shape_exact"] >= 0.80,
        "all_outputs_and_checkpoints_finite": all(
            bool(record["finite"])
            for seed in TRAIN_SEEDS
            for record in _seed_records(evidence["records"], seed)
        ),
    }
    for metric in ("first_mae_m", "complete_mae_m"):
        group_directions = 0
        all_seed_medians = True
        for group_id in (9, 12, 17, 25, 27, 29):
            current_seed_values = []
            dense_seed_values = []
            for seed in TRAIN_SEEDS:
                current = _group_means(_seed_records(evidence["records"], seed), metric)
                dense = _group_means(
                    _seed_records(dense_result["records"]["mamba_full"], seed), metric
                )
                current_seed_values.append(current[group_id])
                dense_seed_values.append(dense[group_id])
            group_directions += float(np.mean(current_seed_values)) < float(
                np.mean(dense_seed_values)
            )
        for seed in TRAIN_SEEDS:
            current = _group_means(_seed_records(evidence["records"], seed), metric)
            dense = _group_means(_seed_records(dense_result["records"]["mamba_full"], seed), metric)
            all_seed_medians &= statistics.median(current.values()) < statistics.median(
                dense.values()
            )
        gates[f"{metric}_improves_5of6_groups_and_all_seed_medians"] = (
            group_directions >= 5 and all_seed_medians
        )
    return {name: bool(value) for name, value in gates.items()}


def outcome_from_gates(gates: Mapping[str, bool]) -> str:
    if all(gates.values()):
        return "event_readout_bottleneck_supported"
    per_slot_shape_pass = all(
        gates[name]
        for name in (
            "first_shape_accuracy_at_least_90pct",
            "second_shape_accuracy_at_least_90pct",
        )
    )
    mae_pass = all(
        gates[name]
        for name in (
            "first_mae_improves_10pct_over_dense",
            "complete_mae_improves_10pct_over_dense",
            "first_mae_m_improves_5of6_groups_and_all_seed_medians",
            "complete_mae_m_improves_5of6_groups_and_all_seed_medians",
        )
    )
    association_pass = (
        gates["ordered_pair_accuracy_at_least_80pct"] and gates["swap_shape_exact_at_least_80pct"]
    )
    if per_slot_shape_pass and not mae_pass:
        return "continuous_geometry_or_readout_bottleneck"
    if not per_slot_shape_pass:
        return "spatial_representation_bottleneck"
    if mae_pass and not association_pass:
        return "slot_association_or_objective_invalid"
    return "technical_or_mixed_gate_failure"


def run_overfit(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    output_root: Path,
    *,
    device: torch.device,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    data = LoadedSegmentSlotData(corpus, cache)
    stats = TargetStats.fit(corpus)
    deadline = time.monotonic() + 15 * 60
    training = train_one(
        data,
        stats,
        seed=230923,
        steps=OVERFIT_STEPS,
        device=device,
        deadline=deadline,
        checkpoint_path=output_root / "checkpoint_final.pt",
        overfit_group=0,
    )
    model, loaded_stats = load_checkpoint(
        output_root / "checkpoint_final.pt",
        seed=230923,
        expected_steps=OVERFIT_STEPS,
        device=device,
    )
    records = evaluate_group(
        model,
        loaded_stats,
        data.features[0],
        data.targets[0],
        device=device,
    )
    first = statistics.median(row["total_loss"] for row in training["curve"][:50])
    final = statistics.median(row["total_loss"] for row in training["curve"][-50:])
    gates = {
        "total_loss_reduction_at_least_90pct": final <= 0.10 * first,
        "ordered_pair_accuracy_6of6": sum(row["ordered_pair_accuracy"] for row in records) == 6,
        "slot_shape_accuracy_12of12": (
            sum(row["first_shape_accuracy"] + row["second_shape_accuracy"] for row in records) == 12
        ),
        "slot_encoder_and_heads_have_finite_nonzero_gradients": all(
            training["gradient_seen"].values()
        ),
        "all_outputs_finite": all(record["finite"] for record in records),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "segment_slot_overfit",
        "group_id": 0,
        "first_50_median_total_loss": first,
        "final_50_median_total_loss": final,
        "training": training,
        "records": records,
        "gates": gates,
        "provenance": dict(provenance),
        "confirmation_allowed": False,
    }
    if time.monotonic() > deadline:
        raise TimeoutError("segment-slot overfit exceeded its frozen wall deadline")
    _enforce_result_budget(output_root, result, OVERFIT_ARTIFACT_LIMIT)
    write_new_json(output_root / "RESULT.json", result)
    return result


def run_matched_dev(
    train_corpus: ShardedMemoryCorpus,
    train_cache: FeatureCacheCorpus,
    dev_corpus: ShardedMemoryCorpus,
    dev_cache: FeatureCacheCorpus,
    dense_result: Mapping[str, Any],
    overfit_receipt: Mapping[str, Any],
    output_root: Path,
    *,
    device: torch.device,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if overfit_receipt.get("status") != "passed" or not all(
        overfit_receipt.get("gates", {}).values()
    ):
        raise PermissionError("segment-slot overfit did not open matched training")
    data = LoadedSegmentSlotData(train_corpus, train_cache)
    stats = TargetStats.fit(train_corpus)
    deadline = time.monotonic() + 60 * 60
    training = []
    checkpoints = {}
    for seed in TRAIN_SEEDS:
        path = output_root / "checkpoints" / f"seed_{seed}" / "checkpoint_final.pt"
        training.append(
            train_one(
                data,
                stats,
                seed=seed,
                steps=TRAIN_STEPS,
                device=device,
                deadline=deadline,
                checkpoint_path=path,
            )
        )
        checkpoints[seed] = path
    evidence = evaluate_checkpoint_set(dev_corpus, dev_cache, checkpoints, device=device)
    gates = evaluate_dev_gates(evidence, dense_result)
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "segment_slot_matched_dev",
        "seeds": list(TRAIN_SEEDS),
        "steps_per_seed": TRAIN_STEPS,
        "training": training,
        **evidence,
        "gates": gates,
        "outcome": outcome_from_gates(gates),
        "provenance": dict(provenance),
        "confirmation_allowed": False,
    }
    if time.monotonic() > deadline:
        raise TimeoutError("segment-slot matched-dev exceeded its frozen wall deadline")
    _enforce_result_budget(output_root, result, MATCHED_ARTIFACT_LIMIT)
    write_new_json(output_root / "RESULT.json", result)
    return result
