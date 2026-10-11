"""Training and evaluation operations for the MIKASA native-RGB memory experiment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Final

import numpy as np
import torch

from gr00t.experimental.mikasa_native_rgb_memory import (
    TARGET_DIM,
    MemoryBatch,
    MikasaNativeRGBMemoryModel,
    temporal_parameter_identity,
)
from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    BATCH_SIZE,
    ROWS_PER_GROUP,
    FeatureCacheCorpus,
    FeatureGroup,
    GroupArrays,
    GroupBalancedSchedule,
    ShardedMemoryCorpus,
    TargetStats,
    build_native_rgb_features,
    concatenate_feature_groups,
    confirmation_spec_identity,
    group_equal_mean,
    physical_metrics,
    serialize_checkpoint,
    sha256_file,
    verify_confirmation_receipts,
    write_new_bytes,
    write_new_json,
)


TRAIN_SEEDS: Final = (230923, 230924, 230925)
CONDITIONS: Final = (
    "mamba_current",
    "mamba_one_frame",
    "mamba_last_segment",
    "mamba_full",
    "gru_full",
    "transformer_full",
)
TRAIN_STEPS: Final = 1500
OVERFIT_STEPS: Final = 500
LEARNING_RATE: Final = 3e-4
FINAL_LEARNING_RATE: Final = 3e-5
WARMUP_STEPS: Final = 100
WEIGHT_DECAY: Final = 1e-4
BETAS: Final = (0.9, 0.95)
GRADIENT_CLIP: Final = 1.0


def _memory_batch_to_device(batch: MemoryBatch, device: torch.device) -> MemoryBatch:
    # The model also accepts CPU grids and transfers frame chunks itself. Post
    # tensors are kept on CPU here to bound host-to-device residency.
    del device
    return batch


def _target_for_groups(groups: Sequence[GroupArrays], device: torch.device) -> torch.Tensor:
    value = np.concatenate(
        [
            group.target["ordered_checkpoints"].reshape(ROWS_PER_GROUP, TARGET_DIM)
            for group in groups
        ],
        axis=0,
    )
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device=device)


def _lr_multiplier(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = (step + 1 - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return (
        FINAL_LEARNING_RATE / LEARNING_RATE + (1.0 - FINAL_LEARNING_RATE / LEARNING_RATE) * cosine
    )


def _finite_nonzero_gradient(module: torch.nn.Module) -> bool:
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
    model: MikasaNativeRGBMemoryModel,
    stats: TargetStats,
    *,
    condition: str,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    return {
        "schema": "g1-mikasa-native-rgb-memory-backbone-r1/checkpoint-v1",
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "dtype": "float32",
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "target_stats": stats.as_json(),
        "temporal_identity": temporal_parameter_identity(model),
    }


def _load_checkpoint(
    path: Path,
    *,
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[MikasaNativeRGBMemoryModel, TargetStats]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("condition") != condition or value.get("seed") != seed:
        raise ValueError(f"checkpoint identity changed: {path}")
    if value.get("dtype") != "float32":
        raise ValueError("checkpoint dtype contract changed")
    model = MikasaNativeRGBMemoryModel(condition, seed=seed).to(device=device, dtype=torch.float32)
    model.load_state_dict(value["model_state_dict"], strict=True)
    if temporal_parameter_identity(model) != value.get("temporal_identity"):
        raise ValueError("checkpoint temporal identity changed")
    return model.eval(), TargetStats.from_json(value["target_stats"])


class LoadedTrainData:
    """Bounded in-memory view loaded one feature/target shard at a time."""

    def __init__(self, corpus: ShardedMemoryCorpus, cache: FeatureCacheCorpus):
        if corpus.split != "train" or cache.split != "train":
            raise ValueError("training data must come from the train split")
        cache.validate_source(corpus)
        self.features: dict[int, FeatureGroup] = {}
        self.targets: dict[int, GroupArrays] = {}
        for group_id in corpus.group_ids:
            self.features[group_id] = cache.load_group(group_id)
            self.targets[group_id] = corpus.load_target_group(group_id)
        self.group_ids = corpus.group_ids

    def batch(self, group_ids: Sequence[int]) -> tuple[MemoryBatch, list[GroupArrays]]:
        if len(group_ids) not in (1, 4):
            raise ValueError("training batches must contain one overfit or four matched groups")
        feature = concatenate_feature_groups([self.features[group_id] for group_id in group_ids])
        targets = [self.targets[group_id] for group_id in group_ids]
        if len(group_ids) == 4 and feature.arrays["temporal_grid"].shape[0] != BATCH_SIZE:
            raise RuntimeError("group-balanced batch is not 24 rows")
        return feature.memory_batch(), targets


def train_one(
    data: LoadedTrainData,
    stats: TargetStats,
    *,
    condition: str,
    seed: int,
    steps: int,
    device: torch.device,
    output_path: Path | None,
    overfit_group: int | None = None,
    wall_limit_seconds: float,
) -> dict[str, Any]:
    if condition not in CONDITIONS or seed not in TRAIN_SEEDS:
        raise ValueError("unfrozen training condition or seed")
    if steps not in (OVERFIT_STEPS, TRAIN_STEPS):
        raise ValueError("unfrozen optimizer step count")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MikasaNativeRGBMemoryModel(condition, seed=seed).to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("model is not entirely FP32")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    schedule = GroupBalancedSchedule(data.group_ids, seed=seed)
    started = time.monotonic()
    curve: list[dict[str, float | int]] = []
    gradient_seen = {"frontend": False, "temporal": False, "readout": False}
    model.train()
    for step in range(steps):
        if time.monotonic() - started > wall_limit_seconds:
            raise TimeoutError(f"{condition}/{seed} exceeded its wall limit")
        group_ids = (overfit_group,) if overfit_group is not None else schedule.batch(step)
        batch, target_groups = data.batch(group_ids)
        target = _target_for_groups(target_groups, device)
        prediction = model(_memory_batch_to_device(batch, device))
        loss = torch.square(prediction - stats.normalize(target)).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("training loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_seen["frontend"] |= _finite_nonzero_gradient(model.temporal_frontend)
        gradient_seen["temporal"] |= _finite_nonzero_gradient(model.temporal)
        gradient_seen["readout"] |= _finite_nonzero_gradient(model.readout)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("gradient norm is nonfinite")
        multiplier = _lr_multiplier(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * multiplier
        optimizer.step()
        curve.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        )
    if not all(gradient_seen.values()):
        raise RuntimeError(f"missing finite nonzero gradients: {gradient_seen}")
    checkpoint_identity = None
    if output_path is not None:
        checkpoint_identity = write_new_bytes(
            output_path,
            serialize_checkpoint(
                _checkpoint_payload(model, stats, condition=condition, seed=seed, steps=steps)
            ),
        )
    return {
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "batch_size": ROWS_PER_GROUP if overfit_group is not None else BATCH_SIZE,
        "gradient_seen": gradient_seen,
        "curve": curve,
        "checkpoint": checkpoint_identity,
        "wall_seconds": time.monotonic() - started,
    }


@torch.inference_mode()
def evaluate_group(
    model: MikasaNativeRGBMemoryModel,
    stats: TargetStats,
    feature: FeatureGroup,
    target: GroupArrays,
    *,
    device: torch.device,
    intervention: str = "none",
) -> list[dict[str, Any]]:
    arrays = {key: value.copy() for key, value in feature.arrays.items()}
    target_indices = torch.arange(ROWS_PER_GROUP, device=device)
    if intervention == "substitute":
        arrays["temporal_grid"] = np.roll(arrays["temporal_grid"], shift=-1, axis=0).copy()
        target_indices = torch.roll(target_indices, shifts=-1)
    elif intervention == "zero":
        arrays["temporal_grid"].fill(0.0)
    elif intervention == "shuffle":
        permutation = np.random.default_rng(230923).permutation(130)
        arrays["temporal_grid"] = arrays["temporal_grid"][:, permutation].copy()
    elif intervention != "none":
        raise ValueError(f"unknown intervention: {intervention}")
    prediction = model(FeatureGroup(feature.group_id, arrays).memory_batch())
    physical_target = torch.from_numpy(
        np.ascontiguousarray(
            target.target["ordered_checkpoints"].reshape(ROWS_PER_GROUP, TARGET_DIM)
        )
    ).to(device=device, dtype=torch.float32)
    candidates = physical_target.unsqueeze(0).expand(ROWS_PER_GROUP, -1, -1)
    metrics = physical_metrics(prediction, physical_target, stats, candidates, target_indices)
    records: list[dict[str, Any]] = []
    for row in range(ROWS_PER_GROUP):
        records.append(
            {
                "group_id": feature.group_id,
                "row": row,
                "intervention": intervention,
                "first_mae_m": float(metrics["first_mae_m"][row].cpu()),
                "second_mae_m": float(metrics["second_mae_m"][row].cpu()),
                "complete_mae_m": float(metrics["complete_mae_m"][row].cpu()),
                "normalized_mse": float(metrics["normalized_mse"][row].cpu()),
                "order_correct": float(metrics["order_correct"][row].cpu()),
                "predicted_candidate_index": int(metrics["predicted_candidate_index"][row].cpu()),
                "target_candidate_index": int(target_indices[row].cpu()),
            }
        )
    return records


def evaluate_checkpoint_set(
    corpus: ShardedMemoryCorpus,
    feature_cache: FeatureCacheCorpus | None,
    checkpoints: Mapping[str, Mapping[int, Path]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if feature_cache is not None:
        feature_cache.validate_source(corpus)
    all_records: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    intervention_records: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for condition in CONDITIONS:
        for seed in TRAIN_SEEDS:
            model, stats = _load_checkpoint(
                checkpoints[condition][seed], condition=condition, seed=seed, device=device
            )
            records: list[dict[str, Any]] = []
            per_intervention: list[dict[str, Any]] = []
            for group_id in corpus.group_ids:
                if feature_cache is None:
                    target = corpus.load_group(group_id)
                    arrays = build_native_rgb_features(target.model)
                    feature = FeatureGroup(group_id, arrays)
                else:
                    target = corpus.load_target_group(group_id)
                    feature = feature_cache.load_group(group_id)
                records.extend(evaluate_group(model, stats, feature, target, device=device))
                if condition == "mamba_full":
                    for intervention in ("substitute", "zero", "shuffle"):
                        per_intervention.extend(
                            evaluate_group(
                                model,
                                stats,
                                feature,
                                target,
                                device=device,
                                intervention=intervention,
                            )
                        )
            all_records[condition][seed] = records
            if condition == "mamba_full":
                intervention_records[condition][seed] = per_intervention
    aggregates: dict[str, Any] = {}
    for condition, by_seed in all_records.items():
        seed_metrics = {
            str(seed): {
                metric: group_equal_mean(records, metric)
                for metric in (
                    "first_mae_m",
                    "second_mae_m",
                    "complete_mae_m",
                    "normalized_mse",
                    "order_correct",
                )
            }
            for seed, records in sorted(by_seed.items())
        }
        aggregates[condition] = {
            "by_seed": seed_metrics,
            "seed_equal": {
                metric: float(np.mean([values[metric] for values in seed_metrics.values()]))
                for metric in (
                    "first_mae_m",
                    "second_mae_m",
                    "complete_mae_m",
                    "normalized_mse",
                    "order_correct",
                )
            },
        }
    interventions: dict[str, Any] = {}
    for seed, records in intervention_records.get("mamba_full", {}).items():
        interventions[str(seed)] = {
            name: {
                metric: group_equal_mean(
                    [record for record in records if record["intervention"] == name], metric
                )
                for metric in ("first_mae_m", "complete_mae_m", "order_correct")
            }
            for name in ("substitute", "zero", "shuffle")
        }
    return {"records": all_records, "aggregates": aggregates, "interventions": interventions}


def _improvement(candidate: float, control: float) -> float:
    return (control - candidate) / control if control > 0 else float("-inf")


def evaluate_dev_gates(evidence: Mapping[str, Any], cost: Mapping[str, Any]) -> dict[str, bool]:
    aggregate = evidence["aggregates"]
    full = aggregate["mamba_full"]["seed_equal"]
    short_controls = ("mamba_current", "mamba_one_frame", "mamba_last_segment")
    gates: dict[str, bool] = {}
    for control in short_controls:
        control_metric = aggregate[control]["seed_equal"]
        gates[f"mamba_full_first_mae_improves_10pct_vs_{control}"] = (
            _improvement(full["first_mae_m"], control_metric["first_mae_m"]) >= 0.10
        )
        gates[f"mamba_full_complete_mae_improves_10pct_vs_{control}"] = (
            _improvement(full["complete_mae_m"], control_metric["complete_mae_m"]) >= 0.10
        )
        gates[f"mamba_full_order_advantage_20pt_vs_{control}"] = (
            full["order_correct"] - control_metric["order_correct"] >= 0.20
        )
    gates["mamba_full_order_accuracy_at_least_80pct"] = full["order_correct"] >= 0.80
    best_full = {
        "first_mae_m": min(
            aggregate["gru_full"]["seed_equal"]["first_mae_m"],
            aggregate["transformer_full"]["seed_equal"]["first_mae_m"],
        ),
        "complete_mae_m": min(
            aggregate["gru_full"]["seed_equal"]["complete_mae_m"],
            aggregate["transformer_full"]["seed_equal"]["complete_mae_m"],
        ),
        "order_correct": max(
            aggregate["gru_full"]["seed_equal"]["order_correct"],
            aggregate["transformer_full"]["seed_equal"]["order_correct"],
        ),
    }
    gates["mamba_first_mae_within_2pct_best_full_control"] = (
        full["first_mae_m"] <= 1.02 * best_full["first_mae_m"]
    )
    gates["mamba_complete_mae_within_2pct_best_full_control"] = (
        full["complete_mae_m"] <= 1.02 * best_full["complete_mae_m"]
    )
    gates["mamba_order_within_2pt_best_full_control"] = (
        full["order_correct"] >= best_full["order_correct"] - 0.02
    )
    intervention = evidence["interventions"]
    substitute = np.mean([value["substitute"]["order_correct"] for value in intervention.values()])
    zero_first = np.mean([value["zero"]["first_mae_m"] for value in intervention.values()])
    shuffle_first = np.mean([value["shuffle"]["first_mae_m"] for value in intervention.values()])
    gates["cue_substitution_follows_order_80pct"] = substitute >= 0.80
    gates["zero_worsens_first_mae_10pct"] = zero_first >= 1.10 * full["first_mae_m"]
    gates["shuffle_worsens_first_mae_10pct"] = shuffle_first >= 1.10 * full["first_mae_m"]
    gates["mamba_long_context_resource_advantage"] = bool(
        cost.get("three_blocks_passed")
        and (
            cost.get("mamba_transformer_p95_ratio_t1040", float("inf")) <= 0.80
            or cost.get("mamba_transformer_peak_allocated_ratio_t1040", float("inf")) <= 0.67
        )
    )
    # Direction gates are computed from each seed/group's group-equal rows.
    for control in short_controls:
        for metric in ("first_mae_m", "complete_mae_m"):
            direction_count = 0
            all_seed_medians = True
            for group_id in sorted(
                {
                    int(record["group_id"])
                    for record in evidence["records"]["mamba_full"][TRAIN_SEEDS[0]]
                }
            ):
                full_by_seed = []
                control_by_seed = []
                for seed in TRAIN_SEEDS:
                    full_by_seed.append(
                        float(
                            np.mean(
                                [
                                    float(record[metric])
                                    for record in evidence["records"]["mamba_full"][seed]
                                    if int(record["group_id"]) == group_id
                                ]
                            )
                        )
                    )
                    control_by_seed.append(
                        float(
                            np.mean(
                                [
                                    float(record[metric])
                                    for record in evidence["records"][control][seed]
                                    if int(record["group_id"]) == group_id
                                ]
                            )
                        )
                    )
                direction_count += float(np.mean(full_by_seed)) < float(np.mean(control_by_seed))
            for seed in TRAIN_SEEDS:
                full_group_values = [
                    np.mean(
                        [
                            float(record[metric])
                            for record in evidence["records"]["mamba_full"][seed]
                            if int(record["group_id"]) == group_id
                        ]
                    )
                    for group_id in sorted(
                        {
                            int(record["group_id"])
                            for record in evidence["records"]["mamba_full"][seed]
                        }
                    )
                ]
                control_group_values = [
                    np.mean(
                        [
                            float(record[metric])
                            for record in evidence["records"][control][seed]
                            if int(record["group_id"]) == group_id
                        ]
                    )
                    for group_id in sorted(
                        {int(record["group_id"]) for record in evidence["records"][control][seed]}
                    )
                ]
                full_seed = statistics.median(full_group_values)
                control_seed = statistics.median(control_group_values)
                all_seed_medians &= full_seed < control_seed
            gates[f"{metric}_direction_5of6_and_all_seeds_vs_{control}"] = (
                direction_count >= 5 and all_seed_medians
            )
    return {name: bool(value) for name, value in gates.items()}


def run_overfit(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    output_root: Path,
    *,
    device: torch.device,
    mamba_overlay: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    data = LoadedTrainData(corpus, cache)
    stats = TargetStats.fit(corpus)
    group_id = data.group_ids[0]
    training = train_one(
        data,
        stats,
        condition="mamba_full",
        seed=230923,
        steps=OVERFIT_STEPS,
        device=device,
        output_path=output_root / "checkpoint_final.pt",
        overfit_group=group_id,
        wall_limit_seconds=15 * 60,
    )
    first = statistics.median(float(row["loss"]) for row in training["curve"][:50])
    final = statistics.median(float(row["loss"]) for row in training["curve"][-50:])
    model, loaded_stats = _load_checkpoint(
        output_root / "checkpoint_final.pt",
        condition="mamba_full",
        seed=230923,
        device=device,
    )
    feature = data.features[group_id]
    target = data.targets[group_id]
    original = evaluate_group(model, loaded_stats, feature, target, device=device)
    substituted = evaluate_group(
        model, loaded_stats, feature, target, device=device, intervention="substitute"
    )
    gates = {
        "loss_reduction_at_least_90pct": final <= 0.10 * first,
        "train_order_accuracy_6of6": sum(row["order_correct"] for row in original) == 6,
        "all_modules_have_finite_nonzero_gradients": all(training["gradient_seen"].values()),
        "order_substitution_changes_predictions": any(
            before["predicted_candidate_index"] != after["predicted_candidate_index"]
            for before, after in zip(original, substituted, strict=True)
        ),
    }
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "overfit",
        "group_id": group_id,
        "first_50_median_loss": first,
        "final_50_median_loss": final,
        "training": training,
        "gates": gates,
        "mamba_overlay": dict(mamba_overlay),
    }
    write_new_json(output_root / "RESULT.json", result)
    return result


def run_matched_training(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    output_root: Path,
    *,
    device: torch.device,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if set(prerequisites) != {"cost", "batch24_preflight", "overfit", "mamba_overlay"}:
        raise PermissionError("matched training prerequisites are incomplete")
    data = LoadedTrainData(corpus, cache)
    stats = TargetStats.fit(corpus)
    runs = []
    for condition in CONDITIONS:
        for seed in TRAIN_SEEDS:
            checkpoint = (
                output_root / "checkpoints" / condition / f"seed_{seed}" / "checkpoint_final.pt"
            )
            runs.append(
                train_one(
                    data,
                    stats,
                    condition=condition,
                    seed=seed,
                    steps=TRAIN_STEPS,
                    device=device,
                    output_path=checkpoint,
                    wall_limit_seconds=20 * 60,
                )
            )
    result = {
        "status": "passed",
        "mode": "matched_train",
        "conditions": list(CONDITIONS),
        "seeds": list(TRAIN_SEEDS),
        "steps_per_run": TRAIN_STEPS,
        "runs": runs,
        "target_stats": stats.as_json(),
        "prerequisites": dict(prerequisites),
    }
    write_new_json(output_root / "RESULT.json", result)
    return result


def checkpoint_paths(root: Path) -> dict[str, dict[int, Path]]:
    return {
        condition: {
            seed: root / "checkpoints" / condition / f"seed_{seed}" / "checkpoint_final.pt"
            for seed in TRAIN_SEEDS
        }
        for condition in CONDITIONS
    }


def run_dev_evaluation(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    checkpoint_root: Path,
    cost_result_path: Path,
    output_root: Path,
    *,
    device: torch.device,
    source_paths: Mapping[str, Path],
    confirmation_spec_path: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    checkpoints = checkpoint_paths(checkpoint_root)
    cost = json.loads(cost_result_path.read_text(encoding="utf-8"))
    evidence = evaluate_checkpoint_set(corpus, cache, checkpoints, device=device)
    gates = evaluate_dev_gates(evidence, cost)
    status = "passed" if all(gates.values()) else "gate_failed"
    result = {"status": status, "mode": "dev_eval", "gates": gates, **evidence}
    write_new_json(output_root / "RESULT.json", result)
    if status == "passed":
        receipt = {
            "status": "passed",
            "dev_gate_passed": True,
            "result_sha256": sha256_file(output_root / "RESULT.json"),
        }
        write_new_json(output_root / "DEV_PASS_RECEIPT.json", receipt)
        _, spec_identity = confirmation_spec_identity(confirmation_spec_path)
        if any(path.is_symlink() or not path.is_file() for path in source_paths.values()):
            raise ValueError("model/evaluator source material is not a regular file")
        freeze = {
            "status": "frozen",
            "checkpoint_sha256": {
                condition: {str(seed): sha256_file(path) for seed, path in sorted(by_seed.items())}
                for condition, by_seed in sorted(checkpoints.items())
            },
            "source_sha256": {
                name: sha256_file(path) for name, path in sorted(source_paths.items())
            },
            "confirmation_spec": spec_identity,
        }
        write_new_json(output_root / "FREEZE_RECEIPT.json", freeze)
    return result


def run_joint_confirmation(
    *,
    dev_receipt_path: Path,
    freeze_receipt_path: Path,
    checkpoint_root: Path,
    source_paths: Mapping[str, Path],
    confirmation_spec_path: Path,
    confirmation_corpus_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Authorize, generate, and evaluate confirmation in one fail-closed entry point."""

    if output_root.exists():
        raise FileExistsError(output_root)
    checkpoints = checkpoint_paths(checkpoint_root)
    receipt = verify_confirmation_receipts(
        dev_receipt_path,
        freeze_receipt_path,
        checkpoints,
        source_paths,
        confirmation_spec_path,
    )
    spec, _ = confirmation_spec_identity(confirmation_spec_path)
    generator_argv = spec["generator_argv"]
    # The confirmation path is first touched only after receipts and all frozen
    # source/checkpoint identities pass above.
    confirmation_split = confirmation_corpus_root.resolve() / "confirmation"
    if confirmation_split.exists():
        raise FileExistsError("confirmation split was materialized before the joint command")
    completed = subprocess.run(list(generator_argv), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"confirmation generator failed: {completed.stderr[-2000:]}")
    corpus = ShardedMemoryCorpus.discover(
        confirmation_corpus_root,
        "confirmation",
        confirmation_authorized=True,
    )
    evidence = evaluate_checkpoint_set(corpus, None, checkpoints, device=device)
    # Confirmation reuses gates 1--5. The cost gate is fixed from dev and is
    # deliberately omitted rather than rerun or reselected.
    dummy_cost = {
        "three_blocks_passed": True,
        "mamba_transformer_p95_ratio_t1040": 0.0,
    }
    gates = evaluate_dev_gates(evidence, dummy_cost)
    gates.pop("mamba_long_context_resource_advantage")
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "joint_confirmation",
        "authorization": receipt,
        "generator_returncode": completed.returncode,
        "gates": gates,
        **evidence,
    }
    write_new_json(output_root / "RESULT.json", result)
    return result


def state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
