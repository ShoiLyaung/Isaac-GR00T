"""Matched temporal-moment training, deferred calibration scoring, and gates."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
from pathlib import Path
import statistics
import time
from typing import Any, Final

import numpy as np
import torch

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    FeatureCacheCorpus,
    ShardedMemoryCorpus,
    TargetStats,
    canonical_json_bytes,
    group_equal_mean,
    write_new_json,
)
from gr00t.experimental.mikasa_segment_slot_bottleneck import MikasaSegmentSlotModel
from gr00t.experimental.mikasa_segment_slot_experiment import (
    TRAIN_SEEDS,
    TRAIN_STEPS,
    load_checkpoint,
)
import gr00t.experimental.mikasa_segment_slot_train_calibration as calibration_training
from gr00t.experimental.mikasa_segment_slot_train_calibration import (
    CALIBRATION_GROUPS,
    FitOnlyData,
    FitTargetCorpus,
    _group_means,
    _seed_records,
    evaluate_calibration_gates,
    fit_target_mean,
    train_fit_seed,
    validate_frozen_split,
)
from gr00t.experimental.mikasa_temporal_moment_cache import CONDITIONS, TemporalMomentCache


HISTORICAL_TOLERANCE: Final = 1e-6
ARTIFACT_LIMIT_BYTES: Final = 2 * 1024 * 1024 * 1024


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def capture_actual_training_initialization():
    original = calibration_training.MikasaSegmentSlotModel
    observed: list[str] = []

    def constructor(*args, **kwargs):
        model = original(*args, **kwargs)
        observed.append(model_state_sha256(model))
        return model

    calibration_training.MikasaSegmentSlotModel = constructor
    try:
        yield observed
    finally:
        calibration_training.MikasaSegmentSlotModel = original


def train_with_actual_initial_hash(*args, seed: int, **kwargs):
    expected = model_state_sha256(MikasaSegmentSlotModel(seed=seed))
    with capture_actual_training_initialization() as observed:
        run = train_fit_seed(*args, seed=seed, **kwargs)
    if observed != [expected]:
        raise RuntimeError("actual training model initialization changed")
    return run, observed[0]


def learning_rate_curve_sha256(curve: list[Mapping[str, Any]]) -> str:
    if len(curve) != TRAIN_STEPS:
        raise ValueError("learning-rate curve length changed")
    digest = hashlib.sha256()
    for expected_step, record in enumerate(curve, start=1):
        if record.get("step") != expected_step:
            raise ValueError("learning-rate curve step identity changed")
        digest.update(np.float64(record["lr"]).tobytes())
    return digest.hexdigest()


def verify_historical_checkpoints(
    checkpoints: Mapping[int, Path], prior_result: Mapping[str, Any]
) -> dict[str, Any]:
    if set(checkpoints) != set(TRAIN_SEEDS):
        raise ValueError("historical checkpoint seed set changed")
    training = prior_result.get("training")
    if not isinstance(training, list) or len(training) != len(TRAIN_SEEDS):
        raise ValueError("historical checkpoint receipts missing")
    receipts = {int(record["seed"]): record["checkpoint"] for record in training}
    if set(receipts) != set(TRAIN_SEEDS):
        raise ValueError("historical checkpoint receipt seeds changed")
    verified = {}
    for seed, path in sorted(checkpoints.items()):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        receipt = receipts[seed]
        raw = path.read_bytes()
        observed = {
            "path": str(path.resolve()),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if observed != receipt:
            raise ValueError(f"historical checkpoint receipt changed for seed {seed}")
        verified[str(seed)] = observed
    return verified


@torch.inference_mode()
def predict_calibration(
    checkpoints: Mapping[int, Path],
    cache: Any,
    *,
    device: torch.device,
) -> dict[int, dict[int, dict[str, Any]]]:
    predictions = {}
    for seed in TRAIN_SEEDS:
        model, stats = load_checkpoint(
            checkpoints[seed], seed=seed, expected_steps=TRAIN_STEPS, device=device
        )
        by_group = {}
        for group_id in CALIBRATION_GROUPS:
            temporal = torch.from_numpy(cache.load_group(group_id).arrays["temporal_grid"]).to(
                device=device, dtype=torch.float32
            )
            slot_maps = model.pool_segments(temporal)
            output = model.forward_slot_maps(slot_maps)
            swapped = model.forward_slot_maps(slot_maps.flip(1))
            by_group[group_id] = {
                "shape_logits": output.shape_logits.cpu(),
                "normalized_checkpoints": output.normalized_checkpoints.cpu(),
                "swapped_shape_logits": swapped.shape_logits.cpu(),
                "swapped_checkpoints": swapped.normalized_checkpoints.cpu(),
                "stats": stats,
            }
        predictions[seed] = by_group
    return predictions


def score_predictions(
    predictions: Mapping[int, Mapping[int, Mapping[str, Any]]],
    corpus: ShardedMemoryCorpus,
    target_mean: np.ndarray,
) -> dict[str, Any]:
    by_seed = {}
    for seed in TRAIN_SEEDS:
        records = []
        for group_id in CALIBRATION_GROUPS:
            prediction = predictions[seed][group_id]
            target_group = corpus.load_target_group(group_id).target
            shape_target = torch.from_numpy(target_group["ordered_shape_ids"])
            checkpoint_target = torch.from_numpy(target_group["ordered_checkpoints"])
            shape = prediction["shape_logits"].argmax(dim=-1)
            swapped_shape = prediction["swapped_shape_logits"].argmax(dim=-1)
            stats = prediction["stats"]
            physical = stats.denormalize(
                prediction["normalized_checkpoints"].reshape(6, 48)
            ).reshape(6, 2, 12, 2)
            error = (physical - checkpoint_target).abs()
            swap_checkpoint_error = (
                prediction["swapped_checkpoints"] - prediction["normalized_checkpoints"].flip(1)
            ).abs()
            finite = (
                torch.isfinite(prediction["shape_logits"]).all(dim=(1, 2))
                & torch.isfinite(prediction["normalized_checkpoints"]).all(dim=(1, 2))
                & torch.isfinite(prediction["swapped_shape_logits"]).all(dim=(1, 2))
                & torch.isfinite(prediction["swapped_checkpoints"]).all(dim=(1, 2))
                & torch.isfinite(physical).all(dim=(1, 2, 3))
            )
            for row in range(6):
                records.append(
                    {
                        "group_id": group_id,
                        "row": row,
                        "first_shape_accuracy": float(shape[row, 0] == shape_target[row, 0]),
                        "second_shape_accuracy": float(shape[row, 1] == shape_target[row, 1]),
                        "ordered_pair_accuracy": float(torch.equal(shape[row], shape_target[row])),
                        "first_mae_m": float(error[row, 0].mean()),
                        "complete_mae_m": float(error[row].mean()),
                        "swap_shape_exact": float(
                            torch.equal(swapped_shape[row], shape[row].flip(0))
                        ),
                        "swap_checkpoint_exact": float(
                            torch.equal(
                                prediction["swapped_checkpoints"][row],
                                prediction["normalized_checkpoints"][row].flip(0),
                            )
                        ),
                        "swap_checkpoint_max_abs": float(swap_checkpoint_error[row].max()),
                        "finite": bool(finite[row]),
                    }
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
    baseline_records = []
    mean = target_mean.reshape(2, 12, 2)
    for group_id in CALIBRATION_GROUPS:
        target = corpus.load_target_group(group_id).target["ordered_checkpoints"]
        error = np.abs(target - mean[None])
        for row in range(6):
            baseline_records.append(
                {
                    "group_id": group_id,
                    "first_mae_m": float(error[row, 0].mean()),
                    "complete_mae_m": float(error[row].mean()),
                }
            )
    return {
        "records": by_seed,
        "aggregates": {"by_seed": per_seed, "seed_equal": seed_equal},
        "train_target_mean_control": {
            "target_mean": target_mean.tolist(),
            "records": baseline_records,
            "group_equal": {
                metric: group_equal_mean(baseline_records, metric)
                for metric in ("first_mae_m", "complete_mae_m")
            },
        },
    }


def reproduction_gates(historical: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, bool]:
    observed = historical["aggregates"]["seed_equal"]
    expected = prior["aggregates"]["seed_equal"]
    if set(observed) != set(expected):
        raise ValueError("historical aggregate metric set changed")
    gates = {
        f"historical_reproduces_{metric}": bool(
            abs(float(observed[metric]) - float(expected[metric])) <= HISTORICAL_TOLERANCE
        )
        for metric in sorted(observed)
    }
    observed_by_seed = historical["aggregates"].get("by_seed")
    expected_by_seed = prior["aggregates"].get("by_seed")
    if not isinstance(observed_by_seed, Mapping) or not isinstance(expected_by_seed, Mapping):
        raise ValueError("historical per-seed aggregates missing")
    if set(observed_by_seed) != set(expected_by_seed):
        raise ValueError("historical aggregate seed set changed")
    for seed in sorted(observed_by_seed):
        observed_seed = observed_by_seed[seed]
        expected_seed = expected_by_seed[seed]
        if set(observed_seed) != set(expected_seed) or set(observed_seed) != set(observed):
            raise ValueError(f"historical metric set changed for seed {seed}")
        for metric in sorted(observed_seed):
            gates[f"historical_reproduces_seed_{seed}_{metric}"] = bool(
                abs(float(observed_seed[metric]) - float(expected_seed[metric]))
                <= HISTORICAL_TOLERANCE
            )
    return gates


def relative_gates(
    ordered: Mapping[str, Any], control: Mapping[str, Any], label: str
) -> dict[str, bool]:
    candidate = ordered["aggregates"]["seed_equal"]
    baseline = control["aggregates"]["seed_equal"]
    gates: dict[str, Any] = {
        f"ordered_first_mae_10pct_better_than_{label}": (
            candidate["first_mae_m"] <= 0.90 * baseline["first_mae_m"]
        ),
        f"ordered_complete_mae_10pct_better_than_{label}": (
            candidate["complete_mae_m"] <= 0.90 * baseline["complete_mae_m"]
        ),
        f"ordered_order_20pt_better_than_{label}": (
            candidate["ordered_pair_accuracy"] >= baseline["ordered_pair_accuracy"] + 0.20
        ),
    }
    for metric in ("first_mae_m", "complete_mae_m"):
        directions = 0
        all_seed_medians = True
        for group_id in CALIBRATION_GROUPS:
            ordered_values = [
                _group_means(_seed_records(ordered["records"], seed), metric)[group_id]
                for seed in TRAIN_SEEDS
            ]
            control_values = [
                _group_means(_seed_records(control["records"], seed), metric)[group_id]
                for seed in TRAIN_SEEDS
            ]
            directions += float(np.mean(ordered_values)) < float(np.mean(control_values))
        for seed in TRAIN_SEEDS:
            a = _group_means(_seed_records(ordered["records"], seed), metric)
            b = _group_means(_seed_records(control["records"], seed), metric)
            all_seed_medians &= statistics.median(a.values()) < statistics.median(b.values())
        gates[f"ordered_{metric}_5of6_and_all_seeds_vs_{label}"] = (
            directions >= 5 and all_seed_medians
        )
    return {name: bool(value) for name, value in gates.items()}


def run_temporal_moment(
    corpus: ShardedMemoryCorpus,
    original_cache: FeatureCacheCorpus,
    condition_caches: Mapping[str, TemporalMomentCache],
    historical_checkpoints: Mapping[int, Path],
    prior_result: Mapping[str, Any],
    output_root: Path,
    *,
    device: torch.device,
    deadline: float,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    split = validate_frozen_split(corpus.group_ids)
    historical_checkpoint_binding = verify_historical_checkpoints(
        historical_checkpoints, prior_result
    )
    stats = TargetStats.fit(FitTargetCorpus(corpus))
    target_mean = fit_target_mean(corpus)
    # Historical forward pass occurs before training, without opening targets.
    historical_predictions = predict_calibration(
        historical_checkpoints, original_cache, device=device
    )
    training, checkpoints = {}, {}
    initial_hashes, schedule_hashes, lr_hashes = {}, {}, {}
    completed_steps = 0
    for condition in CONDITIONS:
        data = FitOnlyData(corpus, condition_caches[condition])
        training[condition], checkpoints[condition] = [], {}
        for seed in TRAIN_SEEDS:
            path = output_root / "checkpoints" / condition / f"seed_{seed}" / "checkpoint_final.pt"
            run, initial_hash = train_with_actual_initial_hash(
                data,
                stats,
                seed=seed,
                device=device,
                deadline=deadline,
                checkpoint_path=path,
            )
            training[condition].append(run)
            checkpoints[condition][seed] = path
            initial_hashes.setdefault(str(seed), []).append(initial_hash)
            schedule_hashes.setdefault(str(seed), []).append(run["schedule_sha256"])
            lr_hashes.setdefault(str(seed), []).append(learning_rate_curve_sha256(run["curve"]))
            completed_steps += int(run["steps"])
    if completed_steps != 3 * len(TRAIN_SEEDS) * TRAIN_STEPS:
        raise RuntimeError("temporal-moment training did not finish all 13,500 steps")
    hash_gates = {
        "initial_state_hashes_matched": all(
            len(set(values)) == 1 for values in initial_hashes.values()
        ),
        "schedule_hashes_matched": all(
            len(set(values)) == 1 for values in schedule_hashes.values()
        ),
        "lr_hashes_matched": all(len(set(values)) == 1 for values in lr_hashes.values()),
    }
    # Calibration targets are opened only after all nine runs finish.
    historical = score_predictions(historical_predictions, corpus, target_mean)
    evidence = {}
    for condition in CONDITIONS:
        predictions = predict_calibration(
            checkpoints[condition], condition_caches[condition], device=device
        )
        evidence[condition] = score_predictions(predictions, corpus, target_mean)
    reproduce = reproduction_gates(historical, prior_result)
    absolute = evaluate_calibration_gates(evidence["ordered_moment"])
    relative = {}
    for label, control in (
        ("endpoint", evidence["endpoint"]),
        ("shuffled_moment", evidence["shuffled_moment"]),
        ("historical_original", historical),
    ):
        relative.update(relative_gates(evidence["ordered_moment"], control, label))
    gates = {
        **{name: bool(value) for name, value in hash_gates.items()},
        **reproduce,
        **{f"ordered_{name}": value for name, value in absolute.items()},
        **relative,
    }
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "mikasa_temporal_moment_sufficiency",
        "split": split,
        "training": training,
        "calibration_opened_after_optimizer_steps": completed_steps,
        "historical_evidence": historical,
        "condition_evidence": evidence,
        "matched_hashes": {
            "initial_state": initial_hashes,
            "schedule": schedule_hashes,
            "learning_rate": lr_hashes,
        },
        "gates": gates,
        "provenance": dict(provenance),
        "historical_checkpoint_binding": historical_checkpoint_binding,
        "original_dev_opened": False,
        "confirmation_opened": False,
        "event_token_mamba_design_allowed": all(gates.values()),
    }
    if time.monotonic() > deadline:
        raise TimeoutError("temporal-moment run exceeded one hour")
    projected = sum(p.stat().st_size for p in output_root.rglob("*") if p.is_file()) + len(
        canonical_json_bytes(result)
    )
    if projected > ARTIFACT_LIMIT_BYTES:
        raise RuntimeError("temporal-moment artifact exceeds 2 GiB")
    write_new_json(output_root / "RESULT.json", result)
    return result
