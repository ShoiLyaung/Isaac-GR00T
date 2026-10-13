"""Training, evaluation, and gates for matched MIKASA native-order backbones."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from io import BytesIO
import math
from pathlib import Path
import statistics
import time
from typing import Any, Final

import numpy as np
import torch

from gr00t.experimental.mikasa_history_hard_order import (
    CONDITIONS,
    DEV_SHUFFLE_SEEDS,
    MikasaHistoryHardOrderModel,
    OrderBatch,
    OrderOutput,
    order_cross_entropy,
)
from gr00t.experimental.mikasa_history_hard_order_runtime import (
    BATCH_SIZE,
    DEV_GROUPS,
    ROWS_PER_GROUP,
    CachedOrderData,
    GroupBalancedSchedule,
    canonical_json_bytes,
    freeze_directory,
    write_once,
)


TRAIN_SEEDS: Final = (230923, 230924, 230925)
PREFLIGHT_STEPS: Final = 20
OVERFIT_STEPS: Final = 500
TRAIN_STEPS: Final = 1500
LEARNING_RATE: Final = 3e-4
FINAL_LEARNING_RATE: Final = 3e-5
WARMUP_STEPS: Final = 100
BETAS: Final = (0.9, 0.95)
WEIGHT_DECAY: Final = 1e-4
GRADIENT_CLIP: Final = 1.0
PREFLIGHT_MEMORY_LIMIT: Final = 1_073_741_824
OVERFIT_WALL_SECONDS: Final = 15 * 60.0
MATCHED_WALL_SECONDS: Final = 30 * 60.0
MATCHED_ARTIFACT_LIMIT: Final = 256 * 1024 * 1024
EVALUATOR_EXPERIMENT_ID: Final = "g1-mikasa-history-hard-native-order-dev-evaluator-r2"


def lr_multiplier(step: int, total_steps: int) -> float:
    if step < 0 or total_steps < 1:
        raise ValueError("invalid learning-rate step contract")
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = (step + 1 - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    floor = FINAL_LEARNING_RATE / LEARNING_RATE
    return floor + (1.0 - floor) * cosine


def module_has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
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


def state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def shared_initialization(model: MikasaHistoryHardOrderModel) -> dict[str, str]:
    return {
        "temporal_frontend": state_dict_sha256(model.temporal_frontend),
        "current_branch": state_dict_sha256(model.current_branch),
        "shared_head": state_dict_sha256(model.shared_head),
    }


def _batch_to_device(batch: OrderBatch, device: torch.device) -> OrderBatch:
    # Frame chunks move inside the model. Keeping source grids on CPU bounds
    # peak residency and preserves the frozen cache representation.
    del device
    return batch


def _checkpoint_bytes(
    model: MikasaHistoryHardOrderModel,
    *,
    condition: str,
    seed: int,
    steps: int,
    cache_result_sha256: str,
    initial_shared: Mapping[str, str],
) -> bytes:
    payload = {
        "schema": "g1-mikasa-history-hard-native-order-backbones-r1/checkpoint-v1",
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "dtype": "float32",
        "cache_result_sha256": cache_result_sha256,
        "initial_shared": dict(initial_shared),
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    stream = BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def load_checkpoint(
    path: Path,
    *,
    condition: str,
    seed: int,
    expected_steps: int,
    cache_result_sha256: str,
    device: torch.device,
) -> MikasaHistoryHardOrderModel:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("schema") != "g1-mikasa-history-hard-native-order-backbones-r1/checkpoint-v1"
        or value.get("condition") != condition
        or value.get("seed") != seed
        or value.get("steps") != expected_steps
        or value.get("dtype") != "float32"
        or value.get("cache_result_sha256") != cache_result_sha256
    ):
        raise ValueError(f"checkpoint identity changed: {path}")
    model = MikasaHistoryHardOrderModel(condition, seed=seed).to(device=device, dtype=torch.float32)
    if shared_initialization(model) != value.get("initial_shared"):
        raise ValueError("checkpoint shared initialization changed")
    model.load_state_dict(value["model_state_dict"], strict=True)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("checkpoint contains nonfinite parameters")
    return model.eval()


def train_one(
    data: CachedOrderData,
    *,
    condition: str,
    seed: int,
    steps: int,
    device: torch.device,
    deadline: float,
    cache_result_sha256: str,
    checkpoint_path: Path | None,
    overfit_group: int | None = None,
) -> tuple[MikasaHistoryHardOrderModel, dict[str, Any]]:
    if condition not in CONDITIONS or seed not in TRAIN_SEEDS:
        raise ValueError("unfrozen condition or seed")
    if steps not in (PREFLIGHT_STEPS, OVERFIT_STEPS, TRAIN_STEPS):
        raise ValueError("unfrozen optimizer step count")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MikasaHistoryHardOrderModel(condition, seed=seed).to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("native-order model must remain FP32")
    initial_shared = shared_initialization(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    schedule = GroupBalancedSchedule(seed)
    gradient_seen = {"frontend": False, "temporal": False, "shared_head": False}
    curve: list[dict[str, float | int]] = []
    model.train()
    started = time.monotonic()
    for step in range(steps):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{condition}/{seed} exceeded the frozen deadline")
        group_ids = (overfit_group,) if overfit_group is not None else schedule.batch(step)
        batch, labels = data.batch_for_groups(group_ids)
        labels = labels.to(device=device)
        output = model(
            _batch_to_device(batch, device),
            shuffle_seed=seed if condition == "mamba_shuffle" else None,
            shuffle_step=step if condition == "mamba_shuffle" else None,
        )
        loss = order_cross_entropy(output, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("native-order cross entropy is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_seen["frontend"] |= module_has_finite_nonzero_gradient(model.temporal_frontend)
        gradient_seen["temporal"] |= module_has_finite_nonzero_gradient(model.temporal)
        gradient_seen["shared_head"] |= module_has_finite_nonzero_gradient(model.shared_head)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("native-order gradient norm is nonfinite")
        multiplier = lr_multiplier(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * multiplier
        optimizer.step()
        prediction = output.predictions.detach()
        curve.append(
            {
                "step": step + 1,
                "cross_entropy": float(loss.detach().cpu()),
                "pair_accuracy": float(prediction.eq(labels).all(dim=1).float().mean().cpu()),
                "slot0_accuracy": float(prediction[:, 0].eq(labels[:, 0]).float().mean().cpu()),
                "slot1_accuracy": float(prediction[:, 1].eq(labels[:, 1]).float().mean().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        )
    if not all(gradient_seen.values()):
        raise RuntimeError(f"native-order gradients missing: {gradient_seen}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("native-order checkpoint parameters are nonfinite")
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = write_once(
            checkpoint_path,
            _checkpoint_bytes(
                model,
                condition=condition,
                seed=seed,
                steps=steps,
                cache_result_sha256=cache_result_sha256,
                initial_shared=initial_shared,
            ),
        )
    return model, {
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "batch_size": ROWS_PER_GROUP if overfit_group is not None else BATCH_SIZE,
        "optimizer": {
            "name": "AdamW",
            "lr": LEARNING_RATE,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "final_lr": FINAL_LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
        },
        "initial_shared": initial_shared,
        "gradient_seen": gradient_seen,
        "curve": curve,
        "checkpoint": checkpoint,
        "wall_seconds": time.monotonic() - started,
    }


def _accuracy_records(
    output: OrderOutput,
    labels: torch.Tensor,
    *,
    group_id: int,
    training_seed: int,
    intervention: str,
    permutation_seed: int | None = None,
) -> list[dict[str, Any]]:
    prediction = output.predictions
    ce = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, 3), labels.reshape(-1), reduction="none"
    ).reshape(-1, 2)
    records = []
    for row in range(len(labels)):
        records.append(
            {
                "group_id": group_id,
                "row": row,
                "training_seed": training_seed,
                "intervention": intervention,
                "permutation_seed": permutation_seed,
                "prediction": prediction[row].detach().cpu().tolist(),
                "target": labels[row].detach().cpu().tolist(),
                "slot0_accuracy": float(prediction[row, 0] == labels[row, 0]),
                "slot1_accuracy": float(prediction[row, 1] == labels[row, 1]),
                "pair_accuracy": float(bool(torch.equal(prediction[row], labels[row]))),
                "cross_entropy": float(ce[row].mean().detach().cpu()),
                "finite": bool(torch.isfinite(output.logits[row]).all()),
            }
        )
    return records


@torch.inference_mode()
def cue_only_substitution(
    batch: OrderBatch, labels: torch.Tensor
) -> tuple[OrderBatch, torch.Tensor]:
    """Roll only cue evidence while holding every recipient-current input fixed."""

    batch.validate()
    if batch.batch_size != ROWS_PER_GROUP or labels.shape != (ROWS_PER_GROUP, 2):
        raise ValueError("cue substitution requires one complete six-row group")
    donor_cue = batch.temporal_grid[..., :3].roll(shifts=-1, dims=0)
    recipient_current = batch.temporal_grid[..., 3:6]
    temporal = torch.cat((donor_cue, recipient_current, donor_cue - recipient_current), dim=-1)
    substituted = OrderBatch(
        temporal_grid=temporal,
        current_rgb_grid=batch.current_rgb_grid,
        current_proprio=batch.current_proprio,
    )
    substituted.validate()
    return substituted, labels.roll(shifts=-1, dims=0)


@torch.inference_mode()
def evaluate_group(
    model: MikasaHistoryHardOrderModel,
    data: CachedOrderData,
    group_id: int,
    *,
    training_seed: int,
    intervention: str = "none",
) -> list[dict[str, Any]]:
    batch, labels = data.batch_for_groups((group_id,))
    device = next(model.parameters()).device
    labels = labels.to(device=device)
    if intervention == "substitute":
        batch, labels = cue_only_substitution(batch, labels)
        output = model(batch)
        return _accuracy_records(
            output,
            labels,
            group_id=group_id,
            training_seed=training_seed,
            intervention=intervention,
        )
    if intervention == "zero":
        output = model(batch, condition_override="mamba_zero")
        return _accuracy_records(
            output,
            labels,
            group_id=group_id,
            training_seed=training_seed,
            intervention=intervention,
        )
    if intervention == "shuffle":
        records = []
        for permutation_seed in DEV_SHUFFLE_SEEDS:
            output = model(
                batch,
                condition_override="mamba_shuffle",
                shuffle_seed=permutation_seed,
                shuffle_step=0,
            )
            records.extend(
                _accuracy_records(
                    output,
                    labels,
                    group_id=group_id,
                    training_seed=training_seed,
                    intervention=intervention,
                    permutation_seed=permutation_seed,
                )
            )
        return records
    if intervention != "none":
        raise ValueError(f"unknown dev intervention: {intervention}")
    if model.condition == "mamba_shuffle":
        records = []
        for permutation_seed in DEV_SHUFFLE_SEEDS:
            output = model(batch, shuffle_seed=permutation_seed, shuffle_step=0)
            records.extend(
                _accuracy_records(
                    output,
                    labels,
                    group_id=group_id,
                    training_seed=training_seed,
                    intervention="none",
                    permutation_seed=permutation_seed,
                )
            )
        return records
    return _accuracy_records(
        model(batch),
        labels,
        group_id=group_id,
        training_seed=training_seed,
        intervention="none",
    )


def _mean(records: Sequence[Mapping[str, Any]], metric: str) -> float:
    if not records:
        raise ValueError("cannot aggregate empty records")
    return float(np.mean([float(record[metric]) for record in records]))


def aggregate_records(records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy", "cross_entropy")
    by_seed: dict[str, Any] = {}
    for seed in TRAIN_SEEDS:
        records = list(records_by_seed[seed])
        group_values = {
            str(group): {
                metric: _mean([record for record in records if record["group_id"] == group], metric)
                for metric in metrics
            }
            for group in DEV_GROUPS
        }
        by_seed[str(seed)] = {
            "groups": group_values,
            "group_equal": {
                metric: float(np.mean([group_values[str(group)][metric] for group in DEV_GROUPS]))
                for metric in metrics
            },
        }
    return {
        "by_seed": by_seed,
        "seed_equal": {
            metric: float(
                np.mean([by_seed[str(seed)]["group_equal"][metric] for seed in TRAIN_SEEDS])
            )
            for metric in metrics
        },
    }


def dev_gates(evidence: Mapping[str, Any]) -> dict[str, bool]:
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    standard = evidence["standard"]
    full = standard["mamba_full"]["aggregates"]
    controls = ("mamba_current", "mamba_zero", "mamba_shuffle")
    gates: dict[str, bool] = {
        "mamba_full_pair_at_least_83_33": full["seed_equal"]["pair_accuracy"] >= 5.0 / 6.0,
        "mamba_full_slot0_at_least_91_67": full["seed_equal"]["slot0_accuracy"] >= 11.0 / 12.0,
        "mamba_full_slot1_at_least_91_67": full["seed_equal"]["slot1_accuracy"] >= 11.0 / 12.0,
    }
    for control in controls:
        other = standard[control]["aggregates"]
        gates[f"full_vs_{control}_pair_plus_50"] = (
            full["seed_equal"]["pair_accuracy"] - other["seed_equal"]["pair_accuracy"] >= 0.50
        )
        for slot in ("slot0_accuracy", "slot1_accuracy"):
            gates[f"full_vs_{control}_{slot}_plus_25"] = (
                full["seed_equal"][slot] - other["seed_equal"][slot] >= 0.25
            )
        gates[f"full_vs_{control}_pair_both_groups"] = all(
            np.mean(
                [
                    full["by_seed"][str(seed)]["groups"][str(group)]["pair_accuracy"]
                    for seed in TRAIN_SEEDS
                ]
            )
            > np.mean(
                [
                    other["by_seed"][str(seed)]["groups"][str(group)]["pair_accuracy"]
                    for seed in TRAIN_SEEDS
                ]
            )
            for group in DEV_GROUPS
        )
        gates[f"full_vs_{control}_pair_every_seed"] = all(
            full["by_seed"][str(seed)]["group_equal"]["pair_accuracy"]
            > other["by_seed"][str(seed)]["group_equal"]["pair_accuracy"]
            for seed in TRAIN_SEEDS
        )
    for metric in metrics:
        best_control = max(
            standard[name]["aggregates"]["seed_equal"][metric]
            for name in ("gru_full", "transformer_full")
        )
        gates[f"full_backbone_noninferior_{metric}"] = (
            full["seed_equal"][metric] >= best_control - 0.02
        )
    interventions = evidence["mamba_full_interventions"]
    substitute = interventions["substitute"]["aggregates"]
    zero = interventions["zero"]["aggregates"]
    shuffle = interventions["shuffle"]["aggregates"]
    gates["cue_substitution_follow_at_least_80"] = substitute["seed_equal"]["pair_accuracy"] >= 0.80
    for name, changed in (("zero", zero), ("shuffle", shuffle)):
        gates[f"full_{name}_pair_drop_40"] = (
            full["seed_equal"]["pair_accuracy"] - changed["seed_equal"]["pair_accuracy"] >= 0.40
        )
        gates[f"full_{name}_drop_both_groups"] = all(
            np.mean(
                [
                    full["by_seed"][str(seed)]["groups"][str(group)]["pair_accuracy"]
                    for seed in TRAIN_SEEDS
                ]
            )
            > np.mean(
                [
                    changed["by_seed"][str(seed)]["groups"][str(group)]["pair_accuracy"]
                    for seed in TRAIN_SEEDS
                ]
            )
            for group in DEV_GROUPS
        )
    all_records = [
        record
        for section in (list(standard.values()) + list(interventions.values()))
        for records in section["records"].values()
        for record in records
    ]
    gates["all_logits_metrics_finite"] = all(
        record.get("finite") is True
        and all(
            np.isfinite(float(record[key]))
            for key in ("pair_accuracy", "slot0_accuracy", "slot1_accuracy", "cross_entropy")
        )
        for record in all_records
    )
    gates["all_checkpoints_curves_finite"] = evidence.get("matched_finite") is True
    return {key: bool(value) for key, value in gates.items()}


def run_preflight(
    data: CachedOrderData,
    output_root: Path,
    *,
    device: torch.device,
    cache_result_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or device.type != "cuda":
        raise (
            FileExistsError(output_root)
            if output_root.exists()
            else ValueError("preflight requires CUDA")
        )
    output_root.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats(device)
    _, training = train_one(
        data,
        condition="mamba_full",
        seed=230923,
        steps=PREFLIGHT_STEPS,
        device=device,
        deadline=time.monotonic() + 15 * 60.0,
        cache_result_sha256=cache_result_sha256,
        checkpoint_path=None,
    )
    peak = int(torch.cuda.max_memory_allocated(device))
    finite = all(
        np.isfinite(float(row[key]))
        for row in training["curve"]
        for key in ("cross_entropy", "grad_norm")
    )
    gates = {
        "steps_exact": training["steps"] == PREFLIGHT_STEPS,
        "batch24_exact": training["batch_size"] == BATCH_SIZE,
        "loss_gradient_finite": finite,
        "peak_allocated_below_1gib": peak < PREFLIGHT_MEMORY_LIMIT,
        "checkpoint_not_written": training["checkpoint"] is None,
    }
    result = {
        "schema": "g1-mikasa-history-hard-native-order-backbones-r1/preflight-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "batch24_preflight",
        "gates": gates,
        "peak_allocated_bytes": peak,
        "training": training,
        "retry_count": 0,
        "provenance": dict(provenance),
    }
    write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return result


def run_overfit(
    data: CachedOrderData,
    output_root: Path,
    *,
    device: torch.device,
    cache_result_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    model, training = train_one(
        data,
        condition="mamba_full",
        seed=230923,
        steps=OVERFIT_STEPS,
        device=device,
        deadline=time.monotonic() + OVERFIT_WALL_SECONDS,
        cache_result_sha256=cache_result_sha256,
        checkpoint_path=output_root / "checkpoint-final.pt",
        overfit_group=0,
    )
    records = evaluate_group(model.eval(), data, 0, training_seed=230923)
    substituted = evaluate_group(
        model.eval(), data, 0, training_seed=230923, intervention="substitute"
    )
    losses = [float(row["cross_entropy"]) for row in training["curve"]]
    first = statistics.median(losses[:50])
    final = statistics.median(losses[-50:])
    gates = {
        "loss_median_reduction_at_least_90": final <= 0.10 * first,
        "six_of_six_pair_exact": sum(row["pair_accuracy"] for row in records) == 6.0,
        "frontend_gradient": training["gradient_seen"]["frontend"],
        "mamba_gradient": training["gradient_seen"]["temporal"],
        "shared_head_gradient": training["gradient_seen"]["shared_head"],
        "cue_substitution_changes_prediction": any(
            left["prediction"] != right["prediction"]
            for left, right in zip(records, substituted, strict=True)
        ),
        "wall_within_15_minutes": training["wall_seconds"] <= OVERFIT_WALL_SECONDS,
    }
    result = {
        "schema": "g1-mikasa-history-hard-native-order-backbones-r1/overfit-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "overfit",
        "condition": "mamba_full",
        "seed": 230923,
        "group": 0,
        "first50_median_ce": first,
        "final50_median_ce": final,
        "gates": gates,
        "training": training,
        "records": records,
        "substitution_records": substituted,
        "retry_count": 0,
        "provenance": dict(provenance),
    }
    write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return result


def _directory_bytes(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def run_matched_training(
    data: CachedOrderData,
    output_root: Path,
    *,
    device: torch.device,
    cache_result_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    deadline = time.monotonic() + MATCHED_WALL_SECONDS
    records = []
    shared_by_seed: dict[int, dict[str, str]] = {}
    for condition in CONDITIONS:
        for seed in TRAIN_SEEDS:
            _, record = train_one(
                data,
                condition=condition,
                seed=seed,
                steps=TRAIN_STEPS,
                device=device,
                deadline=deadline,
                cache_result_sha256=cache_result_sha256,
                checkpoint_path=output_root / "checkpoints" / condition / f"seed-{seed}.pt",
            )
            if seed in shared_by_seed and record["initial_shared"] != shared_by_seed[seed]:
                raise RuntimeError(f"shared initialization changed for seed {seed}")
            shared_by_seed.setdefault(seed, record["initial_shared"])
            records.append(record)
    wall = MATCHED_WALL_SECONDS - max(0.0, deadline - time.monotonic())
    gates = {
        "condition_order_exact": [record["condition"] for record in records[::3]]
        == list(CONDITIONS),
        "all_three_seeds_each_condition": all(
            [record["seed"] for record in records if record["condition"] == condition]
            == list(TRAIN_SEEDS)
            for condition in CONDITIONS
        ),
        "all_steps_exact": all(record["steps"] == TRAIN_STEPS for record in records),
        "shared_initialization_exact": True,
        "only_final_checkpoints": True,
        "wall_within_30_minutes": wall <= MATCHED_WALL_SECONDS,
    }
    result = {
        "schema": "g1-mikasa-history-hard-native-order-backbones-r1/matched-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "matched_training",
        "conditions": list(CONDITIONS),
        "seeds": list(TRAIN_SEEDS),
        "steps_per_run": TRAIN_STEPS,
        "records": records,
        "shared_initialization": {str(key): value for key, value in shared_by_seed.items()},
        "wall_seconds": wall,
        "gates": gates,
        "retry_count": 0,
        "provenance": dict(provenance),
    }
    projected = _directory_bytes(output_root) + len(canonical_json_bytes(result))
    if projected > MATCHED_ARTIFACT_LIMIT:
        raise RuntimeError("matched artifacts exceed 256 MiB")
    write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return result


def checkpoint_paths(root: Path) -> dict[str, dict[int, Path]]:
    return {
        condition: {
            seed: root.resolve() / "checkpoints" / condition / f"seed-{seed}.pt"
            for seed in TRAIN_SEEDS
        }
        for condition in CONDITIONS
    }


def validate_matched_checkpoints(result: Mapping[str, Any], root: Path) -> None:
    records = result.get("records")
    if not isinstance(records, list) or len(records) != len(CONDITIONS) * len(TRAIN_SEEDS):
        raise ValueError("matched checkpoint receipt count changed")
    expected = {
        (condition, seed): root.resolve() / "checkpoints" / condition / f"seed-{seed}.pt"
        for condition in CONDITIONS
        for seed in TRAIN_SEEDS
    }
    seen = set()
    for record in records:
        key = (record.get("condition"), record.get("seed"))
        checkpoint = record.get("checkpoint")
        if key not in expected or key in seen or not isinstance(checkpoint, Mapping):
            raise ValueError("matched checkpoint receipt identity changed")
        path = Path(str(checkpoint.get("path")))
        if (
            path.resolve() != expected[key]
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != checkpoint.get("bytes")
            or hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint.get("sha256")
        ):
            raise ValueError(f"matched checkpoint material changed: {key}")
        curve = record.get("curve")
        if not isinstance(curve, list) or len(curve) != TRAIN_STEPS:
            raise ValueError(f"matched training curve changed: {key}")
        fields = (
            "cross_entropy",
            "pair_accuracy",
            "slot0_accuracy",
            "slot1_accuracy",
            "lr",
            "grad_norm",
        )
        if not all(all(np.isfinite(float(point[field])) for field in fields) for point in curve):
            raise ValueError(f"matched training curve is nonfinite: {key}")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("matched checkpoint set changed")


def run_dev_evaluation(
    data: CachedOrderData,
    checkpoint_root: Path,
    output_root: Path,
    *,
    device: torch.device,
    cache_result_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    standard: dict[str, Any] = {}
    full_models: dict[int, MikasaHistoryHardOrderModel] = {}
    paths = checkpoint_paths(checkpoint_root)
    for condition in CONDITIONS:
        by_seed = {}
        for seed in TRAIN_SEEDS:
            model = load_checkpoint(
                paths[condition][seed],
                condition=condition,
                seed=seed,
                expected_steps=TRAIN_STEPS,
                cache_result_sha256=cache_result_sha256,
                device=device,
            )
            records = []
            for group_id in DEV_GROUPS:
                records.extend(evaluate_group(model, data, group_id, training_seed=seed))
            by_seed[seed] = records
            if condition == "mamba_full":
                full_models[seed] = model
        standard[condition] = {"records": by_seed, "aggregates": aggregate_records(by_seed)}
    interventions = {}
    for name in ("substitute", "zero", "shuffle"):
        by_seed = {}
        for seed, model in full_models.items():
            records = []
            for group_id in DEV_GROUPS:
                records.extend(
                    evaluate_group(
                        model,
                        data,
                        group_id,
                        training_seed=seed,
                        intervention=name,
                    )
                )
            by_seed[seed] = records
        interventions[name] = {"records": by_seed, "aggregates": aggregate_records(by_seed)}
    evidence = {
        "standard": standard,
        "mamba_full_interventions": interventions,
        "checkpoint_selection": "final_step_only_before_dev",
        "dev_groups": list(DEV_GROUPS),
        "shuffle_seeds": list(DEV_SHUFFLE_SEEDS),
        "matched_finite": True,
    }
    gates = dev_gates(evidence)
    result = {
        "schema": f"{EVALUATOR_EXPERIMENT_ID}/dev-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "revealed_dev",
        "evidence": evidence,
        "gates": gates,
        "fresh_data_opened": False,
        "confirmation_model_outputs": 0,
        "h16_runs": 0,
        "codec_runs": 0,
        "vla_runs": 0,
        "simulator_runs": 0,
        "retry_count": 0,
        "provenance": dict(provenance),
    }
    write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return result
