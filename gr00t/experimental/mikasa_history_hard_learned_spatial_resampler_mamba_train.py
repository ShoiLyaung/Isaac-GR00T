"""Train-only helpers for the MIKASA learned spatial-resampler/Mamba pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from pathlib import Path
import stat
import time
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from gr00t.experimental.mikasa_history_hard_learned_spatial_resampler_mamba import (
    CONDITIONS,
    DEV_SHUFFLE_SEEDS,
    ROWS,
    ROWS_PER_GROUP,
    TRAIN_GROUPS,
    TRAIN_SEED,
    LearnedSpatialResamplerMamba,
    mean_shuffle_geometry_metrics,
    order_cross_entropy,
    scientific_gates,
    score_predictions,
    trainable_parameter_count,
)
from gr00t.experimental.mikasa_history_hard_order_runtime import (
    canonical_json_bytes,
    sha256_file,
    write_once,
)


EXPERIMENT_ID: Final = "g1-mikasa-history-hard-learned-spatial-resampler-mamba-train-r1"
CONDITION_ORDER: Final = tuple(CONDITIONS)
FOLD_ORDER: Final = tuple(TRAIN_GROUPS)
TRAIN_STEPS: Final = 1_000
EVAL_SHUFFLE_SEEDS: Final = tuple(DEV_SHUFFLE_SEEDS)
LEARNING_RATE: Final = 3e-4
FINAL_LEARNING_RATE: Final = 3e-5
WARMUP_STEPS: Final = 100
BETAS: Final = (0.9, 0.95)
WEIGHT_DECAY: Final = 1e-4
GRADIENT_CLIP: Final = 1.0
REMAINING_SECONDS: Final = 21_521.13326904003
ARTIFACT_LIMIT: Final = 512 * 1024 * 1024
CHECKPOINT_SCHEMA: Final = f"{EXPERIMENT_ID}/checkpoint-v1"
TECHNICAL_SOURCE_SHA256: Final = "9b344aed0332a3347b173f1d6cb773523ce57b0ec804fbc23901afbf86e66b9b"
TECHNICAL_CACHE_SHA256: Final = "e2bf628f41ab13df865d076682560f41bfcace8428b5851ea5b2abcaa6b9e0d4"
CACHE_FILE_SHA256: Final = {
    "cue": "00f7bbbb1aa4ff54f3612ebeed5c2455b278a23fc3f64d7159cd663b93206223",
    "current": "eb3f49f2f2d425f1c1c9677511690ee97626d6316e0453b28bc39e15af79551d",
}
ModelFactory = Callable[[str, int], nn.Module]
GroupLoader = Callable[
    [Mapping[str, np.ndarray], np.ndarray, int, torch.device],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class TrainContractError(RuntimeError):
    """The frozen train-only model, fold, checkpoint, or metric contract changed."""


@dataclass(frozen=True)
class FoldSpec:
    heldout_group: int
    candidate_groups: tuple[int, ...]
    heldout_rows: tuple[int, ...]
    candidate_rows: tuple[int, ...]

    def validate(self) -> None:
        if self.heldout_group not in FOLD_ORDER:
            raise TrainContractError("held-out geometry escaped train groups")
        expected_groups = tuple(group for group in FOLD_ORDER if group != self.heldout_group)
        if self.candidate_groups != expected_groups:
            raise TrainContractError("candidate geometry order changed")
        expected_heldout = tuple(
            range(
                self.heldout_group * ROWS_PER_GROUP,
                (self.heldout_group + 1) * ROWS_PER_GROUP,
            )
        )
        expected_candidates = tuple(
            row
            for group in expected_groups
            for row in range(group * ROWS_PER_GROUP, (group + 1) * ROWS_PER_GROUP)
        )
        if self.heldout_rows != expected_heldout or self.candidate_rows != expected_candidates:
            raise TrainContractError("candidate/held-out row isolation changed")
        if set(self.heldout_rows) & set(self.candidate_rows):
            raise TrainContractError("candidate rows contain the held-out geometry")


def fold_spec(heldout_group: int) -> FoldSpec:
    heldout = int(heldout_group)
    candidate_groups = tuple(group for group in FOLD_ORDER if group != heldout)
    spec = FoldSpec(
        heldout_group=heldout,
        candidate_groups=candidate_groups,
        heldout_rows=tuple(range(heldout * ROWS_PER_GROUP, (heldout + 1) * ROWS_PER_GROUP)),
        candidate_rows=tuple(
            row
            for group in candidate_groups
            for row in range(group * ROWS_PER_GROUP, (group + 1) * ROWS_PER_GROUP)
        ),
    )
    spec.validate()
    return spec


def lr_multiplier(step: int, total_steps: int = TRAIN_STEPS) -> float:
    if step < 0 or total_steps < 1:
        raise ValueError("invalid learning-rate step contract")
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = (step + 1 - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    floor = FINAL_LEARNING_RATE / LEARNING_RATE
    return floor + (1.0 - floor) * cosine


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_sha256(tensor).encode())
    return digest.hexdigest()


def model_initialization_hash(model: nn.Module) -> str:
    return state_dict_sha256(model)


def _module_parameters(model: LearnedSpatialResamplerMamba) -> dict[str, list[nn.Parameter]]:
    resampler_modules = (
        model.query_norm,
        model.patch_norm,
        model.cross_attention,
        model.ff_norm,
        model.ff,
    )
    return {
        "raw_projection": list(model.raw_projection.parameters()),
        "residual_projection": list(model.residual_projection.parameters()),
        "spatial_resampler": [
            parameter for module in resampler_modules for parameter in module.parameters()
        ],
        "learned_queries": [model.resampler_queries],
        "identity_embeddings": [
            *model.camera_embedding.parameters(),
            *model.query_embedding.parameters(),
            *model.slot_embedding.parameters(),
        ],
        "temporal_mamba": list(model.temporal.parameters()),
        "shared_classifier": list(model.classifier.parameters()),
    }


def module_hashes(model: LearnedSpatialResamplerMamba) -> dict[str, str]:
    result = {}
    for name, parameters in _module_parameters(model).items():
        digest = hashlib.sha256()
        for index, parameter in enumerate(parameters):
            digest.update(str(index).encode())
            digest.update(_tensor_sha256(parameter).encode())
        result[name] = digest.hexdigest()
    result["learned_queries"] = _tensor_sha256(model.resampler_queries)
    return result


def optimizer_schedule_sha256(total_steps: int = TRAIN_STEPS) -> str:
    values = [LEARNING_RATE * lr_multiplier(step, total_steps) for step in range(total_steps)]
    return hashlib.sha256(canonical_json_bytes(values)).hexdigest()


def effective_samples_sha256(spec: FoldSpec, total_steps: int = TRAIN_STEPS) -> str:
    spec.validate()
    digest = hashlib.sha256()
    for step in range(total_steps):
        digest.update(f"step={step}".encode())
        for group in spec.candidate_groups:
            digest.update(f"|group={group}|rows=".encode())
            digest.update(",".join(map(str, range(group * 6, group * 6 + 6))).encode())
    return digest.hexdigest()


def _default_model_factory(condition: str, seed: int) -> LearnedSpatialResamplerMamba:
    return LearnedSpatialResamplerMamba(condition, seed=seed)


def new_model(
    condition: str,
    device: torch.device,
    *,
    model_factory: ModelFactory = _default_model_factory,
) -> nn.Module:
    if condition not in CONDITION_ORDER:
        raise TrainContractError("condition escaped the frozen order")
    torch.manual_seed(TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
    model = model_factory(condition, TRAIN_SEED).to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TrainContractError("formal training must remain FP32")
    if trainable_parameter_count(model) > 2_500_000:
        raise TrainContractError("trainable parameter cap changed")
    return model


def _group_tensors(
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group not in FOLD_ORDER:
        raise TrainContractError("optimizer/evaluator group escaped rows0-35")
    start = group * ROWS_PER_GROUP
    stop = start + ROWS_PER_GROUP
    if labels.shape != (ROWS, 2) or np.any(labels[start:stop] < 0):
        raise TrainContractError("optimizer/evaluator attempted unopened targets")
    cue = torch.from_numpy(np.asarray(arrays["cue"][start:stop])).to(
        device=device, dtype=torch.float32
    )
    current = torch.from_numpy(np.asarray(arrays["current"][start:stop])).to(
        device=device, dtype=torch.float32
    )
    target = torch.from_numpy(labels[start:stop]).to(device=device, dtype=torch.int64)
    return cue, current, target


def _all_gradients_finite(parameters: Sequence[nn.Parameter]) -> bool:
    gradients = [parameter.grad for parameter in parameters if parameter.requires_grad]
    return bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
    )


def _finite_nonzero_gradient(parameters: Sequence[nn.Parameter]) -> bool:
    return _all_gradients_finite(parameters) and any(
        bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    )


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    spec: FoldSpec,
    *,
    step: int,
    device: torch.device,
    group_loader: GroupLoader = _group_tensors,
    module_parameter_fn: Callable[[Any], Mapping[str, Sequence[nn.Parameter]]] = _module_parameters,
) -> tuple[dict[str, float | int], dict[str, bool], bool]:
    """Run five ordered six-row microbatches and one optimizer update."""

    spec.validate()
    if step not in range(TRAIN_STEPS):
        raise TrainContractError("optimizer step escaped exact 1000-step schedule")
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    pairs: list[float] = []
    slot0: list[float] = []
    slot1: list[float] = []
    visited = []
    for group in spec.candidate_groups:
        cue, current, target = group_loader(arrays, labels, group, device)
        if (
            cue.shape[0] != ROWS_PER_GROUP
            or current.shape[0] != ROWS_PER_GROUP
            or target.shape
            != (
                ROWS_PER_GROUP,
                2,
            )
        ):
            raise TrainContractError(
                "each optimizer microbatch must be one complete six-row geometry"
            )
        output = model(
            cue,
            current,
            shuffle_seed=TRAIN_SEED if model.condition == "mamba_shuffle" else None,
            shuffle_step=step if model.condition == "mamba_shuffle" else None,
        )
        loss = order_cross_entropy(output, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("fold cross entropy is nonfinite")
        (loss / len(spec.candidate_groups)).backward()
        prediction = output.predictions.detach()
        losses.append(float(loss.detach().cpu()))
        pairs.append(float(prediction.eq(target).all(dim=1).float().mean().cpu()))
        slot0.append(float(prediction[:, 0].eq(target[:, 0]).float().mean().cpu()))
        slot1.append(float(prediction[:, 1].eq(target[:, 1]).float().mean().cpu()))
        visited.append(group)
        del cue, current, target, output, loss
    if tuple(visited) != spec.candidate_groups:
        raise TrainContractError("effective microbatch group order changed")
    modules = module_parameter_fn(model)
    all_parameters = [parameter for values in modules.values() for parameter in values]
    gradients_finite = _all_gradients_finite(all_parameters)
    nonzero = {name: _finite_nonzero_gradient(values) for name, values in modules.items()}
    if not gradients_finite:
        raise FloatingPointError("fold gradients are missing or nonfinite")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError("fold gradient norm is nonfinite")
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * lr_multiplier(step)
    optimizer.step()
    return (
        {
            "step": step + 1,
            "cross_entropy": float(np.mean(losses)),
            "pair_accuracy": float(np.mean(pairs)),
            "slot0_accuracy": float(np.mean(slot0)),
            "slot1_accuracy": float(np.mean(slot1)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": float(grad_norm.detach().cpu()),
        },
        nonzero,
        gradients_finite,
    )


def checkpoint_bytes(
    model: nn.Module,
    *,
    condition: str,
    spec: FoldSpec,
    source_sha256: str,
    cache_sha256: str,
    initial_hash: str,
    final_hash: str,
) -> bytes:
    spec.validate()
    if condition not in CONDITION_ORDER or any(
        len(value) != 64 for value in (source_sha256, cache_sha256, initial_hash, final_hash)
    ):
        raise TrainContractError("checkpoint identity input changed")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "condition": condition,
        "heldout_group": spec.heldout_group,
        "candidate_groups": list(spec.candidate_groups),
        "heldout_rows": list(spec.heldout_rows),
        "candidate_rows": list(spec.candidate_rows),
        "steps": TRAIN_STEPS,
        "seed": TRAIN_SEED,
        "dtype": "float32",
        "train_source_result_sha256": source_sha256,
        "technical_source_result_sha256": TECHNICAL_SOURCE_SHA256,
        "technical_cache_result_sha256": cache_sha256,
        "cache_file_sha256": dict(CACHE_FILE_SHA256),
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "optimizer_schedule_sha256": optimizer_schedule_sha256(),
        "effective_samples_sha256": effective_samples_sha256(spec),
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
    heldout_group: int,
    source_sha256: str,
    cache_sha256: str,
    device: torch.device,
    expected_receipt: Mapping[str, Any] | None = None,
    model_factory: ModelFactory = _default_model_factory,
) -> nn.Module:
    spec = fold_spec(heldout_group)
    if condition not in CONDITION_ORDER or len(source_sha256) != 64 or len(cache_sha256) != 64:
        raise TrainContractError("checkpoint load identity input changed")
    if path.is_symlink() or not path.is_file():
        raise TrainContractError("checkpoint must be one regular file")
    if expected_receipt is not None and (
        path.resolve() != Path(str(expected_receipt.get("path"))).resolve()
        or stat.S_IMODE(path.stat().st_mode) != 0o444
        or expected_receipt.get("mode") != "0444"
        or path.stat().st_size != expected_receipt.get("bytes")
        or sha256_file(path) != expected_receipt.get("sha256")
    ):
        raise TrainContractError("checkpoint receipt identity changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "condition": condition,
        "heldout_group": spec.heldout_group,
        "candidate_groups": list(spec.candidate_groups),
        "heldout_rows": list(spec.heldout_rows),
        "candidate_rows": list(spec.candidate_rows),
        "steps": TRAIN_STEPS,
        "seed": TRAIN_SEED,
        "dtype": "float32",
        "train_source_result_sha256": source_sha256,
        "technical_source_result_sha256": TECHNICAL_SOURCE_SHA256,
        "technical_cache_result_sha256": cache_sha256,
        "cache_file_sha256": dict(CACHE_FILE_SHA256),
        "optimizer_schedule_sha256": optimizer_schedule_sha256(),
        "effective_samples_sha256": effective_samples_sha256(spec),
    }
    expected_keys = set(expected) | {
        "initial_state_sha256",
        "final_state_sha256",
        "model_state_dict",
    }
    if set(value) != expected_keys or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise TrainContractError("checkpoint frozen identity fields changed")
    if not isinstance(value.get("model_state_dict"), Mapping):
        raise TrainContractError("checkpoint state dict changed")
    model = new_model(condition, device, model_factory=model_factory)
    initial = model_initialization_hash(model)
    if value.get("initial_state_sha256") != initial:
        raise TrainContractError("checkpoint shared initialization changed")
    model.load_state_dict(value.get("model_state_dict"), strict=True)
    observed_final = state_dict_sha256(model)
    if value.get("final_state_sha256") != observed_final:
        raise TrainContractError("checkpoint final parameter hash changed")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise TrainContractError("checkpoint contains nonfinite parameters")
    return model.eval()


@torch.inference_mode()
def _logits_for_group(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group: int,
    device: torch.device,
    *,
    shuffle_seed: int | None = None,
    group_loader: GroupLoader = _group_tensors,
) -> tuple[torch.Tensor, torch.Tensor]:
    cue, current, target = group_loader(arrays, labels, group, device)
    output = model(
        cue,
        current,
        shuffle_seed=shuffle_seed if model.condition == "mamba_shuffle" else None,
        shuffle_step=0 if model.condition == "mamba_shuffle" else None,
    )
    if output.logits.shape != (ROWS_PER_GROUP, 2, 3) or not bool(
        torch.isfinite(output.logits).all()
    ):
        raise TrainContractError("held-out logits are invalid")
    return output.logits.detach(), target.detach()


def train_fold(
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    condition: str,
    heldout_group: int,
    device: torch.device,
    deadline: float,
    checkpoint_path: Path,
    source_sha256: str = TECHNICAL_SOURCE_SHA256,
    cache_sha256: str = TECHNICAL_CACHE_SHA256,
    expected_initial_hash: str,
    model_factory: ModelFactory = _default_model_factory,
    group_loader: GroupLoader = _group_tensors,
    module_parameter_fn: Callable[[Any], Mapping[str, Sequence[nn.Parameter]]] = _module_parameters,
) -> dict[str, Any]:
    spec = fold_spec(heldout_group)
    model = new_model(condition, device, model_factory=model_factory)
    initial_hash = model_initialization_hash(model)
    if initial_hash != expected_initial_hash:
        raise TrainContractError("condition/fold initialization differs from frozen state")
    initial_modules = (
        module_hashes(model) if isinstance(model, LearnedSpatialResamplerMamba) else {}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    curve = []
    gradient_seen = {name: False for name in module_parameter_fn(model)}
    gradients_finite = True
    started = time.monotonic()
    model.train()
    for step in range(TRAIN_STEPS):
        if time.monotonic() >= deadline:
            raise TimeoutError("six-hour cumulative training budget exhausted")
        record, nonzero, finite = optimizer_step(
            model,
            optimizer,
            arrays,
            labels,
            spec,
            step=step,
            device=device,
            group_loader=group_loader,
            module_parameter_fn=module_parameter_fn,
        )
        curve.append(record)
        gradients_finite &= finite
        gradient_seen = {name: gradient_seen[name] or nonzero[name] for name in gradient_seen}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if time.monotonic() > deadline:
        raise TimeoutError("six-hour cumulative training budget exhausted")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("final fold parameters are nonfinite")
    final_hash = state_dict_sha256(model)
    final_modules = module_hashes(model) if isinstance(model, LearnedSpatialResamplerMamba) else {}
    receipt = write_once(
        checkpoint_path,
        checkpoint_bytes(
            model,
            condition=condition,
            spec=spec,
            source_sha256=source_sha256,
            cache_sha256=cache_sha256,
            initial_hash=initial_hash,
            final_hash=final_hash,
        ),
    )
    # Reload parity uses a candidate geometry. Held-out outputs stay unopened
    # until all 24 final checkpoint files have been written and frozen.
    parity_group = spec.candidate_groups[0]
    model.eval()
    original_logits, _ = _logits_for_group(
        model,
        arrays,
        labels,
        parity_group,
        device,
        shuffle_seed=EVAL_SHUFFLE_SEEDS[0],
        group_loader=group_loader,
    )
    reloaded = load_checkpoint(
        checkpoint_path,
        condition=condition,
        heldout_group=heldout_group,
        source_sha256=source_sha256,
        cache_sha256=cache_sha256,
        device=device,
        expected_receipt=receipt,
        model_factory=model_factory,
    )
    reloaded_logits, _ = _logits_for_group(
        reloaded,
        arrays,
        labels,
        parity_group,
        device,
        shuffle_seed=EVAL_SHUFFLE_SEEDS[0],
        group_loader=group_loader,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if time.monotonic() > deadline:
        raise TimeoutError("six-hour cumulative training budget exhausted")
    reload_parity = bool(torch.equal(original_logits, reloaded_logits))
    del reloaded, original_logits, reloaded_logits
    modules_changed = {
        name: initial_modules[name] != final_modules[name] for name in initial_modules
    }
    return {
        "condition": condition,
        "heldout_group": heldout_group,
        "candidate_groups": list(spec.candidate_groups),
        "heldout_rows": list(spec.heldout_rows),
        "candidate_rows": list(spec.candidate_rows),
        "steps": TRAIN_STEPS,
        "seed": TRAIN_SEED,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "initial_module_sha256": initial_modules,
        "final_module_sha256": final_modules,
        "module_changed": modules_changed,
        "gradient_seen": gradient_seen,
        "gradients_finite": gradients_finite,
        "optimizer_schedule_sha256": optimizer_schedule_sha256(),
        "effective_samples_sha256": effective_samples_sha256(spec),
        "curve": curve,
        "checkpoint": receipt,
        "reload_parity": reload_parity,
        "wall_seconds": time.monotonic() - started,
    }


def evaluate_fold(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    condition: str,
    heldout_group: int,
    device: torch.device,
    group_loader: GroupLoader = _group_tensors,
    deadline: float | None = None,
) -> dict[str, Any]:
    spec = fold_spec(heldout_group)
    groups = np.arange(ROWS, dtype=np.int64) // ROWS_PER_GROUP
    if condition == "mamba_shuffle":
        seed_sections = []
        for seed in EVAL_SHUFFLE_SEEDS:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("six-hour cumulative evaluation budget exhausted")
            logits, _ = _logits_for_group(
                model,
                arrays,
                labels,
                heldout_group,
                device,
                shuffle_seed=seed,
                group_loader=group_loader,
            )
            section = score_predictions(
                logits.argmax(dim=-1).cpu().numpy(), labels, groups, spec.heldout_rows
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("six-hour cumulative evaluation budget exhausted")
            for record in section["records"]:
                record["evaluation_seed"] = seed
            seed_sections.append(section)
        section = mean_shuffle_geometry_metrics(seed_sections)
        section["records"] = [
            record for seed_section in seed_sections for record in seed_section["records"]
        ]
        return {
            "condition": condition,
            "heldout_group": heldout_group,
            "evaluation_seeds": list(EVAL_SHUFFLE_SEEDS),
            "section": section,
            "finite": True,
        }
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("six-hour cumulative evaluation budget exhausted")
    logits, _ = _logits_for_group(
        model,
        arrays,
        labels,
        heldout_group,
        device,
        group_loader=group_loader,
    )
    section = score_predictions(
        logits.argmax(dim=-1).cpu().numpy(), labels, groups, spec.heldout_rows
    )
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("six-hour cumulative evaluation budget exhausted")
    return {
        "condition": condition,
        "heldout_group": heldout_group,
        "evaluation_seeds": [],
        "section": section,
        "finite": bool(torch.isfinite(logits).all()),
    }


def combine_condition_evidence(
    condition: str, fold_results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if (
        condition not in CONDITION_ORDER
        or tuple(int(result.get("heldout_group", -1)) for result in fold_results) != FOLD_ORDER
    ):
        raise TrainContractError("condition fold aggregation order changed")
    by_group = {}
    records = []
    for group, result in zip(FOLD_ORDER, fold_results, strict=True):
        if result.get("condition") != condition or result.get("finite") is not True:
            raise TrainContractError("condition fold evidence identity changed")
        section = result["section"]
        group_key = str(group)
        if tuple(section["by_group"]) != (group_key,):
            raise TrainContractError("fold evidence contains another geometry")
        by_group[group_key] = dict(section["by_group"][group_key])
        records.extend(section.get("records", ()))
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    return {
        "condition": condition,
        "records": records,
        "by_group": by_group,
        "group_equal": {
            metric: float(np.mean([by_group[str(group)][metric] for group in FOLD_ORDER]))
            for metric in metrics
        },
        "folds": list(fold_results),
    }


def recompute_evidence(condition: str, evidence: Mapping[str, Any]) -> bool:
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    records = evidence.get("records")
    if not isinstance(records, list):
        return False
    recomputed = {}
    for group in FOLD_ORDER:
        selected = [record for record in records if record.get("group_id") == group]
        if condition == "mamba_shuffle":
            values = []
            for seed in EVAL_SHUFFLE_SEEDS:
                seeded = [record for record in selected if record.get("evaluation_seed") == seed]
                if len(seeded) != ROWS_PER_GROUP:
                    return False
                values.append(
                    {
                        metric: float(np.mean([float(record[metric]) for record in seeded]))
                        for metric in metrics
                    }
                )
            recomputed[str(group)] = {
                "rows": ROWS_PER_GROUP,
                **{
                    metric: float(np.mean([value[metric] for value in values]))
                    for metric in metrics
                },
            }
        else:
            if len(selected) != ROWS_PER_GROUP:
                return False
            recomputed[str(group)] = {
                "rows": ROWS_PER_GROUP,
                **{
                    metric: float(np.mean([float(record[metric]) for record in selected]))
                    for metric in metrics
                },
            }
    if recomputed != evidence.get("by_group"):
        return False
    group_equal = {
        metric: float(np.mean([recomputed[str(group)][metric] for group in FOLD_ORDER]))
        for metric in metrics
    }
    return group_equal == evidence.get("group_equal")


def substitution_batch(
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    group: int,
    recipient_row: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    spec = fold_spec(group)
    if recipient_row not in spec.heldout_rows:
        raise TrainContractError("substitution recipient escaped held-out geometry")
    donors = tuple(row for row in spec.heldout_rows if row != recipient_row)
    cue_np = np.ascontiguousarray(arrays["cue"][list(donors)])
    current_one = np.asarray(arrays["current"][recipient_row])
    current_np = np.broadcast_to(current_one[None], (len(donors), *current_one.shape)).copy()
    target_np = np.ascontiguousarray(labels[list(donors)])
    if len(donors) != 5 or np.any(target_np < 0):
        raise TrainContractError("substitution donor contract changed")
    return (
        torch.from_numpy(cue_np).to(device=device, dtype=torch.float32),
        torch.from_numpy(current_np).to(device=device, dtype=torch.float32),
        torch.from_numpy(target_np).to(device=device, dtype=torch.int64),
        donors,
    )


@torch.inference_mode()
def evaluate_ordered_substitutions(
    models_by_group: Mapping[int, nn.Module],
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    device: torch.device,
    deadline: float | None = None,
) -> dict[str, Any]:
    if tuple(models_by_group) != FOLD_ORDER:
        raise TrainContractError("ordered substitution model fold order changed")
    correct = 0
    records = []
    for group in FOLD_ORDER:
        model = models_by_group[group]
        if model.condition != "mamba_ordered":
            raise TrainContractError("substitution requires ordered checkpoints")
        for recipient in fold_spec(group).heldout_rows:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("six-hour cumulative substitution budget exhausted")
            cue, current, target, donors = substitution_batch(
                arrays,
                labels,
                group=group,
                recipient_row=recipient,
                device=device,
            )
            output = model(cue, current)
            predictions = output.predictions
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("six-hour cumulative substitution budget exhausted")
            if predictions.shape != (5, 2) or not bool(torch.isfinite(output.logits).all()):
                raise TrainContractError("substitution output is invalid")
            for offset, donor in enumerate(donors):
                passed = bool(torch.equal(predictions[offset], target[offset]))
                correct += int(passed)
                records.append(
                    {
                        "group_id": group,
                        "recipient_row": recipient,
                        "donor_row": donor,
                        "prediction": predictions[offset].cpu().tolist(),
                        "donor_target": target[offset].cpu().tolist(),
                        "pair_correct": passed,
                    }
                )
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("six-hour cumulative substitution budget exhausted")
    total = len(records)
    if total != 180:
        raise TrainContractError("ordered substitution denominator changed")
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "records": records,
    }


def train_scientific_gates(
    evidence: Mapping[str, Any], substitution: Mapping[str, Any]
) -> dict[str, bool]:
    gates = scientific_gates(
        evidence,
        expected_groups=6,
        substitution_correct=int(substitution.get("correct", -1)),
        substitution_total=int(substitution.get("total", -1)),
    )
    gates["all_conditions_independently_recomputable"] = all(
        recompute_evidence(condition, evidence[condition]) for condition in CONDITION_ORDER
    )
    return {key: bool(value) for key, value in gates.items()}
