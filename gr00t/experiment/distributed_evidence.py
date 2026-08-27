# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow distributed-conformance evidence for clean-room finetuning runs.

The hook is inert unless GROOT_CLEANROOM_EVIDENCE=1. It does not alter the
optimizer, data path, or model outputs. Rank-local summaries are gathered and
compared before rank 0 writes one immutable JSON file per rank and phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers.trainer_callback import TrainerCallback

from gr00t.experiment.dist_utils import get_collective_trace, record_collective, run_on_rank0


def evidence_enabled() -> bool:
    return os.environ.get("GROOT_CLEANROOM_EVIDENCE") == "1"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]
    trainable = [item for item in parameters if item["requires_grad"]]
    trainable_names = sorted(item["name"] for item in trainable)
    trainable_numel = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "parameter_count": len(parameters),
        "parameters_sha256": _digest(parameters),
        "trainable_parameter_count": len(trainable),
        "trainable_key_set_sha256": _digest(trainable_names),
        "trainable_numel": int(trainable_numel),
        "trainable_names": trainable_names,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_contract_summary(config: Any, train_dataset: Any) -> dict[str, Any]:
    datasets = []
    leaves = getattr(train_dataset, "datasets", [])
    for index, spec in enumerate(config.data.datasets):
        path = Path(spec.dataset_paths[0])
        info_path = path / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        view_paths = sorted((path / "meta").glob("*_view_manifest.json"))
        if len(view_paths) > 1:
            raise ValueError(
                f"dataset has multiple loader view manifests: "
                f"{[view_path.name for view_path in view_paths]}"
            )
        view_path = view_paths[0] if view_paths else None
        adapter_path = Path(spec.adapter_config_path) if spec.adapter_config_path else None
        modality = config.data.modality_configs[spec.embodiment_tag]
        leaf = leaves[index] if index < len(leaves) else None
        entry = {
            "dataset_path": str(path),
            "dataset_adapter": spec.dataset_adapter,
            "adapter_config_path": str(adapter_path) if adapter_path else None,
            "adapter_config_sha256": (_file_sha256(adapter_path) if adapter_path else None),
            "embodiment_tag": spec.embodiment_tag,
            "info": {
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "fps": info.get("fps"),
                "splits": info.get("splits"),
                "state_shape": info.get("features", {}).get("observation.state", {}).get("shape"),
                "action_shape": info.get("features", {}).get("action", {}).get("shape"),
            },
            "modality_config": _jsonable(modality),
            "leaf_dataset_class": (
                f"{type(leaf).__module__}.{type(leaf).__qualname__}" if leaf else None
            ),
            "action_horizon": getattr(leaf, "action_horizon", None),
        }
        if view_path is not None:
            entry["loader_view_manifest_path"] = str(view_path)
            entry["loader_view_manifest"] = json.loads(view_path.read_text(encoding="utf-8"))
            entry["loader_view_manifest_sha256"] = _file_sha256(view_path)
        datasets.append(entry)
    return {
        "datasets": datasets,
        "statistics_sha256": _digest(train_dataset.get_dataset_statistics()),
    }


def _dataset_sampling_summary(config: Any, trainer: Any) -> dict[str, Any]:
    dataloader_num_workers = int(config.training.dataloader_num_workers)
    train_dataset = getattr(trainer, "train_dataset", None)
    get_sampling_evidence = getattr(train_dataset, "get_sampling_evidence", None)
    if not callable(get_sampling_evidence):
        return {
            "available": False,
            "dataloader_num_workers": dataloader_num_workers,
            "rank_local_counts_exact": False,
        }

    evidence = _jsonable(get_sampling_evidence())
    rank_local_counts_exact = bool(
        dataloader_num_workers == 0 and evidence.get("rank_local_counts_available")
    )
    return {
        "available": True,
        "dataloader_num_workers": dataloader_num_workers,
        "rank_local_counts_exact": rank_local_counts_exact,
        "evidence": evidence,
    }


_DECLARED_ZERO_STAGE_CACHE_ATTR = "_cleanroom_declared_deepspeed_zero_stage"


def _resolve_declared_zero_stage(config: Any) -> int | None:
    if config.training.num_gpus <= 1 or config.training.use_ddp:
        return None
    ds_config = config.get_deepspeed_config()
    return int(ds_config["zero_optimization"]["stage"])


def _cache_declared_zero_stage(config: Any) -> int | None:
    """Resolve and cache the declared stage while source config files exist."""
    stage = _resolve_declared_zero_stage(config)
    setattr(config, _DECLARED_ZERO_STAGE_CACHE_ATTR, stage)
    return stage


def _declared_zero_stage(config: Any) -> int | None:
    """Return the pretrain-cached stage, resolving it for standalone callers."""
    if hasattr(config, _DECLARED_ZERO_STAGE_CACHE_ATTR):
        return getattr(config, _DECLARED_ZERO_STAGE_CACHE_ATTR)
    return _cache_declared_zero_stage(config)


def _actual_zero_stage(trainer: Any) -> tuple[int | None, str | None]:
    candidates = [
        getattr(trainer, "deepspeed", None),
        getattr(trainer, "model_wrapped", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        stage = getattr(candidate, "zero_optimization_stage", None)
        if callable(stage):
            stage = stage()
        if stage is None:
            stage = getattr(getattr(candidate, "config", None), "zero_optimization_stage", None)
        if stage is not None:
            return int(stage), f"{type(candidate).__module__}.{type(candidate).__qualname__}"
    return None, None


class FiniteTrainingMonitor(TrainerCallback):
    """Capture one raw loss, gradient norm, and skipped-step bit per step."""

    def __init__(self) -> None:
        self._losses: list[torch.Tensor] = []
        self._grad_norms: list[torch.Tensor] = []
        self._skipped: list[bool] = []
        self._overflow: list[bool] = []
        self._trainer = None
        self._first_batch: dict[str, Any] | None = None

    def capture_loss(self, loss: torch.Tensor) -> None:
        if torch.is_tensor(loss) and loss.numel() == 1:
            self._losses.append(loss.detach().reshape(()))

    def capture_batch(self, inputs: Mapping[str, Any]) -> None:
        if self._first_batch is not None:
            return
        payload = inputs.get("inputs")
        if not isinstance(payload, Mapping):
            payload = inputs
        summary = {}
        for key, value in sorted(payload.items()):
            if not torch.is_tensor(value):
                summary[key] = {"python_type": type(value).__qualname__}
                continue
            item = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": (
                    bool(torch.isfinite(value).all().item()) if value.is_floating_point() else None
                ),
            }
            if key == "action_mask":
                item["sum"] = float(value.sum().item())
            if key == "embodiment_id":
                item["unique"] = sorted(int(entry) for entry in value.unique().tolist())
            summary[key] = item
        self._first_batch = summary

    def bind_trainer(self, trainer: Any) -> None:
        self._trainer = trainer
        trainer._cleanroom_finite_monitor = self

    def on_optimizer_step(
        self,
        args,
        state,
        control,
        model=None,
        optimizer=None,
        **kwargs,
    ):
        del args, state, control
        wrapped_model = (
            getattr(self._trainer, "model_wrapped", None) if self._trainer is not None else model
        )
        norm = getattr(wrapped_model, "get_global_grad_norm", lambda: None)()
        if norm is not None:
            if not torch.is_tensor(norm):
                norm = torch.tensor(
                    float(norm),
                    device=torch.device("cuda", torch.cuda.current_device()),
                )
            self._grad_norms.append(norm.detach().reshape(()))
        skipped = False
        if self._trainer is not None:
            skipped = bool(self._trainer.accelerator.optimizer_step_was_skipped)
        self._skipped.append(skipped)
        overflow = getattr(
            getattr(wrapped_model, "optimizer", optimizer),
            "overflow",
            False,
        )
        self._overflow.append(bool(overflow))

    @staticmethod
    def _values(tensors: list[torch.Tensor]) -> list[float]:
        if not tensors:
            return []
        return torch.stack(tensors).float().cpu().tolist()

    def summary(self, expected_steps: int) -> dict[str, Any]:
        try:
            losses = self._values(self._losses)
            grad_norms = self._values(self._grad_norms)
            error = None
        except BaseException as exc:
            losses = []
            grad_norms = []
            error = f"{type(exc).__name__}: {exc}"
        finite_losses = all(math.isfinite(value) for value in losses)
        finite_grad_norms = all(math.isfinite(value) for value in grad_norms)
        complete = (
            error is None
            and len(losses) == expected_steps
            and len(grad_norms) == expected_steps
            and len(self._skipped) == expected_steps
            and len(self._overflow) == expected_steps
            and finite_losses
            and finite_grad_norms
            and not any(self._skipped)
            and not any(self._overflow)
        )
        return {
            "expected_optimizer_steps": expected_steps,
            "raw_loss_count": len(losses),
            "raw_losses": losses,
            "finite_loss_count": sum(math.isfinite(value) for value in losses),
            "gradient_norm_count": len(grad_norms),
            "gradient_norms": grad_norms,
            "finite_gradient_norm_count": sum(math.isfinite(value) for value in grad_norms),
            "optimizer_step_events": len(self._skipped),
            "skipped_optimizer_steps": sum(self._skipped),
            "overflow_events": sum(self._overflow),
            "first_batch": self._first_batch,
            "complete": complete,
            "error": error,
        }


def create_monitor() -> FiniteTrainingMonitor | None:
    if not evidence_enabled():
        return None
    return FiniteTrainingMonitor()


class EvidenceTrainerMixin:
    """Observe Trainer.training_step's detached scalar without a model hook."""

    def training_step(self, *args, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs is None and len(args) >= 2:
            inputs = args[1]
        monitor = getattr(self, "_cleanroom_finite_monitor", None)
        if monitor is not None and isinstance(inputs, Mapping):
            monitor.capture_batch(inputs)
        loss = super().training_step(*args, **kwargs)
        if monitor is not None:
            monitor.capture_loss(loss)
        return loss


def _rank_identity() -> dict[str, int]:
    return {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "cuda_current_device": (torch.cuda.current_device() if torch.cuda.is_available() else -1),
    }


def _nccl_flight_recorder_summary() -> dict[str, Any]:
    required = os.environ.get("GROOT_CLEANROOM_REQUIRE_NCCL_TRACE") == "1"
    if not required:
        return {"required": False, "complete": None}
    try:
        import torch._C._distributed_c10d as c10d

        raw = c10d._dump_nccl_trace_json(True, False)
        trace = json.loads(raw.decode("utf-8"))
        entries = trace.get("entries", [])
        operations = [
            {
                "process_group": entry.get("process_group"),
                "collective_seq_id": entry.get("collective_seq_id"),
                "p2p_seq_id": entry.get("p2p_seq_id"),
                "is_p2p": entry.get("is_p2p"),
                "op_id": entry.get("op_id"),
                "profiling_name": entry.get("profiling_name"),
                "input_sizes": entry.get("input_sizes"),
                "input_dtypes": entry.get("input_dtypes"),
                "output_sizes": entry.get("output_sizes"),
                "output_dtypes": entry.get("output_dtypes"),
            }
            for entry in entries
        ]
        record_ids = [int(entry["record_id"]) for entry in entries]
        buffer_size = int(os.environ.get("TORCH_NCCL_TRACE_BUFFER_SIZE", "0"))
        contiguous = bool(record_ids) and sorted(record_ids) == list(
            range(min(record_ids), max(record_ids) + 1)
        )
        no_wrap = contiguous and min(record_ids) == 0 and len(record_ids) <= buffer_size
        return {
            "required": True,
            "complete": bool(operations) and no_wrap,
            "schema_version": trace.get("version"),
            "trace_buffer_size": buffer_size,
            "entry_count": len(operations),
            "first_record_id": min(record_ids) if record_ids else None,
            "last_record_id": max(record_ids) if record_ids else None,
            "no_circular_buffer_wrap": no_wrap,
            "operation_sequence_sha256": _digest(operations),
            # The digest, count, and record-id bounds are sufficient to compare
            # collective order across ranks.  Keeping the complete sequence in
            # the gathered post-train object grows to hundreds of megabytes on
            # long runs and adds a large, evidence-only NCCL transfer after the
            # final optimizer step.
            "operation_sequence_included": False,
            "error": None,
        }
    except BaseException as exc:
        return {
            "required": True,
            "complete": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_rank_files(
    output_dir: Path,
    phase: str,
    summaries: list[dict[str, Any]],
) -> None:
    evidence_dir = output_dir / "distributed_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        path = evidence_dir / f"rank_{summary['rank']}_{phase}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    directory_fd = os.open(evidence_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _gather_verify_write(
    output_dir: Path,
    phase: str,
    local_summary: dict[str, Any],
    equal_fields: list[str],
) -> None:
    if dist.is_initialized():
        record_collective(f"distributed_evidence.{phase}", "all_gather_object")
        summaries: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(summaries, local_summary)
        gathered = [item for item in summaries if item is not None]
    else:
        gathered = [local_summary]

    errors = []
    if len(gathered) != local_summary["world_size"]:
        errors.append(
            f"gathered {len(gathered)} summaries for world_size={local_summary['world_size']}"
        )
    if sorted(item["rank"] for item in gathered) != list(range(len(gathered))):
        errors.append("rank identities are incomplete or duplicated")
    for field in equal_fields:
        values = {_canonical_json(item.get(field)) for item in gathered}
        if len(values) != 1:
            errors.append(f"rank mismatch: {field}")

    run_on_rank0(
        _write_rank_files,
        output_dir,
        phase,
        gathered,
        label=f"distributed_evidence.{phase}.write",
    )
    if errors:
        raise RuntimeError("distributed conformance failure: " + "; ".join(errors))


def record_pretrain(
    output_dir: Path,
    config: Any,
    model: torch.nn.Module,
    train_dataset: Any,
) -> None:
    if not evidence_enabled():
        return
    declared_stage = _cache_declared_zero_stage(config)
    summary = {
        "phase": "pretrain",
        **_rank_identity(),
        "resolved_config": _jsonable(config),
        "resolved_config_sha256": _digest(config),
        **_parameter_summary(model),
        "dataset_contract": _dataset_contract_summary(config, train_dataset),
        "declared_deepspeed_zero_stage": declared_stage,
        "application_collective_trace": get_collective_trace(),
    }
    _gather_verify_write(
        output_dir,
        "pretrain",
        summary,
        [
            "resolved_config_sha256",
            "resolved_config",
            "parameters_sha256",
            "trainable_parameter_count",
            "trainable_key_set_sha256",
            "trainable_numel",
            "dataset_contract",
            "declared_deepspeed_zero_stage",
            "application_collective_trace",
        ],
    )


def record_posttrain(
    output_dir: Path,
    config: Any,
    model: torch.nn.Module,
    trainer: Any,
    monitor: FiniteTrainingMonitor | None,
) -> None:
    if not evidence_enabled():
        return
    declared_stage = _declared_zero_stage(config)
    actual_stage, engine_class = _actual_zero_stage(trainer)
    training_monitor = monitor.summary(config.training.max_steps) if monitor else None
    nccl_trace = _nccl_flight_recorder_summary()
    summary = {
        "phase": "posttrain",
        **_rank_identity(),
        "resolved_config_sha256": _digest(config),
        **_parameter_summary(model),
        "declared_deepspeed_zero_stage": declared_stage,
        "actual_deepspeed_zero_stage": actual_stage,
        "deepspeed_engine_class": engine_class,
        "trainer_global_step": int(trainer.state.global_step),
        "training_monitor": training_monitor,
        "dataset_sampling": _dataset_sampling_summary(config, trainer),
        "application_collective_trace": get_collective_trace(),
        "nccl_flight_recorder": nccl_trace,
    }
    _gather_verify_write(
        output_dir,
        "posttrain",
        summary,
        [
            "resolved_config_sha256",
            "parameters_sha256",
            "trainable_parameter_count",
            "trainable_key_set_sha256",
            "trainable_numel",
            "declared_deepspeed_zero_stage",
            "actual_deepspeed_zero_stage",
            "deepspeed_engine_class",
            "trainer_global_step",
            "application_collective_trace",
            "nccl_flight_recorder",
        ],
    )
    if actual_stage != declared_stage:
        raise RuntimeError(
            f"DeepSpeed ZeRO stage mismatch: declared={declared_stage}, actual={actual_stage}"
        )
    if trainer.state.global_step != config.training.max_steps:
        raise RuntimeError(
            f"optimizer-step mismatch: expected={config.training.max_steps}, "
            f"actual={trainer.state.global_step}"
        )
    if not training_monitor or not training_monitor["complete"]:
        raise RuntimeError("loss/gradient/optimizer-step finite monitor did not close")
    if nccl_trace["required"] and not nccl_trace["complete"]:
        raise RuntimeError("NCCL flight-recorder sequence is absent or incomplete")
