#!/usr/bin/env python3
"""Independent online-state scaling runner for the frozen Arena descriptors.

The correctness route is self-contained and model-specific: it compares each
online recurrence to an independent CPU FP64 equation.  It does not import the
closed functional-streaming-controls runner, call its result/gate helpers, or
invoke the opaque full-sequence Transformer endpoint.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator

import functional_mamba_transition as input_runtime_provider
from gr00t.experimental.arena_codec_streaming_backbone import (
    CODEC_FIELDS,
    GRID_HEIGHT,
    GRID_WIDTH,
    LAYERS,
    SEED,
    WIDTH,
    ArenaCodecSpatialTokenizer,
    trainable_parameter_count,
)
from gr00t.experimental.arena_functional_mamba_transition import FunctionalMamba2Backbone
from gr00t.experimental.arena_functional_streaming_controls import (
    PersistentGRUBackbone,
    StreamingCurrentDescriptorBackbone,
    StreamingKVTransformerBackbone,
    state_is_finite,
    state_tensors,
    states_equal,
    storage_identity,
    transformer_kv_expected_bytes,
    unique_storage_state_bytes,
)
from gr00t.experimental.arena_online_state_scaling import (
    KV_CAPACITY,
    ONLINE_LENGTHS,
    PreallocatedKVState,
    PreallocatedKVTransformerBackbone,
    functional_mamba_fp64_oracle,
    persistent_gru_fp64_oracle,
    preallocated_capacity_expected_bytes,
    preallocated_states_equal,
    streaming_transformer_fp64_oracle,
)
import numpy as np
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "g1-arena-online-state-scaling-r3"
BASE_SOURCE_COMMIT = "748404e7749af1f7cca465fb74374f5ca0640235"
ARTIFACT_ROOT = Path("/home/sl/research/icra27-artifacts/g1-arena-online-state-scaling-r3")
PLAN_PATH = ARTIFACT_ROOT / "plan-r3/PLAN.json"
CORRECTNESS_RUN_ID = "correctness-gpu0-r3"
COST_RUN_IDS = {label: f"cost-gpu0-r3{label.lower()}" for label in "ABCDE"}
AGGREGATE_RUN_ID = "aggregate-r3"

CONDITIONS = (
    "functional_mamba",
    "persistent_gru",
    "transformer_functional_cat",
    "transformer_preallocated",
    "current_descriptor",
)
CONDITION_ORDERS = {
    label: CONDITIONS[offset:] + CONDITIONS[:offset] for offset, label in enumerate(COST_RUN_IDS)
}
LENGTHS = ONLINE_LENGTHS
LENGTH_ORDERS = {
    label: LENGTHS[offset % len(LENGTHS) :] + LENGTHS[: offset % len(LENGTHS)]
    for offset, label in enumerate(COST_RUN_IDS)
}
PROCESS_SEEDS = {label: SEED for label in COST_RUN_IDS}
SCOPES = ("end_to_end", "temporal_only")
WARMUP_SAMPLES = 20
MEASURED_SAMPLES = 100
FIXED_CHUNK_SIZE = 73
GRADIENT_LENGTH = 128
MAX_ORACLE_ABS = 1e-4
MAX_TRANSFORMER_CONTROL_ABS = 1e-5
MAX_DIRECTIONAL_GRAD_RELATIVE_ERROR = 0.05
MAX_CORRECTNESS_WALL_SECONDS = 20 * 60
MAX_COST_WALL_SECONDS = 2 * 60 * 60
MAX_AGGREGATE_WALL_SECONDS = 20 * 60
MAX_RESULT_BYTES = 1 << 30
MAX_ARTIFACT_BYTES = 4 << 30

INPUT_PATH = input_runtime_provider.INPUT_PATH
INPUT_BYTES = input_runtime_provider.INPUT_BYTES
INPUT_SHA256 = input_runtime_provider.INPUT_SHA256
INPUT_FIELD = input_runtime_provider.INPUT_FIELD
INPUT_SHAPE = input_runtime_provider.INPUT_SHAPE
EXPECTED_ARCHIVE_FIELDS = input_runtime_provider.EXPECTED_ARCHIVE_FIELDS
FROZEN_DEPENDENCY = copy.deepcopy(input_runtime_provider.FROZEN_DEPENDENCY)
FROZEN_RUNTIME = copy.deepcopy(input_runtime_provider.FROZEN_RUNTIME)
FROZEN_RUNTIME["cuda_visible_devices"] = "0"
FROZEN_RUNTIME["pythonpath"] = f"{input_runtime_provider.OVERLAY_ROOT}:{REPO}"

FUNCTIONAL_AGGREGATE = {
    "path": (
        "/home/sl/research/icra27-artifacts/g1-arena-functional-mamba-transition-r1/"
        "aggregate-r2/RESULT.json"
    ),
    "bytes": 9_249,
    "sha256": "1cc3cebbf621bca0619eaef48f687c91f2bdd8dd39ac377c048d4c8a3813d653",
}
FUNCTIONAL_RECHECK = {
    "path": (
        "/home/sl/research/icra27-artifacts/g1-arena-functional-mamba-transition-r1/"
        "aggregate-r2/RESULT_RECHECK.json"
    ),
    "bytes": 2_134,
    "sha256": "6581ff40fb90f4d795b40a5490db0438c48a9621ce99faa604d409711f61387d",
}

EXPECTED_STATE_BYTES = {
    "functional_mamba": {length: 282_624 for length in LENGTHS},
    "persistent_gru": {length: 2_048 for length in LENGTHS},
    "transformer_functional_cat": {
        128: 524_288,
        256: 1_048_576,
        512: 2_097_152,
        925: 3_788_800,
    },
    "transformer_preallocated": {
        128: 524_288,
        256: 1_048_576,
        512: 2_097_152,
        925: 3_788_800,
    },
    "current_descriptor": {length: 0 for length in LENGTHS},
}

SOURCE_IDENTITY_PATHS = (
    "gr00t/experimental/arena_codec_streaming_backbone.py",
    "gr00t/experimental/arena_functional_mamba_transition.py",
    "gr00t/experimental/arena_functional_streaming_controls.py",
    "gr00t/experimental/arena_online_state_scaling.py",
    "examples/G1_ARENA_N17/functional_mamba_transition.py",
    "examples/G1_ARENA_N17/online_state_scaling.py",
    "tests/gr00t/experimental/test_arena_online_state_scaling.py",
    "tests/examples/test_g1_arena_online_state_scaling.py",
)
NATIVE_VLA_MODULE_PREFIXES = input_runtime_provider.NATIVE_VLA_MODULE_PREFIXES

LOW_LEVEL_PROVENANCE = {
    "input_loader": "examples/G1_ARENA_N17/functional_mamba_transition.py:load_codec_input",
    "dependency_verifier": (
        "examples/G1_ARENA_N17/functional_mamba_transition.py:verify_dependencies"
    ),
    "runtime_verifier": "examples/G1_ARENA_N17/functional_mamba_transition.py:verify_environment",
    "measurement": "examples/G1_ARENA_N17/online_state_scaling.py:new_independent_implementation",
}
IMPLEMENTATION_STATUS = {
    "preallocated_control": True,
    "independent_correctness": True,
    "cost_cli_contract": True,
    "cost_measurement": True,
    "aggregate_decision": True,
    "formal_gpu_run_allowed": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def state_dict_tensor_sha256(module: nn.Module) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in sorted(module.state_dict().items())}


def _assert_closed_route_source(source: str, *, filename: str) -> dict[str, Any]:
    """Audit one source unit without consulting process-global import state."""

    tree = ast.parse(source, filename=filename)
    forbidden = {"run_cost", "evaluate_cost_gates", "verify_passed_result"}
    observed_calls = []
    observed_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name in forbidden:
                observed_calls.append({"name": name, "line": node.lineno})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {
                    "functional_streaming_controls",
                    "examples.G1_ARENA_N17.functional_streaming_controls",
                }:
                    observed_imports.append({"module": alias.name, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports_closed_runner = module in {
                "functional_streaming_controls",
                "examples.G1_ARENA_N17.functional_streaming_controls",
            } or (
                module == "examples.G1_ARENA_N17"
                and any(alias.name == "functional_streaming_controls" for alias in node.names)
            )
            if imports_closed_runner:
                observed_imports.append({"module": module, "line": node.lineno})
    if observed_calls:
        raise RuntimeError(f"closed evidence helper call detected: {observed_calls}")
    if observed_imports:
        raise RuntimeError(
            f"closed functional-streaming-controls runner import: {observed_imports}"
        )
    return {
        "closed_helper_calls": [],
        "closed_runner_imported": False,
        "opaque_full_transformer_calls": 0,
    }


def assert_closed_routes_absent() -> dict[str, Any]:
    """Fail if this runner imports or calls a closed evidence route."""

    return _assert_closed_route_source(
        Path(__file__).read_text(encoding="utf-8"),
        filename=__file__,
    )


def max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = (first.detach().cpu().double() - second.detach().cpu().double()).abs()
    return (
        float(difference.max().item()) if bool(torch.isfinite(difference).all()) else float("inf")
    )


def native_vla_modules_loaded() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in NATIVE_VLA_MODULE_PREFIXES
        )
    )


@contextmanager
def optimizer_construction_guard() -> Iterator[dict[str, Any]]:
    optimizer = torch.optim.Optimizer
    original = optimizer.__init__
    evidence: dict[str, Any] = {"constructor_calls": [], "steps_observed": None}

    def blocked(instance, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        evidence["constructor_calls"].append(type(instance).__qualname__)
        raise RuntimeError("optimizer construction is forbidden in fixed-weight scaling")

    optimizer.__init__ = blocked
    try:
        yield evidence
    finally:
        optimizer.__init__ = original
        evidence["steps_observed"] = 0 if not evidence["constructor_calls"] else None


def build_condition(name: str, *, seed: int = SEED) -> nn.Module:
    if name == "functional_mamba":
        return FunctionalMamba2Backbone(seed=seed)
    if name == "persistent_gru":
        return PersistentGRUBackbone(seed=seed)
    if name == "transformer_functional_cat":
        return StreamingKVTransformerBackbone(seed=seed)
    if name == "transformer_preallocated":
        return PreallocatedKVTransformerBackbone(seed=seed)
    if name == "current_descriptor":
        return StreamingCurrentDescriptorBackbone()
    raise ValueError(f"unknown online condition: {name}")


def model_identity() -> dict[str, Any]:
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval()
    modules = {name: build_condition(name).eval() for name in CONDITIONS}
    counts = {name: trainable_parameter_count(module) for name, module in modules.items()}
    hashes = {name: state_dict_sha256(module) for name, module in modules.items()}
    tensor_hashes = {name: state_dict_tensor_sha256(module) for name, module in modules.items()}
    if hashes["transformer_functional_cat"] != hashes["transformer_preallocated"]:
        raise ValueError("functional-cat and preallocated Transformer weights differ")
    return {
        "tokenizer": {
            "parameters": trainable_parameter_count(tokenizer),
            "state_dict_sha256": state_dict_sha256(tokenizer),
        },
        "conditions": {
            name: {
                "parameters": counts[name],
                "state_dict_sha256": hashes[name],
                "state_dict_tensor_sha256": tensor_hashes[name],
            }
            for name in CONDITIONS
        },
        "transformer_weight_hash_shared": True,
        "transformer_per_tensor_hash_shared": (
            tensor_hashes["transformer_functional_cat"] == tensor_hashes["transformer_preallocated"]
        ),
        "analytic_compute": {
            "variables": {"d": WIDTH, "layers": LAYERS, "heads": 8, "ffn": 320},
            "functional_mamba": {
                "step_mac_expression_per_layer": (
                    "d*1160 + 640*4 + 8*64*64 recurrent update + 512*d output projection"
                ),
                "state": "fixed conv and SSM recurrence",
            },
            "persistent_gru": {
                "step_mac_expression_per_layer": "3*d*d input + 3*d*d hidden",
                "state": "fixed d-dimensional hidden state",
            },
            "transformer_functional_cat": (
                "per-layer step MACs: 4*d*d projections + 2*d*ffn + 2*T*d attention; "
                "live KV grows with T and each step cats"
            ),
            "transformer_preallocated": (
                "same per-layer 4*d*d + 2*d*ffn + 2*T*d MACs; fixed T925 buffers and "
                "live-prefix attention"
            ),
            "current_descriptor": {"step_mac_expression": "0", "state": "0 bytes"},
            "nonmatched_difference": (
                "parameter counts are within the frozen architecture band, but recurrent SSM, GRU, "
                "attention and FFN analytic FLOPs cannot be exactly matched without changing the models"
            ),
        },
    }


def frozen_protocol() -> dict[str, Any]:
    return {
        "seed": SEED,
        "dtype": "float32",
        "batch_size": 1,
        "grid": [GRID_HEIGHT, GRID_WIDTH, CODEC_FIELDS],
        "width": WIDTH,
        "layers": LAYERS,
        "lengths": list(LENGTHS),
        "conditions": list(CONDITIONS),
        "condition_orders": {key: list(value) for key, value in CONDITION_ORDERS.items()},
        "length_orders": {key: list(value) for key, value in LENGTH_ORDERS.items()},
        "process_seeds": PROCESS_SEEDS,
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "scopes": list(SCOPES),
        "fixed_chunk_size": FIXED_CHUNK_SIZE,
        "gradient_length": GRADIENT_LENGTH,
        "transformer_capacity": KV_CAPACITY,
        "expected_state_bytes": EXPECTED_STATE_BYTES,
        "preallocated_semantics": {
            "mutable_inference_only": True,
            "write": "current_live_index_only",
            "read": "valid_prefix_only",
            "unwritten": "finite_sentinel_preserved",
            "return_alias": "same_state_object_and_buffer_storage",
            "reset": "in_place_sentinel_fill_and_live_index_zero",
        },
        "cost_measurement": {
            "fresh_python_processes": 5,
            "condition_order": "five_by_five_cyclic_latin",
            "not_concurrent": True,
            "warmup_samples": WARMUP_SAMPLES,
            "measured_samples": MEASURED_SAMPLES,
            "cuda_synchronize_each_frame": True,
            "raw_per_frame_shape": "[100,T]",
            "complete_prefix_cumulative": (
                "sum of the synchronized per-frame host/CUDA samples within each replicate"
            ),
            "reported_latency": [
                "per_frame_flat_p50_p95",
                "per_position_p50_p95",
                "last_frame_p50_p95",
                "complete_prefix_cumulative_p50_p95",
            ],
            "next_step": {
                "prefill": "T_minus_1",
                "allocator_peak_reset_after_prefill": True,
                "preallocated_rollback_outside_timing": True,
                "independent_profiler_runs": 1,
            },
            "allocator": [
                "baseline_allocated_bytes",
                "baseline_reserved_bytes",
                "absolute_peak_allocated_bytes",
                "absolute_peak_reserved_bytes",
                "delta_peak_allocated_bytes",
                "delta_peak_reserved_bytes",
            ],
        },
        "independent_oracles": {
            "functional_mamba": "manual_CPU_FP64_conv_SSM_gated_RMS_recurrence",
            "persistent_gru": "manual_CPU_FP64_reset_after_GRU_recurrence",
            "transformer": "manual_CPU_FP64_matmul_softmax_causal_recurrence",
            "current_descriptor": "identity",
        },
        "excluded": {
            "opaque_fused_full_transformer": True,
            "closed_controls_runner": True,
            "closed_cost_and_gate_helpers": True,
            "vla_vlm_dit_action_head_optimizer_checkpoint": True,
        },
    }


def resolved_plan(source_commit: str, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "source": {
            "root": str(REPO),
            "commit": source_commit,
            "base_commit": BASE_SOURCE_COMMIT,
        },
        "input": {
            "path": str(INPUT_PATH),
            "bytes": INPUT_BYTES,
            "sha256": INPUT_SHA256,
            "field": INPUT_FIELD,
            "shape": list(INPUT_SHAPE),
        },
        "dependencies": copy.deepcopy(FROZEN_DEPENDENCY),
        "runtime": copy.deepcopy(FROZEN_RUNTIME),
        "qualified_prior_evidence": {
            "functional_aggregate": copy.deepcopy(FUNCTIONAL_AGGREGATE),
            "functional_recheck": copy.deepcopy(FUNCTIONAL_RECHECK),
        },
        "closed_evidence_excluded": {
            "functional_streaming_controls": True,
            "transformer_parity_diagnosis": True,
        },
        "low_level_provenance": copy.deepcopy(LOW_LEVEL_PROVENANCE),
        "implementation_status": copy.deepcopy(IMPLEMENTATION_STATUS),
        "models": model_identity() if identity is None else copy.deepcopy(identity),
        "protocol": frozen_protocol(),
        "thresholds": {
            "oracle_max_abs": MAX_ORACLE_ABS,
            "transformer_control_max_abs": MAX_TRANSFORMER_CONTROL_ABS,
            "mamba_t925_end_to_end_p95_ms": 20.0,
            "mamba_state_fraction_of_preallocated_live_kv": 0.10,
            "mamba_preallocated_next_step_ratio": 0.80,
            "mamba_preallocated_prefix_cumulative_ratio": 0.80,
            "mamba_t925_t128_next_step_ratio": 1.15,
            "preallocated_functional_cat_allocator_delta_ratio": 0.50,
            "process_direction_count": "at_least_4_of_5",
            "correctness_wall_seconds": MAX_CORRECTNESS_WALL_SECONDS,
            "cost_wall_seconds": MAX_COST_WALL_SECONDS,
            "aggregate_wall_seconds": MAX_AGGREGATE_WALL_SECONDS,
            "single_result_bytes": MAX_RESULT_BYTES,
            "artifact_root_bytes": MAX_ARTIFACT_BYTES,
        },
        "outputs": {
            "correctness": str(ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json"),
            "cost": {
                label: str(ARTIFACT_ROOT / run_id / "RESULT.json")
                for label, run_id in COST_RUN_IDS.items()
            },
            "aggregate": str(ARTIFACT_ROOT / AGGREGATE_RUN_ID / "RESULT.json"),
        },
    }


def _state_equal(first: Any, second: Any) -> bool:
    if isinstance(first, PreallocatedKVState) and isinstance(second, PreallocatedKVState):
        return preallocated_states_equal(first, second)
    return states_equal(first, second)


def _state_clone(state: Any) -> Any:
    return state.clone()


def _state_finite(state: Any) -> bool:
    return state.is_finite() if isinstance(state, PreallocatedKVState) else state_is_finite(state)


def _state_logical_bytes(state: Any) -> int:
    return (
        state.logical_nbytes()
        if isinstance(state, PreallocatedKVState)
        else unique_storage_state_bytes(state)
    )


def _state_capacity_bytes(state: Any) -> int:
    return (
        state.capacity_nbytes()
        if isinstance(state, PreallocatedKVState)
        else _state_logical_bytes(state)
    )


def _state_storage_ids(state: Any) -> set[tuple[str, int, int]]:
    tensors = state.tensors() if isinstance(state, PreallocatedKVState) else state_tensors(state)
    return {storage_identity(tensor) for tensor in tensors}


def new_state(name: str, module: nn.Module, batch: int) -> Any:
    if name == "functional_mamba":
        return module.new_functional_state(batch)
    if name == "persistent_gru":
        return module.new_persistent_state(batch)
    if name == "transformer_functional_cat":
        return module.new_kv_state(batch)
    if name == "transformer_preallocated":
        return module.new_preallocated_state(batch)
    if name == "current_descriptor":
        return module.new_state(batch)
    raise ValueError(f"unknown online condition: {name}")


def online_step(
    name: str, module: nn.Module, descriptor: torch.Tensor, state: Any
) -> tuple[torch.Tensor, Any]:
    if name == "functional_mamba":
        return module.functional_step(descriptor, state)
    if name == "persistent_gru":
        return module.persistent_step(descriptor, state)
    if name == "transformer_functional_cat":
        return module.streaming_step(descriptor, state)
    if name == "transformer_preallocated":
        return module.preallocated_step(descriptor, state)
    if name == "current_descriptor":
        return module.streaming_step(descriptor, state)
    raise ValueError(f"unknown online condition: {name}")


def online_stream(
    name: str, module: nn.Module, descriptors: torch.Tensor, state: Any | None = None
) -> tuple[torch.Tensor, Any]:
    current = new_state(name, module, int(descriptors.shape[0])) if state is None else state
    outputs = []
    for descriptor in descriptors.unbind(dim=1):
        output, current = online_step(name, module, descriptor, current)
        outputs.append(output)
    return torch.stack(outputs, dim=1), current


def differentiable_stream(name: str, module: nn.Module, descriptors: torch.Tensor) -> torch.Tensor:
    """Return the differentiable training-path output for gradient diagnosis."""

    if name == "functional_mamba":
        return module.functional_stream(descriptors)[0]
    if name == "persistent_gru":
        return module.persistent_stream(descriptors)[0]
    if name == "transformer_functional_cat":
        return module.streaming_stream(descriptors)[0]
    raise ValueError(f"condition has no differentiable online path: {name}")


@contextmanager
def directional_gradient_training_scope(name: str, module: nn.Module) -> Iterator[None]:
    """Temporarily enable the frozen two-GRU probe path for cuDNN backward."""

    if name != "persistent_gru":
        yield
        return

    submodules = tuple(module.modules())
    training_flags = tuple(submodule.training for submodule in submodules)
    if any(training_flags):
        raise ValueError("persistent GRU gradient probe requires an all-eval module tree")
    grus = tuple(submodule for submodule in submodules if isinstance(submodule, nn.GRU))
    if len(grus) != 2:
        raise ValueError("persistent GRU gradient probe requires exactly two nn.GRU modules")
    if any(gru.num_layers != 1 or float(gru.dropout) != 0.0 for gru in grus):
        raise ValueError(
            "persistent GRU gradient probe requires one-layer nn.GRU modules with zero dropout"
        )

    try:
        module.train(True)
        yield
    finally:
        for submodule, training in zip(submodules, training_flags, strict=True):
            submodule.training = training


def directional_gradient_probe(
    name: str,
    module: nn.Module,
    descriptors: torch.Tensor,
    *,
    seed: int = SEED,
    epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Check one T-length directional derivative against central difference.

    Reverse-mode autograd supplies the scalar directional derivative.  Two
    fresh functional online scans supply the central-difference directional
    JVP.  The mutable preallocated control is intentionally excluded.
    """

    if name not in (
        "functional_mamba",
        "persistent_gru",
        "transformer_functional_cat",
    ):
        raise ValueError("directional gradient probe requires a differentiable condition")
    generator = torch.Generator(device=descriptors.device).manual_seed(seed)
    direction = torch.randn(
        descriptors.shape,
        generator=generator,
        device=descriptors.device,
        dtype=descriptors.dtype,
    )
    direction = direction / direction.norm().clamp_min(torch.finfo(descriptors.dtype).eps)
    cotangent = torch.randn(
        descriptors.shape,
        generator=generator,
        device=descriptors.device,
        dtype=descriptors.dtype,
    )
    cotangent = cotangent / cotangent.norm().clamp_min(torch.finfo(descriptors.dtype).eps)

    with directional_gradient_training_scope(name, module):
        value = descriptors.detach().clone().requires_grad_(True)
        output = differentiable_stream(name, module, value)
        scalar = torch.sum(output * cotangent)
        gradient = torch.autograd.grad(scalar, value, create_graph=False)[0]
        reverse_directional = torch.sum(gradient * direction)
        with torch.no_grad():
            plus = differentiable_stream(name, module, descriptors + epsilon * direction)
            minus = differentiable_stream(name, module, descriptors - epsilon * direction)
            finite_difference = torch.sum(((plus - minus) / (2.0 * epsilon)) * cotangent)
    absolute_error = torch.abs(reverse_directional - finite_difference)
    scale = torch.maximum(
        torch.maximum(torch.abs(reverse_directional), torch.abs(finite_difference)),
        torch.tensor(1e-8, device=descriptors.device, dtype=descriptors.dtype),
    )
    relative_error = absolute_error / scale
    finite = bool(
        torch.isfinite(gradient).all()
        and torch.isfinite(reverse_directional)
        and torch.isfinite(finite_difference)
        and torch.isfinite(relative_error)
    )
    nonzero = bool(gradient.abs().max() > 0 and torch.abs(finite_difference) > 0)
    return {
        "length": int(descriptors.shape[1]),
        "epsilon": epsilon,
        "reverse_directional": float(reverse_directional.detach().cpu().item()),
        "finite_difference_directional_jvp": float(finite_difference.detach().cpu().item()),
        "absolute_error": float(absolute_error.detach().cpu().item()),
        "relative_error": float(relative_error.detach().cpu().item()),
        "max_relative_error": MAX_DIRECTIONAL_GRAD_RELATIVE_ERROR,
        "gradient_finite": finite,
        "gradient_nonzero": nonzero,
        "passed": finite
        and nonzero
        and float(relative_error.detach().cpu().item()) <= MAX_DIRECTIONAL_GRAD_RELATIVE_ERROR,
    }


@torch.inference_mode()
def recurrence_invariants(
    name: str, module: nn.Module, descriptors: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor, Any]:
    length = int(descriptors.shape[1])
    continuous, continuous_state = online_stream(name, module, descriptors)
    pieces = []
    chunk_state = None
    for start in range(0, length, FIXED_CHUNK_SIZE):
        piece, chunk_state = online_stream(
            name,
            module,
            descriptors[:, start : start + FIXED_CHUNK_SIZE],
            chunk_state,
        )
        pieces.append(piece)
    chunked = torch.cat(pieces, dim=1)
    repeat, repeat_state = online_stream(name, module, descriptors)
    prefix = max(1, length // 2)
    edited = descriptors.clone()
    edited[:, prefix:] += 0.125
    edited_output, _ = online_stream(name, module, edited)

    mutable_expected = name == "transformer_preallocated"
    reset_state = _state_clone(continuous_state)
    reset_returned = reset_state.reset()
    reset_returned_observations = reset_returned.observations
    reset_returned_is_input = reset_returned is reset_state
    reset_index_zero = reset_returned_observations == 0
    reset_alias_contract = (
        reset_returned_is_input and reset_state.observations == 0
        if mutable_expected
        else not reset_returned_is_input and reset_state.observations == length
    )
    reset_output, reset_final = online_stream(name, module, descriptors, reset_returned)
    reset_output_exact = torch.equal(continuous, reset_output)
    reset_terminal_state_exact = _state_equal(continuous_state, reset_final)

    initial = new_state(name, module, int(descriptors.shape[0]))
    _, one = online_step(name, module, descriptors[:, 0], initial)
    probe_input = _state_clone(one)
    before = _state_clone(probe_input)
    ids_before = _state_storage_ids(probe_input)
    _, probe_output = online_step(name, module, descriptors[:, min(1, length - 1)], probe_input)
    alias_contract = (
        probe_output is probe_input and _state_storage_ids(probe_output) == ids_before
        if mutable_expected
        else _state_equal(probe_input, before)
        and not (_state_storage_ids(probe_input) & _state_storage_ids(probe_output))
    )
    logical = _state_logical_bytes(continuous_state)
    capacity = _state_capacity_bytes(continuous_state)
    expected_logical = EXPECTED_STATE_BYTES[name][length]
    return (
        {
            "output_shape": list(continuous.shape),
            "output_finite": bool(torch.isfinite(continuous).all()),
            "state_finite": _state_finite(continuous_state),
            "final_observations": continuous_state.observations,
            "continuous_chunk_exact": torch.equal(continuous, chunked)
            and _state_equal(continuous_state, chunk_state),
            "repeat_exact": torch.equal(continuous, repeat)
            and _state_equal(continuous_state, repeat_state),
            "reset_returned_observations": reset_returned_observations,
            "reset_returned_is_input": reset_returned_is_input,
            "reset_alias_contract": reset_alias_contract,
            "episode_reset_output_exact": reset_output_exact,
            "episode_reset_terminal_state_exact": reset_terminal_state_exact,
            "episode_reset_exact": reset_index_zero
            and reset_alias_contract
            and reset_output_exact
            and reset_terminal_state_exact,
            "suffix_change_prefix_exact": torch.equal(
                continuous[:, :prefix], edited_output[:, :prefix]
            ),
            "state_alias_contract": alias_contract,
            "logical_state_bytes": logical,
            "capacity_state_bytes": capacity,
            "expected_logical_state_bytes": expected_logical,
            "state_formula_exact": logical == expected_logical,
            "unwritten_sentinel_preserved": (
                continuous_state.unwritten_is_sentinel()
                if isinstance(continuous_state, PreallocatedKVState)
                else True
            ),
        },
        continuous,
        continuous_state,
    )


def _state_oracle_max_abs(name: str, target_state: Any, oracle_state: Any) -> float:
    if name == "functional_mamba":
        target = tuple(tensor for pair in target_state.layers for tensor in pair)
        expected = tuple(tensor for pair in oracle_state for tensor in pair)
    elif name == "persistent_gru":
        target = target_state.hidden
        expected = oracle_state
    elif name == "transformer_functional_cat":
        target = tuple(tensor for pair in target_state.layers for tensor in pair)
        expected = tuple(tensor for pair in oracle_state for tensor in pair)
    elif name == "transformer_preallocated":
        stop = target_state.live_index
        target = tuple(tensor[:, :, :stop] for tensor in target_state.tensors())
        expected = tuple(tensor for pair in oracle_state for tensor in pair)
    else:
        return 0.0
    return max(max_abs(left, right) for left, right in zip(target, expected, strict=True))


def oracle_result(
    name: str,
    module: nn.Module,
    descriptors: torch.Tensor,
    output: torch.Tensor,
    state: Any,
) -> dict[str, Any]:
    if name == "functional_mamba":
        oracle_output, oracle_state = functional_mamba_fp64_oracle(module, descriptors)
    elif name == "persistent_gru":
        oracle_output, oracle_state = persistent_gru_fp64_oracle(module, descriptors)
    elif name in ("transformer_functional_cat", "transformer_preallocated"):
        oracle_output, oracle_state = streaming_transformer_fp64_oracle(module, descriptors)
    elif name == "current_descriptor":
        oracle_output, oracle_state = descriptors.detach().cpu().double(), None
    else:
        raise ValueError(f"unknown online condition: {name}")
    return {
        "output_max_abs": max_abs(output, oracle_output),
        "state_max_abs": _state_oracle_max_abs(name, state, oracle_state),
        "oracle_output_finite": bool(torch.isfinite(oracle_output).all()),
        "oracle_state_finite": oracle_state is None
        or all(
            bool(torch.isfinite(tensor).all())
            for item in oracle_state
            for tensor in (item if isinstance(item, tuple) else (item,))
        ),
    }


def correctness_probe_from_descriptors(
    descriptors: torch.Tensor,
    *,
    lengths: tuple[int, ...] = LENGTHS,
) -> dict[str, Any]:
    if tuple(lengths) != LENGTHS:
        if any(length < 2 or length > descriptors.shape[1] for length in lengths):
            raise ValueError("test lengths exceed descriptors")
    modules = {
        name: build_condition(name).eval().to(device=descriptors.device, dtype=torch.float32)
        for name in CONDITIONS
    }
    condition_identity = {
        name: {
            "parameters": trainable_parameter_count(module),
            "state_dict_sha256": state_dict_sha256(module),
            "state_dict_tensor_sha256": state_dict_tensor_sha256(module),
        }
        for name, module in modules.items()
    }
    transformer_hashes = {
        name: state_dict_sha256(modules[name])
        for name in ("transformer_functional_cat", "transformer_preallocated")
    }
    transformer_tensor_hashes = {
        name: state_dict_tensor_sha256(modules[name])
        for name in ("transformer_functional_cat", "transformer_preallocated")
    }
    by_length: dict[str, Any] = {}
    for length in lengths:
        prefix = descriptors[:, :length]
        condition_results = {}
        target_cache = {}
        for name, module in modules.items():
            with torch.inference_mode():
                invariants, output, state = recurrence_invariants(name, module, prefix)
                oracle = oracle_result(name, module, prefix, output, state)
            condition_results[name] = {"invariants": invariants, "oracle": oracle}
            target_cache[name] = (output, state)
        cat_output, cat_state = target_cache["transformer_functional_cat"]
        pre_output, pre_state = target_cache["transformer_preallocated"]
        condition_results["transformer_shared_control"] = {
            "output_max_abs": max_abs(cat_output, pre_output),
            "state_max_abs": _state_oracle_max_abs(
                "transformer_preallocated", pre_state, cat_state.layers
            ),
        }
        by_length[str(length)] = condition_results
    gradient_prefix = descriptors[:, :GRADIENT_LENGTH]
    gradient_probe = {
        name: directional_gradient_probe(name, modules[name], gradient_prefix, seed=SEED)
        for name in (
            "functional_mamba",
            "persistent_gru",
            "transformer_functional_cat",
        )
    }
    return {
        "lengths": by_length,
        "transformer_state_dict_sha256": transformer_hashes,
        "transformer_state_dict_tensor_sha256": transformer_tensor_hashes,
        "transformer_weights_shared": len(set(transformer_hashes.values())) == 1,
        "transformer_per_tensor_hash_shared": (
            transformer_tensor_hashes["transformer_functional_cat"]
            == transformer_tensor_hashes["transformer_preallocated"]
        ),
        "preallocated_capacity_bytes": preallocated_capacity_expected_bytes(),
        "condition_identity": condition_identity,
        "gradient_probe": gradient_probe,
    }


def evaluate_correctness(correctness: dict[str, Any], execution: dict[str, Any]) -> dict[str, bool]:
    length_gates = []
    for length in LENGTHS:
        result = correctness.get("lengths", {}).get(str(length), {})
        conditions_ok = True
        for name in CONDITIONS:
            item = result.get(name, {})
            invariants = item.get("invariants", {})
            oracle = item.get("oracle", {})
            conditions_ok = conditions_ok and all(
                (
                    invariants.get("output_shape") == [1, length, WIDTH],
                    invariants.get("output_finite") is True,
                    invariants.get("state_finite") is True,
                    invariants.get("final_observations") == length,
                    invariants.get("continuous_chunk_exact") is True,
                    invariants.get("repeat_exact") is True,
                    invariants.get("reset_returned_observations") == 0,
                    invariants.get("reset_alias_contract") is True,
                    invariants.get("episode_reset_output_exact") is True,
                    invariants.get("episode_reset_terminal_state_exact") is True,
                    invariants.get("episode_reset_exact") is True,
                    invariants.get("suffix_change_prefix_exact") is True,
                    invariants.get("state_alias_contract") is True,
                    invariants.get("state_formula_exact") is True,
                    invariants.get("unwritten_sentinel_preserved") is True,
                    oracle.get("output_max_abs", float("inf")) <= MAX_ORACLE_ABS,
                    oracle.get("state_max_abs", float("inf")) <= MAX_ORACLE_ABS,
                    oracle.get("oracle_output_finite") is True,
                    oracle.get("oracle_state_finite") is True,
                )
            )
        shared = result.get("transformer_shared_control", {})
        conditions_ok = conditions_ok and (
            shared.get("output_max_abs", float("inf")) <= MAX_TRANSFORMER_CONTROL_ABS
            and shared.get("state_max_abs", float("inf")) <= MAX_TRANSFORMER_CONTROL_ABS
        )
        length_gates.append(conditions_ok)
    return {
        "all_lengths_model_specific_oracles_invariants_state_and_shared_transformer_pass": all(
            length_gates
        ),
        "transformer_parameter_hash_shared": correctness.get("transformer_weights_shared") is True,
        "transformer_per_tensor_hash_shared": correctness.get("transformer_per_tensor_hash_shared")
        is True,
        "preallocated_capacity_bytes_exact": correctness.get("preallocated_capacity_bytes")
        == 3_788_800,
        "t128_directional_jvp_gradcheck": set(correctness.get("gradient_probe", {}))
        == {
            "functional_mamba",
            "persistent_gru",
            "transformer_functional_cat",
        }
        and all(
            item.get("length") == GRADIENT_LENGTH and item.get("passed") is True
            for item in correctness.get("gradient_probe", {}).values()
        ),
        "native_vla_modules_absent": not execution.get("native_vla_modules", {}).get("before")
        and not execution.get("native_vla_modules", {}).get("after"),
        "optimizer_steps_zero": not execution.get("optimizer", {}).get("constructor_calls")
        and execution.get("optimizer", {}).get("steps_observed") == 0,
    }


def percentile(values: Any, q: float, *, axis: int | None = None) -> Any:
    array = np.asarray(values, dtype=np.float64)
    result = np.percentile(array, q, axis=axis, method="linear")
    return float(result) if np.ndim(result) == 0 else result.tolist()


def _latency_summary(host_rows: list[list[float]], cuda_rows: list[list[float]]) -> dict[str, Any]:
    host = np.asarray(host_rows, dtype=np.float64)
    cuda = np.asarray(cuda_rows, dtype=np.float64)
    if host.ndim != 2 or cuda.shape != host.shape or host.shape[0] != MEASURED_SAMPLES:
        raise ValueError("latency raw matrices must both have shape [100,T]")
    if not np.isfinite(host).all() or not np.isfinite(cuda).all():
        raise ValueError("latency raw matrices must be finite")
    if np.any(host <= 0) or np.any(cuda < 0):
        raise ValueError("host latency must be positive and CUDA latency nonnegative")
    host_total = host.sum(axis=1)
    cuda_total = cuda.sum(axis=1)
    return {
        "raw": {
            "host_per_frame_ms": host.tolist(),
            "cuda_per_frame_ms": cuda.tolist(),
        },
        "per_frame_flat": {
            "host_p50_ms": percentile(host.reshape(-1), 50),
            "host_p95_ms": percentile(host.reshape(-1), 95),
            "cuda_p50_ms": percentile(cuda.reshape(-1), 50),
            "cuda_p95_ms": percentile(cuda.reshape(-1), 95),
        },
        "per_position": {
            "host_p50_ms": percentile(host, 50, axis=0),
            "host_p95_ms": percentile(host, 95, axis=0),
            "cuda_p50_ms": percentile(cuda, 50, axis=0),
            "cuda_p95_ms": percentile(cuda, 95, axis=0),
        },
        "last_frame": {
            "host_samples_ms": host[:, -1].tolist(),
            "host_p50_ms": percentile(host[:, -1], 50),
            "host_p95_ms": percentile(host[:, -1], 95),
            "cuda_samples_ms": cuda[:, -1].tolist(),
            "cuda_p50_ms": percentile(cuda[:, -1], 50),
            "cuda_p95_ms": percentile(cuda[:, -1], 95),
        },
        "complete_prefix_cumulative": {
            "host_samples_ms": host_total.tolist(),
            "host_p50_ms": percentile(host_total, 50),
            "host_p95_ms": percentile(host_total, 95),
            "cuda_samples_ms": cuda_total.tolist(),
            "cuda_p50_ms": percentile(cuda_total, 50),
            "cuda_p95_ms": percentile(cuda_total, 95),
        },
    }


def _measurement_source(
    codec: np.ndarray,
    *,
    length: int,
    scope: str,
    tokenizer: ArenaCodecSpatialTokenizer,
    device: torch.device,
) -> torch.Tensor:
    raw = (
        torch.from_numpy(np.array(codec[:length], copy=True))
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )
    if scope == "end_to_end":
        return raw
    if scope == "temporal_only":
        with torch.inference_mode():
            descriptors = tokenizer(raw)
        del raw
        return descriptors
    raise ValueError(f"unknown timing scope: {scope}")


def _descriptor_for_step(
    item: torch.Tensor,
    *,
    scope: str,
    tokenizer: ArenaCodecSpatialTokenizer,
) -> torch.Tensor:
    return tokenizer.encode_frame(item) if scope == "end_to_end" else item


@torch.inference_mode()
def _untimed_prefix(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    source: torch.Tensor,
    *,
    scope: str,
    state: Any | None = None,
) -> tuple[torch.Tensor, Any]:
    current = new_state(name, module, int(source.shape[0])) if state is None else state
    output = None
    for item in source.unbind(dim=1):
        descriptor = _descriptor_for_step(item, scope=scope, tokenizer=tokenizer)
        output, current = online_step(name, module, descriptor, current)
    if output is None:
        raise ValueError("prefix must contain at least one observation")
    return output, current


@torch.inference_mode()
def _timed_prefix_frames(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    source: torch.Tensor,
    *,
    scope: str,
    device: torch.device,
) -> tuple[list[float], list[float], Any, torch.Tensor]:
    state = new_state(name, module, int(source.shape[0]))
    host_samples = []
    cuda_samples = []
    output = None
    for item in source.unbind(dim=1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        host_start = time.perf_counter_ns()
        start.record()
        descriptor = _descriptor_for_step(item, scope=scope, tokenizer=tokenizer)
        output, state = online_step(name, module, descriptor, state)
        end.record()
        end.synchronize()
        host_samples.append((time.perf_counter_ns() - host_start) / 1_000_000)
        cuda_samples.append(float(start.elapsed_time(end)))
    if output is None:
        raise ValueError("timed prefix must contain at least one observation")
    return host_samples, cuda_samples, state, output


def _rollback_preallocated_step(state: PreallocatedKVState, expected_live_index: int) -> None:
    if state.live_index != expected_live_index + 1:
        raise RuntimeError("preallocated next-step rollback observed the wrong live index")
    for tensor in state.tensors():
        tensor[:, :, expected_live_index : expected_live_index + 1, :].fill_(state.sentinel)
    state.live_index = expected_live_index
    if not state.unwritten_is_sentinel():
        raise RuntimeError("preallocated next-step rollback corrupted sentinel tail")


def _is_cuda_kernel_event(event: Any) -> bool:
    if not str(getattr(event, "device_type", "")).lower().endswith("cuda"):
        return False
    name = str(getattr(event, "name", "")).lower()
    return "memcpy" not in name and "memset" not in name


@torch.inference_mode()
def _warm_next_step_branch(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    probe: torch.Tensor,
    *,
    scope: str,
    prefilled: Any,
    expected_observations: int,
) -> dict[str, Any]:
    """Consume one branch and return scalars so immutable state can be released."""

    descriptor = _descriptor_for_step(probe, scope=scope, tokenizer=tokenizer)
    output, next_state = online_step(name, module, descriptor, prefilled)
    evidence = {
        "output_finite": bool(torch.isfinite(output).all()),
        "observations_exact": next_state.observations == expected_observations,
        "branch_logical_state_bytes": _state_logical_bytes(next_state),
        "returned_is_prefilled": next_state is prefilled,
    }
    if name == "transformer_preallocated":
        _rollback_preallocated_step(prefilled, expected_observations - 1)
    return evidence


@torch.inference_mode()
def _timed_next_step_branch(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    probe: torch.Tensor,
    *,
    scope: str,
    prefilled: Any,
    expected_observations: int,
    device: torch.device,
) -> dict[str, Any]:
    """Time one branch; tensor locals die before the caller audits baseline."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    host_start = time.perf_counter_ns()
    start.record()
    descriptor = _descriptor_for_step(probe, scope=scope, tokenizer=tokenizer)
    output, next_state = online_step(name, module, descriptor, prefilled)
    end.record()
    end.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    evidence = {
        "host_ms": (time.perf_counter_ns() - host_start) / 1_000_000,
        "cuda_ms": float(start.elapsed_time(end)),
        "output_finite": bool(torch.isfinite(output).all()),
        "observations_exact": next_state.observations == expected_observations,
        "branch_logical_state_bytes": _state_logical_bytes(next_state),
        "returned_is_prefilled": next_state is prefilled,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    if name == "transformer_preallocated":
        _rollback_preallocated_step(prefilled, expected_observations - 1)
    return evidence


@torch.inference_mode()
def measure_prefilled_next_step(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    source: torch.Tensor,
    *,
    scope: str,
    device: torch.device,
) -> dict[str, Any]:
    length = int(source.shape[1])
    if length < 2:
        raise ValueError("next-step measurement requires T>=2")
    _, prefilled = _untimed_prefix(
        name,
        module,
        tokenizer,
        source[:, : length - 1],
        scope=scope,
    )
    probe = source[:, length - 1]
    prefill_observations = prefilled.observations
    prefill_logical_bytes = _state_logical_bytes(prefilled)
    prefill_capacity_bytes = _state_capacity_bytes(prefilled)
    for _ in range(WARMUP_SAMPLES):
        warm = _warm_next_step_branch(
            name,
            module,
            tokenizer,
            probe,
            scope=scope,
            prefilled=prefilled,
            expected_observations=length,
        )
        if not warm["output_finite"] or not warm["observations_exact"]:
            raise RuntimeError("next-step warm-up produced invalid output or state")
        if warm["returned_is_prefilled"] is not (name == "transformer_preallocated"):
            raise RuntimeError("next-step warm-up violated mutable/immutable state semantics")
        del warm
        torch.cuda.synchronize(device)
        gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    host_samples = []
    cuda_samples = []
    sample_baseline_allocated = []
    sample_baseline_reserved = []
    sample_peak_allocated = []
    sample_peak_reserved = []
    sample_restored_allocated = []
    sample_restored_reserved = []
    branch_logical_state_bytes = []
    returned_is_prefilled = []
    outputs_finite = True
    observations_exact = True
    for _ in range(MEASURED_SAMPLES):
        gc.collect()
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        if allocated_before != baseline_allocated:
            raise RuntimeError("next-step branch state survived into the next sample baseline")
        sample_baseline_allocated.append(allocated_before)
        sample_baseline_reserved.append(reserved_before)
        torch.cuda.reset_peak_memory_stats(device)
        sample = _timed_next_step_branch(
            name,
            module,
            tokenizer,
            probe,
            scope=scope,
            prefilled=prefilled,
            expected_observations=length,
            device=device,
        )
        host_samples.append(sample["host_ms"])
        cuda_samples.append(sample["cuda_ms"])
        outputs_finite = outputs_finite and sample["output_finite"]
        observations_exact = observations_exact and sample["observations_exact"]
        branch_logical_state_bytes.append(sample["branch_logical_state_bytes"])
        returned_is_prefilled.append(sample["returned_is_prefilled"])
        sample_peak_allocated.append(sample["peak_allocated_bytes"])
        sample_peak_reserved.append(sample["peak_reserved_bytes"])
        del sample
        gc.collect()
        torch.cuda.synchronize(device)
        restored_allocated = torch.cuda.memory_allocated(device)
        restored_reserved = torch.cuda.memory_reserved(device)
        sample_restored_allocated.append(restored_allocated)
        sample_restored_reserved.append(restored_reserved)
        if restored_allocated != baseline_allocated:
            raise RuntimeError("next-step branch state was not released after the sample")
    absolute_peak_allocated = max(sample_peak_allocated)
    absolute_peak_reserved = max(sample_peak_reserved)
    pre_profiler_baseline_allocated = torch.cuda.memory_allocated(device)
    if pre_profiler_baseline_allocated != baseline_allocated:
        raise RuntimeError("last measured branch survived into profiler baseline")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        with torch.profiler.record_function(f"{name}_{scope}_prefilled_next_step_t{length}"):
            descriptor = _descriptor_for_step(probe, scope=scope, tokenizer=tokenizer)
            profiled_output, profiled_state = online_step(name, module, descriptor, prefilled)
            torch.cuda.synchronize(device)
    profiled_observations = profiled_state.observations
    if name == "transformer_preallocated":
        _rollback_preallocated_step(prefilled, length - 1)
    device_events = [
        event
        for event in profile.events()
        if str(getattr(event, "device_type", "")).lower().endswith("cuda")
    ]
    kernels = [event for event in device_events if _is_cuda_kernel_event(event)]
    del descriptor, profiled_output, profiled_state
    gc.collect()
    torch.cuda.synchronize(device)
    return {
        "prefill_observations": prefill_observations,
        "probe_finishes_at_observations": length,
        "warmup_samples": WARMUP_SAMPLES,
        "measurements": MEASURED_SAMPLES,
        "allocator_peak_reset_after_prefill": True,
        "raw": {"host_samples_ms": host_samples, "cuda_samples_ms": cuda_samples},
        "host_p50_ms": percentile(host_samples, 50),
        "host_p95_ms": percentile(host_samples, 95),
        "cuda_p50_ms": percentile(cuda_samples, 50),
        "cuda_p95_ms": percentile(cuda_samples, 95),
        "logical_state_bytes": prefill_logical_bytes,
        "capacity_state_bytes": prefill_capacity_bytes,
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "absolute_peak_allocated_bytes": absolute_peak_allocated,
            "absolute_peak_reserved_bytes": absolute_peak_reserved,
            "delta_peak_allocated_bytes": absolute_peak_allocated - baseline_allocated,
            "delta_peak_reserved_bytes": absolute_peak_reserved - baseline_reserved,
            "sample_baseline_allocated_bytes": sample_baseline_allocated,
            "sample_baseline_reserved_bytes": sample_baseline_reserved,
            "sample_peak_allocated_bytes": sample_peak_allocated,
            "sample_peak_reserved_bytes": sample_peak_reserved,
            "sample_restored_allocated_bytes": sample_restored_allocated,
            "sample_restored_reserved_bytes": sample_restored_reserved,
            "baseline_allocated_stable": sample_baseline_allocated
            == [baseline_allocated] * MEASURED_SAMPLES,
            "allocated_restored_after_each_sample": sample_restored_allocated
            == [baseline_allocated] * MEASURED_SAMPLES,
            "pre_profiler_baseline_allocated_bytes": pre_profiler_baseline_allocated,
            "prefilled_baseline_recovered_before_profiler": pre_profiler_baseline_allocated
            == baseline_allocated,
        },
        "branch_logical_state_bytes": branch_logical_state_bytes,
        "returned_is_prefilled": returned_is_prefilled,
        "profiler": {
            "runs": 1,
            "cuda_device_event_count": len(device_events),
            "cuda_kernel_count": len(kernels),
            "excluded_memcpy_memset_count": len(device_events) - len(kernels),
            "profiled_state_observations": profiled_observations,
        },
        "outputs_finite": outputs_finite,
        "observations_exact": observations_exact,
        "preallocated_state_rolled_back": name != "transformer_preallocated"
        or prefilled.observations == length - 1,
    }


@torch.inference_mode()
def benchmark_scope_length(
    codec: np.ndarray,
    *,
    name: str,
    tokenizer: ArenaCodecSpatialTokenizer,
    module: nn.Module,
    length: int,
    scope: str,
) -> dict[str, Any]:
    if length not in LENGTHS or scope not in SCOPES:
        raise ValueError("cost length or scope escaped the frozen contract")
    device = torch.device("cuda", 0)
    source = _measurement_source(
        codec,
        length=length,
        scope=scope,
        tokenizer=tokenizer,
        device=device,
    )
    initial_state = new_state(name, module, 1)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    memory_output, memory_state = _untimed_prefix(
        name,
        module,
        tokenizer,
        source,
        scope=scope,
        state=initial_state,
    )
    torch.cuda.synchronize(device)
    absolute_peak_allocated = torch.cuda.max_memory_allocated(device)
    absolute_peak_reserved = torch.cuda.max_memory_reserved(device)
    logical_state_bytes = _state_logical_bytes(memory_state)
    capacity_state_bytes = _state_capacity_bytes(memory_state)
    memory_finite = bool(torch.isfinite(memory_output).all()) and _state_finite(memory_state)
    final_observations = memory_state.observations
    sentinel_tail_preserved = (
        memory_state.unwritten_is_sentinel()
        if isinstance(memory_state, PreallocatedKVState)
        else True
    )
    del memory_output, memory_state, initial_state
    for _ in range(WARMUP_SAMPLES):
        _timed_prefix_frames(
            name,
            module,
            tokenizer,
            source,
            scope=scope,
            device=device,
        )
    warm_telemetry = gpu_telemetry()
    host_rows = []
    cuda_rows = []
    all_finite = memory_finite
    observations_exact = final_observations == length
    measured_logical = []
    measured_capacity = []
    for _ in range(MEASURED_SAMPLES):
        host, cuda, state, output = _timed_prefix_frames(
            name,
            module,
            tokenizer,
            source,
            scope=scope,
            device=device,
        )
        host_rows.append(host)
        cuda_rows.append(cuda)
        measured_logical.append(_state_logical_bytes(state))
        measured_capacity.append(_state_capacity_bytes(state))
        all_finite = all_finite and bool(torch.isfinite(output).all()) and _state_finite(state)
        observations_exact = observations_exact and state.observations == length
    result = {
        "condition": name,
        "length": length,
        "scope": scope,
        "warmup_samples": WARMUP_SAMPLES,
        "measurements": MEASURED_SAMPLES,
        "fresh_state_each_prefix_sample": True,
        "cuda_synchronize_each_frame": True,
        "latency": _latency_summary(host_rows, cuda_rows),
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "absolute_peak_allocated_bytes": absolute_peak_allocated,
            "absolute_peak_reserved_bytes": absolute_peak_reserved,
            "delta_peak_allocated_bytes": absolute_peak_allocated - baseline_allocated,
            "delta_peak_reserved_bytes": absolute_peak_reserved - baseline_reserved,
        },
        "logical_state_bytes": logical_state_bytes,
        "capacity_state_bytes": capacity_state_bytes,
        "measured_logical_state_bytes": measured_logical,
        "measured_capacity_state_bytes": measured_capacity,
        "final_state_observations": final_observations,
        "all_outputs_and_states_finite": all_finite,
        "all_observations_exact": observations_exact,
        "sentinel_tail_preserved": sentinel_tail_preserved,
        "next_step": measure_prefilled_next_step(
            name,
            module,
            tokenizer,
            source,
            scope=scope,
            device=device,
        ),
        "warm_telemetry": warm_telemetry,
    }
    del source
    return result


def _nvidia_smi_csv(query_scope: str, fields: str, *, gpu_id: int | None = None) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-{query_scope}={fields}",
        "--format=csv,noheader,nounits",
    ]
    if gpu_id is not None:
        command.extend(["--id", str(gpu_id)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip() and "No running processes found" not in line
    ]


def gpu_telemetry() -> dict[str, Any]:
    fields = (
        "index,uuid,name,driver_version,utilization.gpu,clocks.current.sm,"
        "temperature.gpu,memory.used,memory.total"
    )
    rows = _nvidia_smi_csv("gpu", fields, gpu_id=0)
    if len(rows) != 1 or len(rows[0]) != 9:
        raise RuntimeError("nvidia-smi returned an unexpected GPU0 row")
    names = (
        "physical_index",
        "uuid",
        "name",
        "driver_version",
        "utilization_percent",
        "sm_clock_mhz",
        "temperature_c",
        "memory_used_mib",
        "memory_total_mib",
    )
    gpu = dict(zip(names, rows[0], strict=True))
    process_rows = _nvidia_smi_csv("compute-apps", "gpu_uuid,pid,process_name,used_gpu_memory")
    processes = [
        dict(
            zip(
                ("gpu_uuid", "pid", "process_name", "used_gpu_memory_mib"),
                row,
                strict=True,
            )
        )
        for row in process_rows
        if len(row) == 4 and row[0] == gpu["uuid"]
    ]
    current_pid = os.getpid()
    return {
        "gpu": gpu,
        "process_pid": current_pid,
        "process_ppid": os.getppid(),
        "gpu0_compute_processes": processes,
        "gpu0_other_compute_processes": [
            row for row in processes if int(row["pid"]) != current_pid
        ],
    }


def expected_execution_order(label: str) -> list[dict[str, Any]]:
    return [
        {"condition": condition, "length": length, "scope": scope}
        for condition in CONDITION_ORDERS[label]
        for length in LENGTH_ORDERS[label]
        for scope in SCOPES
    ]


@torch.inference_mode()
def execute_cost_measurements(
    codec: np.ndarray,
    *,
    label: str,
    expected_models: dict[str, Any],
    expected_descriptor_sha256: str,
) -> dict[str, Any]:
    if label not in COST_RUN_IDS:
        raise ValueError("cost label must be one of A-E")
    device = torch.device("cuda", 0)
    measurements: dict[str, Any] = {}
    execution_order = []
    condition_identities = {}
    tokenizer_identities = {}
    for name in CONDITION_ORDERS[label]:
        gc.collect()
        torch.cuda.empty_cache()
        tokenizer = (
            ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device=device, dtype=torch.float32)
        )
        module = build_condition(name).eval().to(device=device, dtype=torch.float32)
        tokenizer_identity = {
            "parameters": trainable_parameter_count(tokenizer),
            "state_dict_sha256": state_dict_sha256(tokenizer),
        }
        if tokenizer_identity != expected_models["tokenizer"]:
            raise RuntimeError(f"cost tokenizer identity differs from correctness: {name}")
        tokenizer_identities[name] = tokenizer_identity
        identity = {
            "parameters": trainable_parameter_count(module),
            "state_dict_sha256": state_dict_sha256(module),
            "state_dict_tensor_sha256": state_dict_tensor_sha256(module),
        }
        if identity != expected_models["conditions"][name]:
            raise RuntimeError(f"cost model identity differs from correctness: {name}")
        condition_identities[name] = identity
        by_length = {}
        for length in LENGTH_ORDERS[label]:
            by_scope = {}
            for scope in SCOPES:
                by_scope[scope] = benchmark_scope_length(
                    codec,
                    name=name,
                    tokenizer=tokenizer,
                    module=module,
                    length=length,
                    scope=scope,
                )
                execution_order.append({"condition": name, "length": length, "scope": scope})
            by_length[str(length)] = by_scope
        measurements[name] = by_length
        module.to(device="cpu")
        tokenizer.to(device="cpu")
        del module, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device=device, dtype=torch.float32)
    raw = (
        torch.from_numpy(np.array(codec, copy=True))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    descriptors = tokenizer(raw)
    descriptor_sha256 = tensor_sha256(descriptors)
    if descriptor_sha256 != expected_descriptor_sha256:
        raise RuntimeError("cost descriptor hash differs from correctness")
    return {
        "label": label,
        "seed": PROCESS_SEEDS[label],
        "condition_order": list(CONDITION_ORDERS[label]),
        "length_order": list(LENGTH_ORDERS[label]),
        "scope_order": list(SCOPES),
        "execution_order": execution_order,
        "model_identity": expected_models,
        "tokenizer_identity_recomputed": tokenizer_identities,
        "condition_identity_recomputed": condition_identities,
        "descriptor_sha256": descriptor_sha256,
        "measurements": measurements,
    }


def _finite_vector(value: Any, length: int, *, positive: bool) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return (
        array.shape == (length,)
        and bool(np.isfinite(array).all())
        and bool(np.all(array > 0 if positive else array >= 0))
    )


def _finite_matrix(value: Any, rows: int, columns: int, *, positive: bool) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return (
        array.shape == (rows, columns)
        and bool(np.isfinite(array).all())
        and bool(np.all(array > 0 if positive else array >= 0))
    )


def _same_number(first: Any, second: Any, *, atol: float = 1e-9) -> bool:
    try:
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _memory_arithmetic_valid(memory: Any) -> bool:
    if not isinstance(memory, dict):
        return False
    keys = {
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "absolute_peak_allocated_bytes",
        "absolute_peak_reserved_bytes",
        "delta_peak_allocated_bytes",
        "delta_peak_reserved_bytes",
    }
    if not keys.issubset(memory):
        return False
    try:
        values = {key: int(memory[key]) for key in keys}
    except (TypeError, ValueError):
        return False
    return (
        all(value >= 0 for value in values.values())
        and values["absolute_peak_allocated_bytes"] >= values["baseline_allocated_bytes"]
        and values["absolute_peak_reserved_bytes"] >= values["baseline_reserved_bytes"]
        and values["delta_peak_allocated_bytes"]
        == values["absolute_peak_allocated_bytes"] - values["baseline_allocated_bytes"]
        and values["delta_peak_reserved_bytes"]
        == values["absolute_peak_reserved_bytes"] - values["baseline_reserved_bytes"]
    )


def _next_step_memory_lifecycle_valid(memory: Any) -> bool:
    if not _memory_arithmetic_valid(memory):
        return False
    vector_names = (
        "sample_baseline_allocated_bytes",
        "sample_baseline_reserved_bytes",
        "sample_peak_allocated_bytes",
        "sample_peak_reserved_bytes",
        "sample_restored_allocated_bytes",
        "sample_restored_reserved_bytes",
    )
    if not all(
        _finite_vector(memory.get(name), MEASURED_SAMPLES, positive=False) for name in vector_names
    ):
        return False
    baseline_allocated = memory["baseline_allocated_bytes"]
    baseline_allocated_samples = memory["sample_baseline_allocated_bytes"]
    baseline_reserved_samples = memory["sample_baseline_reserved_bytes"]
    peak_allocated = memory["sample_peak_allocated_bytes"]
    peak_reserved = memory["sample_peak_reserved_bytes"]
    restored_allocated = memory["sample_restored_allocated_bytes"]
    return all(
        (
            baseline_allocated_samples == [baseline_allocated] * MEASURED_SAMPLES,
            restored_allocated == [baseline_allocated] * MEASURED_SAMPLES,
            all(
                peak >= baseline
                for peak, baseline in zip(peak_allocated, baseline_allocated_samples, strict=True)
            ),
            all(
                peak >= baseline
                for peak, baseline in zip(peak_reserved, baseline_reserved_samples, strict=True)
            ),
            memory["absolute_peak_allocated_bytes"] == max(peak_allocated),
            memory["absolute_peak_reserved_bytes"] == max(peak_reserved),
            memory.get("baseline_allocated_stable") is True,
            memory.get("allocated_restored_after_each_sample") is True,
            memory.get("pre_profiler_baseline_allocated_bytes") == baseline_allocated,
            memory.get("prefilled_baseline_recovered_before_profiler") is True,
        )
    )


def _latency_payload_valid(latency: Any, length: int) -> bool:
    if not isinstance(latency, dict):
        return False
    try:
        host = latency["raw"]["host_per_frame_ms"]
        cuda = latency["raw"]["cuda_per_frame_ms"]
        recomputed = _latency_summary(host, cuda)
    except (KeyError, TypeError, ValueError):
        return False
    if not _finite_matrix(host, MEASURED_SAMPLES, length, positive=True):
        return False
    if not _finite_matrix(cuda, MEASURED_SAMPLES, length, positive=False):
        return False
    return all(
        latency.get(section) == recomputed[section]
        for section in (
            "per_frame_flat",
            "per_position",
            "last_frame",
            "complete_prefix_cumulative",
        )
    )


def _prefill_expected_bytes(name: str, length: int) -> int:
    if name.startswith("transformer_"):
        return transformer_kv_expected_bytes(batch_size=1, observations=length - 1)
    return EXPECTED_STATE_BYTES[name][length]


def _next_step_payload_valid(next_step: Any, name: str, length: int, scope: str) -> bool:
    if not isinstance(next_step, dict):
        return False
    try:
        host = next_step["raw"]["host_samples_ms"]
        cuda = next_step["raw"]["cuda_samples_ms"]
        profiler = next_step["profiler"]
    except (KeyError, TypeError):
        return False
    expected_live = _prefill_expected_bytes(name, length)
    expected_capacity = (
        preallocated_capacity_expected_bytes()
        if name == "transformer_preallocated"
        else expected_live
    )
    return all(
        (
            next_step.get("prefill_observations") == length - 1,
            next_step.get("probe_finishes_at_observations") == length,
            next_step.get("warmup_samples") == WARMUP_SAMPLES,
            next_step.get("measurements") == MEASURED_SAMPLES,
            next_step.get("allocator_peak_reset_after_prefill") is True,
            _finite_vector(host, MEASURED_SAMPLES, positive=True),
            _finite_vector(cuda, MEASURED_SAMPLES, positive=False),
            _same_number(next_step.get("host_p50_ms"), percentile(host, 50)),
            _same_number(next_step.get("host_p95_ms"), percentile(host, 95)),
            _same_number(next_step.get("cuda_p50_ms"), percentile(cuda, 50)),
            _same_number(next_step.get("cuda_p95_ms"), percentile(cuda, 95)),
            next_step.get("logical_state_bytes") == expected_live,
            next_step.get("capacity_state_bytes") == expected_capacity,
            _next_step_memory_lifecycle_valid(next_step.get("memory")),
            profiler.get("runs") == 1,
            isinstance(profiler.get("cuda_kernel_count"), int),
            profiler.get("cuda_kernel_count", -1) >= 0,
            (
                profiler.get("cuda_kernel_count", 0) > 0
                or (name == "current_descriptor" and scope == "temporal_only")
            ),
            profiler.get("cuda_device_event_count", -1) >= profiler.get("cuda_kernel_count", 0),
            profiler.get("excluded_memcpy_memset_count", -1)
            == profiler.get("cuda_device_event_count", 0) - profiler.get("cuda_kernel_count", 0),
            profiler.get("profiled_state_observations") == length,
            next_step.get("outputs_finite") is True,
            next_step.get("observations_exact") is True,
            next_step.get("preallocated_state_rolled_back") is True,
            next_step.get("branch_logical_state_bytes")
            == [EXPECTED_STATE_BYTES[name][length]] * MEASURED_SAMPLES,
            next_step.get("returned_is_prefilled")
            == [name == "transformer_preallocated"] * MEASURED_SAMPLES,
        )
    )


def _telemetry_point_valid(point: Any) -> bool:
    if not isinstance(point, dict):
        return False
    try:
        gpu = point["gpu"]
        numeric = [
            float(gpu[field])
            for field in (
                "physical_index",
                "utilization_percent",
                "sm_clock_mhz",
                "temperature_c",
                "memory_used_mib",
                "memory_total_mib",
            )
        ]
        processes = point["gpu0_compute_processes"]
        others = point["gpu0_other_compute_processes"]
        process_pid = int(point["process_pid"])
        process_ppid = int(point["process_ppid"])
    except (KeyError, TypeError, ValueError):
        return False

    def process_row_valid(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        try:
            memory = float(row.get("used_gpu_memory_mib", "nan"))
            pid = int(row.get("pid"))
        except (TypeError, ValueError):
            return False
        return (
            row.get("gpu_uuid") == gpu.get("uuid")
            and bool(row.get("process_name"))
            and pid > 0
            and math.isfinite(memory)
            and memory >= 0
        )

    process_rows_valid = isinstance(processes, list) and all(
        process_row_valid(row) for row in processes
    )
    return (
        all(math.isfinite(value) for value in numeric)
        and numeric[0] == 0
        and bool(gpu.get("uuid"))
        and gpu.get("name") == FROZEN_RUNTIME["gpu_name"]
        and bool(gpu.get("driver_version"))
        and process_pid > 0
        and process_ppid >= 0
        and process_rows_valid
        and isinstance(others, list)
        and not others
    )


def _measurement_payload_valid(measurement: Any, name: str, length: int, scope: str) -> bool:
    if not isinstance(measurement, dict):
        return False
    expected_live = EXPECTED_STATE_BYTES[name][length]
    expected_capacity = (
        preallocated_capacity_expected_bytes()
        if name == "transformer_preallocated"
        else expected_live
    )
    return all(
        (
            measurement.get("condition") == name,
            measurement.get("length") == length,
            measurement.get("scope") == scope,
            measurement.get("warmup_samples") == WARMUP_SAMPLES,
            measurement.get("measurements") == MEASURED_SAMPLES,
            measurement.get("fresh_state_each_prefix_sample") is True,
            measurement.get("cuda_synchronize_each_frame") is True,
            _latency_payload_valid(measurement.get("latency"), length),
            _memory_arithmetic_valid(measurement.get("memory")),
            measurement.get("logical_state_bytes") == expected_live,
            measurement.get("capacity_state_bytes") == expected_capacity,
            measurement.get("measured_logical_state_bytes") == [expected_live] * MEASURED_SAMPLES,
            measurement.get("measured_capacity_state_bytes")
            == [expected_capacity] * MEASURED_SAMPLES,
            measurement.get("final_state_observations") == length,
            measurement.get("all_outputs_and_states_finite") is True,
            measurement.get("all_observations_exact") is True,
            measurement.get("sentinel_tail_preserved") is True,
            _next_step_payload_valid(measurement.get("next_step"), name, length, scope),
            _telemetry_point_valid(measurement.get("warm_telemetry")),
        )
    )


def evaluate_cost_result(
    cost: dict[str, Any],
    telemetry: dict[str, Any],
    execution: dict[str, Any],
    *,
    label: str,
    expected_models: dict[str, Any],
    expected_descriptor_sha256: str,
) -> dict[str, bool]:
    measurements_valid = True
    try:
        measurements = cost["measurements"]
        for name in CONDITIONS:
            for length in LENGTHS:
                for scope in SCOPES:
                    measurements_valid = measurements_valid and _measurement_payload_valid(
                        measurements[name][str(length)][scope], name, length, scope
                    )
    except (KeyError, TypeError):
        measurements_valid = False
    return {
        "exact_cyclic_order_and_complete_execution_order": (
            label in COST_RUN_IDS
            and cost.get("label") == label
            and cost.get("seed") == PROCESS_SEEDS[label]
            and cost.get("condition_order") == list(CONDITION_ORDERS[label])
            and cost.get("length_order") == list(LENGTH_ORDERS[label])
            and cost.get("scope_order") == list(SCOPES)
            and cost.get("execution_order") == expected_execution_order(label)
        ),
        "model_and_descriptor_identity_match_correctness": cost.get("model_identity")
        == expected_models
        and cost.get("descriptor_sha256") == expected_descriptor_sha256
        and cost.get("tokenizer_identity_recomputed")
        == {name: expected_models["tokenizer"] for name in CONDITIONS}
        and cost.get("condition_identity_recomputed")
        == {name: expected_models["conditions"][name] for name in CONDITIONS},
        "all_40_measurements_have_complete_raw_latency_memory_state_next_step_and_profiler": measurements_valid,
        "telemetry_finite_gpu0_isolated_and_complete": (
            _telemetry_point_valid(telemetry.get("before"))
            and _telemetry_point_valid(telemetry.get("after"))
            and measurements_valid
        ),
        "native_vla_modules_absent": not execution.get("native_vla_modules", {}).get("before")
        and not execution.get("native_vla_modules", {}).get("after"),
        "optimizer_steps_zero": not execution.get("optimizer", {}).get("constructor_calls")
        and execution.get("optimizer", {}).get("steps_observed") == 0,
    }


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_RESULT_BYTES:
        raise ValueError("JSON output exceeds the per-result byte limit")
    if directory_bytes(ARTIFACT_ROOT) + len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("JSON output would exceed the experiment artifact-root limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
    return {"path": str(path.resolve()), "bytes": len(encoded), "sha256": sha256(path)}


def _finalize_result_artifact(result: dict[str, Any], *, bytes_before: int) -> None:
    result.setdefault("gates", {})["artifact_result_and_root_within_limits"] = False
    result["artifact"] = {
        "root": str(ARTIFACT_ROOT.resolve()),
        "root_bytes_before": bytes_before,
        "result_bytes": 0,
        "root_bytes_after_write": 0,
        "single_result_limit_bytes": MAX_RESULT_BYTES,
        "root_limit_bytes": MAX_ARTIFACT_BYTES,
        "traces_written": 0,
    }
    for _ in range(12):
        core_passed = all(result["gates"].values())
        result["status"] = (
            "failed"
            if result.get("kind") == "failure"
            else "passed"
            if core_passed
            else "gate_failed"
        )
        if result.get("kind") == "cost":
            result["next_cost_allowed"] = core_passed and result["label"] != "E"
            result["aggregate_allowed"] = core_passed and result["label"] == "E"
        encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        total = bytes_before + len(encoded)
        gate = len(encoded) <= MAX_RESULT_BYTES and total <= MAX_ARTIFACT_BYTES
        next_artifact = {
            **result["artifact"],
            "result_bytes": len(encoded),
            "root_bytes_after_write": total,
        }
        stable = (
            next_artifact == result["artifact"]
            and result["gates"]["artifact_result_and_root_within_limits"] == gate
        )
        result["artifact"] = next_artifact
        result["gates"]["artifact_result_and_root_within_limits"] = gate
        if stable:
            break
    else:
        raise RuntimeError("result artifact metadata did not stabilize")
    result["status"] = (
        "failed"
        if result.get("kind") == "failure"
        else "passed"
        if all(result["gates"].values())
        else "gate_failed"
    )
    if result.get("kind") == "cost":
        result["next_cost_allowed"] = result["status"] == "passed" and result["label"] != "E"
        result["aggregate_allowed"] = result["status"] == "passed" and result["label"] == "E"
    elif result.get("kind") == "correctness":
        result["cost_allowed"] = result["status"] == "passed"
    elif result.get("kind") == "aggregate":
        result["online_mamba_engineering_scaling_advantage"] = (
            result["status"] == "passed"
            and result["decision"]["online_mamba_engineering_scaling_advantage"]
        )


def write_result_once(run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    output = ARTIFACT_ROOT / run_id
    if output.exists():
        raise FileExistsError(output)
    bytes_before = directory_bytes(ARTIFACT_ROOT)
    output.mkdir(parents=True)
    path = output / "RESULT.json"
    _finalize_result_artifact(result, bytes_before=bytes_before)
    identity = _write_new_json(path, result)
    if identity["bytes"] != result["artifact"]["result_bytes"]:
        raise RuntimeError("stored result bytes differ from stabilized artifact metadata")
    return identity


def prepare_plan(path: Path) -> dict[str, Any]:
    if path.resolve() != PLAN_PATH.resolve():
        raise ValueError(f"resolved plan path must be {PLAN_PATH}")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    verify_source(commit)
    return _write_new_json(path, resolved_plan(commit))


def verify_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("resolved plan is missing or outside the artifact root")
    identity = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }
    if identity["sha256"] != expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("resolved plan identity changed")
    plan = read_json(resolved)
    commit = plan.get("source", {}).get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("resolved plan source commit is invalid")
    # JSON object keys are strings.  Canonicalize the freshly resolved Python
    # contract through the same JSON boundary before comparing it with the
    # stored plan; in particular, this preserves the 20 integer length keys in
    # ``expected_state_bytes`` as their JSON string representation.
    expected_plan = json.loads(json.dumps(resolved_plan(commit), sort_keys=True))
    if plan != expected_plan:
        raise ValueError("resolved plan contract changed")
    return plan, identity


def verify_source(expected_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != expected_commit:
        raise ValueError("online scaling source commit changed")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "merge-base",
            "--is-ancestor",
            BASE_SOURCE_COMMIT,
            commit,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("online scaling source escaped frozen base ancestry")
    dirty = set()
    for extra in ((), ("--cached",)):
        changed = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "diff",
                *extra,
                "--name-only",
                "--",
                *SOURCE_IDENTITY_PATHS,
            ],
            text=True,
        )
        dirty.update(changed.splitlines())
    untracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_IDENTITY_PATHS,
        ],
        text=True,
    )
    dirty.update(untracked.splitlines())
    if dirty:
        raise ValueError(f"online scaling source is dirty: {sorted(dirty)}")
    assert_closed_routes_absent()
    entries = {}
    for relative in SOURCE_IDENTITY_PATHS:
        path = (REPO / relative).resolve()
        if not path.is_file() or not path.is_relative_to(REPO.resolve()):
            raise ValueError(f"source identity path is missing: {relative}")
        entries[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return {"root": str(REPO.resolve()), "commit": commit, "entries": entries}


def _verify_material(spec: dict[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(spec["path"]).resolve()
    if not path.is_file() or path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"{label} identity changed")
    return copy.deepcopy(spec)


def verify_qualified_prior_evidence() -> dict[str, Any]:
    return {
        "functional_aggregate": _verify_material(
            FUNCTIONAL_AGGREGATE, label="functional aggregate"
        ),
        "functional_recheck": _verify_material(FUNCTIONAL_RECHECK, label="functional recheck"),
    }


def load_codec_input() -> tuple[np.ndarray, dict[str, Any]]:
    return input_runtime_provider.load_codec_input()


def verify_dependencies() -> dict[str, Any]:
    return input_runtime_provider.verify_dependencies()


def verify_environment() -> dict[str, Any]:
    previous = input_runtime_provider.FROZEN_RUNTIME
    input_runtime_provider.FROZEN_RUNTIME = FROZEN_RUNTIME
    try:
        return input_runtime_provider.verify_environment()
    finally:
        input_runtime_provider.FROZEN_RUNTIME = previous


def _artifact_metadata_valid(result: dict[str, Any], identity: dict[str, Any]) -> bool:
    artifact = result.get("artifact", {})
    try:
        return all(
            (
                artifact["root"] == str(ARTIFACT_ROOT.resolve()),
                artifact["result_bytes"] == identity["bytes"],
                artifact["root_bytes_before"] + artifact["result_bytes"]
                == artifact["root_bytes_after_write"],
                artifact["result_bytes"]
                <= artifact["single_result_limit_bytes"]
                == MAX_RESULT_BYTES,
                artifact["root_bytes_after_write"]
                <= artifact["root_limit_bytes"]
                == MAX_ARTIFACT_BYTES,
                artifact["traces_written"] == 0,
            )
        )
    except (KeyError, TypeError):
        return False


def _measurement(
    cost_result: dict[str, Any], condition: str, length: int, scope: str
) -> dict[str, Any]:
    return cost_result["cost"]["measurements"][condition][str(length)][scope]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return float("inf")
    return numerator / denominator


def aggregate_decision_metrics(costs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(costs) != 5:
        raise ValueError("aggregate requires exactly five isolated cost results")
    per_process = {}
    for result in costs:
        label = result["label"]
        mamba_925_e2e = _measurement(result, "functional_mamba", 925, "end_to_end")
        mamba_925_temporal = _measurement(result, "functional_mamba", 925, "temporal_only")
        mamba_128_temporal = _measurement(result, "functional_mamba", 128, "temporal_only")
        preallocated_925 = _measurement(result, "transformer_preallocated", 925, "temporal_only")
        cat_925 = _measurement(result, "transformer_functional_cat", 925, "temporal_only")
        mamba_next = mamba_925_temporal["next_step"]["host_p95_ms"]
        preallocated_next = preallocated_925["next_step"]["host_p95_ms"]
        mamba_prefix = mamba_925_temporal["latency"]["complete_prefix_cumulative"]["host_p95_ms"]
        preallocated_prefix = preallocated_925["latency"]["complete_prefix_cumulative"][
            "host_p95_ms"
        ]
        cat_delta = cat_925["memory"]["delta_peak_allocated_bytes"]
        preallocated_delta = preallocated_925["memory"]["delta_peak_allocated_bytes"]
        per_process[label] = {
            "mamba_t925_end_to_end_per_frame_host_p95_ms": mamba_925_e2e["latency"][
                "per_frame_flat"
            ]["host_p95_ms"],
            "mamba_t925_logical_state_bytes": mamba_925_temporal["logical_state_bytes"],
            "preallocated_t925_live_kv_bytes": preallocated_925["logical_state_bytes"],
            "state_ratio": _safe_ratio(
                mamba_925_temporal["logical_state_bytes"],
                preallocated_925["logical_state_bytes"],
            ),
            "next_step_ratio": _safe_ratio(mamba_next, preallocated_next),
            "prefix_cumulative_ratio": _safe_ratio(mamba_prefix, preallocated_prefix),
            "mamba_t925_t128_next_step_ratio": _safe_ratio(
                mamba_next, mamba_128_temporal["next_step"]["host_p95_ms"]
            ),
            "preallocated_functional_cat_delta_peak_ratio": _safe_ratio(
                preallocated_delta, cat_delta
            ),
        }
    labels = list(COST_RUN_IDS)

    def values(field: str) -> list[float]:
        return [float(per_process[label][field]) for label in labels]

    next_ratios = values("next_step_ratio")
    prefix_ratios = values("prefix_cumulative_ratio")
    allocator_ratios = values("preallocated_functional_cat_delta_peak_ratio")
    gates = {
        "all_five_mamba_t925_end_to_end_p95_below_20ms": all(
            value < 20.0 for value in values("mamba_t925_end_to_end_per_frame_host_p95_ms")
        ),
        "all_five_mamba_state_at_most_10pct_preallocated_live_kv": all(
            value <= 0.10 for value in values("state_ratio")
        ),
        "mamba_next_step_at_least_20pct_lower_in_4of5_and_median": (
            sum(value < 1.0 for value in next_ratios) >= 4 and float(np.median(next_ratios)) <= 0.80
        ),
        "mamba_prefix_cumulative_at_least_20pct_lower_in_4of5_and_median": (
            sum(value < 1.0 for value in prefix_ratios) >= 4
            and float(np.median(prefix_ratios)) <= 0.80
        ),
        "all_five_mamba_t925_t128_next_step_ratios_at_most_1p15": all(
            value <= 1.15 for value in values("mamba_t925_t128_next_step_ratio")
        ),
        "preallocated_allocator_delta_at_least_50pct_lower_in_4of5": (
            sum(value <= 0.50 for value in allocator_ratios) >= 4
        ),
    }
    return {
        "per_process": per_process,
        "process_medians": {
            "next_step_ratio": float(np.median(next_ratios)),
            "prefix_cumulative_ratio": float(np.median(prefix_ratios)),
            "preallocated_functional_cat_delta_peak_ratio": float(np.median(allocator_ratios)),
        },
        "direction_counts": {
            "next_step_mamba_lower": sum(value < 1.0 for value in next_ratios),
            "prefix_cumulative_mamba_lower": sum(value < 1.0 for value in prefix_ratios),
            "preallocated_allocator_at_most_half": sum(value <= 0.50 for value in allocator_ratios),
        },
        "decision_gates": gates,
        "online_mamba_engineering_scaling_advantage": all(gates.values()),
    }


def evaluate_aggregate_result(
    correctness: dict[str, Any],
    costs: list[dict[str, Any]],
    cost_identities: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    decision = aggregate_decision_metrics(costs)
    labels = list(COST_RUN_IDS)
    chain_valid = len(cost_identities) == 5
    if chain_valid:
        correctness_sha = correctness["result_identity"]["sha256"]
        chain_valid = (
            costs[0]["cost_predecessor"] is None
            and all(cost["correctness_predecessor"]["sha256"] == correctness_sha for cost in costs)
            and all(
                costs[index]["cost_predecessor"]["sha256"] == cost_identities[index - 1]["sha256"]
                for index in range(1, 5)
            )
        )
    gates = {
        "correctness_passed_with_all_gates": correctness.get("status") == "passed"
        and all(correctness.get("gates", {}).values()),
        "exact_five_ordered_distinct_cost_processes_passed": (
            [cost.get("label") for cost in costs] == labels
            and [cost.get("run_id") for cost in costs] == list(COST_RUN_IDS.values())
            and len({cost.get("process", {}).get("pid") for cost in costs}) == 5
            and all(cost.get("status") == "passed" for cost in costs)
            and all(all(cost.get("gates", {}).values()) for cost in costs)
        ),
        "cost_results_form_exact_correctness_and_predecessor_hash_chain": chain_valid,
        "all_five_processes_report_no_oom_nan_state_storage_or_telemetry_anomaly": all(
            cost.get("engineering_anomalies") == [] for cost in costs
        ),
        **decision["decision_gates"],
    }
    return gates, decision


def _result_identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"predecessor result is missing: {resolved}")
    identity = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }
    if len(expected_sha256) != 64 or identity["sha256"] != expected_sha256:
        raise ValueError("predecessor result SHA-256 changed")
    return identity


def _cost_chain_valid(
    result: dict[str, Any],
    *,
    label: str,
    correctness_identity: dict[str, Any],
) -> bool:
    if label not in COST_RUN_IDS or result.get("correctness_predecessor") != correctness_identity:
        return False
    labels = list(COST_RUN_IDS)
    index = labels.index(label)
    chain = result.get("predecessor_chain")
    if not isinstance(chain, list) or len(chain) != index:
        return False
    verified = []
    try:
        for previous_label, spec in zip(labels[:index], chain, strict=True):
            expected_path = ARTIFACT_ROOT / COST_RUN_IDS[previous_label] / "RESULT.json"
            if spec.get("path") != str(expected_path.resolve()):
                return False
            actual = _result_identity(expected_path, spec.get("sha256", ""))
            if actual != spec:
                return False
            verified.append(actual)
    except (AttributeError, TypeError, ValueError):
        return False
    expected_immediate = None if index == 0 else verified[-1]
    return result.get("cost_predecessor") == expected_immediate


def verify_passed_component(
    path: Path,
    expected_sha256: str,
    *,
    expected_path: Path,
    expected_run_id: str,
    expected_kind: str,
    plan_identity: dict[str, Any],
    source_commit: str,
    correctness: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.resolve() != expected_path.resolve():
        raise ValueError("predecessor result path changed")
    identity = _result_identity(path, expected_sha256)
    result = read_json(path)
    if (
        result.get("experiment_id") != EXPERIMENT
        or result.get("run_id") != expected_run_id
        or result.get("kind") != expected_kind
        or result.get("source_commit") != source_commit
        or result.get("plan") != plan_identity
        or result.get("status") != "passed"
        or result.get("training_allowed") is not False
        or result.get("vla_model_instantiated") is not False
        or result.get("optimizer_steps") != 0
        or not _artifact_metadata_valid(result, identity)
    ):
        raise ValueError("predecessor result identity or boundary changed")
    wall_limit = (
        MAX_CORRECTNESS_WALL_SECONDS if expected_kind == "correctness" else MAX_COST_WALL_SECONDS
    )
    try:
        wall_valid = 0 <= float(result["wall_seconds"]) <= wall_limit
    except (KeyError, TypeError, ValueError):
        wall_valid = False
    if (
        not wall_valid
        or result.get("gates", {}).get("wall_time_within_limit") is not True
        or result.get("gates", {}).get("artifact_result_and_root_within_limits") is not True
    ):
        raise ValueError("predecessor wall or artifact gate changed")
    if expected_kind == "correctness":
        if result.get("model_identity") != model_identity():
            raise ValueError("correctness predecessor model identity changed")
        recomputed = evaluate_correctness(
            result.get("correctness", {}), result.get("execution_evidence", {})
        )
    else:
        if correctness is None:
            raise ValueError("cost predecessor verification requires correctness result")
        recomputed = evaluate_cost_result(
            result.get("cost", {}),
            result.get("telemetry", {}),
            result.get("execution_evidence", {}),
            label=result.get("label", ""),
            expected_models=correctness["model_identity"],
            expected_descriptor_sha256=correctness["descriptor_sha256"],
        )
        expected_label = next(
            (label for label, run_id in COST_RUN_IDS.items() if run_id == expected_run_id),
            None,
        )
        if expected_label is None or result.get("label") != expected_label:
            raise ValueError("cost predecessor label changed")
        correctness_identity = correctness.get("result_identity")
        if not isinstance(correctness_identity, dict) or not _cost_chain_valid(
            result,
            label=expected_label,
            correctness_identity=correctness_identity,
        ):
            raise ValueError("cost predecessor hash chain changed")
    if any(result.get("gates", {}).get(name) is not passed for name, passed in recomputed.items()):
        raise ValueError("predecessor core gates do not recompute")
    if not all(result.get("gates", {}).values()):
        raise ValueError("predecessor stored gates did not all pass")
    return result, identity


def _common_verified_inputs(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": verify_source(plan["source"]["commit"]),
        "prior_evidence": verify_qualified_prior_evidence(),
        "dependencies": verify_dependencies(),
        "environment": verify_environment(),
    }


def execute_correctness_run(plan: dict[str, Any], plan_identity: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    common = _common_verified_inputs(plan)
    codec, input_evidence = load_codec_input()
    native_before = native_vla_modules_loaded()
    with optimizer_construction_guard() as optimizer_evidence:
        device = torch.device("cuda", 0)
        raw = (
            torch.from_numpy(np.array(codec, copy=True))
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        tokenizer = (
            ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device=device, dtype=torch.float32)
        )
        descriptors = tokenizer(raw)
        descriptor_sha256 = tensor_sha256(descriptors)
        correctness = correctness_probe_from_descriptors(descriptors)
        tokenizer_identity = {
            "parameters": trainable_parameter_count(tokenizer),
            "state_dict_sha256": state_dict_sha256(tokenizer),
        }
        if tokenizer_identity != plan["models"]["tokenizer"]:
            raise RuntimeError("correctness tokenizer identity differs from resolved plan")
        if correctness["condition_identity"] != plan["models"]["conditions"]:
            raise RuntimeError("correctness condition identity differs from resolved plan")
    execution = {
        "native_vla_modules": {
            "before": native_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
        "closed_routes": assert_closed_routes_absent(),
    }
    gates = evaluate_correctness(correctness, execution)
    gates["tokenizer_and_all_condition_identities_match_plan"] = True
    gates["source_input_dependency_runtime_and_prior_identity"] = True
    wall_seconds = time.monotonic() - started
    gates["wall_time_within_limit"] = wall_seconds <= MAX_CORRECTNESS_WALL_SECONDS
    return {
        "kind": "correctness",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": CORRECTNESS_RUN_ID,
        "source_commit": common["source"]["commit"],
        "source": common["source"],
        "plan": plan_identity,
        "prior_evidence": common["prior_evidence"],
        "dependencies": common["dependencies"],
        "environment": common["environment"],
        "input": input_evidence,
        "model_identity": plan["models"],
        "descriptor_sha256": descriptor_sha256,
        "correctness": correctness,
        "execution_evidence": execution,
        "gates": gates,
        "optimizer_steps": optimizer_evidence["steps_observed"],
        "vla_model_instantiated": bool(
            execution["native_vla_modules"]["before"] or execution["native_vla_modules"]["after"]
        ),
        "cost_allowed": all(gates.values()),
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "model-specific fixed-weight online recurrence correctness only",
    }


def execute_cost_run(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
    *,
    label: str,
    correctness_path: Path,
    correctness_sha256: str,
    predecessor_path: Path | None,
    predecessor_sha256: str | None,
) -> dict[str, Any]:
    if label not in COST_RUN_IDS:
        raise ValueError("cost label must be one of A-E")
    started = time.monotonic()
    common = _common_verified_inputs(plan)
    correctness, correctness_identity = verify_passed_component(
        correctness_path,
        correctness_sha256,
        expected_path=ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json",
        expected_run_id=CORRECTNESS_RUN_ID,
        expected_kind="correctness",
        plan_identity=plan_identity,
        source_commit=common["source"]["commit"],
    )
    correctness["result_identity"] = correctness_identity
    labels = tuple(COST_RUN_IDS)
    index = labels.index(label)
    predecessor_identity = None
    predecessor_result = None
    if index == 0:
        if predecessor_path is not None or predecessor_sha256 is not None:
            raise ValueError("cost A must not receive a predecessor")
    else:
        if predecessor_path is None or predecessor_sha256 is None:
            raise ValueError("cost B-E require the immediately preceding result")
        previous = labels[index - 1]
        predecessor_result, predecessor_identity = verify_passed_component(
            predecessor_path,
            predecessor_sha256,
            expected_path=ARTIFACT_ROOT / COST_RUN_IDS[previous] / "RESULT.json",
            expected_run_id=COST_RUN_IDS[previous],
            expected_kind="cost",
            plan_identity=plan_identity,
            source_commit=common["source"]["commit"],
            correctness=correctness,
        )
    predecessor_chain = (
        []
        if predecessor_result is None
        else [*predecessor_result["predecessor_chain"], predecessor_identity]
    )
    telemetry_before = gpu_telemetry()
    if not _telemetry_point_valid(telemetry_before):
        raise RuntimeError("cost process requires isolated finite GPU0 telemetry")
    codec, input_evidence = load_codec_input()
    native_before = native_vla_modules_loaded()
    with optimizer_construction_guard() as optimizer_evidence:
        cost = execute_cost_measurements(
            codec,
            label=label,
            expected_models=correctness["model_identity"],
            expected_descriptor_sha256=correctness["descriptor_sha256"],
        )
    gc.collect()
    torch.cuda.empty_cache()
    telemetry_after = gpu_telemetry()
    telemetry = {"before": telemetry_before, "after": telemetry_after}
    execution = {
        "native_vla_modules": {
            "before": native_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
        "closed_routes": assert_closed_routes_absent(),
    }
    gates = evaluate_cost_result(
        cost,
        telemetry,
        execution,
        label=label,
        expected_models=correctness["model_identity"],
        expected_descriptor_sha256=correctness["descriptor_sha256"],
    )
    gates["correctness_and_immediate_predecessor_identity_verified"] = True
    gates["source_input_dependency_runtime_and_prior_identity"] = True
    wall_seconds = time.monotonic() - started
    gates["wall_time_within_limit"] = wall_seconds <= MAX_COST_WALL_SECONDS
    return {
        "kind": "cost",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": COST_RUN_IDS[label],
        "label": label,
        "source_commit": common["source"]["commit"],
        "source": common["source"],
        "plan": plan_identity,
        "correctness_predecessor": correctness_identity,
        "cost_predecessor": predecessor_identity,
        "predecessor_chain": predecessor_chain,
        "prior_evidence": common["prior_evidence"],
        "dependencies": common["dependencies"],
        "environment": common["environment"],
        "input": input_evidence,
        "process": {"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv},
        "telemetry": telemetry,
        "cost": cost,
        "execution_evidence": execution,
        "engineering_anomalies": [],
        "gates": gates,
        "optimizer_steps": optimizer_evidence["steps_observed"],
        "vla_model_instantiated": bool(
            execution["native_vla_modules"]["before"] or execution["native_vla_modules"]["after"]
        ),
        "next_cost_allowed": False,
        "aggregate_allowed": False,
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "fixed-input specified-GPU online implementation cost only",
    }


def execute_aggregate_run(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
    *,
    correctness_path: Path,
    correctness_sha256: str,
    cost_paths: tuple[Path, ...],
    cost_sha256s: tuple[str, ...],
) -> dict[str, Any]:
    if len(cost_paths) != 5 or len(cost_sha256s) != 5:
        raise ValueError("aggregate requires five cost paths and hashes")
    started = time.monotonic()
    source = verify_source(plan["source"]["commit"])
    correctness, correctness_identity = verify_passed_component(
        correctness_path,
        correctness_sha256,
        expected_path=ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json",
        expected_run_id=CORRECTNESS_RUN_ID,
        expected_kind="correctness",
        plan_identity=plan_identity,
        source_commit=source["commit"],
    )
    correctness["result_identity"] = correctness_identity
    costs = []
    identities = []
    for label, path, digest in zip(COST_RUN_IDS, cost_paths, cost_sha256s, strict=True):
        result, identity = verify_passed_component(
            path,
            digest,
            expected_path=ARTIFACT_ROOT / COST_RUN_IDS[label] / "RESULT.json",
            expected_run_id=COST_RUN_IDS[label],
            expected_kind="cost",
            plan_identity=plan_identity,
            source_commit=source["commit"],
            correctness=correctness,
        )
        costs.append(result)
        identities.append(identity)
    gates, decision = evaluate_aggregate_result(correctness, costs, identities)
    native = native_vla_modules_loaded()
    gates["aggregate_process_native_vla_modules_absent"] = not native
    wall_seconds = time.monotonic() - started
    gates["wall_time_within_limit"] = wall_seconds <= MAX_AGGREGATE_WALL_SECONDS
    return {
        "kind": "aggregate",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": AGGREGATE_RUN_ID,
        "source_commit": source["commit"],
        "source": source,
        "plan": plan_identity,
        "component_results": {
            "correctness": correctness_identity,
            "cost": identities,
        },
        "decision": decision,
        "gates": gates,
        "online_mamba_engineering_scaling_advantage": all(gates.values())
        and decision["online_mamba_engineering_scaling_advantage"],
        "optimizer_steps": 0,
        "vla_model_instantiated": bool(native),
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": (
            "fixed Arena codec descriptor and specified RTX6000 Ada FP32 online implementation "
            "scaling only; no action quality, history use, VLA capability, or general advantage claim"
        ),
    }


def _write_mode_result(run_id: str, execute: Any) -> int:
    started = time.monotonic()
    try:
        result = execute()
    except Exception as error:
        failure = {
            "kind": "failure",
            "status": "failed",
            "experiment_id": EXPERIMENT,
            "run_id": run_id,
            "error": f"{type(error).__name__}: {error}",
            "gates": {"execution_completed": False},
            "optimizer_steps": 0,
            "vla_model_instantiated": bool(native_vla_modules_loaded()),
            "training_allowed": False,
            "wall_seconds": time.monotonic() - started,
        }
        write_result_once(run_id, failure)
        raise
    identity = write_result_once(run_id, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": identity,
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "passed" else 2


def dry_run_payload() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "status": "dry_run_only",
        "closed_routes": assert_closed_routes_absent(),
        "plan": resolved_plan(commit),
        "cost_process_interface": {
            label: {
                "run_id": run_id,
                "condition_order": list(CONDITION_ORDERS[label]),
                "length_order": list(LENGTH_ORDERS[label]),
                "predecessor": None if label == "A" else COST_RUN_IDS[chr(ord(label) - 1)],
            }
            for label, run_id in COST_RUN_IDS.items()
        },
        "aggregate_interface": {
            "run_id": AGGREGATE_RUN_ID,
            "requires": list(COST_RUN_IDS.values()),
        },
        "formal_execution_permitted_after_committed_plan": True,
        "result_schema": {
            "status": "ready_not_run",
            "correctness_not_run": True,
            "cost_not_run": True,
            "cost_not_implemented": False,
            "aggregate_not_run": True,
            "aggregate_not_implemented": False,
            "training_allowed": False,
            "vla_model_instantiated": False,
            "optimizer_steps": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare-plan", type=Path)
    mode.add_argument("--correctness", action="store_true")
    for label in COST_RUN_IDS:
        mode.add_argument(f"--cost-{label.lower()}", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--correctness-result", type=Path)
    parser.add_argument("--expected-correctness-sha256")
    parser.add_argument("--predecessor-result", type=Path)
    parser.add_argument("--expected-predecessor-sha256")
    for label in COST_RUN_IDS:
        parser.add_argument(f"--cost-{label.lower()}-result", type=Path)
        parser.add_argument(f"--expected-cost-{label.lower()}-sha256")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run_payload(), indent=2, sort_keys=True))
        return 0
    if args.prepare_plan is not None:
        if args.plan is not None or args.expected_plan_sha256 is not None:
            parser.error("--prepare-plan cannot be combined with resolved plan arguments")
        print(json.dumps(prepare_plan(args.prepare_plan), sort_keys=True), flush=True)
        return 0
    if args.plan is None or args.expected_plan_sha256 is None:
        parser.error("formal execution requires --plan and --expected-plan-sha256")
    plan, plan_identity = verify_plan(args.plan, args.expected_plan_sha256)
    if args.correctness:
        return _write_mode_result(
            CORRECTNESS_RUN_ID,
            lambda: execute_correctness_run(plan, plan_identity),
        )
    selected = next(
        (label for label in COST_RUN_IDS if getattr(args, f"cost_{label.lower()}")),
        None,
    )
    if selected is not None:
        if args.correctness_result is None or args.expected_correctness_sha256 is None:
            parser.error("cost requires correctness result path and SHA-256")
        return _write_mode_result(
            COST_RUN_IDS[selected],
            lambda: execute_cost_run(
                plan,
                plan_identity,
                label=selected,
                correctness_path=args.correctness_result,
                correctness_sha256=args.expected_correctness_sha256,
                predecessor_path=args.predecessor_result,
                predecessor_sha256=args.expected_predecessor_sha256,
            ),
        )
    aggregate_values = [
        args.correctness_result,
        args.expected_correctness_sha256,
        *[getattr(args, f"cost_{label.lower()}_result") for label in COST_RUN_IDS],
        *[getattr(args, f"expected_cost_{label.lower()}_sha256") for label in COST_RUN_IDS],
    ]
    if any(value is None for value in aggregate_values):
        parser.error("aggregate requires correctness and all five cost result paths and SHA-256s")
    return _write_mode_result(
        AGGREGATE_RUN_ID,
        lambda: execute_aggregate_run(
            plan,
            plan_identity,
            correctness_path=args.correctness_result,
            correctness_sha256=args.expected_correctness_sha256,
            cost_paths=tuple(
                getattr(args, f"cost_{label.lower()}_result") for label in COST_RUN_IDS
            ),
            cost_sha256s=tuple(
                getattr(args, f"expected_cost_{label.lower()}_sha256") for label in COST_RUN_IDS
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
