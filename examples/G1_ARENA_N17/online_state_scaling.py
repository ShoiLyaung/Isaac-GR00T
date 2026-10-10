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
import hashlib
import json
from pathlib import Path
import subprocess
import sys
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
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "g1-arena-online-state-scaling-r1"
BASE_SOURCE_COMMIT = "1a441889907e0c1c3831f717f2748918c8d6877c"
ARTIFACT_ROOT = Path("/home/sl/research/icra27-artifacts/g1-arena-online-state-scaling-r1")
PLAN_PATH = ARTIFACT_ROOT / "plan-r1/PLAN.json"
CORRECTNESS_RUN_ID = "correctness-gpu0-r1"
COST_RUN_IDS = {label: f"cost-gpu0-r1{label.lower()}" for label in "ABCDE"}
AGGREGATE_RUN_ID = "aggregate-r1"

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
PROCESS_SEEDS = {label: 230923 + offset for offset, label in enumerate(COST_RUN_IDS)}
SCOPES = ("end_to_end", "temporal_only")
WARMUP_SAMPLES = 20
MEASURED_SAMPLES = 100
FIXED_CHUNK_SIZE = 73
GRADIENT_LENGTH = 128
MAX_ORACLE_ABS = 1e-4
MAX_TRANSFORMER_CONTROL_ABS = 1e-5
MAX_DIRECTIONAL_GRAD_RELATIVE_ERROR = 0.05
MAX_WALL_SECONDS = 20 * 60
MAX_ARTIFACT_BYTES = 1 << 30

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
    "measurement": "new implementation required before formal cost launch",
}
IMPLEMENTATION_STATUS = {
    "preallocated_control": True,
    "independent_correctness": True,
    "cost_cli_contract": True,
    "cost_measurement": False,
    "aggregate_decision": False,
    "formal_gpu_run_allowed": False,
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
            "functional_mamba": "two recurrent Mamba2 residual layers; fixed recurrent state",
            "persistent_gru": "two d256 GRU residual layers; fixed hidden state",
            "transformer_functional_cat": (
                "two-layer h8 d256 causal attention; live KV grows with T and each step cats"
            ),
            "transformer_preallocated": (
                "same Transformer weights/equations; fixed T925 buffers and live-prefix attention"
            ),
            "current_descriptor": "identity temporal floor",
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

    reset_state = _state_clone(continuous_state)
    reset_returned = reset_state.reset()
    reset_index_zero = reset_state.observations == 0
    reset_output, reset_final = online_stream(name, module, descriptors, reset_returned)

    initial = new_state(name, module, int(descriptors.shape[0]))
    _, one = online_step(name, module, descriptors[:, 0], initial)
    probe_input = _state_clone(one)
    before = _state_clone(probe_input)
    ids_before = _state_storage_ids(probe_input)
    _, probe_output = online_step(name, module, descriptors[:, min(1, length - 1)], probe_input)
    mutable_expected = name == "transformer_preallocated"
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
            "episode_reset_exact": reset_index_zero
            and torch.equal(continuous, reset_output)
            and _state_equal(continuous_state, reset_final),
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
    name: str, module: nn.Module, descriptors: torch.Tensor, output: torch.Tensor, state: Any
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
        "formal_execution_permitted": False,
        "result_schema": {
            "status": "engineering_incomplete",
            "correctness_not_run": True,
            "cost_not_implemented": True,
            "aggregate_not_implemented": True,
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
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run_payload(), indent=2, sort_keys=True))
        return 0
    if args.prepare_plan:
        raise RuntimeError(
            "plan materialization is disabled until the cost measurement implementation is complete"
        )
    if args.correctness:
        raise RuntimeError(
            "formal correctness output is disabled until this source milestone is committed and a "
            "complete resolved plan can be materialized"
        )
    if args.aggregate:
        raise RuntimeError("aggregate is disabled until all five cost processes are implemented")
    raise RuntimeError(
        "formal cost execution is disabled: the frozen A-E interface exists, but measurement is pending"
    )


if __name__ == "__main__":
    raise SystemExit(main())
