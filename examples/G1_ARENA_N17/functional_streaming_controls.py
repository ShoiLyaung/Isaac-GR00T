#!/usr/bin/env python3
"""Run the frozen Arena functional streaming-control comparison.

Formal GPU work is split into one correctness process, four ordered cost
processes, and one aggregate process.  The module is also importable on CPU so
the protocol and gate logic can be tested without constructing Mamba2.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator

import gr00t
import gr00t.experimental
from gr00t.experimental.arena_codec_streaming_backbone import (
    CODEC_FIELDS,
    GRID_HEIGHT,
    GRID_WIDTH,
    LAYERS,
    SEED,
    WIDTH,
    ArenaCodecSpatialTokenizer,
    CausalTransformerBackbone,
    trainable_parameter_count,
)
import gr00t.experimental.arena_functional_streaming_controls as controls_module
from gr00t.experimental.arena_functional_mamba_transition import (
    FunctionalMamba2Backbone,
    FunctionalMamba2State,
)
from gr00t.experimental.arena_functional_streaming_controls import (
    EXPECTED_TEMPORAL_PARAMETERS,
    CurrentDescriptorState,
    PersistentGRUBackbone,
    PersistentGRUState,
    StreamingCurrentDescriptorBackbone,
    StreamingKVState,
    StreamingKVTransformerBackbone,
    state_is_finite,
    state_tensors,
    states_equal,
    storage_identity,
    transformer_kv_expected_bytes,
    unique_storage_state_bytes,
    validate_frozen_parameter_counts,
)
import numpy as np
import torch
from torch import nn

import functional_mamba_transition as predecessor_runner


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "g1-arena-functional-streaming-controls-r1"
BASE_SOURCE_COMMIT = "2c88d3f781ba545416bd7991fbe372800b812465"
ARTIFACT_ROOT = Path(
    "/home/sl/research/icra27-artifacts/g1-arena-functional-streaming-controls-r1"
)
PLAN_PATH = ARTIFACT_ROOT / "plan-r1/PLAN.json"
CORRECTNESS_RUN_ID = "correctness-gpu0-r1"
COST_RUN_IDS = {
    "A": "cost-gpu0-r1a",
    "B": "cost-gpu0-r1b",
    "C": "cost-gpu0-r1c",
    "D": "cost-gpu0-r1d",
}
AGGREGATE_RUN_ID = "aggregate-r1"

CONDITIONS = (
    "functional_mamba",
    "persistent_gru",
    "streaming_kv_transformer",
    "current_descriptor",
)
CONDITION_ORDERS = {
    "A": CONDITIONS,
    "B": CONDITIONS[1:] + CONDITIONS[:1],
    "C": CONDITIONS[2:] + CONDITIONS[:2],
    "D": CONDITIONS[3:] + CONDITIONS[:3],
}
LENGTHS = (128, 256, 512, 925)
LENGTH_ORDERS = {
    "A": LENGTHS,
    "B": LENGTHS[1:] + LENGTHS[:1],
    "C": LENGTHS[2:] + LENGTHS[:2],
    "D": LENGTHS[3:] + LENGTHS[:3],
}
PROCESS_SEEDS = {"A": 230923, "B": 230924, "C": 230925, "D": 230926}
SCOPES = ("end_to_end", "temporal_only")
WARMUP_SAMPLES = 5
MEASURED_SAMPLES = 50
BOOTSTRAP_DRAWS = 10_000
CORRECTNESS_LENGTH = 128
CORRECTNESS_CHUNKS = (47, 81)
CAUSAL_PREFIX = 64

MAX_PARITY_ABS = 1e-4
MAX_FIXED_STATE_GROWTH = 0.01
MAX_KV_RATIO_RELATIVE_ERROR = 0.01
MAX_MAMBA_PEAK_FRACTION = 0.60
MAX_MAMBA_PREFIX_GROWTH_MULTIPLIER = 1.25
MIN_RELATIVE_GROWTH_POINT = 0.10
MIN_RELATIVE_GROWTH_CI95_LOWER = 0.05
MIN_TRANSFORMER_TO_MAMBA_GROWTH = 1.10
MAX_MAMBA_NEXT_STEP_P95_MS = 5.0
MAX_START_TEMPERATURE_C = 65
MAX_WARM_TEMPERATURE_C = 80
MIN_WARM_SM_CLOCK_MHZ = 2000
MAX_WALL_SECONDS = 20 * 60
MAX_ARTIFACT_BYTES = 256 << 20

INPUT_PATH = predecessor_runner.INPUT_PATH
INPUT_BYTES = predecessor_runner.INPUT_BYTES
INPUT_SHA256 = predecessor_runner.INPUT_SHA256
INPUT_FIELD = predecessor_runner.INPUT_FIELD
EXPECTED_ARCHIVE_FIELDS = predecessor_runner.EXPECTED_ARCHIVE_FIELDS
INPUT_SHAPE = predecessor_runner.INPUT_SHAPE
FROZEN_DEPENDENCY = predecessor_runner.FROZEN_DEPENDENCY
FROZEN_RUNTIME = predecessor_runner.FROZEN_RUNTIME

FUNCTIONAL_AGGREGATE_PATH = (
    Path("/home/sl/research/icra27-artifacts/g1-arena-functional-mamba-transition-r1")
    / "aggregate-r2/RESULT.json"
)
FUNCTIONAL_RECHECK_PATH = FUNCTIONAL_AGGREGATE_PATH.with_name("RESULT_RECHECK.json")
FROZEN_PRIOR_EVIDENCE = {
    "functional_aggregate": {
        "path": str(FUNCTIONAL_AGGREGATE_PATH),
        "bytes": 9_249,
        "sha256": "1cc3cebbf621bca0619eaef48f687c91f2bdd8dd39ac377c048d4c8a3813d653",
    },
    "functional_independent_recheck": {
        "path": str(FUNCTIONAL_RECHECK_PATH),
        "bytes": 2_134,
        "sha256": "6581ff40fb90f4d795b40a5490db0438c48a9621ce99faa604d409711f61387d",
    },
}

NATIVE_VLA_MODULE_PREFIXES = predecessor_runner.NATIVE_VLA_MODULE_PREFIXES
SOURCE_IDENTITY_PATHS = (
    "gr00t/experimental/arena_codec_streaming_backbone.py",
    "gr00t/experimental/arena_functional_mamba_transition.py",
    "gr00t/experimental/arena_functional_streaming_controls.py",
    "examples/G1_ARENA_N17/functional_streaming_controls.py",
    "tests/gr00t/experimental/test_arena_functional_streaming_controls.py",
    "tests/examples/test_g1_arena_functional_streaming_controls.py",
)

FROZEN_PROTOCOL = {
    "seed": SEED,
    "dtype": "float32",
    "batch_size": 1,
    "camera_count": 1,
    "grid": [GRID_HEIGHT, GRID_WIDTH, CODEC_FIELDS],
    "width": WIDTH,
    "layers": LAYERS,
    "lengths": list(LENGTHS),
    "conditions": list(CONDITIONS),
    "condition_orders": {key: list(value) for key, value in CONDITION_ORDERS.items()},
    "length_orders": {key: list(value) for key, value in LENGTH_ORDERS.items()},
    "process_seeds": PROCESS_SEEDS,
    "scopes": list(SCOPES),
    "warmup_samples": WARMUP_SAMPLES,
    "measured_samples": MEASURED_SAMPLES,
    "percentile": "numpy_linear",
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "temporal_parameters": EXPECTED_TEMPORAL_PARAMETERS,
    "transformer": {
        "heads": 8,
        "feedforward_width": 320,
        "norm": "pre_norm",
        "position": "shared_sinusoidal",
        "kv_growth": "functional_cat_each_step_no_preallocation",
        "new_trainable_parameters": 0,
    },
    "memory": {
        "allocator_order": [
            "gc",
            "empty_cache",
            "model_input_state_resident",
            "synchronize",
            "reset_peak_memory_stats",
        ],
        "reported": [
            "baseline_allocated",
            "baseline_reserved",
            "absolute_peak_allocated",
            "delta_peak_allocated",
            "absolute_peak_reserved",
            "delta_peak_reserved",
        ],
        "state_bytes": "unique_storage_device_data_ptr_nbytes_alias_rejected",
    },
    "timing": {
        "input_and_model_resident": True,
        "pre_and_post_sample_cuda_synchronize": True,
        "mode": "torch.inference_mode",
        "sample": "fresh_state_complete_prefix",
        "end_to_end": "encode_frame_plus_temporal_step_plus_new_state",
        "temporal_only": "resident_descriptor_plus_temporal_step_plus_new_state",
    },
    "next_step": {
        "prefill_each_length": True,
        "probe": "repeat_final_resident_input",
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "kernel_profiles": 1,
    },
}

FROZEN_THRESHOLDS = {
    "parity_max_abs": MAX_PARITY_ABS,
    "fixed_state_growth_fraction": MAX_FIXED_STATE_GROWTH,
    "kv_ratio_relative_error": MAX_KV_RATIO_RELATIVE_ERROR,
    "mamba_t925_peak_fraction_of_transformer": MAX_MAMBA_PEAK_FRACTION,
    "mamba_prefix_growth_multiplier": MAX_MAMBA_PREFIX_GROWTH_MULTIPLIER,
    "transformer_to_mamba_temporal_growth": MIN_TRANSFORMER_TO_MAMBA_GROWTH,
    "bootstrap_point": MIN_RELATIVE_GROWTH_POINT,
    "bootstrap_ci95_lower": MIN_RELATIVE_GROWTH_CI95_LOWER,
    "mamba_t925_next_step_host_p95_ms": MAX_MAMBA_NEXT_STEP_P95_MS,
    "start_temperature_c": MAX_START_TEMPERATURE_C,
    "warm_temperature_c": MAX_WARM_TEMPERATURE_C,
    "warm_sm_clock_mhz": MIN_WARM_SM_CLOCK_MHZ,
    "wall_seconds": MAX_WALL_SECONDS,
    "artifact_bytes": MAX_ARTIFACT_BYTES,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, payload: dict[str, Any]) -> int:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("JSON artifact exceeds 256 MiB")
    if directory_bytes(ARTIFACT_ROOT) + len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("experiment artifact root would exceed 256 MiB")
    with path.open("xb") as stream:
        stream.write(encoded)
    return len(encoded)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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
    original_init = optimizer.__init__
    evidence: dict[str, Any] = {"constructor_calls": [], "steps_observed": None}

    def blocked_init(instance, *args, **kwargs):
        del args, kwargs
        evidence["constructor_calls"].append(type(instance).__qualname__)
        raise RuntimeError("optimizer construction is forbidden in the fixed-weight protocol")

    optimizer.__init__ = blocked_init
    try:
        yield evidence
    finally:
        optimizer.__init__ = original_init
        evidence["steps_observed"] = 0 if not evidence["constructor_calls"] else None


def resolved_plan(source_commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "source": {
            "root": str(REPO),
            "commit": source_commit,
            "base_commit": BASE_SOURCE_COMMIT,
        },
        "prior_evidence": copy.deepcopy(FROZEN_PRIOR_EVIDENCE),
        "input": {
            "path": str(INPUT_PATH),
            "bytes": INPUT_BYTES,
            "sha256": INPUT_SHA256,
            "field": INPUT_FIELD,
            "shape": list(INPUT_SHAPE),
            "dtype": "float32",
        },
        "dependency": copy.deepcopy(FROZEN_DEPENDENCY),
        "runtime": copy.deepcopy(FROZEN_RUNTIME),
        "protocol": copy.deepcopy(FROZEN_PROTOCOL),
        "thresholds": copy.deepcopy(FROZEN_THRESHOLDS),
        "outputs": {
            "correctness": str(ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json"),
            "cost": {
                label: str(ARTIFACT_ROOT / run_id / "RESULT.json")
                for label, run_id in COST_RUN_IDS.items()
            },
            "aggregate": str(ARTIFACT_ROOT / AGGREGATE_RUN_ID / "RESULT.json"),
        },
    }


def prepare_plan(path: Path) -> dict[str, Any]:
    if path.resolve() != PLAN_PATH.resolve():
        raise ValueError(f"resolved plan path must be {PLAN_PATH}")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    size = write_new(path, resolved_plan(commit))
    return {"path": str(path.resolve()), "bytes": size, "sha256": sha256(path)}


def verify_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("resolved plan must be inside the experiment artifact root")
    actual_sha256 = sha256(resolved)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ValueError("resolved plan identity changed")
    plan = read_json(resolved)
    commit = plan.get("source", {}).get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("resolved source commit must contain 40 hexadecimal characters")
    try:
        int(commit, 16)
    except ValueError as error:
        raise ValueError("resolved source commit is not hexadecimal") from error
    if plan != resolved_plan(commit):
        raise ValueError("resolved plan contract changed")
    return plan, {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
    }


def _source_file_identity(path: Path, commit: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(REPO.resolve()):
        raise ValueError(f"source escaped repository: {resolved}")
    relative = resolved.relative_to(REPO.resolve()).as_posix()
    committed_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", f"{commit}:{relative}"],
        text=True,
    ).strip()
    working_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "hash-object", str(resolved)],
        text=True,
    ).strip()
    if committed_blob != working_blob:
        raise ValueError(f"source differs from committed blob: {relative}")
    return {
        "path": str(resolved),
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
        "git_blob": working_blob,
    }


def verify_source(expected_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != expected_commit:
        raise ValueError("streaming-control source commit changed")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, commit],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("streaming-control source escaped the exact base ancestry")
    dirty: set[str] = set()
    for extra in ((), ("--cached",)):
        changed = subprocess.check_output(
            ["git", "-C", str(REPO), "diff", *extra, "--name-only", "--", *SOURCE_IDENTITY_PATHS],
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
        raise ValueError(f"streaming-control source is dirty: {sorted(dirty)}")
    expected_module = (
        REPO / "gr00t/experimental/arena_functional_streaming_controls.py"
    ).resolve()
    if Path(controls_module.__file__).resolve() != expected_module:
        raise ValueError("imported streaming-control module escaped source root")
    entries = {
        relative: _source_file_identity(REPO / relative, commit)
        for relative in SOURCE_IDENTITY_PATHS
    }
    return {"root": str(REPO.resolve()), "commit": commit, "entries": entries}


def _verify_material(spec: dict[str, Any], expected_path: Path, label: str) -> dict[str, Any]:
    path = Path(spec["path"]).resolve()
    if path != expected_path.resolve() or not path.is_file():
        raise ValueError(f"{label} path changed or is missing")
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"{label} identity changed")
    return {**spec, "verified": True}


def verify_prior_evidence() -> dict[str, Any]:
    aggregate_identity = _verify_material(
        FROZEN_PRIOR_EVIDENCE["functional_aggregate"],
        FUNCTIONAL_AGGREGATE_PATH,
        "functional aggregate",
    )
    recheck_identity = _verify_material(
        FROZEN_PRIOR_EVIDENCE["functional_independent_recheck"],
        FUNCTIONAL_RECHECK_PATH,
        "functional independent recheck",
    )
    aggregate = read_json(FUNCTIONAL_AGGREGATE_PATH)
    recheck = read_json(FUNCTIONAL_RECHECK_PATH)
    if (
        aggregate.get("status") != "passed"
        or aggregate.get("source_commit") != BASE_SOURCE_COMMIT
        or aggregate.get("matched_functional_streaming_cost_experiment_allowed") is not True
        or aggregate.get("optimizer_steps") != 0
        or aggregate.get("vla_model_instantiated") is not False
        or not all(aggregate.get("gates", {}).values())
    ):
        raise ValueError("functional aggregate no longer authorizes the matched cost experiment")
    if (
        recheck.get("source_commit") != BASE_SOURCE_COMMIT
        or recheck.get("aggregate_status") != "passed"
        or recheck.get("all_stored_gates_passed") is not True
        or recheck.get("matched_cost_experiment_allowed") is not True
        or recheck.get("files", {}).get("aggregate", {}).get("sha256")
        != FROZEN_PRIOR_EVIDENCE["functional_aggregate"]["sha256"]
    ):
        raise ValueError("functional independent recheck contract changed")
    return {
        "functional_aggregate": aggregate_identity,
        "functional_independent_recheck": recheck_identity,
    }


def load_codec_input() -> tuple[np.ndarray, dict[str, Any]]:
    return predecessor_runner.load_codec_input()


def verify_dependencies() -> dict[str, Any]:
    return predecessor_runner.verify_dependencies()


def verify_environment() -> dict[str, Any]:
    return predecessor_runner.verify_environment()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = (first.float() - second.float()).abs()
    if not bool(torch.isfinite(difference).all()):
        return float("inf")
    return float(difference.max().item())


def _state_storage_disjoint(first: Any, second: Any) -> bool:
    first_ids = {storage_identity(tensor) for tensor in state_tensors(first)}
    second_ids = {storage_identity(tensor) for tensor in state_tensors(second)}
    return not first_ids & second_ids


def build_condition(name: str, *, seed: int = SEED) -> nn.Module:
    if name == "functional_mamba":
        return FunctionalMamba2Backbone(seed=seed)
    if name == "persistent_gru":
        return PersistentGRUBackbone(seed=seed)
    if name == "streaming_kv_transformer":
        return StreamingKVTransformerBackbone(seed=seed)
    if name == "current_descriptor":
        return StreamingCurrentDescriptorBackbone()
    raise ValueError(f"unknown streaming condition: {name}")


def new_condition_state(name: str, module: nn.Module, batch_size: int) -> Any:
    if name == "functional_mamba":
        assert isinstance(module, FunctionalMamba2Backbone)
        return module.new_functional_state(batch_size)
    if name == "persistent_gru":
        assert isinstance(module, PersistentGRUBackbone)
        return module.new_persistent_state(batch_size)
    if name == "streaming_kv_transformer":
        assert isinstance(module, StreamingKVTransformerBackbone)
        return module.new_kv_state(batch_size)
    if name == "current_descriptor":
        assert isinstance(module, StreamingCurrentDescriptorBackbone)
        return module.new_state(batch_size)
    raise ValueError(f"unknown streaming condition: {name}")


def condition_step(
    name: str,
    module: nn.Module,
    descriptor: torch.Tensor,
    state: Any,
) -> tuple[torch.Tensor, Any]:
    if name == "functional_mamba":
        assert isinstance(module, FunctionalMamba2Backbone)
        assert isinstance(state, FunctionalMamba2State)
        return module.functional_step(descriptor, state)
    if name == "persistent_gru":
        assert isinstance(module, PersistentGRUBackbone)
        assert isinstance(state, PersistentGRUState)
        return module.persistent_step(descriptor, state)
    if name == "streaming_kv_transformer":
        assert isinstance(module, StreamingKVTransformerBackbone)
        assert isinstance(state, StreamingKVState)
        return module.streaming_step(descriptor, state)
    if name == "current_descriptor":
        assert isinstance(module, StreamingCurrentDescriptorBackbone)
        assert isinstance(state, CurrentDescriptorState)
        return module.streaming_step(descriptor, state)
    raise ValueError(f"unknown streaming condition: {name}")


def condition_stream(
    name: str,
    module: nn.Module,
    descriptors: torch.Tensor,
    state: Any | None = None,
) -> tuple[torch.Tensor, Any]:
    if name == "functional_mamba":
        assert isinstance(module, FunctionalMamba2Backbone)
        return module.functional_stream(descriptors, state)
    if name == "persistent_gru":
        assert isinstance(module, PersistentGRUBackbone)
        return module.persistent_stream(descriptors, state)
    if name == "streaming_kv_transformer":
        assert isinstance(module, StreamingKVTransformerBackbone)
        return module.streaming_stream(descriptors, state)
    if name == "current_descriptor":
        assert isinstance(module, StreamingCurrentDescriptorBackbone)
        return module.streaming_stream(descriptors, state)
    raise ValueError(f"unknown streaming condition: {name}")


def _mamba_state_max_abs(first: Any, second: Any) -> float:
    values = [
        max_abs(left, right)
        for first_pair, second_pair in zip(first.layers, second.layers, strict=True)
        for left, right in zip(first_pair, second_pair, strict=True)
    ]
    return max(values)


@torch.inference_mode()
def mamba_official_functional_parity(
    module: FunctionalMamba2Backbone,
    descriptors: torch.Tensor,
) -> dict[str, Any]:
    teacher_state = module.new_official_state(int(descriptors.shape[0]))
    functional_state = module.new_functional_state(int(descriptors.shape[0]))
    output_errors = []
    state_errors = []
    inputs_unchanged = True
    state_storage_disjoint = True
    outputs_finite = True
    states_finite = True
    for descriptor in descriptors.unbind(dim=1):
        before = functional_state.clone()
        teacher_output = module.step(descriptor, teacher_state)
        functional_output, next_state = module.functional_step(descriptor, functional_state)
        output_errors.append(max_abs(functional_output, teacher_output))
        state_errors.append(_mamba_state_max_abs(next_state, teacher_state))
        inputs_unchanged = inputs_unchanged and states_equal(functional_state, before)
        state_storage_disjoint = (
            state_storage_disjoint
            and _state_storage_disjoint(functional_state, next_state)
            and unique_storage_state_bytes(next_state) > 0
        )
        outputs_finite = outputs_finite and bool(
            torch.isfinite(functional_output).all() and torch.isfinite(teacher_output).all()
        )
        states_finite = states_finite and state_is_finite(next_state) and all(
            bool(torch.isfinite(tensor).all())
            for pair in teacher_state.layers
            for tensor in pair
        )
        functional_state = next_state
    return {
        "frames": int(descriptors.shape[1]),
        "output_max_abs": max(output_errors),
        "state_max_abs": max(state_errors),
        "all_outputs_finite": outputs_finite,
        "all_states_finite": states_finite,
        "functional_input_state_unchanged": inputs_unchanged,
        "input_output_state_storage_disjoint": state_storage_disjoint,
        "official_observations": teacher_state.observations,
        "functional_observations": functional_state.observations,
        "functional_state_bytes": unique_storage_state_bytes(functional_state),
    }


@torch.inference_mode()
def recurrence_invariants(
    name: str,
    module: nn.Module,
    descriptors: torch.Tensor,
) -> dict[str, Any]:
    continuous, continuous_state = condition_stream(name, module, descriptors)
    split = CORRECTNESS_CHUNKS[0]
    first, chunk_state = condition_stream(name, module, descriptors[:, :split])
    second, chunk_state = condition_stream(name, module, descriptors[:, split:], chunk_state)
    chunked = torch.cat((first, second), dim=1)
    repeat, repeat_state = condition_stream(name, module, descriptors)
    reset_input = continuous_state.reset()
    reset_output, reset_state = condition_stream(name, module, descriptors, reset_input)

    edited = descriptors.clone()
    edited[:, CAUSAL_PREFIX:] = edited[:, CAUSAL_PREFIX:] + 0.125
    edited_output, _ = condition_stream(name, module, edited)

    initial = new_condition_state(name, module, int(descriptors.shape[0]))
    _, one_step = condition_step(name, module, descriptors[:, 0], initial)
    step_input = one_step.clone()
    before = step_input.clone()
    _, step_output = condition_step(name, module, descriptors[:, 1], step_input)
    input_unchanged = states_equal(step_input, before)
    input_output_disjoint = _state_storage_disjoint(step_input, step_output)
    state_bytes = unique_storage_state_bytes(continuous_state)
    expected_kv_bytes = (
        transformer_kv_expected_bytes(
            batch_size=int(descriptors.shape[0]),
            observations=int(descriptors.shape[1]),
        )
        if name == "streaming_kv_transformer"
        else None
    )
    return {
        "output_shape": list(continuous.shape),
        "output_finite": bool(torch.isfinite(continuous).all()),
        "state_finite": state_is_finite(continuous_state),
        "final_observations": continuous_state.observations,
        "continuous_chunk_exact": torch.equal(continuous, chunked)
        and states_equal(continuous_state, chunk_state),
        "repeat_exact": torch.equal(continuous, repeat)
        and states_equal(continuous_state, repeat_state),
        "reset_exact": (
            reset_input.observations == 0
            and torch.equal(continuous, reset_output)
            and states_equal(continuous_state, reset_state)
        ),
        "suffix_edit_prefix_exact": torch.equal(
            continuous[:, :CAUSAL_PREFIX],
            edited_output[:, :CAUSAL_PREFIX],
        ),
        "input_state_unchanged": input_unchanged,
        "input_output_state_storage_disjoint": input_output_disjoint,
        "state_unique_storage_bytes": state_bytes,
        "expected_transformer_kv_bytes": expected_kv_bytes,
        "transformer_kv_bytes_exact": expected_kv_bytes is None or state_bytes == expected_kv_bytes,
        "episode_reset_clears_observations": continuous_state.reset().observations == 0,
    }


@torch.inference_mode()
def correctness_probe(codec: np.ndarray) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    raw = torch.from_numpy(np.array(codec[:CORRECTNESS_LENGTH], copy=True)).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device=device, dtype=torch.float32)
    descriptors = tokenizer(raw)
    modules = {
        name: build_condition(name, seed=SEED).eval().to(device=device, dtype=torch.float32)
        for name in CONDITIONS
    }
    counts = validate_frozen_parameter_counts(modules)
    tokenizer_hash = state_dict_sha256(tokenizer)
    temporal_hashes = {name: state_dict_sha256(module) for name, module in modules.items()}
    transformer_reference = CausalTransformerBackbone(seed=SEED).eval()
    transformer = modules["streaming_kv_transformer"]
    assert isinstance(transformer, StreamingKVTransformerBackbone)
    transformer_no_new_parameters = (
        transformer.state_dict().keys() == transformer_reference.state_dict().keys()
        and temporal_hashes["streaming_kv_transformer"]
        == state_dict_sha256(transformer_reference)
        and len(tuple(transformer.parameters())) == len(tuple(transformer_reference.parameters()))
    )

    mamba = modules["functional_mamba"]
    gru = modules["persistent_gru"]
    assert isinstance(mamba, FunctionalMamba2Backbone)
    assert isinstance(gru, PersistentGRUBackbone)
    mamba_parity = mamba_official_functional_parity(mamba, descriptors)
    gru_full = gru.forward_explicit(descriptors)
    gru_stream, gru_state = gru.persistent_stream(descriptors)
    transformer_full = transformer(descriptors)
    transformer_stream, transformer_state = transformer.streaming_stream(descriptors)
    invariants = {
        name: recurrence_invariants(name, module, descriptors)
        for name, module in modules.items()
    }
    parameter_ids = [id(parameter) for module in modules.values() for parameter in module.parameters()]
    torch.cuda.synchronize(device)
    return {
        "tokenizer": {
            "parameter_count": trainable_parameter_count(tokenizer),
            "state_dict_sha256": tokenizer_hash,
            "descriptor_shape": list(descriptors.shape),
            "descriptor_dtype": str(descriptors.dtype),
            "descriptor_finite": bool(torch.isfinite(descriptors).all()),
            "descriptor_sha256": tensor_sha256(descriptors),
            "shared_descriptor_tensor_for_all_conditions": True,
        },
        "parameters": {
            "expected_temporal_counts": EXPECTED_TEMPORAL_PARAMETERS,
            "actual_temporal_counts": counts,
            "temporal_state_dict_sha256": temporal_hashes,
            "all_parameter_objects_unique": len(parameter_ids) == len(set(parameter_ids)),
            "transformer_reuses_exact_full_parameter_set": transformer_no_new_parameters,
        },
        "functional_mamba": mamba_parity,
        "persistent_gru": {
            "explicit_full_stream_exact": torch.equal(gru_full, gru_stream),
            "max_abs": max_abs(gru_full, gru_stream),
            "output_finite": bool(torch.isfinite(gru_full).all()),
            "final_observations": gru_state.observations,
        },
        "streaming_kv_transformer": {
            "full_stream_max_abs": max_abs(transformer_full, transformer_stream),
            "output_finite": bool(
                torch.isfinite(transformer_full).all()
                and torch.isfinite(transformer_stream).all()
            ),
            "final_observations": transformer_state.observations,
        },
        "invariants": invariants,
    }


def evaluate_correctness_gates(
    correctness: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, bool]:
    parameters = correctness["parameters"]
    mamba = correctness["functional_mamba"]
    gru = correctness["persistent_gru"]
    transformer = correctness["streaming_kv_transformer"]
    invariants = correctness["invariants"]
    invariant_values = list(invariants.values())
    return {
        "source_input_dependency_runtime_and_prior_identity": True,
        "shared_finite_t128_tokenizer_descriptors": (
            correctness["tokenizer"]["descriptor_shape"] == [1, CORRECTNESS_LENGTH, WIDTH]
            and correctness["tokenizer"]["descriptor_dtype"] == "torch.float32"
            and correctness["tokenizer"]["descriptor_finite"]
            and correctness["tokenizer"]["shared_descriptor_tensor_for_all_conditions"]
        ),
        "frozen_temporal_parameter_counts_and_unique_objects": (
            parameters["actual_temporal_counts"] == EXPECTED_TEMPORAL_PARAMETERS
            and parameters["expected_temporal_counts"] == EXPECTED_TEMPORAL_PARAMETERS
            and parameters["all_parameter_objects_unique"]
        ),
        "streaming_transformer_adds_no_trainable_parameters": parameters[
            "transformer_reuses_exact_full_parameter_set"
        ],
        "mamba_functional_official_t128_output_and_state_at_most_1e4": (
            mamba["frames"] == CORRECTNESS_LENGTH
            and mamba["output_max_abs"] <= MAX_PARITY_ABS
            and mamba["state_max_abs"] <= MAX_PARITY_ABS
            and mamba["all_outputs_finite"]
            and mamba["all_states_finite"]
            and mamba["official_observations"] == CORRECTNESS_LENGTH
            and mamba["functional_observations"] == CORRECTNESS_LENGTH
        ),
        "mamba_functional_state_input_immutable_and_storage_disjoint": (
            mamba["functional_input_state_unchanged"]
            and mamba["input_output_state_storage_disjoint"]
        ),
        "gru_explicit_full_and_persistent_step_exact": (
            gru["explicit_full_stream_exact"]
            and gru["max_abs"] == 0.0
            and gru["output_finite"]
            and gru["final_observations"] == CORRECTNESS_LENGTH
        ),
        "transformer_full_causal_and_streaming_kv_t128_at_most_1e4": (
            transformer["full_stream_max_abs"] <= MAX_PARITY_ABS
            and transformer["output_finite"]
            and transformer["final_observations"] == CORRECTNESS_LENGTH
        ),
        "all_conditions_repeat_reset_chunk_suffix_shape_and_finite": all(
            item["output_shape"] == [1, CORRECTNESS_LENGTH, WIDTH]
            and item["output_finite"]
            and item["state_finite"]
            and item["final_observations"] == CORRECTNESS_LENGTH
            and item["continuous_chunk_exact"]
            and item["repeat_exact"]
            and item["reset_exact"]
            and item["suffix_edit_prefix_exact"]
            and item["episode_reset_clears_observations"]
            for item in invariant_values
        ),
        "all_condition_state_inputs_immutable_unique_and_disjoint": all(
            item["input_state_unchanged"]
            and item["input_output_state_storage_disjoint"]
            and item["transformer_kv_bytes_exact"]
            for item in invariant_values
        ),
        "native_vla_modules_absent": not execution["native_vla_modules"]["before"]
        and not execution["native_vla_modules"]["after"],
        "optimizer_steps_zero": not execution["optimizer"]["constructor_calls"]
        and execution["optimizer"]["steps_observed"] == 0,
    }


def _finalize_artifact_gate(
    result: dict[str, Any],
    *,
    artifact_bytes_before: int,
) -> None:
    for _ in range(3):
        result_bytes = len((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        total_bytes = artifact_bytes_before + result_bytes
        result["artifact"] = {
            "root": str(ARTIFACT_ROOT.resolve()),
            "bytes_before_result": artifact_bytes_before,
            "result_bytes": result_bytes,
            "total_bytes": total_bytes,
            "limit_bytes": MAX_ARTIFACT_BYTES,
            "traces_written": 0,
        }
        result["gates"]["artifact_total_at_most_256mib"] = total_bytes <= MAX_ARTIFACT_BYTES
        result["status"] = "passed" if all(result["gates"].values()) else "gate_failed"


def run_correctness(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_bytes_before = directory_bytes(ARTIFACT_ROOT)
    native_before = native_vla_modules_loaded()
    source = verify_source(plan["source"]["commit"])
    prior_evidence = verify_prior_evidence()
    dependencies = verify_dependencies()
    codec, input_evidence = load_codec_input()
    environment = verify_environment()
    with optimizer_construction_guard() as optimizer_evidence:
        correctness = correctness_probe(codec)
    execution = {
        "native_vla_modules": {
            "before": native_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
    }
    gates = evaluate_correctness_gates(correctness, execution)
    wall_seconds = time.monotonic() - started
    gates["wall_time_at_most_20min"] = wall_seconds <= MAX_WALL_SECONDS
    gates["artifact_total_at_most_256mib"] = True
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": CORRECTNESS_RUN_ID,
        "source_commit": source["commit"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source": source,
        "plan": plan_identity,
        "prior_evidence": prior_evidence,
        "input": input_evidence,
        "dependencies": dependencies,
        "environment": environment,
        "protocol": FROZEN_PROTOCOL,
        "thresholds": FROZEN_THRESHOLDS,
        "correctness": correctness,
        "execution_evidence": execution,
        "gates": gates,
        "optimizer_steps": optimizer_evidence["steps_observed"],
        "vla_model_instantiated": bool(
            execution["native_vla_modules"]["before"]
            or execution["native_vla_modules"]["after"]
        ),
        "cost_allowed": all(gates.values()),
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "fixed-weight VLA-free streaming-control correctness only",
    }
    _finalize_artifact_gate(result, artifact_bytes_before=artifact_bytes_before)
    result["cost_allowed"] = result["status"] == "passed"
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
    gpu_fields = (
        "index,uuid,name,driver_version,utilization.gpu,clocks.current.sm,temperature.gpu"
    )
    gpu_rows = _nvidia_smi_csv("gpu", gpu_fields, gpu_id=0)
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 7:
        raise RuntimeError("nvidia-smi returned an unexpected GPU0 identity row")
    names = (
        "physical_index",
        "uuid",
        "name",
        "driver_version",
        "utilization_percent",
        "sm_clock_mhz",
        "temperature_c",
    )
    gpu = dict(zip(names, gpu_rows[0], strict=True))
    process_rows = _nvidia_smi_csv(
        "compute-apps",
        "gpu_uuid,pid,process_name,used_gpu_memory",
    )
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
    others = [row for row in processes if int(row["pid"]) != current_pid]
    return {
        "gpu": gpu,
        "process_pid": current_pid,
        "process_ppid": os.getppid(),
        "gpu0_compute_processes": processes,
        "gpu0_other_compute_processes": others,
    }


def _is_cuda_kernel_event(event: Any) -> bool:
    if not str(getattr(event, "device_type", "")).lower().endswith("cuda"):
        return False
    name = str(getattr(event, "name", "")).lower()
    return "memcpy" not in name and "memset" not in name


@torch.inference_mode()
def run_prefix(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    source: torch.Tensor,
    *,
    scope: str,
    state: Any | None = None,
    collect_outputs: bool = False,
) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
    if scope not in SCOPES:
        raise ValueError(f"unknown timing scope: {scope}")
    current = new_condition_state(name, module, int(source.shape[0])) if state is None else state
    outputs = []
    last_output = None
    for item in source.unbind(dim=1):
        descriptor = tokenizer.encode_frame(item) if scope == "end_to_end" else item
        last_output, current = condition_step(name, module, descriptor, current)
        if collect_outputs:
            outputs.append(last_output)
    if last_output is None:
        raise ValueError("prefix source must contain at least one frame")
    collected = torch.stack(outputs, dim=1) if collect_outputs else None
    return last_output, current, collected


@torch.inference_mode()
def _timed_prefix_sample(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    source: torch.Tensor,
    *,
    scope: str,
    device: torch.device,
) -> tuple[float, float, Any, torch.Tensor]:
    state = new_condition_state(name, module, int(source.shape[0]))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    host_start = time.perf_counter_ns()
    start_event.record()
    output, state, _ = run_prefix(
        name,
        module,
        tokenizer,
        source,
        scope=scope,
        state=state,
    )
    end_event.record()
    torch.cuda.synchronize(device)
    return (
        (time.perf_counter_ns() - host_start) / 1_000_000,
        float(start_event.elapsed_time(end_event)),
        state,
        output,
    )


@torch.inference_mode()
def profile_prefilled_next_step(
    name: str,
    module: nn.Module,
    tokenizer: ArenaCodecSpatialTokenizer,
    source: torch.Tensor,
    *,
    scope: str,
    device: torch.device,
) -> dict[str, Any]:
    _, prefilled, _ = run_prefix(name, module, tokenizer, source, scope=scope)
    probe = source[:, -1]
    for _ in range(WARMUP_SAMPLES):
        state = prefilled.clone()
        descriptor = tokenizer.encode_frame(probe) if scope == "end_to_end" else probe
        condition_step(name, module, descriptor, state)
    torch.cuda.synchronize(device)

    host_samples = []
    cuda_samples = []
    outputs_finite = True
    observations_exact = True
    for _ in range(MEASURED_SAMPLES):
        state = prefilled.clone()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        host_start = time.perf_counter_ns()
        start_event.record()
        descriptor = tokenizer.encode_frame(probe) if scope == "end_to_end" else probe
        output, next_state = condition_step(name, module, descriptor, state)
        end_event.record()
        torch.cuda.synchronize(device)
        host_samples.append((time.perf_counter_ns() - host_start) / 1_000_000)
        cuda_samples.append(float(start_event.elapsed_time(end_event)))
        outputs_finite = outputs_finite and bool(torch.isfinite(output).all())
        observations_exact = observations_exact and next_state.observations == source.shape[1] + 1

    profile_state = prefilled.clone()
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        with torch.profiler.record_function(f"{name}_{scope}_prefilled_next_step"):
            descriptor = tokenizer.encode_frame(probe) if scope == "end_to_end" else probe
            _, profiled_state = condition_step(name, module, descriptor, profile_state)
            torch.cuda.synchronize(device)
    device_events = [
        event
        for event in profile.events()
        if str(getattr(event, "device_type", "")).lower().endswith("cuda")
    ]
    kernel_events = [event for event in device_events if _is_cuda_kernel_event(event)]
    return {
        "scope": scope,
        "boundary": (
            "encode_frame_plus_temporal_step_plus_new_state"
            if scope == "end_to_end"
            else "resident_descriptor_plus_temporal_step_plus_new_state"
        ),
        "prefill_observations": prefilled.observations,
        "probe": "repeat_final_resident_input",
        "warmup_samples": WARMUP_SAMPLES,
        "measurements": MEASURED_SAMPLES,
        "host_wall_samples_ms": host_samples,
        "host_wall_p95_ms": percentile(host_samples, 95),
        "cuda_event_samples_ms": cuda_samples,
        "cuda_event_p95_ms": percentile(cuda_samples, 95),
        "kernel_profile_count": 1,
        "kernel_count": len(kernel_events),
        "cuda_device_event_count": len(device_events),
        "excluded_memcpy_memset_events": len(device_events) - len(kernel_events),
        "profiled_state_observations": profiled_state.observations,
        "outputs_finite": outputs_finite,
        "observations_exact": observations_exact,
    }


def _module_residency(
    name: str,
    tokenizer: ArenaCodecSpatialTokenizer,
    module: nn.Module,
    source: torch.Tensor,
) -> dict[str, Any]:
    tokenizer_parameters = list(tokenizer.parameters())
    temporal_parameters = list(module.parameters())
    all_parameters = tokenizer_parameters + temporal_parameters
    return {
        "condition": name,
        "temporal_class": type(module).__qualname__,
        "tokenizer_class": type(tokenizer).__qualname__,
        "all_model_parameters_on_cuda0": all(
            parameter.device == torch.device("cuda", 0) for parameter in all_parameters
        ),
        "resident_parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in all_parameters
        ),
        "resident_input_device": str(source.device),
        "resident_input_bytes": source.numel() * source.element_size(),
        "other_condition_models_constructed_concurrently": [],
        "temporal_model_instances_resident": 1,
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda", 0)
    tokenizer.to(device="cpu")
    module.to(device="cpu")
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer.to(device=device, dtype=torch.float32)
    module.to(device=device, dtype=torch.float32)
    raw = torch.from_numpy(np.array(codec[:length], copy=True)).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    if scope == "temporal_only":
        source = tokenizer(raw)
        del raw
    elif scope == "end_to_end":
        source = raw
    else:
        raise ValueError(f"unknown timing scope: {scope}")
    initial_state = new_condition_state(name, module, 1)
    residency = _module_residency(name, tokenizer, module, source)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    memory_output, memory_state, _ = run_prefix(
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
    state_bytes = unique_storage_state_bytes(memory_state)
    memory_finite = bool(torch.isfinite(memory_output).all()) and state_is_finite(memory_state)
    final_observations = memory_state.observations
    del memory_output, memory_state, initial_state

    for _ in range(WARMUP_SAMPLES):
        _timed_prefix_sample(
            name,
            module,
            tokenizer,
            source,
            scope=scope,
            device=device,
        )
    warm_telemetry = gpu_telemetry()

    host_samples = []
    cuda_samples = []
    measured_state_bytes = []
    measured_observations = []
    last_outputs_finite = True
    for _ in range(MEASURED_SAMPLES):
        host_ms, cuda_ms, state, output = _timed_prefix_sample(
            name,
            module,
            tokenizer,
            source,
            scope=scope,
            device=device,
        )
        host_samples.append(host_ms)
        cuda_samples.append(cuda_ms)
        measured_state_bytes.append(unique_storage_state_bytes(state))
        measured_observations.append(state.observations)
        last_outputs_finite = last_outputs_finite and bool(torch.isfinite(output).all())

    _, validation_state, validation_outputs = run_prefix(
        name,
        module,
        tokenizer,
        source,
        scope=scope,
        collect_outputs=True,
    )
    assert validation_outputs is not None
    validation_finite = bool(torch.isfinite(validation_outputs).all()) and state_is_finite(
        validation_state
    )
    next_step = profile_prefilled_next_step(
        name,
        module,
        tokenizer,
        source,
        scope=scope,
        device=device,
    )
    result = {
        "condition": name,
        "length": length,
        "scope": scope,
        "allocator_lifecycle": list(FROZEN_PROTOCOL["memory"]["allocator_order"]),
        "residency": residency,
        "warmup_samples": WARMUP_SAMPLES,
        "measurements": MEASURED_SAMPLES,
        "pre_and_post_sample_cuda_synchronize": True,
        "inference_mode": True,
        "fresh_state_each_prefix_sample": True,
        "host_wall_samples_ms": host_samples,
        "host_wall_p50_ms": percentile(host_samples, 50),
        "host_wall_p95_ms": percentile(host_samples, 95),
        "cuda_event_samples_ms": cuda_samples,
        "cuda_event_p50_ms": percentile(cuda_samples, 50),
        "cuda_event_p95_ms": percentile(cuda_samples, 95),
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "absolute_peak_allocated_bytes": absolute_peak_allocated,
            "delta_peak_allocated_bytes": absolute_peak_allocated - baseline_allocated,
            "absolute_peak_reserved_bytes": absolute_peak_reserved,
            "delta_peak_reserved_bytes": absolute_peak_reserved - baseline_reserved,
        },
        "state_unique_storage_bytes": state_bytes,
        "state_bytes_samples": measured_state_bytes,
        "final_state_observations": final_observations,
        "measured_state_observations": measured_observations,
        "all_outputs_and_states_finite": (
            memory_finite and last_outputs_finite and validation_finite
        ),
        "next_step": next_step,
    }
    tokenizer.to(device="cpu")
    module.to(device="cpu")
    del source, validation_state, validation_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return result, warm_telemetry


def bootstrap_growth_effect(
    mamba_t128: list[float],
    mamba_t925: list[float],
    transformer_t128: list[float],
    transformer_t925: list[float],
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (mamba_t128, mamba_t925, transformer_t128, transformer_t925)
    ]
    if draws != BOOTSTRAP_DRAWS:
        raise ValueError("bootstrap draw count changed")
    if any(array.shape != (MEASURED_SAMPLES,) for array in arrays):
        raise ValueError("bootstrap requires four complete 50-sample vectors")
    if any(not np.isfinite(array).all() or np.any(array <= 0) for array in arrays):
        raise ValueError("bootstrap latency samples must be positive and finite")
    rng = np.random.default_rng(seed)
    p95_draws = []
    for array in arrays:
        indices = rng.integers(0, MEASURED_SAMPLES, size=(draws, MEASURED_SAMPLES))
        p95_draws.append(np.percentile(array[indices], 95, axis=1, method="linear"))
    mamba_growth_draws = p95_draws[1] / p95_draws[0]
    transformer_growth_draws = p95_draws[3] / p95_draws[2]
    effects = transformer_growth_draws / mamba_growth_draws - 1.0
    p95_values = [float(np.percentile(array, 95, method="linear")) for array in arrays]
    mamba_growth = p95_values[1] / p95_values[0]
    transformer_growth = p95_values[3] / p95_values[2]
    return {
        "seed": seed,
        "draws": draws,
        "resampling": "four_independent_vectors_with_replacement",
        "sample_statistic": "numpy_linear_p95",
        "metric": "transformer_p95_growth_over_mamba_p95_growth_minus_one",
        "mamba_growth": mamba_growth,
        "transformer_growth": transformer_growth,
        "point_estimate": transformer_growth / mamba_growth - 1.0,
        "ci95_percentile": [
            float(np.percentile(effects, 2.5, method="linear")),
            float(np.percentile(effects, 97.5, method="linear")),
        ],
        "all_draws_finite": bool(np.isfinite(effects).all()),
    }


@torch.inference_mode()
def benchmark_cost_process(
    codec: np.ndarray,
    *,
    label: str,
    expected_parameter_identity: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if label not in COST_RUN_IDS:
        raise ValueError("cost process label must be A, B, C, or D")
    measurements: dict[str, Any] = {}
    warm_telemetry = []
    execution_order = []
    tokenizer_hashes = []
    temporal_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    for name in CONDITION_ORDERS[label]:
        tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval()
        module = build_condition(name, seed=SEED).eval()
        tokenizer_hashes.append(state_dict_sha256(tokenizer))
        temporal_hashes[name] = state_dict_sha256(module)
        parameter_counts[name] = trainable_parameter_count(module)
        condition_measurements = {}
        for length in LENGTH_ORDERS[label]:
            length_measurements = {}
            for scope in SCOPES:
                measured, telemetry = benchmark_scope_length(
                    codec,
                    name=name,
                    tokenizer=tokenizer,
                    module=module,
                    length=length,
                    scope=scope,
                )
                length_measurements[scope] = measured
                warm_telemetry.append(
                    {
                        "condition": name,
                        "length": length,
                        "scope": scope,
                        "telemetry": telemetry,
                    }
                )
                execution_order.append({"condition": name, "length": length, "scope": scope})
            condition_measurements[str(length)] = length_measurements
        measurements[name] = condition_measurements
        del module, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    parameter_identity = {
        "tokenizer_state_dict_sha256": tokenizer_hashes[0],
        "all_tokenizer_hashes_equal": len(set(tokenizer_hashes)) == 1,
        "temporal_state_dict_sha256": temporal_hashes,
        "temporal_parameter_counts": parameter_counts,
    }
    if parameter_identity != expected_parameter_identity:
        raise RuntimeError("cost process weights do not match correctness predecessor")
    temporal = lambda condition, length: measurements[condition][str(length)]["temporal_only"][
        "host_wall_samples_ms"
    ]
    bootstrap = bootstrap_growth_effect(
        temporal("functional_mamba", 128),
        temporal("functional_mamba", 925),
        temporal("streaming_kv_transformer", 128),
        temporal("streaming_kv_transformer", 925),
        seed=PROCESS_SEEDS[label],
    )
    return {
        "label": label,
        "seed": PROCESS_SEEDS[label],
        "condition_order": list(CONDITION_ORDERS[label]),
        "length_order": list(LENGTH_ORDERS[label]),
        "scope_order": list(SCOPES),
        "execution_order": execution_order,
        "parameter_identity": parameter_identity,
        "measurements": measurements,
        "bootstrap": bootstrap,
    }, warm_telemetry


def _measurement(
    cost: dict[str, Any],
    condition: str,
    length: int,
    scope: str,
) -> dict[str, Any]:
    return cost["measurements"][condition][str(length)][scope]


def _finite_samples(values: Any, expected: int = MEASURED_SAMPLES) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.shape == (expected,) and bool(np.isfinite(array).all()) and bool(
        np.all(array >= 0)
    )


def _positive_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric)) and numeric > 0


def _expected_execution_order(label: str) -> list[dict[str, Any]]:
    return [
        {"condition": condition, "length": length, "scope": scope}
        for condition in CONDITION_ORDERS[label]
        for length in LENGTH_ORDERS[label]
        for scope in SCOPES
    ]


def _measurement_integrity(cost: dict[str, Any]) -> bool:
    if set(cost.get("measurements", {})) != set(CONDITIONS):
        return False
    for condition in CONDITIONS:
        per_length = cost["measurements"][condition]
        if set(per_length) != {str(length) for length in LENGTHS}:
            return False
        for length in LENGTHS:
            if set(per_length[str(length)]) != set(SCOPES):
                return False
            for scope in SCOPES:
                item = per_length[str(length)][scope]
                host = item.get("host_wall_samples_ms")
                cuda = item.get("cuda_event_samples_ms")
                state_bytes = item.get("state_bytes_samples")
                observations = item.get("measured_state_observations")
                next_step = item.get("next_step", {})
                memory = item.get("memory", {})
                residency = item.get("residency", {})
                expected_state_bytes = item.get("state_unique_storage_bytes")
                if condition == "streaming_kv_transformer":
                    state_contract = expected_state_bytes == transformer_kv_expected_bytes(
                        batch_size=1,
                        observations=length,
                    )
                elif condition == "current_descriptor":
                    state_contract = expected_state_bytes == 0
                else:
                    state_contract = _positive_finite(expected_state_bytes)
                kernel_count_valid = next_step.get("kernel_count", -1) > 0 or (
                    condition == "current_descriptor"
                    and scope == "temporal_only"
                    and next_step.get("kernel_count") == 0
                )
                memory_keys = {
                    "baseline_allocated_bytes",
                    "baseline_reserved_bytes",
                    "absolute_peak_allocated_bytes",
                    "delta_peak_allocated_bytes",
                    "absolute_peak_reserved_bytes",
                    "delta_peak_reserved_bytes",
                }
                if not (
                    item.get("condition") == condition
                    and item.get("length") == length
                    and item.get("scope") == scope
                    and item.get("allocator_lifecycle")
                    == FROZEN_PROTOCOL["memory"]["allocator_order"]
                    and item.get("warmup_samples") == WARMUP_SAMPLES
                    and item.get("measurements") == MEASURED_SAMPLES
                    and item.get("pre_and_post_sample_cuda_synchronize") is True
                    and item.get("inference_mode") is True
                    and item.get("fresh_state_each_prefix_sample") is True
                    and _finite_samples(host)
                    and _finite_samples(cuda)
                    and abs(item.get("host_wall_p95_ms", -1) - percentile(host, 95)) <= 1e-12
                    and abs(item.get("cuda_event_p95_ms", -1) - percentile(cuda, 95)) <= 1e-12
                    and isinstance(state_bytes, list)
                    and len(state_bytes) == MEASURED_SAMPLES
                    and all(value == expected_state_bytes for value in state_bytes)
                    and isinstance(observations, list)
                    and observations == [length] * MEASURED_SAMPLES
                    and item.get("final_state_observations") == length
                    and item.get("all_outputs_and_states_finite") is True
                    and state_contract
                    and set(memory) == memory_keys
                    and memory["absolute_peak_allocated_bytes"]
                    >= memory["baseline_allocated_bytes"]
                    and memory["delta_peak_allocated_bytes"]
                    == memory["absolute_peak_allocated_bytes"]
                    - memory["baseline_allocated_bytes"]
                    and memory["absolute_peak_reserved_bytes"]
                    >= memory["baseline_reserved_bytes"]
                    and memory["delta_peak_reserved_bytes"]
                    == memory["absolute_peak_reserved_bytes"]
                    - memory["baseline_reserved_bytes"]
                    and residency.get("condition") == condition
                    and residency.get("all_model_parameters_on_cuda0") is True
                    and residency.get("resident_input_device") == "cuda:0"
                    and residency.get("other_condition_models_constructed_concurrently") == []
                    and residency.get("temporal_model_instances_resident") == 1
                    and next_step.get("prefill_observations") == length
                    and next_step.get("warmup_samples") == WARMUP_SAMPLES
                    and next_step.get("measurements") == MEASURED_SAMPLES
                    and _finite_samples(next_step.get("host_wall_samples_ms"))
                    and _finite_samples(next_step.get("cuda_event_samples_ms"))
                    and abs(
                        next_step.get("host_wall_p95_ms", -1)
                        - percentile(next_step["host_wall_samples_ms"], 95)
                    )
                    <= 1e-12
                    and next_step.get("kernel_profile_count") == 1
                    and kernel_count_valid
                    and next_step.get("profiled_state_observations") == length + 1
                    and next_step.get("outputs_finite") is True
                    and next_step.get("observations_exact") is True
                ):
                    return False
    return True


def _fixed_state_growth(cost: dict[str, Any], condition: str, scope: str) -> float:
    first = _measurement(cost, condition, 128, scope)["state_unique_storage_bytes"]
    last = _measurement(cost, condition, 925, scope)["state_unique_storage_bytes"]
    return abs(last / first - 1.0)


def _kv_ratio_relative_error(cost: dict[str, Any], scope: str) -> float:
    first = _measurement(cost, "streaming_kv_transformer", 128, scope)[
        "state_unique_storage_bytes"
    ]
    last = _measurement(cost, "streaming_kv_transformer", 925, scope)[
        "state_unique_storage_bytes"
    ]
    return abs((last / first) / (925 / 128) - 1.0)


def _latency_growth(cost: dict[str, Any], condition: str, scope: str) -> float:
    first = _measurement(cost, condition, 128, scope)["host_wall_p95_ms"]
    last = _measurement(cost, condition, 925, scope)["host_wall_p95_ms"]
    return last / first


def _peak_scaling_passes(cost: dict[str, Any]) -> bool:
    for scope in SCOPES:
        for metric in ("absolute_peak_allocated_bytes", "delta_peak_allocated_bytes"):
            ratios = []
            for length in LENGTHS:
                mamba = _measurement(cost, "functional_mamba", length, scope)["memory"][metric]
                transformer = _measurement(
                    cost,
                    "streaming_kv_transformer",
                    length,
                    scope,
                )["memory"][metric]
                if mamba <= 0 or transformer <= 0:
                    return False
                ratios.append(transformer / mamba)
            if any(right < left for left, right in zip(ratios, ratios[1:], strict=True)):
                return False
            mamba_t925 = _measurement(cost, "functional_mamba", 925, scope)["memory"][metric]
            transformer_t925 = _measurement(
                cost,
                "streaming_kv_transformer",
                925,
                scope,
            )["memory"][metric]
            if mamba_t925 > MAX_MAMBA_PEAK_FRACTION * transformer_t925:
                return False
    return True


def _telemetry_number(telemetry: dict[str, Any], field: str) -> float:
    return float(telemetry["gpu"][field])


def evaluate_cost_gates(
    cost: dict[str, Any],
    telemetry: dict[str, Any],
    execution: dict[str, Any],
    expected_parameter_identity: dict[str, Any],
) -> dict[str, bool]:
    label = cost["label"]
    bootstrap = cost["bootstrap"]
    temporal_growth_mamba = _latency_growth(cost, "functional_mamba", "temporal_only")
    temporal_growth_transformer = _latency_growth(
        cost,
        "streaming_kv_transformer",
        "temporal_only",
    )
    recomputed_bootstrap = bootstrap_growth_effect(
        _measurement(cost, "functional_mamba", 128, "temporal_only")[
            "host_wall_samples_ms"
        ],
        _measurement(cost, "functional_mamba", 925, "temporal_only")[
            "host_wall_samples_ms"
        ],
        _measurement(cost, "streaming_kv_transformer", 128, "temporal_only")[
            "host_wall_samples_ms"
        ],
        _measurement(cost, "streaming_kv_transformer", 925, "temporal_only")[
            "host_wall_samples_ms"
        ],
        seed=PROCESS_SEEDS[label],
    )
    warm_records = telemetry["after_warmup"]
    start = telemetry["before"]
    after = telemetry["after"]
    try:
        start_environment = (
            _telemetry_number(start, "utilization_percent") == 0
            and _telemetry_number(start, "temperature_c") <= MAX_START_TEMPERATURE_C
            and not start["gpu0_compute_processes"]
        )
        warm_environment = len(warm_records) == len(CONDITIONS) * len(LENGTHS) * len(SCOPES) and all(
            record["condition"] in CONDITIONS
            and record["length"] in LENGTHS
            and record["scope"] in SCOPES
            and _telemetry_number(record["telemetry"], "sm_clock_mhz") >= MIN_WARM_SM_CLOCK_MHZ
            and _telemetry_number(record["telemetry"], "temperature_c")
            <= MAX_WARM_TEMPERATURE_C
            and not record["telemetry"]["gpu0_other_compute_processes"]
            for record in warm_records
        )
        after_environment = not after["gpu0_other_compute_processes"]
    except (KeyError, TypeError, ValueError):
        start_environment = warm_environment = after_environment = False
    measurements_valid = _measurement_integrity(cost)
    return {
        "correctness_predecessor_passed": True,
        "source_input_dependency_runtime_and_prior_identity": True,
        "exact_latin_square_condition_length_and_scope_order": (
            cost["condition_order"] == list(CONDITION_ORDERS[label])
            and cost["length_order"] == list(LENGTH_ORDERS[label])
            and cost["scope_order"] == list(SCOPES)
            and cost["execution_order"] == _expected_execution_order(label)
            and cost["seed"] == PROCESS_SEEDS[label]
        ),
        "weights_match_correctness_and_frozen_parameter_counts": (
            cost["parameter_identity"] == expected_parameter_identity
            and cost["parameter_identity"]["temporal_parameter_counts"]
            == EXPECTED_TEMPORAL_PARAMETERS
            and cost["parameter_identity"]["all_tokenizer_hashes_equal"] is True
        ),
        "all_32_measurements_have_exact_raw_memory_state_profile_and_residency_contract": measurements_valid,
        "mamba_and_gru_state_growth_at_most_1pct": measurements_valid
        and all(
            _fixed_state_growth(cost, condition, scope) <= MAX_FIXED_STATE_GROWTH
            for condition in ("functional_mamba", "persistent_gru")
            for scope in SCOPES
        ),
        "transformer_kv_growth_matches_925_over_128_within_1pct": measurements_valid
        and all(
            _kv_ratio_relative_error(cost, scope) <= MAX_KV_RATIO_RELATIVE_ERROR
            for scope in SCOPES
        ),
        "transformer_over_mamba_absolute_and_delta_peak_ratios_nondecreasing_and_t925_mamba_at_most_60pct": measurements_valid
        and _peak_scaling_passes(cost),
        "mamba_full_prefix_host_p95_growth_within_linear_slack": measurements_valid
        and all(
            _latency_growth(cost, "functional_mamba", scope)
            <= MAX_MAMBA_PREFIX_GROWTH_MULTIPLIER * (925 / 128)
            for scope in SCOPES
        ),
        "temporal_transformer_host_p95_growth_at_least_1p10x_mamba": measurements_valid
        and temporal_growth_transformer
        >= MIN_TRANSFORMER_TO_MAMBA_GROWTH * temporal_growth_mamba,
        "mamba_t925_prefilled_next_step_host_p95_below_5ms": measurements_valid
        and all(
            _measurement(cost, "functional_mamba", 925, scope)["next_step"][
                "host_wall_p95_ms"
            ]
            < MAX_MAMBA_NEXT_STEP_P95_MS
            for scope in SCOPES
        ),
        "bootstrap_recomputed_exactly_with_point_and_ci_lower_gates": (
            bootstrap == recomputed_bootstrap
            and bootstrap["draws"] == BOOTSTRAP_DRAWS
            and bootstrap["all_draws_finite"]
            and bootstrap["point_estimate"] >= MIN_RELATIVE_GROWTH_POINT
            and bootstrap["ci95_percentile"][0] >= MIN_RELATIVE_GROWTH_CI95_LOWER
        ),
        "gpu_start_idle_cool_all_warm_points_clocked_and_end_has_no_other_process": (
            start_environment and warm_environment and after_environment
        ),
        "native_vla_modules_absent": not execution["native_vla_modules"]["before"]
        and not execution["native_vla_modules"]["after"],
        "optimizer_steps_zero": not execution["optimizer"]["constructor_calls"]
        and execution["optimizer"]["steps_observed"] == 0,
    }


def verify_passed_result(
    path: Path,
    expected_sha256: str,
    *,
    expected_path: Path,
    expected_run_id: str,
    plan_identity: dict[str, Any],
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if resolved != expected_path.resolve() or not resolved.is_file():
        raise ValueError("predecessor result path changed or is missing")
    actual_sha256 = sha256(resolved)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ValueError("predecessor result identity changed")
    result = read_json(resolved)
    if (
        result.get("experiment_id") != EXPERIMENT
        or result.get("run_id") != expected_run_id
        or result.get("source_commit") != source_commit
        or result.get("plan", {}).get("sha256") != plan_identity["sha256"]
        or result.get("status") != "passed"
        or not result.get("gates")
        or not all(result["gates"].values())
        or result.get("optimizer_steps") != 0
        or result.get("vla_model_instantiated") is not False
        or result.get("training_allowed") is not False
    ):
        raise ValueError("predecessor result contract changed or did not pass")
    return result, {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
        "run_id": expected_run_id,
    }


def run_cost(
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
        raise ValueError("cost process label must be A, B, C, or D")
    started = time.monotonic()
    artifact_bytes_before = directory_bytes(ARTIFACT_ROOT)
    native_before = native_vla_modules_loaded()
    telemetry_before = gpu_telemetry()
    source = verify_source(plan["source"]["commit"])
    correctness, correctness_identity = verify_passed_result(
        correctness_path,
        correctness_sha256,
        expected_path=ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json",
        expected_run_id=CORRECTNESS_RUN_ID,
        plan_identity=plan_identity,
        source_commit=source["commit"],
    )
    labels = tuple(COST_RUN_IDS)
    index = labels.index(label)
    predecessor_identity = None
    if index == 0:
        if predecessor_path is not None or predecessor_sha256 is not None:
            raise ValueError("cost process A must not receive a cost predecessor")
    else:
        if predecessor_path is None or predecessor_sha256 is None:
            raise ValueError("cost process B/C/D requires the immediately preceding cost result")
        previous_label = labels[index - 1]
        _, predecessor_identity = verify_passed_result(
            predecessor_path,
            predecessor_sha256,
            expected_path=ARTIFACT_ROOT / COST_RUN_IDS[previous_label] / "RESULT.json",
            expected_run_id=COST_RUN_IDS[previous_label],
            plan_identity=plan_identity,
            source_commit=source["commit"],
        )
    prior_evidence = verify_prior_evidence()
    dependencies = verify_dependencies()
    codec, input_evidence = load_codec_input()
    environment = verify_environment()
    correctness_parameters = correctness["correctness"]["parameters"]
    expected_parameter_identity = {
        "tokenizer_state_dict_sha256": correctness["correctness"]["tokenizer"][
            "state_dict_sha256"
        ],
        "all_tokenizer_hashes_equal": True,
        "temporal_state_dict_sha256": correctness_parameters["temporal_state_dict_sha256"],
        "temporal_parameter_counts": correctness_parameters["actual_temporal_counts"],
    }
    with optimizer_construction_guard() as optimizer_evidence:
        cost, warm_telemetry = benchmark_cost_process(
            codec,
            label=label,
            expected_parameter_identity=expected_parameter_identity,
        )
    gc.collect()
    torch.cuda.empty_cache()
    telemetry_after = gpu_telemetry()
    telemetry = {
        "before": telemetry_before,
        "after_warmup": warm_telemetry,
        "after": telemetry_after,
    }
    execution = {
        "native_vla_modules": {
            "before": native_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
    }
    gates = evaluate_cost_gates(cost, telemetry, execution, expected_parameter_identity)
    wall_seconds = time.monotonic() - started
    gates["wall_time_at_most_20min"] = wall_seconds <= MAX_WALL_SECONDS
    gates["artifact_total_at_most_256mib"] = True
    end_to_end_growth = {
        condition: _latency_growth(cost, condition, "end_to_end")
        for condition in ("functional_mamba", "streaming_kv_transformer")
    }
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": COST_RUN_IDS[label],
        "label": label,
        "source_commit": source["commit"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source": source,
        "plan": plan_identity,
        "correctness_predecessor": correctness_identity,
        "cost_predecessor": predecessor_identity,
        "prior_evidence": prior_evidence,
        "input": input_evidence,
        "dependencies": dependencies,
        "environment": environment,
        "process": {"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv},
        "telemetry": telemetry,
        "cost": cost,
        "reported_end_to_end_growth": {
            **end_to_end_growth,
            "transformer_grows_more_than_mamba": end_to_end_growth[
                "streaming_kv_transformer"
            ]
            > end_to_end_growth["functional_mamba"],
            "gate": False,
        },
        "execution_evidence": execution,
        "gates": gates,
        "optimizer_steps": optimizer_evidence["steps_observed"],
        "vla_model_instantiated": bool(
            execution["native_vla_modules"]["before"]
            or execution["native_vla_modules"]["after"]
        ),
        "next_cost_allowed": all(gates.values()) and label != "D",
        "aggregate_allowed": all(gates.values()) and label == "D",
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "fixed-episode fixed-weight engineering scaling only",
    }
    _finalize_artifact_gate(result, artifact_bytes_before=artifact_bytes_before)
    result["next_cost_allowed"] = result["status"] == "passed" and label != "D"
    result["aggregate_allowed"] = result["status"] == "passed" and label == "D"
    return result


def evaluate_aggregate_gates(
    correctness: dict[str, Any],
    costs: list[dict[str, Any]],
    cost_identities: list[dict[str, Any]],
) -> dict[str, bool]:
    expected_labels = list(COST_RUN_IDS)
    expected_run_ids = list(COST_RUN_IDS.values())
    correctness_parameters = correctness["correctness"]["parameters"]
    expected_parameter_identity = {
        "tokenizer_state_dict_sha256": correctness["correctness"]["tokenizer"][
            "state_dict_sha256"
        ],
        "all_tokenizer_hashes_equal": True,
        "temporal_state_dict_sha256": correctness_parameters["temporal_state_dict_sha256"],
        "temporal_parameter_counts": correctness_parameters["actual_temporal_counts"],
    }
    correctness_core = evaluate_correctness_gates(
        correctness["correctness"],
        correctness["execution_evidence"],
    )
    cost_core_matches = []
    for result in costs:
        recomputed = evaluate_cost_gates(
            result["cost"],
            result["telemetry"],
            result["execution_evidence"],
            expected_parameter_identity,
        )
        cost_core_matches.append(
            all(result["gates"].get(name) == value for name, value in recomputed.items())
            and all(recomputed.values())
        )
    chain_valid = len(costs) == 4 and len(cost_identities) == 4
    if chain_valid:
        correctness_sha = correctness["result_identity"]["sha256"]
        chain_valid = (
            costs[0]["cost_predecessor"] is None
            and all(
                result["correctness_predecessor"]["sha256"] == correctness_sha
                for result in costs
            )
            and all(
                costs[index]["cost_predecessor"]["sha256"]
                == cost_identities[index - 1]["sha256"]
                for index in range(1, 4)
            )
        )
    return {
        "correctness_result_passed_and_core_gates_recompute": (
            correctness["status"] == "passed"
            and all(correctness["gates"].values())
            and all(correctness_core.values())
            and all(correctness["gates"].get(name) == value for name, value in correctness_core.items())
        ),
        "exact_four_ordered_cost_results_passed_without_averaging": (
            len(costs) == 4
            and [result["label"] for result in costs] == expected_labels
            and [result["run_id"] for result in costs] == expected_run_ids
            and all(result["status"] == "passed" for result in costs)
            and all(all(result["gates"].values()) for result in costs)
            and all(cost_core_matches)
        ),
        "cost_results_form_required_hash_predecessor_chain": chain_valid,
        "four_cost_python_processes_are_distinct": len(
            {result["process"]["pid"] for result in costs}
        )
        == 4,
        "all_four_have_exact_cyclic_orders_and_fixed_bootstrap_gates": all(
            result["cost"]["condition_order"] == list(CONDITION_ORDERS[label])
            and result["cost"]["length_order"] == list(LENGTH_ORDERS[label])
            and result["cost"]["execution_order"] == _expected_execution_order(label)
            and result["cost"]["bootstrap"]["point_estimate"] >= MIN_RELATIVE_GROWTH_POINT
            and result["cost"]["bootstrap"]["ci95_percentile"][0]
            >= MIN_RELATIVE_GROWTH_CI95_LOWER
            for label, result in zip(expected_labels, costs, strict=True)
        ),
        "all_four_weight_identities_match_correctness": all(
            result["cost"]["parameter_identity"] == expected_parameter_identity
            for result in costs
        ),
        "all_components_vla_free_and_zero_optimizer": (
            correctness["vla_model_instantiated"] is False
            and correctness["optimizer_steps"] == 0
            and all(result["vla_model_instantiated"] is False for result in costs)
            and all(result["optimizer_steps"] == 0 for result in costs)
        ),
        "each_component_wall_at_most_20min": correctness["wall_seconds"] <= MAX_WALL_SECONDS
        and all(result["wall_seconds"] <= MAX_WALL_SECONDS for result in costs),
    }


def run_aggregate(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
    *,
    correctness_path: Path,
    correctness_sha256: str,
    cost_paths: tuple[Path, Path, Path, Path],
    cost_sha256s: tuple[str, str, str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_bytes_before = directory_bytes(ARTIFACT_ROOT)
    source = verify_source(plan["source"]["commit"])
    correctness, correctness_identity = verify_passed_result(
        correctness_path,
        correctness_sha256,
        expected_path=ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json",
        expected_run_id=CORRECTNESS_RUN_ID,
        plan_identity=plan_identity,
        source_commit=source["commit"],
    )
    correctness["result_identity"] = correctness_identity
    costs = []
    cost_identities = []
    for label, path, digest in zip(COST_RUN_IDS, cost_paths, cost_sha256s, strict=True):
        cost, identity = verify_passed_result(
            path,
            digest,
            expected_path=ARTIFACT_ROOT / COST_RUN_IDS[label] / "RESULT.json",
            expected_run_id=COST_RUN_IDS[label],
            plan_identity=plan_identity,
            source_commit=source["commit"],
        )
        costs.append(cost)
        cost_identities.append(identity)
    prior_evidence = verify_prior_evidence()
    dependencies = verify_dependencies()
    _, input_evidence = load_codec_input()
    gates = evaluate_aggregate_gates(correctness, costs, cost_identities)
    native_modules = native_vla_modules_loaded()
    gates["aggregate_process_native_vla_modules_absent"] = not native_modules
    wall_seconds = time.monotonic() - started
    gates["aggregate_wall_time_at_most_20min"] = wall_seconds <= MAX_WALL_SECONDS
    gates["artifact_total_at_most_256mib"] = True
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": AGGREGATE_RUN_ID,
        "source_commit": source["commit"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source": source,
        "plan": plan_identity,
        "prior_evidence": prior_evidence,
        "dependencies": dependencies,
        "input": input_evidence,
        "component_results": {
            "correctness": correctness_identity,
            "cost": cost_identities,
        },
        "summary": {
            "cost_process_pids": [cost["process"]["pid"] for cost in costs],
            "bootstrap_effect": {
                cost["label"]: {
                    "point_estimate": cost["cost"]["bootstrap"]["point_estimate"],
                    "ci95_percentile": cost["cost"]["bootstrap"]["ci95_percentile"],
                }
                for cost in costs
            },
            "host_p95_ms": {
                cost["label"]: {
                    condition: {
                        str(length): {
                            scope: _measurement(cost["cost"], condition, length, scope)[
                                "host_wall_p95_ms"
                            ]
                            for scope in SCOPES
                        }
                        for length in LENGTHS
                    }
                    for condition in CONDITIONS
                }
                for cost in costs
            },
        },
        "gates": gates,
        "optimizer_steps": 0,
        "vla_model_instantiated": bool(native_modules),
        "arena_episode45_single_episode_residual_overfit_design_allowed": all(gates.values()),
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": (
            "fixed episode codec stream engineering scaling relative to this streaming-KV "
            "implementation; no action, task-success, Mamba-over-GRU, or general capability claim"
        ),
    }
    _finalize_artifact_gate(result, artifact_bytes_before=artifact_bytes_before)
    result["arena_episode45_single_episode_residual_overfit_design_allowed"] = (
        result["status"] == "passed"
    )
    return result


def _write_mode_result(run_id: str, run: Any) -> int:
    output = ARTIFACT_ROOT / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result_path = output / "RESULT.json"
    started = time.monotonic()
    try:
        result = run()
        write_new(result_path, result)
    except Exception as error:
        write_new(
            result_path,
            {
                "status": "failed",
                "experiment_id": EXPERIMENT,
                "run_id": run_id,
                "error": f"{type(error).__name__}: {error}",
                "native_vla_evidence": native_vla_modules_loaded(),
                "optimizer_evidence": "unavailable because the run raised before completion",
                "wall_seconds": time.monotonic() - started,
                "training_allowed": False,
            },
        )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "passed" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
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
    if args.prepare_plan is not None:
        if args.plan is not None or args.expected_plan_sha256 is not None:
            parser.error("--prepare-plan cannot be combined with resolved plan arguments")
        print(json.dumps(prepare_plan(args.prepare_plan), sort_keys=True), flush=True)
        return 0
    if args.plan is None or args.expected_plan_sha256 is None:
        parser.error("all execution modes require --plan and --expected-plan-sha256")
    plan, plan_identity = verify_plan(args.plan, args.expected_plan_sha256)
    if args.correctness:
        return _write_mode_result(
            CORRECTNESS_RUN_ID,
            lambda: run_correctness(plan, plan_identity),
        )
    selected_cost = next(
        (label for label in COST_RUN_IDS if getattr(args, f"cost_{label.lower()}")),
        None,
    )
    if selected_cost is not None:
        if args.correctness_result is None or args.expected_correctness_sha256 is None:
            parser.error("cost modes require correctness result path and SHA-256")
        return _write_mode_result(
            COST_RUN_IDS[selected_cost],
            lambda: run_cost(
                plan,
                plan_identity,
                label=selected_cost,
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
        parser.error("aggregate requires correctness and all four cost paths and SHA-256s")
    return _write_mode_result(
        AGGREGATE_RUN_ID,
        lambda: run_aggregate(
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
