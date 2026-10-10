from __future__ import annotations

import ast
import copy
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

from gr00t.experimental.arena_functional_mamba_transition import FunctionalMamba2State
from gr00t.experimental.arena_functional_streaming_controls import StreamingKVState
from gr00t.experimental.arena_online_state_scaling import KV_SENTINEL
import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples/G1_ARENA_N17/online_state_scaling.py"
EXAMPLE_DIR = str(EXAMPLE.parent)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)
SPEC = importlib.util.spec_from_file_location("online_state_scaling_runner", EXAMPLE)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _function_source_bundle_sha256(names: tuple[str, ...]) -> str:
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lines = source.splitlines(keepends=True)
    payload = "".join(
        "".join(lines[functions[name].lineno - 1 : functions[name].end_lineno]) for name in names
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _decision_cost(
    label: str,
    *,
    end_to_end_p95: float = 19.999,
    state_ratio: float = 0.10,
    next_ratio: float = 0.80,
    prefix_ratio: float = 0.80,
    length_ratio: float = 1.15,
    allocator_ratio: float = 0.50,
) -> dict[str, object]:
    preallocated_live = 3_788_800
    mamba_next_128 = 1.0
    mamba_next_925 = mamba_next_128 * length_ratio
    preallocated_next = mamba_next_925 / next_ratio
    preallocated_prefix = 100.0
    cat_delta = 1000
    return {
        "label": label,
        "cost": {
            "measurements": {
                "functional_mamba": {
                    "128": {"temporal_only": {"next_step": {"host_p95_ms": mamba_next_128}}},
                    "925": {
                        "end_to_end": {
                            "latency": {"per_frame_flat": {"host_p95_ms": end_to_end_p95}}
                        },
                        "temporal_only": {
                            "logical_state_bytes": preallocated_live * state_ratio,
                            "next_step": {"host_p95_ms": mamba_next_925},
                            "latency": {
                                "complete_prefix_cumulative": {
                                    "host_p95_ms": preallocated_prefix * prefix_ratio
                                }
                            },
                        },
                    },
                },
                "transformer_preallocated": {
                    "925": {
                        "temporal_only": {
                            "logical_state_bytes": preallocated_live,
                            "next_step": {"host_p95_ms": preallocated_next},
                            "latency": {
                                "complete_prefix_cumulative": {"host_p95_ms": preallocated_prefix}
                            },
                            "memory": {"delta_peak_allocated_bytes": cat_delta * allocator_ratio},
                        }
                    }
                },
                "transformer_functional_cat": {
                    "925": {"temporal_only": {"memory": {"delta_peak_allocated_bytes": cat_delta}}}
                },
            }
        },
    }


def _valid_telemetry() -> dict[str, object]:
    return {
        "gpu": {
            "physical_index": "0",
            "uuid": "GPU-unit",
            "name": runner.FROZEN_RUNTIME["gpu_name"],
            "driver_version": "999.0",
            "utilization_percent": "0",
            "sm_clock_mhz": "2100",
            "temperature_c": "40",
            "memory_used_mib": "10",
            "memory_total_mib": "49140",
        },
        "process_pid": 123,
        "process_ppid": 1,
        "gpu0_compute_processes": [],
        "gpu0_other_compute_processes": [],
    }


class _CPUFunctionalMambaStub(torch.nn.Module):
    """Exercise runner state semantics without invoking GPU-only Mamba kernels."""

    def new_functional_state(self, batch_size: int) -> FunctionalMamba2State:
        return FunctionalMamba2State(
            tuple(
                (
                    torch.zeros(batch_size, 640, 4),
                    torch.zeros(batch_size, 8, 64, 64),
                )
                for _ in range(2)
            ),
            batch_size,
        )

    def functional_step(
        self,
        descriptor: torch.Tensor,
        state: FunctionalMamba2State,
    ) -> tuple[torch.Tensor, FunctionalMamba2State]:
        return descriptor, FunctionalMamba2State(
            tuple((conv.clone(), ssm.clone()) for conv, ssm in state.layers),
            state.batch_size,
            state.observations + 1,
        )


class _FunctionalCatLifecycleStub(torch.nn.Module):
    branch_ref: weakref.ReferenceType[StreamingKVState] | None = None

    def streaming_step(
        self, descriptor: torch.Tensor, state: StreamingKVState
    ) -> tuple[torch.Tensor, StreamingKVState]:
        observations = state.observations + 1
        shape = (1, 8, observations, 32)
        next_state = StreamingKVState(
            tuple((torch.zeros(shape), torch.zeros(shape)) for _ in range(2)),
            batch_size=1,
            observations=observations,
        )
        self.branch_ref = weakref.ref(next_state)
        return descriptor, next_state


class _PreallocatedLifecycleStub(torch.nn.Module):
    branch_ref: weakref.ReferenceType[runner.PreallocatedKVState] | None = None

    def preallocated_step(
        self, descriptor: torch.Tensor, state: runner.PreallocatedKVState
    ) -> tuple[torch.Tensor, runner.PreallocatedKVState]:
        index = state.live_index
        for tensor in state.tensors():
            tensor[:, :, index : index + 1].zero_()
        state.live_index += 1
        self.branch_ref = weakref.ref(state)
        return descriptor, state


def test_frozen_five_process_orders_lengths_and_incomplete_status() -> None:
    assert runner.EXPERIMENT == "g1-arena-online-state-scaling-r5"
    assert runner.BASE_SOURCE_COMMIT == "7c49314cd855dbf9db1dd2505d8b98b741c6cde3"
    assert runner.ARTIFACT_ROOT == Path(
        "/home/sl/research/icra27-artifacts/g1-arena-online-state-scaling-r5"
    )
    assert runner.PLAN_PATH == runner.ARTIFACT_ROOT / "plan-r5/PLAN.json"
    assert runner.CORRECTNESS_RUN_ID == "correctness-gpu0-r5"
    assert runner.AGGREGATE_RUN_ID == "aggregate-r5"
    assert runner.FUNCTIONAL_AGGREGATE["path"].endswith("aggregate-r2/RESULT.json")
    assert runner.FUNCTIONAL_RECHECK["path"].endswith("aggregate-r2/RESULT_RECHECK.json")
    assert runner.GRU_ATTRIBUTION_RESULT == {
        "path": (
            "/home/sl/research/icra27-artifacts/g1-arena-gru-gradient-attribution-r2/"
            "attribution-gpu0-r2/RESULT.json"
        ),
        "bytes": 29_817,
        "sha256": "754a28c592b20ede974ea20c60d70b2d3c43db1364584f9d7a2e6d868165abb8",
    }
    assert runner.GRU_ATTRIBUTION_GRADIENTS == {
        "path": (
            "/home/sl/research/icra27-artifacts/g1-arena-gru-gradient-attribution-r2/"
            "attribution-gpu0-r2/GRADIENTS.npz"
        ),
        "bytes": 525_098,
        "sha256": "5c2c32b288c6e17cc22997af70290e0f21aea302d136b939674850ce31fa47f7",
    }
    assert runner.CONDITIONS == (
        "functional_mamba",
        "persistent_gru",
        "transformer_functional_cat",
        "transformer_preallocated",
        "current_descriptor",
    )
    assert runner.LENGTHS == (128, 256, 512, 925)
    assert runner.COST_RUN_IDS == {
        "A": "cost-gpu0-r5a",
        "B": "cost-gpu0-r5b",
        "C": "cost-gpu0-r5c",
        "D": "cost-gpu0-r5d",
        "E": "cost-gpu0-r5e",
    }
    for offset, label in enumerate(runner.COST_RUN_IDS):
        assert runner.CONDITION_ORDERS[label] == (
            runner.CONDITIONS[offset:] + runner.CONDITIONS[:offset]
        )
        length_offset = offset % len(runner.LENGTHS)
        assert runner.LENGTH_ORDERS[label] == (
            runner.LENGTHS[length_offset:] + runner.LENGTHS[:length_offset]
        )
    assert runner.WARMUP_SAMPLES == 20
    assert runner.MEASURED_SAMPLES == 100
    assert set(runner.PROCESS_SEEDS.values()) == {230923}
    assert runner.IMPLEMENTATION_STATUS == {
        "preallocated_control": True,
        "independent_correctness": True,
        "cost_cli_contract": True,
        "cost_measurement": True,
        "aggregate_decision": True,
        "formal_gpu_run_allowed": True,
    }


def test_model_identity_records_per_tensor_shared_transformer_hashes() -> None:
    identity = runner.model_identity()
    functional = identity["conditions"]["transformer_functional_cat"]
    preallocated = identity["conditions"]["transformer_preallocated"]
    assert functional["parameters"] == preallocated["parameters"] == 857_216
    assert functional["state_dict_sha256"] == preallocated["state_dict_sha256"]
    assert functional["state_dict_tensor_sha256"] == preallocated["state_dict_tensor_sha256"]
    assert identity["transformer_weight_hash_shared"]
    assert identity["transformer_per_tensor_hash_shared"]
    assert (
        len(functional["state_dict_tensor_sha256"]) == len(functional["state_dict_sha256"][:0]) + 24
    )


def test_closed_routes_and_opaque_full_endpoint_are_fail_closed() -> None:
    assert runner.assert_closed_routes_absent() == {
        "closed_helper_calls": [],
        "closed_runner_imported": False,
        "opaque_full_transformer_calls": 0,
    }
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "CausalTransformerBackbone" not in source
    module = runner.build_condition("transformer_preallocated").eval()
    with pytest.raises(RuntimeError, match="full-sequence.*forbidden"):
        module(torch.randn(1, 2, 256))


def test_closed_route_audit_ignores_preimported_runner_from_shared_test_process() -> None:
    imported = importlib.import_module("functional_streaming_controls")
    assert (
        Path(imported.__file__).resolve()
        == (REPO / "examples/G1_ARENA_N17/functional_streaming_controls.py").resolve()
    )
    assert "functional_streaming_controls" in sys.modules
    assert runner.assert_closed_routes_absent()["closed_runner_imported"] is False


def test_closed_route_audit_rejects_import_in_the_new_runner_ast() -> None:
    with pytest.raises(RuntimeError, match="runner import"):
        runner._assert_closed_route_source(
            "import functional_streaming_controls as closed_runner\n",
            filename="synthetic_new_runner.py",
        )
    with pytest.raises(RuntimeError, match="helper call"):
        runner._assert_closed_route_source(
            "def bad():\n    evaluate_cost_gates()\n",
            filename="synthetic_new_runner.py",
        )


def test_generic_dispatch_keeps_preallocated_mutable_semantics_separate() -> None:
    module = runner.build_condition("transformer_preallocated").eval()
    state = runner.new_state("transformer_preallocated", module, 1)
    ids = state.storage_identities()
    with torch.inference_mode():
        output, returned = runner.online_step(
            "transformer_preallocated", module, torch.randn(1, 256), state
        )
    assert output.shape == (1, 256)
    assert returned is state
    assert state.live_index == 1
    assert state.storage_identities() == ids
    assert runner._state_logical_bytes(state) == 4_096
    assert runner._state_capacity_bytes(state) == 3_788_800

    functional = runner.build_condition("transformer_functional_cat").eval()
    functional_state = runner.new_state("transformer_functional_cat", functional, 1)
    with torch.inference_mode():
        _, functional_returned = runner.online_step(
            "transformer_functional_cat",
            functional,
            torch.randn(1, 256),
            functional_state,
        )
    assert functional_returned is not functional_state
    assert functional_returned.observations == 1


def test_next_step_branch_lifecycle_releases_functional_kv_and_keeps_preallocated() -> None:
    prefill_observations = 924
    prefill_shape = (1, 8, prefill_observations, 32)
    functional_prefill = StreamingKVState(
        tuple((torch.zeros(prefill_shape), torch.zeros(prefill_shape)) for _ in range(2)),
        batch_size=1,
        observations=prefill_observations,
    )
    functional_module = _FunctionalCatLifecycleStub()
    functional = runner._warm_next_step_branch(
        "transformer_functional_cat",
        functional_module,
        torch.nn.Identity(),
        torch.zeros(1, 256),
        scope="temporal_only",
        prefilled=functional_prefill,
        expected_observations=925,
    )
    assert functional["branch_logical_state_bytes"] == 3_788_800
    assert functional["returned_is_prefilled"] is False
    assert runner._state_logical_bytes(functional_prefill) == 3_784_704
    gc.collect()
    assert functional_module.branch_ref is not None
    assert functional_module.branch_ref() is None

    shape = (1, 8, 925, 32)
    preallocated = runner.PreallocatedKVState(
        tuple(
            (
                torch.full(shape, KV_SENTINEL),
                torch.full(shape, KV_SENTINEL),
            )
            for _ in range(2)
        ),
        batch_size=1,
        capacity=925,
        live_index=prefill_observations,
    )
    # Mark the live prefix as committed while preserving the one-slot tail.
    for tensor in preallocated.tensors():
        tensor[:, :, :prefill_observations].zero_()
    preallocated_module = _PreallocatedLifecycleStub()
    mutable = runner._warm_next_step_branch(
        "transformer_preallocated",
        preallocated_module,
        torch.nn.Identity(),
        torch.zeros(1, 256),
        scope="temporal_only",
        prefilled=preallocated,
        expected_observations=925,
    )
    assert mutable["branch_logical_state_bytes"] == 3_788_800
    assert mutable["returned_is_prefilled"] is True
    assert preallocated.live_index == prefill_observations
    assert preallocated.unwritten_is_sentinel()
    assert preallocated_module.branch_ref is not None
    assert preallocated_module.branch_ref() is preallocated


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_ONLINE_SCALING_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="explicit opt-in GPU lifecycle check",
)
def test_cuda_functional_t925_next_step_restores_prefill_allocated_baseline() -> None:
    device = torch.device("cuda", 0)
    module = runner.build_condition("transformer_functional_cat").eval().to(device)
    descriptors = torch.randn(1, 925, 256, device=device)
    _, prefilled = runner._untimed_prefix(
        "transformer_functional_cat",
        module,
        torch.nn.Identity(),
        descriptors[:, :924],
        scope="temporal_only",
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    evidence = runner._warm_next_step_branch(
        "transformer_functional_cat",
        module,
        torch.nn.Identity(),
        descriptors[:, 924],
        scope="temporal_only",
        prefilled=prefilled,
        expected_observations=925,
    )
    assert evidence["branch_logical_state_bytes"] == 3_788_800
    del evidence
    gc.collect()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == baseline


@pytest.mark.parametrize("condition", runner.CONDITIONS)
def test_recurrence_invariants_enforce_reset_return_and_replay_contract(
    condition: str,
) -> None:
    module = (
        _CPUFunctionalMambaStub()
        if condition == "functional_mamba"
        else runner.build_condition(condition).eval()
    )
    descriptors = torch.randn(
        1,
        128,
        256,
        generator=torch.Generator().manual_seed(230923),
    )
    invariants, replay_output, replay_state = runner.recurrence_invariants(
        condition,
        module,
        descriptors,
    )
    assert replay_output.shape == (1, 128, 256)
    assert replay_state.observations == 128
    assert invariants["reset_returned_observations"] == 0
    assert invariants["reset_alias_contract"]
    assert invariants["reset_returned_is_input"] is (condition == "transformer_preallocated")
    assert invariants["episode_reset_output_exact"]
    assert invariants["episode_reset_terminal_state_exact"]
    assert invariants["episode_reset_exact"]


def test_plan_and_dry_run_are_explicitly_nonformal_and_cost_fail_closed() -> None:
    payload = runner.dry_run_payload()
    assert payload["status"] == "dry_run_only"
    assert payload["formal_execution_permitted_after_committed_plan"] is True
    assert payload["result_schema"] == {
        "status": "ready_not_run",
        "correctness_not_run": True,
        "cost_not_run": True,
        "cost_not_implemented": False,
        "aggregate_not_run": True,
        "aggregate_not_implemented": False,
        "training_allowed": False,
        "vla_model_instantiated": False,
        "optimizer_steps": 0,
    }
    assert len(payload["cost_process_interface"]) == 5
    assert payload["plan"]["closed_evidence_excluded"] == {
        "functional_streaming_controls": True,
        "transformer_parity_diagnosis": True,
    }
    assert payload["plan"]["implementation_status"]["cost_measurement"] is True


def test_correctness_gates_fail_closed_before_gradient_and_full_lengths() -> None:
    execution = {
        "native_vla_modules": {"before": [], "after": []},
        "optimizer": {"constructor_calls": [], "steps_observed": 0},
    }
    gates = runner.evaluate_correctness({}, execution)
    assert gates["native_vla_modules_absent"]
    assert gates["optimizer_steps_zero"]
    assert not gates[
        "all_lengths_model_specific_oracles_invariants_state_and_shared_transformer_pass"
    ]
    assert not gates["t128_directional_jvp_gradcheck"]
    assert not gates["t128_persistent_gru_full_input_gradient"]
    assert not all(gates.values())


def test_correctness_distinguishes_directional_and_gru_full_gradient_gates() -> None:
    execution = {
        "native_vla_modules": {"before": [], "after": []},
        "optimizer": {"constructor_calls": [], "steps_observed": 0},
    }
    directional = {
        name: {"length": runner.GRADIENT_LENGTH, "passed": True}
        for name in ("functional_mamba", "transformer_functional_cat")
    }
    gru_gates = {
        name: True
        for name in (
            "all_outputs_and_gradients_finite",
            "all_outputs_and_gradients_nonzero",
            "persistent_manual_output_max_abs",
            "persistent_native_output_max_abs",
            "persistent_manual_gradient_relative_l2",
            "persistent_native_gradient_relative_l2",
            "target_modes_restored_to_eval",
        )
    }
    correctness = {
        "gradient_probe": {
            "directional": directional,
            "persistent_gru_full_input_gradient": {
                "kind": "three_path_full_input_gradient",
                "length": runner.GRADIENT_LENGTH,
                "finite_difference": False,
                "gates": gru_gates,
                "passed": True,
            },
        }
    }
    gates = runner.evaluate_correctness(correctness, execution)
    assert gates["t128_directional_jvp_gradcheck"]
    assert gates["t128_persistent_gru_full_input_gradient"]

    correctness["gradient_probe"]["persistent_gru_full_input_gradient"]["passed"] = False
    gates = runner.evaluate_correctness(correctness, execution)
    assert gates["t128_directional_jvp_gradcheck"]
    assert not gates["t128_persistent_gru_full_input_gradient"]


def test_cpu_transformer_directional_gradient_probe_is_finite_nonzero() -> None:
    condition = "transformer_functional_cat"
    module = runner.build_condition(condition).eval()
    descriptors = torch.randn(1, 4, 256, generator=torch.Generator().manual_seed(29))
    result = runner.directional_gradient_probe(condition, module, descriptors)
    assert result["length"] == 4
    assert result["gradient_finite"]
    assert result["gradient_nonzero"]
    assert result["passed"], result


def test_mamba_transformer_directional_probe_source_is_frozen_from_r3() -> None:
    assert _function_source_bundle_sha256(("directional_gradient_probe",)) == (
        "4bd6fe039a671d330f0a05b709ed51ada96cbb3b956cd623f86c907355d2dee0"
    )


def test_r4_gru_probe_manual_oracle_and_cost_sources_are_frozen() -> None:
    assert _function_source_bundle_sha256(("manual_gru_fp64",)) == (
        "f142af9f557d5d4c70d0bdb5fa62ac2c276c1c97e93a45f13188bf9ed0984a9d"
    )
    assert _function_source_bundle_sha256(("persistent_gru_full_gradient_probe",)) == (
        "2d9420909146d161cde784dee57438bfc407330b5cece6791ec9841e2f9f8cbd"
    )
    gru_program = (
        "directional_gradient_training_scope",
        "_manual_layer_norm",
        "manual_gru_fp64",
        "_relative_l2",
        "evaluate_gru_full_gradient_gates",
        "_gru_cuda_input_gradient",
        "persistent_gru_full_gradient_probe",
    )
    assert _function_source_bundle_sha256(gru_program) == (
        "f87a483f82eccc594dcb7570dee21d574c05df3f73c601dc0c434ea7e7b9a58e"
    )
    cost_and_aggregate_program = (
        "percentile",
        "_latency_summary",
        "_measurement_source",
        "_descriptor_for_step",
        "_untimed_prefix",
        "_timed_prefix_frames",
        "_rollback_preallocated_step",
        "_is_cuda_kernel_event",
        "_warm_next_step_branch",
        "_timed_next_step_branch",
        "measure_prefilled_next_step",
        "benchmark_scope_length",
        "_nvidia_smi_csv",
        "gpu_telemetry",
        "expected_execution_order",
        "execute_cost_measurements",
        "_finite_vector",
        "_finite_matrix",
        "_same_number",
        "_memory_arithmetic_valid",
        "_next_step_memory_lifecycle_valid",
        "_latency_payload_valid",
        "_prefill_expected_bytes",
        "_next_step_payload_valid",
        "_telemetry_point_valid",
        "_measurement_payload_valid",
        "evaluate_cost_result",
        "_measurement",
        "_safe_ratio",
        "aggregate_decision_metrics",
        "evaluate_aggregate_result",
        "_cost_chain_valid",
        "execute_cost_run",
        "execute_aggregate_run",
    )
    assert _function_source_bundle_sha256(cost_and_aggregate_program) == (
        "e7111b91b1e06b3c0f130dd60391a185984faac1a9b553f7faed6f6ebaffde3a"
    )


def test_manual_t4_forward_and_input_gradient_match_native_double() -> None:
    module = runner.build_condition("persistent_gru").eval()
    state = {
        name: value.detach().cpu().float().clone() for name, value in module.state_dict().items()
    }
    descriptor = torch.randn(
        1,
        4,
        runner.WIDTH,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(11),
    )
    cotangent = torch.randn(
        descriptor.shape,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(13),
    )

    manual_output, manual_input = runner.manual_gru_fp64(descriptor, state)
    manual_gradient = torch.autograd.grad(
        torch.sum(manual_output * cotangent), manual_input, create_graph=False
    )[0]

    native = runner.build_condition("persistent_gru").double().eval()
    native.load_state_dict(state, strict=True)
    native_input = descriptor.clone().requires_grad_(True)
    native_output = native(native_input)
    native_gradient = torch.autograd.grad(
        torch.sum(native_output * cotangent), native_input, create_graph=False
    )[0]

    torch.testing.assert_close(manual_output, native_output, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(manual_gradient, native_gradient, rtol=2e-10, atol=2e-11)


def test_manual_gru_is_independent_and_keeps_input_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runner.build_condition("persistent_gru").eval()
    state = {
        name: value.detach().cpu().float().clone() for name, value in module.state_dict().items()
    }
    descriptor = torch.randn(1, 4, runner.WIDTH, dtype=torch.float64, requires_grad=True)

    source = inspect.getsource(runner.manual_gru_fp64)
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"detach", "persistent_stream", "persistent_gru_fp64_oracle"} & calls

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise AssertionError("target or existing oracle reached")

    monkeypatch.setattr(torch.nn.GRU, "forward", forbidden)
    monkeypatch.setattr(type(module), "persistent_stream", forbidden)
    monkeypatch.setattr(runner, "persistent_gru_fp64_oracle", forbidden)
    output, manual_input = runner.manual_gru_fp64(descriptor, state)
    gradient = torch.autograd.grad(output.sum(), manual_input)[0]
    assert manual_input is descriptor
    assert output.shape == descriptor.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("persistent", [True, False])
def test_gru_cuda_objective_restores_modes_on_cpu_success(persistent: bool) -> None:
    module = runner.build_condition("persistent_gru").eval()
    descriptors = torch.randn(1, 4, runner.WIDTH)
    cotangent = torch.randn_like(descriptors)
    output, gradient = runner._gru_cuda_input_gradient(
        module,
        descriptors,
        cotangent,
        persistent=persistent,
    )
    assert output.shape == descriptors.shape
    assert torch.isfinite(gradient).all()
    assert all(not child.training for child in module.modules())


@pytest.mark.parametrize("persistent", [True, False])
def test_gru_cuda_objective_restores_modes_after_forced_exception(
    persistent: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runner.build_condition("persistent_gru").eval()
    descriptors = torch.randn(1, 4, runner.WIDTH)
    cotangent = torch.randn_like(descriptors)

    def forced_failure(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        assert all(child.training for child in module.modules())
        raise RuntimeError("forced GRU target failure")

    if persistent:
        monkeypatch.setattr(module, "persistent_stream", forced_failure)
    else:
        monkeypatch.setattr(module, "forward", forced_failure)
    with pytest.raises(RuntimeError, match="forced GRU target failure"):
        runner._gru_cuda_input_gradient(
            module,
            descriptors,
            cotangent,
            persistent=persistent,
        )
    assert all(not child.training for child in module.modules())


def test_non_gru_probe_does_not_change_module_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = runner.build_condition("transformer_functional_cat").eval()
    descriptors = torch.randn(1, 4, 256, generator=torch.Generator().manual_seed(41))
    original = runner.differentiable_stream
    observed_flags: list[tuple[bool, ...]] = []

    def recording_stream(name: str, target: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
        observed_flags.append(tuple(child.training for child in target.modules()))
        return original(name, target, value)

    monkeypatch.setattr(runner, "differentiable_stream", recording_stream)
    result = runner.directional_gradient_probe("transformer_functional_cat", module, descriptors)

    assert result["passed"], result
    assert len(observed_flags) == 3
    assert all(not any(flags) for flags in observed_flags)
    assert all(not child.training for child in module.modules())


@pytest.mark.parametrize("fault", ["entry_training", "count", "layers", "dropout"])
def test_gru_training_scope_structure_is_fail_closed(fault: str) -> None:
    module = runner.build_condition("persistent_gru").eval()
    if fault == "entry_training":
        next(child for child in module.modules() if isinstance(child, torch.nn.GRU)).train()
    elif fault == "count":
        module.layers[1].gru = torch.nn.Identity()
    elif fault == "layers":
        module.layers[0].gru.num_layers = 2
    elif fault == "dropout":
        module.layers[0].gru.dropout = 0.1
    else:  # pragma: no cover - parametrization is frozen above
        raise AssertionError(f"unknown test fault: {fault}")

    descriptors = torch.randn(1, 4, 256, generator=torch.Generator().manual_seed(43))
    with pytest.raises(ValueError, match="persistent GRU gradient probe requires"):
        runner._gru_cuda_input_gradient(
            module,
            descriptors,
            torch.randn_like(descriptors),
            persistent=True,
        )


def test_gru_full_gradient_gate_recomputes_positive_and_negative_cases() -> None:
    base = torch.randn(1, 4, 8, dtype=torch.float64)
    outputs = {
        "cuda_persistent": base.float(),
        "cuda_native_full": base.float(),
        "cpu_fp64_manual": base,
    }
    gradients = {name: value.clone() for name, value in outputs.items()}
    positive, metrics = runner.evaluate_gru_full_gradient_gates(
        outputs, gradients, modes_restored=True
    )
    assert all(positive.values())
    assert metrics["gradient_relative_l2"]["persistent_vs_manual"] < 1e-6

    bad_gradient = {name: value.clone() for name, value in gradients.items()}
    bad_gradient["cpu_fp64_manual"] *= 2
    negative, _ = runner.evaluate_gru_full_gradient_gates(
        outputs, bad_gradient, modes_restored=True
    )
    assert not negative["persistent_manual_gradient_relative_l2"]

    bad_output = {name: value.clone() for name, value in outputs.items()}
    bad_output["cuda_native_full"][0, 0, 0] += 1
    negative, _ = runner.evaluate_gru_full_gradient_gates(
        bad_output, gradients, modes_restored=True
    )
    assert not negative["persistent_native_output_max_abs"]

    zero = {name: value.clone() for name, value in gradients.items()}
    zero["cuda_native_full"].zero_()
    negative, _ = runner.evaluate_gru_full_gradient_gates(outputs, zero, modes_restored=False)
    assert not negative["all_outputs_and_gradients_nonzero"]
    assert not negative["target_modes_restored_to_eval"]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_ONLINE_SCALING_GRU_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="explicit opt-in CUDA T128 GRU three-path full-gradient probe",
)
def test_cuda_persistent_gru_t128_full_gradient_probe_and_mode_contract() -> None:
    runtime = runner.verify_environment()
    assert runtime["matmul_tf32"] is False
    assert runtime["cudnn_tf32"] is False

    device = torch.device("cuda", 0)
    module = runner.build_condition("persistent_gru").eval().to(device)
    descriptors = torch.randn(
        1,
        runner.GRADIENT_LENGTH,
        256,
        device=device,
        generator=torch.Generator(device=device).manual_seed(47),
    )

    result = runner.persistent_gru_full_gradient_probe(module, descriptors)
    shared_generator = torch.Generator(device=device).manual_seed(runner.SEED)
    expected_direction = torch.randn(
        descriptors.shape,
        generator=shared_generator,
        device=device,
        dtype=torch.float32,
    )
    expected_direction /= expected_direction.norm().clamp_min(torch.finfo(torch.float32).eps)
    expected_cotangent = torch.randn(
        descriptors.shape,
        generator=shared_generator,
        device=device,
        dtype=torch.float32,
    )
    expected_cotangent /= expected_cotangent.norm().clamp_min(torch.finfo(torch.float32).eps)
    assert result["kind"] == "three_path_full_input_gradient"
    assert result["length"] == 128
    assert result["finite_difference"] is False
    assert result["rng"]["generation_order"] == ["direction", "cotangent"]
    assert result["rng"]["direction_participates_in_gate"] is False
    assert result["rng"]["direction_sha256"] == runner.tensor_sha256(expected_direction)
    assert result["rng"]["cotangent_sha256"] == runner.tensor_sha256(expected_cotangent)
    assert all(result["gates"].values())
    assert result["passed"], result
    assert all(not child.training for child in module.modules())


def test_cuda_preflight_source_orders_formal_runtime_before_frozen_probe() -> None:
    source = inspect.getsource(test_cuda_persistent_gru_t128_full_gradient_probe_and_mode_contract)
    verify_offset = source.index("runner.verify_environment()")
    probe_offset = source.index("runner.persistent_gru_full_gradient_probe")
    assert verify_offset < probe_offset
    assert 'runtime["matmul_tf32"] is False' in source
    assert 'runtime["cudnn_tf32"] is False' in source


def test_latency_raw_and_memory_arithmetic_recompute_and_fail_closed() -> None:
    host = [[float(frame + sample + 1) for frame in range(4)] for sample in range(100)]
    cuda = [[value / 2 for value in row] for row in host]
    latency = runner._latency_summary(host, cuda)
    assert runner._latency_payload_valid(latency, 4)
    missing_raw = copy.deepcopy(latency)
    missing_raw["raw"].pop("host_per_frame_ms")
    assert not runner._latency_payload_valid(missing_raw, 4)
    changed_summary = copy.deepcopy(latency)
    changed_summary["last_frame"]["host_p95_ms"] += 1e-3
    assert not runner._latency_payload_valid(changed_summary, 4)

    memory = {
        "baseline_allocated_bytes": 100,
        "baseline_reserved_bytes": 200,
        "absolute_peak_allocated_bytes": 150,
        "absolute_peak_reserved_bytes": 260,
        "delta_peak_allocated_bytes": 50,
        "delta_peak_reserved_bytes": 60,
    }
    assert runner._memory_arithmetic_valid(memory)
    corrupted = dict(memory, delta_peak_allocated_bytes=49)
    assert not runner._memory_arithmetic_valid(corrupted)


def test_preallocated_next_step_schema_uses_t_minus_one_live_and_fixed_capacity() -> None:
    host = [1.0] * runner.MEASURED_SAMPLES
    cuda = [0.5] * runner.MEASURED_SAMPLES
    memory = {
        "baseline_allocated_bytes": 100,
        "baseline_reserved_bytes": 200,
        "absolute_peak_allocated_bytes": 120,
        "absolute_peak_reserved_bytes": 240,
        "delta_peak_allocated_bytes": 20,
        "delta_peak_reserved_bytes": 40,
        "sample_baseline_allocated_bytes": [100] * runner.MEASURED_SAMPLES,
        "sample_baseline_reserved_bytes": [200] * runner.MEASURED_SAMPLES,
        "sample_peak_allocated_bytes": [120] * runner.MEASURED_SAMPLES,
        "sample_peak_reserved_bytes": [240] * runner.MEASURED_SAMPLES,
        "sample_restored_allocated_bytes": [100] * runner.MEASURED_SAMPLES,
        "sample_restored_reserved_bytes": [240] * runner.MEASURED_SAMPLES,
        "baseline_allocated_stable": True,
        "allocated_restored_after_each_sample": True,
        "pre_profiler_baseline_allocated_bytes": 100,
        "prefilled_baseline_recovered_before_profiler": True,
    }
    length = 925
    payload = {
        "prefill_observations": length - 1,
        "probe_finishes_at_observations": length,
        "warmup_samples": runner.WARMUP_SAMPLES,
        "measurements": runner.MEASURED_SAMPLES,
        "allocator_peak_reset_after_prefill": True,
        "raw": {"host_samples_ms": host, "cuda_samples_ms": cuda},
        "host_p50_ms": 1.0,
        "host_p95_ms": 1.0,
        "cuda_p50_ms": 0.5,
        "cuda_p95_ms": 0.5,
        "logical_state_bytes": runner.transformer_kv_expected_bytes(
            batch_size=1, observations=length - 1
        ),
        "capacity_state_bytes": runner.preallocated_capacity_expected_bytes(),
        "memory": memory,
        "branch_logical_state_bytes": [
            runner.EXPECTED_STATE_BYTES["transformer_preallocated"][length]
        ]
        * runner.MEASURED_SAMPLES,
        "returned_is_prefilled": [True] * runner.MEASURED_SAMPLES,
        "profiler": {
            "runs": 1,
            "cuda_device_event_count": 10,
            "cuda_kernel_count": 8,
            "excluded_memcpy_memset_count": 2,
            "profiled_state_observations": length,
        },
        "outputs_finite": True,
        "observations_exact": True,
        "preallocated_state_rolled_back": True,
    }
    assert runner._next_step_payload_valid(
        payload, "transformer_preallocated", length, "temporal_only"
    )
    current = copy.deepcopy(payload)
    current["logical_state_bytes"] = 0
    current["capacity_state_bytes"] = 0
    current["branch_logical_state_bytes"] = [0] * runner.MEASURED_SAMPLES
    current["returned_is_prefilled"] = [False] * runner.MEASURED_SAMPLES
    current["profiler"]["cuda_device_event_count"] = 0
    current["profiler"]["cuda_kernel_count"] = 0
    current["profiler"]["excluded_memcpy_memset_count"] = 0
    assert runner._next_step_payload_valid(current, "current_descriptor", length, "temporal_only")
    payload["logical_state_bytes"] += 4096
    assert not runner._next_step_payload_valid(
        payload, "transformer_preallocated", length, "temporal_only"
    )


def test_telemetry_requires_finite_isolated_gpu0_rows() -> None:
    telemetry = _valid_telemetry()
    assert runner._telemetry_point_valid(telemetry)
    busy = copy.deepcopy(telemetry)
    busy["gpu0_other_compute_processes"] = [
        {
            "gpu_uuid": "GPU-unit",
            "pid": "999",
            "process_name": "other",
            "used_gpu_memory_mib": "100",
        }
    ]
    assert not runner._telemetry_point_valid(busy)
    nonfinite = copy.deepcopy(telemetry)
    nonfinite["gpu"]["temperature_c"] = "nan"
    assert not runner._telemetry_point_valid(nonfinite)


def test_aggregate_threshold_boundaries_and_four_of_five_rule() -> None:
    costs = [_decision_cost(label) for label in runner.COST_RUN_IDS]
    decision = runner.aggregate_decision_metrics(costs)
    assert decision["online_mamba_engineering_scaling_advantage"]
    assert all(decision["decision_gates"].values())

    strict_latency = copy.deepcopy(costs)
    strict_latency[0]["cost"]["measurements"]["functional_mamba"]["925"]["end_to_end"]["latency"][
        "per_frame_flat"
    ]["host_p95_ms"] = 20.0
    assert not runner.aggregate_decision_metrics(strict_latency)["decision_gates"][
        "all_five_mamba_t925_end_to_end_p95_below_20ms"
    ]

    four_directional = [
        _decision_cost(label, next_ratio=0.80 if index < 4 else 1.01)
        for index, label in enumerate(runner.COST_RUN_IDS)
    ]
    assert runner.aggregate_decision_metrics(four_directional)["decision_gates"][
        "mamba_next_step_at_least_20pct_lower_in_4of5_and_median"
    ]
    bad_median = [
        _decision_cost(label, next_ratio=0.81 if index < 4 else 1.01)
        for index, label in enumerate(runner.COST_RUN_IDS)
    ]
    assert not runner.aggregate_decision_metrics(bad_median)["decision_gates"][
        "mamba_next_step_at_least_20pct_lower_in_4of5_and_median"
    ]


def test_cost_hash_chain_accepts_exact_files_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    identities = []
    for label in ("A", "B"):
        path = tmp_path / runner.COST_RUN_IDS[label] / "RESULT.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"label": label}) + "\n", encoding="utf-8")
        identities.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": runner.sha256(path),
            }
        )
    correctness_identity = {
        "path": str((tmp_path / runner.CORRECTNESS_RUN_ID / "RESULT.json").resolve()),
        "bytes": 10,
        "sha256": "c" * 64,
    }
    result = {
        "correctness_predecessor": correctness_identity,
        "predecessor_chain": identities,
        "cost_predecessor": identities[-1],
    }
    assert runner._cost_chain_valid(result, label="C", correctness_identity=correctness_identity)
    corrupted = copy.deepcopy(result)
    corrupted["predecessor_chain"][0]["sha256"] = "0" * 64
    assert not runner._cost_chain_valid(
        corrupted, label="C", correctness_identity=correctness_identity
    )


def test_write_once_result_artifact_identity_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    result = {
        "kind": "correctness",
        "status": "passed",
        "gates": {"core": True},
        "cost_allowed": True,
    }
    identity = runner.write_result_once("unit", result)
    stored = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))
    assert stored["status"] == "passed"
    assert stored["artifact"]["result_bytes"] == identity["bytes"]
    assert (
        stored["artifact"]["root_bytes_before"] + identity["bytes"]
        == stored["artifact"]["root_bytes_after_write"]
    )
    with pytest.raises(FileExistsError):
        runner.write_result_once("unit", result)


def test_attribution_binding_is_rechecked_for_prepare_verify_and_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()

    def material(name: str, payload: bytes) -> dict[str, object]:
        path = evidence_root / name
        path.write_bytes(payload)
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": runner.sha256(path),
        }

    functional = material("functional.json", b"functional")
    recheck = material("recheck.json", b"recheck")
    attribution = material("RESULT.json", b"attribution")
    gradients = material("GRADIENTS.npz", b"gradients")
    monkeypatch.setattr(runner, "FUNCTIONAL_AGGREGATE", functional)
    monkeypatch.setattr(runner, "FUNCTIONAL_RECHECK", recheck)
    monkeypatch.setattr(runner, "GRU_ATTRIBUTION_RESULT", attribution)
    monkeypatch.setattr(runner, "GRU_ATTRIBUTION_GRADIENTS", gradients)

    artifact_root = tmp_path / "artifact"
    plan_path = artifact_root / "plan-r5/PLAN.json"
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(runner, "PLAN_PATH", plan_path)
    commit = "1" * 40
    monkeypatch.setattr(runner, "verify_source", lambda expected: {"commit": expected})
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: commit + "\n",
    )

    identity = runner.prepare_plan(plan_path)
    plan, checked = runner.verify_plan(plan_path, identity["sha256"])
    assert checked == identity
    assert plan["qualified_prior_evidence"]["gru_gradient_attribution_result"] == attribution
    assert plan["qualified_prior_evidence"]["gru_gradient_attribution_gradients"] == gradients

    Path(str(gradients["path"])).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="attribution gradients identity changed"):
        runner.verify_plan(plan_path, identity["sha256"])
    with pytest.raises(ValueError, match="attribution gradients identity changed"):
        runner._common_verified_inputs(plan)


def test_resolved_plan_write_read_verify_roundtrip_preserves_json_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifact"
    plan_path = artifact_root / "plan-r5/PLAN.json"
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(runner, "PLAN_PATH", plan_path)
    commit = "1" * 40
    fresh = runner.resolved_plan(commit)
    expected_state_bytes = fresh["protocol"]["expected_state_bytes"]
    assert (
        sum(
            isinstance(length, int)
            for condition in expected_state_bytes.values()
            for length in condition
        )
        == 20
    )

    identity = runner._write_new_json(plan_path, fresh)
    loaded, verified_identity = runner.verify_plan(plan_path, identity["sha256"])

    assert verified_identity == identity
    assert loaded == json.loads(json.dumps(fresh, sort_keys=True))
    assert all(
        isinstance(length, str)
        for condition in loaded["protocol"]["expected_state_bytes"].values()
        for length in condition
    )


def test_cli_dry_run_is_json_and_does_not_materialize_artifact() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = runner.FROZEN_RUNTIME["pythonpath"]
    before = runner.ARTIFACT_ROOT.exists()
    completed = subprocess.run(
        [runner.FROZEN_RUNTIME["python_executable"], str(EXAMPLE), "--dry-run"],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run_only"
    assert payload["result_schema"]["cost_not_implemented"] is False
    assert runner.ARTIFACT_ROOT.exists() is before
