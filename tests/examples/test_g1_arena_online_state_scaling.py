from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from gr00t.experimental.arena_functional_mamba_transition import FunctionalMamba2State
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


def test_frozen_five_process_orders_lengths_and_incomplete_status() -> None:
    assert runner.CONDITIONS == (
        "functional_mamba",
        "persistent_gru",
        "transformer_functional_cat",
        "transformer_preallocated",
        "current_descriptor",
    )
    assert runner.LENGTHS == (128, 256, 512, 925)
    assert runner.COST_RUN_IDS == {
        "A": "cost-gpu0-r1a",
        "B": "cost-gpu0-r1b",
        "C": "cost-gpu0-r1c",
        "D": "cost-gpu0-r1d",
        "E": "cost-gpu0-r1e",
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
    assert not all(gates.values())


@pytest.mark.parametrize("condition", ["persistent_gru", "transformer_functional_cat"])
def test_cpu_directional_gradient_probe_is_finite_nonzero(condition: str) -> None:
    module = runner.build_condition(condition).eval()
    descriptors = torch.randn(1, 4, 256, generator=torch.Generator().manual_seed(29))
    result = runner.directional_gradient_probe(condition, module, descriptors)
    assert result["length"] == 4
    assert result["gradient_finite"]
    assert result["gradient_nonzero"]
    assert result["passed"], result


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
