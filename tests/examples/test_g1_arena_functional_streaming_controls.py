from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_ARENA_N17"
sys.path.insert(0, str(EXAMPLE))
import functional_streaming_controls as runner  # noqa: E402


def _execution() -> dict[str, object]:
    return {
        "native_vla_modules": {"before": [], "after": []},
        "optimizer": {"constructor_calls": [], "steps_observed": 0},
    }


def _passing_correctness_payload() -> dict[str, object]:
    invariant = {
        "output_shape": [1, runner.CORRECTNESS_LENGTH, runner.WIDTH],
        "output_finite": True,
        "state_finite": True,
        "final_observations": runner.CORRECTNESS_LENGTH,
        "continuous_chunk_exact": True,
        "repeat_exact": True,
        "reset_exact": True,
        "suffix_edit_prefix_exact": True,
        "episode_reset_clears_observations": True,
        "input_state_unchanged": True,
        "input_output_state_storage_disjoint": True,
        "transformer_kv_bytes_exact": True,
    }
    return {
        "tokenizer": {
            "parameter_count": 1,
            "state_dict_sha256": "1" * 64,
            "descriptor_shape": [1, runner.CORRECTNESS_LENGTH, runner.WIDTH],
            "descriptor_dtype": "torch.float32",
            "descriptor_finite": True,
            "descriptor_sha256": "2" * 64,
            "shared_descriptor_tensor_for_all_conditions": True,
        },
        "parameters": {
            "expected_temporal_counts": copy.deepcopy(runner.EXPECTED_TEMPORAL_PARAMETERS),
            "actual_temporal_counts": copy.deepcopy(runner.EXPECTED_TEMPORAL_PARAMETERS),
            "temporal_state_dict_sha256": {
                name: f"{index + 3:x}" * 64 for index, name in enumerate(runner.CONDITIONS)
            },
            "all_parameter_objects_unique": True,
            "transformer_reuses_exact_full_parameter_set": True,
        },
        "functional_mamba": {
            "frames": runner.CORRECTNESS_LENGTH,
            "output_max_abs": runner.MAX_PARITY_ABS,
            "state_max_abs": runner.MAX_PARITY_ABS,
            "all_outputs_finite": True,
            "all_states_finite": True,
            "official_observations": runner.CORRECTNESS_LENGTH,
            "functional_observations": runner.CORRECTNESS_LENGTH,
            "functional_input_state_unchanged": True,
            "input_output_state_storage_disjoint": True,
        },
        "persistent_gru": {
            "explicit_full_stream_exact": True,
            "max_abs": 0.0,
            "output_finite": True,
            "final_observations": runner.CORRECTNESS_LENGTH,
        },
        "streaming_kv_transformer": {
            "full_stream_max_abs": runner.MAX_PARITY_ABS,
            "output_finite": True,
            "final_observations": runner.CORRECTNESS_LENGTH,
        },
        "invariants": {name: copy.deepcopy(invariant) for name in runner.CONDITIONS},
    }


def _parameter_identity(correctness: dict[str, object]) -> dict[str, object]:
    parameters = correctness["parameters"]
    assert isinstance(parameters, dict)
    tokenizer = correctness["tokenizer"]
    assert isinstance(tokenizer, dict)
    return {
        "tokenizer_state_dict_sha256": tokenizer["state_dict_sha256"],
        "all_tokenizer_hashes_equal": True,
        "temporal_state_dict_sha256": parameters["temporal_state_dict_sha256"],
        "temporal_parameter_counts": parameters["actual_temporal_counts"],
    }


def _write_result(
    path: Path,
    result: dict[str, object],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_bytes_before = runner.directory_bytes(runner.ARTIFACT_ROOT)
    result["gates"]["artifact_total_at_most_256mib"] = True
    result["status"] = "passed"
    for _ in range(10):
        encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        artifact = {
            "root": str(runner.ARTIFACT_ROOT.resolve()),
            "bytes_before_result": artifact_bytes_before,
            "result_bytes": len(encoded),
            "total_bytes": artifact_bytes_before + len(encoded),
            "limit_bytes": runner.MAX_ARTIFACT_BYTES,
            "traces_written": 0,
        }
        if result.get("artifact") == artifact:
            break
        result["artifact"] = artifact
    else:
        raise AssertionError("test artifact size did not converge")
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert result["artifact"]["result_bytes"] == len(encoded)
    path.write_bytes(encoded)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": runner.sha256(path),
        "run_id": result["run_id"],
    }


def _latency(condition: str, length: int, scope: str) -> float:
    index = runner.LENGTHS.index(length)
    values = {
        "functional_mamba": (1.0, 1.05, 1.10, 1.20),
        "persistent_gru": (1.0, 1.02, 1.04, 1.06),
        "streaming_kv_transformer": (1.0, 1.20, 1.50, 2.00),
        "current_descriptor": (0.10, 0.10, 0.10, 0.10),
    }
    value = values[condition][index]
    return value + (0.25 if scope == "end_to_end" else 0.0)


def _measurement(condition: str, length: int, scope: str) -> dict[str, object]:
    index = runner.LENGTHS.index(length)
    host_value = _latency(condition, length, scope)
    host = [host_value] * runner.MEASURED_SAMPLES
    cuda = [host_value * 0.8] * runner.MEASURED_SAMPLES
    if condition == "streaming_kv_transformer":
        state_bytes = runner.transformer_kv_expected_bytes(batch_size=1, observations=length)
    elif condition == "current_descriptor":
        state_bytes = 0
    else:
        state_bytes = 4_096

    mamba_peak = 1_000 + 100 * index
    if condition == "streaming_kv_transformer":
        allocated_peak = int(mamba_peak * (2.0 + 0.2 * index))
    elif condition == "functional_mamba":
        allocated_peak = mamba_peak
    else:
        allocated_peak = 1_500 + 100 * index
    baseline_allocated = 100
    baseline_reserved = 200
    absolute_peak_reserved = allocated_peak + 500
    next_host = [0.5] * runner.MEASURED_SAMPLES
    next_cuda = [0.4] * runner.MEASURED_SAMPLES
    return {
        "condition": condition,
        "length": length,
        "scope": scope,
        "allocator_lifecycle": list(runner.FROZEN_PROTOCOL["memory"]["allocator_order"]),
        "residency": {
            "condition": condition,
            "all_model_parameters_on_cuda0": True,
            "resident_input_device": "cuda:0",
            "other_condition_models_constructed_concurrently": [],
            "temporal_model_instances_resident": 1,
        },
        "warmup_samples": runner.WARMUP_SAMPLES,
        "measurements": runner.MEASURED_SAMPLES,
        "pre_and_post_sample_cuda_synchronize": True,
        "inference_mode": True,
        "fresh_state_each_prefix_sample": True,
        "host_wall_samples_ms": host,
        "host_wall_p50_ms": runner.percentile(host, 50),
        "host_wall_p95_ms": runner.percentile(host, 95),
        "cuda_event_samples_ms": cuda,
        "cuda_event_p50_ms": runner.percentile(cuda, 50),
        "cuda_event_p95_ms": runner.percentile(cuda, 95),
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "absolute_peak_allocated_bytes": allocated_peak,
            "delta_peak_allocated_bytes": allocated_peak - baseline_allocated,
            "absolute_peak_reserved_bytes": absolute_peak_reserved,
            "delta_peak_reserved_bytes": absolute_peak_reserved - baseline_reserved,
        },
        "state_unique_storage_bytes": state_bytes,
        "state_bytes_samples": [state_bytes] * runner.MEASURED_SAMPLES,
        "final_state_observations": length,
        "measured_state_observations": [length] * runner.MEASURED_SAMPLES,
        "all_outputs_and_states_finite": True,
        "next_step": {
            "prefill_observations": length,
            "warmup_samples": runner.WARMUP_SAMPLES,
            "measurements": runner.MEASURED_SAMPLES,
            "host_wall_samples_ms": next_host,
            "host_wall_p50_ms": runner.percentile(next_host, 50),
            "host_wall_p95_ms": runner.percentile(next_host, 95),
            "cuda_event_samples_ms": next_cuda,
            "cuda_event_p50_ms": runner.percentile(next_cuda, 50),
            "cuda_event_p95_ms": runner.percentile(next_cuda, 95),
            "kernel_profile_count": 1,
            "kernel_count": 1,
            "profiled_state_observations": length + 1,
            "outputs_finite": True,
            "observations_exact": True,
        },
    }


def _passing_cost(
    label: str,
    expected_parameter_identity: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, bool]]:
    measurements = {
        condition: {
            str(length): {scope: _measurement(condition, length, scope) for scope in runner.SCOPES}
            for length in runner.LENGTHS
        }
        for condition in runner.CONDITIONS
    }

    def temporal(condition: str, length: int) -> list[float]:
        return measurements[condition][str(length)]["temporal_only"]["host_wall_samples_ms"]

    cost = {
        "label": label,
        "seed": runner.PROCESS_SEEDS[label],
        "condition_order": list(runner.CONDITION_ORDERS[label]),
        "length_order": list(runner.LENGTH_ORDERS[label]),
        "scope_order": list(runner.SCOPES),
        "execution_order": runner._expected_execution_order(label),
        "parameter_identity": copy.deepcopy(expected_parameter_identity),
        "measurements": measurements,
        "bootstrap": runner.bootstrap_growth_effect(
            temporal("functional_mamba", 128),
            temporal("functional_mamba", 925),
            temporal("streaming_kv_transformer", 128),
            temporal("streaming_kv_transformer", 925),
            seed=runner.PROCESS_SEEDS[label],
        ),
    }
    telemetry = {
        "before": {
            "gpu": {"utilization_percent": 0, "temperature_c": 50},
            "gpu0_compute_processes": [],
        },
        "after_warmup": [
            {
                "condition": condition,
                "length": length,
                "scope": scope,
                "telemetry": {
                    "gpu": {"sm_clock_mhz": 2_100, "temperature_c": 60},
                    "gpu0_other_compute_processes": [],
                },
            }
            for condition in runner.CONDITION_ORDERS[label]
            for length in runner.LENGTH_ORDERS[label]
            for scope in runner.SCOPES
        ],
        "after": {"gpu0_other_compute_processes": []},
    }
    execution = _execution()
    gates = runner.evaluate_cost_gates(cost, telemetry, execution, expected_parameter_identity)
    assert all(gates.values())
    return cost, telemetry, execution, gates


def _passing_aggregate_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    numerical = _passing_correctness_payload()
    execution = _execution()
    correctness_gates = runner.evaluate_correctness_gates(numerical, execution)
    correctness_gates.update(
        {"wall_time_at_most_20min": True, "artifact_total_at_most_256mib": True}
    )
    correctness = {
        "status": "passed",
        "experiment_id": runner.EXPERIMENT,
        "run_id": runner.CORRECTNESS_RUN_ID,
        "source_commit": "1" * 40,
        "plan": {"sha256": "2" * 64},
        "correctness": numerical,
        "execution_evidence": execution,
        "gates": correctness_gates,
        "optimizer_steps": 0,
        "vla_model_instantiated": False,
        "training_allowed": False,
        "wall_seconds": 1.0,
    }
    correctness_identity = _write_result(
        tmp_path / runner.CORRECTNESS_RUN_ID / "RESULT.json",
        correctness,
    )
    correctness["result_identity"] = correctness_identity
    parameter_identity = _parameter_identity(numerical)
    identities = []
    costs = []
    for index, label in enumerate(runner.COST_RUN_IDS):
        cost, telemetry, cost_execution, gates = _passing_cost(label, parameter_identity)
        gates.update({"wall_time_at_most_20min": True, "artifact_total_at_most_256mib": True})
        costs.append(
            {
                "status": "passed",
                "experiment_id": runner.EXPERIMENT,
                "label": label,
                "run_id": runner.COST_RUN_IDS[label],
                "source_commit": "1" * 40,
                "plan": {"sha256": "2" * 64},
                "process": {"pid": 10_000 + index},
                "correctness_predecessor": correctness_identity,
                "cost_predecessor": None if index == 0 else identities[index - 1],
                "cost": cost,
                "telemetry": telemetry,
                "execution_evidence": cost_execution,
                "gates": gates,
                "optimizer_steps": 0,
                "vla_model_instantiated": False,
                "training_allowed": False,
                "wall_seconds": 1.0,
            }
        )
        identities.append(
            _write_result(
                tmp_path / runner.COST_RUN_IDS[label] / "RESULT.json",
                costs[-1],
            )
        )
    return correctness, costs, identities


def test_frozen_run_ids_and_cyclic_orders_match_protocol() -> None:
    assert runner.CORRECTNESS_RUN_ID == "correctness-gpu0-r1"
    assert runner.COST_RUN_IDS == {
        "A": "cost-gpu0-r1a",
        "B": "cost-gpu0-r1b",
        "C": "cost-gpu0-r1c",
        "D": "cost-gpu0-r1d",
    }
    assert runner.AGGREGATE_RUN_ID == "aggregate-r1"
    for offset, label in enumerate(runner.COST_RUN_IDS):
        assert runner.CONDITION_ORDERS[label] == (
            runner.CONDITIONS[offset:] + runner.CONDITIONS[:offset]
        )
        assert runner.LENGTH_ORDERS[label] == (runner.LENGTHS[offset:] + runner.LENGTHS[:offset])
        assert runner._expected_execution_order(label) == [
            {"condition": condition, "length": length, "scope": scope}
            for condition in runner.CONDITION_ORDERS[label]
            for length in runner.LENGTH_ORDERS[label]
            for scope in runner.SCOPES
        ]
    plan = runner.resolved_plan("1" * 40)
    assert plan["protocol"] == runner.FROZEN_PROTOCOL
    assert plan["protocol"]["percentile"] == "numpy_linear"


def test_runtime_rebinds_only_pythonpath_to_current_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor_runtime = runner.predecessor_runner.FROZEN_RUNTIME
    for key, value in predecessor_runtime.items():
        if key != "pythonpath":
            assert runner.FROZEN_RUNTIME[key] == value
    expected = f"{runner.predecessor_runner.OVERLAY_ROOT}:{runner.REPO}"
    assert runner.FROZEN_RUNTIME["pythonpath"] == expected
    assert runner.resolved_plan("1" * 40)["runtime"]["pythonpath"] == expected

    observed = {}

    def fake_verify() -> dict[str, str]:
        observed["runtime"] = runner.predecessor_runner.FROZEN_RUNTIME
        return {"pythonpath": runner.predecessor_runner.FROZEN_RUNTIME["pythonpath"]}

    monkeypatch.setattr(runner.predecessor_runner, "verify_environment", fake_verify)
    assert runner.verify_environment() == {"pythonpath": expected}
    assert observed["runtime"] is runner.FROZEN_RUNTIME
    assert runner.predecessor_runner.FROZEN_RUNTIME is predecessor_runtime


def test_runtime_rebinding_restores_predecessor_contract_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor_runtime = runner.predecessor_runner.FROZEN_RUNTIME

    def fail() -> dict[str, object]:
        assert runner.predecessor_runner.FROZEN_RUNTIME is runner.FROZEN_RUNTIME
        raise RuntimeError("expected test failure")

    monkeypatch.setattr(runner.predecessor_runner, "verify_environment", fail)
    with pytest.raises(RuntimeError, match="expected test failure"):
        runner.verify_environment()
    assert runner.predecessor_runner.FROZEN_RUNTIME is predecessor_runtime


def test_imported_predecessor_runner_is_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "examples/G1_ARENA_N17/functional_mamba_transition.py"
    assert relative in runner.SOURCE_IDENTITY_PATHS
    paths = runner.verify_imported_source_paths()
    assert paths["predecessor_runner"] == str((runner.REPO / relative).resolve())

    monkeypatch.setattr(runner.predecessor_runner, "__file__", str(tmp_path / "escaped.py"))
    with pytest.raises(ValueError, match="escaped source root"):
        runner.verify_imported_source_paths()


def test_bootstrap_uses_numpy_linear_p95_per_draw_not_resampled_means() -> None:
    arrays = [
        np.asarray([1.0] * 46 + [10.0] * 4),
        np.asarray([1.5] * 46 + [30.0] * 4),
        np.asarray([1.0] * 46 + [12.0] * 4),
        np.asarray([2.0] * 46 + [80.0] * 4),
    ]
    seed = runner.PROCESS_SEEDS["A"]
    result = runner.bootstrap_growth_effect(*[array.tolist() for array in arrays], seed=seed)
    rng = np.random.default_rng(seed)
    p95_draws = []
    mean_draws = []
    for array in arrays:
        indices = rng.integers(
            0,
            runner.MEASURED_SAMPLES,
            size=(runner.BOOTSTRAP_DRAWS, runner.MEASURED_SAMPLES),
        )
        samples = array[indices]
        p95_draws.append(np.percentile(samples, 95, axis=1, method="linear"))
        mean_draws.append(samples.mean(axis=1))
    expected_effects = (p95_draws[3] / p95_draws[2]) / (p95_draws[1] / p95_draws[0]) - 1.0
    mean_effects = (mean_draws[3] / mean_draws[2]) / (mean_draws[1] / mean_draws[0]) - 1.0
    expected_ci = np.percentile(expected_effects, [2.5, 97.5], method="linear")
    mean_ci = np.percentile(mean_effects, [2.5, 97.5], method="linear")
    assert result["sample_statistic"] == "numpy_linear_p95"
    assert result["ci95_percentile"] == pytest.approx(expected_ci.tolist())
    assert not np.allclose(expected_ci, mean_ci)


def test_verify_plan_rejects_any_protocol_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    plan = runner.resolved_plan("1" * 40)
    valid = tmp_path / "plan-r1/PLAN.json"
    valid.parent.mkdir(parents=True)
    valid.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert runner.verify_plan(valid, runner.sha256(valid))[0] == plan

    changed = copy.deepcopy(plan)
    changed["protocol"]["measured_samples"] = runner.MEASURED_SAMPLES - 1
    changed_path = tmp_path / "plan-r1/CHANGED.json"
    changed_path.write_text(
        json.dumps(changed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contract changed"):
        runner.verify_plan(changed_path, runner.sha256(changed_path))


def test_finalize_artifact_gate_matches_actual_json_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    result = {
        "run_id": "artifact-unit",
        "status": "passed",
        "wall_seconds": 1.0,
        "gates": {
            "core": True,
            "wall_time_at_most_20min": True,
            "artifact_total_at_most_256mib": True,
        },
        "filler": "x" * 12_345,
    }
    runner._finalize_artifact_gate(result, artifact_bytes_before=0)
    path = tmp_path / "artifact-unit/RESULT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    identity = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": runner.sha256(path),
        "run_id": "artifact-unit",
    }
    assert result["artifact"]["result_bytes"] == path.stat().st_size
    assert runner.stored_result_evidence_matches(result, {"core": True}, identity)


def test_cost_gates_fail_closed_for_samples_memory_telemetry_and_bootstrap() -> None:
    correctness = _passing_correctness_payload()
    expected = _parameter_identity(correctness)
    cost, telemetry, execution, _ = _passing_cost("A", expected)

    bad_samples = copy.deepcopy(cost)
    bad_samples["measurements"]["persistent_gru"]["128"]["end_to_end"]["host_wall_samples_ms"].pop()
    gates = runner.evaluate_cost_gates(bad_samples, telemetry, execution, expected)
    assert not gates[
        "all_32_measurements_have_exact_raw_memory_state_profile_and_residency_contract"
    ]

    bad_memory = copy.deepcopy(cost)
    bad_memory["measurements"]["functional_mamba"]["256"]["temporal_only"]["memory"][
        "delta_peak_allocated_bytes"
    ] += 1
    gates = runner.evaluate_cost_gates(bad_memory, telemetry, execution, expected)
    assert not gates[
        "all_32_measurements_have_exact_raw_memory_state_profile_and_residency_contract"
    ]

    reserved_ratio = copy.deepcopy(cost)
    reserved = reserved_ratio["measurements"]["streaming_kv_transformer"]["512"]["end_to_end"][
        "memory"
    ]
    reserved["absolute_peak_reserved_bytes"] = 2_800
    reserved["delta_peak_reserved_bytes"] = 2_800 - reserved["baseline_reserved_bytes"]
    gates = runner.evaluate_cost_gates(reserved_ratio, telemetry, execution, expected)
    assert not gates[
        "transformer_over_mamba_absolute_and_delta_peak_ratios_nondecreasing_and_t925_mamba_at_most_60pct"
    ]

    reserved_threshold = copy.deepcopy(cost)
    mamba_reserved = reserved_threshold["measurements"]["functional_mamba"]["925"]["temporal_only"][
        "memory"
    ]
    transformer_reserved = reserved_threshold["measurements"]["streaming_kv_transformer"]["925"][
        "temporal_only"
    ]["memory"]["absolute_peak_reserved_bytes"]
    mamba_reserved["absolute_peak_reserved_bytes"] = int(0.61 * transformer_reserved)
    mamba_reserved["delta_peak_reserved_bytes"] = (
        mamba_reserved["absolute_peak_reserved_bytes"] - mamba_reserved["baseline_reserved_bytes"]
    )
    gates = runner.evaluate_cost_gates(reserved_threshold, telemetry, execution, expected)
    assert not gates[
        "transformer_over_mamba_absolute_and_delta_peak_ratios_nondecreasing_and_t925_mamba_at_most_60pct"
    ]

    bad_telemetry = copy.deepcopy(telemetry)
    bad_telemetry["after_warmup"][0]["telemetry"]["gpu"]["sm_clock_mhz"] = 1_999
    gates = runner.evaluate_cost_gates(cost, bad_telemetry, execution, expected)
    assert not gates["gpu_start_idle_cool_all_warm_points_clocked_and_end_has_no_other_process"]

    duplicate_telemetry = copy.deepcopy(telemetry)
    for field in ("condition", "length", "scope"):
        duplicate_telemetry["after_warmup"][0][field] = duplicate_telemetry["after_warmup"][1][
            field
        ]
    gates = runner.evaluate_cost_gates(cost, duplicate_telemetry, execution, expected)
    assert not gates["gpu_start_idle_cool_all_warm_points_clocked_and_end_has_no_other_process"]

    bad_bootstrap = copy.deepcopy(cost)
    bad_bootstrap["bootstrap"]["sample_statistic"] = "mean"
    gates = runner.evaluate_cost_gates(bad_bootstrap, telemetry, execution, expected)
    assert not gates["bootstrap_recomputed_exactly_with_point_and_ci_lower_gates"]


def test_correctness_and_cost_predecessor_raw_evidence_is_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    correctness_payload = _passing_correctness_payload()
    correctness_execution = _execution()
    gates = runner.evaluate_correctness_gates(correctness_payload, correctness_execution)
    gates.update({"wall_time_at_most_20min": True, "artifact_total_at_most_256mib": True})
    result = {
        "experiment_id": runner.EXPERIMENT,
        "run_id": runner.CORRECTNESS_RUN_ID,
        "source_commit": "1" * 40,
        "plan": {"sha256": "2" * 64},
        "status": "passed",
        "correctness": correctness_payload,
        "execution_evidence": correctness_execution,
        "gates": gates,
        "optimizer_steps": 0,
        "vla_model_instantiated": False,
        "training_allowed": False,
        "wall_seconds": 1.0,
    }
    valid = tmp_path / "correctness-valid/RESULT.json"
    valid_identity = _write_result(valid, result)
    runner.verify_passed_result(
        valid,
        valid_identity["sha256"],
        expected_path=valid,
        expected_run_id=runner.CORRECTNESS_RUN_ID,
        plan_identity={"sha256": "2" * 64},
        source_commit="1" * 40,
    )

    raw_tamper = copy.deepcopy(result)
    raw_tamper["correctness"]["functional_mamba"]["output_max_abs"] = 1.01e-4
    tampered = tmp_path / "correctness-raw-tampered/RESULT.json"
    tampered_identity = _write_result(tampered, raw_tamper)
    with pytest.raises(ValueError, match="core gates do not recompute"):
        runner.verify_passed_result(
            tampered,
            tampered_identity["sha256"],
            expected_path=tampered,
            expected_run_id=runner.CORRECTNESS_RUN_ID,
            plan_identity={"sha256": "2" * 64},
            source_commit="1" * 40,
        )

    missing_gate = copy.deepcopy(result)
    missing_gate["gates"].pop("optimizer_steps_zero")
    missing_path = tmp_path / "correctness-missing-gate/RESULT.json"
    missing_identity = _write_result(missing_path, missing_gate)
    with pytest.raises(ValueError, match="core gates do not recompute"):
        runner.verify_passed_result(
            missing_path,
            missing_identity["sha256"],
            expected_path=missing_path,
            expected_run_id=runner.CORRECTNESS_RUN_ID,
            plan_identity={"sha256": "2" * 64},
            source_commit="1" * 40,
        )

    wrong_wall = copy.deepcopy(result)
    wrong_wall["wall_seconds"] = runner.MAX_WALL_SECONDS + 1
    wall_path = tmp_path / "correctness-wrong-wall/RESULT.json"
    wall_identity = _write_result(wall_path, wrong_wall)
    with pytest.raises(ValueError, match="core gates do not recompute"):
        runner.verify_passed_result(
            wall_path,
            wall_identity["sha256"],
            expected_path=wall_path,
            expected_run_id=runner.CORRECTNESS_RUN_ID,
            plan_identity={"sha256": "2" * 64},
            source_commit="1" * 40,
        )

    expected = _parameter_identity(correctness_payload)
    cost, telemetry, execution, cost_gates = _passing_cost("A", expected)
    predecessor = {
        "cost": cost,
        "telemetry": telemetry,
        "execution_evidence": execution,
        "gates": cost_gates,
        "run_id": runner.COST_RUN_IDS["A"],
        "wall_seconds": 1.0,
    }
    predecessor["gates"].update(
        {"wall_time_at_most_20min": True, "artifact_total_at_most_256mib": True}
    )
    predecessor_path = tmp_path / "cost-valid/RESULT.json"
    predecessor_identity = _write_result(predecessor_path, predecessor)
    runner.verify_recomputed_cost_gates(predecessor, expected, predecessor_identity)

    artifact_mutations = {
        "root": str(tmp_path / "wrong-root"),
        "result_bytes": predecessor["artifact"]["result_bytes"] + 1,
        "total_bytes": predecessor["artifact"]["total_bytes"] + 1,
        "limit_bytes": runner.MAX_ARTIFACT_BYTES + 1,
        "traces_written": 1,
    }
    for field, value in artifact_mutations.items():
        changed_artifact = copy.deepcopy(predecessor)
        changed_artifact["artifact"][field] = value
        with pytest.raises(ValueError, match="core gates do not recompute"):
            runner.verify_recomputed_cost_gates(
                changed_artifact,
                expected,
                predecessor_identity,
            )
    changed_identity = copy.deepcopy(predecessor_identity)
    changed_identity["bytes"] += 1
    with pytest.raises(ValueError, match="core gates do not recompute"):
        runner.verify_recomputed_cost_gates(predecessor, expected, changed_identity)

    cost_raw_tamper = copy.deepcopy(predecessor)
    cost_raw_tamper["cost"]["measurements"]["functional_mamba"]["128"]["end_to_end"][
        "state_bytes_samples"
    ].pop()
    cost_tampered_path = tmp_path / "cost-raw-tampered/RESULT.json"
    cost_tampered_identity = _write_result(cost_tampered_path, cost_raw_tamper)
    with pytest.raises(ValueError, match="core gates do not recompute"):
        runner.verify_recomputed_cost_gates(
            cost_raw_tamper,
            expected,
            cost_tampered_identity,
        )

    cost_missing_gate = copy.deepcopy(predecessor)
    cost_missing_gate["gates"].pop("optimizer_steps_zero")
    cost_missing_path = tmp_path / "cost-missing-gate/RESULT.json"
    cost_missing_identity = _write_result(cost_missing_path, cost_missing_gate)
    with pytest.raises(ValueError, match="core gates do not recompute"):
        runner.verify_recomputed_cost_gates(
            cost_missing_gate,
            expected,
            cost_missing_identity,
        )

    cost_bad_artifact = copy.deepcopy(predecessor)
    cost_bad_artifact["artifact"]["traces_written"] = 1
    cost_bad_artifact_path = tmp_path / "cost-bad-artifact/RESULT.json"
    cost_bad_artifact_path.parent.mkdir(parents=True)
    encoded = (json.dumps(cost_bad_artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert len(encoded) == cost_bad_artifact["artifact"]["result_bytes"]
    cost_bad_artifact_path.write_bytes(encoded)
    cost_bad_artifact_identity = {
        "path": str(cost_bad_artifact_path.resolve()),
        "bytes": cost_bad_artifact_path.stat().st_size,
        "sha256": runner.sha256(cost_bad_artifact_path),
        "run_id": runner.COST_RUN_IDS["A"],
    }
    with pytest.raises(ValueError, match="core gates do not recompute"):
        runner.verify_recomputed_cost_gates(
            cost_bad_artifact,
            expected,
            cost_bad_artifact_identity,
        )


def test_aggregate_gates_fail_closed_for_pid_order_chain_and_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correctness, costs, identities = _passing_aggregate_inputs(tmp_path, monkeypatch)
    assert all(runner.evaluate_aggregate_gates(correctness, costs, identities).values())

    duplicate_pid = copy.deepcopy(costs)
    duplicate_pid[2]["process"]["pid"] = duplicate_pid[1]["process"]["pid"]
    assert not runner.evaluate_aggregate_gates(correctness, duplicate_pid, identities)[
        "four_cost_python_processes_are_distinct"
    ]

    wrong_order = [costs[1], costs[0], costs[2], costs[3]]
    assert not runner.evaluate_aggregate_gates(correctness, wrong_order, identities)[
        "exact_four_ordered_cost_results_passed_without_averaging"
    ]

    broken_chain = copy.deepcopy(costs)
    broken_chain[2]["cost_predecessor"]["sha256"] = "f" * 64
    assert not runner.evaluate_aggregate_gates(correctness, broken_chain, identities)[
        "cost_results_form_required_hash_predecessor_chain"
    ]

    changed_weights = copy.deepcopy(costs)
    changed_weights[1]["cost"]["parameter_identity"]["tokenizer_state_dict_sha256"] = "0" * 64
    assert not runner.evaluate_aggregate_gates(correctness, changed_weights, identities)[
        "all_four_weight_identities_match_correctness"
    ]

    missing_gate = copy.deepcopy(costs)
    missing_gate[0]["gates"].pop("optimizer_steps_zero")
    assert not runner.evaluate_aggregate_gates(correctness, missing_gate, identities)[
        "exact_four_ordered_cost_results_passed_without_averaging"
    ]

    wrong_wall = copy.deepcopy(costs)
    wrong_wall[0]["wall_seconds"] = runner.MAX_WALL_SECONDS + 1
    assert not runner.evaluate_aggregate_gates(correctness, wrong_wall, identities)[
        "exact_four_ordered_cost_results_passed_without_averaging"
    ]

    wrong_artifact = copy.deepcopy(costs)
    wrong_artifact[0]["artifact"]["total_bytes"] += 1
    assert not runner.evaluate_aggregate_gates(correctness, wrong_artifact, identities)[
        "exact_four_ordered_cost_results_passed_without_averaging"
    ]


def test_runner_is_vla_free_optimizer_guarded_and_write_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "from_pretrained" not in source
    assert not any(name == "gr00t.model" or name.startswith("gr00t.model.") for name in imported)

    with runner.optimizer_construction_guard() as evidence:
        with pytest.raises(RuntimeError, match="optimizer construction is forbidden"):
            torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)
    assert evidence["constructor_calls"] == ["SGD"]
    assert evidence["steps_observed"] is None

    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    path = tmp_path / "RESULT.json"
    assert runner.write_new(path, {"status": "passed"}) > 0
    with pytest.raises(FileExistsError):
        runner.write_new(path, {"status": "changed"})
