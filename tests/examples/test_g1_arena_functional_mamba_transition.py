from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_ARENA_N17"
sys.path.insert(0, str(EXAMPLE))
import functional_mamba_transition as runner  # noqa: E402


def _passing_correctness() -> tuple[dict[str, object], dict[str, object]]:
    summary = {
        f"layer_{layer}": {
            name: {"max_abs": 0.0, "first_frame_above_1e-4": None}
            for name in ("residual_output", "conv_state", "ssm_state")
        }
        for layer in range(runner.LAYERS)
    }
    gradient_pass = {
        "parameters": {"passed": True},
        "descriptor_gradient": {"finite": True, "frame1_l2_norm": 1e-4},
    }
    per_frame = [
        {
            "frame": frame,
            "layers": [
                {
                    name: {"max_abs": 0.0, "finite": True}
                    for name in ("residual_output", "conv_state", "ssm_state")
                }
                for _ in range(runner.LAYERS)
            ],
        }
        for frame in range(1, runner.INPUT_SHAPE[0] + 1)
    ]
    numerical = {
        "functional_ast": {"passed": True},
        "parameter_contract": {
            "tokenizer_exact": True,
            "temporal_exact": True,
            "temporal_parameter_objects_unique": True,
        },
        "teacher_functional_parity": {
            "frames": runner.INPUT_SHAPE[0],
            "per_frame": per_frame,
            "summary": summary,
            "all_outputs_and_states_finite": True,
            "all_input_states_unchanged": True,
            "all_new_state_storage_independent": True,
            "state_logical_growth_fraction": 0.0,
        },
        "recurrence": {
            "continuous_vs_fixed_chunks": {
                "exact": True,
                "final_state_exact": True,
                "chunks": list(runner.CORRECTNESS_CHUNKS),
            },
            "repeat": {"exact": True, "final_state_exact": True},
            "episode_reset": {"exact": True, "final_state_exact": True},
            "suffix_edit": {"prefix_exact": True},
        },
        "gradients": {
            "probe": {"sha256": runner.PROBE_SHA256},
            "continuous_vs_fixed_chunks": {
                "continuous": copy.deepcopy(gradient_pass),
                "chunked": copy.deepcopy(gradient_pass),
                "all_errors_finite": True,
                "max_abs": 0.0,
            },
            "finite_difference": {"passed": True},
            "detached_state_negative_control": {"frame1_gradient_exact_zero": True},
        },
    }
    execution = {
        "native_vla_modules": {"before": [], "after": []},
        "optimizer": {"constructor_calls": [], "steps_observed": 0},
    }
    return numerical, execution


def _passing_latency() -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, object],
    dict[str, object],
]:
    latency = {
        "input_resident_on_gpu": True,
        "state_allocated_before_timing": True,
        "pre_and_post_sample_cuda_synchronize": True,
        "warmup_frames": runner.LATENCY_WARMUP_FRAMES,
        "measurements": runner.LATENCY_MEASURED_FRAMES,
        "measured_frame_indices": list(
            range(runner.LATENCY_WARMUP_FRAMES + 1, runner.INPUT_SHAPE[0] + 1)
        ),
        "host_wall_samples_ms": [1.0] * runner.LATENCY_MEASURED_FRAMES,
        "cuda_event_samples_ms": [0.5] * runner.LATENCY_MEASURED_FRAMES,
        "kernel_count_samples": [20] * runner.LATENCY_MEASURED_FRAMES,
        "state_logical_bytes_samples": [100] * runner.LATENCY_MEASURED_FRAMES,
        "host_wall_p95_ms": 4.9,
        "output_finite": True,
        "state_logical_bytes_constant": True,
    }
    identities = {
        "tokenizer_state_dict_sha256": "1" * 64,
        "temporal_state_dict_sha256": "2" * 64,
    }
    telemetry = {"gpu0_other_compute_processes": []}
    execution = {
        "native_vla_modules": {"before": [], "after": []},
        "optimizer": {"constructor_calls": [], "steps_observed": 0},
    }
    return latency, identities, telemetry, execution


def test_frozen_run_ids_match_experiment_manifest() -> None:
    assert runner.CORRECTNESS_RUN_ID == "functional-correctness-gpu0-r1"
    assert runner.LATENCY_RUN_IDS == {
        1: "functional-latency-gpu0-r1a",
        2: "functional-latency-gpu0-r1b",
        3: "functional-latency-gpu0-r1c",
    }
    assert runner.AGGREGATE_RUN_ID == "aggregate-r1"
    plan = runner.resolved_plan("1" * 40)
    assert plan["protocol"]["correctness_chunks"] == list(runner.CORRECTNESS_CHUNKS)
    assert plan["protocol"]["gradient_chunks"] == list(runner.GRADIENT_CHUNKS)
    assert plan["protocol"]["latency"]["independent_processes"] == 3


def test_normalized_probe_has_frozen_byte_identity() -> None:
    probe, identity = runner.normalized_probe(torch.device("cpu"))
    assert identity["sha256"] == runner.PROBE_SHA256
    assert identity["sha256"] == hashlib.sha256(probe.numpy().tobytes()).hexdigest()
    assert probe.dtype == torch.float32
    assert float(torch.linalg.vector_norm(probe)) == pytest.approx(1.0)


def test_state_storage_report_rejects_input_output_and_any_internal_overlap() -> None:
    inputs = runner.FunctionalMamba2State(
        ((torch.zeros(2), torch.zeros(3)), (torch.zeros(4), torch.zeros(5))),
        1,
    )
    independent = runner.FunctionalMamba2State(
        ((torch.ones(2), torch.ones(3)), (torch.ones(4), torch.ones(5))),
        1,
    )
    assert runner.state_storage_report(inputs, independent)["passed"]

    input_alias = runner.FunctionalMamba2State(
        ((inputs.layers[0][0], torch.ones(3)), (torch.ones(4), torch.ones(5))),
        1,
    )
    assert not runner.state_storage_report(inputs, input_alias)["passed"]

    shared = torch.zeros(8)
    same_layer_alias = runner.FunctionalMamba2State(
        ((shared[:4], shared[4:]), (torch.ones(4), torch.ones(5))),
        1,
    )
    report = runner.state_storage_report(inputs, same_layer_alias)
    assert not report["passed"]
    assert report["output_internal_overlap"][0]["same_layer"]


def test_correctness_gate_evaluation_is_fail_closed() -> None:
    numerical, execution = _passing_correctness()
    assert all(runner.evaluate_correctness_gates(numerical, execution).values())

    changed = copy.deepcopy(numerical)
    changed["teacher_functional_parity"]["summary"]["layer_1"]["ssm_state"]["max_abs"] = 1.00001e-4
    assert not runner.evaluate_correctness_gates(changed, execution)[
        "teacher_functional_all_925_frames_two_layers_at_most_1e-4"
    ]

    changed = copy.deepcopy(numerical)
    changed["gradients"]["finite_difference"]["passed"] = False
    assert not runner.evaluate_correctness_gates(changed, execution)[
        "finite_difference_same_sign_and_within_tolerance"
    ]

    changed = copy.deepcopy(numerical)
    changed["teacher_functional_parity"]["all_input_states_unchanged"] = False
    assert not runner.evaluate_correctness_gates(changed, execution)[
        "functional_input_state_unchanged_every_step"
    ]

    changed = copy.deepcopy(numerical)
    changed["teacher_functional_parity"]["per_frame"].pop()
    assert not runner.evaluate_correctness_gates(changed, execution)[
        "teacher_functional_parity_has_all_925_frames_and_recomputed_summary"
    ]

    changed = copy.deepcopy(numerical)
    changed["teacher_functional_parity"]["frames"] = runner.INPUT_SHAPE[0] - 1
    assert not runner.evaluate_correctness_gates(changed, execution)[
        "teacher_functional_parity_has_all_925_frames_and_recomputed_summary"
    ]

    changed = copy.deepcopy(numerical)
    comparison = changed["gradients"]["continuous_vs_fixed_chunks"]
    comparison["chunked"]["descriptor_gradient"]["finite"] = False
    comparison["all_errors_finite"] = False
    comparison["max_abs"] = float("inf")
    assert not runner.evaluate_correctness_gates(changed, execution)[
        "continuous_chunked_loss_and_all_gradients_at_most_1e-5"
    ]


def test_tensor_error_rejects_nonfinite_gradient() -> None:
    report = runner._tensor_error(torch.tensor([float("nan")]), torch.zeros(1))
    assert not report["finite"]
    assert report["max_abs"] == float("inf")


def test_latency_gate_requires_strict_p95_and_all_per_sample_metrics() -> None:
    latency, identities, telemetry, execution = _passing_latency()
    gates = runner.evaluate_latency_gates(
        latency,
        identities,
        identities,
        telemetry,
        telemetry,
        execution,
    )
    assert all(gates.values())

    changed = copy.deepcopy(latency)
    changed["host_wall_p95_ms"] = 5.0
    gates = runner.evaluate_latency_gates(
        changed,
        identities,
        identities,
        telemetry,
        telemetry,
        execution,
    )
    assert not gates["host_wall_p95_strictly_below_5ms"]

    changed = copy.deepcopy(latency)
    changed["kernel_count_samples"] = changed["kernel_count_samples"][:-1]
    gates = runner.evaluate_latency_gates(
        changed,
        identities,
        identities,
        telemetry,
        telemetry,
        execution,
    )
    assert not gates["every_sample_synchronized_and_has_host_cuda_kernel_state_metrics"]


def test_aggregate_requires_three_distinct_chained_latency_processes() -> None:
    correctness_sha = "a" * 64
    correctness = {
        "status": "passed",
        "gates": {"correctness": True},
        "result_identity": {"sha256": correctness_sha},
        "vla_model_instantiated": False,
        "optimizer_steps": 0,
        "wall_seconds": 10.0,
    }
    identities = [
        {"sha256": digest, "run_id": runner.LATENCY_RUN_IDS[index]}
        for index, digest in zip((1, 2, 3), ("b" * 64, "c" * 64, "d" * 64), strict=True)
    ]
    weight_identity = {
        "tokenizer_state_dict_sha256": "1" * 64,
        "temporal_state_dict_sha256": "2" * 64,
    }
    latencies = []
    for index in (1, 2, 3):
        latencies.append(
            {
                "run_id": runner.LATENCY_RUN_IDS[index],
                "replicate": index,
                "status": "passed",
                "gates": {"latency": True},
                "process": {"pid": 100 + index},
                "correctness_predecessor": {"sha256": correctness_sha},
                "latency_predecessor": (
                    None if index == 1 else {"sha256": identities[index - 2]["sha256"]}
                ),
                "parameter_identity": copy.deepcopy(weight_identity),
                "latency": {
                    "host_wall_p95_ms": 4.9,
                    "warmup_frames": runner.LATENCY_WARMUP_FRAMES,
                    "measurements": runner.LATENCY_MEASURED_FRAMES,
                    "host_wall_samples_ms": [1.0] * runner.LATENCY_MEASURED_FRAMES,
                    "cuda_event_samples_ms": [0.5] * runner.LATENCY_MEASURED_FRAMES,
                    "kernel_count_samples": [20] * runner.LATENCY_MEASURED_FRAMES,
                    "state_logical_bytes_samples": [100] * runner.LATENCY_MEASURED_FRAMES,
                },
                "telemetry": {
                    "before": {"gpu0_other_compute_processes": []},
                    "after": {"gpu0_other_compute_processes": []},
                },
                "vla_model_instantiated": False,
                "optimizer_steps": 0,
                "wall_seconds": 20.0,
            }
        )
    assert all(runner.evaluate_aggregate_gates(correctness, latencies, identities).values())

    duplicate_pid = copy.deepcopy(latencies)
    duplicate_pid[2]["process"]["pid"] = duplicate_pid[1]["process"]["pid"]
    assert not runner.evaluate_aggregate_gates(correctness, duplicate_pid, identities)[
        "three_latency_python_processes_are_distinct"
    ]

    boundary = copy.deepcopy(latencies)
    boundary[0]["latency"]["host_wall_p95_ms"] = 5.0
    assert not runner.evaluate_aggregate_gates(correctness, boundary, identities)[
        "all_three_host_wall_p95_strictly_below_5ms"
    ]

    broken_chain = copy.deepcopy(latencies)
    broken_chain[2]["latency_predecessor"]["sha256"] = "e" * 64
    assert not runner.evaluate_aggregate_gates(correctness, broken_chain, identities)[
        "latency_results_form_required_predecessor_chain"
    ]

    changed_weights = copy.deepcopy(latencies)
    changed_weights[1]["parameter_identity"]["temporal_state_dict_sha256"] = "9" * 64
    assert not runner.evaluate_aggregate_gates(correctness, changed_weights, identities)[
        "all_three_weights_match"
    ]


def test_codec_loader_rejects_wrong_stored_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "codec.npz"
    shape = (2, 1, runner.GRID_HEIGHT, runner.GRID_WIDTH, runner.CODEC_FIELDS)
    values = {
        name: np.zeros(shape, dtype=np.float64 if name == runner.INPUT_FIELD else np.float32)
        for name in runner.EXPECTED_ARCHIVE_FIELDS
    }
    np.savez(path, **values)
    monkeypatch.setattr(runner, "INPUT_PATH", path)
    monkeypatch.setattr(runner, "INPUT_BYTES", path.stat().st_size)
    monkeypatch.setattr(runner, "INPUT_SHA256", runner.sha256(path))
    monkeypatch.setattr(runner, "INPUT_SHAPE", shape)
    with pytest.raises(ValueError, match="dtype changed"):
        runner.load_codec_input()


def test_runner_is_vla_free_and_exposes_separate_latency_modes() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "from_pretrained" not in source
    assert "optimizer_construction_guard" in source
    assert not any(name == "gr00t.model" or name.startswith("gr00t.model.") for name in imported)
    assert all(flag in source for flag in ("--latency-a", "--latency-b", "--latency-c"))
    assert runner.functional_ast_report()["passed"]
    assert len(runner.EXPECTED_TOKENIZER_PARAMETERS) == 6
    assert len(runner.EXPECTED_TEMPORAL_PARAMETERS) == 20


def test_write_new_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    path = tmp_path / "RESULT.json"
    assert runner.write_new(path, {"status": "passed"}) > 0
    with pytest.raises(FileExistsError):
        runner.write_new(path, {"status": "changed"})


def test_verify_plan_rejects_protocol_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    commit = "1" * 40
    plan = runner.resolved_plan(commit)
    path = tmp_path / "plan-r1/PLAN.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded, _ = runner.verify_plan(path, runner.sha256(path))
    assert loaded == plan
    plan["protocol"]["gradient_chunks"] = [128]
    changed = tmp_path / "plan-r1/CHANGED.json"
    changed.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contract changed"):
        runner.verify_plan(changed, runner.sha256(changed))


def test_correctness_predecessor_recomputes_full_parity_cardinality(tmp_path: Path) -> None:
    numerical, execution = _passing_correctness()
    gates = runner.evaluate_correctness_gates(numerical, execution)
    gates.update({"wall_time_at_most_10min": True, "artifact_total_at_most_256mib": True})
    result = {
        "experiment_id": runner.EXPERIMENT,
        "run_id": runner.CORRECTNESS_RUN_ID,
        "source_commit": "1" * 40,
        "plan": {"sha256": "2" * 64},
        "status": "passed",
        "gates": gates,
        "optimizer_steps": 0,
        "vla_model_instantiated": False,
        "numerical": numerical,
        "execution_evidence": execution,
    }
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    runner.verify_passed_result(
        valid,
        runner.sha256(valid),
        expected_path=valid,
        expected_run_id=runner.CORRECTNESS_RUN_ID,
        plan_identity={"sha256": "2" * 64},
        source_commit="1" * 40,
    )

    result["numerical"]["teacher_functional_parity"]["per_frame"].pop()
    truncated = tmp_path / "truncated.json"
    truncated.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="do not recompute"):
        runner.verify_passed_result(
            truncated,
            runner.sha256(truncated),
            expected_path=truncated,
            expected_run_id=runner.CORRECTNESS_RUN_ID,
            plan_identity={"sha256": "2" * 64},
            source_commit="1" * 40,
        )
