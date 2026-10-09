from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_ARENA_N17"
sys.path.insert(0, str(EXAMPLE))
import codec_streaming_parity_diagnosis as diagnosis  # noqa: E402


def _condition(max_abs: float) -> dict[str, object]:
    return {
        "shape": [1, diagnosis.INPUT_SHAPE[0], diagnosis.WIDTH],
        "finite": True,
        "prefix": {"925": {"max_abs": max_abs}},
        "repeat_determinism": {"full_exact": True, "stream_exact": True},
        "continuous_vs_chunk127": {
            "exact": True,
            "partition_size": diagnosis.STREAM_PARTITION_SIZE,
        },
        "reset": {"exact": True},
        "causality": {"suffix_edit_prefix_max_abs": 0.0},
    }


def _passing_inputs() -> tuple[dict[str, object], dict[str, object]]:
    numerical = {
        "conditions": {
            "mamba_chunk16_current": _condition(diagnosis.R1_MAMBA_T925_MAX_ABS),
            "mamba_chunk1_candidate": _condition(0.0),
            "gru_batched_current": _condition(diagnosis.R1_GRU_T925_MAX_ABS),
            "gru_explicit_candidate": _condition(0.0),
        },
        "weight_identity": {
            name: {
                "keys_equal": True,
                "all_tensors_equal": True,
                "hashes_equal": True,
                "mismatched_tensors": [],
            }
            for name in ("mamba_current_vs_candidate", "gru_current_vs_candidate")
        },
        "candidate_gradients": {
            name: {
                "length": diagnosis.GRADIENT_LENGTH,
                "output_finite": True,
                "tokenizer": {"finite_nonzero": True},
                "temporal": {"finite_nonzero": True},
            }
            for name in diagnosis.CANDIDATE_CONDITIONS
        },
    }
    execution = {
        "native_vla_modules": {"before": [], "after": []},
        "optimizer": {"constructor_calls": [], "steps_observed": 0},
    }
    return numerical, execution


def test_gate_evaluation_enforces_reproduction_candidate_and_gradient_contracts() -> None:
    numerical, execution = _passing_inputs()
    assert all(diagnosis.evaluate_gates(numerical, execution).values())

    changed = copy.deepcopy(numerical)
    changed["conditions"]["mamba_chunk16_current"]["prefix"]["925"]["max_abs"] += 1.1e-6
    gates = diagnosis.evaluate_gates(changed, execution)
    assert not gates["r1_current_t925_reproduced_within_1e-6"]

    changed = copy.deepcopy(numerical)
    changed["conditions"]["gru_explicit_candidate"]["prefix"]["925"]["max_abs"] = 1.00001e-4
    gates = diagnosis.evaluate_gates(changed, execution)
    assert not gates["candidate_t925_full_step_at_most_1e-4"]

    changed = copy.deepcopy(numerical)
    changed["candidate_gradients"]["mamba_chunk1_candidate"]["temporal"]["finite_nonzero"] = False
    gates = diagnosis.evaluate_gates(changed, execution)
    assert not gates["candidate_t128_tokenizer_temporal_gradients_finite_nonzero"]


def test_gate_evaluation_rejects_extra_condition() -> None:
    numerical, execution = _passing_inputs()
    numerical["conditions"]["extra"] = _condition(0.0)
    gates = diagnosis.evaluate_gates(numerical, execution)
    assert not gates["exactly_four_frozen_conditions"]


def test_condition_metrics_reports_all_prefixes_frames_and_first_crossing() -> None:
    full = torch.ones(1, 925, diagnosis.WIDTH)
    streamed = full.clone()
    streamed[:, 11, 0] += 2e-4
    report = diagnosis.condition_metrics(full, streamed)
    assert tuple(map(int, report["prefix"])) == diagnosis.PREFIX_LENGTHS
    assert len(report["per_frame"]) == 925
    assert report["first_frame_max_abs_above_1e-4"] == 12
    assert report["prefix"]["12"]["max_abs"] > diagnosis.MAX_PARITY_ABS
    assert set(report["prefix"]["925"]["final_token"]) == {"max_abs", "rms", "cosine"}
    assert set(report["output_scale"]) == {"full", "stream"}


def test_state_dict_hash_and_tensor_equality_are_ordered_and_sensitive() -> None:
    torch.manual_seed(7)
    current = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    candidate = copy.deepcopy(current)
    equal = diagnosis.state_dict_pair_report(current, candidate)
    assert equal["all_tensors_equal"]
    assert equal["hashes_equal"]
    with torch.no_grad():
        candidate[0].weight[0, 0] += 1
    changed = diagnosis.state_dict_pair_report(current, candidate)
    assert not changed["all_tensors_equal"]
    assert not changed["hashes_equal"]
    assert changed["mismatched_tensors"] == ["0.weight"]


def test_resolved_plan_freezes_only_the_four_conditions_and_gpu0() -> None:
    plan = diagnosis.resolved_plan("1" * 40)
    assert plan["source"]["base_commit"] == diagnosis.BASE_SOURCE_COMMIT
    assert plan["protocol"]["conditions"] == list(diagnosis.CONDITION_NAMES)
    assert plan["protocol"]["mamba_full_chunk_sizes"] == {"current": 16, "candidate": 1}
    assert plan["resources"] == {
        "gpu_physical_id": 0,
        "timeout_seconds": 600,
        "storage_limit_bytes": 268_435_456,
    }
    assert plan["thresholds"]["full_step_max_abs"] == 1e-4


def test_verify_plan_rejects_protocol_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnosis, "ARTIFACT_ROOT", tmp_path)
    commit = "1" * 40
    plan = diagnosis.resolved_plan(commit)
    path = tmp_path / "plan-r1/PLAN.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded, _ = diagnosis.verify_plan(path, diagnosis.sha256(path))
    assert loaded == plan

    plan["protocol"]["conditions"].append("extra")
    changed = tmp_path / "plan-r1/CHANGED.json"
    changed.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contract changed"):
        diagnosis.verify_plan(changed, diagnosis.sha256(changed))


def test_runner_has_no_native_vla_import_or_optimizer_execution_surface() -> None:
    source = Path(diagnosis.__file__).read_text(encoding="utf-8")
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
    assert diagnosis.MAX_WALL_SECONDS == 600
    assert diagnosis.MAX_ARTIFACT_BYTES == 256 << 20


def test_write_new_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnosis, "ARTIFACT_ROOT", tmp_path)
    path = tmp_path / "RESULT.json"
    assert diagnosis.write_new(path, {"status": "passed"}) > 0
    with pytest.raises(FileExistsError):
        diagnosis.write_new(path, {"status": "changed"})
