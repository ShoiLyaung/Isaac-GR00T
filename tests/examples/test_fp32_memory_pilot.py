"""CPU-only pilot lifecycle, causal controls, graph and acceptance contracts."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import fp32_memory_pilot as runner  # noqa: E402
import fp32_pilot_support as support  # noqa: E402
from gr00t.model.modules.causal_visual_gru import CausalVisualGRU, GRUState  # noqa: E402
from gr00t.model.modules.fp32_visual_memory import (  # noqa: E402
    FP32VisualMemory,
    explicit_cache_values,
)
from memory_training_support import selected_prefix  # noqa: E402


class TinyGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(4, 4)
        self.output_projection = nn.Linear(4, 6)

    def full_sequence(self, value):
        return self.input_projection(value).cumsum(0)

    def new_state(self):
        return GRUState([torch.zeros(2, 4)])

    @torch.no_grad()
    def step(self, value, state):
        state.hidden[0].add_(self.input_projection(value))
        state.observations += 1
        result = state.hidden[0].clone()
        return result, self.output_projection(result)


def test_fp32_executor_supports_only_explicit_mamba_or_gru_layout(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    module = TinyGRU()
    execution = FP32VisualMemory(module)
    state = module.new_state()
    latent, residual = execution.step(torch.ones(2, 4), state)
    assert state.observations == 1 and residual.shape == (2, 6)
    assert explicit_cache_values(state) == state.hidden
    assert latent.dtype == torch.float32
    with pytest.raises(ValueError, match="explicit"):
        explicit_cache_values(SimpleNamespace(hidden=[], layers=[]))
    state.hidden[0] = state.hidden[0].bfloat16()
    with pytest.raises(ValueError, match="caches"):
        execution.step(torch.ones(2, 4), state)


def test_causal_controls_keep_current_and_clock_without_mutating_input():
    sequence = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    saved = sequence.clone()
    assert torch.equal(selected_prefix(sequence, "current_reset_mamba"), sequence[-1:])
    zero = selected_prefix(sequence, "zero_history_mamba")
    assert len(zero) == 3 and not torch.count_nonzero(zero[:-1])
    assert torch.equal(zero[-1], sequence[-1]) and torch.equal(sequence, saved)
    for name in ("mamba_history", "gru_history"):
        assert selected_prefix(sequence, name) is sequence
    with pytest.raises(ValueError):
        selected_prefix(sequence, "unknown")


def test_full_step_control_cadence_and_local_fp32_gradient(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    module = TinyGRU()
    sequence = torch.ones(3, 2, 4)
    for condition in support.CONDITIONS:
        full, residual, length, _ = support.memory_outputs(module, sequence, condition)
        step, actual, actual_length, state = support.memory_outputs(
            module, sequence, condition, True
        )
        assert torch.allclose(full, step, atol=1e-6)
        assert torch.allclose(residual, actual, atol=1e-6)
        assert (
            length
            == actual_length
            == state.observations
            == (1 if condition == "current_reset_mamba" else 3)
        )
    residual.square().sum().backward()
    assert module.input_projection.weight.grad is not None
    assert torch.isfinite(module.input_projection.weight.grad).all()


def test_actual_gru_math_full_step_fp32_and_common_projections(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        common = {
            "input_projection.weight": torch.randn(256, 2048) * 0.01,
            "input_projection.bias": torch.zeros(256),
            "output_projection.weight": torch.randn(2048, 256) * 0.001,
            "output_projection.bias": torch.zeros(2048),
        }
        for i in range(2):
            common[f"layers.{i}.norm.weight"] = torch.ones(256)
            common[f"layers.{i}.norm.bias"] = torch.zeros(256)
        module = CausalVisualGRU(common)
        sequence = torch.randn(3, 192, 2048)
    for name, value in module.state_dict().items():
        if ".cell." not in name:
            assert torch.equal(value, common[name])
    full, residual, _, _ = support.memory_outputs(module, sequence, "gru_history")
    step, actual, _, state = support.memory_outputs(module, sequence, "gru_history", True)
    assert torch.allclose(full, step, atol=1e-6)
    assert torch.allclose(residual, actual, atol=1e-6)
    residual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    clone = state.clone()
    clone.hidden[0].zero_()
    assert torch.count_nonzero(state.hidden[0])


def paired_rows():
    schedule = [
        {
            "step": i + 1,
            "episode": 125,
            "anchor": 16,
            "prefix": [0, 16],
            "forward_seed": 10 + i,
            "lr": 1e-4,
        }
        for i in range(500)
    ]
    rows = {
        condition: [
            {
                **row,
                "condition": condition,
                "effective_length": 1 if condition == "current_reset_mamba" else 2,
                "optimizer_steps_completed": row["step"],
                "noise_receipt": {key: row["forward_seed"] for key in support.NOISE_KEYS},
            }
            for row in schedule
        ]
        for condition in support.CONDITIONS
    }
    return schedule, rows


def test_exact_pairing_catches_actual_time_even_with_equal_seed():
    schedule, rows = paired_rows()
    assert support.compare_receipts(rows, schedule)["passed"]
    rows["gru_history"][8]["noise_receipt"]["continuous_time_values"] = 999
    with pytest.raises(ValueError, match="noise/time"):
        support.compare_receipts(rows, schedule)


def test_pairing_catches_missing_steps_wrong_cadence_and_lr():
    schedule, rows = paired_rows()
    rows["zero_history_mamba"][1]["effective_length"] = 1
    with pytest.raises(ValueError, match="cadence"):
        support.compare_receipts(rows, schedule)
    schedule, rows = paired_rows()
    rows["gru_history"][0]["lr"] = 3e-4
    with pytest.raises(ValueError, match="pairing"):
        support.compare_receipts(rows, schedule)
    schedule, rows = paired_rows()
    rows["mamba_history"].pop()
    with pytest.raises(ValueError, match="receipt"):
        support.compare_receipts(rows, schedule)


def evaluation(value):
    return {
        "metrics": dict.fromkeys(support.METRICS, value),
        "per_episode": {str(i): dict.fromkeys(support.METRICS, value) for i in range(4)},
    }


def test_quality_requires_both_metrics_three_controls_and_direction():
    values = {name: evaluation(1.0) for name in ("native", *support.CONDITIONS)}
    values["mamba_history"] = evaluation(0.95)
    assert support.quality_gates(values, list(range(4)))["passed"]
    values["gru_history"]["metrics"]["combined_mae_mixed_units"] = 0.96
    assert not support.quality_gates(values, list(range(4)))["passed"]
    values["gru_history"] = evaluation(1.0)
    for i in (0, 1):
        values["mamba_history"]["per_episode"][str(i)]["flow_loss"] = 1.1
    assert not support.quality_gates(values, list(range(4)))["passed"]


def test_native_retention_checks_arm_and_hand_separately():
    values = {name: evaluation(1.0) for name in ("native", *support.CONDITIONS)}
    values["mamba_history"] = evaluation(0.95)
    values["mamba_history"]["metrics"]["hand_mae_u"] = 1.021
    assert not support.quality_gates(values, list(range(4)))["passed"]


def test_confirmation_prerequisite_fails_before_model_or_cache(tmp_path):
    args = SimpleNamespace(
        artifact_root=tmp_path, command="cache-confirmation", condition=None, cohort=None
    )
    with pytest.raises(FileNotFoundError):
        runner.prerequisites(args)
    for name in ("preflight-r1", *[f"train-{c}-r1" for c in support.CONDITIONS], "dev-joint-r1"):
        path = tmp_path / name / "output"
        path.mkdir(parents=True)
        (path / "RESULT.json").write_text(
            json.dumps(
                {"status": "passed", "continuation_gate": True, "plan_sha256": runner.PLAN_SHA}
            )
        )
    assert len(runner.prerequisites(args)) == 6
    path = tmp_path / "dev-joint-r1/output/RESULT.json"
    value = json.loads(path.read_text())
    value["continuation_gate"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prerequisite"):
        runner.prerequisites(args)


def test_training_prerequisites_enforce_condition_order(tmp_path):
    args = SimpleNamespace(
        artifact_root=tmp_path, command="train", condition="gru_history", cohort=None
    )
    path = tmp_path / "preflight-r1/output"
    path.mkdir(parents=True)
    (path / "RESULT.json").write_text(
        json.dumps({"status": "passed", "continuation_gate": True, "plan_sha256": runner.PLAN_SHA})
    )
    with pytest.raises(FileNotFoundError):
        runner.prerequisites(args)


def test_gru_parameter_witness_uses_real_cell_names():
    common = {
        "input_projection.weight": torch.zeros(256, 2048),
        "input_projection.bias": torch.zeros(256),
        "output_projection.weight": torch.zeros(2048, 256),
        "output_projection.bias": torch.zeros(2048),
    }
    for i in range(2):
        common[f"layers.{i}.norm.weight"] = torch.ones(256)
        common[f"layers.{i}.norm.bias"] = torch.zeros(256)
    module = CausalVisualGRU(common)
    before = {k: v.clone() for k, v in module.state_dict().items()}
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.001)
    seen = dict.fromkeys(
        ("input_projection", "output_projection", "norm0", "norm1", "recurrence0", "recurrence1"),
        True,
    )
    report = support.parameter_witness(module, before, seen)
    assert report["passed"]
    assert all(".cell." in name for name in report["groups"]["recurrence0"]["changed_parameters"])


def test_canonical_output_forbids_renamed_second_attempt(tmp_path):
    args = SimpleNamespace(
        artifact_root=tmp_path,
        command="train",
        condition="mamba_history",
        cohort=None,
        output=tmp_path / "retry/output",
    )
    assert runner.expected_output(args) == tmp_path / "train-mamba_history-r1/output"
    with pytest.raises(ValueError, match="canonical"):
        runner.execute(args)
    assert not args.output.exists()


def test_cost_lifecycle_moves_unused_branch_and_excludes_seed_restore(monkeypatch, tmp_path):
    import fp32_pilot_evaluation as evaluation
    from gr00t.model.modules.causal_visual_memory import MemoryState

    active = {"gpu": False, "timer": False}
    module = TinyGRU()

    def cpu():
        assert not active["timer"]
        active["gpu"] = False
        return module

    def cuda():
        assert not active["timer"]
        active["gpu"] = True
        return module

    monkeypatch.setattr(module, "cpu", cpu)
    monkeypatch.setattr(module, "cuda", cuda)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda: 10000 + (100 if active["gpu"] else 0)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 20000)
    ticks = [0]

    def clock():
        active["timer"] = not active["timer"]
        ticks[0] += 1
        return ticks[0] * 0.001

    monkeypatch.setattr(evaluation.time, "perf_counter", clock)

    def seed(*a, **k):
        assert not active["timer"]

    monkeypatch.setattr(evaluation, "set_seed", seed)
    # Simulate state transfer while retaining the real isolated clone contract.
    monkeypatch.setattr(MemoryState, "to", lambda state, device: state.clone())
    prefix = MemoryState([(torch.zeros(2, 4), torch.zeros(2, 4))], 1)

    def prefill(*a, **k):
        assert active["gpu"] and not active["timer"]
        return prefix.clone(), {"observations": 1}

    monkeypatch.setattr(evaluation, "prefill_snapshot", prefill)
    monkeypatch.setattr(evaluation, "call_audit", lambda *a, **k: {"instrumentation_removed": True})
    calls = {"native": 0, "fp32_step": 0}

    def policy(model, memory, executor, state, observation, mode):
        assert active["timer"]
        assert active["gpu"] == (mode == "fp32_step")
        if state is not None:
            assert state is not prefix
            state.observations += 1
        calls[mode] += 1
        return torch.ones(1, 40, 13)

    monkeypatch.setattr(evaluation, "policy_action", policy)
    monkeypatch.setattr(evaluation.proto, "validate_input_layout", lambda *a, **k: None)
    features = torch.zeros(1, 209, 2048)
    monkeypatch.setattr(
        evaluation.proto,
        "encode_observation",
        lambda *a, **k: {"features": SimpleNamespace(backbone_features=features)},
    )
    monkeypatch.setattr(
        evaluation.proto, "action_record", lambda value, *a, **k: (value.cpu(), {"finite": True})
    )
    batch = {
        key: torch.zeros(1)
        for key in (
            "state",
            "action",
            "action_mask",
            "embodiment_id",
            "input_ids",
            "attention_mask",
            "image_grid_thw",
        )
    }
    frame = {
        "post_vl_features": features,
        "input_identity": {key: evaluation.tensor_identity(value) for key, value in batch.items()},
    }
    context = SimpleNamespace(
        args=SimpleNamespace(output=tmp_path),
        plan={"grids": {"150": [0, 16], "147": [0, 16]}},
        source=SimpleNamespace(
            batch=lambda pairs: (batch, [None]),
            modality={"video": SimpleNamespace(modality_keys=[])},
            processor=None,
        ),
        model=None,
        budget=lambda: None,
    )
    cache = SimpleNamespace(
        frame=lambda ep, anchor: frame, episode=lambda ep: {0: frame, 16: frame}
    )
    reports = evaluation.cost_probes(context, cache, module, [150, 147])
    assert calls == {"native": 110, "fp32_step": 110}
    assert reports[0]["order"] == ["native", "fp32_step"]
    assert reports[1]["order"] == ["fp32_step", "native"]
    assert all(r["passed"] for r in reports)
    assert not active["gpu"] and not active["timer"]
    assert prefix.observations == 1
