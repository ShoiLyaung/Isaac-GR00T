"""CPU interface/graph checks only; no official CUDA kernels or optimizer execution."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import fp32_memory_overfit as runner  # noqa: E402


class TinyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(4, 4)
        self.output_projection = nn.Linear(4, 6)
        self.states = []

    def full_sequence(self, value):
        return self.input_projection(value).cumsum(0)

    def new_state(self):
        state = SimpleNamespace(layers=[(torch.zeros(2, 4), torch.zeros(2, 4))], observations=0)
        self.states.append(state)
        return state

    @torch.no_grad()
    def step(self, value, state):
        hidden = self.input_projection(value)
        state.layers[0][0].add_(hidden)
        state.observations += 1
        return state.layers[0][0].clone(), self.output_projection(state.layers[0][0])


@pytest.fixture
def precision(monkeypatch):
    active = [False]
    original = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    @contextmanager
    def autocast(device, *, dtype=None, enabled=True):
        assert device == "cuda"
        active.append(enabled)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: active[-1])
    try:
        yield active
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


def test_real_executor_graph_and_backward_outside_autocast(monkeypatch, precision):
    memory = TinyMemory()
    seen = []

    def flow(model, frame, residual, seed):
        assert precision[-1] and torch.backends.cuda.matmul.allow_tf32
        residual.register_hook(lambda grad: seen.append(runner.precision_flags()))
        return {"loss": residual.square().mean()}, {
            "forward_seed": seed,
            "actual_noise": "captured",
        }

    monkeypatch.setattr(runner, "native_flow", flow)
    report = runner.objective_backward(
        None, runner.FP32VisualMemory(memory), torch.ones(3, 2, 4), {}, 123, 0.01
    )
    assert len(seen) == 1 and not seen[0]["cuda_autocast_enabled"]
    assert seen[0]["matmul_allow_tf32"] and seen[0]["cudnn_allow_tf32"]
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0)
        for p in memory.parameters()
    )
    assert [row["label"] for row in report["precision"]["forward_memory"]] == [
        "input_projection",
        "readout",
    ]
    assert all(
        not row["cuda_autocast_enabled"]
        and not row["matmul_allow_tf32"]
        and not row["cudnn_allow_tf32"]
        for row in report["precision"]["forward_memory"]
    )
    assert report["precision"]["base_forward"]["cuda_autocast_enabled"]
    assert not report["precision"]["backward_ambient"]["cuda_autocast_enabled"]
    assert report["objective"] == pytest.approx(report["native_loss"] + 0.01 * report["feature_l2"])
    assert (
        not memory.input_projection._forward_pre_hooks
        and not memory.output_projection._forward_pre_hooks
    )


def test_native_failure_restores_precision_and_removes_hooks(monkeypatch, precision):
    memory = TinyMemory()

    def fail(*args):
        raise RuntimeError("native forward failed")

    monkeypatch.setattr(runner, "native_flow", fail)
    with pytest.raises(RuntimeError, match="native forward failed"):
        runner.objective_backward(
            None, runner.FP32VisualMemory(memory), torch.ones(3, 2, 4), {}, 42, 0.01
        )
    assert precision == [False]
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    assert (
        not memory.input_projection._forward_pre_hooks
        and not memory.output_projection._forward_pre_hooks
    )


def test_checked_fresh_initialization_rejects_resume_and_nonzero_readout(tmp_path, monkeypatch):
    values = {
        "input_projection.weight": torch.ones(2, 3),
        "output_projection.weight": torch.zeros(3, 2),
        "output_projection.bias": torch.zeros(3),
    }
    path = tmp_path / "initial.pt"
    torch.save(values, path)
    monkeypatch.setattr(runner, "INITIAL_SHA", runner.proto.content_hash(values))
    assert torch.equal(
        runner.checked_initialization(path)["input_projection.weight"],
        values["input_projection.weight"],
    )
    torch.save({"module": values, "optimizer": {}}, path)
    with pytest.raises(ValueError, match="resume forbidden"):
        runner.checked_initialization(path)
    values["output_projection.weight"][0, 0] = 1
    torch.save(values, path)
    with pytest.raises(ValueError, match="frozen fresh"):
        runner.checked_initialization(path)
    monkeypatch.setattr(runner, "INITIAL_SHA", runner.proto.content_hash(values))
    with pytest.raises(ValueError, match="exactly zero"):
        runner.checked_initialization(path)


def test_causal_prefix_is_complete_current_and_independent(monkeypatch, precision):
    monkeypatch.setattr(runner.proto, "IMAGE_POSITIONS", [1, 3])
    cache = {
        anchor: {"post_vl_features": torch.full((1, 5, 4), float(anchor + 1))}
        for anchor in (0, 16, 32)
    }
    frozen = {a: v["post_vl_features"].clone() for a, v in cache.items()}
    sequence, indices = runner.sequence_for(cache, 32, "cpu")
    assert indices == [0, 16, 32] and tuple(sequence.shape) == (3, 2, 4)
    assert sequence[:, 0, 0].tolist() == [1, 17, 33]
    memory = TinyMemory()
    executor = runner.FP32VisualMemory(memory)
    first, _, state = runner.step_prefix(memory, executor, sequence)
    repeated, _, again = runner.step_prefix(memory, executor, sequence)
    assert state is not again and state.observations == again.observations == 3
    assert torch.equal(first, repeated) and not first.requires_grad
    sequence.zero_()
    assert all(torch.equal(cache[a]["post_vl_features"], frozen[a]) for a in cache)
    with pytest.raises(ValueError, match="Missing causal"):
        runner.sequence_for({0: cache[0], 32: cache[32]}, 32, "cpu")


def test_action_rng_is_reset_after_packet_preparation(monkeypatch):
    events = []
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(runner, "sampling_packet", lambda *args: events.append("packet") or {})
    monkeypatch.setattr(runner, "set_seed", lambda seed, cuda: events.append(("seed", seed, cuda)))
    monkeypatch.setattr(
        runner.proto,
        "sample_packet",
        lambda *args: events.append("action") or torch.zeros(1, 40, 13),
    )
    assert runner.sample_cached(None, {}, None, 812).shape == (1, 40, 13)
    assert events == ["packet", ("seed", 812, True), "action"]


def test_false_numerical_gate_preserves_both_anchors_and_disabled_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "budget", lambda *args: None)
    monkeypatch.setattr(
        runner, "sequence_for", lambda cache, anchor, device: (torch.ones(2, 2, 4), [0, anchor])
    )
    monkeypatch.setattr(
        runner,
        "FP32VisualMemory",
        lambda memory: SimpleNamespace(full=lambda value: (value, torch.zeros(2, 4))),
    )
    state = SimpleNamespace(describe=lambda: {"observations": 2})
    monkeypatch.setattr(
        runner, "step_prefix", lambda *args: (torch.ones(2, 2, 4) + 0.25, torch.zeros(2, 4), state)
    )
    monkeypatch.setattr(runner.proto, "state_hash", lambda value: "statehash")
    sample_calls, disabled = [], []
    monkeypatch.setattr(
        runner,
        "sample_cached",
        lambda *args: sample_calls.append(args[-1]) or torch.zeros(1, 40, 13),
    )
    monkeypatch.setattr(runner, "raw_record", lambda value: value)
    monkeypatch.setattr(runner.proto, "action_record", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "base_parity",
        lambda model, frame, seeds: (disabled.append(frame) or {"status": "passed"}, {}),
    )
    plan = {
        "numerical_anchors": [288, 320],
        "nominal_seeds": [42, 43, 44, 45],
        "additional_direct_seed": 42,
        "latent_limits": {"relative_rms": 0.02, "maxabs": 0.1},
        "action_limits": {"rms": 0.002, "maxabs": 0.02},
    }
    rows, parity = runner.terminal_checks(
        plan, None, TinyMemory(), {288: "a", 320: "b"}, None, tmp_path, 0
    )
    assert len(rows) == len(parity) == 2 and disabled == ["a", "b"] and len(sample_calls) == 20
    assert all(not row["old_latent_gate"] for row in rows)
    assert all(value["old_action_gate"] for row in rows for value in row["active_actions"])
    assert (
        len(list(tmp_path.glob("terminal-*.pt"))) == len(list(tmp_path.glob("disabled-*.pt"))) == 2
    )
    assert (tmp_path / "TERMINAL_NUMERICAL.json").exists() and (
        tmp_path / "DISABLED_PARITY.json"
    ).exists()
    assert not runner.adoption({"quality": True}, rows, {"passed": True})["terminal_numerical"]


def test_failed_aggregate_keeps_completed_result_and_does_not_retry(tmp_path, monkeypatch):
    report = {
        "status": "failed",
        "technical_validity": "passed",
        "measurements_completed": True,
        "optimizer_steps": 200,
        "continuation_gate": False,
        "integrity": {"base_unchanged": True},
    }
    calls = []
    monkeypatch.setattr(runner, "run", lambda *args: calls.append(True) or report)
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    args = SimpleNamespace(output=tmp_path / "output", wall_seconds=10)
    with pytest.raises(SystemExit) as error:
        runner.execute(args)
    result = runner.read(args.output / "RESULT.json")
    assert error.value.code == 1 and calls == [True]
    assert result["optimizer_steps"] == 200 and result["technical_validity"] == "passed"
    assert result["measurements_completed"] and result["integrity"]["base_unchanged"]
    assert "exception" not in result


def test_technical_failure_is_terminal_and_preserves_step_evidence(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "output", wall_seconds=10)

    def fail(*args):
        (tmp_path / "output" / "training_steps.jsonl").write_text(
            '{"optimizer_steps_completed": 3}\n'
        )
        raise ValueError("nonfinite action")

    monkeypatch.setattr(runner, "run", fail)
    with pytest.raises(ValueError, match="nonfinite action"):
        runner.execute(args)
    result = runner.read(args.output / "RESULT.json")
    assert result["status"] == "failed" and result["technical_validity"] == "not_established"
    assert not result["automatic_retry"] and not result["optimizer_resume_authorized"]
    assert len(result["completed_step_evidence"]) == 1
    assert (args.output / "training_steps.jsonl").read_text().endswith("3}\n")


def test_original_quality_and_numeric_limits_are_unchanged():
    before = {
        "metrics": {
            "flow_loss": 1.0,
            "combined_mae_mixed_units": 1.0,
            "arm_mae_rad": 1.0,
            "hand_mae_u": 1.0,
        }
    }
    after = {
        "metrics": {
            "flow_loss": 0.9,
            "combined_mae_mixed_units": 1.0,
            "arm_mae_rad": 1.02,
            "hand_mae_u": 1.02,
        }
    }
    assert all(runner.overfit_gate(before, after).values())
    after["metrics"]["flow_loss"] = 0.90001
    assert not runner.overfit_gate(before, after)["flow_improved_10pct"]
    assert runner.old_gate({"rms": 0.002, "maxabs": 0.02}, {"rms": 0.002, "maxabs": 0.02})
    assert not runner.old_gate(
        {"relative_rms": 0.0067, "maxabs": 0.125}, {"relative_rms": 0.02, "maxabs": 0.1}
    )
