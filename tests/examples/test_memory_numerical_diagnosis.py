"""CPU layout/observation/precision contracts; no official CUDA execution."""

import ast
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
import memory_numerical_diagnosis as runner  # noqa: E402
import memory_numerical_trace as trace  # noqa: E402


def test_exact_official_full_conv_act_layout_and_owned_snapshots():
    torch.set_num_threads(4)
    paths = [Path(p) / "mamba_ssm/modules/mamba2.py" for p in sys.path]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        pytest.skip("Frozen official source unavailable")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Mamba2")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assignment = next(
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "act"
    )
    model = SimpleNamespace(
        conv1d=nn.Conv1d(640, 640, 4, padding=3, groups=640), act=nn.SiLU(), d_conv=4
    )
    observer = trace.MemoryTrace(None, None, "full", 21, 192)
    with observer.context:
        observer.hook(model.conv1d, "conv_module", conv=True)
        observer.hook(model.act, "act")
        observer.context.callback(
            model.act.register_forward_pre_hook(
                lambda _m, args: observer.capture("conv_preact", args[0])
            ).remove
        )
        scope = {"self": model, "xBC": torch.randn(192, 21, 640)}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), scope)
    result = observer.export()
    assert result["metadata"]["conv_module"][0]["native"]["shape"] == [192, 640, 24]
    assert result["metadata"]["act"][0]["native"]["shape"] == [192, 21, 640]
    assert result["stages"]["act"]["values"].shape == (21, 192, 640)
    assert torch.equal(
        result["stages"]["conv_module"]["values"], result["stages"]["conv_preact"]["values"]
    )
    assert torch.equal(result["stages"]["act"]["values"], scope["xBC"].transpose(0, 1))
    expected = result["stages"]["act"]["values"].clone()
    scope["xBC"].zero_()
    assert torch.equal(result["stages"]["act"]["values"], expected)
    assert not model.conv1d._forward_hooks and not model.act._forward_pre_hooks


def test_step_normalization_preserves_time_slot_channel_and_duplicate_readout():
    observer = trace.MemoryTrace(None, None, "step", 3, 2)
    for time in range(3):
        observer.time = time
        observer.capture("norm", torch.full((2, 1, 5), float(time)))
        observer.capture("readout", torch.full((2, 7), float(time)), readout=True)
    observer.time = None
    observer.capture("readout", torch.full((2, 7), 2.0), readout=True)
    exported = observer.export()
    assert exported["stages"]["norm"]["values"].shape == (3, 2, 5)
    assert exported["stages"]["readout"]["times"] == [0, 1, 2]
    assert torch.equal(exported["stages"]["norm"]["values"][:, 0, 0], torch.arange(3.0))
    with pytest.raises(ValueError, match="Repeated"):
        observer.capture("readout", torch.zeros(2, 7), readout=True)


def test_precision_context_restores_both_flags_on_success_and_exception(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError):
            with trace.memory_precision("fp32"):
                assert (
                    not torch.backends.cuda.matmul.allow_tf32
                    and not torch.backends.cudnn.allow_tf32
                )
                raise RuntimeError("probe failure")
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        with trace.memory_precision("bf16"):
            assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


def test_error_positions_quantiles_and_first_threshold_crossing_are_separate():
    reference = torch.ones(3, 2, 5)
    actual = reference.clone()
    actual[0, 1, 2] += 0.01
    actual[2, 0, 4] += 0.25
    result = trace.error_summary(actual, reference)
    assert result["first_nonzero_position"] == [0, 1, 2] and result["max_position"] == [2, 0, 4]
    assert result["maxabs"] == 0.25 and result["max_actual"] == 1.25
    json.dumps(result, allow_nan=False)
    left = {"stages": {"layer0": {"times": [0, 1, 2], "values": actual}}}
    right = {"stages": {"layer0": {"times": [0, 1, 2], "values": reference}}}
    rows = trace.stage_comparisons(left, right, {"maxabs": 0.1, "relative_rms": 0.02})
    assert rows[0]["first_nonzero_time"] == 0 and rows[0]["first_old_threshold_crossing_time"] == 2


def test_old_failure_reproduction_is_not_a_diagnosis_failure_and_rule_is_conjunctive():
    old = {"maxabs": 0.125, "rms": 0.009497059509158134, "relative_rms": 0.006728019334615535}
    plan = {"expected_bf16_latent": old, "statistics_reproduction_atol": 1e-8}
    assert runner.reproduction(old, plan)["passed"]
    assert not runner.old_gate(old, {"relative_rms": 0.02, "maxabs": 0.1})
    assert not runner.reproduction({**old, "maxabs": 0.124999}, plan)["passed"]
    limits = {"relative_rms": 0.02, "maxabs": 0.1}
    small = {"maxabs": 0.001, "rms": 0.00001, "relative_rms": 0.0001}
    assert (
        trace.interpretation(old, small, limits, 0.1)["outcome"]
        == "precision_sensitive_strong_evidence"
    )
    assert (
        trace.interpretation(old, {**small, "rms": 0.005}, limits, 0.1)["outcome"]
        == "partial_convergence"
    )
    assert (
        trace.interpretation(old, {**small, "relative_rms": 0.03}, limits, 0.1)["outcome"]
        == "partial_convergence"
    )
    assert trace.interpretation(old, old, limits, 0.1)["outcome"] == "unresolved"


class ToyMamba(nn.Module):
    def __init__(self, official):
        super().__init__()
        self.official = official
        self.in_proj = nn.Identity()
        self.norm = nn.Identity()
        self.out_proj = nn.Identity()
        self.act = nn.SiLU()
        self.conv1d = nn.Identity()

    def step(self, value, conv, ssm):
        x = self.in_proj(value.squeeze(1))
        x = self.act(x)
        result = self.official.selective_state_update(
            ssm, x, torch.ones_like(x), torch.ones(1), x, x
        )
        return self.out_proj(self.norm(result)).unsqueeze(1), conv, ssm


class ToyLayer(nn.Module):
    def __init__(self, official):
        super().__init__()
        self.norm = nn.Identity()
        self.mamba = ToyMamba(official)

    def step(self, value, state):
        return value + self.mamba.step(self.norm(value), *state)[0]


class ToyMemory(nn.Module):
    def __init__(self, official):
        super().__init__()
        self.input_projection = nn.Identity()
        self.output_projection = nn.Identity()
        self.layers = nn.ModuleList([ToyLayer(official), ToyLayer(official)])

    @torch.no_grad()
    def step(self, value, state):
        hidden = self.input_projection(value).unsqueeze(1)
        for layer, cache in zip(self.layers, state.layers):
            hidden = layer.step(hidden, cache)
        state.observations += 1
        latent = hidden.squeeze(1)
        return latent, self.output_projection(latent)


def test_scoped_step_trace_has_independent_fresh_state_and_restores_methods():
    def update(state, x, dt, A, B, C):
        state.add_(x)
        return state.clone()

    official = SimpleNamespace(selective_state_update=update)
    memory = ToyMemory(official)

    def state():
        return SimpleNamespace(
            layers=[(torch.zeros(2, 3), torch.zeros(2, 3)) for _ in range(2)], observations=0
        )

    inputs = torch.randn(3, 2, 3)
    reference_state = state()
    reference = torch.stack([memory.step(v, reference_state)[0] for v in inputs])
    independent = state()
    with trace.MemoryTrace(memory, official, "step", 3, 2) as observer:
        actual = torch.stack([memory.step(v, independent)[0] for v in inputs])
    assert torch.equal(actual, reference)
    assert [r["observations"] for r in observer.states] == [1, 2, 3]
    assert official.selective_state_update is update
    assert "step" not in memory.__dict__ and all(
        "step" not in layer.__dict__ for layer in memory.layers
    )
    assert not memory.input_projection._forward_hooks
    with pytest.raises(RuntimeError):
        with trace.MemoryTrace(memory, official, "step", 3, 2):
            raise RuntimeError("intentional")
    assert official.selective_state_update is update and "step" not in memory.__dict__


def test_failed_attempt_preserves_terminal_and_refuses_overwrite(monkeypatch, tmp_path):
    def fail(*args):
        raise ValueError("identity mismatch")

    monkeypatch.setattr(runner, "run", fail)
    args = SimpleNamespace(output=tmp_path / "attempt", wall_seconds=10)
    with pytest.raises(ValueError, match="identity"):
        runner.execute(args)
    result = runner.read(args.output / "RESULT.json")
    assert (
        result["status"] == "failed"
        and result["optimizer_steps"] == 0
        and result["old_training_failure_unchanged"] is True
    )
    with pytest.raises(FileExistsError):
        runner.execute(args)


def test_error_summary_uses_cpu_fp64_including_quantiles_and_stage_reductions():
    reference = torch.ones(2, 3, 4, dtype=torch.float64)
    actual = reference.clone()
    actual[1, 2, 3] += 1e-9
    result = trace.error_summary(actual, reference)
    delta = actual - reference
    assert result["reduction"] == "cpu_float64"
    assert result["rms"] == float(delta.square().mean().sqrt()) > 0
    assert result["relative_rms"] == result["rms"]
    assert result["abs_quantiles"]["p99"] == float(
        torch.quantile(delta.abs().flatten(), torch.tensor(0.99, dtype=torch.float64))
    )
    left = {"stages": {"latent": {"times": [0, 1], "values": actual}}}
    right = {"stages": {"latent": {"times": [0, 1], "values": reference}}}
    stage = trace.stage_comparisons(left, right, {"maxabs": 0.1, "relative_rms": 0.02})[0]
    assert stage["summary"]["rms"] == result["rms"]
    assert all(row["reduction"] == "cpu_float64" for row in stage["per_time"])
    assert runner.old_gate(
        {"maxabs": 0.1, "relative_rms": 0.02}, {"maxabs": 0.1, "relative_rms": 0.02}
    )
    assert runner.old_gate({"maxabs": 0.02, "rms": 0.002}, {"maxabs": 0.02, "rms": 0.002})
    with pytest.raises(ValueError, match="Nonfinite"):
        trace.error_summary(torch.full_like(actual, float("nan")), reference)


def test_historical_mismatch_reports_differences_and_continues_but_bad_inputs_fail(monkeypatch):
    import numpy as np

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runner, "native_flow", lambda *args: ({"loss": torch.tensor(1.5)}, {}))
    monkeypatch.setattr(runner, "sampling_packet", lambda *args: None)
    monkeypatch.setattr(runner, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.prototype, "sample_packet", lambda *args: torch.ones(1, 40, 13))
    old_rows = {42: (0, {"paired_rng_seed": 123, "flow_loss": 0.5})}
    old_predictions = np.zeros((1, 40, 13), dtype=np.float32)
    records, outputs = runner.mode_actions(
        None,
        None,
        None,
        "bf16_step",
        [("nominal_42", 123), ("direct_42", 42)],
        old_rows,
        old_predictions,
    )
    assert len(records) == len(outputs) == 2
    compatibility = records[0]["historical_compatibility"]
    assert records[0]["historical_exact"] is False
    assert compatibility["seed_equal"] is True
    assert compatibility["action_exact"] is compatibility["flow_exact"] is False
    assert compatibility["flow_difference"] == 1
    assert compatibility["action_difference"]["maxabs"] == 1
    assert compatibility["action_difference"]["reduction"] == "cpu_float64"
    assert records[1]["historical_exact"] is None
    for invalid in (
        np.zeros((16, 13), dtype=np.float32),
        np.full((40, 13), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="finite"):
            runner.historical_action_compatibility(
                torch.ones(1, 40, 13), 1.5, 123, old_rows[42][1], invalid
            )
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.historical_action_compatibility(
            torch.ones(1, 40, 13),
            1.5,
            123,
            {"paired_rng_seed": 123, "flow_loss": float("nan")},
            old_predictions[0],
        )
    with pytest.raises(KeyError):
        runner.mode_actions(
            None, None, None, "bf16_step", [("nominal_42", 123)], {}, old_predictions
        )
    monkeypatch.setattr(
        runner.prototype, "sample_packet", lambda *args: torch.full((1, 40, 13), float("nan"))
    )
    with pytest.raises(ValueError, match="finite"):
        runner.mode_actions(
            None, None, None, "bf16_step", [("nominal_42", 123)], old_rows, old_predictions
        )
    old = {"maxabs": 0.125, "rms": 0.009497059509158134, "relative_rms": 0.006728019334615535}
    plan = {"expected_bf16_latent": old, "statistics_reproduction_atol": 1e-8}
    report = runner.reproduction({**old, "rms": old["rms"] + 2e-8}, plan)
    assert report["passed"] is False and report["role"] == "historical_compatibility_only"
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.reproduction({**old, "rms": float("nan")}, plan)
