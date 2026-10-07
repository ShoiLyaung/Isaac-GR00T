"""CPU checks for the thin executor and fixed-condition repeat/cost contracts."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.model.modules.fp32_visual_memory import FP32VisualMemory
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import fp32_memory_validation as probe  # noqa: E402
from runner_core import set_seed  # noqa: E402


class State:
    def __init__(self):
        self.layers = [(torch.zeros(2, 4), torch.zeros(2, 4))]
        self.observations = 0


class ToyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(4, 4)
        self.output_projection = nn.Linear(4, 6)

    def full_sequence(self, sequence):
        assert not torch.backends.cuda.matmul.allow_tf32
        return self.input_projection(sequence).cumsum(0)

    @torch.no_grad()
    def step(self, observation, state):
        assert not torch.backends.cuda.matmul.allow_tf32
        hidden = self.input_projection(observation)
        state.layers[0][0].add_(hidden)
        state.observations += 1
        return state.layers[0][0].clone(), self.output_projection(state.layers[0][0])


def test_fp32_full_covers_readout_and_preserves_gradient_then_restores(monkeypatch):
    entered = []

    @contextmanager
    def disabled(device, *, enabled):
        assert device == "cuda" and enabled is False
        entered.append(True)
        try:
            yield
        finally:
            entered.pop()

    monkeypatch.setattr(torch, "autocast", disabled)
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        memory = ToyMemory()
        executor = FP32VisualMemory(memory)

        def readout_scope(_module, args):
            assert (
                entered
                and not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32
            )

        handle = memory.output_projection.register_forward_pre_hook(readout_scope)
        latent, residual = executor.full(torch.randn(3, 2, 4))
        handle.remove()
        assert latent.dtype == residual.dtype == torch.float32
        residual.square().mean().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0)
            for p in memory.parameters()
        )
        assert (
            torch.backends.cuda.matmul.allow_tf32
            and torch.backends.cudnn.allow_tf32
            and not entered
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def test_precision_restores_on_readout_exception_and_state_is_caller_owned(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        memory = ToyMemory()
        executor = FP32VisualMemory(memory)
        a, b, fresh = State(), State(), State()
        value = torch.randn(2, 4)
        executor.step(value, a)
        executor.step(value, a)
        left = executor.step(value, b)[0]
        right = executor.step(value, fresh)[0]
        assert torch.equal(left, right) and a.observations == 2 and b.observations == 1
        assert not left.requires_grad

        def fail(_module, args):
            raise RuntimeError("readout failure")

        handle = memory.output_projection.register_forward_pre_hook(fail)
        with pytest.raises(RuntimeError, match="readout"):
            executor.full(torch.ones(1, 2, 4))
        handle.remove()
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        with pytest.raises(ValueError, match="observations"):
            executor.full(torch.ones(1, 2, 4, dtype=torch.bfloat16))
        a.layers[0] = (a.layers[0][0].bfloat16(), a.layers[0][1])
        with pytest.raises(ValueError, match="caches"):
            executor.step(value, a)
        with pytest.raises(ValueError, match="master"):
            FP32VisualMemory(ToyMemory().bfloat16())
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


class Encoder(nn.Module):
    def forward(self, value, timestep, embodiment):
        return value


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = Encoder()
        self.calls = 0
        self.mutate = False

    def get_action_with_features(self, **kwargs):
        self.calls += 1
        noise = torch.randn(1, 40, 13)
        for step in range(4):
            self.action_encoder(noise + step, torch.tensor([step * 250]), kwargs["embodiment_id"])
        if self.mutate:
            kwargs["backbone_features"].add_(1)
        # Deliberate same-input repeat variation must be reported, not rejected.
        return {"action_pred": noise + kwargs["backbone_features"].mean() + self.calls * 0.0001}


def frozen_kwargs():
    return {
        "backbone_features": torch.ones(1, 209, 2048),
        "state_features": torch.ones(1, 1, 8),
        "embodiment_id": torch.tensor([10]),
        "options": None,
        "backbone_output": BatchFeature(
            data={
                "backbone_features": torch.ones(1, 209, 2048),
                "image_mask": torch.ones(1, 209, dtype=torch.bool),
                "backbone_attention_mask": torch.ones(1, 209, dtype=torch.bool),
            }
        ),
        "action_input": BatchFeature(
            data={"state": torch.ones(1, 1, 13), "embodiment_id": torch.tensor([10])}
        ),
    }


def test_repeated_sampling_uses_one_context_and_actual_initial_noise(monkeypatch):
    torch.set_num_threads(4)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(probe, "set_seed", lambda value, **kwargs: set_seed(value))
    head = Head()
    kwargs = frozen_kwargs()
    initial = probe.sampling_identity(kwargs)
    actions, records, floor = probe.repeated_actions(
        SimpleNamespace(action_head=head), kwargs, 42, 3
    )
    assert actions.shape == (3, 1, 40, 13) and len(records) == len(floor) == 3
    assert all(
        r["context"] == initial and r["initial_noise"] == records[0]["initial_noise"]
        for r in records
    )
    assert any(r["maxabs"] > 0 for r in floor)
    assert probe.sampling_identity(kwargs) == initial and not head.action_encoder._forward_pre_hooks
    head.mutate = True
    with pytest.raises(ValueError, match="fixed context"):
        probe.repeated_actions(SimpleNamespace(action_head=head), kwargs, 42, 3)
    assert not head.action_encoder._forward_pre_hooks


def test_fp32_policy_keeps_native_encode_step_sample_order(monkeypatch):
    order = []
    model = SimpleNamespace(
        get_action=lambda values: order.append("native") or {"action_pred": torch.zeros(1, 40, 13)}
    )
    monkeypatch.setattr(
        probe.proto, "encode_observation", lambda *args: order.append("encode") or "packet"
    )
    monkeypatch.setattr(probe.proto, "slots", lambda packet: torch.zeros(192, 2048))
    monkeypatch.setattr(
        probe.proto, "sample_packet", lambda *args: order.append("sample") or torch.zeros(1, 40, 13)
    )
    memory = SimpleNamespace(step=lambda *args: (order.append("bf16"), torch.zeros(192, 2048)))
    executor = SimpleNamespace(step=lambda *args: (order.append("fp32"), torch.zeros(192, 2048)))
    probe.policy_action(model, memory, executor, None, {}, "native")
    assert order == ["native"]
    order.clear()
    probe.policy_action(model, memory, executor, State(), {}, "fp32_step")
    assert order == ["encode", "fp32", "sample"]


def test_cost_adoption_is_reported_not_an_early_exception():
    plan = {"max_extra_policy_p95_fraction": 0.1, "max_extra_peak_allocated_bytes": 512 * 2**20}
    base = {"p95_ms": 100, "peak_allocated_bytes": 6 * 2**30}
    edge = {"p95_ms": 110, "peak_allocated_bytes": 6 * 2**30 + 512 * 2**20}
    assert probe.cost_gate(base, edge, plan)["p95_passed"]
    assert probe.cost_gate(base, edge, plan)["extra_peak_passed"]
    failed = probe.cost_gate(
        base, {"p95_ms": 111, "peak_allocated_bytes": edge["peak_allocated_bytes"] + 1}, plan
    )
    assert not failed["p95_passed"] and not failed["extra_peak_passed"]


def test_adoption_failure_preserves_completed_result_and_refuses_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(
        probe,
        "run",
        lambda *args: {
            "status": "failed",
            "technical_validity": "passed",
            "limited_repair_supported": False,
            "measurements_completed": True,
            "action_count": 120,
            "flow_count": 40,
            "cost_calls": 330,
        },
    )
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    args = SimpleNamespace(output=tmp_path / "attempt", wall_seconds=10)
    with pytest.raises(SystemExit) as failure:
        probe.execute(args)
    assert failure.value.code == 1
    report = probe.read(args.output / "RESULT.json")
    assert report["technical_validity"] == "passed" and report["measurements_completed"] is True
    assert (report["action_count"], report["flow_count"], report["cost_calls"]) == (120, 40, 330)
    with pytest.raises(FileExistsError):
        probe.execute(args)


def test_cost_native_cleanup_restore_and_timer_boundary(monkeypatch, tmp_path):
    import run_train_eval

    events = []
    parameter = SimpleNamespace(dtype=torch.float32, is_cuda=False)

    class Memory:
        def parameters(self):
            return [parameter]

        def cpu(self):
            parameter.is_cuda = False
            events.append("cpu")
            return self

        def cuda(self):
            parameter.is_cuda = True
            events.append("cuda")
            return self

    class Snapshot:
        def __init__(self, observations):
            self.observations = observations

        def clone(self):
            events.append("clone")
            return Snapshot(self.observations)

        def to(self, device):
            events.append("restore")
            return Snapshot(self.observations)

    memory = Memory()
    keys = (
        "state",
        "action",
        "action_mask",
        "embodiment_id",
        "input_ids",
        "attention_mask",
        "image_grid_thw",
    )
    batch = {key: torch.zeros(1) for key in keys}
    source = SimpleNamespace(
        processor=SimpleNamespace(eval=lambda: None),
        modality={"video": SimpleNamespace(modality_keys=[])},
        batch=lambda pairs: (batch, [None]),
    )
    cache = {
        anchor: {
            "input_identity": {k: probe.tensor_identity(v) for k, v in batch.items()},
            "post_vl_features": torch.zeros(1, 209, 2048),
        }
        for anchor in (288, 320)
    }
    plan = {
        "prepared": str(tmp_path),
        "anchors": [288, 320],
        "prefixes": {"288": list(range(0, 289, 16)), "320": list(range(0, 321, 16))},
        "cost_order": {
            "288": ["native", "bf16_step", "fp32_step"],
            "320": ["fp32_step", "bf16_step", "native"],
        },
        "cost_modes": ["native", "bf16_step", "fp32_step"],
        "cost_warmup": 5,
        "cost_measured": 50,
        "cost_seed": 42,
        "max_extra_policy_p95_fraction": 0.1,
        "max_extra_peak_allocated_bytes": 512 * 2**20,
    }
    monkeypatch.setattr(run_train_eval, "NativeAnchors", lambda *args: source)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(probe, "budget", lambda *args: 0)
    monkeypatch.setattr(probe.proto, "validate_input_layout", lambda *args: None)
    monkeypatch.setattr(
        probe.proto,
        "encode_observation",
        lambda *args: {"features": SimpleNamespace(backbone_features=torch.zeros(1, 209, 2048))},
    )
    monkeypatch.setattr(probe.proto, "state_hash", lambda state: str(state.observations))
    monkeypatch.setattr(
        probe,
        "prefill_snapshot",
        lambda m, e, c, indices, mode: (Snapshot(len(indices)), {"observations": len(indices)}),
    )
    monkeypatch.setattr(probe, "call_audit", lambda *args: {"instrumentation_removed": True})
    monkeypatch.setattr(probe, "set_seed", lambda *args, **kwargs: events.append("seed"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset_peak"))
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda: 120 if parameter.is_cuda else 100
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 140)
    counter = iter(range(10000))

    def clock():
        events.append("clock")
        return next(counter)

    monkeypatch.setattr(probe.time, "perf_counter", clock)

    def policy(model, m, executor, state, observation, mode):
        assert events[-1] == "clock"
        assert events[-4:] == ["seed", "reset_peak", "sync", "clock"]
        if mode == "native":
            assert not parameter.is_cuda and state is None
        else:
            assert parameter.is_cuda and state is not None
            assert events[-5] == "clone"
            state.observations += 1
        events.append("policy")
        return torch.zeros(1, 40, 13)

    monkeypatch.setattr(probe, "policy_action", policy)

    def checked(value, *args):
        assert events[-2:] == ["sync", "clock"]
        events.append("check")
        return value, {"finite": True, "action_identity": probe.tensor_identity(value)}

    monkeypatch.setattr(probe.proto, "action_record", checked)
    reports = probe.cost_probes(plan, None, memory, cache, tmp_path, float("inf"))
    assert len(reports) == 2 and events.count("policy") == 330
    assert events.count("clone") == 220 and not parameter.is_cuda
    assert all(
        r["candidate_gate"]["p95_passed"] and r["candidate_gate"]["extra_peak_passed"]
        for r in reports
    )
