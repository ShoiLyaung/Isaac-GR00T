"""CPU interface/gradient/lifetime checks; actual CUDA captures require formal GPU run."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import budgeted_action_cudagraph as graph  # noqa: E402
import budgeted_action_cudagraph_probe as probe  # noqa: E402
from gr00t.model.modules.budgeted_action_mamba import ActionCondition  # noqa: E402


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)
        self.adapter = nn.Linear(4, 4)
        self.encodes = 0

    def encode_condition(self, features):
        self.encodes += 1
        code = self.adapter(features.mean(1))
        return ActionCondition(code, code)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        image_mask,
        backbone_attention_mask,
        return_all_hidden_states=False,
        condition_cache=None,
    ):
        hidden = (
            hidden_states
            + encoder_hidden_states.mean(1)[:, None]
            + timestep[:, None, None].float() * 0.001
        )
        if condition_cache is not None:
            hidden = hidden + condition_cache.modulation[:, None]
        output = self.projection(hidden)
        return (output, [hidden, output]) if return_all_hidden_states else output


def inputs():
    return (
        torch.randn(1, 41, 4),
        torch.randn(1, 209, 4),
        torch.tensor([250]),
        torch.ones(1, 209, dtype=torch.bool),
        torch.ones(1, 209, dtype=torch.bool),
    )


@pytest.fixture
def amp(monkeypatch):
    calls = []

    @contextmanager
    def context(device, **kwargs):
        assert device == "cuda" and kwargs["dtype"] == torch.bfloat16
        calls.append(kwargs.get("cache_enabled"))
        yield

    monkeypatch.setattr(torch, "autocast", context)
    return calls


def test_tensor_bridge_abi_preserves_native_output_without_changing_core(amp):
    core = Core()
    args = inputs()
    surface = graph.TensorCore(core, student=False, backward=False)
    kwargs = dict(zip(graph.TENSOR_KEYS, args))
    kwargs["return_all_hidden_states"] = True
    expected = core(**kwargs)[0]
    dispatcher = graph.CoreDispatcher(surface)
    actual, ignored = dispatcher(**kwargs)
    assert torch.equal(actual, expected) and ignored == [actual]
    assert dispatcher.calls == 1 and all(value is False for value in amp)
    assert core.forward.__func__ is Core.forward


def test_student_fb_surface_keeps_condition_gradient_and_single_encode(amp):
    core = Core()
    args = inputs()
    surface = graph.TensorCore(core, student=True, backward=True)
    output = surface(*args)
    output.square().mean().backward()
    assert core.encodes == 1
    assert core.adapter.weight.grad is not None and torch.count_nonzero(core.adapter.weight.grad)
    assert core.projection.weight.grad is not None
    assert all(value is False for value in amp)


def test_inference_modulation_explicit_and_request_restoration(amp):
    core = Core()
    args = inputs()
    condition = core.encode_condition(args[1])
    surface = graph.TensorCore(core, student=True, backward=False)
    dispatcher = graph.CoreDispatcher(surface)
    kwargs = {
        **dict(zip(graph.TENSOR_KEYS, args)),
        "condition_cache": condition,
        "return_all_hidden_states": True,
    }
    actual, _ = dispatcher(**kwargs)
    expected = core(**kwargs)[0]
    assert torch.equal(actual, expected) and core.encodes == 1
    owner = SimpleNamespace(action_head=SimpleNamespace(model=core))
    with pytest.raises(RuntimeError):
        with graph.replace_core(owner, dispatcher):
            raise RuntimeError("test")
    assert owner.action_head.model is core
    copied = graph.tensor_args(kwargs, student=True, backward=False)
    assert len(copied) == 6 and all(not value.requires_grad for value in copied)
    assert copied[-1].data_ptr() != condition.modulation.data_ptr()


def test_forward_replay_copies_all_dynamic_inputs_and_rejects_closed_graph():
    lease = graph.ForwardGraph.__new__(graph.ForwardGraph)
    a = inputs()
    lease.static = tuple(torch.zeros_like(x) for x in a)
    lease.closed = False
    lease.calls = 0
    lease.output = torch.zeros(1)
    snapshots = []
    lease.graph = SimpleNamespace(
        replay=lambda: snapshots.append(tuple(x.clone() for x in lease.static))
    )
    b = tuple(x + 1 if x.dtype != torch.bool else ~x for x in a)
    for values in (a, b, a):
        lease(*values)
    assert lease.calls == 3
    assert all(torch.equal(x, y) for x, y in zip(snapshots[0], snapshots[2]))
    assert all(torch.equal(x, y) for x, y in zip(snapshots[1], b))
    lease.closed = True
    with pytest.raises(RuntimeError):
        lease(*a)


def test_factory_only_patches_wrapper_and_release_restores_it(monkeypatch, amp):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    core = Core()
    surface = graph.TensorCore(core, student=True, backward=True)
    bridge = graph.TensorCore(core, student=True, backward=True)
    old = surface.forward.__func__
    core_old = core.forward.__func__

    def factory(module, args, num_warmup_iters, allow_unused_input):
        assert (
            not torch.is_autocast_enabled("cuda") and num_warmup_iters == 3 and allow_unused_input
        )
        original = module.forward
        module.forward = lambda *values: original(*values)
        return module

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", factory)
    lease = graph.BackwardGraph(surface, inputs())
    assert bridge.forward.__func__ is graph.TensorCore.forward and core.forward.__func__ is core_old
    lease.close()
    assert surface.forward.__func__ is old and lease.closed and lease.callable is None


def test_capture_requires_hook_free_module():
    core = Core()
    handle = core.projection.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="hooks"):
        graph.require_hook_free(core)
    handle.remove()
    graph.require_hook_free(core)


def test_group_gradient_gate_catches_small_disconnected_condition():
    reference = {"large": torch.ones(1000), "small": torch.ones(1) * 0.001}
    candidate = {"large": torch.ones(1000), "small": torch.zeros(1)}
    limits = {"relative_l2": 0.02, "cosine_min": 0.999, "zero_reference_norm": 1e-12}
    report = probe.gradient_comparison(
        reference, candidate, {"large": "backbone", "small": "condition"}, limits
    )
    assert report["groups"]["GLOBAL"]["passed"]
    assert not report["groups"]["condition"]["passed"] and not report["passed"]
    assert report["parameters"][0]["sample_indices"] == [0, 142, 285, 428, 570, 713, 856, 999]


def test_gradient_none_pattern_zero_reference_and_clear_contract():
    limits = {"relative_l2": 0.02, "cosine_min": 0.999, "zero_reference_norm": 1e-12}
    report = probe.gradient_comparison({"p": None}, {"p": torch.zeros(2)}, {"p": "g"}, limits)
    assert not report["passed"] and not report["none_pattern_match"]
    assert probe.metric_from_sums(
        {"reference_sq": 0.0, "candidate_sq": 0.0, "dot": 0.0, "error_sq": 0.0}, limits
    )["passed"]
    core = Core()
    core.projection.weight.grad = torch.ones_like(core.projection.weight)
    with pytest.raises(ValueError, match="accumulation"):
        probe.assert_gradients_cleared(core)
    core.zero_grad(set_to_none=True)
    probe.assert_gradients_cleared(core)
