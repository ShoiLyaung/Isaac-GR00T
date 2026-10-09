from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types

from gr00t.experimental.arena_codec_streaming_backbone import SEED, WIDTH, CausalMamba2Backbone
from gr00t.experimental.arena_functional_mamba_transition import (
    FunctionalMamba2Backbone,
    functional_mamba2_step,
)
import pytest
import torch
from torch import nn
from torch.nn import functional as F


class _FakeGatedRMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = 1e-5

    def forward(self, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gated = value * F.silu(gate)
        return F.rms_norm(gated, (gated.shape[-1],), self.weight, self.eps)


class _FakeMamba2(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        ngroups: int,
        chunk_size: int,
        use_mem_eff_path: bool,
        layer_idx: int,
        device: str,
        dtype: torch.dtype,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = None
        self.expand = expand
        self.process_group = None
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner
        self.ngroups = ngroups
        self.nheads = self.d_ssm // headdim
        self.D_has_hdim = False
        self.rmsnorm = True
        self.norm_before_gate = False
        self.dt_limit = (0.0, float("inf"))
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        projected = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, projected, bias=False, device=device, dtype=dtype)
        conv_width = self.d_ssm + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_width,
            conv_width,
            d_conv,
            groups=conv_width,
            padding=d_conv - 1,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.A_log = nn.Parameter(torch.rand(self.nheads, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.randn(self.nheads, device=device, dtype=dtype))
        self.D = nn.Parameter(torch.randn(self.nheads, device=device, dtype=dtype))
        self.norm = _FakeGatedRMSNorm(self.d_ssm).to(device=device, dtype=dtype)
        self.out_proj = nn.Linear(
            self.d_inner,
            d_model,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del max_seqlen
        device = self.out_proj.weight.device
        return (
            torch.zeros(batch_size, 640, 4, device=device, dtype=dtype),
            torch.zeros(batch_size, 8, 64, 64, device=device, dtype=dtype),
        )


def _install_fake_mamba(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_package = types.ModuleType("mamba_ssm")
    fake_package.Mamba2 = _FakeMamba2
    monkeypatch.setitem(sys.modules, "mamba_ssm", fake_package)


def _mutating_teacher_step(
    module: _FakeMamba2,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = hidden_states.dtype
    projected = module.in_proj(hidden_states.squeeze(1))
    z, xbc, dt = torch.split(projected, (512, 640, 8), dim=-1)
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = xbc
    xbc = torch.sum(conv_state * module.conv1d.weight.squeeze(1), dim=-1)
    xbc = F.silu(xbc + module.conv1d.bias).to(dtype=dtype)
    x_flat, b_group, c_group = torch.split(xbc, (512, 64, 64), dim=-1)
    a = -torch.exp(module.A_log.float())
    dt = F.softplus(dt + module.dt_bias.to(dtype=dt.dtype))
    alpha = torch.exp(dt * a)
    x = x_flat.reshape(hidden_states.shape[0], 8, 64)
    state_input = torch.einsum("bh,bn,bhp->bhpn", dt, b_group, x)
    ssm_state.copy_(ssm_state * alpha[:, :, None, None] + state_input)
    y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), c_group)
    y = y + module.D.to(dtype).reshape(1, 8, 1) * x
    output = module.out_proj(module.norm(y.reshape(hidden_states.shape[0], 512), z))
    return output.unsqueeze(1), conv_state, ssm_state


def _storage_ids(state) -> list[int]:
    return [tensor.untyped_storage().data_ptr() for pair in state.layers for tensor in pair]


def test_functional_step_matches_official_equation_and_preserves_inputs() -> None:
    torch.manual_seed(7)
    module = _FakeMamba2(
        d_model=256,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=16,
        use_mem_eff_path=False,
        layer_idx=0,
        device="cpu",
        dtype=torch.float32,
    )
    hidden = torch.randn(2, 1, WIDTH)
    conv = torch.randn(2, 640, 4)
    ssm = torch.randn(2, 8, 64, 64)
    conv_before = conv.clone()
    ssm_before = ssm.clone()
    teacher = _mutating_teacher_step(module, hidden, conv.clone(), ssm.clone())
    functional = functional_mamba2_step(module, hidden, conv, ssm)

    for actual, expected in zip(functional, teacher, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(conv, conv_before)
    assert torch.equal(ssm, ssm_before)
    assert functional[1].untyped_storage().data_ptr() != conv.untyped_storage().data_ptr()
    assert functional[2].untyped_storage().data_ptr() != ssm.untyped_storage().data_ptr()


def test_functional_backbone_reuses_exact_parameter_set_and_new_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mamba(monkeypatch)
    official = CausalMamba2Backbone(seed=SEED)
    functional = FunctionalMamba2Backbone(seed=SEED)
    assert official.state_dict().keys() == functional.state_dict().keys()
    assert all(
        torch.equal(official.state_dict()[name], functional.state_dict()[name])
        for name in official.state_dict()
    )
    assert len(tuple(functional.named_parameters())) == 20

    state = functional.new_functional_state(1)
    teacher_state = functional.new_official_state(1)
    assert len(set(_storage_ids(state))) == 4
    assert not set(_storage_ids(state)) & set(_storage_ids(teacher_state))
    before_ids = _storage_ids(state)
    before = state.clone()
    output, after = functional.functional_step(torch.randn(1, WIDTH), state)
    assert output.shape == (1, WIDTH)
    assert state.observations == 0
    assert after.observations == 1
    assert all(
        torch.equal(first, second)
        for first_pair, second_pair in zip(state.layers, before.layers, strict=True)
        for first, second in zip(first_pair, second_pair, strict=True)
    )
    assert not set(before_ids) & set(_storage_ids(after))
    assert len(set(_storage_ids(after))) == 4
    assert state.tensor_nbytes() == after.tensor_nbytes()


def test_functional_stream_is_chunk_exact_and_keeps_cross_frame_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mamba(monkeypatch)
    module = FunctionalMamba2Backbone(seed=SEED)
    descriptors = torch.randn(1, 4, WIDTH, requires_grad=True)
    continuous, _ = module.functional_stream(descriptors)
    first, state = module.functional_stream(descriptors[:, :1])
    second, state = module.functional_stream(descriptors[:, 1:], state)
    chunked = torch.cat((first, second), dim=1)
    torch.testing.assert_close(continuous, chunked, rtol=0, atol=0)
    assert state.observations == 4

    continuous[:, -1].square().mean().backward()
    assert descriptors.grad is not None
    assert torch.isfinite(descriptors.grad).all()
    assert float(descriptors.grad[:, 0].norm()) > 0.0
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_functional_transition_source_has_no_forbidden_graph_operation() -> None:
    functions = (
        functional_mamba2_step,
        FunctionalMamba2Backbone.functional_step_with_trace,
        FunctionalMamba2Backbone.functional_step,
        FunctionalMamba2Backbone.functional_stream,
    )
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "detach" not in calls
        assert "copy_" not in calls
        assert "no_grad" not in calls
        assert not any(
            isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
            for node in ast.walk(tree)
        )
