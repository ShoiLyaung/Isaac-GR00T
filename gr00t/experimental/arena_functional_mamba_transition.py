"""Functional, immutable single-token transition for the frozen Arena Mamba2.

The transition reads parameters directly from the official ``Mamba2`` modules
owned by :class:`CausalMamba2Backbone`.  It creates no trainable parameter copy
and expresses both recurrent state updates with ordinary PyTorch tensor
operations so autograd can cross frame and chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.experimental.arena_codec_streaming_backbone import (
    LAYERS,
    WIDTH,
    CausalMamba2Backbone,
    Mamba2State,
    _validate_descriptor,
    _validate_descriptor_sequence,
)


@dataclass(frozen=True)
class FunctionalMamba2State:
    """Persistent state whose transition always returns new tensor storage."""

    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    batch_size: int
    observations: int = 0

    def clone(self) -> FunctionalMamba2State:
        return FunctionalMamba2State(
            tuple(tuple(tensor.clone() for tensor in pair) for pair in self.layers),
            self.batch_size,
            self.observations,
        )

    def reset(self) -> FunctionalMamba2State:
        return FunctionalMamba2State(
            tuple(tuple(torch.zeros_like(tensor) for tensor in pair) for pair in self.layers),
            self.batch_size,
            0,
        )

    def tensor_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for pair in self.layers for tensor in pair
        )


def _validate_official_mamba2(module: nn.Module) -> None:
    expected = {
        "d_model": WIDTH,
        "d_state": 64,
        "d_conv": 4,
        "conv_init": None,
        "expand": 2,
        "d_inner": 512,
        "headdim": 64,
        "d_ssm": 512,
        "ngroups": 1,
        "nheads": 8,
        "D_has_hdim": False,
        "rmsnorm": True,
        "norm_before_gate": False,
        "dt_limit": (0.0, float("inf")),
        "activation": "silu",
        "use_mem_eff_path": False,
    }
    changed = {
        name: getattr(module, name, None)
        for name, value in expected.items()
        if getattr(module, name, None) != value
    }
    if changed:
        raise ValueError(f"official Mamba2 functional contract changed: {changed}")
    if getattr(module, "process_group", None) is not None:
        raise ValueError("tensor-parallel Mamba2 is outside the frozen functional contract")
    if module.in_proj.bias is not None or module.out_proj.bias is not None:
        raise ValueError("frozen Mamba2 input and output projections must be bias-free")
    if module.conv1d.bias is None:
        raise ValueError("frozen Mamba2 depthwise convolution bias is required")
    if tuple(module.in_proj.weight.shape) != (1160, WIDTH):
        raise ValueError("frozen Mamba2 input projection shape changed")
    if tuple(module.conv1d.weight.shape) != (640, 1, 4):
        raise ValueError("frozen Mamba2 convolution shape changed")
    if tuple(module.out_proj.weight.shape) != (WIDTH, 512):
        raise ValueError("frozen Mamba2 output projection shape changed")


def functional_mamba2_step(
    module: nn.Module,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the official Mamba2 step equation without mutating input state."""

    if hidden_states.ndim != 3 or tuple(hidden_states.shape[1:]) != (1, WIDTH):
        raise ValueError("functional Mamba2 expects hidden state [batch,1,256]")
    batch = int(hidden_states.shape[0])
    if tuple(conv_state.shape) != (batch, 640, 4):
        raise ValueError("functional Mamba2 conv state must have shape [batch,640,4]")
    if tuple(ssm_state.shape) != (batch, 8, 64, 64):
        raise ValueError("functional Mamba2 SSM state must have shape [batch,8,64,64]")
    dtype = hidden_states.dtype
    projected = module.in_proj(hidden_states.squeeze(1))
    z, xbc_input, dt_raw = torch.split(projected, (512, 640, 8), dim=-1)

    new_conv_state = torch.cat(
        (conv_state[..., 1:], xbc_input.unsqueeze(-1)),
        dim=-1,
    )
    conv_weight = module.conv1d.weight.squeeze(1)
    convolved = torch.sum(new_conv_state * conv_weight, dim=-1)
    convolved = F.silu(convolved + module.conv1d.bias).to(dtype=dtype)
    x_flat, b_flat, c_flat = torch.split(convolved, (512, 64, 64), dim=-1)

    x = x_flat.reshape(batch, 8, 64)
    b_group = b_flat.reshape(batch, 1, 64)
    c_group = c_flat.reshape(batch, 1, 64)
    a = -torch.exp(module.A_log.float())
    dt = F.softplus(dt_raw + module.dt_bias.to(dtype=dt_raw.dtype))
    alpha = torch.exp(dt * a)
    # ngroups=1: the B/C group is broadcast over all eight heads, matching
    # official ``einsum('bh,bn,bhp->bhpn', ...)`` without parameter copies.
    state_input = torch.einsum("bh,bn,bhp->bhpn", dt, b_group.squeeze(1), x)
    new_ssm_state = ssm_state * alpha[:, :, None, None] + state_input
    y = torch.einsum("bhpn,bn->bhp", new_ssm_state.to(dtype), c_group.squeeze(1))
    y = y + module.D.to(dtype).reshape(1, 8, 1) * x
    y = y.reshape(batch, 512)
    y = module.norm(y, z)
    output = module.out_proj(y).unsqueeze(1)
    return output, new_conv_state, new_ssm_state


class FunctionalMamba2Backbone(CausalMamba2Backbone):
    """The frozen two-layer residual Mamba2 with an autograd-capable step path."""

    def __init__(self, *, seed: int, chunk_size: int = 16):
        super().__init__(seed=seed, chunk_size=chunk_size)
        if len(self.layers) != LAYERS:
            raise ValueError("Arena functional Mamba2 must contain exactly two layers")
        for index, layer in enumerate(self.layers):
            _validate_official_mamba2(layer.mamba)
            if layer.mamba.layer_idx != index:
                raise ValueError("Arena functional Mamba2 layer_idx contract changed")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("Arena functional Mamba2 must initialize in FP32")

    def new_official_state(self, batch_size: int) -> Mamba2State:
        """Allocate and clone independent caches for the mutating teacher path."""

        allocated = super().new_state(batch_size)
        return allocated.clone()

    def new_functional_state(self, batch_size: int) -> FunctionalMamba2State:
        """Allocate via official modules, then clone into independent storage."""

        if batch_size < 1:
            raise ValueError("functional Mamba2 state batch must be positive")
        dtype = next(self.parameters()).dtype
        allocated = tuple(
            layer.mamba.allocate_inference_cache(batch_size, max_seqlen=1, dtype=dtype)
            for layer in self.layers
        )
        return FunctionalMamba2State(
            tuple(tuple(tensor.clone() for tensor in pair) for pair in allocated),
            batch_size,
        )

    def functional_step_with_trace(
        self,
        descriptor: torch.Tensor,
        state: FunctionalMamba2State,
    ) -> tuple[torch.Tensor, FunctionalMamba2State, tuple[torch.Tensor, ...]]:
        """Return one output, new immutable state, and both residual-layer outputs."""

        batch = _validate_descriptor(descriptor)
        if batch != state.batch_size or len(state.layers) != len(self.layers):
            raise ValueError("functional Mamba2 state does not match batch or layer count")
        value = descriptor.unsqueeze(1)
        new_layers = []
        residual_outputs = []
        for layer, (conv_state, ssm_state) in zip(
            self.layers,
            state.layers,
            strict=True,
        ):
            update, new_conv, new_ssm = functional_mamba2_step(
                layer.mamba,
                layer.norm(value),
                conv_state,
                ssm_state,
            )
            value = value + update
            new_layers.append((new_conv, new_ssm))
            residual_outputs.append(value.squeeze(1))
        new_state = FunctionalMamba2State(
            tuple(new_layers),
            state.batch_size,
            state.observations + 1,
        )
        return value.squeeze(1), new_state, tuple(residual_outputs)

    def functional_step(
        self,
        descriptor: torch.Tensor,
        state: FunctionalMamba2State,
    ) -> tuple[torch.Tensor, FunctionalMamba2State]:
        output, new_state, _ = self.functional_step_with_trace(descriptor, state)
        return output, new_state

    def functional_stream(
        self,
        descriptors: torch.Tensor,
        state: FunctionalMamba2State | None = None,
    ) -> tuple[torch.Tensor, FunctionalMamba2State]:
        batch, _ = _validate_descriptor_sequence(descriptors)
        current = self.new_functional_state(batch) if state is None else state
        outputs = []
        for descriptor in descriptors.unbind(dim=1):
            output, current = self.functional_step(descriptor, current)
            outputs.append(output)
        return torch.stack(outputs, dim=1), current
