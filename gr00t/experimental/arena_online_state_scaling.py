"""Online recurrent-state controls for the Arena scaling experiment.

This module deliberately exposes only online transitions.  In particular, the
Transformer implementation below never calls the opaque full-sequence
``TransformerEncoder`` endpoint.  ``PreallocatedKVTransformerBackbone`` reuses
the exact parameters and equations of ``StreamingKVTransformerBackbone`` while
storing projected keys and values in fixed-capacity, mutable inference buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import torch
from torch.nn import functional as F

from gr00t.experimental.arena_codec_streaming_backbone import LAYERS, SEED, WIDTH
from gr00t.experimental.arena_functional_streaming_controls import (
    ATTENTION_HEAD_WIDTH,
    ATTENTION_HEADS,
    EXPECTED_TEMPORAL_PARAMETERS,
    StreamingKVTransformerBackbone,
    storage_identity,
    transformer_kv_expected_bytes,
)


ONLINE_LENGTHS = (128, 256, 512, 925)
KV_CAPACITY = 925
# A finite, exactly representable FP32 value that is far outside initialized
# activation magnitudes.  NaN would make the experiment's finite-state gate
# impossible to interpret.
KV_SENTINEL = -32_768.0


KVBufferPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]


@dataclass
class PreallocatedKVState:
    """Mutable inference-only KV state with stable, non-aliased storage.

    ``live_index`` is the number of committed observations.  The valid prefix
    of every key/value buffer is ``:live_index``; every later slot must retain
    ``sentinel``.  A transition mutates and returns the same state object.  This
    alias is intentional and is the deployment behavior under study.
    """

    layers: tuple[KVBufferPair, ...]
    batch_size: int
    capacity: int
    live_index: int = 0
    sentinel: float = KV_SENTINEL

    @property
    def observations(self) -> int:
        return self.live_index

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for pair in self.layers for tensor in pair)

    def storage_identities(self) -> tuple[tuple[str, int, int], ...]:
        identities = tuple(storage_identity(tensor) for tensor in self.tensors())
        if len(identities) != len(set(identities)):
            raise ValueError("preallocated KV buffers contain aliased storage")
        return identities

    def capacity_nbytes(self) -> int:
        identities = self.storage_identities()
        return sum(identity[2] for identity in identities)

    def logical_nbytes(self) -> int:
        return transformer_kv_expected_bytes(
            batch_size=self.batch_size,
            observations=self.live_index,
            element_size=self.tensors()[0].element_size(),
        )

    def clone(self) -> PreallocatedKVState:
        return PreallocatedKVState(
            tuple((key.clone(), value.clone()) for key, value in self.layers),
            self.batch_size,
            self.capacity,
            self.live_index,
            self.sentinel,
        )

    def reset_(self) -> PreallocatedKVState:
        for tensor in self.tensors():
            tensor.fill_(self.sentinel)
        self.live_index = 0
        return self

    def reset(self) -> PreallocatedKVState:
        """Reset in place and return ``self`` to make alias semantics explicit."""

        return self.reset_()

    def unwritten_is_sentinel(self) -> bool:
        if not 0 <= self.live_index <= self.capacity:
            return False
        return all(
            bool(torch.eq(tensor[:, :, self.live_index :, :], self.sentinel).all())
            for tensor in self.tensors()
        )

    def is_finite(self) -> bool:
        return all(bool(torch.isfinite(tensor).all()) for tensor in self.tensors())


class PreallocatedKVTransformerBackbone(StreamingKVTransformerBackbone):
    """Shared-weight streaming Transformer with preallocated mutable KV.

    This is an inference control, not a differentiable training path.  Calls
    fail closed when gradients are enabled, when capacity differs from 925, or
    when buffer identity/sentinel contracts have changed.
    """

    def __init__(self, *, seed: int = SEED):
        super().__init__(seed=seed)
        if EXPECTED_TEMPORAL_PARAMETERS["streaming_kv_transformer"] != sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        ):
            raise ValueError("preallocated Transformer parameter contract changed")

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise RuntimeError("opaque full-sequence Transformer endpoint is forbidden")

    def streaming_step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise RuntimeError("functional-cat endpoint is not the preallocated control")

    def streaming_stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise RuntimeError("functional-cat endpoint is not the preallocated control")

    def new_preallocated_state(
        self,
        batch_size: int,
        *,
        capacity: int = KV_CAPACITY,
        sentinel: float = KV_SENTINEL,
    ) -> PreallocatedKVState:
        if batch_size < 1:
            raise ValueError("preallocated KV state batch must be positive")
        if capacity != KV_CAPACITY:
            raise ValueError(f"preallocated KV capacity must remain {KV_CAPACITY}")
        if not torch.isfinite(torch.tensor(sentinel, dtype=torch.float64)):
            raise ValueError("preallocated KV sentinel must be finite")
        parameter = next(self.parameters())
        shape = (batch_size, ATTENTION_HEADS, capacity, ATTENTION_HEAD_WIDTH)
        state = PreallocatedKVState(
            tuple(
                (
                    torch.full(shape, sentinel, device=parameter.device, dtype=parameter.dtype),
                    torch.full(shape, sentinel, device=parameter.device, dtype=parameter.dtype),
                )
                for _ in range(LAYERS)
            ),
            batch_size=batch_size,
            capacity=capacity,
            sentinel=float(sentinel),
        )
        self._validate_preallocated_state(state, reference=parameter, require_sentinel=True)
        expected = transformer_kv_expected_bytes(
            batch_size=batch_size,
            observations=capacity,
            element_size=parameter.element_size(),
        )
        if state.capacity_nbytes() != expected:
            raise ValueError("preallocated KV capacity byte formula changed")
        return state

    @staticmethod
    def _validate_preallocated_state(
        state: PreallocatedKVState,
        *,
        reference: torch.Tensor,
        require_sentinel: bool,
    ) -> None:
        if state.capacity != KV_CAPACITY:
            raise ValueError(f"preallocated KV capacity must remain {KV_CAPACITY}")
        if not 0 <= state.live_index <= state.capacity:
            raise ValueError("preallocated KV live index escaped capacity")
        if len(state.layers) != LAYERS or state.batch_size < 1:
            raise ValueError("preallocated KV batch or layer count changed")
        expected = (
            state.batch_size,
            ATTENTION_HEADS,
            state.capacity,
            ATTENTION_HEAD_WIDTH,
        )
        for key, value in state.layers:
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("preallocated KV buffer shape changed")
            if (
                key.device != reference.device
                or value.device != reference.device
                or key.dtype != reference.dtype
                or value.dtype != reference.dtype
            ):
                raise ValueError("preallocated KV buffer device or dtype changed")
        state.storage_identities()
        if require_sentinel and not state.unwritten_is_sentinel():
            raise ValueError("preallocated KV unwritten capacity changed from sentinel")

    def preallocated_step(
        self,
        descriptor: torch.Tensor,
        state: PreallocatedKVState,
    ) -> tuple[torch.Tensor, PreallocatedKVState]:
        if torch.is_grad_enabled():
            raise RuntimeError("preallocated KV is an inference-only mutable control")
        batch = int(descriptor.shape[0]) if descriptor.ndim == 2 else -1
        if tuple(descriptor.shape) != (batch, WIDTH) or batch < 1:
            raise ValueError("preallocated Transformer expects descriptor [batch,256]")
        if not descriptor.is_floating_point() or not bool(torch.isfinite(descriptor).all()):
            raise ValueError("preallocated Transformer descriptor must be finite floating point")
        if batch != state.batch_size:
            raise ValueError("preallocated KV state does not match descriptor batch")
        # Full sentinel-tail audits are episode-boundary correctness checks.
        # Scanning all unused capacity here would add O(capacity) work to the
        # deployment transition whose memory/runtime scaling is under test.
        self._validate_preallocated_state(state, reference=descriptor, require_sentinel=False)
        if state.live_index >= state.capacity:
            raise ValueError("preallocated KV capacity exhausted")

        identities_before = state.storage_identities()
        index = state.live_index
        position = self._position_at(index, device=descriptor.device, dtype=descriptor.dtype)
        current = (descriptor + position).unsqueeze(1)
        for layer, (key_buffer, value_buffer) in zip(
            self.layers.layers,
            state.layers,
            strict=True,
        ):
            normalized = layer.norm1(current)
            query, key, value = self._project_qkv(normalized, layer.self_attn)
            query_heads = self._as_heads(query)
            key_heads = self._as_heads(key)
            value_heads = self._as_heads(value)
            key_buffer[:, :, index : index + 1, :].copy_(key_heads)
            value_buffer[:, :, index : index + 1, :].copy_(value_heads)
            # Only the committed prefix participates; the finite sentinel tail
            # remains physically resident but is outside both attention inputs.
            valid_key = key_buffer[:, :, : index + 1, :]
            valid_value = value_buffer[:, :, : index + 1, :]
            attention = torch.nn.functional.scaled_dot_product_attention(
                query_heads,
                valid_key,
                valid_value,
                dropout_p=0.0,
                is_causal=False,
            )
            attention = attention.transpose(1, 2).reshape(batch, 1, WIDTH)
            attention = layer.self_attn.out_proj(attention)
            current = current + layer.dropout1(attention)
            feedforward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(current))))
            )
            current = current + layer.dropout2(feedforward)

        state.live_index = index + 1
        if state.storage_identities() != identities_before:
            raise ValueError("preallocated KV transition replaced buffer storage")
        return current.squeeze(1), state

    def preallocated_stream(
        self,
        descriptors: torch.Tensor,
        state: PreallocatedKVState | None = None,
    ) -> tuple[torch.Tensor, PreallocatedKVState]:
        if descriptors.ndim != 3 or descriptors.shape[-1] != WIDTH:
            raise ValueError("preallocated Transformer expects [batch,time,256]")
        if descriptors.shape[1] < 1:
            raise ValueError("preallocated Transformer sequence cannot be empty")
        current = self.new_preallocated_state(int(descriptors.shape[0])) if state is None else state
        if current.live_index + int(descriptors.shape[1]) > current.capacity:
            raise ValueError("preallocated KV stream exceeds capacity")
        outputs = []
        for descriptor in descriptors.unbind(dim=1):
            output, returned = self.preallocated_step(descriptor, current)
            if returned is not current:
                raise ValueError("preallocated KV transition broke mutable alias contract")
            outputs.append(output)
        if not current.unwritten_is_sentinel():
            raise ValueError("preallocated KV stream wrote beyond live prefix")
        return torch.stack(outputs, dim=1), current


def preallocated_states_equal(
    first: PreallocatedKVState,
    second: PreallocatedKVState,
    *,
    include_unwritten: bool = True,
) -> bool:
    if (
        first.batch_size != second.batch_size
        or first.capacity != second.capacity
        or first.live_index != second.live_index
        or (include_unwritten and first.sentinel != second.sentinel)
    ):
        return False
    stop = first.capacity if include_unwritten else first.live_index
    return all(
        torch.equal(left[:, :, :stop, :], right[:, :, :stop, :])
        for left, right in zip(first.tensors(), second.tensors(), strict=True)
    )


def preallocated_capacity_expected_bytes(*, batch_size: int = 1, element_size: int = 4) -> int:
    return transformer_kv_expected_bytes(
        batch_size=batch_size,
        observations=KV_CAPACITY,
        element_size=element_size,
    )


def _layer_norm_fp64(value: torch.Tensor, norm: torch.nn.LayerNorm) -> torch.Tensor:
    return F.layer_norm(
        value,
        norm.normalized_shape,
        norm.weight.detach().cpu().double(),
        None if norm.bias is None else norm.bias.detach().cpu().double(),
        norm.eps,
    )


def functional_mamba_fp64_oracle(
    module: torch.nn.Module,
    descriptors: torch.Tensor,
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    """Independent CPU FP64 recurrence for the frozen two-layer Mamba2.

    The implementation intentionally does not call ``functional_mamba2_step``,
    an official Mamba forward/step method, or a fused scan kernel.
    """

    if descriptors.ndim != 3 or descriptors.shape[-1] != WIDTH:
        raise ValueError("Mamba oracle expects [batch,time,256]")
    value_sequence = descriptors.detach().cpu().double()
    batch = int(value_sequence.shape[0])
    states = [
        (
            torch.zeros(batch, 640, 4, dtype=torch.float64),
            torch.zeros(batch, 8, 64, 64, dtype=torch.float64),
        )
        for _ in range(LAYERS)
    ]
    outputs = []
    with torch.no_grad():
        for descriptor in value_sequence.unbind(dim=1):
            current = descriptor
            next_states = []
            for residual_layer, (conv_state, ssm_state) in zip(module.layers, states, strict=True):
                normalized = _layer_norm_fp64(current, residual_layer.norm)
                mamba = residual_layer.mamba
                projected = F.linear(
                    normalized,
                    mamba.in_proj.weight.detach().cpu().double(),
                    None,
                )
                z, xbc_input, dt_raw = torch.split(projected, (512, 640, 8), dim=-1)
                next_conv = torch.cat((conv_state[..., 1:], xbc_input.unsqueeze(-1)), dim=-1)
                convolved = torch.sum(
                    next_conv * mamba.conv1d.weight.detach().cpu().double().squeeze(1),
                    dim=-1,
                )
                convolved = F.silu(convolved + mamba.conv1d.bias.detach().cpu().double())
                x_flat, b_flat, c_flat = torch.split(convolved, (512, 64, 64), dim=-1)
                x = x_flat.reshape(batch, 8, 64)
                b_group = b_flat.reshape(batch, 64)
                c_group = c_flat.reshape(batch, 64)
                a = -torch.exp(mamba.A_log.detach().cpu().double())
                dt = F.softplus(dt_raw + mamba.dt_bias.detach().cpu().double())
                alpha = torch.exp(dt * a)
                state_input = torch.einsum("bh,bn,bhp->bhpn", dt, b_group, x)
                next_ssm = ssm_state * alpha[:, :, None, None] + state_input
                y = torch.einsum("bhpn,bn->bhp", next_ssm, c_group)
                y = y + mamba.D.detach().cpu().double().reshape(1, 8, 1) * x
                y = y.reshape(batch, 512)
                # Frozen Mamba2 uses norm_before_gate=False: gate first, then
                # RMS normalization.  This is the documented reference
                # equation, written independently from the Triton operator.
                y = y * F.silu(z)
                rms = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + mamba.norm.eps)
                y = y * rms * mamba.norm.weight.detach().cpu().double()
                update = F.linear(y, mamba.out_proj.weight.detach().cpu().double(), None)
                current = current + update
                next_states.append((next_conv, next_ssm))
            states = next_states
            outputs.append(current)
    return torch.stack(outputs, dim=1), tuple(states)


def persistent_gru_fp64_oracle(
    module: torch.nn.Module,
    descriptors: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Independent CPU FP64 recurrence for PyTorch's reset-after GRU."""

    if descriptors.ndim != 3 or descriptors.shape[-1] != WIDTH:
        raise ValueError("GRU oracle expects [batch,time,256]")
    sequence = descriptors.detach().cpu().double()
    batch = int(sequence.shape[0])
    hidden = [torch.zeros(batch, WIDTH, dtype=torch.float64) for _ in range(LAYERS)]
    outputs = []
    with torch.no_grad():
        for descriptor in sequence.unbind(dim=1):
            current = descriptor
            for index, residual_layer in enumerate(module.layers):
                normalized = _layer_norm_fp64(current, residual_layer.norm)
                gru = residual_layer.gru
                input_gates = F.linear(
                    normalized,
                    gru.weight_ih_l0.detach().cpu().double(),
                    gru.bias_ih_l0.detach().cpu().double(),
                )
                hidden_gates = F.linear(
                    hidden[index],
                    gru.weight_hh_l0.detach().cpu().double(),
                    gru.bias_hh_l0.detach().cpu().double(),
                )
                ir, iz, inn = input_gates.chunk(3, dim=-1)
                hr, hz, hn = hidden_gates.chunk(3, dim=-1)
                reset = torch.sigmoid(ir + hr)
                update = torch.sigmoid(iz + hz)
                candidate = torch.tanh(inn + reset * hn)
                next_hidden = (1.0 - update) * candidate + update * hidden[index]
                hidden[index] = next_hidden
                current = current + next_hidden
            outputs.append(current)
    return torch.stack(outputs, dim=1), tuple(value.unsqueeze(0) for value in hidden)


def streaming_transformer_fp64_oracle(
    module: torch.nn.Module,
    descriptors: torch.Tensor,
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    """Independent CPU FP64 online causal Transformer recurrence.

    Attention is expressed as explicit matmul/scale/softmax rather than SDPA;
    K/V history is accumulated as Python lists and stacked, not by the target
    functional-cat or preallocated state implementation.
    """

    if descriptors.ndim != 3 or descriptors.shape[-1] != WIDTH:
        raise ValueError("Transformer oracle expects [batch,time,256]")
    sequence = descriptors.detach().cpu().double()
    batch = int(sequence.shape[0])
    key_history: list[list[torch.Tensor]] = [[] for _ in range(LAYERS)]
    value_history: list[list[torch.Tensor]] = [[] for _ in range(LAYERS)]
    outputs = []
    with torch.no_grad():
        for index, descriptor in enumerate(sequence.unbind(dim=1)):
            position = torch.empty(WIDTH, dtype=torch.float64)
            divisor = torch.exp(
                torch.arange(0, WIDTH, 2, dtype=torch.float64) * (-math.log(10_000.0) / WIDTH)
            )
            position[0::2] = torch.sin(float(index) * divisor)
            position[1::2] = torch.cos(float(index) * divisor)
            current = descriptor + position
            for layer_index, layer in enumerate(module.layers.layers):
                normalized = _layer_norm_fp64(current, layer.norm1)
                packed = F.linear(
                    normalized,
                    layer.self_attn.in_proj_weight.detach().cpu().double(),
                    None
                    if layer.self_attn.in_proj_bias is None
                    else layer.self_attn.in_proj_bias.detach().cpu().double(),
                )
                query, key, value = packed.chunk(3, dim=-1)
                query = query.reshape(batch, ATTENTION_HEADS, ATTENTION_HEAD_WIDTH)
                key = key.reshape(batch, ATTENTION_HEADS, ATTENTION_HEAD_WIDTH)
                value = value.reshape(batch, ATTENTION_HEADS, ATTENTION_HEAD_WIDTH)
                key_history[layer_index].append(key)
                value_history[layer_index].append(value)
                keys = torch.stack(key_history[layer_index], dim=2)
                values = torch.stack(value_history[layer_index], dim=2)
                scores = torch.einsum("bhd,bhtd->bht", query, keys) / math.sqrt(
                    ATTENTION_HEAD_WIDTH
                )
                probabilities = torch.softmax(scores, dim=-1)
                attended = torch.einsum("bht,bhtd->bhd", probabilities, values)
                attended = attended.reshape(batch, WIDTH)
                attended = F.linear(
                    attended,
                    layer.self_attn.out_proj.weight.detach().cpu().double(),
                    None
                    if layer.self_attn.out_proj.bias is None
                    else layer.self_attn.out_proj.bias.detach().cpu().double(),
                )
                current = current + attended
                normalized_ff = _layer_norm_fp64(current, layer.norm2)
                feedforward = F.linear(
                    F.gelu(
                        F.linear(
                            normalized_ff,
                            layer.linear1.weight.detach().cpu().double(),
                            layer.linear1.bias.detach().cpu().double(),
                        )
                    ),
                    layer.linear2.weight.detach().cpu().double(),
                    layer.linear2.bias.detach().cpu().double(),
                )
                current = current + feedforward
            outputs.append(current)
    states = tuple(
        (
            torch.stack(key_history[index], dim=2),
            torch.stack(value_history[index], dim=2),
        )
        for index in range(LAYERS)
    )
    return torch.stack(outputs, dim=1), states
