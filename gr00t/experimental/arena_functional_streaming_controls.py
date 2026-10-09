"""Fixed-weight online controls for the Arena functional Mamba comparison.

The controls consume the exact frame descriptors produced by
``ArenaCodecSpatialTokenizer``.  They do not import or construct the native
VLM, DiT, action head, optimizer, or dataset stack.  Every online transition
returns fresh state storage so timing includes the state/KV materialization
specified by the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.experimental.arena_codec_streaming_backbone import (
    LAYERS,
    SEED,
    WIDTH,
    CausalGRUBackbone,
    CausalTransformerBackbone,
    CurrentDescriptorBackbone,
    _validate_descriptor,
    _validate_descriptor_sequence,
    trainable_parameter_count,
)
from gr00t.experimental.arena_functional_mamba_transition import FunctionalMamba2State


ATTENTION_HEADS = 8
ATTENTION_HEAD_WIDTH = WIDTH // ATTENTION_HEADS
FEEDFORWARD_WIDTH = 320
EXPECTED_TEMPORAL_PARAMETERS = {
    "functional_mamba": 864_560,
    "persistent_gru": 790_528,
    "streaming_kv_transformer": 857_216,
    "current_descriptor": 0,
}


@dataclass(frozen=True)
class PersistentGRUState:
    """Immutable persistent hidden state for the two residual GRU layers."""

    hidden: tuple[torch.Tensor, ...]
    batch_size: int
    observations: int = 0

    def clone(self) -> PersistentGRUState:
        return PersistentGRUState(
            tuple(tensor.clone() for tensor in self.hidden),
            self.batch_size,
            self.observations,
        )

    def reset(self) -> PersistentGRUState:
        return PersistentGRUState(
            tuple(torch.zeros_like(tensor) for tensor in self.hidden),
            self.batch_size,
            0,
        )

    def tensor_nbytes(self) -> int:
        return unique_storage_state_bytes(self)


class PersistentGRUBackbone(CausalGRUBackbone):
    """The frozen GRU weights with fresh persistent state on every step."""

    def __init__(self, *, seed: int = SEED):
        super().__init__(seed=seed)
        if len(self.layers) != LAYERS:
            raise ValueError("Arena persistent GRU must contain exactly two layers")
        if trainable_parameter_count(self) != EXPECTED_TEMPORAL_PARAMETERS["persistent_gru"]:
            raise ValueError("Arena persistent GRU parameter contract changed")

    def new_persistent_state(self, batch_size: int) -> PersistentGRUState:
        if batch_size < 1:
            raise ValueError("persistent GRU state batch must be positive")
        parameter = next(self.parameters())
        return PersistentGRUState(
            tuple(
                torch.zeros(
                    1,
                    batch_size,
                    WIDTH,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                for _ in self.layers
            ),
            batch_size,
        )

    def persistent_step(
        self,
        descriptor: torch.Tensor,
        state: PersistentGRUState,
    ) -> tuple[torch.Tensor, PersistentGRUState]:
        batch = _validate_descriptor(descriptor)
        if batch != state.batch_size or len(state.hidden) != len(self.layers):
            raise ValueError("persistent GRU state does not match batch or layer count")
        unique_storage_state_bytes(state)
        value = descriptor.unsqueeze(1)
        updated_hidden = []
        for layer, hidden in zip(self.layers, state.hidden, strict=True):
            expected = (1, batch, WIDTH)
            if tuple(hidden.shape) != expected:
                raise ValueError("persistent GRU hidden state shape changed")
            if hidden.device != value.device or hidden.dtype != value.dtype:
                raise ValueError("persistent GRU hidden state device or dtype changed")
            update, next_hidden = layer.gru(layer.norm(value), hidden)
            value = value + update
            updated_hidden.append(next_hidden)
        next_state = PersistentGRUState(
            tuple(updated_hidden),
            state.batch_size,
            state.observations + 1,
        )
        unique_storage_state_bytes(next_state)
        return value.squeeze(1), next_state

    def persistent_stream(
        self,
        descriptors: torch.Tensor,
        state: PersistentGRUState | None = None,
    ) -> tuple[torch.Tensor, PersistentGRUState]:
        batch, _ = _validate_descriptor_sequence(descriptors)
        current = self.new_persistent_state(batch) if state is None else state
        outputs = []
        for descriptor in descriptors.unbind(dim=1):
            output, current = self.persistent_step(descriptor, current)
            outputs.append(output)
        return torch.stack(outputs, dim=1), current


KVPair: TypeAlias = tuple[torch.Tensor | None, torch.Tensor | None]


@dataclass(frozen=True)
class StreamingKVState:
    """Per-layer projected keys and values, grown by functional concatenation."""

    layers: tuple[KVPair, ...]
    batch_size: int
    observations: int = 0

    def clone(self) -> StreamingKVState:
        return StreamingKVState(
            tuple(
                (
                    None if key is None else key.clone(),
                    None if value is None else value.clone(),
                )
                for key, value in self.layers
            ),
            self.batch_size,
            self.observations,
        )

    def reset(self) -> StreamingKVState:
        return StreamingKVState(
            tuple((None, None) for _ in self.layers),
            self.batch_size,
            0,
        )

    def tensor_nbytes(self) -> int:
        return unique_storage_state_bytes(self)


class StreamingKVTransformerBackbone(CausalTransformerBackbone):
    """Causal Transformer step path that directly reuses full-path parameters."""

    def __init__(self, *, seed: int = SEED):
        super().__init__(seed=seed)
        if len(self.layers.layers) != LAYERS:
            raise ValueError("Arena streaming KV Transformer must contain exactly two layers")
        for layer in self.layers.layers:
            attention = layer.self_attn
            if (
                attention.embed_dim != WIDTH
                or attention.num_heads != ATTENTION_HEADS
                or attention.head_dim != ATTENTION_HEAD_WIDTH
                or layer.linear1.out_features != FEEDFORWARD_WIDTH
                or not layer.norm_first
                or attention.dropout != 0.0
            ):
                raise ValueError("Arena streaming KV Transformer architecture changed")
        if trainable_parameter_count(self) != EXPECTED_TEMPORAL_PARAMETERS[
            "streaming_kv_transformer"
        ]:
            raise ValueError("Arena streaming KV Transformer parameter contract changed")

    def new_kv_state(self, batch_size: int) -> StreamingKVState:
        if batch_size < 1:
            raise ValueError("streaming KV state batch must be positive")
        return StreamingKVState(tuple((None, None) for _ in self.layers.layers), batch_size)

    @staticmethod
    def _project_qkv(
        normalized: torch.Tensor,
        attention: nn.MultiheadAttention,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q/K/V separately from the existing packed parameter tensor.

        Separate calls keep the persisted K and V in independent storages.  The
        weight slices are views of the original parameter and introduce no new
        parameter or copied trainable tensor.
        """

        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        if weight is None or tuple(weight.shape) != (3 * WIDTH, WIDTH):
            raise ValueError("streaming KV Transformer packed QKV weight changed")
        query = F.linear(
            normalized,
            weight[:WIDTH],
            None if bias is None else bias[:WIDTH],
        )
        key = F.linear(
            normalized,
            weight[WIDTH : 2 * WIDTH],
            None if bias is None else bias[WIDTH : 2 * WIDTH],
        )
        value = F.linear(
            normalized,
            weight[2 * WIDTH :],
            None if bias is None else bias[2 * WIDTH :],
        )
        return query, key, value

    @staticmethod
    def _as_heads(value: torch.Tensor) -> torch.Tensor:
        batch, time, width = value.shape
        if width != WIDTH:
            raise ValueError("streaming KV projection width changed")
        return value.reshape(batch, time, ATTENTION_HEADS, ATTENTION_HEAD_WIDTH).transpose(1, 2)

    @staticmethod
    def _validate_pair(
        pair: KVPair,
        *,
        batch: int,
        observations: int,
        reference: torch.Tensor,
    ) -> None:
        key, value = pair
        if observations == 0:
            if key is not None or value is not None:
                raise ValueError("empty streaming KV state unexpectedly contains a cache")
            return
        expected = (batch, ATTENTION_HEADS, observations, ATTENTION_HEAD_WIDTH)
        if key is None or value is None or tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError("streaming KV cache shape does not match observation count")
        if (
            key.device != reference.device
            or value.device != reference.device
            or key.dtype != reference.dtype
            or value.dtype != reference.dtype
        ):
            raise ValueError("streaming KV cache device or dtype changed")

    def streaming_step(
        self,
        descriptor: torch.Tensor,
        state: StreamingKVState,
    ) -> tuple[torch.Tensor, StreamingKVState]:
        batch = _validate_descriptor(descriptor)
        if batch != state.batch_size or len(state.layers) != len(self.layers.layers):
            raise ValueError("streaming KV state does not match batch or layer count")
        unique_storage_state_bytes(state)
        position = self.position(
            state.observations + 1,
            WIDTH,
            descriptor.device,
            descriptor.dtype,
        )[-1]
        current = (descriptor + position).unsqueeze(1)
        updated_layers = []
        for layer, pair in zip(self.layers.layers, state.layers, strict=True):
            self._validate_pair(
                pair,
                batch=batch,
                observations=state.observations,
                reference=current,
            )
            normalized = layer.norm1(current)
            query, key, value = self._project_qkv(normalized, layer.self_attn)
            query_heads = self._as_heads(query)
            key_heads = self._as_heads(key)
            value_heads = self._as_heads(value)
            previous_key, previous_value = pair
            cache_key = (
                key_heads
                if previous_key is None
                else torch.cat((previous_key, key_heads), dim=2)
            )
            cache_value = (
                value_heads
                if previous_value is None
                else torch.cat((previous_value, value_heads), dim=2)
            )
            attention = F.scaled_dot_product_attention(
                query_heads,
                cache_key,
                cache_value,
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
            updated_layers.append((cache_key, cache_value))
        next_state = StreamingKVState(
            tuple(updated_layers),
            state.batch_size,
            state.observations + 1,
        )
        unique_storage_state_bytes(next_state)
        return current.squeeze(1), next_state

    def streaming_stream(
        self,
        descriptors: torch.Tensor,
        state: StreamingKVState | None = None,
    ) -> tuple[torch.Tensor, StreamingKVState]:
        batch, _ = _validate_descriptor_sequence(descriptors)
        current = self.new_kv_state(batch) if state is None else state
        outputs = []
        for descriptor in descriptors.unbind(dim=1):
            output, current = self.streaming_step(descriptor, current)
            outputs.append(output)
        return torch.stack(outputs, dim=1), current


@dataclass(frozen=True)
class CurrentDescriptorState:
    batch_size: int
    observations: int = 0

    def clone(self) -> CurrentDescriptorState:
        return CurrentDescriptorState(self.batch_size, self.observations)

    def reset(self) -> CurrentDescriptorState:
        return CurrentDescriptorState(self.batch_size, 0)

    def tensor_nbytes(self) -> int:
        return 0


class StreamingCurrentDescriptorBackbone(CurrentDescriptorBackbone):
    """Parameter-free tokenizer-output lower bound with the same step API."""

    def new_state(self, batch_size: int) -> CurrentDescriptorState:
        if batch_size < 1:
            raise ValueError("current descriptor state batch must be positive")
        return CurrentDescriptorState(batch_size)

    def streaming_step(
        self,
        descriptor: torch.Tensor,
        state: CurrentDescriptorState,
    ) -> tuple[torch.Tensor, CurrentDescriptorState]:
        batch = _validate_descriptor(descriptor)
        if batch != state.batch_size:
            raise ValueError("current descriptor state does not match batch")
        return descriptor, CurrentDescriptorState(state.batch_size, state.observations + 1)

    def streaming_stream(
        self,
        descriptors: torch.Tensor,
        state: CurrentDescriptorState | None = None,
    ) -> tuple[torch.Tensor, CurrentDescriptorState]:
        batch, _ = _validate_descriptor_sequence(descriptors)
        current = self.new_state(batch) if state is None else state
        outputs = []
        for descriptor in descriptors.unbind(dim=1):
            output, current = self.streaming_step(descriptor, current)
            outputs.append(output)
        return torch.stack(outputs, dim=1), current


StreamingState: TypeAlias = (
    FunctionalMamba2State | PersistentGRUState | StreamingKVState | CurrentDescriptorState
)


def state_tensors(state: StreamingState) -> tuple[torch.Tensor, ...]:
    """Return every logical state tensor without silently deduplicating aliases."""

    if isinstance(state, FunctionalMamba2State):
        return tuple(tensor for pair in state.layers for tensor in pair)
    if isinstance(state, PersistentGRUState):
        return state.hidden
    if isinstance(state, StreamingKVState):
        return tuple(tensor for pair in state.layers for tensor in pair if tensor is not None)
    if isinstance(state, CurrentDescriptorState):
        return ()
    raise TypeError(f"unsupported streaming state: {type(state).__qualname__}")


def storage_identity(value: torch.Tensor) -> tuple[str, int, int]:
    storage = value.untyped_storage()
    return str(value.device), storage.data_ptr(), storage.nbytes()


def unique_storage_state_bytes(state: StreamingState) -> int:
    """Count unique state storages and reject every internal alias."""

    identities = [storage_identity(tensor) for tensor in state_tensors(state)]
    if len(identities) != len(set(identities)):
        raise ValueError("streaming state contains repeated or aliased storage")
    return sum(identity[2] for identity in identities)


def state_is_finite(state: StreamingState) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in state_tensors(state))


def states_equal(first: StreamingState, second: StreamingState) -> bool:
    if type(first) is not type(second):
        return False
    if first.batch_size != second.batch_size or first.observations != second.observations:
        return False
    first_tensors = state_tensors(first)
    second_tensors = state_tensors(second)
    return len(first_tensors) == len(second_tensors) and all(
        torch.equal(left, right)
        for left, right in zip(first_tensors, second_tensors, strict=True)
    )


def relative_parameter_difference(reference: int, candidate: int) -> float:
    if reference <= 0 or candidate < 0:
        raise ValueError("parameter counts must be nonnegative with a positive reference")
    return abs(candidate - reference) / reference


def validate_frozen_parameter_counts(modules: dict[str, nn.Module]) -> dict[str, int]:
    if set(modules) != set(EXPECTED_TEMPORAL_PARAMETERS):
        raise ValueError("streaming control condition set changed")
    counts = {name: trainable_parameter_count(module) for name, module in modules.items()}
    if counts != EXPECTED_TEMPORAL_PARAMETERS:
        raise ValueError(f"streaming control parameter counts changed: {counts}")
    reference = counts["functional_mamba"]
    if any(
        relative_parameter_difference(reference, counts[name]) > 0.15
        for name in ("persistent_gru", "streaming_kv_transformer")
    ):
        raise ValueError("temporal parameter count escaped the 15 percent matching band")
    return counts


def transformer_kv_expected_bytes(
    *,
    batch_size: int,
    observations: int,
    element_size: int = 4,
) -> int:
    if batch_size < 1 or observations < 0 or element_size < 1:
        raise ValueError("invalid Transformer KV byte-count arguments")
    return LAYERS * 2 * batch_size * observations * WIDTH * element_size

