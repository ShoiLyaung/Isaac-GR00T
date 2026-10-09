from __future__ import annotations

import pytest
import torch

from gr00t.experimental.arena_codec_streaming_backbone import (
    SEED,
    WIDTH,
    CausalTransformerBackbone,
    trainable_parameter_count,
)
from gr00t.experimental.arena_functional_streaming_controls import (
    EXPECTED_TEMPORAL_PARAMETERS,
    CurrentDescriptorState,
    PersistentGRUBackbone,
    StreamingCurrentDescriptorBackbone,
    StreamingKVState,
    StreamingKVTransformerBackbone,
    states_equal,
    transformer_kv_expected_bytes,
    unique_storage_state_bytes,
)


def test_streaming_kv_reuses_exact_full_parameter_set_and_matches_t128() -> None:
    full = CausalTransformerBackbone(seed=SEED).eval()
    streaming = StreamingKVTransformerBackbone(seed=SEED).eval()
    assert full.state_dict().keys() == streaming.state_dict().keys()
    assert all(
        torch.equal(full.state_dict()[name], streaming.state_dict()[name])
        for name in full.state_dict()
    )
    assert trainable_parameter_count(streaming) == EXPECTED_TEMPORAL_PARAMETERS[
        "streaming_kv_transformer"
    ]
    assert len(tuple(streaming.parameters())) == len(tuple(full.parameters()))

    descriptors = torch.randn(1, 128, WIDTH)
    with torch.inference_mode():
        expected = streaming(descriptors)
        actual, state = streaming.streaming_stream(descriptors)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
    assert state.observations == 128
    assert unique_storage_state_bytes(state) == transformer_kv_expected_bytes(
        batch_size=1,
        observations=128,
    )


def test_streaming_kv_is_chunk_exact_immutable_and_linear_through_t925() -> None:
    module = StreamingKVTransformerBackbone(seed=SEED).eval()
    descriptors = torch.randn(1, 925, WIDTH)
    with torch.inference_mode():
        continuous, continuous_state = module.streaming_stream(descriptors)
        first, state = module.streaming_stream(descriptors[:, :128])
        bytes_t128 = unique_storage_state_bytes(state)
        state_before = state.clone()
        second, state = module.streaming_stream(descriptors[:, 128:], state)
        bytes_t925 = unique_storage_state_bytes(state)
    assert states_equal(state_before, continuous_state) is False
    assert states_equal(state_before, state_before.clone())
    assert torch.equal(continuous, torch.cat((first, second), dim=1))
    assert states_equal(state, continuous_state)
    assert bytes_t128 == transformer_kv_expected_bytes(batch_size=1, observations=128)
    assert bytes_t925 == transformer_kv_expected_bytes(batch_size=1, observations=925)
    assert bytes_t925 / bytes_t128 == pytest.approx(925 / 128)

    second_step_input = state_before.clone()
    with torch.inference_mode():
        _, second_step_output = module.streaming_step(
            descriptors[:, 128],
            second_step_input,
        )
    assert states_equal(second_step_input, state_before)
    input_storages = {tensor.untyped_storage().data_ptr() for pair in state_before.layers for tensor in pair}
    output_storages = {
        tensor.untyped_storage().data_ptr() for pair in second_step_output.layers for tensor in pair
    }
    assert not input_storages & output_storages


def test_unique_storage_bytes_rejects_any_kv_alias() -> None:
    shared = torch.zeros(1, 8, 4, 32)
    state = StreamingKVState(((shared, shared), (torch.ones_like(shared), torch.randn_like(shared))), 1, 4)
    with pytest.raises(ValueError, match="aliased storage"):
        unique_storage_state_bytes(state)


def test_persistent_gru_full_step_exact_and_fixed_state_storage() -> None:
    module = PersistentGRUBackbone(seed=SEED).eval()
    assert trainable_parameter_count(module) == EXPECTED_TEMPORAL_PARAMETERS["persistent_gru"]
    descriptors = torch.randn(1, 128, WIDTH)
    with torch.inference_mode():
        expected = module.forward_explicit(descriptors)
        actual, state = module.persistent_stream(descriptors)
    assert torch.equal(actual, expected)
    assert state.observations == 128
    assert unique_storage_state_bytes(state) == 2 * WIDTH * torch.tensor([], dtype=torch.float32).element_size()

    input_state = state.clone()
    before = input_state.clone()
    with torch.inference_mode():
        _, output_state = module.persistent_step(torch.randn(1, WIDTH), input_state)
    assert states_equal(input_state, before)
    assert unique_storage_state_bytes(output_state) == unique_storage_state_bytes(input_state)
    assert not {
        tensor.untyped_storage().data_ptr() for tensor in input_state.hidden
    } & {tensor.untyped_storage().data_ptr() for tensor in output_state.hidden}


def test_current_descriptor_is_parameter_free_and_tracks_episode_reset() -> None:
    module = StreamingCurrentDescriptorBackbone()
    descriptors = torch.randn(2, 7, WIDTH)
    output, state = module.streaming_stream(descriptors)
    assert torch.equal(output, descriptors)
    assert trainable_parameter_count(module) == EXPECTED_TEMPORAL_PARAMETERS["current_descriptor"]
    assert state == CurrentDescriptorState(batch_size=2, observations=7)
    assert state.reset() == CurrentDescriptorState(batch_size=2, observations=0)
    assert unique_storage_state_bytes(state) == 0

