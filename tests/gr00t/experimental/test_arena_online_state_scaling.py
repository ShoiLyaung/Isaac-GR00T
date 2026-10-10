from __future__ import annotations

import copy
import inspect

from gr00t.experimental.arena_codec_streaming_backbone import SEED, WIDTH
from gr00t.experimental.arena_functional_mamba_transition import FunctionalMamba2Backbone
from gr00t.experimental.arena_functional_streaming_controls import (
    PersistentGRUBackbone,
    StreamingKVTransformerBackbone,
    transformer_kv_expected_bytes,
)
from gr00t.experimental.arena_online_state_scaling import (
    KV_CAPACITY,
    KV_SENTINEL,
    PreallocatedKVState,
    PreallocatedKVTransformerBackbone,
    functional_mamba_fp64_oracle,
    persistent_gru_fp64_oracle,
    preallocated_capacity_expected_bytes,
    preallocated_states_equal,
    streaming_transformer_fp64_oracle,
)
import pytest
import torch


def _models() -> tuple[StreamingKVTransformerBackbone, PreallocatedKVTransformerBackbone]:
    functional = StreamingKVTransformerBackbone(seed=SEED).eval()
    preallocated = PreallocatedKVTransformerBackbone(seed=SEED).eval()
    assert functional.state_dict().keys() == preallocated.state_dict().keys()
    assert all(
        torch.equal(value, preallocated.state_dict()[name])
        for name, value in functional.state_dict().items()
    )
    return functional, preallocated


def test_preallocated_matches_functional_cat_and_preserves_sentinel_tail() -> None:
    functional, preallocated = _models()
    descriptors = torch.randn(1, 17, WIDTH, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        expected, expected_state = functional.streaming_stream(descriptors)
        state = preallocated.new_preallocated_state(1)
        identities = state.storage_identities()
        actual, returned = preallocated.preallocated_stream(descriptors, state)

    assert returned is state
    assert state.storage_identities() == identities
    assert torch.equal(actual, expected)
    assert state.live_index == 17
    assert state.logical_nbytes() == transformer_kv_expected_bytes(batch_size=1, observations=17)
    assert state.capacity_nbytes() == preallocated_capacity_expected_bytes()
    assert state.capacity_nbytes() == transformer_kv_expected_bytes(
        batch_size=1, observations=KV_CAPACITY
    )
    assert state.unwritten_is_sentinel()
    assert state.is_finite()
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        state.layers, expected_state.layers, strict=True
    ):
        assert torch.equal(actual_key[:, :, :17], expected_key)
        assert torch.equal(actual_value[:, :, :17], expected_value)
        assert bool(torch.eq(actual_key[:, :, 17:], KV_SENTINEL).all())
        assert bool(torch.eq(actual_value[:, :, 17:], KV_SENTINEL).all())


def test_unwritten_capacity_cannot_affect_valid_prefix_attention() -> None:
    _, module = _models()
    descriptors = torch.randn(1, 9, WIDTH, generator=torch.Generator().manual_seed(11))
    with torch.inference_mode():
        prefix, state = module.preallocated_stream(descriptors[:, :8])
        del prefix
        poisoned = state.clone()
        poisoned.sentinel = 12_345.0
        for tensor in poisoned.tensors():
            tensor[:, :, poisoned.live_index :, :].fill_(poisoned.sentinel)
        state_out, returned = module.preallocated_step(descriptors[:, 8], state)
        poisoned_out, poisoned_returned = module.preallocated_step(descriptors[:, 8], poisoned)

    assert returned is state
    assert poisoned_returned is poisoned
    assert torch.equal(state_out, poisoned_out)
    assert state.live_index == poisoned.live_index == 9
    # Poison after the valid prefix is intentionally retained: equality of the
    # outputs proves those resident bytes did not enter attention.
    assert not preallocated_states_equal(state, poisoned, include_unwritten=True)
    assert preallocated_states_equal(state, poisoned, include_unwritten=False)


def test_reset_is_in_place_restores_sentinel_and_replays_exactly() -> None:
    _, module = _models()
    descriptors = torch.randn(1, 6, WIDTH, generator=torch.Generator().manual_seed(13))
    with torch.inference_mode():
        first, state = module.preallocated_stream(descriptors)
        identities = state.storage_identities()
        returned = state.reset()
        second, second_state = module.preallocated_stream(descriptors, state)

    assert returned is state is second_state
    assert state.storage_identities() == identities
    assert state.live_index == 6
    assert torch.equal(first, second)
    assert state.unwritten_is_sentinel()


def test_mutable_alias_and_capacity_fail_closed() -> None:
    _, module = _models()
    with pytest.raises(ValueError, match="capacity must remain 925"):
        module.new_preallocated_state(1, capacity=924)
    with pytest.raises(ValueError, match="sentinel must be finite"):
        module.new_preallocated_state(1, sentinel=float("nan"))

    state = module.new_preallocated_state(1)
    shared = state.layers[0][0]
    aliased = PreallocatedKVState(
        ((shared, shared), state.layers[1]),
        batch_size=1,
        capacity=KV_CAPACITY,
    )
    with pytest.raises(ValueError, match="aliased storage"):
        aliased.storage_identities()

    broken = state.clone()
    broken.layers[0][0][:, :, 4].zero_()
    with torch.inference_mode(), pytest.raises(ValueError, match="unwritten capacity"):
        module.preallocated_step(torch.randn(1, WIDTH), broken)

    full = state.clone()
    full.live_index = KV_CAPACITY
    for tensor in full.tensors():
        tensor.zero_()
    with torch.inference_mode(), pytest.raises(ValueError, match="capacity exhausted"):
        module.preallocated_step(torch.randn(1, WIDTH), full)


def test_inference_only_and_forbidden_noncontrol_endpoints() -> None:
    _, module = _models()
    state = module.new_preallocated_state(1)
    with pytest.raises(RuntimeError, match="inference-only"):
        module.preallocated_step(torch.randn(1, WIDTH), state)
    with pytest.raises(RuntimeError, match="full-sequence.*forbidden"):
        module(torch.randn(1, 2, WIDTH))
    with pytest.raises(RuntimeError, match="not the preallocated control"):
        module.streaming_stream(torch.randn(1, 2, WIDTH))
    with pytest.raises(RuntimeError, match="not the preallocated control"):
        module.streaming_step(torch.randn(1, WIDTH), copy.deepcopy(object()))


def test_clone_has_independent_storage_and_live_prefix_equality() -> None:
    _, module = _models()
    state = module.new_preallocated_state(2)
    with torch.inference_mode():
        module.preallocated_stream(torch.randn(2, 3, WIDTH), state)
    clone = state.clone()
    assert preallocated_states_equal(state, clone)
    assert set(state.storage_identities()).isdisjoint(clone.storage_identities())
    assert clone.capacity_nbytes() == 2 * preallocated_capacity_expected_bytes()
    clone.layers[0][0][:, :, 0].add_(1)
    assert not preallocated_states_equal(state, clone, include_unwritten=False)


def test_gru_fp64_oracle_is_independent_and_matches_online_target() -> None:
    module = PersistentGRUBackbone(seed=SEED).eval()
    descriptors = torch.randn(1, 11, WIDTH, generator=torch.Generator().manual_seed(17))
    with torch.inference_mode():
        target, target_state = module.persistent_stream(descriptors)
    oracle, oracle_state = persistent_gru_fp64_oracle(module, descriptors)
    torch.testing.assert_close(target.double(), oracle, rtol=0, atol=1e-5)
    for target_hidden, expected_hidden in zip(target_state.hidden, oracle_state, strict=True):
        torch.testing.assert_close(target_hidden.double(), expected_hidden, rtol=0, atol=1e-5)
    source = inspect.getsource(persistent_gru_fp64_oracle)
    assert ".gru(" not in source
    assert "forward_explicit" not in source


def test_transformer_fp64_oracle_uses_no_sdpa_or_full_endpoint() -> None:
    functional, _ = _models()
    descriptors = torch.randn(1, 13, WIDTH, generator=torch.Generator().manual_seed(19))
    with torch.inference_mode():
        target, target_state = functional.streaming_stream(descriptors)
    oracle, oracle_state = streaming_transformer_fp64_oracle(functional, descriptors)
    torch.testing.assert_close(target.double(), oracle, rtol=0, atol=1e-5)
    for target_pair, expected_pair in zip(target_state.layers, oracle_state, strict=True):
        for target_tensor, expected_tensor in zip(target_pair, expected_pair, strict=True):
            torch.testing.assert_close(target_tensor.double(), expected_tensor, rtol=0, atol=1e-5)
    source = inspect.getsource(streaming_transformer_fp64_oracle)
    assert "scaled_dot_product_attention" not in source
    assert ".forward(" not in source


def test_mamba_fp64_oracle_is_cpu_finite_and_excludes_target_transition() -> None:
    module = FunctionalMamba2Backbone(seed=SEED).eval()
    descriptors = torch.randn(1, 3, WIDTH, generator=torch.Generator().manual_seed(23))
    output, state = functional_mamba_fp64_oracle(module, descriptors)
    assert output.shape == (1, 3, WIDTH)
    assert output.dtype == torch.float64
    assert bool(torch.isfinite(output).all())
    assert len(state) == 2
    assert all(
        tuple(conv.shape) == (1, 640, 4)
        and tuple(ssm.shape) == (1, 8, 64, 64)
        and bool(torch.isfinite(conv).all())
        and bool(torch.isfinite(ssm).all())
        for conv, ssm in state
    )
    source = inspect.getsource(functional_mamba_fp64_oracle)
    assert "functional_mamba2_step(" not in source
    assert ".functional_step" not in source
    assert ".mamba.step" not in source
