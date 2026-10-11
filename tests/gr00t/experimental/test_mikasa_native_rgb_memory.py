from __future__ import annotations

import copy

from gr00t.experimental.arena_codec_streaming_backbone import trainable_parameter_count
from gr00t.experimental.mikasa_native_rgb_memory import (
    EXPECTED_TEMPORAL_PARAMETERS,
    GridCameraEncoder,
    MemoryBatch,
    MikasaNativeRGBMemoryModel,
    NativeRGBTemporalFrontend,
    apply_history_condition,
    build_temporal_backbone,
    post_as_temporal_grid,
)
import pytest
import torch


def _grids(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    temporal = torch.rand(batch, 130, 2, 8, 8, 9, generator=generator)
    post = torch.rand(batch, 2, 8, 8, 3, generator=generator)
    proprio = torch.rand(batch, 7, generator=generator)
    return temporal, post, proprio


def test_six_conditions_have_exact_fixed_length_visibility_masks() -> None:
    temporal, post, _ = _grids(1)
    replacement = post_as_temporal_grid(post).unsqueeze(1).expand_as(temporal)
    current = apply_history_condition(temporal, post, "mamba_current")
    one = apply_history_condition(temporal, post, "mamba_one_frame")
    last = apply_history_condition(temporal, post, "mamba_last_segment")
    for condition in ("mamba_full", "gru_full", "transformer_full"):
        torch.testing.assert_close(
            apply_history_condition(temporal, post, condition), temporal, rtol=0, atol=0
        )
    torch.testing.assert_close(current, replacement, rtol=0, atol=0)
    torch.testing.assert_close(one[:, :129], replacement[:, :129], rtol=0, atol=0)
    torch.testing.assert_close(one[:, 129:], temporal[:, 129:], rtol=0, atol=0)
    torch.testing.assert_close(last[:, :66], replacement[:, :66], rtol=0, atol=0)
    torch.testing.assert_close(last[:, 66:], temporal[:, 66:], rtol=0, atol=0)
    assert all(value.shape[1] == 130 for value in (current, one, last))


def test_shared_camera_encoder_preserves_ordered_camera_identity() -> None:
    frontend = NativeRGBTemporalFrontend().eval()
    assert isinstance(frontend.camera_encoder, GridCameraEncoder)
    value = torch.zeros(1, 2, 8, 8, 9)
    value[:, 0, :4, :4] = 1.0
    swapped = value.flip(1)
    with torch.inference_mode():
        original = frontend(value)
        reversed_cameras = frontend(swapped)
    assert original.shape == (1, 256)
    assert not torch.equal(original, reversed_cameras)
    assert frontend.camera_encoder.convs[0].kernel_size == (3, 3)
    assert frontend.camera_encoder.projection.in_features == 128 * 2 * 2


def test_gru_and_transformer_match_frozen_temporal_parameter_counts() -> None:
    for kind in ("gru", "transformer"):
        module = build_temporal_backbone(kind, seed=230923)
        assert trainable_parameter_count(module) == EXPECTED_TEMPORAL_PARAMETERS[kind]
    assert EXPECTED_TEMPORAL_PARAMETERS == {
        "mamba": 864_560,
        "gru": 790_528,
        "transformer": 857_216,
    }


def test_frontend_current_and_readout_initialization_are_condition_matched() -> None:
    gru = MikasaNativeRGBMemoryModel("gru_full", seed=230923)
    transformer = MikasaNativeRGBMemoryModel("transformer_full", seed=230923)
    for name in ("temporal_frontend", "current_branch", "readout"):
        left = getattr(gru, name).state_dict()
        right = getattr(transformer, name).state_dict()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)


def test_model_is_stateless_across_calls_and_row_order() -> None:
    model = MikasaNativeRGBMemoryModel("gru_full", seed=230923, frame_chunk_size=65).eval()
    temporal, post, proprio = _grids(2)
    batch = MemoryBatch(temporal, post, proprio)
    reversed_batch = MemoryBatch(temporal.flip(0), post.flip(0), proprio.flip(0))
    with torch.inference_mode():
        first = model(batch)
        repeated = model(batch)
        reversed_rows = model(reversed_batch)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first.flip(0), reversed_rows, rtol=0, atol=2e-6)
    assert not any(key in model.__dict__ for key in ("state", "hidden", "cache"))


def test_model_returns_finite_48d_checkpoint_prediction_and_gradients() -> None:
    model = MikasaNativeRGBMemoryModel("gru_full", seed=230923, frame_chunk_size=65)
    temporal, post, proprio = _grids(1)
    prediction = model(MemoryBatch(temporal, post, proprio))
    assert prediction.shape == (1, 48)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    for module in (model.temporal_frontend, model.current_branch, model.temporal, model.readout):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        assert any(torch.count_nonzero(gradient) for gradient in gradients if gradient is not None)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda temporal, post: (temporal[:, :-1], post), "invalid shape"),
        (lambda temporal, post: (temporal.to(torch.int64), post), "floating point"),
        (lambda temporal, post: (temporal, post[:, :1]), "invalid shape"),
    ],
)
def test_condition_transform_rejects_shape_and_dtype_drift(mutator, match: str) -> None:
    temporal, post, _ = _grids(1)
    temporal, post = mutator(temporal, post)
    with pytest.raises(ValueError, match=match):
        apply_history_condition(temporal, post, "mamba_full")


def test_no_native_vla_modules_are_imported() -> None:
    import sys

    forbidden = (
        "gr00t.model.gr00t_n1d7",
        "gr00t.model.modules.dit",
        "gr00t.policy",
    )
    assert not [name for name in sys.modules if name.startswith(forbidden)]


def test_module_deepcopy_does_not_create_persistent_runtime_state() -> None:
    model = MikasaNativeRGBMemoryModel("gru_full", seed=230923)
    cloned = copy.deepcopy(model)
    assert model.state_dict().keys() == cloned.state_dict().keys()
    assert all(
        torch.equal(model.state_dict()[key], cloned.state_dict()[key]) for key in model.state_dict()
    )
