from __future__ import annotations

from gr00t.experimental.arena_codec_streaming_backbone import (
    CODEC_FIELDS,
    LAYERS,
    MAMBA2_CONFIG,
    WIDTH,
    ArenaCodecSpatialTokenizer,
    CausalGRUBackbone,
    CausalTransformerBackbone,
    CurrentDescriptorBackbone,
    trainable_parameter_count,
)
import pytest
import torch


EXPECTED_OFFICIAL_MAMBA2_PARAMETERS = 864_560


def test_spatial_tokenizer_keeps_2x2_layout_and_supports_variable_cameras() -> None:
    tokenizer = ArenaCodecSpatialTokenizer().eval()
    assert tokenizer.conv1.kernel_size == (3, 3)
    assert tokenizer.conv1.stride == (2, 2)
    assert tokenizer.conv2.kernel_size == (3, 3)
    assert tokenizer.conv2.stride == (2, 2)
    assert tokenizer.projection.in_features == 128 * 2 * 2

    one_camera = torch.randn(2, 5, 1, 8, 8, CODEC_FIELDS)
    three_cameras = torch.randn(2, 5, 3, 8, 8, CODEC_FIELDS)
    with torch.inference_mode():
        one_output = tokenizer(one_camera)
        three_output = tokenizer(three_cameras)
        frame_output = tokenizer.encode_frame(three_cameras[:, 0])
    assert one_output.shape == (2, 5, WIDTH)
    assert three_output.shape == (2, 5, WIDTH)
    torch.testing.assert_close(frame_output, three_output[:, 0])
    assert torch.isfinite(one_output).all()
    assert torch.isfinite(three_output).all()


def test_spatial_tokenizer_is_not_direct_cell_mean_pooling() -> None:
    tokenizer = ArenaCodecSpatialTokenizer().eval()
    first = torch.zeros(1, 1, 1, 8, 8, CODEC_FIELDS)
    second = first.clone()
    first[..., 0, 0, 0] = 1.0
    second[..., 7, 7, 0] = 1.0
    assert torch.equal(first.mean(dim=(-3, -2)), second.mean(dim=(-3, -2)))
    with torch.inference_mode():
        first_output = tokenizer(first)
        second_output = tokenizer(second)
    assert not torch.equal(first_output, second_output)


def test_gru_full_step_chunk_and_reset_match_on_cpu() -> None:
    module = CausalGRUBackbone().eval()
    descriptors = torch.randn(2, 13, WIDTH)
    with torch.inference_mode():
        full = module(descriptors)
        streamed, state = module.stream(descriptors)
        first_bytes = state.tensor_nbytes()
        state.reset()
        first_reset, state = module.stream(descriptors[:, :5], state)
        state.reset()
        second_reset, state = module.stream(descriptors[:, :5], state)
    torch.testing.assert_close(streamed, full, rtol=0, atol=1e-5)
    torch.testing.assert_close(first_reset, second_reset, rtol=0, atol=0)
    assert state.observations == 5
    assert state.tensor_nbytes() == first_bytes


def test_transformer_is_causal_and_all_outputs_are_finite() -> None:
    transformer = CausalTransformerBackbone().eval()
    identity = CurrentDescriptorBackbone().eval()
    descriptors = torch.randn(1, 16, WIDTH)
    changed = descriptors.clone()
    changed[:, 8:] += 0.25
    with torch.inference_mode():
        original = transformer(descriptors)
        perturbed = transformer(changed)
        lower_bound = identity(descriptors)
    torch.testing.assert_close(original[:, :8], perturbed[:, :8], rtol=0, atol=1e-6)
    assert original.shape == (1, 16, WIDTH)
    assert torch.isfinite(original).all()
    assert torch.equal(lower_bound, descriptors)


def test_temporal_controls_match_frozen_official_mamba_parameter_count() -> None:
    gru_parameters = trainable_parameter_count(CausalGRUBackbone())
    transformer_parameters = trainable_parameter_count(CausalTransformerBackbone())
    assert gru_parameters == 790_528
    assert transformer_parameters == 857_216
    for count in (gru_parameters, transformer_parameters):
        assert (
            abs(count - EXPECTED_OFFICIAL_MAMBA2_PARAMETERS) / EXPECTED_OFFICIAL_MAMBA2_PARAMETERS
            <= 0.15
        )
    assert LAYERS == 2
    assert MAMBA2_CONFIG == {
        "d_model": 256,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
        "chunk_size": 16,
        "use_mem_eff_path": False,
    }


@pytest.mark.parametrize(
    "value,match",
    [
        (torch.zeros(1, 2, 1, 8, 8, CODEC_FIELDS - 1), "expected codec sequence"),
        (torch.zeros(1, 2, 1, 8, 8, CODEC_FIELDS, dtype=torch.int64), "floating point"),
    ],
)
def test_tokenizer_rejects_invalid_inputs(value: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ArenaCodecSpatialTokenizer()(value)


def test_tokenizer_rejects_nonfinite_input() -> None:
    value = torch.zeros(1, 2, 1, 8, 8, CODEC_FIELDS)
    value[0, 0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        ArenaCodecSpatialTokenizer()(value)
