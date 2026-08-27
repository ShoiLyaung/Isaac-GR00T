from __future__ import annotations

from dataclasses import asdict

from gr00t.model.gr00t_n1d7.codec_dynamics import Gr00tCodecConditioningStack, size_codec_modules
import pytest
import torch
from torch import nn


def _legacy_batch(*, batch_size: int = 2, sample_valid: bool = True) -> dict:
    history = 8
    mv_count = 3
    frame_valid = torch.ones((batch_size, history), dtype=torch.bool)
    validity = {
        name: frame_valid.clone()
        for name in (
            "frame_valid",
            "packet_valid",
            "reference_valid",
            "dependency_valid",
            "alignment_valid",
            "mv_valid",
            "rpe_valid",
        )
    }
    motion_fields = {
        "source": torch.zeros((batch_size, history, mv_count), dtype=torch.int32),
        "block_width": torch.full((batch_size, history, mv_count), 16, dtype=torch.uint8),
        "block_height": torch.full((batch_size, history, mv_count), 16, dtype=torch.uint8),
        "source_x": torch.zeros((batch_size, history, mv_count), dtype=torch.int16),
        "source_y": torch.zeros((batch_size, history, mv_count), dtype=torch.int16),
        "destination_x": torch.tensor([0, 400, 0], dtype=torch.int16)
        .view(1, 1, mv_count)
        .expand(batch_size, history, -1),
        "destination_y": torch.tensor([0, 0, 300], dtype=torch.int16)
        .view(1, 1, mv_count)
        .expand(batch_size, history, -1),
        "flags": torch.zeros((batch_size, history, mv_count), dtype=torch.int64),
        "motion_x": torch.ones((batch_size, history, mv_count), dtype=torch.int32),
        "motion_y": torch.ones((batch_size, history, mv_count), dtype=torch.int32),
        "motion_scale": torch.full((batch_size, history, mv_count), 4, dtype=torch.int32),
    }
    frame_index = torch.arange(history, dtype=torch.int64).expand(batch_size, -1)
    return {
        "observation_row": torch.full((batch_size,), history - 1, dtype=torch.int64),
        "time_and_reference": {
            "frame_index": frame_index,
            "pts": frame_index * 1000,
            "dts": frame_index * 1000,
            "gop_id": torch.zeros_like(frame_index),
            "gop_position": frame_index,
            "reference_index": frame_index - 1,
        },
        "frame_type": torch.full((batch_size, history), 2, dtype=torch.uint8),
        "frame_validity": validity,
        "frame_reason_code": torch.zeros((batch_size, history), dtype=torch.int16),
        "motion_vector_fields": motion_fields,
        "mv_mask": torch.ones((batch_size, history, mv_count), dtype=torch.bool),
        "mv_count": torch.full((batch_size, history), mv_count, dtype=torch.int32),
        "rpe": torch.zeros((batch_size, history, 2, 2, 4), dtype=torch.float32),
        "rpe_mask": torch.ones((batch_size, history, 2, 2), dtype=torch.bool),
        "i_frame_appearance": torch.zeros((batch_size, 3, 256, 256), dtype=torch.float32),
        "i_frame_mask": torch.ones((batch_size,), dtype=torch.bool),
        "i_frame_index": torch.zeros((batch_size,), dtype=torch.int64),
        "camera_motion_summary": torch.zeros((batch_size, history, 4), dtype=torch.float32),
        "camera_motion_mask": torch.ones((batch_size, history), dtype=torch.bool),
        "sample_valid": torch.full((batch_size,), sample_valid, dtype=torch.bool),
        "sample_reason_code": torch.zeros((batch_size,), dtype=torch.int16),
    }


def _episode_batch(
    *, batch_size: int = 2, frame_valid: bool = True, sample_valid: bool = True
) -> dict:
    history = 8
    return {
        "frame_mask": torch.ones((batch_size, history), dtype=torch.bool),
        "frame_index": torch.arange(history, dtype=torch.int64).expand(batch_size, -1),
        "frame_type": torch.full((batch_size, history), 2, dtype=torch.uint8),
        "frame_validity": torch.full((batch_size, history, 7), frame_valid, dtype=torch.bool),
        "frame_reason_code": torch.zeros((batch_size, history), dtype=torch.int16),
        "sample_valid": torch.full((batch_size,), sample_valid, dtype=torch.bool),
        "sample_reason_code": torch.zeros((batch_size,), dtype=torch.int16),
        "mv_cell_features": torch.zeros((batch_size, history, 2, 2, 11), dtype=torch.float32),
        "mv_cell_mask": torch.ones((batch_size, history, 2, 2), dtype=torch.bool),
        "rpe": torch.zeros((batch_size, history, 2, 2, 4), dtype=torch.float32),
        "rpe_mask": torch.ones((batch_size, history, 2, 2), dtype=torch.bool),
        "metadata_features": torch.zeros((batch_size, history, 21), dtype=torch.float32),
        "anchor_appearance": torch.zeros((batch_size, 2, 3, 256, 256), dtype=torch.float32),
        "anchor_mask": torch.ones((batch_size, 2), dtype=torch.bool),
        "frame_anchor_slot": torch.cat(
            (
                torch.zeros((batch_size, 4), dtype=torch.int64),
                torch.ones((batch_size, 4), dtype=torch.int64),
            ),
            dim=1,
        ),
    }


@pytest.mark.parametrize("batch_factory", [_legacy_batch, _episode_batch])
def test_t8_core_projects_to_32_action_width_tokens(batch_factory):
    torch.manual_seed(0)
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)

    context, mask = stack.encode(batch_factory())

    assert context.shape == (2, 32, 64)
    assert mask.shape == (2, 32)
    assert mask.all()
    assert torch.isfinite(context).all()


@pytest.mark.parametrize(
    "batch",
    [_legacy_batch(sample_valid=False), _episode_batch(sample_valid=False)],
)
def test_all_invalid_context_is_exact_native_fallback(batch):
    torch.manual_seed(1)
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)
    context, mask = stack.encode(batch)
    action = torch.arange(2 * 40 * 64, dtype=torch.float32).reshape(2, 40, 64)

    assert not mask.any()
    assert torch.count_nonzero(context) == 0
    assert torch.equal(stack.condition(action, context, mask), action)


def test_padding_masks_rpe_camera_and_episode_metadata():
    torch.manual_seed(2)
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)
    legacy = _legacy_batch(batch_size=1)
    legacy["frame_validity"]["frame_valid"][:, :4] = False
    legacy["rpe"][:, :4] = 1234.0
    legacy["camera_motion_summary"][:, :4] = 1234.0
    clean_legacy = _legacy_batch(batch_size=1)
    clean_legacy["frame_validity"]["frame_valid"][:, :4] = False
    legacy_context, legacy_mask = stack.encode(legacy)
    clean_context, clean_mask = stack.encode(clean_legacy)
    assert torch.equal(legacy_mask, clean_mask)
    assert torch.equal(legacy_context, clean_context)

    episode = _episode_batch(batch_size=1)
    episode["frame_mask"][:, :4] = False
    episode["rpe"][:, :4] = 1234.0
    episode["metadata_features"][:, :4] = 1234.0
    clean_episode = _episode_batch(batch_size=1)
    clean_episode["frame_mask"][:, :4] = False
    episode_context, episode_mask = stack.encode(episode)
    clean_context, clean_mask = stack.encode(clean_episode)
    assert torch.equal(episode_mask, clean_mask)
    assert torch.equal(episode_context, clean_context)


def test_native_width_sizing_contract_is_locked():
    sizing = asdict(size_codec_modules())

    assert sizing["d_codec"] == 64
    assert sizing["action_width"] == 1536
    assert sizing["conditioner_heads"] == 8
    assert sizing["parameters"] == 9_571_649
    assert sizing["encode_flops_batch1"] == 7_487_488
    assert sizing["condition_flops_per_call_batch1"] == 683_409_408
    assert sizing["training_forward_flops_batch1"] == 690_896_896
    assert sizing["policy_query_four_euler_flops_batch1"] == 2_741_125_120


class _NaN(nn.Module):
    def forward(self, value):
        return torch.full_like(value, float("nan"))


def test_conditioner_checks_proposal_finite_before_zero_gate():
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)
    stack.conditioner.output_norm = _NaN()
    action = torch.zeros((1, 40, 64))
    context = torch.zeros((1, 32, 64))
    mask = torch.ones((1, 32), dtype=torch.bool)

    with pytest.raises(FloatingPointError, match="proposal"):
        stack.condition(action, context, mask)
