from __future__ import annotations

from dataclasses import fields

from gr00t.experimental.mikasa_native_rgb_memory_runtime import TargetStats
from gr00t.experimental.mikasa_segment_slot_bottleneck import (
    SLOT_BOUNDARIES,
    MikasaSegmentSlotModel,
    SegmentSlotBatch,
)
from gr00t.experimental.mikasa_segment_slot_experiment import segment_slot_loss
import numpy as np
import pytest
import torch


def _temporal(batch: int = 2) -> torch.Tensor:
    return torch.rand(batch, 130, 2, 8, 8, 9, generator=torch.Generator().manual_seed(11))


def test_exact_segment_indices_produce_mean_max_18_channel_maps() -> None:
    temporal = torch.zeros(1, 130, 2, 8, 8, 9)
    temporal[:, :66] = 2.0
    temporal[:, 66:] = 5.0
    pooled = MikasaSegmentSlotModel.pool_segments(temporal)
    assert SLOT_BOUNDARIES == ((0, 66), (66, 130))
    assert pooled.shape == (1, 2, 2, 8, 8, 18)
    torch.testing.assert_close(pooled[:, 0, ..., :9], torch.full_like(pooled[:, 0, ..., :9], 2.0))
    torch.testing.assert_close(pooled[:, 0, ..., 9:], torch.full_like(pooled[:, 0, ..., 9:], 2.0))
    torch.testing.assert_close(pooled[:, 1, ..., :9], torch.full_like(pooled[:, 1, ..., :9], 5.0))
    torch.testing.assert_close(pooled[:, 1, ..., 9:], torch.full_like(pooled[:, 1, ..., 9:], 5.0))


def test_model_abi_contains_only_temporal_grid_and_outputs_two_shared_slots() -> None:
    assert [field.name for field in fields(SegmentSlotBatch)] == ["temporal_grid"]
    model = MikasaSegmentSlotModel(seed=230923).eval()
    with torch.inference_mode():
        output = model(SegmentSlotBatch(_temporal(2)))
    assert output.shape_logits.shape == (2, 2, 3)
    assert output.normalized_checkpoints.shape == (2, 2, 24)
    assert not hasattr(model, "slot0_encoder")
    assert not hasattr(model, "slot1_encoder")
    assert not hasattr(model, "base_camera_encoder")
    assert not hasattr(model, "hand_camera_encoder")
    with pytest.raises(TypeError):
        SegmentSlotBatch(temporal_grid=_temporal(1), group_id=torch.tensor([0]))


def test_ordered_camera_fusion_does_not_average_camera_identity() -> None:
    model = MikasaSegmentSlotModel(seed=230923).eval()
    slot_maps = torch.zeros(1, 2, 2, 8, 8, 18)
    slot_maps[:, :, 0, :4, :4] = 1.0
    with torch.inference_mode():
        original = model.encode_slot_maps(slot_maps)
        swapped = model.encode_slot_maps(slot_maps.flip(2))
    assert original.shape == (1, 2, 256)
    assert not torch.equal(original, swapped)


def test_shared_slot_model_is_exactly_swap_equivariant() -> None:
    model = MikasaSegmentSlotModel(seed=230923).eval()
    slot_maps = model.pool_segments(_temporal(3))
    with torch.inference_mode():
        original = model.forward_slot_maps(slot_maps)
        swapped = model.forward_slot_maps(slot_maps.flip(1))
    assert torch.equal(swapped.shape_logits, original.shape_logits.flip(1))
    assert torch.equal(
        swapped.normalized_checkpoints,
        original.normalized_checkpoints.flip(1),
    )


def test_one_to_one_loss_and_all_modules_receive_finite_nonzero_gradients() -> None:
    model = MikasaSegmentSlotModel(seed=230923)
    output = model(SegmentSlotBatch(_temporal(2)))
    shape_target = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
    checkpoint_target = torch.randn(2, 2, 12, 2)
    stats = TargetStats(np.zeros(48, np.float32), np.ones(48, np.float32))
    total, checkpoint_mse, shape_ce = segment_slot_loss(
        output,
        shape_target,
        checkpoint_target,
        stats,
    )
    torch.testing.assert_close(total, checkpoint_mse + shape_ce)
    total.backward()
    for module in (
        model.camera_encoder,
        model.camera_fusion,
        model.shape_head,
        model.checkpoint_head,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        assert any(torch.count_nonzero(gradient) for gradient in gradients if gradient is not None)


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(1, 129, 2, 8, 8, 9),
        torch.zeros(1, 130, 2, 8, 8, 8),
        torch.zeros(1, 130, 2, 8, 8, 9, dtype=torch.int64),
    ],
)
def test_segment_pool_rejects_shape_and_dtype_drift(value: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        MikasaSegmentSlotModel.pool_segments(value)
