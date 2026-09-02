from __future__ import annotations

from dataclasses import asdict, replace

from gr00t.data.codec_evidence.episode_bundle import make_synthetic_episode_bundle
from gr00t.data.codec_evidence.processor import CodecEvidenceProcessorCore
from gr00t.data.codec_evidence.torch_backend import TorchCodecEvidenceCollator
from gr00t.model.gr00t_n1d7.codec_dynamics import (
    Gr00tCodecConditioningStack,
    ZeroGatedActionConditioner,
    size_codec_modules,
)
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from torch import nn


def test_dynamics_shape_finite_and_gate_zero_exact_parity(synthetic_codec_record):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    batch = TorchCodecEvidenceCollator()([sample, sample])
    torch.manual_seed(42)
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)
    context, mask = stack.encode(batch)
    assert context.shape == (2, 32, 64)
    assert mask.shape == (2, 32)
    assert torch.isfinite(context).all()
    action = torch.randn(2, 40, 64)
    output = stack.condition(action, context, mask)
    assert torch.equal(output, action)
    assert stack.conditioner.attention.dropout == 0.0


def test_all_invalid_is_exact_native_fallback(synthetic_codec_record):
    invalid = replace(
        CodecEvidenceProcessorCore().process(synthetic_codec_record),
        sample_valid=torch.tensor(False).numpy(),
    )
    batch = TorchCodecEvidenceCollator()([invalid])
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)
    context, mask = stack.encode(batch)
    assert not mask.any()
    assert torch.count_nonzero(context) == 0
    action = torch.randn(1, 40, 64)
    assert torch.equal(stack.condition(action, context, mask), action)


def test_bfloat16_model_boundary_preserves_gate_zero_parity(synthetic_codec_record):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    batch = TorchCodecEvidenceCollator()([sample])
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8).to(
        dtype=torch.bfloat16
    )
    context, mask = stack.encode(batch)
    assert context.dtype == torch.bfloat16
    action = torch.randn(1, 40, 64, dtype=torch.bfloat16)
    assert torch.equal(stack.condition(action, context, mask), action)


def test_conditioner_rejects_nonfinite_proposal_before_gate():
    conditioner = ZeroGatedActionConditioner(64, 8, gate_init=0.0)
    action = torch.zeros(1, 40, 64)
    context = torch.zeros(1, 32, 64)
    context[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        conditioner(action, context, torch.ones(1, 32, dtype=torch.bool))


def test_native_width_implementation_sizing_is_explicit():
    sizing = asdict(size_codec_modules())
    assert sizing["action_width"] == 1536
    assert sizing["parameters"] == 9_571_841
    assert sizing["encode_flops_batch1"] == 7_499_776
    assert sizing["condition_flops_per_call_batch1"] == 683_409_408
    assert sizing["training_forward_flops_batch1"] == 690_909_184
    assert sizing["policy_query_four_euler_flops_batch1"] == 2_741_137_408
    assert sizing["status"] == "IMPLEMENTATION_SIZED_NOT_RUNTIME_BENCHMARKED"


def test_mv_and_camera_masks_are_consumed(synthetic_codec_record):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    batch = TorchCodecEvidenceCollator()([sample])
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)

    masked = {
        **batch,
        "frame_validity": dict(batch["frame_validity"]),
    }
    masked["frame_validity"]["mv_valid"] = torch.zeros_like(batch["frame_validity"]["mv_valid"])
    aggregate = stack.core._aggregate_mv(masked, torch.float32)
    assert torch.count_nonzero(aggregate) == 0

    masked["camera_motion_mask"] = torch.zeros_like(batch["camera_motion_mask"])
    masked["camera_motion_summary"] = torch.full_like(batch["camera_motion_summary"], 1234.0)
    zero_camera = dict(masked)
    zero_camera["camera_motion_summary"] = torch.zeros_like(batch["camera_motion_summary"])
    masked_context, masked_tokens = stack.encode(masked)
    zero_context, zero_tokens = stack.encode(zero_camera)
    assert torch.equal(masked_tokens, zero_tokens)
    assert torch.equal(masked_context, zero_context)


def test_action_head_disabled_is_identity_and_does_not_read_codec():
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(head)
    head.codec_evidence_enabled = False
    head.codec_conditioning = None

    class NoRead:
        def __contains__(self, key):
            raise AssertionError("disabled path read action_input")

    action = torch.randn(1, 40, 64)
    assert head._encode_codec_context(NoRead()) is None
    assert head._condition_action_features(action, None) is action


def test_episode_bundle_dynamics_uses_bounded_cells_and_k2_anchor_bank():
    bundle = make_synthetic_episode_bundle(num_frames=60)
    view = bundle.view(30)
    assert view.anchor_mask.tolist() == [True, True]
    assert view.anchor_frame_index.tolist() == [0, 30]
    batch = TorchCodecEvidenceCollator()([view])
    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)
    context, mask = stack.encode(batch)
    assert context.shape == (1, 32, 64)
    assert mask.shape == (1, 32)
    assert torch.isfinite(context).all()
    action = torch.randn(1, 40, 64)
    assert torch.equal(stack.condition(action, context, mask), action)
