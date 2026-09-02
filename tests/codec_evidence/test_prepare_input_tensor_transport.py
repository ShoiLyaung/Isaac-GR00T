from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from gr00t.data.codec_evidence.episode_bundle import (
    collate_episode_views_numpy,
    make_synthetic_episode_bundle,
)
from gr00t.data.codec_evidence.processor import CodecEvidenceProcessorCore
from gr00t.data.codec_evidence.torch_backend import (
    EPISODE_AUDIT_IDENTITY_KEYS,
    EPISODE_MODEL_TENSOR_KEYS,
    TorchCodecEvidenceCollator,
)
from gr00t.model.gr00t_n1d7.codec_dynamics import Gr00tCodecConditioningStack
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
import torch
from torch import nn
from transformers import BatchFeature


def _real_prepare_input(inputs: dict[str, Any]) -> tuple[BatchFeature, BatchFeature]:
    """Call the real unbound Gr00tN1d7 device-transfer implementation."""

    class Backbone:
        @staticmethod
        def prepare_input(batch):
            return BatchFeature(data={"state": batch["state"]})

    action_head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(action_head)
    harness = SimpleNamespace(
        backbone=Backbone(),
        action_head=action_head,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return Gr00tN1d7.prepare_input(harness, dict(inputs))


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _tensor_leaves(item)]
    if isinstance(value, BatchFeature):
        return _tensor_leaves(value.data)
    assert isinstance(value, torch.Tensor)
    return [value]


def test_episode_identity_stays_cpu_side_and_real_prepare_input_receives_only_tensors():
    view = make_synthetic_episode_bundle(num_frames=60).view(30)
    audit_batch = collate_episode_views_numpy([view])
    assert set(EPISODE_AUDIT_IDENTITY_KEYS).issubset(audit_batch)
    assert audit_batch["episode_uid"] == (view.key.episode_uid,)
    assert audit_batch["camera_id"] == (view.key.camera_id,)
    assert audit_batch["source_kind"] == (view.key.source_kind,)

    model_payload = TorchCodecEvidenceCollator()([view])
    assert set(model_payload) == set(EPISODE_MODEL_TENSOR_KEYS)
    assert not set(EPISODE_AUDIT_IDENTITY_KEYS).intersection(model_payload)
    assert all(isinstance(value, torch.Tensor) for value in model_payload.values())

    _, action_inputs = _real_prepare_input(
        {
            "state": torch.zeros((1, 1, 132), dtype=torch.float64),
            "codec_evidence": model_payload,
        }
    )
    prepared = action_inputs["codec_evidence"]
    assert set(prepared) == set(EPISODE_MODEL_TENSOR_KEYS)
    assert not set(EPISODE_AUDIT_IDENTITY_KEYS).intersection(prepared)
    assert all(isinstance(leaf, torch.Tensor) for leaf in _tensor_leaves(prepared))
    assert prepared["mv_cell_features"].shape == (1, 8, 2, 2, 11)
    assert prepared["rpe"].shape == (1, 8, 2, 2, 4)
    assert prepared["metadata_features"].shape == (1, 8, 21)
    assert prepared["anchor_appearance"].shape == (1, 2, 3, 256, 256)

    stack = Gr00tCodecConditioningStack(
        action_width=64, d_codec=16, conditioner_heads=8, gate_init=0.0
    )
    context, context_mask = stack.encode(prepared)
    assert context.shape == (1, 32, 64)
    assert context_mask.shape == (1, 32)
    assert torch.isfinite(context).all()


def test_record_v2_and_disabled_real_prepare_input_paths_are_unchanged(synthetic_codec_record):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    record_payload = TorchCodecEvidenceCollator()([sample])
    assert "time_and_reference" in record_payload
    assert "motion_vector_fields" in record_payload
    assert all(isinstance(leaf, torch.Tensor) for leaf in _tensor_leaves(record_payload))

    _, enabled_action_inputs = _real_prepare_input(
        {
            "state": torch.zeros((1, 1, 132), dtype=torch.float32),
            "codec_evidence": record_payload,
        }
    )
    assert enabled_action_inputs["codec_evidence"].keys() == record_payload.keys()
    assert all(
        isinstance(leaf, torch.Tensor)
        for leaf in _tensor_leaves(enabled_action_inputs["codec_evidence"])
    )

    _, disabled_action_inputs = _real_prepare_input(
        {"state": torch.zeros((1, 1, 132), dtype=torch.float32)}
    )
    assert "codec_evidence" not in disabled_action_inputs
