"""Torch conversion for the framework-neutral codec-evidence batch."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .contract import CodecEvidenceSample, assert_batch_conforms, collate_codec_evidence_numpy
from .episode_bundle import (
    EpisodeT8View,
    assert_episode_batch_conforms,
    collate_episode_views_numpy,
)


EPISODE_AUDIT_IDENTITY_KEYS = frozenset({"episode_uid", "camera_id", "source_kind"})
EPISODE_MODEL_TENSOR_KEYS = frozenset(
    {
        "observation_row",
        "frame_mask",
        "frame_index",
        "frame_type",
        "frame_validity",
        "frame_reason_code",
        "mv_cell_features",
        "mv_cell_mask",
        "rpe",
        "rpe_mask",
        "metadata_features",
        "anchor_appearance",
        "anchor_mask",
        "anchor_frame_index",
        "frame_anchor_slot",
        "sample_valid",
        "sample_reason_code",
    }
)


def _to_torch(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_torch(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise TypeError(f"Unsupported codec-evidence batch value: {type(value).__name__}")


def _episode_model_payload(numpy_batch: dict[str, Any]) -> dict[str, np.ndarray]:
    """Remove CPU-side audit identity before the model/device-transfer boundary."""

    if not EPISODE_AUDIT_IDENTITY_KEYS.issubset(numpy_batch):
        raise ValueError("Episode batch lacks its CPU audit identity")
    payload = {
        key: value for key, value in numpy_batch.items() if key not in EPISODE_AUDIT_IDENTITY_KEYS
    }
    if set(payload) != EPISODE_MODEL_TENSOR_KEYS:
        raise ValueError("Episode model tensor keys do not match the frozen transport contract")
    if not all(isinstance(value, np.ndarray) for value in payload.values()):
        raise TypeError("Episode model payload must contain only NumPy arrays")
    return payload


class TorchCodecEvidenceCollator:
    """Deterministic collator: NumPy contract check first, Torch conversion second."""

    def __call__(self, samples: Sequence[CodecEvidenceSample | EpisodeT8View]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty codec-evidence batch")
        if all(isinstance(sample, EpisodeT8View) for sample in samples):
            numpy_batch = collate_episode_views_numpy(samples)
            assert_episode_batch_conforms(numpy_batch)
            numpy_batch = _episode_model_payload(numpy_batch)
        elif all(isinstance(sample, CodecEvidenceSample) for sample in samples):
            numpy_batch = collate_codec_evidence_numpy(samples)
            assert_batch_conforms(numpy_batch)
        else:
            raise TypeError("Codec-evidence batches may not mix record-v2 and episode views")
        return _to_torch(numpy_batch)
