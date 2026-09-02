from __future__ import annotations

from copy import deepcopy

from gr00t.data.codec_evidence.contract import (
    MV_FIELD_DTYPES,
    CodecEvidenceKey,
    CodecEvidenceRecord,
)
import numpy as np
import pytest


@pytest.fixture
def synthetic_codec_record() -> CodecEvidenceRecord:
    key = CodecEvidenceKey(
        episode_uid="a" * 64,
        observation_row=7,
        camera_id="cam_left_high",
    )
    frames = []
    for index in range(8):
        frame_type = "I" if index == 0 else "P"
        frames.append(
            {
                "slot": index,
                "frame_index": index,
                "pts": index * 1000,
                "dts": index * 1000,
                "frame_type": frame_type,
                "gop_id": 0,
                "gop_position": index,
                "reference_index": -1 if index == 0 else index - 1,
                "source_sha256": "b" * 64,
                "decoded_sha256": "c" * 64,
                "validity": {
                    "packet": True,
                    "frame": True,
                    "reference": True,
                    "dependency": True,
                    "alignment": True,
                    "mv": True,
                    "rpe": True,
                },
                "reason_code": "NONE",
                "mv_offset_begin": index * 2,
                "mv_offset_end": (index + 1) * 2,
                "mv_count": 2,
            }
        )
    payload = {
        "schema_version": "codec-evidence-record-v2",
        "episode": {
            "episode_uid": key.episode_uid,
            "dataset_name": "synthetic",
            "cohort_id": "synthetic",
            "source_episode_index": 0,
        },
        "camera_id": key.camera_id,
        "observation_row": key.observation_row,
        "history": {"length": 8, "frames": frames},
        "i_frame": {
            "frame_index": 0,
            "source_sha256": "b" * 64,
            "decoded_sha256": "c" * 64,
        },
        "sample_valid": True,
        "sample_reason_code": "NONE",
    }
    offsets = np.arange(0, 18, 2, dtype=np.int64)
    fields = {
        name: np.arange(16, dtype=np.int64).astype(dtype) for name, dtype in MV_FIELD_DTYPES.items()
    }
    fields["block_width"][:] = 16
    fields["block_height"][:] = 16
    fields["destination_x"][:] = np.tile([100, 500], 8)
    fields["destination_y"][:] = np.repeat([100, 300], 8)
    fields["motion_scale"][:] = 4
    return CodecEvidenceRecord(
        key=key,
        payload=deepcopy(payload),
        mv_offsets=offsets,
        mv_fields=fields,
        rpe=np.ones((8, 2, 2, 4), dtype=np.float32),
        i_frame_appearance=np.full((480, 640, 3), 127, dtype=np.uint8),
        camera_motion_summary=np.ones((8, 4), dtype=np.float32),
    )
