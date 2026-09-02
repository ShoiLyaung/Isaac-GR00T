"""Shared constants for the active causal episode-bundle Codec contract."""

from __future__ import annotations

import re
from types import MappingProxyType

import numpy as np


HISTORY = 8
MAX_MV_PER_FRAME = 19_200
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
REASON_CODES = MappingProxyType(
    {
        "NONE": 0,
        "LEFT_PAD": 1,
        "PACKET_INVALID": 2,
        "FRAME_INVALID": 3,
        "REFERENCE_INVALID": 4,
        "REFERENCE_CHAIN_INVALID": 5,
        "ALIGNMENT_INVALID": 6,
        "MV_OVERFLOW": 7,
        "RPE_INVALID": 8,
        "I_FRAME_UNAVAILABLE": 9,
        "MISSING_RECORD": 10,
        "SCHEMA_INVALID": 11,
        "HASH_MISMATCH": 12,
    }
)
MV_FIELD_DTYPES = MappingProxyType(
    {
        "source": np.dtype("int32"),
        "block_width": np.dtype("uint8"),
        "block_height": np.dtype("uint8"),
        "source_x": np.dtype("int16"),
        "source_y": np.dtype("int16"),
        "destination_x": np.dtype("int16"),
        "destination_y": np.dtype("int16"),
        "flags": np.dtype("uint64"),
        "motion_x": np.dtype("int32"),
        "motion_y": np.dtype("int32"),
        "motion_scale": np.dtype("uint16"),
    }
)
INVALID_REASON_PRIORITY = (
    ("packet_valid", "PACKET_INVALID"),
    ("frame_valid", "FRAME_INVALID"),
    ("reference_valid", "REFERENCE_INVALID"),
    ("dependency_valid", "REFERENCE_CHAIN_INVALID"),
    ("alignment_valid", "ALIGNMENT_INVALID"),
    ("mv_valid", "MV_OVERFLOW"),
    ("rpe_valid", "RPE_INVALID"),
)
TRANSPORT_NEUTRAL_REASON_CODES = frozenset(
    {
        REASON_CODES["MISSING_RECORD"],
        REASON_CODES["SCHEMA_INVALID"],
        REASON_CODES["HASH_MISMATCH"],
    }
)


__all__ = [
    "HISTORY",
    "INVALID_REASON_PRIORITY",
    "MAX_MV_PER_FRAME",
    "MV_FIELD_DTYPES",
    "REASON_CODES",
    "SHA256_RE",
    "TRANSPORT_NEUTRAL_REASON_CODES",
]
