"""NumPy codec-evidence record, sample, and batch contracts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np


HISTORY = 8
MAX_MV_PER_FRAME = 19_200
MAX_CODEC_BATCH_SIZE = 64
MAX_PADDED_MV_ELEMENTS = 2_000_000
MAX_CODEC_TRANSPORT_BYTES = 128 * 1024 * 1024
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
FRAME_TYPE_CODES = MappingProxyType({"PAD": 0, "I": 1, "P": 2})
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
TIME_REFERENCE_FIELDS = (
    "frame_index",
    "pts",
    "dts",
    "gop_id",
    "gop_position",
    "reference_index",
)
VALIDITY_FIELDS = (
    "frame_valid",
    "packet_valid",
    "reference_valid",
    "dependency_valid",
    "alignment_valid",
    "mv_valid",
    "rpe_valid",
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


@dataclass(frozen=True, order=True)
class CodecEvidenceKey:
    episode_uid: str
    observation_row: int
    camera_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_uid, str) or not SHA256_RE.fullmatch(self.episode_uid):
            raise ValueError("episode_uid must be a lowercase SHA-256")
        if type(self.observation_row) is not int or self.observation_row < 0:
            raise ValueError("observation_row must be a non-negative integer")
        if self.camera_id != "cam_left_high":
            raise ValueError("camera_id must be cam_left_high")

    @property
    def canonical(self) -> str:
        return f"{self.episode_uid}:{self.observation_row}:{self.camera_id}"


@dataclass(frozen=True)
class CodecEvidenceRecord:
    """Validated disk/in-memory facts plus typed artifact arrays."""

    key: CodecEvidenceKey
    payload: Mapping[str, Any]
    mv_offsets: np.ndarray
    mv_fields: Mapping[str, np.ndarray]
    rpe: np.ndarray
    i_frame_appearance: np.ndarray
    camera_motion_summary: np.ndarray


@dataclass(frozen=True)
class CodecEvidenceSample:
    """Framework-neutral per-observation T8 sample."""

    key: CodecEvidenceKey
    observation_row: np.int64
    time_and_reference: Mapping[str, np.ndarray]
    frame_type: np.ndarray
    frame_validity: Mapping[str, np.ndarray]
    frame_reason_code: np.ndarray
    mv_offsets: np.ndarray
    motion_vector_fields: Mapping[str, np.ndarray]
    rpe: np.ndarray
    rpe_mask: np.ndarray
    i_frame_appearance: np.ndarray
    i_frame_mask: np.bool_
    i_frame_index: np.int64
    camera_motion_summary: np.ndarray
    camera_motion_mask: np.ndarray
    sample_valid: np.bool_
    sample_reason_code: np.int16


CodecEvidenceBatch: TypeAlias = dict[str, Any]
"""Framework-neutral nested batch produced by ``collate_codec_evidence_numpy``."""


def preflight_codec_batch_allocation(batch_size: int, max_mv_count: int) -> int:
    """Bound the legacy padded-MV transport before allocating its arrays."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("codec batch_size must be a positive integer")
    if type(max_mv_count) is not int or not 0 <= max_mv_count <= MAX_MV_PER_FRAME:
        raise ValueError("max_mv_count is outside the frozen per-frame bound")
    if batch_size > MAX_CODEC_BATCH_SIZE:
        raise ValueError(f"codec batch_size exceeds MAX_CODEC_BATCH_SIZE={MAX_CODEC_BATCH_SIZE}")
    padded_elements = batch_size * HISTORY * max_mv_count
    if padded_elements > MAX_PADDED_MV_ELEMENTS:
        raise ValueError(
            f"padded MV elements exceed MAX_PADDED_MV_ELEMENTS={MAX_PADDED_MV_ELEMENTS}"
        )
    mv_element_bytes = np.dtype("bool").itemsize + sum(
        dtype.itemsize for dtype in MV_FIELD_DTYPES.values()
    )
    fixed_bytes_per_sample = (
        np.dtype("int64").itemsize
        + len(TIME_REFERENCE_FIELDS) * HISTORY * np.dtype("int64").itemsize
        + HISTORY * np.dtype("uint8").itemsize
        + len(VALIDITY_FIELDS) * HISTORY * np.dtype("bool").itemsize
        + HISTORY * np.dtype("int16").itemsize
        + HISTORY * np.dtype("int32").itemsize
        + HISTORY * 2 * 2 * 4 * np.dtype("float32").itemsize
        + HISTORY * 2 * 2 * np.dtype("bool").itemsize
        + 3 * 256 * 256 * np.dtype("float32").itemsize
        + np.dtype("bool").itemsize
        + np.dtype("int64").itemsize
        + HISTORY * 4 * np.dtype("float32").itemsize
        + HISTORY * np.dtype("bool").itemsize
        + np.dtype("bool").itemsize
        + np.dtype("int16").itemsize
    )
    transport_bytes = padded_elements * mv_element_bytes + batch_size * fixed_bytes_per_sample
    if transport_bytes > MAX_CODEC_TRANSPORT_BYTES:
        raise ValueError(
            f"codec transport exceeds MAX_CODEC_TRANSPORT_BYTES={MAX_CODEC_TRANSPORT_BYTES}"
        )
    return int(transport_bytes)


def _array(value: Any, *, dtype: np.dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _expected_frame_reason_code(frame_validity: Mapping[str, np.ndarray], slot: int) -> int:
    for validity_name, reason_name in INVALID_REASON_PRIORITY:
        if not bool(frame_validity[validity_name][slot]):
            return REASON_CODES[reason_name]
    return REASON_CODES["NONE"]


def validate_sample(sample: CodecEvidenceSample) -> None:
    """Validate every invariant still observable after Record -> Sample.

    Artifact and source hashes are intentionally not present in
    :class:`CodecEvidenceSample`; their verification remains the store and
    ``CodecEvidenceProcessorCore`` boundary.  The collator revalidates all
    structural, causal, mask, reason-code, and fallback invariants below so a
    hand-built or mutated sample cannot bypass that boundary.
    """

    observation_row = _array(
        sample.observation_row, dtype=np.dtype("int64"), shape=(), name="observation_row"
    )
    if sample.key.observation_row != int(observation_row):
        raise ValueError("Sample key observation_row disagrees with observation_row")
    if not isinstance(sample.time_and_reference, Mapping) or set(sample.time_and_reference) != set(
        TIME_REFERENCE_FIELDS
    ):
        raise ValueError("Sample time_and_reference fields differ from the contract")
    time_and_reference = {
        name: _array(
            sample.time_and_reference[name],
            dtype=np.dtype("int64"),
            shape=(HISTORY,),
            name=name,
        )
        for name in TIME_REFERENCE_FIELDS
    }
    frame_type = _array(
        sample.frame_type, dtype=np.dtype("uint8"), shape=(HISTORY,), name="frame_type"
    )
    if not isinstance(sample.frame_validity, Mapping) or set(sample.frame_validity) != set(
        VALIDITY_FIELDS
    ):
        raise ValueError("Sample frame_validity fields differ from the contract")
    frame_validity = {
        name: _array(
            sample.frame_validity[name],
            dtype=np.dtype("bool"),
            shape=(HISTORY,),
            name=name,
        )
        for name in VALIDITY_FIELDS
    }
    frame_reason_code = _array(
        sample.frame_reason_code,
        dtype=np.dtype("int16"),
        shape=(HISTORY,),
        name="frame_reason_code",
    )
    offsets = _array(
        sample.mv_offsets, dtype=np.dtype("int64"), shape=(HISTORY + 1,), name="mv_offsets"
    )
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("mv_offsets must start at zero and be monotonic")
    total = int(offsets[-1])
    counts = np.diff(offsets)
    if np.any(counts > MAX_MV_PER_FRAME):
        raise ValueError("MV count exceeds MAX_MV_PER_FRAME; truncation is forbidden")
    if set(sample.motion_vector_fields) != set(MV_FIELD_DTYPES):
        raise ValueError("Motion-vector fields do not match the frozen 11-field contract")
    for name, dtype in MV_FIELD_DTYPES.items():
        _array(
            sample.motion_vector_fields[name],
            dtype=dtype,
            shape=(total,),
            name=f"motion_vector_fields.{name}",
        )
    rpe = _array(sample.rpe, dtype=np.dtype("float32"), shape=(8, 2, 2, 4), name="rpe")
    rpe_mask = _array(sample.rpe_mask, dtype=np.dtype("bool"), shape=(8, 2, 2), name="rpe_mask")
    i_frame_appearance = _array(
        sample.i_frame_appearance,
        dtype=np.dtype("float32"),
        shape=(3, 256, 256),
        name="i_frame_appearance",
    )
    i_frame_mask = _array(
        sample.i_frame_mask, dtype=np.dtype("bool"), shape=(), name="i_frame_mask"
    )
    i_frame_index = _array(
        sample.i_frame_index, dtype=np.dtype("int64"), shape=(), name="i_frame_index"
    )
    camera_motion_summary = _array(
        sample.camera_motion_summary,
        dtype=np.dtype("float32"),
        shape=(8, 4),
        name="camera_motion_summary",
    )
    camera_motion_mask = _array(
        sample.camera_motion_mask,
        dtype=np.dtype("bool"),
        shape=(8,),
        name="camera_motion_mask",
    )
    sample_valid = _array(
        sample.sample_valid, dtype=np.dtype("bool"), shape=(), name="sample_valid"
    )
    sample_reason_code = _array(
        sample.sample_reason_code,
        dtype=np.dtype("int16"),
        shape=(),
        name="sample_reason_code",
    )

    if int(sample_reason_code) in TRANSPORT_NEUTRAL_REASON_CODES:
        neutral_zero_values = (
            frame_type,
            frame_reason_code,
            offsets,
            rpe,
            rpe_mask,
            i_frame_appearance,
            camera_motion_summary,
            camera_motion_mask,
        )
        if bool(sample_valid):
            raise ValueError("Transport-neutral record sample must be sample_valid=false")
        if any(np.any(value != 0) for value in neutral_zero_values):
            raise ValueError("Transport-neutral record evidence must be exact zero/false")
        if any(np.any(values != -1) for values in time_and_reference.values()):
            raise ValueError("Transport-neutral record temporal IDs must be -1")
        if any(values.any() for values in frame_validity.values()):
            raise ValueError("Transport-neutral record validity must be false")
        if any(value.size for value in sample.motion_vector_fields.values()):
            raise ValueError("Transport-neutral record motion vectors must be empty")
        if bool(i_frame_mask) or int(i_frame_index) != -1:
            raise ValueError("Transport-neutral record I-frame identity must be unavailable")
        return

    if not set(int(value) for value in np.unique(frame_type)).issubset(FRAME_TYPE_CODES.values()):
        raise ValueError("Sample frame_type contains an unknown code")
    pad = frame_type == FRAME_TYPE_CODES["PAD"]
    active = ~pad
    if not active.any():
        raise ValueError("Sample history must contain its observation frame")
    first_active = int(np.flatnonzero(active)[0])
    if pad[first_active:].any() or not pad[:first_active].all():
        raise ValueError("Sample padding must be one exact left prefix")
    for name, values in time_and_reference.items():
        if np.any(values[pad] != -1):
            raise ValueError(f"PAD {name} values must be -1")
    for name, values in frame_validity.items():
        if values[pad].any():
            raise ValueError(f"PAD {name} values must be false")
    if np.any(frame_reason_code[pad] != REASON_CODES["LEFT_PAD"]):
        raise ValueError("PAD frame_reason_code must be LEFT_PAD")
    if np.any(counts[pad] != 0):
        raise ValueError("PAD frames must contain zero motion vectors")
    if np.any(rpe[pad] != 0) or rpe_mask[pad].any():
        raise ValueError("PAD RPE values and masks must be exact zero/false")
    if np.any(camera_motion_summary[pad] != 0) or camera_motion_mask[pad].any():
        raise ValueError("PAD camera values and masks must be exact zero/false")

    frame_index = time_and_reference["frame_index"]
    active_rows = frame_index[active]
    if np.any(active_rows < 0) or np.any(active_rows > int(observation_row)):
        raise ValueError("Sample contains a future or negative active frame")
    if not np.array_equal(
        active_rows,
        np.arange(
            int(observation_row) - active_rows.size + 1,
            int(observation_row) + 1,
            dtype=np.int64,
        ),
    ):
        raise ValueError("Sample active frame rows must be contiguous and end at observation")
    pts = time_and_reference["pts"]
    dts = time_and_reference["dts"]
    if np.any(pts[active] < 0) or np.any(dts[active] < 0):
        raise ValueError("Active PTS and DTS must be non-negative")
    if active_rows.size > 1 and (
        np.any(np.diff(pts[active]) <= 0) or np.any(np.diff(dts[active]) <= 0)
    ):
        raise ValueError("Active PTS and DTS must be strictly increasing")
    expected_timestamp = active_rows * 1000
    if not np.array_equal(pts[active], expected_timestamp) or not np.array_equal(
        dts[active], expected_timestamp
    ):
        raise ValueError("Active PTS and DTS must equal frame_index * 1000")

    for slot in np.flatnonzero(active):
        row = int(frame_index[slot])
        expected_gop, expected_position = divmod(row, 30)
        if (
            int(time_and_reference["gop_id"][slot]) != expected_gop
            or int(time_and_reference["gop_position"][slot]) != expected_position
        ):
            raise ValueError("Sample GOP identity must equal divmod(frame_index, 30)")
        expected_type = "I" if expected_position == 0 else "P"
        if int(frame_type[slot]) != FRAME_TYPE_CODES[expected_type]:
            raise ValueError("Sample I/P type disagrees with the GOP30 row")
        reference = int(time_and_reference["reference_index"][slot])
        expected_reference = -1 if expected_type == "I" else row - 1
        if reference != expected_reference:
            raise ValueError("Sample reference_index is noncausal")
        expected_reason = _expected_frame_reason_code(frame_validity, int(slot))
        if int(frame_reason_code[slot]) != expected_reason:
            raise ValueError("Sample frame_reason_code disagrees with validity priority")

    expected_rpe_mask = np.broadcast_to(
        (
            frame_validity["frame_valid"]
            & frame_validity["alignment_valid"]
            & frame_validity["rpe_valid"]
        )[:, None, None],
        (HISTORY, 2, 2),
    )
    if not np.array_equal(rpe_mask, expected_rpe_mask):
        raise ValueError("Sample rpe_mask is not exactly derived from validity")
    expected_camera_mask = frame_validity["frame_valid"] & frame_validity["mv_valid"]
    if not np.array_equal(camera_motion_mask, expected_camera_mask):
        raise ValueError("Sample camera_motion_mask is not exactly derived from validity")

    if np.any(i_frame_appearance < 0) or np.any(i_frame_appearance > 1):
        raise ValueError("Sample I-frame appearance must lie in [0,1]")
    if bool(i_frame_mask):
        expected_i_frame = int(observation_row) // 30 * 30
        if int(i_frame_index) != expected_i_frame:
            raise ValueError("Sample I-frame index is not the nearest legal past GOP30 anchor")
        in_window_anchor = (
            active & (frame_index == int(i_frame_index)) & (frame_type == FRAME_TYPE_CODES["I"])
        )
        if int(in_window_anchor.sum()) != 1:
            raise ValueError("record-v2 sample cannot bind an I-frame outside T8")
    else:
        if int(i_frame_index) != -1 or np.any(i_frame_appearance != 0):
            raise ValueError("Unavailable I-frame requires index -1 and zero appearance")

    active_reasons = frame_reason_code[active]
    invalid_reasons = active_reasons[active_reasons != REASON_CODES["NONE"]]
    if not bool(i_frame_mask):
        expected_sample_reason = REASON_CODES["I_FRAME_UNAVAILABLE"]
    elif invalid_reasons.size:
        expected_sample_reason = int(invalid_reasons[0])
    else:
        expected_sample_reason = REASON_CODES["NONE"]
    expected_sample_valid = expected_sample_reason == REASON_CODES["NONE"]
    if bool(sample_valid) != expected_sample_valid:
        raise ValueError("sample_valid disagrees with frame and I-frame validity")
    if int(sample_reason_code) != expected_sample_reason:
        raise ValueError("sample_reason_code disagrees with the sample fallback reason")


def make_neutral_codec_evidence_sample(
    key: CodecEvidenceKey, reason_code: int
) -> CodecEvidenceSample:
    """Build an online-only all-zero record-v2 transport fallback."""

    if type(reason_code) is not int or reason_code not in TRANSPORT_NEUTRAL_REASON_CODES:
        raise ValueError("Neutral record reason must be MISSING/SCHEMA/HASH")
    sample = CodecEvidenceSample(
        key=key,
        observation_row=np.int64(key.observation_row),
        time_and_reference={
            name: np.full((HISTORY,), -1, dtype=np.int64) for name in TIME_REFERENCE_FIELDS
        },
        frame_type=np.zeros((HISTORY,), dtype=np.uint8),
        frame_validity={name: np.zeros((HISTORY,), dtype=np.bool_) for name in VALIDITY_FIELDS},
        frame_reason_code=np.zeros((HISTORY,), dtype=np.int16),
        mv_offsets=np.zeros((HISTORY + 1,), dtype=np.int64),
        motion_vector_fields={
            name: np.empty((0,), dtype=dtype) for name, dtype in MV_FIELD_DTYPES.items()
        },
        rpe=np.zeros((HISTORY, 2, 2, 4), dtype=np.float32),
        rpe_mask=np.zeros((HISTORY, 2, 2), dtype=np.bool_),
        i_frame_appearance=np.zeros((3, 256, 256), dtype=np.float32),
        i_frame_mask=np.bool_(False),
        i_frame_index=np.int64(-1),
        camera_motion_summary=np.zeros((HISTORY, 4), dtype=np.float32),
        camera_motion_mask=np.zeros((HISTORY,), dtype=np.bool_),
        sample_valid=np.bool_(False),
        sample_reason_code=np.int16(reason_code),
    )
    validate_sample(sample)
    return sample


def collate_codec_evidence_numpy(samples: Sequence[CodecEvidenceSample]) -> CodecEvidenceBatch:
    """Pad only the ragged MV axis and return the backend-neutral batch."""

    if not samples:
        raise ValueError("Cannot collate an empty codec-evidence batch")
    for sample in samples:
        validate_sample(sample)
    batch_size = len(samples)
    max_count = max(int(np.diff(sample.mv_offsets).max(initial=0)) for sample in samples)
    if max_count > MAX_MV_PER_FRAME:
        raise ValueError("MV count exceeds the frozen maximum")
    preflight_codec_batch_allocation(batch_size, max_count)

    def stack_attr(name: str) -> np.ndarray:
        return np.stack([np.asarray(getattr(sample, name)) for sample in samples], axis=0)

    batch: CodecEvidenceBatch = {
        "observation_row": stack_attr("observation_row"),
        "time_and_reference": {
            name: np.stack([sample.time_and_reference[name] for sample in samples])
            for name in TIME_REFERENCE_FIELDS
        },
        "frame_type": stack_attr("frame_type"),
        "frame_validity": {
            name: np.stack([sample.frame_validity[name] for sample in samples])
            for name in VALIDITY_FIELDS
        },
        "frame_reason_code": stack_attr("frame_reason_code"),
        "motion_vector_fields": {
            name: np.zeros((batch_size, HISTORY, max_count), dtype=dtype)
            for name, dtype in MV_FIELD_DTYPES.items()
        },
        "mv_mask": np.zeros((batch_size, HISTORY, max_count), dtype=np.bool_),
        "mv_count": np.zeros((batch_size, HISTORY), dtype=np.int32),
        "rpe": stack_attr("rpe"),
        "rpe_mask": stack_attr("rpe_mask"),
        "i_frame_appearance": stack_attr("i_frame_appearance"),
        "i_frame_mask": stack_attr("i_frame_mask"),
        "i_frame_index": stack_attr("i_frame_index"),
        "camera_motion_summary": stack_attr("camera_motion_summary"),
        "camera_motion_mask": stack_attr("camera_motion_mask"),
        "sample_valid": stack_attr("sample_valid"),
        "sample_reason_code": stack_attr("sample_reason_code"),
    }
    for batch_index, sample in enumerate(samples):
        for frame_index, (begin, end) in enumerate(
            zip(sample.mv_offsets[:-1], sample.mv_offsets[1:])
        ):
            count = int(end - begin)
            batch["mv_count"][batch_index, frame_index] = count
            batch["mv_mask"][batch_index, frame_index, :count] = True
            for name in MV_FIELD_DTYPES:
                batch["motion_vector_fields"][name][batch_index, frame_index, :count] = (
                    sample.motion_vector_fields[name][begin:end]
                )
    assert_batch_conforms(batch)
    return batch


def assert_batch_conforms(batch: Mapping[str, Any]) -> None:
    """Fail closed on dtype/shape/mask drift for NumPy or array-like backends."""

    observation_row = np.asarray(batch["observation_row"])
    if observation_row.dtype != np.dtype("int64") or observation_row.ndim != 1:
        raise ValueError("observation_row must be int64 [B]")
    batch_size = observation_row.shape[0]
    frame_shape = (batch_size, HISTORY)
    for name in TIME_REFERENCE_FIELDS:
        _array(
            batch["time_and_reference"][name],
            dtype=np.dtype("int64"),
            shape=frame_shape,
            name=name,
        )
    _array(batch["frame_type"], dtype=np.dtype("uint8"), shape=frame_shape, name="frame_type")
    for name in VALIDITY_FIELDS:
        _array(
            batch["frame_validity"][name],
            dtype=np.dtype("bool"),
            shape=frame_shape,
            name=name,
        )
    _array(
        batch["frame_reason_code"],
        dtype=np.dtype("int16"),
        shape=frame_shape,
        name="frame_reason_code",
    )
    mv_count = _array(
        batch["mv_count"], dtype=np.dtype("int32"), shape=frame_shape, name="mv_count"
    )
    mv_mask = np.asarray(batch["mv_mask"])
    if mv_mask.dtype != np.dtype("bool") or mv_mask.shape[:2] != frame_shape:
        raise ValueError("mv_mask must be bool [B,8,M_batch]")
    m_batch = mv_mask.shape[2]
    if m_batch > MAX_MV_PER_FRAME or np.any(mv_count > m_batch):
        raise ValueError("mv_count exceeds M_batch")
    expected_mask = np.arange(m_batch)[None, None, :] < mv_count[:, :, None]
    if not np.array_equal(mv_mask, expected_mask):
        raise ValueError("mv_mask is not the exact prefix mask derived from mv_count")
    for name, dtype in MV_FIELD_DTYPES.items():
        _array(
            batch["motion_vector_fields"][name],
            dtype=dtype,
            shape=(batch_size, HISTORY, m_batch),
            name=f"motion_vector_fields.{name}",
        )
    fixed = {
        "rpe": (np.dtype("float32"), (batch_size, 8, 2, 2, 4)),
        "rpe_mask": (np.dtype("bool"), (batch_size, 8, 2, 2)),
        "i_frame_appearance": (np.dtype("float32"), (batch_size, 3, 256, 256)),
        "i_frame_mask": (np.dtype("bool"), (batch_size,)),
        "i_frame_index": (np.dtype("int64"), (batch_size,)),
        "camera_motion_summary": (np.dtype("float32"), (batch_size, 8, 4)),
        "camera_motion_mask": (np.dtype("bool"), (batch_size, 8)),
        "sample_valid": (np.dtype("bool"), (batch_size,)),
        "sample_reason_code": (np.dtype("int16"), (batch_size,)),
    }
    for name, (dtype, shape) in fixed.items():
        _array(batch[name], dtype=dtype, shape=shape, name=name)
