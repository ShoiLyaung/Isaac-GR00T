"""Immutable episode-level codec evidence and bounded NumPy T8 views.

Raw encoder motion vectors are intentionally outside the training surface.  A
bundle stores the bounded 2x2 cell aggregation used by training and keeps the
raw vectors in a separately named, lazily loaded audit artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    SYNTHETIC_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
    validate_codec_contract,
)
from .contract import (
    HISTORY,
    INVALID_REASON_PRIORITY,
    MAX_MV_PER_FRAME,
    MV_FIELD_DTYPES,
    REASON_CODES,
    SHA256_RE,
    TRANSPORT_NEUTRAL_REASON_CODES,
)
from .identity import (
    EPISODE_IDENTITY_SCHEMA_VERSION,
    episode_uid_from_preimage,
    validate_episode_identity_preimage,
)


BUNDLE_SCHEMA_VERSION = "codec-evidence-episode-bundle-v1"
K_MAX = 2
CELL_ROWS = 2
CELL_COLUMNS = 2
MV_CELL_WIDTH = 11
RPE_WIDTH = 4
METADATA_WIDTH = 21
MAX_EPISODE_BATCH_SIZE = 64
MAX_EPISODE_TRANSPORT_BYTES = 128 * 1024 * 1024

SOURCE_KIND_RAW_JPEG = "RAW_JPEG_DERIVED"
SOURCE_KIND_NATIVE_CANONICAL = "NATIVE_CANONICAL_DECODED_DERIVED"
SOURCE_KINDS = frozenset({SOURCE_KIND_RAW_JPEG, SOURCE_KIND_NATIVE_CANONICAL})

FRAME_TYPE_PAD = 0
FRAME_TYPE_IDR = 1
FRAME_TYPE_INTRA_NON_IDR = 2
FRAME_TYPE_P = 3
NON_PAD_FRAME_TYPES = frozenset({FRAME_TYPE_IDR, FRAME_TYPE_INTRA_NON_IDR, FRAME_TYPE_P})

FRAME_VALIDITY_NAMES = (
    "frame_valid",
    "packet_valid",
    "reference_valid",
    "dependency_valid",
    "alignment_valid",
    "mv_valid",
    "rpe_valid",
)
METADATA_FEATURE_NAMES = (
    "is_idr",
    "is_intra_non_idr",
    "is_p",
    "relative_frame_index",
    "relative_pts_seconds",
    "relative_dts_seconds",
    "normalized_gop_position",
    "normalized_reference_distance",
    *FRAME_VALIDITY_NAMES,
    "camera_valid",
    "anchor_valid",
    "camera_motion_0",
    "camera_motion_1",
    "camera_motion_2",
    "camera_motion_3",
)

_METADATA_KEYS = frozenset(
    {
        "split",
        "cohort_id",
        "dataset_name",
        "source_episode_index",
        "width",
        "height",
        "time_base_num",
        "time_base_den",
        "gop_size",
        "source_frame_sha256",
        "canonical_frame_sha256",
        "fixture_frame_sha256",
        "tool_identity",
        "encoder_identity",
        "decoder_identity",
        "extractor_identity",
        "episode_identity",
        "codec_contract",
        "codec_contract_sha256",
    }
)
_ARRAY_KEYS = frozenset(
    {
        "frame_index",
        "pts",
        "dts",
        "frame_type",
        "gop_id",
        "gop_position",
        "reference_index",
        "frame_validity",
        "frame_reason_code",
        "mv_cell_features",
        "mv_cell_mask",
        "rpe",
        "rpe_mask",
        "camera_motion_summary",
        "camera_motion_mask",
        "anchor_index_per_frame",
        "anchor_frame_index",
        "anchor_appearance",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "episode_uid",
        "camera_id",
        "source_kind",
        "episode_identity",
        "codec_contract",
        "codec_contract_sha256",
        "split",
        "cohort_id",
        "dataset_name",
        "source_episode_index",
        "num_frames",
        "num_anchors",
        "width",
        "height",
        "time_base_num",
        "time_base_den",
        "gop_size",
        "k_max",
        "frame_hashes",
        "identities",
        "arrays",
        "accepted_bitstream",
        "raw_motion_vector_audit",
    }
)
_BATCH_KEYS = frozenset(
    {
        "episode_uid",
        "camera_id",
        "source_kind",
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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_plain_component(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be a non-empty plain path component")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _deep_freeze(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} keys must be non-empty strings")
            result[key] = _deep_freeze(item, f"{name}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item, name) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite JSON data")
        return value
    raise TypeError(f"{name} must contain only JSON-compatible values")


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _frozen_array(
    value: Any,
    *,
    dtype: np.dtype | str,
    shape: tuple[int, ...],
    name: str,
    finite: bool = False,
) -> np.ndarray:
    expected_dtype = np.dtype(dtype)
    array = np.asarray(value)
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


def _validate_frame_reason_codes(
    frame_validity: np.ndarray,
    frame_reason_code: np.ndarray,
    active_mask: np.ndarray,
    anchor_available: np.ndarray,
    *,
    context: str,
) -> None:
    """Bind every active frame's reason to the first failed validity term."""

    validity_index = {name: index for index, name in enumerate(FRAME_VALIDITY_NAMES)}
    for frame in np.flatnonzero(active_mask):
        expected_name = "NONE"
        for validity_name, reason_name in INVALID_REASON_PRIORITY:
            if not bool(frame_validity[frame, validity_index[validity_name]]):
                expected_name = reason_name
                break
        if expected_name == "NONE" and not bool(anchor_available[frame]):
            expected_name = "I_FRAME_UNAVAILABLE"
        expected_code = REASON_CODES[expected_name]
        if int(frame_reason_code[frame]) != expected_code:
            raise ValueError(
                f"{context} frame_reason_code disagrees with validity priority "
                f"at frame {int(frame)}: expected {expected_name}"
            )


@dataclass(frozen=True, order=True)
class EpisodeBundleKey:
    episode_uid: str
    camera_id: str
    source_kind: str

    def __post_init__(self) -> None:
        _require_sha(self.episode_uid, "episode_uid")
        _require_plain_component(self.camera_id, "camera_id")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")

    @property
    def canonical(self) -> str:
        return f"{self.episode_uid}:{self.camera_id}:{self.source_kind}"


@dataclass(frozen=True)
class EpisodeBundle:
    """One immutable episode's bounded training facts.

    ``anchor_frame_index`` contains every available verified anchor in the
    episode.  A T8 view selects only the distinct anchors its frames name and
    fails instead of truncating if that set exceeds :data:`K_MAX`.
    """

    key: EpisodeBundleKey
    split: str
    cohort_id: str
    dataset_name: str
    source_episode_index: int
    width: int
    height: int
    time_base_num: int
    time_base_den: int
    gop_size: int
    source_frame_sha256: tuple[str, ...]
    canonical_frame_sha256: tuple[str, ...]
    fixture_frame_sha256: tuple[str, ...]
    tool_identity: Mapping[str, Any]
    encoder_identity: Mapping[str, Any]
    decoder_identity: Mapping[str, Any]
    extractor_identity: Mapping[str, Any]
    episode_identity: Mapping[str, Any]
    codec_contract: Mapping[str, Any]
    codec_contract_sha256: str
    frame_index: np.ndarray
    pts: np.ndarray
    dts: np.ndarray
    frame_type: np.ndarray
    gop_id: np.ndarray
    gop_position: np.ndarray
    reference_index: np.ndarray
    frame_validity: np.ndarray
    frame_reason_code: np.ndarray
    mv_cell_features: np.ndarray
    mv_cell_mask: np.ndarray
    rpe: np.ndarray
    rpe_mask: np.ndarray
    camera_motion_summary: np.ndarray
    camera_motion_mask: np.ndarray
    anchor_index_per_frame: np.ndarray
    anchor_frame_index: np.ndarray
    anchor_appearance: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeBundleKey):
            raise TypeError("key must be an EpisodeBundleKey")
        for name in ("split", "cohort_id", "dataset_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        _require_int(self.source_episode_index, "source_episode_index")
        _require_int(self.width, "width", minimum=1)
        _require_int(self.height, "height", minimum=1)
        _require_int(self.time_base_num, "time_base_num", minimum=1)
        _require_int(self.time_base_den, "time_base_den", minimum=1)
        _require_int(self.gop_size, "gop_size", minimum=1)

        frame_index = np.asarray(self.frame_index)
        if frame_index.dtype != np.dtype("int64") or frame_index.ndim != 1:
            raise TypeError("frame_index must be int64 [N]")
        num_frames = frame_index.shape[0]
        if num_frames == 0:
            raise ValueError("An episode bundle must contain at least one frame")
        anchor_index = np.asarray(self.anchor_frame_index)
        if anchor_index.dtype != np.dtype("int64") or anchor_index.ndim != 1:
            raise TypeError("anchor_frame_index must be int64 [A]")
        num_anchors = anchor_index.shape[0]

        vector_specs = {
            "frame_index": ("int64", (num_frames,), False),
            "pts": ("int64", (num_frames,), False),
            "dts": ("int64", (num_frames,), False),
            "frame_type": ("uint8", (num_frames,), False),
            "gop_id": ("int64", (num_frames,), False),
            "gop_position": ("int64", (num_frames,), False),
            "reference_index": ("int64", (num_frames,), False),
            "frame_validity": ("bool", (num_frames, len(FRAME_VALIDITY_NAMES)), False),
            "frame_reason_code": ("int16", (num_frames,), False),
            "mv_cell_features": (
                "float32",
                (num_frames, CELL_ROWS, CELL_COLUMNS, MV_CELL_WIDTH),
                True,
            ),
            "mv_cell_mask": ("bool", (num_frames, CELL_ROWS, CELL_COLUMNS), False),
            "rpe": (
                "float32",
                (num_frames, CELL_ROWS, CELL_COLUMNS, RPE_WIDTH),
                True,
            ),
            "rpe_mask": ("bool", (num_frames, CELL_ROWS, CELL_COLUMNS), False),
            "camera_motion_summary": ("float32", (num_frames, 4), True),
            "camera_motion_mask": ("bool", (num_frames,), False),
            "anchor_index_per_frame": ("int64", (num_frames,), False),
            "anchor_frame_index": ("int64", (num_anchors,), False),
            "anchor_appearance": (
                "float32",
                (num_anchors, 3, 256, 256),
                True,
            ),
        }
        for name, (dtype, shape, finite) in vector_specs.items():
            object.__setattr__(
                self,
                name,
                _frozen_array(
                    getattr(self, name),
                    dtype=dtype,
                    shape=shape,
                    name=name,
                    finite=finite,
                ),
            )
        for name in (
            "tool_identity",
            "encoder_identity",
            "decoder_identity",
            "extractor_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping")
            object.__setattr__(self, name, _deep_freeze(value, name))

        episode_identity = validate_episode_identity_preimage(self.episode_identity)
        if episode_uid_from_preimage(episode_identity) != self.key.episode_uid:
            raise ValueError("Episode identity preimage does not derive the bundle key UID")
        if (
            episode_identity["camera_id"] != self.key.camera_id
            or episode_identity["episode_index"] != self.source_episode_index
            or episode_identity["episode_length"] != num_frames
        ):
            raise ValueError("Episode identity preimage disagrees with bundle metadata")
        object.__setattr__(
            self, "episode_identity", _deep_freeze(episode_identity, "episode_identity")
        )
        codec_contract = validate_codec_contract(self.codec_contract)
        if codec_contract_sha256(codec_contract) != self.codec_contract_sha256:
            raise ValueError("Bundle codec_contract SHA-256 mismatch")
        contract_matches = {
            "camera_id": self.key.camera_id,
            "source_kind": self.key.source_kind,
            "width": self.width,
            "height": self.height,
            "time_base_num": self.time_base_num,
            "time_base_den": self.time_base_den,
            "gop_size": self.gop_size,
        }
        expected_contract = {
            "camera_id": codec_contract["camera_id"],
            "source_kind": codec_contract["source_kind"],
            "width": codec_contract["width"],
            "height": codec_contract["height"],
            "time_base_num": codec_contract["time_base"]["numerator"],
            "time_base_den": codec_contract["time_base"]["denominator"],
            "gop_size": codec_contract["gop_size"],
        }
        if contract_matches != expected_contract:
            raise ValueError("Bundle dimensions/time/GOP differ from codec_contract")
        for field, identity_name in (
            ("tool_identity", "tool_identity"),
            ("encoder_identity", "encoder_identity"),
            ("decoder_identity", "decoder_identity"),
            ("extractor_identity", "extractor_identity"),
        ):
            if _deep_thaw(getattr(self, field)) != codec_contract[identity_name]:
                raise ValueError(f"Bundle {field} differs from codec_contract")
        object.__setattr__(self, "codec_contract", _deep_freeze(codec_contract, "codec_contract"))
        _require_sha(self.codec_contract_sha256, "codec_contract_sha256")

        for name in (
            "source_frame_sha256",
            "canonical_frame_sha256",
            "fixture_frame_sha256",
        ):
            values = tuple(getattr(self, name))
            if len(values) != num_frames:
                raise ValueError(f"{name} must contain exactly N hashes")
            for index, value in enumerate(values):
                _require_sha(value, f"{name}[{index}]")
            object.__setattr__(self, name, values)
        self._validate_graph_and_masks()

    @property
    def num_frames(self) -> int:
        return int(self.frame_index.shape[0])

    @property
    def num_anchors(self) -> int:
        return int(self.anchor_frame_index.shape[0])

    def _validate_graph_and_masks(self) -> None:
        num_frames = self.num_frames
        if not np.array_equal(self.frame_index, np.arange(num_frames, dtype=np.int64)):
            raise ValueError("frame_index must be the complete canonical 0..N-1 row grid")
        if np.any(self.pts < 0) or np.any(self.dts < 0):
            raise ValueError("PTS and DTS must be non-negative")
        if num_frames > 1 and (
            np.any(self.pts[1:] <= self.pts[:-1]) or np.any(self.dts[1:] <= self.dts[:-1])
        ):
            raise ValueError("PTS and DTS must be strictly increasing")
        expected_timestamp = self.frame_index * 1000
        if not np.array_equal(self.pts, expected_timestamp) or not np.array_equal(
            self.dts, expected_timestamp
        ):
            raise ValueError("Episode PTS and DTS must equal frame_index * 1000")
        if not set(np.unique(self.frame_type)).issubset(NON_PAD_FRAME_TYPES):
            raise ValueError("Episode frame_type must contain only IDR, intra-non-IDR, or P")
        if not set(int(value) for value in np.unique(self.frame_reason_code)).issubset(
            REASON_CODES.values()
        ):
            raise ValueError("frame_reason_code contains an unknown code")
        _validate_frame_reason_codes(
            self.frame_validity,
            self.frame_reason_code,
            np.ones((num_frames,), dtype=np.bool_),
            self.anchor_index_per_frame >= 0,
            context="Episode bundle",
        )
        if np.any(self.gop_id < 0) or np.any(self.gop_position < 0):
            raise ValueError("GOP fields must be non-negative")
        if np.any(self.gop_position >= self.gop_size):
            raise ValueError("gop_position must be smaller than gop_size")
        if self.frame_type[0] != FRAME_TYPE_IDR:
            raise ValueError("The first episode frame must be an IDR")
        if self.codec_contract["fixture_mode"] == PRODUCTION_FIXTURE_MODE:
            expected_gop_id, expected_gop_position = np.divmod(self.frame_index, 30)
            expected_type = np.full((num_frames,), FRAME_TYPE_P, dtype=np.uint8)
            expected_type[expected_gop_position == 0] = FRAME_TYPE_IDR
            expected_reference = self.frame_index - 1
            expected_reference[expected_gop_position == 0] = -1
            if (
                not np.array_equal(self.gop_id, expected_gop_id)
                or not np.array_equal(self.gop_position, expected_gop_position)
                or not np.array_equal(self.frame_type, expected_type)
                or not np.array_equal(self.reference_index, expected_reference)
            ):
                raise ValueError("Production bundle requires IDR iff row%30==0, else P/ref=row-1")
        for index in range(num_frames):
            frame_type = int(self.frame_type[index])
            reference = int(self.reference_index[index])
            position = int(self.gop_position[index])
            if frame_type in {FRAME_TYPE_IDR, FRAME_TYPE_INTRA_NON_IDR}:
                if reference != -1:
                    raise ValueError("Intra frames must have reference_index=-1")
            elif reference != index - 1:
                raise ValueError("P frames must reference the episode-global predecessor")
            if frame_type == FRAME_TYPE_IDR and position != 0:
                raise ValueError("IDR frames must start a GOP at position zero")
            if index == 0:
                if int(self.gop_id[index]) != 0 or position != 0:
                    raise ValueError("The episode must start at GOP 0 position 0")
                continue
            previous_gop = int(self.gop_id[index - 1])
            current_gop = int(self.gop_id[index])
            if current_gop == previous_gop:
                if position != int(self.gop_position[index - 1]) + 1:
                    raise ValueError("Within-GOP positions must be contiguous")
                if frame_type == FRAME_TYPE_IDR:
                    raise ValueError("IDR may only occur on a GOP transition")
            elif current_gop == previous_gop + 1:
                if frame_type != FRAME_TYPE_IDR or position != 0:
                    raise ValueError("A GOP transition requires an IDR at position zero")
            else:
                raise ValueError("gop_id must stay equal or increase exactly once")
            if frame_type == FRAME_TYPE_P and current_gop != previous_gop:
                raise ValueError("P-frame predecessor may not cross a GOP boundary")

        frame_valid = self.frame_validity[:, 0]
        mv_valid = self.frame_validity[:, 5]
        rpe_valid = self.frame_validity[:, 6]
        alignment_valid = self.frame_validity[:, 4]
        allowed_mv = (frame_valid & mv_valid)[:, None, None]
        if np.any(self.mv_cell_mask & ~allowed_mv):
            raise ValueError("mv_cell_mask contradicts frame/mv validity")
        if np.any(self.mv_cell_features[~self.mv_cell_mask] != 0):
            raise ValueError("Masked MV cell features must be exact numeric zero")
        allowed_rpe = (frame_valid & alignment_valid & rpe_valid)[:, None, None]
        if np.any(self.rpe_mask & ~allowed_rpe):
            raise ValueError("rpe_mask contradicts frame/alignment/RPE validity")
        if np.any(self.rpe[~self.rpe_mask] != 0):
            raise ValueError("Masked RPE values must be exact numeric zero")
        if np.any(self.camera_motion_mask & ~frame_valid):
            raise ValueError("camera_motion_mask contradicts frame validity")
        if np.any(self.camera_motion_summary[~self.camera_motion_mask] != 0):
            raise ValueError("Masked camera motion must be exact numeric zero")
        if np.any(self.anchor_appearance < 0) or np.any(self.anchor_appearance > 1):
            raise ValueError("anchor_appearance must lie in [0,1]")

        anchors = self.anchor_frame_index
        if np.any(anchors < 0) or np.any(anchors >= num_frames):
            raise ValueError("Anchor frame indices must lie inside the episode")
        if anchors.size and np.any(anchors[1:] <= anchors[:-1]):
            raise ValueError("Anchor frame indices must be strictly increasing")
        if anchors.size and np.any(self.frame_type[anchors] != FRAME_TYPE_IDR):
            raise ValueError("Every available anchor must refer to an IDR frame")
        named = self.anchor_index_per_frame
        if np.any(named < -1):
            raise ValueError("anchor_index_per_frame must contain -1 or an anchor frame index")
        anchor_set = set(int(value) for value in anchors)
        if any(int(value) not in anchor_set for value in named if value >= 0):
            raise ValueError("anchor_index_per_frame names an unavailable anchor")
        for frame, named_anchor in enumerate(named):
            available = anchors[anchors <= frame]
            expected = int(available[-1]) if available.size else -1
            if int(named_anchor) != expected:
                raise ValueError("anchor_index_per_frame must name the nearest available past IDR")

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "cohort_id": self.cohort_id,
            "dataset_name": self.dataset_name,
            "source_episode_index": self.source_episode_index,
            "width": self.width,
            "height": self.height,
            "time_base_num": self.time_base_num,
            "time_base_den": self.time_base_den,
            "gop_size": self.gop_size,
            "source_frame_sha256": list(self.source_frame_sha256),
            "canonical_frame_sha256": list(self.canonical_frame_sha256),
            "fixture_frame_sha256": list(self.fixture_frame_sha256),
            "tool_identity": _deep_thaw(self.tool_identity),
            "encoder_identity": _deep_thaw(self.encoder_identity),
            "decoder_identity": _deep_thaw(self.decoder_identity),
            "extractor_identity": _deep_thaw(self.extractor_identity),
            "episode_identity": _deep_thaw(self.episode_identity),
            "codec_contract": _deep_thaw(self.codec_contract),
            "codec_contract_sha256": self.codec_contract_sha256,
        }

    def arrays_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in _ARRAY_KEYS}

    def view(self, observation_row: int) -> "EpisodeT8View":
        if type(observation_row) is not int or not 0 <= observation_row < self.num_frames:
            raise IndexError("observation_row is outside the episode")
        begin = max(0, observation_row - HISTORY + 1)
        source = np.arange(begin, observation_row + 1, dtype=np.int64)
        left_pad = HISTORY - source.size
        destination = slice(left_pad, HISTORY)
        frame_mask = np.zeros((HISTORY,), dtype=np.bool_)
        frame_mask[destination] = True

        def padded(name: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
            result = np.zeros((HISTORY, *shape), dtype=dtype)
            result[destination] = getattr(self, name)[source]
            return result

        frame_index = padded("frame_index", (), "int64")
        frame_type = padded("frame_type", (), "uint8")
        frame_validity = padded("frame_validity", (len(FRAME_VALIDITY_NAMES),), "bool")
        frame_reason_code = padded("frame_reason_code", (), "int16")
        mv_cell_features = padded(
            "mv_cell_features", (CELL_ROWS, CELL_COLUMNS, MV_CELL_WIDTH), "float32"
        )
        mv_cell_mask = padded("mv_cell_mask", (CELL_ROWS, CELL_COLUMNS), "bool")
        rpe = padded("rpe", (CELL_ROWS, CELL_COLUMNS, RPE_WIDTH), "float32")
        rpe_mask = padded("rpe_mask", (CELL_ROWS, CELL_COLUMNS), "bool")

        named_anchors = self.anchor_index_per_frame[source]
        unique_anchors = np.unique(named_anchors[named_anchors >= 0])
        if unique_anchors.size > K_MAX:
            raise ValueError("A T8 view requires more than K_MAX anchors; truncation is forbidden")
        anchor_mask = np.zeros((K_MAX,), dtype=np.bool_)
        anchor_frame_index = np.zeros((K_MAX,), dtype=np.int64)
        anchor_appearance = np.zeros((K_MAX, 3, 256, 256), dtype=np.float32)
        frame_anchor_slot = np.full((HISTORY,), -1, dtype=np.int64)
        anchor_to_slot: dict[int, int] = {}
        global_anchor_to_row = {
            int(frame): row for row, frame in enumerate(self.anchor_frame_index)
        }
        for slot, frame in enumerate(unique_anchors):
            frame_value = int(frame)
            anchor_to_slot[frame_value] = slot
            anchor_mask[slot] = True
            anchor_frame_index[slot] = frame_value
            anchor_appearance[slot] = self.anchor_appearance[global_anchor_to_row[frame_value]]
        for offset, frame in enumerate(named_anchors, start=left_pad):
            if int(frame) >= 0:
                frame_anchor_slot[offset] = anchor_to_slot[int(frame)]

        metadata = np.zeros((HISTORY, METADATA_WIDTH), dtype=np.float32)
        current_pts = int(self.pts[observation_row])
        current_dts = int(self.dts[observation_row])
        time_scale = self.time_base_num / self.time_base_den
        for offset, frame in enumerate(source, start=left_pad):
            frame_value = int(frame)
            frame_type_value = int(self.frame_type[frame_value])
            metadata[offset, frame_type_value - 1] = np.float32(1.0)
            metadata[offset, 3] = np.float32((frame_value - observation_row) / 7.0)
            metadata[offset, 4] = np.float32(
                (int(self.pts[frame_value]) - current_pts) * time_scale
            )
            metadata[offset, 5] = np.float32(
                (int(self.dts[frame_value]) - current_dts) * time_scale
            )
            metadata[offset, 6] = np.float32(
                int(self.gop_position[frame_value]) / max(self.gop_size - 1, 1)
            )
            reference = int(self.reference_index[frame_value])
            metadata[offset, 7] = np.float32(
                (frame_value - reference) / max(self.gop_size, 1) if reference >= 0 else 0.0
            )
            metadata[offset, 8:15] = self.frame_validity[frame_value].astype(np.float32)
            camera_valid = bool(self.camera_motion_mask[frame_value])
            anchor_valid = int(self.anchor_index_per_frame[frame_value]) >= 0
            metadata[offset, 15] = np.float32(camera_valid)
            metadata[offset, 16] = np.float32(anchor_valid)
            if camera_valid:
                metadata[offset, 17:21] = self.camera_motion_summary[frame_value]

        view = EpisodeT8View(
            key=self.key,
            observation_row=np.int64(observation_row),
            frame_mask=frame_mask,
            frame_index=frame_index,
            frame_type=frame_type,
            frame_validity=frame_validity,
            frame_reason_code=frame_reason_code,
            mv_cell_features=mv_cell_features,
            mv_cell_mask=mv_cell_mask,
            rpe=rpe,
            rpe_mask=rpe_mask,
            metadata_features=metadata,
            anchor_appearance=anchor_appearance,
            anchor_mask=anchor_mask,
            anchor_frame_index=anchor_frame_index,
            frame_anchor_slot=frame_anchor_slot,
            sample_valid=np.bool_(np.all(frame_reason_code[destination] == REASON_CODES["NONE"])),
            sample_reason_code=np.int16(
                next(
                    (
                        int(code)
                        for code in frame_reason_code[destination]
                        if int(code) != REASON_CODES["NONE"]
                    ),
                    REASON_CODES["NONE"],
                )
            ),
        )
        validate_episode_view(view)
        return view


@dataclass(frozen=True)
class EpisodeT8View:
    key: EpisodeBundleKey
    observation_row: np.int64
    frame_mask: np.ndarray
    frame_index: np.ndarray
    frame_type: np.ndarray
    frame_validity: np.ndarray
    frame_reason_code: np.ndarray
    mv_cell_features: np.ndarray
    mv_cell_mask: np.ndarray
    rpe: np.ndarray
    rpe_mask: np.ndarray
    metadata_features: np.ndarray
    anchor_appearance: np.ndarray
    anchor_mask: np.ndarray
    anchor_frame_index: np.ndarray
    frame_anchor_slot: np.ndarray
    sample_valid: np.bool_
    sample_reason_code: np.int16

    def __post_init__(self) -> None:
        specs = {
            "observation_row": ("int64", (), False),
            "frame_mask": ("bool", (HISTORY,), False),
            "frame_index": ("int64", (HISTORY,), False),
            "frame_type": ("uint8", (HISTORY,), False),
            "frame_validity": ("bool", (HISTORY, len(FRAME_VALIDITY_NAMES)), False),
            "frame_reason_code": ("int16", (HISTORY,), False),
            "mv_cell_features": (
                "float32",
                (HISTORY, CELL_ROWS, CELL_COLUMNS, MV_CELL_WIDTH),
                True,
            ),
            "mv_cell_mask": ("bool", (HISTORY, CELL_ROWS, CELL_COLUMNS), False),
            "rpe": (
                "float32",
                (HISTORY, CELL_ROWS, CELL_COLUMNS, RPE_WIDTH),
                True,
            ),
            "rpe_mask": ("bool", (HISTORY, CELL_ROWS, CELL_COLUMNS), False),
            "metadata_features": ("float32", (HISTORY, METADATA_WIDTH), True),
            "anchor_appearance": ("float32", (K_MAX, 3, 256, 256), True),
            "anchor_mask": ("bool", (K_MAX,), False),
            "anchor_frame_index": ("int64", (K_MAX,), False),
            "frame_anchor_slot": ("int64", (HISTORY,), False),
            "sample_valid": ("bool", (), False),
            "sample_reason_code": ("int16", (), False),
        }
        for name, (dtype, shape, finite) in specs.items():
            object.__setattr__(
                self,
                name,
                _frozen_array(
                    getattr(self, name),
                    dtype=dtype,
                    shape=shape,
                    name=name,
                    finite=finite,
                ),
            )


def validate_episode_view(view: EpisodeT8View) -> None:
    if not isinstance(view.key, EpisodeBundleKey):
        raise TypeError("Episode view key must be an EpisodeBundleKey")
    observation_value = np.asarray(view.observation_row)
    if (
        observation_value.dtype != np.dtype("int64")
        or observation_value.shape != ()
        or int(observation_value) < 0
    ):
        raise ValueError("Episode view observation_row must be non-negative int64 scalar")
    if int(view.sample_reason_code) in TRANSPORT_NEUTRAL_REASON_CODES:
        if bool(view.sample_valid):
            raise ValueError("Transport-neutral episode view must be sample_valid=false")
        zero_names = (
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
        )
        if any(np.any(getattr(view, name) != 0) for name in zero_names):
            raise ValueError("Transport-neutral episode evidence must be exact zero/false")
        if np.any(view.frame_anchor_slot != -1):
            raise ValueError("Transport-neutral frame_anchor_slot must be -1")
        return
    mask = view.frame_mask
    if mask.any():
        first = int(np.flatnonzero(mask)[0])
        if not mask[first:].all():
            raise ValueError("frame_mask must be an exact left-padding suffix mask")
    else:
        raise ValueError("A T8 view must contain its observation frame")
    observation_row = int(view.observation_row)
    if observation_row < 0:
        raise ValueError("observation_row must be non-negative")
    active_count = int(mask.sum())
    expected_rows = np.arange(
        observation_row - active_count + 1, observation_row + 1, dtype=np.int64
    )
    if not np.array_equal(view.frame_index[mask], expected_rows):
        raise ValueError("Non-PAD frame_index must be the canonical T8 suffix ending at row")
    pad = ~mask
    numeric_pad_names = (
        "frame_index",
        "frame_type",
        "frame_validity",
        "frame_reason_code",
        "mv_cell_features",
        "mv_cell_mask",
        "rpe",
        "rpe_mask",
        "metadata_features",
    )
    for name in numeric_pad_names:
        if np.any(getattr(view, name)[pad] != 0):
            raise ValueError(f"PAD values in {name} must be exact numeric zero")
    if np.any(view.frame_anchor_slot[pad] != -1):
        raise ValueError("PAD frame_anchor_slot values must be -1")
    if np.any((view.frame_type[mask] < 1) | (view.frame_type[mask] > 3)):
        raise ValueError("Non-PAD frame types are invalid")
    if not set(int(value) for value in np.unique(view.frame_reason_code[mask])).issubset(
        REASON_CODES.values()
    ):
        raise ValueError("Non-PAD frame_reason_code contains an unknown code")
    _validate_frame_reason_codes(
        view.frame_validity,
        view.frame_reason_code,
        mask,
        view.frame_anchor_slot >= 0,
        context="Episode view",
    )
    active_reasons = view.frame_reason_code[mask]
    invalid_reasons = active_reasons[active_reasons != REASON_CODES["NONE"]]
    expected_sample_reason = (
        int(invalid_reasons[0]) if invalid_reasons.size else REASON_CODES["NONE"]
    )
    if int(view.sample_reason_code) != expected_sample_reason:
        raise ValueError("Episode view sample_reason_code disagrees with frame/anchor reason")
    if bool(view.sample_valid) != (expected_sample_reason == REASON_CODES["NONE"]):
        raise ValueError("Episode view sample_valid disagrees with frame/anchor reason")
    if np.any(view.mv_cell_features[~view.mv_cell_mask] != 0):
        raise ValueError("Masked MV cells must be exact zero")
    allowed_mv = (view.frame_validity[:, 0] & view.frame_validity[:, 5])[:, None, None]
    if np.any(view.mv_cell_mask & ~allowed_mv):
        raise ValueError("View MV mask contradicts frame/mv validity")
    if np.any(view.rpe[~view.rpe_mask] != 0):
        raise ValueError("Masked RPE cells must be exact zero")
    allowed_rpe = (
        view.frame_validity[:, 0] & view.frame_validity[:, 4] & view.frame_validity[:, 6]
    )[:, None, None]
    if np.any(view.rpe_mask & ~allowed_rpe):
        raise ValueError("View RPE mask contradicts frame/alignment/RPE validity")
    if np.any(view.anchor_appearance[~view.anchor_mask] != 0):
        raise ValueError("Unused anchor bank slots must be exact zero")
    if np.any(view.anchor_appearance < 0) or np.any(view.anchor_appearance > 1):
        raise ValueError("View anchor appearance must lie in [0,1]")
    if np.any(view.anchor_frame_index[~view.anchor_mask] != 0):
        raise ValueError("Unused anchor indices must be exact zero")
    if view.anchor_mask.any():
        count = int(view.anchor_mask.sum())
        if not view.anchor_mask[:count].all() or view.anchor_mask[count:].any():
            raise ValueError("anchor_mask must be a prefix mask")
        if count > 1 and np.any(
            view.anchor_frame_index[1:count] <= view.anchor_frame_index[: count - 1]
        ):
            raise ValueError("Active anchor indices must be strictly increasing")
    slots = view.frame_anchor_slot
    invalid_slot = (slots < -1) | (slots >= K_MAX)
    if invalid_slot.any():
        raise ValueError("frame_anchor_slot is outside [-1,K_MAX)")
    for index in np.flatnonzero(slots >= 0):
        if not view.anchor_mask[int(slots[index])]:
            raise ValueError("frame_anchor_slot names an inactive anchor")
        if view.anchor_frame_index[int(slots[index])] > view.frame_index[index]:
            raise ValueError("frame_anchor_slot names a future anchor")
    active_slots = slots[mask]
    if np.any(active_slots[1:] < active_slots[:-1]):
        raise ValueError("frame_anchor_slot must be causal and nondecreasing")
    used_slots = set(int(slot) for slot in active_slots if slot >= 0)
    if used_slots != set(range(int(view.anchor_mask.sum()))):
        raise ValueError("Every active anchor bank slot must be named by a frame")
    expected_anchor_valid = (slots >= 0).astype(np.float32)
    if not np.array_equal(view.metadata_features[:, 16], expected_anchor_valid):
        raise ValueError("metadata anchor validity disagrees with frame_anchor_slot")
    expected_types = np.zeros((HISTORY, 3), dtype=np.float32)
    expected_types[np.flatnonzero(mask), view.frame_type[mask].astype(np.int64) - 1] = 1.0
    if not np.array_equal(view.metadata_features[:, :3], expected_types):
        raise ValueError("Metadata frame indicators disagree with frame_type")
    expected_relative_rows = np.zeros((HISTORY,), dtype=np.float32)
    expected_relative_rows[mask] = (
        view.frame_index[mask].astype(np.float32) - np.float32(observation_row)
    ) / np.float32(7.0)
    if not np.array_equal(view.metadata_features[:, 3], expected_relative_rows):
        raise ValueError("Metadata relative frame index disagrees with the T8 row")
    if not np.array_equal(view.metadata_features[:, 8:15], view.frame_validity.astype(np.float32)):
        raise ValueError("Metadata validity columns disagree with frame_validity")
    if np.any(~np.isin(view.metadata_features[:, 15:17], (0.0, 1.0))):
        raise ValueError("Metadata camera/anchor validity must be binary")
    if np.any(view.metadata_features[view.metadata_features[:, 15] == 0, 17:21] != 0):
        raise ValueError("Invalid camera metadata must be exact zero")
    if np.any(view.metadata_features[mask, 4:6] > 0):
        raise ValueError("Historical PTS/DTS metadata may not be later than the observation")
    if np.any(view.metadata_features[mask, 6:8] < 0) or np.any(
        view.metadata_features[mask, 6:8] > 1
    ):
        raise ValueError("Normalized GOP/reference metadata must lie in [0,1]")
    if np.any(view.metadata_features[pad] != 0):
        raise ValueError("PAD metadata must be exact zero")


def make_neutral_episode_view(
    key: EpisodeBundleKey, observation_row: int, reason_code: int
) -> EpisodeT8View:
    """Build an online-only all-zero episode transport fallback."""

    if type(observation_row) is not int or observation_row < 0:
        raise ValueError("Neutral episode observation_row must be non-negative integer")
    if type(reason_code) is not int or reason_code not in TRANSPORT_NEUTRAL_REASON_CODES:
        raise ValueError("Neutral episode reason must be MISSING/SCHEMA/HASH")
    view = EpisodeT8View(
        key=key,
        observation_row=np.int64(observation_row),
        frame_mask=np.zeros((HISTORY,), dtype=np.bool_),
        frame_index=np.zeros((HISTORY,), dtype=np.int64),
        frame_type=np.zeros((HISTORY,), dtype=np.uint8),
        frame_validity=np.zeros((HISTORY, len(FRAME_VALIDITY_NAMES)), dtype=np.bool_),
        frame_reason_code=np.zeros((HISTORY,), dtype=np.int16),
        mv_cell_features=np.zeros(
            (HISTORY, CELL_ROWS, CELL_COLUMNS, MV_CELL_WIDTH), dtype=np.float32
        ),
        mv_cell_mask=np.zeros((HISTORY, CELL_ROWS, CELL_COLUMNS), dtype=np.bool_),
        rpe=np.zeros((HISTORY, CELL_ROWS, CELL_COLUMNS, RPE_WIDTH), dtype=np.float32),
        rpe_mask=np.zeros((HISTORY, CELL_ROWS, CELL_COLUMNS), dtype=np.bool_),
        metadata_features=np.zeros((HISTORY, METADATA_WIDTH), dtype=np.float32),
        anchor_appearance=np.zeros((K_MAX, 3, 256, 256), dtype=np.float32),
        anchor_mask=np.zeros((K_MAX,), dtype=np.bool_),
        anchor_frame_index=np.zeros((K_MAX,), dtype=np.int64),
        frame_anchor_slot=np.full((HISTORY,), -1, dtype=np.int64),
        sample_valid=np.bool_(False),
        sample_reason_code=np.int16(reason_code),
    )
    validate_episode_view(view)
    return view


def collate_episode_views_numpy(views: Sequence[EpisodeT8View]) -> dict[str, Any]:
    """Collate fixed-size T8/K2 views without exposing raw motion vectors."""

    if not views:
        raise ValueError("Cannot collate an empty episode-view batch")
    if len(views) > MAX_EPISODE_BATCH_SIZE:
        raise ValueError("Episode batch exceeds MAX_EPISODE_BATCH_SIZE")
    for view in views:
        validate_episode_view(view)

    array_names = (
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
    )
    transport_bytes = sum(
        np.asarray(getattr(view, name)).nbytes for view in views for name in array_names
    )
    if transport_bytes > MAX_EPISODE_TRANSPORT_BYTES:
        raise ValueError("Episode batch exceeds MAX_EPISODE_TRANSPORT_BYTES")

    def stack(name: str) -> np.ndarray:
        return np.stack([getattr(view, name) for view in views], axis=0)

    batch: dict[str, Any] = {
        "episode_uid": tuple(view.key.episode_uid for view in views),
        "camera_id": tuple(view.key.camera_id for view in views),
        "source_kind": tuple(view.key.source_kind for view in views),
        "observation_row": stack("observation_row"),
        "frame_mask": stack("frame_mask"),
        "frame_index": stack("frame_index"),
        "frame_type": stack("frame_type"),
        "frame_validity": stack("frame_validity"),
        "frame_reason_code": stack("frame_reason_code"),
        "mv_cell_features": stack("mv_cell_features"),
        "mv_cell_mask": stack("mv_cell_mask"),
        "rpe": stack("rpe"),
        "rpe_mask": stack("rpe_mask"),
        "metadata_features": stack("metadata_features"),
        "anchor_appearance": stack("anchor_appearance"),
        "anchor_mask": stack("anchor_mask"),
        "anchor_frame_index": stack("anchor_frame_index"),
        "frame_anchor_slot": stack("frame_anchor_slot"),
        "sample_valid": stack("sample_valid"),
        "sample_reason_code": stack("sample_reason_code"),
    }
    assert_episode_batch_conforms(batch)
    return batch


def assert_episode_batch_conforms(batch: Mapping[str, Any]) -> None:
    if not isinstance(batch, Mapping) or set(batch) != _BATCH_KEYS:
        raise ValueError("Episode batch keys do not match the episode-bundle-v1 contract")
    rows = np.asarray(batch["observation_row"])
    if rows.dtype != np.dtype("int64") or rows.ndim != 1 or rows.shape[0] == 0:
        raise ValueError("observation_row must be int64 [B] with B>0")
    batch_size = rows.shape[0]
    for name in ("episode_uid", "camera_id", "source_kind"):
        values = batch[name]
        if not isinstance(values, tuple) or len(values) != batch_size:
            raise ValueError(f"{name} must be a tuple of length B")
    for uid in batch["episode_uid"]:
        _require_sha(uid, "episode_uid")
    for camera in batch["camera_id"]:
        _require_plain_component(camera, "camera_id")
    if any(source not in SOURCE_KINDS for source in batch["source_kind"]):
        raise ValueError("Batch source_kind is invalid")

    specs = {
        "frame_mask": ("bool", (batch_size, HISTORY), False),
        "frame_index": ("int64", (batch_size, HISTORY), False),
        "frame_type": ("uint8", (batch_size, HISTORY), False),
        "frame_validity": (
            "bool",
            (batch_size, HISTORY, len(FRAME_VALIDITY_NAMES)),
            False,
        ),
        "frame_reason_code": ("int16", (batch_size, HISTORY), False),
        "mv_cell_features": (
            "float32",
            (batch_size, HISTORY, CELL_ROWS, CELL_COLUMNS, MV_CELL_WIDTH),
            True,
        ),
        "mv_cell_mask": (
            "bool",
            (batch_size, HISTORY, CELL_ROWS, CELL_COLUMNS),
            False,
        ),
        "rpe": (
            "float32",
            (batch_size, HISTORY, CELL_ROWS, CELL_COLUMNS, RPE_WIDTH),
            True,
        ),
        "rpe_mask": (
            "bool",
            (batch_size, HISTORY, CELL_ROWS, CELL_COLUMNS),
            False,
        ),
        "metadata_features": (
            "float32",
            (batch_size, HISTORY, METADATA_WIDTH),
            True,
        ),
        "anchor_appearance": (
            "float32",
            (batch_size, K_MAX, 3, 256, 256),
            True,
        ),
        "anchor_mask": ("bool", (batch_size, K_MAX), False),
        "anchor_frame_index": ("int64", (batch_size, K_MAX), False),
        "frame_anchor_slot": ("int64", (batch_size, HISTORY), False),
        "sample_valid": ("bool", (batch_size,), False),
        "sample_reason_code": ("int16", (batch_size,), False),
    }
    for name, (dtype, shape, finite) in specs.items():
        array = np.asarray(batch[name])
        if array.dtype != np.dtype(dtype) or array.shape != shape:
            raise ValueError(f"{name} must be {dtype} {shape}")
        if finite and not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
    for index in range(batch_size):
        view = EpisodeT8View(
            key=EpisodeBundleKey(
                batch["episode_uid"][index],
                batch["camera_id"][index],
                batch["source_kind"][index],
            ),
            observation_row=np.asarray(batch["observation_row"])[index],
            **{name: np.asarray(batch[name])[index] for name in specs},
        )
        validate_episode_view(view)


@dataclass(frozen=True)
class RawMotionVectorAudit:
    """Raw MV CSR data, returned only by the explicit store audit API."""

    offsets: np.ndarray
    fields: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets)
        if offsets.dtype != np.dtype("int64") or offsets.ndim != 1 or offsets.size < 2:
            raise TypeError("Raw MV audit offsets must be int64 [N+1]")
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("Raw MV offsets must start at zero and be monotonic")
        if np.any(np.diff(offsets) > MAX_MV_PER_FRAME):
            raise ValueError("Raw MV count exceeds MAX_MV_PER_FRAME")
        frozen_offsets = _frozen_array(
            offsets,
            dtype="int64",
            shape=offsets.shape,
            name="raw_mv_offsets",
        )
        if not isinstance(self.fields, Mapping) or set(self.fields) != set(MV_FIELD_DTYPES):
            raise ValueError("Raw MV audit requires the exact eleven fields")
        total = int(offsets[-1])
        frozen_fields: dict[str, np.ndarray] = {}
        for name, dtype in MV_FIELD_DTYPES.items():
            frozen_fields[name] = _frozen_array(
                self.fields[name],
                dtype=dtype,
                shape=(total,),
                name=f"raw_mv_fields.{name}",
            )
        object.__setattr__(self, "offsets", frozen_offsets)
        object.__setattr__(self, "fields", MappingProxyType(frozen_fields))


def _episode_from_components(
    *,
    episode_uid: str,
    camera_id: str,
    source_kind: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> EpisodeBundle:
    if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_KEYS:
        raise ValueError("metadata keys do not match the frozen episode-bundle contract")
    if not isinstance(arrays, Mapping) or set(arrays) != _ARRAY_KEYS:
        raise ValueError("array keys do not match the frozen episode-bundle contract")
    return EpisodeBundle(
        key=EpisodeBundleKey(episode_uid, camera_id, source_kind),
        **{name: metadata[name] for name in _METADATA_KEYS if not name.endswith("sha256")},
        source_frame_sha256=tuple(metadata["source_frame_sha256"]),
        canonical_frame_sha256=tuple(metadata["canonical_frame_sha256"]),
        fixture_frame_sha256=tuple(metadata["fixture_frame_sha256"]),
        codec_contract_sha256=metadata["codec_contract_sha256"],
        **dict(arrays),
    )


def _write_array(path: Path, value: np.ndarray, *, relative_path: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return {
        "relative_path": relative_path,
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def write_episode_bundle(
    root: str | Path,
    *,
    episode_uid: str,
    camera_id: str,
    source_kind: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    raw_mv_offsets: np.ndarray | None = None,
    raw_mv_fields: Mapping[str, np.ndarray] | None = None,
    accepted_bitstream: str | Path | None = None,
) -> Path:
    """Validate and atomically write one non-overwritable episode bundle."""

    bundle = _episode_from_components(
        episode_uid=episode_uid,
        camera_id=camera_id,
        source_kind=source_kind,
        metadata=metadata,
        arrays=arrays,
    )
    if (raw_mv_offsets is None) != (raw_mv_fields is None):
        raise ValueError("raw_mv_offsets and raw_mv_fields must be provided together")
    raw_audit = (
        None
        if raw_mv_offsets is None
        else RawMotionVectorAudit(raw_mv_offsets, raw_mv_fields or {})
    )
    if bundle.codec_contract["fixture_mode"] == PRODUCTION_FIXTURE_MODE and (
        raw_audit is None or accepted_bitstream is None
    ):
        raise ValueError("Production episode bundles require raw MV audit and accepted bitstream")
    if raw_audit is not None and raw_audit.offsets.shape != (bundle.num_frames + 1,):
        raise ValueError("Raw MV offsets must have shape [N+1]")

    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    destination = root_path / bundle.key.episode_uid / bundle.key.camera_id / bundle.key.source_kind
    if destination.exists():
        raise FileExistsError(f"Episode bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{source_kind}.tmp-", dir=destination.parent))
    try:
        descriptors: dict[str, Any] = {}
        for name in sorted(_ARRAY_KEYS):
            relative = f"arrays/{name}.npy"
            descriptors[name] = _write_array(
                stage / relative,
                getattr(bundle, name),
                relative_path=relative,
            )
        raw_descriptor: dict[str, Any] | None = None
        if raw_audit is not None:
            offsets_relative = "audit/raw_mv_offsets.npy"
            raw_descriptor = {
                "offsets": _write_array(
                    stage / offsets_relative,
                    raw_audit.offsets,
                    relative_path=offsets_relative,
                ),
                "fields": {},
            }
            for name in MV_FIELD_DTYPES:
                relative = f"audit/raw_mv_{name}.npy"
                raw_descriptor["fields"][name] = _write_array(
                    stage / relative,
                    raw_audit.fields[name],
                    relative_path=relative,
                )

        bitstream_descriptor: dict[str, Any] | None = None
        if accepted_bitstream is not None:
            source_bitstream = Path(accepted_bitstream).resolve(strict=True)
            if not source_bitstream.is_file():
                raise ValueError("accepted_bitstream must be a regular file")
            relative = "artifacts/accepted_bitstream.bin"
            destination_bitstream = stage / relative
            destination_bitstream.parent.mkdir(parents=True, exist_ok=True)
            with source_bitstream.open("rb") as source, destination_bitstream.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            bitstream_descriptor = {
                "relative_path": relative,
                "sha256": _sha256_path(destination_bitstream),
                "bytes": destination_bitstream.stat().st_size,
                "dtype": "bytes",
                "shape": [],
            }

        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "episode_uid": bundle.key.episode_uid,
            "camera_id": bundle.key.camera_id,
            "source_kind": bundle.key.source_kind,
            "episode_identity": _deep_thaw(bundle.episode_identity),
            "codec_contract": _deep_thaw(bundle.codec_contract),
            "codec_contract_sha256": bundle.codec_contract_sha256,
            "split": bundle.split,
            "cohort_id": bundle.cohort_id,
            "dataset_name": bundle.dataset_name,
            "source_episode_index": bundle.source_episode_index,
            "num_frames": bundle.num_frames,
            "num_anchors": bundle.num_anchors,
            "width": bundle.width,
            "height": bundle.height,
            "time_base_num": bundle.time_base_num,
            "time_base_den": bundle.time_base_den,
            "gop_size": bundle.gop_size,
            "k_max": K_MAX,
            "frame_hashes": {
                "source": list(bundle.source_frame_sha256),
                "canonical": list(bundle.canonical_frame_sha256),
                "fixture": list(bundle.fixture_frame_sha256),
            },
            "identities": {
                "tool": _deep_thaw(bundle.tool_identity),
                "encoder": _deep_thaw(bundle.encoder_identity),
                "decoder": _deep_thaw(bundle.decoder_identity),
                "extractor": _deep_thaw(bundle.extractor_identity),
            },
            "arrays": descriptors,
            "accepted_bitstream": bitstream_descriptor,
            "raw_motion_vector_audit": raw_descriptor,
        }
        manifest_path = stage / "bundle.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        for file_path in stage.rglob("*"):
            if file_path.is_file():
                file_path.chmod(0o444)
        for directory in sorted((path for path in stage.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o555)
        stage.chmod(0o555)
        if destination.exists():
            raise FileExistsError(f"Episode bundle already exists: {destination}")
        os.rename(stage, destination)
        return destination / "bundle.json"
    except Exception:
        if stage.exists():
            for directory in [stage, *stage.rglob("*")]:
                if directory.is_dir():
                    try:
                        directory.chmod(0o755)
                    except OSError:
                        pass
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_descriptor(
    descriptor: Any, *, name: str, expected_relative: str
) -> Mapping[str, Any]:
    fields = {"relative_path", "sha256", "bytes", "dtype", "shape"}
    if not isinstance(descriptor, Mapping) or set(descriptor) != fields:
        raise ValueError(f"{name} descriptor fields do not match contract")
    if descriptor["relative_path"] != expected_relative:
        raise ValueError(f"{name} descriptor path is not canonical")
    _require_sha(descriptor["sha256"], f"{name}.sha256")
    _require_int(descriptor["bytes"], f"{name}.bytes")
    if not isinstance(descriptor["dtype"], str):
        raise ValueError(f"{name}.dtype must be a string")
    if not isinstance(descriptor["shape"], list) or any(
        type(item) is not int or item < 0 for item in descriptor["shape"]
    ):
        raise ValueError(f"{name}.shape must be a list of non-negative integers")
    return descriptor


class EpisodeBundleStore:
    """Fail-closed loader rooted at immutable, identity-qualified bundles."""

    def __init__(
        self,
        root: str | Path,
        *,
        manifest_sha256_by_key: Mapping[EpisodeBundleKey, str] | None = None,
    ):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Episode bundle root must be a directory")
        pins: dict[EpisodeBundleKey, str] = {}
        for key, value in (manifest_sha256_by_key or {}).items():
            if not isinstance(key, EpisodeBundleKey):
                raise TypeError("Manifest pins must be keyed by EpisodeBundleKey")
            pins[key] = _require_sha(value, f"manifest pin for {key.canonical}")
        self._manifest_pins = MappingProxyType(pins)

    @property
    def keys(self) -> frozenset[EpisodeBundleKey]:
        """Return the immutable key closure supplied by the validated index."""

        return frozenset(self._manifest_pins)

    def _bundle_dir(self, key: EpisodeBundleKey) -> Path:
        if not isinstance(key, EpisodeBundleKey):
            raise TypeError("key must be an EpisodeBundleKey")
        relative = Path(key.episode_uid) / key.camera_id / key.source_kind
        cursor = self.root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Symlinks are forbidden in episode bundle paths")
        resolved = (self.root / relative).resolve(strict=True)
        if self.root not in resolved.parents:
            raise ValueError("Episode bundle path escapes its root")
        if not resolved.is_dir():
            raise ValueError("Episode bundle path is not a directory")
        return resolved

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
        if not isinstance(value, dict):
            raise ValueError("Episode bundle manifest must contain an object")
        return value

    @staticmethod
    def _load_array(bundle_dir: Path, descriptor: Mapping[str, Any]) -> np.ndarray:
        relative = Path(descriptor["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Array descriptor path escapes the bundle")
        cursor = bundle_dir
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Symlink array artifacts are forbidden")
        path = (bundle_dir / relative).resolve(strict=True)
        if bundle_dir not in path.parents:
            raise ValueError("Array descriptor path escapes the bundle")
        if path.stat().st_size != descriptor["bytes"]:
            raise ValueError("Episode bundle array byte count mismatch")
        if _sha256_path(path) != descriptor["sha256"]:
            raise ValueError("Episode bundle array SHA-256 mismatch")
        value = np.load(path, allow_pickle=False)
        if str(value.dtype) != descriptor["dtype"]:
            raise ValueError("Episode bundle array dtype differs from descriptor")
        if list(value.shape) != descriptor["shape"]:
            raise ValueError("Episode bundle array shape differs from descriptor")
        return value

    def _manifest(self, key: EpisodeBundleKey) -> tuple[Path, dict[str, Any]]:
        bundle_dir = self._bundle_dir(key)
        manifest_path = bundle_dir / "bundle.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("Episode bundle manifest is missing or a symlink")
        pin = self._manifest_pins.get(key)
        if pin is not None and _sha256_path(manifest_path) != pin:
            raise ValueError("Episode bundle manifest SHA-256 mismatch")
        manifest = self._load_json(manifest_path)
        if set(manifest) != _MANIFEST_KEYS:
            raise ValueError("Episode bundle manifest keys do not match episode-bundle-v1")
        if manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported episode bundle schema")
        if manifest["k_max"] != K_MAX:
            raise ValueError("Episode bundle K_MAX differs from the frozen contract")
        if (
            manifest["episode_uid"] != key.episode_uid
            or manifest["camera_id"] != key.camera_id
            or manifest["source_kind"] != key.source_kind
        ):
            raise ValueError("Episode bundle manifest identity differs from its path/key")
        episode_identity = validate_episode_identity_preimage(manifest["episode_identity"])
        if episode_uid_from_preimage(episode_identity) != key.episode_uid:
            raise ValueError("Episode bundle identity preimage does not derive its key")
        codec_contract = validate_codec_contract(manifest["codec_contract"])
        if codec_contract_sha256(codec_contract) != manifest["codec_contract_sha256"]:
            raise ValueError("Episode bundle codec_contract hash mismatch")
        if (
            codec_contract["camera_id"] != key.camera_id
            or codec_contract["source_kind"] != key.source_kind
        ):
            raise ValueError("Episode bundle codec_contract differs from its key")
        if codec_contract["fixture_mode"] == PRODUCTION_FIXTURE_MODE and (
            manifest["accepted_bitstream"] is None or manifest["raw_motion_vector_audit"] is None
        ):
            raise ValueError(
                "Production episode bundle is missing raw MV audit or accepted bitstream"
            )
        return bundle_dir, manifest

    def identity(self, key: EpisodeBundleKey) -> Mapping[str, Any]:
        """Read only the hash-bound identity surface, without loading arrays."""

        _bundle_dir, manifest = self._manifest(key)
        return _deep_freeze(
            {
                "episode_identity": manifest["episode_identity"],
                "codec_contract": manifest["codec_contract"],
                "codec_contract_sha256": manifest["codec_contract_sha256"],
            },
            "bundle_identity",
        )

    def get(self, key: EpisodeBundleKey) -> EpisodeBundle:
        bundle_dir, manifest = self._manifest(key)
        descriptors = manifest["arrays"]
        if not isinstance(descriptors, Mapping) or set(descriptors) != _ARRAY_KEYS:
            raise ValueError("Episode bundle array descriptors do not match contract")
        arrays: dict[str, np.ndarray] = {}
        for name in _ARRAY_KEYS:
            expected_relative = f"arrays/{name}.npy"
            descriptor = _validate_descriptor(
                descriptors[name], name=f"arrays.{name}", expected_relative=expected_relative
            )
            arrays[name] = self._load_array(bundle_dir, descriptor)
        frame_hashes = manifest["frame_hashes"]
        identities = manifest["identities"]
        if not isinstance(frame_hashes, Mapping) or set(frame_hashes) != {
            "source",
            "canonical",
            "fixture",
        }:
            raise ValueError("Episode bundle frame_hashes block is malformed")
        if not isinstance(identities, Mapping) or set(identities) != {
            "tool",
            "encoder",
            "decoder",
            "extractor",
        }:
            raise ValueError("Episode bundle identities block is malformed")
        metadata = {
            "split": manifest["split"],
            "cohort_id": manifest["cohort_id"],
            "dataset_name": manifest["dataset_name"],
            "source_episode_index": manifest["source_episode_index"],
            "width": manifest["width"],
            "height": manifest["height"],
            "time_base_num": manifest["time_base_num"],
            "time_base_den": manifest["time_base_den"],
            "gop_size": manifest["gop_size"],
            "source_frame_sha256": frame_hashes["source"],
            "canonical_frame_sha256": frame_hashes["canonical"],
            "fixture_frame_sha256": frame_hashes["fixture"],
            "tool_identity": identities["tool"],
            "encoder_identity": identities["encoder"],
            "decoder_identity": identities["decoder"],
            "extractor_identity": identities["extractor"],
            "episode_identity": manifest["episode_identity"],
            "codec_contract": manifest["codec_contract"],
            "codec_contract_sha256": manifest["codec_contract_sha256"],
        }
        bundle = _episode_from_components(
            episode_uid=key.episode_uid,
            camera_id=key.camera_id,
            source_kind=key.source_kind,
            metadata=metadata,
            arrays=arrays,
        )
        if manifest["num_frames"] != bundle.num_frames:
            raise ValueError("Manifest num_frames differs from episode arrays")
        if manifest["num_anchors"] != bundle.num_anchors:
            raise ValueError("Manifest num_anchors differs from episode arrays")
        return bundle

    def get_view(self, key: EpisodeBundleKey, observation_row: int) -> EpisodeT8View:
        return self.get(key).view(observation_row)

    def audit_raw_motion_vectors(
        self, key: EpisodeBundleKey, *, frame_index: int | None = None
    ) -> RawMotionVectorAudit:
        bundle_dir, manifest = self._manifest(key)
        descriptor = manifest["raw_motion_vector_audit"]
        if descriptor is None:
            raise KeyError(f"Bundle has no raw MV audit artifact: {key.canonical}")
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"offsets", "fields"}:
            raise ValueError("Raw MV audit descriptor is malformed")
        fields = descriptor["fields"]
        if not isinstance(fields, Mapping) or set(fields) != set(MV_FIELD_DTYPES):
            raise ValueError("Raw MV field descriptors do not match contract")
        offsets_descriptor = _validate_descriptor(
            descriptor["offsets"],
            name="raw_motion_vector_audit.offsets",
            expected_relative="audit/raw_mv_offsets.npy",
        )
        offsets = self._load_array(bundle_dir, offsets_descriptor)
        values: dict[str, np.ndarray] = {}
        for name in MV_FIELD_DTYPES:
            field_descriptor = _validate_descriptor(
                fields[name],
                name=f"raw_motion_vector_audit.fields.{name}",
                expected_relative=f"audit/raw_mv_{name}.npy",
            )
            values[name] = self._load_array(bundle_dir, field_descriptor)
        audit = RawMotionVectorAudit(offsets, values)
        if audit.offsets.shape != (int(manifest["num_frames"]) + 1,):
            raise ValueError("Raw MV audit offsets do not span the episode")
        if frame_index is None:
            return audit
        if type(frame_index) is not int or not 0 <= frame_index < manifest["num_frames"]:
            raise IndexError("Raw MV audit frame_index is outside the episode")
        begin = int(audit.offsets[frame_index])
        end = int(audit.offsets[frame_index + 1])
        return RawMotionVectorAudit(
            np.asarray([0, end - begin], dtype=np.int64),
            {name: value[begin:end] for name, value in audit.fields.items()},
        )

    def audit_accepted_bitstream(self, key: EpisodeBundleKey) -> Path:
        """Verify and return the retained bitstream path for explicit audit use."""

        bundle_dir, manifest = self._manifest(key)
        descriptor = manifest["accepted_bitstream"]
        if descriptor is None:
            raise KeyError(f"Bundle has no accepted bitstream: {key.canonical}")
        descriptor = _validate_descriptor(
            descriptor,
            name="accepted_bitstream",
            expected_relative="artifacts/accepted_bitstream.bin",
        )
        if descriptor["dtype"] != "bytes" or descriptor["shape"] != []:
            raise ValueError("Accepted bitstream descriptor type is malformed")
        path = bundle_dir / descriptor["relative_path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Accepted bitstream is missing or a symlink")
        if path.stat().st_size != descriptor["bytes"]:
            raise ValueError("Accepted bitstream byte count mismatch")
        if _sha256_path(path) != descriptor["sha256"]:
            raise ValueError("Accepted bitstream SHA-256 mismatch")
        return path


class InMemoryEpisodeBundleStore:
    """Small deterministic store for synthetic tests without filesystem I/O."""

    def __init__(self, bundles: Sequence[EpisodeBundle]):
        mapping: dict[EpisodeBundleKey, EpisodeBundle] = {}
        for bundle in bundles:
            if bundle.key in mapping:
                raise ValueError(f"Duplicate episode bundle: {bundle.key.canonical}")
            mapping[bundle.key] = bundle
        self._bundles = MappingProxyType(mapping)

    @property
    def keys(self) -> frozenset[EpisodeBundleKey]:
        return frozenset(self._bundles)

    def get(self, key: EpisodeBundleKey) -> EpisodeBundle:
        try:
            return self._bundles[key]
        except KeyError as error:
            raise KeyError(f"Missing episode bundle: {key.canonical}") from error

    def get_view(self, key: EpisodeBundleKey, observation_row: int) -> EpisodeT8View:
        return self.get(key).view(observation_row)

    def identity(self, key: EpisodeBundleKey) -> Mapping[str, Any]:
        bundle = self.get(key)
        return _deep_freeze(
            {
                "episode_identity": _deep_thaw(bundle.episode_identity),
                "codec_contract": _deep_thaw(bundle.codec_contract),
                "codec_contract_sha256": bundle.codec_contract_sha256,
            },
            "bundle_identity",
        )


def make_synthetic_episode_bundle(
    *,
    num_frames: int = 40,
    episode_uid: str = "a" * 64,
    camera_id: str = "cam_left_high",
    source_kind: str = SOURCE_KIND_NATIVE_CANONICAL,
    gop_size: int = 30,
) -> EpisodeBundle:
    """Return a deterministic CPU-only bundle for tests and API smoke checks."""

    _require_int(num_frames, "num_frames", minimum=1)
    _require_int(gop_size, "gop_size", minimum=1)
    rows = np.arange(num_frames, dtype=np.int64)
    gop_id, gop_position = np.divmod(rows, gop_size)
    frame_type = np.full((num_frames,), FRAME_TYPE_P, dtype=np.uint8)
    frame_type[gop_position == 0] = FRAME_TYPE_IDR
    reference = rows - 1
    reference[gop_position == 0] = -1
    anchors = rows[gop_position == 0].copy()
    anchor_per_frame = anchors[np.searchsorted(anchors, rows, side="right") - 1]
    validity = np.ones((num_frames, len(FRAME_VALIDITY_NAMES)), dtype=np.bool_)
    mv = np.zeros((num_frames, CELL_ROWS, CELL_COLUMNS, MV_CELL_WIDTH), dtype=np.float32)
    mv[..., 0] = rows[:, None, None]
    mv_mask = np.ones((num_frames, CELL_ROWS, CELL_COLUMNS), dtype=np.bool_)
    rpe = np.zeros((num_frames, CELL_ROWS, CELL_COLUMNS, RPE_WIDTH), dtype=np.float32)
    rpe[..., 0] = np.float32(1.0)
    rpe_mask = np.ones((num_frames, CELL_ROWS, CELL_COLUMNS), dtype=np.bool_)
    camera = np.zeros((num_frames, 4), dtype=np.float32)
    camera[:, 0] = rows.astype(np.float32) / max(num_frames - 1, 1)
    camera_mask = np.ones((num_frames,), dtype=np.bool_)
    appearance = np.zeros((anchors.size, 3, 256, 256), dtype=np.float32)
    for index in range(anchors.size):
        appearance[index] = np.float32((index + 1) / (anchors.size + 1))

    def hashes(label: str) -> tuple[str, ...]:
        return tuple(
            sha256(f"synthetic:{label}:{index}".encode("utf-8")).hexdigest()
            for index in range(num_frames)
        )

    identity = {"name": "synthetic", "version": "1", "sha256": "d" * 64}
    episode_identity = validate_episode_identity_preimage(
        {
            "schema_version": EPISODE_IDENTITY_SCHEMA_VERSION,
            "source_id": f"synthetic-{episode_uid[:16]}",
            "canonical_dataset_identity": {
                "info_sha256": "e" * 64,
                "modality_sha256": "f" * 64,
            },
            "episode_index": 0,
            "episode_length": num_frames,
            "camera_id": camera_id,
            "parquet": {
                "relative_path": "synthetic/episode.parquet",
                "bytes": 0,
                "sha256": "1" * 64,
            },
            "video": {
                "relative_path": "synthetic/cam_left_high.mp4",
                "bytes": 0,
                "sha256": "2" * 64,
            },
        }
    )
    derived_uid = episode_uid_from_preimage(episode_identity)
    codec_contract = make_codec_contract(
        fixture_mode=SYNTHETIC_FIXTURE_MODE,
        source_kind=source_kind,
        gop_size=gop_size,
        identity=identity,
    )
    return EpisodeBundle(
        key=EpisodeBundleKey(derived_uid, camera_id, source_kind),
        split="synthetic",
        cohort_id="synthetic",
        dataset_name="synthetic",
        source_episode_index=0,
        width=640,
        height=480,
        time_base_num=1,
        time_base_den=30_000,
        gop_size=gop_size,
        source_frame_sha256=hashes("source"),
        canonical_frame_sha256=hashes("canonical"),
        fixture_frame_sha256=hashes("fixture"),
        tool_identity=identity,
        encoder_identity=identity,
        decoder_identity=identity,
        extractor_identity=identity,
        episode_identity=episode_identity,
        codec_contract=codec_contract,
        codec_contract_sha256=codec_contract_sha256(codec_contract),
        frame_index=rows,
        pts=rows * 1000,
        dts=rows * 1000,
        frame_type=frame_type,
        gop_id=gop_id.astype(np.int64),
        gop_position=gop_position.astype(np.int64),
        reference_index=reference,
        frame_validity=validity,
        frame_reason_code=np.zeros((num_frames,), dtype=np.int16),
        mv_cell_features=mv,
        mv_cell_mask=mv_mask,
        rpe=rpe,
        rpe_mask=rpe_mask,
        camera_motion_summary=camera,
        camera_motion_mask=camera_mask,
        anchor_index_per_frame=anchor_per_frame.astype(np.int64),
        anchor_frame_index=anchors.astype(np.int64),
        anchor_appearance=appearance,
    )


# Explicit aliases make the transport naming discoverable without widening the
# contract or adding raw MV data to it.
EpisodeBundleView = EpisodeT8View
collate_episode_bundle_numpy = collate_episode_views_numpy
assert_episode_bundle_batch_conforms = assert_episode_batch_conforms


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "FRAME_TYPE_IDR",
    "FRAME_TYPE_INTRA_NON_IDR",
    "FRAME_TYPE_P",
    "FRAME_TYPE_PAD",
    "FRAME_VALIDITY_NAMES",
    "K_MAX",
    "METADATA_FEATURE_NAMES",
    "METADATA_WIDTH",
    "MAX_EPISODE_BATCH_SIZE",
    "MAX_EPISODE_TRANSPORT_BYTES",
    "MV_CELL_WIDTH",
    "SOURCE_KINDS",
    "SOURCE_KIND_NATIVE_CANONICAL",
    "SOURCE_KIND_RAW_JPEG",
    "EpisodeBundle",
    "EpisodeBundleKey",
    "EpisodeBundleStore",
    "EpisodeBundleView",
    "EpisodeT8View",
    "InMemoryEpisodeBundleStore",
    "RawMotionVectorAudit",
    "assert_episode_batch_conforms",
    "assert_episode_bundle_batch_conforms",
    "collate_episode_bundle_numpy",
    "collate_episode_views_numpy",
    "make_synthetic_episode_bundle",
    "validate_episode_view",
    "write_episode_bundle",
]
