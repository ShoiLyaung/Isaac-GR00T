"""Hash-bound, stateless causal-prefix materializer for online evaluation.

The callable returned by :func:`build_prefix_materializer` reuses the exact
production x264 encoder and evidence array builders.  Every invocation sees
only the supplied episode prefix, writes its two deterministic encodes into a
temporary directory, and returns only the immutable model-facing T8 view.
No bitstream or episode state survives the call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
from gr00t.data.codec_evidence.contract import HISTORY, REASON_CODES
from gr00t.data.codec_evidence.episode_bundle import (
    FRAME_VALIDITY_NAMES,
    K_MAX,
    METADATA_WIDTH,
    SOURCE_KIND_NATIVE_CANONICAL,
    EpisodeBundleKey,
    EpisodeT8View,
    validate_episode_view,
)
from gr00t.data.codec_evidence.production import (
    CAMERA_ID,
    GOP,
    X264_CAUSAL_PARAMETERS,
    build_episode_arrays,
    encode_h264_twice,
    extract_mvs_and_decoded_rgb,
    probe_packets,
    probe_x264_encoder_options,
    sha256_path,
    trace_h264_headers,
    validate_rgb_sequence,
)


FACTORY_SCHEMA_VERSION = "groot-causal-prefix-materializer-factory-v1"
TRAINING_CONTRACT_SCHEMA_VERSION = "groot-causal-codec-training-contract-v1"
EXPECTED_MATERIALIZER_ID = "causal-prefix-v1"
EXPECTED_SOURCE_KIND = SOURCE_KIND_NATIVE_CANONICAL
EXPECTED_HEIGHT = 480
EXPECTED_WIDTH = 640
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_CONTRACT_KEYS = {
    "schema_version",
    "online_prefix_equivalent",
    "materializer_id",
    "materializer_sha256",
    "source_kind",
    "camera_id",
    "evidence_hz",
    "history_length",
    "future_frame_fallback_allowed",
}
_EPISODE_ARRAY_KEYS = {
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


class PrefixMaterializerError(RuntimeError):
    """The factory contract or visible-prefix boundary is not exact."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PrefixMaterializerError(f"{name} must be a lowercase SHA-256")
    return value


def _module_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PrefixMaterializerError(f"factory dependency must be a regular file: {path}")
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_path(resolved),
        "bytes": resolved.stat().st_size,
    }


def _validate_contract(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(contract) is not dict:
        raise PrefixMaterializerError("causal materializer contract must be a plain dict")
    keys = set(contract)
    if keys != _TRAINING_CONTRACT_KEYS:
        raise PrefixMaterializerError("causal materializer contract keys are not exact")
    if contract.get("schema_version") != TRAINING_CONTRACT_SCHEMA_VERSION:
        raise PrefixMaterializerError("causal materializer contract schema is not exact")
    if (
        contract.get("online_prefix_equivalent") is not True
        or contract.get("future_frame_fallback_allowed") is not False
    ):
        raise PrefixMaterializerError("causal materializer contract is not prefix-safe")
    expected = {
        "materializer_id": EXPECTED_MATERIALIZER_ID,
        "source_kind": EXPECTED_SOURCE_KIND,
        "camera_id": CAMERA_ID,
        "evidence_hz": 30,
        "history_length": HISTORY,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise PrefixMaterializerError(f"causal materializer contract {name} is not exact")
    factory_path = Path(__file__)
    factory_identity = _module_identity(factory_path)
    materializer_sha256 = _require_sha256(
        contract.get("materializer_sha256"), "materializer_sha256"
    )
    if materializer_sha256 != factory_identity["sha256"]:
        raise PrefixMaterializerError("contract materializer SHA-256 differs from this factory")
    production_path = Path(sys_module_file(build_episode_arrays))
    episode_path = Path(sys_module_file(validate_episode_view))
    codec_contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind="NATIVE_CANONICAL_DECODED_DERIVED",
    )
    identity = {
        "schema_version": FACTORY_SCHEMA_VERSION,
        "materializer_id": EXPECTED_MATERIALIZER_ID,
        "factory": factory_identity,
        "contract_sha256": _canonical_sha256(contract),
        "source_kind": EXPECTED_SOURCE_KIND,
        "camera_id": CAMERA_ID,
        "history_length": HISTORY,
        "frame_shape": [EXPECTED_HEIGHT, EXPECTED_WIDTH, 3],
        "x264_parameters": list(X264_CAUSAL_PARAMETERS),
        "production_codec_contract_sha256": codec_contract_sha256(codec_contract),
        "production_module": _module_identity(production_path),
        "episode_view_module": _module_identity(episode_path),
        "stateless_reset_isolation": True,
        "future_rows_accepted": False,
        "retained_bitstream": False,
        "execution": "CPU_OFFLINE_PREFIX_REENCODE_REFERENCE",
    }
    return dict(contract), identity


def sys_module_file(value: Any) -> str:
    module_name = getattr(value, "__module__", None)
    if type(module_name) is not str:
        raise PrefixMaterializerError("factory dependency module identity is unavailable")
    module = __import__(module_name, fromlist=["__file__"])
    path = getattr(module, "__file__", None)
    if type(path) is not str:
        raise PrefixMaterializerError("factory dependency file identity is unavailable")
    return path


def episode_arrays_to_view(
    arrays: Mapping[str, np.ndarray],
    *,
    episode_uid: str,
    source_kind: str,
    observation_row: int,
) -> EpisodeT8View:
    """Construct the exact training-compatible T8 projection from production arrays."""

    if not isinstance(arrays, Mapping) or set(arrays) != _EPISODE_ARRAY_KEYS:
        raise PrefixMaterializerError("production episode arrays are not exact")
    episode_uid = _require_sha256(episode_uid, "episode_uid")
    if source_kind != EXPECTED_SOURCE_KIND:
        raise PrefixMaterializerError("online source_kind is not exact")
    frame_index_all = np.asarray(arrays["frame_index"])
    if frame_index_all.dtype != np.int64 or frame_index_all.ndim != 1:
        raise PrefixMaterializerError("production frame_index must be int64 [N]")
    num_frames = int(frame_index_all.size)
    if type(observation_row) is not int or not 0 <= observation_row < num_frames:
        raise PrefixMaterializerError("observation row lies outside production arrays")
    if not np.array_equal(frame_index_all, np.arange(num_frames, dtype=np.int64)):
        raise PrefixMaterializerError("production frame index is not contiguous")

    begin = max(0, observation_row - HISTORY + 1)
    source = np.arange(begin, observation_row + 1, dtype=np.int64)
    left_pad = HISTORY - source.size
    destination = slice(left_pad, HISTORY)
    frame_mask = np.zeros((HISTORY,), dtype=np.bool_)
    frame_mask[destination] = True

    def padded(name: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
        value = np.asarray(arrays[name])
        if value.shape != (num_frames, *shape) or value.dtype != np.dtype(dtype):
            raise PrefixMaterializerError(f"production array {name} dtype/shape drifted")
        result = np.zeros((HISTORY, *shape), dtype=dtype)
        result[destination] = value[source]
        return result

    frame_index = padded("frame_index", (), "int64")
    frame_type = padded("frame_type", (), "uint8")
    frame_validity = padded("frame_validity", (len(FRAME_VALIDITY_NAMES),), "bool")
    frame_reason_code = padded("frame_reason_code", (), "int16")
    mv_cell_features = padded("mv_cell_features", (2, 2, 11), "float32")
    mv_cell_mask = padded("mv_cell_mask", (2, 2), "bool")
    rpe = padded("rpe", (2, 2, 4), "float32")
    rpe_mask = padded("rpe_mask", (2, 2), "bool")

    anchor_index_per_frame = np.asarray(arrays["anchor_index_per_frame"])
    anchor_frame_index_all = np.asarray(arrays["anchor_frame_index"])
    anchor_appearance_all = np.asarray(arrays["anchor_appearance"])
    if (
        anchor_index_per_frame.dtype != np.int64
        or anchor_index_per_frame.shape != (num_frames,)
        or anchor_frame_index_all.dtype != np.int64
        or anchor_frame_index_all.ndim != 1
        or anchor_appearance_all.dtype != np.float32
        or anchor_appearance_all.shape != (anchor_frame_index_all.size, 3, 256, 256)
    ):
        raise PrefixMaterializerError("production anchor arrays drifted")
    named_anchors = anchor_index_per_frame[source]
    unique_anchors = np.unique(named_anchors[named_anchors >= 0])
    if unique_anchors.size > K_MAX:
        raise PrefixMaterializerError("T8 requires more anchors than K_MAX")
    anchor_mask = np.zeros((K_MAX,), dtype=np.bool_)
    anchor_frame_index = np.zeros((K_MAX,), dtype=np.int64)
    anchor_appearance = np.zeros((K_MAX, 3, 256, 256), dtype=np.float32)
    frame_anchor_slot = np.full((HISTORY,), -1, dtype=np.int64)
    global_anchor_to_row = {int(frame): index for index, frame in enumerate(anchor_frame_index_all)}
    anchor_to_slot: dict[int, int] = {}
    for slot, frame in enumerate(unique_anchors):
        frame_value = int(frame)
        if frame_value not in global_anchor_to_row:
            raise PrefixMaterializerError("frame names an unavailable anchor")
        anchor_to_slot[frame_value] = slot
        anchor_mask[slot] = True
        anchor_frame_index[slot] = frame_value
        anchor_appearance[slot] = anchor_appearance_all[global_anchor_to_row[frame_value]]
    for offset, frame in enumerate(named_anchors, start=left_pad):
        if int(frame) >= 0:
            frame_anchor_slot[offset] = anchor_to_slot[int(frame)]

    pts = np.asarray(arrays["pts"])
    dts = np.asarray(arrays["dts"])
    gop_position = np.asarray(arrays["gop_position"])
    reference_index = np.asarray(arrays["reference_index"])
    camera_motion_summary = np.asarray(arrays["camera_motion_summary"])
    camera_motion_mask = np.asarray(arrays["camera_motion_mask"])
    expected_vectors = (pts, dts, gop_position, reference_index, camera_motion_mask)
    if any(value.shape != (num_frames,) for value in expected_vectors):
        raise PrefixMaterializerError("production metadata vectors drifted")
    if camera_motion_summary.dtype != np.float32 or camera_motion_summary.shape != (num_frames, 4):
        raise PrefixMaterializerError("production camera motion summary drifted")
    metadata = np.zeros((HISTORY, METADATA_WIDTH), dtype=np.float32)
    current_pts = int(pts[observation_row])
    current_dts = int(dts[observation_row])
    for offset, frame in enumerate(source, start=left_pad):
        frame_value = int(frame)
        frame_type_value = int(np.asarray(arrays["frame_type"])[frame_value])
        metadata[offset, frame_type_value - 1] = np.float32(1.0)
        metadata[offset, 3] = np.float32((frame_value - observation_row) / 7.0)
        metadata[offset, 4] = np.float32((int(pts[frame_value]) - current_pts) / 30_000)
        metadata[offset, 5] = np.float32((int(dts[frame_value]) - current_dts) / 30_000)
        metadata[offset, 6] = np.float32(int(gop_position[frame_value]) / max(GOP - 1, 1))
        reference = int(reference_index[frame_value])
        metadata[offset, 7] = np.float32((frame_value - reference) / GOP if reference >= 0 else 0.0)
        metadata[offset, 8:15] = frame_validity[offset].astype(np.float32)
        camera_valid = bool(camera_motion_mask[frame_value])
        anchor_valid = int(anchor_index_per_frame[frame_value]) >= 0
        metadata[offset, 15] = np.float32(camera_valid)
        metadata[offset, 16] = np.float32(anchor_valid)
        if camera_valid:
            metadata[offset, 17:21] = camera_motion_summary[frame_value]

    active_reasons = frame_reason_code[destination]
    view = EpisodeT8View(
        key=EpisodeBundleKey(episode_uid, CAMERA_ID, source_kind),
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
        sample_valid=np.bool_(np.all(active_reasons == REASON_CODES["NONE"])),
        sample_reason_code=np.int16(
            next(
                (int(code) for code in active_reasons if int(code) != REASON_CODES["NONE"]),
                REASON_CODES["NONE"],
            )
        ),
    )
    validate_episode_view(view)
    return view


def _materialize_visible_prefix(
    frames: Sequence[np.ndarray],
    episode_uid: str,
    observation_row: int,
    *,
    source_kind: str,
) -> EpisodeT8View:
    if type(frames) is not tuple or not frames:
        raise PrefixMaterializerError("visible episode prefix must be a non-empty tuple")
    if type(observation_row) is not int or observation_row < 0:
        raise PrefixMaterializerError("observation_row must be a non-negative plain integer")
    episode_uid = _require_sha256(episode_uid, "episode_uid")
    if len(frames) != observation_row + 1:
        raise PrefixMaterializerError(
            "future rows are forbidden: prefix length must equal observation_row + 1"
        )
    copied = []
    for index, frame in enumerate(frames):
        if type(frame) is not np.ndarray:
            raise PrefixMaterializerError(f"prefix frame {index} must be a plain numpy array")
        if frame.dtype != np.uint8 or frame.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
            raise PrefixMaterializerError(
                f"prefix frame {index} must be uint8 [{EXPECTED_HEIGHT},{EXPECTED_WIDTH},3]"
            )
        copied.append(np.array(frame, dtype=np.uint8, order="C", copy=True))
    rgb = validate_rgb_sequence(np.stack(copied))
    with tempfile.TemporaryDirectory(prefix="groot-causal-prefix-") as temporary:
        scratch = Path(temporary)
        encoded = encode_h264_twice(rgb, scratch)
        probe_x264_encoder_options(encoded.bitstream_path)
        packets = probe_packets(encoded.bitstream_path, num_frames=len(frames))
        headers = trace_h264_headers(encoded.bitstream_path, num_frames=len(frames))
        extracted = extract_mvs_and_decoded_rgb(encoded.bitstream_path, num_frames=len(frames))
        arrays = build_episode_arrays(extracted, packets, headers)
        view = episode_arrays_to_view(
            arrays,
            episode_uid=episode_uid,
            source_kind=source_kind,
            observation_row=observation_row,
        )
    return view


def build_prefix_materializer(
    contract: Mapping[str, Any],
) -> Callable[[tuple[np.ndarray, ...], str, int], EpisodeT8View]:
    """Build the evaluator's hash-bound, future-blind prefix callback."""

    normalized_contract, identity = _validate_contract(contract)
    source_kind = str(normalized_contract["source_kind"])

    def materialize_prefix(
        frames: tuple[np.ndarray, ...], episode_uid: str, observation_row: int
    ) -> EpisodeT8View:
        return _materialize_visible_prefix(
            frames,
            episode_uid,
            observation_row,
            source_kind=source_kind,
        )

    materialize_prefix.materializer_id = identity["materializer_id"]  # type: ignore[attr-defined]
    materialize_prefix.materializer_sha256 = identity["factory"]["sha256"]  # type: ignore[attr-defined]
    materialize_prefix.factory_identity = identity  # type: ignore[attr-defined]
    return materialize_prefix


__all__ = [
    "EXPECTED_MATERIALIZER_ID",
    "EXPECTED_SOURCE_KIND",
    "FACTORY_SCHEMA_VERSION",
    "PrefixMaterializerError",
    "TRAINING_CONTRACT_SCHEMA_VERSION",
    "build_prefix_materializer",
    "episode_arrays_to_view",
]
