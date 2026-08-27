"""Fail-closed stores for codec-evidence records."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contract import HISTORY, MV_FIELD_DTYPES, REASON_CODES, CodecEvidenceKey, CodecEvidenceRecord


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


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _require_exact_fields(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields do not match the frozen record-v2 contract")
    return value


def _require_sha(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_int(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_record_payload_v2(payload: Mapping[str, Any]) -> None:
    """Strict dependency-free validation of the frozen record-v2 JSON surface."""

    _require_exact_fields(
        payload,
        {
            "schema_version",
            "synthetic_only",
            "fixture_manifest_sha256",
            "fixture_schema_sha256",
            "encoder_extractor_identity",
            "episode",
            "camera_id",
            "view_claim",
            "observation_row",
            "history",
            "i_frame",
            "mv_storage",
            "rpe",
            "camera_motion_summary",
            "sample_valid",
            "sample_reason_code",
            "terminology",
        },
        "record",
    )
    if payload["schema_version"] != "codec-evidence-record-v2":
        raise ValueError("Unsupported codec-evidence record schema")
    if type(payload["synthetic_only"]) is not bool or type(payload["sample_valid"]) is not bool:
        raise ValueError("Record synthetic_only and sample_valid must be boolean")
    _require_sha(payload["fixture_manifest_sha256"], "fixture_manifest_sha256")
    _require_sha(payload["fixture_schema_sha256"], "fixture_schema_sha256")
    if payload["sample_reason_code"] not in REASON_CODES:
        raise ValueError("Unknown record sample reason code")
    _require_int(payload["observation_row"], "observation_row")
    if payload["camera_id"] != "cam_left_high":
        raise ValueError("Record camera_id must be cam_left_high")
    if payload["view_claim"] != "HEAD_FRONT_EGOCENTRIC_CANDIDATE_VIEW_NOT_CALIBRATION_PROVEN":
        raise ValueError("Record view claim differs from the frozen contract")

    episode = _require_exact_fields(
        payload["episode"],
        {"episode_uid", "dataset_name", "cohort_id", "source_episode_index"},
        "episode",
    )
    _require_sha(episode["episode_uid"], "episode_uid")
    if not isinstance(episode["dataset_name"], str) or not episode["dataset_name"]:
        raise ValueError("episode.dataset_name must be non-empty")
    if not isinstance(episode["cohort_id"], str) or not episode["cohort_id"]:
        raise ValueError("episode.cohort_id must be non-empty")
    _require_int(episode["source_episode_index"], "episode.source_episode_index")

    encoder = _require_exact_fields(
        payload["encoder_extractor_identity"],
        {
            "container",
            "codec",
            "encoder",
            "encoder_binary_sha256",
            "encoder_build_identity",
            "extractor_identity",
            "extractor_sha256",
            "bframes",
            "references",
            "gop",
            "scene_cut",
            "open_gop",
            "time_base_num",
            "time_base_den",
        },
        "encoder_extractor_identity",
    )
    frozen_encoder = {
        "container": "mp4",
        "codec": "h264",
        "encoder": "libx264",
        "bframes": 0,
        "references": 1,
        "gop": 30,
        "scene_cut": 0,
        "open_gop": 0,
        "time_base_num": 1,
        "time_base_den": 30000,
    }
    if any(encoder[name] != value for name, value in frozen_encoder.items()):
        raise ValueError("Encoder identity violates the causal I/P fixture contract")
    _require_sha(encoder["encoder_binary_sha256"], "encoder_binary_sha256")
    _require_sha(encoder["extractor_sha256"], "extractor_sha256")
    for name in ("encoder_build_identity", "extractor_identity"):
        if not isinstance(encoder[name], str) or not encoder[name]:
            raise ValueError(f"{name} must be non-empty")

    history = _require_exact_fields(payload["history"], {"length", "padding", "frames"}, "history")
    if history["length"] != HISTORY or history["padding"] != (
        "LEFT_ZERO_NEGATIVE_ONE_TEMPORAL_IDS_ALL_VALIDITY_FALSE_NO_REPEAT"
    ):
        raise ValueError("History length or padding contract mismatch")
    frames = history["frames"]
    if not isinstance(frames, list) or len(frames) != HISTORY:
        raise ValueError("History must contain exactly eight frames")
    frame_fields = {
        "slot",
        "frame_index",
        "pts",
        "dts",
        "frame_type",
        "gop_id",
        "gop_position",
        "reference_index",
        "source_sha256",
        "decoded_sha256",
        "validity",
        "reason_code",
        "mv_offset_begin",
        "mv_offset_end",
        "mv_count",
    }
    validity_fields = {
        "packet",
        "frame",
        "reference",
        "dependency",
        "alignment",
        "mv",
        "rpe",
    }
    for slot, frame_value in enumerate(frames):
        frame = _require_exact_fields(frame_value, frame_fields, f"history.frames[{slot}]")
        if frame["slot"] != slot or frame["frame_type"] not in {"PAD", "I", "P"}:
            raise ValueError("History slot or frame type is invalid")
        if frame["reason_code"] not in REASON_CODES:
            raise ValueError("Unknown frame reason code")
        for name in ("frame_index", "pts", "dts", "gop_id", "gop_position", "reference_index"):
            if type(frame[name]) is not int or frame[name] < -1:
                raise ValueError(f"Frame {name} must be an integer >= -1")
        if frame["gop_position"] > 29:
            raise ValueError("Frame gop_position exceeds 29")
        for name in ("mv_offset_begin", "mv_offset_end", "mv_count"):
            _require_int(frame[name], f"frame.{name}")
        if frame["mv_count"] > 19_200:
            raise ValueError("Frame MV count exceeds 19200")
        _require_sha(frame["source_sha256"], "frame.source_sha256", nullable=True)
        _require_sha(frame["decoded_sha256"], "frame.decoded_sha256", nullable=True)
        validity = _require_exact_fields(frame["validity"], validity_fields, "frame.validity")
        if any(type(validity[name]) is not bool for name in validity_fields):
            raise ValueError("Every frame validity field must be boolean")

    i_frame = _require_exact_fields(
        payload["i_frame"],
        {"frame_index", "selection", "source_sha256", "decoded_sha256", "appearance"},
        "i_frame",
    )
    if type(i_frame["frame_index"]) is not int or i_frame["frame_index"] < -1:
        raise ValueError("i_frame.frame_index must be >= -1")
    if i_frame["selection"] != "NEAREST_VERIFIED_PAST_IDR_AT_OR_BEFORE_OBSERVATION":
        raise ValueError("I-frame selection contract mismatch")
    _require_sha(i_frame["source_sha256"], "i_frame.source_sha256", nullable=True)
    _require_sha(i_frame["decoded_sha256"], "i_frame.decoded_sha256", nullable=True)

    storage = _require_exact_fields(
        payload["mv_storage"],
        {"layout", "pickle_allowed", "max_per_frame", "offsets", "fields"},
        "mv_storage",
    )
    if (
        storage["layout"] != "CSR_RAGGED_PER_FRAME_NO_TRUNCATION"
        or storage["pickle_allowed"] is not False
        or storage["max_per_frame"] != 19_200
        or not isinstance(storage["fields"], Mapping)
        or set(storage["fields"]) != set(MV_FIELD_DTYPES)
    ):
        raise ValueError("MV storage contract mismatch")

    terminology = _require_exact_fields(
        payload["terminology"], {"motion_vectors", "rpe", "row_grid"}, "terminology"
    )
    expected_terminology = {
        "motion_vectors": "CODEC_ENCODER_DECISIONS_NOT_OPTICAL_FLOW",
        "rpe": "DECODED_LUMA_MV_WARP_PROXY_NOT_BITSTREAM_RESIDUAL",
        "row_grid": "SYNTHETIC_30_HZ_REPRESENTATION_NOT_PROVEN_HARDWARE_CAPTURE_CLOCK",
    }
    if terminology != expected_terminology:
        raise ValueError("Codec terminology contract mismatch")


@runtime_checkable
class CodecEvidenceStore(Protocol):
    """Minimal baseline-neutral lookup protocol."""

    def get(self, key: CodecEvidenceKey) -> CodecEvidenceRecord: ...


class InMemoryCodecEvidenceStore:
    """Deterministic synthetic/conformance store with no filesystem access."""

    def __init__(self, records: Sequence[CodecEvidenceRecord]):
        mapping: dict[CodecEvidenceKey, CodecEvidenceRecord] = {}
        for record in records:
            if record.key in mapping:
                raise ValueError(f"Duplicate codec-evidence key: {record.key.canonical}")
            mapping[record.key] = record
        self._records = mapping

    def get(self, key: CodecEvidenceKey) -> CodecEvidenceRecord:
        try:
            return self._records[key]
        except KeyError as error:
            raise KeyError(f"Missing codec-evidence record: {key.canonical}") from error


class FixtureManifestCodecEvidenceStore:
    """Load immutable record entries rooted below one explicitly pinned fixture."""

    def __init__(
        self,
        root: Path,
        *,
        fixture_manifest_sha256: str,
        record_schema_sha256: str,
    ):
        self.root = root.resolve(strict=True)
        self._record_schema_sha256 = record_schema_sha256
        manifest_path = self._safe_path("fixture_manifest.json")
        if _sha256_path(manifest_path) != fixture_manifest_sha256:
            raise ValueError("Fixture manifest SHA-256 mismatch")
        manifest = self._load_json(manifest_path)
        if manifest.get("schema_version") != "codec-evidence-fixture-manifest-v1":
            raise ValueError("Unsupported codec-evidence fixture manifest")
        required_manifest = {
            "schema_version",
            "fixture_id",
            "identity",
            "fixture_binding_sha256",
            "record_schema",
            "records",
        }
        if set(manifest) != required_manifest:
            raise ValueError("Fixture manifest fields do not match the v1 contract")
        identity = manifest["identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "record_schema_sha256",
            "encoder_extractor_identity",
        }:
            raise ValueError("Fixture manifest identity block is malformed")
        if identity.get("record_schema_sha256") != record_schema_sha256:
            raise ValueError("Fixture manifest record schema SHA-256 mismatch")
        binding_payload = {
            "fixture_id": manifest["fixture_id"],
            "record_schema_sha256": record_schema_sha256,
            "encoder_extractor_identity": identity["encoder_extractor_identity"],
        }
        if _canonical_sha256(binding_payload) != manifest["fixture_binding_sha256"]:
            raise ValueError("Fixture identity binding SHA-256 mismatch")
        self._fixture_binding_sha256 = manifest["fixture_binding_sha256"]
        self._encoder_identity = identity["encoder_extractor_identity"]
        self._validate_artifact_descriptor(manifest["record_schema"], expected_dtype="json")
        schema_path = self._safe_path(manifest["record_schema"]["relative_path"])
        if (
            schema_path.stat().st_size != manifest["record_schema"]["bytes"]
            or _sha256_path(schema_path) != record_schema_sha256
            or manifest["record_schema"]["sha256"] != record_schema_sha256
        ):
            raise ValueError("Record schema artifact identity mismatch")
        record_schema = self._load_json(schema_path)
        if record_schema.get("$id") != (
            "https://icra27.local/schemas/codec-evidence-record-v2.schema.json"
        ):
            raise ValueError("Record schema artifact has an unexpected identity")
        entries = manifest.get("records")
        if not isinstance(entries, list):
            raise ValueError("Fixture manifest records must be a list")
        self._manifest_sha256 = fixture_manifest_sha256
        mapping: dict[CodecEvidenceKey, Mapping[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Fixture manifest record entries must be objects")
            if set(entry) != {"episode_uid", "observation_row", "camera_id", "record"}:
                raise ValueError("Fixture manifest record entry fields do not match contract")
            key = CodecEvidenceKey(
                episode_uid=entry["episode_uid"],
                observation_row=entry["observation_row"],
                camera_id=entry["camera_id"],
            )
            if key in mapping:
                raise ValueError(f"Duplicate codec-evidence key: {key.canonical}")
            self._validate_artifact_descriptor(entry["record"], expected_dtype="json")
            mapping[key] = entry["record"]
        self._entries = mapping

    def _safe_path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("Artifact relative_path must be non-empty")
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Artifact paths must be fixture-root-relative")
        unresolved = self.root / candidate
        cursor = self.root
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Symlink artifacts are forbidden")
        resolved = unresolved.resolve(strict=True)
        if self.root != resolved and self.root not in resolved.parents:
            raise ValueError("Artifact path escapes fixture root")
        return resolved

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
        if not isinstance(result, dict):
            raise ValueError("JSON artifact must contain one object")
        return result

    @staticmethod
    def _validate_artifact_descriptor(
        descriptor: Mapping[str, Any], *, expected_dtype: str
    ) -> None:
        if not isinstance(descriptor, Mapping):
            raise ValueError("Artifact descriptor must be an object")
        required = {"relative_path", "sha256", "bytes", "dtype", "shape"}
        if set(descriptor) != required:
            raise ValueError("Artifact descriptor fields do not match contract")
        if descriptor["dtype"] != expected_dtype:
            raise ValueError("Artifact descriptor dtype mismatch")
        if (
            not isinstance(descriptor["relative_path"], str)
            or not isinstance(descriptor["sha256"], str)
            or len(descriptor["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in descriptor["sha256"])
            or not isinstance(descriptor["bytes"], int)
            or descriptor["bytes"] < 0
            or not isinstance(descriptor["shape"], list)
        ):
            raise ValueError("Artifact descriptor identity fields are malformed")

    def _load_artifact(self, descriptor: Mapping[str, Any], *, dtype: np.dtype) -> np.ndarray:
        self._validate_artifact_descriptor(descriptor, expected_dtype=str(dtype))
        path = self._safe_path(descriptor["relative_path"])
        stat = path.stat()
        if stat.st_size != descriptor["bytes"] or _sha256_path(path) != descriptor["sha256"]:
            raise ValueError("Artifact bytes or SHA-256 mismatch")
        value = np.load(path, allow_pickle=False)
        expected_shape = tuple(
            value.shape[0] if item == "N_total" else item for item in descriptor["shape"]
        )
        if value.dtype != dtype or value.shape != expected_shape:
            raise ValueError("Artifact dtype or shape mismatch")
        return np.ascontiguousarray(value)

    def get(self, key: CodecEvidenceKey) -> CodecEvidenceRecord:
        try:
            descriptor = self._entries[key]
        except KeyError as error:
            raise KeyError(f"Missing codec-evidence record: {key.canonical}") from error
        record_path = self._safe_path(descriptor["relative_path"])
        if record_path.stat().st_size != descriptor["bytes"]:
            raise ValueError("Record bytes mismatch")
        if _sha256_path(record_path) != descriptor["sha256"]:
            raise ValueError("Record SHA-256 mismatch")
        payload = self._load_json(record_path)
        _validate_record_payload_v2(payload)
        if payload.get("fixture_manifest_sha256") != self._fixture_binding_sha256:
            raise ValueError("Record fixture identity binding SHA-256 mismatch")
        if payload.get("fixture_schema_sha256") != self._record_schema_sha256:
            raise ValueError("Record schema SHA-256 mismatch")
        if payload.get("schema_version") != "codec-evidence-record-v2":
            raise ValueError("Unsupported codec-evidence record schema")
        encoder = payload.get("encoder_extractor_identity", {})
        if encoder != self._encoder_identity:
            raise ValueError("Record encoder/extractor identity differs from fixture manifest")
        frozen = {
            "container": "mp4",
            "codec": "h264",
            "encoder": "libx264",
            "bframes": 0,
            "references": 1,
            "gop": 30,
            "scene_cut": 0,
            "open_gop": 0,
        }
        if any(encoder.get(name) != value for name, value in frozen.items()):
            raise ValueError("Encoder identity violates the causal I/P fixture contract")

        episode = payload.get("episode", {})
        record_key = CodecEvidenceKey(
            episode_uid=episode.get("episode_uid"),
            observation_row=payload.get("observation_row"),
            camera_id=payload.get("camera_id"),
        )
        if record_key != key:
            raise ValueError("Record identity does not match manifest lookup key")
        storage = payload["mv_storage"]
        offsets = self._load_artifact(storage["offsets"], dtype=np.dtype("int64"))
        fields = {
            name: self._load_artifact(storage["fields"][name], dtype=dtype)
            for name, dtype in MV_FIELD_DTYPES.items()
        }
        rpe = self._load_artifact(payload["rpe"], dtype=np.dtype("float32"))
        appearance = self._load_artifact(payload["i_frame"]["appearance"], dtype=np.dtype("uint8"))
        camera = self._load_artifact(payload["camera_motion_summary"], dtype=np.dtype("float32"))
        if offsets.shape != (HISTORY + 1,):
            raise ValueError("MV offsets must have shape [9]")
        return CodecEvidenceRecord(
            key=key,
            payload=payload,
            mv_offsets=offsets,
            mv_fields=fields,
            rpe=rpe,
            i_frame_appearance=appearance,
            camera_motion_summary=camera,
        )
