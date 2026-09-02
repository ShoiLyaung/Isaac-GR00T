"""Content-addressed, split-neutral codec-evidence warehouse.

The legacy R5 pilot bundle remains supported by :mod:`episode_bundle`.  This
module is the production storage boundary for complete episodes.  An immutable
object contains only stable episode/source/codec facts.  Dataset generation and
split identities live exclusively in an explicitly pinned snapshot manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import MV_FIELD_DTYPES, SHA256_RE
from .episode_bundle import SOURCE_KIND_NATIVE_CANONICAL, EpisodeBundle, EpisodeBundleKey


WAREHOUSE_OBJECT_SCHEMA_VERSION = "codec-evidence-warehouse-object-v1"
SNAPSHOT_SCHEMA_VERSION = "codec-evidence-fixture-snapshot-v1"
ATTEMPT_SCHEMA_VERSION = "codec-evidence-warehouse-attempt-v1"
DERIVATION_SCHEMA_VERSION = "codec-evidence-derivation-binding-v1"
SNAPSHOT_FILENAME = "fixture_snapshot_manifest.json"
OBJECT_MANIFEST_FILENAME = "object_manifest.json"
CANARY_COVERAGE = "CANARY_SUBSET_NONTRAINING"
FULL_SPLIT_COVERAGE = "FULL_SPLIT"
SOURCE_KIND = SOURCE_KIND_NATIVE_CANONICAL

ARRAY_NAMES = (
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
)

_OBJECT_FIELDS = {
    "schema_version",
    "derivation_key_sha256",
    "object_content_sha256",
    "manifest_payload_sha256",
    "manifest_payload",
    "artifacts",
}
_SNAPSHOT_FIELDS = {
    "schema_version",
    "snapshot_id",
    "complete",
    "coverage_kind",
    "training_eligible",
    "camera_id",
    "source_kind",
    "dataset_binding",
    "covered_episode_uids",
    "covered_episode_uid_digest",
    "covered_episode_count",
    "objects",
    "producer_identity",
}
_BINDING_FIELDS = {
    "dataset_root",
    "catalog_database_path",
    "catalog_database_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "payload_manifest_path",
    "payload_manifest_sha256",
    "train_statistics_path",
    "train_statistics_sha256",
    "train_episode_uid_digest",
    "resolved_train_episode_count",
}
_SNAPSHOT_OBJECT_FIELDS = {
    "episode_uid",
    "camera_id",
    "source_kind",
    "derivation_key_sha256",
    "object_content_sha256",
    "num_frames",
}
_DERIVATION_PAYLOAD_FIELDS = {
    "schema_version",
    "episode_uid",
    "camera_id",
    "source_kind",
    "episode",
    "canonical_video",
    "canonical_parquet",
    "codec_recipe",
    "toolchain_identity",
}
_MANIFEST_PAYLOAD_FIELDS = {
    "schema_version",
    "derivation_payload_sha256",
    "episode_uid",
    "camera_id",
    "source_kind",
    "num_frames",
    "episode",
    "width",
    "height",
    "time_base_num",
    "time_base_den",
    "gop_size",
    "input_identity",
    "frame_hashes",
    "identities",
}
_FORBIDDEN_OBJECT_KEY_PATTERNS = (
    "split",
    "catalog",
    "payload_manifest",
    "train_statistics",
    "raw_root",
    "raw_jpeg",
    "raw_source",
    "source_jpeg",
    "upstream_raw",
    "upstream_jpeg",
    "dual_source",
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one JSON encoding used by all warehouse identities."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def episode_uid_digest(episode_uids: Sequence[str]) -> str:
    values = sorted(episode_uids)
    if len(values) != len(set(values)):
        raise ValueError("episode_uids must be unique")
    for value in values:
        _require_sha(value, "episode_uid")
    return canonical_json_sha256(values)


def frozen_train_episode_uid_digest(episode_uids: Sequence[str]) -> str:
    """Match the frozen adapter manifest's newline-delimited train UID digest."""

    values = sorted(episode_uids)
    if len(values) != len(set(values)):
        raise ValueError("episode_uids must be unique")
    for value in values:
        _require_sha(value, "episode_uid")
    return sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


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
        raise ValueError(f"{name} must be one plain path component")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact is missing, non-regular, or a symlink: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_child(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Artifact path is not a safe relative path: {relative_path!r}")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"Artifact is missing, non-regular, or a symlink: {relative_path}")
    return candidate


def _array_descriptor(path: Path, relative_path: str, value: np.ndarray) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "dtype": np.dtype(value.dtype).name,
        "shape": list(value.shape),
        "format": "npy-v1",
    }


def _file_descriptor(path: Path, relative_path: str, kind: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "kind": kind,
    }


def derivation_key_sha256(payload: Mapping[str, Any]) -> str:
    """Hash recipe/input identity, never artifact bytes or dataset split."""

    if set(payload) != _DERIVATION_PAYLOAD_FIELDS:
        raise ValueError("Derivation payload fields differ from the frozen canonical schema")
    _reject_dataset_identity_leaks(payload)
    return canonical_json_sha256(payload)


def _reject_dataset_identity_leaks(value: Any, path: str = "object") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            lowered = key.lower()
            if any(pattern in lowered for pattern in _FORBIDDEN_OBJECT_KEY_PATTERNS):
                raise ValueError(
                    f"Dataset/split identity is forbidden in split-neutral object: {path}.{key}"
                )
            _reject_dataset_identity_leaks(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_dataset_identity_leaks(item, f"{path}[{index}]")


@dataclass(frozen=True)
class PublishedObject:
    episode_uid: str
    camera_id: str
    source_kind: str
    derivation_key_sha256: str
    object_content_sha256: str
    num_frames: int
    object_path: Path
    reused: bool

    def snapshot_entry(self) -> dict[str, Any]:
        return {
            "episode_uid": self.episode_uid,
            "camera_id": self.camera_id,
            "source_kind": self.source_kind,
            "derivation_key_sha256": self.derivation_key_sha256,
            "object_content_sha256": self.object_content_sha256,
            "num_frames": self.num_frames,
        }


def ensure_warehouse_layout(root: str | Path) -> Path:
    requested = Path(root)
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    warehouse = requested.resolve(strict=False)
    if not warehouse.is_absolute():
        raise ValueError("warehouse root must be absolute")
    warehouse.mkdir(mode=0o755, parents=False, exist_ok=True)
    if warehouse.is_symlink() or not warehouse.is_dir():
        raise ValueError("warehouse root must be a real directory")
    for name in ("objects", "derivations", "attempts", "canaries", "snapshots", ".staging"):
        path = warehouse / name
        path.mkdir(mode=0o755, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"warehouse namespace is not a real directory: {name}")
    return warehouse.resolve(strict=True)


def _validate_array_surface(arrays: Mapping[str, np.ndarray]) -> int:
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Production array keys differ from the episode-bundle surface")
    frame_index = np.asarray(arrays["frame_index"])
    if frame_index.dtype != np.int64 or frame_index.ndim != 1 or frame_index.size == 0:
        raise ValueError("frame_index must be non-empty int64 [N]")
    num_frames = int(frame_index.size)
    if not np.array_equal(frame_index, np.arange(num_frames, dtype=np.int64)):
        raise ValueError("frame_index must cover the complete 0..N-1 episode")
    anchor_index = np.asarray(arrays["anchor_frame_index"])
    if anchor_index.dtype != np.int64 or anchor_index.ndim != 1 or anchor_index.size == 0:
        raise ValueError("anchor_frame_index must be non-empty int64 [A]")
    # Reuse the strict episode validator without publishing its split-bearing schema.
    metadata = {
        "split": "train",
        "cohort_id": "warehouse-validation",
        "dataset_name": "warehouse-validation",
        "source_episode_index": 0,
        "width": 2,
        "height": 2,
        "time_base_num": 1,
        "time_base_den": 30_000,
        "gop_size": 30,
        "source_frame_sha256": ["0" * 64] * num_frames,
        "canonical_frame_sha256": ["0" * 64] * num_frames,
        "fixture_frame_sha256": ["0" * 64] * num_frames,
        "tool_identity": {"validator": "warehouse"},
        "encoder_identity": {"validator": "warehouse"},
        "decoder_identity": {"validator": "warehouse"},
        "extractor_identity": {"validator": "warehouse"},
    }
    EpisodeBundle(
        key=EpisodeBundleKey("0" * 64, "validation", SOURCE_KIND),
        **metadata,
        **arrays,
    )
    return num_frames


def publish_object(
    warehouse_root: str | Path,
    *,
    derivation_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    raw_mv_offsets: np.ndarray,
    raw_mv_fields: Mapping[str, np.ndarray],
    accepted_bitstream: str | Path,
) -> PublishedObject:
    """Atomically publish one complete split-neutral object, or reuse exact bytes."""

    root = ensure_warehouse_layout(warehouse_root)
    if manifest_payload.get("schema_version") != WAREHOUSE_OBJECT_SCHEMA_VERSION:
        raise ValueError("manifest_payload schema_version mismatch")
    if set(manifest_payload) != _MANIFEST_PAYLOAD_FIELDS:
        raise ValueError("Manifest payload fields differ from the frozen canonical schema")
    _reject_dataset_identity_leaks(manifest_payload)
    episode_uid = _require_sha(manifest_payload.get("episode_uid"), "episode_uid")
    camera_id = _require_plain_component(manifest_payload.get("camera_id"), "camera_id")
    if manifest_payload.get("source_kind") != SOURCE_KIND:
        raise ValueError("Production C0 object must be canonical-derived")
    num_frames = _validate_array_surface(arrays)
    if manifest_payload.get("num_frames") != num_frames:
        raise ValueError("manifest_payload num_frames differs from complete arrays")
    accepted_input = Path(accepted_bitstream)
    if accepted_input.is_symlink():
        raise ValueError("accepted_bitstream must not be a symlink")
    accepted = accepted_input.resolve(strict=True)
    if not accepted.is_file() or accepted.stat().st_size == 0:
        raise ValueError("accepted_bitstream must be a non-empty regular file")
    offsets = np.asarray(raw_mv_offsets)
    if offsets.dtype != np.int64 or offsets.shape != (num_frames + 1,):
        raise ValueError("raw_mv_offsets must be int64 [N+1]")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("raw_mv_offsets must be monotonic and start at zero")
    if set(raw_mv_fields) != set(MV_FIELD_DTYPES):
        raise ValueError("raw_mv_fields must preserve the exact eleven-field audit surface")
    total_vectors = int(offsets[-1])
    for name, dtype in MV_FIELD_DTYPES.items():
        value = np.asarray(raw_mv_fields[name])
        if value.dtype != dtype or value.shape != (total_vectors,):
            raise ValueError(f"raw_mv_fields.{name} has wrong dtype or shape")

    derivation_key = derivation_key_sha256(derivation_payload)
    if manifest_payload.get("derivation_payload_sha256") != derivation_key:
        raise ValueError("manifest payload does not bind the exact derivation payload")

    staging: Path | None = Path(tempfile.mkdtemp(prefix="object-", dir=root / ".staging"))
    try:
        artifacts: list[dict[str, Any]] = []
        arrays_root = staging / "arrays"
        arrays_root.mkdir()
        for name in ARRAY_NAMES:
            value = np.ascontiguousarray(arrays[name])
            relative = f"arrays/{name}.npy"
            path = staging / relative
            with path.open("xb") as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            artifacts.append(_array_descriptor(path, relative, value))

        audit_root = staging / "raw_motion_vectors"
        fields_root = audit_root / "fields"
        fields_root.mkdir(parents=True)
        offset_relative = "raw_motion_vectors/offsets.npy"
        offset_path = staging / offset_relative
        with offset_path.open("xb") as stream:
            np.lib.format.write_array(stream, offsets, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        artifacts.append(_array_descriptor(offset_path, offset_relative, offsets))
        for name in MV_FIELD_DTYPES:
            value = np.ascontiguousarray(raw_mv_fields[name])
            relative = f"raw_motion_vectors/fields/{name}.npy"
            path = staging / relative
            with path.open("xb") as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            artifacts.append(_array_descriptor(path, relative, value))

        bitstream_path = staging / "accepted.mp4"
        with accepted.open("rb") as source, bitstream_path.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        artifacts.append(_file_descriptor(bitstream_path, "accepted.mp4", "accepted_bitstream"))
        artifacts.sort(key=lambda descriptor: descriptor["relative_path"])

        payload = json.loads(canonical_json_bytes(manifest_payload))
        payload_sha = canonical_json_sha256(payload)
        content_digest = canonical_json_sha256(
            {
                "schema_version": WAREHOUSE_OBJECT_SCHEMA_VERSION,
                "manifest_payload_sha256": payload_sha,
                "artifacts": artifacts,
            }
        )
        object_manifest = {
            "schema_version": WAREHOUSE_OBJECT_SCHEMA_VERSION,
            "derivation_key_sha256": derivation_key,
            "object_content_sha256": content_digest,
            "manifest_payload_sha256": payload_sha,
            "manifest_payload": payload,
            "artifacts": artifacts,
        }
        _write_json_exclusive(staging / OBJECT_MANIFEST_FILENAME, object_manifest)
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((item for item in staging.rglob("*") if item.is_dir()), reverse=True):
            path.chmod(0o555)
        # Keep the staging root writable through rename for filesystems that
        # reject moving a non-writable source directory.  Freeze the published
        # destination immediately after the atomic namespace operation.
        staging.chmod(0o755)

        # Reserve the derivation mapping before publishing the object namespace.
        # A crash now leaves a fail-closed pointer to a missing object that an
        # exact retry can complete; a different digest for the same recipe is
        # rejected before it can create an orphan object.
        binding = {
            "schema_version": DERIVATION_SCHEMA_VERSION,
            "derivation_key_sha256": derivation_key,
            "object_content_sha256": content_digest,
        }
        binding_path = root / "derivations" / f"{derivation_key}.json"
        try:
            _write_json_exclusive(binding_path, binding)
            binding_path.chmod(0o444)
            _fsync_directory(binding_path.parent)
        except FileExistsError:
            existing_binding = _load_json(binding_path)
            if existing_binding != binding:
                raise ValueError(
                    "One derivation key produced a different content digest; refusing reuse"
                )

        destination = root / "objects" / content_digest
        reused = False
        try:
            os.rename(staging, destination)
            staging = None
            destination.chmod(0o555)
            _fsync_directory(root / "objects")
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            reused = True
            staging.chmod(0o755)
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                else:
                    path.chmod(0o644)
            shutil.rmtree(staging)
            staging = None
            existing = verify_object(root, content_digest)
            if canonical_json_bytes(existing) != canonical_json_bytes(object_manifest):
                raise ValueError("Existing object path does not contain the same content closure")

        return PublishedObject(
            episode_uid=episode_uid,
            camera_id=camera_id,
            source_kind=SOURCE_KIND,
            derivation_key_sha256=derivation_key,
            object_content_sha256=content_digest,
            num_frames=num_frames,
            object_path=destination,
            reused=reused,
        )
    finally:
        if staging is not None and staging.exists():
            staging.chmod(0o755)
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                else:
                    path.chmod(0o644)
            shutil.rmtree(staging)


def verify_object(warehouse_root: str | Path, object_content_sha256: str) -> dict[str, Any]:
    requested_root = Path(warehouse_root)
    if requested_root.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    root = requested_root.resolve(strict=True)
    digest = _require_sha(object_content_sha256, "object_content_sha256")
    object_root = root / "objects" / digest
    if object_root.is_symlink() or not object_root.is_dir():
        raise ValueError("Warehouse object is missing or a symlink")
    manifest = _load_json(object_root / OBJECT_MANIFEST_FILENAME)
    if (
        set(manifest) != _OBJECT_FIELDS
        or manifest["schema_version"] != WAREHOUSE_OBJECT_SCHEMA_VERSION
    ):
        raise ValueError("Warehouse object manifest keys/schema mismatch")
    if manifest["object_content_sha256"] != digest:
        raise ValueError("Object directory name and manifest digest differ")
    _require_sha(manifest["derivation_key_sha256"], "derivation_key_sha256")
    payload = manifest["manifest_payload"]
    if not isinstance(payload, dict):
        raise ValueError("manifest_payload must be an object")
    if (
        set(payload) != _MANIFEST_PAYLOAD_FIELDS
        or payload.get("schema_version") != WAREHOUSE_OBJECT_SCHEMA_VERSION
    ):
        raise ValueError("Warehouse manifest_payload fields/schema mismatch")
    _reject_dataset_identity_leaks(payload)
    if payload.get("derivation_payload_sha256") != manifest["derivation_key_sha256"]:
        raise ValueError("Object payload does not bind its derivation key")
    if canonical_json_sha256(payload) != manifest["manifest_payload_sha256"]:
        raise ValueError("manifest_payload SHA-256 mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifacts must be a non-empty list")
    relative_paths: list[str] = []
    for descriptor in artifacts:
        if not isinstance(descriptor, dict):
            raise ValueError("artifact descriptor must be an object")
        relative = descriptor.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("artifact relative_path must be a string")
        path = _regular_child(object_root, relative)
        if type(descriptor.get("bytes")) is not int or descriptor["bytes"] != path.stat().st_size:
            raise ValueError(f"Artifact byte-count mismatch: {relative}")
        if sha256_path(path) != descriptor.get("sha256"):
            raise ValueError(f"Artifact SHA-256 mismatch: {relative}")
        relative_paths.append(relative)
    if relative_paths != sorted(relative_paths) or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Artifact descriptors must be sorted and unique")
    recomputed = canonical_json_sha256(
        {
            "schema_version": WAREHOUSE_OBJECT_SCHEMA_VERSION,
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "artifacts": artifacts,
        }
    )
    if recomputed != digest:
        raise ValueError("Object artifact/Merkle content digest mismatch")
    derivation_key = manifest["derivation_key_sha256"]
    binding_path = root / "derivations" / f"{derivation_key}.json"
    binding = _load_json(binding_path)
    expected_binding = {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "derivation_key_sha256": derivation_key,
        "object_content_sha256": digest,
    }
    if binding != expected_binding:
        raise ValueError("Derivation binding is missing, corrupt, or maps to another object")
    expected_files = set(relative_paths) | {OBJECT_MANIFEST_FILENAME}
    observed_files: set[str] = set()
    for current, directory_names, file_names in os.walk(object_root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if (current_path / directory_name).is_symlink():
                raise ValueError("Symlinks are forbidden inside warehouse objects")
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Only regular files are allowed inside warehouse objects")
            observed_files.add(path.relative_to(object_root).as_posix())
    if observed_files != expected_files:
        raise ValueError("Object contains unclosed or missing files")
    return manifest


def _validate_dataset_binding(binding: Any) -> dict[str, Any]:
    if not isinstance(binding, dict) or set(binding) != _BINDING_FIELDS:
        raise ValueError("dataset_binding fields differ from the snapshot contract")
    for name in (
        "catalog_database_sha256",
        "split_manifest_sha256",
        "payload_manifest_sha256",
        "train_statistics_sha256",
        "train_episode_uid_digest",
    ):
        _require_sha(binding[name], f"dataset_binding.{name}")
    if (
        type(binding["resolved_train_episode_count"]) is not int
        or binding["resolved_train_episode_count"] <= 0
    ):
        raise ValueError("resolved_train_episode_count must be positive")
    for name in _BINDING_FIELDS - {
        "catalog_database_sha256",
        "split_manifest_sha256",
        "payload_manifest_sha256",
        "train_statistics_sha256",
        "train_episode_uid_digest",
        "resolved_train_episode_count",
    }:
        if not isinstance(binding[name], str) or not binding[name]:
            raise ValueError(f"dataset_binding.{name} must be non-empty")
    return binding


def write_snapshot_manifest(
    warehouse_root: str | Path,
    *,
    snapshot_id: str,
    coverage_kind: str,
    dataset_binding: Mapping[str, Any],
    objects: Sequence[PublishedObject | Mapping[str, Any]],
    producer_identity: Mapping[str, Any],
) -> Path:
    """Publish the sole loader commit for a canary subset or a full split."""

    root = ensure_warehouse_layout(warehouse_root)
    snapshot_id = _require_plain_component(snapshot_id, "snapshot_id")
    if coverage_kind not in {CANARY_COVERAGE, FULL_SPLIT_COVERAGE}:
        raise ValueError("Unknown coverage_kind")
    training_eligible = coverage_kind == FULL_SPLIT_COVERAGE
    namespace = "snapshots" if training_eligible else "canaries"
    entries = [
        item.snapshot_entry() if isinstance(item, PublishedObject) else dict(item)
        for item in objects
    ]
    if not entries:
        raise ValueError("A snapshot must contain at least one object")
    entries.sort(key=lambda item: (item["episode_uid"], item["camera_id"], item["source_kind"]))
    keys: set[tuple[str, str, str]] = set()
    for entry in entries:
        if set(entry) != _SNAPSHOT_OBJECT_FIELDS:
            raise ValueError("Snapshot object entry fields differ from the contract")
        key = (
            _require_sha(entry["episode_uid"], "objects.episode_uid"),
            _require_plain_component(entry["camera_id"], "objects.camera_id"),
            entry["source_kind"],
        )
        if key[2] != SOURCE_KIND or key in keys:
            raise ValueError("Snapshot object keys must be unique canonical-derived identities")
        keys.add(key)
        _require_sha(entry["derivation_key_sha256"], "objects.derivation_key_sha256")
        object_manifest = verify_object(
            root, _require_sha(entry["object_content_sha256"], "objects.content")
        )
        if object_manifest["derivation_key_sha256"] != entry["derivation_key_sha256"]:
            raise ValueError("Snapshot entry derivation key differs from its object")
        payload = object_manifest["manifest_payload"]
        for name in ("episode_uid", "camera_id", "source_kind", "num_frames"):
            if payload.get(name) != entry[name]:
                raise ValueError(f"Snapshot entry {name} differs from its object payload")
        if type(entry["num_frames"]) is not int or entry["num_frames"] <= 0:
            raise ValueError("objects.num_frames must be positive")
    uids = sorted({key[0] for key in keys})
    if len(uids) != len(entries):
        raise ValueError("C0 snapshots support exactly one camera/source object per episode")
    cameras = {entry["camera_id"] for entry in entries}
    sources = {entry["source_kind"] for entry in entries}
    if len(cameras) != 1 or sources != {SOURCE_KIND}:
        raise ValueError("Snapshot objects must share one exact camera/source identity")
    if not isinstance(producer_identity, Mapping) or not producer_identity:
        raise ValueError("producer_identity must be a non-empty mapping")
    binding = _validate_dataset_binding(dict(dataset_binding))
    if training_eligible:
        if len(uids) != binding["resolved_train_episode_count"]:
            raise ValueError("FULL_SPLIT snapshot count differs from resolved train split")
        if frozen_train_episode_uid_digest(uids) != binding["train_episode_uid_digest"]:
            raise ValueError("FULL_SPLIT snapshot UIDs differ from resolved train split")
    destination = root / namespace / snapshot_id
    destination.mkdir(mode=0o755, exist_ok=False)
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "complete": True,
        "coverage_kind": coverage_kind,
        "training_eligible": training_eligible,
        "camera_id": entries[0]["camera_id"],
        "source_kind": SOURCE_KIND,
        "dataset_binding": binding,
        "covered_episode_uids": uids,
        "covered_episode_uid_digest": episode_uid_digest(uids),
        "covered_episode_count": len(uids),
        "objects": entries,
        "producer_identity": dict(producer_identity),
    }
    path = destination / SNAPSHOT_FILENAME
    _write_json_exclusive(path, manifest)
    path.chmod(0o444)
    _fsync_directory(destination)
    destination.chmod(0o555)
    _fsync_directory(root / namespace)
    return path


def validate_snapshot_manifest(
    warehouse_root: str | Path,
    manifest_path: str | Path,
    *,
    manifest_sha256: str,
    verifier_only: bool = False,
) -> dict[str, Any]:
    requested_root = Path(warehouse_root)
    requested_path = Path(manifest_path)
    if requested_root.is_symlink() or requested_path.is_symlink():
        raise ValueError("Warehouse and snapshot manifest paths must not be symlinks")
    root = requested_root.resolve(strict=True)
    path = requested_path.resolve(strict=True)
    if path.name != SNAPSHOT_FILENAME or not path.is_file():
        raise ValueError("Snapshot manifest filename/type mismatch")
    expected_namespace = path.parent.parent.name
    if expected_namespace not in {"canaries", "snapshots"} or path.parent.parent.parent != root:
        raise ValueError("Snapshot manifest is outside the warehouse namespace")
    if sha256_path(path) != _require_sha(manifest_sha256, "manifest_sha256"):
        raise ValueError("Snapshot manifest SHA-256 mismatch")
    manifest = _load_json(path)
    if set(manifest) != _SNAPSHOT_FIELDS or manifest["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Snapshot manifest keys/schema mismatch")
    if manifest["complete"] is not True:
        raise ValueError("Incomplete snapshot is never loadable")
    if manifest["snapshot_id"] != path.parent.name:
        raise ValueError("Snapshot manifest ID differs from its directory")
    _require_plain_component(manifest["snapshot_id"], "snapshot_id")
    camera_id = _require_plain_component(manifest["camera_id"], "camera_id")
    if manifest["source_kind"] != SOURCE_KIND:
        raise ValueError("Snapshot top-level source_kind is not canonical-derived")
    if not isinstance(manifest["producer_identity"], dict) or not manifest["producer_identity"]:
        raise ValueError("Snapshot producer_identity must be a non-empty object")
    coverage = manifest["coverage_kind"]
    eligible = manifest["training_eligible"]
    if coverage == CANARY_COVERAGE:
        if eligible is not False or expected_namespace != "canaries":
            raise ValueError("Canary snapshot namespace/eligibility mismatch")
        if not verifier_only:
            raise ValueError("CANARY_SUBSET_NONTRAINING is verifier-only")
    elif coverage == FULL_SPLIT_COVERAGE:
        if eligible is not True or expected_namespace != "snapshots":
            raise ValueError("FULL_SPLIT snapshot namespace/eligibility mismatch")
    else:
        raise ValueError("Unknown snapshot coverage_kind")
    binding = _validate_dataset_binding(manifest["dataset_binding"])
    uids = manifest["covered_episode_uids"]
    if (
        not isinstance(uids, list)
        or uids != sorted(uids)
        or len(uids) != len(set(uids))
        or manifest["covered_episode_count"] != len(uids)
        or manifest["covered_episode_uid_digest"] != episode_uid_digest(uids)
    ):
        raise ValueError("Snapshot covered UID closure mismatch")
    entries = manifest["objects"]
    if not isinstance(entries, list) or len(entries) != len(uids):
        raise ValueError("Snapshot object coverage count mismatch")
    if sorted(entry.get("episode_uid") for entry in entries) != uids:
        raise ValueError("Snapshot object UIDs differ from covered UIDs")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _SNAPSHOT_OBJECT_FIELDS:
            raise ValueError("Snapshot object entry mismatch")
        object_manifest = verify_object(root, entry["object_content_sha256"])
        if object_manifest["derivation_key_sha256"] != entry["derivation_key_sha256"]:
            raise ValueError("Snapshot derivation key differs from immutable object")
        payload = object_manifest["manifest_payload"]
        for name in ("episode_uid", "camera_id", "source_kind", "num_frames"):
            if payload.get(name) != entry[name]:
                raise ValueError(f"Snapshot object {name} differs from immutable object")
        if entry["camera_id"] != camera_id or entry["source_kind"] != SOURCE_KIND:
            raise ValueError("Snapshot top-level camera/source differs from an object entry")
    if coverage == FULL_SPLIT_COVERAGE:
        if len(uids) != binding["resolved_train_episode_count"]:
            raise ValueError("FULL_SPLIT count differs from resolved train split")
        if frozen_train_episode_uid_digest(uids) != binding["train_episode_uid_digest"]:
            raise ValueError("FULL_SPLIT exact UID coverage failed")
    return manifest


class WarehouseEpisodeBundleStore:
    """Read production objects through one pinned dataset-bound snapshot."""

    def __init__(
        self,
        warehouse_root: str | Path,
        manifest_path: str | Path,
        *,
        manifest_sha256: str,
        verifier_only: bool = False,
    ) -> None:
        requested_root = Path(warehouse_root)
        if requested_root.is_symlink():
            raise ValueError("warehouse root must not be a symlink")
        self.root = requested_root.resolve(strict=True)
        self.manifest = validate_snapshot_manifest(
            self.root,
            manifest_path,
            manifest_sha256=manifest_sha256,
            verifier_only=verifier_only,
        )
        self._entries = {
            EpisodeBundleKey(entry["episode_uid"], entry["camera_id"], entry["source_kind"]): entry
            for entry in self.manifest["objects"]
        }

    def get(self, key: EpisodeBundleKey) -> EpisodeBundle:
        try:
            entry = self._entries[key]
        except KeyError as error:
            raise KeyError(f"Snapshot has no object for {key.canonical}") from error
        object_root = self.root / "objects" / entry["object_content_sha256"]
        manifest = verify_object(self.root, entry["object_content_sha256"])
        payload = manifest["manifest_payload"]
        descriptor_by_path = {
            descriptor["relative_path"]: descriptor for descriptor in manifest["artifacts"]
        }
        arrays: dict[str, np.ndarray] = {}
        for name in ARRAY_NAMES:
            relative = f"arrays/{name}.npy"
            descriptor = descriptor_by_path.get(relative)
            if descriptor is None:
                raise ValueError(f"Object is missing array descriptor: {name}")
            with _regular_child(object_root, relative).open("rb") as stream:
                value = np.load(stream, allow_pickle=False)
            if value.dtype.name != descriptor.get("dtype") or list(value.shape) != descriptor.get(
                "shape"
            ):
                raise ValueError(f"Object array dtype/shape differs from descriptor: {name}")
            value.flags.writeable = False
            arrays[name] = value
        frame_hashes = payload["frame_hashes"]
        identities = payload["identities"]
        episode = payload["episode"]
        return EpisodeBundle(
            key=key,
            split="train",
            cohort_id=episode["cohort_id"],
            dataset_name=episode["dataset_name"],
            source_episode_index=episode["source_episode_index"],
            width=payload["width"],
            height=payload["height"],
            time_base_num=payload["time_base_num"],
            time_base_den=payload["time_base_den"],
            gop_size=payload["gop_size"],
            source_frame_sha256=tuple(frame_hashes["source"]),
            canonical_frame_sha256=tuple(frame_hashes["canonical"]),
            fixture_frame_sha256=tuple(frame_hashes["fixture"]),
            tool_identity=identities["tool"],
            encoder_identity=identities["encoder"],
            decoder_identity=identities["decoder"],
            extractor_identity=identities["extractor"],
            **arrays,
        )

    def get_view(self, key: EpisodeBundleKey, observation_row: int):
        return self.get(key).view(observation_row)


def open_fixture_snapshot(
    snapshot_root: str | Path,
    *,
    snapshot_manifest_sha256: str,
    expected_dataset_binding: Mapping[str, Any],
    expected_train_episode_uids: Sequence[str],
    camera_id: str,
    source_kind: str,
    verifier_only: bool = False,
) -> WarehouseEpisodeBundleStore:
    """Open one pinned snapshot after closing its live dataset identity.

    ``snapshot_root`` is exactly ``<warehouse>/{snapshots,canaries}/<id>``;
    callers never infer or scan for a snapshot.  The expected binding must be
    resolved from the base dataset actually constructed by ``DatasetFactory``.
    """

    requested_snapshot = Path(snapshot_root)
    if requested_snapshot.is_symlink():
        raise ValueError("snapshot_root must not be a symlink")
    snapshot = requested_snapshot.resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError("snapshot_root must be a real directory")
    if snapshot.parent.name not in {"snapshots", "canaries"}:
        raise ValueError("snapshot_root must be inside snapshots/ or canaries/")
    warehouse_root = snapshot.parent.parent
    manifest_path = snapshot / SNAPSHOT_FILENAME
    manifest = validate_snapshot_manifest(
        warehouse_root,
        manifest_path,
        manifest_sha256=snapshot_manifest_sha256,
        verifier_only=verifier_only,
    )
    expected_binding = _validate_dataset_binding(dict(expected_dataset_binding))
    if manifest["dataset_binding"] != expected_binding:
        raise ValueError("Snapshot dataset binding differs from the live DatasetFactory inputs")
    train_uids = sorted(expected_train_episode_uids)
    if len(train_uids) != len(set(train_uids)):
        raise ValueError("Expected train episode UIDs must be unique")
    if frozen_train_episode_uid_digest(train_uids) != expected_binding["train_episode_uid_digest"]:
        raise ValueError("Live train UID set differs from its declared digest")
    if len(train_uids) != expected_binding["resolved_train_episode_count"]:
        raise ValueError("Live train UID count differs from its dataset binding")
    if manifest["camera_id"] != _require_plain_component(camera_id, "camera_id"):
        raise ValueError("Snapshot camera_id differs from adapter configuration")
    if source_kind != SOURCE_KIND or manifest["source_kind"] != source_kind:
        raise ValueError("Snapshot source_kind differs from adapter configuration")
    if manifest["coverage_kind"] == FULL_SPLIT_COVERAGE:
        if manifest["covered_episode_uids"] != train_uids:
            raise ValueError("FULL_SPLIT snapshot lacks exact live train UID coverage")
    else:
        if not verifier_only:
            raise ValueError("A canary subset cannot be opened by the training loader")
        if not set(manifest["covered_episode_uids"]).issubset(train_uids):
            raise ValueError("Canary snapshot contains an episode outside the live train split")
    return WarehouseEpisodeBundleStore(
        warehouse_root,
        manifest_path,
        manifest_sha256=snapshot_manifest_sha256,
        verifier_only=verifier_only,
    )


class AttemptJournal:
    """Minimal append-only attempt record with explicit STARTED semantics."""

    def __init__(self, warehouse_root: str | Path, attempt_id: str, plan: Mapping[str, Any]):
        root = ensure_warehouse_layout(warehouse_root)
        self.attempt_id = _require_plain_component(attempt_id, "attempt_id")
        self.path = root / "attempts" / self.attempt_id
        self.path.mkdir(mode=0o755, exist_ok=False)
        self.started_path = self.path / "ATTEMPT_STARTED.json"
        _write_json_exclusive(
            self.started_path,
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": self.attempt_id,
                "status": "STARTED",
                "plan_sha256": canonical_json_sha256(plan),
                "plan": dict(plan),
            },
        )
        self.started_path.chmod(0o444)
        _fsync_directory(self.path)
        _fsync_directory(self.path.parent)

    def finish(self, status: str, result: Mapping[str, Any]) -> Path:
        if status not in {"PASS", "FAILED", "STOPPED"}:
            raise ValueError("Attempt terminal status must be PASS/FAILED/STOPPED")
        path = self.path / f"ATTEMPT_{status}.json"
        _write_json_exclusive(
            path,
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": self.attempt_id,
                "status": status,
                "result": dict(result),
            },
        )
        path.chmod(0o444)
        _fsync_directory(self.path)
        return path

    def ready(self, result: Mapping[str, Any]) -> Path:
        """Publish a nonterminal result for authoritative outer finalization."""

        path = self.path / "ATTEMPT_READY.json"
        _write_json_exclusive(
            path,
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": self.attempt_id,
                "status": "READY",
                "result": dict(result),
            },
        )
        path.chmod(0o444)
        _fsync_directory(self.path)
        return path


class WarehouseExclusiveLease:
    """Fail-fast process coordination for one producer/publisher invocation."""

    def __init__(self, warehouse_root: str | Path) -> None:
        root = ensure_warehouse_layout(warehouse_root)
        self.path = root / "WAREHOUSE_PRODUCER.lock"
        self._stream = None

    def __enter__(self) -> "WarehouseExclusiveLease":
        self._stream = self.path.open("a+b")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise RuntimeError("Another warehouse producer owns the exclusive lease") from error
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


__all__ = [
    "ARRAY_NAMES",
    "ATTEMPT_SCHEMA_VERSION",
    "CANARY_COVERAGE",
    "DERIVATION_SCHEMA_VERSION",
    "FULL_SPLIT_COVERAGE",
    "OBJECT_MANIFEST_FILENAME",
    "SNAPSHOT_FILENAME",
    "SNAPSHOT_SCHEMA_VERSION",
    "SOURCE_KIND",
    "WAREHOUSE_OBJECT_SCHEMA_VERSION",
    "AttemptJournal",
    "PublishedObject",
    "WarehouseEpisodeBundleStore",
    "WarehouseExclusiveLease",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "derivation_key_sha256",
    "ensure_warehouse_layout",
    "episode_uid_digest",
    "frozen_train_episode_uid_digest",
    "open_fixture_snapshot",
    "publish_object",
    "sha256_path",
    "validate_snapshot_manifest",
    "verify_object",
    "write_snapshot_manifest",
]
