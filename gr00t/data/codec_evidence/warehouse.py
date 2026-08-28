"""Content-addressed storage for complete episode bundles.

This module deliberately wraps :mod:`episode_bundle` instead of defining a
second bundle format.  A warehouse object contains one complete, current
``codec-evidence-episode-bundle-v1`` tree.  The object name is the digest of
that tree's closed file descriptor set; a separate immutable binding enforces
that one derivation can never resolve to two different objects.

The only loader-visible commit is ``fixture_manifest.json``.  It is published
with a hard link, which is atomic and cannot replace an existing manifest.
Attempts and manual resumes are small provenance records only: they do not
infer a "latest" attempt and they never retry work automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .codec_contract import PRODUCTION_FIXTURE_MODE, codec_contract_sha256
from .contract import SHA256_RE
from .episode_bundle import (
    SOURCE_KINDS,
    EpisodeBundle,
    EpisodeBundleKey,
    EpisodeBundleStore,
    RawMotionVectorAudit,
    write_episode_bundle,
)


WAREHOUSE_OBJECT_SCHEMA_VERSION = "codec-evidence-warehouse-object-v1"
OBJECT_COMMIT_SCHEMA_VERSION = "codec-evidence-warehouse-object-commit-v1"
DERIVATION_BINDING_SCHEMA_VERSION = "codec-evidence-derivation-binding-v1"
FULL_FIXTURE_MANIFEST_SCHEMA_VERSION = "codec-evidence-full-fixture-manifest-v1"
ATTEMPT_SCHEMA_VERSION = "codec-evidence-warehouse-attempt-v1"
ATTEMPT_TERMINAL_FILENAME = "ATTEMPT_TERMINAL.json"

OBJECT_MANIFEST_FILENAME = "object_manifest.json"
OBJECT_COMMIT_FILENAME = "OBJECT_COMMITTED.json"
FINAL_MANIFEST_FILENAME = "fixture_manifest.json"
FULL_COVERAGE_KIND = "FULL_EPISODE_SET"

_OBJECT_MANIFEST_FIELDS = {
    "schema_version",
    "object_content_sha256",
    "episode_uid",
    "camera_id",
    "source_kind",
    "num_frames",
    "bundle_manifest",
    "artifacts",
}
_ARTIFACT_FIELDS = {"relative_path", "bytes", "sha256"}
_BUNDLE_DESCRIPTOR_FIELDS = {"relative_path", "bytes", "sha256"}
_COMMIT_FIELDS = {
    "schema_version",
    "object_content_sha256",
    "object_manifest_sha256",
}
_DERIVATION_BINDING_FIELDS = {
    "schema_version",
    "derivation_sha256",
    "object_content_sha256",
    "episode_uid",
    "camera_id",
    "source_kind",
}
_MANIFEST_OBJECT_FIELDS = {
    "episode_uid",
    "camera_id",
    "source_kind",
    "num_frames",
    "derivation_sha256",
    "object_content_sha256",
    "bundle_manifest_sha256",
    "bundle_manifest_bytes",
}
_FULL_MANIFEST_FIELDS = {
    "schema_version",
    "fixture_id",
    "complete",
    "coverage_kind",
    "camera_id",
    "source_kind",
    "expected_episode_count",
    "covered_episode_count",
    "covered_episode_uids",
    "covered_episode_uid_digest",
    "input_snapshot_sha256",
    "budget_sha256",
    "codec_contract",
    "codec_contract_sha256",
    "objects",
    "producer_identity",
}
_ATTEMPT_STARTED_FIELDS = {
    "schema_version",
    "attempt_id",
    "mode",
    "resume_from_attempt_id",
    "plan_sha256",
    "plan",
}
_ATTEMPT_TERMINAL_FIELDS = {
    "schema_version",
    "attempt_id",
    "plan_sha256",
    "status",
    "result",
}


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one byte encoding used for warehouse identities."""

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


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, canonical_json_bytes(value) + b"\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hardlink_commit(final_path: Path, payload: bytes, staging_root: Path) -> None:
    """Atomically create ``final_path`` without ever replacing it."""

    descriptor, temporary_name = tempfile.mkstemp(prefix="commit-", dir=staging_root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, final_path, follow_symlinks=False)
        _fsync_directory(final_path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_relative_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe warehouse artifact path: {relative_path!r}")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Symlinks are forbidden in warehouse objects: {relative_path}")
    resolved = (root / relative).resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"Warehouse artifact escapes its object: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"Warehouse artifact is not a regular file: {relative_path}")
    return resolved


def ensure_warehouse_layout(root: str | Path) -> Path:
    requested = Path(root)
    if not requested.is_absolute():
        raise ValueError("warehouse root must be absolute")
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    requested.mkdir(mode=0o755, parents=False, exist_ok=True)
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("warehouse root must be a real directory")
    for name in ("objects", "derivations", "attempts", ".staging"):
        path = resolved / name
        path.mkdir(mode=0o755, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"warehouse namespace is not a real directory: {name}")
    return resolved


def derivation_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the explicit input/recipe/toolchain derivation payload."""

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("derivation payload must be a non-empty mapping")
    # Round-tripping also rejects NaN and non-JSON values while detaching the
    # identity from mutable caller-owned containers.
    normalized = json.loads(canonical_json_bytes(dict(payload)))
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("derivation payload must normalize to a non-empty object")
    return canonical_json_sha256(normalized)


def episode_uid_digest(episode_uids: Sequence[str]) -> str:
    values = sorted(episode_uids)
    if len(values) != len(set(values)):
        raise ValueError("episode_uids must be unique")
    for value in values:
        _require_sha(value, "episode_uid")
    return canonical_json_sha256(values)


@dataclass(frozen=True)
class PublishedObject:
    key: EpisodeBundleKey
    derivation_sha256: str
    object_content_sha256: str
    num_frames: int
    bundle_manifest_sha256: str
    bundle_manifest_bytes: int
    object_path: Path
    reused: bool

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "episode_uid": self.key.episode_uid,
            "camera_id": self.key.camera_id,
            "source_kind": self.key.source_kind,
            "num_frames": self.num_frames,
            "derivation_sha256": self.derivation_sha256,
            "object_content_sha256": self.object_content_sha256,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "bundle_manifest_bytes": self.bundle_manifest_bytes,
        }


def _artifact_descriptors(root: Path) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden inside warehouse objects")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Only regular files are allowed inside warehouse objects")
        descriptors.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return descriptors


def _content_digest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "episode_uid": manifest["episode_uid"],
        "camera_id": manifest["camera_id"],
        "source_kind": manifest["source_kind"],
        "num_frames": manifest["num_frames"],
        "bundle_manifest": manifest["bundle_manifest"],
        "artifacts": manifest["artifacts"],
    }


def _object_paths(root: Path, digest: str) -> tuple[Path, Path, Path]:
    object_root = root / "objects" / digest
    return (
        object_root,
        object_root / "payload" / OBJECT_MANIFEST_FILENAME,
        object_root / OBJECT_COMMIT_FILENAME,
    )


def _validate_object_payload(
    root: Path,
    digest: str,
    *,
    require_commit: bool,
    full_bundle: bool,
) -> tuple[dict[str, Any], EpisodeBundleStore]:
    object_root, manifest_path, commit_path = _object_paths(root, digest)
    if object_root.is_symlink() or not object_root.is_dir():
        raise ValueError("Warehouse object is missing or a symlink")
    payload_root = object_root / "payload"
    if payload_root.is_symlink() or not payload_root.is_dir():
        raise ValueError("Warehouse object payload is missing or a symlink")
    manifest = _load_json(manifest_path)
    if (
        set(manifest) != _OBJECT_MANIFEST_FIELDS
        or manifest.get("schema_version") != WAREHOUSE_OBJECT_SCHEMA_VERSION
    ):
        raise ValueError("Warehouse object manifest keys/schema mismatch")
    if manifest.get("object_content_sha256") != digest:
        raise ValueError("Object directory and content digest differ")
    key = EpisodeBundleKey(
        manifest.get("episode_uid"),
        manifest.get("camera_id"),
        manifest.get("source_kind"),
    )
    if type(manifest.get("num_frames")) is not int or manifest["num_frames"] <= 0:
        raise ValueError("Warehouse object num_frames must be positive")
    descriptor = manifest.get("bundle_manifest")
    if not isinstance(descriptor, dict) or set(descriptor) != _BUNDLE_DESCRIPTOR_FIELDS:
        raise ValueError("Warehouse bundle manifest descriptor is malformed")
    expected_relative = f"bundle/{key.episode_uid}/{key.camera_id}/{key.source_kind}/bundle.json"
    if descriptor["relative_path"] != expected_relative:
        raise ValueError("Warehouse bundle manifest path is not canonical")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Warehouse object artifacts must be a non-empty list")
    observed_paths: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise ValueError("Warehouse object artifact descriptor is malformed")
        relative = artifact["relative_path"]
        if not isinstance(relative, str):
            raise ValueError("Warehouse object artifact path must be a string")
        path = _safe_relative_file(payload_root, relative)
        if type(artifact["bytes"]) is not int or artifact["bytes"] != path.stat().st_size:
            raise ValueError(f"Warehouse artifact byte count mismatch: {relative}")
        if sha256_path(path) != _require_sha(artifact["sha256"], "artifact.sha256"):
            raise ValueError(f"Warehouse artifact SHA-256 mismatch: {relative}")
        observed_paths.append(relative)
    if observed_paths != sorted(observed_paths) or len(observed_paths) != len(set(observed_paths)):
        raise ValueError("Warehouse artifact descriptors must be sorted and unique")
    if descriptor not in artifacts:
        raise ValueError("Bundle manifest descriptor is absent from object artifacts")
    if canonical_json_sha256(_content_digest_payload(manifest)) != digest:
        raise ValueError("Warehouse object content digest mismatch")
    expected_payload_files = set(observed_paths) | {OBJECT_MANIFEST_FILENAME}
    if require_commit:
        commit = _load_json(commit_path)
        if (
            set(commit) != _COMMIT_FIELDS
            or commit.get("schema_version") != OBJECT_COMMIT_SCHEMA_VERSION
        ):
            raise ValueError("Warehouse object commit marker keys/schema mismatch")
        expected_commit = {
            "schema_version": OBJECT_COMMIT_SCHEMA_VERSION,
            "object_content_sha256": digest,
            "object_manifest_sha256": sha256_path(manifest_path),
        }
        if commit != expected_commit:
            raise ValueError("Warehouse object commit marker does not bind its manifest")
    elif commit_path.exists() or commit_path.is_symlink():
        raise ValueError("Uncommitted object unexpectedly contains a commit marker")
    observed_payload_files: set[str] = set()
    for current, directory_names, file_names in os.walk(payload_root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if (current_path / directory_name).is_symlink():
                raise ValueError("Symlinks are forbidden inside warehouse objects")
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Only regular files are allowed inside warehouse objects")
            observed_payload_files.add(path.relative_to(payload_root).as_posix())
    if observed_payload_files != expected_payload_files:
        raise ValueError("Warehouse object contains unclosed or missing files")
    root_entries = {path.name for path in object_root.iterdir()}
    expected_root_entries = {"payload"}
    if require_commit:
        expected_root_entries.add(OBJECT_COMMIT_FILENAME)
    if root_entries != expected_root_entries:
        raise ValueError("Warehouse object root contains unclosed or missing entries")

    bundle_root = payload_root / "bundle"
    store = EpisodeBundleStore(
        bundle_root,
        manifest_sha256_by_key={key: descriptor["sha256"]},
    )
    identity = store.identity(key)
    if full_bundle:
        bundle = store.get(key)
        if bundle.num_frames != manifest["num_frames"]:
            raise ValueError("Warehouse object frame count differs from its bundle")
        bundle_manifest = _load_json(_safe_relative_file(payload_root, expected_relative))
        if bundle_manifest.get("raw_motion_vector_audit") is not None:
            store.audit_raw_motion_vectors(key)
        if bundle_manifest.get("accepted_bitstream") is not None:
            store.audit_accepted_bitstream(key)
        if identity["codec_contract"]["fixture_mode"] == PRODUCTION_FIXTURE_MODE and (
            bundle_manifest.get("raw_motion_vector_audit") is None
            or bundle_manifest.get("accepted_bitstream") is None
        ):
            raise ValueError("Production warehouse object lacks required audit artifacts")
    return manifest, store


def verify_object(
    warehouse_root: str | Path,
    object_content_sha256: str,
    *,
    full_bundle: bool = True,
) -> dict[str, Any]:
    requested = Path(warehouse_root)
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    root = requested.resolve(strict=True)
    digest = _require_sha(object_content_sha256, "object_content_sha256")
    manifest, _store = _validate_object_payload(
        root,
        digest,
        require_commit=True,
        full_bundle=full_bundle,
    )
    return manifest


def _binding_path(root: Path, digest: str) -> Path:
    return root / "derivations" / f"{digest}.json"


def _validate_binding(root: Path, derivation_digest: str) -> dict[str, Any]:
    binding = _load_json(_binding_path(root, derivation_digest))
    if (
        set(binding) != _DERIVATION_BINDING_FIELDS
        or binding.get("schema_version") != DERIVATION_BINDING_SCHEMA_VERSION
        or binding.get("derivation_sha256") != derivation_digest
    ):
        raise ValueError("Derivation binding keys/schema/digest mismatch")
    _require_sha(binding.get("object_content_sha256"), "object_content_sha256")
    EpisodeBundleKey(
        binding.get("episode_uid"),
        binding.get("camera_id"),
        binding.get("source_kind"),
    )
    return binding


def _reserve_derivation(root: Path, binding: dict[str, Any]) -> None:
    path = _binding_path(root, binding["derivation_sha256"])
    try:
        _hardlink_commit(path, canonical_json_bytes(binding) + b"\n", root / ".staging")
    except FileExistsError:
        existing = _validate_binding(root, binding["derivation_sha256"])
        if existing != binding:
            raise ValueError(
                "One derivation key produced a different content digest; refusing reuse"
            )


def resolve_derivation(
    warehouse_root: str | Path,
    derivation_payload: Mapping[str, Any],
    *,
    expected_key: EpisodeBundleKey | None = None,
) -> PublishedObject:
    """Resolve and fully verify one reusable derivation; never scan the warehouse."""

    requested = Path(warehouse_root)
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    root = requested.resolve(strict=True)
    derivation_digest = derivation_sha256(derivation_payload)
    binding = _validate_binding(root, derivation_digest)
    key = EpisodeBundleKey(binding["episode_uid"], binding["camera_id"], binding["source_kind"])
    if expected_key is not None and key != expected_key:
        raise ValueError("Derivation binding key differs from the expected episode")
    object_digest = binding["object_content_sha256"]
    manifest = verify_object(root, object_digest, full_bundle=True)
    if (
        manifest["episode_uid"] != key.episode_uid
        or manifest["camera_id"] != key.camera_id
        or manifest["source_kind"] != key.source_kind
    ):
        raise ValueError("Derivation binding key differs from its immutable object")
    descriptor = manifest["bundle_manifest"]
    return PublishedObject(
        key=key,
        derivation_sha256=derivation_digest,
        object_content_sha256=object_digest,
        num_frames=manifest["num_frames"],
        bundle_manifest_sha256=descriptor["sha256"],
        bundle_manifest_bytes=descriptor["bytes"],
        object_path=root / "objects" / object_digest,
        reused=True,
    )


def reusable_derivation(
    warehouse_root: str | Path,
    derivation_payload: Mapping[str, Any],
    *,
    expected_key: EpisodeBundleKey | None = None,
) -> PublishedObject | None:
    """Return one fully committed reusable object, or ``None`` for pending work.

    This is the explicit manual-resume lookup.  It addresses exactly one
    caller-supplied derivation and never scans for a latest attempt or a nearby
    object.  An absent binding, a binding reserved before a crash, or an object
    whose payload was published without its commit marker is pending work.
    Corrupt committed state still fails closed through :func:`resolve_derivation`.
    """

    requested = Path(warehouse_root)
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    root = requested.resolve(strict=True)
    derivation_digest = derivation_sha256(derivation_payload)
    binding_path = _binding_path(root, derivation_digest)
    if not binding_path.exists() and not binding_path.is_symlink():
        return None
    binding = _validate_binding(root, derivation_digest)
    key = EpisodeBundleKey(binding["episode_uid"], binding["camera_id"], binding["source_kind"])
    if expected_key is not None and key != expected_key:
        raise ValueError("Derivation binding key differs from the expected episode")
    object_root, _manifest_path, commit_path = _object_paths(root, binding["object_content_sha256"])
    if not object_root.exists() and not object_root.is_symlink():
        return None
    if object_root.is_symlink() or not object_root.is_dir():
        raise ValueError("Derivation object path is not a real directory")
    if not commit_path.exists() and not commit_path.is_symlink():
        return None
    return resolve_derivation(root, derivation_payload, expected_key=expected_key)


def _restore_writable_tree(path: Path) -> None:
    if not path.exists():
        return
    for current in path.rglob("*"):
        if current.is_dir():
            current.chmod(0o755)
        elif current.exists():
            current.chmod(0o644)
    path.chmod(0o755)


def publish_episode_object(
    warehouse_root: str | Path,
    *,
    derivation_payload: Mapping[str, Any],
    episode_uid: str,
    camera_id: str,
    source_kind: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    raw_mv_offsets: np.ndarray | None = None,
    raw_mv_fields: Mapping[str, np.ndarray] | None = None,
    accepted_bitstream: str | Path | None = None,
    prepublish_guard: Callable[[int], None] | None = None,
) -> PublishedObject:
    """Publish one complete bundle as a non-overwritable content object."""

    root = ensure_warehouse_layout(warehouse_root)
    key = EpisodeBundleKey(episode_uid, camera_id, source_kind)
    derivation_digest = derivation_sha256(derivation_payload)
    stage: Path | None = Path(tempfile.mkdtemp(prefix="object-", dir=root / ".staging"))
    try:
        bundle_manifest_path = write_episode_bundle(
            stage / "bundle",
            episode_uid=episode_uid,
            camera_id=camera_id,
            source_kind=source_kind,
            metadata=metadata,
            arrays=arrays,
            raw_mv_offsets=raw_mv_offsets,
            raw_mv_fields=raw_mv_fields,
            accepted_bitstream=accepted_bitstream,
        )
        bundle_manifest_relative = bundle_manifest_path.relative_to(stage).as_posix()
        artifacts = _artifact_descriptors(stage)
        bundle_descriptor = next(
            descriptor
            for descriptor in artifacts
            if descriptor["relative_path"] == bundle_manifest_relative
        )
        bundle_store = EpisodeBundleStore(
            stage / "bundle",
            manifest_sha256_by_key={key: bundle_descriptor["sha256"]},
        )
        bundle = bundle_store.get(key)
        if bundle.num_frames <= 0:
            raise ValueError("Warehouse objects require a non-empty complete episode")
        object_manifest: dict[str, Any] = {
            "schema_version": WAREHOUSE_OBJECT_SCHEMA_VERSION,
            "object_content_sha256": "0" * 64,
            "episode_uid": key.episode_uid,
            "camera_id": key.camera_id,
            "source_kind": key.source_kind,
            "num_frames": bundle.num_frames,
            "bundle_manifest": bundle_descriptor,
            "artifacts": artifacts,
        }
        object_digest = canonical_json_sha256(_content_digest_payload(object_manifest))
        object_manifest["object_content_sha256"] = object_digest
        _write_json_exclusive(stage / OBJECT_MANIFEST_FILENAME, object_manifest)
        stage_bytes = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
        if prepublish_guard is not None:
            prepublish_guard(stage_bytes)

        binding = {
            "schema_version": DERIVATION_BINDING_SCHEMA_VERSION,
            "derivation_sha256": derivation_digest,
            "object_content_sha256": object_digest,
            "episode_uid": key.episode_uid,
            "camera_id": key.camera_id,
            "source_kind": key.source_kind,
        }
        _reserve_derivation(root, binding)

        destination, destination_manifest, destination_commit = _object_paths(root, object_digest)
        reused = False
        try:
            destination.mkdir(mode=0o755, exist_ok=False)
            _fsync_directory(root / "objects")
        except FileExistsError:
            reused = True
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Object digest path is not a real directory")

        payload_destination = destination / "payload"
        if not payload_destination.exists() and not payload_destination.is_symlink():
            if set(destination.iterdir()):
                raise ValueError("Reserved object path contains an unexpected partial closure")
            try:
                os.rename(stage, payload_destination)
                stage = None
                _fsync_directory(destination)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                reused = True
        if stage is not None:
            reused = True
            existing, _store = _validate_object_payload(
                root,
                object_digest,
                require_commit=destination_commit.exists(),
                full_bundle=True,
            )
            if canonical_json_bytes(existing) != canonical_json_bytes(object_manifest):
                raise ValueError("Existing object digest path contains a different closure")
        elif destination_commit.exists():
            # This is possible only when a concurrent exact publisher committed
            # after our payload rename.  Full verification below remains the
            # authority; no path is ever overwritten.
            reused = True

        if not payload_destination.is_dir() or payload_destination.is_symlink():
            raise ValueError("Object payload publish did not produce one real directory")

        if not destination_commit.exists():
            commit = {
                "schema_version": OBJECT_COMMIT_SCHEMA_VERSION,
                "object_content_sha256": object_digest,
                "object_manifest_sha256": sha256_path(destination_manifest),
            }
            try:
                _hardlink_commit(
                    destination_commit,
                    canonical_json_bytes(commit) + b"\n",
                    root / ".staging",
                )
            except FileExistsError:
                # A concurrent publisher may only have committed the same exact
                # object; the full verification below decides that fact.
                pass
        verify_object(root, object_digest, full_bundle=True)
        for path in destination.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(
            (item for item in destination.rglob("*") if item.is_dir()), reverse=True
        ):
            path.chmod(0o555)
        destination.chmod(0o555)
        _fsync_directory(root / "objects")
        return PublishedObject(
            key=key,
            derivation_sha256=derivation_digest,
            object_content_sha256=object_digest,
            num_frames=bundle.num_frames,
            bundle_manifest_sha256=bundle_descriptor["sha256"],
            bundle_manifest_bytes=bundle_descriptor["bytes"],
            object_path=destination,
            reused=reused,
        )
    finally:
        if stage is not None and stage.exists():
            _restore_writable_tree(stage)
            shutil.rmtree(stage)


# The shorter donor-era name remains a convenient API without carrying any of
# the donor's obsolete EpisodeBundle constructor assumptions.
publish_object = publish_episode_object


class AttemptJournal:
    """One exact plan-bound attempt or explicitly named manual resume."""

    def __init__(self, root: Path, path: Path, started: Mapping[str, Any]):
        self.root = root
        self.path = path
        self.attempt_id = started["attempt_id"]
        self.plan_sha256 = started["plan_sha256"]

    @staticmethod
    def _normalize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, Mapping) or not plan:
            raise ValueError("attempt plan must be a non-empty mapping")
        value = json.loads(canonical_json_bytes(dict(plan)))
        if not isinstance(value, dict) or not value:
            raise ValueError("attempt plan must normalize to a non-empty object")
        return value

    @classmethod
    def start(
        cls,
        warehouse_root: str | Path,
        *,
        attempt_id: str,
        plan: Mapping[str, Any],
    ) -> "AttemptJournal":
        return cls._create(
            warehouse_root,
            attempt_id=attempt_id,
            plan=plan,
            resume_from_attempt_id=None,
        )

    @classmethod
    def manual_resume(
        cls,
        warehouse_root: str | Path,
        *,
        attempt_id: str,
        resume_from_attempt_id: str,
        plan: Mapping[str, Any],
    ) -> "AttemptJournal":
        if resume_from_attempt_id is None:
            raise ValueError("manual resume requires one explicit resume_from_attempt_id")
        return cls._create(
            warehouse_root,
            attempt_id=attempt_id,
            plan=plan,
            resume_from_attempt_id=resume_from_attempt_id,
        )

    @classmethod
    def _create(
        cls,
        warehouse_root: str | Path,
        *,
        attempt_id: str,
        plan: Mapping[str, Any],
        resume_from_attempt_id: str | None,
    ) -> "AttemptJournal":
        root = ensure_warehouse_layout(warehouse_root)
        attempt_id = _require_plain_component(attempt_id, "attempt_id")
        normalized = cls._normalize_plan(plan)
        plan_digest = canonical_json_sha256(normalized)
        mode = "INITIAL" if resume_from_attempt_id is None else "MANUAL_RESUME"
        if resume_from_attempt_id is not None:
            resume_from_attempt_id = _require_plain_component(
                resume_from_attempt_id, "resume_from_attempt_id"
            )
            if resume_from_attempt_id == attempt_id:
                raise ValueError("manual resume must use a new attempt_id")
            prior_path = root / "attempts" / resume_from_attempt_id / "ATTEMPT_STARTED.json"
            prior = _load_json(prior_path)
            if (
                set(prior) != _ATTEMPT_STARTED_FIELDS
                or prior.get("schema_version") != ATTEMPT_SCHEMA_VERSION
            ):
                raise ValueError("Prior attempt start record fields differ from the contract")
            if prior.get("attempt_id") != resume_from_attempt_id:
                raise ValueError("Prior attempt ID differs from its explicit directory")
            if canonical_json_sha256(prior.get("plan")) != prior.get("plan_sha256"):
                raise ValueError("Prior attempt plan digest is internally inconsistent")
            if prior.get("plan_sha256") != plan_digest or prior.get("plan") != normalized:
                raise ValueError("Manual resume plan differs from the exact prior plan")
            prior_terminal = _load_json(
                root / "attempts" / resume_from_attempt_id / ATTEMPT_TERMINAL_FILENAME
            )
            if (
                set(prior_terminal) != _ATTEMPT_TERMINAL_FIELDS
                or prior_terminal.get("schema_version") != ATTEMPT_SCHEMA_VERSION
                or prior_terminal.get("attempt_id") != resume_from_attempt_id
                or prior_terminal.get("plan_sha256") != plan_digest
                or prior_terminal.get("status") not in {"FAILED", "STOPPED"}
                or not isinstance(prior_terminal.get("result"), dict)
            ):
                raise ValueError("Manual resume requires an exact FAILED or STOPPED prior attempt")
        if (root / FINAL_MANIFEST_FILENAME).exists() or (
            root / FINAL_MANIFEST_FILENAME
        ).is_symlink():
            raise FileExistsError("A finalized fixture cannot be resumed")
        path = root / "attempts" / attempt_id
        path.mkdir(mode=0o755, exist_ok=False)
        started = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "mode": mode,
            "resume_from_attempt_id": resume_from_attempt_id,
            "plan_sha256": plan_digest,
            "plan": normalized,
        }
        _write_json_exclusive(path / "ATTEMPT_STARTED.json", started)
        _fsync_directory(path)
        _fsync_directory(path.parent)
        return cls(root, path, started)

    def record_object(self, published: PublishedObject) -> Path:
        if not isinstance(published, PublishedObject):
            raise TypeError("published must be a PublishedObject")
        path = self.path / "objects" / f"{published.key.episode_uid}.json"
        _write_json_exclusive(
            path,
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": self.attempt_id,
                "plan_sha256": self.plan_sha256,
                "object": published.manifest_entry(),
            },
        )
        _fsync_directory(path.parent)
        return path

    def reusable_object(
        self,
        derivation_payload: Mapping[str, Any],
        *,
        expected_key: EpisodeBundleKey | None = None,
    ) -> PublishedObject | None:
        return reusable_derivation(
            self.root,
            derivation_payload,
            expected_key=expected_key,
        )

    def finish(self, status: str, result: Mapping[str, Any]) -> Path:
        if status not in {"READY", "FAILED", "STOPPED"}:
            raise ValueError("attempt status must be READY, FAILED, or STOPPED")
        if not isinstance(result, Mapping):
            raise TypeError("attempt result must be a mapping")
        path = self.path / ATTEMPT_TERMINAL_FILENAME
        _write_json_exclusive(
            path,
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": self.attempt_id,
                "plan_sha256": self.plan_sha256,
                "status": status,
                "result": dict(result),
            },
        )
        _fsync_directory(self.path)
        return path


def _entry_from_object(item: PublishedObject | Mapping[str, Any]) -> dict[str, Any]:
    entry = item.manifest_entry() if isinstance(item, PublishedObject) else dict(item)
    if set(entry) != _MANIFEST_OBJECT_FIELDS:
        raise ValueError("Fixture object entry fields differ from the contract")
    return entry


def _validate_manifest_mapping(
    root: Path,
    manifest: Any,
    *,
    verify_objects: bool,
) -> dict[str, Any]:
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _FULL_MANIFEST_FIELDS
        or manifest.get("schema_version") != FULL_FIXTURE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Full fixture manifest fields/schema mismatch")
    _require_plain_component(manifest.get("fixture_id"), "fixture_id")
    if manifest.get("complete") is not True or manifest.get("coverage_kind") != FULL_COVERAGE_KIND:
        raise ValueError("Full fixture manifest is incomplete or not full episode coverage")
    camera_id = _require_plain_component(manifest.get("camera_id"), "camera_id")
    source_kind = manifest.get("source_kind")
    if source_kind not in SOURCE_KINDS:
        raise ValueError("Full fixture manifest source_kind is unsupported")
    expected_count = manifest.get("expected_episode_count")
    covered_count = manifest.get("covered_episode_count")
    if type(expected_count) is not int or expected_count <= 0 or covered_count != expected_count:
        raise ValueError("Full fixture expected/covered episode counts differ")
    uids = manifest.get("covered_episode_uids")
    if (
        not isinstance(uids, list)
        or uids != sorted(uids)
        or len(uids) != expected_count
        or len(uids) != len(set(uids))
        or manifest.get("covered_episode_uid_digest") != episode_uid_digest(uids)
    ):
        raise ValueError("Full fixture UID closure mismatch")
    _require_sha(manifest.get("input_snapshot_sha256"), "input_snapshot_sha256")
    _require_sha(manifest.get("budget_sha256"), "budget_sha256")
    contract = manifest.get("codec_contract")
    if not isinstance(contract, dict):
        raise ValueError("Full fixture codec_contract must be an object")
    if codec_contract_sha256(contract) != _require_sha(
        manifest.get("codec_contract_sha256"), "codec_contract_sha256"
    ):
        raise ValueError("Full fixture codec contract digest mismatch")
    if contract.get("fixture_mode") != PRODUCTION_FIXTURE_MODE:
        raise ValueError("A complete full fixture requires production fixture mode")
    if contract.get("camera_id") != camera_id or contract.get("source_kind") != source_kind:
        raise ValueError("Full fixture codec contract differs from camera/source identity")
    if not isinstance(manifest.get("producer_identity"), dict) or not manifest["producer_identity"]:
        raise ValueError("Full fixture producer_identity must be a non-empty object")
    entries = manifest.get("objects")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError("Full fixture object count differs from expected coverage")
    if entries != sorted(
        entries,
        key=lambda entry: (
            entry.get("episode_uid"),
            entry.get("camera_id"),
            entry.get("source_kind"),
        ),
    ):
        raise ValueError("Full fixture objects must be in canonical key order")
    observed_uids: list[str] = []
    seen_keys: set[EpisodeBundleKey] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _MANIFEST_OBJECT_FIELDS:
            raise ValueError("Full fixture object entry is malformed")
        key = EpisodeBundleKey(entry["episode_uid"], entry["camera_id"], entry["source_kind"])
        if key in seen_keys or key.camera_id != camera_id or key.source_kind != source_kind:
            raise ValueError("Full fixture object keys are duplicate or mixed")
        seen_keys.add(key)
        observed_uids.append(key.episode_uid)
        if type(entry["num_frames"]) is not int or entry["num_frames"] <= 0:
            raise ValueError("Full fixture object num_frames must be positive")
        derivation_digest = _require_sha(entry["derivation_sha256"], "derivation_sha256")
        object_digest = _require_sha(entry["object_content_sha256"], "object_content_sha256")
        binding = _validate_binding(root, derivation_digest)
        expected_binding = {
            "schema_version": DERIVATION_BINDING_SCHEMA_VERSION,
            "derivation_sha256": derivation_digest,
            "object_content_sha256": object_digest,
            "episode_uid": key.episode_uid,
            "camera_id": key.camera_id,
            "source_kind": key.source_kind,
        }
        if binding != expected_binding:
            raise ValueError("Full fixture derivation binding differs from its entry")
        object_manifest = verify_object(
            root,
            object_digest,
            full_bundle=verify_objects,
        )
        descriptor = object_manifest["bundle_manifest"]
        for name, expected in (
            ("episode_uid", key.episode_uid),
            ("camera_id", key.camera_id),
            ("source_kind", key.source_kind),
            ("num_frames", entry["num_frames"]),
        ):
            if object_manifest[name] != expected:
                raise ValueError(f"Full fixture object {name} differs from its entry")
        if (
            descriptor["sha256"] != entry["bundle_manifest_sha256"]
            or descriptor["bytes"] != entry["bundle_manifest_bytes"]
        ):
            raise ValueError("Full fixture bundle descriptor differs from its object")
        identity = EpisodeBundleStore(
            root / "objects" / object_digest / "payload" / "bundle",
            manifest_sha256_by_key={key: descriptor["sha256"]},
        ).identity(key)
        if (
            identity["codec_contract_sha256"] != manifest["codec_contract_sha256"]
            or identity["codec_contract"] != contract
        ):
            raise ValueError("Full fixture object codec contract drifted")
    if observed_uids != uids:
        raise ValueError("Full fixture object UIDs differ from covered UID closure")
    return manifest


def commit_full_fixture_manifest(
    warehouse_root: str | Path,
    *,
    fixture_id: str,
    expected_episode_count: int,
    expected_episode_uids: Sequence[str],
    objects: Sequence[PublishedObject | Mapping[str, Any]],
    input_snapshot_sha256: str,
    budget_sha256: str,
    codec_contract: Mapping[str, Any],
    producer_identity: Mapping[str, Any],
) -> Path:
    """Fully verify and atomically hard-link the one training manifest."""

    root = ensure_warehouse_layout(warehouse_root)
    fixture_id = _require_plain_component(fixture_id, "fixture_id")
    if type(expected_episode_count) is not int or expected_episode_count <= 0:
        raise ValueError("expected_episode_count must be a positive integer")
    uids = sorted(expected_episode_uids)
    if not uids:
        raise ValueError("A full fixture must cover at least one episode")
    if len(uids) != len(set(uids)):
        raise ValueError("expected_episode_uids must be unique")
    for uid in uids:
        _require_sha(uid, "expected_episode_uids")
    if len(uids) != expected_episode_count:
        raise ValueError("Frozen expected episode count differs from its UID closure")
    entries = [_entry_from_object(item) for item in objects]
    entries.sort(key=lambda entry: (entry["episode_uid"], entry["camera_id"], entry["source_kind"]))
    if not entries:
        raise ValueError("A full fixture must contain at least one object")
    normalized_contract = json.loads(canonical_json_bytes(dict(codec_contract)))
    manifest = {
        "schema_version": FULL_FIXTURE_MANIFEST_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "complete": True,
        "coverage_kind": FULL_COVERAGE_KIND,
        "camera_id": entries[0]["camera_id"],
        "source_kind": entries[0]["source_kind"],
        "expected_episode_count": expected_episode_count,
        "covered_episode_count": len(entries),
        "covered_episode_uids": uids,
        "covered_episode_uid_digest": episode_uid_digest(uids),
        "input_snapshot_sha256": _require_sha(input_snapshot_sha256, "input_snapshot_sha256"),
        "budget_sha256": _require_sha(budget_sha256, "budget_sha256"),
        "codec_contract": normalized_contract,
        "codec_contract_sha256": codec_contract_sha256(normalized_contract),
        "objects": entries,
        "producer_identity": json.loads(canonical_json_bytes(dict(producer_identity))),
    }
    _validate_manifest_mapping(root, manifest, verify_objects=True)
    final_path = root / FINAL_MANIFEST_FILENAME
    _hardlink_commit(
        final_path,
        canonical_json_bytes(manifest) + b"\n",
        root / ".staging",
    )
    return final_path


def validate_full_fixture_manifest(
    warehouse_root: str | Path,
    *,
    fixture_manifest_sha256: str,
    verify_objects: bool = True,
) -> dict[str, Any]:
    requested = Path(warehouse_root)
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    root = requested.resolve(strict=True)
    manifest_path = root / FINAL_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Final fixture manifest is missing or a symlink")
    if sha256_path(manifest_path) != _require_sha(
        fixture_manifest_sha256, "fixture_manifest_sha256"
    ):
        raise ValueError("Final fixture manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path)
    return _validate_manifest_mapping(root, manifest, verify_objects=verify_objects)


class WarehouseEpisodeBundleStore:
    """Read current EpisodeBundle objects through one pinned full manifest."""

    def __init__(
        self,
        warehouse_root: str | Path,
        *,
        fixture_manifest_sha256: str,
    ) -> None:
        requested = Path(warehouse_root)
        if requested.is_symlink():
            raise ValueError("warehouse root must not be a symlink")
        self.root = requested.resolve(strict=True)
        self.manifest = validate_full_fixture_manifest(
            self.root,
            fixture_manifest_sha256=fixture_manifest_sha256,
            verify_objects=False,
        )
        self.coverage_kind = self.manifest["coverage_kind"]
        self.fixture_manifest_schema_version = self.manifest["schema_version"]
        self._entries = {
            EpisodeBundleKey(entry["episode_uid"], entry["camera_id"], entry["source_kind"]): entry
            for entry in self.manifest["objects"]
        }
        self._stores: dict[EpisodeBundleKey, EpisodeBundleStore] = {}

    @property
    def keys(self) -> frozenset[EpisodeBundleKey]:
        return frozenset(self._entries)

    def _store(self, key: EpisodeBundleKey) -> EpisodeBundleStore:
        if key not in self._entries:
            raise KeyError(f"Full fixture has no object for {key.canonical}")
        cached = self._stores.get(key)
        if cached is not None:
            return cached
        entry = self._entries[key]
        manifest = verify_object(
            self.root,
            entry["object_content_sha256"],
            full_bundle=True,
        )
        descriptor = manifest["bundle_manifest"]
        store = EpisodeBundleStore(
            self.root / "objects" / entry["object_content_sha256"] / "payload" / "bundle",
            manifest_sha256_by_key={key: descriptor["sha256"]},
        )
        self._stores[key] = store
        return store

    def get(self, key: EpisodeBundleKey) -> EpisodeBundle:
        return self._store(key).get(key)

    def get_view(self, key: EpisodeBundleKey, observation_row: int):
        return self._store(key).get_view(key, observation_row)

    def identity(self, key: EpisodeBundleKey) -> Mapping[str, Any]:
        return self._store(key).identity(key)

    def audit_raw_motion_vectors(
        self,
        key: EpisodeBundleKey,
        *,
        frame_index: int | None = None,
    ) -> RawMotionVectorAudit:
        return self._store(key).audit_raw_motion_vectors(key, frame_index=frame_index)

    def audit_accepted_bitstream(self, key: EpisodeBundleKey) -> Path:
        return self._store(key).audit_accepted_bitstream(key)


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "ATTEMPT_TERMINAL_FILENAME",
    "DERIVATION_BINDING_SCHEMA_VERSION",
    "FINAL_MANIFEST_FILENAME",
    "FULL_COVERAGE_KIND",
    "FULL_FIXTURE_MANIFEST_SCHEMA_VERSION",
    "OBJECT_COMMIT_FILENAME",
    "OBJECT_COMMIT_SCHEMA_VERSION",
    "OBJECT_MANIFEST_FILENAME",
    "WAREHOUSE_OBJECT_SCHEMA_VERSION",
    "AttemptJournal",
    "PublishedObject",
    "WarehouseEpisodeBundleStore",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "commit_full_fixture_manifest",
    "derivation_sha256",
    "ensure_warehouse_layout",
    "episode_uid_digest",
    "publish_episode_object",
    "publish_object",
    "reusable_derivation",
    "resolve_derivation",
    "sha256_path",
    "validate_full_fixture_manifest",
    "verify_object",
]
