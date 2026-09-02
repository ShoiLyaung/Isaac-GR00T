"""Exact executable/source closure for the one authorized C1 attempt.

The implementation manifest is intentionally independent from a produced
fixture snapshot.  It freezes every source, test, executable, and loaded
shared library selected by the C1 runner, then re-hashes the same exact paths
before the attempt reaches a terminal state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .warehouse import canonical_json_sha256, sha256_path


IMPLEMENTATION_MANIFEST_SCHEMA_VERSION = "codec-evidence-c1-implementation-manifest-v1"
_MANIFEST_FIELDS = {"schema_version", "run_id", "complete", "files"}
_FILE_FIELDS = {"path", "sha256", "bytes"}
_ROLE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def implementation_file_identity(path: str | Path) -> dict[str, Any]:
    """Resolve and hash one regular implementation artifact."""

    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"Implementation artifact must not be a symlink: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Implementation artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_path(resolved),
        "bytes": resolved.stat().st_size,
    }


def implementation_file_identities(
    paths_by_role: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    """Create one deterministically ordered exact role-to-file mapping."""

    if not paths_by_role:
        raise ValueError("Implementation file set must be non-empty")
    result: dict[str, dict[str, Any]] = {}
    for role in sorted(paths_by_role):
        if not isinstance(role, str) or _ROLE_RE.fullmatch(role) is None:
            raise ValueError(f"Invalid implementation role: {role!r}")
        result[role] = implementation_file_identity(paths_by_role[role])
    return result


def build_implementation_manifest(
    *, run_id: str, files: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Build (but do not publish) the exact frozen manifest payload."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be non-empty")
    normalized = _validate_files(files)
    return {
        "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "complete": True,
        "files": normalized,
    }


def write_implementation_manifest_exclusive(
    path: str | Path, *, run_id: str, files: Mapping[str, Mapping[str, Any]]
) -> Path:
    """Publish a C1 implementation manifest without overwriting any file."""

    destination = Path(path)
    if not destination.is_absolute() or not destination.parent.is_dir():
        raise ValueError("Implementation manifest needs an existing absolute parent")
    payload = build_implementation_manifest(run_id=run_id, files=files)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    destination.chmod(0o444)
    return destination.resolve(strict=True)


def _validate_files(files: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(files, dict) or not files:
        raise ValueError("Implementation manifest files must be a non-empty object")
    normalized: dict[str, dict[str, Any]] = {}
    observed_paths: set[str] = set()
    for role in sorted(files):
        if not isinstance(role, str) or _ROLE_RE.fullmatch(role) is None:
            raise ValueError(f"Invalid implementation role: {role!r}")
        descriptor = files[role]
        if not isinstance(descriptor, dict) or set(descriptor) != _FILE_FIELDS:
            raise ValueError(f"Implementation descriptor fields differ for role {role}")
        path = descriptor["path"]
        digest = descriptor["sha256"]
        size = descriptor["bytes"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"Implementation path must be absolute for role {role}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Implementation SHA-256 is invalid for role {role}")
        if type(size) is not int or size < 0:
            raise ValueError(f"Implementation byte count is invalid for role {role}")
        if path in observed_paths:
            raise ValueError(f"One implementation file has multiple roles: {path}")
        observed_paths.add(path)
        normalized[role] = {"path": path, "sha256": digest, "bytes": size}
    return normalized


def verify_implementation_manifest(
    path: str | Path,
    *,
    expected_run_id: str,
    expected_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the exact manifest and re-hash every frozen artifact now."""

    requested = Path(path)
    if requested.is_symlink():
        raise ValueError("Implementation manifest must not be a symlink")
    manifest_path = requested.resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError("Implementation manifest is missing, non-regular, or a symlink")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("Implementation manifest fields differ from the frozen schema")
    if (
        manifest["schema_version"] != IMPLEMENTATION_MANIFEST_SCHEMA_VERSION
        or manifest["complete"] is not True
        or manifest["run_id"] != expected_run_id
    ):
        raise ValueError("Implementation manifest schema/run/completeness mismatch")
    declared = _validate_files(manifest["files"])
    expected = _validate_files(dict(expected_files))
    if declared != expected:
        raise ValueError("Implementation manifest does not equal the executable file closure")
    for role, frozen in expected.items():
        current = implementation_file_identity(frozen["path"])
        if current != frozen:
            raise ValueError(f"Implementation artifact drifted after freeze: {role}")
    return {
        "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
        "path": str(manifest_path),
        "sha256": sha256_path(manifest_path),
        "payload_sha256": canonical_json_sha256(manifest),
        "run_id": expected_run_id,
        "file_count": len(expected),
        "files": expected,
    }


__all__ = [
    "IMPLEMENTATION_MANIFEST_SCHEMA_VERSION",
    "build_implementation_manifest",
    "implementation_file_identities",
    "implementation_file_identity",
    "verify_implementation_manifest",
    "write_implementation_manifest_exclusive",
]
