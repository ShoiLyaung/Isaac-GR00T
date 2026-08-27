"""Canonical, baseline-neutral episode identity preimages."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping


EPISODE_IDENTITY_SCHEMA_VERSION = "codec-evidence-episode-identity-v1"
_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREIMAGE_KEYS = {
    "schema_version",
    "source_id",
    "canonical_dataset_identity",
    "episode_index",
    "episode_length",
    "camera_id",
    "parquet",
    "video",
}
_DATASET_IDENTITY_KEYS = {"info_sha256", "modality_sha256"}
_ARTIFACT_KEYS = {"relative_path", "bytes", "sha256"}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _artifact(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_KEYS:
        raise ValueError(f"{name} fields do not match the episode identity contract")
    relative = value["relative_path"]
    path = Path(relative) if isinstance(relative, str) else Path("")
    if not isinstance(relative, str) or not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name}.relative_path must be a contained relative path")
    size = value["bytes"]
    if type(size) is not int or size < 0:
        raise ValueError(f"{name}.bytes must be a non-negative integer")
    return {
        "relative_path": path.as_posix(),
        "bytes": size,
        "sha256": _sha(value["sha256"], f"{name}.sha256"),
    }


def validate_episode_identity_preimage(value: Any) -> dict[str, Any]:
    """Return a canonical copy after exact schema and type validation."""

    if not isinstance(value, Mapping) or set(value) != _PREIMAGE_KEYS:
        raise ValueError("episode_identity fields do not match the v1 preimage schema")
    if value["schema_version"] != EPISODE_IDENTITY_SCHEMA_VERSION:
        raise ValueError("Unsupported episode identity preimage schema")
    source_id = value["source_id"]
    if not isinstance(source_id, str) or _SOURCE_ID_RE.fullmatch(source_id) is None:
        raise ValueError("episode_identity.source_id is invalid")
    dataset = value["canonical_dataset_identity"]
    if not isinstance(dataset, Mapping) or set(dataset) != _DATASET_IDENTITY_KEYS:
        raise ValueError("canonical_dataset_identity fields differ from the contract")
    episode_index = value["episode_index"]
    episode_length = value["episode_length"]
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode_identity.episode_index must be non-negative integer")
    if type(episode_length) is not int or episode_length < 1:
        raise ValueError("episode_identity.episode_length must be positive integer")
    if value["camera_id"] != "cam_left_high":
        raise ValueError("episode_identity.camera_id must be cam_left_high")
    return {
        "schema_version": EPISODE_IDENTITY_SCHEMA_VERSION,
        "source_id": source_id,
        "canonical_dataset_identity": {
            "info_sha256": _sha(dataset["info_sha256"], "info_sha256"),
            "modality_sha256": _sha(dataset["modality_sha256"], "modality_sha256"),
        },
        "episode_index": episode_index,
        "episode_length": episode_length,
        "camera_id": "cam_left_high",
        "parquet": _artifact(value["parquet"], "parquet"),
        "video": _artifact(value["video"], "video"),
    }


def episode_uid_from_preimage(value: Any) -> str:
    canonical = validate_episode_identity_preimage(value)
    return sha256(canonical_json_bytes(canonical)).hexdigest()


def episode_identity_with_uid(value: Any) -> dict[str, Any]:
    canonical = validate_episode_identity_preimage(value)
    return {**canonical, "episode_uid": episode_uid_from_preimage(canonical)}


__all__ = [
    "EPISODE_IDENTITY_SCHEMA_VERSION",
    "canonical_json_bytes",
    "episode_identity_with_uid",
    "episode_uid_from_preimage",
    "validate_episode_identity_preimage",
]
