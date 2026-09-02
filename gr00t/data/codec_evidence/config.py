"""Strict optional configuration for codec evidence.

An absent block and enabled=false are intentionally equivalent. Parsing a
disabled configuration performs no filesystem access and constructs no store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_ALLOWED_MODALITIES = frozenset({"mv", "rpe", "i_frame", "metadata"})
_ALLOWED_KEYS = frozenset(
    {
        "enabled",
        "fixture_root",
        "fixture_manifest_sha256",
        "record_schema_sha256",
        "modalities",
        "gate_init",
        "schema_version",
        "camera_id",
        "source_kind",
        "d_codec",
        "conditioner_heads",
    }
)


@dataclass(frozen=True)
class CodecEvidenceConfig:
    enabled: bool = False
    fixture_root: Path | None = None
    fixture_manifest_sha256: str | None = None
    record_schema_sha256: str | None = None
    modalities: tuple[str, ...] = ("mv", "rpe", "i_frame", "metadata")
    gate_init: float = 0.0
    schema_version: str = "codec-evidence-record-v2"
    camera_id: str = "cam_left_high"
    source_kind: str | None = None
    d_codec: int = 64
    conditioner_heads: int = 8

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "modalities": list(self.modalities),
            "gate_init": self.gate_init,
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "source_kind": self.source_kind,
            "d_codec": self.d_codec,
            "conditioner_heads": self.conditioner_heads,
        }


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def parse_codec_evidence_config(value: Mapping[str, Any] | None) -> CodecEvidenceConfig:
    """Parse one optional adapter block without touching the filesystem."""

    if value is None:
        return CodecEvidenceConfig()
    if not isinstance(value, Mapping):
        raise TypeError("codec_evidence must be an object")
    unknown = set(value) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown codec_evidence fields: {sorted(unknown)}")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("codec_evidence.enabled must be boolean")
    modalities_value = value.get("modalities", ["mv", "rpe", "i_frame", "metadata"])
    if not isinstance(modalities_value, list) or not modalities_value:
        raise ValueError("codec_evidence.modalities must be a non-empty list")
    modalities = tuple(modalities_value)
    if len(set(modalities)) != len(modalities) or not set(modalities).issubset(_ALLOWED_MODALITIES):
        raise ValueError("codec_evidence.modalities contain duplicates or unknown values")
    gate_init = float(value.get("gate_init", 0.0))
    if gate_init != 0.0:
        raise ValueError("codec_evidence.gate_init must be exactly 0.0")
    schema_version = value.get("schema_version", "codec-evidence-record-v2")
    supported_schemas = {
        "codec-evidence-record-v2",
        "codec-evidence-episode-bundle-v1",
    }
    if schema_version not in supported_schemas:
        raise ValueError(f"Unsupported codec-evidence schema: {schema_version!r}")
    camera_id = value.get("camera_id", "cam_left_high")
    if not isinstance(camera_id, str) or not camera_id:
        raise ValueError("codec_evidence.camera_id must be non-empty")
    source_kind = value.get("source_kind")
    source_kinds = {
        "RAW_JPEG_DERIVED",
        "NATIVE_CANONICAL_DECODED_DERIVED",
    }
    if source_kind is not None and source_kind not in source_kinds:
        raise ValueError(f"codec_evidence.source_kind must be one of {sorted(source_kinds)}")
    if schema_version == "codec-evidence-record-v2" and source_kind is not None:
        raise ValueError("record-v2 does not accept codec_evidence.source_kind")
    d_codec = value.get("d_codec", 64)
    conditioner_heads = value.get("conditioner_heads", 8)
    if not isinstance(d_codec, int) or d_codec <= 0:
        raise ValueError("codec_evidence.d_codec must be a positive integer")
    if not isinstance(conditioner_heads, int) or conditioner_heads <= 0:
        raise ValueError("codec_evidence.conditioner_heads must be a positive integer")

    if not enabled:
        return CodecEvidenceConfig(
            enabled=False,
            modalities=modalities,
            gate_init=gate_init,
            schema_version=schema_version,
            camera_id=camera_id,
            source_kind=source_kind,
            d_codec=d_codec,
            conditioner_heads=conditioner_heads,
        )

    root_value = value.get("fixture_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("Enabled codec evidence requires fixture_root")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise ValueError("codec_evidence.fixture_root must be absolute")
    manifest_sha = _sha256(
        value.get("fixture_manifest_sha256"), "codec_evidence.fixture_manifest_sha256"
    )
    schema_sha = (
        _sha256(value.get("record_schema_sha256"), "codec_evidence.record_schema_sha256")
        if schema_version == "codec-evidence-record-v2"
        else None
    )
    if schema_version == "codec-evidence-episode-bundle-v1" and source_kind is None:
        raise ValueError("Enabled episode-bundle codec evidence requires source_kind")
    return CodecEvidenceConfig(
        enabled=True,
        fixture_root=root,
        fixture_manifest_sha256=manifest_sha,
        record_schema_sha256=schema_sha,
        modalities=modalities,
        gate_init=gate_init,
        schema_version=schema_version,
        camera_id=camera_id,
        source_kind=source_kind,
        d_codec=d_codec,
        conditioner_heads=conditioner_heads,
    )
