"""Strict optional adapter and model configuration for codec evidence.

Parsing never resolves a fixture path.  An absent adapter block and
``enabled=false`` therefore remain filesystem- and RNG-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


_ALLOWED_MODALITIES = frozenset({"mv", "rpe", "i_frame", "metadata"})
_DEFAULT_MODALITIES = ("mv", "rpe", "i_frame", "metadata")
_SUPPORTED_SCHEMAS = frozenset({"codec-evidence-record-v2", "codec-evidence-episode-bundle-v1"})
_SOURCE_KINDS = frozenset({"RAW_JPEG_DERIVED", "NATIVE_CANONICAL_DECODED_DERIVED"})
_MODEL_KEYS = frozenset(
    {
        "enabled",
        "modalities",
        "gate_init",
        "schema_version",
        "camera_id",
        "source_kind",
        "d_codec",
        "conditioner_heads",
    }
)
_ADAPTER_KEYS = _MODEL_KEYS | {
    "fixture_root",
    "fixture_manifest_sha256",
    "record_schema_sha256",
}


@dataclass(frozen=True)
class CodecEvidenceConfig:
    enabled: bool = False
    fixture_root: Path | None = None
    fixture_manifest_sha256: str | None = None
    record_schema_sha256: str | None = None
    modalities: tuple[str, ...] = _DEFAULT_MODALITIES
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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _semantic_config(value: Mapping[str, Any]) -> CodecEvidenceConfig:
    enabled = value.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("codec_evidence.enabled must be boolean")

    modalities_value = value.get("modalities", list(_DEFAULT_MODALITIES))
    if not isinstance(modalities_value, list) or not modalities_value:
        raise ValueError("codec_evidence.modalities must be a non-empty list")
    if any(type(item) is not str for item in modalities_value):
        raise TypeError("codec_evidence.modalities entries must be strings")
    modalities = tuple(modalities_value)
    if len(set(modalities)) != len(modalities) or not set(modalities).issubset(_ALLOWED_MODALITIES):
        raise ValueError("codec_evidence.modalities contain duplicates or unknown values")

    raw_gate = value.get("gate_init", 0.0)
    if isinstance(raw_gate, bool) or not isinstance(raw_gate, Real):
        raise TypeError("codec_evidence.gate_init must be a real number, not boolean")
    gate_init = float(raw_gate)
    if not isfinite(gate_init) or gate_init != 0.0:
        raise ValueError("codec_evidence.gate_init must be finite and exactly 0.0")

    schema_version = value.get("schema_version", "codec-evidence-record-v2")
    if schema_version not in _SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported codec-evidence schema: {schema_version!r}")
    camera_id = value.get("camera_id", "cam_left_high")
    if camera_id != "cam_left_high":
        raise ValueError("codec_evidence.camera_id requires cam_left_high exactly")
    source_kind = value.get("source_kind")
    if schema_version == "codec-evidence-record-v2":
        if source_kind is not None:
            raise ValueError("record-v2 requires codec_evidence.source_kind=null")
    elif source_kind not in _SOURCE_KINDS:
        raise ValueError(f"episode-bundle requires source_kind in {sorted(_SOURCE_KINDS)}")

    d_codec = value.get("d_codec", 64)
    conditioner_heads = value.get("conditioner_heads", 8)
    if type(d_codec) is not int or d_codec <= 0:
        raise ValueError("codec_evidence.d_codec must be a positive non-boolean integer")
    if type(conditioner_heads) is not int or conditioner_heads <= 0:
        raise ValueError("codec_evidence.conditioner_heads must be a positive non-boolean integer")
    return CodecEvidenceConfig(
        enabled=enabled,
        modalities=modalities,
        gate_init=gate_init,
        schema_version=schema_version,
        camera_id=camera_id,
        source_kind=source_kind,
        d_codec=d_codec,
        conditioner_heads=conditioner_heads,
    )


def parse_codec_model_config(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact fixture-free model/processor serialization surface.

    Only enabled model dictionaries are accepted.  Disabled adapter blocks
    canonicalize to absence before any model serialization is constructed.
    """

    if type(value) is not dict:
        raise TypeError("model codec_evidence must be a dict")
    if set(value) != _MODEL_KEYS:
        raise ValueError("model codec_evidence keys differ from the exact fixture-free contract")
    if value["enabled"] is not True:
        raise ValueError("model codec_evidence must be enabled=true")
    return _semantic_config(value).to_model_dict()


def parse_codec_evidence_config(
    value: Mapping[str, Any] | None,
) -> CodecEvidenceConfig:
    """Parse one optional adapter block without touching the filesystem."""

    if value is None:
        return CodecEvidenceConfig()
    if not isinstance(value, Mapping):
        raise TypeError("codec_evidence must be an object")
    enabled = value.get("enabled", False)
    if type(enabled) is not bool:
        raise TypeError("codec_evidence.enabled must be boolean")
    if not enabled:
        return CodecEvidenceConfig()
    unknown = set(value) - _ADAPTER_KEYS
    if unknown:
        raise ValueError(f"Unknown codec_evidence fields: {sorted(unknown)}")
    semantic = _semantic_config(value)
    if (
        semantic.schema_version == "codec-evidence-episode-bundle-v1"
        and "record_schema_sha256" in value
    ):
        raise ValueError("episode-bundle must not carry record_schema_sha256")
    if not semantic.enabled:
        return semantic

    root_value = value.get("fixture_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("Enabled codec evidence requires fixture_root")
    root = Path(root_value)
    if not root.is_absolute():
        raise ValueError("codec_evidence.fixture_root must be absolute")
    manifest_sha = _sha256(
        value.get("fixture_manifest_sha256"),
        "codec_evidence.fixture_manifest_sha256",
    )
    schema_sha = None
    if semantic.schema_version == "codec-evidence-record-v2":
        schema_sha = _sha256(
            value.get("record_schema_sha256"),
            "codec_evidence.record_schema_sha256",
        )
    return CodecEvidenceConfig(
        enabled=True,
        fixture_root=root,
        fixture_manifest_sha256=manifest_sha,
        record_schema_sha256=schema_sha,
        modalities=semantic.modalities,
        gate_init=semantic.gate_init,
        schema_version=semantic.schema_version,
        camera_id=semantic.camera_id,
        source_kind=semantic.source_kind,
        d_codec=semantic.d_codec,
        conditioner_heads=semantic.conditioner_heads,
    )


__all__ = [
    "CodecEvidenceConfig",
    "parse_codec_evidence_config",
    "parse_codec_model_config",
]
