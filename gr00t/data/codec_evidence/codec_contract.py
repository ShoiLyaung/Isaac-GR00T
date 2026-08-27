"""Hashable global codec semantics shared by an index and every bundle."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Mapping

from .identity import canonical_json_bytes


CODEC_CONTRACT_SCHEMA_VERSION = "codec-evidence-global-contract-v1"
PRODUCTION_FIXTURE_MODE = "PRODUCTION_NATIVE_CANONICAL"
SYNTHETIC_FIXTURE_MODE = "SYNTHETIC_CONFORMANCE"
_FIXTURE_MODES = {PRODUCTION_FIXTURE_MODE, SYNTHETIC_FIXTURE_MODE}
_SOURCE_KINDS = {"RAW_JPEG_DERIVED", "NATIVE_CANONICAL_DECODED_DERIVED"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = {"name", "version", "sha256"}
_CONTRACT_KEYS = {
    "schema_version",
    "fixture_mode",
    "camera_id",
    "source_kind",
    "width",
    "height",
    "container",
    "codec",
    "encoder",
    "pixel_format",
    "preset",
    "crf",
    "frame_rate",
    "time_base",
    "gop_size",
    "keyint",
    "min_keyint",
    "bframes",
    "reference_frames",
    "scene_cut",
    "open_gop",
    "threads",
    "lookahead_threads",
    "sliced_threads",
    "sync_lookahead",
    "pts_step",
    "audio",
    "metadata",
    "pts_dts_rule",
    "frame_type_rule",
    "tool_identity",
    "encoder_identity",
    "decoder_identity",
    "extractor_identity",
}


def _identity(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise ValueError(f"{name} fields do not match the codec contract")
    if any(not isinstance(value[field], str) or not value[field] for field in ("name", "version")):
        raise ValueError(f"{name} name/version must be non-empty strings")
    digest = value["sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256")
    return {"name": value["name"], "version": value["version"], "sha256": digest}


def validate_codec_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONTRACT_KEYS:
        raise ValueError("codec_contract fields do not match the global v1 schema")
    if value["schema_version"] != CODEC_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Unsupported global codec contract schema")
    mode = value["fixture_mode"]
    if mode not in _FIXTURE_MODES:
        raise ValueError("codec_contract.fixture_mode is unsupported")
    source_kind = value["source_kind"]
    if source_kind not in _SOURCE_KINDS:
        raise ValueError("codec_contract.source_kind is unsupported")
    for name, expected in (
        ("camera_id", "cam_left_high"),
        ("container", "mp4"),
        ("codec", "h264"),
        ("encoder", "libx264"),
        ("pixel_format", "yuv420p"),
        ("preset", "medium"),
        ("pts_dts_rule", "PTS_EQUALS_DTS_EQUALS_FRAME_INDEX_TIMES_1000"),
    ):
        if value[name] != expected:
            raise ValueError(f"codec_contract.{name} must be {expected!r}")
    integer_expected = {
        "width": 640,
        "height": 480,
        "crf": 18,
        "bframes": 0,
        "reference_frames": 1,
        "scene_cut": 0,
        "open_gop": 0,
        "threads": 1,
        "lookahead_threads": 1,
        "sliced_threads": 0,
        "sync_lookahead": 0,
        "pts_step": 1000,
    }
    for name, expected in integer_expected.items():
        if type(value[name]) is not int or value[name] != expected:
            raise ValueError(f"codec_contract.{name} must be integer {expected}")
    for name in ("audio", "metadata"):
        if type(value[name]) is not bool or value[name] is not False:
            raise ValueError(f"codec_contract.{name} must be false")
    if value["frame_rate"] != {"numerator": 30, "denominator": 1} or any(
        type(item) is not int for item in value["frame_rate"].values()
    ):
        raise ValueError("codec_contract.frame_rate must be exact 30/1")
    if value["time_base"] != {"numerator": 1, "denominator": 30000} or any(
        type(item) is not int for item in value["time_base"].values()
    ):
        raise ValueError("codec_contract.time_base must be exact 1/30000")
    gop_size = value["gop_size"]
    if type(gop_size) is not int or gop_size < 1:
        raise ValueError("codec_contract.gop_size must be a positive integer")
    for name in ("keyint", "min_keyint"):
        if type(value[name]) is not int or value[name] != gop_size:
            raise ValueError(f"codec_contract.{name} must equal gop_size")
    frame_type_rule = value["frame_type_rule"]
    if mode == PRODUCTION_FIXTURE_MODE:
        if source_kind != "NATIVE_CANONICAL_DECODED_DERIVED":
            raise ValueError("Production contract requires native-canonical source_kind")
        if gop_size != 30 or frame_type_rule != "IDR_IFF_ROW_MOD_30_ZERO_ELSE_P_REF_ROW_MINUS_1":
            raise ValueError("Production contract requires exact causal GOP30")
    elif frame_type_rule != "SYNTHETIC_CAUSAL_GOP":
        raise ValueError("Synthetic contract requires explicit synthetic frame_type_rule")
    canonical = {
        key: value[key]
        for key in (
            "schema_version",
            "fixture_mode",
            "camera_id",
            "source_kind",
            "width",
            "height",
            "container",
            "codec",
            "encoder",
            "pixel_format",
            "preset",
            "crf",
            "gop_size",
            "keyint",
            "min_keyint",
            "bframes",
            "reference_frames",
            "scene_cut",
            "open_gop",
            "threads",
            "lookahead_threads",
            "sliced_threads",
            "sync_lookahead",
            "pts_step",
            "audio",
            "metadata",
            "pts_dts_rule",
            "frame_type_rule",
        )
    }
    canonical["frame_rate"] = {"numerator": 30, "denominator": 1}
    canonical["time_base"] = {"numerator": 1, "denominator": 30000}
    for name in ("tool_identity", "encoder_identity", "decoder_identity", "extractor_identity"):
        canonical[name] = _identity(value[name], name)
    return canonical


def codec_contract_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(validate_codec_contract(value))).hexdigest()


def make_codec_contract(
    *,
    fixture_mode: str,
    source_kind: str,
    gop_size: int = 30,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shared_identity = dict(identity or {"name": "synthetic", "version": "1", "sha256": "d" * 64})
    return validate_codec_contract(
        {
            "schema_version": CODEC_CONTRACT_SCHEMA_VERSION,
            "fixture_mode": fixture_mode,
            "camera_id": "cam_left_high",
            "source_kind": source_kind,
            "width": 640,
            "height": 480,
            "container": "mp4",
            "codec": "h264",
            "encoder": "libx264",
            "pixel_format": "yuv420p",
            "preset": "medium",
            "crf": 18,
            "frame_rate": {"numerator": 30, "denominator": 1},
            "time_base": {"numerator": 1, "denominator": 30000},
            "gop_size": gop_size,
            "keyint": gop_size,
            "min_keyint": gop_size,
            "bframes": 0,
            "reference_frames": 1,
            "scene_cut": 0,
            "open_gop": 0,
            "threads": 1,
            "lookahead_threads": 1,
            "sliced_threads": 0,
            "sync_lookahead": 0,
            "pts_step": 1000,
            "audio": False,
            "metadata": False,
            "pts_dts_rule": "PTS_EQUALS_DTS_EQUALS_FRAME_INDEX_TIMES_1000",
            "frame_type_rule": (
                "IDR_IFF_ROW_MOD_30_ZERO_ELSE_P_REF_ROW_MINUS_1"
                if fixture_mode == PRODUCTION_FIXTURE_MODE
                else "SYNTHETIC_CAUSAL_GOP"
            ),
            "tool_identity": shared_identity,
            "encoder_identity": shared_identity,
            "decoder_identity": shared_identity,
            "extractor_identity": shared_identity,
        }
    )


__all__ = [
    "CODEC_CONTRACT_SCHEMA_VERSION",
    "PRODUCTION_FIXTURE_MODE",
    "SYNTHETIC_FIXTURE_MODE",
    "codec_contract_sha256",
    "make_codec_contract",
    "validate_codec_contract",
]
