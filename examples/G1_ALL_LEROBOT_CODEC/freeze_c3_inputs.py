#!/usr/bin/env python3
"""Read-only freeze of all 806 five-source episodes and the C3 budget model."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from gr00t.data.codec_evidence.contract import MAX_MV_PER_FRAME, MV_FIELD_DTYPES
from gr00t.data.codec_evidence.identity import EPISODE_IDENTITY_SCHEMA_VERSION
from gr00t.data.codec_evidence.production import sha256_path, validate_represented_30hz_timestamps
from gr00t.data.dataset.g1_all_lerobot_h40_adapter import build_codec_episode_identity
import numpy as np
import pandas as pd


SCHEMA_VERSION = "g1-all-lerobot-c3-input-freeze-v1"
RECIPE_RELATIVE = Path("examples/G1_ALL_LEROBOT_H40/recipe.20k.json")
OUTPUT_RELATIVE = Path("examples/G1_ALL_LEROBOT_CODEC/c3_full_input_freeze.json")
RECIPE_SHA256 = "cecca3d46676ad973e9779677ecfb353ff745617ed3784160fbb4e8b685a344d"
C2_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c2-r1")
C2_TERMINAL_SHA256 = "b79f2e72e3812759f381e10b18955eecac37ba55f10ee5010646cb72589d2aca"
C2_SUMMARY_SHA256 = "ee7fa65e7ed9aa9f3ee2ed0ad080002043f1dcaed8e446aec2ca5f28bf50a65b"
C2_MANIFEST_SHA256 = "23ff5969dc150c8f809870e82679a296f1a81206140b49c4ede0a6e43d8d78c6"
EXPECTED_EPISODES = 806
EXPECTED_FRAMES = 297_340
CAMERA_ID = "cam_left_high"
VIDEO_KEY = "observation.images.cam_left_high"
GOP = 30


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be regular and non-symlink: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            result.append(value)
    return result


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Input payload must be regular and non-symlink: {path}")
    resolved = path.resolve(strict=True)
    return {
        "relative_path": None,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_path(resolved),
    }


def _format_path(
    pattern: str,
    *,
    episode_index: int,
    chunk_size: int,
    video_key: str | None = None,
) -> str:
    chunk = episode_index // chunk_size
    return pattern.format(
        chunk_index=chunk,
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def _contained_payload(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Payload path escapes source root: {relative_path}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"Payload may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"Payload resolved outside source root: {relative_path}")
    return resolved


def _probe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,"
                "nb_frames,nb_read_frames"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-2000:]}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    stream = streams[0]
    expected = {
        "codec_name": "h264",
        "profile": "High",
        "pix_fmt": "yuv420p",
        "width": 640,
        "height": 480,
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "nb_frames": str(expected_frames),
        "nb_read_frames": str(expected_frames),
    }
    if any(stream.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Video stream contract drifted: {path}: {stream}")
    return stream


def _validate_parquet(path: Path, episode_index: int, length: int) -> None:
    table = pd.read_parquet(path)
    required = {
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
        "annotation.human.validity",
    }
    if not required.issubset(table.columns) or len(table) != length:
        raise ValueError(f"Parquet row/schema mismatch: {path}")
    rows = np.arange(length, dtype=np.int64)
    if not np.array_equal(table["frame_index"].to_numpy(dtype=np.int64), rows):
        raise ValueError(f"Parquet frame_index is not complete 0..N-1: {path}")
    if not np.array_equal(
        table["episode_index"].to_numpy(dtype=np.int64),
        np.full(length, episode_index, dtype=np.int64),
    ):
        raise ValueError(f"Parquet episode_index mismatch: {path}")
    if np.any(table["task_index"].to_numpy(dtype=np.int64) != 0):
        raise ValueError(f"Parquet task_index is not zero: {path}")
    if np.any(table["annotation.human.validity"].to_numpy(dtype=np.int64) != 1):
        raise ValueError(f"Parquet contains invalid demonstration rows: {path}")
    try:
        validate_represented_30hz_timestamps(table["timestamp"].to_numpy(), length)
    except ValueError as error:
        raise ValueError(
            f"Parquet timestamp differs from represented 30 Hz grid: {path}"
        ) from error


def _c2_basis() -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    if sha256_path(C2_ROOT / "ATTEMPT_TERMINAL.json") != C2_TERMINAL_SHA256:
        raise ValueError("C2 terminal identity drifted")
    if sha256_path(C2_ROOT / "summary.json") != C2_SUMMARY_SHA256:
        raise ValueError("C2 summary identity drifted")
    if sha256_path(C2_ROOT / "fixture_manifest.json") != C2_MANIFEST_SHA256:
        raise ValueError("C2 fixture manifest identity drifted")
    terminal = _load_json(C2_ROOT / "ATTEMPT_TERMINAL.json")
    if terminal.get("status") != "ACCEPTED":
        raise ValueError("C2 terminal is not ACCEPTED")
    if (
        terminal.get("summary_sha256") != C2_SUMMARY_SHA256
        or terminal.get("fixture_manifest_sha256") != C2_MANIFEST_SHA256
    ):
        raise ValueError("C2 terminal does not bind the frozen summary/manifest")
    summary = _load_json(C2_ROOT / "summary.json")
    resources = summary.get("resources")
    episodes = summary.get("episodes")
    if (
        summary.get("total_frames") != 1_803
        or not isinstance(resources, dict)
        or not isinstance(episodes, list)
        or len(episodes) != 5
    ):
        raise ValueError("C2 summary basis fields differ from the accepted canary")
    wall_seconds = resources.get("wall_seconds")
    output_bytes = resources.get("output_bytes")
    peak_rss_bytes = resources.get("peak_rss_bytes")
    if (
        not isinstance(wall_seconds, (int, float))
        or type(output_bytes) is not int
        or type(peak_rss_bytes) is not int
    ):
        raise ValueError("C2 summary resource basis is malformed")
    pooled_bytes_per_frame = output_bytes / summary["total_frames"]
    pooled_wall_per_frame = float(wall_seconds) / summary["total_frames"]
    rates: dict[str, dict[str, float]] = {}
    for episode in episodes:
        source_id = episode.get("source_id")
        frames = episode.get("num_frames")
        raw_mv_count = episode.get("raw_mv_count")
        if (
            not isinstance(source_id, str)
            or type(frames) is not int
            or frames <= 0
            or type(raw_mv_count) is not int
            or raw_mv_count < 0
            or source_id in rates
        ):
            raise ValueError("C2 per-source summary basis is malformed")
        rates[source_id] = {
            "c2_frames": frames,
            "c2_allocated_output_bytes": pooled_bytes_per_frame * frames,
            "bytes_per_frame": pooled_bytes_per_frame,
            "raw_mv_per_frame": raw_mv_count / frames,
            "wall_seconds_per_frame": pooled_wall_per_frame,
        }
    basis = {
        "terminal_sha256": C2_TERMINAL_SHA256,
        "summary_sha256": C2_SUMMARY_SHA256,
        "fixture_manifest_sha256": C2_MANIFEST_SHA256,
        "wall_seconds": float(wall_seconds),
        "peak_rss_bytes": peak_rss_bytes,
        "frames": summary["total_frames"],
        "output_bytes_before_terminal_receipts": output_bytes,
    }
    return basis, rates


def build_freeze(repository: Path) -> dict[str, Any]:
    recipe_path = (repository / RECIPE_RELATIVE).resolve(strict=True)
    if sha256_path(recipe_path) != RECIPE_SHA256:
        raise ValueError("Five-source recipe identity drifted")
    recipe = _load_json(recipe_path)
    dataset_root = Path(recipe["dataset_root"]).resolve(strict=True)
    c2_basis, c2_rates = _c2_basis()
    sources = []
    episodes = []
    uid_set: set[str] = set()
    for recipe_order, source in enumerate(recipe["datasets"]):
        source_id = source["source_id"]
        root = (dataset_root / source["relative_path"]).resolve(strict=True)
        info_path = root / "meta/info.json"
        modality_path = root / "meta/modality.json"
        episodes_path = root / "meta/episodes.jsonl"
        info = _load_json(info_path)
        metadata_rows = sorted(
            _load_jsonl(episodes_path), key=lambda row: int(row["episode_index"])
        )
        if len(metadata_rows) != source["episodes"]:
            raise ValueError(f"Episode count drifted: {source_id}")
        info_identity = _file_identity(info_path)
        modality_identity = _file_identity(modality_path)
        episodes_identity = _file_identity(episodes_path)
        if info_identity["sha256"] != source["expected_info_sha256"]:
            raise ValueError(f"info.json identity drifted: {source_id}")
        if modality_identity["sha256"] != source["expected_modality_sha256"]:
            raise ValueError(f"modality.json identity drifted: {source_id}")
        source_frames = 0
        source_anchors = 0
        lengths = []
        seen_indices: set[int] = set()
        for row in metadata_rows:
            episode_index = row.get("episode_index")
            length = row.get("length")
            if (
                type(episode_index) is not int
                or type(length) is not int
                or episode_index < 0
                or length < 40
                or episode_index in seen_indices
            ):
                raise ValueError(f"Invalid full episode metadata: {source_id}:{row}")
            seen_indices.add(episode_index)
            parquet_relative = _format_path(
                info["data_path"],
                episode_index=episode_index,
                chunk_size=info["chunks_size"],
            )
            video_relative = _format_path(
                info["video_path"],
                episode_index=episode_index,
                chunk_size=info["chunks_size"],
                video_key=VIDEO_KEY,
            )
            parquet = _contained_payload(root, parquet_relative)
            video = _contained_payload(root, video_relative)
            _validate_parquet(parquet, episode_index, length)
            probe = _probe_video(video, length)
            parquet_identity = _file_identity(parquet)
            video_identity = _file_identity(video)
            parquet_identity["relative_path"] = parquet_relative
            video_identity["relative_path"] = video_relative
            identity = build_codec_episode_identity(
                source_id=source_id,
                canonical_root=root,
                canonical_info_sha256=info_identity["sha256"],
                canonical_modality_sha256=modality_identity["sha256"],
                data_path_pattern=info["data_path"],
                video_path_pattern=info["video_path"],
                chunk_size=info["chunks_size"],
                episode_index=episode_index,
                episode_length=length,
            )
            uid = identity["episode_uid"]
            identity_preimage = {
                key: value for key, value in identity.items() if key != "episode_uid"
            }
            if uid in uid_set:
                raise ValueError(f"Duplicate C3 episode UID: {uid}")
            uid_set.add(uid)
            anchors = math.ceil(length / GOP)
            source_frames += length
            source_anchors += anchors
            lengths.append(length)
            episodes.append(
                {
                    "recipe_order": recipe_order,
                    "source_id": source_id,
                    "episode_index": episode_index,
                    "num_frames": length,
                    "gop30_anchors": anchors,
                    "episode_uid": uid,
                    "episode_identity": identity_preimage,
                    "parquet": parquet_identity,
                    "video": video_identity,
                    "video_probe": probe,
                }
            )
        if source_frames != source["frames"]:
            raise ValueError(f"Source frame total drifted: {source_id}")
        rate = c2_rates[source_id]
        sources.append(
            {
                "recipe_order": recipe_order,
                "source_id": source_id,
                "canonical_root": str(root),
                "episode_count": len(metadata_rows),
                "frames": source_frames,
                "h40_anchors": source["h40_anchors"],
                "gop30_anchors": source_anchors,
                "min_frames": min(lengths),
                "max_frames": max(lengths),
                "mean_frames": source_frames / len(lengths),
                "metadata": {
                    "info": info_identity,
                    "modality": modality_identity,
                    "episodes": episodes_identity,
                },
                "c2_rate_basis": rate,
                "projected_object_bytes": math.ceil(rate["bytes_per_frame"] * source_frames),
                "projected_raw_mv_count": math.ceil(rate["raw_mv_per_frame"] * source_frames),
                "projected_wall_seconds": rate["wall_seconds_per_frame"] * source_frames,
            }
        )
    if len(episodes) != EXPECTED_EPISODES or len(uid_set) != EXPECTED_EPISODES:
        raise ValueError("C3 exact episode coverage is not 806 unique UIDs")
    total_frames = sum(item["num_frames"] for item in episodes)
    if total_frames != EXPECTED_FRAMES:
        raise ValueError("C3 total frame count drifted")
    total_h40_anchors = sum(source["h40_anchors"] for source in recipe["datasets"])
    if total_h40_anchors != recipe["total_h40_anchors"]:
        raise ValueError("C3 H40 anchor total drifted")
    central_bytes = sum(item["projected_object_bytes"] for item in sources)
    central_wall = sum(item["projected_wall_seconds"] for item in sources)
    max_rate = max(item["c2_rate_basis"]["bytes_per_frame"] for item in sources)
    conservative_bytes = math.ceil(max(central_bytes * 1.25, max_rate * total_frames * 1.10))
    pooled_bytes = math.ceil(
        c2_basis["output_bytes_before_terminal_receipts"] / c2_basis["frames"] * total_frames
    )
    max_rate_bytes = math.ceil(max_rate * total_frames)
    raw_mv_contract_bound_bytes = (
        total_frames * MAX_MV_PER_FRAME * sum(dtype.itemsize for dtype in MV_FIELD_DTYPES.values())
    )
    max_episode_frames = max(item["num_frames"] for item in episodes)
    projected_rss = math.ceil(
        c2_basis["peak_rss_bytes"]
        * max_episode_frames
        / max(item["c2_frames"] for item in c2_rates.values())
    )
    episode_freeze_core = {
        "schema_version": "g1-all-lerobot-c3-episode-input-freeze-core-v1",
        "camera_id": CAMERA_ID,
        "episode_identity_schema_version": EPISODE_IDENTITY_SCHEMA_VERSION,
        "episodes": [
            {
                "source_id": item["source_id"],
                "episode_index": item["episode_index"],
                "episode_uid": item["episode_uid"],
                "episode_identity": item["episode_identity"],
            }
            for item in episodes
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_READ_ONLY_PREPARATION",
        "recipe": {
            "path": str(recipe_path),
            "sha256": RECIPE_SHA256,
        },
        "dataset_root": str(dataset_root),
        "camera_id": CAMERA_ID,
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "episode_count": len(episodes),
        "episode_uid_digest": canonical_json_sha256(sorted(uid_set)),
        "episode_freeze_core_sha256": canonical_json_sha256(episode_freeze_core),
        "total_frames": total_frames,
        "total_h40_anchors": total_h40_anchors,
        "total_codec_anchors": sum(item["gop30_anchors"] for item in episodes),
        "max_episode_frames": max_episode_frames,
        "sources": sources,
        "episodes": episodes,
        "budget": {
            "basis": c2_basis,
            "central_projected_persistent_bytes": central_bytes,
            "pooled_projected_persistent_bytes": pooled_bytes,
            "max_observed_rate_projected_persistent_bytes": max_rate_bytes,
            "conservative_projected_persistent_bytes": conservative_bytes,
            "recommended_persistent_limit_bytes": 30_000_000_000,
            "central_projected_wall_seconds": central_wall,
            "recommended_cumulative_wall_limit_seconds": 10_800,
            "projected_peak_rss_bytes_linear": projected_rss,
            "recommended_peak_rss_limit_bytes": 16_000_000_000,
            "recommended_scratch_limit_bytes": 256_000_000,
            "estimated_bundle_file_count": 32 * EXPECTED_EPISODES,
            "raw_mv_contract_bound_bytes_not_recommended_budget": raw_mv_contract_bound_bytes,
            "projection_method": (
                "C2_PINNED_SUMMARY_POOLED_BYTES_AND_WALL_WITH_PER_SOURCE_RAW_MV_RATES"
            ),
            "conservative_margin": 1.25,
            "max_workers": 1,
            "automatic_retries": 0,
            "resume": "EXPLICIT_MANUAL_ONLY_SAME_PLAN",
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("C3 freeze output parent must be an existing real directory")
    if path.is_symlink():
        raise ValueError("C3 freeze output must not be a symlink")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_output_path(repository: Path, requested: Path) -> Path:
    expected = (repository / OUTPUT_RELATIVE).resolve(strict=False)
    candidate = requested if requested.is_absolute() else repository / requested
    if candidate.resolve(strict=False) != expected:
        raise ValueError(f"C3 freeze output is fixed to {expected}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--check", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve(strict=True).parents[2]
    payload = build_freeze(repository)
    if args.output is not None:
        _write_new(_validate_output_path(repository, args.output), payload)
    else:
        if _load_json(_validate_output_path(repository, args.check)) != payload:
            raise RuntimeError("Existing C3 input freeze differs from current read-only inputs")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "episode_count",
                    "total_frames",
                    "total_codec_anchors",
                    "content_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
