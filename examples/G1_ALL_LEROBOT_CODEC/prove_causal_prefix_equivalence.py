#!/usr/bin/env python3
"""Freeze and prove the causal libx264 scientific-evidence contract on CPU.

The proof is deliberately completed before a full fixture is materialized.  It
compares decoded RGB, packet/header topology, all eleven raw MV fields, the
aggregated MV/RPE/camera evidence, and anchor appearance.  MP4 container bytes
are not used as the prefix-equivalence criterion.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.production import (
    CRF,
    FPS,
    GOP,
    TIME_BASE_DEN,
    X264_CAUSAL_PARAMETERS,
    build_episode_arrays,
    encode_h264_twice,
    extract_mvs_and_decoded_rgb,
    probe_packets,
    probe_x264_encoder_options,
    sha256_path,
    toolchain_identity,
    trace_h264_headers,
)


SCHEMA_VERSION = "g1-all-lerobot-causal-prefix-equivalence-proof-v1"
EXPECTED_OUTPUT_ROOT = Path(
    "/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-contract-r1"
)
INPUT_FREEZE = Path(__file__).with_name("c3_full_input_freeze.json")
PREFIX_LENGTHS = (1, 2, 8, 9, 29, 30, 31, 39, 46)
FULL_FRAMES = 47
COMMON_SUFFIX_PREFIX = 31
ACTUAL_PREFIX = 31


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be a regular non-symlink file: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()


def _git_identity(repository: Path) -> dict[str, Any]:
    dirty = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"Causal proof requires a clean committed worktree:\n{dirty}")
    return {
        "repository": str(repository.resolve(strict=True)),
        "branch": _git(repository, "branch", "--show-current"),
        "head": _git(repository, "rev-parse", "HEAD"),
        "dirty_paths": [],
    }


def _force_cpu_offline() -> None:
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        }
    )


def _moving_rgb(num_frames: int, *, height: int = 96, width: int = 128) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = (rows + columns) % 256
    base[..., 1] = (2 * rows + columns) % 256
    base[..., 2] = (rows + 2 * columns) % 256
    frames = []
    for row in range(num_frames):
        frame = np.roll(base, shift=(row % 7, row % 11), axis=(0, 1)).copy()
        left = (row * 5) % (width - 24)
        frame[16:40, left : left + 24] = np.asarray([255, 20, 30], dtype=np.uint8)
        frames.append(frame)
    return np.stack(frames)


def _different_suffix(frames: np.ndarray, prefix_frames: int) -> np.ndarray:
    changed = frames.copy()
    suffix = changed[prefix_frames:]
    suffix[:] = np.flip(suffix, axis=(1, 2))
    suffix[:, ::4, ::4] ^= np.uint8(0xA5)
    return changed


def _decode_actual_prefix(path: Path, num_frames: int) -> np.ndarray:
    import av

    frames = []
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise RuntimeError("Actual proof video must expose exactly one video stream")
        for frame in container.decode(streams[0]):
            frames.append(np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8))
            if len(frames) == num_frames:
                break
    if len(frames) != num_frames:
        raise RuntimeError("Actual proof video is shorter than the frozen proof length")
    return np.stack(frames)


def _actual_source() -> tuple[Path, dict[str, Any]]:
    frozen = _load_json(INPUT_FREEZE)
    sources = {row["source_id"]: row for row in frozen["sources"]}
    episode = next(row for row in frozen["episodes"] if row["num_frames"] >= FULL_FRAMES)
    source = sources[episode["source_id"]]
    root = Path(source["canonical_root"]).resolve(strict=True)
    path = (root / episode["video"]["relative_path"]).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or root not in path.parents:
        raise RuntimeError("Frozen actual proof video is not a contained regular file")
    observed = {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
    expected = {key: episode["video"][key] for key in ("bytes", "sha256")}
    if observed != expected:
        raise RuntimeError("Frozen actual proof video identity drifted")
    return path, {
        "source_id": episode["source_id"],
        "episode_index": episode["episode_index"],
        "episode_uid": episode["episode_uid"],
        "path": str(path),
        **observed,
    }


def _x264_user_data(path: Path) -> dict[str, Any]:
    text = path.read_bytes().decode("latin1", errors="ignore")
    values = {}
    for name in (
        "ref",
        "threads",
        "lookahead_threads",
        "sliced_threads",
        "bframes",
        "keyint",
        "keyint_min",
        "scenecut",
        "rc_lookahead",
        "mbtree",
    ):
        match = re.search(rf"\b{name}=(\d+)", text)
        values[name] = int(match.group(1)) if match else None
    return values


def _legacy_encode(frames: np.ndarray, output: Path) -> Path:
    height, width = frames.shape[1:3]
    parameters = ":".join(
        value
        for value in X264_CAUSAL_PARAMETERS
        if not value.startswith("rc-lookahead=") and not value.startswith("mbtree=")
    )
    command = (
        "/usr/bin/ffmpeg",
        "-nostdin",
        "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(CRF),
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(GOP),
        "-keyint_min",
        "16",
        "-bf",
        "0",
        "-refs",
        "1",
        "-sc_threshold",
        "0",
        "-threads",
        "1",
        "-flags",
        "+cgop",
        "-x264-params",
        parameters,
        "-fps_mode",
        "cfr",
        "-r",
        str(FPS),
        "-video_track_timescale",
        str(TIME_BASE_DEN),
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(output),
    )
    result = subprocess.run(
        command,
        input=np.ascontiguousarray(frames).tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=os.environ.copy(),
        timeout=1_800,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-8_000:])
    return output


def _evidence(frames: np.ndarray, root: Path, *, legacy: bool = False) -> dict[str, Any]:
    root.mkdir()
    if legacy:
        bitstream = _legacy_encode(frames, root / "legacy.mp4")
        bitstream_sha256 = sha256_path(bitstream)
    else:
        encoded = encode_h264_twice(frames, root)
        bitstream = encoded.bitstream_path
        bitstream_sha256 = encoded.bitstream_sha256
    count = int(frames.shape[0])
    packets = probe_packets(bitstream, num_frames=count)
    headers = trace_h264_headers(bitstream, num_frames=count)
    extracted = extract_mvs_and_decoded_rgb(bitstream, num_frames=count)
    arrays = build_episode_arrays(extracted, packets, headers)
    return {
        "bitstream": bitstream,
        "bitstream_sha256": bitstream_sha256,
        "packets": packets,
        "headers": headers,
        "extracted": extracted,
        "arrays": arrays,
        "x264_user_data": (
            _x264_user_data(bitstream)
            if legacy
            else probe_x264_encoder_options(bitstream)["observed"]
        ),
    }


def _update_array_digest(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(_canonical_json_bytes(list(array.shape)) + b"\0")
    digest.update(array.tobytes())


def _prefix_digest(evidence: Mapping[str, Any], prefix_frames: int) -> str:
    digest = sha256()
    packets = evidence["packets"]
    headers = evidence["headers"]
    extracted = evidence["extracted"]
    arrays = evidence["arrays"]
    for name in ("pts", "dts", "duration", "key_frame"):
        _update_array_digest(digest, f"packets.{name}", getattr(packets, name)[:prefix_frames])
    for name in ("frame_type", "reference_index"):
        _update_array_digest(digest, f"headers.{name}", getattr(headers, name)[:prefix_frames])
    _update_array_digest(digest, "decoded_rgb", extracted.rgb[:prefix_frames])
    offsets = extracted.mv_offsets[: prefix_frames + 1]
    _update_array_digest(digest, "raw_mv_offsets", offsets)
    mv_end = int(offsets[-1])
    for name in MV_FIELD_DTYPES:
        _update_array_digest(digest, f"raw_mv_fields.{name}", extracted.mv_fields[name][:mv_end])
    for name in (
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
    ):
        _update_array_digest(digest, f"arrays.{name}", arrays[name][:prefix_frames])
    anchor_count = int(np.count_nonzero(arrays["anchor_frame_index"] < prefix_frames))
    for name in ("anchor_frame_index", "anchor_appearance"):
        _update_array_digest(digest, f"arrays.{name}", arrays[name][:anchor_count])
    return digest.hexdigest()


def _compare(left: Mapping[str, Any], right: Mapping[str, Any], prefix_frames: int) -> dict[str, Any]:
    mismatches = []
    left_digest = _prefix_digest(left, prefix_frames)
    right_digest = _prefix_digest(right, prefix_frames)
    if left_digest != right_digest:
        mismatches.append("scientific_evidence_digest")
    return {
        "prefix_frames": prefix_frames,
        "left_prefix_sha256": left_digest,
        "right_prefix_sha256": right_digest,
        "mismatches": mismatches,
        "prefix_equivalent": not mismatches,
    }


def prove(repository: Path) -> dict[str, Any]:
    _force_cpu_offline()
    git = _git_identity(repository)
    actual_path, actual_identity = _actual_source()
    actual_before = sha256_path(actual_path)
    synthetic = _moving_rgb(FULL_FRAMES)
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind="NATIVE_CANONICAL_DECODED_DERIVED",
    )
    tools = toolchain_identity(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe")

    with tempfile.TemporaryDirectory(prefix="groot-causal-prefix-proof-") as temporary:
        root = Path(temporary)
        full = _evidence(synthetic, root / "synthetic-full")
        full_vs_prefix = []
        for prefix in PREFIX_LENGTHS:
            standalone = _evidence(synthetic[:prefix], root / f"prefix-{prefix:03d}")
            full_vs_prefix.append(_compare(full, standalone, prefix))

        suffix_b = _different_suffix(synthetic, COMMON_SUFFIX_PREFIX)
        suffix_a_evidence = _evidence(synthetic, root / "suffix-a")
        suffix_b_evidence = _evidence(suffix_b, root / "suffix-b")
        suffix_comparison = _compare(
            suffix_a_evidence, suffix_b_evidence, COMMON_SUFFIX_PREFIX
        )

        episode_a = _moving_rgb(17)
        episode_b = _moving_rgb(41)
        _evidence(episode_a, root / "reset-episode-a")
        reset_b_first = _evidence(episode_b, root / "reset-episode-b-first")
        reset_b_second = _evidence(episode_b, root / "reset-episode-b-second")
        reset_comparison = _compare(reset_b_first, reset_b_second, len(episode_b))
        reset_comparison["first_frame_is_idr"] = bool(
            reset_b_second["headers"].reference_index[0] == -1
        )
        reset_comparison["bitstream_repeat_equal"] = (
            reset_b_first["bitstream_sha256"] == reset_b_second["bitstream_sha256"]
        )

        actual_rgb = _decode_actual_prefix(actual_path, FULL_FRAMES)
        actual_full = _evidence(actual_rgb, root / "actual-full")
        actual_prefix = _evidence(actual_rgb[:ACTUAL_PREFIX], root / "actual-prefix")
        actual_comparison = _compare(actual_full, actual_prefix, ACTUAL_PREFIX)

        legacy_full = _evidence(synthetic, root / "legacy-full", legacy=True)
        legacy_prefix = _evidence(synthetic[:9], root / "legacy-prefix", legacy=True)
        legacy_comparison = _compare(legacy_full, legacy_prefix, 9)

        production_options = full["x264_user_data"]
        legacy_options = legacy_full["x264_user_data"]
        expected_effective = {
            "ref": 1,
            "threads": 1,
            "lookahead_threads": 1,
            "sliced_threads": 0,
            "bframes": 0,
            "keyint": 30,
            "keyint_min": 16,
            "scenecut": 0,
            "mbtree": 0,
        }
        effective_match = all(production_options.get(key) == value for key, value in expected_effective.items())
        rc_zero_is_explicit_and_empirical = (
            "rc-lookahead=0" in X264_CAUSAL_PARAMETERS
            and production_options.get("rc_lookahead") is None
            and all(item["prefix_equivalent"] for item in full_vs_prefix)
            and suffix_comparison["prefix_equivalent"]
            and actual_comparison["prefix_equivalent"]
        )
        negative_detected = (
            not legacy_comparison["prefix_equivalent"]
            and legacy_options.get("rc_lookahead", 0) > 0
            and legacy_options.get("mbtree") == 1
        )
        reset_pass = (
            reset_comparison["prefix_equivalent"]
            and reset_comparison["first_frame_is_idr"]
            and reset_comparison["bitstream_repeat_equal"]
        )
        passed = (
            effective_match
            and rc_zero_is_explicit_and_empirical
            and negative_detected
            and reset_pass
            and all(item["prefix_equivalent"] for item in full_vs_prefix)
        )

    if sha256_path(actual_path) != actual_before:
        raise RuntimeError("Actual source video changed during causal proof")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "technical_valid": passed,
        "scientific_scope": "ENCODER_EVIDENCE_PREFIX_EQUIVALENCE_NOT_POLICY_QUALITY",
        "git": git,
        "input_freeze": {
            "path": str(INPUT_FREEZE.resolve(strict=True)),
            "sha256": sha256_path(INPUT_FREEZE),
        },
        "actual_source": actual_identity,
        "contract": contract,
        "contract_sha256": codec_contract_sha256(contract),
        "x264_requested_parameters": list(X264_CAUSAL_PARAMETERS),
        "x264_effective_user_data": production_options,
        "x264_effective_expected": expected_effective,
        "x264_effective_match": effective_match,
        "rc_lookahead_zero_proof": {
            "command_parameter_explicit": "rc-lookahead=0" in X264_CAUSAL_PARAMETERS,
            "zero_is_omitted_from_this_x264_user_data": production_options.get("rc_lookahead") is None,
            "empirical_prefix_equivalence": rc_zero_is_explicit_and_empirical,
        },
        "toolchain": tools,
        "tests": {
            "full_vs_prefix": full_vs_prefix,
            "different_suffix_same_prefix": suffix_comparison,
            "gop_boundary_prefixes": [29, 30, 31],
            "episode_reset": reset_comparison,
            "actual_frozen_video": actual_comparison,
            "legacy_negative_control": {
                **legacy_comparison,
                "x264_effective_user_data": legacy_options,
                "known_noncausal_detected": negative_detected,
            },
        },
        "claim_boundary": (
            "Equality covers decoded RGB, packet/header timing and references, all eleven raw "
            "motion-vector fields, MV cells, RPE, camera summary, and anchor appearance. "
            "Container bytes and policy task success are outside this proof."
        ),
    }
    result["proof_payload_sha256"] = _canonical_sha256(result)
    if not passed:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def _write_new(root: Path, result: Mapping[str, Any]) -> Path:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Causal proof output already exists: {root}")
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise FileNotFoundError("Causal proof output parent must be a real directory")
    root.mkdir()
    output = root / "CAUSAL_PREFIX_EQUIVALENCE.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError(f"Causal proof output root is frozen to {EXPECTED_OUTPUT_ROOT}")
    repository = Path(__file__).resolve(strict=True).parents[2]
    if not args.execute:
        print(
            json.dumps(
                {
                    "output_root": str(EXPECTED_OUTPUT_ROOT),
                    "output_exists": EXPECTED_OUTPUT_ROOT.exists(),
                    "prefix_lengths": PREFIX_LENGTHS,
                    "actual_frames": FULL_FRAMES,
                    "gpu": False,
                    "network": "offline_no_download",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = prove(repository)
    output = _write_new(EXPECTED_OUTPUT_ROOT, result)
    print(json.dumps({"status": result["status"], "output": str(output), "sha256": sha256_path(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
