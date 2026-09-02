"""Build the fixed, real-data, CPU-only R5 codec-evidence pilot.

The production command deliberately has no episode, camera, row-range, codec,
or device switches.  Those choices are evidence contracts rather than tuning
knobs: three frozen train episodes, ``cam_left_high``, rows 0..59, and a
single-reference closed-GOP H.264 stream encoded by system libx264.

The helper functions are intentionally importable so the bitstream proof and
the extractor can be tested on synthetic frames without reading the real
dataset or publishing a fixture.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


FRAME_COUNT = 60
CAMERA_ID = "cam_left_high"
FPS = 30
TIME_BASE_NUM = 1
TIME_BASE_DEN = 30_000
PTS_STEP = 1_000
GOP = 30
CRF = 18
SSIM_MEDIAN_MINIMUM = 0.95
ALIGNMENT_OFFSETS = tuple(range(-3, 4))
MAX_MV_PER_FRAME = 19_200
SPLIT_SCHEMA_VERSION = "icra27-dual-vla-split-manifest-v1"
PAYLOAD_SCHEMA_VERSION = "icra27-dual-vla-payload-manifest-v1"
DATASET_CONTRACT_ID = "ICRA27-DUAL-VLA-DATASET-R1"
SPLIT_MANIFEST_ID = "ICRA27-DUAL-VLA-SPLIT-R1"
SPLIT_ALGORITHM_ID = "FAMILY_HASH_HAMILTON_V1"
SPLIT_SEED = 42
CONTRACT_INSTRUCTION = "Insert the shaft into the manipulator housing."

MV_FIELD_DTYPES: Mapping[str, np.dtype] = {
    "source": np.dtype("int32"),
    "block_width": np.dtype("uint8"),
    "block_height": np.dtype("uint8"),
    "source_x": np.dtype("int16"),
    "source_y": np.dtype("int16"),
    "destination_x": np.dtype("int16"),
    "destination_y": np.dtype("int16"),
    "flags": np.dtype("uint64"),
    "motion_x": np.dtype("int32"),
    "motion_y": np.dtype("int32"),
    "motion_scale": np.dtype("uint16"),
}
PYAV_MV_NAMES = {
    "source": "source",
    "block_width": "w",
    "block_height": "h",
    "source_x": "src_x",
    "source_y": "src_y",
    "destination_x": "dst_x",
    "destination_y": "dst_y",
    "flags": "flags",
    "motion_x": "motion_x",
    "motion_y": "motion_y",
    "motion_scale": "motion_scale",
}


@dataclass(frozen=True)
class PilotEpisode:
    episode_uid: str
    cohort_id: str
    dataset_name: str
    source_episode_index: int
    expected_rows: int
    expected_global_index_start: int
    raw_relative_path: str
    parquet_relative_path: str
    video_relative_path: str


PILOT_EPISODES = (
    PilotEpisode(
        episode_uid="93265c0fe622c7dc2b11964343961247c8c4cd2a0853238874180014b4a815c0",
        cohort_id="main_300_20260602",
        dataset_name="insert the shaft into the housing",
        source_episode_index=1,
        expected_rows=270,
        expected_global_index_start=284,
        raw_relative_path="raw/insert the shaft into the housing/episode_0001",
        parquet_relative_path=(
            "lerobot/insert the shaft into the housing/data/chunk-000/episode_000001.parquet"
        ),
        video_relative_path=(
            "lerobot/insert the shaft into the housing/videos/chunk-000/"
            "observation.images.cam_left_high/episode_000001.mp4"
        ),
    ),
    PilotEpisode(
        episode_uid="a7eca2c9c84a7bb38a1d679440a4a3bbcb2a6acd5da31b383bb7a748061308e0",
        cohort_id="aug_1p2mm_220_20260625_26",
        dataset_name="aug1.2mm_insert the shaft into the housing",
        source_episode_index=1,
        expected_rows=457,
        expected_global_index_start=439,
        raw_relative_path=(
            "raw/aug1.2mm_insertion/aug1.2mm_insert the shaft into the housing/episode_0001"
        ),
        parquet_relative_path=(
            "lerobot/aug1.2mm_insert the shaft into the housing/data/chunk-000/"
            "episode_000001.parquet"
        ),
        video_relative_path=(
            "lerobot/aug1.2mm_insert the shaft into the housing/videos/chunk-000/"
            "observation.images.cam_left_high/episode_000001.mp4"
        ),
    ),
    PilotEpisode(
        episode_uid="46d9379c0158f52a4339b8e1b3b57e133b6079436fcd2dcb005a3968f03756e6",
        cohort_id="aug_5p4mm_153_20260624",
        dataset_name="aug5.4mm_insert the shaft into the housing",
        source_episode_index=0,
        expected_rows=355,
        expected_global_index_start=0,
        raw_relative_path=(
            "raw/aug5.4mm_insertion_row_data/insert the shaft into the housing/episode_0000"
        ),
        parquet_relative_path=(
            "lerobot/aug5.4mm_insert the shaft into the housing/data/chunk-000/"
            "episode_000000.parquet"
        ),
        video_relative_path=(
            "lerobot/aug5.4mm_insert the shaft into the housing/videos/chunk-000/"
            "observation.images.cam_left_high/episode_000000.mp4"
        ),
    ),
)


@dataclass(frozen=True)
class AlignmentEvidence:
    per_row_ssim: np.ndarray
    per_row_psnr_db: np.ndarray
    offsets: np.ndarray
    offset_mean_ssim: np.ndarray
    median_same_row_ssim: float


@dataclass(frozen=True)
class EncodeResult:
    bitstream_path: Path
    bitstream_sha256: str
    attempt_sha256: tuple[str, str]
    attempts: int


@dataclass(frozen=True)
class PacketProof:
    pts: np.ndarray
    dts: np.ndarray
    duration: np.ndarray
    key_frame: np.ndarray
    stream: Mapping[str, Any]


@dataclass(frozen=True)
class HeaderProof:
    frame_type: np.ndarray
    reference_index: np.ndarray
    nal_unit_type: np.ndarray
    nal_ref_idc: np.ndarray
    slice_type: np.ndarray
    num_ref_idx_active_override_flag: np.ndarray
    ref_pic_list_modification_flag_l0: np.ndarray
    adaptive_ref_pic_marking_mode_flag: np.ndarray
    long_term_reference_flag: np.ndarray
    max_num_ref_frames: int
    num_ref_idx_l0_default_active_minus1: int
    gaps_in_frame_num_allowed_flag: int
    max_frame_num: int
    normalized_trace_sha256: str
    proof_claims: Mapping[str, bool]


@dataclass(frozen=True)
class ExtractedVideo:
    rgb: np.ndarray
    pts: np.ndarray
    frame_type: np.ndarray
    mv_offsets: np.ndarray
    mv_fields: Mapping[str, np.ndarray]
    pyav_identity: Mapping[str, Any]


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _raster_sha256(frame: np.ndarray) -> str:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise TypeError("Raster hashes require uint8 HxWx3 RGB frames")
    return sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _resolved_executable(path: str | Path, *, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be an absolute system executable path")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{name} is not an executable file: {resolved}")
    return resolved


def _cpu_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["ROCR_VISIBLE_DEVICES"] = ""
    environment["HIP_VISIBLE_DEVICES"] = ""
    return environment


def _run(
    command: Sequence[str | Path],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        [str(value) for value in command],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        env=_cpu_environment(),
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"Command failed with status {result.returncode}: {stderr}")
    return result


def _validate_rgb_sequence(frames: np.ndarray) -> np.ndarray:
    value = np.asarray(frames)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise TypeError("Frames must be uint8 [60,H,W,3] RGB")
    if value.shape[0] != FRAME_COUNT:
        raise ValueError(f"Exactly {FRAME_COUNT} frames are required")
    if value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError("Frame dimensions must be positive")
    if value.shape[1] % 2 or value.shape[2] % 2:
        raise ValueError("yuv420p requires even frame dimensions")
    return np.ascontiguousarray(value)


def _encode_once(frames: np.ndarray, output: Path, *, ffmpeg: Path) -> None:
    frames = _validate_rgb_sequence(frames)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite encoded attempt: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    height, width = frames.shape[1:3]
    x264_parameters = ":".join(
        (
            "bframes=0",
            "ref=1",
            "keyint=30",
            "min-keyint=30",
            "scenecut=0",
            "open-gop=0",
            "threads=1",
            "lookahead-threads=1",
            "sliced-threads=0",
            "sync-lookahead=0",
        )
    )
    command = (
        ffmpeg,
        "-nostdin",
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
        str(GOP),
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
        x264_parameters,
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
    _run(command, input_bytes=frames.tobytes(), timeout=600)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("libx264 did not create a non-empty MP4")


def encode_h264_twice(
    frames: np.ndarray,
    attempt_directory: str | Path,
    *,
    ffmpeg: str | Path = "/usr/bin/ffmpeg",
) -> EncodeResult:
    """Encode the frozen stream twice and require byte-identical MP4 output."""

    ffmpeg_path = _resolved_executable(ffmpeg, name="ffmpeg")
    attempt_directory = Path(attempt_directory)
    if not attempt_directory.is_dir():
        raise FileNotFoundError(attempt_directory)
    first = attempt_directory / "libx264-attempt-1.mp4"
    second = attempt_directory / "libx264-attempt-2.mp4"
    _encode_once(frames, first, ffmpeg=ffmpeg_path)
    _encode_once(frames, second, ffmpeg=ffmpeg_path)
    digests = (_sha256_path(first), _sha256_path(second))
    if digests[0] != digests[1]:
        raise ValueError(
            "Deterministic libx264 gate failed: two encodes have different SHA-256 values"
        )
    return EncodeResult(
        bitstream_path=first,
        bitstream_sha256=digests[0],
        attempt_sha256=digests,
        attempts=2,
    )


def probe_packets(
    bitstream: str | Path,
    *,
    ffprobe: str | Path = "/usr/bin/ffprobe",
) -> PacketProof:
    """Prove the 1/30000 packet clock and the no-reorder PTS/DTS sequence."""

    ffprobe_path = _resolved_executable(ffprobe, name="ffprobe")
    result = _run(
        (
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=index,codec_name,codec_type,time_base,avg_frame_rate,r_frame_rate,"
                "width,height,has_b_frames,refs,nb_frames:packet=pts,dts,duration,flags"
            ),
            "-of",
            "json",
            Path(bitstream),
        )
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    packets = payload.get("packets", [])
    if len(streams) != 1 or len(packets) != FRAME_COUNT:
        raise ValueError("Encoded MP4 must contain one video stream and exactly 60 packets")
    stream = streams[0]
    expected_stream = {
        "codec_name": "h264",
        "codec_type": "video",
        "time_base": "1/30000",
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "has_b_frames": 0,
        "refs": 1,
        "nb_frames": "60",
    }
    for name, expected in expected_stream.items():
        if stream.get(name) != expected:
            raise ValueError(f"Encoded stream {name}={stream.get(name)!r}, expected {expected!r}")
    pts = np.asarray([int(packet["pts"]) for packet in packets], dtype=np.int64)
    dts = np.asarray([int(packet["dts"]) for packet in packets], dtype=np.int64)
    duration = np.asarray([int(packet["duration"]) for packet in packets], dtype=np.int64)
    key_frame = np.asarray(["K" in packet.get("flags", "") for packet in packets], dtype=np.bool_)
    expected_time = np.arange(FRAME_COUNT, dtype=np.int64) * PTS_STEP
    expected_key = np.zeros((FRAME_COUNT,), dtype=np.bool_)
    expected_key[[0, 30]] = True
    if not np.array_equal(pts, expected_time) or not np.array_equal(dts, expected_time):
        raise ValueError("Every encoded packet must have PTS=DTS=row*1000")
    if not np.array_equal(duration, np.full(FRAME_COUNT, PTS_STEP, dtype=np.int64)):
        raise ValueError("Every encoded packet must have duration 1000")
    if not np.array_equal(key_frame, expected_key):
        raise ValueError("Packet key-frame flags must identify rows 0 and 30 only")
    return PacketProof(pts=pts, dts=dts, duration=duration, key_frame=key_frame, stream=stream)


_TRACE_NAL = re.compile(r"\[trace_headers[^]]*\]\s+nal_unit_type:\s*(\d+).*nal_ref_idc:\s*(\d+)")
_TRACE_FIELD = re.compile(
    r"\[trace_headers[^]]*\]\s+\d+\s+([A-Za-z0-9_\[\]]+)\s+[01]+\s+=\s+(-?\d+)\s*$"
)


def trace_h264_headers(
    bitstream: str | Path,
    *,
    ffmpeg: str | Path = "/usr/bin/ffmpeg",
) -> HeaderProof:
    """Parse ``trace_headers`` and prove the exact causal DPB/reference contract.

    ``AVMotionVector.source`` is intentionally absent from this function.  The
    immediately-previous reference identity follows from H.264 syntax: a
    one-picture DPB, one default L0 entry, no per-slice override, no list
    modification, no long-term marking/MMCO, referenced P slices, and IDR GOP
    resets at rows 0 and 30.
    """

    ffmpeg_path = _resolved_executable(ffmpeg, name="ffmpeg")
    result = _run(
        (
            ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "trace",
            "-i",
            Path(bitstream),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-bsf:v",
            "trace_headers",
            "-f",
            "null",
            "-",
        )
    )
    trace_bytes = result.stderr
    trace = trace_bytes.decode("utf-8", errors="strict")
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_nal: tuple[int, int] | None = None
    for line in trace.splitlines():
        match = _TRACE_NAL.search(line)
        if match:
            pending_nal = (int(match.group(1)), int(match.group(2)))
            current = None
            continue
        if "[trace_headers" in line and "Sequence Parameter Set" in line:
            current = {"nal_unit_type": 7, "nal_ref_idc": 3, "fields": {}}
            entries.append(current)
            continue
        if "[trace_headers" in line and "Picture Parameter Set" in line:
            current = {"nal_unit_type": 8, "nal_ref_idc": 3, "fields": {}}
            entries.append(current)
            continue
        if "[trace_headers" in line and "Slice Header" in line:
            if pending_nal is None or pending_nal[0] not in {1, 5}:
                raise ValueError("trace_headers Slice Header has no preceding VCL NAL identity")
            current = {
                "nal_unit_type": pending_nal[0],
                "nal_ref_idc": pending_nal[1],
                "fields": {},
            }
            entries.append(current)
            continue
        match = _TRACE_FIELD.search(line)
        if match and current is not None:
            fields = current["fields"]
            name = match.group(1)
            value = int(match.group(2))
            # Alignment/CABAC payload fields can repeat.  Every proof-bearing
            # slice/SPS/PPS field below is single-valued by grammar, so retain
            # its first traced value and validate it explicitly later.
            fields.setdefault(name, value)

    sps = [entry for entry in entries if entry["nal_unit_type"] == 7]
    pps = [entry for entry in entries if entry["nal_unit_type"] == 8]
    vcl = [entry for entry in entries if entry["nal_unit_type"] in {1, 5}]
    if not sps or not pps or len(vcl) != FRAME_COUNT:
        raise ValueError("trace_headers did not expose SPS, PPS, and exactly 60 VCL NAL units")
    sps_refs = {entry["fields"].get("max_num_ref_frames") for entry in sps}
    sps_gaps = {entry["fields"].get("gaps_in_frame_num_allowed_flag") for entry in sps}
    sps_log2_frame_num = {entry["fields"].get("log2_max_frame_num_minus4") for entry in sps}
    pps_l0 = {entry["fields"].get("num_ref_idx_l0_default_active_minus1") for entry in pps}
    if sps_refs != {1}:
        raise ValueError("SPS must prove max_num_ref_frames=1")
    if sps_gaps != {0}:
        raise ValueError("SPS must prove gaps_in_frame_num_allowed_flag=0")
    if len(sps_log2_frame_num) != 1 or None in sps_log2_frame_num:
        raise ValueError("SPS must expose one unambiguous max_frame_num modulus")
    max_frame_num = 1 << (int(next(iter(sps_log2_frame_num))) + 4)
    if pps_l0 != {0}:
        raise ValueError("PPS must prove exactly one default active L0 reference")

    expected_nal = np.asarray(
        [5 if row % GOP == 0 else 1 for row in range(FRAME_COUNT)], dtype=np.uint8
    )
    nal_unit_type = np.asarray([entry["nal_unit_type"] for entry in vcl], dtype=np.uint8)
    nal_ref_idc = np.asarray([entry["nal_ref_idc"] for entry in vcl], dtype=np.uint8)
    slice_type = np.asarray(
        [entry["fields"].get("slice_type", -1) for entry in vcl], dtype=np.int16
    )
    if not np.array_equal(nal_unit_type, expected_nal):
        raise ValueError("VCL NAL sequence must be exactly 2 IDR plus 58 non-IDR slices")
    expected_slice_modulo = np.asarray(
        [2 if row % GOP == 0 else 0 for row in range(FRAME_COUNT)], dtype=np.int16
    )
    if np.any(slice_type < 0) or not np.array_equal(slice_type % 5, expected_slice_modulo):
        raise ValueError("Slice headers must prove exactly I at GOP starts and P elsewhere")
    frame_num = np.asarray([entry["fields"].get("frame_num", -1) for entry in vcl], dtype=np.int64)
    expected_frame_num = (np.arange(FRAME_COUNT, dtype=np.int64) % GOP) % max_frame_num
    if not np.array_equal(frame_num, expected_frame_num):
        raise ValueError("VCL frame_num must be contiguous modulo max_frame_num within each GOP")
    p_rows = np.flatnonzero(expected_nal == 1)
    if np.any(nal_ref_idc[p_rows] == 0):
        raise ValueError("Every P picture must itself be a reference picture")

    def p_field(name: str) -> np.ndarray:
        values = np.asarray([vcl[row]["fields"].get(name, -1) for row in p_rows], dtype=np.int8)
        if np.any(values < 0):
            raise ValueError(f"Every P slice must expose trace_headers field {name}")
        return values

    override = p_field("num_ref_idx_active_override_flag")
    list_modification = p_field("ref_pic_list_modification_flag_l0")
    adaptive_marking = p_field("adaptive_ref_pic_marking_mode_flag")
    if np.any(override != 0):
        raise ValueError("P slices may not override the default one-entry L0 list")
    if np.any(list_modification != 0):
        raise ValueError("P slices may not modify reference picture list L0")
    if np.any(adaptive_marking != 0):
        raise ValueError("P slices may not signal adaptive marking/MMCO")

    idr_rows = np.flatnonzero(expected_nal == 5)
    idr_long_term = np.asarray(
        [vcl[row]["fields"].get("long_term_reference_flag", -1) for row in idr_rows],
        dtype=np.int8,
    )
    if not np.array_equal(idr_long_term, np.zeros((2,), dtype=np.int8)):
        raise ValueError("Both IDRs must prove long_term_reference_flag=0")
    forbidden_syntax = (
        "memory_management_control_operation",
        "difference_of_pic_nums_minus1",
        "long_term_pic_num",
        "long_term_frame_idx",
        "max_long_term_frame_idx_plus1",
        "modification_of_pic_nums_idc",
        "abs_diff_pic_num_minus1",
    )
    if any(name in trace for name in forbidden_syntax):
        raise ValueError("Long-term, MMCO, or reference-list modification syntax is forbidden")

    # Episode-bundle codes are 1=IDR, 2=intra non-IDR, 3=P.
    frame_type = np.asarray(
        [1 if row % GOP == 0 else 3 for row in range(FRAME_COUNT)], dtype=np.uint8
    )
    reference_index = np.arange(FRAME_COUNT, dtype=np.int64) - 1
    reference_index[idr_rows] = -1
    full_override = np.full((FRAME_COUNT,), -1, dtype=np.int8)
    full_list_modification = np.full((FRAME_COUNT,), -1, dtype=np.int8)
    full_adaptive_marking = np.full((FRAME_COUNT,), -1, dtype=np.int8)
    full_long_term = np.full((FRAME_COUNT,), -1, dtype=np.int8)
    full_override[p_rows] = override
    full_list_modification[p_rows] = list_modification
    full_adaptive_marking[p_rows] = adaptive_marking
    full_long_term[idr_rows] = idr_long_term
    claims = {
        "two_idr_58_p": True,
        "all_p_are_reference_pictures": True,
        "max_num_ref_frames_is_one": True,
        "frame_num_gaps_forbidden": True,
        "frame_num_contiguous_within_gop": True,
        "default_l0_has_one_entry": True,
        "no_l0_override": True,
        "no_reference_list_modification": True,
        "no_long_term_reference": True,
        "no_mmco": True,
        "closed_idr_gop_30": True,
        "reference_identity_from_bitstream_not_mv_source": True,
    }
    normalized_trace = {
        "max_num_ref_frames": 1,
        "gaps_in_frame_num_allowed_flag": 0,
        "max_frame_num": max_frame_num,
        "num_ref_idx_l0_default_active_minus1": 0,
        "forbidden_syntax_absent": list(forbidden_syntax),
        "vcl": [
            {
                "nal_unit_type": entry["nal_unit_type"],
                "nal_ref_idc": entry["nal_ref_idc"],
                "slice_type": entry["fields"].get("slice_type"),
                "frame_num": entry["fields"].get("frame_num"),
                "num_ref_idx_active_override_flag": entry["fields"].get(
                    "num_ref_idx_active_override_flag"
                ),
                "ref_pic_list_modification_flag_l0": entry["fields"].get(
                    "ref_pic_list_modification_flag_l0"
                ),
                "adaptive_ref_pic_marking_mode_flag": entry["fields"].get(
                    "adaptive_ref_pic_marking_mode_flag"
                ),
                "long_term_reference_flag": entry["fields"].get("long_term_reference_flag"),
            }
            for entry in vcl
        ],
    }
    normalized_trace_bytes = json.dumps(
        normalized_trace, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return HeaderProof(
        frame_type=frame_type,
        reference_index=reference_index,
        nal_unit_type=nal_unit_type,
        nal_ref_idc=nal_ref_idc,
        slice_type=slice_type,
        num_ref_idx_active_override_flag=full_override,
        ref_pic_list_modification_flag_l0=full_list_modification,
        adaptive_ref_pic_marking_mode_flag=full_adaptive_marking,
        long_term_reference_flag=full_long_term,
        max_num_ref_frames=1,
        num_ref_idx_l0_default_active_minus1=0,
        gaps_in_frame_num_allowed_flag=0,
        max_frame_num=max_frame_num,
        normalized_trace_sha256=sha256(normalized_trace_bytes).hexdigest(),
        proof_claims=claims,
    )


def _pyav_identity(av_module: Any) -> dict[str, Any]:
    module_path = Path(av_module.__file__).resolve(strict=True)
    versions = {
        str(name): list(version) if isinstance(version, tuple) else str(version)
        for name, version in sorted(av_module.library_versions.items())
    }
    return {
        "implementation": "PyAV export_mvs",
        "pyav_version": str(av_module.__version__),
        "module_path": str(module_path),
        "module_sha256": _sha256_path(module_path),
        "library_versions": versions,
    }


def extract_mvs_and_decoded_rgb(bitstream: str | Path) -> ExtractedVideo:
    """Decode on CPU and extract the frozen eleven ``AVMotionVector`` fields."""

    try:
        import av
    except ImportError as error:  # pragma: no cover - environment failure is explicit
        raise RuntimeError("PyAV is required for R5 export_mvs extraction") from error

    rgb_frames: list[np.ndarray] = []
    pts_values: list[int] = []
    frame_types: list[int] = []
    offsets = [0]
    field_chunks: dict[str, list[np.ndarray]] = {name: [] for name in MV_FIELD_DTYPES}
    with av.open(str(Path(bitstream)), mode="r") as container:
        video_streams = list(container.streams.video)
        if len(video_streams) != 1:
            raise ValueError("Encoded MP4 must expose exactly one video stream to PyAV")
        stream = video_streams[0]
        stream.codec_context.options = {"flags2": "+export_mvs"}
        for frame in container.decode(stream):
            if len(rgb_frames) >= FRAME_COUNT:
                raise ValueError("PyAV decoded more than 60 frames")
            if frame.pts is None:
                raise ValueError("Decoded frame is missing PTS")
            rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("PyAV did not produce an RGB24 raster")
            rgb_frames.append(np.ascontiguousarray(rgb))
            pts_values.append(int(frame.pts))
            picture_code = int(frame.pict_type)
            if picture_code not in {1, 2}:
                raise ValueError("Decoded stream contains a picture other than I or P")
            frame_types.append(picture_code)
            side_data = frame.side_data.get("MOTION_VECTORS")
            count = 0
            if side_data is not None:
                vectors = np.asarray(side_data.to_ndarray())
                if vectors.ndim != 1 or vectors.dtype.names is None:
                    raise ValueError("PyAV motion-vector side data has an unexpected layout")
                if set(vectors.dtype.names) != set(PYAV_MV_NAMES.values()):
                    raise ValueError(
                        "PyAV motion-vector fields differ from the frozen 11-field set"
                    )
                count = int(vectors.shape[0])
                if count > MAX_MV_PER_FRAME:
                    raise ValueError("Motion-vector count exceeds 19200; truncation is forbidden")
                for output_name, dtype in MV_FIELD_DTYPES.items():
                    source_name = PYAV_MV_NAMES[output_name]
                    field_chunks[output_name].append(
                        np.ascontiguousarray(vectors[source_name], dtype=dtype)
                    )
            else:
                for output_name, dtype in MV_FIELD_DTYPES.items():
                    field_chunks[output_name].append(np.empty((0,), dtype=dtype))
            if picture_code == 2 and count == 0:
                raise ValueError(
                    "Every encoded P frame must expose non-empty PyAV export_mvs side data"
                )
            offsets.append(offsets[-1] + count)

    if len(rgb_frames) != FRAME_COUNT:
        raise ValueError(f"PyAV decoded {len(rgb_frames)} frames, expected 60")
    rgb_array = np.stack(rgb_frames, axis=0)
    pts = np.asarray(pts_values, dtype=np.int64)
    expected_pts = np.arange(FRAME_COUNT, dtype=np.int64) * PTS_STEP
    if not np.array_equal(pts, expected_pts):
        raise ValueError("PyAV decoded PTS values differ from row*1000")
    pyav_frame_type = np.asarray(frame_types, dtype=np.uint8)
    expected_pyav_frame_type = np.asarray(
        [1 if row % GOP == 0 else 2 for row in range(FRAME_COUNT)], dtype=np.uint8
    )
    if not np.array_equal(pyav_frame_type, expected_pyav_frame_type):
        raise ValueError("PyAV picture types disagree with the two-IDR/58-P contract")
    frame_type = np.asarray(
        [1 if row % GOP == 0 else 3 for row in range(FRAME_COUNT)], dtype=np.uint8
    )
    mv_fields = {
        name: (
            np.ascontiguousarray(np.concatenate(chunks), dtype=dtype)
            if chunks
            else np.empty((0,), dtype=dtype)
        )
        for (name, dtype), chunks in zip(MV_FIELD_DTYPES.items(), field_chunks.values())
    }
    return ExtractedVideo(
        rgb=rgb_array,
        pts=pts,
        frame_type=frame_type,
        mv_offsets=np.asarray(offsets, dtype=np.int64),
        mv_fields=mv_fields,
        pyav_identity=_pyav_identity(av),
    )


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:  # pragma: no cover - environment failure is explicit
        raise RuntimeError("scikit-image is required for the R5 SSIM gate") from error
    return float(structural_similarity(left, right, channel_axis=-1, data_range=255))


def _psnr_db(left: np.ndarray, right: np.ndarray) -> float:
    difference = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.mean(difference * difference))
    return float("inf") if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def prove_alignment(
    raw_rgb: np.ndarray,
    canonical_rgb: np.ndarray,
    *,
    median_minimum: float = SSIM_MEDIAN_MINIMUM,
) -> AlignmentEvidence:
    """Gate dual-source alignment without requiring unique per-frame matches.

    Near-static adjacent frames can legitimately tie.  The frozen gate records
    every same-row score, requires median SSIM >= 0.95, and requires offset zero
    to be the *strict* maximum of sequence-mean SSIM for offsets -3..3.
    """

    raw_rgb = _validate_rgb_sequence(raw_rgb)
    canonical_rgb = _validate_rgb_sequence(canonical_rgb)
    if raw_rgb.shape != canonical_rgb.shape:
        raise ValueError("Raw color_0 and canonical cam_left_high shapes differ")
    score_by_offset: dict[int, np.ndarray] = {}
    for offset in ALIGNMENT_OFFSETS:
        if offset < 0:
            pairs = zip(raw_rgb[-offset:], canonical_rgb[: FRAME_COUNT + offset])
        else:
            pairs = zip(raw_rgb[: FRAME_COUNT - offset], canonical_rgb[offset:])
        values = np.asarray([_ssim(left, right) for left, right in pairs], dtype=np.float64)
        if values.size != FRAME_COUNT - abs(offset) or not np.isfinite(values).all():
            raise ValueError("SSIM offset scan produced an invalid score vector")
        score_by_offset[offset] = values
    per_row_ssim = score_by_offset[0]
    median = float(np.median(per_row_ssim))
    if median < median_minimum:
        raise ValueError(
            f"Raw/canonical same-row median SSIM {median:.8f} is below {median_minimum:.8f}"
        )
    mean_scores = np.asarray(
        [float(np.mean(score_by_offset[offset])) for offset in ALIGNMENT_OFFSETS],
        dtype=np.float64,
    )
    zero_index = ALIGNMENT_OFFSETS.index(0)
    alternatives = np.delete(mean_scores, zero_index)
    if not bool(mean_scores[zero_index] > np.max(alternatives)):
        raise ValueError("Offset zero is not the unique maximum mean SSIM over offsets -3..3")
    psnr = np.asarray(
        [_psnr_db(left, right) for left, right in zip(raw_rgb, canonical_rgb)],
        dtype=np.float64,
    )
    if np.isnan(psnr).any():
        raise ValueError("PSNR produced NaN")
    return AlignmentEvidence(
        per_row_ssim=per_row_ssim,
        per_row_psnr_db=psnr,
        offsets=np.asarray(ALIGNMENT_OFFSETS, dtype=np.int8),
        offset_mean_ssim=mean_scores,
        median_same_row_ssim=median,
    )


def aggregate_motion_vectors_2x2(
    mv_offsets: np.ndarray,
    mv_fields: Mapping[str, np.ndarray],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CPU NumPy 2x2x11 features plus a four-value camera summary.

    The per-vector normalization and cell assignment intentionally mirror
    ``CodecDynamicsCore._aggregate_mv`` exactly before taking each cell mean:
    source/4, w/64, h/64, src/dst x/640 and y/480, flags/16,
    motion_x/scale, motion_y/scale, and log2(scale)/16.  ``source`` remains a
    numeric model feature only and is never interpreted as reference identity.
    """

    offsets = np.asarray(mv_offsets)
    if offsets.dtype != np.int64 or offsets.shape != (FRAME_COUNT + 1,):
        raise ValueError("MV offsets must be int64 [61]")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("MV offsets must start at zero and be monotonic")
    total = int(offsets[-1])
    if set(mv_fields) != set(MV_FIELD_DTYPES):
        raise ValueError("MV fields must match the exact frozen eleven-field set")
    for name, dtype in MV_FIELD_DTYPES.items():
        value = np.asarray(mv_fields[name])
        if value.dtype != dtype or value.shape != (total,):
            raise ValueError(f"MV field {name} has wrong dtype or shape")

    features = np.zeros((FRAME_COUNT, 2, 2, 11), dtype=np.float32)
    mask = np.zeros((FRAME_COUNT, 2, 2), dtype=np.bool_)
    camera = np.zeros((FRAME_COUNT, 4), dtype=np.float32)
    camera_mask = np.zeros((FRAME_COUNT,), dtype=np.bool_)
    for row in range(FRAME_COUNT):
        begin, end = int(offsets[row]), int(offsets[row + 1])
        if begin == end:
            continue
        scale = np.asarray(mv_fields["motion_scale"][begin:end], dtype=np.float32)
        scale = np.maximum(scale, np.float32(1.0))
        normalized = np.stack(
            (
                np.asarray(mv_fields["source"][begin:end], dtype=np.float32) / np.float32(4.0),
                np.asarray(mv_fields["block_width"][begin:end], dtype=np.float32)
                / np.float32(64.0),
                np.asarray(mv_fields["block_height"][begin:end], dtype=np.float32)
                / np.float32(64.0),
                np.asarray(mv_fields["source_x"][begin:end], dtype=np.float32) / np.float32(640.0),
                np.asarray(mv_fields["source_y"][begin:end], dtype=np.float32) / np.float32(480.0),
                np.asarray(mv_fields["destination_x"][begin:end], dtype=np.float32)
                / np.float32(640.0),
                np.asarray(mv_fields["destination_y"][begin:end], dtype=np.float32)
                / np.float32(480.0),
                np.asarray(mv_fields["flags"][begin:end], dtype=np.float32) / np.float32(16.0),
                np.asarray(mv_fields["motion_x"][begin:end], dtype=np.float32) / scale,
                np.asarray(mv_fields["motion_y"][begin:end], dtype=np.float32) / scale,
                np.log2(scale) / np.float32(16.0),
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        x = np.asarray(mv_fields["destination_x"][begin:end], dtype=np.float32)
        y = np.asarray(mv_fields["destination_y"][begin:end], dtype=np.float32)
        cell_x = (x >= np.float32(320.0)).astype(np.int64)
        cell_y = (y >= np.float32(240.0)).astype(np.int64)
        for cell_row in range(2):
            for cell_column in range(2):
                selected = (cell_y == cell_row) & (cell_x == cell_column)
                if not np.any(selected):
                    continue
                mask[row, cell_row, cell_column] = True
                features[row, cell_row, cell_column] = np.mean(
                    normalized[selected], axis=0, dtype=np.float32
                )

        motion_scale = np.asarray(mv_fields["motion_scale"][begin:end], dtype=np.float64)
        motion_scale = np.maximum(motion_scale, 1.0)
        dx = np.asarray(mv_fields["motion_x"][begin:end], dtype=np.float64) / motion_scale
        dy = np.asarray(mv_fields["motion_y"][begin:end], dtype=np.float64) / motion_scale
        block_area = np.asarray(mv_fields["block_width"][begin:end], dtype=np.float64) * np.asarray(
            mv_fields["block_height"][begin:end], dtype=np.float64
        )
        if np.any(block_area <= 0):
            raise ValueError("Every exported MV block must have positive area")
        weight = block_area / np.sum(block_area)
        coverage = min(1.0, float(np.sum(block_area)) / float(width * height))
        camera[row] = np.asarray(
            [
                np.sum(weight * dx),
                np.sum(weight * dy),
                np.sum(weight * np.hypot(dx, dy)),
                coverage,
            ],
            dtype=np.float32,
        )
        camera_mask[row] = True
    if not np.isfinite(features).all() or not np.isfinite(camera).all():
        raise ValueError("MV aggregation produced non-finite output")
    return features, mask, camera, camera_mask


def _block_bounds(center: int, size: int) -> tuple[int, int]:
    start = int(center) - int(size) // 2
    return start, start + int(size)


def compute_rpe_2x2(
    decoded_rgb: np.ndarray,
    mv_offsets: np.ndarray,
    mv_fields: Mapping[str, np.ndarray],
    reference_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the decoded-luma MV-warp proxy (not a bitstream residual).

    Per quadrant the four features are signed mean, mean absolute, RMS, and
    p95 absolute luma error, each divided by 255.  Only pixels covered by an
    exported motion-vector block contribute.
    """

    frames = _validate_rgb_sequence(decoded_rgb)
    offsets = np.asarray(mv_offsets)
    references = np.asarray(reference_index)
    if offsets.dtype != np.int64 or offsets.shape != (FRAME_COUNT + 1,):
        raise ValueError("MV offsets must be int64 [61]")
    if references.dtype != np.int64 or references.shape != (FRAME_COUNT,):
        raise ValueError("reference_index must be int64 [60]")
    height, width = frames.shape[1:3]
    # Deterministic integer BT.601-like luma proxy over decoded RGB24 rasters.
    rgb32 = frames.astype(np.uint32)
    luma = ((77 * rgb32[..., 0] + 150 * rgb32[..., 1] + 29 * rgb32[..., 2] + 128) >> 8).astype(
        np.float32
    )
    rpe = np.zeros((FRAME_COUNT, 2, 2, 4), dtype=np.float32)
    rpe_mask = np.zeros((FRAME_COUNT, 2, 2), dtype=np.bool_)
    for row in range(FRAME_COUNT):
        reference = int(references[row])
        if reference < 0:
            continue
        if reference != row - 1:
            raise ValueError(
                "RPE only supports the bitstream-proven immediately previous reference"
            )
        residual = np.zeros((height, width), dtype=np.float32)
        covered = np.zeros((height, width), dtype=np.bool_)
        begin, end = int(offsets[row]), int(offsets[row + 1])
        for vector_index in range(begin, end):
            block_width = int(mv_fields["block_width"][vector_index])
            block_height = int(mv_fields["block_height"][vector_index])
            destination_x0, destination_x1 = _block_bounds(
                int(mv_fields["destination_x"][vector_index]), block_width
            )
            destination_y0, destination_y1 = _block_bounds(
                int(mv_fields["destination_y"][vector_index]), block_height
            )
            source_x0, source_x1 = _block_bounds(
                int(mv_fields["source_x"][vector_index]), block_width
            )
            source_y0, source_y1 = _block_bounds(
                int(mv_fields["source_y"][vector_index]), block_height
            )
            left_clip = max(0, -destination_x0, -source_x0)
            right_clip = max(0, destination_x1 - width, source_x1 - width)
            top_clip = max(0, -destination_y0, -source_y0)
            bottom_clip = max(0, destination_y1 - height, source_y1 - height)
            copy_width = block_width - left_clip - right_clip
            copy_height = block_height - top_clip - bottom_clip
            if copy_width <= 0 or copy_height <= 0:
                continue
            dx0 = destination_x0 + left_clip
            dy0 = destination_y0 + top_clip
            sx0 = source_x0 + left_clip
            sy0 = source_y0 + top_clip
            destination_slice = np.s_[dy0 : dy0 + copy_height, dx0 : dx0 + copy_width]
            source_slice = np.s_[sy0 : sy0 + copy_height, sx0 : sx0 + copy_width]
            residual[destination_slice] = (
                luma[row][destination_slice] - luma[reference][source_slice]
            )
            covered[destination_slice] = True

        for cell_row, row_slice in enumerate((slice(0, height // 2), slice(height // 2, height))):
            for cell_column, column_slice in enumerate(
                (slice(0, width // 2), slice(width // 2, width))
            ):
                selected = covered[row_slice, column_slice]
                if not np.any(selected):
                    continue
                values = residual[row_slice, column_slice][selected].astype(np.float64)
                absolute = np.abs(values)
                rpe[row, cell_row, cell_column] = np.asarray(
                    [
                        np.mean(values) / 255.0,
                        np.mean(absolute) / 255.0,
                        np.sqrt(np.mean(values * values)) / 255.0,
                        np.percentile(absolute, 95) / 255.0,
                    ],
                    dtype=np.float32,
                )
                rpe_mask[row, cell_row, cell_column] = True
    if not np.isfinite(rpe).all():
        raise ValueError("RPE aggregation produced non-finite output")
    return rpe, rpe_mask


def _resize_anchor(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise TypeError("Anchor source must be uint8 RGB")
    rows = np.rint(np.linspace(0, frame.shape[0] - 1, 256)).astype(np.int64)
    columns = np.rint(np.linspace(0, frame.shape[1] - 1, 256)).astype(np.int64)
    resized = frame[rows[:, None], columns[None, :], :]
    return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0


def build_episode_arrays(
    extracted: ExtractedVideo,
    packet: PacketProof,
    headers: HeaderProof,
) -> dict[str, np.ndarray]:
    """Build the exact episode-bundle array surface from absolute row facts."""

    if not np.array_equal(extracted.pts, packet.pts):
        raise ValueError("PyAV and ffprobe disagree on decoded PTS")
    if not np.array_equal(extracted.frame_type, headers.frame_type):
        raise ValueError("PyAV and trace_headers disagree on picture types")
    height, width = extracted.rgb.shape[1:3]
    mv_features, mv_mask, camera, camera_mask = aggregate_motion_vectors_2x2(
        extracted.mv_offsets,
        extracted.mv_fields,
        width=width,
        height=height,
    )
    rpe, rpe_mask = compute_rpe_2x2(
        extracted.rgb,
        extracted.mv_offsets,
        extracted.mv_fields,
        headers.reference_index,
    )
    frame_index = np.arange(FRAME_COUNT, dtype=np.int64)
    gop_id = frame_index // GOP
    gop_position = frame_index % GOP
    validity = np.ones((FRAME_COUNT, 7), dtype=np.bool_)
    anchors = np.asarray([0, 30], dtype=np.int64)
    anchor_per_frame = np.where(frame_index < 30, 0, 30).astype(np.int64)
    anchor_appearance = np.stack(
        [_resize_anchor(extracted.rgb[int(index)]) for index in anchors], axis=0
    ).astype(np.float32, copy=False)
    return {
        "frame_index": frame_index,
        "pts": np.ascontiguousarray(packet.pts, dtype=np.int64),
        "dts": np.ascontiguousarray(packet.dts, dtype=np.int64),
        "frame_type": np.ascontiguousarray(headers.frame_type, dtype=np.uint8),
        "gop_id": np.ascontiguousarray(gop_id, dtype=np.int64),
        "gop_position": np.ascontiguousarray(gop_position, dtype=np.int64),
        "reference_index": np.ascontiguousarray(headers.reference_index, dtype=np.int64),
        "frame_validity": validity,
        "frame_reason_code": np.zeros((FRAME_COUNT,), dtype=np.int16),
        "mv_cell_features": mv_features,
        "mv_cell_mask": mv_mask,
        "rpe": rpe,
        "rpe_mask": rpe_mask,
        "camera_motion_summary": camera,
        "camera_motion_mask": camera_mask,
        "anchor_index_per_frame": anchor_per_frame,
        "anchor_frame_index": anchors,
        "anchor_appearance": anchor_appearance,
    }


def _raw_state_action(row: Mapping[str, Any], group: str) -> np.ndarray:
    value = row.get(group)
    if not isinstance(value, Mapping):
        raise ValueError(f"Raw row {group} must be an object")
    pieces: list[np.ndarray] = []
    for key in ("left_arm", "right_arm", "left_ee", "right_ee"):
        item = value.get(key)
        if not isinstance(item, Mapping) or not isinstance(item.get("qpos"), list):
            raise ValueError(f"Raw row {group}.{key}.qpos is missing")
        pieces.append(np.asarray(item["qpos"], dtype=np.float32))
    result = np.concatenate(pieces)
    if result.shape != (26,) or not np.isfinite(result).all():
        raise ValueError(f"Raw row {group} must flatten to finite float32 [26]")
    return result


def _read_raw_episode(
    raw_root: Path,
    *,
    expected_rows: int,
    dataset_relative_root: str,
) -> tuple[np.ndarray, list[Mapping[str, Any]], list[dict[str, Any]]]:
    data_path = raw_root / "data.json"
    payload = _load_json(data_path)
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError("Raw data.json row count differs from the frozen pilot identity")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - explicit environment failure
        raise RuntimeError("Pillow is required to read raw color_0 JPEGs") from error

    frames: list[np.ndarray] = []
    selected_rows: list[Mapping[str, Any]] = []
    jpeg_ledger: list[dict[str, Any]] = []
    for row_index in range(FRAME_COUNT):
        row = rows[row_index]
        if not isinstance(row, Mapping) or row.get("idx") != row_index:
            raise ValueError("Raw data.json rows must have contiguous idx=0..59")
        colors = row.get("colors")
        expected_relative = f"colors/{row_index:06d}_color_0.jpg"
        if not isinstance(colors, Mapping) or colors.get("color_0") != expected_relative:
            raise ValueError("Raw color_0 path does not match the frozen per-row naming contract")
        image_path = raw_root / expected_relative
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        jpeg_ledger.append(
            {
                "relative_path": f"{dataset_relative_root}/{expected_relative}",
                "sha256": _sha256_path(image_path),
                "bytes": image_path.stat().st_size,
            }
        )
        with Image.open(image_path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("Raw color_0 JPEG did not decode as RGB")
        frames.append(np.ascontiguousarray(frame))
        selected_rows.append(row)
    result = np.stack(frames, axis=0)
    _validate_rgb_sequence(result)
    return result, selected_rows, jpeg_ledger


def _read_canonical_parquet(
    parquet_path: Path,
    episode: PilotEpisode,
    raw_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - explicit environment failure
        raise RuntimeError("pandas/pyarrow are required for canonical provenance") from error

    table = pd.read_parquet(parquet_path)
    required = {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "annotation.human.validity",
    }
    if not required.issubset(table.columns):
        raise ValueError(
            f"Canonical Parquet lacks columns: {sorted(required - set(table.columns))}"
        )
    if len(table) != episode.expected_rows:
        raise ValueError("Canonical Parquet row count differs from the frozen pilot identity")
    first = table.iloc[:FRAME_COUNT]
    expected_rows = np.arange(FRAME_COUNT, dtype=np.int64)
    if not np.array_equal(first["frame_index"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError("Canonical Parquet frame_index is not 0..59")
    if not np.array_equal(
        first["episode_index"].to_numpy(dtype=np.int64),
        np.full(FRAME_COUNT, episode.source_episode_index, dtype=np.int64),
    ):
        raise ValueError("Canonical Parquet episode_index differs from the frozen source")
    expected_global = expected_rows + episode.expected_global_index_start
    if not np.array_equal(first["index"].to_numpy(dtype=np.int64), expected_global):
        raise ValueError("Canonical Parquet global index differs from the frozen catalog mapping")
    if np.any(first["task_index"].to_numpy(dtype=np.int64) != 0):
        raise ValueError("Canonical pilot rows must have task_index=0")
    if np.any(first["annotation.human.validity"].to_numpy(dtype=np.int64) != 1):
        raise ValueError("Canonical pilot rows must have annotation validity=1")
    timestamps = first["timestamp"].to_numpy(dtype=np.float64)
    if not np.allclose(timestamps, expected_rows / FPS, rtol=0.0, atol=1e-6):
        raise ValueError("Canonical Parquet timestamp differs from the represented 30 Hz row grid")
    for row_index, raw_row in enumerate(raw_rows):
        raw_state = _raw_state_action(raw_row, "states")
        raw_action = _raw_state_action(raw_row, "actions")
        canonical_state = np.asarray(first.iloc[row_index]["observation.state"], dtype=np.float32)
        canonical_action = np.asarray(first.iloc[row_index]["action"], dtype=np.float32)
        if not np.array_equal(raw_state, canonical_state):
            raise ValueError(f"Raw/canonical float32 state mismatch at row {row_index}")
        if not np.array_equal(raw_action, canonical_action):
            raise ValueError(f"Raw/canonical float32 action mismatch at row {row_index}")
    return timestamps


def _decode_canonical_video(
    video_path: Path,
    *,
    expected_rows: int,
    canonical_timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import av
    except ImportError as error:  # pragma: no cover - explicit environment failure
        raise RuntimeError("PyAV is required to decode canonical cam_left_high") from error

    frames: list[np.ndarray] = []
    decoded_seconds: list[float] = []
    with av.open(str(video_path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("Canonical MP4 must contain exactly one video stream")
        stream = streams[0]
        if int(stream.frames) != expected_rows:
            raise ValueError("Canonical MP4 frame count differs from its frozen episode row count")
        stream_identity = {
            "codec_sniffed_by_pyav": str(stream.codec_context.name),
            "profile": str(stream.codec_context.profile),
            "pixel_format": str(stream.codec_context.format.name),
            "time_base": str(stream.time_base),
            "declared_frames": int(stream.frames),
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
        }
        for frame in container.decode(stream):
            if len(frames) >= FRAME_COUNT:
                break
            if frame.pts is None:
                raise ValueError("Canonical decoded frame is missing PTS")
            rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            frames.append(np.ascontiguousarray(rgb))
            decoded_seconds.append(float(Fraction(int(frame.pts)) * frame.time_base))
    if len(frames) != FRAME_COUNT:
        raise ValueError("Canonical MP4 did not decode rows 0..59")
    rgb = np.stack(frames, axis=0)
    _validate_rgb_sequence(rgb)
    decoded_seconds_array = np.asarray(decoded_seconds, dtype=np.float64)
    if not np.allclose(decoded_seconds_array, canonical_timestamps, rtol=0.0, atol=1e-6):
        raise ValueError("Canonical MP4 PTS and Parquet timestamp disagree")
    identity = {
        "source_file_sha256": _sha256_path(video_path),
        "stream": stream_identity,
        "pyav": _pyav_identity(av),
        "decode_policy": "SNIFF_ACTUAL_CODEC_THEN_RGB24_CPU",
    }
    return rgb, identity


def _resolve_manifest_inputs(
    split_manifest_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    split_manifest_path = split_manifest_path.resolve(strict=True)
    split_manifest = _load_json(split_manifest_path)
    expected_split_identity = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "contract_id": DATASET_CONTRACT_ID,
        "split_manifest_id": SPLIT_MANIFEST_ID,
        "split_algorithm_id": SPLIT_ALGORITHM_ID,
        "split_seed": SPLIT_SEED,
        "contract_instruction": CONTRACT_INSTRUCTION,
    }
    for name, expected in expected_split_identity.items():
        if split_manifest.get(name) != expected:
            raise ValueError(f"Split manifest identity changed: {name}")
    if split_manifest.get("selected_camera_key") != "observation.images.cam_left_high":
        raise ValueError("Split manifest must select observation.images.cam_left_high")
    dataset_root = Path(str(split_manifest.get("dataset_root", ""))).resolve(strict=True)
    if not dataset_root.is_dir():
        raise ValueError("Split manifest dataset_root is not a directory")
    if (dataset_root / "catalog/STALE_AFTER_SYNC.json").exists():
        raise ValueError("Dataset catalog is marked stale after rsync")
    catalog = split_manifest.get("catalog")
    if not isinstance(catalog, Mapping):
        raise ValueError("Split manifest catalog identity is missing")
    catalog_path = Path(str(catalog.get("database_path", ""))).resolve()
    expected_catalog_path = dataset_root / "catalog/current.sqlite3"
    if catalog_path != expected_catalog_path or not catalog_path.is_file():
        raise ValueError("Split manifest does not bind dataset_root/catalog/current.sqlite3")
    if _sha256_path(catalog_path) != catalog.get("database_sha256"):
        raise ValueError("Current catalog SHA-256 differs from the split manifest")
    payload_sha = split_manifest.get("payload_manifest_sha256")
    if not isinstance(payload_sha, str) or len(payload_sha) != 64:
        raise ValueError("Split manifest lacks a valid payload_manifest_sha256")
    sibling = split_manifest_path.with_name("payload_manifest.json")
    declared = Path(str(split_manifest.get("payload_manifest_path", "")))
    candidates = [sibling]
    if declared.is_absolute():
        candidates.append(declared)
    payload_path = next(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and _sha256_path(path) == payload_sha
        ),
        None,
    )
    if payload_path is None:
        raise ValueError("Payload manifest is missing or its SHA-256 changed")
    payload_manifest = _load_json(payload_path)
    expected_payload_identity = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "contract_id": DATASET_CONTRACT_ID,
        "selected_camera_key": "observation.images.cam_left_high",
        "contract_instruction": CONTRACT_INSTRUCTION,
    }
    for name, expected in expected_payload_identity.items():
        if payload_manifest.get(name) != expected:
            raise ValueError(f"Payload manifest identity changed: {name}")
    if payload_manifest.get("catalog") != catalog:
        raise ValueError("Payload and split manifests bind different catalog identities")
    if Path(str(payload_manifest.get("dataset_root", ""))).resolve() != dataset_root:
        raise ValueError("Split and payload manifests disagree on dataset_root")
    assignments = split_manifest.get("assignments")
    payload_episodes = payload_manifest.get("episodes")
    if not isinstance(assignments, list) or not isinstance(payload_episodes, list):
        raise ValueError("Split/payload manifests lack episode records")
    assignment_by_uid = {
        row.get("episode_uid"): row for row in assignments if isinstance(row, Mapping)
    }
    payload_by_uid = {
        row.get("episode_uid"): row for row in payload_episodes if isinstance(row, Mapping)
    }
    if (
        len(assignment_by_uid) != len(assignments)
        or len(payload_by_uid) != len(payload_episodes)
        or set(assignment_by_uid) != set(payload_by_uid)
    ):
        raise ValueError("Split/payload episode UID coverage is duplicated or incomplete")
    train_uids = sorted(
        row["episode_uid"]
        for row in assignments
        if row.get("eligible") is True and row.get("split") == "train"
    )
    train_digest = sha256(("\n".join(train_uids) + "\n").encode("utf-8")).hexdigest()
    if split_manifest.get("train_episode_uid_digest") != train_digest:
        raise ValueError("Split manifest train UID digest is inconsistent")
    for episode in PILOT_EPISODES:
        assignment = assignment_by_uid.get(episode.episode_uid)
        payload_row = payload_by_uid.get(episode.episode_uid)
        if assignment is None or payload_row is None:
            raise ValueError(f"Frozen pilot UID missing from manifests: {episode.episode_uid}")
        exact = {
            "cohort_id": episode.cohort_id,
            "dataset_name": episode.dataset_name,
            "episode_index": episode.source_episode_index,
            "rows": episode.expected_rows,
        }
        for name, expected in exact.items():
            if assignment.get(name) != expected or payload_row.get(name) != expected:
                raise ValueError(f"Frozen pilot manifest field changed: {name}")
        uid_material = (
            f"icra27-episode-v1\0{episode.dataset_name}\0{episode.source_episode_index}\0"
            f"{payload_row.get('parquet_sha256')}\0"
            f"{payload_row.get('cam_left_high_sha256')}"
        )
        if sha256(uid_material.encode("utf-8")).hexdigest() != episode.episode_uid:
            raise ValueError("Frozen pilot UID no longer derives from payload file identities")
        if assignment.get("split") != "train" or assignment.get("eligible") is not True:
            raise ValueError("Every frozen pilot episode must remain eligible train data")
        if payload_row.get("parquet_relative_path") != episode.parquet_relative_path:
            raise ValueError("Frozen pilot Parquet path changed")
        if payload_row.get("cam_left_high_relative_path") != episode.video_relative_path:
            raise ValueError("Frozen pilot cam_left_high path changed")
    provenance = {
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": _sha256_path(split_manifest_path),
        "payload_manifest_path": str(payload_path),
        "payload_manifest_sha256": payload_sha,
        "catalog": split_manifest.get("catalog"),
    }
    return dataset_root, payload_manifest, payload_by_uid, provenance


def _binary_identity(executable: Path, *, encoder_help: bool = False) -> dict[str, Any]:
    version = _run((executable, "-hide_banner", "-version"), timeout=60).stdout.decode(
        "utf-8", errors="strict"
    )
    identity: dict[str, Any] = {
        "path": str(executable),
        "sha256": _sha256_path(executable),
        "version_output": version,
        "version_output_sha256": sha256(version.encode()).hexdigest(),
    }
    if encoder_help:
        help_result = _run((executable, "-hide_banner", "-h", "encoder=libx264"), timeout=60)
        help_bytes = help_result.stdout + help_result.stderr
        if b"libx264" not in help_bytes:
            raise ValueError("System ffmpeg does not expose the libx264 encoder")
        identity["libx264_help_sha256"] = sha256(help_bytes).hexdigest()
    return identity


def reserve_output_root(output_root: str | Path, *, dataset_root: str | Path) -> Path:
    """Reserve a new output root; an existing path is never reused or replaced."""

    output = Path(output_root)
    dataset = Path(dataset_root).resolve(strict=True)
    if not output.is_absolute():
        raise ValueError("output_root must be absolute")
    unresolved = output.resolve(strict=False)
    if unresolved == dataset or dataset in unresolved.parents:
        raise ValueError("Codec pilot output may not be inside raw/LeRobot dataset_root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite pilot output root: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    output.mkdir(mode=0o755, exist_ok=False)
    return output.resolve(strict=True)


def _header_identity(headers: HeaderProof) -> dict[str, Any]:
    return {
        "method": "SYSTEM_FFMPEG_TRACE_HEADERS_H264_SYNTAX",
        "normalized_trace_proof_sha256": headers.normalized_trace_sha256,
        "normalized_trace_hash_definition": (
            "SHA256_OF_CANONICAL_JSON_PROOF_FIELDS_WITH_RUNTIME_POINTERS_AND_LOG_NOISE_REMOVED"
        ),
        "proof_claims": dict(headers.proof_claims),
        "max_num_ref_frames": headers.max_num_ref_frames,
        "gaps_in_frame_num_allowed_flag": headers.gaps_in_frame_num_allowed_flag,
        "max_frame_num": headers.max_frame_num,
        "num_ref_idx_l0_default_active_minus1": (headers.num_ref_idx_l0_default_active_minus1),
        "nal_unit_type": headers.nal_unit_type.tolist(),
        "nal_ref_idc": headers.nal_ref_idc.tolist(),
        "slice_type": headers.slice_type.tolist(),
        "num_ref_idx_active_override_flag": (headers.num_ref_idx_active_override_flag.tolist()),
        "ref_pic_list_modification_flag_l0": (headers.ref_pic_list_modification_flag_l0.tolist()),
        "adaptive_ref_pic_marking_mode_flag": (headers.adaptive_ref_pic_marking_mode_flag.tolist()),
        "long_term_reference_flag": headers.long_term_reference_flag.tolist(),
        "derived_reference_index": headers.reference_index.tolist(),
        "reference_identity_exclusion": "AVMotionVector.source is not used",
    }


def _alignment_identity(alignment: AlignmentEvidence) -> dict[str, Any]:
    psnr_json = [
        float(value) if np.isfinite(value) else None for value in alignment.per_row_psnr_db
    ]
    return {
        "source_a": "raw colors.color_0 JPEG decoded RGB24",
        "source_b": "canonical observation.images.cam_left_high decoded RGB24",
        "same_row_ssim": alignment.per_row_ssim.tolist(),
        "same_row_psnr_db": psnr_json,
        "psnr_positive_infinity_representation": "null_for_exact_raster_identity",
        "median_same_row_ssim": alignment.median_same_row_ssim,
        "median_same_row_ssim_minimum": SSIM_MEDIAN_MINIMUM,
        "offsets": alignment.offsets.tolist(),
        "offset_mean_ssim": alignment.offset_mean_ssim.tolist(),
        "strict_unique_maximum_offset": 0,
    }


def _encode_source_bundle(
    output_root: Path,
    episode: PilotEpisode,
    source_rgb: np.ndarray,
    canonical_rgb: np.ndarray,
    source_kind: str,
    alignment: AlignmentEvidence,
    structural_provenance: Mapping[str, Any],
    canonical_decoder_identity: Mapping[str, Any],
    *,
    ffmpeg: Path,
    ffprobe: Path,
    common_tool_identity: Mapping[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix=f"r5-{episode.episode_uid[:12]}-{source_kind.lower()}-"
    ) as temporary:
        encode = encode_h264_twice(source_rgb, temporary, ffmpeg=ffmpeg)
        packet = probe_packets(encode.bitstream_path, ffprobe=ffprobe)
        headers = trace_h264_headers(encode.bitstream_path, ffmpeg=ffmpeg)
        extracted = extract_mvs_and_decoded_rgb(encode.bitstream_path)
        arrays = build_episode_arrays(extracted, packet, headers)

        source_hashes = [_raster_sha256(frame) for frame in source_rgb]
        canonical_hashes = [_raster_sha256(frame) for frame in canonical_rgb]
        fixture_hashes = [_raster_sha256(frame) for frame in extracted.rgb]
        height, width = source_rgb.shape[1:3]
        encoder_identity = {
            "container": "mp4",
            "codec": "h264",
            "encoder": "libx264",
            "preset": "medium",
            "crf": CRF,
            "pixel_format": "yuv420p",
            "bframes": 0,
            "references": 1,
            "gop": GOP,
            "keyint_min": GOP,
            "scene_cut": 0,
            "open_gop": 0,
            "threads": 1,
            "lookahead_threads": 1,
            "sliced_threads": 0,
            "time_base_num": TIME_BASE_NUM,
            "time_base_den": TIME_BASE_DEN,
            "source_kind": source_kind,
            "determinism_encode_repetitions": encode.attempts,
            "attempt_sha256": list(encode.attempt_sha256),
            "deterministic_sha256_equal": True,
            "bitstream_sha256": encode.bitstream_sha256,
            "accepted_bitstream_policy": (
                "IMMUTABLE_BUNDLE_AUDIT_ARTIFACT_NOT_READ_BY_TRAINING_VIEW"
            ),
            "packet_proof": {
                "pts_equals_dts_equals_row_times_1000": True,
                "duration_is_1000": True,
                "stream": dict(packet.stream),
            },
        }
        decoder_identity = {
            "canonical": dict(canonical_decoder_identity),
            "fixture": extracted.pyav_identity,
            "source_raster_kind": source_kind,
            "canonical_codec_policy": "SNIFF_DO_NOT_TRUST_META_INFO_CODEC_LABEL",
            "dual_source_alignment": _alignment_identity(alignment),
        }
        extractor_identity = {
            "pyav_export_mvs": extracted.pyav_identity,
            "trace_headers": _header_identity(headers),
            "raw_mv_fields": list(MV_FIELD_DTYPES),
            "raw_mv_layout": "CSR_RAGGED_NO_TRUNCATION",
            "mv_cell_features": (
                "EXACT_CodecDynamicsCore._aggregate_mv_PER_VECTOR_NORMALIZATION_THEN_"
                "CELL_MEAN:source/4,w/64,h/64,src_x/640,src_y/480,dst_x/640,"
                "dst_y/480,flags/16,motion_x/max(scale,1),motion_y/max(scale,1),"
                "log2(max(scale,1))/16;cells=(dst_y>=240)*2+(dst_x>=320)"
            ),
            "camera_motion_summary": ("AREA_WEIGHTED_DX_DY_MAGNITUDE_AND_BLOCK_COVERAGE"),
            "rpe": ("DECODED_LUMA_MV_WARP_PROXY_SIGNED_MEAN_MAE_RMS_P95_NOT_BITSTREAM_RESIDUAL"),
        }
        tool_identity = {
            **dict(common_tool_identity),
            "structural_provenance": {
                **dict(structural_provenance),
                "source_kind": source_kind,
            },
            "cpu_only": True,
            "gpu_api_used": False,
        }
        metadata = {
            "split": "train",
            "cohort_id": episode.cohort_id,
            "dataset_name": episode.dataset_name,
            "source_episode_index": episode.source_episode_index,
            "width": width,
            "height": height,
            "time_base_num": TIME_BASE_NUM,
            "time_base_den": TIME_BASE_DEN,
            "gop_size": GOP,
            "source_frame_sha256": source_hashes,
            "canonical_frame_sha256": canonical_hashes,
            "fixture_frame_sha256": fixture_hashes,
            "tool_identity": tool_identity,
            "encoder_identity": encoder_identity,
            "decoder_identity": decoder_identity,
            "extractor_identity": extractor_identity,
        }

        from gr00t.data.codec_evidence.episode_bundle import write_episode_bundle

        bundle_path = write_episode_bundle(
            output_root,
            episode_uid=episode.episode_uid,
            camera_id=CAMERA_ID,
            source_kind=source_kind,
            metadata=metadata,
            arrays=arrays,
            raw_mv_offsets=extracted.mv_offsets,
            raw_mv_fields=extracted.mv_fields,
            accepted_bitstream=encode.bitstream_path,
        )
    return {
        "episode_uid": episode.episode_uid,
        "camera_id": CAMERA_ID,
        "source_kind": source_kind,
        "relative_path": bundle_path.relative_to(output_root).as_posix(),
        "sha256": _sha256_path(bundle_path),
        "bytes": bundle_path.stat().st_size,
        "frames": FRAME_COUNT,
        "determinism_encode_repetitions": 2,
        "deterministic_bitstream_sha256": encoder_identity["bitstream_sha256"],
        "accepted_bitstream_in_bundle": True,
        "median_same_row_ssim": alignment.median_same_row_ssim,
        "unique_best_offset": 0,
        "frame_count": FRAME_COUNT,
        "raw_mv_count": int(extracted.mv_offsets[-1]),
    }


def _build_one_episode(
    output_root: Path,
    dataset_root: Path,
    episode: PilotEpisode,
    payload_row: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    ffmpeg: Path,
    ffprobe: Path,
    common_tool_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_root = dataset_root / episode.raw_relative_path
    parquet_path = dataset_root / episode.parquet_relative_path
    video_path = dataset_root / episode.video_relative_path
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    for path in (parquet_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    declared_files = (
        (parquet_path, "parquet_sha256", "parquet_bytes"),
        (video_path, "cam_left_high_sha256", "cam_left_high_bytes"),
    )
    for path, sha_field, bytes_field in declared_files:
        if _sha256_path(path) != payload_row.get(sha_field):
            raise ValueError(f"Payload-manifest SHA-256 mismatch for {path}")
        if path.stat().st_size != payload_row.get(bytes_field):
            raise ValueError(f"Payload-manifest byte count mismatch for {path}")

    raw_rgb, raw_rows, raw_jpeg_ledger = _read_raw_episode(
        raw_root,
        expected_rows=episode.expected_rows,
        dataset_relative_root=episode.raw_relative_path,
    )
    canonical_timestamps = _read_canonical_parquet(
        parquet_path,
        episode,
        raw_rows,
    )
    canonical_rgb, canonical_decoder_identity = _decode_canonical_video(
        video_path,
        expected_rows=episode.expected_rows,
        canonical_timestamps=canonical_timestamps,
    )
    alignment = prove_alignment(raw_rgb, canonical_rgb)
    raw_jpeg_ledger_bytes = json.dumps(
        raw_jpeg_ledger,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    structural_provenance = {
        **dict(provenance),
        "episode_uid": episode.episode_uid,
        "raw_root": str(raw_root),
        "raw_data_json_sha256": _sha256_path(raw_root / "data.json"),
        "raw_color_0_jpeg_ledger": raw_jpeg_ledger,
        "raw_color_0_jpeg_ledger_sha256": sha256(raw_jpeg_ledger_bytes).hexdigest(),
        "parquet_path": str(parquet_path),
        "parquet_sha256": _sha256_path(parquet_path),
        "canonical_video_path": str(video_path),
        "canonical_video_sha256": _sha256_path(video_path),
        "raw_idx_0_through_59": True,
        "canonical_frame_index_0_through_59": True,
        "canonical_episode_index": episode.source_episode_index,
        "canonical_global_index_start": episode.expected_global_index_start,
        "raw_canonical_float32_state_action_exact": True,
        "parquet_timestamp_matches_canonical_pts": True,
        "frame_hash_definition": {
            "source_frame_sha256": ("SHA256_OF_DECODED_SOURCE_RGB24_UINT8_C_ORDER_RASTER_BYTES"),
            "canonical_frame_sha256": (
                "SHA256_OF_DECODED_CANONICAL_RGB24_UINT8_C_ORDER_RASTER_BYTES"
            ),
            "fixture_frame_sha256": (
                "SHA256_OF_DECODED_ACCEPTED_FIXTURE_RGB24_UINT8_C_ORDER_RASTER_BYTES"
            ),
            "raw_jpeg_ledger_sha256": (
                "SHA256_OF_CANONICAL_JSON_LEDGER_OF_SOURCE_FILE_PATH_SHA256_AND_BYTES"
            ),
        },
    }
    sources = (
        ("RAW_JPEG_DERIVED", raw_rgb),
        ("NATIVE_CANONICAL_DECODED_DERIVED", canonical_rgb),
    )
    return [
        _encode_source_bundle(
            output_root,
            episode,
            source_rgb,
            canonical_rgb,
            source_kind,
            alignment,
            structural_provenance,
            canonical_decoder_identity,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            common_tool_identity=common_tool_identity,
        )
        for source_kind, source_rgb in sources
    ]


def build_pilot(
    split_manifest: str | Path,
    output_root: str | Path,
    *,
    ffmpeg: str | Path = "/usr/bin/ffmpeg",
    ffprobe: str | Path = "/usr/bin/ffprobe",
) -> dict[str, Any]:
    """Build and publish exactly six no-clobber episode/source bundles."""

    ffmpeg_path = _resolved_executable(ffmpeg, name="ffmpeg")
    ffprobe_path = _resolved_executable(ffprobe, name="ffprobe")
    dataset_root, _, payload_by_uid, provenance = _resolve_manifest_inputs(Path(split_manifest))
    producer_path = Path(__file__).resolve(strict=True)
    catalog_identity = provenance.get("catalog")
    output = reserve_output_root(output_root, dataset_root=dataset_root)
    _write_json_new(
        output / "ATTEMPT_STARTED.json",
        {
            "schema_version": "codec-evidence-r5-pilot-attempt-v1",
            "pilot_attempt": 1,
            "status": "STARTED",
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "local_time": datetime.now().astimezone().isoformat(),
            "argv": list(sys.argv),
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "payload_manifest_sha256": provenance["payload_manifest_sha256"],
            "catalog_database_sha256": (
                catalog_identity.get("database_sha256")
                if isinstance(catalog_identity, Mapping)
                else None
            ),
            "producer_sha256": _sha256_path(producer_path),
            "cpu_only_limits": {
                "gpu_api_allowed": False,
                "hardware_acceleration_allowed": False,
                "ffmpeg_encoder": "libx264",
                "ffmpeg_threads": 1,
                "cuda_visible_devices_for_children": "",
                "rocr_visible_devices_for_children": "",
                "hip_visible_devices_for_children": "",
            },
            "forbidden_mutations": [
                str(dataset_root / "raw"),
                str(dataset_root / "lerobot"),
                str(dataset_root / "catalog"),
            ],
            "output_root": str(output),
            "output_policy": "NO_CLOBBER_PRESERVE_STARTED_AND_TERMINAL_ON_FAILURE",
        },
    )
    from gr00t.data.codec_evidence import episode_bundle as episode_bundle_module

    episode_bundle_writer_path = Path(episode_bundle_module.__file__).resolve(strict=True)
    codec_dynamics_path = (
        producer_path.parents[2] / "gr00t/model/gr00t_n1d7/codec_dynamics.py"
    ).resolve(strict=True)
    common_tool_identity = {
        "ffmpeg": _binary_identity(ffmpeg_path, encoder_help=True),
        "ffprobe": _binary_identity(ffprobe_path),
        "producer_path": str(producer_path),
        "producer_sha256": _sha256_path(producer_path),
        "episode_bundle_writer": {
            "path": str(episode_bundle_writer_path),
            "sha256": _sha256_path(episode_bundle_writer_path),
        },
        "mv_normalization_reference": {
            "path": str(codec_dynamics_path),
            "sha256": _sha256_path(codec_dynamics_path),
            "symbol": "CodecDynamicsCore._aggregate_mv",
        },
        "numpy_version": np.__version__,
        "execution": "CPU_ONLY_NO_TORCH_NO_CUDA_NO_HWACCEL",
    }
    summaries: list[dict[str, Any]] = []
    for episode in PILOT_EPISODES:
        summaries.extend(
            _build_one_episode(
                output,
                dataset_root,
                episode,
                payload_by_uid[episode.episode_uid],
                provenance,
                ffmpeg=ffmpeg_path,
                ffprobe=ffprobe_path,
                common_tool_identity=common_tool_identity,
            )
        )
    bundle_keys = {(row["episode_uid"], row["camera_id"], row["source_kind"]) for row in summaries}
    if len(summaries) != 6 or len(bundle_keys) != 6:
        raise AssertionError("Pilot must contain exactly six unique episode/source bundles")
    manifest = {
        "schema_version": "codec-evidence-r5-pilot-v1",
        "complete": True,
        "pilot_generation_attempt": 1,
        "source_kinds": [
            "RAW_JPEG_DERIVED",
            "NATIVE_CANONICAL_DECODED_DERIVED",
        ],
        "camera_id": CAMERA_ID,
        "rows": [0, FRAME_COUNT - 1],
        "bundles": summaries,
        "bundle_count": 6,
        "episode_uids": [episode.episode_uid for episode in PILOT_EPISODES],
        "determinism_encode_repetitions_per_bundle": 2,
        "total_internal_encode_executions": 12,
        "output_policy": "CREATE_NEW_ROOT_AND_WRITER_NO_CLOBBER",
        "dataset_mutations": 0,
        "gpu_used": False,
    }
    manifest_path = output / "pilot_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for file_path in output.rglob("*"):
        if file_path.is_file():
            file_path.chmod(0o444)
    for directory in sorted((path for path in output.rglob("*") if path.is_dir()), reverse=True):
        directory.chmod(0o555)
    output.chmod(0o555)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed three-episode CPU-only R5 codec-evidence pilot"
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = build_pilot(
            args.split_manifest,
            args.output_root,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
    except Exception as error:
        output = Path(args.output_root)
        if output.is_dir():
            failure_path = output / "ATTEMPT_FAILED.json"
            if not failure_path.exists():
                _write_json_new(
                    failure_path,
                    {
                        "schema_version": "codec-evidence-r5-pilot-attempt-v1",
                        "pilot_attempt": 1,
                        "status": "FAILED",
                        "utc_time": datetime.now(timezone.utc).isoformat(),
                        "local_time": datetime.now().astimezone().isoformat(),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "output_preserved": True,
                        "cleanup_performed": False,
                    },
                )
        raise
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
