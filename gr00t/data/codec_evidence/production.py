"""CPU-only production primitives for causal episode-bundle materialization.

The module is deliberately independent of any dataset adapter, warehouse,
supervisor, model, or GPU runtime. Its image inputs are canonical local MP4
files or explicit RGB arrays. Motion vectors are codec encoder decisions, not
optical flow; RPE is a decoded-luma motion-warp proxy, not a bitstream residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import MAX_MV_PER_FRAME, MV_FIELD_DTYPES
from .episode_bundle import FRAME_TYPE_IDR, FRAME_TYPE_P


FPS = 30
TIME_BASE_NUM = 1
TIME_BASE_DEN = 30_000
PTS_STEP = 1_000
GOP = 30
CRF = 18
CAMERA_ID = "cam_left_high"
X264_CAUSAL_PARAMETERS = (
    "bframes=0",
    "ref=1",
    "keyint=30",
    "min-keyint=16",
    "scenecut=0",
    "open-gop=0",
    "threads=1",
    "lookahead-threads=1",
    "sliced-threads=0",
    "sync-lookahead=0",
    "rc-lookahead=0",
    "mbtree=0",
)

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
class CanonicalEpisodeInput:
    episode_uid: str
    cohort_id: str
    dataset_name: str
    source_episode_index: int
    num_frames: int
    parquet_path: Path
    parquet_sha256: str
    parquet_bytes: int
    canonical_video_path: Path
    canonical_video_sha256: str
    canonical_video_bytes: int


@dataclass(frozen=True)
class EncodeResult:
    bitstream_path: Path
    bitstream_sha256: str
    attempt_sha256: tuple[str, str]


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
    normalized_trace_sha256: str
    normalized_trace: Mapping[str, Any]
    proof_claims: Mapping[str, bool]


@dataclass(frozen=True)
class ExtractedVideo:
    rgb: np.ndarray
    pts: np.ndarray
    frame_type: np.ndarray
    mv_offsets: np.ndarray
    mv_fields: Mapping[str, np.ndarray]
    pyav_identity: Mapping[str, Any]


def sha256_path(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _resolved_executable(path: str | Path, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{name} is not executable: {resolved}")
    return resolved


def _cpu_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        }
    )
    return environment


def _run(
    command: Sequence[str | Path],
    *,
    input_bytes: bytes | None = None,
    timeout_seconds: int = 1_800,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        [str(item) for item in command],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=_cpu_environment(),
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-8_000:]
        raise RuntimeError(f"Command failed with status {result.returncode}: {stderr}")
    return result


def validate_rgb_sequence(frames: np.ndarray) -> np.ndarray:
    value = np.asarray(frames)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise TypeError("Frames must be uint8 [N,H,W,3] RGB")
    if value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError("Frame count and dimensions must be positive")
    if value.shape[1] % 2 or value.shape[2] % 2:
        raise ValueError("yuv420p requires even dimensions")
    return np.ascontiguousarray(value)


def raster_sha256(frame: np.ndarray) -> str:
    value = np.asarray(frame)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise TypeError("Raster hash requires uint8 HxWx3 RGB")
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_represented_30hz_timestamps(value: Any, num_frames: int) -> np.ndarray:
    """Accept only the exact ideal-float64 or stored-float32 30 Hz row grid."""

    timestamps = np.asarray(value)
    if timestamps.shape != (num_frames,) or not np.issubdtype(timestamps.dtype, np.floating):
        raise ValueError("Represented timestamps must be floating [N]")
    observed = timestamps.astype(np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("Represented timestamps must be finite")
    rows = np.arange(num_frames, dtype=np.float64)
    ideal = rows / FPS
    stored_float32 = ideal.astype(np.float32).astype(np.float64)
    if not np.array_equal(observed, ideal) and not np.array_equal(observed, stored_float32):
        raise ValueError("Timestamp differs from exact float64/float32 30 Hz row grid")
    return observed


def _encode_once(frames: np.ndarray, output: Path, *, ffmpeg: Path) -> None:
    frames = validate_rgb_sequence(frames)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite encode attempt: {output}")
    height, width = frames.shape[1:3]
    x264_parameters = ":".join(X264_CAUSAL_PARAMETERS)
    command = (
        ffmpeg,
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
        output,
    )
    _run(command, input_bytes=frames.tobytes())
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("libx264 did not create a non-empty stream")


def encode_h264_twice(
    frames: np.ndarray,
    attempt_directory: str | Path,
    *,
    ffmpeg: str | Path = "/usr/bin/ffmpeg",
) -> EncodeResult:
    frames = validate_rgb_sequence(frames)
    directory = Path(attempt_directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ffmpeg_path = _resolved_executable(ffmpeg, "ffmpeg")
    first = directory / "libx264-attempt-1.mp4"
    second = directory / "libx264-attempt-2.mp4"
    _encode_once(frames, first, ffmpeg=ffmpeg_path)
    _encode_once(frames, second, ffmpeg=ffmpeg_path)
    digests = (sha256_path(first), sha256_path(second))
    if digests[0] != digests[1]:
        raise ValueError("Two deterministic encodes produced different SHA-256 values")
    return EncodeResult(first, digests[0], digests)


def probe_x264_encoder_options(bitstream: str | Path) -> dict[str, Any]:
    """Fail closed unless the x264 SEI matches the effective causal profile.

    This x264 build omits ``rc_lookahead`` from its user-data string when the
    value is zero.  The returned receipt records that omission explicitly;
    prefix-equivalence remains the independent behavioral proof that the
    command's frozen ``rc-lookahead=0`` request took effect.
    """

    path = Path(bitstream)
    if path.is_symlink() or not path.is_file():
        raise ValueError("x264 option probe requires a regular non-symlink bitstream")
    text = path.read_bytes().decode("latin1", errors="ignore")
    observed: dict[str, int | None] = {}
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
        observed[name] = int(match.group(1)) if match else None
    expected = {
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
    mismatches = {
        name: {"expected": value, "observed": observed.get(name)}
        for name, value in expected.items()
        if observed.get(name) != value
    }
    if observed["rc_lookahead"] not in {None, 0}:
        mismatches["rc_lookahead"] = {
            "expected": "zero_or_x264_zero_omission",
            "observed": observed["rc_lookahead"],
        }
    if mismatches:
        raise ValueError(f"x264 effective options differ from causal contract: {mismatches}")
    return {
        "observed": observed,
        "expected": expected,
        "rc_lookahead_zero_representation": (
            "EXPLICIT_ZERO" if observed["rc_lookahead"] == 0 else "OMITTED_WHEN_ZERO"
        ),
        "requested_parameters": list(X264_CAUSAL_PARAMETERS),
        "effective_options_match": True,
    }


def probe_packets(
    bitstream: str | Path,
    *,
    num_frames: int,
    ffprobe: str | Path = "/usr/bin/ffprobe",
) -> PacketProof:
    ffprobe_path = _resolved_executable(ffprobe, "ffprobe")
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
    if len(streams) != 1 or len(packets) != num_frames:
        raise ValueError("Encoded MP4 stream/packet count mismatch")
    stream = streams[0]
    expected = {
        "codec_name": "h264",
        "codec_type": "video",
        "time_base": "1/30000",
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "has_b_frames": 0,
        "refs": 1,
        "nb_frames": str(num_frames),
    }
    for name, value in expected.items():
        if stream.get(name) != value:
            raise ValueError(f"Encoded stream {name} differs from {value!r}")
    pts = np.asarray([int(packet["pts"]) for packet in packets], dtype=np.int64)
    dts = np.asarray([int(packet["dts"]) for packet in packets], dtype=np.int64)
    duration = np.asarray([int(packet["duration"]) for packet in packets], dtype=np.int64)
    key_frame = np.asarray(["K" in packet.get("flags", "") for packet in packets], dtype=np.bool_)
    expected_time = np.arange(num_frames, dtype=np.int64) * PTS_STEP
    expected_key = np.zeros(num_frames, dtype=np.bool_)
    expected_key[::GOP] = True
    if not np.array_equal(pts, expected_time) or not np.array_equal(dts, expected_time):
        raise ValueError("Packets must satisfy PTS=DTS=row*1000")
    if not np.array_equal(duration, np.full(num_frames, PTS_STEP, dtype=np.int64)):
        raise ValueError("Packet durations must equal 1000")
    if not np.array_equal(key_frame, expected_key):
        raise ValueError("Key packets must occur exactly at row%30==0")
    return PacketProof(pts, dts, duration, key_frame, stream)


_TRACE_NAL = re.compile(r"\[trace_headers[^]]*\]\s+nal_unit_type:\s*(\d+).*nal_ref_idc:\s*(\d+)")
_TRACE_FIELD = re.compile(
    r"\[trace_headers[^]]*\]\s+\d+\s+([A-Za-z0-9_\[\]]+)\s+[01]+\s+=\s+(-?\d+)\s*$"
)


def trace_h264_headers(
    bitstream: str | Path,
    *,
    num_frames: int,
    ffmpeg: str | Path = "/usr/bin/ffmpeg",
) -> HeaderProof:
    ffmpeg_path = _resolved_executable(ffmpeg, "ffmpeg")
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
    trace = result.stderr.decode("utf-8", errors="strict")
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_nal: tuple[int, int] | None = None
    for line in trace.splitlines():
        nal_match = _TRACE_NAL.search(line)
        if nal_match:
            pending_nal = (int(nal_match.group(1)), int(nal_match.group(2)))
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
                raise ValueError("Slice header has no preceding VCL NAL identity")
            current = {
                "nal_unit_type": pending_nal[0],
                "nal_ref_idc": pending_nal[1],
                "fields": {},
            }
            entries.append(current)
            continue
        field_match = _TRACE_FIELD.search(line)
        if field_match and current is not None:
            current["fields"].setdefault(field_match.group(1), int(field_match.group(2)))

    sps = [entry for entry in entries if entry["nal_unit_type"] == 7]
    pps = [entry for entry in entries if entry["nal_unit_type"] == 8]
    vcl = [entry for entry in entries if entry["nal_unit_type"] in {1, 5}]
    if not sps or not pps or len(vcl) != num_frames:
        raise ValueError("trace_headers did not expose exact SPS/PPS/VCL closure")
    if {entry["fields"].get("max_num_ref_frames") for entry in sps} != {1}:
        raise ValueError("SPS must prove max_num_ref_frames=1")
    if {entry["fields"].get("gaps_in_frame_num_allowed_flag") for entry in sps} != {0}:
        raise ValueError("SPS must forbid frame_num gaps")
    max_frame_values = {entry["fields"].get("log2_max_frame_num_minus4") for entry in sps}
    if len(max_frame_values) != 1 or None in max_frame_values:
        raise ValueError("SPS max_frame_num is ambiguous")
    max_frame_num = 1 << (int(next(iter(max_frame_values))) + 4)
    if {entry["fields"].get("num_ref_idx_l0_default_active_minus1") for entry in pps} != {0}:
        raise ValueError("PPS must prove one default active L0 reference")

    rows = np.arange(num_frames, dtype=np.int64)
    expected_nal = np.where(rows % GOP == 0, 5, 1).astype(np.uint8)
    nal = np.asarray([entry["nal_unit_type"] for entry in vcl], dtype=np.uint8)
    nal_ref_idc = np.asarray([entry["nal_ref_idc"] for entry in vcl], dtype=np.uint8)
    slices = np.asarray([entry["fields"].get("slice_type", -1) for entry in vcl], dtype=np.int16)
    frame_num = np.asarray([entry["fields"].get("frame_num", -1) for entry in vcl], dtype=np.int64)
    if not np.array_equal(nal, expected_nal):
        raise ValueError("VCL NAL sequence differs from closed GOP30 I/P-only contract")
    expected_slice = np.where(rows % GOP == 0, 2, 0).astype(np.int16)
    if np.any(slices < 0) or not np.array_equal(slices % 5, expected_slice):
        raise ValueError("Slice types differ from I-at-GOP-start/P-elsewhere")
    if not np.array_equal(frame_num, (rows % GOP) % max_frame_num):
        raise ValueError("frame_num is not contiguous within each GOP")
    p_rows = np.flatnonzero(expected_nal == 1)
    idr_rows = np.flatnonzero(expected_nal == 5)
    if np.any(nal_ref_idc[p_rows] == 0):
        raise ValueError("Every P picture must be a reference picture")

    def p_field(name: str) -> np.ndarray:
        values = np.asarray([vcl[row]["fields"].get(name, -1) for row in p_rows], dtype=np.int8)
        if np.any(values < 0):
            raise ValueError(f"Every P slice must expose {name}")
        return values

    l0_override = p_field("num_ref_idx_active_override_flag")
    l0_modification = p_field("ref_pic_list_modification_flag_l0")
    adaptive_marking = p_field("adaptive_ref_pic_marking_mode_flag")
    if np.any(l0_override != 0):
        raise ValueError("P slice overrides the one-entry default L0 list")
    if np.any(l0_modification != 0):
        raise ValueError("P slice modifies reference list L0")
    if np.any(adaptive_marking != 0):
        raise ValueError("P slice signals MMCO")
    idr_long_term = np.asarray(
        [vcl[row]["fields"].get("long_term_reference_flag", -1) for row in idr_rows],
        dtype=np.int8,
    )
    if not np.array_equal(idr_long_term, np.zeros(len(idr_rows), dtype=np.int8)):
        raise ValueError("IDR long_term_reference_flag must be zero")
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
        raise ValueError("Long-term, MMCO, or list-modification syntax is forbidden")
    frame_type = np.where(rows % GOP == 0, FRAME_TYPE_IDR, FRAME_TYPE_P).astype(np.uint8)
    references = rows - 1
    references[idr_rows] = -1
    normalized = {
        "num_frames": num_frames,
        "gop_size": GOP,
        "max_num_ref_frames": 1,
        "max_frame_num": max_frame_num,
        "nal_unit_type": nal.tolist(),
        "nal_ref_idc": nal_ref_idc.tolist(),
        "slice_type": slices.tolist(),
        "frame_num": frame_num.tolist(),
        "reference_index": references.tolist(),
        "num_ref_idx_active_override_flag": l0_override.tolist(),
        "ref_pic_list_modification_flag_l0": l0_modification.tolist(),
        "adaptive_ref_pic_marking_mode_flag": adaptive_marking.tolist(),
        "idr_long_term_reference_flag": idr_long_term.tolist(),
        "forbidden_syntax_absent": list(forbidden_syntax),
    }
    claims = {
        "idr_only_at_row_mod_gop_zero": True,
        "all_other_frames_are_reference_p": True,
        "max_num_ref_frames_is_one": True,
        "default_l0_has_one_entry": True,
        "no_l0_override_or_modification": True,
        "no_long_term_reference_or_mmco": True,
        "closed_gop_30": True,
        "reference_is_immediately_previous_row": True,
        "reference_identity_does_not_use_mv_source": True,
    }
    return HeaderProof(
        frame_type, references, _canonical_json_sha256(normalized), normalized, claims
    )


def _pyav_identity(av_module: Any) -> dict[str, Any]:
    module_path = Path(av_module.__file__).resolve(strict=True)
    return {
        "implementation": "PyAV export_mvs / RGB24 CPU",
        "pyav_version": str(av_module.__version__),
        "path": str(module_path),
        "sha256": sha256_path(module_path),
        "module_path": str(module_path),
        "module_sha256": sha256_path(module_path),
        "library_versions": {
            str(name): list(version) if isinstance(version, tuple) else str(version)
            for name, version in sorted(av_module.library_versions.items())
        },
    }


def extract_mvs_and_decoded_rgb(bitstream: str | Path, *, num_frames: int) -> ExtractedVideo:
    try:
        import av
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyAV is required for codec extraction") from error
    rgb_frames: list[np.ndarray] = []
    pts_values: list[int] = []
    frame_types: list[int] = []
    offsets = [0]
    field_chunks: dict[str, list[np.ndarray]] = {name: [] for name in MV_FIELD_DTYPES}
    with av.open(str(Path(bitstream)), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("Accepted fixture must expose one video stream")
        stream = streams[0]
        stream.codec_context.options = {"flags2": "+export_mvs"}
        for frame in container.decode(stream):
            if len(rgb_frames) >= num_frames:
                raise ValueError("PyAV decoded more than the expected complete episode")
            if frame.pts is None:
                raise ValueError("Fixture frame is missing PTS")
            rgb_frames.append(
                np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            )
            pts_values.append(int(frame.pts))
            picture_code = int(frame.pict_type)
            if picture_code not in {1, 2}:
                raise ValueError("Fixture contains a picture other than I or P")
            frame_types.append(picture_code)
            side_data = frame.side_data.get("MOTION_VECTORS")
            count = 0
            if side_data is not None:
                vectors = np.asarray(side_data.to_ndarray())
                if vectors.ndim != 1 or vectors.dtype.names is None:
                    raise ValueError("PyAV motion-vector layout is unexpected")
                if set(vectors.dtype.names) != set(PYAV_MV_NAMES.values()):
                    raise ValueError(
                        "PyAV motion-vector fields differ from the exact eleven fields"
                    )
                count = int(vectors.shape[0])
                if count > MAX_MV_PER_FRAME:
                    raise ValueError("Motion-vector count exceeds the no-truncation bound")
                for output_name, dtype in MV_FIELD_DTYPES.items():
                    field_chunks[output_name].append(
                        np.ascontiguousarray(vectors[PYAV_MV_NAMES[output_name]], dtype=dtype)
                    )
            else:
                for output_name, dtype in MV_FIELD_DTYPES.items():
                    field_chunks[output_name].append(np.empty(0, dtype=dtype))
            offsets.append(offsets[-1] + count)
    if len(rgb_frames) != num_frames:
        raise ValueError("PyAV did not decode the complete expected episode")
    rgb = validate_rgb_sequence(np.stack(rgb_frames))
    pts = np.asarray(pts_values, dtype=np.int64)
    if not np.array_equal(pts, np.arange(num_frames, dtype=np.int64) * PTS_STEP):
        raise ValueError("Fixture PyAV PTS differs from row*1000")
    expected_picture = np.where(np.arange(num_frames) % GOP == 0, 1, 2).astype(np.uint8)
    if not np.array_equal(np.asarray(frame_types, dtype=np.uint8), expected_picture):
        raise ValueError("Fixture PyAV picture types differ from GOP30 I/P-only")
    fields = {
        name: np.ascontiguousarray(np.concatenate(chunks), dtype=dtype)
        for (name, dtype), chunks in zip(MV_FIELD_DTYPES.items(), field_chunks.values())
    }
    return ExtractedVideo(
        rgb=rgb,
        pts=pts,
        frame_type=np.where(np.arange(num_frames) % GOP == 0, FRAME_TYPE_IDR, FRAME_TYPE_P).astype(
            np.uint8
        ),
        mv_offsets=np.asarray(offsets, dtype=np.int64),
        mv_fields=fields,
        pyav_identity=_pyav_identity(av),
    )


def aggregate_motion_vectors_2x2(
    mv_offsets: np.ndarray,
    mv_fields: Mapping[str, np.ndarray],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(mv_offsets)
    if offsets.dtype != np.int64 or offsets.ndim != 1 or offsets.size < 2:
        raise ValueError("MV offsets must be int64 [N+1]")
    num_frames = offsets.size - 1
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("MV offsets must start at zero and be monotonic")
    total = int(offsets[-1])
    if set(mv_fields) != set(MV_FIELD_DTYPES):
        raise ValueError("MV fields differ from the exact eleven-field set")
    for name, dtype in MV_FIELD_DTYPES.items():
        value = np.asarray(mv_fields[name])
        if value.dtype != dtype or value.shape != (total,):
            raise ValueError(f"MV field {name} has wrong dtype/shape")
    features = np.zeros((num_frames, 2, 2, 11), dtype=np.float32)
    mask = np.zeros((num_frames, 2, 2), dtype=np.bool_)
    camera = np.zeros((num_frames, 4), dtype=np.float32)
    camera_mask = np.zeros(num_frames, dtype=np.bool_)
    for row in range(num_frames):
        begin, end = int(offsets[row]), int(offsets[row + 1])
        if begin == end:
            continue
        scale = np.maximum(np.asarray(mv_fields["motion_scale"][begin:end], dtype=np.float32), 1.0)
        normalized = np.stack(
            (
                np.asarray(mv_fields["source"][begin:end], dtype=np.float32) / 4.0,
                np.asarray(mv_fields["block_width"][begin:end], dtype=np.float32) / 64.0,
                np.asarray(mv_fields["block_height"][begin:end], dtype=np.float32) / 64.0,
                np.asarray(mv_fields["source_x"][begin:end], dtype=np.float32) / float(width),
                np.asarray(mv_fields["source_y"][begin:end], dtype=np.float32) / float(height),
                np.asarray(mv_fields["destination_x"][begin:end], dtype=np.float32) / float(width),
                np.asarray(mv_fields["destination_y"][begin:end], dtype=np.float32) / float(height),
                np.asarray(mv_fields["flags"][begin:end], dtype=np.float32) / 16.0,
                np.asarray(mv_fields["motion_x"][begin:end], dtype=np.float32) / scale,
                np.asarray(mv_fields["motion_y"][begin:end], dtype=np.float32) / scale,
                np.log2(scale) / 16.0,
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        x = np.asarray(mv_fields["destination_x"][begin:end], dtype=np.float32)
        y = np.asarray(mv_fields["destination_y"][begin:end], dtype=np.float32)
        cell_x = (x >= 320.0).astype(np.int64)
        cell_y = (y >= 240.0).astype(np.int64)
        for cell_row in range(2):
            for cell_column in range(2):
                selected = (cell_y == cell_row) & (cell_x == cell_column)
                if np.any(selected):
                    mask[row, cell_row, cell_column] = True
                    features[row, cell_row, cell_column] = np.mean(
                        normalized[selected], axis=0, dtype=np.float32
                    )
        scale64 = np.maximum(
            np.asarray(mv_fields["motion_scale"][begin:end], dtype=np.float64), 1.0
        )
        dx = np.asarray(mv_fields["motion_x"][begin:end], dtype=np.float64) / scale64
        dy = np.asarray(mv_fields["motion_y"][begin:end], dtype=np.float64) / scale64
        area = np.asarray(mv_fields["block_width"][begin:end], dtype=np.float64) * np.asarray(
            mv_fields["block_height"][begin:end], dtype=np.float64
        )
        if np.any(area <= 0):
            raise ValueError("Every exported MV block must have positive area")
        weight = area / np.sum(area)
        camera[row] = np.asarray(
            [
                np.sum(weight * dx),
                np.sum(weight * dy),
                np.sum(weight * np.hypot(dx, dy)),
                min(1.0, float(np.sum(area)) / float(width * height)),
            ],
            dtype=np.float32,
        )
        camera_mask[row] = True
    if not np.isfinite(features).all() or not np.isfinite(camera).all():
        raise ValueError("MV aggregation produced non-finite values")
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
    frames = validate_rgb_sequence(decoded_rgb)
    num_frames, height, width = frames.shape[:3]
    offsets = np.asarray(mv_offsets)
    references = np.asarray(reference_index)
    if offsets.dtype != np.int64 or offsets.shape != (num_frames + 1,):
        raise ValueError("MV offsets must be int64 [N+1]")
    if references.dtype != np.int64 or references.shape != (num_frames,):
        raise ValueError("reference_index must be int64 [N]")
    rgb32 = frames.astype(np.uint32)
    luma = ((77 * rgb32[..., 0] + 150 * rgb32[..., 1] + 29 * rgb32[..., 2] + 128) >> 8).astype(
        np.float32
    )
    rpe = np.zeros((num_frames, 2, 2, 4), dtype=np.float32)
    mask = np.zeros((num_frames, 2, 2), dtype=np.bool_)
    for row in range(num_frames):
        reference = int(references[row])
        if reference < 0:
            continue
        if reference != row - 1:
            raise ValueError("RPE requires an immediately-previous bitstream reference")
        residual = np.zeros((height, width), dtype=np.float32)
        covered = np.zeros((height, width), dtype=np.bool_)
        begin, end = int(offsets[row]), int(offsets[row + 1])
        for vector in range(begin, end):
            block_width = int(mv_fields["block_width"][vector])
            block_height = int(mv_fields["block_height"][vector])
            dx0, dx1 = _block_bounds(int(mv_fields["destination_x"][vector]), block_width)
            dy0, dy1 = _block_bounds(int(mv_fields["destination_y"][vector]), block_height)
            sx0, sx1 = _block_bounds(int(mv_fields["source_x"][vector]), block_width)
            sy0, sy1 = _block_bounds(int(mv_fields["source_y"][vector]), block_height)
            left = max(0, -dx0, -sx0)
            right = max(0, dx1 - width, sx1 - width)
            top = max(0, -dy0, -sy0)
            bottom = max(0, dy1 - height, sy1 - height)
            copy_width = block_width - left - right
            copy_height = block_height - top - bottom
            if copy_width <= 0 or copy_height <= 0:
                continue
            destination = np.s_[
                dy0 + top : dy0 + top + copy_height, dx0 + left : dx0 + left + copy_width
            ]
            source = np.s_[
                sy0 + top : sy0 + top + copy_height, sx0 + left : sx0 + left + copy_width
            ]
            residual[destination] = luma[row][destination] - luma[reference][source]
            covered[destination] = True
        row_slices = (slice(0, height // 2), slice(height // 2, height))
        column_slices = (slice(0, width // 2), slice(width // 2, width))
        for cell_row, row_slice in enumerate(row_slices):
            for cell_column, column_slice in enumerate(column_slices):
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
                mask[row, cell_row, cell_column] = True
    if not np.isfinite(rpe).all():
        raise ValueError("RPE produced non-finite values")
    return rpe, mask


def _resize_anchor(frame: np.ndarray) -> np.ndarray:
    rows = np.rint(np.linspace(0, frame.shape[0] - 1, 256)).astype(np.int64)
    columns = np.rint(np.linspace(0, frame.shape[1] - 1, 256)).astype(np.int64)
    resized = frame[rows[:, None], columns[None, :], :]
    return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0


def build_episode_arrays(
    extracted: ExtractedVideo, packet: PacketProof, headers: HeaderProof
) -> dict[str, np.ndarray]:
    num_frames = extracted.rgb.shape[0]
    if not np.array_equal(extracted.pts, packet.pts):
        raise ValueError("PyAV and ffprobe PTS differ")
    if not np.array_equal(extracted.frame_type, headers.frame_type):
        raise ValueError("PyAV and trace_headers frame types differ")
    height, width = extracted.rgb.shape[1:3]
    mv_features, mv_mask, camera, camera_mask = aggregate_motion_vectors_2x2(
        extracted.mv_offsets, extracted.mv_fields, width=width, height=height
    )
    rpe, rpe_mask = compute_rpe_2x2(
        extracted.rgb, extracted.mv_offsets, extracted.mv_fields, headers.reference_index
    )
    rows = np.arange(num_frames, dtype=np.int64)
    anchors = rows[rows % GOP == 0]
    anchor_per_frame = (rows // GOP) * GOP
    anchor_appearance = np.stack(
        [_resize_anchor(extracted.rgb[int(index)]) for index in anchors]
    ).astype(np.float32, copy=False)
    return {
        "frame_index": rows,
        "pts": np.ascontiguousarray(packet.pts, dtype=np.int64),
        "dts": np.ascontiguousarray(packet.dts, dtype=np.int64),
        "frame_type": np.ascontiguousarray(headers.frame_type, dtype=np.uint8),
        "gop_id": rows // GOP,
        "gop_position": rows % GOP,
        "reference_index": np.ascontiguousarray(headers.reference_index, dtype=np.int64),
        "frame_validity": np.ones((num_frames, 7), dtype=np.bool_),
        "frame_reason_code": np.zeros(num_frames, dtype=np.int16),
        "mv_cell_features": mv_features,
        "mv_cell_mask": mv_mask,
        "rpe": rpe,
        "rpe_mask": rpe_mask,
        "camera_motion_summary": camera,
        "camera_motion_mask": camera_mask,
        "anchor_index_per_frame": anchor_per_frame.astype(np.int64),
        "anchor_frame_index": anchors.astype(np.int64),
        "anchor_appearance": anchor_appearance,
    }


def validate_canonical_parquet(episode: CanonicalEpisodeInput) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("pandas/pyarrow are required for canonical provenance") from error
    if episode.parquet_path.is_symlink():
        raise ValueError("Canonical Parquet must not be a symlink")
    path = episode.parquet_path.resolve(strict=True)
    if sha256_path(path) != episode.parquet_sha256:
        raise ValueError("Canonical Parquet identity mismatch")
    if path.stat().st_size != episode.parquet_bytes:
        raise ValueError("Canonical Parquet byte count mismatch")
    table = pd.read_parquet(path)
    required = {
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
        "annotation.human.validity",
    }
    if not required.issubset(table.columns) or len(table) != episode.num_frames:
        raise ValueError("Canonical Parquet row/schema mismatch")
    rows = np.arange(episode.num_frames, dtype=np.int64)
    if not np.array_equal(table["frame_index"].to_numpy(dtype=np.int64), rows):
        raise ValueError("Canonical frame_index must be complete 0..N-1")
    if not np.array_equal(
        table["episode_index"].to_numpy(dtype=np.int64),
        np.full(episode.num_frames, episode.source_episode_index, dtype=np.int64),
    ):
        raise ValueError("Canonical episode_index mismatch")
    if np.any(table["task_index"].to_numpy(dtype=np.int64) != 0):
        raise ValueError("Canonical task_index must be zero")
    if np.any(table["annotation.human.validity"].to_numpy(dtype=np.int64) != 1):
        raise ValueError("Canonical episode contains invalid rows")
    return validate_represented_30hz_timestamps(table["timestamp"].to_numpy(), episode.num_frames)


def decode_canonical_pyav(
    episode: CanonicalEpisodeInput, timestamps: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import av
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyAV is required for canonical decode") from error
    if episode.canonical_video_path.is_symlink():
        raise ValueError("Canonical MP4 must not be a symlink")
    path = episode.canonical_video_path.resolve(strict=True)
    if sha256_path(path) != episode.canonical_video_sha256:
        raise ValueError("Canonical MP4 identity mismatch")
    if path.stat().st_size != episode.canonical_video_bytes:
        raise ValueError("Canonical MP4 byte count mismatch")
    frames: list[np.ndarray] = []
    seconds: list[float] = []
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("Canonical MP4 must expose one video stream")
        stream = streams[0]
        identity = {
            "codec_sniffed_by_pyav": str(stream.codec_context.name),
            "profile": str(stream.codec_context.profile),
            "pixel_format": str(stream.codec_context.format.name),
            "time_base": str(stream.time_base),
            "declared_frames": int(stream.frames),
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
        }
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Canonical decoded frame is missing PTS")
            frames.append(np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8))
            seconds.append(float(Fraction(int(frame.pts)) * frame.time_base))
    if len(frames) != episode.num_frames:
        raise ValueError("Canonical MP4 decode count differs from complete episode")
    rgb = validate_rgb_sequence(np.stack(frames))
    validate_represented_30hz_timestamps(timestamps, episode.num_frames)
    decoded_seconds = np.asarray(seconds, dtype=np.float64)
    expected_seconds = np.arange(episode.num_frames, dtype=np.float64) / FPS
    if not np.allclose(decoded_seconds, expected_seconds, rtol=0.0, atol=1e-12):
        raise ValueError("Canonical MP4 PTS differs from the exact 30 Hz row grid")
    return rgb, {
        "source_file_sha256": episode.canonical_video_sha256,
        "stream": identity,
        "pyav": _pyav_identity(av),
        "decode_policy": "SNIFF_ACTUAL_CODEC_THEN_RGB24_CPU",
    }


def assert_torchcodec_pyav_exact(
    video_path: str | Path,
    pyav_rgb: np.ndarray,
    *,
    dataset_factory_rgb: np.ndarray | None = None,
    dataset_factory_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare with the exact pre-transform utility used by LeRobot loader."""

    from gr00t.utils.video_utils import get_frames_by_indices

    pyav_rgb = validate_rgb_sequence(pyav_rgb)
    frame_count = pyav_rgb.shape[0]
    if dataset_factory_rgb is None:
        decoded = get_frames_by_indices(
            str(Path(video_path)),
            np.arange(frame_count, dtype=np.int64),
            video_backend="torchcodec",
            video_backend_kwargs={},
        )
        torchcodec_rgb = np.asarray(decoded)
        path_evidence: dict[str, Any] = {
            "dataset_factory_constructed": False,
            "claim_scope": (
                "DIRECT_LEROBOT_TORCHCODEC_UTILITY_CONFORMANCE_NOT_DATASET_FACTORY_CONSTRUCTION"
            ),
        }
    else:
        if (
            not isinstance(dataset_factory_evidence, Mapping)
            or dataset_factory_evidence.get("dataset_factory_constructed") is not True
        ):
            raise ValueError("C1 parity requires real DatasetFactory provenance")
        torchcodec_rgb = np.asarray(dataset_factory_rgb)
        path_evidence = dict(dataset_factory_evidence)
    if torchcodec_rgb.dtype != np.uint8 or torchcodec_rgb.shape != pyav_rgb.shape:
        raise ValueError("TorchCodec pre-transform dtype/shape differs from PyAV RGB24")
    if not np.array_equal(torchcodec_rgb, pyav_rgb):
        mismatch = np.abs(torchcodec_rgb.astype(np.int16) - pyav_rgb.astype(np.int16))
        raise ValueError(f"TorchCodec/PyAV exact parity failed: max_abs_diff={int(mismatch.max())}")
    hashes = [raster_sha256(frame) for frame in pyav_rgb]
    raster_digest = sha256(b"".join(bytes.fromhex(value) for value in hashes)).hexdigest()
    try:
        import torchcodec
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("TorchCodec disappeared after successful decode") from error
    module_path = Path(torchcodec.__file__).resolve(strict=True)
    return {
        "backend": "torchcodec",
        "api": (
            "gr00t.utils.video_utils.get_frames_by_indices -> "
            "VideoDecoder(path,device=cpu,dimension_order=NHWC,num_ffmpeg_threads=0)."
            "get_frames_at(0..N-1)"
        ),
        "module_path": str(module_path),
        "module_sha256": sha256_path(module_path),
        "version": str(torchcodec.__version__),
        "num_frames": frame_count,
        "dtype": "uint8",
        "shape": list(torchcodec_rgb.shape),
        "all_rows_array_equal": True,
        "max_abs_diff": 0,
        "c_order_raster_digest": raster_digest,
        "per_row_pyav_rgb_sha256": hashes,
        "per_row_dataset_factory_torchcodec_rgb_sha256": hashes,
        "per_row_ledger_sha256": _canonical_json_sha256(hashes),
        "per_row_hash_equality_count": frame_count,
        "dataset_factory": path_evidence,
    }


def _module_identity(module: Any) -> dict[str, Any]:
    path = Path(module.__file__).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Module identity is not a regular file: {path}")
    return {
        "name": str(module.__name__),
        "path": str(path),
        "sha256": sha256_path(path),
    }


def _binary_identity(executable: Path) -> dict[str, Any]:
    version = _run((executable, "-hide_banner", "-version"), timeout_seconds=60).stdout.decode(
        "utf-8", errors="strict"
    )
    first_line = version.splitlines()[0] if version.splitlines() else ""
    if not first_line:
        raise ValueError(f"Could not identify executable version: {executable}")
    return {
        "path": str(executable),
        "sha256": sha256_path(executable),
        "version": first_line,
        "version_output": version,
        "version_output_sha256": sha256(version.encode("utf-8")).hexdigest(),
    }


def _loaded_shared_libraries(tokens: Sequence[str]) -> list[dict[str, Any]]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        raise RuntimeError("/proc/self/maps is required to freeze loaded shared libraries")
    normalized_tokens = tuple(token.lower() for token in tokens)
    seen: set[Path] = set()
    identities: list[dict[str, Any]] = []
    for line in maps_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1]).resolve(strict=False)
        if path in seen or not path.is_file():
            continue
        lowered = path.name.lower()
        if not any(token in lowered for token in normalized_tokens):
            continue
        seen.add(path)
        identities.append({"path": str(path), "sha256": sha256_path(path)})
    identities.sort(key=lambda item: item["path"])
    return identities


def _libx264_identity(ffmpeg: Path) -> dict[str, Any]:
    ldd = _resolved_executable("/usr/bin/ldd", "ldd")
    output = _run((ldd, ffmpeg), timeout_seconds=60).stdout.decode("utf-8", errors="strict")
    candidates: dict[Path, dict[str, Any]] = {}
    for line in output.splitlines():
        if "libx264" not in line or "=>" not in line:
            continue
        target = line.split("=>", 1)[1].strip().split(" ", 1)[0]
        path = Path(target).resolve(strict=True)
        if not path.is_file():
            raise ValueError("ldd resolved libx264 to a non-regular file")
        candidates[path] = {
            "soname": line.split("=>", 1)[0].strip(),
            "path": str(path),
            "sha256": sha256_path(path),
        }
    if len(candidates) != 1:
        raise ValueError("ffmpeg must resolve exactly one libx264 shared library")
    return next(iter(candidates.values()))


def _system_ffmpeg_codec_libraries(ffmpeg: Path) -> list[dict[str, Any]]:
    ldd = _resolved_executable("/usr/bin/ldd", "ldd")
    output = _run((ldd, ffmpeg), timeout_seconds=60).stdout.decode("utf-8", errors="strict")
    prefixes = ("libavcodec", "libavformat", "libavutil", "libswscale", "libx264")
    identities: dict[Path, dict[str, Any]] = {}
    for line in output.splitlines():
        soname = line.split("=>", 1)[0].strip()
        if not soname.startswith(prefixes) or "=>" not in line:
            continue
        target = line.split("=>", 1)[1].strip().split(" ", 1)[0]
        path = Path(target).resolve(strict=True)
        identities[path] = {
            "soname": soname,
            "path": str(path),
            "sha256": sha256_path(path),
        }
    if not identities or not any(
        item["soname"].startswith("libx264") for item in identities.values()
    ):
        raise ValueError("ffmpeg codec-library closure is incomplete")
    return [identities[path] for path in sorted(identities, key=str)]


def toolchain_identity(
    *,
    ffmpeg: str | Path = "/usr/bin/ffmpeg",
    ffprobe: str | Path = "/usr/bin/ffprobe",
) -> dict[str, Any]:
    """Return a read-only identity closure for every production codec tool.

    Importing TorchCodec loads its extension libraries. The closure then hashes
    the exact libraries observed in this process rather than merely enumerating
    package candidates on disk.
    """

    ffmpeg_path = _resolved_executable(ffmpeg, "ffmpeg")
    ffprobe_path = _resolved_executable(ffprobe, "ffprobe")
    try:
        import av
        from av.sidedata import motionvectors as pyav_motionvectors
        import torchcodec
        from torchcodec import _core as torchcodec_core
        from torchcodec.decoders import _video_decoder as torchcodec_video_decoder
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("PyAV and TorchCodec are required for production identity") from error

    ffmpeg_identity = _binary_identity(ffmpeg_path)
    help_result = _run(
        (ffmpeg_path, "-hide_banner", "-h", "encoder=libx264"),
        timeout_seconds=60,
    )
    help_bytes = help_result.stdout + help_result.stderr
    if b"libx264" not in help_bytes:
        raise ValueError("ffmpeg does not expose the libx264 encoder")

    torchcodec_libraries = _loaded_shared_libraries(("torchcodec",))
    if not torchcodec_libraries:
        raise RuntimeError("No actually loaded TorchCodec shared library was observed")
    decoder_libraries = _loaded_shared_libraries(
        ("libavcodec", "libavformat", "libavutil", "libswscale", "libswresample")
    )
    pyav_extensions = []
    for name, module in sorted(sys.modules.items()):
        path_value = getattr(module, "__file__", None)
        if not (name == "av" or name.startswith("av.")) or not isinstance(path_value, str):
            continue
        path = Path(path_value).resolve(strict=True)
        if path.suffix == ".so":
            pyav_extensions.append({"name": name, "path": str(path), "sha256": sha256_path(path)})
    production_module = Path(__file__).resolve(strict=True)
    identity = {
        "schema_version": "codec-evidence-production-toolchain-v1",
        "execution": "CPU_ONLY_OFFLINE_LOCAL_FILES",
        "ffmpeg": {
            **ffmpeg_identity,
            "libx264_help_sha256": sha256(help_bytes).hexdigest(),
        },
        "ffprobe": _binary_identity(ffprobe_path),
        "libx264": _libx264_identity(ffmpeg_path),
        "system_ffmpeg_codec_libraries": _system_ffmpeg_codec_libraries(ffmpeg_path),
        "pyav": {
            **_module_identity(av),
            "version": str(av.__version__),
            "library_versions": {
                str(name): list(value) if isinstance(value, tuple) else str(value)
                for name, value in sorted(av.library_versions.items())
            },
            "loaded_extension_modules": pyav_extensions,
        },
        "motion_vector_extractor": {
            "implementation": "PyAV export_mvs exact eleven-field CSR",
            "production_module": {
                "path": str(production_module),
                "sha256": sha256_path(production_module),
                "symbol": "extract_mvs_and_decoded_rgb",
            },
            "pyav_motionvectors_module": _module_identity(pyav_motionvectors),
        },
        "torchcodec": {
            "version": str(torchcodec.__version__),
            "package": _module_identity(torchcodec),
            "core": _module_identity(torchcodec_core),
            "video_decoder": _module_identity(torchcodec_video_decoder),
            "actually_loaded_shared_libraries": torchcodec_libraries,
            "actually_loaded_decoder_libraries": decoder_libraries,
        },
        "numpy_version": str(np.__version__),
    }
    return {
        **identity,
        "identity_sha256": _canonical_json_sha256(identity),
    }


def verify_canonical_input_identity(
    episode: CanonicalEpisodeInput, *, stage: str
) -> dict[str, Any]:
    """Re-hash both canonical inputs at one named read-only boundary."""

    if not isinstance(stage, str) or not stage:
        raise ValueError("Input identity stage must be non-empty")
    result: dict[str, Any] = {"stage": stage}
    for name, path, expected_sha, expected_bytes in (
        (
            "canonical_video",
            episode.canonical_video_path,
            episode.canonical_video_sha256,
            episode.canonical_video_bytes,
        ),
        (
            "canonical_parquet",
            episode.parquet_path,
            episode.parquet_sha256,
            episode.parquet_bytes,
        ),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{stage}: {name} is missing, non-regular, or a symlink")
        observed_bytes = path.stat().st_size
        observed_sha = sha256_path(path)
        if observed_bytes != expected_bytes or observed_sha != expected_sha:
            raise ValueError(f"{stage}: {name} identity drifted")
        result[name] = {"sha256": observed_sha, "bytes": observed_bytes}
    return result


__all__ = [
    "CAMERA_ID",
    "CRF",
    "FPS",
    "GOP",
    "PTS_STEP",
    "X264_CAUSAL_PARAMETERS",
    "CanonicalEpisodeInput",
    "EncodeResult",
    "ExtractedVideo",
    "HeaderProof",
    "PacketProof",
    "aggregate_motion_vectors_2x2",
    "assert_torchcodec_pyav_exact",
    "build_episode_arrays",
    "compute_rpe_2x2",
    "decode_canonical_pyav",
    "encode_h264_twice",
    "extract_mvs_and_decoded_rgb",
    "probe_packets",
    "probe_x264_encoder_options",
    "raster_sha256",
    "sha256_path",
    "toolchain_identity",
    "trace_h264_headers",
    "validate_canonical_parquet",
    "validate_represented_30hz_timestamps",
    "validate_rgb_sequence",
    "verify_canonical_input_identity",
]
