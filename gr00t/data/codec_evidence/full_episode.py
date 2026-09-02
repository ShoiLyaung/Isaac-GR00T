"""CPU-only canonical-derived codec evidence for complete episodes.

This module deliberately has no upstream JPEG source argument.  Its only image
input is the payload-manifest-bound canonical MP4 that GR00T decodes through
TorchCodec.  Motion vectors are codec encoder decisions, not optical flow; RPE
is a decoded-luma warp proxy, not a bitstream residual.
"""

from __future__ import annotations

from contextlib import nullcontext
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
from .warehouse import (
    SOURCE_KIND,
    WAREHOUSE_OBJECT_SCHEMA_VERSION,
    PublishedObject,
    publish_object,
    sha256_path,
)


FPS = 30
TIME_BASE_NUM = 1
TIME_BASE_DEN = 30_000
PTS_STEP = 1_000
GOP = 30
CRF = 18
CAMERA_ID = "cam_left_high"

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


@dataclass
class DatasetReadAudit:
    forbidden_root: Path
    forbidden_open_count: int = 0
    observed_forbidden_paths: tuple[str, ...] = ()


def install_forbidden_upstream_jpeg_audit(dataset_root: str | Path) -> DatasetReadAudit:
    """Fail any Python-audited open below ``dataset_root/raw`` for this process."""

    forbidden = Path(dataset_root).resolve(strict=True) / "raw"
    audit = DatasetReadAudit(forbidden_root=forbidden)
    observed: list[str] = []

    def hook(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        candidate = arguments[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(candidate)).resolve(strict=False)
        except (OSError, ValueError):
            return
        if path == forbidden or forbidden in path.parents:
            observed.append(str(path))
            audit.forbidden_open_count = len(observed)
            audit.observed_forbidden_paths = tuple(observed)
            raise PermissionError(
                f"Canonical-only producer forbids upstream JPEG source open: {path}"
            )

    sys.addaudithook(hook)
    return audit


class _DatasetConstructionProcessor:
    """Minimum processor boundary needed by DatasetFactory construction only."""

    def __init__(self) -> None:
        self.statistics_calls = 0

    def set_statistics(self, statistics: Mapping[str, Any], *, override: bool = False) -> None:
        del statistics, override
        self.statistics_calls += 1


class DatasetFactoryCanonicalDecoder:
    """Obtain complete pre-transform RGB from the real training dataset topology."""

    def __init__(self, resolved_launch_config: Any) -> None:
        from gr00t.data.dataset.factory import DatasetFactory

        config = resolved_launch_config
        if len(config.data.datasets) != 1:
            raise ValueError("C1 requires exactly one resolved DatasetFactory specification")
        spec = config.data.datasets[0]
        if spec.dataset_adapter != "g1_id10_h40_native30_v1":
            raise ValueError("Resolved launch config does not use the frozen H40 adapter")
        if spec.embodiment_tag != "new_embodiment":
            raise ValueError("Resolved launch config does not use NEW_EMBODIMENT")
        if config.data.video_backend != "torchcodec":
            raise ValueError("Resolved launch config video_backend must remain torchcodec")
        if config.model.action_horizon != 40:
            raise ValueError("Resolved launch config action_horizon must remain 40")
        processor = _DatasetConstructionProcessor()
        train_dataset, eval_dataset = DatasetFactory(config).build(processor)
        if eval_dataset is not None or len(train_dataset.datasets) != 1:
            raise ValueError("Resolved DatasetFactory topology differs from one train dataset")
        if processor.statistics_calls != 1:
            raise ValueError("DatasetFactory did not bind train statistics exactly once")
        single_dataset = train_dataset.datasets[0]
        collection = single_dataset.episode_loader
        if getattr(collection, "adapter_config", {}).get("split") != "train":
            raise ValueError("DatasetFactory collection is not manifest-filtered train")
        self.config = config
        self.single_dataset = single_dataset
        self.collection = collection
        self._bound_generation_guard: Any | None = None
        self._prior_dataset_guard = getattr(single_dataset, "dataset_generation_guard", None)
        self._prior_collection_guard = getattr(collection, "dataset_generation_guard", None)
        self._global_index_by_uid: dict[str, int] = {}
        for global_index, (loader_index, local_index) in enumerate(collection.mapping):
            loader = collection.loaders[loader_index]
            if loader.video_backend != "torchcodec":
                raise ValueError("A real LeRobotEpisodeLoader changed video_backend")
            uid = loader.selected_episode_uids[local_index]
            if uid in self._global_index_by_uid:
                raise ValueError("DatasetFactory emitted a duplicate train episode UID")
            self._global_index_by_uid[uid] = global_index
        self.evidence = {
            "capture_path": "runpy launch_finetune.py -> intercepted run(config)",
            "factory_class": f"{DatasetFactory.__module__}.{DatasetFactory.__qualname__}",
            "dataset_adapter": spec.dataset_adapter,
            "embodiment_tag": spec.embodiment_tag,
            "action_horizon": config.model.action_horizon,
            "video_backend": config.data.video_backend,
            "loader_class": type(collection).__name__,
            "pre_transform_decode_path": (
                "G1Id10H40ManifestEpisodeCollection.__getitem__ -> "
                "LeRobotEpisodeLoader.__getitem__ -> _load_video_data -> "
                "get_frames_by_indices(torchcodec)"
            ),
            "dataset_factory_constructed": True,
        }
        self.runtime_evidence = {
            **self.evidence,
            "adapter_config_path": str(spec.adapter_config_path),
            "dataset_paths": list(spec.dataset_paths),
            "manifest_filtered_train_episode_count": len(self._global_index_by_uid),
        }

    @property
    def train_episode_uids(self) -> tuple[str, ...]:
        return tuple(sorted(self._global_index_by_uid))

    def release_generation_lease(self) -> None:
        """Release the factory dataset's reader lease between producer episodes."""

        if self._bound_generation_guard is not None:
            raise RuntimeError("Cannot release a caller-owned generation guard while it is bound")
        guards = {
            guard
            for guard in (
                getattr(self.single_dataset, "dataset_generation_guard", None),
                getattr(self.collection, "dataset_generation_guard", None),
            )
            if guard is not None
        }
        for guard in guards:
            guard.close()
        self.single_dataset.dataset_generation_guard = None
        self.collection.dataset_generation_guard = None
        self._prior_dataset_guard = None
        self._prior_collection_guard = None

    def bind_generation_guard(self, guard: Any) -> None:
        """Bind one caller-owned shared lease to the complete real loader path."""

        if self._bound_generation_guard is not None:
            raise RuntimeError("A generation guard is already bound")
        guard.ensure_current_process()
        guard.verify_frozen_identity()
        current_dataset = getattr(self.single_dataset, "dataset_generation_guard", None)
        current_collection = getattr(self.collection, "dataset_generation_guard", None)
        if (current_dataset is None) != (current_collection is None) or (
            current_dataset is not None and current_dataset is not current_collection
        ):
            raise RuntimeError("Dataset and collection have inconsistent generation guards")
        if current_dataset is not None and current_dataset is not guard:
            current_dataset.close()
            current_dataset = None
            current_collection = None
        self._prior_dataset_guard = current_dataset
        self._prior_collection_guard = current_collection
        self.single_dataset.dataset_generation_guard = guard
        self.collection.dataset_generation_guard = guard
        self._bound_generation_guard = guard

    def unbind_generation_guard(self, guard: Any) -> None:
        """Remove the exact caller-owned lease without closing it."""

        if self._bound_generation_guard is not guard:
            raise RuntimeError("Attempted to unbind a different generation guard")
        guard.verify_frozen_identity()
        self.single_dataset.dataset_generation_guard = self._prior_dataset_guard
        self.collection.dataset_generation_guard = self._prior_collection_guard
        self._bound_generation_guard = None
        self._prior_dataset_guard = None
        self._prior_collection_guard = None

    def decode_episode(self, episode: CanonicalEpisodeInput) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            global_index = self._global_index_by_uid[episode.episode_uid]
        except KeyError as error:
            raise ValueError(
                "Authorized episode is absent from the real train DatasetFactory"
            ) from error
        metadata = self.collection.episodes_metadata[global_index]
        if (
            metadata.get("dataset_name") != episode.dataset_name
            or int(metadata.get("source_episode_index")) != episode.source_episode_index
        ):
            raise ValueError("DatasetFactory episode identity differs from payload manifest")
        frame = self.collection[global_index]
        if (
            len(frame) != episode.num_frames
            or "video.ego_view" not in frame
            or "frame_index" not in frame
            or "timestamp" not in frame
        ):
            raise ValueError("DatasetFactory did not expose the complete ego_view episode")
        rows = np.arange(episode.num_frames, dtype=np.int64)
        if not np.array_equal(frame["frame_index"].to_numpy(dtype=np.int64), rows):
            raise ValueError("DatasetFactory frame_index is not complete 0..N-1")
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        if not np.allclose(timestamps, rows / FPS, rtol=0.0, atol=1e-6):
            raise ValueError("DatasetFactory timestamp differs from the represented row grid")
        rgb = validate_rgb_sequence(np.stack(frame["video.ego_view"].to_numpy()))
        if rgb.shape[1:] != (480, 640, 3) or not rgb.flags.c_contiguous:
            raise ValueError("C1 DatasetFactory RGB must be C-order uint8 [N,480,640,3]")
        evidence = {
            **self.evidence,
            "episode_uid": episode.episode_uid,
            "dataset_name": episode.dataset_name,
            "source_episode_index": episode.source_episode_index,
            "num_frames": episode.num_frames,
            "dtype": str(rgb.dtype),
            "shape": list(rgb.shape),
            "frame_index_complete": True,
            "timestamp_row_grid_exact": True,
            "c_contiguous": True,
        }
        return rgb, evidence


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


def _encode_once(frames: np.ndarray, output: Path, *, ffmpeg: Path) -> None:
    frames = validate_rgb_sequence(frames)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite encode attempt: {output}")
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

    if np.any(p_field("num_ref_idx_active_override_flag") != 0):
        raise ValueError("P slice overrides the one-entry default L0 list")
    if np.any(p_field("ref_pic_list_modification_flag_l0") != 0):
        raise ValueError("P slice modifies reference list L0")
    if np.any(p_field("adaptive_ref_pic_marking_mode_flag") != 0):
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
            if picture_code == 2 and count == 0:
                raise ValueError("Every fixture P frame must expose codec encoder decisions")
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
                np.asarray(mv_fields["source_x"][begin:end], dtype=np.float32) / 640.0,
                np.asarray(mv_fields["source_y"][begin:end], dtype=np.float32) / 480.0,
                np.asarray(mv_fields["destination_x"][begin:end], dtype=np.float32) / 640.0,
                np.asarray(mv_fields["destination_y"][begin:end], dtype=np.float32) / 480.0,
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
    timestamps = table["timestamp"].to_numpy(dtype=np.float64)
    if not np.allclose(timestamps, rows / FPS, rtol=0.0, atol=1e-6):
        raise ValueError("Synthetic 30 Hz row-grid timestamp mismatch")
    return timestamps


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
    decoded_seconds = np.asarray(seconds, dtype=np.float64)
    if not np.allclose(decoded_seconds, timestamps, rtol=0.0, atol=1e-6):
        raise ValueError("Canonical MP4 PTS differs from represented row-grid timestamps")
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
            "claim_scope": "C0_TEMPORARY_VIDEO_CONFORMANCE_ONLY",
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


def _binary_identity(executable: Path, *, encoder_help: bool = False) -> dict[str, Any]:
    version = _run((executable, "-hide_banner", "-version"), timeout_seconds=60).stdout.decode(
        "utf-8", errors="strict"
    )
    identity: dict[str, Any] = {
        "path": str(executable),
        "sha256": sha256_path(executable),
        "version_output_sha256": sha256(version.encode()).hexdigest(),
        "version_output": version,
    }
    if encoder_help:
        help_result = _run(
            (executable, "-hide_banner", "-h", "encoder=libx264"), timeout_seconds=60
        )
        help_bytes = help_result.stdout + help_result.stderr
        if b"libx264" not in help_bytes:
            raise ValueError("ffmpeg does not expose libx264")
        identity["libx264_help_sha256"] = sha256(help_bytes).hexdigest()
        ldd = _run(("/usr/bin/ldd", executable), timeout_seconds=60).stdout.decode(
            "utf-8", errors="strict"
        )
        candidates = []
        for line in ldd.splitlines():
            if "libx264" not in line or "=>" not in line:
                continue
            candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
            path = Path(candidate).resolve(strict=True)
            candidates.append({"path": str(path), "sha256": sha256_path(path)})
        if len(candidates) != 1:
            raise ValueError("Unable to pin exactly one loaded libx264 shared library")
        identity["loaded_libx264"] = candidates[0]
    return identity


def toolchain_identity(
    *, ffmpeg: str | Path = "/usr/bin/ffmpeg", ffprobe: str | Path = "/usr/bin/ffprobe"
) -> dict[str, Any]:
    ffmpeg_path = _resolved_executable(ffmpeg, "ffmpeg")
    ffprobe_path = _resolved_executable(ffprobe, "ffprobe")
    module_path = Path(__file__).resolve(strict=True)
    from gr00t.data.dataset import (
        dataset_generation_guard,
        factory as dataset_factory_module,
        g1_id10_h40_native30_adapter,
        lerobot_episode_loader,
    )
    from gr00t.model.gr00t_n1d7 import (
        codec_dynamics,
        gr00t_n1d7 as gr00t_model_module,
        processing_gr00t_n1d7,
    )
    from gr00t.utils import video_utils

    from . import (
        config as codec_config,
        contract,
        dataset as codec_dataset,
        episode_bundle,
        implementation_manifest,
        processor as codec_processor,
        torch_backend,
        warehouse,
    )

    try:
        import av
        import torchcodec
        from torchcodec import _core as torchcodec_core
        from torchcodec.decoders import _video_decoder as torchcodec_video_decoder
    except (ImportError, RuntimeError) as error:  # pragma: no cover
        raise RuntimeError("PyAV and TorchCodec must be loaded before toolchain pinning") from error

    def source_identity(module: Any) -> dict[str, Any]:
        path = Path(module.__file__).resolve(strict=True)
        return {"path": str(path), "sha256": sha256_path(path)}

    loaded_shared_libraries: list[dict[str, Any]] = []
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        raise RuntimeError("/proc/self/maps is required to pin loaded decoder libraries")
    seen: set[Path] = set()
    for line in maps_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1]).resolve(strict=False)
        lowered = path.name.lower()
        if not any(
            token in lowered
            for token in (
                "torchcodec",
                "libavcodec",
                "libavformat",
                "libavutil",
                "libswscale",
                "libswresample",
            )
        ):
            continue
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        loaded_shared_libraries.append({"path": str(path), "sha256": sha256_path(path)})
    loaded_shared_libraries.sort(key=lambda item: item["path"])
    if not any("torchcodec" in Path(item["path"]).name for item in loaded_shared_libraries):
        raise RuntimeError("No actually loaded TorchCodec shared library was observed")

    writer_path = Path(episode_bundle.__file__).resolve(strict=True)
    verifier_path = module_path.with_name("full_episode_verifier.py")
    launch_finetune_path = module_path.parents[2] / "experiment/launch_finetune.py"
    if not verifier_path.is_file():
        raise FileNotFoundError("Production verifier source must exist before C1")
    return {
        "ffmpeg": _binary_identity(ffmpeg_path, encoder_help=True),
        "ffprobe": _binary_identity(ffprobe_path),
        "producer_module": {"path": str(module_path), "sha256": sha256_path(module_path)},
        "warehouse_writer": source_identity(warehouse),
        "implementation_manifest_verifier": source_identity(implementation_manifest),
        "production_verifier": {"path": str(verifier_path), "sha256": sha256_path(verifier_path)},
        "launch_finetune": {
            "path": str(launch_finetune_path.resolve(strict=True)),
            "sha256": sha256_path(launch_finetune_path),
        },
        "episode_bundle_validator": {"path": str(writer_path), "sha256": sha256_path(writer_path)},
        "codec_contract": source_identity(contract),
        "codec_config": source_identity(codec_config),
        "pyav": _pyav_identity(av),
        "torchcodec": {
            "version": str(torchcodec.__version__),
            "package": source_identity(torchcodec),
            "core": source_identity(torchcodec_core),
            "video_decoder": source_identity(torchcodec_video_decoder),
            "actually_loaded_shared_libraries": loaded_shared_libraries,
        },
        "training_decode_sources": {
            "codec_dataset_wrapper": source_identity(codec_dataset),
            "codec_processor": source_identity(codec_processor),
            "video_utils": source_identity(video_utils),
            "lerobot_episode_loader": source_identity(lerobot_episode_loader),
            "dataset_factory": source_identity(dataset_factory_module),
            "g1_h40_adapter": source_identity(g1_id10_h40_native30_adapter),
            "dataset_generation_guard": source_identity(dataset_generation_guard),
        },
        "transport_sources": {
            "processor_collator": source_identity(processing_gr00t_n1d7),
            "codec_torch_backend": source_identity(torch_backend),
            "gr00t_prepare_input": source_identity(gr00t_model_module),
        },
        "mv_normalization_reference": {
            **source_identity(codec_dynamics),
            "symbol": "CodecDynamicsCore._aggregate_mv",
        },
        "numpy_version": np.__version__,
        "execution": "CPU_ONLY_NO_HARDWARE_ACCELERATION",
    }


def publish_verified_canonical_object(
    episode: CanonicalEpisodeInput,
    warehouse_root: str | Path,
    *,
    derivation_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    raw_mv_offsets: np.ndarray,
    raw_mv_fields: Mapping[str, np.ndarray],
    accepted_bitstream: str | Path,
) -> tuple[PublishedObject, dict[str, Any]]:
    """Gate immutable publication on a post-read canonical identity check."""

    identity = verify_canonical_input_identity(episode, stage="BEFORE_PUBLISH")
    published = publish_object(
        warehouse_root,
        derivation_payload=derivation_payload,
        manifest_payload=manifest_payload,
        arrays=arrays,
        raw_mv_offsets=raw_mv_offsets,
        raw_mv_fields=raw_mv_fields,
        accepted_bitstream=accepted_bitstream,
    )
    return published, identity


def build_canonical_object(
    warehouse_root: str | Path,
    episode: CanonicalEpisodeInput,
    *,
    scratch_root: str | Path,
    ffmpeg: str | Path = "/usr/bin/ffmpeg",
    ffprobe: str | Path = "/usr/bin/ffprobe",
    tool_identity: Mapping[str, Any] | None = None,
    dataset_factory_decoder: DatasetFactoryCanonicalDecoder,
) -> tuple[PublishedObject, dict[str, Any]]:
    """Build one complete canonical-only episode object."""

    input_identity_before = verify_canonical_input_identity(episode, stage="BEFORE_READ")
    timestamps = validate_canonical_parquet(episode)
    canonical_rgb, canonical_decoder = decode_canonical_pyav(episode, timestamps)
    training_rgb, factory_evidence = dataset_factory_decoder.decode_episode(episode)
    decoder_parity = assert_torchcodec_pyav_exact(
        episode.canonical_video_path,
        canonical_rgb,
        dataset_factory_rgb=training_rgb,
        dataset_factory_evidence=factory_evidence,
    )
    tools = dict(tool_identity or toolchain_identity(ffmpeg=ffmpeg, ffprobe=ffprobe))
    scratch = Path(scratch_root)
    if not scratch.is_dir():
        raise FileNotFoundError(scratch)
    episode_scratch = scratch / f"episode-{episode.episode_uid}"
    episode_scratch.mkdir(mode=0o755, exist_ok=False)
    # nullcontext deliberately preserves both deterministic encodes and all
    # intermediate failure evidence in the append-only attempt scratch root.
    with nullcontext(episode_scratch) as temp:
        encode = encode_h264_twice(canonical_rgb, temp, ffmpeg=ffmpeg)
        packet = probe_packets(
            encode.bitstream_path, num_frames=episode.num_frames, ffprobe=ffprobe
        )
        headers = trace_h264_headers(
            encode.bitstream_path, num_frames=episode.num_frames, ffmpeg=ffmpeg
        )
        extracted = extract_mvs_and_decoded_rgb(
            encode.bitstream_path, num_frames=episode.num_frames
        )
        arrays = build_episode_arrays(extracted, packet, headers)
        source_hashes = [raster_sha256(frame) for frame in canonical_rgb]
        fixture_hashes = [raster_sha256(frame) for frame in extracted.rgb]
        derivation_payload = {
            "schema_version": "codec-evidence-canonical-derivation-v1",
            "episode_uid": episode.episode_uid,
            "camera_id": CAMERA_ID,
            "source_kind": SOURCE_KIND,
            "episode": {
                "cohort_id": episode.cohort_id,
                "dataset_name": episode.dataset_name,
                "source_episode_index": episode.source_episode_index,
                "num_frames": episode.num_frames,
            },
            "canonical_video": {
                "sha256": episode.canonical_video_sha256,
                "bytes": episode.canonical_video_bytes,
            },
            "canonical_parquet": {
                "sha256": episode.parquet_sha256,
                "bytes": episode.parquet_bytes,
            },
            "codec_recipe": {
                "codec": "h264",
                "encoder": "libx264",
                "preset": "medium",
                "crf": CRF,
                "pixel_format": "yuv420p",
                "bframes": 0,
                "references": 1,
                "gop": GOP,
                "scene_cut": 0,
                "open_gop": 0,
                "threads": 1,
                "time_base": "1/30000",
                "fps": FPS,
            },
            "toolchain_identity": tools,
        }
        derivation_digest = _canonical_json_sha256(derivation_payload)
        height, width = canonical_rgb.shape[1:3]
        manifest_payload = {
            "schema_version": WAREHOUSE_OBJECT_SCHEMA_VERSION,
            "derivation_payload_sha256": derivation_digest,
            "episode_uid": episode.episode_uid,
            "camera_id": CAMERA_ID,
            "source_kind": SOURCE_KIND,
            "num_frames": episode.num_frames,
            "episode": {
                "cohort_id": episode.cohort_id,
                "dataset_name": episode.dataset_name,
                "source_episode_index": episode.source_episode_index,
            },
            "width": width,
            "height": height,
            "time_base_num": TIME_BASE_NUM,
            "time_base_den": TIME_BASE_DEN,
            "gop_size": GOP,
            "input_identity": {
                "canonical_video_sha256": episode.canonical_video_sha256,
                "canonical_video_bytes": episode.canonical_video_bytes,
                "canonical_parquet_sha256": episode.parquet_sha256,
                "canonical_parquet_bytes": episode.parquet_bytes,
            },
            "frame_hashes": {
                "source": source_hashes,
                "canonical": source_hashes,
                "fixture": fixture_hashes,
            },
            "identities": {
                "tool": tools,
                "encoder": {
                    "recipe": derivation_payload["codec_recipe"],
                    "attempt_sha256": list(encode.attempt_sha256),
                    "deterministic_sha256_equal": True,
                    "bitstream_sha256": encode.bitstream_sha256,
                    "packet_stream": dict(packet.stream),
                },
                "decoder": {
                    "canonical_pyav": canonical_decoder,
                    "torchcodec_exact_parity": decoder_parity,
                    "fixture_pyav": extracted.pyav_identity,
                },
                "extractor": {
                    "trace_headers_sha256": headers.normalized_trace_sha256,
                    "trace_headers": dict(headers.normalized_trace),
                    "proof_claims": dict(headers.proof_claims),
                    "motion_vector_term": "codec encoder decision; not optical flow",
                    "rpe_term": "decoded-luma MV-warp proxy; not bitstream residual",
                    "raw_mv_fields": list(MV_FIELD_DTYPES),
                    "raw_mv_layout": "CSR_RAGGED_NO_TRUNCATION",
                },
            },
        }
        # No immutable warehouse object may be published until every canonical
        # input has been re-hashed after all reads and derivation work.
        published, input_identity_before_publish = publish_verified_canonical_object(
            episode,
            warehouse_root,
            derivation_payload=derivation_payload,
            manifest_payload=manifest_payload,
            arrays=arrays,
            raw_mv_offsets=extracted.mv_offsets,
            raw_mv_fields=extracted.mv_fields,
            accepted_bitstream=encode.bitstream_path,
        )
    input_identity_after_publish = verify_canonical_input_identity(episode, stage="AFTER_PUBLISH")
    evidence = {
        "episode_uid": episode.episode_uid,
        "num_frames": episode.num_frames,
        "anchors": int((episode.num_frames + GOP - 1) // GOP),
        "tail": int(episode.num_frames % GOP or GOP),
        "derivation_key_sha256": published.derivation_key_sha256,
        "object_content_sha256": published.object_content_sha256,
        "deterministic_bitstream_sha256": encode.bitstream_sha256,
        "torchcodec_pyav_exact": True,
        "input_sha256_unchanged": True,
        "input_identity_checks": [
            input_identity_before,
            input_identity_before_publish,
            input_identity_after_publish,
        ],
        "dataset_factory_constructed": True,
        "dataset_factory_decode": factory_evidence,
        "toolchain_identity": tools,
        "scratch_directory": str(episode_scratch),
        "scratch_preserved": True,
    }
    return published, evidence


def verify_canonical_input_identity(
    episode: CanonicalEpisodeInput, *, stage: str
) -> dict[str, Any]:
    """Re-hash both canonical inputs at one named producer boundary."""

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
    "CanonicalEpisodeInput",
    "DatasetFactoryCanonicalDecoder",
    "DatasetReadAudit",
    "EncodeResult",
    "ExtractedVideo",
    "HeaderProof",
    "PacketProof",
    "aggregate_motion_vectors_2x2",
    "assert_torchcodec_pyav_exact",
    "build_canonical_object",
    "build_episode_arrays",
    "compute_rpe_2x2",
    "decode_canonical_pyav",
    "encode_h264_twice",
    "extract_mvs_and_decoded_rgb",
    "install_forbidden_upstream_jpeg_audit",
    "probe_packets",
    "publish_verified_canonical_object",
    "raster_sha256",
    "toolchain_identity",
    "trace_h264_headers",
    "validate_canonical_parquet",
    "validate_rgb_sequence",
    "verify_canonical_input_identity",
]
