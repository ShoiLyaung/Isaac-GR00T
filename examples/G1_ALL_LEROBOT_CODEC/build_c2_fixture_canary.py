#!/usr/bin/env python3
"""Build the one authorized five-source CPU Codec fixture canary.

The command has a read-only ``--preflight`` mode and one no-clobber
``--execute`` mode.  It never writes the canonical LeRobot inputs, never uses
network or GPU execution, and never retries a failed materialization attempt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
    validate_codec_contract,
)
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.dataset import (
    EPISODE_BUNDLE_INDEX_SCHEMA_VERSION,
    episode_bundle_store_from_index_manifest,
)
from gr00t.data.codec_evidence.episode_bundle import (
    SOURCE_KIND_NATIVE_CANONICAL,
    EpisodeBundleKey,
    EpisodeBundleStore,
    write_episode_bundle,
)
from gr00t.data.codec_evidence.identity import EPISODE_IDENTITY_SCHEMA_VERSION
from gr00t.data.codec_evidence.production import (
    CAMERA_ID,
    CRF,
    GOP,
    CanonicalEpisodeInput,
    assert_torchcodec_pyav_exact,
    build_episode_arrays,
    decode_canonical_pyav,
    encode_h264_twice,
    extract_mvs_and_decoded_rgb,
    probe_packets,
    raster_sha256,
    sha256_path,
    toolchain_identity,
    trace_h264_headers,
    validate_canonical_parquet,
    verify_canonical_input_identity,
)
from gr00t.data.dataset.g1_all_lerobot_h40_adapter import build_codec_episode_identity


SCHEMA_VERSION = "g1-all-lerobot-c2-fixture-canary-v1"
ATTEMPT_ID = "G1-ALL-LEROBOT-C2-FIXTURE-CANARY-R1-ATTEMPT1"
EXPECTED_OUTPUT_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c2-r1")
RECIPE_PATH = Path("examples/G1_ALL_LEROBOT_H40/recipe.20k.json")
RECIPE_SHA256 = "cecca3d46676ad973e9779677ecfb353ff745617ed3784160fbb4e8b685a344d"
PREP_ROOT = Path("/home/sl/icra27-runs/groot-n17-all-lerobot-20k-r1/prep-20260827T171352+0800")
PREP_MANIFEST_SHA256 = "57769ef4469e3f094240299ae4f2c5935f7c9c452a06ff2349122f509776bdbe"
EXPECTED_BASELINE = "dd86aa8deb7c5ee314ca0cf082ab9468b0b61119"
EXPECTED_SOURCE_COUNT = 5
EXPECTED_TOTAL_FRAMES = 1_803
MAX_WALL_SECONDS = 7_200
MAX_NEW_BYTES = 5_000_000_000
FINAL_RECEIPT_RESERVE_BYTES = 1_000_000
MARKERS = (
    Path("/home/sl/groot_datasets/catalog/SYNC_IN_PROGRESS.json"),
    Path("/home/sl/groot_datasets/catalog/INDEX_IN_PROGRESS.json"),
    Path("/home/sl/groot_datasets/catalog/STALE_AFTER_SYNC.json"),
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"JSON input must be one regular non-symlink file: {requested}")
    with requested.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {requested}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    else:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return (text + "\n").encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value, pretty=False).rstrip(b"\n")).hexdigest()


def _write_bytes_new(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def _write_json_new(path: Path, value: Any, *, mode: int = 0o444) -> None:
    _write_bytes_new(path, _canonical_json_bytes(value, pretty=True), mode=mode)


def _append_event(path: Path, event: str, **fields: Any) -> None:
    value = {
        "event": event,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("ab") as stream:
        stream.write(_canonical_json_bytes(value, pretty=False))
        stream.flush()
        os.fsync(stream.fileno())


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _observed_resources(started: float, output_root: Path) -> dict[str, Any]:
    return {
        "wall_seconds": time.monotonic() - started,
        "output_bytes": _directory_bytes(output_root),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _check_budget(started: float, output_root: Path) -> dict[str, Any]:
    observed = _observed_resources(started, output_root)
    if observed["wall_seconds"] > MAX_WALL_SECONDS:
        raise RuntimeError(f"C2 wall-time budget exceeded: {observed}")
    if observed["output_bytes"] > MAX_NEW_BYTES - FINAL_RECEIPT_RESERVE_BYTES:
        raise RuntimeError(f"C2 storage budget exceeded: {observed}")
    return observed


def _check_dataset_markers() -> None:
    present = [str(path) for path in MARKERS if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"Dataset generation marker blocks C2 read: {present}")


def _output_filesystem_identity() -> dict[str, Any]:
    ancestor = EXPECTED_OUTPUT_ROOT
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            raise FileNotFoundError("No existing ancestor for the C2 output root")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise ValueError(f"C2 output ancestor must be a real directory: {ancestor}")
    usage = shutil.disk_usage(ancestor)
    if usage.free < MAX_NEW_BYTES:
        raise RuntimeError(
            f"C2 output filesystem has less than the authorized 5 GB budget: {usage.free}"
        )
    return {
        "existing_ancestor": str(ancestor.resolve(strict=True)),
        "free_bytes_at_preflight": usage.free,
        "required_free_bytes": MAX_NEW_BYTES,
    }


def _file_identity(path: str | Path) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"Identity input must be one regular non-symlink file: {requested}")
    resolved = requested.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_path(resolved),
    }


def _verify_file_closure(closure: Mapping[str, Mapping[str, Any]]) -> None:
    for role, identity in closure.items():
        current = _file_identity(identity["path"])
        if current != dict(identity):
            raise RuntimeError(f"Frozen file identity drifted: {role}")


def _verify_toolchain_identity(toolchain: Mapping[str, Any]) -> None:
    expected_digest = toolchain.get("identity_sha256")
    observed_digest = _canonical_json_sha256(
        {key: value for key, value in toolchain.items() if key != "identity_sha256"}
    )
    if expected_digest != observed_digest:
        raise RuntimeError("Frozen toolchain identity digest is internally inconsistent")
    seen: set[Path] = set()

    def visit(value: Any, role: str) -> None:
        if isinstance(value, Mapping):
            path_value = value.get("path")
            digest = value.get("sha256")
            if isinstance(path_value, str) and isinstance(digest, str):
                path = Path(path_value)
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"Frozen toolchain path disappeared: {role}:{path}")
                resolved = path.resolve(strict=True)
                if resolved not in seen:
                    seen.add(resolved)
                    if sha256_path(resolved) != digest:
                        raise RuntimeError(f"Frozen toolchain file drifted: {role}:{resolved}")
            for key, item in value.items():
                visit(item, f"{role}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{role}.{index}")

    visit(toolchain, "toolchain")
    if not seen:
        raise RuntimeError("Frozen toolchain identity contains no exact file closure")


def _contained_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Dataset payload path escapes its source root: {relative_path}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"Dataset payload may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"Dataset payload resolved outside source root: {relative_path}")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"Dataset payload must be a regular non-symlink file: {resolved}")
    return resolved


def _format_episode_path(
    pattern: str,
    *,
    episode_index: int,
    chunk_size: int,
    video_key: str | None = None,
) -> str:
    chunk = episode_index // chunk_size
    values = {
        "chunk_index": chunk,
        "episode_chunk": chunk,
        "episode_index": episode_index,
        "video_key": video_key,
    }
    try:
        return pattern.format(**values)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(f"Unsupported LeRobot episode path pattern: {pattern}") from error


def _probe_source_video(path: Path, expected_frames: int, ffprobe: str | Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,"
                "time_base,nb_frames,nb_read_frames,has_b_frames"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        },
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe source probe failed: {result.stderr[-4000:]}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Source MP4 must expose one video stream: {path}")
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
    for name, value in expected.items():
        if stream.get(name) != value:
            raise ValueError(f"Source MP4 {name} differs from {value!r}: {path}")
    return stream


def _git_identity(repository: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"C2 execution requires a clean implementation worktree:\n{dirty}")
    head = git("rev-parse", "HEAD")
    baseline = git("merge-base", "HEAD", "feat/groot-n17-all-lerobot-20k-r1")
    if baseline != EXPECTED_BASELINE:
        raise RuntimeError(f"Codec branch baseline drifted: {baseline}")
    return {
        "repository": str(repository),
        "branch": git("branch", "--show-current"),
        "head": head,
        "baseline": baseline,
        "dirty_paths": [],
    }


def _verify_outer_timeout_supervisor() -> None:
    if os.environ.get("C2_OUTER_TIMEOUT_SECONDS") != str(MAX_WALL_SECONDS):
        raise RuntimeError("C2 execute requires the frozen outer 7200-second timeout")
    parent_executable = Path(f"/proc/{os.getppid()}/exe").resolve(strict=True)
    expected = Path("/usr/bin/timeout").resolve(strict=True)
    if parent_executable != expected:
        raise RuntimeError(f"C2 execute parent must be /usr/bin/timeout, got {parent_executable}")
    command = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    expected_prefix = [
        b"/usr/bin/timeout",
        b"--signal=TERM",
        b"--kill-after=30s",
        b"7200s",
    ]
    if command[:4] != expected_prefix:
        raise RuntimeError("C2 outer timeout argv must be --signal=TERM --kill-after=30s 7200s")


def _preparation_identity(
    source_id: str,
    canonical_root: Path,
    closure: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_root = PREP_ROOT / "sources" / source_id
    adapter_path = source_root / "ADAPTER_CONFIG.json"
    view_manifest_path = source_root / "LOADER_VIEW/meta/g1_all_lerobot_h40_view_manifest.json"
    adapter = _load_json(adapter_path)
    view = _load_json(view_manifest_path)
    if (
        adapter.get("source_id") != source_id
        or Path(adapter.get("canonical_root", "")).resolve(strict=True) != canonical_root
        or adapter.get("horizon") != 40
        or adapter.get("row_grid_rate_hz") != 30
    ):
        raise ValueError(f"Prepared adapter identity drifted: {source_id}")
    if view.get("camera_mapping") != {
        "model_key": "ego_view",
        "canonical_key": "cam_left_high",
        "original_key": "observation.images.cam_left_high",
    }:
        raise ValueError(f"Prepared loader camera mapping drifted: {source_id}")
    for payload_name in ("data", "videos"):
        link = source_root / "LOADER_VIEW" / payload_name
        expected = (canonical_root / payload_name).resolve(strict=True)
        if not link.is_symlink() or link.resolve(strict=True) != expected:
            raise ValueError(f"Prepared loader {payload_name} target drifted: {source_id}")
    closure[f"prep.{source_id}.adapter"] = _file_identity(adapter_path)
    closure[f"prep.{source_id}.view_manifest"] = _file_identity(view_manifest_path)
    return {
        "adapter": closure[f"prep.{source_id}.adapter"],
        "view_manifest": closure[f"prep.{source_id}.view_manifest"],
        "loader_view": str((source_root / "LOADER_VIEW").resolve(strict=True)),
        "payload_symlink_targets_exact": True,
    }


def resolve_sources(
    repository: Path,
    *,
    ffprobe: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _check_dataset_markers()
    recipe_path = (repository / RECIPE_PATH).resolve(strict=True)
    if sha256_path(recipe_path) != RECIPE_SHA256:
        raise ValueError("Five-source recipe identity drifted")
    prep_manifest = PREP_ROOT / "DATASET_MANIFEST.json"
    if sha256_path(prep_manifest) != PREP_MANIFEST_SHA256:
        raise ValueError("Prepared five-source DATASET_MANIFEST identity drifted")
    recipe = _load_json(recipe_path)
    dataset_root = Path(recipe["dataset_root"]).resolve(strict=True)
    dataset_specs = recipe.get("datasets")
    if not isinstance(dataset_specs, list) or len(dataset_specs) != EXPECTED_SOURCE_COUNT:
        raise ValueError("C2 requires exactly the five frozen recipe sources")
    closure: dict[str, dict[str, Any]] = {
        "recipe": _file_identity(recipe_path),
        "prep.dataset_manifest": _file_identity(prep_manifest),
    }
    sources: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for dataset in dataset_specs:
        source_id = dataset["source_id"]
        canonical_root = (dataset_root / dataset["relative_path"]).resolve(strict=True)
        if dataset_root not in canonical_root.parents or not canonical_root.is_dir():
            raise ValueError(f"Canonical source root is outside the recipe root: {source_id}")
        info_path = canonical_root / "meta/info.json"
        modality_path = canonical_root / "meta/modality.json"
        episodes_path = canonical_root / "meta/episodes.jsonl"
        info_identity = _file_identity(info_path)
        modality_identity = _file_identity(modality_path)
        if info_identity["sha256"] != dataset["expected_info_sha256"]:
            raise ValueError(f"Canonical info.json identity drifted: {source_id}")
        if modality_identity["sha256"] != dataset["expected_modality_sha256"]:
            raise ValueError(f"Canonical modality.json identity drifted: {source_id}")
        info = _load_json(info_path)
        episodes = _load_jsonl(episodes_path)
        eligible = []
        for row in episodes:
            index = row.get("episode_index")
            length = row.get("length")
            if type(index) is int and type(length) is int and index >= 0 and length >= 40:
                eligible.append((index, length))
        if not eligible:
            raise ValueError(f"Source has no complete episode of length >=40: {source_id}")
        episode_index, num_frames = min(eligible)
        chunk_size = info["chunks_size"]
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError(f"Invalid LeRobot chunks_size: {source_id}")
        parquet_relative = _format_episode_path(
            info["data_path"], episode_index=episode_index, chunk_size=chunk_size
        )
        video_relative = _format_episode_path(
            info["video_path"],
            episode_index=episode_index,
            chunk_size=chunk_size,
            video_key="observation.images.cam_left_high",
        )
        parquet_path = _contained_file(canonical_root, parquet_relative)
        video_path = _contained_file(canonical_root, video_relative)
        parquet_identity = _file_identity(parquet_path)
        video_identity = _file_identity(video_path)
        episode_identity = build_codec_episode_identity(
            source_id=source_id,
            canonical_root=canonical_root,
            canonical_info_sha256=info_identity["sha256"],
            canonical_modality_sha256=modality_identity["sha256"],
            data_path_pattern=info["data_path"],
            video_path_pattern=info["video_path"],
            chunk_size=chunk_size,
            episode_index=episode_index,
            episode_length=num_frames,
        )
        episode_uid = episode_identity["episode_uid"]
        if episode_uid in seen_uids:
            raise ValueError("Five-source C2 episode UIDs must be unique")
        seen_uids.add(episode_uid)
        canonical = CanonicalEpisodeInput(
            episode_uid=episode_uid,
            cohort_id=source_id,
            dataset_name=dataset["relative_path"],
            source_episode_index=episode_index,
            num_frames=num_frames,
            parquet_path=parquet_path,
            parquet_sha256=parquet_identity["sha256"],
            parquet_bytes=parquet_identity["bytes"],
            canonical_video_path=video_path,
            canonical_video_sha256=video_identity["sha256"],
            canonical_video_bytes=video_identity["bytes"],
        )
        validate_canonical_parquet(canonical)
        source_video_probe = _probe_source_video(video_path, num_frames, ffprobe)
        closure[f"source.{source_id}.info"] = info_identity
        closure[f"source.{source_id}.modality"] = modality_identity
        closure[f"source.{source_id}.episodes"] = _file_identity(episodes_path)
        closure[f"source.{source_id}.parquet"] = parquet_identity
        closure[f"source.{source_id}.video"] = video_identity
        preparation = _preparation_identity(source_id, canonical_root, closure)
        sources.append(
            {
                "source_id": source_id,
                "canonical_root": str(canonical_root),
                "episode_index": episode_index,
                "num_frames": num_frames,
                "episode_uid": episode_uid,
                "episode_identity": episode_identity,
                "parquet": parquet_identity,
                "video": video_identity,
                "canonical_info": info_identity,
                "canonical_modality": modality_identity,
                "source_video_probe": source_video_probe,
                "preparation": preparation,
            }
        )
    if sum(source["num_frames"] for source in sources) != EXPECTED_TOTAL_FRAMES:
        raise ValueError("C2 five-source total frame count drifted")
    _verify_file_closure(closure)
    return sources, closure


def _implementation_closure(repository: Path) -> dict[str, dict[str, Any]]:
    relative_paths = {
        "producer_cli": "examples/G1_ALL_LEROBOT_CODEC/build_c2_fixture_canary.py",
        "production_primitives": "gr00t/data/codec_evidence/production.py",
        "episode_bundle_writer": "gr00t/data/codec_evidence/episode_bundle.py",
        "codec_contract": "gr00t/data/codec_evidence/codec_contract.py",
        "episode_identity": "gr00t/data/codec_evidence/identity.py",
        "adapter_identity": "gr00t/data/dataset/g1_all_lerobot_h40_adapter.py",
        "mv_normalization_reference": "gr00t/model/gr00t_n1d7/codec_dynamics.py",
        "bundle_schema": (
            "gr00t/data/codec_evidence/schemas/codec-evidence-episode-bundle-v1.schema.json"
        ),
        "index_schema": (
            "gr00t/data/codec_evidence/schemas/codec-evidence-episode-bundle-index-v1.schema.json"
        ),
        "test_production_fixture": "tests/codec_evidence/test_production_fixture.py",
        "test_c2_fixture_cli": "tests/codec_evidence/test_c2_fixture_cli.py",
        "test_dataset_protocol": "tests/codec_evidence/test_dataset_protocol.py",
        "project_configuration": "pyproject.toml",
    }
    return {
        role: _file_identity(repository / relative) for role, relative in relative_paths.items()
    }


def _compact_identity(name: str, value: Any) -> dict[str, str]:
    return {"name": name, "version": "c2-r1", "sha256": _canonical_json_sha256(value)}


def _codec_contract(toolchain: Mapping[str, Any]) -> dict[str, Any]:
    encoder_details = {
        "ffmpeg": toolchain["ffmpeg"],
        "libx264": toolchain["libx264"],
        "recipe": {
            "codec": "h264",
            "container": "mp4",
            "pixel_format": "yuv420p",
            "preset": "medium",
            "crf": CRF,
            "gop": GOP,
            "bframes": 0,
            "refs": 1,
            "threads": 1,
            "time_base": "1/30000",
            "pts_step": 1000,
        },
    }
    decoder_details = {
        "pyav": toolchain["pyav"],
        "torchcodec": toolchain["torchcodec"],
        "parity": "EVERY_SELECTED_SOURCE_ROW_MUST_BE_EXACT_UINT8_RGB_EQUAL",
    }
    extractor_details = {
        "motion_vector_extractor": toolchain["motion_vector_extractor"],
        "mv_fields": list(MV_FIELD_DTYPES),
        "mv_term": "codec encoder decision; not optical flow",
        "rpe_term": "decoded-luma MV-warp proxy; not bitstream residual",
        "aggregation_reference": toolchain["implementation"]["mv_normalization_reference"],
    }
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
    )
    contract.update(
        {
            "tool_identity": _compact_identity("g1-all-lerobot-c2-toolchain", toolchain),
            "encoder_identity": _compact_identity("g1-all-lerobot-c2-encoder", encoder_details),
            "decoder_identity": _compact_identity("g1-all-lerobot-c2-decoder", decoder_details),
            "extractor_identity": _compact_identity(
                "g1-all-lerobot-c2-mv-extractor", extractor_details
            ),
        }
    )
    return validate_codec_contract(contract)


def build_plan(repository: Path) -> dict[str, Any]:
    _check_dataset_markers()
    if EXPECTED_OUTPUT_ROOT.exists() or EXPECTED_OUTPUT_ROOT.is_symlink():
        raise FileExistsError(
            f"Authorized C2 output root is already consumed: {EXPECTED_OUTPUT_ROOT}"
        )
    git = _git_identity(repository)
    output_filesystem = _output_filesystem_identity()
    tools = toolchain_identity(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe")
    implementation = _implementation_closure(repository)
    tools = {
        **tools,
        "runtime_python": _file_identity(Path(sys.executable).resolve(strict=True)),
        "outer_timeout": _file_identity("/usr/bin/timeout"),
        "implementation": implementation,
        "git": git,
    }
    tools["identity_sha256"] = _canonical_json_sha256(
        {key: value for key, value in tools.items() if key != "identity_sha256"}
    )
    sources, input_closure = resolve_sources(repository, ffprobe="/usr/bin/ffprobe")
    contract = _codec_contract(tools)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": ATTEMPT_ID,
        "attempts": 1,
        "automatic_retries": 0,
        "output_root": str(EXPECTED_OUTPUT_ROOT),
        "execution": {
            "cpu_only": True,
            "network": "offline_no_download",
            "gpu": False,
            "max_wall_seconds": MAX_WALL_SECONDS,
            "max_new_bytes": MAX_NEW_BYTES,
            "ffmpeg_threads": 1,
            "source_inputs_read_only": True,
            "output_filesystem": output_filesystem,
        },
        "selection_policy": "minimum episode_index with complete length>=40 per frozen source",
        "source_count": len(sources),
        "total_frames": sum(source["num_frames"] for source in sources),
        "sources": sources,
        "input_file_closure": input_closure,
        "toolchain": tools,
        "codec_contract": contract,
        "codec_contract_sha256": codec_contract_sha256(contract),
        "fixture_index_schema_version": EPISODE_BUNDLE_INDEX_SCHEMA_VERSION,
        "training_eligible": False,
        "scope": "FIVE_SOURCE_ONE_EPISODE_EACH_TECHNICAL_CANARY_NOT_FULL_TRAIN_COVERAGE",
        "forbidden": [
            "network_or_download",
            "gpu_or_hardware_acceleration",
            "checkpoint_or_dataset_mutation",
            "training",
            "evaluator",
            "automatic_retry",
        ],
    }
    plan["plan_sha256"] = _canonical_json_sha256(plan)
    return plan


def _episode_from_source(source: Mapping[str, Any]) -> CanonicalEpisodeInput:
    return CanonicalEpisodeInput(
        episode_uid=source["episode_uid"],
        cohort_id=source["source_id"],
        dataset_name=Path(source["canonical_root"]).name,
        source_episode_index=source["episode_index"],
        num_frames=source["num_frames"],
        parquet_path=Path(source["parquet"]["path"]),
        parquet_sha256=source["parquet"]["sha256"],
        parquet_bytes=source["parquet"]["bytes"],
        canonical_video_path=Path(source["video"]["path"]),
        canonical_video_sha256=source["video"]["sha256"],
        canonical_video_bytes=source["video"]["bytes"],
    )


def _reserve_output_root() -> Path:
    parent = EXPECTED_OUTPUT_ROOT.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"C2 output parent must be a real directory: {parent}")
    EXPECTED_OUTPUT_ROOT.mkdir(mode=0o755, exist_ok=False)
    return EXPECTED_OUTPUT_ROOT.resolve(strict=True)


def _remove_success_scratch(path: Path, output_root: Path) -> None:
    resolved = path.resolve(strict=True)
    scratch_root = (output_root / ".scratch").resolve(strict=True)
    if scratch_root not in resolved.parents:
        raise RuntimeError("Refusing to clean scratch outside the C2 attempt root")
    shutil.rmtree(resolved)


def _verified_encode_source_identity(encoded: Any) -> dict[str, Any]:
    identity = _file_identity(encoded.bitstream_path)
    if (
        identity["sha256"] != encoded.bitstream_sha256
        or encoded.attempt_sha256 != (encoded.bitstream_sha256,) * 2
    ):
        raise RuntimeError("Accepted encode source drifted after the deterministic double hash")
    return identity


def _content_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {"bytes": identity["bytes"], "sha256": identity["sha256"]}


def _write_bundle(
    output_root: Path,
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    event_path: Path,
) -> tuple[dict[str, Any], EpisodeBundleKey]:
    source_id = source["source_id"]
    episode = _episode_from_source(source)
    _check_dataset_markers()
    _verify_file_closure(plan["toolchain"]["implementation"])
    _verify_toolchain_identity(plan["toolchain"])
    before = verify_canonical_input_identity(episode, stage="BEFORE_READ")
    timestamps = validate_canonical_parquet(episode)
    canonical_rgb, canonical_decoder = decode_canonical_pyav(episode, timestamps)
    parity = assert_torchcodec_pyav_exact(episode.canonical_video_path, canonical_rgb)
    scratch = output_root / ".scratch" / source_id
    scratch.mkdir(parents=True, exist_ok=False)
    _append_event(event_path, "SOURCE_DECODED", source_id=source_id, frames=episode.num_frames)
    encoded = encode_h264_twice(canonical_rgb, scratch)
    packets = probe_packets(encoded.bitstream_path, num_frames=episode.num_frames)
    headers = trace_h264_headers(encoded.bitstream_path, num_frames=episode.num_frames)
    extracted = extract_mvs_and_decoded_rgb(encoded.bitstream_path, num_frames=episode.num_frames)
    arrays = build_episode_arrays(extracted, packets, headers)
    encode_source_identity = _verified_encode_source_identity(encoded)
    contract = plan["codec_contract"]
    metadata = {
        "split": "canary",
        "cohort_id": source_id,
        "dataset_name": Path(source["canonical_root"]).name,
        "source_episode_index": episode.source_episode_index,
        "width": int(canonical_rgb.shape[2]),
        "height": int(canonical_rgb.shape[1]),
        "time_base_num": 1,
        "time_base_den": 30_000,
        "gop_size": GOP,
        "source_frame_sha256": [raster_sha256(frame) for frame in canonical_rgb],
        "canonical_frame_sha256": [raster_sha256(frame) for frame in canonical_rgb],
        "fixture_frame_sha256": [raster_sha256(frame) for frame in extracted.rgb],
        "tool_identity": contract["tool_identity"],
        "encoder_identity": contract["encoder_identity"],
        "decoder_identity": contract["decoder_identity"],
        "extractor_identity": contract["extractor_identity"],
        "episode_identity": {
            key: value for key, value in source["episode_identity"].items() if key != "episode_uid"
        },
        "codec_contract": contract,
        "codec_contract_sha256": plan["codec_contract_sha256"],
    }
    bundle_manifest = write_episode_bundle(
        output_root,
        episode_uid=episode.episode_uid,
        camera_id=CAMERA_ID,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
        metadata=metadata,
        arrays=arrays,
        raw_mv_offsets=extracted.mv_offsets,
        raw_mv_fields=extracted.mv_fields,
        accepted_bitstream=encoded.bitstream_path,
    )
    after = verify_canonical_input_identity(episode, stage="AFTER_WRITE")
    key = EpisodeBundleKey(episode.episode_uid, CAMERA_ID, SOURCE_KIND_NATIVE_CANONICAL)
    store = EpisodeBundleStore(output_root)
    loaded = store.get(key)
    raw_audit = store.audit_raw_motion_vectors(key)
    retained_bitstream = store.audit_accepted_bitstream(key)
    retained_identity = _file_identity(retained_bitstream)
    if _content_identity(retained_identity) != _content_identity(encode_source_identity):
        raise RuntimeError("Retained accepted bitstream differs from the frozen encode source")
    manifest = _load_json(bundle_manifest)
    if manifest["accepted_bitstream"] is None or manifest["raw_motion_vector_audit"] is None:
        raise RuntimeError("C2 bundle omitted required production audit artifacts")
    if set(manifest["raw_motion_vector_audit"]["fields"]) != set(MV_FIELD_DTYPES):
        raise RuntimeError("C2 bundle raw-MV audit fields are incomplete")
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "episode_uid": episode.episode_uid,
        "episode_index": episode.source_episode_index,
        "num_frames": episode.num_frames,
        "anchors": loaded.num_anchors,
        "tail_gop_frames": episode.num_frames % GOP or GOP,
        "input_identity_checks": [before, after],
        "input_identity_unchanged": before["canonical_video"] == after["canonical_video"]
        and before["canonical_parquet"] == after["canonical_parquet"],
        "canonical_decoder": canonical_decoder,
        "torchcodec_pyav_exact": parity,
        "deterministic_encode_sha256": list(encoded.attempt_sha256),
        "deterministic_encode_equal": len(set(encoded.attempt_sha256)) == 1,
        "accepted_bitstream": retained_identity,
        "packet_stream": dict(packets.stream),
        "header_proof_sha256": headers.normalized_trace_sha256,
        "header_proof_claims": dict(headers.proof_claims),
        "raw_mv_count": int(raw_audit.offsets[-1]),
        "raw_mv_fields": list(raw_audit.fields),
        "bundle_manifest": _file_identity(bundle_manifest),
        "bundle_runtime_roundtrip_valid": True,
    }
    _write_json_new(output_root / "episode_results" / f"{source_id}.json", result)
    _remove_success_scratch(scratch, output_root)
    return (
        {
            "episode_uid": episode.episode_uid,
            "camera_id": CAMERA_ID,
            "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
            "relative_path": bundle_manifest.relative_to(output_root).as_posix(),
            "sha256": sha256_path(bundle_manifest),
            "bytes": bundle_manifest.stat().st_size,
        },
        key,
    )


def execute(plan: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    del repository
    started = time.monotonic()
    output_root: Path | None = None
    event_path: Path | None = None
    try:
        _verify_file_closure(plan["toolchain"]["implementation"])
        _verify_toolchain_identity(plan["toolchain"])
        _verify_file_closure(plan["input_file_closure"])
        _check_dataset_markers()
        output_root = _reserve_output_root()
        event_path = output_root / "events.jsonl"
        _append_event(event_path, "ATTEMPT_STARTED", attempt_id=ATTEMPT_ID)
        _write_json_new(
            output_root / "ATTEMPT_STARTED.json",
            {
                "schema_version": SCHEMA_VERSION,
                "attempt_id": ATTEMPT_ID,
                "status": "STARTED",
                "attempts": 1,
                "automatic_retries": 0,
                "argv": list(sys.argv),
                "plan_sha256": plan["plan_sha256"],
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "output_root": str(output_root),
            },
        )
        _write_json_new(output_root / "PLAN.json", plan)
        _write_json_new(output_root / "TOOLCHAIN_IDENTITY.json", plan["toolchain"])
        _write_json_new(
            output_root / "SOURCE_SELECTION.json",
            {
                "schema_version": SCHEMA_VERSION,
                "selection_policy": plan["selection_policy"],
                "sources": plan["sources"],
                "input_file_closure": plan["input_file_closure"],
            },
        )
        _write_json_new(output_root / "CODEC_CONTRACT.json", plan["codec_contract"])
        entries: list[dict[str, Any]] = []
        keys: list[EpisodeBundleKey] = []
        episode_summaries: list[dict[str, Any]] = []
        for source in plan["sources"]:
            _check_budget(started, output_root)
            _append_event(
                event_path,
                "SOURCE_STARTED",
                source_id=source["source_id"],
                frames=source["num_frames"],
            )
            entry, key = _write_bundle(output_root, source, plan, event_path)
            entries.append(entry)
            keys.append(key)
            episode_result = _load_json(
                output_root / "episode_results" / f"{source['source_id']}.json"
            )
            episode_summaries.append(
                {
                    "source_id": source["source_id"],
                    "episode_uid": source["episode_uid"],
                    "num_frames": source["num_frames"],
                    "anchors": episode_result["anchors"],
                    "raw_mv_count": episode_result["raw_mv_count"],
                    "bitstream_sha256": episode_result["accepted_bitstream"]["sha256"],
                    "torchcodec_pyav_exact": True,
                }
            )
            observed = _check_budget(started, output_root)
            _append_event(
                event_path,
                "SOURCE_ACCEPTED",
                source_id=source["source_id"],
                observed=observed,
            )

        expected_uids = {source["episode_uid"] for source in plan["sources"]}
        if (
            len(entries) != EXPECTED_SOURCE_COUNT
            or {entry["episode_uid"] for entry in entries} != expected_uids
        ):
            raise RuntimeError("C2 bundle index coverage is not the exact five-source selection")
        _verify_file_closure(plan["input_file_closure"])
        _verify_file_closure(plan["toolchain"]["implementation"])
        _verify_toolchain_identity(plan["toolchain"])
        _check_dataset_markers()
        fixture_manifest = {
            "schema_version": EPISODE_BUNDLE_INDEX_SCHEMA_VERSION,
            "complete": True,
            "camera_id": CAMERA_ID,
            "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
            "identity": {
                "episode_identity_schema_version": EPISODE_IDENTITY_SCHEMA_VERSION,
                "codec_contract_sha256": plan["codec_contract_sha256"],
            },
            "codec_contract": plan["codec_contract"],
            "bundles": entries,
        }
        fixture_bytes = _canonical_json_bytes(fixture_manifest, pretty=False)
        fixture_sha256 = sha256(fixture_bytes).hexdigest()
        _write_json_new(
            output_root / "ATTEMPT_READY_FOR_INDEX_COMMIT.json",
            {
                "schema_version": SCHEMA_VERSION,
                "attempt_id": ATTEMPT_ID,
                "status": "READY_FOR_INDEX_COMMIT",
                "fixture_manifest_sha256": fixture_sha256,
                "bundle_count": len(entries),
            },
        )
        _write_bytes_new(output_root / "fixture_manifest.json", fixture_bytes)
        verified_store = episode_bundle_store_from_index_manifest(
            output_root,
            fixture_manifest_sha256=fixture_sha256,
            camera_id=CAMERA_ID,
            source_kind=SOURCE_KIND_NATIVE_CANONICAL,
        )
        if verified_store.keys != frozenset(keys):
            raise RuntimeError("Committed C2 index key closure differs from five-source plan")
        for key in keys:
            verified_store.get(key)
            verified_store.audit_raw_motion_vectors(key)
            verified_store.audit_accepted_bitstream(key)
        observed = _check_budget(started, output_root)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": ATTEMPT_ID,
            "status": "TECHNICALLY_VERIFIED_READY_FOR_TERMINAL_COMMIT",
            "scope": plan["scope"],
            "training_eligible": False,
            "source_count": len(entries),
            "episode_count": len(entries),
            "total_frames": sum(item["num_frames"] for item in episode_summaries),
            "total_anchors": sum(item["anchors"] for item in episode_summaries),
            "fixture_manifest": {
                "path": str(output_root / "fixture_manifest.json"),
                "sha256": fixture_sha256,
            },
            "codec_contract_sha256": plan["codec_contract_sha256"],
            "toolchain_identity_sha256": plan["toolchain"]["identity_sha256"],
            "input_identity_unchanged": True,
            "torchcodec_pyav_exact_all_rows": True,
            "deterministic_double_encode_all_bundles": True,
            "raw_mv_exact_eleven_fields_all_bundles": True,
            "runtime_bundle_and_index_verification": True,
            "episodes": episode_summaries,
            "resources": observed,
            "network": "offline_no_download",
            "gpu_used": False,
            "training_run": False,
            "evaluator_run": False,
        }
        _write_json_new(output_root / "summary.json", summary)
        _append_event(event_path, "ATTEMPT_VERIFIED_READY", summary=summary)
        _write_json_new(
            output_root / "ATTEMPT_TERMINAL.json",
            {
                "schema_version": SCHEMA_VERSION,
                "attempt_id": ATTEMPT_ID,
                "status": "ACCEPTED",
                "summary_sha256": sha256_path(output_root / "summary.json"),
                "fixture_manifest_sha256": fixture_sha256,
                "utc_time": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except BaseException as error:
        if output_root is not None and output_root.is_dir():
            failure = {
                "schema_version": SCHEMA_VERSION,
                "attempt_id": ATTEMPT_ID,
                "status": "FAILED_ATTEMPT_CONSUMED_NO_RETRY",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "resources": _observed_resources(started, output_root),
            }
            failure_path = output_root / "ATTEMPT_FAILED.json"
            if not failure_path.exists():
                _write_json_new(failure_path, failure)
            if event_path is not None and event_path.exists():
                _append_event(event_path, "ATTEMPT_FAILED", error=failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=EXPECTED_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError(f"C2 output root is frozen to {EXPECTED_OUTPUT_ROOT}")
    repository = Path(__file__).resolve(strict=True).parents[2]
    plan = build_plan(repository)
    if args.preflight:
        print(_canonical_json_bytes(plan, pretty=True).decode("utf-8"), end="")
        return 0
    _verify_outer_timeout_supervisor()
    summary = execute(plan, repository)
    print(_canonical_json_bytes(summary, pretty=True).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
