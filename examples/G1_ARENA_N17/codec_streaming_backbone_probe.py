#!/usr/bin/env python3
"""Run the frozen untrained Arena Codec IR streaming-backbone probe."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterator

import gr00t
import gr00t.experimental
import gr00t.experimental.arena_codec_streaming_backbone as arena_codec_module
from gr00t.experimental.arena_codec_streaming_backbone import (
    CODEC_FIELDS,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAMBA2_CONFIG,
    SEED,
    WIDTH,
    ArenaCodecSpatialTokenizer,
    CausalGRUBackbone,
    CausalMamba2Backbone,
    CausalTransformerBackbone,
    CurrentDescriptorBackbone,
    trainable_parameter_count,
)
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "g1-arena-codec-streaming-backbone-r1"
RUN_ID = "untrained-gpu1-r1"
BASE_SOURCE_COMMIT = "455323ac28043e95206135ee3a22c4486ef786e8"
ARTIFACT_ROOT = Path("/home/sl/research/icra27-artifacts/g1-arena-codec-streaming-backbone-r1")
INPUT_PATH = Path(
    "/home/sl/research/icra27-artifacts/g1-arena-native-first-codec-ir-r1/"
    "episode45-native-first-r1/episode_000045.native_first_codec_ir_v1.npz"
)
DEPENDENCY_MANIFEST_PATH = ARTIFACT_ROOT / "dependencies/DEPENDENCY.json"
OFFICIAL_WHEEL_PATH = (
    ARTIFACT_ROOT
    / "dependencies/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
OVERLAY_ROOT = ARTIFACT_ROOT / "dependencies/python-overlay"
OVERLAY_RECORD_PATH = OVERLAY_ROOT / "mamba_ssm-2.3.2.post1.dist-info/RECORD"
MAMBA2_FILE_PATH = OVERLAY_ROOT / "mamba_ssm/modules/mamba2.py"
SELECTIVE_SCAN_EXTENSION_PATH = OVERLAY_ROOT / "selective_scan_cuda.cpython-312-x86_64-linux-gnu.so"
INPUT_BYTES = 5_632_357
INPUT_SHA256 = "c7b420aa2ee549b4163a3fc2a1f7335c601e2b769f6eaa16d0ff629d130b6797"
INPUT_FIELD = "codec_v1"
EXPECTED_ARCHIVE_FIELDS = (
    "codec_v0",
    "codec_v1",
    "decoded_rgb_teacher",
    "reconstruction_v0",
    "reconstruction_v1",
)
INPUT_SHAPE = (925, 1, GRID_HEIGHT, GRID_WIDTH, CODEC_FIELDS)
LENGTHS = (128, 256, 512, 925)
CHUNK_SIZE = 127
MAX_WALL_SECONDS = 30 * 60
MAX_ARTIFACT_BYTES = 1 << 30
MAX_PARITY_ABS = 1e-4
MAX_CAUSAL_ABS = 1e-5
MAX_PARAMETER_GAP = 0.15
MAX_STATE_GROWTH = 0.01
MAX_ALLOCATED_GROWTH_BYTES = 1 << 20
MAX_MAMBA_TRANSFORMER_PEAK_RATIO = 0.60
MAX_STREAM_STEP_P95_MS = 5.0
SOURCE_IDENTITY_PATHS = (
    "gr00t/__init__.py",
    "gr00t/experimental/__init__.py",
    "gr00t/experimental/arena_codec_streaming_backbone.py",
    "examples/G1_ARENA_N17/codec_streaming_backbone_probe.py",
    "tests/gr00t/experimental/test_arena_codec_streaming_backbone.py",
    "tests/gr00t/experimental/test_arena_codec_streaming_mamba_target.py",
    "tests/examples/test_g1_arena_codec_streaming_backbone_probe.py",
)
NATIVE_VLA_MODULE_PREFIXES = (
    "gr00t.model.gr00t_n1d7",
    "gr00t.model.modules.dit",
    "gr00t.model.modules.qwen3_backbone",
    "gr00t.policy",
)
FROZEN_RUNTIME = {
    "python_executable": "/home/sl/research/icra27-worktrees/Isaac-GR00T/main/.venv/bin/python",
    "python": "3.12.13",
    "torch": "2.9.0+cu128",
    "torch_cuda": "12.8",
    "cxx11_abi": True,
    "mamba_ssm": "2.3.2.post1",
    "cuda_visible_devices": "1",
    "pythonpath": f"{OVERLAY_ROOT}:{REPO}",
    "python_dont_write_bytecode": True,
    "gpu_name": "NVIDIA RTX 6000 Ada Generation",
    "cost_dtype": "bfloat16",
}
FROZEN_PROTOCOL = {
    "lengths": list(LENGTHS),
    "seed": SEED,
    "layers": 2,
    "width": WIDTH,
    "chunk_size": CHUNK_SIZE,
    "warmup": 20,
    "measurements": 50,
    "spatial_tokenizer": "conv3x3_s2_silu_x2_2x2_flatten_linear_d256",
    "temporal_backbones": ["mamba2", "gru", "transformer", "identity"],
    "optimizer_steps": 0,
    "vla_model_instantiated": False,
}
FROZEN_THRESHOLDS = {
    "full_step_chunk_max_abs": MAX_PARITY_ABS,
    "suffix_prefix_max_abs": MAX_CAUSAL_ABS,
    "reset_exact": True,
    "mamba_state_serialized_growth_max_fraction": MAX_STATE_GROWTH,
    "mamba_allocated_second_half_growth_max_bytes": MAX_ALLOCATED_GROWTH_BYTES,
    "temporal_parameter_gap_max_fraction": MAX_PARAMETER_GAP,
    "mamba_t925_peak_allocated_max_transformer_fraction": MAX_MAMBA_TRANSFORMER_PEAK_RATIO,
    "transformer_mamba_memory_ratio_nondecreasing": True,
    "mamba_stream_step_host_p95_max_ms": MAX_STREAM_STEP_P95_MS,
}
FROZEN_RESOURCES = {
    "gpu_physical_id": 1,
    "timeout_seconds": MAX_WALL_SECONDS,
    "storage_limit_bytes": MAX_ARTIFACT_BYTES,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def native_vla_modules_loaded() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in NATIVE_VLA_MODULE_PREFIXES
        )
    )


@contextmanager
def optimizer_construction_guard() -> Iterator[dict[str, Any]]:
    optimizer = torch.optim.Optimizer
    original_init = optimizer.__init__
    evidence: dict[str, Any] = {
        "constructor_calls": [],
        "steps_observed": None,
        "guard": "torch.optim.Optimizer.__init__ blocked for the measured probe",
    }

    def blocked_init(instance, *args, **kwargs):
        del args, kwargs
        evidence["constructor_calls"].append(type(instance).__qualname__)
        raise RuntimeError("optimizer construction is forbidden in this untrained probe")

    optimizer.__init__ = blocked_init
    try:
        yield evidence
    finally:
        optimizer.__init__ = original_init
        evidence["steps_observed"] = 0 if not evidence["constructor_calls"] else None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_material_spec(
    spec: dict[str, Any],
    *,
    expected_path: Path,
    label: str,
) -> Path:
    if set(spec) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} material spec keys changed")
    path = Path(spec["path"]).resolve()
    if path != expected_path.resolve():
        raise ValueError(f"{label} path changed")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if path.stat().st_size != int(spec["bytes"]) or sha256(path) != spec["sha256"]:
        raise ValueError(f"{label} identity changed")
    return path


def _record_sha256(path: Path) -> str:
    digest = bytes.fromhex(sha256(path))
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def verify_overlay_record(record_path: Path, overlay_root: Path) -> dict[str, Any]:
    root = overlay_root.resolve()
    record = record_path.resolve()
    if not record.is_file() or not record.is_relative_to(root):
        raise ValueError("overlay RECORD is missing or outside the overlay")
    recorded: set[str] = set()
    verified_bytes = 0
    with record.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 3:
                raise ValueError("overlay RECORD row schema changed")
            relative_text, expected_hash, expected_size = row
            relative = PurePosixPath(relative_text)
            if relative.is_absolute() or ".." in relative.parts or relative_text in recorded:
                raise ValueError("overlay RECORD contains an unsafe or duplicate path")
            material = root.joinpath(*relative.parts).resolve()
            if not material.is_relative_to(root) or not material.is_file():
                raise ValueError(f"overlay RECORD material is missing: {relative_text}")
            recorded.add(relative_text)
            if not expected_hash and not expected_size:
                if material != record:
                    raise ValueError("only overlay RECORD may omit its own hash and size")
                continue
            if not expected_hash.startswith("sha256=") or not expected_size:
                raise ValueError("overlay RECORD hash or size contract changed")
            size = int(expected_size)
            if material.stat().st_size != size:
                raise ValueError(f"overlay RECORD size mismatch: {relative_text}")
            if _record_sha256(material) != expected_hash.removeprefix("sha256="):
                raise ValueError(f"overlay RECORD hash mismatch: {relative_text}")
            verified_bytes += size

    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    extras = actual - recorded
    allowed_extras = {relative for relative in extras if relative == ".lock"}
    unexpected = extras - allowed_extras
    if unexpected:
        raise ValueError(f"overlay contains files outside RECORD: {sorted(unexpected)}")
    return {
        "root": str(root),
        "record": str(record),
        "record_entries": len(recorded),
        "verified_hashed_files": len(recorded) - 1,
        "verified_hashed_bytes": verified_bytes,
        "allowed_unrecorded_files": sorted(allowed_extras),
        "unexpected_unrecorded_files": [],
    }


def verify_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("resolved plan must be a file inside the experiment artifact root")
    actual_sha256 = sha256(resolved)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ValueError("resolved plan identity changed")
    plan = read_json(resolved)
    if set(plan) != {
        "schema_version",
        "experiment_id",
        "run_id",
        "source",
        "input",
        "dependency",
        "runtime",
        "protocol",
        "thresholds",
        "resources",
        "output",
    }:
        raise ValueError("resolved plan top-level schema changed")
    if plan["schema_version"] != 1 or plan["experiment_id"] != EXPERIMENT:
        raise ValueError("wrong resolved plan schema or experiment")
    if plan["run_id"] != RUN_ID:
        raise ValueError("wrong resolved plan run id")

    source = plan["source"]
    if set(source) != {"root", "commit", "base_commit"}:
        raise ValueError("resolved source schema changed")
    if Path(source["root"]).resolve() != REPO.resolve():
        raise ValueError("resolved source root changed")
    try:
        int(source["commit"], 16)
    except (TypeError, ValueError) as error:
        raise ValueError("resolved source commit is not a Git object id") from error
    if source["base_commit"] != BASE_SOURCE_COMMIT or len(source["commit"]) != 40:
        raise ValueError("resolved source commit contract changed")

    expected_input = {
        "path": str(INPUT_PATH),
        "bytes": INPUT_BYTES,
        "sha256": INPUT_SHA256,
        "field": INPUT_FIELD,
        "shape": list(INPUT_SHAPE),
        "dtype": "float32",
    }
    if plan["input"] != expected_input:
        raise ValueError("resolved input contract changed")

    dependency = plan["dependency"]
    if set(dependency) != {
        "manifest",
        "official_wheel",
        "overlay_record",
        "mamba_file",
        "selective_scan_extension",
    }:
        raise ValueError("resolved dependency schema changed")
    manifest_path = _verify_material_spec(
        dependency["manifest"],
        expected_path=DEPENDENCY_MANIFEST_PATH,
        label="dependency manifest",
    )
    mamba_file = _verify_material_spec(
        dependency["mamba_file"],
        expected_path=MAMBA2_FILE_PATH,
        label="Mamba2 source",
    )
    wheel_path = _verify_material_spec(
        dependency["official_wheel"],
        expected_path=OFFICIAL_WHEEL_PATH,
        label="official Mamba wheel",
    )
    record_path = _verify_material_spec(
        dependency["overlay_record"],
        expected_path=OVERLAY_RECORD_PATH,
        label="overlay RECORD",
    )
    extension_path = _verify_material_spec(
        dependency["selective_scan_extension"],
        expected_path=SELECTIVE_SCAN_EXTENSION_PATH,
        label="selective_scan CUDA extension",
    )
    closure = verify_overlay_record(record_path, OVERLAY_ROOT)
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != EXPERIMENT
        or manifest.get("overlay", {}).get("mamba_version") != "2.3.2.post1"
        or manifest.get("overlay", {}).get("causal_conv1d_present") is not False
        or Path(manifest.get("overlay", {}).get("mamba2_file", "")).resolve() != mamba_file
        or manifest.get("overlay", {}).get("mamba2_sha256") != dependency["mamba_file"]["sha256"]
        or Path(manifest.get("overlay", {}).get("path", "")).resolve() != OVERLAY_ROOT.resolve()
        or Path(manifest.get("wheel", {}).get("path", "")).resolve() != wheel_path
        or manifest.get("wheel", {}).get("bytes") != dependency["official_wheel"]["bytes"]
        or manifest.get("wheel", {}).get("sha256") != dependency["official_wheel"]["sha256"]
        or manifest.get("runtime", {}).get("python_version") != FROZEN_RUNTIME["python"]
        or manifest.get("runtime", {}).get("torch") != FROZEN_RUNTIME["torch"]
        or manifest.get("runtime", {}).get("torch_cuda") != FROZEN_RUNTIME["torch_cuda"]
        or manifest.get("runtime", {}).get("cxx11_abi") is not FROZEN_RUNTIME["cxx11_abi"]
        or extension_path.relative_to(OVERLAY_ROOT).as_posix()
        not in {
            row.split(",", 1)[0] for row in record_path.read_text(encoding="utf-8").splitlines()
        }
    ):
        raise ValueError("dependency manifest runtime or overlay contract changed")

    if plan["runtime"] != FROZEN_RUNTIME:
        raise ValueError("resolved runtime contract changed")
    if plan["protocol"] != FROZEN_PROTOCOL:
        raise ValueError("resolved protocol changed")
    if plan["thresholds"] != FROZEN_THRESHOLDS:
        raise ValueError("resolved thresholds changed")
    if plan["resources"] != FROZEN_RESOURCES:
        raise ValueError("resolved resources changed")
    expected_output = {
        "root": str(ARTIFACT_ROOT / RUN_ID),
        "result_file": "RESULT.json",
    }
    if plan["output"] != expected_output:
        raise ValueError("resolved output contract changed")
    return plan, {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
        "dependency_manifest_sha256": dependency["manifest"]["sha256"],
        "official_wheel_sha256": dependency["official_wheel"]["sha256"],
        "overlay_record_sha256": dependency["overlay_record"]["sha256"],
        "mamba_file_sha256": dependency["mamba_file"]["sha256"],
        "selective_scan_extension_sha256": dependency["selective_scan_extension"]["sha256"],
        "overlay_closure": closure,
    }


def _source_file_identity(path: Path, commit: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(REPO.resolve()):
        raise ValueError(f"imported source escaped repository: {resolved}")
    relative = resolved.relative_to(REPO.resolve()).as_posix()
    committed_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", f"{commit}:{relative}"],
        text=True,
    ).strip()
    working_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "hash-object", str(resolved)],
        text=True,
    ).strip()
    if working_blob != committed_blob:
        raise ValueError(f"imported source differs from committed blob: {relative}")
    return {
        "path": str(resolved),
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
        "git_blob": working_blob,
    }


def verify_imported_source_identity(commit: str) -> dict[str, Any]:
    expected_arena_module = (
        REPO / "gr00t/experimental/arena_codec_streaming_backbone.py"
    ).resolve()
    actual_arena_module = Path(arena_codec_module.__file__).resolve()
    if actual_arena_module != expected_arena_module:
        raise ValueError("imported Arena codec module escaped the current source root")
    entries = {
        "arena_codec_module": _source_file_identity(actual_arena_module, commit),
        "gr00t_package": _source_file_identity(Path(gr00t.__file__), commit),
        "experimental_package": _source_file_identity(Path(gr00t.experimental.__file__), commit),
        "runner": _source_file_identity(Path(__file__), commit),
    }
    return {"root": str(REPO.resolve()), "commit": commit, "entries": entries}


def verify_source(expected_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != expected_commit:
        raise ValueError("Arena codec backbone source commit changed")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "merge-base",
            "--is-ancestor",
            BASE_SOURCE_COMMIT,
            commit,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("Arena codec backbone source is not based on frozen N1.7 contract")
    dirty: set[str] = set()
    for extra in ((), ("--cached",)):
        changed = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "diff",
                *extra,
                "--name-only",
                "--",
                *SOURCE_IDENTITY_PATHS,
            ],
            text=True,
        )
        dirty.update(changed.splitlines())
    untracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_IDENTITY_PATHS,
        ],
        text=True,
    )
    dirty.update(untracked.splitlines())
    if dirty:
        raise ValueError(f"Arena codec backbone source is dirty: {sorted(dirty)}")
    return verify_imported_source_identity(commit)


def load_codec_input(
    path: Path,
    *,
    expected_bytes: int = INPUT_BYTES,
    expected_sha256: str = INPUT_SHA256,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Arena Codec IR input is missing: {resolved}")
    actual_bytes = resolved.stat().st_size
    actual_sha256 = sha256(resolved)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError("Arena Codec IR input identity changed")
    with np.load(resolved, allow_pickle=False) as archive:
        present = tuple(archive.files)
        if set(present) != set(EXPECTED_ARCHIVE_FIELDS) or len(present) != len(
            EXPECTED_ARCHIVE_FIELDS
        ):
            raise ValueError("Arena Codec IR archive field whitelist changed")
        codec = np.array(archive[INPUT_FIELD], copy=True)
    if codec.shape != INPUT_SHAPE:
        raise ValueError(f"expected codec_v1 shape {INPUT_SHAPE}, got {codec.shape}")
    if codec.dtype != np.float32:
        raise ValueError(f"expected codec_v1 float32, got {codec.dtype}")
    if not np.isfinite(codec).all():
        raise ValueError("codec_v1 contains nonfinite values")
    metadata = {
        "path": str(resolved),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "fields_present": list(present),
        "fields_consumed_by_backbone": [INPUT_FIELD],
        "fields_excluded_from_backbone": sorted(set(present) - {INPUT_FIELD}),
        "shape": list(codec.shape),
        "dtype": str(codec.dtype),
    }
    return codec, metadata


def verify_environment(
    expected_mamba_file: Path,
    expected_extension_file: Path,
) -> dict[str, Any]:
    import mamba_ssm
    import mamba_ssm.modules.mamba2 as official
    import mamba_ssm.ops.triton.ssd_combined as ssd_combined
    import selective_scan_cuda

    mamba_version = importlib.metadata.version("mamba-ssm")
    actual_mamba_file = Path(official.__file__).resolve()
    actual_extension_file = Path(selective_scan_cuda.__file__).resolve()
    actual_ssd_file = Path(ssd_combined.__file__).resolve()
    actual_package_file = Path(mamba_ssm.__file__).resolve()
    if official.selective_state_update is None:
        raise ValueError("Mamba2 persistent step op is unavailable")
    state_update_module = sys.modules[official.selective_state_update.__module__]
    actual_state_update_file = Path(state_update_module.__file__).resolve()
    if Path(sys.executable).resolve() != Path(FROZEN_RUNTIME["python_executable"]).resolve():
        raise ValueError("formal Python executable changed")
    if platform.python_version() != FROZEN_RUNTIME["python"]:
        raise ValueError("expected frozen Python 3.12.13 target runtime")
    if torch.__version__ != FROZEN_RUNTIME["torch"]:
        raise ValueError("expected frozen Torch 2.9.0+cu128 target runtime")
    if torch.version.cuda != FROZEN_RUNTIME["torch_cuda"]:
        raise ValueError("expected frozen CUDA 12.8 runtime")
    if torch.compiled_with_cxx11_abi() is not FROZEN_RUNTIME["cxx11_abi"]:
        raise ValueError("Torch CXX11 ABI contract changed")
    if mamba_version != "2.3.2.post1":
        raise ValueError("expected official mamba-ssm 2.3.2.post1")
    if MAMBA2_CONFIG["use_mem_eff_path"] is not False:
        raise ValueError("frozen Mamba2 requires the non-fused explicit path")
    if actual_mamba_file != expected_mamba_file.resolve():
        raise ValueError("Mamba2 import escaped the resolved official runtime overlay")
    if actual_extension_file != expected_extension_file.resolve():
        raise ValueError("selective_scan extension escaped the resolved official runtime overlay")
    for path in (actual_package_file, actual_ssd_file, actual_state_update_file):
        if not path.is_relative_to(OVERLAY_ROOT.resolve()):
            raise ValueError(f"critical Mamba import escaped the overlay: {path}")
    if official.causal_conv1d_fn is not None or official.causal_conv1d_update is not None:
        raise ValueError("frozen explicit Mamba path requires causal-conv1d to be absent")
    if official.mamba_chunk_scan_combined.__module__ != ssd_combined.__name__:
        raise ValueError("Mamba2 full scan op escaped the verified SSD implementation")
    if os.environ.get("PYTHONPATH") != FROZEN_RUNTIME["pythonpath"]:
        raise ValueError("formal PYTHONPATH contract changed")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ValueError("formal probe must disable unrecorded Python bytecode writes")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("formal probe must expose only physical GPU1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("formal probe requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("formal GPU does not support BF16")
    if torch.cuda.get_device_name(0) != FROZEN_RUNTIME["gpu_name"]:
        raise ValueError("formal GPU identity changed")
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "python_dont_write_bytecode": sys.dont_write_bytecode,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mamba_ssm": mamba_version,
        "mamba_file": str(actual_mamba_file),
        "mamba_package_file": str(actual_package_file),
        "ssd_combined_file": str(actual_ssd_file),
        "selective_state_update_file": str(actual_state_update_file),
        "selective_scan_extension_file": str(actual_extension_file),
        "causal_conv1d_fn": None,
        "causal_conv1d_update": None,
        "use_mem_eff_path": MAMBA2_CONFIG["use_mem_eff_path"],
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
    }


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max().item())


def _finite_shape(value: torch.Tensor, length: int) -> bool:
    return list(value.shape) == [1, length, WIDTH] and bool(torch.isfinite(value).all())


@torch.inference_mode()
def _chunked_stream(
    module: CausalMamba2Backbone | CausalGRUBackbone,
    descriptors: torch.Tensor,
) -> torch.Tensor:
    state = module.new_state(int(descriptors.shape[0]))
    outputs = []
    for start in range(0, int(descriptors.shape[1]), CHUNK_SIZE):
        current, state = module.stream(descriptors[:, start : start + CHUNK_SIZE], state)
        outputs.append(current)
    return torch.cat(outputs, dim=1)


@torch.inference_mode()
def numerical_probe(codec: np.ndarray) -> tuple[dict[str, Any], dict[str, int]]:
    device = torch.device("cuda", 0)
    raw = torch.from_numpy(codec).unsqueeze(0).to(device=device, dtype=torch.float32)
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device)
    backbones: dict[str, torch.nn.Module] = {
        "mamba2": CausalMamba2Backbone(seed=SEED).eval().to(device),
        "gru": CausalGRUBackbone(seed=SEED).eval().to(device),
        "transformer": CausalTransformerBackbone(seed=SEED).eval().to(device),
        "identity": CurrentDescriptorBackbone().eval().to(device),
    }
    descriptors = tokenizer(raw)
    if not _finite_shape(descriptors, INPUT_SHAPE[0]):
        raise RuntimeError("shared tokenizer output is invalid")

    outputs: dict[str, torch.Tensor] = {}
    output_reports: dict[str, Any] = {}
    for name, module in backbones.items():
        output = module(descriptors)
        outputs[name] = output
        output_reports[name] = {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(torch.isfinite(output).all()),
        }

    parity: dict[str, Any] = {}
    reset: dict[str, Any] = {}
    for name in ("mamba2", "gru"):
        module = backbones[name]
        streamed, state = module.stream(descriptors)
        chunked = _chunked_stream(module, descriptors)
        fresh_reset, _ = module.stream(descriptors[:, : LENGTHS[0]])
        state.reset()
        first_reset, state = module.stream(descriptors[:, : LENGTHS[0]], state)
        state.reset()
        second_reset, state = module.stream(descriptors[:, : LENGTHS[0]], state)
        parity[name] = {
            "full_vs_step_max_abs": _max_abs(outputs[name], streamed),
            "full_vs_chunked_max_abs": _max_abs(outputs[name], chunked),
            "chunk_size": CHUNK_SIZE,
        }
        reset[name] = {
            "exact": bool(
                torch.equal(fresh_reset, first_reset) and torch.equal(first_reset, second_reset)
            ),
            "fresh_vs_reset_max_abs": _max_abs(fresh_reset, first_reset),
            "max_abs": _max_abs(first_reset, second_reset),
            "observations_after_second_episode": state.observations,
        }

    causal_raw = raw[:, : LENGTHS[1]].clone()
    causal_changed = causal_raw.clone()
    causal_changed[:, LENGTHS[0] :] = causal_changed[:, LENGTHS[0] :] + 0.125
    original_descriptors = tokenizer(causal_raw)
    changed_descriptors = tokenizer(causal_changed)
    causality = {}
    for name, module in backbones.items():
        original = module(original_descriptors)[:, : LENGTHS[0]]
        changed = module(changed_descriptors)[:, : LENGTHS[0]]
        causality[name] = {"suffix_edit_prefix_max_abs": _max_abs(original, changed)}

    mamba = backbones["mamba2"]
    state = mamba.new_state(1)
    allocated = []
    state_at_128: dict[str, Any] | None = None
    for index, descriptor in enumerate(descriptors.unbind(dim=1), start=1):
        current = mamba.step(descriptor, state)
        del current
        torch.cuda.synchronize(device)
        allocated.append(int(torch.cuda.memory_allocated(device)))
        if index == LENGTHS[0]:
            state_at_128 = state.describe()
    state_at_925 = state.describe()
    if state_at_128 is None:
        raise RuntimeError("failed to capture Mamba2 T128 state")
    serialized_growth = (
        int(state_at_925["serialized_bytes"]) - int(state_at_128["serialized_bytes"])
    ) / int(state_at_128["serialized_bytes"])
    second_half = allocated[len(allocated) // 2 :]
    allocated_growth = max(second_half) - min(second_half)
    state_report = {
        "at_128": state_at_128,
        "at_925": state_at_925,
        "serialized_growth_fraction": serialized_growth,
        "allocated_second_half_growth_bytes": allocated_growth,
        "allocated_second_half_min_bytes": min(second_half),
        "allocated_second_half_max_bytes": max(second_half),
        "allocated_at_frame_463_bytes": allocated[462],
        "allocated_at_frame_925_bytes": allocated[924],
    }

    parameters = {name: trainable_parameter_count(module) for name, module in backbones.items()}
    mamba_parameters = parameters["mamba2"]
    parameter_gap = {
        name: abs(parameters[name] - mamba_parameters) / mamba_parameters
        for name in ("gru", "transformer")
    }
    result = {
        "tokenizer": {
            "shape": list(descriptors.shape),
            "dtype": str(descriptors.dtype),
            "finite": bool(torch.isfinite(descriptors).all()),
            "parameters": trainable_parameter_count(tokenizer),
            "camera_reduction": "mean_after_per_camera_2x2_flatten_projection",
        },
        "outputs": output_reports,
        "parity": parity,
        "causality": causality,
        "reset": reset,
        "mamba_state": state_report,
        "parameter_gap_from_mamba": parameter_gap,
    }
    for module in backbones.values():
        module.cpu()
    tokenizer.cpu()
    del raw, descriptors, outputs, backbones, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result, parameters


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _is_cuda_kernel_event(event: Any) -> bool:
    if not str(getattr(event, "device_type", "")).lower().endswith("cuda"):
        return False
    name = str(getattr(event, "name", "")).lower()
    return "memcpy" not in name and "memset" not in name


def _kernel_summary(call: Callable[[], torch.Tensor]) -> dict[str, int]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as profile:
        output = call()
        torch.cuda.synchronize()
        del output
    device_events = [
        event
        for event in profile.events()
        if str(getattr(event, "device_type", "")).lower().endswith("cuda")
    ]
    kernel_events = [event for event in device_events if _is_cuda_kernel_event(event)]
    if not kernel_events:
        raise RuntimeError("CUDA profiler recorded no kernels")
    return {
        "kernel_count": len(kernel_events),
        "cuda_device_event_count": len(device_events),
        "excluded_memcpy_memset_events": len(device_events) - len(kernel_events),
    }


@torch.inference_mode()
def benchmark_full(
    tokenizer: ArenaCodecSpatialTokenizer,
    backbone: torch.nn.Module,
    value: torch.Tensor,
    *,
    warmup: int,
    measurements: int,
) -> dict[str, Any]:
    def call() -> torch.Tensor:
        return backbone(tokenizer(value))

    validation = call()
    if not _finite_shape(validation, int(value.shape[1])):
        raise RuntimeError("BF16 benchmark output is invalid")
    output_dtype = str(validation.dtype)
    del validation
    for _ in range(warmup):
        output = call()
        del output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    samples = []
    host_samples = []
    for _ in range(measurements):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter_ns()
        start.record()
        output = call()
        end.record()
        end.synchronize()
        host_samples.append((time.perf_counter_ns() - host_start) / 1_000_000)
        samples.append(float(start.elapsed_time(end)))
        del output
    peak = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    kernel_summary = _kernel_summary(call)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 95),
        "cuda_event_samples_ms": samples,
        "host_p50_ms": statistics.median(host_samples),
        "host_p95_ms": percentile(host_samples, 95),
        "host_samples_ms": host_samples,
        "output_dtype": output_dtype,
        "output_finite": True,
        "measurements": measurements,
        "warmup": warmup,
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_delta_bytes": peak - baseline,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "kernel_count_measured_separately_from_latency": True,
        **kernel_summary,
    }


@torch.inference_mode()
def benchmark_streaming_step(
    tokenizer: ArenaCodecSpatialTokenizer,
    mamba: CausalMamba2Backbone,
    codec: np.ndarray,
    *,
    warmup_frames: int,
) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    value = torch.from_numpy(codec).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    state = mamba.new_state(1)
    samples = []
    host_samples = []
    all_finite = True
    for index, frame in enumerate(value.unbind(dim=1)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter_ns()
        start.record()
        descriptor = tokenizer.encode_frame(frame)
        output = mamba.step(descriptor, state)
        end.record()
        end.synchronize()
        host_elapsed = (time.perf_counter_ns() - host_start) / 1_000_000
        all_finite = all_finite and bool(torch.isfinite(output).all())
        if index >= warmup_frames:
            samples.append(float(start.elapsed_time(end)))
            host_samples.append(host_elapsed)
        del descriptor, output
    if not samples:
        raise ValueError("streaming warmup consumed all frames")
    profile_state = state.clone()
    profile_frame = value[:, -1]

    def profile_call() -> torch.Tensor:
        return mamba.step(tokenizer.encode_frame(profile_frame), profile_state)

    kernel_summary = _kernel_summary(profile_call)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 95),
        "cuda_event_samples_ms": samples,
        "host_p50_ms": statistics.median(host_samples),
        "host_p95_ms": percentile(host_samples, 95),
        "host_samples_ms": host_samples,
        "measurements": len(samples),
        "warmup_frames": warmup_frames,
        "observations": state.observations,
        "state_tensor_bytes": state.tensor_nbytes(),
        "output_finite": all_finite,
        "kernel_count_measured_separately_from_latency": True,
        **kernel_summary,
    }


def _make_backbone(name: str) -> torch.nn.Module:
    if name == "mamba2":
        return CausalMamba2Backbone(seed=SEED)
    if name == "gru":
        return CausalGRUBackbone(seed=SEED)
    if name == "transformer":
        return CausalTransformerBackbone(seed=SEED)
    if name == "identity":
        return CurrentDescriptorBackbone()
    raise ValueError(f"unknown backbone: {name}")


def cost_probe(
    codec: np.ndarray,
    *,
    warmup: int,
    measurements: int,
) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device=device, dtype=torch.bfloat16)
    result: dict[str, Any] = {}
    for name in ("mamba2", "gru", "transformer", "identity"):
        backbone = _make_backbone(name).eval().to(device=device, dtype=torch.bfloat16)
        rows = {}
        for length in LENGTHS:
            value = (
                torch.from_numpy(codec[:length])
                .unsqueeze(0)
                .to(device=device, dtype=torch.bfloat16)
            )
            rows[str(length)] = benchmark_full(
                tokenizer,
                backbone,
                value,
                warmup=warmup,
                measurements=measurements,
            )
            del value
            gc.collect()
            torch.cuda.empty_cache()
        result[name] = rows
        if name == "mamba2":
            result["mamba2_streaming_step"] = benchmark_streaming_step(
                tokenizer,
                backbone,
                codec,
                warmup_frames=warmup,
            )
        backbone.cpu()
        del backbone
        gc.collect()
        torch.cuda.empty_cache()
    tokenizer.cpu()
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_gates(
    numerical: dict[str, Any],
    parameters: dict[str, int],
    cost: dict[str, Any],
    execution_evidence: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    parity_values = [
        numerical["parity"][name][metric]
        for name in ("mamba2", "gru")
        for metric in ("full_vs_step_max_abs", "full_vs_chunked_max_abs")
    ]
    memory_ratios = {
        str(length): cost["transformer"][str(length)]["peak_allocated_bytes"]
        / cost["mamba2"][str(length)]["peak_allocated_bytes"]
        for length in LENGTHS
    }
    delta_ratios = {
        str(length): cost["transformer"][str(length)]["peak_delta_bytes"]
        / cost["mamba2"][str(length)]["peak_delta_bytes"]
        for length in LENGTHS
    }
    ordered_ratios = [memory_ratios[str(length)] for length in LENGTHS]
    parameter_gaps = [
        abs(parameters[name] - parameters["mamba2"]) / parameters["mamba2"]
        for name in ("gru", "transformer")
    ]
    gates = {
        "all_outputs_shape_finite": all(
            row["shape"] == [1, INPUT_SHAPE[0], WIDTH] and row["finite"]
            for row in numerical["outputs"].values()
        ),
        "mamba_gru_full_step_chunk_parity": max(parity_values) <= MAX_PARITY_ABS,
        "all_suffix_edits_are_causal": max(
            row["suffix_edit_prefix_max_abs"] for row in numerical["causality"].values()
        )
        <= MAX_CAUSAL_ABS,
        "mamba_gru_reset_exact": all(row["exact"] for row in numerical["reset"].values()),
        "mamba_state_serialized_growth": numerical["mamba_state"]["serialized_growth_fraction"]
        <= MAX_STATE_GROWTH,
        "mamba_allocated_second_half_growth": numerical["mamba_state"][
            "allocated_second_half_growth_bytes"
        ]
        <= MAX_ALLOCATED_GROWTH_BYTES,
        "temporal_parameter_match": max(parameter_gaps) <= MAX_PARAMETER_GAP,
        "mamba_t925_peak_at_most_60pct_transformer": cost["mamba2"]["925"]["peak_allocated_bytes"]
        <= MAX_MAMBA_TRANSFORMER_PEAK_RATIO * cost["transformer"]["925"]["peak_allocated_bytes"],
        "transformer_mamba_memory_ratio_nondecreasing": all(
            later >= earlier for earlier, later in zip(ordered_ratios, ordered_ratios[1:])
        ),
        "mamba_stream_step_host_p95_below_5ms": cost["mamba2_streaming_step"]["host_p95_ms"]
        < MAX_STREAM_STEP_P95_MS,
        "native_vla_modules_absent": not execution_evidence["native_vla_modules"]["before"]
        and not execution_evidence["native_vla_modules"]["after"],
        "optimizer_steps_zero": not execution_evidence["optimizer"]["constructor_calls"]
        and execution_evidence["optimizer"]["steps_observed"] == 0,
    }
    derived = {
        "memory_gate_metric": "absolute_peak_allocated_bytes",
        "transformer_over_mamba_peak_allocated_ratio": memory_ratios,
        "supplemental_transformer_over_mamba_peak_delta_ratio": delta_ratios,
        "parameter_gap_from_mamba": {
            name: abs(parameters[name] - parameters["mamba2"]) / parameters["mamba2"]
            for name in ("gru", "transformer")
        },
    }
    return gates, derived


def write_new(path: Path, payload: dict[str, Any]) -> int:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("result artifact exceeds 1 GiB")
    artifact_root = path.parent.parent
    if directory_bytes(artifact_root) + len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("experiment artifact root would exceed 1 GiB")
    with path.open("xb") as stream:
        stream.write(encoded)
    return len(encoded)


def run(plan: dict[str, Any], plan_identity: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    artifact_root = ARTIFACT_ROOT.resolve()
    artifact_bytes_before_result = directory_bytes(artifact_root)
    native_modules_before = native_vla_modules_loaded()
    source_evidence = verify_source(plan["source"]["commit"])
    codec, input_metadata = load_codec_input(Path(plan["input"]["path"]))
    environment = verify_environment(
        Path(plan["dependency"]["mamba_file"]["path"]),
        Path(plan["dependency"]["selective_scan_extension"]["path"]),
    )
    with optimizer_construction_guard() as optimizer_evidence:
        numerical, parameters = numerical_probe(codec)
        cost = cost_probe(
            codec,
            warmup=plan["protocol"]["warmup"],
            measurements=plan["protocol"]["measurements"],
        )
    execution_evidence = {
        "native_vla_modules": {
            "before": native_modules_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
    }
    gates, derived = evaluate_gates(numerical, parameters, cost, execution_evidence)
    wall_seconds = time.monotonic() - started
    gates["wall_time_at_most_30min"] = wall_seconds <= MAX_WALL_SECONDS
    gates["artifact_total_at_most_1g"] = True
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "source_commit": source_evidence["commit"],
        "source": source_evidence,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "plan": plan_identity,
        "seed": SEED,
        "input": input_metadata,
        "environment": environment,
        "lengths": list(LENGTHS),
        "parameters": parameters,
        "parameter_accounting": {
            "temporal_only": parameters,
            "shared_tokenizer": numerical["tokenizer"]["parameters"],
            "tokenizer_plus_temporal": {
                name: numerical["tokenizer"]["parameters"] + count
                for name, count in parameters.items()
            },
        },
        "instantiated_modules": {
            "tokenizer": "ArenaCodecSpatialTokenizer",
            "temporal": [
                "CausalMamba2Backbone",
                "CausalGRUBackbone",
                "CausalTransformerBackbone",
                "CurrentDescriptorBackbone",
            ],
            "native_vla": sorted(
                set(execution_evidence["native_vla_modules"]["before"])
                | set(execution_evidence["native_vla_modules"]["after"])
            ),
        },
        "execution_evidence": execution_evidence,
        "numerical": numerical,
        "cost_bf16": cost,
        "derived": derived,
        "gates": gates,
        "optimizer_steps": optimizer_evidence["steps_observed"],
        "vla_model_instantiated": bool(
            execution_evidence["native_vla_modules"]["before"]
            or execution_evidence["native_vla_modules"]["after"]
        ),
        "training_allowed": all(gates.values()),
        "wall_seconds": wall_seconds,
        "claim_boundary": "untrained VLA-free codec tokenizer/backbone engineering and memory scaling only",
    }
    for _ in range(3):
        result_bytes = len((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        total_bytes = artifact_bytes_before_result + result_bytes
        result["artifact"] = {
            "root": str(artifact_root),
            "bytes_before_result": artifact_bytes_before_result,
            "result_bytes": result_bytes,
            "total_bytes": total_bytes,
            "limit_bytes": MAX_ARTIFACT_BYTES,
            "chrome_traces_written": 0,
        }
        result["gates"]["artifact_total_at_most_1g"] = total_bytes <= MAX_ARTIFACT_BYTES
        result["status"] = "passed" if all(result["gates"].values()) else "gate_failed"
        result["training_allowed"] = all(result["gates"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    plan, plan_identity = verify_plan(args.plan, args.expected_plan_sha256)
    output = Path(plan["output"]["root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    result_path = output / plan["output"]["result_file"]
    try:
        result = run(plan, plan_identity)
        write_new(result_path, result)
    except Exception as error:
        write_new(
            result_path,
            {
                "status": "failed",
                "experiment_id": EXPERIMENT,
                "plan": plan_identity,
                "error": f"{type(error).__name__}: {error}",
                "optimizer_evidence": "unavailable because the probe raised before completion",
                "native_vla_evidence": native_vla_modules_loaded(),
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "wall_seconds": result["wall_seconds"],
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
