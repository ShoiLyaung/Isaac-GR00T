#!/usr/bin/env python3
"""Run the frozen Arena functional Mamba2 correctness and latency protocol."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterator

import gr00t
import gr00t.experimental
from gr00t.experimental.arena_codec_streaming_backbone import (
    CODEC_FIELDS,
    GRID_HEIGHT,
    GRID_WIDTH,
    LAYERS,
    SEED,
    WIDTH,
    ArenaCodecSpatialTokenizer,
)
import gr00t.experimental.arena_functional_mamba_transition as functional_module
from gr00t.experimental.arena_functional_mamba_transition import (
    FunctionalMamba2Backbone,
    FunctionalMamba2State,
)
import numpy as np
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "g1-arena-functional-mamba-transition-r1"
BASE_SOURCE_COMMIT = "9d29f636fbcffbcf5a2b1a26f9fa4df9cbec65f3"
ARTIFACT_ROOT = Path("/home/sl/research/icra27-artifacts/g1-arena-functional-mamba-transition-r1")
CORRECTNESS_RUN_ID = "functional-correctness-gpu0-r1"
LATENCY_RUN_IDS = {
    1: "functional-latency-gpu0-r1a",
    2: "functional-latency-gpu0-r1b",
    3: "functional-latency-gpu0-r1c",
}
AGGREGATE_RUN_ID = "aggregate-r1"

STREAMING_ARTIFACT_ROOT = Path(
    "/home/sl/research/icra27-artifacts/g1-arena-codec-streaming-backbone-r1"
)
PARITY_ARTIFACT_ROOT = Path(
    "/home/sl/research/icra27-artifacts/g1-arena-codec-streaming-parity-diagnosis-r1"
)
STREAMING_RESULT_PATH = STREAMING_ARTIFACT_ROOT / "untrained-gpu1-r1/RESULT.json"
PARITY_RESULT_PATH = PARITY_ARTIFACT_ROOT / "fixed-weight-gpu0-r3/RESULT.json"
FROZEN_PRIOR_EVIDENCE = {
    "streaming_r1": {
        "path": str(STREAMING_RESULT_PATH),
        "bytes": 106_681,
        "sha256": "8957277625c168afd8b229b5c641702cfae97cb29a3fc6a3ca489b26336dbcd6",
    },
    "parity_diagnosis_r3": {
        "path": str(PARITY_RESULT_PATH),
        "bytes": 682_693,
        "sha256": "8e4cd391ce6f606a767b2a0f40f09ae00ce3007d13a88d9d4c2089e72392cb4f",
    },
}

INPUT_PATH = Path(
    "/home/sl/research/icra27-artifacts/g1-arena-native-first-codec-ir-r1/"
    "episode45-native-first-r1/episode_000045.native_first_codec_ir_v1.npz"
)
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

DEPENDENCY_MANIFEST_PATH = STREAMING_ARTIFACT_ROOT / "dependencies/DEPENDENCY.json"
OFFICIAL_WHEEL_PATH = (
    STREAMING_ARTIFACT_ROOT
    / "dependencies/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
OVERLAY_ROOT = STREAMING_ARTIFACT_ROOT / "dependencies/python-overlay"
OVERLAY_RECORD_PATH = OVERLAY_ROOT / "mamba_ssm-2.3.2.post1.dist-info/RECORD"
MAMBA2_FILE_PATH = OVERLAY_ROOT / "mamba_ssm/modules/mamba2.py"
SELECTIVE_SCAN_EXTENSION_PATH = OVERLAY_ROOT / "selective_scan_cuda.cpython-312-x86_64-linux-gnu.so"
FROZEN_DEPENDENCY = {
    "manifest": {
        "path": str(DEPENDENCY_MANIFEST_PATH),
        "bytes": 2_932,
        "sha256": "559ec49f9ec564e2e4cc2d7045ff3d8d42e614bc53bcae2ddf9d47120629dece",
    },
    "official_wheel": {
        "path": str(OFFICIAL_WHEEL_PATH),
        "bytes": 322_246_976,
        "sha256": "8fd21085d668a3bdbb1f7903c81f64434d7ed2c763240f8328a465a42370a7b6",
    },
    "overlay_record": {
        "path": str(OVERLAY_RECORD_PATH),
        "bytes": 5_974,
        "sha256": "c6aedc73d29161fdb121caae8ec793617bfd05ed48992706f7981a0df19b4d4d",
    },
    "mamba_file": {
        "path": str(MAMBA2_FILE_PATH),
        "bytes": 17_479,
        "sha256": "605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707",
    },
    "selective_scan_extension": {
        "path": str(SELECTIVE_SCAN_EXTENSION_PATH),
        "bytes": 419_076_688,
        "sha256": "e9489a6a560600d87f540636a46bf7f0ac86b69b19b5231eb838c5de06166de6",
    },
}
FROZEN_RUNTIME = {
    "python_executable": "/home/sl/research/icra27-worktrees/Isaac-GR00T/main/.venv/bin/python",
    "python": "3.12.13",
    "torch": "2.9.0+cu128",
    "torch_cuda": "12.8",
    "cxx11_abi": True,
    "mamba_ssm": "2.3.2.post1",
    "cuda_visible_devices": "0",
    "pythonpath": f"{OVERLAY_ROOT}:{REPO}",
    "python_dont_write_bytecode": True,
    "gpu_name": "NVIDIA RTX 6000 Ada Generation",
    "dtype": "float32",
    "autocast": False,
    "matmul_tf32": False,
    "cudnn_tf32": False,
}

CORRECTNESS_CHUNKS = (1, 11, 37, 128, 7, 64, 256, 421)
GRADIENT_CHUNKS = (1, 11, 37, 79)
GRADIENT_LENGTH = 128
CAUSAL_PREFIX_LENGTH = 128
FINITE_DIFFERENCE_EPSILON = 1e-2
PROBE_SHA256 = "fe909ce41aa44ea9cf8b884e446351e0ff4a5b839c7a506008015aabeeccc06b"
MAX_PARITY_ABS = 1e-4
MAX_GRADIENT_ABS = 1e-5
MIN_FRAME1_GRADIENT_NORM = 1e-8
MAX_STATE_GROWTH = 0.01
MAX_HOST_P95_MS = 5.0
LATENCY_WARMUP_FRAMES = 20
LATENCY_MEASURED_FRAMES = 905
MAX_WALL_SECONDS = 10 * 60
MAX_ARTIFACT_BYTES = 256 << 20

EXPECTED_TOKENIZER_PARAMETERS = (
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "projection.weight",
    "projection.bias",
)
EXPECTED_LAYER_PARAMETERS = (
    "norm.weight",
    "norm.bias",
    "mamba.A_log",
    "mamba.D",
    "mamba.conv1d.weight",
    "mamba.conv1d.bias",
    "mamba.dt_bias",
    "mamba.in_proj.weight",
    "mamba.norm.weight",
    "mamba.out_proj.weight",
)
EXPECTED_TEMPORAL_PARAMETERS = tuple(
    f"layers.{layer}.{name}" for layer in range(LAYERS) for name in EXPECTED_LAYER_PARAMETERS
)
NATIVE_VLA_MODULE_PREFIXES = (
    "gr00t.model.gr00t_n1d7",
    "gr00t.model.modules.dit",
    "gr00t.model.modules.qwen3_backbone",
    "gr00t.policy",
)
SOURCE_IDENTITY_PATHS = (
    "gr00t/__init__.py",
    "gr00t/experimental/__init__.py",
    "gr00t/experimental/arena_codec_streaming_backbone.py",
    "gr00t/experimental/arena_functional_mamba_transition.py",
    "examples/G1_ARENA_N17/functional_mamba_transition.py",
    "tests/gr00t/experimental/test_arena_functional_mamba_transition.py",
    "tests/examples/test_g1_arena_functional_mamba_transition.py",
)

FROZEN_PROTOCOL = {
    "seed": SEED,
    "dtype": "float32",
    "batch_size": 1,
    "frames": INPUT_SHAPE[0],
    "layers": LAYERS,
    "width": WIDTH,
    "state": {"conv": [1, 640, 4], "ssm": [1, 8, 64, 64]},
    "functional_path": "ordinary_pytorch_immutable_step",
    "teacher_path": "official_Mamba2.step",
    "official_full_path": "official_Mamba2.forward_chunk16_report_only",
    "correctness_chunks": list(CORRECTNESS_CHUNKS),
    "gradient_length": GRADIENT_LENGTH,
    "gradient_chunks": list(GRADIENT_CHUNKS),
    "finite_difference_epsilon": FINITE_DIFFERENCE_EPSILON,
    "probe_sha256": PROBE_SHA256,
    "latency": {
        "independent_processes": 3,
        "warmup_frames": LATENCY_WARMUP_FRAMES,
        "measured_frames": LATENCY_MEASURED_FRAMES,
        "boundary": "resident_input_encode_frame_plus_two_functional_layers_plus_new_state",
    },
    "optimizer_steps": 0,
    "vla_model_instantiated": False,
}
FROZEN_THRESHOLDS = {
    "teacher_functional_max_abs": MAX_PARITY_ABS,
    "continuous_chunk_gradient_max_abs": MAX_GRADIENT_ABS,
    "frame1_descriptor_gradient_norm_min": MIN_FRAME1_GRADIENT_NORM,
    "finite_difference_abs_plus_relative": "1e-5 + 0.05 * max(abs(analytic), abs(fd))",
    "state_logical_growth_max_fraction": MAX_STATE_GROWTH,
    "latency_host_p95_strict_ms": MAX_HOST_P95_MS,
    "wall_seconds_max": MAX_WALL_SECONDS,
    "artifact_bytes_max": MAX_ARTIFACT_BYTES,
}
FROZEN_RESOURCES = {
    "gpu_physical_id": 0,
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
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, payload: dict[str, Any]) -> int:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("JSON artifact exceeds 256 MiB")
    if directory_bytes(ARTIFACT_ROOT) + len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("experiment artifact root would exceed 256 MiB")
    with path.open("xb") as stream:
        stream.write(encoded)
    return len(encoded)


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
    evidence: dict[str, Any] = {"constructor_calls": [], "steps_observed": None}

    def blocked_init(instance, *args, **kwargs):
        del args, kwargs
        evidence["constructor_calls"].append(type(instance).__qualname__)
        raise RuntimeError("optimizer construction is forbidden in the fixed-weight protocol")

    optimizer.__init__ = blocked_init
    try:
        yield evidence
    finally:
        optimizer.__init__ = original_init
        evidence["steps_observed"] = 0 if not evidence["constructor_calls"] else None


def resolved_plan(source_commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "source": {
            "root": str(REPO),
            "commit": source_commit,
            "base_commit": BASE_SOURCE_COMMIT,
        },
        "prior_evidence": copy.deepcopy(FROZEN_PRIOR_EVIDENCE),
        "input": {
            "path": str(INPUT_PATH),
            "bytes": INPUT_BYTES,
            "sha256": INPUT_SHA256,
            "field": INPUT_FIELD,
            "shape": list(INPUT_SHAPE),
            "dtype": "float32",
        },
        "dependency": copy.deepcopy(FROZEN_DEPENDENCY),
        "runtime": copy.deepcopy(FROZEN_RUNTIME),
        "protocol": copy.deepcopy(FROZEN_PROTOCOL),
        "thresholds": copy.deepcopy(FROZEN_THRESHOLDS),
        "resources": copy.deepcopy(FROZEN_RESOURCES),
        "outputs": {
            "correctness": str(ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json"),
            "latency": {
                str(index): str(ARTIFACT_ROOT / run_id / "RESULT.json")
                for index, run_id in LATENCY_RUN_IDS.items()
            },
            "aggregate": str(ARTIFACT_ROOT / AGGREGATE_RUN_ID / "RESULT.json"),
        },
    }


def prepare_plan(path: Path) -> dict[str, Any]:
    expected = ARTIFACT_ROOT / "plan-r1/PLAN.json"
    if path.resolve() != expected.resolve():
        raise ValueError(f"resolved plan path must be {expected}")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    size = write_new(path, resolved_plan(commit))
    return {"path": str(path.resolve()), "bytes": size, "sha256": sha256(path)}


def verify_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("resolved plan must be inside the experiment artifact root")
    actual_sha256 = sha256(resolved)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ValueError("resolved plan identity changed")
    plan = read_json(resolved)
    commit = plan.get("source", {}).get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("resolved source commit must contain 40 hex characters")
    try:
        int(commit, 16)
    except ValueError as error:
        raise ValueError("resolved source commit is not hexadecimal") from error
    if plan != resolved_plan(commit):
        raise ValueError("resolved plan contract changed")
    return plan, {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
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


def verify_source(expected_commit: str) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != expected_commit:
        raise ValueError("functional transition source commit changed")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, commit],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("functional transition source escaped the frozen base")
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
        raise ValueError(f"functional transition source is dirty: {sorted(dirty)}")
    expected_module = (REPO / "gr00t/experimental/arena_functional_mamba_transition.py").resolve()
    if Path(functional_module.__file__).resolve() != expected_module:
        raise ValueError("imported functional transition module escaped source root")
    return {
        "root": str(REPO.resolve()),
        "commit": commit,
        "entries": {
            "gr00t": _source_file_identity(Path(gr00t.__file__), commit),
            "experimental": _source_file_identity(Path(gr00t.experimental.__file__), commit),
            "functional_module": _source_file_identity(expected_module, commit),
            "runner": _source_file_identity(Path(__file__), commit),
        },
    }


def _verify_material_spec(spec: dict[str, Any], *, expected_path: Path, label: str) -> Path:
    if set(spec) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} material spec keys changed")
    path = Path(spec["path"]).resolve()
    if path != expected_path.resolve() or not path.is_file():
        raise ValueError(f"{label} path changed or is missing")
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
    unexpected = actual - recorded - {".lock"}
    if unexpected:
        raise ValueError(f"overlay contains files outside RECORD: {sorted(unexpected)}")
    return {
        "record_entries": len(recorded),
        "verified_hashed_bytes": verified_bytes,
        "unexpected_unrecorded_files": [],
    }


def verify_prior_evidence() -> dict[str, Any]:
    paths = {
        "streaming_r1": STREAMING_RESULT_PATH,
        "parity_diagnosis_r3": PARITY_RESULT_PATH,
    }
    verified = {
        name: _verify_material_spec(spec, expected_path=paths[name], label=name)
        for name, spec in FROZEN_PRIOR_EVIDENCE.items()
    }
    streaming = read_json(verified["streaming_r1"])
    parity = read_json(verified["parity_diagnosis_r3"])
    conditions = parity.get("numerical", {}).get("conditions", {})
    if (
        streaming.get("experiment_id") != "g1-arena-codec-streaming-backbone-r1"
        or streaming.get("optimizer_steps") != 0
        or streaming.get("vla_model_instantiated") is not False
    ):
        raise ValueError("streaming r1 result contract changed")
    if (
        parity.get("experiment_id") != "g1-arena-codec-streaming-parity-diagnosis-r1"
        or parity.get("source_commit") != BASE_SOURCE_COMMIT
        or parity.get("optimizer_steps") != 0
        or parity.get("vla_model_instantiated") is not False
        or conditions.get("mamba_chunk16_current", {})
        .get("prefix", {})
        .get("925", {})
        .get("max_abs")
        != 0.00025087594985961914
        or conditions.get("mamba_chunk1_candidate", {})
        .get("prefix", {})
        .get("925", {})
        .get("max_abs")
        != 0.00028705596923828125
    ):
        raise ValueError("parity diagnosis r3 result contract changed")
    return {
        name: {**FROZEN_PRIOR_EVIDENCE[name], "verified": True} for name in FROZEN_PRIOR_EVIDENCE
    }


def verify_dependencies() -> dict[str, Any]:
    expected_paths = {
        "manifest": DEPENDENCY_MANIFEST_PATH,
        "official_wheel": OFFICIAL_WHEEL_PATH,
        "overlay_record": OVERLAY_RECORD_PATH,
        "mamba_file": MAMBA2_FILE_PATH,
        "selective_scan_extension": SELECTIVE_SCAN_EXTENSION_PATH,
    }
    paths = {
        name: _verify_material_spec(
            spec,
            expected_path=expected_paths[name],
            label=name,
        )
        for name, spec in FROZEN_DEPENDENCY.items()
    }
    manifest = read_json(paths["manifest"])
    if (
        manifest.get("overlay", {}).get("mamba_version") != "2.3.2.post1"
        or manifest.get("overlay", {}).get("causal_conv1d_present") is not False
        or Path(manifest.get("overlay", {}).get("path", "")).resolve() != OVERLAY_ROOT.resolve()
        or Path(manifest.get("overlay", {}).get("mamba2_file", "")).resolve()
        != MAMBA2_FILE_PATH.resolve()
        or manifest.get("runtime", {}).get("torch") != FROZEN_RUNTIME["torch"]
        or manifest.get("runtime", {}).get("torch_cuda") != FROZEN_RUNTIME["torch_cuda"]
    ):
        raise ValueError("frozen dependency manifest contract changed")
    return {
        "materials": copy.deepcopy(FROZEN_DEPENDENCY),
        "overlay_closure": verify_overlay_record(paths["overlay_record"], OVERLAY_ROOT),
    }


def load_codec_input() -> tuple[np.ndarray, dict[str, Any]]:
    if INPUT_PATH.stat().st_size != INPUT_BYTES or sha256(INPUT_PATH) != INPUT_SHA256:
        raise ValueError("frozen codec input identity changed")
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        present = tuple(sorted(archive.files))
        if present != tuple(sorted(EXPECTED_ARCHIVE_FIELDS)):
            raise ValueError("frozen codec archive fields changed")
        stored = archive[INPUT_FIELD]
        if stored.dtype != np.dtype(np.float32):
            raise ValueError("frozen codec_v1 dtype changed")
        codec = np.array(stored, copy=True)
    if codec.shape != INPUT_SHAPE or not np.isfinite(codec).all():
        raise ValueError("frozen codec_v1 shape, dtype, or finiteness changed")
    return codec, {
        "path": str(INPUT_PATH),
        "bytes": INPUT_BYTES,
        "sha256": INPUT_SHA256,
        "field": INPUT_FIELD,
        "shape": list(codec.shape),
        "dtype": str(codec.dtype),
        "fields_present": list(present),
        "fields_consumed": [INPUT_FIELD],
        "fields_excluded": sorted(set(present) - {INPUT_FIELD}),
    }


def verify_environment() -> dict[str, Any]:
    import mamba_ssm
    import mamba_ssm.modules.mamba2 as official
    import mamba_ssm.ops.triton.ssd_combined as ssd_combined
    import selective_scan_cuda

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    actual_mamba_file = Path(official.__file__).resolve()
    actual_extension_file = Path(selective_scan_cuda.__file__).resolve()
    if Path(sys.executable).resolve() != Path(FROZEN_RUNTIME["python_executable"]).resolve():
        raise ValueError("formal Python executable changed")
    if platform.python_version() != FROZEN_RUNTIME["python"]:
        raise ValueError("formal Python version changed")
    if (
        torch.__version__ != FROZEN_RUNTIME["torch"]
        or torch.version.cuda != FROZEN_RUNTIME["torch_cuda"]
        or torch.compiled_with_cxx11_abi() is not FROZEN_RUNTIME["cxx11_abi"]
    ):
        raise ValueError("formal Torch, CUDA, or ABI contract changed")
    if importlib.metadata.version("mamba-ssm") != FROZEN_RUNTIME["mamba_ssm"]:
        raise ValueError("official Mamba version changed")
    if actual_mamba_file != MAMBA2_FILE_PATH.resolve():
        raise ValueError("Mamba2 import escaped the verified overlay")
    if actual_extension_file != SELECTIVE_SCAN_EXTENSION_PATH.resolve():
        raise ValueError("selective scan extension escaped the verified overlay")
    if official.selective_state_update is None:
        raise ValueError("official Mamba2 deployment step op is unavailable")
    critical_paths = (
        Path(mamba_ssm.__file__).resolve(),
        Path(ssd_combined.__file__).resolve(),
        Path(sys.modules[official.selective_state_update.__module__].__file__).resolve(),
    )
    if any(not path.is_relative_to(OVERLAY_ROOT.resolve()) for path in critical_paths):
        raise ValueError("critical official Mamba import escaped the overlay")
    if official.causal_conv1d_fn is not None or official.causal_conv1d_update is not None:
        raise ValueError("frozen explicit convolution path requires causal-conv1d to be absent")
    if official.mamba_chunk_scan_combined.__module__ != ssd_combined.__name__:
        raise ValueError("official full scan escaped the frozen SSD implementation")
    if os.environ.get("PYTHONPATH") != FROZEN_RUNTIME["pythonpath"]:
        raise ValueError("formal PYTHONPATH contract changed")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ValueError("formal run must disable Python bytecode writes")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != FROZEN_RUNTIME["cuda_visible_devices"]:
        raise ValueError("formal run must expose only physical GPU0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("formal run requires exactly one visible CUDA device")
    if torch.cuda.get_device_name(0) != FROZEN_RUNTIME["gpu_name"]:
        raise ValueError("formal GPU identity changed")
    if (
        torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cuda")
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise ValueError("FP32 autocast or TF32 contract changed")
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "mamba_ssm": importlib.metadata.version("mamba-ssm"),
        "mamba_file": str(actual_mamba_file),
        "selective_scan_extension": str(actual_extension_file),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "autocast": False,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state: FunctionalMamba2State) -> str:
    digest = hashlib.sha256()
    for layer, pair in enumerate(state.layers):
        for kind, tensor in zip(("conv", "ssm"), pair, strict=True):
            digest.update(f"{layer}:{kind}:".encode("ascii"))
            digest.update(bytes.fromhex(tensor_sha256(tensor)))
    return digest.hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _storage_id(value: torch.Tensor) -> tuple[str, int]:
    return str(value.device), value.untyped_storage().data_ptr()


def _state_tensors(
    state: FunctionalMamba2State,
) -> list[tuple[int, str, torch.Tensor]]:
    return [
        (layer, kind, tensor)
        for layer, pair in enumerate(state.layers)
        for kind, tensor in zip(("conv", "ssm"), pair, strict=True)
    ]


def state_storage_report(
    input_state: FunctionalMamba2State,
    output_state: FunctionalMamba2State,
) -> dict[str, Any]:
    inputs = _state_tensors(input_state)
    outputs = _state_tensors(output_state)
    input_output_overlap = [
        {
            "input": [input_layer, input_kind],
            "output": [output_layer, output_kind],
        }
        for input_layer, input_kind, input_tensor in inputs
        for output_layer, output_kind, output_tensor in outputs
        if _storage_id(input_tensor) == _storage_id(output_tensor)
    ]
    output_internal_overlap = [
        {
            "first": [first_layer, first_kind],
            "second": [second_layer, second_kind],
            "same_layer": first_layer == second_layer,
        }
        for first_index, (first_layer, first_kind, first_tensor) in enumerate(outputs)
        for second_layer, second_kind, second_tensor in outputs[first_index + 1 :]
        if _storage_id(first_tensor) == _storage_id(second_tensor)
    ]
    return {
        "input_output_overlap": input_output_overlap,
        "output_internal_overlap": output_internal_overlap,
        "passed": not input_output_overlap and not output_internal_overlap,
    }


def independent_state_storage_report(first: Any, second: Any) -> dict[str, Any]:
    first_ids = {_storage_id(tensor) for pair in first.layers for tensor in pair}
    second_ids = {_storage_id(tensor) for pair in second.layers for tensor in pair}
    overlap = sorted(first_ids & second_ids)
    return {"overlap": overlap, "passed": not overlap}


def max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max().item())


def states_equal(first: FunctionalMamba2State, second: FunctionalMamba2State) -> bool:
    return first.observations == second.observations and all(
        torch.equal(first_tensor, second_tensor)
        for first_pair, second_pair in zip(first.layers, second.layers, strict=True)
        for first_tensor, second_tensor in zip(first_pair, second_pair, strict=True)
    )


def _error_metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    error = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(error.max().item()),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
    }


def functional_ast_report() -> dict[str, Any]:
    import ast

    source_path = Path(functional_module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    targets = {
        "functional_mamba2_step",
        "functional_step_with_trace",
        "functional_step",
        "functional_stream",
    }
    reports: dict[str, Any] = {}
    for node in ast.walk(tree):
        if (
            not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            or node.name not in targets
        ):
            continue
        calls = [
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        ]
        decorators = [
            decorator.id for decorator in node.decorator_list if isinstance(decorator, ast.Name)
        ]
        slice_assignments = [
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Subscript) and isinstance(child.ctx, ast.Store)
        ]
        forbidden_calls = sorted({name for name in calls if name in {"detach", "copy_", "no_grad"}})
        reports[node.name] = {
            "forbidden_calls": forbidden_calls,
            "forbidden_decorators": sorted(set(decorators) & {"no_grad", "inference_mode"}),
            "slice_assignment_lines": slice_assignments,
            "passed": not forbidden_calls
            and not slice_assignments
            and not (set(decorators) & {"no_grad", "inference_mode"}),
        }
    missing = sorted(targets - set(reports))
    return {
        "source": str(source_path),
        "functions": reports,
        "missing_functions": missing,
        "passed": not missing and all(report["passed"] for report in reports.values()),
    }


@torch.inference_mode()
def official_step_with_trace(
    module: FunctionalMamba2Backbone,
    descriptor: torch.Tensor,
    state: Any,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    value = descriptor.unsqueeze(1)
    residual_outputs = []
    for layer, cache in zip(module.layers, state.layers, strict=True):
        update, conv_state, ssm_state = layer.mamba.step(layer.norm(value), *cache)
        if conv_state is not cache[0] or ssm_state is not cache[1]:
            raise RuntimeError("official Mamba2.step stopped mutating its supplied cache")
        value = value + update
        residual_outputs.append(value.squeeze(1))
    state.observations += 1
    return value.squeeze(1), tuple(residual_outputs)


def _first_crossing(values: list[float], threshold: float) -> int | None:
    return next((index + 1 for index, value in enumerate(values) if value > threshold), None)


@torch.inference_mode()
def teacher_functional_parity(
    module: FunctionalMamba2Backbone,
    descriptors: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, FunctionalMamba2State]:
    teacher_state = module.new_official_state(int(descriptors.shape[0]))
    functional_state = module.new_functional_state(int(descriptors.shape[0]))
    initial_storage = independent_state_storage_report(teacher_state, functional_state)
    if any(
        tensor.dtype != torch.float32 or bool(torch.count_nonzero(tensor).item())
        for state in (teacher_state, functional_state)
        for pair in state.layers
        for tensor in pair
    ):
        raise RuntimeError("teacher and functional caches must start as independent FP32 zeros")

    per_frame = []
    teacher_outputs = []
    functional_outputs = []
    metric_series = {
        layer: {name: [] for name in ("residual_output", "conv_state", "ssm_state")}
        for layer in range(LAYERS)
    }
    all_finite = True
    all_inputs_unchanged = True
    all_storage_independent = initial_storage["passed"]
    first_state_bytes = None
    for frame, descriptor in enumerate(descriptors.unbind(dim=1), start=1):
        input_state = functional_state
        snapshot = input_state.clone()
        input_hash_before = state_sha256(input_state)
        input_observations_before = input_state.observations
        teacher_output, teacher_layers = official_step_with_trace(
            module,
            descriptor,
            teacher_state,
        )
        functional_output, functional_state, functional_layers = module.functional_step_with_trace(
            descriptor, input_state
        )
        input_hash_after = state_sha256(input_state)
        tensors_unchanged = all(
            torch.equal(actual, expected)
            for actual_pair, expected_pair in zip(input_state.layers, snapshot.layers, strict=True)
            for actual, expected in zip(actual_pair, expected_pair, strict=True)
        )
        observations_unchanged = input_state.observations == input_observations_before
        storage = state_storage_report(input_state, functional_state)
        all_inputs_unchanged = all_inputs_unchanged and (
            input_hash_before == input_hash_after and tensors_unchanged and observations_unchanged
        )
        all_storage_independent = all_storage_independent and storage["passed"]
        layer_rows = []
        for layer in range(LAYERS):
            metrics = {
                "residual_output": _error_metric(
                    functional_layers[layer],
                    teacher_layers[layer],
                ),
                "conv_state": _error_metric(
                    functional_state.layers[layer][0],
                    teacher_state.layers[layer][0],
                ),
                "ssm_state": _error_metric(
                    functional_state.layers[layer][1],
                    teacher_state.layers[layer][1],
                ),
            }
            for name, metric in metrics.items():
                metric_series[layer][name].append(metric["max_abs"])
                all_finite = all_finite and metric["finite"]
            layer_rows.append(metrics)
        all_finite = all_finite and bool(
            torch.isfinite(teacher_output).all() and torch.isfinite(functional_output).all()
        )
        state_bytes = functional_state.tensor_nbytes()
        if first_state_bytes is None:
            first_state_bytes = state_bytes
        per_frame.append(
            {
                "frame": frame,
                "layers": layer_rows,
                "input_state_sha256_before": input_hash_before,
                "input_state_sha256_after": input_hash_after,
                "input_state_tensors_unchanged": tensors_unchanged,
                "input_state_observations_unchanged": observations_unchanged,
                "new_state_storage": storage,
                "state_logical_bytes": state_bytes,
            }
        )
        teacher_outputs.append(teacher_output)
        functional_outputs.append(functional_output)
    if first_state_bytes is None:
        raise RuntimeError("parity probe produced no state")
    summary = {
        f"layer_{layer}": {
            name: {
                "max_abs": max(values),
                "first_frame_above_1e-4": _first_crossing(values, MAX_PARITY_ABS),
            }
            for name, values in rows.items()
        }
        for layer, rows in metric_series.items()
    }
    last_state_bytes = functional_state.tensor_nbytes()
    return (
        {
            "frames": int(descriptors.shape[1]),
            "per_frame": per_frame,
            "summary": summary,
            "all_outputs_and_states_finite": all_finite,
            "all_input_states_unchanged": all_inputs_unchanged,
            "all_new_state_storage_independent": all_storage_independent,
            "initial_teacher_functional_storage": initial_storage,
            "state_logical_bytes_t1": first_state_bytes,
            "state_logical_bytes_t925": last_state_bytes,
            "state_logical_growth_fraction": (
                (last_state_bytes - first_state_bytes) / first_state_bytes
            ),
        },
        torch.stack(teacher_outputs, dim=1),
        torch.stack(functional_outputs, dim=1),
        functional_state,
    )


def functional_run(
    module: FunctionalMamba2Backbone,
    descriptors: torch.Tensor,
    *,
    chunks: tuple[int, ...] | None = None,
    state: FunctionalMamba2State | None = None,
) -> tuple[torch.Tensor, FunctionalMamba2State]:
    lengths = (int(descriptors.shape[1]),) if chunks is None else chunks
    if sum(lengths) != int(descriptors.shape[1]) or any(length < 1 for length in lengths):
        raise ValueError("functional stream chunks do not cover the sequence exactly")
    outputs = []
    current = state
    start = 0
    for length in lengths:
        output, current = module.functional_stream(
            descriptors[:, start : start + length],
            current,
        )
        outputs.append(output)
        start += length
    if current is None:
        raise RuntimeError("functional stream returned no state")
    return torch.cat(outputs, dim=1), current


@torch.inference_mode()
def recurrence_invariants(
    module: FunctionalMamba2Backbone,
    tokenizer: ArenaCodecSpatialTokenizer,
    raw: torch.Tensor,
    descriptors: torch.Tensor,
    official_teacher: torch.Tensor,
    continuous: torch.Tensor,
    final_state: FunctionalMamba2State,
) -> dict[str, Any]:
    chunked, chunked_state = functional_run(module, descriptors, chunks=CORRECTNESS_CHUNKS)
    repeated, repeated_state = functional_run(module, descriptors)
    reset, reset_state = functional_run(module, descriptors, state=final_state.reset())
    changed_raw = torch.cat(
        (raw[:, :CAUSAL_PREFIX_LENGTH], raw[:, CAUSAL_PREFIX_LENGTH:] + 0.125),
        dim=1,
    )
    changed_descriptors = tokenizer(changed_raw)
    changed, _ = functional_run(module, changed_descriptors)
    official_full = module(descriptors)
    return {
        "continuous_vs_fixed_chunks": {
            "chunks": list(CORRECTNESS_CHUNKS),
            "exact": bool(torch.equal(continuous, chunked)),
            "max_abs": max_abs(continuous, chunked),
            "final_state_exact": all(
                torch.equal(first, second)
                for first_pair, second_pair in zip(
                    final_state.layers,
                    chunked_state.layers,
                    strict=True,
                )
                for first, second in zip(first_pair, second_pair, strict=True)
            ),
        },
        "repeat": {
            "exact": bool(torch.equal(continuous, repeated)),
            "max_abs": max_abs(continuous, repeated),
            "final_state_exact": states_equal(final_state, repeated_state),
            "final_state_observations": repeated_state.observations,
        },
        "episode_reset": {
            "exact": bool(torch.equal(continuous, reset)),
            "max_abs": max_abs(continuous, reset),
            "final_state_exact": states_equal(final_state, reset_state),
            "final_state_observations": reset_state.observations,
        },
        "suffix_edit": {
            "prefix_length": CAUSAL_PREFIX_LENGTH,
            "prefix_exact": bool(
                torch.equal(
                    continuous[:, :CAUSAL_PREFIX_LENGTH],
                    changed[:, :CAUSAL_PREFIX_LENGTH],
                )
            ),
            "prefix_max_abs": max_abs(
                continuous[:, :CAUSAL_PREFIX_LENGTH],
                changed[:, :CAUSAL_PREFIX_LENGTH],
            ),
        },
        "official_full_chunk16_report_only": {
            "observed_full_vs_official_step_teacher_max_abs": max_abs(
                official_full, official_teacher
            ),
            "observed_full_vs_functional_max_abs": max_abs(official_full, continuous),
            "participates_in_pass_gate": False,
            "prior_r3_full_vs_official_step_t925_max_abs": 0.00025087594985961914,
            "prior_r3_chunk1_vs_official_step_t925_max_abs": 0.00028705596923828125,
        },
        "current_descriptor_identity": {
            "temporal_parameters": 0,
            "role": "cost_lower_bound_only",
            "participates_in_capability_or_parameter_gate": False,
        },
    }


def parameter_contract(
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
) -> dict[str, Any]:
    tokenizer_names = tuple(name for name, _ in tokenizer.named_parameters())
    temporal_names = tuple(name for name, _ in temporal.named_parameters())
    parameter_ids = [id(parameter) for parameter in temporal.parameters()]
    return {
        "expected_tokenizer": list(EXPECTED_TOKENIZER_PARAMETERS),
        "actual_tokenizer": list(tokenizer_names),
        "expected_temporal": list(EXPECTED_TEMPORAL_PARAMETERS),
        "actual_temporal": list(temporal_names),
        "tokenizer_exact": set(tokenizer_names) == set(EXPECTED_TOKENIZER_PARAMETERS)
        and len(tokenizer_names) == len(EXPECTED_TOKENIZER_PARAMETERS),
        "temporal_exact": set(temporal_names) == set(EXPECTED_TEMPORAL_PARAMETERS)
        and len(temporal_names) == len(EXPECTED_TEMPORAL_PARAMETERS),
        "temporal_parameter_objects_unique": len(parameter_ids) == len(set(parameter_ids)),
        "tokenizer_parameter_tensors": len(tokenizer_names),
        "temporal_parameter_tensors": len(temporal_names),
        "temporal_state_dict_sha256": state_dict_sha256(temporal),
        "functional_reads_same_official_module_instances": True,
    }


def normalized_probe(device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    values = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)
    values = values / np.linalg.norm(values)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    if digest != PROBE_SHA256:
        raise RuntimeError("fixed normalized gradient probe identity changed")
    probe = torch.from_numpy(values.copy()).to(device=device, dtype=torch.float32)
    return probe, {
        "construction": "np.linspace(-1,1,256,dtype=float32)/np.linalg.norm",
        "dtype": str(probe.dtype),
        "shape": list(probe.shape),
        "l2_norm": float(torch.linalg.vector_norm(probe).item()),
        "sha256": digest,
    }


def fixed_loss(output: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    score = torch.sum(output[:, -1].float() * probe)
    return (score - 0.25).square()


def _all_named_parameters(
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
) -> dict[str, nn.Parameter]:
    return {
        **{f"tokenizer.{name}": parameter for name, parameter in tokenizer.named_parameters()},
        **{f"temporal.{name}": parameter for name, parameter in temporal.named_parameters()},
    }


def _gradient_summary(parameters: dict[str, nn.Parameter]) -> dict[str, Any]:
    missing = [name for name, parameter in parameters.items() if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in parameters.items()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    zero = [
        name
        for name, parameter in parameters.items()
        if parameter.grad is not None and not bool(torch.count_nonzero(parameter.grad).item())
    ]
    return {
        "expected_parameter_tensors": len(EXPECTED_TOKENIZER_PARAMETERS)
        + len(EXPECTED_TEMPORAL_PARAMETERS),
        "actual_parameter_tensors": len(parameters),
        "names": list(parameters),
        "missing": missing,
        "nonfinite": nonfinite,
        "zero": zero,
        "l2_norms": {
            name: float(torch.linalg.vector_norm(parameter.grad.float()).item())
            for name, parameter in parameters.items()
            if parameter.grad is not None
        },
        "passed": not missing
        and not nonfinite
        and not zero
        and len(parameters)
        == len(EXPECTED_TOKENIZER_PARAMETERS) + len(EXPECTED_TEMPORAL_PARAMETERS),
    }


def _clear_gradients(
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
) -> None:
    tokenizer.zero_grad(set_to_none=True)
    temporal.zero_grad(set_to_none=True)


def gradient_pass(
    raw: torch.Tensor,
    *,
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
    probe: torch.Tensor,
    chunks: tuple[int, ...] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor, float]:
    _clear_gradients(tokenizer, temporal)
    descriptors = tokenizer(raw[:, :GRADIENT_LENGTH])
    descriptors.retain_grad()
    output, final_state = functional_run(temporal, descriptors, chunks=chunks)
    loss = fixed_loss(output, probe)
    loss.backward()
    parameters = _all_named_parameters(tokenizer, temporal)
    summary = _gradient_summary(parameters)
    if descriptors.grad is None:
        raise RuntimeError("descriptor gradient is missing")
    descriptor_gradient = descriptors.grad.detach().clone()
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in parameters.items()
        if parameter.grad is not None
    }
    frame1_norm = float(torch.linalg.vector_norm(descriptor_gradient[:, 0].float()).item())
    report = {
        "chunks": None if chunks is None else list(chunks),
        "loss": float(loss.detach().item()),
        "output_finite": bool(torch.isfinite(output).all()),
        "final_state_observations": final_state.observations,
        "parameters": summary,
        "descriptor_gradient": {
            "shape": list(descriptor_gradient.shape),
            "finite": bool(torch.isfinite(descriptor_gradient).all()),
            "frame1_l2_norm": frame1_norm,
            "global_l2_norm": float(torch.linalg.vector_norm(descriptor_gradient.float()).item()),
        },
    }
    return report, parameter_gradients, descriptor_gradient, float(loss.detach().item())


def _tensor_error(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())
    if not finite:
        return {"max_abs": float("inf"), "max_relative": float("inf"), "finite": False}
    absolute = max_abs(first, second)
    denominator = max(
        float(first.float().abs().max().item()),
        float(second.float().abs().max().item()),
        torch.finfo(torch.float32).tiny,
    )
    return {"max_abs": absolute, "max_relative": absolute / denominator, "finite": True}


def gradient_chunk_comparison(
    continuous_report: dict[str, Any],
    continuous_parameters: dict[str, torch.Tensor],
    continuous_descriptors: torch.Tensor,
    continuous_loss: float,
    chunked_report: dict[str, Any],
    chunked_parameters: dict[str, torch.Tensor],
    chunked_descriptors: torch.Tensor,
    chunked_loss: float,
) -> dict[str, Any]:
    if set(continuous_parameters) != set(chunked_parameters):
        raise RuntimeError("continuous and chunked parameter gradient names differ")
    parameter_errors = {
        name: _tensor_error(continuous_parameters[name], chunked_parameters[name])
        for name in continuous_parameters
    }
    loss_abs = abs(continuous_loss - chunked_loss)
    loss_relative = loss_abs / max(
        abs(continuous_loss),
        abs(chunked_loss),
        torch.finfo(torch.float32).tiny,
    )
    descriptor_error = _tensor_error(continuous_descriptors, chunked_descriptors)
    all_errors_finite = (
        math.isfinite(loss_abs)
        and descriptor_error["finite"]
        and all(row["finite"] for row in parameter_errors.values())
    )
    max_parameter_abs = max(row["max_abs"] for row in parameter_errors.values())
    combined_max_abs = max(loss_abs, max_parameter_abs, descriptor_error["max_abs"])
    return {
        "continuous": continuous_report,
        "chunked": chunked_report,
        "loss": {"abs": loss_abs, "relative": loss_relative},
        "parameters": parameter_errors,
        "descriptor_gradient": descriptor_error,
        "all_errors_finite": all_errors_finite,
        "max_abs": combined_max_abs if all_errors_finite else float("inf"),
    }


def finite_difference_report(
    raw: torch.Tensor,
    *,
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
    probe: torch.Tensor,
    analytic_descriptor_gradient: torch.Tensor,
) -> dict[str, Any]:
    _clear_gradients(tokenizer, temporal)
    descriptors = tokenizer(raw[:, :GRADIENT_LENGTH])
    direction = probe.reshape(1, 1, WIDTH)
    plus = torch.cat(
        (descriptors[:, :1] + FINITE_DIFFERENCE_EPSILON * direction, descriptors[:, 1:]),
        dim=1,
    )
    minus = torch.cat(
        (descriptors[:, :1] - FINITE_DIFFERENCE_EPSILON * direction, descriptors[:, 1:]),
        dim=1,
    )
    plus_output, _ = functional_run(temporal, plus)
    minus_output, _ = functional_run(temporal, minus)
    plus_loss = fixed_loss(plus_output, probe)
    minus_loss = fixed_loss(minus_output, probe)
    finite_difference = float(
        ((plus_loss - minus_loss) / (2 * FINITE_DIFFERENCE_EPSILON)).detach().item()
    )
    analytic = float(torch.sum(analytic_descriptor_gradient[0, 0].float() * probe).detach().item())
    absolute_error = abs(analytic - finite_difference)
    tolerance = 1e-5 + 0.05 * max(abs(analytic), abs(finite_difference))
    return {
        "epsilon": FINITE_DIFFERENCE_EPSILON,
        "analytic_directional_derivative": analytic,
        "finite_difference_directional_derivative": finite_difference,
        "plus_loss": float(plus_loss.detach().item()),
        "minus_loss": float(minus_loss.detach().item()),
        "same_nonzero_sign": analytic * finite_difference > 0.0,
        "abs_error": absolute_error,
        "tolerance": tolerance,
        "passed": analytic * finite_difference > 0.0 and absolute_error <= tolerance,
    }


def detached_state_negative_control(
    raw: torch.Tensor,
    *,
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
    probe: torch.Tensor,
) -> dict[str, Any]:
    _clear_gradients(tokenizer, temporal)
    descriptors = tokenizer(raw[:, :GRADIENT_LENGTH])
    state = temporal.new_functional_state(int(descriptors.shape[0]))
    _, state = temporal.functional_step(descriptors[:, 0], state)
    detached_state = FunctionalMamba2State(
        tuple((conv_state.detach(), ssm_state.detach()) for conv_state, ssm_state in state.layers),
        state.batch_size,
        state.observations,
    )
    output, _ = temporal.functional_stream(descriptors[:, 1:], detached_state)
    loss = fixed_loss(output, probe)
    descriptor_gradient = torch.autograd.grad(loss, descriptors)[0]
    frame1 = descriptor_gradient[:, 0]
    return {
        "detached_after_frame": 1,
        "detached_layer_states": len(detached_state.layers),
        "frame1_gradient_nonzero_elements": int(torch.count_nonzero(frame1).item()),
        "frame1_gradient_l2_norm": float(torch.linalg.vector_norm(frame1.float()).item()),
        "frame1_gradient_exact_zero": bool(torch.equal(frame1, torch.zeros_like(frame1))),
    }


def gradient_probe(
    raw: torch.Tensor,
    *,
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
) -> dict[str, Any]:
    tokenizer.train()
    temporal.train()
    probe, probe_identity = normalized_probe(raw.device)
    continuous = gradient_pass(
        raw,
        tokenizer=tokenizer,
        temporal=temporal,
        probe=probe,
        chunks=None,
    )
    chunked = gradient_pass(
        raw,
        tokenizer=tokenizer,
        temporal=temporal,
        probe=probe,
        chunks=GRADIENT_CHUNKS,
    )
    comparison = gradient_chunk_comparison(
        continuous[0],
        continuous[1],
        continuous[2],
        continuous[3],
        chunked[0],
        chunked[1],
        chunked[2],
        chunked[3],
    )
    finite_difference = finite_difference_report(
        raw,
        tokenizer=tokenizer,
        temporal=temporal,
        probe=probe,
        analytic_descriptor_gradient=continuous[2],
    )
    negative_control = detached_state_negative_control(
        raw,
        tokenizer=tokenizer,
        temporal=temporal,
        probe=probe,
    )
    _clear_gradients(tokenizer, temporal)
    return {
        "probe": probe_identity,
        "loss_contract": "(dot(final_output,probe)-0.25)^2",
        "continuous_vs_fixed_chunks": comparison,
        "finite_difference": finite_difference,
        "detached_state_negative_control": negative_control,
    }


def _parity_maxima(parity: dict[str, Any]) -> list[float]:
    return [
        parity["summary"][f"layer_{layer}"][name]["max_abs"]
        for layer in range(LAYERS)
        for name in ("residual_output", "conv_state", "ssm_state")
    ]


def parity_cardinality_and_summary_report(parity: dict[str, Any]) -> dict[str, Any]:
    metric_names = ("residual_output", "conv_state", "ssm_state")
    rows = parity.get("per_frame")
    if (
        parity.get("frames") != INPUT_SHAPE[0]
        or not isinstance(rows, list)
        or len(rows) != INPUT_SHAPE[0]
        or [row.get("frame") for row in rows] != list(range(1, INPUT_SHAPE[0] + 1))
    ):
        return {"passed": False, "recomputed_maxima": [float("inf")], "reason": "cardinality"}
    recomputed: dict[str, Any] = {}
    try:
        for layer in range(LAYERS):
            layer_key = f"layer_{layer}"
            recomputed[layer_key] = {}
            for name in metric_names:
                values = []
                for row in rows:
                    layers = row["layers"]
                    if len(layers) != LAYERS or set(layers[layer]) != set(metric_names):
                        raise ValueError("per-frame parity layer schema changed")
                    metric = layers[layer][name]
                    value = float(metric["max_abs"])
                    if metric.get("finite") is not True or not math.isfinite(value):
                        raise ValueError("per-frame parity metric is nonfinite")
                    values.append(value)
                recomputed[layer_key][name] = {
                    "max_abs": max(values),
                    "first_frame_above_1e-4": _first_crossing(values, MAX_PARITY_ABS),
                }
    except (KeyError, TypeError, ValueError):
        return {"passed": False, "recomputed_maxima": [float("inf")], "reason": "schema"}
    summary_matches = parity.get("summary") == recomputed
    maxima = [
        recomputed[f"layer_{layer}"][name]["max_abs"]
        for layer in range(LAYERS)
        for name in metric_names
    ]
    return {
        "passed": summary_matches,
        "recomputed_summary": recomputed,
        "recomputed_maxima": maxima,
        "reason": None if summary_matches else "summary_mismatch",
    }


def evaluate_correctness_gates(
    numerical: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, bool]:
    parity = numerical["teacher_functional_parity"]
    recurrence = numerical["recurrence"]
    gradients = numerical["gradients"]
    comparison = gradients["continuous_vs_fixed_chunks"]
    continuous = comparison["continuous"]
    chunked = comparison["chunked"]
    parameter_contract_report = numerical["parameter_contract"]
    parity_integrity = parity_cardinality_and_summary_report(parity)
    return {
        "source_input_dependency_runtime_identity": True,
        "functional_ast_has_no_forbidden_graph_operations": numerical["functional_ast"]["passed"],
        "exact_6_tokenizer_and_20_temporal_parameters": (
            parameter_contract_report["tokenizer_exact"]
            and parameter_contract_report["temporal_exact"]
            and parameter_contract_report["temporal_parameter_objects_unique"]
        ),
        "teacher_functional_parity_has_all_925_frames_and_recomputed_summary": parity_integrity[
            "passed"
        ],
        "teacher_functional_all_925_frames_two_layers_at_most_1e-4": (
            parity_integrity["passed"]
            and max(parity_integrity["recomputed_maxima"]) <= MAX_PARITY_ABS
            and parity["all_outputs_and_states_finite"]
        ),
        "functional_input_state_unchanged_every_step": parity["all_input_states_unchanged"],
        "functional_state_storage_never_aliases": parity["all_new_state_storage_independent"],
        "functional_state_logical_growth_at_most_1pct": parity["state_logical_growth_fraction"]
        <= MAX_STATE_GROWTH,
        "continuous_fixed_t925_chunks_exact": (
            recurrence["continuous_vs_fixed_chunks"]["exact"]
            and recurrence["continuous_vs_fixed_chunks"]["final_state_exact"]
            and recurrence["continuous_vs_fixed_chunks"]["chunks"] == list(CORRECTNESS_CHUNKS)
        ),
        "repeat_and_episode_reset_exact": (
            recurrence["repeat"]["exact"]
            and recurrence["repeat"]["final_state_exact"]
            and recurrence["episode_reset"]["exact"]
            and recurrence["episode_reset"]["final_state_exact"]
        ),
        "suffix_edit_prefix_exact": recurrence["suffix_edit"]["prefix_exact"],
        "fixed_probe_identity": gradients["probe"]["sha256"] == PROBE_SHA256,
        "continuous_all_26_parameter_gradients_finite_nonzero": continuous["parameters"]["passed"],
        "chunked_all_26_parameter_gradients_finite_nonzero": chunked["parameters"]["passed"],
        "frame1_descriptor_gradient_finite_and_at_least_1e-8": (
            continuous["descriptor_gradient"]["finite"]
            and chunked["descriptor_gradient"]["finite"]
            and continuous["descriptor_gradient"]["frame1_l2_norm"] >= MIN_FRAME1_GRADIENT_NORM
        ),
        "finite_difference_same_sign_and_within_tolerance": gradients["finite_difference"][
            "passed"
        ],
        "detached_state_negative_control_frame1_gradient_zero": gradients[
            "detached_state_negative_control"
        ]["frame1_gradient_exact_zero"],
        "continuous_chunked_loss_and_all_gradients_at_most_1e-5": comparison["all_errors_finite"]
        and comparison["max_abs"] <= MAX_GRADIENT_ABS,
        "native_vla_modules_absent": not execution["native_vla_modules"]["before"]
        and not execution["native_vla_modules"]["after"],
        "optimizer_steps_zero": not execution["optimizer"]["constructor_calls"]
        and execution["optimizer"]["steps_observed"] == 0,
    }


def correctness_probe(codec: np.ndarray) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    raw = torch.from_numpy(codec).unsqueeze(0).to(device=device, dtype=torch.float32)
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device=device, dtype=torch.float32)
    temporal = (
        FunctionalMamba2Backbone(seed=SEED)
        .eval()
        .to(
            device=device,
            dtype=torch.float32,
        )
    )
    parameters = parameter_contract(tokenizer, temporal)
    with torch.inference_mode():
        descriptors = tokenizer(raw)
    if list(descriptors.shape) != [1, INPUT_SHAPE[0], WIDTH] or not bool(
        torch.isfinite(descriptors).all()
    ):
        raise RuntimeError("shared Arena codec tokenizer output is invalid")
    parity, official_teacher, continuous, final_state = teacher_functional_parity(
        temporal, descriptors
    )
    recurrence = recurrence_invariants(
        temporal,
        tokenizer,
        raw,
        descriptors,
        official_teacher,
        continuous,
        final_state,
    )
    gradients = gradient_probe(raw, tokenizer=tokenizer, temporal=temporal)
    torch.cuda.synchronize(device)
    return {
        "tokenizer": {
            "descriptor_shape": list(descriptors.shape),
            "descriptor_dtype": str(descriptors.dtype),
            "descriptor_finite": bool(torch.isfinite(descriptors).all()),
            "state_dict_sha256": state_dict_sha256(tokenizer),
        },
        "parameter_contract": parameters,
        "functional_ast": functional_ast_report(),
        "teacher_functional_parity": parity,
        "recurrence": recurrence,
        "gradients": gradients,
    }


def _finalize_artifact_gate(
    result: dict[str, Any],
    *,
    artifact_bytes_before: int,
    gate_name: str,
) -> None:
    for _ in range(3):
        result_bytes = len((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        total_bytes = artifact_bytes_before + result_bytes
        result["artifact"] = {
            "root": str(ARTIFACT_ROOT.resolve()),
            "bytes_before_result": artifact_bytes_before,
            "result_bytes": result_bytes,
            "total_bytes": total_bytes,
            "limit_bytes": MAX_ARTIFACT_BYTES,
            "traces_written": 0,
        }
        result["gates"][gate_name] = total_bytes <= MAX_ARTIFACT_BYTES
        result["status"] = "passed" if all(result["gates"].values()) else "gate_failed"


def run_correctness(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_bytes_before = directory_bytes(ARTIFACT_ROOT)
    native_before = native_vla_modules_loaded()
    source = verify_source(plan["source"]["commit"])
    prior_evidence = verify_prior_evidence()
    dependencies = verify_dependencies()
    codec, input_evidence = load_codec_input()
    environment = verify_environment()
    with optimizer_construction_guard() as optimizer_evidence:
        numerical = correctness_probe(codec)
    execution = {
        "native_vla_modules": {
            "before": native_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
    }
    gates = evaluate_correctness_gates(numerical, execution)
    wall_seconds = time.monotonic() - started
    gates["wall_time_at_most_10min"] = wall_seconds <= MAX_WALL_SECONDS
    gates["artifact_total_at_most_256mib"] = True
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": CORRECTNESS_RUN_ID,
        "source_commit": source["commit"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source": source,
        "plan": plan_identity,
        "prior_evidence": prior_evidence,
        "input": input_evidence,
        "dependencies": dependencies,
        "environment": environment,
        "protocol": FROZEN_PROTOCOL,
        "thresholds": FROZEN_THRESHOLDS,
        "execution_evidence": execution,
        "numerical": numerical,
        "gates": gates,
        "optimizer_steps": optimizer_evidence["steps_observed"],
        "vla_model_instantiated": bool(
            execution["native_vla_modules"]["before"] or execution["native_vla_modules"]["after"]
        ),
        "latency_allowed": all(gates.values()),
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "fixed-weight functional transition engineering only",
    }
    _finalize_artifact_gate(
        result,
        artifact_bytes_before=artifact_bytes_before,
        gate_name="artifact_total_at_most_256mib",
    )
    result["latency_allowed"] = result["status"] == "passed"
    return result


def verify_passed_result(
    path: Path,
    expected_sha256: str,
    *,
    expected_path: Path,
    expected_run_id: str,
    plan_identity: dict[str, Any],
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if resolved != expected_path.resolve() or not resolved.is_file():
        raise ValueError(f"required predecessor result path must be {expected_path}")
    actual_sha256 = sha256(resolved)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ValueError("predecessor result identity changed")
    result = read_json(resolved)
    if (
        result.get("experiment_id") != EXPERIMENT
        or result.get("run_id") != expected_run_id
        or result.get("source_commit") != source_commit
        or result.get("plan", {}).get("sha256") != plan_identity["sha256"]
        or result.get("status") != "passed"
        or not result.get("gates")
        or not all(result["gates"].values())
        or result.get("optimizer_steps") != 0
        or result.get("vla_model_instantiated") is not False
    ):
        raise ValueError("predecessor result did not pass its frozen contract")
    if expected_run_id == CORRECTNESS_RUN_ID:
        recomputed = evaluate_correctness_gates(
            result["numerical"],
            result["execution_evidence"],
        )
        if any(not value for value in recomputed.values()) or any(
            result["gates"].get(name) != value for name, value in recomputed.items()
        ):
            raise ValueError("correctness predecessor gates do not recompute from raw evidence")
    return result, {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
        "run_id": expected_run_id,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _nvidia_smi_csv(query: str, *, gpu_id: int | None = None) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-{query.split(':', maxsplit=1)[0]}={query.split(':', maxsplit=1)[1]}",
        "--format=csv,noheader,nounits",
    ]
    if gpu_id is not None:
        command.extend(["--id", str(gpu_id)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip() and "No running processes found" not in line
    ]


def gpu_telemetry() -> dict[str, Any]:
    gpu_fields = "index,uuid,name,driver_version,clocks.current.sm,temperature.gpu"
    gpu_rows = _nvidia_smi_csv(f"gpu:{gpu_fields}", gpu_id=0)
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 6:
        raise RuntimeError("nvidia-smi returned an unexpected GPU0 identity row")
    gpu = dict(
        zip(
            (
                "physical_index",
                "uuid",
                "name",
                "driver_version",
                "sm_clock_mhz",
                "temperature_c",
            ),
            gpu_rows[0],
            strict=True,
        )
    )
    process_fields = "gpu_uuid,pid,process_name,used_gpu_memory"
    process_rows = _nvidia_smi_csv(f"compute-apps:{process_fields}")
    processes = [
        dict(
            zip(
                ("gpu_uuid", "pid", "process_name", "used_gpu_memory_mib"),
                row,
                strict=True,
            )
        )
        for row in process_rows
        if len(row) == 4 and row[0] == gpu["uuid"]
    ]
    current_pid = os.getpid()
    others = [row for row in processes if int(row["pid"]) != current_pid]
    return {
        "gpu": gpu,
        "process_pid": current_pid,
        "process_ppid": os.getppid(),
        "gpu0_compute_processes": processes,
        "gpu0_other_compute_processes": others,
    }


def _is_cuda_kernel_event(event: Any) -> bool:
    if not str(getattr(event, "device_type", "")).lower().endswith("cuda"):
        return False
    name = str(getattr(event, "name", "")).lower()
    return "memcpy" not in name and "memset" not in name


def _event_latency_label(event: Any, labels: set[str]) -> str | None:
    parent = getattr(event, "cpu_parent", None)
    while parent is not None:
        if getattr(parent, "name", None) in labels:
            return str(parent.name)
        parent = getattr(parent, "cpu_parent", None)
    return None


@torch.inference_mode()
def profile_kernel_counts(
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: FunctionalMamba2Backbone,
    frames: tuple[torch.Tensor, ...],
) -> list[int]:
    state = temporal.new_functional_state(1)
    for frame in frames[:LATENCY_WARMUP_FRAMES]:
        descriptor = tokenizer.encode_frame(frame)
        _, state = temporal.functional_step(descriptor, state)
    torch.cuda.synchronize(torch.device("cuda", 0))
    labels = [f"functional_latency_frame_{index + 1}" for index in range(LATENCY_MEASURED_FRAMES)]
    label_set = set(labels)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for label, frame in zip(
            labels,
            frames[LATENCY_WARMUP_FRAMES:],
            strict=True,
        ):
            torch.cuda.synchronize(torch.device("cuda", 0))
            with torch.profiler.record_function(label):
                descriptor = tokenizer.encode_frame(frame)
                _, state = temporal.functional_step(descriptor, state)
            torch.cuda.synchronize(torch.device("cuda", 0))
    counts = {label: 0 for label in labels}
    unattributed_kernels = 0
    for event in profile.events():
        if not _is_cuda_kernel_event(event):
            continue
        label = _event_latency_label(event, label_set)
        if label is None:
            unattributed_kernels += 1
        else:
            counts[label] += 1
    if unattributed_kernels or any(counts[label] < 1 for label in labels):
        raise RuntimeError(
            "CUDA profiler could not attribute every functional latency sample "
            f"(unattributed={unattributed_kernels})"
        )
    if state.observations != INPUT_SHAPE[0]:
        raise RuntimeError("kernel-count replay did not cover the full trajectory")
    return [counts[label] for label in labels]


@torch.inference_mode()
def benchmark_latency(
    codec: np.ndarray,
    *,
    expected_tokenizer_sha256: str,
    expected_temporal_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    device = torch.device("cuda", 0)
    raw = torch.from_numpy(codec).unsqueeze(0).to(device=device, dtype=torch.float32)
    frames = tuple(raw.unbind(dim=1))
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device=device, dtype=torch.float32)
    temporal = (
        FunctionalMamba2Backbone(seed=SEED)
        .eval()
        .to(
            device=device,
            dtype=torch.float32,
        )
    )
    identities = {
        "tokenizer_state_dict_sha256": state_dict_sha256(tokenizer),
        "temporal_state_dict_sha256": state_dict_sha256(temporal),
    }
    if (
        identities["tokenizer_state_dict_sha256"] != expected_tokenizer_sha256
        or identities["temporal_state_dict_sha256"] != expected_temporal_sha256
    ):
        raise RuntimeError("latency modules do not match the passed correctness weights")

    state = temporal.new_functional_state(1)
    state_allocated_before_timing = True
    host_samples = []
    cuda_samples = []
    state_bytes = []
    all_outputs_finite = True
    measured_frames = []
    for frame_index, frame in enumerate(frames, start=1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        host_start = time.perf_counter_ns()
        start.record()
        descriptor = tokenizer.encode_frame(frame)
        output, state = temporal.functional_step(descriptor, state)
        end.record()
        torch.cuda.synchronize(device)
        host_ms = (time.perf_counter_ns() - host_start) / 1_000_000
        cuda_ms = float(start.elapsed_time(end))
        all_outputs_finite = all_outputs_finite and bool(torch.isfinite(output).all())
        if frame_index > LATENCY_WARMUP_FRAMES:
            measured_frames.append(frame_index)
            host_samples.append(host_ms)
            cuda_samples.append(cuda_ms)
            state_bytes.append(state.tensor_nbytes())
    if len(host_samples) != LATENCY_MEASURED_FRAMES or state.observations != INPUT_SHAPE[0]:
        raise RuntimeError("latency trajectory warm-up or measured frame count changed")
    kernel_counts = profile_kernel_counts(tokenizer, temporal, frames)
    return {
        "boundary": FROZEN_PROTOCOL["latency"]["boundary"],
        "input_resident_on_gpu": raw.device == device,
        "state_allocated_before_timing": state_allocated_before_timing,
        "pre_and_post_sample_cuda_synchronize": True,
        "no_grad_execution": True,
        "warmup_frames": LATENCY_WARMUP_FRAMES,
        "measurements": LATENCY_MEASURED_FRAMES,
        "measured_frame_indices": measured_frames,
        "host_wall_samples_ms": host_samples,
        "host_wall_p50_ms": statistics.median(host_samples),
        "host_wall_p95_ms": percentile(host_samples, 95),
        "cuda_event_samples_ms": cuda_samples,
        "cuda_event_p50_ms": statistics.median(cuda_samples),
        "cuda_event_p95_ms": percentile(cuda_samples, 95),
        "kernel_count_samples": kernel_counts,
        "kernel_count_profiled_on_separate_same_trajectory_replay": True,
        "state_logical_bytes_samples": state_bytes,
        "state_logical_bytes_constant": len(set(state_bytes)) == 1,
        "output_finite": all_outputs_finite,
        "final_state_observations": state.observations,
    }, identities


def evaluate_latency_gates(
    latency: dict[str, Any],
    identities: dict[str, str],
    expected_identities: dict[str, str],
    telemetry_before: dict[str, Any],
    telemetry_after: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, bool]:
    return {
        "correctness_predecessor_passed": True,
        "source_input_dependency_runtime_identity": True,
        "weights_match_correctness_result": identities == expected_identities,
        "resident_input_and_preallocated_initial_state": latency["input_resident_on_gpu"]
        and latency["state_allocated_before_timing"],
        "exact_warmup20_then_905_measured_frames": (
            latency["warmup_frames"] == LATENCY_WARMUP_FRAMES
            and latency["measurements"] == LATENCY_MEASURED_FRAMES
            and latency["measured_frame_indices"]
            == list(range(LATENCY_WARMUP_FRAMES + 1, INPUT_SHAPE[0] + 1))
        ),
        "every_sample_synchronized_and_has_host_cuda_kernel_state_metrics": (
            latency["pre_and_post_sample_cuda_synchronize"]
            and len(latency["host_wall_samples_ms"]) == LATENCY_MEASURED_FRAMES
            and len(latency["cuda_event_samples_ms"]) == LATENCY_MEASURED_FRAMES
            and len(latency["kernel_count_samples"]) == LATENCY_MEASURED_FRAMES
            and all(count > 0 for count in latency["kernel_count_samples"])
            and len(latency["state_logical_bytes_samples"]) == LATENCY_MEASURED_FRAMES
        ),
        "host_wall_p95_strictly_below_5ms": latency["host_wall_p95_ms"] < MAX_HOST_P95_MS,
        "outputs_finite_and_state_bytes_constant": latency["output_finite"]
        and latency["state_logical_bytes_constant"],
        "gpu0_has_no_other_compute_process": not telemetry_before["gpu0_other_compute_processes"]
        and not telemetry_after["gpu0_other_compute_processes"],
        "native_vla_modules_absent": not execution["native_vla_modules"]["before"]
        and not execution["native_vla_modules"]["after"],
        "optimizer_steps_zero": not execution["optimizer"]["constructor_calls"]
        and execution["optimizer"]["steps_observed"] == 0,
    }


def run_latency(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
    *,
    replicate: int,
    correctness_path: Path,
    correctness_sha256: str,
    predecessor_path: Path | None,
    predecessor_sha256: str | None,
) -> dict[str, Any]:
    if replicate not in LATENCY_RUN_IDS:
        raise ValueError("latency replicate must be 1, 2, or 3")
    started = time.monotonic()
    artifact_bytes_before = directory_bytes(ARTIFACT_ROOT)
    native_before = native_vla_modules_loaded()
    source = verify_source(plan["source"]["commit"])
    correctness, correctness_identity = verify_passed_result(
        correctness_path,
        correctness_sha256,
        expected_path=ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json",
        expected_run_id=CORRECTNESS_RUN_ID,
        plan_identity=plan_identity,
        source_commit=source["commit"],
    )
    predecessor_identity = None
    if replicate == 1:
        if predecessor_path is not None or predecessor_sha256 is not None:
            raise ValueError("latency replicate A must not receive a latency predecessor")
    else:
        if predecessor_path is None or predecessor_sha256 is None:
            raise ValueError("latency replicate B/C requires its preceding latency result")
        _, predecessor_identity = verify_passed_result(
            predecessor_path,
            predecessor_sha256,
            expected_path=ARTIFACT_ROOT / LATENCY_RUN_IDS[replicate - 1] / "RESULT.json",
            expected_run_id=LATENCY_RUN_IDS[replicate - 1],
            plan_identity=plan_identity,
            source_commit=source["commit"],
        )
    prior_evidence = verify_prior_evidence()
    dependencies = verify_dependencies()
    codec, input_evidence = load_codec_input()
    environment = verify_environment()
    expected_identities = {
        "tokenizer_state_dict_sha256": correctness["numerical"]["tokenizer"]["state_dict_sha256"],
        "temporal_state_dict_sha256": correctness["numerical"]["parameter_contract"][
            "temporal_state_dict_sha256"
        ],
    }
    telemetry_before = gpu_telemetry()
    with optimizer_construction_guard() as optimizer_evidence:
        latency, identities = benchmark_latency(
            codec,
            expected_tokenizer_sha256=expected_identities["tokenizer_state_dict_sha256"],
            expected_temporal_sha256=expected_identities["temporal_state_dict_sha256"],
        )
    telemetry_after = gpu_telemetry()
    execution = {
        "native_vla_modules": {
            "before": native_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
    }
    gates = evaluate_latency_gates(
        latency,
        identities,
        expected_identities,
        telemetry_before,
        telemetry_after,
        execution,
    )
    wall_seconds = time.monotonic() - started
    gates["wall_time_at_most_10min"] = wall_seconds <= MAX_WALL_SECONDS
    gates["artifact_total_at_most_256mib"] = True
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": LATENCY_RUN_IDS[replicate],
        "replicate": replicate,
        "source_commit": source["commit"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source": source,
        "plan": plan_identity,
        "correctness_predecessor": correctness_identity,
        "latency_predecessor": predecessor_identity,
        "prior_evidence": prior_evidence,
        "input": input_evidence,
        "dependencies": dependencies,
        "environment": environment,
        "protocol": FROZEN_PROTOCOL["latency"],
        "thresholds": {"host_p95_strict_ms": MAX_HOST_P95_MS},
        "process": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "argv": sys.argv,
        },
        "telemetry": {"before": telemetry_before, "after": telemetry_after},
        "parameter_identity": identities,
        "latency": latency,
        "execution_evidence": execution,
        "gates": gates,
        "optimizer_steps": optimizer_evidence["steps_observed"],
        "vla_model_instantiated": bool(
            execution["native_vla_modules"]["before"] or execution["native_vla_modules"]["after"]
        ),
        "next_latency_allowed": all(gates.values()) and replicate < 3,
        "aggregate_allowed": all(gates.values()) and replicate == 3,
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "fixed-weight functional transition latency only",
    }
    _finalize_artifact_gate(
        result,
        artifact_bytes_before=artifact_bytes_before,
        gate_name="artifact_total_at_most_256mib",
    )
    result["next_latency_allowed"] = result["status"] == "passed" and replicate < 3
    result["aggregate_allowed"] = result["status"] == "passed" and replicate == 3
    return result


def evaluate_aggregate_gates(
    correctness: dict[str, Any],
    latencies: list[dict[str, Any]],
    latency_identities: list[dict[str, Any]],
) -> dict[str, bool]:
    expected_runs = [LATENCY_RUN_IDS[index] for index in (1, 2, 3)]
    unique_pids = {result["process"]["pid"] for result in latencies}
    correctness_sha = correctness["result_identity"]["sha256"]
    chain_valid = (
        latencies[0]["latency_predecessor"] is None
        and latencies[1]["latency_predecessor"]["sha256"] == latency_identities[0]["sha256"]
        and latencies[2]["latency_predecessor"]["sha256"] == latency_identities[1]["sha256"]
        and all(
            result["correctness_predecessor"]["sha256"] == correctness_sha for result in latencies
        )
    )
    parameter_identities = [result["parameter_identity"] for result in latencies]
    return {
        "correctness_result_passed_all_gates": correctness["status"] == "passed"
        and all(correctness["gates"].values()),
        "exact_three_ordered_latency_results_passed": (
            [result["run_id"] for result in latencies] == expected_runs
            and [result["replicate"] for result in latencies] == [1, 2, 3]
            and all(result["status"] == "passed" for result in latencies)
            and all(all(result["gates"].values()) for result in latencies)
        ),
        "latency_results_form_required_predecessor_chain": chain_valid,
        "three_latency_python_processes_are_distinct": len(unique_pids) == 3,
        "all_three_host_wall_p95_strictly_below_5ms": all(
            result["latency"]["host_wall_p95_ms"] < MAX_HOST_P95_MS for result in latencies
        ),
        "all_three_have_exact_warmup_and_measurement_contract": all(
            result["latency"]["warmup_frames"] == LATENCY_WARMUP_FRAMES
            and result["latency"]["measurements"] == LATENCY_MEASURED_FRAMES
            and len(result["latency"]["host_wall_samples_ms"]) == LATENCY_MEASURED_FRAMES
            and len(result["latency"]["cuda_event_samples_ms"]) == LATENCY_MEASURED_FRAMES
            and len(result["latency"]["kernel_count_samples"]) == LATENCY_MEASURED_FRAMES
            and len(result["latency"]["state_logical_bytes_samples"]) == LATENCY_MEASURED_FRAMES
            for result in latencies
        ),
        "all_three_weights_match": parameter_identities[1:] == parameter_identities[:-1],
        "all_gpu0_other_compute_process_lists_empty": all(
            not result["telemetry"][phase]["gpu0_other_compute_processes"]
            for result in latencies
            for phase in ("before", "after")
        ),
        "all_component_runs_vla_free_and_zero_optimizer": (
            correctness["vla_model_instantiated"] is False
            and correctness["optimizer_steps"] == 0
            and all(result["vla_model_instantiated"] is False for result in latencies)
            and all(result["optimizer_steps"] == 0 for result in latencies)
        ),
        "each_gpu_component_wall_at_most_10min": correctness["wall_seconds"] <= MAX_WALL_SECONDS
        and all(result["wall_seconds"] <= MAX_WALL_SECONDS for result in latencies),
    }


def run_aggregate(
    plan: dict[str, Any],
    plan_identity: dict[str, Any],
    *,
    correctness_path: Path,
    correctness_sha256: str,
    latency_paths: tuple[Path, Path, Path],
    latency_sha256s: tuple[str, str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_bytes_before = directory_bytes(ARTIFACT_ROOT)
    source = verify_source(plan["source"]["commit"])
    correctness, correctness_identity = verify_passed_result(
        correctness_path,
        correctness_sha256,
        expected_path=ARTIFACT_ROOT / CORRECTNESS_RUN_ID / "RESULT.json",
        expected_run_id=CORRECTNESS_RUN_ID,
        plan_identity=plan_identity,
        source_commit=source["commit"],
    )
    correctness["result_identity"] = correctness_identity
    latencies = []
    latency_identities = []
    for replicate, (path, digest) in enumerate(
        zip(latency_paths, latency_sha256s, strict=True),
        start=1,
    ):
        result, identity = verify_passed_result(
            path,
            digest,
            expected_path=ARTIFACT_ROOT / LATENCY_RUN_IDS[replicate] / "RESULT.json",
            expected_run_id=LATENCY_RUN_IDS[replicate],
            plan_identity=plan_identity,
            source_commit=source["commit"],
        )
        latencies.append(result)
        latency_identities.append(identity)
    prior_evidence = verify_prior_evidence()
    dependencies = verify_dependencies()
    _, input_evidence = load_codec_input()
    gates = evaluate_aggregate_gates(correctness, latencies, latency_identities)
    native_modules = native_vla_modules_loaded()
    gates["aggregate_process_native_vla_modules_absent"] = not native_modules
    wall_seconds = time.monotonic() - started
    gates["aggregate_wall_time_at_most_120s"] = wall_seconds <= 120
    gates["artifact_total_at_most_256mib"] = True
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": AGGREGATE_RUN_ID,
        "source_commit": source["commit"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source": source,
        "plan": plan_identity,
        "prior_evidence": prior_evidence,
        "dependencies": dependencies,
        "input": input_evidence,
        "component_results": {
            "correctness": correctness_identity,
            "latency": latency_identities,
        },
        "summary": {
            "latency_host_wall_p95_ms": {
                result["run_id"]: result["latency"]["host_wall_p95_ms"] for result in latencies
            },
            "latency_process_pids": [result["process"]["pid"] for result in latencies],
            "correctness_parity_max_abs": max(
                _parity_maxima(correctness["numerical"]["teacher_functional_parity"])
            ),
            "gradient_continuous_chunk_max_abs": correctness["numerical"]["gradients"][
                "continuous_vs_fixed_chunks"
            ]["max_abs"],
        },
        "gates": gates,
        "optimizer_steps": 0,
        "vla_model_instantiated": bool(native_modules),
        "matched_functional_streaming_cost_experiment_allowed": all(gates.values()),
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "fixed-weight functional transition engineering only",
    }
    _finalize_artifact_gate(
        result,
        artifact_bytes_before=artifact_bytes_before,
        gate_name="artifact_total_at_most_256mib",
    )
    result["matched_functional_streaming_cost_experiment_allowed"] = result["status"] == "passed"
    return result


def _write_mode_result(
    run_id: str,
    run: Any,
) -> int:
    output = ARTIFACT_ROOT / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result_path = output / "RESULT.json"
    started = time.monotonic()
    try:
        result = run()
        write_new(result_path, result)
    except Exception as error:
        write_new(
            result_path,
            {
                "status": "failed",
                "experiment_id": EXPERIMENT,
                "run_id": run_id,
                "error": f"{type(error).__name__}: {error}",
                "native_vla_evidence": native_vla_modules_loaded(),
                "optimizer_evidence": "unavailable because the run raised before completion",
                "wall_seconds": time.monotonic() - started,
                "training_allowed": False,
            },
        )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "passed" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-plan", type=Path)
    mode.add_argument("--correctness", action="store_true")
    mode.add_argument("--latency-a", action="store_true")
    mode.add_argument("--latency-b", action="store_true")
    mode.add_argument("--latency-c", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--correctness-result", type=Path)
    parser.add_argument("--expected-correctness-sha256")
    parser.add_argument("--predecessor-result", type=Path)
    parser.add_argument("--expected-predecessor-sha256")
    parser.add_argument("--latency-a-result", type=Path)
    parser.add_argument("--expected-latency-a-sha256")
    parser.add_argument("--latency-b-result", type=Path)
    parser.add_argument("--expected-latency-b-sha256")
    parser.add_argument("--latency-c-result", type=Path)
    parser.add_argument("--expected-latency-c-sha256")
    args = parser.parse_args()
    if args.prepare_plan is not None:
        if args.plan is not None or args.expected_plan_sha256 is not None:
            parser.error("--prepare-plan cannot be combined with resolved plan arguments")
        print(json.dumps(prepare_plan(args.prepare_plan), sort_keys=True), flush=True)
        return 0
    if args.plan is None or args.expected_plan_sha256 is None:
        parser.error("all execution modes require --plan and --expected-plan-sha256")
    plan, plan_identity = verify_plan(args.plan, args.expected_plan_sha256)
    if args.correctness:
        return _write_mode_result(
            CORRECTNESS_RUN_ID,
            lambda: run_correctness(plan, plan_identity),
        )
    if args.latency_a or args.latency_b or args.latency_c:
        if args.correctness_result is None or args.expected_correctness_sha256 is None:
            parser.error("latency modes require correctness result path and SHA-256")
        replicate = 1 if args.latency_a else 2 if args.latency_b else 3
        return _write_mode_result(
            LATENCY_RUN_IDS[replicate],
            lambda: run_latency(
                plan,
                plan_identity,
                replicate=replicate,
                correctness_path=args.correctness_result,
                correctness_sha256=args.expected_correctness_sha256,
                predecessor_path=args.predecessor_result,
                predecessor_sha256=args.expected_predecessor_sha256,
            ),
        )
    aggregate_values = (
        args.correctness_result,
        args.expected_correctness_sha256,
        args.latency_a_result,
        args.expected_latency_a_sha256,
        args.latency_b_result,
        args.expected_latency_b_sha256,
        args.latency_c_result,
        args.expected_latency_c_sha256,
    )
    if any(value is None for value in aggregate_values):
        parser.error("aggregate requires correctness and all three latency paths and SHA-256s")
    return _write_mode_result(
        AGGREGATE_RUN_ID,
        lambda: run_aggregate(
            plan,
            plan_identity,
            correctness_path=args.correctness_result,
            correctness_sha256=args.expected_correctness_sha256,
            latency_paths=(
                args.latency_a_result,
                args.latency_b_result,
                args.latency_c_result,
            ),
            latency_sha256s=(
                args.expected_latency_a_sha256,
                args.expected_latency_b_sha256,
                args.expected_latency_c_sha256,
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
