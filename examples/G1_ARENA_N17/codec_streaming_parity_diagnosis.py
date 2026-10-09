#!/usr/bin/env python3
"""Run the fixed-weight Arena Codec streaming parity diagnosis."""

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
    LAYERS,
    MAMBA2_CONFIG,
    SEED,
    WIDTH,
    ArenaCodecSpatialTokenizer,
    CausalGRUBackbone,
    CausalMamba2Backbone,
)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "g1-arena-codec-streaming-parity-diagnosis-r1"
RUN_ID = "fixed-weight-gpu0-r2"
BASE_SOURCE_COMMIT = "8dd32c8538d758874697e6cbc4086824c6c932ff"
ARTIFACT_ROOT = Path(
    "/home/sl/research/icra27-artifacts/g1-arena-codec-streaming-parity-diagnosis-r1"
)
R1_ARTIFACT_ROOT = Path("/home/sl/research/icra27-artifacts/g1-arena-codec-streaming-backbone-r1")
R1_PLAN_PATH = R1_ARTIFACT_ROOT / "plan-r1/PLAN.json"
R1_RESULT_PATH = R1_ARTIFACT_ROOT / "untrained-gpu1-r1/RESULT.json"
R1_PLAN_BYTES = 4_001
R1_RESULT_BYTES = 106_681
R1_PLAN_SHA256 = "917b0177884f114e480442275c9e56ffda2f147a5e39a26a7108b640c7b59356"
R1_RESULT_SHA256 = "8957277625c168afd8b229b5c641702cfae97cb29a3fc6a3ca489b26336dbcd6"
R1_MAMBA_T925_MAX_ABS = 0.00025087594985961914
R1_GRU_T925_MAX_ABS = 0.0007979720830917358

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

DEPENDENCY_MANIFEST_PATH = R1_ARTIFACT_ROOT / "dependencies/DEPENDENCY.json"
OFFICIAL_WHEEL_PATH = (
    R1_ARTIFACT_ROOT
    / "dependencies/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
OVERLAY_ROOT = R1_ARTIFACT_ROOT / "dependencies/python-overlay"
OVERLAY_RECORD_PATH = OVERLAY_ROOT / "mamba_ssm-2.3.2.post1.dist-info/RECORD"
MAMBA2_FILE_PATH = OVERLAY_ROOT / "mamba_ssm/modules/mamba2.py"
SELECTIVE_SCAN_EXTENSION_PATH = OVERLAY_ROOT / "selective_scan_cuda.cpython-312-x86_64-linux-gnu.so"

PREFIX_LENGTHS = (1, 12, 16, 32, 64, 128, 256, 512, 925)
STREAM_PARTITION_SIZE = 127
GRADIENT_LENGTH = 128
CAUSAL_PREFIX_LENGTH = 128
CAUSAL_TOTAL_LENGTH = 256
MAX_PARITY_ABS = 1e-4
MAX_CAUSAL_ABS = 1e-5
R1_REPRODUCTION_ATOL = 1e-6
MAX_WALL_SECONDS = 10 * 60
MAX_ARTIFACT_BYTES = 256 << 20
CONDITION_NAMES = (
    "mamba_chunk16_current",
    "mamba_chunk1_candidate",
    "gru_batched_current",
    "gru_explicit_candidate",
)
CURRENT_CONDITIONS = ("mamba_chunk16_current", "gru_batched_current")
CANDIDATE_CONDITIONS = ("mamba_chunk1_candidate", "gru_explicit_candidate")
SOURCE_IDENTITY_PATHS = (
    "gr00t/__init__.py",
    "gr00t/experimental/__init__.py",
    "gr00t/experimental/arena_codec_streaming_backbone.py",
    "examples/G1_ARENA_N17/codec_streaming_parity_diagnosis.py",
    "tests/gr00t/experimental/test_arena_codec_streaming_parity_diagnosis.py",
    "tests/gr00t/experimental/test_arena_codec_streaming_parity_target.py",
    "tests/examples/test_g1_arena_codec_streaming_parity_diagnosis.py",
)
NATIVE_VLA_MODULE_PREFIXES = (
    "gr00t.model.gr00t_n1d7",
    "gr00t.model.modules.dit",
    "gr00t.model.modules.qwen3_backbone",
    "gr00t.policy",
)

FROZEN_INPUT = {
    "path": str(INPUT_PATH),
    "bytes": INPUT_BYTES,
    "sha256": INPUT_SHA256,
    "field": INPUT_FIELD,
    "shape": list(INPUT_SHAPE),
    "dtype": "float32",
}
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
    "diagnostic_dtype": "float32",
}
FROZEN_PROTOCOL = {
    "conditions": list(CONDITION_NAMES),
    "prefix_lengths": list(PREFIX_LENGTHS),
    "seed": SEED,
    "layers": LAYERS,
    "width": WIDTH,
    "mamba_full_chunk_sizes": {"current": 16, "candidate": 1},
    "gru_full_paths": {
        "current": "nn.GRU_batched_sequence",
        "candidate": "differentiable_per_timestep_nn.GRU_recurrence",
    },
    "step_paths": {"mamba": "official_Mamba2.step", "gru": "per_timestep_nn.GRU"},
    "persistent_stream_partition_size": STREAM_PARTITION_SIZE,
    "repeat_count": 2,
    "gradient_length": GRADIENT_LENGTH,
    "optimizer_steps": 0,
    "vla_model_instantiated": False,
}
FROZEN_THRESHOLDS = {
    "full_step_max_abs": MAX_PARITY_ABS,
    "r1_reproduction_atol": R1_REPRODUCTION_ATOL,
    "r1_mamba_t925_max_abs": R1_MAMBA_T925_MAX_ABS,
    "r1_gru_t925_max_abs": R1_GRU_T925_MAX_ABS,
    "suffix_prefix_max_abs": MAX_CAUSAL_ABS,
    "repeat_exact": True,
    "continuous_chunk127_exact": True,
    "reset_exact": True,
    "state_dict_tensor_equal": True,
    "candidate_gradient_finite_nonzero": True,
}
FROZEN_RESOURCES = {
    "gpu_physical_id": 0,
    "timeout_seconds": MAX_WALL_SECONDS,
    "storage_limit_bytes": MAX_ARTIFACT_BYTES,
}
FROZEN_R1_EVIDENCE = {
    "plan": {"path": str(R1_PLAN_PATH), "bytes": R1_PLAN_BYTES, "sha256": R1_PLAN_SHA256},
    "result": {
        "path": str(R1_RESULT_PATH),
        "bytes": R1_RESULT_BYTES,
        "sha256": R1_RESULT_SHA256,
    },
}
FAILED_ENGINEERING_ATTEMPT = {
    "plan": {
        "path": str(ARTIFACT_ROOT / "plan-r1/PLAN.json"),
        "bytes": 4_881,
        "sha256": "234358bd44e1a110e7988878a090d61f3fa94dcf167b22224136a86aba3bb34e",
    },
    "result": {
        "path": str(ARTIFACT_ROOT / "fixed-weight-gpu0-r1/RESULT.json"),
        "bytes": 664,
        "sha256": "83e68bfd7d018e8e931f29af8b71e4e70264c470e1a72e3d6d49bc9599de433f",
    },
    "required_status": "failed",
    "required_error": (
        "ValueError: diagnosis source is dirty: "
        "['tests/gr00t/experimental/test_arena_codec_streaming_parity_target.py']"
    ),
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
        "guard": "torch.optim.Optimizer.__init__ blocked for the measured diagnosis",
    }

    def blocked_init(instance, *args, **kwargs):
        del args, kwargs
        evidence["constructor_calls"].append(type(instance).__qualname__)
        raise RuntimeError("optimizer construction is forbidden in this fixed-weight diagnosis")

    optimizer.__init__ = blocked_init
    try:
        yield evidence
    finally:
        optimizer.__init__ = original_init
        evidence["steps_observed"] = 0 if not evidence["constructor_calls"] else None


def _verify_material_spec(spec: dict[str, Any], *, expected_path: Path, label: str) -> Path:
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


def resolved_plan(source_commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "run_id": RUN_ID,
        "source": {
            "root": str(REPO),
            "commit": source_commit,
            "base_commit": BASE_SOURCE_COMMIT,
        },
        "r1_evidence": copy.deepcopy(FROZEN_R1_EVIDENCE),
        "failed_engineering_attempt": copy.deepcopy(FAILED_ENGINEERING_ATTEMPT),
        "input": copy.deepcopy(FROZEN_INPUT),
        "dependency": copy.deepcopy(FROZEN_DEPENDENCY),
        "runtime": copy.deepcopy(FROZEN_RUNTIME),
        "protocol": copy.deepcopy(FROZEN_PROTOCOL),
        "thresholds": copy.deepcopy(FROZEN_THRESHOLDS),
        "resources": copy.deepcopy(FROZEN_RESOURCES),
        "output": {
            "root": str(ARTIFACT_ROOT / RUN_ID),
            "result_file": "RESULT.json",
        },
    }


def write_new(path: Path, payload: dict[str, Any]) -> int:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("JSON artifact exceeds 256 MiB")
    if directory_bytes(ARTIFACT_ROOT) + len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("experiment artifact root would exceed 256 MiB")
    with path.open("xb") as stream:
        stream.write(encoded)
    return len(encoded)


def prepare_plan(path: Path) -> dict[str, Any]:
    expected = ARTIFACT_ROOT / "plan-r2/PLAN.json"
    if path.resolve() != expected.resolve():
        raise ValueError(f"resolved plan path must be {expected}")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    plan = resolved_plan(commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = write_new(path, plan)
    return {"path": str(path.resolve()), "bytes": size, "sha256": sha256(path)}


def verify_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("resolved plan must be inside the diagnosis artifact root")
    actual_sha256 = sha256(resolved)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ValueError("resolved plan identity changed")
    plan = read_json(resolved)
    if set(plan) != set(resolved_plan("0" * 40)):
        raise ValueError("resolved plan top-level schema changed")
    source = plan.get("source", {})
    commit = source.get("commit")
    try:
        int(commit, 16)
    except (TypeError, ValueError) as error:
        raise ValueError("resolved source commit is not a Git object id") from error
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("resolved source commit must contain 40 hex characters")
    expected = resolved_plan(commit)
    if plan != expected:
        raise ValueError("resolved plan contract changed")
    return plan, {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual_sha256}


def _source_file_identity(path: Path, commit: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(REPO.resolve()):
        raise ValueError(f"imported source escaped repository: {resolved}")
    relative = resolved.relative_to(REPO.resolve()).as_posix()
    committed_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", f"{commit}:{relative}"], text=True
    ).strip()
    working_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "hash-object", str(resolved)], text=True
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
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != expected_commit:
        raise ValueError("diagnosis source commit changed")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, commit],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("diagnosis source is not based on the frozen r1 source commit")
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
        raise ValueError(f"diagnosis source is dirty: {sorted(dirty)}")
    expected_module = (REPO / "gr00t/experimental/arena_codec_streaming_backbone.py").resolve()
    actual_module = Path(arena_codec_module.__file__).resolve()
    if actual_module != expected_module:
        raise ValueError("imported Arena codec module escaped the diagnosis source root")
    entries = {
        "arena_codec_module": _source_file_identity(actual_module, commit),
        "gr00t_package": _source_file_identity(Path(gr00t.__file__), commit),
        "experimental_package": _source_file_identity(Path(gr00t.experimental.__file__), commit),
        "runner": _source_file_identity(Path(__file__), commit),
    }
    return {"root": str(REPO.resolve()), "commit": commit, "entries": entries}


def verify_r1_evidence() -> dict[str, Any]:
    _verify_material_spec(
        FROZEN_R1_EVIDENCE["plan"], expected_path=R1_PLAN_PATH, label="r1 resolved plan"
    )
    _verify_material_spec(
        FROZEN_R1_EVIDENCE["result"], expected_path=R1_RESULT_PATH, label="r1 result"
    )
    plan = read_json(R1_PLAN_PATH)
    result = read_json(R1_RESULT_PATH)
    if (
        plan.get("experiment_id") != "g1-arena-codec-streaming-backbone-r1"
        or plan.get("source", {}).get("commit") != BASE_SOURCE_COMMIT
        or plan.get("input") != FROZEN_INPUT
        or plan.get("dependency") != FROZEN_DEPENDENCY
    ):
        raise ValueError("r1 resolved plan contract changed")
    parity = result.get("numerical", {}).get("parity", {})
    if (
        result.get("experiment_id") != "g1-arena-codec-streaming-backbone-r1"
        or result.get("source_commit") != BASE_SOURCE_COMMIT
        or result.get("optimizer_steps") != 0
        or result.get("vla_model_instantiated") is not False
        or parity.get("mamba2", {}).get("full_vs_step_max_abs") != R1_MAMBA_T925_MAX_ABS
        or parity.get("gru", {}).get("full_vs_step_max_abs") != R1_GRU_T925_MAX_ABS
    ):
        raise ValueError("r1 result evidence contract changed")
    return {
        "plan": FROZEN_R1_EVIDENCE["plan"],
        "result": FROZEN_R1_EVIDENCE["result"],
        "source_commit": BASE_SOURCE_COMMIT,
        "t925_full_step_max_abs": {
            "mamba2": R1_MAMBA_T925_MAX_ABS,
            "gru": R1_GRU_T925_MAX_ABS,
        },
    }


def verify_failed_engineering_attempt() -> dict[str, Any]:
    _verify_material_spec(
        FAILED_ENGINEERING_ATTEMPT["plan"],
        expected_path=ARTIFACT_ROOT / "plan-r1/PLAN.json",
        label="failed engineering attempt plan",
    )
    result_path = _verify_material_spec(
        FAILED_ENGINEERING_ATTEMPT["result"],
        expected_path=ARTIFACT_ROOT / "fixed-weight-gpu0-r1/RESULT.json",
        label="failed engineering attempt result",
    )
    result = read_json(result_path)
    if (
        result.get("status") != FAILED_ENGINEERING_ATTEMPT["required_status"]
        or result.get("error") != FAILED_ENGINEERING_ATTEMPT["required_error"]
        or result.get("training_allowed") is not False
    ):
        raise ValueError("failed engineering attempt evidence changed")
    return {
        "plan": FAILED_ENGINEERING_ATTEMPT["plan"],
        "result": FAILED_ENGINEERING_ATTEMPT["result"],
        "failure_before_model_probe": True,
        "error": result["error"],
    }


def verify_dependencies() -> dict[str, Any]:
    paths = {
        "manifest": _verify_material_spec(
            FROZEN_DEPENDENCY["manifest"],
            expected_path=DEPENDENCY_MANIFEST_PATH,
            label="dependency manifest",
        ),
        "official_wheel": _verify_material_spec(
            FROZEN_DEPENDENCY["official_wheel"],
            expected_path=OFFICIAL_WHEEL_PATH,
            label="official Mamba wheel",
        ),
        "overlay_record": _verify_material_spec(
            FROZEN_DEPENDENCY["overlay_record"],
            expected_path=OVERLAY_RECORD_PATH,
            label="overlay RECORD",
        ),
        "mamba_file": _verify_material_spec(
            FROZEN_DEPENDENCY["mamba_file"],
            expected_path=MAMBA2_FILE_PATH,
            label="Mamba2 source",
        ),
        "selective_scan_extension": _verify_material_spec(
            FROZEN_DEPENDENCY["selective_scan_extension"],
            expected_path=SELECTIVE_SCAN_EXTENSION_PATH,
            label="selective scan CUDA extension",
        ),
    }
    manifest = read_json(paths["manifest"])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != "g1-arena-codec-streaming-backbone-r1"
        or manifest.get("overlay", {}).get("mamba_version") != "2.3.2.post1"
        or manifest.get("overlay", {}).get("causal_conv1d_present") is not False
        or Path(manifest.get("overlay", {}).get("path", "")).resolve() != OVERLAY_ROOT.resolve()
        or Path(manifest.get("overlay", {}).get("mamba2_file", "")).resolve()
        != MAMBA2_FILE_PATH.resolve()
        or manifest.get("overlay", {}).get("mamba2_sha256")
        != FROZEN_DEPENDENCY["mamba_file"]["sha256"]
        or Path(manifest.get("wheel", {}).get("path", "")).resolve()
        != OFFICIAL_WHEEL_PATH.resolve()
        or manifest.get("wheel", {}).get("bytes") != FROZEN_DEPENDENCY["official_wheel"]["bytes"]
        or manifest.get("wheel", {}).get("sha256") != FROZEN_DEPENDENCY["official_wheel"]["sha256"]
    ):
        raise ValueError("dependency manifest contract changed")
    return {
        "materials": FROZEN_DEPENDENCY,
        "overlay_closure": verify_overlay_record(paths["overlay_record"], OVERLAY_ROOT),
    }


def load_codec_input() -> tuple[np.ndarray, dict[str, Any]]:
    resolved = INPUT_PATH.resolve()
    if not resolved.is_file():
        raise ValueError(f"Arena Codec IR input is missing: {resolved}")
    if resolved.stat().st_size != INPUT_BYTES or sha256(resolved) != INPUT_SHA256:
        raise ValueError("Arena Codec IR input identity changed")
    with np.load(resolved, allow_pickle=False) as archive:
        present = tuple(archive.files)
        if set(present) != set(EXPECTED_ARCHIVE_FIELDS) or len(present) != len(
            EXPECTED_ARCHIVE_FIELDS
        ):
            raise ValueError("Arena Codec IR archive field whitelist changed")
        codec = np.array(archive[INPUT_FIELD], copy=True)
    if codec.shape != INPUT_SHAPE or codec.dtype != np.float32 or not np.isfinite(codec).all():
        raise ValueError("codec_v1 shape, dtype, or finite contract changed")
    return codec, {
        **FROZEN_INPUT,
        "fields_present": list(present),
        "fields_consumed_by_backbone": [INPUT_FIELD],
        "fields_excluded_from_backbone": sorted(set(present) - {INPUT_FIELD}),
    }


def verify_environment() -> dict[str, Any]:
    import mamba_ssm
    import mamba_ssm.modules.mamba2 as official
    import mamba_ssm.ops.triton.ssd_combined as ssd_combined
    import selective_scan_cuda

    mamba_version = importlib.metadata.version("mamba-ssm")
    actual_mamba_file = Path(official.__file__).resolve()
    actual_extension_file = Path(selective_scan_cuda.__file__).resolve()
    actual_package_file = Path(mamba_ssm.__file__).resolve()
    actual_ssd_file = Path(ssd_combined.__file__).resolve()
    if official.selective_state_update is None:
        raise ValueError("Mamba2 persistent step op is unavailable")
    state_update_module = sys.modules[official.selective_state_update.__module__]
    actual_state_update_file = Path(state_update_module.__file__).resolve()
    if Path(sys.executable).resolve() != Path(FROZEN_RUNTIME["python_executable"]).resolve():
        raise ValueError("formal Python executable changed")
    if platform.python_version() != FROZEN_RUNTIME["python"]:
        raise ValueError("formal Python version changed")
    if (
        torch.__version__ != FROZEN_RUNTIME["torch"]
        or torch.version.cuda != FROZEN_RUNTIME["torch_cuda"]
    ):
        raise ValueError("formal Torch or CUDA runtime changed")
    if torch.compiled_with_cxx11_abi() is not FROZEN_RUNTIME["cxx11_abi"]:
        raise ValueError("Torch CXX11 ABI contract changed")
    if mamba_version != FROZEN_RUNTIME["mamba_ssm"]:
        raise ValueError("official Mamba version changed")
    if actual_mamba_file != MAMBA2_FILE_PATH.resolve():
        raise ValueError("Mamba2 import escaped the verified overlay")
    if actual_extension_file != SELECTIVE_SCAN_EXTENSION_PATH.resolve():
        raise ValueError("selective scan extension escaped the verified overlay")
    for path in (actual_package_file, actual_ssd_file, actual_state_update_file):
        if not path.is_relative_to(OVERLAY_ROOT.resolve()):
            raise ValueError(f"critical Mamba import escaped the overlay: {path}")
    if official.causal_conv1d_fn is not None or official.causal_conv1d_update is not None:
        raise ValueError("frozen explicit Mamba path requires causal-conv1d to be absent")
    if official.mamba_chunk_scan_combined.__module__ != ssd_combined.__name__:
        raise ValueError("Mamba2 full scan op escaped the verified SSD implementation")
    if MAMBA2_CONFIG["chunk_size"] != 16 or MAMBA2_CONFIG["use_mem_eff_path"] is not False:
        raise ValueError("frozen Mamba2 current execution contract changed")
    if os.environ.get("PYTHONPATH") != FROZEN_RUNTIME["pythonpath"]:
        raise ValueError("formal PYTHONPATH contract changed")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ValueError("formal diagnosis must disable Python bytecode writes")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("formal diagnosis must expose only physical GPU0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("formal diagnosis requires exactly one visible CUDA device")
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
        "selective_scan_extension_file": str(actual_extension_file),
        "causal_conv1d_fn": None,
        "causal_conv1d_update": None,
        "mamba_current_chunk_size": MAMBA2_CONFIG["chunk_size"],
        "mamba_candidate_chunk_size": 1,
        "use_mem_eff_path": MAMBA2_CONFIG["use_mem_eff_path"],
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
    }


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        shape_bytes = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        for field in (
            name_bytes,
            dtype_bytes,
            shape_bytes,
            value.view(torch.uint8).numpy().tobytes(),
        ):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def state_dict_pair_report(current: nn.Module, candidate: nn.Module) -> dict[str, Any]:
    current_state = current.state_dict()
    candidate_state = candidate.state_dict()
    current_names = tuple(sorted(current_state))
    candidate_names = tuple(sorted(candidate_state))
    mismatches = []
    if current_names == candidate_names:
        mismatches = [
            name
            for name in current_names
            if not torch.equal(current_state[name], candidate_state[name])
        ]
    else:
        mismatches = sorted(set(current_names).symmetric_difference(candidate_names))
    current_hash = state_dict_sha256(current)
    candidate_hash = state_dict_sha256(candidate)
    return {
        "current_sha256": current_hash,
        "candidate_sha256": candidate_hash,
        "keys_equal": current_names == candidate_names,
        "tensor_count": len(current_names),
        "mismatched_tensors": mismatches,
        "all_tensors_equal": current_names == candidate_names and not mismatches,
        "hashes_equal": current_hash == candidate_hash,
    }


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max().item())


def _output_scale(value: torch.Tensor) -> dict[str, float]:
    floating = value.float()
    return {
        "max_abs": float(floating.abs().max().item()),
        "mean_abs": float(floating.abs().mean().item()),
        "rms": float(floating.square().mean().sqrt().item()),
    }


def condition_metrics(full: torch.Tensor, streamed: torch.Tensor) -> dict[str, Any]:
    if full.shape != streamed.shape or full.ndim != 3:
        raise ValueError("full and stream outputs must have the same [batch,time,width] shape")
    error = (full.float() - streamed.float()).abs()
    frame_max = error.amax(dim=(0, 2))
    frame_mean = error.mean(dim=(0, 2))
    frame_rms = error.square().mean(dim=(0, 2)).sqrt()
    crossings = torch.nonzero(frame_max > MAX_PARITY_ABS, as_tuple=False).flatten()
    prefix: dict[str, Any] = {}
    for length in PREFIX_LENGTHS:
        current_error = error[:, :length]
        full_final = full[:, length - 1].float()
        stream_final = streamed[:, length - 1].float()
        prefix[str(length)] = {
            "max_abs": float(current_error.max().item()),
            "mean_abs": float(current_error.mean().item()),
            "rms": float(current_error.square().mean().sqrt().item()),
            "final_token": {
                "max_abs": float((full_final - stream_final).abs().max().item()),
                "rms": float((full_final - stream_final).square().mean().sqrt().item()),
                "cosine": float(
                    F.cosine_similarity(full_final, stream_final, dim=-1).mean().item()
                ),
            },
            "output_scale": {
                "full": _output_scale(full[:, :length]),
                "stream": _output_scale(streamed[:, :length]),
            },
        }
    per_frame = [
        {
            "frame": index + 1,
            "max_abs": float(frame_max[index].item()),
            "mean_abs": float(frame_mean[index].item()),
            "rms": float(frame_rms[index].item()),
        }
        for index in range(int(full.shape[1]))
    ]
    return {
        "shape": list(full.shape),
        "dtype": str(full.dtype),
        "finite": bool(torch.isfinite(full).all() and torch.isfinite(streamed).all()),
        "prefix": prefix,
        "per_frame": per_frame,
        "first_frame_max_abs_above_1e-4": (
            int(crossings[0].item()) + 1 if crossings.numel() else None
        ),
        "output_scale": {"full": _output_scale(full), "stream": _output_scale(streamed)},
    }


@torch.inference_mode()
def _partitioned_stream(
    module: CausalMamba2Backbone | CausalGRUBackbone,
    descriptors: torch.Tensor,
) -> torch.Tensor:
    state = module.new_state(int(descriptors.shape[0]))
    outputs = []
    for start in range(0, int(descriptors.shape[1]), STREAM_PARTITION_SIZE):
        current, state = module.stream(descriptors[:, start : start + STREAM_PARTITION_SIZE], state)
        outputs.append(current)
    return torch.cat(outputs, dim=1)


@torch.inference_mode()
def evaluate_condition(
    module: CausalMamba2Backbone | CausalGRUBackbone,
    full_path: Callable[[torch.Tensor], torch.Tensor],
    descriptors: torch.Tensor,
    causal_original: torch.Tensor,
    causal_changed: torch.Tensor,
) -> dict[str, Any]:
    full_first = full_path(descriptors)
    full_second = full_path(descriptors)
    stream_first, used_state = module.stream(descriptors)
    stream_second, _ = module.stream(descriptors)
    partitioned = _partitioned_stream(module, descriptors)

    used_state.reset()
    reset_first, used_state = module.stream(descriptors[:, :GRADIENT_LENGTH], used_state)
    used_state.reset()
    reset_second, used_state = module.stream(descriptors[:, :GRADIENT_LENGTH], used_state)
    reset_fresh, _ = module.stream(descriptors[:, :GRADIENT_LENGTH])

    causal_first = full_path(causal_original)[:, :CAUSAL_PREFIX_LENGTH]
    causal_second = full_path(causal_changed)[:, :CAUSAL_PREFIX_LENGTH]
    report = condition_metrics(full_first, stream_first)
    report.update(
        {
            "repeat_determinism": {
                "full_exact": bool(torch.equal(full_first, full_second)),
                "stream_exact": bool(torch.equal(stream_first, stream_second)),
                "full_max_abs": _max_abs(full_first, full_second),
                "stream_max_abs": _max_abs(stream_first, stream_second),
            },
            "continuous_vs_chunk127": {
                "exact": bool(torch.equal(stream_first, partitioned)),
                "max_abs": _max_abs(stream_first, partitioned),
                "partition_size": STREAM_PARTITION_SIZE,
            },
            "reset": {
                "exact": bool(
                    torch.equal(reset_first, reset_second) and torch.equal(reset_first, reset_fresh)
                ),
                "first_vs_second_max_abs": _max_abs(reset_first, reset_second),
                "reset_vs_fresh_max_abs": _max_abs(reset_first, reset_fresh),
                "observations_after_second_reset_run": used_state.observations,
            },
            "causality": {
                "suffix_edit_prefix_max_abs": _max_abs(causal_first, causal_second),
                "prefix_length": CAUSAL_PREFIX_LENGTH,
                "total_length": CAUSAL_TOTAL_LENGTH,
            },
        }
    )
    return report


def _gradient_group_report(module: nn.Module) -> dict[str, Any]:
    parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    missing = [name for name, parameter in parameters if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in parameters
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    nonzero = [
        name
        for name, parameter in parameters
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
    ]
    squared_norm = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for _, parameter in parameters
        if parameter.grad is not None
    )
    return {
        "trainable_parameter_tensors": len(parameters),
        "gradient_tensors": len(parameters) - len(missing),
        "finite_gradient_tensors": len(parameters) - len(missing) - len(nonfinite),
        "nonzero_gradient_tensors": len(nonzero),
        "missing_gradient_parameters": missing,
        "nonfinite_gradient_parameters": nonfinite,
        "global_l2_norm": math.sqrt(squared_norm),
        "finite_nonzero": not missing and not nonfinite and bool(nonzero) and squared_norm > 0.0,
    }


def candidate_gradient_probe(
    raw: torch.Tensor,
    *,
    tokenizer: ArenaCodecSpatialTokenizer,
    temporal: CausalMamba2Backbone | CausalGRUBackbone,
    full_path: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    tokenizer.zero_grad(set_to_none=True)
    temporal.zero_grad(set_to_none=True)
    descriptors = tokenizer(raw[:, :GRADIENT_LENGTH])
    output = full_path(descriptors)
    loss = output.float().square().mean()
    loss.backward()
    report = {
        "length": GRADIENT_LENGTH,
        "loss": float(loss.detach().item()),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "tokenizer": _gradient_group_report(tokenizer),
        "temporal": _gradient_group_report(temporal),
    }
    tokenizer.zero_grad(set_to_none=True)
    temporal.zero_grad(set_to_none=True)
    return report


def numerical_probe(codec: np.ndarray) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    raw = torch.from_numpy(codec).unsqueeze(0).to(device=device, dtype=torch.float32)
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device)
    mamba_current = CausalMamba2Backbone(seed=SEED, chunk_size=16).eval().to(device)
    mamba_candidate = CausalMamba2Backbone(seed=SEED, chunk_size=1).eval().to(device)
    mamba_candidate.load_state_dict(mamba_current.state_dict(), strict=True)
    gru_current = CausalGRUBackbone(seed=SEED).eval().to(device)
    gru_candidate = CausalGRUBackbone(seed=SEED).eval().to(device)
    gru_candidate.load_state_dict(gru_current.state_dict(), strict=True)
    modules: dict[str, CausalMamba2Backbone | CausalGRUBackbone] = {
        "mamba_chunk16_current": mamba_current,
        "mamba_chunk1_candidate": mamba_candidate,
        "gru_batched_current": gru_current,
        "gru_explicit_candidate": gru_candidate,
    }
    if tuple(modules) != CONDITION_NAMES:
        raise RuntimeError("diagnosis condition set changed")
    weight_identity = {
        "mamba_current_vs_candidate": state_dict_pair_report(
            modules["mamba_chunk16_current"], modules["mamba_chunk1_candidate"]
        ),
        "gru_current_vs_candidate": state_dict_pair_report(
            modules["gru_batched_current"], modules["gru_explicit_candidate"]
        ),
    }
    with torch.inference_mode():
        descriptors = tokenizer(raw)
        causal_raw = raw[:, :CAUSAL_TOTAL_LENGTH].clone()
        causal_changed_raw = causal_raw.clone()
        causal_changed_raw[:, CAUSAL_PREFIX_LENGTH:] += 0.125
        causal_original = tokenizer(causal_raw)
        causal_changed = tokenizer(causal_changed_raw)
    if list(descriptors.shape) != [1, INPUT_SHAPE[0], WIDTH] or not bool(
        torch.isfinite(descriptors).all()
    ):
        raise RuntimeError("shared tokenizer output is invalid")
    full_paths: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "mamba_chunk16_current": modules["mamba_chunk16_current"].forward,
        "mamba_chunk1_candidate": modules["mamba_chunk1_candidate"].forward,
        "gru_batched_current": modules["gru_batched_current"].forward,
        "gru_explicit_candidate": modules["gru_explicit_candidate"].forward_explicit,
    }
    conditions = {
        name: evaluate_condition(
            modules[name],
            full_paths[name],
            descriptors,
            causal_original,
            causal_changed,
        )
        for name in CONDITION_NAMES
    }
    gradients = {
        name: candidate_gradient_probe(
            raw,
            tokenizer=tokenizer,
            temporal=modules[name],
            full_path=full_paths[name],
        )
        for name in CANDIDATE_CONDITIONS
    }
    torch.cuda.synchronize(device)
    return {
        "tokenizer": {
            "shape": list(descriptors.shape),
            "dtype": str(descriptors.dtype),
            "finite": bool(torch.isfinite(descriptors).all()),
            "state_dict_sha256": state_dict_sha256(tokenizer),
        },
        "weight_identity": weight_identity,
        "conditions": conditions,
        "candidate_gradients": gradients,
    }


def evaluate_gates(numerical: dict[str, Any], execution: dict[str, Any]) -> dict[str, bool]:
    conditions = numerical["conditions"]
    exact_condition_set = tuple(conditions) == CONDITION_NAMES
    shape_finite = exact_condition_set and all(
        report["shape"] == [1, INPUT_SHAPE[0], WIDTH] and report["finite"]
        for report in conditions.values()
    )
    repeat_exact = exact_condition_set and all(
        report["repeat_determinism"]["full_exact"] and report["repeat_determinism"]["stream_exact"]
        for report in conditions.values()
    )
    partition_exact = exact_condition_set and all(
        report["continuous_vs_chunk127"]["exact"]
        and report["continuous_vs_chunk127"]["partition_size"] == STREAM_PARTITION_SIZE
        for report in conditions.values()
    )
    reset_exact = exact_condition_set and all(
        report["reset"]["exact"] for report in conditions.values()
    )
    causal = exact_condition_set and all(
        report["causality"]["suffix_edit_prefix_max_abs"] <= MAX_CAUSAL_ABS
        for report in conditions.values()
    )
    mamba_current = conditions.get("mamba_chunk16_current", {}).get("prefix", {}).get("925", {})
    gru_current = conditions.get("gru_batched_current", {}).get("prefix", {}).get("925", {})
    current_reproduction = (
        abs(mamba_current.get("max_abs", float("inf")) - R1_MAMBA_T925_MAX_ABS)
        <= R1_REPRODUCTION_ATOL
        and abs(gru_current.get("max_abs", float("inf")) - R1_GRU_T925_MAX_ABS)
        <= R1_REPRODUCTION_ATOL
    )
    candidate_parity = all(
        conditions.get(name, {}).get("prefix", {}).get("925", {}).get("max_abs", float("inf"))
        <= MAX_PARITY_ABS
        for name in CANDIDATE_CONDITIONS
    )
    weights = numerical["weight_identity"]
    weight_identity = all(
        report["keys_equal"]
        and report["all_tensors_equal"]
        and report["hashes_equal"]
        and not report["mismatched_tensors"]
        for report in weights.values()
    )
    gradients = numerical["candidate_gradients"]
    candidate_gradients = set(gradients) == set(CANDIDATE_CONDITIONS) and all(
        report["length"] == GRADIENT_LENGTH
        and report["output_finite"]
        and report["tokenizer"]["finite_nonzero"]
        and report["temporal"]["finite_nonzero"]
        for report in gradients.values()
    )
    return {
        "exactly_four_frozen_conditions": exact_condition_set,
        "source_input_dependency_runtime_identity": True,
        "r1_current_t925_reproduced_within_1e-6": current_reproduction,
        "condition_repeat_full_stream_exact": repeat_exact,
        "continuous_vs_chunk127_stream_exact": partition_exact,
        "candidate_t925_full_step_at_most_1e-4": candidate_parity,
        "candidate_t128_tokenizer_temporal_gradients_finite_nonzero": candidate_gradients,
        "current_candidate_state_dicts_equal_and_hash_equal": weight_identity,
        "all_conditions_reset_exact": reset_exact,
        "all_conditions_suffix_causal": causal,
        "all_outputs_shape_and_finite": shape_finite,
        "native_vla_modules_absent": not (
            execution["native_vla_modules"]["before"] or execution["native_vla_modules"]["after"]
        ),
        "optimizer_steps_zero": (
            execution["optimizer"]["steps_observed"] == 0
            and not execution["optimizer"]["constructor_calls"]
        ),
    }


def run(plan: dict[str, Any], plan_identity: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    artifact_bytes_before_result = directory_bytes(ARTIFACT_ROOT)
    native_modules_before = native_vla_modules_loaded()
    source = verify_source(plan["source"]["commit"])
    r1_evidence = verify_r1_evidence()
    failed_engineering_attempt = verify_failed_engineering_attempt()
    dependencies = verify_dependencies()
    codec, input_evidence = load_codec_input()
    environment = verify_environment()
    with optimizer_construction_guard() as optimizer_evidence:
        numerical = numerical_probe(codec)
    execution = {
        "native_vla_modules": {
            "before": native_modules_before,
            "after": native_vla_modules_loaded(),
        },
        "optimizer": optimizer_evidence,
    }
    gates = evaluate_gates(numerical, execution)
    wall_seconds = time.monotonic() - started
    gates["wall_time_at_most_10min"] = wall_seconds <= MAX_WALL_SECONDS
    gates["artifact_total_at_most_256mib"] = True
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "run_id": RUN_ID,
        "source_commit": source["commit"],
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source": source,
        "plan": plan_identity,
        "r1_evidence": r1_evidence,
        "failed_engineering_attempt": failed_engineering_attempt,
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
        "r2_allowed": all(gates.values()),
        "training_allowed": False,
        "wall_seconds": wall_seconds,
        "claim_boundary": "fixed-weight FP32 numerical execution diagnosis only",
    }
    for _ in range(3):
        result_bytes = len((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        total_bytes = artifact_bytes_before_result + result_bytes
        result["artifact"] = {
            "root": str(ARTIFACT_ROOT.resolve()),
            "bytes_before_result": artifact_bytes_before_result,
            "result_bytes": result_bytes,
            "total_bytes": total_bytes,
            "limit_bytes": MAX_ARTIFACT_BYTES,
        }
        result["gates"]["artifact_total_at_most_256mib"] = total_bytes <= MAX_ARTIFACT_BYTES
        result["status"] = "passed" if all(result["gates"].values()) else "gate_failed"
        result["r2_allowed"] = all(result["gates"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-plan", type=Path)
    mode.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    if args.prepare_plan is not None:
        if args.expected_plan_sha256 is not None:
            parser.error("--expected-plan-sha256 is only valid with --plan")
        identity = prepare_plan(args.prepare_plan)
        print(json.dumps(identity, sort_keys=True), flush=True)
        return 0
    if args.expected_plan_sha256 is None:
        parser.error("--expected-plan-sha256 is required with --plan")
    plan, plan_identity = verify_plan(args.plan, args.expected_plan_sha256)
    output = Path(plan["output"]["root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result_path = output / plan["output"]["result_file"]
    started = time.monotonic()
    try:
        result = run(plan, plan_identity)
        write_new(result_path, result)
    except Exception as error:
        write_new(
            result_path,
            {
                "status": "failed",
                "experiment_id": EXPERIMENT,
                "run_id": RUN_ID,
                "plan": plan_identity,
                "error": f"{type(error).__name__}: {error}",
                "optimizer_evidence": "unavailable because diagnosis raised before completion",
                "native_vla_evidence": native_vla_modules_loaded(),
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


if __name__ == "__main__":
    raise SystemExit(main())
