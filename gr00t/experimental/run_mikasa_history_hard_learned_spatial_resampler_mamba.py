#!/usr/bin/env python3
"""Run the full-grid learned spatial resampler plus causal-Mamba experiment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import stat
import statistics
import subprocess
import sys
import time
from typing import Any, Final
from unittest.mock import patch


OVERLAY_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/g1-dit-mamba-yuqi-5000step-r1/dependencies/python-overlay"
)
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))

import numpy as np
from PIL import Image
import torch

import gr00t
from gr00t.experimental.mikasa_history_hard_learned_spatial_resampler_mamba import (
    BOUNDARY_INDICES,
    CONDITIONS,
    DEV_GROUPS,
    DEV_ROWS,
    EXPERIMENT_ID,
    FRAMES,
    INPUT_DIM,
    PAIR_CODES,
    PARAMETER_LIMIT,
    ROWS,
    ROWS_PER_GROUP,
    TRAIN_GROUPS,
    TRAIN_SEED,
    LearnedSpatialContractError,
    LearnedSpatialResamplerMamba,
    order_cross_entropy,
    score_predictions,
    trainable_parameter_count,
)
from gr00t.experimental.mikasa_history_hard_order_runtime import (
    CORPUS_ROOT,
    MODEL_FIELDS,
    TARGET_FIELDS,
    HistoryHardSources,
    RuntimeContractError,
    array_identity,
    canonical_json_bytes,
    freeze_directory,
    sha256_bytes,
    sha256_file,
    validate_frozen_evidence,
    write_once,
)
from gr00t.experimental.mikasa_history_hard_pretrained_vision_topk_events import (
    encode_public_merged_tokens,
)


REPO: Final = Path(__file__).resolve().parents[2]
CAMERA_ORDER: Final = ("base_camera", "hand_camera")
CUE_FRAMES: Final = FRAMES
EMBED_DIM: Final = INPUT_DIM
BASE_COMMIT: Final = "2f792adfd8b5f44ff4a0a3d2c8401a77cdb49d46"
EXPECTED_BRANCH: Final = "codex/g1-mikasa-history-hard-learned-spatial-resampler-mamba-r1"
SOURCE_PATHS: Final = (
    "gr00t/experimental/mikasa_history_hard_learned_spatial_resampler_mamba.py",
    "gr00t/experimental/run_mikasa_history_hard_learned_spatial_resampler_mamba.py",
    "tests/test_mikasa_history_hard_learned_spatial_resampler_mamba.py",
)
DEPENDENCY_PATHS: Final = (
    "gr00t/experimental/mikasa_history_hard_order.py",
    "gr00t/experimental/mikasa_history_hard_order_runtime.py",
    "gr00t/experimental/mikasa_history_hard_pretrained_vision_topk_events.py",
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
    "gr00t/model/gr00t_n1d7/image_augmentations.py",
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "gr00t/model/modules/qwen3_backbone.py",
)
PHASES: Final = (
    "source_acceptance",
    "processor_smoke",
    "layout",
    "cache",
    "preflight",
    "overfit",
)

PREDECESSOR_IDENTITIES: Final = {
    "topk_cache": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vision-topk-events-r1/"
            "target-blind-cache-gpu0-r1/RESULT.json"
        ),
        25_778,
        "832da23be4ad405f9de9df832c2b2c406e1baf907b432c2867f8d68de9987b09",
    ),
    "topk_train": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vision-topk-events-r1/"
            "train-loo-audit-cpu-r1/RESULT.json"
        ),
        214_126,
        "3428edc09da3e4d73e3c2a1b4b5de413ab6ecb9c7f8265c449492bb4675c8ab9",
    ),
    "spatial_pyramid_cache": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vision-spatial-pyramid-r1/"
            "target-blind-cache-gpu0-r1/RESULT.json"
        ),
        25_426,
        "21c6799601cf8d8af830ba35328ed9361565672f8bfda7ebbf51e23196e46133",
    ),
    "spatial_pyramid_train": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vision-spatial-pyramid-r1/"
            "train-loo-audit-cpu-r1/RESULT.json"
        ),
        214_106,
        "ae84adbff0b4e7ab798783476593be4ace0756737bc0cf60d22a3719f4da1413",
    ),
    "camera_mean_cache": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vlm-cue-audit-cache-bool-repair-r1/"
            "target-blind-cache-gpu0-r1/RESULT.json"
        ),
        24_648,
        "3970fb8ed1c3a1fc47b303a309aca4766a226dda4e896e9cceca2c6bae1ab11d",
    ),
    "camera_mean_train": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vlm-cue-audit-cache-bool-repair-r1/"
            "train-loo-audit-cpu-r1/RESULT.json"
        ),
        214_075,
        "52081108886cc766eb0e3c175dee46371d9c09a3ce5ba6830af7f32d4233a942",
    ),
    "r1_failure": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vlm-cue-audit-r1/"
            "layout-probe-gpu0-r1.FAILED.json"
        ),
        1_291,
        "fa3fd4b235c5812317e6fb7b92440acac9de6ca101ebc1776c0ebf7b0edb7a8f",
    ),
    "offline_repair_failure": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vlm-cue-audit-offline-repair-r1/"
            "layout-probe-gpu0-r1.FAILED.json"
        ),
        1_387,
        "f730a0d5197edc08bf821b1a6ae55f50a85d857218a677a5a7057dd711c7c2c3",
    ),
    "local_processor_cache_failure": (
        Path(
            "/home/sl/research/icra27-artifacts/"
            "g1-mikasa-history-hard-pretrained-vlm-cue-audit-local-processor-r1/"
            "target-blind-cache-gpu0-r1.FAILED.json"
        ),
        1_390,
        "03b77aaa0d2b09cd42dfd9b1c35a558405b34006c0464bd06c1a8007e2d9523c",
    ),
}
QUALIFIED_PREDECESSOR_LAYOUT: Final = (
    Path(
        "/home/sl/research/icra27-artifacts/"
        "g1-mikasa-history-hard-pretrained-vision-topk-events-r1/"
        "layout-probe-gpu0-r2/RESULT.json"
    ),
    15_272,
    "6e69effbc93c38ac752872c14ff2ba4594e6909993646053924ea0b06011f853",
)
FORBIDDEN_PARTIAL_DESCRIPTOR: Final = (
    Path(
        "/home/sl/research/icra27-artifacts/"
        "g1-mikasa-history-hard-pretrained-vision-topk-events-r1/"
        "target-blind-cache-gpu0-r1/DESCRIPTORS.npz"
    ),
    468_311_479,
    "f368d5d4f5aaf244805fc7d3d9ce8cf88fe8af56e335fa04d08ef00a912fdcd5",
)
LOCAL_PROCESSOR_PARTIAL_SHA256: Final = (
    "9bf07ca5e0523c4bc034dcc6628dc47cc61ae248cdbd2d70a60532a1a72f0833"
)
REMOTE_MODEL_ID: Final = "nvidia/Cosmos-Reason2-2B"
PROCESSOR_SMOKE_WALL_SECONDS: Final = 5 * 60.0
PROCESSOR_SMOKE_ARTIFACT_LIMIT: Final = 16 * 1024 * 1024

PRE_RUN: Final = Path(
    "/home/sl/research/icra27-artifacts/"
    "g1-pretrained-causal-mamba-memory-prototype-r1/PRE_RUN_IDENTITIES.json"
)
PRE_RUN_BYTES: Final = 11_107
PRE_RUN_SHA256: Final = "0570e37475c640a375a687e38a6615fdfd32e1816608bbf8c469398b8b84b132"
PRIVILEGED_LABEL_MANIFEST_SHA256: Final = (
    "8e414d9663ae32dcc3cb88af8624ac508aebac5bea03071420531d4d6e8e7aa9"
)
CHECKPOINT: Final = Path(
    "/home/yuqi/Isaac-GR00T-N1.7/"
    "runs_left13_n1d7_visual_compact_1.2mm_from_ckpt29000_20k/checkpoint-17000"
)
PROCESSOR_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/g1-dit-mamba-yuqi-5000step-r1/input-candidate-r3/processor"
)
FORMAL_PYTHON: Final = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv/bin/python")
FORMAL_TORCH: Final = "2.7.1+cu128"
FORMAL_TRANSFORMERS: Final = "4.57.3"
MAMBA_VERSION: Final = "2.3.2.post1"
MAMBA_OVERLAY_MATERIALS: Final = {
    "mamba2.py": (
        OVERLAY_ROOT / "mamba_ssm/modules/mamba2.py",
        17_479,
        "605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707",
    ),
    "selective_scan_cuda.so": (
        OVERLAY_ROOT / "selective_scan_cuda.cpython-310-x86_64-linux-gnu.so",
        418_507_320,
        "48cd62932a7b3ccd5e57eb80c40fcced8e3f35ec7e5ff219fc90d20021961f66",
    ),
    "RECORD": (
        OVERLAY_ROOT / "mamba_ssm-2.3.2.post1.dist-info/RECORD",
        5_974,
        "ba0b92cde35e0c47bd6e04a65588a14c3c0383745d20b13e95217cdcb0335a5d",
    ),
}
HF_REVISION: Final = "9ce19a195e423419c349abfc86fd07178b230561"
HF_CACHE_ROOT: Final = Path("/home/sl/.cache/huggingface/hub")
HF_MODEL_ROOT: Final = HF_CACHE_ROOT / "models--nvidia--Cosmos-Reason2-2B"
HF_SNAPSHOT_ROOT: Final = HF_MODEL_ROOT / "snapshots" / HF_REVISION
PROMPT: Final = "Trace the demonstrated paths in the shown order."
ALTERNATE_PROMPT: Final = "This fixed text must not change the visual descriptor."
FRAME_BATCH_SIZE: Final = 8
LAYOUT_WALL_SECONDS: Final = 10 * 60.0
LAYOUT_ARTIFACT_LIMIT: Final = 32 * 1024 * 1024
CACHE_WALL_SECONDS: Final = 45 * 60.0
CACHE_ARTIFACT_LIMIT: Final = 8 * 1024 * 1024 * 1024
CACHE_PEAK_ALLOCATED_LIMIT: Final = 12 * 1024 * 1024 * 1024
PREFLIGHT_STEPS: Final = 20
OVERFIT_STEPS: Final = 500
TRAIN_STEPS: Final = 1000
MICROBATCH: Final = 6
ACCUMULATION: Final = 5
LEARNING_RATE: Final = 3e-4
FINAL_LEARNING_RATE: Final = 3e-5
WARMUP_STEPS: Final = 100
BETAS: Final = (0.9, 0.95)
WEIGHT_DECAY: Final = 1e-4
GRADIENT_CLIP: Final = 1.0
PREFLIGHT_PEAK_LIMIT: Final = 20 * 1024 * 1024 * 1024
CUMULATIVE_TRAIN_SECONDS: Final = 6 * 60 * 60.0
OVERFIT_WALL_SECONDS: Final = 30 * 60.0
TRAIN_ARTIFACT_LIMIT: Final = 512 * 1024 * 1024
CACHE_GATE_KEYS: Final = (
    "all_48_rows",
    "descriptor_shapes_exact",
    "all_finite_float32",
    "six_current_exact_each_geometry",
    "six_unique_cue_hashes_each_geometry",
    "first_batch_repeat_exact",
    "public_boundary_exact",
    "local_processor_repair_exact",
    "forbidden_paths_zero",
    "parameter_versions_unchanged",
    "gradients_zero",
    "wall_within_45_minutes",
    "peak_allocated_below_12gib",
    "forbidden_partial_inode_not_reused",
)
PROCESSOR_SMOKE_GATE_KEYS: Final = (
    "official_model_id_preserved",
    "one_local_builder_call",
    "network_calls_zero",
    "left_padding_preserved",
    "gpu_contexts_zero",
    "wall_within_5_minutes",
)
LAYOUT_GATE_KEYS: Final = (
    "four_ordered_images",
    "grid_exact",
    "text_pixel_values_exact",
    "text_grid_exact",
    "text_visual_tokens_exact",
    "repeat_pixel_values_exact",
    "repeat_grid_exact",
    "repeat_visual_tokens_exact",
    "public_boundary_exact",
    "learned_sequence_exact",
    "shared_logits_exact",
    "boundary_indices_exact",
    "all_learned_outputs_finite",
    "actual_mamba_causal_prefix_exact",
    "future_perturbation_reaches_final",
    "residual_projection_exact_zero",
    "parameter_count_below_limit",
    "mamba_config_exact",
    "local_processor_repair_exact",
    "forbidden_paths_zero",
    "vision_parameters_and_gradients_unchanged",
    "wall_within_10_minutes",
)
PREFLIGHT_GATE_KEYS: Final = (
    "exact_20_optimizer_steps",
    "microbatch6_accumulation5",
    "all_losses_and_gradients_finite",
    "all_six_modules_have_nonzero_gradient",
    "all_six_modules_changed",
    "learned_queries_changed",
    "peak_allocation_below_20gib",
    "projected_train_only_below_6_gpu_hours",
    "checkpoint_not_written",
    "fp32_no_autocast",
)

PROCESSOR_IDENTITIES: Final = {
    "config.json": (2_190, "e1b5c231f440e5cc6b2a899f27f623558e441bdea9e3c37593d781a2bda3876e"),
    "embodiment_id.json": (
        2_169,
        "6f01100d50c52602f0c7c68304eea48f79a1f8c2426a960c04824c0ed4557869",
    ),
    "processor_config.json": (
        26_594,
        "7b940f9e8d216d29e8a28a4c2f55345e3bba0db1638f754d7300ff1bb5cd7764",
    ),
    "statistics.json": (
        4_298_426,
        "4e058f83bef04f3180ab006c791967fe0d36e0e6a3902f030d338b37069a0483",
    ),
}
HF_SNAPSHOT_IDENTITIES: Final = {
    "chat_template.json": (
        5_502,
        "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    ),
    "config.json": (1_505, "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"),
    "generation_config.json": (
        269,
        "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4",
    ),
    "merges.txt": (1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "model.safetensors": (
        4_877_470_304,
        "fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede",
    ),
    "preprocessor_config.json": (
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "tokenizer_config.json": (
        10_868,
        "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    ),
    "tokenizer.json": (
        7_032_403,
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    ),
    "video_preprocessor_config.json": (
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    "vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode(errors="replace").strip()) from exc


def _bind_repo_root(repo: Path) -> Path:
    resolved = repo.resolve(strict=True)
    expected = REPO.resolve(strict=True)
    if resolved != expected:
        raise RuntimeError(f"cue-audit repo-root differs from imported runner: {resolved}")
    import_roots = {
        Path(__file__).resolve(),
        Path(gr00t.__file__).resolve(),
        Path(sys.modules[HistoryHardSources.__module__].__file__).resolve(),
        Path(sys.modules[LearnedSpatialResamplerMamba.__module__].__file__).resolve(),
    }
    if any(expected not in path.parents for path in import_roots):
        raise RuntimeError(f"cue-audit import escaped repo-root: {sorted(map(str, import_roots))}")
    return resolved


def source_identity(repo: Path) -> dict[str, Any]:
    repo = _bind_repo_root(repo)
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    _git(repo, "merge-base", "--is-ancestor", BASE_COMMIT, head)
    raw = _git(repo, "diff", "--name-status", f"{BASE_COMMIT}..{head}").decode()
    observed = {}
    for line in raw.splitlines():
        status_code, path = line.split("\t", maxsplit=1)
        observed[path] = status_code
    expected = {path: "A" for path in SOURCE_PATHS}
    if observed != expected:
        raise RuntimeError(f"cue-audit exact source diff changed: {observed}")
    bound_paths = (*SOURCE_PATHS, *DEPENDENCY_PATHS)
    dirty = (
        _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--", *bound_paths)
        .decode()
        .splitlines()
    )
    if dirty:
        raise RuntimeError(f"cue-audit source is dirty: {dirty}")
    files = {}
    for relative in bound_paths:
        path = repo / relative
        value = path.read_bytes()
        if path.is_symlink() or value != _git(repo, "show", f"HEAD:{relative}"):
            raise RuntimeError(f"cue-audit source differs from HEAD: {relative}")
        files[relative] = {"bytes": len(value), "sha256": sha256_bytes(value)}
    branch = _git(repo, "branch", "--show-current").decode().strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"learned-spatial branch changed: {branch}")
    return {
        "base_commit": BASE_COMMIT,
        "head_commit": head,
        "branch": branch,
        "exact_diff": observed,
        "files": {path: files[path] for path in SOURCE_PATHS},
        "dependency_files": {path: files[path] for path in DEPENDENCY_PATHS},
        "owned_source_clean": True,
    }


def _regular_identity(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size != expected_bytes:
        raise RuntimeError(f"material shape/type changed: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"material hash changed: {path}")
    return {"path": str(path.resolve()), "bytes": expected_bytes, "sha256": observed}


def _snapshot_identity(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"HF snapshot entry must remain a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to((HF_MODEL_ROOT / "blobs").resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"HF snapshot entry escapes the frozen model cache: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size != expected_bytes:
        raise RuntimeError(f"HF snapshot material size/type changed: {path}")
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise RuntimeError(f"HF snapshot material hash changed: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": expected_bytes,
        "sha256": observed,
    }


def validate_pre_run() -> dict[str, Any]:
    receipt = _regular_identity(PRE_RUN, PRE_RUN_BYTES, PRE_RUN_SHA256)
    value = json.loads(PRE_RUN.read_bytes())
    if (
        value.get("status") != "passed"
        or Path(str(value.get("base_checkpoint"))).resolve() != CHECKPOINT.resolve()
        or value.get("optimizer_steps") != 0
    ):
        raise RuntimeError("prototype PRE_RUN boundary changed")
    source_files = {Path(record["path"]).name: record for record in value.get("source_files", [])}
    required_checkpoint = {
        "config.json": "e1b5c231f440e5cc6b2a899f27f623558e441bdea9e3c37593d781a2bda3876e",
        "model-00001-of-00002.safetensors": "9c317868603b586b312b20873a8d490683ec4f8711184c936a39d507b400b92f",
        "model-00002-of-00002.safetensors": "a34c207c113649e19569cd0e559bf37d450eda2fa54965dd2b599c6b914b0a78",
        "model.safetensors.index.json": "1b0db0b3cb34172eb63d6999ef01090357ae363efe663983fab7bbf591aa5b4c",
        "processor_config.json": "516e938ce7970c7d1b66b6ebafc5b28fdb64c6b1266f40c275ec81ab715f14d3",
        "statistics.json": "4e058f83bef04f3180ab006c791967fe0d36e0e6a3902f030d338b37069a0483",
    }
    checkpoint_files = {}
    for name, digest in required_checkpoint.items():
        record = source_files.get(name)
        if record is None or record.get("sha256") != digest:
            raise RuntimeError(f"PRE_RUN checkpoint identity changed: {name}")
        path = Path(str(record["path"]))
        if path.parent.resolve() != CHECKPOINT.resolve() or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"PRE_RUN checkpoint path changed: {name}")
        checkpoint_files[name] = _regular_identity(path, int(record["bytes"]), digest)
    processor = {
        name: _regular_identity(PROCESSOR_ROOT / name, size, digest)
        for name, (size, digest) in PROCESSOR_IDENTITIES.items()
    }
    config = json.loads((CHECKPOINT / "config.json").read_bytes())
    if (
        config.get("tune_visual") is not True
        or config.get("tune_llm") is not False
        or config.get("backbone_embedding_dim") != EMBED_DIM
        or config.get("max_state_dim") != 13
        or config.get("max_action_dim") != 13
    ):
        raise RuntimeError("Yuqi17000 frozen visual/checkpoint contract changed")
    processor_config = json.loads((PROCESSOR_ROOT / "processor_config.json").read_bytes())
    kwargs = processor_config.get("processor_kwargs", {})
    if (
        kwargs.get("letter_box_transform") is not True
        or kwargs.get("use_albumentations") is not True
        or kwargs.get("shortest_image_edge") != 256
        or kwargs.get("crop_fraction") != 0.95
    ):
        raise RuntimeError("Yuqi compatibility processor contract changed")
    return {"pre_run": receipt, "checkpoint_files": checkpoint_files, "processor": processor}


def validate_hf_snapshot() -> dict[str, Any]:
    if not HF_SNAPSHOT_ROOT.is_dir():
        raise RuntimeError(f"frozen HF snapshot is missing: {HF_SNAPSHOT_ROOT}")
    observed_names = {path.name for path in HF_SNAPSHOT_ROOT.iterdir()}
    if observed_names != set(HF_SNAPSHOT_IDENTITIES):
        raise RuntimeError(f"HF snapshot file set changed: {sorted(observed_names)}")
    files = {
        name: _snapshot_identity(HF_SNAPSHOT_ROOT / name, size, digest)
        for name, (size, digest) in HF_SNAPSHOT_IDENTITIES.items()
    }
    config = json.loads((HF_SNAPSHOT_ROOT / "config.json").read_bytes())
    if (
        config.get("model_type") != "qwen3_vl"
        or config.get("text_config", {}).get("hidden_size") != EMBED_DIM
        or config.get("vision_config", {}).get("out_hidden_size") != EMBED_DIM
    ):
        raise RuntimeError("frozen Cosmos/Qwen3-VL snapshot config changed")
    return {
        "model_id": "nvidia/Cosmos-Reason2-2B",
        "revision": HF_REVISION,
        "snapshot_root": str(HF_SNAPSHOT_ROOT),
        "files": files,
    }


def validate_predecessors() -> dict[str, Any]:
    identities = {
        name: _regular_identity(path, size, digest)
        for name, (path, size, digest) in PREDECESSOR_IDENTITIES.items()
    }
    failure_names = {
        "r1_failure",
        "offline_repair_failure",
        "local_processor_cache_failure",
    }
    for name in failure_names:
        path, _size, _digest = PREDECESSOR_IDENTITIES[name]
        value = json.loads(path.read_bytes())
        common_failed = (
            value.get("status") == "technical_failed"
            and value.get("target_npz_opened") == []
            and value.get("retry_count") == 0
        )
        boundary_ok = (
            value.get("result_written") is False
            and value.get("partial_artifact", {}).get("sha256") == LOCAL_PROCESSOR_PARTIAL_SHA256
            if name == "local_processor_cache_failure"
            else value.get("model_forward_calls") == 0
        )
        if not common_failed or not boundary_ok:
            raise RuntimeError(f"predecessor failure boundary changed: {name}")
    # The camera-mean and spatial-pyramid cache/train results are target-derived
    # historical receipts.
    # Bind them only by their preregistered byte/hash identities here; source
    # acceptance must not deserialize their label-bearing payloads.
    layout_path, layout_size, layout_sha = QUALIFIED_PREDECESSOR_LAYOUT
    layout_identity = _regular_identity(layout_path, layout_size, layout_sha)
    layout = json.loads(layout_path.read_bytes())
    if layout.get("status") != "passed" or not all(layout.get("gates", {}).values()):
        raise RuntimeError("qualified predecessor layout changed")
    partial_path, partial_size, partial_sha = FORBIDDEN_PARTIAL_DESCRIPTOR
    partial_identity = _regular_identity(partial_path, partial_size, partial_sha)
    if (
        stat.S_IMODE(partial_path.stat().st_mode) != 0o444
        or stat.S_IMODE(partial_path.parent.stat().st_mode) != 0o555
    ):
        raise RuntimeError("forbidden partial descriptor is not read-only frozen")
    return {
        "failures": {name: identities[name] for name in sorted(failure_names)},
        "camera_mean_results": {
            name: identities[name] for name in ("camera_mean_cache", "camera_mean_train")
        },
        "spatial_pyramid_results": {
            name: identities[name] for name in ("spatial_pyramid_cache", "spatial_pyramid_train")
        },
        "topk_results": {name: identities[name] for name in ("topk_cache", "topk_train")},
        "qualified_layout": layout_identity,
        "forbidden_partial_descriptor": partial_identity,
    }


def verify_mamba_overlay(*, import_package: bool) -> dict[str, Any]:
    records = {}
    for name, (path, size, digest) in MAMBA_OVERLAY_MATERIALS.items():
        records[name] = _regular_identity(path, size, digest)
    if importlib.util.find_spec("causal_conv1d") is not None:
        raise RuntimeError("causal_conv1d must remain absent for the frozen Torch Conv1d path")
    package_file = None
    version = MAMBA_VERSION
    if import_package:
        package = importlib.import_module("mamba_ssm")
        package_file = Path(package.__file__).resolve()
        if not package_file.is_relative_to(OVERLAY_ROOT.resolve()):
            raise RuntimeError("mamba_ssm import escaped the frozen Python 3.10 overlay")
        version = importlib.metadata.version("mamba-ssm")
        if version != MAMBA_VERSION:
            raise RuntimeError("mamba-ssm version changed")
    return {
        "root": str(OVERLAY_ROOT),
        "version": version,
        "package_file": None if package_file is None else str(package_file),
        "causal_conv1d_absent": True,
        "materials": records,
    }


def validate_external_materials() -> dict[str, Any]:
    return {
        "predecessors": validate_predecessors(),
        "checkpoint_and_processor": validate_pre_run(),
        "hf_snapshot": validate_hf_snapshot(),
        "mamba_overlay": verify_mamba_overlay(import_package=False),
    }


def bind_runtime_external_materials(source: Mapping[str, Any]) -> dict[str, Any]:
    accepted = source.get("external_materials")
    current = validate_external_materials()
    if current != accepted:
        raise PermissionError("cue-audit external checkpoint/processor/snapshot identity changed")
    return current


def protocol() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "representation_variable": "full64 patches plus fixed-query spatial resampler and 2-layer causal Mamba",
        "predecessor_result_sha256": {
            name: identity[2] for name, identity in PREDECESSOR_IDENTITIES.items()
        },
        "qualified_predecessor_layout_sha256": QUALIFIED_PREDECESSOR_LAYOUT[2],
        "forbidden_partial_descriptor_sha256": FORBIDDEN_PARTIAL_DESCRIPTOR[2],
        "phases": list(PHASES),
        "implementation_slice": "technical_plus_preflight_overfit",
        "deferred_phases_fail_closed": ["train", "dev"],
        "camera_order": list(CAMERA_ORDER),
        "prompt": PROMPT,
        "alternate_prompt": ALTERNATE_PROMPT,
        "public_boundary": "model.backbone.model.get_image_features",
        "pre_run_sha256": PRE_RUN_SHA256,
        "hf_revision": HF_REVISION,
        "processor_smoke_wall_seconds": PROCESSOR_SMOKE_WALL_SECONDS,
        "processor_smoke_artifact_limit_bytes": PROCESSOR_SMOKE_ARTIFACT_LIMIT,
        "hf_snapshot_files": sorted(HF_SNAPSHOT_IDENTITIES),
        "privileged_label_manifest_sha256_frozen_but_unopened": (PRIVILEGED_LABEL_MANIFEST_SHA256),
        "merged_tokens_per_image": 64,
        "merged_token_grid": [8, 8],
        "full_grid_shape": [FRAMES, 2, 64, INPUT_DIM],
        "embedding_dim": EMBED_DIM,
        "cue_frames": CUE_FRAMES,
        "slot_frames": 65,
        "frame_batch_size": FRAME_BATCH_SIZE,
        "conditions": list(CONDITIONS),
        "train_groups": list(TRAIN_GROUPS),
        "dev_groups": list(DEV_GROUPS),
        "dev_rows_identity_only_and_unopened": list(DEV_ROWS),
        "resampler": {
            "queries": 4,
            "width": 128,
            "heads": 4,
            "ffn_width": 256,
            "pre_norm": True,
        },
        "mamba": {
            "layers": 2,
            "d_model": 128,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "headdim": 64,
            "ngroups": 1,
            "chunk_size": 16,
            "use_mem_eff_path": False,
            "causal": True,
        },
        "sequence_order": "time-major,camera-major,query-major",
        "sequence_length": 1040,
        "boundary_indices": [519, 1039],
        "parameter_limit": PARAMETER_LIMIT,
        "training": {
            "seed": TRAIN_SEED,
            "preflight_steps": PREFLIGHT_STEPS,
            "overfit_steps": OVERFIT_STEPS,
            "fold_steps": TRAIN_STEPS,
            "microbatch": MICROBATCH,
            "accumulation": ACCUMULATION,
            "optimizer": "AdamW",
            "lr": LEARNING_RATE,
            "final_lr": FINAL_LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "cumulative_seconds": CUMULATIVE_TRAIN_SECONDS,
        },
        "formal_runtime": {
            "python": str(FORMAL_PYTHON),
            "mamba_version": MAMBA_VERSION,
            "mamba_overlay": str(OVERLAY_ROOT),
            "torch": FORMAL_TORCH,
            "transformers": FORMAL_TRANSFORMERS,
            "device": "physical cuda:0",
        },
        "layout_wall_seconds": LAYOUT_WALL_SECONDS,
        "layout_artifact_limit_bytes": LAYOUT_ARTIFACT_LIMIT,
        "cache_wall_seconds": CACHE_WALL_SECONDS,
        "cache_artifact_limit_bytes": CACHE_ARTIFACT_LIMIT,
        "cache_peak_allocated_limit_bytes": CACHE_PEAK_ALLOCATED_LIMIT,
        "train_artifact_limit_bytes": TRAIN_ARTIFACT_LIMIT,
        "action_outputs": 0,
        "retry_count": 0,
    }


def _run_focused_tests(repo: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        "tests/test_mikasa_history_hard_learned_spatial_resampler_mamba.py",
    ]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repo),
        }
    )
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"focused tests failed:\n{completed.stdout}\n{completed.stderr}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2_000:],
        "cuda_visible_devices": "",
    }


def run_source_acceptance(output_root: Path, repo: Path = REPO) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    identity = source_identity(repo)
    external_materials = validate_external_materials()
    evidence = validate_frozen_evidence(include_target=False)
    tests = _run_focused_tests(repo)
    result = {
        "schema": f"{EXPERIMENT_ID}/source-acceptance-v1",
        "status": "passed",
        "mode": "source_acceptance",
        "source_identity": identity,
        "external_materials": external_materials,
        "frozen_evidence": evidence,
        "focused_tests": tests,
        "protocol": protocol(),
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "fresh_data_opened": False,
        "gpu_contexts": 0,
        "model_calls": 0,
        "action_outputs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return result


def _read_receipt(
    path: Path, expected_sha256: str, mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise PermissionError(f"{mode} receipt must be regular mode 0444")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if len(expected_sha256) != 64 or observed != expected_sha256:
        raise PermissionError(f"{mode} receipt identity changed")
    value = json.loads(raw)
    if value.get("status") != "passed" or value.get("mode") != mode:
        raise PermissionError(f"{mode} did not pass")
    return value, {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": observed,
        "mode": "0444",
    }


def bind_runtime_source(source: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    if (
        source.get("schema") != f"{EXPERIMENT_ID}/source-acceptance-v1"
        or source.get("protocol") != protocol()
        or source.get("target_manifest_opened") is not False
        or source.get("fresh_data_opened") is not False
    ):
        raise PermissionError("cue-audit source acceptance contract changed")
    current = source_identity(repo)
    if current != source.get("source_identity"):
        raise PermissionError("cue-audit runtime source identity changed")
    return current


def _formal_runtime(device_name: str) -> tuple[torch.device, dict[str, Any]]:
    import transformers

    import gr00t

    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError("model phases require the frozen Yuqi Python executable")
    if platform.python_version_tuple()[:2] != ("3", "10"):
        raise RuntimeError("model phases require Python 3.10")
    if torch.__version__ != FORMAL_TORCH or transformers.__version__ != FORMAL_TRANSFORMERS:
        raise RuntimeError("model phase torch/transformers identity changed")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("model phases require frozen offline Hugging Face resolution")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("model phases require CUDA_VISIBLE_DEVICES=0")
    gr00t_file = Path(gr00t.__file__).resolve()
    if REPO.resolve() not in gr00t_file.parents:
        raise RuntimeError(f"GR00T import escaped the active cue-audit worktree: {gr00t_file}")
    device = torch.device(device_name)
    if (
        device != torch.device("cuda:0")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("model phases require one visible physical cuda:0")
    torch.cuda.set_device(0)
    overlay = verify_mamba_overlay(import_package=True)
    return device, {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gr00t_file": str(gr00t_file),
        "mamba_overlay": overlay,
    }


def _frozen_loading_kwargs() -> dict[str, Any]:
    return {
        "local_files_only": True,
        "trust_remote_code": True,
        "revision": HF_REVISION,
        "cache_dir": str(HF_CACHE_ROOT),
    }


@contextmanager
def _forbidden_network_calls():
    calls: list[str] = []

    def forbid(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise RuntimeError(f"network call forbidden during local processor repair: {name}")

        return fail

    with (
        patch("huggingface_hub.model_info", forbid("model_info")),
        patch("huggingface_hub.hf_api.HfApi.model_info", forbid("HfApi.model_info")),
        patch("requests.sessions.Session.request", forbid("requests.Session.request")),
    ):
        yield calls


@contextmanager
def _local_collator_processor_builder():
    from transformers import Qwen3VLProcessor

    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing

    calls: list[dict[str, Any]] = []

    def local_builder(model_name: str, loading_kwargs: Mapping[str, Any]):
        if model_name != REMOTE_MODEL_ID:
            raise RuntimeError(f"collator builder expected the official model ID: {model_name}")
        expected = _frozen_loading_kwargs()
        if dict(loading_kwargs) != expected:
            raise RuntimeError("collator builder loading kwargs changed")
        processor = Qwen3VLProcessor.from_pretrained(str(HF_SNAPSHOT_ROOT), **expected)
        calls.append(
            {
                "requested_model_name": model_name,
                "resolved_snapshot": str(HF_SNAPSHOT_ROOT),
                "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
                "tokenizer_name_or_path": str(processor.tokenizer.name_or_path),
            }
        )
        return processor

    with patch.object(processing, "build_processor", local_builder):
        yield calls


def _validate_local_builder_calls(calls: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        len(calls) == 1
        and calls[0].get("requested_model_name") == REMOTE_MODEL_ID
        and calls[0].get("resolved_snapshot") == str(HF_SNAPSHOT_ROOT)
        and str(calls[0].get("processor_class", "")).endswith(".Qwen3VLProcessor")
        and calls[0].get("tokenizer_name_or_path") == str(HF_SNAPSHOT_ROOT)
    )


def _construct_collator_smoke() -> dict[str, Any]:
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator

    with _forbidden_network_calls() as network_calls:
        with _local_collator_processor_builder() as builder_calls:
            collator = Gr00tN1d7DataCollator(
                model_name=REMOTE_MODEL_ID,
                model_type="qwen",
                transformers_loading_kwargs=_frozen_loading_kwargs(),
            )
    receipt = {
        "collator_class": f"{type(collator).__module__}.{type(collator).__qualname__}",
        "collator_model_name": collator.model_name,
        "collator_model_type": collator.model_type,
        "builder_calls": list(builder_calls),
        "network_calls": list(network_calls),
        "padding_side": collator.processor.tokenizer.padding_side,
    }
    if (
        receipt["collator_model_name"] != REMOTE_MODEL_ID
        or receipt["collator_model_type"] != "qwen"
        or receipt["padding_side"] != "left"
        or not _validate_local_builder_calls(builder_calls)
        or network_calls
    ):
        raise RuntimeError(f"local collator processor smoke failed: {receipt}")
    return receipt


def _load_model_and_processor(device: torch.device):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    loading = _frozen_loading_kwargs()
    with _forbidden_network_calls() as network_calls:
        with _local_collator_processor_builder() as builder_calls:
            model, info = Gr00tN1d7.from_pretrained(
                CHECKPOINT,
                local_files_only=True,
                output_loading_info=True,
                ignore_mismatched_sizes=False,
                transformers_loading_kwargs=loading,
            )
        if any(
            info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise RuntimeError(f"strict Yuqi17000 checkpoint load failed: {info}")
        processor = Gr00tN1d7Processor.from_pretrained(
            PROCESSOR_ROOT,
            local_files_only=True,
            model_name=str(HF_SNAPSHOT_ROOT),
            transformers_loading_kwargs=loading,
        )
    if network_calls or not _validate_local_builder_calls(builder_calls):
        raise RuntimeError("local collator processor repair did not remain exact and offline")
    processor.eval()
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    if (
        model.config.tune_visual is not True
        or model.config.tune_llm is not False
        or model.config.backbone_embedding_dim != EMBED_DIM
        or model.config.model_name != REMOTE_MODEL_ID
        or model.collator.model_name != REMOTE_MODEL_ID
        or processor.model_name != str(HF_SNAPSHOT_ROOT)
    ):
        raise RuntimeError("loaded Yuqi visual contract changed")
    if any(
        parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()
    ):
        raise RuntimeError("frozen model has trainable parameters or gradients")
    loading_info = dict(info)
    loading_info["local_processor_repair"] = {
        "serialized_model_name": model.config.model_name,
        "collator_model_name": model.collator.model_name,
        "compatibility_processor_model_name": processor.model_name,
        "builder_calls": list(builder_calls),
        "network_calls": list(network_calls),
    }
    return model, processor, loading_info


@contextmanager
def _forbidden_model_paths(model):
    calls: list[str] = []

    def forbid(name: str):
        def hook(_module, _args):
            calls.append(name)
            raise RuntimeError(f"forbidden model path called during visual cache: {name}")

        return hook

    modules = {
        "language_model": model.backbone.model.language_model,
        "vlln": model.action_head.vlln,
        "vl_self_attention": model.action_head.vl_self_attention,
        "state_encoder": model.action_head.state_encoder,
        "action_head": model.action_head,
    }
    handles = [module.register_forward_pre_hook(forbid(name)) for name, module in modules.items()]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def _parameter_versions(model) -> dict[str, int]:
    return {name: int(parameter._version) for name, parameter in model.named_parameters()}


def _package_rgb(processor, rgb: np.ndarray, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay

    if rgb.ndim != 5 or rgb.shape[1:] != (len(CAMERA_ORDER), 128, 128, 3) or rgb.dtype != np.uint8:
        raise LearnedSpatialContractError("RGB packaging requires uint8 [time,2,128,128,3]")
    images = [Image.fromarray(frame) for frame_pair in rgb for frame in frame_pair]
    transformed, _ = apply_with_replay(processor.eval_image_transform, images)
    stacked = torch.stack(transformed)
    content = processor._apply_vlm_processing(stacked, prompt)
    batch = processor.collator([content])["inputs"]
    pixel_values = batch["pixel_values"].detach().cpu().contiguous()
    image_grid_thw = batch["image_grid_thw"].detach().cpu().contiguous()
    expected = torch.tensor([[1, 16, 16]] * (len(rgb) * len(CAMERA_ORDER)), dtype=torch.int64)
    if image_grid_thw.dtype != torch.int64 or not torch.equal(image_grid_thw, expected):
        raise LearnedSpatialContractError("native processor image grid/order changed")
    if pixel_values.dtype != torch.float32 or not bool(torch.isfinite(pixel_values).all()):
        raise LearnedSpatialContractError("native processor pixel_values changed")
    return pixel_values, image_grid_thw


def _elapsed_indices(elapsed: np.ndarray, required: Sequence[int]) -> dict[int, int]:
    if elapsed.ndim != 1 or elapsed.dtype != np.int64:
        raise RuntimeContractError("elapsed_steps must be int64 vector")
    result = {}
    for value in required:
        found = np.flatnonzero(elapsed == value)
        if found.shape != (1,):
            raise RuntimeContractError(f"elapsed {value} is not unique")
        result[value] = int(found[0])
    return result


def _load_row_rgb(row, sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    row.model.verify(root=CORPUS_ROOT)
    with np.load(row.model.path, allow_pickle=False) as archive:
        if set(archive.files) != set(MODEL_FIELDS):
            raise RuntimeContractError(f"row {row.row_id} model field allowlist changed")
        elapsed = np.asarray(archive["elapsed_steps"])
        indices = _elapsed_indices(elapsed, (*range(1, 131), 132))
        rgb = np.asarray(archive["rgb"])
        valid = np.asarray(archive["valid_mask"])
        cameras = tuple(str(value) for value in np.asarray(archive["camera_order"]).tolist())
        if (
            rgb.shape != (len(elapsed), 128, 128, 6)
            or rgb.dtype != np.uint8
            or valid.shape != (len(elapsed),)
            or valid.dtype != np.bool_
            or cameras != CAMERA_ORDER
        ):
            raise RuntimeContractError(f"row {row.row_id} target-blind RGB contract changed")
        selected = np.asarray([indices[value] for value in (*range(1, 131), 132)])
        if not bool(valid[selected].all()):
            raise RuntimeContractError(f"row {row.row_id} selected RGB is invalid")
        cue_packed = np.ascontiguousarray(rgb[[indices[value] for value in range(1, 131)]])
        current_packed = np.ascontiguousarray(rgb[indices[132]])
    if sample.get("temporal_rgb_1_130") != array_identity(cue_packed):
        raise RuntimeContractError(f"row {row.row_id} cue RGB identity changed")
    if sample.get("current_rgb_132") != array_identity(current_packed):
        raise RuntimeContractError(f"row {row.row_id} current RGB identity changed")
    cue = np.stack((cue_packed[..., :3], cue_packed[..., 3:]), axis=1)
    current = np.stack((current_packed[..., :3], current_packed[..., 3:]), axis=0)
    return np.ascontiguousarray(cue), np.ascontiguousarray(current)


def _encode_rgb_batch(
    model,
    processor,
    rgb: np.ndarray,
    prompt: str,
    device: torch.device,
    *,
    hash_tokens: bool = False,
):
    pixel_values, image_grid_thw = _package_rgb(processor, rgb, prompt)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        descriptor, receipt = encode_public_merged_tokens(
            model.backbone,
            pixel_values.to(device),
            image_grid_thw.to(device),
            frames=len(rgb),
            hash_tokens=hash_tokens,
        )
    return descriptor.detach().cpu().contiguous(), pixel_values, image_grid_thw, receipt


def _directory_bytes(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _public_boundary_exact(receipt: Mapping[str, Any], images: int) -> bool:
    return bool(
        receipt.get("images") == images
        and receipt.get("merged_tokens_per_image") == 64
        and receipt.get("embedding_dim") == EMBED_DIM
        and receipt.get("grid_side") == 8
        and str(receipt.get("owner_type", "")).endswith(".Qwen3VLForConditionalGeneration")
        and str(receipt.get("method_qualname", "")).endswith(
            "Qwen3VLForConditionalGeneration.get_image_features"
        )
    )


def _local_processor_repair_exact(loading: Mapping[str, Any]) -> bool:
    repair = loading.get("local_processor_repair")
    return bool(
        isinstance(repair, Mapping)
        and repair.get("serialized_model_name") == REMOTE_MODEL_ID
        and repair.get("collator_model_name") == REMOTE_MODEL_ID
        and repair.get("compatibility_processor_model_name") == str(HF_SNAPSHOT_ROOT)
        and _validate_local_builder_calls(repair.get("builder_calls", ()))
        and repair.get("network_calls") == []
    )


def _native_python_bool_gates(values: Mapping[str, Any]) -> dict[str, bool]:
    if tuple(values) != CACHE_GATE_KEYS:
        raise RuntimeError(f"cache gate key/order changed: {tuple(values)}")
    result = {key: bool(value) for key, value in values.items()}
    if any(type(value) is not bool for value in result.values()):
        raise RuntimeError("cache gate normalization did not produce exact Python bool")
    canonical_json_bytes(result)
    return result


def _write_result(
    output_root: Path, result: Mapping[str, Any], *, limit: int | None = None
) -> None:
    payload = canonical_json_bytes(result)
    projected = _directory_bytes(output_root) + len(payload)
    if limit is not None and projected > limit:
        raise RuntimeError(f"artifact exceeds frozen byte limit: {projected} > {limit}")
    write_once(output_root / "RESULT.json", payload)
    freeze_directory(output_root)


def run_processor_smoke(output_root: Path, *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic()
    receipt = _construct_collator_smoke()
    elapsed = time.monotonic() - started
    gates = {
        "official_model_id_preserved": receipt["collator_model_name"] == REMOTE_MODEL_ID,
        "one_local_builder_call": _validate_local_builder_calls(receipt["builder_calls"]),
        "network_calls_zero": receipt["network_calls"] == [],
        "left_padding_preserved": receipt["padding_side"] == "left",
        "gpu_contexts_zero": not torch.cuda.is_initialized(),
        "wall_within_5_minutes": elapsed <= PROCESSOR_SMOKE_WALL_SECONDS,
    }
    result = {
        "schema": f"{EXPERIMENT_ID}/processor-smoke-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "processor_smoke",
        "receipt": receipt,
        "gates": gates,
        "wall_seconds": elapsed,
        "provenance": dict(provenance),
        "gpu_contexts": 0,
        "model_weight_loads": 0,
        "model_calls": 0,
        "image_outputs": 0,
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "fresh_data_opened": False,
        "action_outputs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    _write_result(output_root, result, limit=PROCESSOR_SMOKE_ARTIFACT_LIMIT)
    return result


def run_layout(
    output_root: Path,
    *,
    device: torch.device,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the public vision boundary and the learned 1040-token layout."""

    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic()
    sources = HistoryHardSources.discover()
    cue_rgb, current_rgb = _load_row_rgb(sources.rows[0], sources.input_samples[0])
    layout_rgb = np.concatenate((cue_rgb[:1], current_rgb[None]), axis=0)
    vision, processor, loading = _load_model_and_processor(device)
    versions = _parameter_versions(vision)
    with _forbidden_model_paths(vision) as forbidden:
        first, pixels_a, grid_a, boundary = _encode_rgb_batch(
            vision, processor, layout_rgb, PROMPT, device, hash_tokens=True
        )
        alternate, pixels_b, grid_b, alternate_boundary = _encode_rgb_batch(
            vision, processor, layout_rgb, ALTERNATE_PROMPT, device, hash_tokens=True
        )
        repeated, pixels_c, grid_c, repeated_boundary = _encode_rgb_batch(
            vision, processor, layout_rgb, PROMPT, device, hash_tokens=True
        )
    frozen_vision_ok = versions == _parameter_versions(vision) and all(
        parameter.grad is None for parameter in vision.parameters()
    )
    del vision, processor
    torch.cuda.empty_cache()

    learned = (
        LearnedSpatialResamplerMamba("mamba_ordered").to(device=device, dtype=torch.float32).eval()
    )
    parameter_count = trainable_parameter_count(learned)
    residual_zero = bool(
        torch.count_nonzero(learned.residual_projection.weight) == 0
        and torch.count_nonzero(learned.residual_projection.bias) == 0
    )
    cue_tokens = (
        first[0]
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, FRAMES, len(CAMERA_ORDER), 64, INPUT_DIM)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
    current_tokens = first[1].unsqueeze(0).to(device=device, dtype=torch.float32)
    future_perturbed = cue_tokens.clone()
    future_perturbed[:, FRAMES // 2 :] = -future_perturbed[:, FRAMES // 2 :]
    with torch.inference_mode():
        output = learned(cue_tokens, current_tokens)
        perturbed_output = learned(future_perturbed, current_tokens)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    mixer_config = [
        {
            "d_model": int(layer.mixer.d_model),
            "d_state": int(layer.mixer.d_state),
            "d_conv": int(layer.mixer.d_conv),
            "expand": int(layer.mixer.expand),
            "headdim": int(layer.mixer.headdim),
            "ngroups": int(layer.mixer.ngroups),
            "chunk_size": int(layer.mixer.chunk_size),
            "use_mem_eff_path": bool(layer.mixer.use_mem_eff_path),
        }
        for layer in learned.temporal
    ]
    gates = {
        "four_ordered_images": first.shape == (2, len(CAMERA_ORDER), 64, INPUT_DIM),
        "grid_exact": torch.equal(grid_a, torch.tensor([[1, 16, 16]] * (2 * len(CAMERA_ORDER)))),
        "text_pixel_values_exact": torch.equal(pixels_a, pixels_b),
        "text_grid_exact": torch.equal(grid_a, grid_b),
        "text_visual_tokens_exact": boundary["merged_token_sha256"]
        == alternate_boundary["merged_token_sha256"]
        and torch.equal(first, alternate),
        "repeat_pixel_values_exact": torch.equal(pixels_a, pixels_c),
        "repeat_grid_exact": torch.equal(grid_a, grid_c),
        "repeat_visual_tokens_exact": boundary["merged_token_sha256"]
        == repeated_boundary["merged_token_sha256"]
        and torch.equal(first, repeated),
        "public_boundary_exact": _public_boundary_exact(boundary, 4),
        "learned_sequence_exact": output.sequence.shape == (1, 1040, 128),
        "shared_logits_exact": output.logits.shape == (1, 2, 3),
        "boundary_indices_exact": list(BOUNDARY_INDICES) == [519, 1039],
        "all_learned_outputs_finite": bool(torch.isfinite(output.sequence).all())
        and bool(torch.isfinite(output.logits).all())
        and bool(torch.isfinite(perturbed_output.sequence).all())
        and bool(torch.isfinite(perturbed_output.logits).all()),
        "actual_mamba_causal_prefix_exact": torch.allclose(
            output.sequence[:, BOUNDARY_INDICES[0]],
            perturbed_output.sequence[:, BOUNDARY_INDICES[0]],
            rtol=1e-6,
            atol=1e-6,
        ),
        "future_perturbation_reaches_final": not torch.allclose(
            output.sequence[:, BOUNDARY_INDICES[1]],
            perturbed_output.sequence[:, BOUNDARY_INDICES[1]],
            rtol=1e-6,
            atol=1e-6,
        ),
        "residual_projection_exact_zero": residual_zero,
        "parameter_count_below_limit": parameter_count <= PARAMETER_LIMIT,
        "mamba_config_exact": mixer_config
        == [
            {
                "d_model": 128,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "headdim": 64,
                "ngroups": 1,
                "chunk_size": 16,
                "use_mem_eff_path": False,
            }
        ]
        * 2,
        "local_processor_repair_exact": _local_processor_repair_exact(loading),
        "forbidden_paths_zero": forbidden == [],
        "vision_parameters_and_gradients_unchanged": frozen_vision_ok,
        "wall_within_10_minutes": elapsed <= LAYOUT_WALL_SECONDS,
    }
    result = {
        "schema": f"{EXPERIMENT_ID}/layout-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "layout_probe",
        "row_id": 0,
        "camera_order": list(CAMERA_ORDER),
        "image_grid_thw": grid_a.tolist(),
        "merged_token_shape": list(first.shape),
        "merged_token_dtype": str(first.dtype),
        "learned_sequence_shape": list(output.sequence.shape),
        "logit_shape": list(output.logits.shape),
        "boundary_indices": list(BOUNDARY_INDICES),
        "trainable_parameters": parameter_count,
        "mamba_config": mixer_config,
        "public_boundary": boundary,
        "gates": gates,
        "wall_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "loading_info": loading,
        "runtime": dict(runtime),
        "provenance": dict(provenance),
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "fresh_data_opened": False,
        "language_model_calls": 0,
        "action_outputs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    _write_result(output_root, result, limit=LAYOUT_ARTIFACT_LIMIT)
    return result


def _npy_receipt(path: Path) -> dict[str, Any]:
    path.chmod(0o444)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mode": "0444",
    }


def run_cache(
    output_root: Path,
    *,
    device: torch.device,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the write-once, target-blind full 64-patch vision cache."""

    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic()
    sources = HistoryHardSources.discover()
    vision, processor, loading = _load_model_and_processor(device)
    versions = _parameter_versions(vision)
    output_root.mkdir(parents=True)
    cue_path = output_root / "CUE.npy"
    current_path = output_root / "CURRENT.npy"
    cue_map = np.lib.format.open_memmap(
        cue_path,
        mode="w+",
        dtype=np.float32,
        shape=(ROWS, FRAMES, len(CAMERA_ORDER), 64, INPUT_DIM),
    )
    current_map = np.lib.format.open_memmap(
        current_path,
        mode="w+",
        dtype=np.float32,
        shape=(ROWS, len(CAMERA_ORDER), 64, INPUT_DIM),
    )
    current_rows: list[np.ndarray] = []
    cue_hashes: dict[str, list[str]] = {str(group): [] for group in range(8)}
    first_batch_exact: bool | None = None
    boundary_receipt: dict[str, Any] | None = None
    with _forbidden_model_paths(vision) as forbidden:
        for row, sample in zip(sources.rows, sources.input_samples, strict=True):
            cue_rgb, current_rgb = _load_row_rgb(row, sample)
            frames = np.concatenate((cue_rgb, current_rgb[None]), axis=0)
            encoded = []
            for offset in range(0, len(frames), FRAME_BATCH_SIZE):
                if time.monotonic() - started > CACHE_WALL_SECONDS:
                    raise TimeoutError("full-grid target-blind cache exceeded 45 minutes")
                batch = frames[offset : offset + FRAME_BATCH_SIZE]
                tokens, _, _, receipt = _encode_rgb_batch(
                    vision,
                    processor,
                    batch,
                    PROMPT,
                    device,
                    hash_tokens=first_batch_exact is None,
                )
                boundary_receipt = boundary_receipt or receipt
                if first_batch_exact is None:
                    repeated, _, _, repeat_receipt = _encode_rgb_batch(
                        vision, processor, batch, PROMPT, device, hash_tokens=True
                    )
                    first_batch_exact = bool(
                        torch.equal(tokens, repeated)
                        and receipt["merged_token_sha256"] == repeat_receipt["merged_token_sha256"]
                    )
                encoded.append(tokens)
            row_value = torch.cat(encoded, dim=0).numpy().astype(np.float32, copy=False)
            expected_shape = (FRAMES + 1, len(CAMERA_ORDER), 64, INPUT_DIM)
            if row_value.shape != expected_shape:
                raise LearnedSpatialContractError("row full-grid token count changed")
            if not np.isfinite(row_value).all():
                raise LearnedSpatialContractError("row full-grid token cache is nonfinite")
            cue_map[row.row_id] = row_value[:FRAMES]
            current_map[row.row_id] = row_value[FRAMES]
            current_rows.append(np.ascontiguousarray(row_value[FRAMES]))
            cue_bytes = memoryview(np.ascontiguousarray(row_value[:FRAMES])).cast("B")
            cue_hashes[str(row.row_id // ROWS_PER_GROUP)].append(
                hashlib.sha256(cue_bytes).hexdigest()
            )
    current_array = np.stack(current_rows)
    current_equal = {
        str(group): all(
            np.array_equal(
                current_array[group * ROWS_PER_GROUP],
                current_array[group * ROWS_PER_GROUP + offset],
            )
            for offset in range(1, ROWS_PER_GROUP)
        )
        for group in range(8)
    }
    peak = int(torch.cuda.max_memory_allocated(device))
    descriptor_shapes_exact = cue_map.shape == (48, 130, 2, 64, 2048) and current_map.shape == (
        48,
        2,
        64,
        2048,
    )
    all_finite_float32 = cue_map.dtype == current_map.dtype == np.float32
    cue_map.flush()
    current_map.flush()
    del cue_map, current_map
    cue_receipt = _npy_receipt(cue_path)
    current_receipt = _npy_receipt(current_path)
    inode_reused = os.path.samefile(cue_path, FORBIDDEN_PARTIAL_DESCRIPTOR[0])
    elapsed = time.monotonic() - started
    gates = {
        "all_48_rows": len(current_rows) == ROWS,
        "descriptor_shapes_exact": descriptor_shapes_exact,
        "all_finite_float32": all_finite_float32,
        "six_current_exact_each_geometry": all(current_equal.values()),
        "six_unique_cue_hashes_each_geometry": all(
            len(set(values)) == ROWS_PER_GROUP for values in cue_hashes.values()
        ),
        "first_batch_repeat_exact": first_batch_exact is True,
        "public_boundary_exact": boundary_receipt is not None
        and _public_boundary_exact(boundary_receipt, FRAME_BATCH_SIZE * len(CAMERA_ORDER)),
        "local_processor_repair_exact": _local_processor_repair_exact(loading),
        "forbidden_paths_zero": forbidden == [],
        "parameter_versions_unchanged": versions == _parameter_versions(vision),
        "gradients_zero": all(parameter.grad is None for parameter in vision.parameters()),
        "wall_within_45_minutes": elapsed <= CACHE_WALL_SECONDS,
        "peak_allocated_below_12gib": peak < CACHE_PEAK_ALLOCATED_LIMIT,
    }
    gates["forbidden_partial_inode_not_reused"] = not inode_reused
    gates = _native_python_bool_gates(gates)
    result = {
        "schema": f"{EXPERIMENT_ID}/cache-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "target_blind_cache",
        "rows": ROWS,
        "features": {"cue": cue_receipt, "current": current_receipt},
        "sample_ids": [row.sample_id for row in sources.rows],
        "source_model_sha256": [row.model.sha256 for row in sources.rows],
        "camera_order": list(CAMERA_ORDER),
        "frame_batch_size": FRAME_BATCH_SIZE,
        "public_boundary": boundary_receipt,
        "current_equal_by_geometry": current_equal,
        "cue_hashes_by_geometry": cue_hashes,
        "gates": gates,
        "predecessor_descriptor_comparison": {
            "cue_inode_reused": inode_reused,
            "source_recomputed": True,
        },
        "wall_seconds": elapsed,
        "peak_allocated_bytes": peak,
        "loading_info": loading,
        "runtime": dict(runtime),
        "provenance": dict(provenance),
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "fresh_data_opened": False,
        "language_model_calls": 0,
        "action_outputs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    _write_result(output_root, result, limit=CACHE_ARTIFACT_LIMIT)
    return result


def _exact_passed_gates(value: Mapping[str, Any], expected: Sequence[str]) -> bool:
    gates = value.get("gates")
    return bool(
        isinstance(gates, Mapping)
        and len(gates) == len(expected)
        and set(gates) == set(expected)
        and all(type(gate) is bool and gate for gate in gates.values())
    )


def _load_cache(
    cache_root: Path, expected_sha256: str
) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, Any], dict[str, Any]]:
    result, identity = _read_receipt(
        cache_root / "RESULT.json", expected_sha256, "target_blind_cache"
    )
    if (
        result.get("schema") != f"{EXPERIMENT_ID}/cache-v1"
        or result.get("target_manifest_opened") is not False
        or result.get("target_npz_opened") != []
        or result.get("fresh_data_opened") is not False
        or not _exact_passed_gates(result, CACHE_GATE_KEYS)
        or result.get("rows") != ROWS
        or result.get("camera_order") != list(CAMERA_ORDER)
        or result.get("frame_batch_size") != FRAME_BATCH_SIZE
    ):
        raise PermissionError("full-grid cache boundary did not pass")
    sources = HistoryHardSources.discover()
    sample_ids = tuple(str(value) for value in result.get("sample_ids", ()))
    if sample_ids != tuple(row.sample_id for row in sources.rows):
        raise PermissionError("full-grid cache sample identity changed")
    if result.get("source_model_sha256") != [row.model.sha256 for row in sources.rows]:
        raise PermissionError("full-grid cache source model identity changed")
    arrays: dict[str, np.ndarray] = {}
    expected = {
        "cue": ((ROWS, FRAMES, len(CAMERA_ORDER), 64, INPUT_DIM), np.dtype(np.float32), "CUE.npy"),
        "current": ((ROWS, len(CAMERA_ORDER), 64, INPUT_DIM), np.dtype(np.float32), "CURRENT.npy"),
    }
    receipts = result.get("features")
    if not isinstance(receipts, Mapping) or set(receipts) != set(expected):
        raise PermissionError("full-grid cache feature receipts changed")
    for name, (shape, dtype, filename) in expected.items():
        receipt = receipts[name]
        path = Path(str(receipt.get("path")))
        if (
            path.resolve() != (cache_root.resolve() / filename)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != receipt.get("bytes")
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or sha256_file(path) != receipt.get("sha256")
        ):
            raise PermissionError(f"full-grid cache {name} identity changed")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != dtype:
            raise PermissionError(f"full-grid cache {name} shape/dtype changed")
        arrays[name] = value
    return arrays, sample_ids, result, identity


_LABEL_ROWS: Final = {
    "preflight": tuple(range(30)),
    "overfit": tuple(range(6)),
}


def _validate_label_rows(row_ids: Sequence[int], phase: str) -> tuple[int, ...]:
    rows = tuple(int(value) for value in row_ids)
    if phase not in _LABEL_ROWS or rows != _LABEL_ROWS[phase]:
        raise PermissionError(f"{phase} may open only its exact frozen target rows")
    return rows


def _load_labels(row_ids: Sequence[int], phase: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Open only the explicitly allowed target NPZs; never open TARGET_MANIFEST."""

    rows = _validate_label_rows(row_ids, phase)
    sources = HistoryHardSources.discover()
    labels = np.full((ROWS, 2), -1, dtype=np.int64)
    opened = []
    for row_id in rows:
        row = sources.rows[row_id]
        row.target.verify(root=CORPUS_ROOT)
        with np.load(row.target.path, allow_pickle=False) as archive:
            if set(archive.files) != set(TARGET_FIELDS):
                raise RuntimeContractError(f"row {row_id} target field allowlist changed")
            target_row = int(np.asarray(archive["row_id"]).item())
            group = int(np.asarray(archive["group_id"]).item())
            pair_code = str(np.asarray(archive["pair_code"]).item())
            ordered = np.asarray(archive["ordered_path_ids"])
        expected_group = row_id // ROWS_PER_GROUP
        expected_pair = PAIR_CODES[row_id % ROWS_PER_GROUP]
        if (
            target_row != row_id
            or group != expected_group
            or pair_code != expected_pair
            or ordered.shape != (2,)
            or ordered.dtype != np.int64
            or ordered.tolist() != [int(expected_pair[0]), int(expected_pair[1])]
        ):
            raise RuntimeContractError(f"row {row_id} ordered target contract changed")
        labels[row_id] = ordered
        opened.append(
            {
                "row_id": row_id,
                "path": str(row.target.path.resolve()),
                "bytes": row.target.bytes,
                "sha256": row.target.sha256,
            }
        )
    return labels, opened


def lr_multiplier(step: int, total_steps: int) -> float:
    if step < 0 or total_steps < 1:
        raise ValueError("invalid learning-rate step contract")
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = (step + 1 - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    floor = FINAL_LEARNING_RATE / LEARNING_RATE
    return floor + (1.0 - floor) * cosine


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_sha256(tensor).encode())
    return digest.hexdigest()


def _module_parameters(model: LearnedSpatialResamplerMamba) -> dict[str, list[torch.nn.Parameter]]:
    resampler_modules = (
        model.query_norm,
        model.patch_norm,
        model.cross_attention,
        model.ff_norm,
        model.ff,
    )
    return {
        "raw_projection": list(model.raw_projection.parameters()),
        "residual_projection": list(model.residual_projection.parameters()),
        "spatial_resampler": [model.resampler_queries]
        + [parameter for module in resampler_modules for parameter in module.parameters()],
        "identity_embeddings": [
            *model.camera_embedding.parameters(),
            *model.query_embedding.parameters(),
            *model.slot_embedding.parameters(),
        ],
        "temporal_mamba": list(model.temporal.parameters()),
        "shared_classifier": list(model.classifier.parameters()),
    }


def _module_hashes(model: LearnedSpatialResamplerMamba) -> dict[str, str]:
    modules = {
        "raw_projection": model.raw_projection,
        "residual_projection": model.residual_projection,
        "spatial_resampler": torch.nn.ModuleList(
            [model.query_norm, model.patch_norm, model.cross_attention, model.ff_norm, model.ff]
        ),
        "identity_embeddings": torch.nn.ModuleList(
            [model.camera_embedding, model.query_embedding, model.slot_embedding]
        ),
        "temporal_mamba": model.temporal,
        "shared_classifier": model.classifier,
    }
    result = {name: _state_dict_sha256(module) for name, module in modules.items()}
    query_hash = _tensor_sha256(model.resampler_queries)
    result["spatial_resampler"] = hashlib.sha256(
        f"{result['spatial_resampler']}|{query_hash}".encode()
    ).hexdigest()
    result["resampler_queries"] = query_hash
    return result


def _finite_nonzero_gradients(parameters: Sequence[torch.nn.Parameter]) -> bool:
    gradients = [parameter.grad for parameter in parameters if parameter.requires_grad]
    return (
        bool(gradients)
        and all(
            gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        and any(
            bool(torch.count_nonzero(gradient)) for gradient in gradients if gradient is not None
        )
    )


def _group_tensors(
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    group: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group not in TRAIN_GROUPS:
        raise PermissionError("optimizer group escaped frozen training groups")
    start = group * ROWS_PER_GROUP
    stop = start + ROWS_PER_GROUP
    if np.any(labels[start:stop] < 0):
        raise PermissionError("optimizer attempted to use unopened target rows")
    cue = torch.from_numpy(np.asarray(arrays["cue"][start:stop])).to(
        device=device, dtype=torch.float32
    )
    current = torch.from_numpy(np.asarray(arrays["current"][start:stop])).to(
        device=device, dtype=torch.float32
    )
    target = torch.from_numpy(labels[start:stop]).to(device=device, dtype=torch.int64)
    return cue, current, target


def _optimizer_step(
    model: LearnedSpatialResamplerMamba,
    optimizer: torch.optim.Optimizer,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    groups: Sequence[int],
    step: int,
    total_steps: int,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, bool]]:
    groups = tuple(int(group) for group in groups)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("optimizer groups must be nonempty and unique")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    pair = []
    slot0 = []
    slot1 = []
    for group in groups:
        cue, current, target = _group_tensors(arrays, labels, group, device)
        output = model(
            cue,
            current,
            shuffle_seed=TRAIN_SEED if model.condition == "mamba_shuffle" else None,
            shuffle_step=step if model.condition == "mamba_shuffle" else None,
        )
        loss = order_cross_entropy(output, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("learned-spatial cross entropy is nonfinite")
        (loss / len(groups)).backward()
        prediction = output.predictions.detach()
        losses.append(float(loss.detach().cpu()))
        pair.append(float(prediction.eq(target).all(dim=1).float().mean().cpu()))
        slot0.append(float(prediction[:, 0].eq(target[:, 0]).float().mean().cpu()))
        slot1.append(float(prediction[:, 1].eq(target[:, 1]).float().mean().cpu()))
        del cue, current, target, output, loss
    gradient_status = {
        name: _finite_nonzero_gradients(parameters)
        for name, parameters in _module_parameters(model).items()
    }
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError("learned-spatial gradient norm is nonfinite")
    multiplier = lr_multiplier(step, total_steps)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = LEARNING_RATE * multiplier
    optimizer.step()
    return (
        {
            "step": step + 1,
            "cross_entropy": float(np.mean(losses)),
            "pair_accuracy": float(np.mean(pair)),
            "slot0_accuracy": float(np.mean(slot0)),
            "slot1_accuracy": float(np.mean(slot1)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": float(grad_norm.detach().cpu()),
        },
        gradient_status,
    )


def _new_training_model(condition: str, device: torch.device) -> LearnedSpatialResamplerMamba:
    torch.manual_seed(TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
    model = LearnedSpatialResamplerMamba(condition, seed=TRAIN_SEED).to(
        device=device, dtype=torch.float32
    )
    if trainable_parameter_count(model) > PARAMETER_LIMIT:
        raise RuntimeError("learned-spatial trainable parameter cap exceeded")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("learned-spatial training must remain FP32")
    return model


def _new_optimizer(model: LearnedSpatialResamplerMamba) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def run_preflight(
    output_root: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic()
    labels, opened = _load_labels(_LABEL_ROWS["preflight"], "preflight")
    model = _new_training_model("mamba_ordered", device)
    initial = _module_hashes(model)
    optimizer = _new_optimizer(model)
    gradient_seen = {name: False for name in _module_parameters(model)}
    curve = []
    durations = []
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    setup_seconds = time.monotonic() - started
    for step in range(PREFLIGHT_STEPS):
        torch.cuda.synchronize(device)
        step_started = time.monotonic()
        record, gradients = _optimizer_step(
            model,
            optimizer,
            arrays,
            labels,
            groups=(0, 1, 2, 3, 4),
            step=step,
            total_steps=PREFLIGHT_STEPS,
            device=device,
        )
        torch.cuda.synchronize(device)
        durations.append(time.monotonic() - step_started)
        curve.append(record)
        gradient_seen = {name: gradient_seen[name] or gradients[name] for name in gradient_seen}
    torch.cuda.synchronize(device)
    wall = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    steady = durations[5:] if len(durations) > 5 else durations[1:]
    steady_median = float(statistics.median(steady))
    remaining_projected_steps = OVERFIT_STEPS + len(CONDITIONS) * len(TRAIN_GROUPS) * TRAIN_STEPS
    projected_steps = PREFLIGHT_STEPS + remaining_projected_steps
    projected_seconds = wall + steady_median * remaining_projected_steps
    final_hash = _module_hashes(model)
    gates = {
        "exact_20_optimizer_steps": len(curve) == PREFLIGHT_STEPS,
        "microbatch6_accumulation5": MICROBATCH == 6 and ACCUMULATION == 5,
        "all_losses_and_gradients_finite": all(
            np.isfinite(record["cross_entropy"]) and np.isfinite(record["grad_norm"])
            for record in curve
        ),
        "all_six_modules_have_nonzero_gradient": all(gradient_seen.values()),
        "all_six_modules_changed": all(final_hash[name] != initial[name] for name in gradient_seen),
        "learned_queries_changed": final_hash["resampler_queries"] != initial["resampler_queries"],
        "peak_allocation_below_20gib": peak < PREFLIGHT_PEAK_LIMIT,
        "projected_train_only_below_6_gpu_hours": projected_seconds < CUMULATIVE_TRAIN_SECONDS,
        "checkpoint_not_written": True,
        "fp32_no_autocast": all(
            parameter.dtype == torch.float32 for parameter in model.parameters()
        ),
    }
    result = {
        "schema": f"{EXPERIMENT_ID}/preflight-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "training_preflight",
        "condition": "mamba_ordered",
        "seed": TRAIN_SEED,
        "groups": [0, 1, 2, 3, 4],
        "microbatch": MICROBATCH,
        "gradient_accumulation": ACCUMULATION,
        "curve": curve,
        "step_seconds": durations,
        "setup_seconds": setup_seconds,
        "first_step_seconds": durations[0],
        "steady_step_median_seconds": steady_median,
        "projected_optimizer_steps": projected_steps,
        "projected_train_seconds": projected_seconds,
        "peak_allocated_bytes": peak,
        "trainable_parameters": trainable_parameter_count(model),
        "initial_module_sha256": initial,
        "final_module_sha256": final_hash,
        "gradient_seen": gradient_seen,
        "gates": gates,
        "wall_seconds": wall,
        "runtime": dict(runtime),
        "provenance": dict(provenance),
        "target_manifest_opened": False,
        "target_npz_opened": opened,
        "fresh_data_opened": False,
        "action_outputs": 0,
        "optimizer_steps": PREFLIGHT_STEPS,
        "checkpoint_written": False,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    _write_result(output_root, result, limit=TRAIN_ARTIFACT_LIMIT)
    return result


@torch.inference_mode()
def _evaluate_group0(
    model: LearnedSpatialResamplerMamba,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    cue, current, target = _group_tensors(arrays, labels, 0, device)
    output = model(cue, current)
    predictions = output.predictions.detach().cpu().numpy()
    section = score_predictions(
        predictions,
        labels,
        np.arange(ROWS, dtype=np.int64) // ROWS_PER_GROUP,
        tuple(range(ROWS_PER_GROUP)),
    )
    return {
        "section": section,
        "logits_finite": bool(torch.isfinite(output.logits).all()),
        "cross_entropy": float(order_cross_entropy(output, target).cpu()),
    }


@torch.inference_mode()
def _group0_substitution(
    model: LearnedSpatialResamplerMamba,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    correct = 0
    total = 0
    records = []
    cue_source = arrays["cue"][:ROWS_PER_GROUP]
    current_source = arrays["current"][:ROWS_PER_GROUP]
    for recipient in range(ROWS_PER_GROUP):
        donors = [donor for donor in range(ROWS_PER_GROUP) if donor != recipient]
        cue = torch.from_numpy(np.asarray(cue_source[donors])).to(
            device=device, dtype=torch.float32
        )
        current_np = np.broadcast_to(
            np.asarray(current_source[recipient])[None],
            (len(donors), *current_source.shape[1:]),
        ).copy()
        current = torch.from_numpy(current_np).to(device=device, dtype=torch.float32)
        output = model(cue, current)
        prediction = output.predictions.detach().cpu().numpy()
        for offset, donor in enumerate(donors):
            passed = bool(np.array_equal(prediction[offset], labels[donor]))
            correct += int(passed)
            total += 1
            records.append(
                {
                    "recipient_row": recipient,
                    "donor_row": donor,
                    "prediction": prediction[offset].tolist(),
                    "donor_target": labels[donor].tolist(),
                    "pair_correct": passed,
                }
            )
    if total != 30:
        raise LearnedSpatialContractError("group0 substitution denominator changed")
    return {"correct": correct, "total": total, "accuracy": correct / total, "records": records}


def run_overfit(
    output_root: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic()
    deadline = started + OVERFIT_WALL_SECONDS
    labels, opened = _load_labels(_LABEL_ROWS["overfit"], "overfit")
    model = _new_training_model("mamba_ordered", device)
    initial = _module_hashes(model)
    optimizer = _new_optimizer(model)
    gradient_seen = {name: False for name in _module_parameters(model)}
    curve = []
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    setup_seconds = time.monotonic() - started
    for step in range(OVERFIT_STEPS):
        if time.monotonic() > deadline:
            raise TimeoutError("group0 overfit exceeded 30 minutes")
        record, gradients = _optimizer_step(
            model,
            optimizer,
            arrays,
            labels,
            groups=(0,),
            step=step,
            total_steps=OVERFIT_STEPS,
            device=device,
        )
        curve.append(record)
        gradient_seen = {name: gradient_seen[name] or gradients[name] for name in gradient_seen}
    model.eval()
    evaluation = _evaluate_group0(model, arrays, labels, device)
    substitution = _group0_substitution(model, arrays, labels, device)
    torch.cuda.synchronize(device)
    wall = time.monotonic() - started
    final_hash = _module_hashes(model)
    first_median = float(statistics.median(record["cross_entropy"] for record in curve[:50]))
    final_median = float(statistics.median(record["cross_entropy"] for record in curve[-50:]))
    gates = {
        "exact_500_optimizer_steps": len(curve) == OVERFIT_STEPS,
        "final50_ce_reduced_at_least_90pct": final_median <= 0.10 * first_median,
        "all_six_pairs_exact": evaluation["section"]["group_equal"]["pair_accuracy"] == 1.0,
        "all_six_modules_have_nonzero_gradient": all(gradient_seen.values()),
        "all_six_modules_changed": all(final_hash[name] != initial[name] for name in gradient_seen),
        "learned_queries_changed": final_hash["resampler_queries"] != initial["resampler_queries"],
        "donor_substitution_at_least_80pct": substitution["total"] == 30
        and substitution["accuracy"] >= 0.80,
        "all_outputs_and_metrics_finite": evaluation["logits_finite"]
        and all(np.isfinite(record["cross_entropy"]) for record in curve),
        "wall_within_30_minutes": wall <= OVERFIT_WALL_SECONDS,
        "checkpoint_not_written": True,
    }
    result = {
        "schema": f"{EXPERIMENT_ID}/overfit-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "group0_overfit",
        "condition": "mamba_ordered",
        "seed": TRAIN_SEED,
        "group": 0,
        "curve": curve,
        "setup_seconds": setup_seconds,
        "first50_median_cross_entropy": first_median,
        "final50_median_cross_entropy": final_median,
        "evaluation": evaluation,
        "substitution": substitution,
        "initial_module_sha256": initial,
        "final_module_sha256": final_hash,
        "gradient_seen": gradient_seen,
        "trainable_parameters": trainable_parameter_count(model),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gates": gates,
        "wall_seconds": wall,
        "runtime": dict(runtime),
        "provenance": dict(provenance),
        "target_manifest_opened": False,
        "target_npz_opened": opened,
        "fresh_data_opened": False,
        "action_outputs": 0,
        "optimizer_steps": OVERFIT_STEPS,
        "checkpoint_written": False,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    _write_result(output_root, result, limit=TRAIN_ARTIFACT_LIMIT)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--processor-smoke-result", type=Path)
    parser.add_argument("--processor-smoke-sha256")
    parser.add_argument("--layout-result", type=Path)
    parser.add_argument("--layout-sha256")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-sha256")
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--preflight-sha256")
    return parser.parse_args(argv)


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"missing required {label}")
    return value


def _require_phase_receipt(
    path: Path,
    sha256: str,
    *,
    mode: str,
    schema: str,
    expected_gate_keys: Sequence[str],
    provenance_key: str,
    provenance_value: Mapping[str, Any],
    expected_fields: Mapping[str, Any] | None = None,
    expected_target_rows: Sequence[int] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, receipt = _read_receipt(path, sha256, mode)
    if value.get("schema") != schema or not _exact_passed_gates(value, expected_gate_keys):
        raise PermissionError(f"{mode} exact gates did not all pass")
    if value.get("provenance", {}).get(provenance_key) != provenance_value:
        raise PermissionError(f"{mode}/{provenance_key} provenance changed")
    if expected_fields is not None and any(
        value.get(key) != expected for key, expected in expected_fields.items()
    ):
        raise PermissionError(f"{mode} frozen result fields changed")
    opened = value.get("target_npz_opened")
    if not isinstance(opened, list) or tuple(record.get("row_id") for record in opened) != tuple(
        expected_target_rows
    ):
        raise PermissionError(f"{mode} target row boundary changed")
    if (
        value.get("target_manifest_opened") is not False
        or value.get("fresh_data_opened") is not False
    ):
        raise PermissionError(f"{mode} privileged data boundary changed")
    return value, receipt


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.phase == "source_acceptance":
        result = run_source_acceptance(args.output_root, args.repo_root)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0

    source, source_receipt = _read_receipt(
        _required(args.source_result, "--source-result"),
        _required(args.source_sha256, "--source-sha256"),
        "source_acceptance",
    )
    current_source = bind_runtime_source(source, args.repo_root)
    provenance: dict[str, Any] = {
        "source_acceptance": source_receipt,
        "runtime_source_identity": current_source,
    }
    if args.phase == "processor_smoke":
        provenance["runtime_external_materials"] = bind_runtime_external_materials(source)
        result = run_processor_smoke(args.output_root, provenance=provenance)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result["status"] == "passed" else 2

    smoke, smoke_receipt = _require_phase_receipt(
        _required(args.processor_smoke_result, "--processor-smoke-result"),
        _required(args.processor_smoke_sha256, "--processor-smoke-sha256"),
        mode="processor_smoke",
        schema=f"{EXPERIMENT_ID}/processor-smoke-v1",
        expected_gate_keys=PROCESSOR_SMOKE_GATE_KEYS,
        provenance_key="source_acceptance",
        provenance_value=source_receipt,
        expected_fields={"optimizer_steps": 0, "action_outputs": 0},
    )
    del smoke
    provenance["processor_smoke"] = smoke_receipt
    if args.phase == "layout":
        provenance["runtime_external_materials"] = bind_runtime_external_materials(source)
        device, runtime = _formal_runtime(args.device)
        torch.cuda.reset_peak_memory_stats(device)
        result = run_layout(args.output_root, device=device, runtime=runtime, provenance=provenance)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result["status"] == "passed" else 2

    layout, layout_receipt = _require_phase_receipt(
        _required(args.layout_result, "--layout-result"),
        _required(args.layout_sha256, "--layout-sha256"),
        mode="layout_probe",
        schema=f"{EXPERIMENT_ID}/layout-v1",
        expected_gate_keys=LAYOUT_GATE_KEYS,
        provenance_key="processor_smoke",
        provenance_value=smoke_receipt,
        expected_fields={
            "boundary_indices": [519, 1039],
            "optimizer_steps": 0,
            "action_outputs": 0,
        },
    )
    if layout.get("provenance", {}).get("source_acceptance") != source_receipt:
        raise PermissionError("layout/source provenance changed")
    provenance["layout_probe"] = layout_receipt
    if args.phase == "cache":
        provenance["runtime_external_materials"] = bind_runtime_external_materials(source)
        device, runtime = _formal_runtime(args.device)
        torch.cuda.reset_peak_memory_stats(device)
        result = run_cache(args.output_root, device=device, runtime=runtime, provenance=provenance)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result["status"] == "passed" else 2

    cache_root = _required(args.cache_root, "--cache-root")
    arrays, _sample_ids, cache, cache_receipt = _load_cache(
        cache_root, _required(args.cache_sha256, "--cache-sha256")
    )
    if cache.get("provenance", {}).get("layout_probe") != layout_receipt:
        raise PermissionError("cache/layout provenance changed")
    if cache.get("provenance", {}).get("source_acceptance") != source_receipt:
        raise PermissionError("cache/source provenance changed")
    provenance["target_blind_cache"] = cache_receipt
    provenance["runtime_external_materials"] = bind_runtime_external_materials(source)
    device, runtime = _formal_runtime(args.device)
    if args.phase == "preflight":
        result = run_preflight(
            args.output_root,
            arrays=arrays,
            device=device,
            runtime=runtime,
            provenance=provenance,
        )
    else:
        preflight, preflight_receipt = _require_phase_receipt(
            _required(args.preflight_result, "--preflight-result"),
            _required(args.preflight_sha256, "--preflight-sha256"),
            mode="training_preflight",
            schema=f"{EXPERIMENT_ID}/preflight-v1",
            expected_gate_keys=PREFLIGHT_GATE_KEYS,
            provenance_key="target_blind_cache",
            provenance_value=cache_receipt,
            expected_fields={
                "condition": "mamba_ordered",
                "seed": TRAIN_SEED,
                "groups": [0, 1, 2, 3, 4],
                "microbatch": MICROBATCH,
                "gradient_accumulation": ACCUMULATION,
                "optimizer_steps": PREFLIGHT_STEPS,
                "checkpoint_written": False,
            },
            expected_target_rows=_LABEL_ROWS["preflight"],
        )
        del preflight
        provenance["training_preflight"] = preflight_receipt
        result = run_overfit(
            args.output_root,
            arrays=arrays,
            device=device,
            runtime=runtime,
            provenance=provenance,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
