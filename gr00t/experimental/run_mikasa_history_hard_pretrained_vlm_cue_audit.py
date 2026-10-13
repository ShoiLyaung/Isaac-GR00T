#!/usr/bin/env python3
"""Run one write-once phase of the MIKASA pretrained-VLM cue audit."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Any, Final

import numpy as np
from PIL import Image
import torch

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
from gr00t.experimental.mikasa_history_hard_pretrained_vlm_cue_audit import (
    ALL_CONDITIONS,
    CAMERAS,
    CUE_FRAMES,
    DESCRIPTIVE_CONDITION,
    DEV_GROUPS,
    DEV_ROWS,
    EMBED_DIM,
    EXPERIMENT_ID,
    PAIR_CODES,
    PRIMARY_CONDITION,
    ROWS,
    ROWS_PER_GROUP,
    SHUFFLE_SEEDS,
    TRAIN_GROUPS,
    TRAIN_ROWS,
    CueAuditContractError,
    Standardizer,
    apply_standardizer,
    audit_gates,
    build_neighbor_mapping,
    condition_slots,
    encode_public_merged_descriptors,
    fit_standardizer,
    mean_score_sections,
    score_neighbor_mapping,
    validate_descriptor_cache,
)


REPO: Final = Path(__file__).resolve().parents[2]
BASE_COMMIT: Final = "7aa3ec5a31d6c3276abda0e660c378ce939aa70b"
SOURCE_PATHS: Final = (
    "gr00t/experimental/mikasa_history_hard_pretrained_vlm_cue_audit.py",
    "gr00t/experimental/run_mikasa_history_hard_pretrained_vlm_cue_audit.py",
    "tests/test_mikasa_history_hard_pretrained_vlm_cue_audit.py",
)
PHASES: Final = ("source_acceptance", "layout", "cache", "train_audit", "dev")

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
PROMPT: Final = "Trace the demonstrated paths in the shown order."
ALTERNATE_PROMPT: Final = "This fixed text must not change the visual descriptor."
FRAME_BATCH_SIZE: Final = 8
LAYOUT_WALL_SECONDS: Final = 10 * 60.0
LAYOUT_ARTIFACT_LIMIT: Final = 32 * 1024 * 1024
CACHE_WALL_SECONDS: Final = 45 * 60.0
CACHE_ARTIFACT_LIMIT: Final = 512 * 1024 * 1024

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


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode(errors="replace").strip()) from exc


def source_identity(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
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
    dirty = (
        _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_PATHS)
        .decode()
        .splitlines()
    )
    if dirty:
        raise RuntimeError(f"cue-audit source is dirty: {dirty}")
    files = {}
    for relative in SOURCE_PATHS:
        path = repo / relative
        value = path.read_bytes()
        if path.is_symlink() or value != _git(repo, "show", f"HEAD:{relative}"):
            raise RuntimeError(f"cue-audit source differs from HEAD: {relative}")
        files[relative] = {"bytes": len(value), "sha256": sha256_bytes(value)}
    return {
        "base_commit": BASE_COMMIT,
        "head_commit": head,
        "branch": _git(repo, "branch", "--show-current").decode().strip(),
        "exact_diff": observed,
        "files": files,
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


def protocol() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "phases": list(PHASES),
        "camera_order": list(CAMERAS),
        "prompt": PROMPT,
        "alternate_prompt": ALTERNATE_PROMPT,
        "public_boundary": "model.backbone.model.get_image_features",
        "pre_run_sha256": PRE_RUN_SHA256,
        "privileged_label_manifest_sha256_frozen_but_unopened": (PRIVILEGED_LABEL_MANIFEST_SHA256),
        "merged_tokens_per_image": 64,
        "embedding_dim": EMBED_DIM,
        "cue_frames": CUE_FRAMES,
        "slot_frames": 65,
        "frame_batch_size": FRAME_BATCH_SIZE,
        "conditions": list(ALL_CONDITIONS),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "train_groups": list(TRAIN_GROUPS),
        "dev_groups": list(DEV_GROUPS),
        "standardization": {
            "axes": "candidate rows x slots x time; separate camera x feature dimensions",
            "train_loo": "five candidate geometry groups per held-out fold",
            "dev": "all train groups 0..5",
            "constant_scale": 1.0,
        },
        "formal_runtime": {
            "python": str(FORMAL_PYTHON),
            "torch": FORMAL_TORCH,
            "transformers": FORMAL_TRANSFORMERS,
            "device": "physical cuda:0",
        },
        "layout_wall_seconds": LAYOUT_WALL_SECONDS,
        "layout_artifact_limit_bytes": LAYOUT_ARTIFACT_LIMIT,
        "cache_wall_seconds": CACHE_WALL_SECONDS,
        "cache_artifact_limit_bytes": CACHE_ARTIFACT_LIMIT,
        "optimizer_steps": 0,
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
        "tests/test_mikasa_history_hard_pretrained_vlm_cue_audit.py",
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
    pre_run = validate_pre_run()
    evidence = validate_frozen_evidence(include_target=False)
    tests = _run_focused_tests(repo)
    result = {
        "schema": f"{EXPERIMENT_ID}/source-acceptance-v1",
        "status": "passed",
        "mode": "source_acceptance",
        "source_identity": identity,
        "pre_run_identity": pre_run,
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
    return device, {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gr00t_file": str(gr00t_file),
    }


def _load_model_and_processor(device: torch.device):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    loading = {"local_files_only": True, "trust_remote_code": True}
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
        transformers_loading_kwargs=loading,
    )
    processor.eval()
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    if (
        model.config.tune_visual is not True
        or model.config.tune_llm is not False
        or model.config.backbone_embedding_dim != EMBED_DIM
    ):
        raise RuntimeError("loaded Yuqi visual contract changed")
    if any(
        parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()
    ):
        raise RuntimeError("frozen model has trainable parameters or gradients")
    return model, processor, info


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

    if rgb.ndim != 5 or rgb.shape[1:] != (len(CAMERAS), 128, 128, 3) or rgb.dtype != np.uint8:
        raise CueAuditContractError("RGB packaging requires uint8 [time,2,128,128,3]")
    images = [Image.fromarray(frame) for frame_pair in rgb for frame in frame_pair]
    transformed, _ = apply_with_replay(processor.eval_image_transform, images)
    stacked = torch.stack(transformed)
    content = processor._apply_vlm_processing(stacked, prompt)
    batch = processor.collator([content])["inputs"]
    pixel_values = batch["pixel_values"].detach().cpu().contiguous()
    image_grid_thw = batch["image_grid_thw"].detach().cpu().contiguous()
    expected = torch.tensor([[1, 16, 16]] * (len(rgb) * len(CAMERAS)), dtype=torch.int64)
    if image_grid_thw.dtype != torch.int64 or not torch.equal(image_grid_thw, expected):
        raise CueAuditContractError("native processor image grid/order changed")
    if pixel_values.dtype != torch.float32 or not bool(torch.isfinite(pixel_values).all()):
        raise CueAuditContractError("native processor pixel_values changed")
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
            or cameras != CAMERAS
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
        descriptor, receipt = encode_public_merged_descriptors(
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
        and str(receipt.get("owner_type", "")).endswith(".Qwen3VLForConditionalGeneration")
        and str(receipt.get("method_qualname", "")).endswith(
            "Qwen3VLForConditionalGeneration.get_image_features"
        )
    )


def _write_result(
    output_root: Path, result: Mapping[str, Any], *, limit: int | None = None
) -> None:
    payload = canonical_json_bytes(result)
    projected = _directory_bytes(output_root) + len(payload)
    if limit is not None and projected > limit:
        raise RuntimeError(f"artifact exceeds frozen byte limit: {projected} > {limit}")
    write_once(output_root / "RESULT.json", payload)
    freeze_directory(output_root)


def run_layout(
    output_root: Path,
    *,
    device: torch.device,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic()
    sources = HistoryHardSources.discover()
    cue, _ = _load_row_rgb(sources.rows[0], sources.input_samples[0])
    model, processor, loading = _load_model_and_processor(device)
    versions = _parameter_versions(model)
    with _forbidden_model_paths(model) as forbidden:
        first, pixels_a, grid_a, boundary = _encode_rgb_batch(
            model, processor, cue[:1], PROMPT, device, hash_tokens=True
        )
        alternate, pixels_b, grid_b, alternate_boundary = _encode_rgb_batch(
            model, processor, cue[:1], ALTERNATE_PROMPT, device, hash_tokens=True
        )
        repeated, pixels_c, grid_c, repeated_boundary = _encode_rgb_batch(
            model, processor, cue[:1], PROMPT, device, hash_tokens=True
        )
    elapsed = time.monotonic() - started
    gates = {
        "two_ordered_images": first.shape == (1, 2, EMBED_DIM),
        "grid_exact": torch.equal(grid_a, torch.tensor([[1, 16, 16], [1, 16, 16]])),
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
        "public_boundary_exact": _public_boundary_exact(boundary, 2),
        "forbidden_paths_zero": forbidden == [],
        "parameter_versions_unchanged": versions == _parameter_versions(model),
        "gradients_zero": all(parameter.grad is None for parameter in model.parameters()),
        "wall_within_10_minutes": elapsed <= LAYOUT_WALL_SECONDS,
    }
    result = {
        "schema": f"{EXPERIMENT_ID}/layout-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "layout_probe",
        "row_id": 0,
        "camera_order": list(CAMERAS),
        "image_grid_thw": grid_a.tolist(),
        "descriptor_shape": list(first.shape),
        "descriptor_dtype": str(first.dtype),
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


def _npz_bytes(values: Mapping[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **{key: values[key] for key in sorted(values)})
    return stream.getvalue()


def run_cache(
    output_root: Path,
    *,
    device: torch.device,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic()
    sources = HistoryHardSources.discover()
    model, processor, loading = _load_model_and_processor(device)
    versions = _parameter_versions(model)
    all_cue, all_current = [], []
    first_batch_exact = None
    boundary_receipt = None
    with _forbidden_model_paths(model) as forbidden:
        for row, sample in zip(sources.rows, sources.input_samples, strict=True):
            cue_rgb, current_rgb = _load_row_rgb(row, sample)
            frames = np.concatenate((cue_rgb, current_rgb[None]), axis=0)
            encoded = []
            for start in range(0, len(frames), FRAME_BATCH_SIZE):
                if time.monotonic() - started > CACHE_WALL_SECONDS:
                    raise TimeoutError("target-blind descriptor cache exceeded 45 minutes")
                batch = frames[start : start + FRAME_BATCH_SIZE]
                descriptor, _, _, receipt = _encode_rgb_batch(
                    model,
                    processor,
                    batch,
                    PROMPT,
                    device,
                    hash_tokens=first_batch_exact is None,
                )
                boundary_receipt = boundary_receipt or receipt
                if first_batch_exact is None:
                    repeated, _, _, repeated_receipt = _encode_rgb_batch(
                        model, processor, batch, PROMPT, device, hash_tokens=True
                    )
                    first_batch_exact = bool(
                        torch.equal(descriptor, repeated)
                        and receipt["merged_token_sha256"]
                        == repeated_receipt["merged_token_sha256"]
                    )
                encoded.append(descriptor)
            row_value = torch.cat(encoded, dim=0).numpy().astype(np.float32, copy=False)
            if row_value.shape != (CUE_FRAMES + 1, len(CAMERAS), EMBED_DIM):
                raise CueAuditContractError("row descriptor count changed")
            all_cue.append(np.ascontiguousarray(row_value[:CUE_FRAMES]))
            all_current.append(np.ascontiguousarray(row_value[CUE_FRAMES]))
    cue = np.ascontiguousarray(np.stack(all_cue), dtype=np.float32)
    current = np.ascontiguousarray(np.stack(all_current), dtype=np.float32)
    sample_ids = tuple(row.sample_id for row in sources.rows)
    validate_descriptor_cache(cue, current, sample_ids)
    current_equal = {
        str(group): all(
            np.array_equal(
                current[group * ROWS_PER_GROUP], current[group * ROWS_PER_GROUP + offset]
            )
            for offset in range(1, ROWS_PER_GROUP)
        )
        for group in range(8)
    }
    cue_hashes = {
        str(group): [
            hashlib.sha256(memoryview(cue[group * ROWS_PER_GROUP + offset]).cast("B")).hexdigest()
            for offset in range(ROWS_PER_GROUP)
        ]
        for group in range(8)
    }
    elapsed = time.monotonic() - started
    gates = {
        "all_48_rows": cue.shape[0] == ROWS,
        "descriptor_shapes_exact": cue.shape == (48, 130, 2, 2048)
        and current.shape == (48, 2, 2048),
        "all_finite_float32": cue.dtype == current.dtype == np.float32
        and np.isfinite(cue).all()
        and np.isfinite(current).all(),
        "six_current_exact_each_geometry": all(current_equal.values()),
        "six_unique_cue_hashes_each_geometry": all(
            len(set(values)) == ROWS_PER_GROUP for values in cue_hashes.values()
        ),
        "first_batch_repeat_exact": first_batch_exact is True,
        "public_boundary_exact": boundary_receipt is not None
        and _public_boundary_exact(boundary_receipt, FRAME_BATCH_SIZE * len(CAMERAS)),
        "forbidden_paths_zero": forbidden == [],
        "parameter_versions_unchanged": versions == _parameter_versions(model),
        "gradients_zero": all(parameter.grad is None for parameter in model.parameters()),
        "wall_within_45_minutes": elapsed <= CACHE_WALL_SECONDS,
    }
    feature_payload = _npz_bytes({"cue": cue, "current": current})
    if len(feature_payload) > CACHE_ARTIFACT_LIMIT - 1024 * 1024:
        raise RuntimeError("descriptor NPZ leaves insufficient room for the frozen cache result")
    output_root.mkdir(parents=True)
    feature = write_once(
        output_root / "DESCRIPTORS.npz",
        feature_payload,
    )
    result = {
        "schema": f"{EXPERIMENT_ID}/cache-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "target_blind_cache",
        "rows": ROWS,
        "feature": feature,
        "sample_ids": list(sample_ids),
        "source_model_sha256": [row.model.sha256 for row in sources.rows],
        "camera_order": list(CAMERAS),
        "frame_batch_size": FRAME_BATCH_SIZE,
        "public_boundary": boundary_receipt,
        "current_equal_by_geometry": current_equal,
        "cue_hashes_by_geometry": cue_hashes,
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
    _write_result(output_root, result, limit=CACHE_ARTIFACT_LIMIT)
    return result


def _load_cache(
    cache_root: Path, expected_sha256: str
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any], dict[str, Any]]:
    result, identity = _read_receipt(
        cache_root / "RESULT.json", expected_sha256, "target_blind_cache"
    )
    if (
        result.get("schema") != f"{EXPERIMENT_ID}/cache-v1"
        or result.get("target_manifest_opened") is not False
        or result.get("target_npz_opened") != []
        or result.get("fresh_data_opened") is not False
        or not all(result.get("gates", {}).values())
        or result.get("rows") != ROWS
        or result.get("camera_order") != list(CAMERAS)
        or result.get("frame_batch_size") != FRAME_BATCH_SIZE
    ):
        raise PermissionError("descriptor cache boundary did not pass")
    sources = HistoryHardSources.discover()
    expected_sample_ids = tuple(row.sample_id for row in sources.rows)
    expected_model_sha256 = [row.model.sha256 for row in sources.rows]
    if (
        tuple(result.get("sample_ids", ())) != expected_sample_ids
        or result.get("source_model_sha256") != expected_model_sha256
    ):
        raise PermissionError("descriptor cache/source row identity changed")
    feature = result.get("feature")
    if not isinstance(feature, Mapping):
        raise PermissionError("descriptor cache feature receipt missing")
    path = Path(str(feature.get("path")))
    if (
        path.resolve() != (cache_root.resolve() / "DESCRIPTORS.npz")
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != feature.get("bytes")
        or sha256_file(path) != feature.get("sha256")
    ):
        raise PermissionError("descriptor cache feature identity changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"cue", "current"}:
            raise PermissionError("descriptor cache field set changed")
        cue = np.ascontiguousarray(archive["cue"])
        current = np.ascontiguousarray(archive["current"])
    sample_ids = tuple(str(value) for value in result.get("sample_ids", ()))
    validate_descriptor_cache(cue, current, sample_ids)
    return cue, current, sample_ids, result, identity


def _validate_label_rows(row_ids: Sequence[int], phase: str) -> tuple[int, ...]:
    rows = tuple(int(value) for value in row_ids)
    expected = TRAIN_ROWS if phase == "train_audit" else DEV_ROWS if phase == "dev" else ()
    if rows != expected:
        raise PermissionError(f"{phase} may open only its exact frozen target rows")
    return rows


def _load_labels(row_ids: Sequence[int], phase: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
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


def _standardizer_arrays(standardizers: Mapping[str, Standardizer]) -> dict[str, np.ndarray]:
    values = {}
    for name, standardizer in standardizers.items():
        standardizer.validate()
        values[f"{name}_mean"] = standardizer.mean
        values[f"{name}_scale"] = standardizer.scale
    return values


def _load_standardizers(path: Path, receipt: Mapping[str, Any]) -> dict[str, Standardizer]:
    if (
        path.resolve() != Path(str(receipt.get("path"))).resolve()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != receipt.get("bytes")
        or sha256_file(path) != receipt.get("sha256")
    ):
        raise PermissionError("train standardizer identity changed")
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            f"{name}_{suffix}"
            for name in (
                PRIMARY_CONDITION,
                "current_repeat",
                "endpoint_repeat",
                DESCRIPTIVE_CONDITION,
            )
            for suffix in ("mean", "scale")
        }
        if set(archive.files) != expected:
            raise PermissionError("train standardizer field set changed")
        result = {
            name: Standardizer(
                np.ascontiguousarray(archive[f"{name}_mean"]),
                np.ascontiguousarray(archive[f"{name}_scale"]),
            )
            for name in (
                PRIMARY_CONDITION,
                "current_repeat",
                "endpoint_repeat",
                DESCRIPTIVE_CONDITION,
            )
        }
    for value in result.values():
        value.validate()
    return result


def _condition_evidence(
    cue: np.ndarray,
    current: np.ndarray,
    sample_ids: Sequence[str],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    query_rows: Sequence[int],
    candidate_rows: Sequence[int],
    standardizers: Mapping[str, Standardizer],
) -> dict[str, Any]:
    fitted = dict(standardizers)
    evidence: dict[str, Any] = {}
    for condition in (
        PRIMARY_CONDITION,
        "current_repeat",
        "endpoint_repeat",
        DESCRIPTIVE_CONDITION,
    ):
        slots = condition_slots(cue, current, sample_ids, condition)
        if condition not in fitted:
            raise CueAuditContractError(f"missing frozen train standardizer: {condition}")
        standardized = apply_standardizer(slots, fitted[condition])
        mapping = build_neighbor_mapping(
            standardized,
            sample_ids,
            groups,
            query_rows=query_rows,
            candidate_rows=candidate_rows,
        )
        evidence[condition] = score_neighbor_mapping(mapping, labels, groups)
    shuffle_sections = []
    for seed in SHUFFLE_SEEDS:
        slots = condition_slots(
            cue,
            current,
            sample_ids,
            "sample_shuffle",
            shuffle_seed=seed,
        )
        standardized = apply_standardizer(slots, fitted[PRIMARY_CONDITION])
        mapping = build_neighbor_mapping(
            standardized,
            sample_ids,
            groups,
            query_rows=query_rows,
            candidate_rows=candidate_rows,
        )
        shuffle_sections.append(score_neighbor_mapping(mapping, labels, groups))
    evidence["sample_shuffle"] = mean_score_sections(shuffle_sections)
    return evidence


def _merge_single_group_sections(sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    by_group: dict[str, Any] = {}
    records = []
    for section in sections:
        if len(section["by_group"]) != 1:
            raise CueAuditContractError("LOO fold must score exactly one held-out group")
        group, values = next(iter(section["by_group"].items()))
        if group in by_group:
            raise CueAuditContractError("LOO held-out group repeated")
        by_group[group] = dict(values)
        records.extend(section["records"])
    return {
        "records": records,
        "by_group": by_group,
        "group_equal": {
            metric: float(np.mean([values[metric] for values in by_group.values()]))
            for metric in metrics
        },
    }


def _train_loo_evidence(
    cue: np.ndarray,
    current: np.ndarray,
    sample_ids: Sequence[str],
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Fit every fold only on its five candidate geometries, then score the sixth."""

    evidence: dict[str, Any] = {}
    for condition in (
        PRIMARY_CONDITION,
        "current_repeat",
        "endpoint_repeat",
        DESCRIPTIVE_CONDITION,
    ):
        slots = condition_slots(cue, current, sample_ids, condition)
        folds = []
        for held_out in TRAIN_GROUPS:
            query_rows = tuple(row for row in TRAIN_ROWS if int(groups[row]) == held_out)
            candidate_rows = tuple(row for row in TRAIN_ROWS if int(groups[row]) != held_out)
            standardizer = fit_standardizer(slots, candidate_rows)
            standardized = apply_standardizer(slots, standardizer)
            mapping = build_neighbor_mapping(
                standardized,
                sample_ids,
                groups,
                query_rows=query_rows,
                candidate_rows=candidate_rows,
            )
            folds.append(score_neighbor_mapping(mapping, labels, groups))
        evidence[condition] = _merge_single_group_sections(folds)
    shuffle_sections = []
    for seed in SHUFFLE_SEEDS:
        slots = condition_slots(
            cue,
            current,
            sample_ids,
            "sample_shuffle",
            shuffle_seed=seed,
        )
        folds = []
        for held_out in TRAIN_GROUPS:
            query_rows = tuple(row for row in TRAIN_ROWS if int(groups[row]) == held_out)
            candidate_rows = tuple(row for row in TRAIN_ROWS if int(groups[row]) != held_out)
            standardizer = fit_standardizer(slots, candidate_rows)
            standardized = apply_standardizer(slots, standardizer)
            mapping = build_neighbor_mapping(
                standardized,
                sample_ids,
                groups,
                query_rows=query_rows,
                candidate_rows=candidate_rows,
            )
            folds.append(score_neighbor_mapping(mapping, labels, groups))
        shuffle_sections.append(_merge_single_group_sections(folds))
    evidence["sample_shuffle"] = mean_score_sections(shuffle_sections)
    return evidence


def run_train_audit(
    output_root: Path,
    *,
    cue: np.ndarray,
    current: np.ndarray,
    sample_ids: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    labels, opened = _load_labels(TRAIN_ROWS, "train_audit")
    groups = np.arange(ROWS, dtype=np.int64) // ROWS_PER_GROUP
    evidence = _train_loo_evidence(cue, current, sample_ids, labels, groups)
    standardizers = {
        condition: fit_standardizer(
            condition_slots(cue, current, sample_ids, condition), TRAIN_ROWS
        )
        for condition in (
            PRIMARY_CONDITION,
            "current_repeat",
            "endpoint_repeat",
            DESCRIPTIVE_CONDITION,
        )
    }
    gates = audit_gates(evidence, expected_groups=len(TRAIN_GROUPS))
    output_root.mkdir(parents=True)
    stats = write_once(
        output_root / "TRAIN_STANDARDIZERS.npz",
        _npz_bytes(_standardizer_arrays(standardizers)),
    )
    train_labels = [
        {
            "row_id": row,
            "sample_id": str(sample_ids[row]),
            "group_id": int(groups[row]),
            "ordered_path_ids": labels[row].tolist(),
        }
        for row in TRAIN_ROWS
    ]
    result = {
        "schema": f"{EXPERIMENT_ID}/train-audit-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "train_only_audit",
        "evidence": evidence,
        "gates": gates,
        "standardizers": stats,
        "train_labels": train_labels,
        "provenance": dict(provenance),
        "target_manifest_opened": False,
        "target_npz_opened": opened,
        "dev_target_npz_opened": 0,
        "fresh_data_opened": False,
        "model_calls": 0,
        "action_outputs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    _write_result(output_root, result)
    return result


def _train_labels_from_receipt(train: Mapping[str, Any], sample_ids: Sequence[str]) -> np.ndarray:
    records = train.get("train_labels")
    if not isinstance(records, list) or len(records) != len(TRAIN_ROWS):
        raise PermissionError("train candidate labels are missing")
    labels = np.full((ROWS, 2), -1, dtype=np.int64)
    for expected_row, record in zip(TRAIN_ROWS, records, strict=True):
        if (
            record.get("row_id") != expected_row
            or record.get("sample_id") != sample_ids[expected_row]
            or record.get("group_id") != expected_row // ROWS_PER_GROUP
        ):
            raise PermissionError("train candidate label identity changed")
        ordered = np.asarray(record.get("ordered_path_ids"), dtype=np.int64)
        if ordered.shape != (2,) or ordered.tolist() != [
            int(PAIR_CODES[expected_row % ROWS_PER_GROUP][0]),
            int(PAIR_CODES[expected_row % ROWS_PER_GROUP][1]),
        ]:
            raise PermissionError("train candidate ordered label changed")
        labels[expected_row] = ordered
    return labels


def run_dev(
    output_root: Path,
    *,
    cue: np.ndarray,
    current: np.ndarray,
    sample_ids: Sequence[str],
    train: Mapping[str, Any],
    train_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    if not all(train.get("gates", {}).values()):
        raise PermissionError("train-only audit gates did not all pass")
    labels = _train_labels_from_receipt(train, sample_ids)
    dev_labels, opened = _load_labels(DEV_ROWS, "dev")
    labels[np.asarray(DEV_ROWS)] = dev_labels[np.asarray(DEV_ROWS)]
    standardizer_receipt = train.get("standardizers")
    if not isinstance(standardizer_receipt, Mapping):
        raise PermissionError("train standardizer receipt missing")
    standardizers = _load_standardizers(
        train_root / "TRAIN_STANDARDIZERS.npz", standardizer_receipt
    )
    groups = np.arange(ROWS, dtype=np.int64) // ROWS_PER_GROUP
    evidence = _condition_evidence(
        cue,
        current,
        sample_ids,
        labels,
        groups,
        query_rows=DEV_ROWS,
        candidate_rows=TRAIN_ROWS,
        standardizers=standardizers,
    )
    gates = audit_gates(evidence, expected_groups=len(DEV_GROUPS))
    result = {
        "schema": f"{EXPERIMENT_ID}/dev-v1",
        "status": "passed" if all(gates.values()) else "gate_failed",
        "mode": "revealed_dev",
        "evidence": evidence,
        "gates": gates,
        "provenance": dict(provenance),
        "target_manifest_opened": False,
        "target_npz_opened": opened,
        "train_target_npz_reopened": 0,
        "fresh_data_opened": False,
        "model_calls": 0,
        "action_outputs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    _write_result(output_root, result)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--layout-result", type=Path)
    parser.add_argument("--layout-sha256")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-sha256")
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--train-sha256")
    return parser.parse_args(argv)


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"missing required {label}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.phase == "source_acceptance":
        result = run_source_acceptance(args.output_root, args.repo_root)
    else:
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
        if args.phase == "layout":
            device, runtime = _formal_runtime(args.device)
            torch.cuda.reset_peak_memory_stats(device)
            result = run_layout(
                args.output_root,
                device=device,
                runtime=runtime,
                provenance=provenance,
            )
        else:
            layout, layout_receipt = _read_receipt(
                _required(args.layout_result, "--layout-result"),
                _required(args.layout_sha256, "--layout-sha256"),
                "layout_probe",
            )
            if layout.get("schema") != f"{EXPERIMENT_ID}/layout-v1" or not all(
                layout.get("gates", {}).values()
            ):
                raise PermissionError("layout probe gates did not all pass")
            if layout.get("provenance", {}).get("source_acceptance") != source_receipt:
                raise PermissionError("layout/source-acceptance provenance changed")
            provenance["layout_probe"] = layout_receipt
            if args.phase == "cache":
                device, runtime = _formal_runtime(args.device)
                torch.cuda.reset_peak_memory_stats(device)
                result = run_cache(
                    args.output_root,
                    device=device,
                    runtime=runtime,
                    provenance=provenance,
                )
            else:
                cache_root = _required(args.cache_root, "--cache-root")
                cue, current, sample_ids, cache, cache_receipt = _load_cache(
                    cache_root, _required(args.cache_sha256, "--cache-sha256")
                )
                if cache.get("provenance", {}).get("layout_probe") != layout_receipt:
                    raise PermissionError("cache/layout provenance changed")
                if cache.get("provenance", {}).get("source_acceptance") != source_receipt:
                    raise PermissionError("cache/source-acceptance provenance changed")
                provenance["target_blind_cache"] = cache_receipt
                if args.phase == "train_audit":
                    result = run_train_audit(
                        args.output_root,
                        cue=cue,
                        current=current,
                        sample_ids=sample_ids,
                        provenance=provenance,
                    )
                else:
                    train_root = _required(args.train_root, "--train-root")
                    train, train_receipt = _read_receipt(
                        train_root / "RESULT.json",
                        _required(args.train_sha256, "--train-sha256"),
                        "train_only_audit",
                    )
                    if train.get("schema") != f"{EXPERIMENT_ID}/train-audit-v1":
                        raise PermissionError("train-only audit schema changed")
                    if train.get("provenance", {}).get("target_blind_cache") != cache_receipt:
                        raise PermissionError("train/cache provenance changed")
                    provenance["train_only_audit"] = train_receipt
                    result = run_dev(
                        args.output_root,
                        cue=cue,
                        current=current,
                        sample_ids=sample_ids,
                        train=train,
                        train_root=train_root,
                        provenance=provenance,
                    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
