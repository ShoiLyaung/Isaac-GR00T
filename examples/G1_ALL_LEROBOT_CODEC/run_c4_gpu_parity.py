#!/usr/bin/env python3
"""Run one bounded C4 native-vs-Codec full-model GPU parity attempt."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "g1-all-lerobot-c4-gpu-parity-r3"
ATTEMPT_ID = "G1-ALL-LEROBOT-C4-GPU-PARITY-R3-ATTEMPT1"
REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path("/home/sl/icra27-runs/codec-gpu-parity/g1-all-lerobot-c4-r3")
CHECKPOINT = Path(
    "/home/sl/icra27-runs/research-control/groot-n17-all-lerobot-20k-r1/"
    "experiments/groot-n17-all-lerobot-20k-r1/runs/"
    "train-20k-attempt-1-20260827T192825p0800.artifacts/attempt/train/checkpoint-20000"
)
FIXTURE_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-full-r2")
FIXTURE_MANIFEST_SHA256 = "c05736e1e33745ee0580d45aeebf651891a6533701297f23f41a1bd5090c3a88"
LOADER_VIEW = Path(
    "/home/sl/icra27-runs/groot-n17-all-lerobot-20k-r1/"
    "prep-20260827T171352+0800/sources/no_torque_100/LOADER_VIEW"
)
ADAPTER_CONFIG = Path(
    "/home/sl/icra27-runs/groot-n17-all-lerobot-20k-r1/"
    "prep-20260827T171352+0800/sources/no_torque_100/ADAPTER_CONFIG.json"
)
EPISODE_UID = "cb927e3e88cfe78e4bfe72d42c0ea60fc23de5274478dcd9e9937be6b3e538c9"
EPISODE_INDEX = 64
OBSERVATION_ROW = 64
SEED = 42
MAX_WALL_SECONDS = 1_800
MAX_OUTPUT_BYTES = 1_000_000_000
EMBODIMENT_TAG = "new_embodiment"
CAMERA_ID = "cam_left_high"
SOURCE_KIND = "NATIVE_CANONICAL_DECODED_DERIVED"
SUMMARY_NAME = "C4_PARITY_SUMMARY.json"
TERMINAL_NAME = "C4_TERMINAL.json"
ACTION_NAME = "ACTION_PRED.npy"
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "GROOT_HF_LOCAL_FIRST": "1",
    "GROOT_PATCH_MISTRAL": "1",
    "NO_ALBUMENTATIONS_UPDATE": "1",
}
EXPECTED_SHARDS = {
    "model-00001-of-00002.safetensors": "5524cf9933dd6ec40687637444b22db6dc76921dfe0e1843c7140e8701763cf7",
    "model-00002-of-00002.safetensors": "51784715d68c870fbbcdb8b5790c4d4514b86f91722506e3d04d6df1b594739d",
}
MODEL_CODEC_CONFIG = {
    "enabled": True,
    "modalities": ["mv", "rpe", "i_frame", "metadata"],
    "gate_init": 0.0,
    "schema_version": "codec-evidence-episode-bundle-v1",
    "camera_id": CAMERA_ID,
    "source_kind": SOURCE_KIND,
    "d_codec": 64,
    "conditioner_heads": 8,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected one regular non-symlink JSON file: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Identity input must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_path(resolved),
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def _checkpoint_identity() -> dict[str, Any]:
    checkpoint = CHECKPOINT.resolve(strict=True)
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError("C4 checkpoint must be one real directory")
    config = _load_json(checkpoint / "config.json")
    processor = _load_json(checkpoint / "processor_config.json")
    index = _load_json(checkpoint / "model.safetensors.index.json")
    if config.get("architectures") != ["Gr00tN1d7"] or config.get("action_horizon") != 40:
        raise ValueError("C4 checkpoint architecture/H40 identity drifted")
    saved_codec = config.get("codec_evidence")
    if saved_codec is not None and saved_codec != {"enabled": False}:
        raise ValueError("C4 requires one native checkpoint without Codec weights")
    if processor.get("processor_kwargs", {}).get("codec_evidence") is not None:
        raise ValueError("C4 native checkpoint processor unexpectedly enables Codec evidence")
    if set(index.get("weight_map", {}).values()) != set(EXPECTED_SHARDS):
        raise ValueError("C4 checkpoint shard index differs from the frozen pair")
    shards = {}
    for name, expected in EXPECTED_SHARDS.items():
        identity = _file_identity(checkpoint / name)
        if identity["sha256"] != expected:
            raise RuntimeError(f"C4 checkpoint shard drifted: {name}")
        shards[name] = identity
    return {
        "path": str(checkpoint),
        "config": _file_identity(checkpoint / "config.json"),
        "processor_config": _file_identity(checkpoint / "processor_config.json"),
        "index": _file_identity(checkpoint / "model.safetensors.index.json"),
        "shards": shards,
    }


def _fixture_identity() -> dict[str, Any]:
    from gr00t.data.codec_evidence.episode_bundle import EpisodeBundleKey
    from gr00t.data.codec_evidence.warehouse import validate_full_fixture_manifest, verify_object

    root = FIXTURE_ROOT.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("C4 fixture root must be one real directory")
    manifest_path = root / "fixture_manifest.json"
    if _sha256_path(manifest_path) != FIXTURE_MANIFEST_SHA256:
        raise RuntimeError("C4 fixture manifest identity drifted")
    manifest = validate_full_fixture_manifest(
        root,
        fixture_manifest_sha256=FIXTURE_MANIFEST_SHA256,
        verify_objects=False,
    )
    matches = [item for item in manifest["objects"] if item["episode_uid"] == EPISODE_UID]
    if len(matches) != 1:
        raise RuntimeError("C4 fixed episode UID is absent or duplicate")
    entry = matches[0]
    key = EpisodeBundleKey(EPISODE_UID, CAMERA_ID, SOURCE_KIND)
    object_manifest = verify_object(root, entry["object_content_sha256"], full_bundle=True)
    if (
        object_manifest["episode_uid"] != key.episode_uid
        or object_manifest["num_frames"] != 1_028
        or OBSERVATION_ROW + 40 > object_manifest["num_frames"]
    ):
        raise RuntimeError("C4 selected fixture object length/row drifted")
    return {
        "root": str(root),
        "manifest": _file_identity(manifest_path),
        "selected_entry": entry,
        "selected_object_manifest_sha256": _canonical_sha256(object_manifest),
        "episode_index": EPISODE_INDEX,
        "observation_row": OBSERVATION_ROW,
    }


def _loader_identity() -> dict[str, Any]:
    view = LOADER_VIEW.resolve(strict=True)
    adapter = ADAPTER_CONFIG.resolve(strict=True)
    if not view.is_dir() or not adapter.is_file():
        raise FileNotFoundError("C4 loader view/adapter is missing")
    config = _load_json(adapter)
    if config.get("source_id") != "no_torque_100" or config.get("horizon") != 40:
        raise ValueError("C4 adapter source/H40 identity drifted")
    if config.get("codec_evidence") is not None:
        raise ValueError("C4 source adapter must remain native and read-only")
    for name in ("data", "videos"):
        path = view / name
        if not path.is_symlink():
            raise ValueError(f"C4 loader view {name} must remain a symlink")
    return {
        "loader_view": str(view),
        "adapter": _file_identity(adapter),
        "view_manifest": _file_identity(view / "meta/g1_all_lerobot_h40_view_manifest.json"),
    }


def _gpu_inventory(selector: str) -> dict[str, Any]:
    if not selector.isdigit() or "," in selector:
        raise ValueError("C4 CUDA_VISIBLE_DEVICES must be one physical numeric GPU")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 7:
            raise ValueError("Unexpected nvidia-smi GPU row")
        rows.append(
            {
                "index": values[0],
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": int(values[3]),
                "memory_used_mib": int(values[4]),
                "memory_free_mib": int(values[5]),
                "utilization_percent": int(values[6]),
            }
        )
    selected = [row for row in rows if row["index"] == selector]
    if len(selected) != 1:
        raise RuntimeError("C4 selected physical GPU is absent")
    return selected[0]


def build_plan() -> dict[str, Any]:
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"C4 execution requires a clean worktree:\n{dirty}")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"C4 output root is already consumed: {OUTPUT_ROOT}")
    missing_offline = [key for key, value in OFFLINE_ENV.items() if os.environ.get(key) != value]
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": ATTEMPT_ID,
        "claim_scope": "ONE_REAL_BATCH_NATIVE_VS_ZERO_GATE_CODEC_FULL_MODEL_GPU_EXACT_PARITY",
        "git": {
            "repository": str(REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "dirty_paths": [],
        },
        "checkpoint": _checkpoint_identity(),
        "fixture": _fixture_identity(),
        "loader": _loader_identity(),
        "codec_model_config": dict(MODEL_CODEC_CONFIG),
        "seed": SEED,
        "output_root": str(OUTPUT_ROOT),
        "resources": {
            "gpu_count": 1,
            "cuda_visible_devices": selector,
            "gpu": _gpu_inventory(selector),
            "timeout_seconds": MAX_WALL_SECONDS,
            "maximum_new_bytes": MAX_OUTPUT_BYTES,
            "automatic_retries": 0,
            "offline": True,
        },
        "offline_environment_missing_or_invalid": missing_offline,
        "training": False,
        "evaluator": False,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _set_seed(torch: Any) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _tensor_tree_identity(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        cpu = value.detach().contiguous().cpu()
        return {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "sha256": sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {key: _tensor_tree_identity(item) for key, item in sorted(value.items())}
    raise TypeError(f"Unexpected parity batch value: {type(value).__name__}")


def _assert_native_batch_equal(native: Mapping[str, Any], codec: Mapping[str, Any]) -> None:
    import torch

    if set(codec) != set(native) | {"codec_evidence"}:
        raise RuntimeError("C4 Codec batch differs from native batch beyond codec_evidence")
    for key, value in native.items():
        other = codec[key]
        if not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor):
            raise TypeError(f"C4 native batch key {key} is not a tensor")
        if not torch.equal(value, other):
            raise RuntimeError(f"C4 native preprocessed tensor differs: {key}")


def _build_batches(plan: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    from gr00t.data.codec_evidence.episode_bundle import EpisodeBundleKey
    from gr00t.data.codec_evidence.warehouse import WarehouseEpisodeBundleStore
    from gr00t.data.dataset.g1_all_lerobot_h40_adapter import G1AllLerobotH40EpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    import gr00t.model  # noqa: F401
    import torch
    from transformers import AutoProcessor

    checkpoint = plan["checkpoint"]["path"]
    native_processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    codec_processor = AutoProcessor.from_pretrained(
        checkpoint,
        codec_evidence=dict(MODEL_CODEC_CONFIG),
        local_files_only=True,
    )
    native_processor.eval()
    codec_processor.eval()
    tag = EmbodimentTag(EMBODIMENT_TAG)
    native_modalities = native_processor.get_modality_configs()[tag.value]
    codec_modalities = codec_processor.get_modality_configs()[tag.value]
    if native_modalities != codec_modalities:
        raise RuntimeError("C4 native/Codec modality configs differ")
    loader = G1AllLerobotH40EpisodeLoader(
        plan["loader"]["loader_view"],
        native_modalities,
        plan["loader"]["adapter"]["path"],
        require_runtime_verified_fixed_state=True,
    )
    episode = loader[EPISODE_INDEX]
    step = extract_step_data(
        episode,
        OBSERVATION_ROW,
        native_modalities,
        tag,
        allow_padding=False,
    )
    store = WarehouseEpisodeBundleStore(
        FIXTURE_ROOT,
        fixture_manifest_sha256=FIXTURE_MANIFEST_SHA256,
    )
    key = EpisodeBundleKey(EPISODE_UID, CAMERA_ID, SOURCE_KIND)
    evidence = store.get(key).view(OBSERVATION_ROW)
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
    _set_seed(torch)
    native_feature = native_processor(messages)
    _set_seed(torch)
    codec_feature = codec_processor(messages)
    codec_feature["codec_evidence_sample"] = evidence
    _set_seed(torch)
    native_batch = native_processor.collator([native_feature])
    _set_seed(torch)
    codec_batch = codec_processor.collator([codec_feature])
    native_inputs = native_batch["inputs"]
    codec_inputs = codec_batch["inputs"]
    _assert_native_batch_equal(native_inputs, codec_inputs)
    audit = {
        "episode_uid": EPISODE_UID,
        "episode_index": EPISODE_INDEX,
        "observation_row": OBSERVATION_ROW,
        "native_batch_identity": _tensor_tree_identity(native_inputs),
        "codec_evidence_identity": _tensor_tree_identity(codec_inputs["codec_evidence"]),
    }
    del loader, native_processor, codec_processor
    gc.collect()
    return native_batch, codec_batch, audit


def _loading_failures(info: Mapping[str, Any], allowed_missing: set[str]) -> dict[str, Any]:
    missing = set(info.get("missing_keys", []))
    failures = {
        "missing_other": sorted(missing - allowed_missing),
        "allowed_missing": sorted(missing & allowed_missing),
        "unexpected_keys": list(info.get("unexpected_keys", [])),
        "mismatched_keys": list(info.get("mismatched_keys", [])),
        "error_msgs": list(info.get("error_msgs", [])),
    }
    if failures["missing_other"] or any(
        failures[key] for key in ("unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"C4 checkpoint loading mismatch: {failures}")
    return failures


def _inference_batch(training_batch: Mapping[str, Any]) -> dict[str, Any]:
    inputs = training_batch.get("inputs")
    if not isinstance(inputs, Mapping) or "action" not in inputs:
        raise ValueError("C4 training batch must contain ground-truth action")
    inference_inputs = {key: value for key, value in inputs.items() if key != "action"}
    if "action" in inference_inputs or "action_mask" not in inference_inputs:
        raise RuntimeError("C4 inference input action/action_mask contract drifted")
    return {"inputs": inference_inputs}


def _run_model_phase(
    *,
    codec_enabled: bool,
    batch: Any,
    device: str,
) -> tuple[dict[str, Any], Any, Any]:
    import gr00t.model  # noqa: F401
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    import torch
    from transformers import AutoModel

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)
    _set_seed(torch)
    kwargs = {"local_files_only": True, "output_loading_info": True}
    if codec_enabled:
        kwargs["codec_evidence"] = dict(MODEL_CODEC_CONFIG)
    started = time.monotonic()
    model, info = AutoModel.from_pretrained(CHECKPOINT, **kwargs)
    expected_codec = set()
    if codec_enabled:
        stack = model.action_head.codec_conditioning
        expected_codec = {f"action_head.codec_conditioning.{key}" for key in stack.state_dict()}
        if not expected_codec:
            raise RuntimeError("C4 Codec module identity is invalid")
    loading = _loading_failures(info, expected_codec)
    if codec_enabled and set(loading["allowed_missing"]) != expected_codec:
        raise RuntimeError("C4 native-to-Codec load lacks the complete new module key set")
    if codec_enabled:
        Gr00tN1d7Pipeline._initialize_native_to_codec_gate(
            model,
            "native_to_codec",
            expected_codec,
        )
        if stack.conditioner.gate.item() != 0.0:
            raise RuntimeError("C4 Codec gate is not exact zero after native extension")
    if not codec_enabled and loading["allowed_missing"]:
        raise RuntimeError("C4 native load unexpectedly allows missing keys")
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    calls = {"encode": 0, "condition": 0}
    if codec_enabled:
        original_encode = stack.encode
        original_condition = stack.condition

        def counted_encode(evidence):
            calls["encode"] += 1
            return original_encode(evidence)

        def counted_condition(action, context, mask):
            calls["condition"] += 1
            return original_condition(action, context, mask)

        stack.encode = counted_encode
        stack.condition = counted_condition
    _set_seed(torch)
    with torch.inference_mode():
        forward_output = model(**batch)
    loss = forward_output["loss"].detach().float().cpu()
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise RuntimeError("C4 forward loss is non-finite or non-scalar")
    _set_seed(torch)
    with torch.inference_mode():
        action = model.get_action(**_inference_batch(batch))["action_pred"].detach().float().cpu()
    if action.shape != (1, 40, 132) or not torch.isfinite(action).all():
        raise RuntimeError(f"C4 action prediction shape/finite mismatch: {tuple(action.shape)}")
    torch.cuda.synchronize()
    phase = {
        "codec_enabled": codec_enabled,
        "load_seconds": load_seconds,
        "loss": float(loss.item()),
        "action_shape": list(action.shape),
        "loading": loading,
        "codec_calls": calls,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
    }
    del forward_output, model
    torch.cuda.empty_cache()
    gc.collect()
    return phase, loss, action


@contextlib.contextmanager
def _gpu_lease(plan: Mapping[str, Any]):
    gpu = plan["resources"]["gpu"]
    lease_root = Path("/home/sl/icra27-runs/research-control/leases")
    lease_root.mkdir(parents=True, exist_ok=True)
    lease = lease_root / f"gpu-{gpu['index']}-{gpu['uuid']}.lease"
    descriptor = os.open(lease, os.O_RDWR | os.O_CREAT, 0o664)
    handle = os.fdopen(descriptor, "r+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"C4 GPU lease is held: {lease}") from error
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"attempt_id": ATTEMPT_ID, "pid": os.getpid()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield str(lease)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def execute(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan["offline_environment_missing_or_invalid"]:
        raise RuntimeError(
            f"C4 strict offline environment is incomplete: "
            f"{plan['offline_environment_missing_or_invalid']}"
        )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"C4 expected one visible GPU, got {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    started = time.monotonic()
    with _gpu_lease(plan) as lease:
        native_batch, codec_batch, batch_audit = _build_batches(plan)
        native_phase, native_loss, native_action = _run_model_phase(
            codec_enabled=False,
            batch=native_batch,
            device="cuda:0",
        )
        codec_phase, codec_loss, codec_action = _run_model_phase(
            codec_enabled=True,
            batch=codec_batch,
            device="cuda:0",
        )
        loss_equal = torch.equal(native_loss, codec_loss)
        action_equal = torch.equal(native_action, codec_action)
        if not loss_equal or not action_equal:
            raise RuntimeError(
                f"C4 exact parity failed: loss_equal={loss_equal}, action_equal={action_equal}, "
                f"max_action_diff={float((native_action - codec_action).abs().max())}"
            )
        if codec_phase["codec_calls"] != {"encode": 2, "condition": 5}:
            raise RuntimeError(f"C4 Codec execution count drifted: {codec_phase['codec_calls']}")
        action_path = OUTPUT_ROOT / ACTION_NAME
        with action_path.open("xb") as stream:
            np.save(stream, native_action.numpy(), allow_pickle=False)
        result = {
            **dict(plan),
            "status": "READY",
            "technical_valid": True,
            "native_phase": native_phase,
            "codec_phase": codec_phase,
            "batch_audit": batch_audit,
            "parity": {
                "loss_torch_equal": loss_equal,
                "action_torch_equal": action_equal,
                "action_max_abs_diff": 0.0,
                "action_artifact": _file_identity(action_path),
            },
            "gpu_runtime": {
                "logical_device": 0,
                "name": torch.cuda.get_device_name(0),
                "lease": lease,
            },
            "wall_seconds": time.monotonic() - started,
            "checkpoint_unchanged": _checkpoint_identity() == plan["checkpoint"],
            "fixture_unchanged": _sha256_path(FIXTURE_ROOT / "fixture_manifest.json")
            == FIXTURE_MANIFEST_SHA256,
        }
    return result


def _verify_outer_timeout() -> None:
    if os.environ.get("C4_OUTER_TIMEOUT_SECONDS") != str(MAX_WALL_SECONDS):
        raise RuntimeError("C4 execute requires the frozen outer 1800-second timeout")
    parent = Path(f"/proc/{os.getppid()}/exe").resolve(strict=True)
    if parent != Path("/usr/bin/timeout").resolve(strict=True):
        raise RuntimeError("C4 execute parent must be /usr/bin/timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_plan()
    if args.preflight:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    _verify_outer_timeout()
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(mode=0o755, exist_ok=False)
    _write_json_new(OUTPUT_ROOT / "PLAN.json", plan)
    try:
        result = execute(plan)
        _write_json_new(OUTPUT_ROOT / SUMMARY_NAME, result)
        terminal = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": ATTEMPT_ID,
            "status": "READY",
            "summary_sha256": _sha256_path(OUTPUT_ROOT / SUMMARY_NAME),
            "finished_utc": _utc_now(),
        }
        _write_json_new(OUTPUT_ROOT / TERMINAL_NAME, terminal)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": ATTEMPT_ID,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "finished_utc": _utc_now(),
        }
        if not (OUTPUT_ROOT / TERMINAL_NAME).exists():
            _write_json_new(OUTPUT_ROOT / TERMINAL_NAME, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
