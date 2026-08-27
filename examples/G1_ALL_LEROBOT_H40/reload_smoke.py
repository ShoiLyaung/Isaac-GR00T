#!/usr/bin/env python3
"""Strictly reload one joint H40 checkpoint and request one finite action.

The default ``plan`` action is CPU-only: it resolves and validates local input
references, writes a plan summary, and never imports torch or the GR00T runtime.
``execute`` requires an explicitly isolated single visible GPU and offline-only
environment flags.  It loads the checkpoint through ``Gr00tPolicy``'s strict
checkpoint path and calls the public ``policy.get_action`` method exactly once.

This is a connectivity smoke, not an evaluator and not evidence of task quality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "g1-all-lerobot-h40-reload-smoke-v1"
SUMMARY_NAME = "RELOAD_SMOKE_SUMMARY.json"
ACTIVE_ACTION_NAME = "ACTIVE_ACTION_26D.npy"
EMBODIMENT_TAG = "new_embodiment"
HORIZON = 40
ACTION_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
    "waist": 3,
    "base_height_command": 1,
    "navigate_command": 3,
}
ACTIVE_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "GROOT_HF_LOCAL_FIRST": "1",
    "GROOT_PATCH_MISTRAL": "1",
    "NO_ALBUMENTATIONS_UPDATE": "1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_real_directory(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _resolve_real_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _checkpoint_inventory(checkpoint: Path) -> dict[str, Any]:
    required = (
        "config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint root is incomplete: {missing}")

    index = load_json_object(checkpoint / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index has no non-empty weight_map")
    shards = sorted(set(weight_map.values()))
    if any(
        not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors")
        for name in shards
    ):
        raise ValueError(f"Unsafe checkpoint shard inventory: {shards!r}")
    absent_shards = [name for name in shards if not (checkpoint / name).is_file()]
    if absent_shards:
        raise FileNotFoundError(f"Checkpoint shards are missing: {absent_shards}")

    config = load_json_object(checkpoint / "config.json")
    processor = load_json_object(checkpoint / "processor_config.json")
    embodiment_ids = load_json_object(checkpoint / "embodiment_id.json")
    processor_kwargs = processor.get("processor_kwargs")
    if config.get("architectures") != ["Gr00tN1d7"]:
        raise ValueError("Checkpoint architecture is not exactly Gr00tN1d7")
    if config.get("action_horizon") != HORIZON:
        raise ValueError(f"Checkpoint action_horizon is not {HORIZON}")
    if (
        not isinstance(processor_kwargs, dict)
        or processor_kwargs.get("max_action_horizon") != HORIZON
    ):
        raise ValueError(f"Processor max_action_horizon is not {HORIZON}")
    if embodiment_ids.get(EMBODIMENT_TAG) != 10:
        raise ValueError("Checkpoint new_embodiment projector id is not 10")

    small_files = {
        name: {
            "bytes": (checkpoint / name).stat().st_size,
            "sha256": sha256_file(checkpoint / name),
        }
        for name in required
    }
    return {
        "architecture": "Gr00tN1d7",
        "action_horizon": HORIZON,
        "new_embodiment_id": 10,
        "tune_visual": config.get("tune_visual"),
        "weight_map_entries": len(weight_map),
        "shards": [{"name": name, "bytes": (checkpoint / name).stat().st_size} for name in shards],
        "small_files": small_files,
    }


def _input_contract(loader_view: Path, adapter_config: Path) -> dict[str, Any]:
    required_view = (
        "meta/info.json",
        "meta/modality.json",
        "meta/tasks.jsonl",
        "meta/g1_all_lerobot_h40_view_manifest.json",
        "data",
        "videos",
    )
    missing = [name for name in required_view if not (loader_view / name).exists()]
    if missing:
        raise FileNotFoundError(f"Loader view is incomplete: {missing}")
    adapter = load_json_object(adapter_config)
    manifest_path = loader_view / "meta/g1_all_lerobot_h40_view_manifest.json"
    manifest = load_json_object(manifest_path)
    exact = {
        "adapter_kind": "g1_all_lerobot_h40_v1",
        "embodiment_tag": EMBODIMENT_TAG,
        "embodiment_id": 10,
        "horizon": HORIZON,
        "active_action_dims": 26,
    }
    for key, expected in exact.items():
        if adapter.get(key) != expected:
            raise ValueError(f"Adapter {key}={adapter.get(key)!r}, expected {expected!r}")
    for key in (
        "source_id",
        "canonical_root",
        "canonical_info_sha256",
        "canonical_modality_sha256",
        "raw_task_labels",
        "training_language",
    ):
        if manifest.get(key) != adapter.get(key):
            raise ValueError(f"Loader-view/adapter mismatch: {key}")
    if manifest.get("camera_mapping", {}).get("model_key") != "ego_view":
        raise ValueError("Loader view does not map its common camera to ego_view")
    return {
        "source_id": adapter["source_id"],
        "canonical_root": adapter["canonical_root"],
        "training_language": adapter["training_language"],
        "view_manifest_sha256": sha256_file(manifest_path),
        "adapter_config_sha256": sha256_file(adapter_config),
        "payload_links": {
            name: {
                "is_symlink": (loader_view / name).is_symlink(),
                "resolved": str((loader_view / name).resolve(strict=True)),
            }
            for name in ("data", "videos")
        },
    }


def resolve_plan(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_real_directory(args.checkpoint, "checkpoint")
    loader_view = _resolve_real_directory(args.loader_view, "loader view")
    adapter_config = _resolve_real_file(args.adapter_config, "adapter config")
    output_dir = _resolve_real_directory(args.output_dir, "output directory")
    for input_root in (checkpoint, loader_view):
        if output_dir == input_root or output_dir.is_relative_to(input_root):
            raise ValueError(f"Output directory must be external to input root: {input_root}")
    for filename in (SUMMARY_NAME, ACTIVE_ACTION_NAME):
        if (output_dir / filename).exists():
            raise FileExistsError(output_dir / filename)
    if args.device != "cuda:0":
        raise ValueError("This smoke accepts exactly one logical device: cuda:0")
    if args.episode < 0 or args.anchor < 0:
        raise ValueError("episode and anchor must be non-negative")

    env = {key: os.environ.get(key) for key in (*OFFLINE_ENV, "HF_HOME", "CUDA_VISIBLE_DEVICES")}
    missing_offline = [key for key, expected in OFFLINE_ENV.items() if env[key] != expected]
    selector = env["CUDA_VISIBLE_DEVICES"]
    selector_is_single = bool(selector) and "," not in str(selector)
    return {
        "schema_version": SCHEMA_VERSION,
        "action": args.action,
        "claim_scope": "STRICT_CHECKPOINT_RELOAD_AND_ONE_POLICY_ACTION_CONNECTIVITY_ONLY",
        "checkpoint": str(checkpoint),
        "checkpoint_inventory": _checkpoint_inventory(checkpoint),
        "loader_view": str(loader_view),
        "adapter_config": str(adapter_config),
        "input_contract": _input_contract(loader_view, adapter_config),
        "output_dir": str(output_dir),
        "summary": str(output_dir / SUMMARY_NAME),
        "active_action": str(output_dir / ACTIVE_ACTION_NAME),
        "device": args.device,
        "seed": args.seed,
        "episode": args.episode,
        "anchor": args.anchor,
        "expected_action_groups": ACTION_DIMS,
        "offline_environment": env,
        "offline_environment_missing_or_invalid": missing_offline,
        "single_visible_gpu_selector": selector_is_single,
        "gpu_or_model_touched": False,
    }


def require_execute_environment(plan: dict[str, Any]) -> None:
    missing = plan["offline_environment_missing_or_invalid"]
    if missing:
        raise RuntimeError(f"Strict offline environment is incomplete: {missing}")
    if not plan["single_visible_gpu_selector"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one physical GPU")
    hf_home = plan["offline_environment"].get("HF_HOME")
    if not hf_home or not Path(hf_home).is_absolute() or not Path(hf_home).is_dir():
        raise RuntimeError("HF_HOME must be one existing absolute local cache directory")


def build_observation(step: Any) -> dict[str, Any]:
    return {
        "video": {
            key: np.asarray(value, dtype=np.uint8)[None, ...] for key, value in step.images.items()
        },
        "state": {
            key: np.asarray(value, dtype=np.float32)[None, ...]
            for key, value in step.states.items()
        },
        "language": {"annotation.human.task_description": [[str(step.text)]]},
    }


def call_policy_once(policy: Any, step: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the public policy boundary exactly once; kept small for CPU tests."""

    return policy.get_action(build_observation(step))


def summarize_actions(action: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if set(action) != set(ACTION_DIMS):
        raise ValueError(
            f"Decoded action groups differ: actual={sorted(action)}, expected={sorted(ACTION_DIMS)}"
        )
    arrays: dict[str, np.ndarray] = {}
    groups: dict[str, Any] = {}
    for key, dim in ACTION_DIMS.items():
        value = np.asarray(action[key])
        expected_shape = (1, HORIZON, dim)
        if value.dtype != np.float32:
            raise ValueError(f"{key} dtype is {value.dtype}, expected float32")
        if value.shape != expected_shape:
            raise ValueError(f"{key} shape is {value.shape}, expected {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains a non-finite value")
        arrays[key] = value
        groups[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": True,
            "min": float(value.min()),
            "max": float(value.max()),
        }

    # Deliberately concatenate the raw decoded groups.  Do not clip, clamp, or
    # otherwise mutate the model result in this reload connectivity probe.
    active = np.concatenate([arrays[key] for key in ACTIVE_GROUPS], axis=-1)
    if active.shape != (1, HORIZON, 26) or not np.isfinite(active).all():
        raise AssertionError("Raw active action is not finite [1,40,26]")
    hand_values = np.concatenate([arrays["left_hand"], arrays["right_hand"]], axis=-1)
    return active, {
        "groups": groups,
        "active_26d": {
            "shape": list(active.shape),
            "dtype": str(active.dtype),
            "finite": True,
            "min": float(active.min()),
            "max": float(active.max()),
            "construction": "raw left_arm+right_arm+left_hand+right_hand concatenation",
            "clamp_applied": False,
        },
        "diagnostics": {
            "hand_unit_interval_oob_elements": int(
                np.count_nonzero((hand_values < 0.0) | (hand_values > 1.0))
            ),
            "oob_policy": "record_only_no_clamp",
        },
    }


def execute(plan: dict[str, Any]) -> dict[str, Any]:
    require_execute_environment(plan)

    # Runtime imports are intentionally below the plan/execute boundary.
    from gr00t.data.dataset.g1_all_lerobot_h40_adapter import G1AllLerobotH40EpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA device, observed {torch.cuda.device_count()}"
        )
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.cuda.manual_seed_all(plan["seed"])
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    started = time.monotonic()
    policy = Gr00tPolicy(
        embodiment_tag=EMBODIMENT_TAG,
        model_path=plan["checkpoint"],
        device=plan["device"],
        strict=True,
        strict_checkpoint_loading=True,
    )
    model_load_seconds = time.monotonic() - started
    modality = policy.get_modality_config()
    if list(modality["action"].modality_keys) != list(ACTION_DIMS):
        raise RuntimeError("Checkpoint action modality order does not match the joint H40 contract")
    if list(modality["action"].delta_indices) != list(range(HORIZON)):
        raise RuntimeError("Checkpoint action offsets are not native H40 [0..39]")

    loader = G1AllLerobotH40EpisodeLoader(
        plan["loader_view"],
        modality,
        plan["adapter_config"],
        require_runtime_verified_fixed_state=True,
    )
    episode = loader[plan["episode"]]
    step = extract_step_data(
        episode,
        plan["anchor"],
        modality,
        EmbodimentTag(EMBODIMENT_TAG),
        allow_padding=False,
    )
    action_started = time.monotonic()
    action, info = call_policy_once(policy, step)
    action_seconds = time.monotonic() - action_started
    active, action_summary = summarize_actions(action)

    active_path = Path(plan["active_action"])
    with active_path.open("xb") as stream:
        np.save(stream, active, allow_pickle=False)
    action_summary["active_26d"]["path"] = str(active_path)
    action_summary["active_26d"]["sha256"] = sha256_file(active_path)
    result = {
        **plan,
        "status": "PASS",
        "technical_valid": True,
        "gpu_or_model_touched": True,
        "strict_checkpoint_loading": "PASS_NO_MISSING_UNEXPECTED_MISMATCHED_OR_ERROR_KEYS",
        "policy_get_action_calls": 1,
        "input": {
            "language": step.text,
            "video_shapes": {
                key: list(np.asarray(value).shape) for key, value in step.images.items()
            },
            "state_shapes": {
                key: list(np.asarray(value).shape) for key, value in step.states.items()
            },
        },
        "action_output": action_summary,
        "policy_info_keys": sorted(info),
        "gpu": {
            "visible_device_count": torch.cuda.device_count(),
            "logical_device": 0,
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        },
        "timing_seconds": {
            "model_strict_load": model_load_seconds,
            "one_policy_get_action": action_seconds,
            "total": time.monotonic() - started,
        },
    }
    del loader, policy
    torch.cuda.empty_cache()
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("plan", "execute"))
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--loader-view", required=True, type=Path)
    result.add_argument("--adapter-config", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--episode", type=int, default=0)
    result.add_argument("--anchor", type=int, default=0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = args.output_dir.resolve(strict=True)
    summary_path = output_dir / SUMMARY_NAME
    plan: dict[str, Any] | None = None
    try:
        plan = resolve_plan(args)
        if args.action == "plan":
            result = {
                **plan,
                "status": "DRY_RUN_READY",
                "technical_valid": None,
                "policy_get_action_calls": 0,
                "finished_utc": utc_now(),
            }
        else:
            result = {**execute(plan), "finished_utc": utc_now()}
        write_json_new(summary_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "technical_valid": False,
            "action": args.action,
            "claim_scope": "STRICT_CHECKPOINT_RELOAD_AND_ONE_POLICY_ACTION_CONNECTIVITY_ONLY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "policy_get_action_calls": 0 if plan is None or args.action == "plan" else None,
            "finished_utc": utc_now(),
        }
        if plan is not None:
            failure["resolved_plan"] = plan
        if not summary_path.exists():
            write_json_new(summary_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
