#!/usr/bin/env python3
"""Strictly load the ID10/H40 model and emit one finite active action chunk.

This is a bounded model-connectivity probe.  It does not train, save weights,
start a policy server, run a simulator, or establish task quality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel

import gr00t.model  # noqa: F401 - register the pinned model class
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTION_GROUPS,
    ACTIVE_ACTION_DIM,
    BASELINE_ID,
    EMBODIMENT_TAG,
    G1Id10H40Native30EpisodeLoader,
    HORIZON,
    LANGUAGE,
    PADDED_DIM,
    TEMPORAL_OPTION,
    h40_native30_offsets,
    resolve_decoded_action_26d,
)
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.policy.gr00t_policy import _rec_to_dtype


SCHEMA_VERSION = "g1-id10-h40-native30-strict-load-finite-action-result-v1"
EXPECTED_SEMANTIC_MD_SHA256 = "16bb85aa8e5a8d84a59cfd851ba21308847f225270ec050893dac03bb741834c"
EXPECTED_SEMANTIC_JSON_SHA256 = "d3cb9304f75568c5dca73acc8aaaa596cdcd7ae436a1f97ef9f16d9f7b6ef930"
EXPECTED_FIXED_STATE_MD_SHA256 = "11bbddf4580ae9af31ad1acfcfea5e77c52ac5c107d1db200f60aeee7702375a"
EXPECTED_FIXED_STATE_JSON_SHA256 = "df7d61428ad0090801004a0ffb7d56e6a099acfbc76363d9b5395b9a2c686c8c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_modality(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("g1_id10_h40_strict_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modality = MODALITY_CONFIGS[EMBODIMENT_TAG]
    if modality["action"].delta_indices != h40_native30_offsets():
        raise RuntimeError("Frozen H40 native-row offsets changed")
    if modality["action"].modality_keys != ACTION_GROUPS:
        raise RuntimeError("Frozen ID10 action-group order changed")
    if modality["video"].delta_indices != [0] or modality["state"].delta_indices != [0]:
        raise RuntimeError("Frozen current-frame video/state contract changed")
    return modality


def strict_loading_summary(loading_info: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "missing": "missing_keys",
        "unexpected": "unexpected_keys",
        "mismatched": "mismatched_keys",
        "error": "error_msgs",
    }
    result: dict[str, Any] = {}
    failures = []
    for label, key in mapping.items():
        values = list(loading_info.get(key, []) or [])
        result[label] = {"count": len(values), "values": values}
        if values:
            failures.append(f"{label}={values}")
    result["all_zero"] = not failures
    if failures:
        raise RuntimeError("Strict checkpoint loading mismatch: " + "; ".join(failures))
    return result


def assert_exact_runtime_contract(contract: dict[str, Any]) -> None:
    exact = {
        "embodiment_id": 10,
        "model_action_horizon": 40,
        "processor_action_horizon": 40,
        "model_max_state_dim": 132,
        "model_max_action_dim": 132,
        "processor_max_state_dim": 132,
        "processor_max_action_dim": 132,
        "action_head": "Gr00tN1d7ActionHead",
        "alternate_vl_dit": "AlternateVLDiT",
        "alternate_vl_dit_blocks": 32,
        "num_inference_timesteps": 4,
        "tune_llm": False,
        "tune_visual": False,
        "tune_top_llm_layers": 0,
        "backbone_trainable_parameters": 0,
        "temporal_option": TEMPORAL_OPTION,
        "true_30hz_hardware_clock_proven": False,
    }
    for key, expected in exact.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Contract mismatch {key}: {contract.get(key)!r} != {expected!r}")
    if contract.get("action_delta_indices") != h40_native30_offsets():
        raise RuntimeError("Processor H40 action offsets mismatch")
    if contract.get("video_delta_indices") != [0] or contract.get("state_delta_indices") != [0]:
        raise RuntimeError("Processor video/state offsets mismatch")
    if contract.get("action_group_order") != ACTION_GROUPS:
        raise RuntimeError("Processor action-group order mismatch")


def _parameter_summary(module: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
    }


def _artifact_identity(checkpoint: Path, processor_dir: Path) -> dict[str, Any]:
    index_name = "model.safetensors.index.json"
    with (checkpoint / index_name).open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Checkpoint safetensors index has no non-empty weight_map")
    shard_names = sorted(set(weight_map.values()))
    if not shard_names or any(
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".safetensors")
        for name in shard_names
    ):
        raise RuntimeError(f"Invalid checkpoint shard inventory: {shard_names!r}")
    checkpoint_files = ["config.json", index_name, *shard_names]
    missing = [name for name in checkpoint_files if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint artifact inventory is incomplete: {missing}")
    processor_files = ["processor_config.json", "statistics.json", "embodiment_id.json"]
    return {
        "checkpoint": {
            name: {"bytes": (checkpoint / name).stat().st_size, "sha256": _sha256(checkpoint / name)}
            for name in checkpoint_files
        },
        "processor": {
            name: {"bytes": (processor_dir / name).stat().st_size, "sha256": _sha256(processor_dir / name)}
            for name in processor_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--semantic-contract-md", type=Path, required=True)
    parser.add_argument("--semantic-contract-json", type=Path, required=True)
    parser.add_argument("--fixed-state-contract-md", type=Path, required=True)
    parser.add_argument("--fixed-state-contract-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--anchor", type=int, default=0)
    args = parser.parse_args()

    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Launcher-created output directory missing: {output_dir}")
    for name in (
        "PRELAUNCH.json",
        "RESULT.json",
        "ACTION_CHUNK_ACTIVE_26D.npy",
        "ACTION_CHUNK_PADDED_132.npy",
    ):
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)

    worktree = args.worktree.resolve()
    checkpoint = args.checkpoint.resolve()
    processor_dir = args.processor_dir.resolve()
    view_root = args.view_root.resolve()
    adapter_config = args.adapter_config.resolve()
    modality_config = args.modality_config.resolve()
    semantic_md = args.semantic_contract_md.resolve()
    semantic_json = args.semantic_contract_json.resolve()
    fixed_md = args.fixed_state_contract_md.resolve()
    fixed_json = args.fixed_state_contract_json.resolve()

    head = subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != args.expected_commit:
        raise RuntimeError(f"Worktree HEAD mismatch: {head} != {args.expected_commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(worktree), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError(f"Strict-load worktree is dirty:\n{dirty}")
    for path, expected in (
        (semantic_md, EXPECTED_SEMANTIC_MD_SHA256),
        (semantic_json, EXPECTED_SEMANTIC_JSON_SHA256),
        (fixed_md, EXPECTED_FIXED_STATE_MD_SHA256),
        (fixed_json, EXPECTED_FIXED_STATE_JSON_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"Frozen contract hash mismatch: {path}")
    interfaces = [name for _, name in socket.if_nameindex()]
    if interfaces != ["lo"]:
        raise RuntimeError(f"Expected loopback-only network namespace: {interfaces}")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Offline environment flags are required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(False)

    modality = _load_modality(modality_config)
    loader = G1Id10H40Native30EpisodeLoader(
        view_root,
        modality,
        adapter_config,
        require_runtime_verified_fixed_state=True,
    )
    episode = loader[args.episode]
    step = extract_step_data(
        episode,
        args.anchor,
        modality,
        EmbodimentTag(EMBODIMENT_TAG),
        allow_padding=False,
    )
    if step.text != LANGUAGE:
        raise RuntimeError("Frozen task-language literal mismatch")

    prelaunch = {
        "schema_version": "g1-id10-h40-native30-strict-load-prelaunch-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_id": BASELINE_ID,
        "source_commit": head,
        "checkpoint": str(checkpoint),
        "processor_dir": str(processor_dir),
        "view_root": str(view_root),
        "adapter_config": str(adapter_config),
        "modality_config": str(modality_config),
        "semantic_contract_md_sha256": _sha256(semantic_md),
        "semantic_contract_json_sha256": _sha256(semantic_json),
        "fixed_state_contract_md_sha256": _sha256(fixed_md),
        "fixed_state_contract_json_sha256": _sha256(fixed_json),
        "view_manifest_sha256": _sha256(
            view_root / "meta/g1_id10_h40_native30_view_manifest.json"
        ),
        "adapter_config_sha256": _sha256(adapter_config),
        "modality_config_sha256": _sha256(modality_config),
        "python": os.path.realpath(os.sys.executable),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_request": args.device,
        "seed": args.seed,
        "episode": args.episode,
        "anchor": args.anchor,
        "network_interfaces": interfaces,
        "offline": True,
    }
    _write_json_new(output_dir / "PRELAUNCH.json", prelaunch)

    loading_kwargs = {"local_files_only": True, "trust_remote_code": True}
    load_started = time.monotonic()
    model, loading_info = AutoModel.from_pretrained(
        checkpoint,
        action_horizon=HORIZON,
        tune_llm=False,
        tune_visual=False,
        tune_top_llm_layers=0,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        transformers_loading_kwargs=loading_kwargs,
        output_loading_info=True,
        **loading_kwargs,
    )
    strict_loading = strict_loading_summary(loading_info)
    load_seconds = time.monotonic() - load_started
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)

    processor = Gr00tN1d7Processor.from_pretrained(
        processor_dir,
        transformers_loading_kwargs=loading_kwargs,
    )
    processor.eval()

    backbone_params = _parameter_summary(model.backbone)
    contract = {
        "embodiment_tag": EMBODIMENT_TAG,
        "embodiment_id": int(processor.embodiment_id_mapping[EMBODIMENT_TAG]),
        "model_action_horizon": int(model.config.action_horizon),
        "processor_action_horizon": int(processor.max_action_horizon),
        "model_max_state_dim": int(model.config.max_state_dim),
        "model_max_action_dim": int(model.config.max_action_dim),
        "processor_max_state_dim": int(processor.max_state_dim),
        "processor_max_action_dim": int(processor.max_action_dim),
        "action_head": type(model.action_head).__name__,
        "alternate_vl_dit": type(model.action_head.model).__name__,
        "alternate_vl_dit_blocks": len(model.action_head.model.transformer_blocks),
        "state_encoder": type(model.action_head.state_encoder).__name__,
        "action_encoder": type(model.action_head.action_encoder).__name__,
        "action_decoder": type(model.action_head.action_decoder).__name__,
        "num_inference_timesteps": int(model.action_head.num_inference_timesteps),
        "tune_llm": bool(model.config.tune_llm),
        "tune_visual": bool(model.config.tune_visual),
        "tune_top_llm_layers": int(model.config.tune_top_llm_layers),
        "tune_projector": bool(model.config.tune_projector),
        "tune_diffusion_model": bool(model.config.tune_diffusion_model),
        "backbone_trainable_parameters": backbone_params["trainable"],
        "action_delta_indices": list(
            processor.modality_configs[EMBODIMENT_TAG]["action"].delta_indices
        ),
        "video_delta_indices": list(
            processor.modality_configs[EMBODIMENT_TAG]["video"].delta_indices
        ),
        "state_delta_indices": list(
            processor.modality_configs[EMBODIMENT_TAG]["state"].delta_indices
        ),
        "action_group_order": list(
            processor.modality_configs[EMBODIMENT_TAG]["action"].modality_keys
        ),
        "temporal_option": TEMPORAL_OPTION,
        "true_30hz_hardware_clock_proven": False,
    }
    assert_exact_runtime_contract(contract)

    vla_step = VLAStepData(
        images=step.images,
        states=step.states,
        actions={},
        text=step.text,
        embodiment=EmbodimentTag(EMBODIMENT_TAG),
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step}])
    collated = _rec_to_dtype(processor.collator([processed]), dtype=torch.bfloat16)

    inference_started = time.monotonic()
    with torch.inference_mode():
        prediction = model.get_action(**collated)
    inference_seconds = time.monotonic() - inference_started
    padded = prediction["action_pred"].float().cpu().numpy()
    if padded.shape != (1, HORIZON, PADDED_DIM) or not np.isfinite(padded).all():
        raise RuntimeError(f"Padded action is not finite [1,40,132]: {padded.shape}")

    batched_states = {key: np.asarray(value)[None, ...] for key, value in step.states.items()}
    decoded = processor.decode_action(
        padded,
        EmbodimentTag(EMBODIMENT_TAG),
        state=batched_states,
    )
    unbatched = {
        key: np.asarray(value)[0] if np.asarray(value).ndim == 3 else np.asarray(value)
        for key, value in decoded.items()
    }
    active_unbatched, clamp_count = resolve_decoded_action_26d(unbatched)
    active = np.asarray(active_unbatched, dtype=np.float32)[None, ...]
    if active.shape != (1, HORIZON, ACTIVE_ACTION_DIM) or not np.isfinite(active).all():
        raise RuntimeError(f"Decoded action is not finite [1,40,26]: {active.shape}")

    padded_path = output_dir / "ACTION_CHUNK_PADDED_132.npy"
    active_path = output_dir / "ACTION_CHUNK_ACTIVE_26D.npy"
    with padded_path.open("xb") as stream:
        np.save(stream, padded, allow_pickle=False)
    with active_path.open("xb") as stream:
        np.save(stream, active, allow_pickle=False)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "STRICT_LOAD_ZERO_MISMATCH_FINITE_H40_ACTION_PASS",
        "claim_scope": "MODEL_LOAD_AND_ONE_ACTION_CONNECTIVITY_NOT_QUALITY_OR_TASK_SUCCESS",
        "baseline_id": BASELINE_ID,
        "source_commit": head,
        "strict_mode": True,
        "strict_loading": strict_loading,
        "contract": contract,
        "native_invariants": {
            "flow_velocity_target": "SOURCE_DEFINED_ACTION_MINUS_NOISE_NOT_EXECUTED_IN_INFERENCE",
            "masked_mse": "SOURCE_DEFINED_NOT_EXECUTED_IN_INFERENCE",
            "alternate_vl_dit": "RUNTIME_TYPE_AND_32_BLOCK_COUNT_OBSERVED",
            "decoder": "RUNTIME_NATIVE_ACTION_DECODER_TYPE_OBSERVED",
            "euler_evaluations": "RUNTIME_CONFIG_4_AND_NATIVE_GET_ACTION_USED",
            "vlm": "RUNTIME_BACKBONE_TRAINABLE_PARAMETER_COUNT_ZERO",
        },
        "parameters": {
            "model": _parameter_summary(model),
            "backbone": backbone_params,
            "action_head": _parameter_summary(model.action_head),
        },
        "input": {
            "episode": args.episode,
            "anchor": args.anchor,
            "language": step.text,
            "camera_shapes": {
                key: list(np.asarray(value).shape) for key, value in step.images.items()
            },
            "state_shapes": {
                key: list(np.asarray(value).shape) for key, value in step.states.items()
            },
        },
        "padded_action": {
            "path": padded_path.name,
            "shape": list(padded.shape),
            "dtype": str(padded.dtype),
            "finite": bool(np.isfinite(padded).all()),
            "min": float(padded.min()),
            "max": float(padded.max()),
            "mean": float(padded.mean()),
            "sha256": _sha256(padded_path),
        },
        "decoded_action_26d": {
            "path": active_path.name,
            "shape": list(active.shape),
            "dtype": str(active.dtype),
            "finite": bool(np.isfinite(active).all()),
            "min": float(active.min()),
            "max": float(active.max()),
            "mean": float(active.mean()),
            "hand_clamp_count": int(clamp_count),
            "sha256": _sha256(active_path),
        },
        "timing_seconds": {
            "model_strict_load": load_seconds,
            "one_action_inference": inference_seconds,
            "total": time.monotonic() - started,
        },
        "material_identity": _artifact_identity(checkpoint, processor_dir),
        "ended_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(output_dir / "RESULT.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "strict_loading",
                    "contract",
                    "padded_action",
                    "decoded_action_26d",
                    "timing_seconds",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
