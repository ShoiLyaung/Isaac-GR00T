# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter-aware open-loop entry for the G1 ID10/H50 baseline candidate.

This module is intentionally separate from the generic upstream evaluator.  It
uses the exact candidate loader and preserves the synthetic 30 Hz -> nominal 50 Hz
ZOH provenance.  Metrics produced here are connectivity diagnostics, not task
success or paper-quality claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.g1_id10_h50_adapter import (
    ACTION_GROUPS,
    ACTIVE_ACTION_DIM,
    BASELINE_ID,
    EMBODIMENT_TAG,
    G1Id10H50EpisodeLoader,
    HORIZON,
    LANGUAGE,
    TEMPORAL_OPTION,
    h50_zoh_offsets,
    resolve_decoded_action_26d,
)
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy


RESULT_SCHEMA = "g1-id10-h50-open-loop-result-v1"
ACTIVE_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_modality_config(path: str | Path) -> dict[str, Any]:
    """Load the exact project registration and reject semantic drift."""

    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("g1_id10_h50_eval_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modality = MODALITY_CONFIGS[EMBODIMENT_TAG]
    if modality["action"].delta_indices != h50_zoh_offsets():
        raise ValueError("Open-loop action indices differ from frozen synthetic ZOH")
    if modality["action"].modality_keys != ACTION_GROUPS:
        raise ValueError("Open-loop action-group order differs from frozen ID10 contract")
    if modality["language"].modality_keys != ["annotation.human.task_description"]:
        raise ValueError("Open-loop language key differs from frozen contract")
    return modality


def build_open_loop_loader(
    *,
    view_root: str | Path,
    adapter_config_path: str | Path,
    modality_config_path: str | Path,
    require_runtime_verified_fixed_state: bool = True,
    video_backend: str = "torchcodec",
) -> G1Id10H50EpisodeLoader:
    """Construct only the accepted adapter loader, never the generic loader."""

    modality = load_frozen_modality_config(modality_config_path)
    return G1Id10H50EpisodeLoader(
        dataset_path=Path(view_root),
        modality_configs=modality,
        adapter_config_path=Path(adapter_config_path),
        video_backend=video_backend,
        require_runtime_verified_fixed_state=require_runtime_verified_fixed_state,
    )


def _policy_observation(step: Any) -> dict[str, Any]:
    return {
        "video": {key: np.asarray(value)[None, ...] for key, value in step.images.items()},
        "state": {
            key: np.asarray(value, dtype=np.float32)[None, ...]
            for key, value in step.states.items()
        },
        "language": {"annotation.human.task_description": [[step.text]]},
    }


def _ground_truth_26d(step: Any) -> np.ndarray:
    groups = []
    for key in ACTIVE_GROUPS:
        value = np.asarray(step.actions[key], dtype=np.float32)
        expected_dim = 7 if key.endswith("arm") else 6
        if value.shape != (HORIZON, expected_dim) or not np.isfinite(value).all():
            raise ValueError(f"Ground truth {key} must be finite [50,{expected_dim}]")
        groups.append(value)
    result = np.concatenate(groups, axis=-1)
    if result.shape != (HORIZON, ACTIVE_ACTION_DIM):
        raise AssertionError(result.shape)
    return result


def _unbatch_decoded(action: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for key, value in action.items():
        array = np.asarray(value)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError(f"Decoded {key} requires batch size one, got {array.shape}")
            array = array[0]
        result[key] = array
    return result


def validate_loaded_policy_contract(policy: Any) -> None:
    """Reject a fresh process that did not reload the exact H50/ID10 contract."""

    if int(policy.model.config.action_horizon) != HORIZON:
        raise RuntimeError("Reloaded model action horizon is not H50")
    if int(policy.processor.max_action_horizon) != HORIZON:
        raise RuntimeError("Reloaded processor action horizon is not H50")
    if int(policy.processor.embodiment_id_mapping[EMBODIMENT_TAG]) != 10:
        raise RuntimeError("Reloaded processor embodiment mapping is not ID10")
    modality = policy.modality_configs
    if modality["action"].delta_indices != h50_zoh_offsets():
        raise RuntimeError("Reloaded processor lost frozen synthetic-ZOH indices")
    if modality["action"].modality_keys != ACTION_GROUPS:
        raise RuntimeError("Reloaded processor action-group order changed")


def evaluate_g1_id10_h50_anchor(
    policy: BasePolicy,
    loader: G1Id10H50EpisodeLoader,
    *,
    episode_index: int = 0,
    anchor_index: int = 0,
) -> dict[str, Any]:
    """Run one H50 policy call against one accepted dataset anchor."""

    modality = loader.modality_configs
    episode = loader[episode_index]
    step = extract_step_data(
        episode,
        anchor_index,
        modality,
        EmbodimentTag(EMBODIMENT_TAG),
        allow_padding=False,
    )
    if step.text != LANGUAGE:
        raise ValueError("Open-loop language literal mismatch")
    action, policy_info = policy.get_action(_policy_observation(step))
    predicted, clamp_count = resolve_decoded_action_26d(_unbatch_decoded(action))
    target = _ground_truth_26d(step)
    error = predicted.astype(np.float64) - target.astype(np.float64)
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "OPEN_LOOP_CONNECTIVITY_PASS_NO_QUALITY_CLAIM",
        "metric_scope": "CONNECTIVITY_DIAGNOSTIC_NOT_TASK_SUCCESS_OR_PAPER_RESULT",
        "baseline_id": BASELINE_ID,
        "embodiment_id": 10,
        "action_horizon": HORIZON,
        "active_action_dim": ACTIVE_ACTION_DIM,
        "temporal_option": TEMPORAL_OPTION,
        "source_index_formula": "n+floor(3*k/5)",
        "true_50hz_capture": False,
        "episode_index": episode_index,
        "anchor_index": anchor_index,
        "source_offsets": h50_zoh_offsets(),
        "prediction_shape": list(predicted.shape),
        "prediction_dtype": str(predicted.dtype),
        "ground_truth_shape": list(target.shape),
        "clamp_count": clamp_count,
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "predicted_action_26d": predicted.tolist(),
        "ground_truth_action_26d": target.tolist(),
        "policy_info_keys": sorted(policy_info) if isinstance(policy_info, dict) else [],
    }


@dataclass
class OpenLoopConfig:
    model_path: str
    view_root: str
    adapter_config_path: str
    modality_config_path: str = "examples/G1_ID10_H50/g1_id10_h50_config.py"
    output: str = "/tmp/g1-id10-h50-open-loop-result.json"
    episode_index: int = 0
    anchor_index: int = 0
    device: str = "cuda"


def main(config: OpenLoopConfig) -> None:
    output = Path(config.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    loader = build_open_loop_loader(
        view_root=config.view_root,
        adapter_config_path=config.adapter_config_path,
        modality_config_path=config.modality_config_path,
        require_runtime_verified_fixed_state=True,
    )
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        model_path=config.model_path,
        device=config.device,
        strict_checkpoint_loading=True,
    )
    validate_loaded_policy_contract(policy)
    result = evaluate_g1_id10_h50_anchor(
        policy,
        loader,
        episode_index=config.episode_index,
        anchor_index=config.anchor_index,
    )
    result["input_identity"] = {
        "model_path": str(Path(config.model_path).resolve()),
        "view_manifest_sha256": _sha256(
            Path(config.view_root).resolve() / "meta/g1_id10_h50_view_manifest.json"
        ),
        "adapter_config_sha256": _sha256(Path(config.adapter_config_path).resolve()),
        "modality_config_sha256": _sha256(Path(config.modality_config_path).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main(tyro.cli(OpenLoopConfig))
