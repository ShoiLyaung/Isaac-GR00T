from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "examples/G1_ID10_H40_NATIVE30/strict_load_finite_action.py"
SPEC = importlib.util.spec_from_file_location("g1_h40_strict_load_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _contract() -> dict:
    return {
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
        "action_delta_indices": list(range(40)),
        "video_delta_indices": [0],
        "state_delta_indices": [0],
        "action_group_order": MODULE.ACTION_GROUPS,
        "temporal_option": MODULE.TEMPORAL_OPTION,
        "true_30hz_hardware_clock_proven": False,
    }


def test_strict_loading_requires_all_four_categories_empty():
    clean = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    summary = MODULE.strict_loading_summary(clean)
    assert summary["all_zero"] is True
    assert all(
        summary[key]["count"] == 0 for key in ("missing", "unexpected", "mismatched", "error")
    )
    for key in clean:
        dirty = copy.deepcopy(clean)
        dirty[key] = [f"sentinel-{key}"]
        with pytest.raises(RuntimeError, match="Strict checkpoint loading mismatch"):
            MODULE.strict_loading_summary(dirty)


def test_policy_strict_loading_rejects_every_mismatch_category(tmp_path):
    from gr00t.policy.gr00t_policy import _validate_checkpoint_loading_info

    clean = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    _validate_checkpoint_loading_info(clean, tmp_path)
    for key in clean:
        dirty = copy.deepcopy(clean)
        dirty[key] = [f"sentinel-{key}"]
        with pytest.raises(RuntimeError, match="Strict checkpoint reload mismatch"):
            _validate_checkpoint_loading_info(dirty, tmp_path)


def test_exact_h40_runtime_contract_accepts_only_frozen_values():
    contract = _contract()
    MODULE.assert_exact_runtime_contract(contract)
    mutations = {
        "embodiment_id": 25,
        "model_action_horizon": 50,
        "processor_action_horizon": 50,
        "model_max_action_dim": 131,
        "alternate_vl_dit": "DiT",
        "alternate_vl_dit_blocks": 16,
        "num_inference_timesteps": 5,
        "tune_llm": True,
        "backbone_trainable_parameters": 1,
        "true_30hz_hardware_clock_proven": True,
    }
    for key, value in mutations.items():
        changed = copy.deepcopy(contract)
        changed[key] = value
        with pytest.raises(RuntimeError, match="Contract mismatch"):
            MODULE.assert_exact_runtime_contract(changed)


def test_exact_h40_runtime_contract_rejects_offsets_and_group_order():
    for key, value in (
        ("action_delta_indices", list(range(39))),
        ("video_delta_indices", [-1, 0]),
        ("state_delta_indices", [0, 1]),
        ("action_group_order", list(reversed(MODULE.ACTION_GROUPS))),
    ):
        changed = copy.deepcopy(_contract())
        changed[key] = value
        with pytest.raises(RuntimeError):
            MODULE.assert_exact_runtime_contract(changed)


def test_artifact_identity_uses_index_declared_shards(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    processor = tmp_path / "processor"
    checkpoint.mkdir()
    processor.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    shards = ["model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors"]
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1], "c": shards[0]}}),
        encoding="utf-8",
    )
    for name in shards:
        (checkpoint / name).write_bytes(name.encode())
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (processor / name).write_text("{}\n", encoding="utf-8")

    identity = MODULE._artifact_identity(checkpoint, processor)
    assert list(identity["checkpoint"]) == [
        "config.json",
        "model.safetensors.index.json",
        *shards,
    ]
    assert (
        identity["checkpoint"][shards[0]]["sha256"]
        == hashlib.sha256(shards[0].encode()).hexdigest()
    )


def test_artifact_identity_rejects_missing_or_unsafe_shards(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    processor = tmp_path / "processor"
    checkpoint.mkdir()
    processor.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (processor / name).write_text("{}\n", encoding="utf-8")

    index = checkpoint / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        MODULE._artifact_identity(checkpoint, processor)

    index.write_text(json.dumps({"weight_map": {"a": "../escape.safetensors"}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid checkpoint shard inventory"):
        MODULE._artifact_identity(checkpoint, processor)
