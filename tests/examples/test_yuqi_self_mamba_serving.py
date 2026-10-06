"""CPU validation of the new serving condition mapping and checkpoint gate."""

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/G1_DIT_MAMBA_YUQI"))
from self_mamba_roundtrip import tiny_config
from serve_checkpoint import selftest, training_identity


def test_serving_selftest_includes_hybrid_and_reset_mapping():
    selftest()


@pytest.mark.parametrize("condition", ["dit_self_mamba2", "dit_reinit_self_attention"])
def test_serving_requires_complete_strictly_reloaded_checkpoint(tmp_path, condition):
    checkpoint = tmp_path / "checkpoint-5000"
    result = tmp_path / "TRAIN_RESULT.json"
    payload = {
        "condition": condition,
        "checkpoint": str(checkpoint),
        "status": "completed",
        "steps": 5000,
        "strict_reload_action_max_abs": 0.0,
    }
    result.write_text(json.dumps(payload))
    assert training_identity(checkpoint, result)[0] == condition
    for key, bad in (
        ("steps", 4999),
        ("strict_reload_action_max_abs", 1e-5),
        ("condition", "unknown"),
        ("status", "running"),
    ):
        result.write_text(json.dumps({**payload, key: bad}))
        with pytest.raises(ValueError):
            training_identity(checkpoint, result)


def test_official_roundtrip_shape_contract():
    cfg = tiny_config()
    assert cfg.diffusion_model_cfg["num_layers"] == 32
    assert cfg.hidden_size == cfg.input_embedding_dim == cfg.backbone_embedding_dim == 32
    assert cfg.action_horizon == 40 and cfg.max_action_dim == cfg.max_state_dim == 13
    assert cfg.num_inference_timesteps == 4
    assert cfg.action_mamba2_cfg["d_state"] == cfg.action_mamba2_cfg["headdim"] == 16
    assert cfg.action_mamba2_cfg["use_mem_eff_path"] is False
