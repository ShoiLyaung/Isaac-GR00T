"""Input identity, cache boundary, and compact full-head checkpoint contracts."""

from argparse import Namespace
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

from G1_DIT_MAMBA_BASELINE import run_baseline as runner, run_train_eval as training  # noqa: E402
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
from safetensors.torch import save_file
import torch


def tiny_config(**kwargs):
    return Gr00tN1d7Config(
        backbone_embedding_dim=16,
        input_embedding_dim=16,
        hidden_size=16,
        max_state_dim=7,
        max_action_dim=7,
        max_num_embodiments=2,
        state_history_length=1,
        action_horizon=4,
        num_inference_timesteps=2,
        max_seq_len=8,
        state_dropout_prob=0.0,
        diffusion_model_cfg=dict(
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=8,
            output_dim=16,
            dropout=0.0,
            final_dropout=False,
            interleave_self_attention=True,
            positional_embeddings=None,
        ),
        **kwargs,
    )


def test_checkpoint_identity_binds_weights_not_just_config(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "part.safetensors"}})
    )
    save_file({"x": torch.ones(2)}, tmp_path / "part.safetensors")
    manifest = {"base_checkpoint_files": runner.checkpoint_identity(tmp_path)}
    runner.validate_checkpoint_identity(tmp_path, manifest)
    save_file({"x": torch.zeros(2)}, tmp_path / "part.safetensors")
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        runner.validate_checkpoint_identity(tmp_path, manifest)


def test_cache_packing_does_not_mutate_source():
    record = {
        "raw": {"backbone_features": torch.ones(1, 3, 16)},
        "action_input": {"state": torch.ones(1, 1, 132)},
    }
    raw, action = runner.pack_records([record, record], device="cpu")
    raw.backbone_features[0] = 0
    action.state = action.state.reshape(2, 132)
    assert record["raw"]["backbone_features"].sum() == 48
    assert record["action_input"]["state"].shape == (1, 1, 132)


def test_attention_free_model_rejects_unmasked_native_route():
    with pytest.raises(ValueError, match="masked alternate-VL"):
        Gr00tN1d7ActionHead(
            tiny_config(action_backbone="conditional_mamba2", use_alternate_vl_dit=False)
        )


def test_compact_checkpoint_roundtrip_and_tamper_rejection(tmp_path):
    head = Gr00tN1d7ActionHead(tiny_config())
    cache = tmp_path / "cache"
    processor = cache / "processor"
    processor.mkdir(parents=True)
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        (processor / name).write_text("{}")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    args = Namespace(cache_root=cache, base_checkpoint=base, condition="dit_pretrained")
    target = tmp_path / "checkpoint"
    training.save_head(head, target, args, {})
    reloaded, metadata = training.load_compact_head(target)
    assert metadata["condition"] == "dit_pretrained"
    assert set(head.state_dict()) == set(reloaded.state_dict())
    assert all(
        torch.equal(value, reloaded.state_dict()[name]) for name, value in head.state_dict().items()
    )
    with (target / "head.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        training.load_compact_head(target)
