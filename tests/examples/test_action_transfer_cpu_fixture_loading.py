"""CPU fixture construction uses a per-load native override, never a GPU/global patch."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_checkpoint as checkpoint  # noqa:E402
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa:E402
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa:E402
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone  # noqa:E402
from run_train_eval import strict_reload  # noqa:E402


@pytest.fixture(autouse=True)
def actual_cpu_weights(monkeypatch):
    # The repository CPU conftest normally skips HF weights; these loader tests
    # deliberately exercise real pretrained loading and its explicit skip guard.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("GROOT_SKIP_HF_MODEL_WEIGHTS", raising=False)


def config_file(tmp_path):
    config = Gr00tN1d7Config(use_flash_attention=True)
    config.save_pretrained(tmp_path)
    return tmp_path / "config.json"


def test_native_config_override_is_local_and_default_config_remains_flash(tmp_path):
    path = config_file(tmp_path)
    original = path.read_bytes()
    cpu = Gr00tN1d7Config.from_pretrained(
        tmp_path, local_files_only=True, use_flash_attention=False
    )
    normal = Gr00tN1d7Config.from_pretrained(tmp_path, local_files_only=True)
    assert cpu.use_flash_attention is False and normal.use_flash_attention is True
    assert path.read_bytes() == original and json.loads(original)["use_flash_attention"] is True
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", [None, "factory", "missing_keys", "wrong_backend"])
def test_cpu_loader_scope_strictness_and_ordinary_loader_restoration(
    monkeypatch, tmp_path, failure
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path = config_file(tmp_path)
    original = path.read_bytes()
    calls = []

    def factory(cls, source, **kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] and kwargs["output_loading_info"]
        assert kwargs["ignore_mismatched_sizes"] is False
        assert kwargs["transformers_loading_kwargs"] == {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        cpu = kwargs.get("use_flash_attention") is False
        if cpu and failure == "factory":
            raise RuntimeError("strict load failed")
        cfg = Gr00tN1d7Config.from_pretrained(
            source, local_files_only=True, **({"use_flash_attention": False} if cpu else {})
        )
        backend = "sdpa" if cpu else "flash_attention_2"
        if cpu and failure == "wrong_backend":
            backend = "eager"
        model = SimpleNamespace(
            config=cfg,
            backbone=SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=backend))
            ),
        )
        return model, {"missing_keys": ["missing"] if cpu and failure == "missing_keys" else []}

    monkeypatch.setattr(Gr00tN1d7, "from_pretrained", classmethod(factory))
    binding = Gr00tN1d7.__dict__["from_pretrained"]
    if failure:
        messages = {
            "factory": "strict load failed",
            "missing_keys": "Strict CPU checkpoint reload failed",
            "wrong_backend": "native SDPA backbone",
        }
        with pytest.raises((RuntimeError, ValueError), match=messages[failure]):
            checkpoint.strict_cpu_fixture_reload(tmp_path)
    else:
        model, info = checkpoint.strict_cpu_fixture_reload(tmp_path)
        assert model.config.use_flash_attention is False
        assert info["cpu_fixture_attention_backend"] == "sdpa"
    assert Gr00tN1d7.__dict__["from_pretrained"] is binding
    ordinary, _ = strict_reload(tmp_path)
    assert ordinary.config.use_flash_attention is True
    assert ordinary.backbone.model.config._attn_implementation == "flash_attention_2"
    assert "use_flash_attention" not in calls[-1]
    assert path.read_bytes() == original and not torch.cuda.is_initialized()


@pytest.mark.parametrize("visible,skip", [("0", False), ("", True)])
def test_cpu_loader_never_enables_cuda_or_skips_weights(monkeypatch, tmp_path, visible, skip):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    if skip:
        monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "1")
    else:
        monkeypatch.delenv("GROOT_SKIP_HF_MODEL_WEIGHTS", raising=False)
    monkeypatch.setattr(
        Gr00tN1d7,
        "from_pretrained",
        lambda *a, **k: pytest.fail("Factory reached despite CPU guard"),
    )
    with pytest.raises(ValueError):
        checkpoint.strict_cpu_fixture_reload(tmp_path)
    assert not torch.cuda.is_initialized()


def test_real_tiny_qwen_backbone_can_strictly_reload_on_cpu_with_native_flag(tmp_path, monkeypatch):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "max_position_embeddings": 32,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 0]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
            "out_hidden_size": 8,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    )
    tiny = Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()
    before = {key: value.clone() for key, value in tiny.state_dict().items()}
    tiny.save_pretrained(tmp_path)
    loaded = Qwen3Backbone(
        model_name=str(tmp_path),
        select_layer=1,
        use_flash_attention=False,
        load_bf16=True,
        transformers_loading_kwargs={"local_files_only": True},
    )
    assert loaded.model.config._attn_implementation == "sdpa"
    assert all(torch.equal(value, loaded.model.state_dict()[key]) for key, value in before.items())
    assert not torch.cuda.is_initialized()
