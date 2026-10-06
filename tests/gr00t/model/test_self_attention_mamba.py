"""CPU stand-in tests for the limited self-attention replacement boundary."""

from unittest.mock import patch

from diffusers.models.attention import Attention
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.modules import mamba_action_mixer
from gr00t.model.modules.dit import AlternateVLDiT
from gr00t.model.modules.self_attention_mamba import modify_native_self_attention
import pytest
import torch
from torch import nn


class StandIn(nn.Module):
    def __init__(self, d_model, device=None, dtype=None, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, value):
        return self.proj(value).cumsum(1) / value.shape[1]


class TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def config(**overrides):
    options = dict(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=13,
        max_action_dim=13,
        action_horizon=40,
        state_history_length=1,
        num_inference_timesteps=4,
        max_num_embodiments=16,
        max_seq_len=64,
        use_alternate_vl_dit=True,
        use_vlln=True,
        state_dropout_prob=0.0,
        action_mamba2_cfg=dict(
            d_state=16, d_conv=4, expand=2, headdim=16, chunk_size=16, use_mem_eff_path=False
        ),
        diffusion_model_cfg=dict(
            num_layers=32,
            num_attention_heads=2,
            attention_head_dim=16,
            norm_type="ada_norm",
            dropout=0.0,
            final_dropout=False,
            output_dim=32,
            positional_embeddings=None,
            interleave_self_attention=True,
        ),
    )
    options.update(overrides)
    return Gr00tN1d7Config(**options)


@pytest.fixture(autouse=True)
def cpu_double(monkeypatch):
    monkeypatch.setattr(mamba_action_mixer, "_mamba2_class", lambda: StandIn)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def holder():
    model = nn.Module()
    model.config = config()
    model.action_head = Gr00tN1d7ActionHead(model.config)
    model.register_buffer("audit_buffer", torch.tensor([7]))
    return model


@pytest.mark.parametrize("reset_only", [False, True])
def test_exact_sixteen_module_boundary_and_content_audit(reset_only):
    model = holder()
    original = model.action_head.model
    before = {name: value.clone() for name, value in model.state_dict().items()}
    audit = modify_native_self_attention(model, reset_only=reset_only)
    prefixes = tuple(audit["replacement_prefixes"])
    assert len(prefixes) == 16
    assert audit["retained_content_sha256_before"] == audit["retained_content_sha256_after"]
    assert audit["retained_identity_and_version_equal"]
    assert original is model.action_head.model
    assert type(original).forward is AlternateVLDiT.forward
    for name, value in model.state_dict().items():
        if not name.startswith(prefixes):
            assert torch.equal(value, before[name]), name
    for index, block in enumerate(original.transformer_blocks):
        if index % 2 == 0 or reset_only:
            assert isinstance(block.attn1, Attention)
        else:
            assert isinstance(block.attn1, mamba_action_mixer.BidirectionalMamba2SelfMixer)
    if reset_only:
        assert len(audit["reset_linear_modules"]) == 64
        assert model.config.action_backbone == "dit"
        for prefix in prefixes:
            assert any(
                not torch.equal(v, before[k])
                for k, v in model.state_dict().items()
                if k.startswith(prefix)
            )
    else:
        assert model.config.action_backbone == "dit_self_mamba2"


def test_forward_abi_mask_exclusion_and_all_sixteen_gradients():
    model = holder()
    modify_native_self_attention(model)
    backbone = model.action_head.model.eval()
    hidden = torch.randn(1, 41, 32)
    condition = torch.randn(1, 6, 32)
    mask = torch.tensor([[True, True, True, True, False, False]])
    images = torch.tensor([[True, True, False, False, True, False]])
    time = torch.tensor([30])
    first, intermediates = backbone(
        hidden,
        condition,
        time,
        image_mask=images,
        backbone_attention_mask=mask,
        return_all_hidden_states=True,
    )
    altered = condition.clone()
    altered[:, 4:] += 1000
    second = backbone(hidden, altered, time, image_mask=images, backbone_attention_mask=mask)
    assert torch.equal(first, second)
    assert first.shape == (1, 41, 32) and len(intermediates) == 33
    first.square().mean().backward()
    for index in range(1, 32, 2):
        grads = [p.grad for p in backbone.transformer_blocks[index].attn1.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    with pytest.raises(ValueError, match="unmasked"):
        backbone.transformer_blocks[1].attn1(hidden, attention_mask=mask)


def test_hf_save_constructor_strict_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    with (
        patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls", return_value=TinyBackbone),
        patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"),
    ):
        native = Gr00tN1d7(config())
        modify_native_self_attention(native)
        native.save_pretrained(tmp_path)
        restored, info = Gr00tN1d7.from_pretrained(
            tmp_path, local_files_only=True, output_loading_info=True
        )
    assert not any(
        info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    assert native.state_dict().keys() == restored.state_dict().keys()
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in native.state_dict().items()
    )
    assert restored.config.action_backbone == "dit_self_mamba2"


def test_fail_closed_without_alternating_native_interface():
    with pytest.raises(ValueError, match="AlternateVLDiT"):
        Gr00tN1d7ActionHead(config(action_backbone="dit_self_mamba2", use_alternate_vl_dit=False))
    model = holder()
    model.action_head.model.transformer_blocks[1].cross_attention_dim = 32
    with pytest.raises(ValueError, match="order"):
        modify_native_self_attention(model)
