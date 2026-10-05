"""CPU regression tests for version-scoped Qwen patch-embed layout compatibility."""

import hashlib
import json
from pathlib import Path
import sys

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
import pytest
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/G1_DIT_MAMBA_YUQI"))
from preflight import initialization_evidence  # noqa: E402


def shell():
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    nn.Module.__init__(backbone)
    backbone.model = nn.Module()
    backbone.model.visual = nn.Module()
    backbone.model.visual.patch_embed = nn.Module()
    backbone.model.visual.patch_embed.proj = nn.Conv3d(2, 3, kernel_size=(2, 3, 3))
    backbone.model.visual.rotary_pos_emb = nn.Module()
    backbone.model.visual.rotary_pos_emb.register_buffer(
        "inv_freq", torch.tensor([1.0, 0.01]), persistent=False
    )
    backbone.model.language_model = nn.Module()
    backbone.model.language_model.rotary_emb = nn.Module()
    backbone.model.language_model.rotary_emb.register_buffer(
        "inv_freq", torch.tensor([1.0, 0.1, 0.01]), persistent=False
    )
    backbone.model.language_model.rotary_emb.original_inv_freq = torch.tensor([1.0, 0.1, 0.01])
    model = nn.Module()
    model.backbone = backbone
    return model, backbone, backbone.model.visual.patch_embed.proj


@pytest.mark.parametrize("version", ["2.7.1+cu128", "2.8.0+cu128"])
def test_pre29_keeps_native_ncdhw_and_has_no_conversion_hook(monkeypatch, version):
    _, backbone, convolution = shell()
    before, stride, pointer = (
        convolution.weight.detach().clone(),
        convolution.weight.stride(),
        convolution.weight.data_ptr(),
    )
    monkeypatch.setattr(torch, "__version__", version)
    backbone._apply_vision_patch_embed_channels_last()
    assert convolution.weight.is_contiguous()
    assert convolution.weight.stride() == stride
    assert convolution.weight.data_ptr() == pointer
    assert torch.equal(convolution.weight, before)
    assert len(convolution._forward_pre_hooks) == 0


@pytest.mark.parametrize("version", ["2.9.0+cu128", "2.9.0.dev20250801", "2.10.0", "3.0.0"])
def test_29_and_newer_retain_existing_weight_and_input_conversion(monkeypatch, version):
    _, backbone, convolution = shell()
    before = convolution.weight.detach().clone()
    monkeypatch.setattr(torch, "__version__", version)
    backbone._apply_vision_patch_embed_channels_last()
    assert convolution.weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert not convolution.weight.is_contiguous()
    assert torch.equal(convolution.weight, before)
    assert len(convolution._forward_pre_hooks) == 1
    hook = next(iter(convolution._forward_pre_hooks.values()))
    original = torch.arange(2 * 2 * 4 * 8 * 8).reshape(2, 2, 4, 8, 8).float()
    marker = object()
    converted, kept = hook(convolution, (original, marker))
    assert converted.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(converted, original)
    assert original.is_contiguous() and kept is marker


def test_initialization_evidence_reads_live_stride_hooks_dtype_and_nonpersistent_values():
    model, backbone, convolution = shell()
    before_rng = torch.get_rng_state().clone()
    before_values = backbone.model.visual.rotary_pos_emb.inv_freq.clone()
    before = initialization_evidence(model)
    json.dumps(before)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(before_values, backbone.model.visual.rotary_pos_emb.inv_freq)
    assert before["vision_conv3d"][0]["weight_stride"] == list(convolution.weight.stride())
    assert before["vision_conv3d"][0]["forward_pre_hook_count"] == 0
    vision = before["rope"]["vision"]["values"]["inv_freq"]
    assert vision["dtype"] == "torch.float32" and vision["registered_buffer"]
    assert vision["persistent"] is False
    assert vision["sha256"] == hashlib.sha256(before_values.numpy().tobytes()).hexdigest()
    original_text = before["rope"]["text"]["values"]["original_inv_freq"]
    assert original_text["registered_buffer"] is False and original_text["persistent"] is None
    assert not any("inv_freq" in key for key in model.state_dict())
    # Reflect actual post-cast buffers; do not assume every RoPE tensor stays FP32.
    model.to(dtype=torch.bfloat16)
    after = initialization_evidence(model)
    assert after["vision_conv3d"][0]["weight_dtype"] == "torch.bfloat16"
    assert after["rope"]["vision"]["values"]["inv_freq"]["dtype"] == "torch.bfloat16"
    assert after["rope"]["text"]["values"]["original_inv_freq"]["dtype"] == "torch.float32"
    previous_hash = after["rope"]["vision"]["values"]["inv_freq"]["sha256"]
    backbone.model.visual.rotary_pos_emb.inv_freq[0] = 2
    assert (
        initialization_evidence(model)["rope"]["vision"]["values"]["inv_freq"]["sha256"]
        != previous_hash
    )
