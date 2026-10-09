from __future__ import annotations

import sys
import types

from gr00t.experimental.arena_codec_streaming_backbone import (
    CODEC_FIELDS,
    WIDTH,
    ArenaCodecSpatialTokenizer,
    CausalGRUBackbone,
    CausalMamba2Backbone,
)
import pytest
import torch
from torch import nn


def _assert_finite_nonzero_gradients(module: nn.Module) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(torch.count_nonzero(gradient) for gradient in gradients if gradient is not None)


def test_gru_explicit_recurrence_matches_step_and_preserves_autograd() -> None:
    tokenizer = ArenaCodecSpatialTokenizer().eval()
    module = CausalGRUBackbone().eval()
    raw = torch.randn(1, 12, 1, 8, 8, CODEC_FIELDS)
    descriptors = tokenizer(raw)
    explicit = module.forward_explicit(descriptors)
    with torch.inference_mode():
        streamed, _ = module.stream(descriptors.detach())
    torch.testing.assert_close(explicit.detach(), streamed, rtol=0, atol=0)
    explicit.square().mean().backward()
    _assert_finite_nonzero_gradients(tokenizer)
    _assert_finite_nonzero_gradients(module)


class _FakeMamba2(nn.Module):
    def __init__(self, *, d_model: int, chunk_size: int, **kwargs):
        super().__init__()
        del kwargs
        self.chunk_size = chunk_size
        self.projection = nn.Linear(d_model, d_model, bias=False)


def test_mamba_chunk_size_override_preserves_state_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_package = types.ModuleType("mamba_ssm")
    fake_package.Mamba2 = _FakeMamba2
    monkeypatch.setitem(sys.modules, "mamba_ssm", fake_package)
    default = CausalMamba2Backbone()
    current = CausalMamba2Backbone(chunk_size=16)
    candidate = CausalMamba2Backbone(chunk_size=1)
    assert default.chunk_size == 16
    assert [layer.mamba.chunk_size for layer in default.layers] == [16, 16]
    assert [layer.mamba.chunk_size for layer in current.layers] == [16, 16]
    assert [layer.mamba.chunk_size for layer in candidate.layers] == [1, 1]
    assert current.state_dict().keys() == candidate.state_dict().keys()
    assert all(
        torch.equal(current.state_dict()[name], candidate.state_dict()[name])
        for name in current.state_dict()
    )


def test_mamba_rejects_unfrozen_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk size must be 1 or"):
        CausalMamba2Backbone(chunk_size=8)


def test_gru_explicit_shape_validation_matches_full_path() -> None:
    module = CausalGRUBackbone()
    with pytest.raises(ValueError, match="frame descriptors"):
        module.forward_explicit(torch.zeros(1, 3, WIDTH - 1))
