from __future__ import annotations

import importlib.metadata
import platform

from gr00t.experimental.arena_codec_streaming_backbone import (
    CODEC_FIELDS,
    SEED,
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


@torch.inference_mode()
def _partitioned_stream(module, descriptors: torch.Tensor) -> torch.Tensor:
    state = module.new_state(int(descriptors.shape[0]))
    first, state = module.stream(descriptors[:, :127], state)
    second, state = module.stream(descriptors[:, 127:], state)
    assert state.observations == descriptors.shape[1]
    return torch.cat((first, second), dim=1)


@pytest.mark.gpu
def test_fixed_candidates_target_preflight() -> None:
    if not torch.cuda.is_available():
        pytest.skip("target-runtime preflight requires CUDA")
    try:
        import mamba_ssm.modules.mamba2 as official
    except Exception as error:
        pytest.skip(f"official Mamba2 target runtime unavailable: {error}")

    assert platform.python_version() == "3.12.13"
    assert torch.__version__ == "2.9.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.compiled_with_cxx11_abi()
    assert importlib.metadata.version("mamba-ssm") == "2.3.2.post1"
    assert official.causal_conv1d_fn is None
    assert official.causal_conv1d_update is None

    device = torch.device("cuda", 0)
    raw = torch.linspace(
        -0.25,
        0.25,
        steps=128 * 8 * 8 * CODEC_FIELDS,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 128, 1, 8, 8, CODEC_FIELDS)
    tokenizer = ArenaCodecSpatialTokenizer(seed=SEED).eval().to(device)
    mamba_current = CausalMamba2Backbone(seed=SEED, chunk_size=16).eval().to(device)
    mamba_candidate = CausalMamba2Backbone(seed=SEED, chunk_size=1).eval().to(device)
    mamba_candidate.load_state_dict(mamba_current.state_dict(), strict=True)
    gru_current = CausalGRUBackbone(seed=SEED).eval().to(device)
    gru_candidate = CausalGRUBackbone(seed=SEED).eval().to(device)
    gru_candidate.load_state_dict(gru_current.state_dict(), strict=True)
    assert all(
        torch.equal(mamba_current.state_dict()[name], mamba_candidate.state_dict()[name])
        for name in mamba_current.state_dict()
    )
    assert all(
        torch.equal(gru_current.state_dict()[name], gru_candidate.state_dict()[name])
        for name in gru_current.state_dict()
    )

    with torch.inference_mode():
        descriptors = tokenizer(raw)
        mamba_full_first = mamba_candidate(descriptors)
        mamba_full_second = mamba_candidate(descriptors)
        mamba_stream_first, _ = mamba_candidate.stream(descriptors)
        mamba_stream_second, _ = mamba_candidate.stream(descriptors)
        mamba_partitioned = _partitioned_stream(mamba_candidate, descriptors)
        gru_full_first = gru_candidate.forward_explicit(descriptors)
        gru_full_second = gru_candidate.forward_explicit(descriptors)
        gru_stream_first, _ = gru_candidate.stream(descriptors)
        gru_stream_second, _ = gru_candidate.stream(descriptors)
        gru_partitioned = _partitioned_stream(gru_candidate, descriptors)
    assert torch.equal(mamba_full_first, mamba_full_second)
    assert torch.equal(mamba_stream_first, mamba_stream_second)
    assert torch.equal(mamba_stream_first, mamba_partitioned)
    assert (mamba_full_first - mamba_stream_first).abs().max() <= 1e-4
    assert torch.equal(gru_full_first, gru_full_second)
    assert torch.equal(gru_stream_first, gru_stream_second)
    assert torch.equal(gru_stream_first, gru_partitioned)
    assert torch.equal(gru_full_first, gru_stream_first)

    for temporal, full_path in (
        (mamba_candidate, mamba_candidate.forward),
        (gru_candidate, gru_candidate.forward_explicit),
    ):
        tokenizer.train()
        temporal.train()
        tokenizer.zero_grad(set_to_none=True)
        temporal.zero_grad(set_to_none=True)
        output = full_path(tokenizer(raw))
        assert torch.isfinite(output).all()
        output.float().square().mean().backward()
        _assert_finite_nonzero_gradients(tokenizer)
        _assert_finite_nonzero_gradients(temporal)
