from __future__ import annotations

import importlib.metadata
import platform

from gr00t.experimental.arena_codec_streaming_backbone import (
    MAMBA2_CONFIG,
    SEED,
    WIDTH,
    CausalMamba2Backbone,
)
import pytest
import torch


@pytest.mark.gpu
def test_official_two_layer_mamba_full_step_and_reset_preflight() -> None:
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
    assert MAMBA2_CONFIG["use_mem_eff_path"] is False

    torch.manual_seed(SEED)
    module = CausalMamba2Backbone(seed=SEED).cuda().eval()
    descriptors = torch.linspace(
        -0.25,
        0.25,
        steps=4 * WIDTH,
        device="cuda",
        dtype=torch.float32,
    ).reshape(1, 4, WIDTH)
    with torch.inference_mode():
        full = module(descriptors)
        streamed, state = module.stream(descriptors)
        state.reset()
        reset_output, state = module.stream(descriptors, state)
        fresh_output, _ = module.stream(descriptors)
    assert full.shape == (1, 4, WIDTH)
    assert torch.isfinite(full).all()
    torch.testing.assert_close(streamed, full, rtol=0, atol=1e-4)
    torch.testing.assert_close(reset_output, fresh_output, rtol=0, atol=0)
    assert state.observations == 4
