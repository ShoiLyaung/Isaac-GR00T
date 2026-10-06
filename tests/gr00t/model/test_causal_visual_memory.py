"""CPU contracts for slot-time orientation, official cache layout and zero recovery."""

import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.model.modules.causal_visual_memory import (
    CACHE_ELEMENTS,
    CausalVisualMemory,
    inject_image_residual,
)
import pytest
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/G1_DIT_MAMBA_YUQI"))
import memory_prototype as probe  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_constructor_contract(monkeypatch):
    # Triton autotuner import requires an active CUDA driver. Execute only the
    # official CPU constructor/allocator source bodies in an isolated namespace;
    # this is source-contract validation, NOT an official runtime/kernel probe.
    package = next(
        (
            Path(p) / "mamba_ssm"
            for p in sys.path
            if (Path(p) / "mamba_ssm/modules/mamba2.py").is_file()
        ),
        None,
    )
    if package is None:
        pytest.skip("Frozen Mamba source unavailable for isolated CPU contract test")
    scope = {"torch": torch, "nn": nn, "math": math}
    for path, name, methods in (
        (package / "ops/triton/layernorm_gated.py", "RMSNorm", {"__init__", "reset_parameters"}),
        (package / "modules/mamba2.py", "Mamba2", {"__init__", "allocate_inference_cache"}),
    ):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        cls.bases = cls.bases[:1]
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
        if name == "RMSNorm":
            scope["RMSNormGated"] = scope[name]
    monkeypatch.setitem(sys.modules, "mamba_ssm", SimpleNamespace(Mamba2=scope["Mamba2"]))


class ToyMamba(nn.Module):
    """Differentiable cumulative recurrence, not a substitute for official CUDA evidence."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((256,), 0.2))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype):
        return (
            torch.zeros(batch_size, 4, 640, dtype=dtype).transpose(1, 2),
            torch.zeros(batch_size, 8, 64, 64, dtype=dtype),
        )

    def forward(self, value):
        return (value * self.weight).cumsum(dim=1)

    def step(self, value, conv, ssm):
        result = conv[:, :256, -1] + value.squeeze(1) * self.weight
        conv[:, :256, -1].copy_(result)
        return result.unsqueeze(1), conv, ssm


def toy_memory():
    module = CausalVisualMemory()
    for layer in module.layers:
        layer.mamba = ToyMamba()
    return module


def test_isolated_constructor_and_allocator_source_contract():
    torch.set_num_threads(4)
    torch.manual_seed(987)
    before = torch.get_rng_state().clone()
    memory = CausalVisualMemory()
    assert torch.equal(torch.get_rng_state(), before)
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in memory.parameters())
    assert torch.count_nonzero(memory.output_projection.weight) == 0
    assert torch.count_nonzero(memory.output_projection.bias) == 0
    another = CausalVisualMemory()
    assert all(torch.equal(p, q) for p, q in zip(memory.parameters(), another.parameters()))
    for layer in memory.layers:
        m = layer.mamba
        assert (m.d_model, m.d_state, m.d_conv, m.expand, m.headdim, m.ngroups, m.chunk_size) == (
            256,
            64,
            4,
            2,
            64,
            1,
            16,
        )
        assert m.dt_limit == (0.0, float("inf")) and not m.use_mem_eff_path
    state = memory.new_state()
    report = state.describe()
    assert report["elements"] == CACHE_ELEMENTS == 13_565_952
    assert report["bytes"] == 54_263_808
    assert len({t.data_ptr() for pair in state.layers for t in pair}) == 4
    cloned = state.clone()
    moved = state.to("cpu")
    for a, b, c in zip(
        [t for p in state.layers for t in p],
        [t for p in cloned.layers for t in p],
        [t for p in moved.layers for t in p],
    ):
        assert len({a.data_ptr(), b.data_ptr(), c.data_ptr()}) == 3
        assert a.stride() == b.stride() == c.stride()
    cloned.layers[0][0].fill_(7)
    assert torch.count_nonzero(state.layers[0][0]) == 0
    cloned.observations = 19
    cloned.reset()
    assert cloned.observations == 0 and all(
        torch.count_nonzero(t) == 0 for pair in cloned.layers for t in pair
    )


def test_slots_are_independent_and_sequence_dimension_is_observation_time():
    memory = toy_memory()
    sequence = torch.randn(3, 192, 2048)
    altered = sequence.clone()
    altered[2, 0] += torch.randn(2048)
    first = memory.full_sequence(sequence)
    second = memory.full_sequence(altered)
    assert torch.equal(first[:2], second[:2])
    assert torch.equal(first[:, 1:], second[:, 1:])
    assert not torch.equal(first[2, 0], second[2, 0])
    with torch.enable_grad():
        step, state = probe.step_sequence(memory, sequence)
    assert not step.requires_grad and state.observations == 3
    assert torch.allclose(first, step, atol=2e-6, rtol=1e-5)
    left, carried = probe.step_sequence(memory, sequence[:1])
    right, carried = probe.step_sequence(memory, sequence[1:], carried)
    assert torch.equal(torch.cat([left, right]), step)
    assert probe.state_hash(state) == probe.state_hash(carried)
    state.reset()
    reset, _ = memory.step(sequence[0], state)
    fresh = memory.new_state()
    expected, _ = memory.step(sequence[0], fresh)
    assert torch.equal(reset, expected)
    assert probe.state_hash(state) == probe.state_hash(fresh)
    with pytest.raises(ValueError, match="observation_time"):
        memory.full_sequence(sequence.transpose(0, 1))


def test_zero_projection_preserves_original_tokens_and_has_trainable_readout():
    memory = toy_memory()
    sequence = torch.randn(2, 192, 2048)
    latent = memory.full_sequence(sequence)
    residual = memory.output_projection(latent[-1])
    value = torch.randn(1, 209, 2048)
    positions = torch.tensor(probe.IMAGE_POSITIONS)
    output = inject_image_residual(value, positions, residual)
    assert torch.equal(output, value)
    output.square().mean().backward()
    assert memory.output_projection.weight.grad.abs().sum() > 0
    assert memory.input_projection.weight.grad.abs().sum() == 0
    memory.zero_grad(set_to_none=True)
    with torch.no_grad():
        memory.output_projection.weight.normal_(std=1e-3)
    output = inject_image_residual(
        value, positions, memory.output_projection(memory.full_sequence(sequence)[-1])
    )
    assert torch.equal(output[:, probe.NONIMAGE_POSITIONS], value[:, probe.NONIMAGE_POSITIONS])
    assert not torch.equal(output[:, probe.IMAGE_POSITIONS], value[:, probe.IMAGE_POSITIONS])
    output.square().mean().backward()
    for group in [
        memory.input_projection,
        *[layer.mamba for layer in memory.layers],
        memory.output_projection,
    ]:
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0)
            for p in group.parameters()
        )
