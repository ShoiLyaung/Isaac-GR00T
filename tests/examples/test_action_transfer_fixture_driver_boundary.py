"""CPU mocks for the explicit fixture CUDA-driver boundary; no GPU is started."""

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_checkpoint as checkpoint  # noqa:E402


def mocked_driver(monkeypatch, *, visible="1", count=1, initialized=False, allocated=0):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: allocated)


@pytest.mark.parametrize("initialized", [False, True])
def test_driver_initialization_is_allowed_and_recorded_without_tensor_allocation(
    monkeypatch, initialized
):
    real_initialized = torch.cuda.is_initialized
    mocked_driver(monkeypatch, initialized=initialized)
    assert checkpoint.fixture_driver_contract(1, "cpu_weights_cuda_driver_only") == {
        "cuda_initialized": initialized,
        "torch_cuda_memory_allocated_bytes": 0,
    }
    assert not real_initialized()


@pytest.mark.parametrize(
    "gpu,execution,visible,count",
    [
        (0, "cpu_weights_cuda_driver_only", "1", 1),
        (1, "cpu_only", "1", 1),
        (1, "cpu_weights_cuda_driver_only", "0", 1),
        (1, "cpu_weights_cuda_driver_only", "", 0),
        (1, "cpu_weights_cuda_driver_only", "1", 0),
        (1, "cpu_weights_cuda_driver_only", "1", 2),
    ],
)
def test_wrong_gpu_schema_or_visible_device_count_is_rejected(
    monkeypatch, gpu, execution, visible, count
):
    mocked_driver(monkeypatch, visible=visible, count=count)
    with pytest.raises(ValueError):
        checkpoint.fixture_driver_contract(gpu, execution)


def test_any_pytorch_cuda_allocation_is_rejected_and_reportable(monkeypatch):
    mocked_driver(monkeypatch, initialized=True, allocated=2048)
    assert checkpoint.fixture_cuda_state() == {
        "cuda_initialized": True,
        "torch_cuda_memory_allocated_bytes": 2048,
    }
    with pytest.raises(ValueError, match="PyTorch CUDA allocations: 2048"):
        checkpoint.fixture_driver_contract(1, checkpoint.FIXTURE_EXECUTION)


@pytest.mark.parametrize("escape", ["parameter", "buffer"])
def test_all_fixture_parameters_and_buffers_must_stay_cpu(monkeypatch, escape):
    mocked_driver(monkeypatch, initialized=True)
    owner = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    state = checkpoint.assert_fixture_cpu_residency(owner, "student")
    assert state["all_parameters_and_buffers_cpu"] and state["cuda_initialized"]
    assert state["torch_cuda_memory_allocated_bytes"] == 0
    if escape == "parameter":
        owner.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))
    else:
        owner.register_buffer("escaped", torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="escaped CPU"):
        checkpoint.assert_fixture_cpu_residency(owner, "student")
