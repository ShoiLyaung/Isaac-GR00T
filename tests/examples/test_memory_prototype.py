"""CPU boundary tests; actual Triton numerics/backward remain formal GPU gates."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/G1_DIT_MAMBA_YUQI"))
import memory_prototype as probe  # noqa: E402


class Features(dict):
    __getattr__ = dict.__getitem__


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.calls = []

    @torch.no_grad()
    def get_action_with_features(self, **kwargs):
        self.calls.append(kwargs)
        value = kwargs["backbone_features"].mean() * self.weight
        return {"action_pred": value.expand(1, 40, 13)}


def packet():
    return {
        "features": Features(
            backbone_features=torch.ones(1, 209, 2048), state_features=torch.ones(1, 1, 8)
        ),
        "backbone": Features(
            image_mask=torch.ones(1, 209, dtype=torch.bool),
            backbone_attention_mask=torch.ones(1, 209, dtype=torch.bool),
        ),
        "action": Features(state=torch.ones(1, 1, 13), embodiment_id=torch.tensor([10])),
    }


def test_gradient_dry_path_bypasses_decorator_and_preserves_native_arguments():
    model = SimpleNamespace(action_head=Head())
    values = packet()
    leaf = torch.zeros(192, 2048, requires_grad=True)
    plain = probe.sample_packet(model, values, leaf)
    assert not plain.requires_grad
    predicted = probe.sample_packet(model, values, leaf, gradient_body=True)
    predicted[:, :16].square().mean().backward()
    assert leaf.grad is not None and leaf.grad.abs().sum() > 0 and torch.isfinite(leaf.grad).all()
    assert model.action_head.weight.grad is None
    call = model.action_head.calls[-1]
    assert call["state_features"] is values["features"].state_features
    assert call["embodiment_id"] is values["action"].embodiment_id
    assert call["action_input"] is values["action"]
    assert call["backbone_output"] is values["backbone"]
    assert call["options"] is None
    values["action"]["action"] = torch.zeros(1, 40, 13)
    with pytest.raises(ValueError, match="teacher/RTC"):
        probe.sample_packet(model, values)


def test_disable_never_encodes_or_updates_memory_and_reset_is_not_disable(monkeypatch):
    calls = []

    class Model:
        def get_action(self, obs, options=None):
            calls.append("native")
            return {"action_pred": torch.zeros(1, 40, 13)}

    class State:
        observations = 7

        def reset(self):
            calls.append("reset")
            self.observations = 0

    class Memory:
        def step(self, slots, state):
            assert not torch.is_grad_enabled()
            calls.append("step")
            state.observations += 1
            return slots, torch.zeros(192, 2048)

    state = State()
    memory = Memory()
    model = Model()
    monkeypatch.setattr(probe, "encode_observation", lambda *a: calls.append("encode") or packet())
    monkeypatch.setattr(
        probe, "sample_packet", lambda *a: calls.append("sample") or torch.zeros(1, 40, 13)
    )
    probe.policy_with_memory(model, memory, state, {}, enabled=False, current_reset=True)
    assert calls == ["native"] and state.observations == 7
    calls.clear()
    probe.policy_with_memory(model, memory, state, {}, current_reset=True)
    assert calls == ["reset", "encode", "step", "sample"] and state.observations == 1


def good_input():
    ids = torch.full((1, 209), 151655, dtype=torch.long)
    ids[0, probe.NONIMAGE_POSITIONS] = torch.tensor(probe.NONIMAGE_IDS)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "image_grid_thw": torch.tensor([[1, 16, 16]] * 3),
        "state": torch.zeros(1, 1, 13),
    }


def test_full_token_camera_and_grid_identity_fail_closed():
    values = good_input()
    probe.validate_input_layout(values, probe.CAMERAS)
    with pytest.raises(ValueError, match="camera"):
        probe.validate_input_layout(values, list(reversed(probe.CAMERAS)))
    values["input_ids"][0, 202] += 1
    with pytest.raises(ValueError, match="layout"):
        probe.validate_input_layout(values, probe.CAMERAS)
    values = good_input()
    values["image_grid_thw"][2, 2] = 8
    with pytest.raises(ValueError, match="grids"):
        probe.validate_input_layout(values, probe.CAMERAS)
    values = good_input()
    values["action_mask"] = torch.ones(1)
    with pytest.raises(ValueError, match="targets"):
        probe.validate_input_layout(values, probe.CAMERAS)


def test_frozen_numerical_tolerances_and_cost_gate_boundaries():
    probe.check_tolerance({"rms": 0.01, "relative_rms": 0.02, "maxabs": 0.1})
    probe.check_tolerance({"rms": 0.002, "relative_rms": 10, "maxabs": 0.02}, action=True)
    with pytest.raises(ValueError, match="latent"):
        probe.check_tolerance({"rms": 0.01, "relative_rms": 0.02001, "maxabs": 0.1})
    with pytest.raises(ValueError, match="action"):
        probe.check_tolerance({"rms": 0.002, "relative_rms": 0, "maxabs": 0.02001}, action=True)
    native = {"p95_ms": 100, "peak_allocated_bytes": 6 * 2**30}
    enabled = {"p95_ms": 110, "peak_allocated_bytes": 6 * 2**30 + 512 * 2**20}
    gate = probe.cost_gate(native, enabled)
    assert gate["latency_passed"] and gate["extra_peak_allocated_passed"]
    enabled["p95_ms"] = 110.001
    enabled["peak_allocated_bytes"] += 1
    gate = probe.cost_gate(native, enabled)
    assert not gate["latency_passed"] and not gate["extra_peak_allocated_passed"]
    enabled["peak_allocated_bytes"] = 45 * 2**30
    assert not probe.cost_gate(native, enabled)["absolute_allocated_passed"]


def test_warmups_are_excluded_but_memory_peaks_are_not():
    rows = [
        {
            "wall_ms": 10000 if i < 5 else 100,
            "warmup": i < 5,
            "peak_allocated_bytes": 999 if i == 0 else 1,
            "peak_reserved_bytes": 1234,
        }
        for i in range(55)
    ]
    result = probe.timing_stats(rows)
    assert (
        result["count"] == 50 and result["p95_ms"] == 100 and result["peak_allocated_bytes"] == 999
    )
    with pytest.raises(ValueError, match="fifty"):
        probe.timing_stats(rows[:-1])


def test_failed_attempt_keeps_terminal_evidence_and_cannot_be_reused(monkeypatch, tmp_path):
    calls = []

    def fail(args, deadline):
        calls.append(deadline)
        probe.write_new(args.output / "TECHNICAL_RESULT.json", {"status": "passed"})
        raise ValueError("Frozen memory cost gate failed")

    monkeypatch.setattr(probe, "run", fail)
    args = SimpleNamespace(output=tmp_path / "attempt", wall_seconds=10)
    with pytest.raises(ValueError, match="cost gate"):
        probe.execute(args)
    path = args.output / "RESULT.json"
    original = path.read_bytes()
    result = json.loads(original)
    assert result["status"] == "failed" and result["technical_validity"] == "passed"
    assert result["optimizer_steps"] == 0 and result["automatic_retry"] is False
    with pytest.raises(FileExistsError):
        probe.execute(args)
    assert len(calls) == 1 and path.read_bytes() == original


def test_wall_and_storage_budget_are_fail_closed(monkeypatch, tmp_path):
    with pytest.raises(TimeoutError):
        probe.check_budget(-1, tmp_path, cuda=False)
    monkeypatch.setattr(probe, "MAX_BYTES", 1)
    (tmp_path / "bytes").write_bytes(b"12")
    with pytest.raises(RuntimeError, match="artifact"):
        probe.check_budget(float("inf"), tmp_path, cuda=False)
