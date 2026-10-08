"""CPU checks of scoped native draws, FP32 discrete state semantics and paired inference evidence."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_trajectory_kd as runner  # noqa:E402
import action_transfer_trajectory_kd_support as support  # noqa:E402


class Core(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        value = self.scale * (hidden_states.float() + encoder_hidden_states.mean(1, keepdim=True))
        return (value, [value]) if kwargs.get("return_all_hidden_states") else value


class Encoder(nn.Module):
    def forward(self, values, time, emb):
        return values.float() + time[:, None, None].float() * 0.0001


class Decoder(nn.Module):
    def forward(self, values, emb):
        return values


class Head(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.model = teacher
        self.action_encoder = Encoder()
        self.action_decoder = Decoder()

    def sample_time(self, size, *, device, dtype):
        return torch.rand(size, device=device, dtype=torch.float32).to(dtype)

    def forward(self, raw, action):
        target = action["action"]
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype)
        t = self.sample_time(len(target), device=target.device, dtype=target.dtype)[:, None, None]
        x = (1 - t) * noise + t * target
        time = (t[:, 0, 0] * 1000).long()
        post = raw["backbone_features"].float() * 1.5
        state = action["state"].float()
        hidden = torch.cat((state, self.action_encoder(x, time, action["embodiment_id"])), 1)
        output, _ = self.model(
            hidden_states=hidden,
            encoder_hidden_states=post,
            timestep=time,
            image_mask=raw["image_mask"],
            backbone_attention_mask=raw["backbone_attention_mask"],
            return_all_hidden_states=True,
        )
        velocity = self.action_decoder(output, action["embodiment_id"])[:, -40:]
        mask = action["action_mask"].float()
        loss = ((velocity - (target - noise)).float().square() * mask).sum() / (mask.sum() + 1e-6)
        return {
            "loss": loss,
            "action_mask": mask,
            "state_features": state,
            "backbone_features": post,
        }


def fixture(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    teacher, student = Core(0.1).eval().requires_grad_(False), Core(0.2).eval()
    head = Head(teacher).eval()
    model = SimpleNamespace(action_head=head, device=torch.device("cpu"))
    mask = torch.zeros(4, 40, 13, dtype=torch.bfloat16)
    mask[:, :16] = 1
    frame = {
        "raw": {
            "backbone_features": torch.ones(4, 209, 13, dtype=torch.bfloat16),
            "image_mask": torch.ones(4, 209, dtype=torch.bool),
            "backbone_attention_mask": torch.ones(4, 209, dtype=torch.bool),
        },
        "action_input": {
            "action": torch.ones(4, 40, 13, dtype=torch.bfloat16) * 0.5,
            "action_mask": mask,
            "state": torch.zeros(4, 1, 13, dtype=torch.bfloat16),
            "embodiment_id": torch.tensor([10] * 4),
        },
    }

    def packet(model, frame):
        return {
            "features": SimpleNamespace(
                backbone_features=frame["raw"]["backbone_features"].float() * 1.5,
                state_features=frame["action_input"]["state"].float(),
            ),
            "action": SimpleNamespace(embodiment_id=frame["action_input"]["embodiment_id"]),
        }

    def sampler(model, packet):
        z = packet["features"].backbone_features
        state = packet["features"].state_features
        emb = packet["action"].embodiment_id
        actions = torch.randn(size=(4, 40, 13), device=z.device, dtype=z.dtype)
        for bucket in (0, 250, 500, 750):
            t = torch.full((4,), bucket)
            encoded = head.action_encoder(actions, t, emb)
            output = head.model(
                hidden_states=torch.cat((state, encoded), 1),
                encoder_hidden_states=z,
                timestep=t,
                image_mask=frame["raw"]["image_mask"],
                backbone_attention_mask=frame["raw"]["backbone_attention_mask"],
            )
            actions = actions + 0.25 * head.action_decoder(output, emb)[:, -40:]
        return actions

    monkeypatch.setattr(support, "sampling_packet", packet)
    monkeypatch.setattr(support.proto, "sample_packet", sampler)
    return model, teacher, student, frame


def test_scoped_draws_use_inference_fp32_and_restore_originals(monkeypatch):
    model, teacher, student, frame = fixture(monkeypatch)
    original_randn, original_time = torch.randn, model.action_head.sample_time
    epsilon = torch.randn(4, 40, 13)
    source = {
        **frame,
        "action_input": {
            **frame["action_input"],
            "action": frame["action_input"]["action"].float(),
        },
    }
    with support.discrete_draws(model.action_head, epsilon, 750) as counts:
        values = support.batch_forward(model, teacher, student, source)
    assert counts == {"noise": 1, "time": 1}
    assert values["evidence"]["noisy_action"].dtype == torch.float32
    assert torch.equal(values["evidence"]["discrete_time"], torch.tensor([750] * 4))
    assert torch.equal(
        values["evidence"]["noisy_action"], 0.25 * epsilon + 0.75 * source["action_input"]["action"]
    )
    assert torch.randn is original_randn and model.action_head.sample_time == original_time
    assert int((torch.tensor(0.75, dtype=torch.bfloat16) * 1000).long()) == 752
    values["kd"].backward()
    assert student.scale.grad is not None and teacher.scale.grad is None


def test_draw_override_rejects_dtype_count_and_restores_after_failure(monkeypatch):
    model, _, _, _ = fixture(monkeypatch)
    original = torch.randn
    epsilon = torch.zeros(4, 40, 13)
    with pytest.raises(ValueError, match="dtype"):
        with support.discrete_draws(model.action_head, epsilon, 0):
            torch.randn((4, 40, 13), device="cpu", dtype=torch.bfloat16)
    assert torch.randn is original and "sample_time" not in model.action_head.__dict__
    with pytest.raises(ValueError, match="exactly one"):
        with support.discrete_draws(model.action_head, epsilon, 0):
            pass
    with pytest.raises(RuntimeError, match="native failure"):
        with support.discrete_draws(model.action_head, epsilon, 0):
            raise RuntimeError("native failure")
    assert torch.randn is original and "sample_time" not in model.action_head.__dict__


def test_teacher_capture_is_original_sampler_with_cloned_fp32_states(monkeypatch):
    model, teacher, _, frame = fixture(monkeypatch)
    original = (
        model.action_head.model,
        model.action_head.action_encoder,
        model.action_head.action_decoder,
    )
    torch.manual_seed(42)
    trace = support.capture_teacher(model, frame)
    torch.manual_seed(42)
    normal = support.proto.sample_packet(model, support.sampling_packet(model, frame))
    assert torch.equal(trace["final_action"], normal)
    assert original == (
        model.action_head.model,
        model.action_head.action_encoder,
        model.action_head.action_decoder,
    )
    assert all(
        s["x_t"].dtype == torch.float32 and not s["x_t"].requires_grad for s in trace["steps"]
    )
    assert trace["epsilon"].data_ptr() != trace["steps"][0]["x_t"].data_ptr()
    assert all(
        a["x_t"].data_ptr() != b["x_t"].data_ptr()
        for a, b in zip(trace["steps"], trace["steps"][1:])
    )


def test_same_weight_t0_inputs_targets_outputs_exact_but_late_states_differ(monkeypatch):
    model, teacher, student, frame = fixture(monkeypatch)
    torch.manual_seed(42)
    trace = support.capture_teacher(model, frame)
    a = support.discrete_query(model, teacher, student, frame, trace, 0, "data_line_mixed")
    b = support.discrete_query(model, teacher, student, frame, trace, 0, "teacher_trajectory_mixed")
    assert support.assert_t0_equal(a, b)["passed"]
    a = support.discrete_query(model, teacher, student, frame, trace, 750, "data_line_mixed")
    b = support.discrete_query(
        model, teacher, student, frame, trace, 750, "teacher_trajectory_mixed"
    )
    assert not torch.equal(a["evidence"]["noisy_action"], b["evidence"]["noisy_action"])
    sa, _, _ = support.query_receipts(a, frame, 42)
    sb, _, _ = support.query_receipts(b, frame, 42)
    assert sa == sb
    b["kd"].backward()
    assert student.scale.grad is not None and teacher.scale.grad is None


def test_common_branch_preserves_bf16_native_draws_and_receipts(monkeypatch):
    model, teacher, student, frame = fixture(monkeypatch)
    torch.manual_seed(42)
    a = support.continuous_query(model, teacher, student, frame)
    assert a["epsilon"].dtype == a["evidence"]["continuous_time"].dtype == torch.bfloat16
    assert a["evidence"]["noisy_action"].dtype == torch.bfloat16
    assert a["teacher_trace"] is None and a["native_loss_diagnostic"] is not None
    torch.manual_seed(42)
    b = support.batch_forward(model, teacher, student, frame)
    assert torch.equal(a["kd"], b["kd"]) and torch.equal(
        a["student_velocity"], b["student_velocity"]
    )


def test_episode_bootstrap_is_paired_deterministic_and_reports_imitation():
    episodes = list(range(8))
    keys = ("flow_loss", "combined_mae_mixed_units", "teacher_action_physical_mae")
    control = {"per_episode": {str(ep): {k: float(ep + 2) for k in keys} for ep in episodes}}
    candidate = {"per_episode": {str(ep): {k: float(ep + 1) for k in keys} for ep in episodes}}
    spec = {"resamples": 10000, "seed": 20261009, "unit": "episode", "confidence": 0.95}
    value = runner.paired_bootstrap(candidate, control, episodes, spec)
    assert value == runner.paired_bootstrap(candidate, control, episodes, spec)
    assert set(value["metrics"]) == set(keys)
    for row in value["metrics"].values():
        assert (
            row["mean_difference"] == -1
            and row["paired_mean_difference_ci95"] == [-1.0, -1.0]
            and row["episodes_better"] == 8
        )


def test_bootstrap_relative_gain_uses_ratio_of_means_and_defines_zero_policy():
    episodes = list(range(8))
    keys = ("flow_loss", "combined_mae_mixed_units", "teacher_action_physical_mae")
    control = {"per_episode": {str(ep): {k: float(ep + 1) for k in keys} for ep in episodes}}
    candidate = {
        "per_episode": {str(ep): {k: float(ep + 1) * 0.8 for k in keys} for ep in episodes}
    }
    spec = {"resamples": 10000, "seed": 20261009, "unit": "episode", "confidence": 0.95}
    result = runner.paired_bootstrap(candidate, control, episodes, spec)
    for row in result["metrics"].values():
        assert row["relative_gain"] == pytest.approx(0.2)
        assert row["paired_relative_gain_ci95"] == pytest.approx([0.2, 0.2])
    for ep in episodes:
        for key in keys:
            control["per_episode"][str(ep)][key] = 0.0
    result = runner.paired_bootstrap(candidate, control, episodes, spec)
    for row in result["metrics"].values():
        assert row["relative_gain"] is None and row["paired_relative_gain_ci95"] is None
        assert row["nonpositive_bootstrap_denominators"] == 10000
