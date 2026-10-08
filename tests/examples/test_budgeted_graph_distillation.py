"""CPU tests for KD delegation, true-input replay, restoration and fresh gates."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
from budgeted_action_cudagraph import storage_identity  # noqa: E402
from budgeted_action_support import capture_velocity  # noqa: E402
import budgeted_graph_distillation as support  # noqa: E402
import budgeted_graph_distillation_runner as runner  # noqa: E402
import memory_prototype as proto  # noqa: E402
import memory_training_support  # noqa: E402


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.calls = []
        self.owner = None

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        self.calls.append(
            (
                hidden_states.detach().clone(),
                encoder_hidden_states.detach().clone(),
                timestep.detach().clone(),
            )
        )
        output = self.linear(hidden_states + encoder_hidden_states.mean(1)[:, None])
        return (output, [output]) if kwargs.get("return_all_hidden_states", False) else output


class Encoder(nn.Linear):
    def forward(self, value, time, embodiment):
        return super().forward(value) + time[:, None, None].float() * 0.001


class Decoder(nn.Linear):
    def forward(self, value, embodiment):
        return super().forward(value)


class Head(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.model = teacher
        self.action_encoder = Encoder(13, 4).requires_grad_(False)
        self.action_decoder = Decoder(4, 13).requires_grad_(False)
        self.time_calls = 0

    def sample_time(self, batch, device, dtype):
        self.time_calls += 1
        return torch.rand(batch, device=device, dtype=dtype) * 0.8

    def forward(self, raw, action):
        noise = torch.randn_like(action["action"])
        time = self.sample_time(1, "cpu", torch.float32)
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * action["action"]
        discrete = (time * 1000).long()
        hidden = torch.cat(
            (action["state"], self.action_encoder(noisy, discrete, action["embodiment_id"])), 1
        )
        output, _ = self.model(
            hidden_states=hidden,
            encoder_hidden_states=raw["backbone_features"],
            timestep=discrete,
            return_all_hidden_states=True,
            image_mask=raw["image_mask"],
            backbone_attention_mask=raw["backbone_attention_mask"],
            encoder_attention_mask=raw["backbone_attention_mask"],
        )
        predicted = self.action_decoder(output, action["embodiment_id"])[:, -40:]
        loss = (
            (predicted - (action["action"] - noise)).float().square() * action["action_mask"]
        ).sum() / (action["action_mask"].sum() + 1e-6)
        return {
            "loss": loss,
            "action_mask": action["action_mask"],
            "state_features": action["state"],
            "backbone_features": raw["backbone_features"],
        }


def fixture():
    teacher = Core().eval().requires_grad_(False)
    student = Core().eval()
    head = Head(teacher).eval()
    model = SimpleNamespace(action_head=head, device="cpu")
    mask = torch.zeros(1, 40, 13)
    mask[:, :16] = 1
    frame = {
        "raw": {
            "backbone_features": torch.ones(1, 209, 4),
            "backbone_attention_mask": torch.ones(1, 209, dtype=torch.bool),
            "image_mask": torch.ones(1, 209, dtype=torch.bool),
        },
        "action_input": {
            "action": torch.zeros(1, 40, 13),
            "action_mask": mask,
            "state": torch.ones(1, 1, 4),
            "embodiment_id": torch.tensor([10]),
        },
    }
    return model, teacher, student, frame


def test_clean_delegates_match_independent_eager_kd_and_true_receipts(monkeypatch):
    model, teacher, student, frame = fixture()
    head = model.action_head
    monkeypatch.setattr(
        memory_training_support,
        "set_seed",
        lambda seed, **kw: torch.random.default_generator.manual_seed(seed),
    )
    # Existing helper independently captures the same native head invocation.
    head.model = student
    reference = capture_velocity(model, frame, 73, teacher=teacher, replay_teacher=True)
    head.model = teacher
    expected_receipt = support.eager_receipt(reference, frame, 73)
    original_decoder = head.action_decoder
    original_encoder = head.action_encoder

    def no_hooks(*a, **k):
        raise AssertionError("Clean step cannot install module hooks")

    monkeypatch.setattr(nn.Module, "register_forward_hook", no_hooks)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", no_hooks)
    torch.random.default_generator.manual_seed(73)
    values = support.clean_graph_kd(model, teacher, student, frame)
    assert torch.equal(values["student_velocity"], reference["velocity"])
    assert torch.equal(values["teacher_velocity"], reference["teacher_velocity"])
    assert torch.equal(values["kd"], reference["kd"])
    assert support.evidence_receipt(values, 73) == expected_receipt
    assert (
        head.model is teacher
        and head.action_decoder is original_decoder
        and head.action_encoder is original_encoder
    )
    assert "sample_time" not in head.__dict__
    values["kd"].backward()
    assert student.linear.weight.grad is not None and torch.count_nonzero(
        student.linear.weight.grad
    )
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in original_decoder.parameters())
    assert head.time_calls == 2
    assert all(torch.equal(a, b) for a, b in zip(student.calls[-1], teacher.calls[-1]))


def test_native_bindings_restore_if_student_forward_fails():
    model, teacher, student, frame = fixture()
    head = model.action_head
    encoder, decoder = head.action_encoder, head.action_decoder

    def fail(**kwargs):
        raise RuntimeError("intentional student failure")

    student.forward = fail
    with pytest.raises(RuntimeError, match="intentional"):
        support.clean_graph_kd(model, teacher, student, frame)
    assert (
        head.model is teacher and head.action_encoder is encoder and head.action_decoder is decoder
    )
    assert "sample_time" not in head.__dict__


def test_teacher_runs_only_after_student_proxies_restore():
    model, teacher, student, frame = fixture()
    head = model.action_head
    decoder, encoder = head.action_decoder, head.action_encoder
    original = teacher.forward

    def checked(**kwargs):
        assert (
            head.model is teacher
            and head.action_decoder is decoder
            and head.action_encoder is encoder
        )
        assert "sample_time" not in head.__dict__
        return original(**kwargs)

    teacher.forward = checked
    values = support.clean_graph_kd(model, teacher, student, frame)
    assert values["student_velocity"].requires_grad and not values["teacher_velocity"].requires_grad


def test_storage_can_remain_stable_while_optimizer_style_values_change():
    core = Core()
    before = storage_identity(core)
    content = proto.content_hash(core.state_dict())
    with torch.no_grad():
        core.linear.weight.add_(0.001)
    assert storage_identity(core) == before
    assert proto.content_hash(core.state_dict()) != content


def test_configured_dropout_is_preserved_but_eval_is_required():
    core = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2)).eval().requires_grad_(True)
    inventory = support.dropout_inventory(core)
    assert inventory["1"] == {"p": 0.2, "training": False, "effective_dropout": 0.0}
    core.train()
    with pytest.raises(ValueError, match="eval"):
        support.dropout_inventory(core)


def test_overfit_requires_matching_three_update_source_plan_gate(tmp_path):
    path = tmp_path / "RESULT.json"
    plan = {"compatibility_result": str(path)}
    with pytest.raises(FileNotFoundError):
        runner.compatibility_required(plan, "head")
    value = {
        "status": "passed",
        "continuation_gate": True,
        "plan_sha256": runner.PLAN_SHA,
        "source_commit": "head",
        "optimizer_steps": 3,
    }
    path.write_text(json.dumps(value))
    assert runner.compatibility_required(plan, "head")["path"] == str(path)
    with pytest.raises(ValueError):
        runner.compatibility_required(plan, "different-head")
    value["optimizer_steps"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.compatibility_required(plan, "head")


def test_step_count_survives_failure_before_evidence_logging(monkeypatch, tmp_path):
    context = SimpleNamespace(optimizer_steps_completed=0, source_commit="test-source")
    optimizer = SimpleNamespace(step=lambda: None)
    monkeypatch.setattr(runner, "context_for", lambda *args: context)

    def failure(ctx):
        runner.apply_optimizer_step(ctx, optimizer)
        raise RuntimeError("after step before logging")

    monkeypatch.setattr(runner, "compatibility", failure)
    output = tmp_path / "compatibility-r1/output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "compatibility",
            "--plan",
            str(tmp_path / "PLAN.json"),
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(output),
            "--wall-seconds",
            "850",
        ],
    )
    with pytest.raises(RuntimeError, match="after step"):
        runner.main()
    saved = json.loads((output / "RESULT.json").read_text())
    assert saved["optimizer_steps_completed"] == saved["optimizer_steps"] == 1
    assert not saved["optimizer_resume_authorized"]
