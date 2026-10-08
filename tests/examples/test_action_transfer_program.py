"""CPU tests of long-run continuation, immutable resume and sampler observers."""

from contextlib import nullcontext
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_program_runner as runner  # noqa: E402
import action_transfer_program_support as support  # noqa: E402
import action_transfer_program_trajectory as trajectory  # noqa: E402
from gr00t.model.modules.budgeted_action_mamba import (  # noqa: E402
    ActionCondition,
    ConditionedActionRequest,
)


def test_quality_failure_is_report_only():
    result = runner.quality_report(
        {"metrics": {"teacher_velocity_kd": 1.0, "teacher_action_physical_mae": 1.0}},
        {"metrics": {"teacher_velocity_kd": 2.0, "teacher_action_physical_mae": 2.0}},
        {
            "kd_max_final_initial_ratio": 0.7,
            "teacher_physical_action_mae_max_final_initial_ratio": 0.9,
        },
    )
    assert (
        not result["kd_reduction_30pct"]
        and not result["teacher_physical_action_mae_reduction_10pct"]
    )
    assert "do not terminate" in result["scope"]


@pytest.fixture
def fake_cuda_rng(monkeypatch):
    states = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: [value.clone() for value in states]
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda values: states.__setitem__(slice(None), [value.clone() for value in values]),
    )
    return states


def step(core, optimizer, lr):
    optimizer.param_groups[0]["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    value = torch.randn(3, 2) + random.random() + np.random.rand()
    loss = core(value).square().mean()
    loss.backward()
    optimizer.step()


def test_safe_checkpoint_resume_keeps_global_step_lr_optimizer_and_rng(tmp_path, fake_cuda_rng):
    random.seed(42)
    np.random.seed(42)
    torch.random.default_generator.manual_seed(42)
    core = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(core.parameters(), lr=0.01)
    step(core, optimizer, 0.01)
    step(core, optimizer, 0.02)
    plan = {
        "resume_checkpoint_steps": [2],
        "schedule_sha256": {"overfit": "schedule"},
        "cache_report_sha256": "cache",
        "training_steps": 5,
    }
    args = SimpleNamespace(output=tmp_path, run_id="original")
    context = SimpleNamespace(
        args=args, student=core, source_commit="source", plan=plan, base_before="base"
    )
    saved = support.checkpoint(context, optimizer, 2, {}, {})
    index = support.read(saved["path"])
    assert index["completed_step"] == 2 and index["next_step"] == 3
    step(core, optimizer, 0.003)
    expected = {name: value.detach().clone() for name, value in core.state_dict().items()}
    resumed = nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=123)
    resume_args = SimpleNamespace(
        resume_index=Path(saved["path"]), resume_index_sha256=saved["sha256"], run_id="resume-1"
    )
    loaded, state = support.load_resume(resume_args, plan, "source")
    rng = support.restore_training(resumed, resumed_optimizer, loaded, state)
    assert resumed_optimizer.param_groups[0]["lr"] == 0.02
    # Simulate model/capture initialization consuming RNG before its final restore.
    random.seed(999)
    np.random.seed(999)
    torch.random.default_generator.manual_seed(999)
    support.restore_rng(rng)
    step(resumed, resumed_optimizer, 0.003)
    assert resumed_optimizer.param_groups[0]["lr"] == 0.003
    assert all(torch.equal(value, expected[name]) for name, value in resumed.state_dict().items())
    assert all(int(value["step"]) == 3 for value in resumed_optimizer.state.values())
    with pytest.raises(FileExistsError):
        support.checkpoint(context, optimizer, 2, {}, {})
    resume_args.resume_index_sha256 = "bad"
    with pytest.raises(ValueError, match="SHA"):
        support.load_resume(resume_args, plan, "source")
    resume_args.resume_index_sha256 = saved["sha256"]
    with pytest.raises(ValueError, match="source/PLAN"):
        support.load_resume(resume_args, plan, "different-source")
    bad_optimizer = torch.optim.AdamW(list(reversed(list(resumed.parameters()))), lr=0.01)
    with pytest.raises(ValueError, match="ordering"):
        support.restore_training(resumed, bad_optimizer, loaded, state)


def test_resume_attempt_requires_parent_and_new_identity(tmp_path):
    plan = {"initial_run_id": "long-kd-r1"}
    args = SimpleNamespace(
        run_id="long-kd-r1",
        artifact_root=tmp_path,
        output=tmp_path / "long-kd-r1/output",
        resume_index=None,
        resume_index_sha256=None,
    )
    support.validate_run_id(args, plan)
    args.resume_index = tmp_path / "index.json"
    with pytest.raises(ValueError, match="together"):
        support.validate_run_id(args, plan)
    args.resume_index_sha256 = "hash"
    with pytest.raises(ValueError, match="new immutable"):
        support.validate_run_id(args, plan)


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def encode_condition(self, features):
        return ActionCondition(features.mean(1), features.mean(1))

    def forward(
        self, hidden_states, encoder_hidden_states, timestep, condition_cache=None, **kwargs
    ):
        value = hidden_states + encoder_hidden_states.mean(1)[:, None]
        if condition_cache is not None:
            value = value + condition_cache.modulation[:, None]
        return self.linear(value)


class Encoder(nn.Linear):
    def forward(self, value, time, emb):
        return super().forward(value) + time[:, None, None] * 0.001


class Decoder(nn.Linear):
    def forward(self, value, emb):
        return super().forward(value)


def fixture(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(trajectory, "set_seed", lambda *args, **kwargs: None)
    teacher = TinyCore().eval().requires_grad_(False)
    student = TinyCore().eval()
    head = nn.Module()
    head.model = teacher
    head.action_encoder = Encoder(13, 4)
    head.action_decoder = Decoder(4, 13)
    model = SimpleNamespace(action_head=head, device="cpu")
    features = torch.ones(1, 209, 4)
    state = torch.ones(1, 1, 4)
    emb = torch.tensor([10])
    packet = {
        "features": SimpleNamespace(backbone_features=features, state_features=state),
        "action": SimpleNamespace(embodiment_id=emb),
    }

    def sampler(model, packet):
        # A test-only producer supplies four states; production delegates the
        # real native sampler and contains no update/integration formula.
        output = None
        for i in range(4):
            x = torch.full((1, 40, 13), float(i) / 4)
            time = torch.tensor([i * 250])
            hidden = torch.cat((state, model.action_head.action_encoder(x, time, emb)), 1)
            core = model.action_head.model(
                hidden_states=hidden, encoder_hidden_states=features, timestep=time
            )
            output = model.action_head.action_decoder(core, emb)[:, -40:]
        return output

    def baseline(model, frame, seed, student=None):
        original = head.model
        try:
            if student is not None:
                head.model = ConditionedActionRequest(student, student.encode_condition(features))
            return sampler(model, packet).detach().cpu()
        finally:
            head.model = original

    monkeypatch.setattr(trajectory, "cached_action", baseline)
    monkeypatch.setattr(trajectory, "sampling_packet", lambda *args: packet)
    monkeypatch.setattr(trajectory.proto, "sample_packet", sampler)
    mask = torch.zeros(1, 40, 13)
    mask[:, :16] = 1
    return model, teacher, student, {"action_input": {"action_mask": mask}}


@pytest.mark.parametrize("driver", ["teacher", "student"])
def test_trajectory_captures_full40_shared_state_queries_and_native_parity(monkeypatch, driver):
    model, teacher, student, frame = fixture(monkeypatch)
    encoder, decoder = model.action_head.action_encoder, model.action_head.action_decoder
    result, values = trajectory.capture_shared_states(model, student, frame, 42, driver)
    assert result["final_action_exact"] and len(values["steps"]) == 4
    assert [row["time_bucket"] for row in result["steps"]] == [0, 250, 500, 750]
    for index, row in enumerate(values["steps"]):
        assert row["x_t"].shape == (1, 40, 13)
        assert torch.equal(row["x_t"], torch.full((1, 40, 13), float(index) / 4))
        assert row["teacher_velocity"].shape == row["student_velocity"].shape == (1, 40, 13)
    assert (
        model.action_head.model is teacher
        and model.action_head.action_encoder is encoder
        and model.action_head.action_decoder is decoder
    )


def test_trajectory_proxies_restore_after_failure(monkeypatch):
    model, teacher, student, frame = fixture(monkeypatch)
    encoder, decoder = model.action_head.action_encoder, model.action_head.action_decoder
    monkeypatch.setattr(trajectory, "cached_action", lambda *args, **kwargs: torch.zeros(1, 40, 13))

    def fail(**kwargs):
        raise RuntimeError("inside delegated sampler")

    student.forward = fail
    with pytest.raises(RuntimeError, match="delegated"):
        trajectory.capture_shared_states(model, student, frame, 42, "student")
    assert (
        model.action_head.model is teacher
        and model.action_head.action_encoder is encoder
        and model.action_head.action_decoder is decoder
    )
