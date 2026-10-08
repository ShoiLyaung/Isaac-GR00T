"""CPU checks of real batch binding, objective controls and immutable phase resume."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_matched_support as support  # noqa: E402
from action_transfer_program_support import optimizer_names, rng_state  # noqa: E402
from budgeted_action_cudagraph import TENSOR_KEYS  # noqa: E402
from contract import sha  # noqa: E402
import memory_prototype as proto  # noqa: E402
from memory_training_support import frame_hash  # noqa: E402


def frame(episode, anchor=0):
    mask = torch.zeros(1, 40, 13)
    mask[:, :16] = 1
    value = {
        "episode": episode,
        "anchor": anchor,
        "raw": {
            "backbone_features": torch.full((1, 209, 4), float(episode)),
            "backbone_attention_mask": torch.ones(1, 209, dtype=torch.bool),
            "image_mask": torch.ones(1, 209, dtype=torch.bool),
        },
        "post_vl_features": torch.full((1, 209, 4), -999.0),
        "action_input": {
            "action": torch.ones(1, 40, 13) * episode,
            "action_mask": mask,
            "state": torch.ones(1, 1, 4),
            "embodiment_id": torch.tensor([10]),
        },
        "physical_states": {"q": torch.zeros(1, 13)},
        "physical_actions": {"q": torch.zeros(16, 13)},
    }
    value["content_sha256"] = frame_hash(value)
    return value


def batch():
    frames = {i: frame(i) for i in range(4)}
    cache = SimpleNamespace(frame=lambda ep, anchor: frames[ep])
    return support.assemble_batch(cache, [{"episode": i, "anchor": 0} for i in range(4)]), frames


def test_batch_uses_raw_and_never_substitutes_stored_b1_post():
    value, originals = batch()
    assert value["raw"]["backbone_features"].shape == (4, 209, 4)
    assert "post_vl_features" not in value
    value["raw"]["backbone_features"].zero_()
    value["action_input"]["action"].add_(2)
    assert all(frame_hash(v) == v["content_sha256"] for v in originals.values())
    cache = SimpleNamespace(frame=lambda ep, anchor: originals[ep])
    with pytest.raises(ValueError, match="distinct"):
        support.assemble_batch(cache, [{"episode": 0, "anchor": 0}] * 4)


def test_batched_kd_active_mask_denominator_and_detached_teacher():
    a = torch.randn(4, 40, 13, requires_grad=True)
    b = torch.randn(4, 40, 13, requires_grad=True)
    mask = torch.zeros_like(a)
    mask[:, :16] = 1
    with torch.no_grad():
        a[:, 16:] = float("nan")
        b[:, 16:] = float("inf")
    loss = support.masked_batch_kd(a, b, mask)
    expected = (a[:, :16].float() - b[:, :16].detach().float()).square().sum() / (mask.sum() + 1e-6)
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    loss.backward()
    assert b.grad is None and torch.isfinite(a.grad).all()
    assert torch.count_nonzero(a.grad[:, 16:]) == 0
    assert torch.count_nonzero(a.grad[:, :16]) > 0
    with pytest.raises(ValueError, match="Bx40x13"):
        support.masked_batch_kd(a[:1], b, mask)


class Surface(nn.Module):
    student = True
    backward_surface = True

    def forward(self, *args):
        return args[0] * 2


def test_dispatcher_rejects_broadcast_dtype_and_grad_changes_before_backend():
    values = (
        torch.ones(4, 41, 4),
        torch.ones(4, 209, 4),
        torch.zeros(4, dtype=torch.long),
        torch.ones(4, 209, dtype=torch.bool),
        torch.ones(4, 209, dtype=torch.bool),
    )
    calls = []
    route = support.StrictBatchDispatcher(
        Surface(), lambda *args: calls.append(1) or args[0], values
    )
    kwargs = dict(zip(TENSOR_KEYS, values))
    assert torch.equal(route(**kwargs), values[0]) and calls == [1]
    for replacement in (values[0][:1], values[0].double(), values[0].clone().requires_grad_()):
        with pytest.raises(ValueError, match="signature"):
            route(**{**kwargs, "hidden_states": replacement})
    assert calls == [1]


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.inputs = None

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        self.inputs = (
            hidden_states.detach().clone(),
            encoder_hidden_states.detach().clone(),
            timestep.detach().clone(),
        )
        output = self.linear(hidden_states + encoder_hidden_states.mean(1)[:, None])
        return (output, [output]) if kwargs.get("return_all_hidden_states") else output


class Encoder(nn.Linear):
    def forward(self, x, time, emb):
        return super().forward(x) + time[:, None, None] * 0.001


class Decoder(nn.Linear):
    def forward(self, x, emb):
        return super().forward(x)


class Head(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.model = core
        self.action_encoder = Encoder(13, 4).requires_grad_(False)
        self.action_decoder = Decoder(4, 13).requires_grad_(False)

    def sample_time(self, size):
        return torch.rand(size)

    def forward(self, raw, action):
        noise = torch.randn_like(action["action"])
        time = self.sample_time(len(noise))
        discrete = (time * 1000).long()
        noisy = time[:, None, None] * action["action"] + (1 - time[:, None, None]) * noise
        # Actual native postprocessing can differ from stored B1 post features.
        post = raw["backbone_features"] * 1.5 + len(noise)
        output, _ = self.model(
            hidden_states=torch.cat(
                (action["state"], self.action_encoder(noisy, discrete, action["embodiment_id"])), 1
            ),
            encoder_hidden_states=post,
            timestep=discrete,
            return_all_hidden_states=True,
            image_mask=raw["image_mask"],
            backbone_attention_mask=raw["backbone_attention_mask"],
        )
        velocity = self.action_decoder(output, action["embodiment_id"])[:, -40:]
        loss = ((velocity - (action["action"] - noise)).square() * action["action_mask"]).sum() / (
            action["action_mask"].sum() + 1e-6
        )
        return {
            "loss": loss,
            "action_mask": action["action_mask"],
            "state_features": action["state"],
            "backbone_features": post,
        }


def test_b4_teacher_actual_inputs_gradients_and_proxy_restoration():
    teacher, student = Core().eval().requires_grad_(False), Core().eval()
    head = Head(teacher)
    model = SimpleNamespace(action_head=head, device="cpu")
    inputs, _ = batch()
    encoder, decoder = head.action_encoder, head.action_decoder
    original = teacher.forward

    def replay(**kwargs):
        assert (
            head.model is teacher
            and head.action_decoder is decoder
            and head.action_encoder is encoder
        )
        assert "sample_time" not in head.__dict__
        return original(**kwargs)

    teacher.forward = replay
    values = support.batch_forward(model, teacher, student, inputs)
    assert values["student_velocity"].shape == (4, 40, 13)
    assert torch.equal(student.inputs[1], inputs["raw"]["backbone_features"] * 1.5 + 4)
    assert all(torch.equal(a, b) for a, b in zip(student.inputs, teacher.inputs))
    values["kd"].backward()
    assert torch.count_nonzero(student.linear.weight.grad) and all(
        p.grad is None for p in teacher.parameters()
    )
    assert all(p.grad is None for p in decoder.parameters())
    assert head.model is teacher and head.action_encoder is encoder

    def fail(**kwargs):
        raise RuntimeError("deliberate B4 failure")

    student.forward = fail
    with pytest.raises(RuntimeError, match="deliberate"):
        support.batch_forward(model, teacher, student, inputs)
    assert (
        head.model is teacher and head.action_decoder is decoder and head.action_encoder is encoder
    )
    assert "sample_time" not in head.__dict__


def test_cache_requires_report_and_frame_identity_and_excludes_other_episodes(tmp_path):
    value = frame(1)
    path = tmp_path / "episode-1.pt"
    torch.save({0: value}, path)
    report = {
        "status": "passed",
        "cohort": "train_dev",
        "plan_sha256": "frozen-cache",
        "episodes": [1],
        "files": [
            {"episode": 1, "file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
        ],
    }
    report_path = tmp_path / "CACHE_RESULT.json"
    report_path.write_text(json.dumps(report))
    plan = {"cache_plan_sha256": "frozen-cache", "train": [1], "dev": [], "grids": {"1": [0]}}
    with pytest.raises(ValueError, match="SHA"):
        support.MMapCache(tmp_path, plan, "wrong")
    cache = support.MMapCache(tmp_path, plan, sha(report_path))
    assert cache.frame(1, 0)["content_sha256"] == value["content_sha256"]
    with pytest.raises(ValueError, match="outside"):
        cache.frame(2, 0)
    cache.assert_readonly()
    cache.frame(1, 0)["raw"]["backbone_features"].add_(1)
    with pytest.raises(ValueError, match="modified"):
        cache.assert_readonly()
    # mmap is private: an accidental in-memory change never edits the frozen file.
    assert sha(path) == report["files"][0]["sha256"]


def test_matched_phase_objective_boundaries():
    assert support.objective_for("distill_then_task", 10000) == "kd"
    assert support.objective_for("distill_then_task", 10001) == "task"
    assert (
        support.objective_for("task_only", 1) == support.objective_for("task_only", 20000) == "task"
    )
    assert support.phase_for_step(10000) == 1 and support.phase_for_step(10001) == 2
    with pytest.raises(ValueError):
        support.phase_for_step(20001)


@pytest.mark.parametrize("boundary", [10000, 15000])
def test_resume_phase_boundary_fresh_or_restored_optimizer(tmp_path, monkeypatch, boundary):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    core = nn.Linear(2, 1)
    opt = torch.optim.AdamW(core.parameters(), lr=0.001)
    core(torch.ones(1, 2)).square().mean().backward()
    opt.step()
    weights = tmp_path / "weights.pt"
    torch.save(core.state_dict(), weights)
    payload = {
        "optimizer_parameter_names": optimizer_names(core, opt),
        "optimizer": opt.state_dict(),
        "rng": rng_state(),
    }
    index = {
        "weights": {"path": str(weights)},
        "student_content_sha256": proto.content_hash(core.state_dict()),
        "next_phase_reinitialize_optimizer": boundary == 10000,
    }
    restored = nn.Linear(2, 1)
    fresh = torch.optim.AdamW(restored.parameters(), lr=0.123)
    support.restore_checkpoint(restored, fresh, index, payload)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in core.state_dict().items())
    if boundary == 10000:
        assert not fresh.state and fresh.param_groups[0]["lr"] == 0.123
    else:
        assert fresh.state and fresh.param_groups[0]["lr"] == 0.001
    bad = torch.optim.AdamW(list(reversed(list(restored.parameters()))), lr=0.001)
    with pytest.raises(ValueError, match="order"):
        support.restore_checkpoint(restored, bad, index, payload)


def test_checkpoint_log_prefix_tracks_explicit_parents_without_mutable_latest(tmp_path):
    first = tmp_path / "first.jsonl"
    first.write_text('{"step":1}\n{"step":2}\n')
    prefix = support.log_prefix(first)
    first.write_text(first.read_text() + '{"step":3}\n')
    identity = {"condition": "task_only", "source_commit": "head"}

    def index(name, data):
        path = tmp_path / name
        path.write_text(json.dumps({**identity, **data}))
        return {"path": str(path), "sha256": sha(path)}

    parent = index(
        "first-index.json",
        {"parent_index": None, "starting_step": 0, "completed_step": 2, "training_log": prefix},
    )
    second = tmp_path / "second.jsonl"
    second.write_text('{"step":3}\n{"step":4}\n')
    child = index(
        "second-index.json",
        {
            "parent_index": parent,
            "starting_step": 2,
            "completed_step": 4,
            "training_log": support.log_prefix(second),
        },
    )
    assert [r["step"] for r in support.checkpoint_log_rows(child, identity)] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="identity"):
        support.checkpoint_log_rows(child, {**identity, "source_commit": "other"})
    first.write_text('{"step":9}\n{"step":2}\n')
    with pytest.raises(ValueError, match="prefix"):
        support.checkpoint_log_rows(child, identity)


def test_checkpoint_requires_explicit_parent_hash_before_loading(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{}")
    args = SimpleNamespace(resume_index=path, resume_index_sha256="bad")
    with pytest.raises(ValueError, match="SHA"):
        support.read_resume(args, {}, "head")


def test_shared_dev512_baseline_is_before_graph_and_any_update(tmp_path, monkeypatch):
    import action_transfer_matched_runner as runner

    plan = {
        "dev": list(range(8)),
        "eval_anchors": {str(i): list(range(0, 64, 4)) for i in range(8)},
        "noise_seeds": [42, 43, 44, 45],
    }
    core = nn.Linear(2, 1)
    plan["student_initialization"] = {"content_sha256": proto.content_hash(core.state_dict())}
    calls = []
    output = tmp_path / "dev-step0"

    def evaluate(ctx, pairs, path):
        assert ctx.optimizer_steps_completed == 0 and ctx.lease is None
        assert len(pairs) == 512 and path == output
        calls.append(1)
        path.mkdir()
        (path / "details.json").write_text("[]")
        value = {
            "status": "passed",
            "rows": 512,
            "checkpoint_content_sha256": plan["student_initialization"]["content_sha256"],
            "cache_report_sha256": "completed-cache",
            "array_sha256": {},
            "details_sha256": sha(path / "details.json"),
        }
        result_path = path / "RESULT.json"
        result_path.write_text(json.dumps(value))
        return {"path": str(result_path), "sha256": sha(result_path), "result": value}

    monkeypatch.setattr(runner, "evaluate_pairs", evaluate)
    ctx = SimpleNamespace(
        student=core,
        plan=plan,
        optimizer_steps_completed=0,
        lease=None,
        args=SimpleNamespace(output=tmp_path),
    )
    record = runner.initial_dev_baseline(ctx)
    # Both fresh conditions resolve the same immutable preflight reference.
    a = runner.validate_initial_dev(record, plan, "completed-cache")
    b = runner.validate_initial_dev(record, plan, "completed-cache")
    assert a["sha256"] == b["sha256"] and calls == [1]
    ctx.optimizer_steps_completed = 3
    with pytest.raises(ValueError, match="precede"):
        runner.initial_dev_baseline(ctx)
    assert calls == [1]
    with pytest.raises(ValueError, match="fresh initialization"):
        runner.validate_initial_dev(record, plan, "different-cache")
