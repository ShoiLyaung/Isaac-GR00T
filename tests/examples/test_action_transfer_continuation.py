"""CPU checks for no-reset continuation, exact restored evidence and old/new lineage."""

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_continuation as continuation  # noqa:E402
from action_transfer_matched_support import log_prefix  # noqa:E402
from action_transfer_program_support import cpu_tree, optimizer_names  # noqa:E402
from contract import sha  # noqa:E402


def test_all_30000_learning_rates_match_frozen_arithmetic_without_restart():
    values = []
    for step in range(20001, 50001):
        u = (step - 20001) / 29999
        expected = 1e-6 + 9e-6 * (1 + math.cos(math.pi * u)) / 2
        actual = continuation.continuation_lr(step)
        assert actual == expected
        values.append(actual)
    assert values[0] == 1e-5 and values[-1] == 1e-6
    assert all(a >= b for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        continuation.continuation_lr(20000)


def make_optimizer_state():
    torch.manual_seed(42)
    core = nn.Linear(2, 1)
    opt = torch.optim.AdamW(core.parameters(), lr=1e-5, betas=(0.9, 0.95))
    core(torch.ones(1, 2)).square().sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    for state in opt.state.values():
        state["step"].fill_(10000)
    return core, opt


def test_restored_adamw_preserves_moments_and_first_update_is_10001():
    core, opt = make_optimizer_state()
    payload = {
        "optimizer": cpu_tree(opt.state_dict()),
        "optimizer_parameter_names": optimizer_names(core, opt),
    }
    restored = nn.Linear(2, 1)
    restored.load_state_dict(core.state_dict())
    fresh = torch.optim.AdamW(restored.parameters(), lr=0.1, betas=(0.5, 0.5))
    report = continuation.restore_optimizer(restored, fresh, payload, 10000, expected_count=2)
    assert report == {"states": 2, "min_step": 10000, "max_step": 10000}
    assert fresh.param_groups[0]["lr"] == 1e-5 and fresh.param_groups[0]["betas"] == (0.9, 0.95)
    for a, b in zip(opt.state.values(), fresh.state.values()):
        assert torch.equal(a["exp_avg"], b["exp_avg"]) and torch.equal(
            a["exp_avg_sq"], b["exp_avg_sq"]
        )
    for model, optimizer in ((core, opt), (restored, fresh)):
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
    assert continuation.optimizer_steps(fresh, 2, 10001)["min_step"] == 10001
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in core.state_dict().items())
    with pytest.raises(ValueError, match="count"):
        continuation.optimizer_steps(fresh, 158, 10001)


def test_wrong_optimizer_order_or_phase_step_is_rejected():
    core, opt = make_optimizer_state()
    payload = {
        "optimizer": cpu_tree(opt.state_dict()),
        "optimizer_parameter_names": optimizer_names(core, opt),
    }
    wrong = torch.optim.AdamW(list(reversed(list(core.parameters()))))
    with pytest.raises(ValueError, match="Ordered"):
        continuation.restore_optimizer(core, wrong, payload, 10000, 2)
    with pytest.raises(ValueError, match="step"):
        continuation.optimizer_steps(opt, 2, 0)


def quality(root):
    root.mkdir()
    array = np.arange(512 * 2, dtype=np.float32).reshape(512, 2)
    np.save(root / "student_actions.npy", array)
    details = [
        {
            "episode": i // 64,
            "anchor": i // 4,
            "seed": 42 + i % 4,
            "noise_receipt": {"input_hash": str(i)},
        }
        for i in range(512)
    ]
    (root / "details.json").write_text(json.dumps(details))
    (root / "NORMALIZATION.json").write_text(json.dumps({"mode": "absolute"}))
    value = {
        "rows": 512,
        "metrics": {"flow_loss": 0.1},
        "per_episode": {},
        "teacher_metrics": {},
        "teacher_per_episode": {},
        "checkpoint_content_sha256": "restored20k",
        "cache_report_sha256": "cache",
        "array_sha256": {"student_actions": sha(root / "student_actions.npy")},
        "details_sha256": sha(root / "details.json"),
        "normalization_sha256": sha(root / "NORMALIZATION.json"),
    }
    path = root / "RESULT.json"
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": sha(path)}


def test_full512_reproduction_requires_exact_arrays_metrics_and_input_hashes(tmp_path):
    left, right = quality(tmp_path / "old"), quality(tmp_path / "restored")
    assert continuation.compare_reproduction(left, right)["all_input_receipts_exact"]
    path = Path(right["path"])
    r = json.loads(path.read_text())
    arr = np.load(path.parent / "student_actions.npy")
    arr[0, 0] = 1e-9
    np.save(path.parent / "student_actions.npy", arr)
    r["array_sha256"]["student_actions"] = sha(path.parent / "student_actions.npy")
    path.write_text(json.dumps(r))
    right["sha256"] = sha(path)
    with pytest.raises(ValueError, match="not exact"):
        continuation.compare_reproduction(left, right)


def test_reproduction_then_batch_gate_performs_no_optimizer_update(monkeypatch, tmp_path):
    events = []
    ctx = SimpleNamespace(
        starting_step=20000,
        optimizer_steps_completed=20000,
        plan={"preflight_batch_indices": [0, 1], "restoration_batch_indices": [20000]},
        args=SimpleNamespace(output=tmp_path),
        cache=object(),
    )
    schedule = [None] * 20000 + [{"batch": [], "forward_seed": 42}]
    monkeypatch.setattr(continuation, "assemble_batch", lambda *a: {})
    monkeypatch.setattr(
        continuation.matched, "create_batch_graph", lambda *a: events.append("capture")
    )
    monkeypatch.setattr(continuation.matched, "dev_pairs", lambda *a: list(range(512)))
    monkeypatch.setattr(
        continuation.matched,
        "evaluate_pairs",
        lambda *a: events.append("dev512") or {"ref": "actual"},
    )
    monkeypatch.setattr(
        continuation,
        "compare_reproduction",
        lambda *a: events.append("exact_replay") or {"passed": True},
    )

    def batch(*a):
        assert ctx.plan["preflight_batch_indices"] == [20000]
        events.append("b4_gradients")
        return {"passed": True}

    monkeypatch.setattr(continuation.matched, "batch_checks", batch)
    result = continuation.restoration_gate(
        ctx, {"evaluations": {"dev-20000": {"ref": "old"}}}, schedule
    )
    assert (
        events == ["capture", "dev512", "exact_replay", "b4_gradients"]
        and result["optimizer_updates"] == 0
    )
    assert ctx.plan["preflight_batch_indices"] == [0, 1]


def test_lineage_accepts_only_pinned_old_parent_and_new_suffix(tmp_path):
    old_log = tmp_path / "old.jsonl"
    old_log.write_text('{"step":1}\n{"step":2}\n')
    common = {"condition": "task_only", "cache_report_sha256": "cache"}
    old_identity = {
        **common,
        "plan_sha256": "oldplan",
        "source_commit": "oldsource",
        "schedule_sha256": "oldschedule",
    }
    old = {
        **old_identity,
        "completed_step": 2,
        "starting_step": 0,
        "parent_index": None,
        "training_log": log_prefix(old_log),
    }
    old_path = tmp_path / "old-index.json"
    old_path.write_text(json.dumps(old))
    plan = {
        "cache_report_sha256": "cache",
        "schedule_sha256": "newschedule",
        "parents": {
            "task_only": {
                "index_path": str(old_path),
                "index_sha256": sha(old_path),
                "plan_sha256": "oldplan",
                "source_commit": "oldsource",
                "schedule_sha256": "oldschedule",
            }
        },
    }
    new_log = tmp_path / "new.jsonl"
    new_log.write_text('{"step":3}\n{"step":4}\n')
    new = {
        **common,
        "plan_sha256": "newplan",
        "source_commit": "newsource",
        "schedule_sha256": "newschedule",
        "completed_step": 4,
        "starting_step": 2,
        "parent_index": continuation.parent_record(plan, "task_only"),
        "training_log": log_prefix(new_log),
    }
    new_path = tmp_path / "new-index.json"
    new_path.write_text(json.dumps(new))
    record = {"path": str(new_path), "sha256": sha(new_path)}
    assert [
        r["step"]
        for r in continuation.lineage_rows(record, plan, "task_only", "newplan", "newsource")
    ] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="identity"):
        continuation.lineage_rows(record, plan, "task_only", "wrong-plan", "newsource")
    new_log.write_text('{"step":3}\n{"step":3}\n')
    with pytest.raises(ValueError, match="prefix"):
        continuation.lineage_rows(record, plan, "task_only", "newplan", "newsource")
