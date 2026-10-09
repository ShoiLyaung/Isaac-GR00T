"""CPU checks for bounded evaluation ownership, private runtime and honest denominators."""

from contextlib import nullcontext
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_evaluation as evaluation  # noqa:E402
from contract import sha  # noqa:E402


def evidence(tmp_path, count=3, aborted=False):
    expected = {
        "episodes": 3,
        "seeds": [11, 22, 33],
        "expected_config": {"action_chunk_steps": 16, "expected_env_steps_per_action_chunk": 27},
    }
    rows = [
        {
            "episode_index": i,
            "seed": expected["seeds"][i],
            "success": i != 1,
            "stable_success": i == 0,
            "failure_reason": "success" if i == 0 else "timeout",
        }
        for i in range(count)
    ]
    (tmp_path / "evaluation_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "episodes": count,
                "successes": sum(r["success"] for r in rows),
                "stable_successes": sum(r["stable_success"] for r in rows),
                "aborted": aborted,
                "config": expected["expected_config"],
            }
        )
    )
    return expected


def test_complete_summary_reconciles_episode_seeds_and_distinguishes_pose_stable(tmp_path):
    plan = evidence(tmp_path)
    result = evaluation.summarize_simulation(tmp_path, plan, 0)
    assert result["complete"] and result["completed_episodes"] == 3 and result["not_completed"] == 0
    assert result["successes"] == 2 and result["stable_successes"] == 1
    assert result["stable_success_fraction_of_planned"] == pytest.approx(1 / 3)
    rows = (tmp_path / "evaluation_results.jsonl").read_text().replace('"seed": 22', '"seed": 999')
    (tmp_path / "evaluation_results.jsonl").write_text(rows)
    assert not evaluation.summarize_simulation(tmp_path, plan, 0)["complete"]


def test_exit_zero_cannot_hide_partial_aborted_or_mismatched_summary(tmp_path):
    plan = evidence(tmp_path, count=2, aborted=True)
    result = evaluation.summarize_simulation(tmp_path, plan, 0)
    assert (
        not result["complete"]
        and result["completed_episodes"] == 2
        and result["not_completed"] == 1
    )
    assert result["stable_success_rate_completed"] == 0.5
    assert result["stable_success_fraction_of_planned"] == pytest.approx(1 / 3)
    (tmp_path / "summary.json").unlink()
    with (tmp_path / "evaluation_results.jsonl").open("a") as stream:
        stream.write('{"partial":')
    result = evaluation.summarize_simulation(tmp_path, plan, -15)
    assert (
        result["completed_episodes"] == 2
        and result["planned_episodes"] == 3
        and not result["complete"]
    )


def kit_fixture(tmp_path):
    root = tmp_path / "kit"
    path = root / "data/user.config"
    path.parent.mkdir(parents=True)
    path.write_text("private")
    rows = [
        {"relative_path": "data/user.config", "bytes": path.stat().st_size, "sha256": sha(path)}
    ]
    manifest = tmp_path / "kit-initial.json"
    manifest.write_text(json.dumps(rows))
    return {
        "kit_root": str(root),
        "kit_manifest_path": str(manifest),
        "kit_manifest_sha256": sha(manifest),
    }


def test_private_kit_rejects_extra_files_and_hardlinks(tmp_path):
    spec = kit_fixture(tmp_path)
    assert evaluation.validate_kit(spec, tmp_path)["private_regular_files"]
    extra = Path(spec["kit_root"]) / "log.txt"
    extra.write_text("not copied initially")
    with pytest.raises(ValueError, match="inventory"):
        evaluation.validate_kit(spec, tmp_path)
    extra.unlink()
    original = Path(spec["kit_root"]) / "data/user.config"
    (tmp_path / "aliased.config").hardlink_to(original)
    with pytest.raises(ValueError, match="aliases"):
        evaluation.validate_kit(spec, tmp_path)


def test_child_env_clears_wall_override_without_changing_parent(monkeypatch):
    monkeypatch.setenv("INSERTION_EVAL_ROOM_WALLS_USD", "shared-override")
    spec = {
        "environment": {
            "CUDA_VISIBLE_DEVICES": "1",
            "INSERTION_EVAL_ROOM_WALLS_USD": "still-forbidden",
        }
    }
    result = evaluation.child_environment(spec, 1)
    assert (
        "INSERTION_EVAL_ROOM_WALLS_USD" not in result and result["PYTHONDONTWRITEBYTECODE"] == "1"
    )
    assert evaluation.os.environ["INSERTION_EVAL_ROOM_WALLS_USD"] == "shared-override"


def test_kill_only_own_live_popen_group(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(evaluation.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    proc = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: 0, returncode=0)
    assert evaluation.terminate_owned(proc)["terminated"] and calls == [(123, signal.SIGTERM)]
    calls.clear()
    monkeypatch.setattr(evaluation.os, "getpgid", lambda pid: 999)
    with pytest.raises(RuntimeError, match="not owned"):
        evaluation.terminate_owned(proc)
    assert calls == []
    proc.poll = lambda: 0
    assert not evaluation.terminate_owned(proc)["terminated"]


def test_owned_timeout_escalates_only_same_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(evaluation.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    waits = []

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("owned", timeout)
        return -9

    proc = SimpleNamespace(pid=321, poll=lambda: None, wait=wait, returncode=-9)
    assert evaluation.terminate_owned(proc)["signal"] == "SIGKILL"
    assert calls == [(321, signal.SIGTERM), (321, signal.SIGKILL)]


def test_ready_requires_selected_generator_and_source(tmp_path):
    spec = {"plan_sha256": "plan", "mode": "native_graph", "source_commit": "commit", "port": 62071}
    path = tmp_path / "READY.json"
    value = {
        "status": "ready",
        "plan_sha256": "plan",
        "mode": "native_graph",
        "source": {"commit": "commit"},
        "host": "127.0.0.1",
        "port": 62071,
        "effective_action_backbone": "dit",
    }
    path.write_text(json.dumps(value))
    assert evaluation.validate_ready(path, spec)["sha256"] == sha(path)
    value["effective_action_backbone"] = "budgeted_mamba"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="generator"):
        evaluation.validate_ready(path, spec)


def test_timed_gpu_adapter_keeps_output_on_device_and_restores_native_binding(monkeypatch):
    import action_transfer_policy as policy_module

    original, route = nn.Identity(), nn.Identity()
    model = SimpleNamespace(action_head=SimpleNamespace(model=original))
    policy = policy_module.ActionTransferPolicy.__new__(policy_module.ActionTransferPolicy)
    policy.model, policy.route, policy.student = model, route, nn.Linear(2, 2)
    policy.mode, policy.closed, policy.lease = "native_graph", False, object()
    output = torch.ones(1, 40, 13)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())

    def call(model, inputs, student):
        assert model.action_head.model is route and student is None
        return output

    monkeypatch.setattr(policy_module, "current_policy", call)

    def forbidden(*a, **k):
        raise AssertionError("CPU evidence entered timer")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    assert evaluation.gpu_action(policy, {}, graph=True) is output
    assert model.action_head.model is original


def test_server_exit_before_ready_never_kills_endpoint(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    kit = kit_fixture(tmp_path)
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    server = {
        "argv": ["/frozen/python", "serve.py"],
        "cwd": str(tmp_path),
        "environment": {"CUDA_VISIBLE_DEVICES": "1"},
        "ready_path": str(output / "server/READY.json"),
        "plan_path": str(tmp_path / "server-plan.json"),
        "plan_sha256": "",
        "ready_timeout_seconds": 1,
    }
    Path(server["plan_path"]).write_text("{}")
    server["plan_sha256"] = sha(server["plan_path"])
    simulator = {
        **kit,
        "argv": [
            "/frozen/python",
            "--output-dir",
            str(output / "evaluation"),
            "--kit_args",
            "--portable-root " + kit["kit_root"],
        ],
        "cwd": str(tmp_path),
        "environment": {"CUDA_VISIBLE_DEVICES": "1", "TMPDIR": str(temporary)},
        "output_dir": str(output / "evaluation"),
        "episodes": 2,
        "seeds": [1, 2],
        "expected_config": {},
    }
    proc = SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *a, **k: proc)

    def forbidden(*a, **k):
        raise AssertionError("Endpoint used without owned READY")

    monkeypatch.setattr(evaluation, "metadata_handshake", forbidden)
    args = SimpleNamespace(output=output, artifact_root=tmp_path)
    plan = {
        "server": server,
        "simulator": simulator,
        "physical_gpu": 1,
        "max_artifact_bytes": 1000000,
    }
    result = evaluation.simulate(args, plan, evaluation.time.monotonic() + 10)
    assert result["status"] == "failed" and result["server_ready"] is None
    assert "owned_kill_endpoint" not in result["cleanup"]
    assert result["simulation"]["not_completed"] == 2
