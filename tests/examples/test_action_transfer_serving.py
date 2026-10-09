"""CPU serving-contract tests; neural loopback remains a separately authorized run."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_policy as policy_module  # noqa: E402
import action_transfer_serving_probe as probe  # noqa: E402
from contract import CAMERAS, sha  # noqa: E402
from gr00t.policy.server_client import MsgSerializer  # noqa: E402
import serve_action_transfer as serve  # noqa: E402
from yuqi_wire import install_legacy_yuqi_response_encoding  # noqa: E402


TASK = "insert the shaft into the housing"


class Processor:
    def __init__(self):
        def modality(keys, delta):
            return SimpleNamespace(modality_keys=list(keys), delta_indices=delta)

        self.modalities = {
            "video": modality(CAMERAS, [0]),
            "state": modality(["left_arm", "left_hand"], [0]),
            "language": modality([policy_module.LANGUAGE_KEY], [0]),
            "action": modality(["left_arm", "left_hand"], list(range(16))),
        }
        self.seen = []
        self.training = True

    def eval(self):
        self.training = False
        # Match Gr00tN1d7Processor.eval(), which returns None, unlike nn.Module.
        return None

    def get_modality_configs(self):
        return {"new_embodiment": self.modalities}

    def __call__(self, messages):
        raw = messages[0]["content"]
        self.seen.append(raw)
        return {
            "state": torch.from_numpy(np.concatenate(list(raw.states.values()), -1))[None],
            "input_ids": torch.tensor([[1, 2]]),
        }

    def collator(self, items):
        assert len(items) == 1
        return {"inputs": items[0]}

    def decode_action(self, values, embodiment, states):
        # Deliberately return views: policy must own its output and never add q.
        return {"left_arm": values[:, :16, :7], "left_hand": values[:, :16, 7:]}


def fixture(monkeypatch, mode="native_eager"):
    monkeypatch.setattr(
        policy_module, "validate_policy", lambda *a: {"returned_action_horizon": 16}
    )
    monkeypatch.setattr(policy_module.proto, "assert_frozen", lambda *a: None)
    model = SimpleNamespace(action_head=SimpleNamespace(model=nn.Identity()))
    student = nn.Linear(2, 2).float().eval()
    processor = Processor()
    policy = policy_module.ActionTransferPolicy(
        model, processor, mode, student=student, task_text=TASK
    )
    observation = {
        "video": {k: np.full((1, 1, 2, 3, 3), i, np.uint8) for i, k in enumerate(CAMERAS)},
        "state": {
            "left_arm": np.full((1, 1, 7), 100, np.float32),
            "left_hand": np.full((1, 1, 6), 200, np.float32),
        },
        "language": {policy_module.LANGUAGE_KEY: [[TASK]]},
    }
    return policy, observation


def test_native_observation_contract_and_exact_lowercase_language(monkeypatch):
    policy, obs = fixture(monkeypatch)
    assert isinstance(policy.processor, Processor) and not policy.processor.training
    inputs, states = policy.prepare_observation(obs)
    assert inputs["state"].shape == (1, 1, 13) and inputs["state"].dtype == torch.bfloat16
    assert policy.processor.seen[-1].text == TASK
    assert states["left_arm"].shape == (1, 1, 7)
    obs["language"][policy_module.LANGUAGE_KEY] = [[TASK.upper()]]
    with pytest.raises(ValueError, match="task text"):
        policy.prepare_observation(obs)
    obs["language"][policy_module.LANGUAGE_KEY] = [[TASK]]
    obs["state"]["left_arm"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy.check_observation(obs)


def test_absolute_h16_cpu_outputs_survive_next_request(monkeypatch):
    policy, obs = fixture(monkeypatch)
    buffer = np.zeros((1, 40, 13), np.float32)
    count = []

    def predict(inputs):
        count.append(1)
        buffer.fill(len(count) / 10)
        return buffer

    monkeypatch.setattr(policy, "predict_normalized", predict)
    first, _ = policy.get_action(obs)
    second, _ = policy.get_action(obs)
    assert np.all(first["left_arm"] == np.float32(0.1)) and np.all(
        second["left_arm"] == np.float32(0.2)
    )
    assert first["left_arm"].shape == (1, 16, 7) and first["left_hand"].shape == (1, 16, 6)
    assert not np.shares_memory(first["left_arm"], buffer)
    assert policy.requests_completed == 2
    # Large input q is not added to absolute normalized-decoded action.
    assert np.max(second["left_hand"]) == np.float32(0.2)


def test_metadata_and_reset_do_not_draw_or_reseed_rng(monkeypatch):
    policy, _ = fixture(monkeypatch)
    torch.manual_seed(73)
    before = torch.get_rng_state().clone()
    assert policy.reset({"seed": 999}) == {}
    policy.get_modality_config()
    assert torch.equal(torch.get_rng_state(), before) and policy.requests_completed == 0


def test_concurrent_requests_rejected_and_lock_released_after_error(monkeypatch):
    policy, obs = fixture(monkeypatch)
    policy.request_lock.acquire()
    with pytest.raises(RuntimeError, match="Concurrent"):
        policy.get_action(obs)
    policy.request_lock.release()

    def fail(*args):
        raise RuntimeError("during model")

    monkeypatch.setattr(policy, "predict_normalized", fail)
    with pytest.raises(RuntimeError, match="during model"):
        policy.get_action(obs)
    assert policy.request_lock.acquire(blocking=False)
    policy.request_lock.release()


def test_graph_request_binding_restores_native_core_on_failure(monkeypatch):
    policy, _ = fixture(monkeypatch, "native_graph")
    original = policy.model.action_head.model
    route = nn.Identity()
    policy.route, policy.lease = route, SimpleNamespace()
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())

    def fail(model, inputs, student):
        assert model.action_head.model is route and student is None
        raise RuntimeError("graph failure")

    monkeypatch.setattr(policy_module, "current_policy", fail)
    with pytest.raises(RuntimeError, match="graph failure"):
        policy.predict_normalized({})
    assert policy.model.action_head.model is original


def test_fixture_npz_is_safe_and_preserves_a_b_a(tmp_path, monkeypatch):
    _, a = fixture(monkeypatch)
    _, b = fixture(monkeypatch)
    b["state"]["left_arm"] += 1
    path = tmp_path / "fixtures.npz"
    result = probe.save_fixtures(path, [a, b], [0, 1, 0])
    loaded = np.load(path, allow_pickle=False)
    assert loaded["order"].tolist() == [0, 1, 0]
    assert loaded["0.text"].item() == TASK and result["sha256"] == sha(path)
    assert np.array_equal(loaded["1.state.left_arm"], b["state"]["left_arm"])
    compile(probe.OLD_CLIENT, "real-old-client", "exec")


def test_legacy_response_install_keeps_safe_decoder_and_rejects_pickle(monkeypatch):
    decoder = MsgSerializer.from_bytes
    original = MsgSerializer.to_bytes
    try:
        install_legacy_yuqi_response_encoding()
        assert MsgSerializer.from_bytes is decoder
        assert probe.rejection_checks() == {"object_encode": True, "pickle_decode": True}
    finally:
        monkeypatch.setattr(MsgSerializer, "to_bytes", staticmethod(original))


def test_serving_plan_requires_explicit_sha_source_and_loopback(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test identity only")
    plan = {
        "source_worktree": str(serve.REPO),
        "host": "127.0.0.1",
        "records": [],
        "immutable_source_relative_paths": [],
        "student_checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
    }
    path = tmp_path / "PLAN.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="SHA"):
        serve.verify_plan(path, "incorrect")
    assert serve.verify_plan(path, sha(path))["host"] == "127.0.0.1"
    plan["host"] = "0.0.0.0"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="loopback"):
        serve.verify_plan(path, sha(path))


def test_old_client_and_new_server_same_relative_name_use_exact_identity(tmp_path, monkeypatch):
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    relative = "gr00t/policy/server_client.py"
    paths = [new_root / relative, old_root / relative]
    for path, content in zip(paths, ("safe-new-server", "legacy-old-client")):
        path.parent.mkdir(parents=True)
        path.write_text(content)
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"explicit candidate")
    plan = {
        "source_worktree": str(new_root),
        "host": "127.0.0.1",
        "records": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths],
        "immutable_source_relative_paths": [relative],
        "student_checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
    }
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(serve, "REPO", new_root)
    assert serve.verify_plan(plan_path, sha(plan_path))["source_worktree"] == str(new_root)


def test_legacy_dependency_is_appended_and_bound_to_frozen_version_origin(tmp_path):
    dependency = tmp_path / "legacy-site-packages"
    origin = dependency / "filelock/__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text('__version__ = "3.13.1"\n')
    old = tmp_path / "old-client"
    plan = {
        "client_pythonpath": str(old),
        "client_dependency_path": str(dependency),
        "client_filelock_version": "3.13.1",
        "records": [{"path": str(origin), "sha256": sha(origin)}],
    }
    result = {
        "old_client_source": str(old / "gr00t/policy/server_client.py"),
        "dependency_path": str(dependency),
        "filelock_origin": str(origin),
        "filelock_version": "3.13.1",
    }
    probe.validate_client_identity(result, plan)
    assert probe.OLD_CLIENT.index("sys.path.append(sys.argv[4])") < probe.OLD_CLIENT.index(
        "from gr00t.policy.server_client import PolicyClient"
    )
    assert "sys.path.insert" not in probe.OLD_CLIENT
    with pytest.raises(ValueError, match="version"):
        probe.validate_client_identity({**result, "filelock_version": "other"}, plan)
