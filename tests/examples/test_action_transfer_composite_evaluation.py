"""CPU-only composite identity, complete-head routing and final pairing regressions."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
sys.path.insert(0, str(ROOT / "tests/examples"))
import action_transfer_checkpoint as checkpoint  # noqa:E402
import action_transfer_final_evaluation as final  # noqa:E402
import action_transfer_final_support as support  # noqa:E402
from action_transfer_periphery_support import NativeStudent  # noqa:E402
import action_transfer_policy as policy_module  # noqa:E402
from budgeted_action_support import current_policy  # noqa:E402
from contract import sha, write_new  # noqa:E402
from gr00t.model.modules.budgeted_action_mamba import ActionCondition  # noqa:E402
import memory_prototype as proto  # noqa:E402
from test_action_transfer_periphery import head_and_frame  # noqa:E402


def record_for(root, state, kind, *, fixture=False, name="candidate"):
    root.mkdir()
    path = root / "student.pt"
    torch.save(state, path)
    record = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
        "kind": kind,
        "name": name,
        "expected_step": 70000,
        "source_commit": "cpu-source",
        "fixture_only": fixture,
        **checkpoint.semantic_identity(state, kind),
    }
    index = {
        "completed_step": 70000,
        "source_commit": "cpu-source",
        "fixture_only": fixture,
        "student_content_sha256": record["content_sha256"],
        "core_content_sha256": record["core_content_sha256"],
        "periphery_content_sha256": record["periphery_content_sha256"],
        "weights": {k: record[k] for k in ("path", "bytes", "sha256")},
    }
    index_path = root / "index.json"
    write_new(index_path, index)
    record["training_index"] = {"path": str(index_path), "sha256": sha(index_path)}
    return record


@pytest.mark.parametrize("kind", checkpoint.KINDS)
def test_strict_checkpoint_index_shapes_and_three_semantics(tmp_path, kind):
    state = (
        {"weight": torch.ones(2, 3)}
        if kind == "core_only"
        else {"core.weight": torch.ones(2, 3), "periphery.action_decoder.weight": torch.zeros(3, 2)}
    )
    record = record_for(tmp_path / kind, state, kind)
    signature = checkpoint.signature(state)
    assert checkpoint.verify_candidate(record, signature)["all_tensor_keys_shapes_dtypes_verified"]
    bad = deepcopy(record)
    bad["core_content_sha256"] = "wrong-core"
    with pytest.raises(ValueError, match="semantic"):
        checkpoint.verify_candidate(bad, signature)
    with pytest.raises(ValueError, match="key inventory"):
        checkpoint.verify_candidate(record, {})
    bad = deepcopy(signature)
    bad[next(iter(bad))] = ((10, 10), torch.float32)
    with pytest.raises(ValueError, match="shape/dtype"):
        checkpoint.verify_candidate(record, bad)
    changed = deepcopy(record)
    changed["expected_step"] = 50000
    with pytest.raises(ValueError, match="expected-step"):
        checkpoint.verify_candidate(changed, signature)


def test_explicit_fixture_can_be_structure_but_never_a_final_candidate(tmp_path):
    state = {"core.weight": torch.ones(2), "periphery.vlln.weight": torch.ones(2)}
    record = record_for(tmp_path / "fixture", state, "core_periphery", fixture=True)
    signature = checkpoint.load_structure_template(record)
    checkpoint.verify_candidate(record, signature, allow_fixture=True)
    with pytest.raises(ValueError, match="not a final trained candidate"):
        checkpoint.verify_candidate(record, signature)
    disguised = deepcopy(record)
    disguised["fixture_only"] = False
    with pytest.raises(ValueError, match="not a final trained candidate"):
        checkpoint.verify_candidate(disguised, signature)
    with pytest.raises(ValueError, match="explicit"):
        checkpoint.verify_candidate(disguised, signature, allow_fixture=True)


def test_cpu_fixture_derives_from_real_parent_and_b0_without_training_identity(
    tmp_path, monkeypatch
):
    head, _ = head_and_frame()
    model = nn.Module()
    model.action_head = head
    model.eval().requires_grad_(False)
    state = {k: v.float() for k, v in head.model.state_dict().items()}
    parent = record_for(tmp_path / "parent", state, "core_only")
    monkeypatch.setattr(checkpoint, "BudgetedActionMamba", lambda native: deepcopy(native).float())
    base_hash = proto.content_hash(model.state_dict())
    record = checkpoint.save_deployment_fixture(
        model, parent, base_hash, tmp_path / "derived", "derived-source"
    )
    index = checkpoint.read(record["training_index"]["path"])
    assert record["fixture_only"] and index["fixture_only"]
    assert index["optimizer_steps"] == index["added_steps"] == 0
    assert index["residency"]["all_parameters_and_buffers_cpu"]
    assert index["residency"]["torch_cuda_memory_allocated_bytes"] == 0
    assert index["core_parent"] == parent and index["base_content_sha256"] == base_hash
    assert record["core_content_sha256"] == parent["core_content_sha256"]
    assert proto.content_hash(model.state_dict()) == base_hash
    owner = checkpoint.load_student(head, record, allow_fixture=True)
    assert proto.content_hash(owner.state_dict()) == record["content_sha256"]
    with pytest.raises(ValueError, match="not a final trained candidate"):
        checkpoint.load_student(head, record)
    plan = {
        "student_checkpoint": record,
        "train": [1],
        "dev": [2],
        "grids": {"1": [0], "2": [4]},
        "fixtures": [{"episode": 2, "anchor": 4}],
    }
    checkpoint.verify_fixture_scope(plan)
    plan["fixtures"] = [{"episode": 2000, "anchor": 0}]
    with pytest.raises(ValueError, match="train/dev"):
        checkpoint.verify_fixture_scope(plan)
    assert not torch.cuda.is_initialized()


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.condition_calls = 0
        self.calls = 0

    def encode_condition(self, features):
        self.condition_calls += 1
        code = features.mean(1)
        return ActionCondition(code, code[:, None, None, :])

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        *,
        condition_cache=None,
        return_all_hidden_states=False,
        **kwargs,
    ):
        self.calls += 1
        condition = (
            encoder_hidden_states.mean(1)
            if condition_cache is None
            else condition_cache.modulation.mean((1, 2))
        )
        value = self.linear(
            hidden_states + hidden_states.mean(1, keepdim=True) + condition[:, None]
        )
        return (value, [value]) if return_all_hidden_states else value


class ToyModel(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.action_head = head
        self.seen_heads = []
        self.fail = False

    def prepare_input(self, observation):
        self.seen_heads.append(self.action_head)
        return observation["raw"], BatchFeature(
            data={
                k: v.clone()
                for k, v in observation["action_input"].items()
                if k not in ("action", "action_mask")
            }
        )

    def backbone(self, value):
        if self.fail:
            raise RuntimeError("VLM failure")
        return BatchFeature(data={k: v.clone() for k, v in value.items()})


class CPUGraph:
    def __init__(self, surface, samples):
        self.surface = surface
        self.setup = {"cpu_test_only": True}
        self.closed = False

    def __call__(self, *args):
        assert not self.closed
        return self.surface(*args)

    def close(self):
        self.closed = True


def composite_policy(monkeypatch):
    original_autocast = torch.autocast
    monkeypatch.setattr(
        torch, "autocast", lambda _device, **kwargs: original_autocast("cpu", **kwargs)
    )
    monkeypatch.setattr(policy_module, "ForwardGraph", CPUGraph)
    head, frame = head_and_frame()
    head.action_dim, head.action_horizon, head.num_inference_timesteps = 13, 40, 4
    head.config.action_horizon = 40
    owner = NativeStudent(head, ToyCore(), "core_only_continuation").eval().requires_grad_(False)
    # Diagnostic private perturbation must force the complete learned-head route.
    with torch.no_grad():
        owner.periphery["action_decoder"].layer2.b.add_(0.75)
        owner.periphery["state_encoder"].layer2.b.add_(0.5)
        owner.periphery["vlln"].bias.add_(0.25)
    record = {
        "kind": "core_periphery",
        **checkpoint.semantic_identity(owner.state_dict(), "core_periphery"),
    }
    deployment = checkpoint.DeployedStudent(owner, head, record)
    model = ToyModel(head)
    deployment.place(head, "student_graph", device="cpu")
    policy = policy_module.ActionTransferPolicy.__new__(policy_module.ActionTransferPolicy)
    policy.model, policy.student, policy.mode = model, deployment, "student_graph"
    policy.closed, policy.lease, policy.route = False, None, None
    inputs = {kind: {k: v[:1].clone() for k, v in values.items()} for kind, values in frame.items()}
    policy.prepare_observation = lambda observation: (observation, {})
    return policy, inputs, original_autocast


def test_full_head_binding_materialized_once_four_euler_one_condition_and_failure_restore(
    monkeypatch,
):
    policy, inputs, autocast = composite_policy(monkeypatch)
    deployment, model = policy.student, policy.model
    teacher = model.action_head
    storage = {k: v.data_ptr() for k, v in deployment.head.state_dict().items()}
    torch.random.default_generator.manual_seed(42)
    with autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        # Incorrect historical core-only binding must disagree under the perturbation.
        wrong = current_policy(model, inputs, deployment.core).detach().clone()
    deployment.core.calls = deployment.core.condition_calls = 0
    torch.random.default_generator.manual_seed(42)
    actual = policy.gpu_action(inputs, graph=False).detach().clone()
    assert not torch.equal(wrong, actual)
    assert model.action_head is teacher and model.seen_heads[-1] is deployment.head
    assert deployment.core.condition_calls == 1 and deployment.core.calls == 4
    # Match the trained FP32 masters' differentiable BF16 view exactly.
    torch.random.default_generator.manual_seed(42)
    model.action_head = deployment.owner.head
    try:
        with autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
            reference = current_policy(model, inputs, deployment.core)
    finally:
        model.action_head = teacher
    assert torch.equal(actual, reference)
    policy.initialize_graph(inputs)
    with pytest.raises(ValueError, match="Close the action graph"):
        deployment.place(teacher, "native_graph", device="cpu")
    actions = []
    other = deepcopy(inputs)
    other["raw"]["backbone_features"].mul_(-3)
    for observation in (inputs, other, inputs):
        deployment.core.calls = deployment.core.condition_calls = 0
        torch.random.default_generator.manual_seed(42)
        actions.append(policy.gpu_action(observation, graph=True).detach().clone())
        assert deployment.core.calls == 4 and deployment.core.condition_calls == 1
    assert torch.equal(actions[0], actions[2]) and not torch.equal(actions[0], actions[1])
    assert storage == {k: v.data_ptr() for k, v in deployment.head.state_dict().items()}
    assert model.action_head is teacher
    model.fail = True
    with pytest.raises(RuntimeError, match="VLM failure"):
        policy.gpu_action(inputs, graph=True)
    assert model.action_head is teacher
    policy.close()
    assert deployment.active_graphs == 0
    deployment.place(teacher, "native_graph", device="cpu")
    assert not torch.cuda.is_initialized()


def toy_final_plan():
    episodes = list(range(1000, 1016))
    return {
        "final": episodes,
        "noise_seeds": [42, 43, 44, 45],
        "eval_anchors": {str(ep): list(range(0, 128, 4)) for ep in episodes},
        "lengths": {str(ep): 144 for ep in episodes},
        "candidates": [{"name": f"candidate{i}"} for i in range(6)],
        "comparisons": [{"candidate": "candidate1", "control": "candidate0"}],
        "quality_gates": {
            "max_flow_ratio": 0.98,
            "max_combined_ratio": 0.98,
            "min_episodes_better": 12,
        },
        "native_retention": {"max_metric_ratio": 1.02, "metrics": list(support.NATIVE_METRICS)},
        "paired_bootstrap": {
            "resamples": 10000,
            "seed": 20261009,
            "unit": "episode",
            "confidence": 0.95,
        },
    }


def test_all_six_candidate_identities_precede_any_final_input(monkeypatch, tmp_path):
    plan = toy_final_plan()
    cohort = {"final": plan["final"], "reserve": list(range(2000, 2016))}
    assert support.verify_final_contract(plan, cohort) == 512
    bad = deepcopy(plan)
    bad["candidates"].append({"name": "seventh"})
    with pytest.raises(ValueError, match="2..6"):
        support.verify_final_contract(bad, cohort)
    base, prepared = tmp_path / "base", tmp_path / "prepared"
    (prepared / "processor").mkdir(parents=True)
    base.mkdir()
    write_new(
        base / "config.json",
        {
            "action_horizon": 40,
            "max_state_dim": 13,
            "max_action_dim": 13,
            "num_inference_timesteps": 4,
        },
    )
    write_new(prepared / "processor/processor_config.json", {})
    write_new(prepared / "processor/statistics.json", {})
    write_new(tmp_path / "cohort.json", cohort)
    state = {"core.weight": torch.ones(2), "periphery.vlln.weight": torch.ones(2)}
    plan["candidates"] = [
        record_for(tmp_path / f"c{i}", state, "core_periphery", name=f"candidate{i}")
        for i in range(6)
    ]
    template = plan["candidates"][0]
    # Last candidate fails: no earlier valid candidate may cause final data access.
    plan["candidates"][-1]["core_content_sha256"] = "wrong"
    output = tmp_path / "artifacts" / "final" / "output"
    paths = [
        base / "config.json",
        prepared / "processor/processor_config.json",
        prepared / "processor/statistics.json",
    ]
    plan.update(
        source_worktree=str(final.REPO),
        artifact_root=str(tmp_path / "artifacts"),
        run_id="final",
        optimizer_steps=0,
        base_checkpoint=str(base),
        prepared=str(prepared),
        records=[{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths],
        immutable_source_relative_paths=[],
        program_cohort_file=str(tmp_path / "cohort.json"),
        program_cohort_sha256=sha(tmp_path / "cohort.json"),
        student_structure_templates={"core_periphery": template},
    )
    write_new(tmp_path / "PLAN.json", plan)
    seen = []
    original = final.verify_candidate

    def verify(candidate, signature):
        seen.append(candidate["name"])
        return original(candidate, signature)

    monkeypatch.setattr(final, "verify_candidate", verify)
    monkeypatch.setattr(
        final, "strict_reload", lambda *a: pytest.fail("Model touched before all candidates passed")
    )
    monkeypatch.setattr(final, "encode_frame", lambda *a: pytest.fail("Final input accessed"))
    args = SimpleNamespace(
        plan=tmp_path / "PLAN.json",
        plan_sha256=sha(tmp_path / "PLAN.json"),
        artifact_root=tmp_path / "artifacts",
        output=output,
    )
    with pytest.raises(ValueError, match="semantic"):
        final.verify_plan(args)
    assert len(seen) == 6 and not output.exists() and not torch.cuda.is_initialized()


def test_joint_student_condition_changes_do_not_change_teacher_pairing(monkeypatch):
    shared = {
        "continuous_time": "t",
        "discrete_time": "td",
        "noisy_action": "noise",
        "forward_seed": 42,
    }
    row = {
        "input_frame_content_sha256": "frozen-raw-state-target-mask",
        "teacher_metrics": {"flow_loss": 1.0},
        "teacher_noise_receipt": {**shared, "effective_condition": "B0"},
        "noise_receipt": {**shared, "effective_condition": "control"},
    }
    other = deepcopy(row)
    other["noise_receipt"]["effective_condition"] = "adapted"
    result = {
        "array_sha256": {
            k: k
            for k in (
                "teacher_actions",
                "teacher_physical",
                "teacher_velocity",
                "physical_targets",
                "action_masks",
            )
        },
        "teacher_metrics": {},
        "teacher_per_episode": {},
        "normalization_sha256": "norm",
        "cache_report_sha256": "raw",
    }
    monkeypatch.setattr(
        support,
        "verify_evaluation",
        lambda record, pairs: (result, [row if record == "a" else other]),
    )
    assert support.assert_teacher_equal("a", "b", [])
    other["noise_receipt"]["noisy_action"] = "different"
    with pytest.raises(ValueError, match="pairing differs"):
        support.assert_teacher_equal("a", "b", [])
    other["noise_receipt"]["noisy_action"] = "noise"
    other["teacher_noise_receipt"]["effective_condition"] = "changed-teacher"
    with pytest.raises(ValueError, match="pairing differs"):
        support.assert_teacher_equal("a", "b", [])


def test_final_composite_evaluator_uses_independent_teacher_and_full_metadata(
    monkeypatch, tmp_path
):
    state = {"core.weight": torch.ones(2), "periphery.vlln.weight": torch.ones(2)}
    ctx = SimpleNamespace(
        candidate={"kind": "core_periphery"},
        student=SimpleNamespace(state_dict=lambda: state),
        cache=SimpleNamespace(frame=lambda ep, anchor: {"content_sha256": "shared-input"}),
    )
    output = tmp_path / "composite"

    def evaluate(ctx, pairs, path):
        path.mkdir()
        value = {
            "scope": support.OLD_SCOPE,
            "checkpoint_content_sha256": "old-core-only",
            "metrics": {"flow_loss": 1.0},
        }
        support.periphery.write_new(path / "details.json", [{"episode": 1000, "anchor": 0}])
        support.periphery.write_new(path / "RESULT.json", value)
        return {"result": value}

    monkeypatch.setattr(support.periphery, "evaluate_pairs", evaluate)
    monkeypatch.setattr(
        support.matched,
        "evaluate_pairs",
        lambda *a: pytest.fail("Joint used old shared-decoder teacher replay"),
    )
    actual = support.evaluate_final_candidate(ctx, [], output)["result"]
    hashes = checkpoint.semantic_identity(state, "core_periphery")
    assert (
        actual["checkpoint_content_sha256"]
        == actual["student_content_sha256"]
        == hashes["content_sha256"]
    )
    assert actual["core_content_sha256"] == hashes["core_content_sha256"]
    assert actual["periphery_content_sha256"] == hashes["periphery_content_sha256"]
    assert actual["metrics"] == {"flow_loss": 1.0} and actual["cohort"] == "final"
    assert support.read(output / "details.json")[0]["input_frame_content_sha256"] == "shared-input"


def test_fixture_entrypoint_refuses_visible_gpu_before_loading(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "action_transfer_checkpoint.py",
            "fixture",
            "--plan",
            str(tmp_path / "absent-plan"),
            "--plan-sha256",
            "unread",
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--wall-seconds",
            "10",
        ],
    )
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        checkpoint.fixture_main()
    assert not (tmp_path / "output").exists() and not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["core_only", "core_periphery"])
def test_development_adapter_is_two_dev_pairs_and_never_final(monkeypatch, tmp_path, kind):
    state = (
        {"weight": torch.ones(2)}
        if kind == "core_only"
        else {"core.weight": torch.ones(2), "periphery.vlln.weight": torch.ones(2)}
    )
    pairs = [{"episode": ep, "anchor": 0, "nominal_seed": 42} for ep in (1, 2)]
    plan = {
        "dev": [1, 2],
        "train": [3],
        "final": [4],
        "reserve": [5],
        "grids": {str(ep): [0] for ep in range(1, 6)},
        "fixtures": [{"episode": ep, "anchor": 0} for ep in (1, 2)],
    }
    ctx = SimpleNamespace(
        plan=plan,
        candidate={"kind": kind, "fixture_only": True},
        student=SimpleNamespace(state_dict=lambda: state),
        cache=SimpleNamespace(frame=lambda ep, anchor: {"content_sha256": f"frame{ep}"}),
    )
    calls = []
    evaluator = support.periphery if kind == "core_periphery" else support.matched

    def evaluate(ctx, selected, path):
        calls.append(selected)
        path.mkdir()
        rows = [
            {"episode": pair["episode"], "anchor": pair["anchor"], "seed": pair["nominal_seed"]}
            for pair in selected
        ]
        evaluator.write_new(path / "details.json", rows)
        evaluator.write_new(path / "NORMALIZATION.json", {"normalization": "unchanged"})
        result = {
            "status": "passed",
            "scope": support.OLD_SCOPE,
            "rows": 2,
            "metrics": {"flow_loss": 1.0},
            "array_sha256": {},
            "details_sha256": sha(path / "details.json"),
            "normalization_sha256": sha(path / "NORMALIZATION.json"),
        }
        evaluator.write_new(path / "RESULT.json", result)
        return {
            "path": str(path / "RESULT.json"),
            "sha256": sha(path / "RESULT.json"),
            "result": result,
        }

    monkeypatch.setattr(evaluator, "evaluate_pairs", evaluate)
    assert support.development_pairs(plan) == pairs
    record = support.evaluate_development_candidate(ctx, pairs, tmp_path / "development")
    result, rows = support.verify_development_evaluation(record, pairs)
    assert result["development_only"] is True and result["cohort"] == "dev"
    assert result["checkpoint_kind"] == kind and rows[0]["input_frame_content_sha256"] == "frame1"
    with pytest.raises(ValueError, match="mislabeled"):
        support.verify_evaluation(record, pairs)
    with pytest.raises(ValueError, match="cannot use the final evaluator"):
        support.evaluate_final_candidate(ctx, pairs, tmp_path / "forbidden-final")
    for episode in (3, 4, 5):
        wrong = [{**pairs[0], "episode": episode}, pairs[1]]
        with pytest.raises(ValueError, match="escaped"):
            support.evaluate_development_candidate(ctx, wrong, tmp_path / f"forbidden{episode}")
    with pytest.raises(ValueError, match="exactly two"):
        support.evaluate_development_candidate(ctx, pairs[:1], tmp_path / "one")
    assert len(calls) == 1 and not torch.cuda.is_initialized()
