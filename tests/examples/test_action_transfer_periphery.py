"""CPU regression tests for native BF16 views, live gradients and composite state."""

from copy import deepcopy
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
import action_transfer_periphery as runner  # noqa:E402
from action_transfer_periphery_support import (  # noqa:E402
    BF16View,
    NativeStudent,
    adaptation_lr,
    live_tensor_args,
    make_optimizer,
    optimizer_steps,
    storage_inventory,
    task_forward,
)
from action_transfer_program_support import cpu_tree, optimizer_names  # noqa:E402
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead  # noqa:E402
from gr00t.model.modules.embodiment_conditioned_mlp import (  # noqa:E402
    CategorySpecificLinear,
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from memory_training import optimizer_for  # noqa:E402
from memory_training_support import fresh  # noqa:E402


PLAN = {
    "optimizer": {
        "lr": 1e-5,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "matrix_weight_decay": 1e-4,
        "bias_norm_no_weight_decay": 0,
    }
}


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(
        self, hidden_states, encoder_hidden_states, return_all_hidden_states=False, **kwargs
    ):
        value = self.linear(
            hidden_states
            + hidden_states.mean(1, keepdim=True)
            + encoder_hidden_states.mean(1, keepdim=True)
        )
        return (value, [value]) if return_all_hidden_states else value


def head_and_frame():
    torch.manual_seed(321)
    # Small dimensions, actual unchanged native forward/time sampling/modules.
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(head)
    head.config = SimpleNamespace(
        state_history_length=1, add_pos_embed=True, use_alternate_vl_dit=True, noise_s=0.999
    )
    head.model = Core()
    head.vlln = nn.LayerNorm(4)
    head.vl_self_attention = nn.Linear(4, 4)
    head.state_encoder = CategorySpecificMLP(2, 13, 4, 4)
    head.action_encoder = MultiEmbodimentActionEncoder(13, 4, 2)
    head.action_decoder = CategorySpecificMLP(2, 4, 4, 13)
    head.position_embedding = nn.Embedding(40, 4)
    head.beta_dist = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
    head.num_timestep_buckets = 1000
    head.state_dropout_prob = 0.2
    head.to(torch.bfloat16).eval().requires_grad_(False)
    frame = {
        "raw": {
            "backbone_features": torch.randn(4, 7, 4).to(torch.bfloat16),
            "backbone_attention_mask": torch.ones(4, 7, dtype=torch.bool),
            "image_mask": torch.ones(4, 7, dtype=torch.bool),
        },
        "action_input": {
            "state": torch.randn(4, 1, 13).to(torch.bfloat16),
            "action": torch.randn(4, 40, 13).to(torch.bfloat16),
            "action_mask": torch.cat((torch.ones(4, 16, 13), torch.zeros(4, 24, 13)), 1),
            "embodiment_id": torch.zeros(4, dtype=torch.long),
        },
    }
    return head, frame


def parent_payload(core):
    opt = optimizer_for(core, PLAN)
    for p in core.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)
    for state in opt.state.values():
        state["step"].fill_(40000)
    return {
        "optimizer": cpu_tree(opt.state_dict()),
        "optimizer_parameter_names": optimizer_names(core, opt),
    }


def test_bf16_view_preserves_category_bias_and_master_gradients():
    native = CategorySpecificLinear(2, 3, 5).bfloat16()
    with torch.no_grad():
        native.b.normal_()
    view = BF16View(deepcopy(native).float())
    x = torch.randn(4, 7, 3).bfloat16()
    ids = torch.tensor([0, 1, 1, 0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, actual = native(x, ids), view(x, ids)
        promoted = view.master(x, ids)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert promoted.dtype == torch.float32
    assert torch.equal(actual, expected)
    actual.float().square().sum().backward()
    assert all(
        p.dtype == torch.float32
        and p.grad is not None
        and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in view.parameters()
    )
    with torch.no_grad():
        view.master.b.add_(0.1)
    assert not torch.equal(view(x, ids), expected)  # Views cannot cache stale weights.


@pytest.mark.parametrize("condition", ["core_only_continuation", "joint_periphery"])
def test_actual_native_b4_forward_is_exact_at_b0_and_live_paths_are_preserved(condition):
    head, frame = head_and_frame()
    core = deepcopy(head.model).float()
    student = NativeStudent(head, core, condition)
    baseline = cpu_tree(head.state_dict())
    torch.manual_seed(9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = head(fresh(frame["raw"], "cpu"), fresh(frame["action_input"], "cpu"))
    torch.manual_seed(9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = task_forward(student, core, frame, retain_inputs=True)
    assert torch.equal(expected["loss"], actual["native"]["loss"])
    samples = live_tensor_args(actual["live_inputs"])
    assert [s.requires_grad for s in samples] == [condition == "joint_periphery"] * 2 + [False] * 3
    assert all(s.is_leaf for s in samples)
    actual["native"]["loss"].backward()
    if condition == "joint_periphery":
        assert all(
            actual["live_inputs"][k].grad.abs().sum() > 0
            for k in ("hidden_states", "encoder_hidden_states")
        )
        assert all(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
            for module in student.periphery.values()
        )
    else:
        assert all(p.grad is None for p in student.periphery.parameters())
    assert all(torch.equal(v, head.state_dict()[k]) for k, v in baseline.items())
    assert storage_inventory(student, head)["disjoint"]
    assert set(student.state_dict()) == {"core." + k for k in core.state_dict()} | {
        "periphery." + k for k in student.periphery.state_dict()
    }
    # Deployment materialization remains exact and restores the trainable view.
    with student.materialized() as view:
        torch.manual_seed(9)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            deployed = view.action_head(
                fresh(frame["raw"], "cpu"), fresh(frame["action_input"], "cpu")
            )
    assert torch.equal(expected["loss"], deployed["loss"])
    assert isinstance(student.head.action_decoder, BF16View)


@pytest.mark.parametrize("condition", ["core_only_continuation", "joint_periphery"])
def test_optimizer_restores_core_moments_and_starts_periphery_from_zero(condition):
    head, frame = head_and_frame()
    student = NativeStudent(head, deepcopy(head.model).float(), condition)
    payload = parent_payload(student.core)
    opt = make_optimizer(student, PLAN, payload)
    assert optimizer_steps(student, opt, 0)["core_step"] == 40000
    decay_by_id = {
        id(p): group["weight_decay"] for group in opt.param_groups for p in group["params"]
    }
    if condition == "joint_periphery":
        for name, p in student.periphery.named_parameters():
            if name.rsplit(".", 1)[-1] in ("b", "bias"):
                assert decay_by_id[id(p)] == 0
    for source_group, actual_group in zip(
        payload["optimizer"]["param_groups"], opt.param_groups[:2]
    ):
        for source_id, parameter in zip(source_group["params"], actual_group["params"]):
            assert torch.equal(
                payload["optimizer"]["state"][source_id]["exp_avg"], opt.state[parameter]["exp_avg"]
            )
    before = cpu_tree(student.periphery.state_dict())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        values = task_forward(student, student.core, frame)
    values["native"]["loss"].backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1)
    opt.step()
    assert optimizer_steps(student, opt, 1)["core_step"] == 40001
    if condition == "joint_periphery":
        assert all(int(opt.state[p]["step"]) == 1 for p in student.periphery.parameters())
        assert all(
            any(not torch.equal(v, before[name + "." + k]) for k, v in module.state_dict().items())
            for name, module in student.periphery.items()
        )
    else:
        assert all(torch.equal(v, before[k]) for k, v in student.periphery.state_dict().items())
    bad = deepcopy(payload)
    bad["optimizer_parameter_names"][0].reverse()
    # This tiny core has one matrix; break the name explicitly.
    bad["optimizer_parameter_names"][0][0] += "wrong"
    with pytest.raises(ValueError, match="inventory"):
        make_optimizer(student, PLAN, bad)


def test_composite_checkpoint_roundtrip_keeps_periphery_optimizer_and_names(tmp_path):
    head, frame = head_and_frame()
    student = NativeStudent(head, deepcopy(head.model).float(), "joint_periphery")
    opt = make_optimizer(student, PLAN, parent_payload(student.core))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        task_forward(student, student.core, frame)["native"]["loss"].backward()
    opt.step()
    path = tmp_path / "composite.pt"
    torch.save(
        {
            "weights": student.state_dict(),
            "optimizer": opt.state_dict(),
            "names": optimizer_names(student, opt),
        },
        path,
    )
    restored = torch.load(path, map_location="cpu", weights_only=True)
    other = NativeStudent(head, deepcopy(head.model).float(), "joint_periphery")
    other_opt = make_optimizer(other, PLAN, parent_payload(other.core))
    other.load_state_dict(restored["weights"], strict=True)
    assert optimizer_names(other, other_opt) == restored["names"]
    other_opt.load_state_dict(restored["optimizer"])
    optimizer_steps(other, other_opt, 1)
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in student.state_dict().items())
    assert all(
        torch.equal(opt.state[p]["exp_avg"], other_opt.state[q]["exp_avg"])
        for p, q in zip(student.parameters(), other.parameters())
    )
    broken = {
        k: v
        for k, v in restored["weights"].items()
        if not k.startswith("periphery.action_decoder.")
    }
    with pytest.raises(RuntimeError, match="Missing key"):
        other.load_state_dict(broken, strict=True)


def assert_tree_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree_exact(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("condition", ["core_only_continuation", "joint_periphery"])
def test_proof_checkpoint_reload_then_reset_preserves_parent_payload(
    tmp_path, monkeypatch, condition
):
    # Exercise the production save/reload/reset functions on the native tiny B4
    # head. Only the CUDA-only replay surface/device is replaced for this CPU test.
    from integrity_diagnosis_support import BaseAudit
    from runner_core import set_seed

    autocast = torch.autocast
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch, "autocast", lambda _device, **kwargs: autocast("cpu", **kwargs))
    monkeypatch.setattr(runner, "eager_route", lambda student: student.core)
    monkeypatch.setattr(runner, "set_seed", lambda seed, cuda: set_seed(seed, cuda=False))
    head, frame = head_and_frame()
    teacher = nn.Module()
    teacher.action_head = head
    student = NativeStudent(head, deepcopy(head.model).float(), condition)
    payload = parent_payload(student.core)
    payload["rng"] = runner.rng_state()
    original_payload = cpu_tree(payload)
    initial_core = cpu_tree(student.core.state_dict())
    initial_periphery = cpu_tree(student.periphery.state_dict())
    optimizer = make_optimizer(student, PLAN, payload)
    audit = BaseAudit(
        teacher,
        tmp_path,
        runner.proto.content_hash(teacher.state_dict()),
        synchronize=lambda: None,
    )
    cache = SimpleNamespace(
        frame=lambda ep, anchor: {
            "raw": {key: value[ep : ep + 1] for key, value in frame["raw"].items()},
            "action_input": {
                key: value[ep : ep + 1] for key, value in frame["action_input"].items()
            },
            "content_sha256": str(ep),
        }
    )
    ctx = SimpleNamespace(
        student=student,
        optimizer=optimizer,
        parent_payload=payload,
        initial_core=initial_core,
        initial_periphery=initial_periphery,
        condition=condition,
        completed_step=50000,
        cache=cache,
        output=tmp_path,
        audit=audit,
        budget=lambda: None,
        source_commit="cpu-test-source",
        args=SimpleNamespace(plan_sha256="cpu-test-plan"),
        plan={**PLAN, "cache_report_sha256": "cache", "schedule_sha256": "schedule", "parent": {}},
    )
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cpu", dtype=torch.bfloat16):
            values = task_forward(student, student.core, frame)
        values["native"]["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1)
        optimizer.step()
        ctx.completed_step += 1
    optimizer_steps(student, optimizer, 3)
    checkpoint = runner.save_checkpoint(ctx, tmp_path / "proof-checkpoint", {})
    planned = {"batch": [{"episode": ep, "anchor": 0} for ep in range(4)], "forward_seed": 912}
    restored = runner.reload_checkpoint(ctx, checkpoint, planned, "cpu-proof")
    assert restored["passed"] and restored["rng_exact"]
    report = runner.reset_parent(ctx)
    assert ctx.completed_step == 50000 and report["optimizer"]["core_step"] == 40000
    assert_tree_exact(payload, original_payload)
    assert_tree_exact(student.core.state_dict(), initial_core)
    assert_tree_exact(student.periphery.state_dict(), initial_periphery)
    assert_tree_exact(runner.rng_state(), original_payload["rng"])
    for group, actual in zip(
        original_payload["optimizer"]["param_groups"], ctx.optimizer.param_groups[:2]
    ):
        for identifier, parameter in zip(group["params"], actual["params"]):
            assert_tree_exact(
                ctx.optimizer.state[parameter], original_payload["optimizer"]["state"][identifier]
            )
    assert all(p not in ctx.optimizer.state for p in student.periphery.parameters())
    assert not torch.cuda.is_initialized()


def test_lr_endpoints_and_fixed_paired_bootstrap():
    values = [adaptation_lr(s) for s in range(50001, 70001)]
    assert values[0] == 1e-5 and values[-1] == 1e-6
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[123] == 1e-6 + 9e-6 * (1 + math.cos(math.pi * 123 / 19999)) / 2
    with pytest.raises(ValueError):
        adaptation_lr(50000)
    a = {str(i): {k: float(i + 1) for k in runner.METRICS} for i in range(8)}
    b = {str(i): {k: float(i + 1) * 0.97 for k in runner.METRICS} for i in range(8)}
    result = runner.paired_bootstrap(a, b, list(range(8)))
    assert result == runner.paired_bootstrap(a, b, list(range(8)))
    assert np.isclose(result["flow_loss"]["joint_over_core_only"], 0.97)
    assert result["flow_loss"]["episodes_better"] == 8
