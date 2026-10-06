"""CPU native-flow/cache/pairing/GRU contracts; never starts an optimizer or GPU."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.model.modules.causal_visual_gru import CausalVisualGRU
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import memory_training as training  # noqa: E402
import memory_training_support as support  # noqa: E402
from runner_core import set_seed  # noqa: E402


def native_method(name):
    path = ROOT / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gr00tN1d7ActionHead"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class IdentityCategory(nn.Module):
    def forward(self, value, *args):
        return value


class Twice(nn.Module):
    def forward(self, value):
        return value * 2


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        value = kwargs["hidden_states"] + kwargs["encoder_hidden_states"].mean() * self.weight
        return value, []


class NativeHead(nn.Module):
    forward = native_method("forward")
    sample_time = native_method("sample_time")
    process_backbone_output = native_method("process_backbone_output")

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            state_history_length=1, add_pos_embed=False, use_alternate_vl_dit=True, noise_s=0.999
        )
        self.beta_dist = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        self.num_timestep_buckets = 1000
        self.state_dropout_prob = 0.2
        self.state_encoder = IdentityCategory()
        self.action_encoder = IdentityCategory()
        self.action_decoder = IdentityCategory()
        self.vlln = Twice()
        self.vl_self_attention = nn.Identity()
        self.model = TinyDiT()
        self.eval()

    def set_frozen_modules_to_eval_mode(self):
        pass


def frame():
    mask = torch.zeros(1, 40, 13)
    mask[:, :16] = 1
    value = {
        "raw": {
            "backbone_features": torch.randn(1, 209, 2048),
            "backbone_attention_mask": torch.ones(1, 209, dtype=torch.bool),
            "image_mask": torch.ones(1, 209, dtype=torch.bool),
        },
        "post_vl_features": torch.zeros(1, 209, 2048),
        "action_input": {
            "state": torch.randn(1, 1, 13),
            "action": torch.randn(1, 40, 13),
            "action_mask": mask,
            "embodiment_id": torch.tensor([10]),
        },
        "physical_states": {"left_arm": torch.zeros(1, 7), "left_hand": torch.zeros(1, 6)},
        "physical_actions": {"left_arm": torch.zeros(16, 7), "left_hand": torch.zeros(16, 6)},
    }
    value["post_vl_features"] = value["raw"]["backbone_features"] * 2
    value["content_sha256"] = support.frame_hash(value)
    return value


def test_actual_native_flow_body_zero_hook_loss_rng_receipts_and_cache_immutability(monkeypatch):
    torch.set_num_threads(4)
    calls = []

    def cpu_seed(seed, *, cuda):
        calls.append((seed, cuda))
        set_seed(seed)

    monkeypatch.setattr(support, "set_seed", cpu_seed)
    model = SimpleNamespace(action_head=NativeHead(), device=torch.device("cpu"))
    cached = frame()
    before = support.frame_hash(cached)
    original, receipt = support.native_flow(model, cached, None, 123)
    torch.randn(125)
    zero = torch.zeros(192, 2048, requires_grad=True)
    injected, again = support.native_flow(model, cached, zero, 123)
    assert calls == [(123, True), (123, True)]
    assert receipt == again
    assert receipt["continuous_time_values"] and receipt["noisy_action"]["shape"] == [1, 40, 13]
    for key in ("loss", "action_loss", "action_mask", "state_features", "backbone_features"):
        assert torch.equal(original[key], injected[key])
    assert injected["loss"].dtype == torch.float32
    assert torch.equal(
        injected["loss"], injected["action_loss"].sum() / (injected["action_mask"].sum() + 1e-6)
    )
    injected["loss"].backward()
    assert zero.grad is not None and zero.grad.abs().sum() > 0
    assert model.action_head.model.weight.grad is None
    assert support.frame_hash(cached) == before
    assert (
        not model.action_head.model._forward_pre_hooks
        and not model.action_head.action_encoder._forward_pre_hooks
    )
    nonzero = torch.full((192, 2048), 0.1, requires_grad=True)
    changed, changed_receipt = support.native_flow(model, cached, nonzero, 123)
    assert changed_receipt["noisy_action"] == receipt["noisy_action"]
    assert changed_receipt["continuous_time"] == receipt["continuous_time"]
    assert changed_receipt["effective_condition"] != changed_receipt["preinjection_post"]
    assert torch.equal(changed["backbone_features"], cached["post_vl_features"])
    assert not torch.equal(changed["loss"], original["loss"])
    assert model.action_head.model.last_kwargs["return_all_hidden_states"] is True
    assert torch.equal(
        model.action_head.model.last_kwargs["image_mask"], cached["raw"]["image_mask"]
    )


def test_hook_cleanup_on_native_forward_error(monkeypatch):
    monkeypatch.setattr(support, "set_seed", lambda seed, **kwargs: set_seed(seed))
    head = NativeHead()
    cached = frame()
    cached["action_input"]["action_mask"].zero_()
    original = head.sample_time.__func__
    with pytest.raises(ValueError, match="empty"):
        support.native_flow(SimpleNamespace(action_head=head, device="cpu"), cached, None, 42)
    assert head.sample_time.__func__ is original
    assert not head.model._forward_pre_hooks and not head.action_encoder._forward_pre_hooks


def test_control_prefix_keeps_null_transitions_and_does_not_mutate_cached_history():
    sequence = torch.randn(7, 192, 2048)
    before = sequence.clone()
    reset = support.selected_prefix(sequence, "current_reset_mamba")
    zero = support.selected_prefix(sequence, "zero_history_mamba")
    assert reset.shape[0] == 1 and torch.equal(reset[0], sequence[-1])
    assert zero.shape[0] == 7 and torch.count_nonzero(zero[:-1]) == 0
    assert torch.equal(zero[-1], sequence[-1]) and torch.equal(sequence, before)
    projection = nn.Linear(2048, 256)
    with torch.no_grad():
        projection.bias.fill_(0.5)
    assert torch.count_nonzero(projection(zero[:-1])) > 0
    assert support.selected_prefix(sequence, "mamba_history") is sequence
    with pytest.raises(ValueError):
        support.selected_prefix(sequence, "native")


def common_initialization():
    generator = torch.Generator().manual_seed(42)
    return {
        "input_projection.weight": torch.randn(256, 2048, generator=generator) * 0.02,
        "input_projection.bias": torch.randn(256, generator=generator) * 0.02,
        "output_projection.weight": torch.zeros(2048, 256),
        "output_projection.bias": torch.zeros(2048),
        **{
            f"layers.{i}.norm.{key}": torch.ones(256) if key == "weight" else torch.zeros(256)
            for i in (0, 1)
            for key in ("weight", "bias")
        },
    }


def test_gru_common_initialization_recurrence_state_and_full_bptt():
    initial = common_initialization()
    one = CausalVisualGRU(initial)
    two = CausalVisualGRU(initial)
    for name, value in one.state_dict().items():
        assert value.dtype == torch.float32 and torch.equal(value, two.state_dict()[name])
        if ".cell." not in name:
            assert torch.equal(value, initial[name])
    state = one.new_state()
    assert state.describe()["bytes"] == 393216
    sequence = torch.randn(4, 192, 2048)
    full = one.full_sequence(sequence)
    steps = torch.stack([one.step(current, state)[0] for current in sequence])
    assert state.observations == 4 and not steps.requires_grad
    assert torch.allclose(full, steps, atol=2e-6, rtol=1e-5)
    cloned = state.clone()
    cloned.hidden[0].zero_()
    assert torch.any(state.hidden[0] != 0)
    state.reset()
    assert state.observations == 0 and all(torch.count_nonzero(v) == 0 for v in state.hidden)
    with torch.no_grad():
        one.output_projection.weight.normal_(std=1e-3)
    one.output_projection(one.full_sequence(sequence)[-1]).square().mean().backward()
    for layer in one.layers:
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0)
            for p in layer.cell.parameters()
        )


def test_cache_identity_and_confirmation_partition_fail_closed(tmp_path):
    from contract import sha, write_new

    values = {0: frame()}
    path = tmp_path / "episode-000125.pt"
    torch.save(values, path)
    plan = {"train": [125], "dev": [], "confirmation": [52], "grids": {"125": [0]}}
    write_new(
        tmp_path / "CACHE_RESULT.json",
        {
            "status": "passed",
            "plan_sha256": "fixed",
            "cohort": "train_dev",
            "episodes": [125],
            "files": [{"episode": 125, "file": path.name, "sha256": sha(path)}],
        },
    )
    cache = support.FeatureCache(tmp_path, plan, "fixed")
    assert cache.frame(125, 0)["content_sha256"] == values[0]["content_sha256"]
    with pytest.raises(ValueError, match="outside"):
        cache.episode(52)
    with pytest.raises(ValueError, match="Confirmation"):
        support.authorize_cache_cohort(plan, "confirmation")
    with pytest.raises(ValueError, match="Confirmation"):
        support.authorize_cache_cohort(
            plan, "confirmation", {"status": "failed", "continuation_gate": True}
        )
    cache.frame(125, 0)["raw"]["backbone_features"].zero_()
    with pytest.raises(ValueError, match="mutated"):
        cache.assert_unchanged()


def test_overfit_gate_is_conjunction_and_does_not_hide_arm_hand_regression():
    before = {
        "metrics": {
            "flow_loss": 1.0,
            "combined_mae_mixed_units": 1.0,
            "arm_mae_rad": 1.0,
            "hand_mae_u": 1.0,
        }
    }
    after = {
        "metrics": {
            "flow_loss": 0.9,
            "combined_mae_mixed_units": 1.0,
            "arm_mae_rad": 1.02,
            "hand_mae_u": 1.02,
        }
    }
    assert all(support.overfit_gate(before, after).values())
    after["metrics"]["hand_mae_u"] = 1.021
    gate = support.overfit_gate(before, after)
    assert not gate["hand_within_2pct"] and gate["flow_improved_10pct"]


def test_failure_result_is_exclusive_no_retry_or_optimizer(monkeypatch, tmp_path):
    def fail(*args):
        raise ValueError("missing preflight gate")

    monkeypatch.setattr(training, "context_for", fail)
    args = SimpleNamespace(output=tmp_path / "attempt", command="overfit", wall_seconds=10)
    with pytest.raises(ValueError, match="preflight"):
        training.execute(args)
    result = training.read(args.output / "OVERFIT_RESULT.json")
    assert (
        result["status"] == "failed"
        and result["continuation_gate"] is False
        and result["automatic_retry"] is False
    )
    with pytest.raises(FileExistsError):
        training.execute(args)


def test_frozen_shared_schedules_have_no_condition_specific_rng_or_future_prefix():
    from contract import sha

    root = Path(
        "/home/sl/research/icra27-artifacts/g1-pretrained-causal-mamba-memory-training-r1/plan-r1"
    )
    if not root.exists():
        pytest.skip("Frozen local experiment plan unavailable")
    plan = training.read(root / "PLAN.json")
    assert sha(root / "PLAN.json") == training.PLAN_SHA
    for name, expected in training.SCHEDULE_SHAS.items():
        assert sha(root / name) == expected
        rows = training.read(root / name)
        assert len(rows) == (200 if name.startswith("OVERFIT") else 500)
        for index, row in enumerate(rows, 1):
            assert row["step"] == index and row["prefix"] == list(range(0, row["anchor"] + 1, 16))
            assert row["anchor"] in plan["grids"][str(row["episode"])]
            assert set(row) == {"step", "episode", "anchor", "prefix", "forward_seed", "lr"}
        assert rows[0]["lr"] == pytest.approx(1e-5)
        assert rows[9]["lr"] == pytest.approx(1e-4)
        assert rows[-1]["lr"] == pytest.approx(1e-5)


def test_cache_command_captures_raw_before_mutating_postprocess(monkeypatch, tmp_path):
    from contextlib import nullcontext

    import numpy as np
    from transformers.feature_extraction_utils import BatchFeature

    ids = torch.full((1, 209), 151655, dtype=torch.long)
    ids[0, training.prototype.NONIMAGE_POSITIONS] = torch.tensor(training.prototype.NONIMAGE_IDS)
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "image_grid_thw": torch.tensor([[1, 16, 16]] * 3),
        "state": torch.ones(1, 1, 13),
        "action": torch.zeros(1, 40, 13),
        "action_mask": torch.cat([torch.ones(1, 16, 13), torch.zeros(1, 24, 13)], dim=1),
        "embodiment_id": torch.tensor([10]),
    }
    physical = SimpleNamespace(
        states={"left_arm": np.zeros((1, 7)), "left_hand": np.zeros((1, 6))},
        actions={"left_arm": np.zeros((16, 7)), "left_hand": np.zeros((16, 6))},
    )

    class Head:
        def _encode_features(self, raw, action):
            raw.backbone_features.mul_(2)
            action.state.mul_(4)
            return BatchFeature(
                data={"backbone_features": raw.backbone_features, "state_features": action.state}
            )

    class Model:
        device = "cpu"
        action_head = Head()

        def prepare_input(self, values):
            return values, BatchFeature(data={k: values[k] for k in support.ACTION_KEYS})

        def backbone(self, values):
            return BatchFeature(
                data={
                    "backbone_features": torch.full((1, 209, 2048), 3.0),
                    "backbone_attention_mask": values["attention_mask"] == 1,
                    "image_mask": values["input_ids"] == 151655,
                }
            )

    source = SimpleNamespace(
        batch=lambda pairs: (batch, [physical]),
        modality={"video": SimpleNamespace(modality_keys=training.prototype.CAMERAS)},
    )
    context = SimpleNamespace(
        model=Model(),
        source=source,
        plan={"train": [125], "dev": [], "grids": {"125": [0]}},
        args=SimpleNamespace(output=tmp_path),
    )
    monkeypatch.setattr(training, "budget", lambda _: 0)
    monkeypatch.setattr(training, "close_integrity", lambda _: {})
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    result = training.cache_command(context)
    cached = torch.load(tmp_path / "episode-000125.pt", map_location="cpu", weights_only=True)[0]
    assert result["observations"] == 1 and result["raw_post_separate"] is True
    assert torch.all(cached["raw"]["backbone_features"] == 3)
    assert torch.all(cached["post_vl_features"] == 6)
    assert torch.all(cached["action_input"]["state"] == 1)
    assert torch.all(batch["state"] == 1)
    assert support.frame_hash(cached) == cached["content_sha256"]
