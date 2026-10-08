"""CPU contracts for migrated action core, request-local cache and velocity KD."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import budgeted_action_runner as runner  # noqa: E402
import budgeted_action_support as support  # noqa: E402
from diffusers.models.attention import FeedForward  # noqa: E402
from gr00t.model.modules import budgeted_action_mamba as module  # noqa: E402
from gr00t.model.modules.dit import AdaLayerNorm, TimestepEncoder  # noqa: E402


class FakeOfficialMamba(nn.Module):
    """CPU shape-compatible test double, never a runtime fallback."""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        assert d_model == 512 and kwargs["expand"] == 1 and kwargs["ngroups"] == 1
        self.in_proj = nn.Linear(512, 1160, bias=False)
        self.conv1d = nn.Conv1d(640, 640, 4, groups=640, bias=True)
        self.A_log = nn.Parameter(torch.zeros(8))
        self.D = nn.Parameter(torch.ones(8))
        self.dt_bias = nn.Parameter(torch.zeros(8))
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(512))
        self.out_proj = nn.Linear(512, 512, bias=False)
        self.calls = []

    def forward(self, value):
        self.calls.append(value.detach().clone())
        return self.out_proj(self.in_proj(value)[..., :512])


@pytest.fixture
def fake_official(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(module, "_mamba2_class", lambda: FakeOfficialMamba)


def test_narrow_bidirection_projects_once_and_shares_complete_mixer(fake_official):
    mixer = module.NarrowBidirectionalMixer()
    counts = {"down": 0, "up": 0}
    handles = []
    for name in counts:

        def count(_m, args, name=name):
            counts[name] += 1

        handles.append(getattr(mixer, name).register_forward_pre_hook(count))
    value = torch.randn(1, 41, 1536)
    result = mixer(value)
    for handle in handles:
        handle.remove()
    assert counts == {"down": 1, "up": 1}
    assert len(mixer.mamba.calls) == 2
    assert torch.equal(mixer.mamba.calls[0], mixer.mamba.calls[1].flip(1))
    assert result.shape == (1, 41, 1536)
    result.square().mean().backward()
    assert mixer.down.weight.grad is not None and torch.isfinite(mixer.down.weight.grad).all()
    assert mixer.up.weight.grad is not None and torch.isfinite(mixer.up.weight.grad).all()
    assert not hasattr(mixer, "state")


def test_condition_adapter_uses_fixed_permutation_and_preserves_graph():
    torch.set_num_threads(2)
    adapter = module.ActionConditionAdapter()
    features = torch.randn(1, 209, 2048)
    condition = adapter(features)
    tokens = torch.nn.functional.silu(adapter.token(features[:, list(module.PERMUTATION)]))
    expected = torch.nn.functional.silu(adapter.code(tokens.flatten(1)))
    assert torch.equal(condition.code, expected)
    assert condition.modulation.shape == (1, 8, 2, 1536)
    condition.modulation.square().mean().backward()
    assert adapter.token.weight.grad is not None
    assert tuple(module.PERMUTATION[:192]) == module.IMAGE_POSITIONS
    assert len(set(module.PERMUTATION)) == 209
    assert not hasattr(adapter, "cache")
    with pytest.raises(ValueError, match="fixed-layout"):
        adapter(torch.zeros(1, 210, 2048))


class NativeCopyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = AdaLayerNorm(1536)
        self.norm3 = nn.LayerNorm(1536, elementwise_affine=False)
        self.ff = FeedForward(
            1536, dropout=0.2, activation_fn="gelu-approximate", final_dropout=True
        )
        self.final_dropout = nn.Dropout(0.2)


def test_exact_migration_parameter_budget_and_frozen_teacher_ownership(fake_official):
    # Native-compatible source sections only; no VLM/attention/model is invoked.
    teacher = nn.Module()
    teacher.inner_dim = 1536
    teacher.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(32)])
    for index in module.SELECTED_LAYERS:
        teacher.transformer_blocks[index] = NativeCopyBlock().bfloat16()
    teacher.timestep_encoder = TimestepEncoder(1536).bfloat16()
    teacher.norm_out = nn.LayerNorm(1536, elementwise_affine=False, eps=1e-6)
    teacher.proj_out_1 = nn.Linear(1536, 3072).bfloat16()
    teacher.proj_out_2 = nn.Linear(1536, 1024).bfloat16()
    teacher.eval().requires_grad_(False)
    versions = {name: p._version for name, p in teacher.named_parameters()}
    student = module.BudgetedActionMamba(teacher)
    assert sum(p.numel() for p in student.parameters()) == 230_785_600
    assert all(p.dtype == torch.float32 and p.requires_grad for p in student.parameters())
    audit = support.transfer_audit(student, teacher)
    assert audit["copied_tensors"] == 56 and audit["copied_parameters"] == 197_880_832
    assert all(
        p._version == versions[name] and not p.requires_grad
        for name, p in teacher.named_parameters()
    )
    assert not any("attention" in type(m).__name__.lower() for m in student.modules())
    assert student.proj_out_2.out_features == 1024
    assert student.condition_adapter.modulation.out_features == 8 * 2 * 1536
    # Original native layer remains a distinct object/tensor, not a shared training alias.
    assert student.blocks[0].ff is not teacher.transformer_blocks[3].ff
    assert (
        next(student.blocks[0].ff.parameters()).data_ptr()
        != next(teacher.transformer_blocks[3].ff.parameters()).data_ptr()
    )


def test_native_mask_applied_before_square_and_teacher_detached():
    predicted = torch.ones(1, 40, 13, requires_grad=True)
    teacher = torch.zeros(1, 40, 13, requires_grad=True)
    mask = torch.zeros(1, 40, 13)
    mask[:, :16] = 1
    with torch.no_grad():
        predicted[:, 16:] = float("inf")
        teacher[:, 16:] = float("nan")
    value = support.masked_velocity_kd(predicted, teacher, mask)
    assert value.dtype == torch.float32 and torch.isfinite(value)
    assert float(value) == pytest.approx(208 / (208 + 1e-6))
    value.backward()
    assert teacher.grad is None and not torch.count_nonzero(predicted.grad[:, 16:])
    assert torch.isfinite(predicted.grad).all()


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.inputs = []

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        return_all_hidden_states=True,
        **kwargs,
    ):
        self.inputs.append(
            (
                hidden_states.detach().clone(),
                encoder_hidden_states.detach().clone(),
                timestep.detach().clone(),
            )
        )
        output = self.linear(hidden_states)
        return (output, [output]) if return_all_hidden_states else output


class ToyStudent(ToyCore):
    def encode_condition(self, features):
        return module.ActionCondition(features.mean(1), features.mean(1))


class ToyDecoder(nn.Linear):
    def forward(self, value, embodiment):
        return super().forward(value)


def test_single_native_draw_teacher_exact_replay_and_decoder_gradient(monkeypatch):
    teacher = ToyCore().eval().requires_grad_(False)
    student = ToyStudent()
    decoder = ToyDecoder(4, 13).eval().requires_grad_(False)
    head = nn.Module()
    head.model = teacher
    head.action_decoder = decoder
    model = SimpleNamespace(action_head=head, device="cpu")
    frame = {
        "post_vl_features": torch.ones(1, 209, 4),
        "action_input": {"embodiment_id": torch.tensor([10])},
    }
    draws = []

    def native_flow(model, frame, residual, seed):
        draws.append(seed)
        kwargs = {
            "hidden_states": torch.arange(164, dtype=torch.float32).reshape(1, 41, 4) / 100,
            "encoder_hidden_states": frame["post_vl_features"],
            "timestep": torch.tensor([234]),
            "return_all_hidden_states": True,
        }
        output, _ = model.action_head.model(**kwargs)
        prediction = model.action_head.action_decoder(
            output, frame["action_input"]["embodiment_id"]
        )[:, -40:]
        mask = torch.zeros_like(prediction)
        mask[:, :16] = 1
        return {"loss": prediction.square().mean(), "action_mask": mask}, {"seed": seed}

    monkeypatch.setattr(support, "native_flow", native_flow)
    result = support.capture_velocity(
        model, frame, 77, student=student, teacher=teacher, replay_teacher=True
    )
    assert draws == [77]
    assert len(student.inputs) == len(teacher.inputs) == 1
    assert all(torch.equal(a, b) for a, b in zip(student.inputs[0], teacher.inputs[0]))
    assert result["velocity"].requires_grad and not result["teacher_velocity"].requires_grad
    result["kd"].backward()
    assert student.linear.weight.grad is not None and torch.count_nonzero(
        student.linear.weight.grad
    )
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in decoder.parameters())
    assert head.model is teacher and not decoder._forward_hooks


def test_request_binding_restores_on_exception():
    teacher = ToyCore()
    student = ToyStudent()
    head = nn.Module()
    head.model = teacher
    model = SimpleNamespace(action_head=head)
    condition = module.ActionCondition(torch.zeros(1, 4), torch.zeros(1, 4))
    with pytest.raises(RuntimeError):
        with support.bind_request(model, student, condition) as request:
            assert request.condition is condition
            raise RuntimeError("intentional request failure")
    assert head.model is teacher and not hasattr(student, "condition")


def test_pilot_api_cannot_launch_before_implementation(tmp_path):
    args = SimpleNamespace(command="train", output=tmp_path / "train/output")
    with pytest.raises(NotImplementedError, match="preflight/overfit"):
        runner.execute(args)
    assert not args.output.exists()


def test_missing_cost_gate_closes_optimizer_entry(tmp_path):
    args = SimpleNamespace(artifact_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.require_preflight(args)


def test_clean_backward_cost_has_no_evidence_hooks_hashes_or_inner_reseed(monkeypatch, tmp_path):
    from contextlib import nullcontext

    import budgeted_action_evaluation as evaluation

    teacher = ToyCore()
    student = ToyStudent()

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = teacher
            self.decoder = ToyDecoder(4, 13).requires_grad_(False)

        def forward(self, raw, action):
            output, _ = self.model(
                hidden_states=raw["hidden"],
                encoder_hidden_states=raw["features"],
                timestep=action["time"],
                return_all_hidden_states=True,
            )
            prediction = self.decoder(output, action["embodiment_id"])
            return {"loss": prediction.float().square().mean()}

    head = Head()
    model = SimpleNamespace(action_head=head, device="cpu")
    frame = {
        "raw": {"hidden": torch.randn(1, 41, 4), "features": torch.ones(1, 209, 4)},
        "action_input": {"time": torch.tensor([234]), "embodiment_id": torch.tensor([10])},
        "post_vl_features": torch.ones(1, 209, 4),
    }

    def forbidden(*a, **k):
        raise AssertionError("Evidence/seed/hook operation entered clean cost")

    for target, name in (
        (support, "capture_velocity"),
        (support, "native_flow"),
        (support, "tensor_identity"),
        (support, "set_seed"),
        (evaluation, "capture_velocity"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_hook", forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", forbidden)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    clock = {"inside": False, "ticks": 0, "seeds": 0}

    def tick():
        clock["inside"] = not clock["inside"]
        clock["ticks"] += 1
        return clock["ticks"] * 0.001

    def seed(*a, **k):
        assert not clock["inside"]
        clock["seeds"] += 1

    monkeypatch.setattr(evaluation.time, "perf_counter", tick)
    monkeypatch.setattr(evaluation, "set_seed", seed)

    def activate(context, mode, backward=False):
        teacher.requires_grad_(mode == "native").eval()
        student.requires_grad_(mode == "student").eval()

    monkeypatch.setattr(evaluation, "activate", activate)
    monkeypatch.setattr(evaluation, "restore_teacher", lambda context: None)
    context = SimpleNamespace(
        args=SimpleNamespace(output=tmp_path),
        plan={"fb_input": {"episode": 150, "anchor": 336}},
        teacher=teacher,
        student=student,
        model=model,
        budget=lambda: None,
    )
    report = evaluation.backward_cost(context, SimpleNamespace(frame=lambda ep, anchor: frame))
    assert report["timings"]["native"]["count"] == report["timings"]["student"]["count"] == 20
    assert clock["seeds"] == 50 and not clock["inside"]
    assert head.model is teacher
    assert len(teacher.inputs) == len(student.inputs) == 25
    assert all(p.grad is None for p in head.decoder.parameters())
    # Explicit gradient assertion after the same uninstrumented helper; the
    # cost loop clears its gradients at each block boundary by design.
    student.requires_grad_(True)
    loss = support.clean_task_loss(model, frame, student=student)
    loss.backward()
    assert student.linear.weight.grad is not None and torch.count_nonzero(
        student.linear.weight.grad
    )
    assert head.model is teacher
