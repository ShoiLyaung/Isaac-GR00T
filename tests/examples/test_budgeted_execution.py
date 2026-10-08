"""CPU tests for fixed-weight casting, instrumentation cleanup and timing contracts."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import budgeted_action_execution_probe as probe  # noqa: E402
from budgeted_action_profile import ExecutionScopes, event_row, profiler_summary  # noqa: E402
from gr00t.model.modules.budgeted_action_mamba import ActionCondition  # noqa: E402


class CastCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.blocks = nn.ModuleList()
        for _ in range(8):
            block = nn.Module()
            block.mixer = nn.Module()
            block.mixer.mamba = nn.Module()
            for name in ("A_log", "D", "dt_bias"):
                block.mixer.mamba.register_parameter(
                    name, nn.Parameter(torch.tensor([0.123456789, 1.00123456]))
                )
            self.blocks.append(block)
        self.register_buffer("floating_buffer", torch.tensor([0.123456789]))
        self.register_buffer("integer_buffer", torch.tensor([17], dtype=torch.int64))


def test_derived_bf16_preserves24_sensitive_parameters_without_master_mutation():
    original = CastCore().eval()
    before = {k: v.clone() for k, v in original.state_dict().items()}
    candidate = probe.derived_bf16(original)
    assert all(
        torch.equal(value, before[name]) and value.dtype == before[name].dtype
        for name, value in original.state_dict().items()
    )
    kept = 0
    for name, value in candidate.named_parameters():
        if name.endswith(probe.SENSITIVE_SUFFIXES):
            kept += 1
            assert value.dtype == torch.float32 and torch.equal(value, original.get_parameter(name))
        else:
            assert value.dtype == torch.bfloat16 and torch.equal(
                value, original.get_parameter(name).bfloat16()
            )
        assert value.data_ptr() != original.get_parameter(name).data_ptr()
        assert value.requires_grad
    assert kept == 24
    assert candidate.floating_buffer.dtype == torch.bfloat16
    assert candidate.integer_buffer.dtype == torch.int64
    assert (
        probe.module_identity(candidate)["state_content_sha256"]
        != probe.module_identity(original)["state_content_sha256"]
    )


def test_missing_sensitive_parameter_layout_fails_closed():
    with pytest.raises(ValueError, match="24"):
        probe.derived_bf16(nn.Linear(3, 4))


def test_cyclic_three_mode_order_is_frozen():
    assert probe.order_for(0) == probe.MODES
    assert probe.order_for(1) == ("student_original", "student_bf16_storage", "native")
    assert probe.order_for(2) == ("student_bf16_storage", "native", "student_original")
    assert probe.order_for(3) == probe.MODES


@pytest.mark.parametrize("use_student", [True, False])
def test_segmented_policy_uses_sequential_events_and_one_explicit_condition(
    monkeypatch, use_student
):
    sequence = []
    event_count = [0]
    active = [None]

    class Event:
        def __init__(self, enable_timing):
            self.index = event_count[0]
            event_count[0] += 1

        def record(self):
            if self.index % 2 == 0:
                assert active[0] is None
                active[0] = self.index // 2
                sequence.append(("start", active[0]))
            else:
                assert active[0] == self.index // 2

        def synchronize(self):
            assert active[0] == self.index // 2
            sequence.append(("end", active[0]))
            active[0] = None

        def elapsed_time(self, other):
            return 0.5

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sequence.append(("total_sync", None)))
    native = nn.Identity()

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.encodes = 0

        def encode_condition(self, value):
            self.encodes += 1
            return ActionCondition(value, value)

    student = Student()
    features = SimpleNamespace(backbone_features=torch.ones(1, 209, 2))
    head = SimpleNamespace(model=native, _encode_features=lambda backbone, action: features)
    model = SimpleNamespace(
        prepare_input=lambda observation: ("raw", "action"),
        backbone=lambda raw: "encoded",
        action_head=head,
    )
    expected = torch.ones(1, 40, 13)

    def sample(model, packet):
        if use_student:
            assert model.action_head.model.student is student
        else:
            assert model.action_head.model is native
        assert packet["features"] is features
        return expected

    monkeypatch.setattr(probe.proto, "sample_packet", sample)
    actual, total, stages = probe.segmented_policy(model, {}, student if use_student else None)
    assert actual is expected and total > 0
    assert list(stages) == ["H2D", "VLM", "VL_q", "condition", "Euler4"]
    assert sequence == [("total_sync", None)] + [
        item for i in range(5) if use_student or i != 3 for item in [("start", i), ("end", i)]
    ]
    assert student.encodes == int(use_student) and model.action_head.model is native
    assert all(
        row["cuda_event_ms"] == 0.5
        for name, row in stages.items()
        if use_student or name != "condition"
    )
    if not use_student:
        assert stages["condition"] == {"wall_ms": 0.0, "cuda_event_ms": 0.0}


class ProfileHead(nn.Module):
    def __init__(self):
        super().__init__()
        for name in (
            "vlln",
            "vl_self_attention",
            "state_encoder",
            "action_encoder",
            "action_decoder",
        ):
            setattr(self, name, nn.Linear(2, 2))

    def _encode_features(self, value):
        return self.state_encoder(self.vl_self_attention(self.vlln(value)))


class ProfileMamba(nn.Module):
    def __init__(self, official):
        super().__init__()
        self.official = official
        for name in ("in_proj", "conv1d", "norm", "out_proj"):
            setattr(self, name, nn.Linear(2, 2))

    def forward(self, value):
        value = self.norm(self.conv1d(self.in_proj(value)))
        return self.out_proj(self.official.mamba_chunk_scan_combined(value))


class ProfileCore(nn.Module):
    def __init__(self, official):
        super().__init__()
        self.condition_adapter = nn.Linear(2, 2)
        block = nn.Module()
        block.norm1 = nn.Linear(2, 2)
        block.ff = nn.Linear(2, 2)
        block.mixer = nn.Module()
        block.mixer.down = nn.Linear(2, 2)
        block.mixer.up = nn.Linear(2, 2)
        block.mixer.mamba = ProfileMamba(official)
        self.blocks = nn.ModuleList([block])

    def forward(self, value):
        value = self.condition_adapter(value)
        block = self.blocks[0]
        return block.ff(block.mixer.up(block.mixer.mamba(block.mixer.down(block.norm1(value)))))


def profile_fixture():
    official = SimpleNamespace(mamba_chunk_scan_combined=lambda value: value * 2)
    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.action_head = ProfileHead()
    core = ProfileCore(official)
    return official, model, core


def test_profiler_scopes_keep_output_and_restore_hooks_namespace():
    official, model, core = profile_fixture()
    value = torch.randn(1, 2)
    scan = official.mamba_chunk_scan_combined
    encode = model.action_head._encode_features.__func__
    expected = core(model.action_head._encode_features(model.backbone(value)))
    with ExecutionScopes(model, core, student=True, official_module=official) as scopes:
        actual = core(model.action_head._encode_features(model.backbone(value)))
    assert torch.equal(actual, expected)
    assert scopes.counts["official_scan"] == 1 and scopes.counts["condition"] == 1
    assert scopes.metadata["layer0/Mamba/in_proj"]["args"][0]["dtype"] == "torch.float32"
    assert official.mamba_chunk_scan_combined is scan
    assert model.action_head._encode_features.__func__ is encode
    assert not any(scopes.active.values())
    assert all(
        not m._forward_pre_hooks and not m._forward_hooks
        for root in (model, core)
        for m in root.modules()
    )


def test_profiler_scopes_restore_after_forward_exception():
    official, model, core = profile_fixture()

    def fail(value):
        raise RuntimeError("intentional scan error")

    official.mamba_chunk_scan_combined = fail
    with pytest.raises(RuntimeError, match="intentional"):
        with ExecutionScopes(model, core, student=True, official_module=official) as scopes:
            core(torch.ones(1, 2))
    assert official.mamba_chunk_scan_combined is fail
    assert not any(scopes.active.values())
    assert all(
        not m._forward_pre_hooks and not m._forward_hooks
        for root in (model, core)
        for m in root.modules()
    )


def test_profile_labels_device_events_and_keeps_inclusive_self_distinct():
    event = SimpleNamespace(
        key="aten::_to_copy",
        count=3,
        self_cpu_time_total=4,
        cpu_time_total=7,
        device_time_total=8,
        self_device_time_total=2,
    )
    row = event_row(event)
    assert row["self_device_us"] == 2 and row["inclusive_device_us"] == 8
    cuda = SimpleNamespace(
        device_type="DeviceType.CUDA",
        name="Memcpy HtoD",
        time_range=SimpleNamespace(elapsed_us=lambda: 9),
    )
    profiler = SimpleNamespace(key_averages=lambda: [event], events=lambda: [cuda])
    report = profiler_summary(profiler, SimpleNamespace(counts={}, metadata={}))
    assert report["cuda_device_events"] == 1
    assert report["cuda_device_event_top40"][0]["name"] == "Memcpy HtoD"
    assert "Memcpy/Memset" in report["scope_warning"]
    assert report["cast_copy_ops"][0]["name"] == "aten::_to_copy"


def test_numeric_failure_does_not_reopen_training_even_if_costs_pass():
    limits = {
        "cast_max_p95_ratio_vs_original": 0.9,
        "cast_max_p95_ratio_vs_native": 0.9,
        "cast_max_allocated_ratio_vs_native": 1.0,
    }
    context = SimpleNamespace(plan={"decision_limits": limits, "cost_inputs": [{"episode": 150}]})
    clean = {
        150: {
            "native": {"p95_ms": 100, "peak_allocated_bytes": 100},
            "student_original": {"p95_ms": 110, "peak_allocated_bytes": 110},
            "student_bf16_storage": {"p95_ms": 80, "peak_allocated_bytes": 90},
        }
    }
    report = probe.decisions(context, {"numeric_gate": False}, clean)
    assert report["cast_10pct_faster_all_inputs"] and report["cast_native_cost_gate_all_inputs"]
    assert not report["dtype_adoption_eligible"] and not report["old_training_gate_opened"]


def test_timing_summary_excludes_warmup_but_peak_includes_it():
    rows = [
        {"warmup": True, "wall_ms": 50, "peak_allocated_bytes": 200},
        {"warmup": False, "wall_ms": 1, "peak_allocated_bytes": 100},
        {"warmup": False, "wall_ms": 3, "peak_allocated_bytes": 150},
    ]
    result = probe.summary(rows)
    assert (
        result["count"] == 2 and result["mean_ms"] == 2 and result["p95_ms"] == pytest.approx(2.9)
    )
    assert result["peak_allocated_bytes"] == 200
