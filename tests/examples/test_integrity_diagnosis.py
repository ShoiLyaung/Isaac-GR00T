"""CPU-only canonical-scope, read-only RNG and first-mutation stop tests."""

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
from action_transfer_program_trajectory import TraceCore  # noqa:E402
import integrity_diagnosis as runner  # noqa:E402
from integrity_diagnosis_support import BaseAudit, BaseMutation, difference  # noqa:E402
import memory_prototype as proto  # noqa:E402


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Module()
        self.action_head.model = nn.Linear(2, 2)
        self.action_head.action_encoder = nn.Linear(2, 2)
        self.action_head.action_decoder = nn.Linear(2, 2)
        self.register_buffer("persistent", torch.arange(4, dtype=torch.float32))
        self.eval().requires_grad_(False)


def audit(model, root):
    root.mkdir()
    return BaseAudit(
        model,
        root,
        proto.content_hash(model.state_dict()),
        chunk_bytes=4,
        samples=4,
        synchronize=lambda: None,
    )


def test_legal_trace_scope_uses_original_owners_then_checks_restored_keys(tmp_path):
    model = Base()
    check = audit(model, tmp_path / "a")
    original = model.action_head.model
    with patch.object(model.action_head, "model", TraceCore(original)):
        check.check("inside_trace", stable=False)
    check.check("restored", stable=True)
    assert not check.failed
    with patch.object(model.action_head, "model", TraceCore(original)):
        with pytest.raises(BaseMutation):
            check.check("unexpected_stable_wrapper", stable=True)
    value = json.loads((tmp_path / "a/FIRST_MUTATION.json").read_text())
    assert value["structural_change"]["keys"]["added"] and value["structural_change"]["bindings"]


def test_passed_cpu_audit_changes_neither_rng_nor_values(tmp_path):
    model = Base()
    check = audit(model, tmp_path / "a")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state().clone()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    check.check("readonly", stable=True)
    assert python == random.getstate() and np.array_equal(numpy[1], np.random.get_state()[1])
    assert numpy[2:] == np.random.get_state()[2:] and torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_mutation_saves_bounded_fragments_hashes_and_aliases(tmp_path):
    model = Base()
    check = audit(model, tmp_path / "a")
    check.extra = lambda: {"accidental_alias": model.persistent}
    model.persistent[2] += 1
    with pytest.raises(BaseMutation):
        check.check("after_write", stable=True)
    report = json.loads((tmp_path / "a/FIRST_MUTATION.json").read_text())
    row = next(r for r in report["changed_tensors"] if r["name"] == "persistent")
    assert (
        row["before"]["kind"] == "persistent_buffer"
        and row["before"]["sha256"] != row["after"]["sha256"]
    )
    assert (
        row["difference"]["changed_bytes"] > 0
        and len(row["difference"]["byte_offsets_sample"]) <= 4
    )
    assert report["alias_information"]["base_external_storage_overlaps"]
    fragments = torch.load(tmp_path / "a/DIFFERENCE_FRAGMENTS.pt", weights_only=True)
    assert fragments["persistent"]["before"].numel() <= 4
    assert not report["full_base_saved_to_disk"]


def test_chunk_difference_scans_all_bytes_but_bounds_saved_samples():
    a = torch.zeros(100, dtype=torch.uint8)
    b = torch.ones_like(a)
    value = difference(a, b, chunk_bytes=7, samples=5)
    assert value["changed_bytes"] == 100 and value["byte_offsets_sample"] == [0, 1, 2, 3, 4]


def test_first_mutation_stops_remaining_matrix(monkeypatch, tmp_path):
    plan = {
        "run_id": "toy",
        "cache_report_sha256": "cache",
        "diagnostic_matrix": [
            {"name": "first", "variant": "teacher_trajectory_mixed"},
            {"name": "must_not_run", "variant": "data_line_mixed"},
        ],
    }
    args = SimpleNamespace(output=tmp_path, plan_sha256="plan")
    monkeypatch.setattr(runner, "environment", lambda plan: {"commit": "toy"})
    monkeypatch.setattr(
        runner.original, "import_parent", lambda *a: (None, {"completed_step": 20000}, {})
    )
    called = []

    def entry(args, plan, entry, parent, source, deadline, totals):
        called.append(entry["name"])
        raise BaseMutation("first exact tensor difference")

    result = runner.run_matrix(args, plan, 0, entry_runner=entry)
    assert called == ["first"] and result["status"] == "mutation_reproduced"
    assert result["formal_optimizer_steps"] == 0 and not result["historical_r1_requalified"]


def test_after_release_mutation_evidence_survives_none_samples(tmp_path):
    model = Base()
    check = audit(model, tmp_path / "release")
    ctx = SimpleNamespace(samples=None, observed={"observed_alias": model.persistent})
    check.extra = lambda: runner.extra_tensors(ctx)
    model.persistent[0] += 1
    with pytest.raises(BaseMutation, match="after_release"):
        check.check("after_release", stable=True)
    report = json.loads((tmp_path / "release/FIRST_MUTATION.json").read_text())
    assert (
        report["stage"] == "after_release"
        and report["alias_information"]["base_external_storage_overlaps"]
    )
    assert (tmp_path / "release/DIFFERENCE_FRAGMENTS.pt").exists()
