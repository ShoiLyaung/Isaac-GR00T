from __future__ import annotations

import json
from pathlib import Path

from gr00t.experimental import (
    mikasa_history_hard_order_experiment as experiment,
    run_mikasa_history_hard_order as runner,
)
import pytest
import torch


def _aggregates(pair: float, slot0: float, slot1: float) -> dict:
    metrics = {
        "pair_accuracy": pair,
        "slot0_accuracy": slot0,
        "slot1_accuracy": slot1,
        "cross_entropy": 0.1,
    }
    return {
        "by_seed": {
            str(seed): {
                "groups": {str(group): dict(metrics) for group in experiment.DEV_GROUPS},
                "group_equal": dict(metrics),
            }
            for seed in experiment.TRAIN_SEEDS
        },
        "seed_equal": dict(metrics),
    }


def _section(pair: float, slot0: float, slot1: float) -> dict:
    records = {
        seed: [
            {
                "group_id": group,
                "pair_accuracy": pair,
                "slot0_accuracy": slot0,
                "slot1_accuracy": slot1,
                "cross_entropy": 0.1,
                "finite": True,
            }
            for group in experiment.DEV_GROUPS
        ]
        for seed in experiment.TRAIN_SEEDS
    }
    return {"records": records, "aggregates": _aggregates(pair, slot0, slot1)}


def _passing_evidence() -> dict:
    standard = {
        "mamba_current": _section(0.20, 0.60, 0.60),
        "mamba_zero": _section(0.10, 0.55, 0.55),
        "mamba_shuffle": _section(0.25, 0.65, 0.65),
        "gru_full": _section(0.90, 0.95, 0.95),
        "transformer_full": _section(0.91, 0.96, 0.96),
        "mamba_full": _section(0.92, 0.97, 0.97),
    }
    return {
        "standard": standard,
        "matched_finite": True,
        "mamba_full_interventions": {
            "substitute": _section(0.85, 0.92, 0.92),
            "zero": _section(0.20, 0.60, 0.60),
            "shuffle": _section(0.25, 0.60, 0.60),
        },
    }


def test_dev_gate_aggregation_passes_and_each_primary_gate_can_fail() -> None:
    evidence = _passing_evidence()
    gates = experiment.dev_gates(evidence)
    assert gates and all(gates.values())
    evidence["standard"]["mamba_full"] = _section(0.82, 0.97, 0.97)
    assert not experiment.dev_gates(evidence)["mamba_full_pair_at_least_83_33"]


def test_frozen_recipe_schedule_and_phase_order() -> None:
    assert experiment.TRAIN_SEEDS == (230923, 230924, 230925)
    assert experiment.PREFLIGHT_STEPS == 20
    assert experiment.OVERFIT_STEPS == 500
    assert experiment.TRAIN_STEPS == 1500
    assert experiment.LEARNING_RATE == 3e-4
    assert experiment.FINAL_LEARNING_RATE == 3e-5
    assert experiment.WARMUP_STEPS == 100
    assert experiment.BETAS == (0.9, 0.95)
    assert experiment.WEIGHT_DECAY == 1e-4
    assert experiment.GRADIENT_CLIP == 1.0
    assert runner.PHASES == (
        "source_acceptance",
        "cache",
        "preflight",
        "overfit",
        "matched",
        "dev",
    )
    assert runner.protocol()["retry_count"] == 0


def test_lr_has_exact_warmup_and_cosine_endpoints() -> None:
    assert experiment.lr_multiplier(0, 1500) == pytest.approx(0.01)
    assert experiment.lr_multiplier(99, 1500) == pytest.approx(1.0)
    assert experiment.lr_multiplier(1499, 1500) == pytest.approx(0.1)


def test_source_acceptance_exactly_owns_new_files_and_binds_overlay() -> None:
    assert runner.BASE_COMMIT == "dfd928df5b7c398b34b1e1c5326cdefe07e13385"
    assert len(runner.SOURCE_PATHS) == 7
    assert all(
        path.startswith("gr00t/experimental/") or path.startswith("tests/")
        for path in runner.SOURCE_PATHS
    )
    overlay = runner.verify_overlay(import_package=False)
    assert overlay["version"] == "2.3.2.post1"
    assert overlay["causal_conv1d_absent"] is True
    assert set(overlay["materials"]) == {"mamba2.py", "selective_scan_cuda.so", "RECORD"}


def test_existing_source_output_rejected_before_git_or_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    git_calls = []
    monkeypatch.setattr(runner, "source_identity", lambda *_args: git_calls.append(True))
    with pytest.raises(FileExistsError):
        runner.run_source_acceptance(output)
    assert git_calls == []


def test_receipt_tamper_and_failed_gate_rejected(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    path.write_text(json.dumps({"status": "passed", "mode": "overfit"}))
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="identity"):
        runner._read_receipt(path, "0" * 64, "overfit")


def test_gradient_validator_requires_every_parameter_gradient() -> None:
    module = torch.nn.Linear(3, 2)
    assert not experiment.module_has_finite_nonzero_gradient(module)
    module(torch.ones(1, 3)).sum().backward()
    assert experiment.module_has_finite_nonzero_gradient(module)
    next(module.parameters()).grad.fill_(float("nan"))
    assert not experiment.module_has_finite_nonzero_gradient(module)
