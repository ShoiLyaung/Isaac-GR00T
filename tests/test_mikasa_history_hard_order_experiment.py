from __future__ import annotations

import copy
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


def test_cue_substitution_rolls_only_cue_and_recomputes_recipient_delta() -> None:
    temporal = torch.arange(6 * 130 * 2 * 8 * 8 * 9, dtype=torch.float32).reshape(
        6, 130, 2, 8, 8, 9
    )
    current_rgb = torch.arange(6 * 2 * 8 * 8 * 3, dtype=torch.float32).reshape(6, 2, 8, 8, 3)
    current_proprio = torch.arange(6 * 7, dtype=torch.float32).reshape(6, 7)
    labels = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
    batch = experiment.OrderBatch(temporal, current_rgb, current_proprio)
    before_temporal = temporal.clone()
    before_rgb = current_rgb.clone()
    before_proprio = current_proprio.clone()

    substituted, substituted_labels = experiment.cue_only_substitution(batch, labels)

    donor_cue = before_temporal[..., :3].roll(shifts=-1, dims=0)
    recipient_current = before_temporal[..., 3:6]
    assert torch.equal(substituted.temporal_grid[..., :3], donor_cue)
    assert torch.equal(substituted.temporal_grid[..., 3:6], recipient_current)
    assert torch.equal(substituted.temporal_grid[..., 6:9], donor_cue - recipient_current)
    assert torch.equal(substituted.current_rgb_grid, before_rgb)
    assert torch.equal(substituted.current_proprio, before_proprio)
    assert torch.equal(substituted_labels, labels.roll(shifts=-1, dims=0))
    assert torch.equal(batch.temporal_grid, before_temporal)
    assert torch.equal(batch.current_rgb_grid, before_rgb)
    assert torch.equal(batch.current_proprio, before_proprio)


def test_noninferiority_uses_per_metric_max_when_controls_win_different_metrics() -> None:
    evidence = _passing_evidence()
    evidence["standard"]["gru_full"] = _section(0.94, 0.95, 0.99)
    evidence["standard"]["transformer_full"] = _section(0.93, 0.99, 0.96)
    evidence["standard"]["mamba_full"] = _section(0.93, 0.965, 0.98)

    gates = experiment.dev_gates(evidence)

    assert gates["full_backbone_noninferior_pair_accuracy"]
    assert not gates["full_backbone_noninferior_slot0_accuracy"]
    assert gates["full_backbone_noninferior_slot1_accuracy"]


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
    assert experiment.DEV_WALL_SECONDS == 15 * 60.0
    assert experiment.DEV_ARTIFACT_LIMIT == 64 * 1024 * 1024
    assert runner.PHASES == ("source_acceptance", "dev")
    assert runner.protocol()["contract"] == "evaluator-r2"
    assert runner.protocol()["retry_count"] == 0


def test_lr_has_exact_warmup_and_cosine_endpoints() -> None:
    assert experiment.lr_multiplier(0, 1500) == pytest.approx(0.01)
    assert experiment.lr_multiplier(99, 1500) == pytest.approx(1.0)
    assert experiment.lr_multiplier(1499, 1500) == pytest.approx(0.1)


def test_source_acceptance_exactly_owns_new_files_and_binds_overlay() -> None:
    assert runner.BASE_COMMIT == "6f9bf220132fc896d3ec0ce3decd13d316bd0692"
    assert len(runner.SOURCE_PATHS) == 7
    assert runner.EVALUATOR_REPAIR_PATHS == (
        "gr00t/experimental/mikasa_history_hard_order_experiment.py",
        "gr00t/experimental/run_mikasa_history_hard_order.py",
        "tests/test_mikasa_history_hard_order_experiment.py",
    )
    assert all(
        path.startswith("gr00t/experimental/") or path.startswith("tests/")
        for path in runner.SOURCE_PATHS
    )
    overlay = runner.verify_overlay(import_package=False)
    assert overlay["version"] == "2.3.2.post1"
    assert overlay["causal_conv1d_absent"] is True
    assert set(overlay["materials"]) == {"mamba2.py", "selective_scan_cuda.so", "RECORD"}
    training = runner.training_source_identity()
    assert training["head_commit"] == runner.BASE_COMMIT
    assert training["matched_training_result_sha256"] == runner.TRAINING_MATCHED_RESULT_SHA256


@pytest.mark.parametrize("drift", ("dirty", "head", "file"))
def test_runtime_source_drift_fails_before_cache_target_or_model_access(
    drift: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = {
        "training_source_commit": runner.TRAINING_SOURCE_COMMIT,
        "head_commit": "evaluator-head",
        "branch": "codex/evaluator",
        "exact_diff": {path: "M" for path in runner.EVALUATOR_REPAIR_PATHS},
        "files": {"material.py": {"bytes": 7, "sha256": "1" * 64}},
        "owned_source_clean": True,
    }
    source = {
        "schema": f"{runner.EXPERIMENT_ID}/source-acceptance-v1",
        "status": "passed",
        "mode": "source_acceptance",
        "source_identity": accepted,
        "evaluator_source_identity": accepted,
        "training_source_identity": runner.training_source_identity(),
        "protocol": runner.protocol(),
    }

    def current_identity(_repo: Path) -> dict:
        if drift == "dirty":
            raise RuntimeError("native-order evaluator source is dirty")
        current = copy.deepcopy(accepted)
        if drift == "head":
            current["head_commit"] = "changed-head"
        else:
            current["files"]["material.py"]["sha256"] = "2" * 64
        return current

    downstream_calls: list[str] = []
    monkeypatch.setattr(runner, "source_identity", current_identity)
    monkeypatch.setattr(
        runner,
        "_read_receipt",
        lambda *_args, **_kwargs: (source, {"sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        runner,
        "_cache_identity",
        lambda *_args, **_kwargs: downstream_calls.append("cache"),
    )

    output_root = tmp_path / "dev-output"
    argv = [
        "--phase",
        "dev",
        "--output-root",
        str(output_root),
        "--repo-root",
        str(tmp_path),
        "--source-result",
        str(tmp_path / "source.json"),
        "--source-sha256",
        "a" * 64,
        "--cache-root",
        str(tmp_path / "cache"),
        "--cache-sha256",
        runner.TRAINING_CACHE_RESULT_SHA256,
        "--matched-root",
        str(tmp_path / "matched"),
        "--matched-sha256",
        runner.TRAINING_MATCHED_RESULT_SHA256,
    ]
    with pytest.raises((PermissionError, RuntimeError)):
        runner.main(argv)
    assert downstream_calls == []
    assert not output_root.exists()


def test_exact_runtime_source_and_frozen_training_lineage_are_both_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = {
        "training_source_commit": runner.TRAINING_SOURCE_COMMIT,
        "head_commit": "evaluator-head",
        "branch": "codex/evaluator",
        "exact_diff": {path: "M" for path in runner.EVALUATOR_REPAIR_PATHS},
        "files": {"material.py": {"bytes": 7, "sha256": "1" * 64}},
        "owned_source_clean": True,
    }
    source = {
        "schema": f"{runner.EXPERIMENT_ID}/source-acceptance-v1",
        "source_identity": evaluator,
        "evaluator_source_identity": evaluator,
        "training_source_identity": runner.training_source_identity(),
        "protocol": runner.protocol(),
    }
    monkeypatch.setattr(runner, "source_identity", lambda _repo: copy.deepcopy(evaluator))
    assert runner.bind_runtime_source(source, Path("/unused")) == evaluator

    cache = {"source_acceptance": {"sha256": runner.TRAINING_SOURCE_RESULT_SHA256}}
    matched = {
        "provenance": {
            "source_acceptance": {"sha256": runner.TRAINING_SOURCE_RESULT_SHA256},
            "cache": {"sha256": runner.TRAINING_CACHE_RESULT_SHA256},
        }
    }
    runner.validate_training_lineage(cache, matched)
    matched["provenance"]["cache"]["sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="lineage"):
        runner.validate_training_lineage(cache, matched)


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


def test_dev_limits_fail_closed_before_result_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "dev"
    output.mkdir()
    oversized = {"payload": "x" * 32}
    monkeypatch.setattr(experiment, "DEV_ARTIFACT_LIMIT", 1)
    with pytest.raises(RuntimeError, match="64 MiB"):
        experiment._dev_result_payload(output, oversized, wall_seconds=0.0)
    monkeypatch.setattr(experiment, "DEV_ARTIFACT_LIMIT", 64 * 1024 * 1024)
    with pytest.raises(TimeoutError, match="15 minutes"):
        experiment._dev_result_payload(
            output, {"status": "passed"}, wall_seconds=experiment.DEV_WALL_SECONDS + 1.0
        )
    assert not (output / "RESULT.json").exists()
