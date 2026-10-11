from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    GroupArrays,
    GroupBalancedSchedule,
    TargetStats,
    canonical_json_bytes,
    serialize_checkpoint,
    write_new_bytes,
    write_new_json,
)
from gr00t.experimental.mikasa_segment_slot_bottleneck import MikasaSegmentSlotModel
import gr00t.experimental.mikasa_segment_slot_experiment as experiment
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_MIKASA_MEMORY"
sys.path.insert(0, str(EXAMPLE))
runner = importlib.import_module("segment_slot_bottleneck_diagnosis")


def test_cpu_synthetic_backward_is_finite_swap_exact_and_vla_free() -> None:
    result = runner.synthetic_cpu_smoke()
    assert result["status"] == "passed"
    assert result["shape"] == [2, 2, 3]
    assert result["checkpoint_shape"] == [2, 2, 24]
    assert result["finite"] is True
    assert result["swap_shape_exact"] is True
    assert result["swap_checkpoint_exact"] is True
    assert result["mamba_loaded"] is False
    assert result["native_vla_loaded"] is False


class _StatsCorpus:
    def __init__(self, split: str):
        self.split = split

    def iter_target_groups(self):
        target = np.arange(6 * 48, dtype=np.float32).reshape(6, 2, 12, 2)
        yield GroupArrays(0, {}, {"ordered_checkpoints": target})
        yield GroupArrays(1, {}, {"ordered_checkpoints": target + 1.0})


def test_train_only_stats_group_balancing_and_three_seed_identity() -> None:
    with pytest.raises(ValueError, match="train split"):
        TargetStats.fit(_StatsCorpus("dev"))
    stats = TargetStats.fit(_StatsCorpus("train"))
    assert stats.mean.shape == (48,)
    schedule = GroupBalancedSchedule(tuple(range(24)), seed=230923)
    assert len(schedule.batch(0)) == 4
    assert len(set(schedule.batch(0))) == 4
    assert experiment.TRAIN_SEEDS == (230923, 230924, 230925)
    assert experiment.OVERFIT_STEPS == 500
    assert experiment.TRAIN_STEPS == 1500


def _passing_gate_evidence() -> tuple[dict, dict]:
    dev_groups = (9, 12, 17, 25, 27, 29)
    records = {
        seed: [
            {
                "group_id": group_id,
                "first_mae_m": 0.05,
                "complete_mae_m": 0.05,
                "finite": True,
            }
            for group_id in dev_groups
        ]
        for seed in experiment.TRAIN_SEEDS
    }
    evidence = {
        "records": records,
        "aggregates": {
            "seed_equal": {
                "first_shape_accuracy": 0.95,
                "second_shape_accuracy": 0.95,
                "ordered_pair_accuracy": 0.90,
                "first_mae_m": 0.05,
                "complete_mae_m": 0.05,
                "swap_shape_exact": np.float64(0.90),
            }
        },
    }
    dense_records = {
        str(seed): [
            {"group_id": group_id, "first_mae_m": 0.07, "complete_mae_m": 0.07}
            for group_id in dev_groups
        ]
        for seed in experiment.TRAIN_SEEDS
    }
    dense = {
        "records": {"mamba_full": dense_records},
        "aggregates": {"mamba_full": {"seed_equal": {"first_mae_m": 0.07, "complete_mae_m": 0.07}}},
    }
    return evidence, dense


def test_metric_gate_aggregation_is_serializable_and_maps_outcome() -> None:
    evidence, dense = _passing_gate_evidence()
    gates = experiment.evaluate_dev_gates(evidence, dense)
    assert gates and all(gates.values())
    assert all(type(value) is bool for value in gates.values())
    assert json.loads(canonical_json_bytes(gates)) == gates
    assert experiment.outcome_from_gates(gates) == "event_readout_bottleneck_supported"

    failed = dict(gates)
    failed["first_shape_accuracy_at_least_90pct"] = False
    assert experiment.outcome_from_gates(failed) == "spatial_representation_bottleneck"
    association = dict(gates)
    association["ordered_pair_accuracy_at_least_80pct"] = False
    assert experiment.outcome_from_gates(association) == "slot_association_or_objective_invalid"


def test_write_once_and_confirmation_paths_are_rejected(tmp_path: Path) -> None:
    result = tmp_path / "RESULT.json"
    write_new_json(result, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_new_json(result, {"status": "changed"})
    with pytest.raises(PermissionError, match="confirmation"):
        runner._reject_confirmation_path(tmp_path / "confirmation" / "secret")


def test_final_checkpoint_is_write_once_and_carries_exact_step_identity(tmp_path: Path) -> None:
    model = MikasaSegmentSlotModel(seed=230923)
    stats = TargetStats(np.zeros(48, np.float32), np.ones(48, np.float32))
    payload = experiment._checkpoint_payload(model, stats, seed=230923, steps=500)
    value = serialize_checkpoint(payload)
    path = tmp_path / "checkpoint_final.pt"
    write_new_bytes(path, value)
    with pytest.raises(FileExistsError):
        write_new_bytes(path, value)
    assert payload["seed"] == 230923
    assert payload["steps"] == 500


def test_runner_has_no_confirmation_mode_and_uses_final_checkpoints_only() -> None:
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    experiment_source = Path(experiment.__file__).read_text(encoding="utf-8")
    assert 'mode.add_argument("--confirmation"' not in runner_source
    assert "import mamba_ssm" not in runner_source
    assert "from gr00t.model" not in runner_source
    assert "import gr00t.model" not in runner_source
    assert '"checkpoint_final.pt"' in experiment_source
    assert "checkpoint-{step}" not in experiment_source
