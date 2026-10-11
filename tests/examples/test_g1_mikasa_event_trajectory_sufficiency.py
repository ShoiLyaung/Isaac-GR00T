from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

from gr00t.experimental.mikasa_event_trajectory import CALIBRATION_GROUPS, CONDITIONS, gates
from gr00t.experimental.mikasa_native_rgb_memory_runtime import canonical_json_bytes, write_new_json
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_MIKASA_MEMORY"))
runner = importlib.import_module("event_trajectory_sufficiency")


def _records(shape: float, order: float, mae: float) -> list[dict]:
    return [
        {
            "group_id": group,
            "first_shape_accuracy": shape,
            "second_shape_accuracy": shape,
            "ordered_pair_accuracy": order,
            "first_mae_m": mae,
            "complete_mae_m": mae,
        }
        for group in CALIBRATION_GROUPS
    ]


def test_absolute_relative_group_gates_and_serialization() -> None:
    ordered_records = _records(1.0, 1.0, 0.02)
    conditions = {
        "ordered": {
            "records": ordered_records,
            "group_equal": {
                "first_shape_accuracy": 1.0,
                "second_shape_accuracy": 1.0,
                "ordered_pair_accuracy": 1.0,
                "first_mae_m": 0.02,
                "complete_mae_m": 0.02,
            },
        }
    }
    for condition in CONDITIONS[1:]:
        conditions[condition] = {
            "records": _records(0.8, 0.7, 0.03),
            "group_equal": {
                "first_shape_accuracy": 0.8,
                "second_shape_accuracy": 0.8,
                "ordered_pair_accuracy": 0.7,
                "first_mae_m": 0.03,
                "complete_mae_m": 0.03,
            },
        }
    baseline = _records(0, 0, 0.05)
    evidence = {
        "conditions": conditions,
        "fit_target_mean_control": {"first_mae_m": 0.05, "complete_mae_m": 0.05},
        "baseline_records": baseline,
        "technical": {
            "mapping_descriptors_distances_finite": True,
            "metrics_finite": True,
        },
    }
    result = gates(evidence)
    assert result and all(result.values())
    assert all(type(value) is bool for value in result.values())
    assert json.loads(canonical_json_bytes(result)) == result


def test_smoke_write_once_and_path_rejection(tmp_path: Path) -> None:
    assert runner.smoke()["valid"] is True
    result = tmp_path / "RESULT.json"
    write_new_json(result, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_new_json(result, {"status": "changed"})
    for name in ("dev", "confirmation"):
        with pytest.raises(PermissionError):
            runner.reject_path(tmp_path / name / "secret")


def test_runner_exposes_exact_three_cpu_stages() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'mode.add_argument("--cache"' in source
    assert 'mode.add_argument("--map"' in source
    assert 'mode.add_argument("--score"' in source
    assert "verify_privileged_targets=False" in source
