from __future__ import annotations

from collections import Counter

from gr00t.experimental.mikasa_native_rgb_memory_runtime import GroupArrays, TargetStats
import gr00t.experimental.mikasa_segment_slot_train_calibration as calibration
import numpy as np
import pytest


TRAIN_GROUPS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    10,
    11,
    14,
    16,
    18,
    19,
    20,
    22,
    23,
    24,
    26,
    28,
    31,
    33,
    34,
    35,
)


def test_frozen_split_selection_hash_and_salted_ranking_are_exact() -> None:
    value = calibration.validate_frozen_split(TRAIN_GROUPS)
    assert calibration.SPLIT_SHA256 == (
        "c9fd3d603d02a73182e5649c6a9a9477af4f6758d398d9126acdc844b33246d2"
    )
    assert tuple(value["fit_groups"]) == calibration.FIT_GROUPS
    assert tuple(value["calibration_groups"]) == calibration.CALIBRATION_GROUPS
    ranked = sorted(
        [group_id for group_id in TRAIN_GROUPS if group_id != 0],
        key=calibration.selection_digest,
        reverse=True,
    )[:6]
    assert set(ranked) == set(calibration.CALIBRATION_GROUPS)
    assert 0 in calibration.FIT_GROUPS


def test_fit_schedule_is_balanced_and_never_contains_calibration() -> None:
    schedule = calibration.FitGroupBalancedSchedule(seed=230923)
    counts = schedule.cycle_counts(0)
    assert counts == Counter({group_id: 2 for group_id in calibration.FIT_GROUPS})
    assert all(len(set(schedule.batch(step))) == 4 for step in range(18))
    assert not set(calibration.CALIBRATION_GROUPS) & set(counts)
    assert schedule.batch(0) == calibration.FitGroupBalancedSchedule(seed=230923).batch(0)


def test_fit_data_rejects_calibration_before_loading_feature_or_target() -> None:
    data = calibration.FitOnlyData.__new__(calibration.FitOnlyData)
    data.group_ids = calibration.FIT_GROUPS
    data.features = {}
    data.targets = {}
    with pytest.raises(PermissionError, match="calibration"):
        data.batch((0, 2, 3, 1))


class _TargetCorpus:
    split = "train"

    def __init__(self):
        self.opened = []

    def load_target_group(self, group_id: int) -> GroupArrays:
        self.opened.append(group_id)
        value = np.full((6, 2, 12, 2), float(group_id), dtype=np.float32)
        return GroupArrays(group_id, {}, {"ordered_checkpoints": value})


def test_stats_and_target_mean_use_fit_groups_only() -> None:
    corpus = _TargetCorpus()
    stats = TargetStats.fit(calibration.FitTargetCorpus(corpus))
    mean = calibration.fit_target_mean(corpus)
    assert np.array_equal(stats.mean, mean)
    assert set(corpus.opened) == set(calibration.FIT_GROUPS)
    assert not set(corpus.opened) & set(calibration.CALIBRATION_GROUPS)


def _passing_evidence() -> dict:
    records = {
        seed: [
            {
                "group_id": group_id,
                "first_shape_accuracy": 1.0,
                "second_shape_accuracy": 1.0,
                "ordered_pair_accuracy": 1.0,
                "first_mae_m": 0.02,
                "complete_mae_m": 0.02,
                "swap_shape_exact": 1.0,
                "finite": True,
            }
            for group_id in calibration.CALIBRATION_GROUPS
        ]
        for seed in calibration.TRAIN_SEEDS
    }
    baseline_records = [
        {
            "group_id": group_id,
            "first_mae_m": 0.04,
            "complete_mae_m": 0.04,
        }
        for group_id in calibration.CALIBRATION_GROUPS
    ]
    return {
        "records": records,
        "aggregates": {
            "seed_equal": {
                "first_shape_accuracy": 1.0,
                "second_shape_accuracy": 1.0,
                "ordered_pair_accuracy": 1.0,
                "first_mae_m": 0.02,
                "complete_mae_m": 0.02,
                "swap_shape_exact": np.float64(1.0),
            }
        },
        "train_target_mean_control": {
            "records": baseline_records,
            "group_equal": {"first_mae_m": 0.04, "complete_mae_m": 0.04},
        },
    }


def test_all_frozen_calibration_gates_and_physical_threshold() -> None:
    evidence = _passing_evidence()
    gates = calibration.evaluate_calibration_gates(evidence)
    assert gates and all(gates.values())
    assert all(type(value) is bool for value in gates.values())
    changed = _passing_evidence()
    changed["aggregates"]["seed_equal"]["first_mae_m"] = 0.036
    failed = calibration.evaluate_calibration_gates(changed)
    assert failed["first_mae_at_most_0p035m"] is False


def test_optimizer_contract_constants_are_inherited_unchanged() -> None:
    assert calibration.TRAIN_SEEDS == (230923, 230924, 230925)
    assert calibration.TRAIN_STEPS == 1500
    assert calibration.LEARNING_RATE == 3e-4
    assert calibration.BETAS == (0.9, 0.95)
    assert calibration.WEIGHT_DECAY == 1e-4
    assert calibration.WARMUP_STEPS == 100
    assert calibration.FINAL_LEARNING_RATE == 3e-5
    assert calibration.GRADIENT_CLIP == 1.0
