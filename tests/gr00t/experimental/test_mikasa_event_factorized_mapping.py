from __future__ import annotations

import hashlib
import inspect

import gr00t.experimental.mikasa_event_factorized_mapping as factorized
from gr00t.experimental.mikasa_event_trajectory import gates as fused_gates
from gr00t.experimental.mikasa_native_rgb_memory_runtime import GroupArrays, MaterialIdentity
import numpy as np


def _arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    return {
        "raw": rng.random((6, 2, 2, 65, 3), dtype=np.float32),
        "canonical": np.zeros((6, 2, 2, 65, 2), np.float32),
        "valid": np.ones((6, 2, 2, 65), np.bool_),
        "post_proprio": rng.random((6, 7), dtype=np.float32),
    }


def test_exact_factorized_masks_dimensions_and_field_separation() -> None:
    arrays = _arrays()
    arrays["valid"][0, 0, 0, 0] = False
    shape, checkpoint, _ = factorized.factorized_views(arrays, "ordered", 0)
    assert shape.shape == (6, 780)
    assert checkpoint.shape == (6, 1047)
    shape_fields = shape[:, :780].reshape(6, 2, 2, 65, 3)
    checkpoint_fields = checkpoint[:, :-7].reshape(6, 2, 2, 65, 4)
    assert np.all(shape_fields[0, 0, 0, 0, :2] == 0)
    assert shape_fields[0, 0, 0, 0, 2] == 0
    assert np.all(checkpoint_fields[0, 0, 0, 0, :3] == 0)
    assert checkpoint_fields[0, 0, 0, 0, 3] == 0
    assert not np.shares_memory(shape, checkpoint)


def test_endpoint_recanonicalizes_shape_view_to_zero() -> None:
    shape, _, _ = factorized.factorized_views(_arrays(), "endpoint_repeat", 0)
    fields = shape.reshape(6, 2, 2, 65, 3)
    assert np.allclose(fields[..., :2], 0, atol=1e-6)


def test_shape_and_checkpoint_neighbors_are_frozen_independently() -> None:
    shape_fit = np.asarray([[0.0], [10.0]], np.float32)
    checkpoint_fit = np.asarray([[10.0], [0.0]], np.float32)
    query = np.asarray([[0.1]], np.float32)
    ids = [(2, 0), (3, 0)]
    shape = factorized._freeze_view(shape_fit, query, ids)
    checkpoint = factorized._freeze_view(checkpoint_fit, query, ids)
    assert shape["neighbor_group"].tolist() == [2]
    assert checkpoint["neighbor_group"].tolist() == [3]


def test_equal_distance_ties_are_lexicographic_for_each_independent_view() -> None:
    fit_ids = [(4, 5), (2, 3), (2, 1)]
    query = np.zeros((1, 2), np.float32)
    shape = factorized._freeze_view(np.zeros((3, 2), np.float32), query, fit_ids)
    checkpoint = factorized._freeze_view(np.zeros((3, 2), np.float32), query.copy(), fit_ids)
    assert (shape["neighbor_group"][0], shape["neighbor_row"][0]) == (2, 1)
    assert (
        checkpoint["neighbor_group"][0],
        checkpoint["neighbor_row"][0],
    ) == (2, 1)
    assert shape["neighbor_group"] is not checkpoint["neighbor_group"]


def test_mapping_freeze_source_contains_no_target_access() -> None:
    assert "load_target_group" not in inspect.getsource(factorized.freeze_factorized_mapping)


def test_gate_reuse_is_exact_functional_alias() -> None:
    assert factorized.unchanged_gates.__code__.co_names == ("gates",)
    assert fused_gates is factorized.gates


class _TargetCorpus:
    split = "train"

    def __init__(self):
        self.opens = []

    def load_target_group(self, group: int) -> GroupArrays:
        self.opens.append(group)
        return GroupArrays(
            group,
            {},
            {
                "ordered_shape_ids": np.tile(np.asarray([[0, 1]], np.int64), (6, 1)),
                "ordered_checkpoints": np.zeros((6, 2, 12, 2), np.float32),
            },
        )


def test_score_uses_independent_neighbors_and_fit_then_cal_target_order(tmp_path) -> None:
    materials = {}
    for condition in factorized.CONDITIONS:
        payload = {
            "cal_group": np.repeat(factorized.CALIBRATION_GROUPS, 6).astype(np.int64),
            "cal_row": np.tile(np.arange(6), 6).astype(np.int64),
            "shape_neighbor_group": np.full(36, factorized.FIT_GROUPS[0], np.int64),
            "shape_neighbor_row": np.zeros(36, np.int64),
            "checkpoint_neighbor_group": np.full(36, factorized.FIT_GROUPS[1], np.int64),
            "checkpoint_neighbor_row": np.ones(36, np.int64),
        }
        path = tmp_path / f"{condition}.npz"
        body = factorized._npz_bytes(payload)
        path.write_bytes(body)
        materials[condition] = MaterialIdentity(path, len(body), hashlib.sha256(body).hexdigest())
    corpus = _TargetCorpus()
    result = factorized.score_factorized(factorized.FactorizedMapping(tmp_path, materials), corpus)
    assert corpus.opens == list(factorized.FIT_GROUPS) * 2 + list(factorized.CALIBRATION_GROUPS)
    assert set(result["conditions"]) == set(factorized.CONDITIONS)
