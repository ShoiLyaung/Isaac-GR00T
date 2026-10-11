from __future__ import annotations

import gr00t.experimental.mikasa_event_trajectory as event
from gr00t.experimental.mikasa_native_rgb_memory_runtime import GroupArrays, MaterialIdentity
import numpy as np
import pytest


def test_zero_motion_and_stable_top64_energy_centroid() -> None:
    assert event.extraction_contract() == {
        "schema": "g1-mikasa-event-trajectory-sufficiency-r1/extraction-v1",
        "top_pixels": 64,
        "slot_boundaries": [[0, 66], [66, 130]],
        "tokens_per_slot": 65,
        "descriptor_dim": 1567,
        "epsilon": 1e-6,
        "cameras": 2,
    }
    previous = np.zeros((128, 128, 3), np.uint8)
    token, valid = event.stable_top64_token(previous, previous.copy())
    np.testing.assert_array_equal(token, np.asarray((0.5, 0.5, 0.0), np.float32))
    assert valid is False

    current = previous.copy()
    current.reshape(-1, 3)[:65] = 255
    token, valid = event.stable_top64_token(previous, current)
    assert valid is True
    selected = np.arange(64)
    y, x = np.divmod(selected, 128)
    assert token[0] == pytest.approx(float(np.mean((x + 0.5) / 128)))
    assert token[1] == pytest.approx(float(np.mean((y + 0.5) / 128)))


def test_segment_local_transitions_padding_and_canonicalization() -> None:
    rgb = np.zeros((130, 128, 128, 3), np.uint8)
    for frame in range(130):
        rgb[frame, frame % 128, 0] = 255
    raw, canonical, valid = event.extract_row_camera(rgb)
    assert raw.shape == (2, 65, 3)
    assert canonical.shape == (2, 65, 2)
    assert valid.shape == (2, 65)
    assert valid[0].sum() == 65
    assert valid[1].sum() == 63
    assert np.array_equal(raw[1, 63:], np.broadcast_to(raw[1, 62], (2, 3)))
    assert not valid[1, 63:].any()
    assert np.all(canonical[1, 63:] == 0)


def test_endpoint_recanonicalizes_to_zero_and_shuffle_is_group_deterministic() -> None:
    rng = np.random.default_rng(2)
    arrays = {
        "raw": rng.random((6, 2, 2, 65, 3), dtype=np.float32),
        "canonical": np.zeros((6, 2, 2, 65, 2), np.float32),
        "valid": np.ones((6, 2, 2, 65), np.bool_),
        "post_proprio": np.zeros((6, 7), np.float32),
    }
    endpoint, _ = event.condition_view(arrays, "endpoint_repeat", 4)
    fields = endpoint[:, :-7].reshape(6, 2, 2, 65, 6)
    assert np.allclose(fields[..., 3:5], 0, atol=1e-6)
    first, permutation1 = event.condition_view(arrays, "shuffle_230923", 4)
    second, permutation2 = event.condition_view(arrays, "shuffle_230923", 4)
    assert np.array_equal(first, second)
    assert np.array_equal(permutation1, permutation2)
    assert not np.array_equal(permutation1, event.group_permutations(230923, 5))
    assert first.shape[1] == endpoint.shape[1] == event.DESCRIPTOR_DIM


def test_sha_seed_contract_is_target_independent() -> None:
    assert event.shuffle_seed(230923, 4, 0, 0) == event.shuffle_seed(230923, 4, 0, 0)
    assert event.shuffle_seed(230923, 4, 0, 0) != event.shuffle_seed(230923, 4, 0, 1)


def test_nearest_neighbor_ties_break_by_group_then_row() -> None:
    query = np.zeros(2, np.float32)
    fit = np.zeros((3, 2), np.float32)
    distance, group, row = event.lexicographic_neighbor(query, fit, [(4, 3), (2, 5), (2, 1)])
    assert distance == 0.0
    assert (group, row) == (2, 1)


class _ScoreCorpus:
    split = "train"

    def __init__(self):
        self.opens = []

    def load_target_group(self, group_id: int) -> GroupArrays:
        self.opens.append(group_id)
        shapes = np.tile(np.asarray([[0, 1]], np.int64), (6, 1))
        checkpoints = np.zeros((6, 2, 12, 2), np.float32)
        return GroupArrays(
            group_id,
            {},
            {"ordered_shape_ids": shapes, "ordered_checkpoints": checkpoints},
        )


def test_score_opens_all_fit_targets_before_calibration_once(tmp_path) -> None:
    materials = {}
    for condition in event.CONDITIONS:
        payload = {
            "mean": np.zeros(event.DESCRIPTOR_DIM, np.float32),
            "std": np.ones(event.DESCRIPTOR_DIM, np.float32),
            "neighbor_group": np.full(36, event.FIT_GROUPS[0], np.int64),
            "neighbor_row": np.zeros(36, np.int64),
            "distance": np.zeros(36, np.float32),
            "cal_group": np.repeat(event.CALIBRATION_GROUPS, 6).astype(np.int64),
            "cal_row": np.tile(np.arange(6), 6).astype(np.int64),
        }
        path = tmp_path / f"{condition}.npz"
        body = event._npz_bytes(payload)
        path.write_bytes(body)
        materials[condition] = MaterialIdentity(
            path, len(body), __import__("hashlib").sha256(body).hexdigest()
        )
    corpus = _ScoreCorpus()
    result = event.score_mapping(event.FrozenMapping(tmp_path, materials), corpus)
    assert corpus.opens == list(event.FIT_GROUPS) * 2 + list(event.CALIBRATION_GROUPS)
    assert set(result["conditions"]) == set(event.CONDITIONS)


def test_mapping_material_validation_rejects_negative_distance_and_wrong_cal_order(
    tmp_path,
) -> None:
    payload = {
        "mean": np.zeros(event.DESCRIPTOR_DIM, np.float32),
        "std": np.ones(event.DESCRIPTOR_DIM, np.float32),
        "neighbor_group": np.full(36, event.FIT_GROUPS[0], np.int64),
        "neighbor_row": np.zeros(36, np.int64),
        "distance": np.zeros(36, np.float32),
        "cal_group": np.repeat(event.CALIBRATION_GROUPS, 6).astype(np.int64),
        "cal_row": np.tile(np.arange(6), 6).astype(np.int64),
    }
    path = tmp_path / "ordered.npz"
    path.write_bytes(event._npz_bytes(payload))
    event.validate_mapping_material("ordered", path)
    payload["distance"][0] = -1
    path.write_bytes(event._npz_bytes(payload))
    with pytest.raises(ValueError, match="nonnegative"):
        event.validate_mapping_material("ordered", path)


def test_event_cache_load_validates_shapes_dtypes_and_canonical_mask(tmp_path) -> None:
    arrays = {
        "raw": np.zeros((6, 2, 2, 65, 3), np.float32),
        "canonical": np.zeros((6, 2, 2, 65, 2), np.float32),
        "valid": np.zeros((6, 2, 2, 65), np.bool_),
        "post_proprio": np.zeros((6, 7), np.float32),
    }
    path = tmp_path / "events.npz"
    body = event._npz_bytes(arrays)
    path.write_bytes(body)
    material = MaterialIdentity(path, len(body), __import__("hashlib").sha256(body).hexdigest())
    cache = event.EventCache.__new__(event.EventCache)
    cache.materials = {0: material}
    assert cache.load_group(0)["raw"].shape == (6, 2, 2, 65, 3)
    arrays["canonical"][0, 0, 0, 0, 0] = 1
    body = event._npz_bytes(arrays)
    path.write_bytes(body)
    cache.materials[0] = MaterialIdentity(
        path, len(body), __import__("hashlib").sha256(body).hexdigest()
    )
    with pytest.raises(ValueError, match="zero canonical"):
        cache.load_group(0)
