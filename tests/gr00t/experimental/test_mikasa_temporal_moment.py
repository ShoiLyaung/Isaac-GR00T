from __future__ import annotations

import hashlib

import gr00t.experimental.mikasa_temporal_moment_cache as cache
import numpy as np


def test_centered_unit_rms_weights_and_frozen_pcg64_permutations() -> None:
    for length in (66, 64):
        value = cache.time_weights(length)
        assert value.dtype == np.float32
        assert abs(float(value.mean())) < 1e-6
        assert abs(float(np.sqrt(np.mean(value**2))) - 1.0) < 1e-6
    first = cache.frozen_permutations()
    second = cache.frozen_permutations()
    for slot, value in enumerate(first):
        assert np.array_equal(value, second[slot])
        assert sorted(value.tolist()) == list(range((66, 64)[slot]))
        expected = np.random.Generator(np.random.PCG64(230923 + slot)).permutation((66, 64)[slot])
        assert np.array_equal(value, expected)


def test_endpoint_ordered_and_shuffled_formulas_preserve_channels_zero_to_six() -> None:
    source = np.random.default_rng(4).normal(size=(6, 130, 2, 8, 8, 9)).astype(np.float32)
    permutations = cache.frozen_permutations()
    endpoint = cache.transform_temporal_grid(
        source, condition="endpoint", permutations=permutations
    )
    ordered = cache.transform_temporal_grid(
        source, condition="ordered_moment", permutations=permutations
    )
    shuffled = cache.transform_temporal_grid(
        source, condition="shuffled_moment", permutations=permutations
    )
    for value in (endpoint, ordered, shuffled):
        assert np.array_equal(value[..., :6], source[..., :6])
    for slot, (start, stop) in enumerate(cache.SLOT_BOUNDARIES):
        difference = source[:, start:stop, ..., 6:9]
        expected_endpoint = np.broadcast_to(difference[:, -1:], difference.shape)
        assert np.array_equal(endpoint[:, start:stop, ..., 6:9], expected_endpoint)
        weights = cache.time_weights(stop - start)
        np.testing.assert_array_equal(
            ordered[:, start:stop, ..., 6:9],
            difference * weights[None, :, None, None, None, None],
        )
        np.testing.assert_array_equal(
            shuffled[:, start:stop, ..., 6:9],
            difference * weights[permutations[slot]][None, :, None, None, None, None],
        )


def test_permutation_array_hash_binds_dtype_shape_and_content() -> None:
    first, second = cache.frozen_permutations()
    assert cache.array_sha256(first) == cache.array_sha256(first.copy())
    changed = first.copy()
    changed[[0, 1]] = changed[[1, 0]]
    assert cache.array_sha256(first) != cache.array_sha256(changed)
    assert cache.array_sha256(first) != cache.array_sha256(second)


def test_old_slot_source_remains_frozen() -> None:
    path = (
        cache.Path(__file__).resolve().parents[3]
        / "gr00t/experimental/mikasa_segment_slot_bottleneck.py"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "1cc259d06be74957a55f00f93dcbb99ad927b15b82f404e64052609f321f6042"
