from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    EXPECTED_SPLIT_GROUPS,
    MaterialIdentity,
)
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


def test_cache_builder_has_no_target_access_and_receipt_binds_all_materials(
    tmp_path: Path,
) -> None:
    assert "load_target_group" not in inspect.getsource(cache.build_temporal_moment_caches)
    records, manifests, sources = {}, {}, {}
    source_records = {}
    condition_records = []
    for group_id in EXPECTED_SPLIT_GROUPS["train"]:
        group_root = tmp_path / "cache/endpoint/train" / f"group_{group_id:03d}"
        group_root.mkdir(parents=True)
        feature_path = group_root / "features.npz"
        manifest_path = group_root / "MANIFEST.json"
        feature_path.write_bytes(f"feature-{group_id}".encode())
        manifest_path.write_bytes(f"manifest-{group_id}".encode())
        feature = MaterialIdentity(
            feature_path,
            feature_path.stat().st_size,
            hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        )
        manifest = MaterialIdentity(
            manifest_path,
            manifest_path.stat().st_size,
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        source_sha = hashlib.sha256(f"source-{group_id}".encode()).hexdigest()
        records[group_id], manifests[group_id], sources[group_id] = feature, manifest, source_sha
        source_records[group_id] = SimpleNamespace(sha256=source_sha)
        condition_records.append(
            {
                "group_id": group_id,
                "source_feature_sha256": source_sha,
                "feature": {
                    "path": str(feature_path.resolve()),
                    "bytes": feature.bytes,
                    "sha256": feature.sha256,
                },
                "manifest": {
                    "path": str(manifest_path.resolve()),
                    "bytes": manifest.bytes,
                    "sha256": manifest.sha256,
                },
            }
        )
    permutation_path = tmp_path / "cache/PERMUTATIONS.npz"
    permutation_path.parent.mkdir(exist_ok=True)
    permutation_path.write_bytes(b"permutations")
    permutation = MaterialIdentity(
        permutation_path,
        permutation_path.stat().st_size,
        hashlib.sha256(permutation_path.read_bytes()).hexdigest(),
    )
    hashes = tuple(cache.array_sha256(value) for value in cache.frozen_permutations())
    source_cache = SimpleNamespace(root=tmp_path / "source", _records=source_records)
    instance = cache.TemporalMomentCache(
        tmp_path / "cache",
        "endpoint",
        records,
        manifests,
        sources,
        hashes,
        permutation,
        source_cache.root,
    )
    result = {
        "status": "passed",
        "mode": "temporal_moment_cache",
        "cache_root": str((tmp_path / "cache").resolve()),
        "conditions": list(cache.CONDITIONS),
        "targets_opened": False,
        "original_dev_opened": False,
        "confirmation_opened": False,
        "permutation_sha256": list(hashes),
        "permutation_identity": {
            "path": str(permutation_path.resolve()),
            "bytes": permutation.bytes,
            "sha256": permutation.sha256,
        },
        "condition_records": {
            "endpoint": condition_records,
            "ordered_moment": [],
            "shuffled_moment": [],
        },
    }
    instance.verify_receipt(result, source_cache)
    result["permutation_identity"]["sha256"] = "0" * 64
    try:
        instance.verify_receipt(result, source_cache)
    except ValueError as error:
        assert "permutation material" in str(error)
    else:
        raise AssertionError("mutated permutation receipt unexpectedly passed")
