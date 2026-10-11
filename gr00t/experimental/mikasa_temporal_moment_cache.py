"""Target-blind temporal endpoint and first-moment cache transforms."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    EXPECTED_SPLIT_GROUPS,
    FEATURE_KEYS,
    FeatureCacheCorpus,
    FeatureGroup,
    MaterialIdentity,
    ShardedMemoryCorpus,
    write_new_bytes,
    write_new_json,
)


CONDITIONS: Final = ("endpoint", "ordered_moment", "shuffled_moment")
SLOT_BOUNDARIES: Final = ((0, 66), (66, 130))
PERMUTATION_SEEDS: Final = (230923, 230924)
CACHE_SCHEMA: Final = "g1-mikasa-temporal-moment-sufficiency-r1/cache-v1"


def time_weights(length: int) -> np.ndarray:
    coordinate = np.arange(length, dtype=np.float64)
    centered = coordinate - coordinate.mean()
    rms = np.sqrt(np.mean(np.square(centered)))
    value = (centered / rms).astype(np.float32)
    if abs(float(value.mean())) > 1e-6 or abs(float(np.sqrt(np.mean(value**2))) - 1.0) > 1e-6:
        raise RuntimeError("centered unit-RMS time-weight contract changed")
    return value


def frozen_permutations() -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        np.random.Generator(np.random.PCG64(seed)).permutation(stop - start).astype(np.int64)
        for seed, (start, stop) in zip(PERMUTATION_SEEDS, SLOT_BOUNDARIES, strict=True)
    )


def array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def transform_temporal_grid(
    temporal_grid: np.ndarray,
    *,
    condition: str,
    permutations: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError("unknown temporal-moment condition")
    if temporal_grid.shape != (6, 130, 2, 8, 8, 9) or temporal_grid.dtype != np.float32:
        raise ValueError("temporal grid must be float32 [6,130,2,8,8,9]")
    if not np.isfinite(temporal_grid).all():
        raise ValueError("temporal grid contains nonfinite values")
    permutation_values = frozen_permutations() if permutations is None else permutations
    output = temporal_grid.copy()
    for slot, (start, stop) in enumerate(SLOT_BOUNDARIES):
        difference = temporal_grid[:, start:stop, ..., 6:9]
        if condition == "endpoint":
            output[:, start:stop, ..., 6:9] = difference[:, -1:, ...]
        else:
            weights = time_weights(stop - start)
            if condition == "shuffled_moment":
                permutation = permutation_values[slot]
                if permutation.shape != (stop - start,) or sorted(permutation.tolist()) != list(
                    range(stop - start)
                ):
                    raise ValueError("frozen PCG64 permutation contract changed")
                weights = weights[permutation]
            output[:, start:stop, ..., 6:9] = difference * weights[None, :, None, None, None, None]
    if not np.array_equal(output[..., :6], temporal_grid[..., :6]):
        raise RuntimeError("temporal-moment transform changed channels 0:6")
    return np.ascontiguousarray(output)


def transform_feature_group(
    arrays: Mapping[str, np.ndarray],
    *,
    condition: str,
    permutations: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(arrays) != FEATURE_KEYS:
        raise ValueError("source feature key contract changed")
    return {
        "temporal_grid": transform_temporal_grid(
            arrays["temporal_grid"], condition=condition, permutations=permutations
        ),
        "post_rgb_grid": arrays["post_rgb_grid"].copy(),
        "post_proprio": arrays["post_proprio"].copy(),
    }


def _serialize_npz(value: Mapping[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **{key: value[key] for key in sorted(value)})
    return stream.getvalue()


def build_temporal_moment_caches(
    source_cache: FeatureCacheCorpus,
    corpus: ShardedMemoryCorpus,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if source_cache.split != "train" or corpus.split != "train":
        raise PermissionError("temporal-moment cache is train-only")
    source_cache.validate_source(corpus)
    permutations = frozen_permutations()
    permutation_bytes = _serialize_npz(
        {f"slot{slot}": value for slot, value in enumerate(permutations)}
    )
    permutation_identity = write_new_bytes(output_root / "PERMUTATIONS.npz", permutation_bytes)
    records = {condition: [] for condition in CONDITIONS}
    for group_id in source_cache.group_ids:
        source = source_cache.load_group(group_id)
        source_material = source_cache._records[group_id]
        for condition in CONDITIONS:
            arrays = transform_feature_group(
                source.arrays, condition=condition, permutations=permutations
            )
            group_root = output_root / condition / "train" / f"group_{group_id:03d}"
            body = _serialize_npz(arrays)
            feature = write_new_bytes(group_root / "features.npz", body)
            manifest = {
                "schema": CACHE_SCHEMA,
                "condition": condition,
                "split": "train",
                "group_id": group_id,
                "feature_keys": sorted(FEATURE_KEYS),
                "source_feature_sha256": source_material.sha256,
                "permutation_sha256": [array_sha256(value) for value in permutations],
                "feature": {
                    "name": "features.npz",
                    "bytes": feature["bytes"],
                    "sha256": feature["sha256"],
                },
                "targets_opened": False,
            }
            manifest_identity = write_new_json(group_root / "MANIFEST.json", manifest)
            records[condition].append(
                {
                    "group_id": group_id,
                    "source_feature_sha256": source_material.sha256,
                    "feature": feature,
                    "manifest": manifest_identity,
                }
            )
    return {
        "records": records,
        "permutation_identity": permutation_identity,
        "permutation_sha256": [array_sha256(value) for value in permutations],
    }


class TemporalMomentCache:
    def __init__(
        self,
        root: Path,
        condition: str,
        records: Mapping[int, MaterialIdentity],
        manifests: Mapping[int, MaterialIdentity],
        source_sha256: Mapping[int, str],
        permutation_sha256: tuple[str, str],
        permutation_material: MaterialIdentity,
        source_cache_root: Path,
    ):
        self.root = root.resolve()
        self.condition = condition
        self.split = "train"
        self._records = dict(sorted(records.items()))
        self._manifests = dict(sorted(manifests.items()))
        self.source_feature_sha256 = dict(sorted(source_sha256.items()))
        self.permutation_sha256 = permutation_sha256
        self.permutation_material = permutation_material
        self.source_cache_root = source_cache_root.resolve()
        if tuple(self._records) != EXPECTED_SPLIT_GROUPS["train"]:
            raise ValueError("temporal-moment cache group set changed")

    @classmethod
    def discover(
        cls, root: Path, condition: str, source_cache: FeatureCacheCorpus
    ) -> "TemporalMomentCache":
        if condition not in CONDITIONS or root.is_symlink():
            raise ValueError("invalid temporal-moment cache root/condition")
        if {entry.name for entry in root.resolve().iterdir()} != {
            "PERMUTATIONS.npz",
            "RESULT.json",
            *CONDITIONS,
        }:
            raise ValueError("temporal-moment cache root entry set changed")
        split_root = root.resolve() / condition / "train"
        if split_root.is_symlink() or not split_root.is_dir():
            raise ValueError("temporal-moment train root missing or unsafe")
        permutation_path = root.resolve() / "PERMUTATIONS.npz"
        if permutation_path.is_symlink() or not permutation_path.is_file():
            raise ValueError("permutation material missing or unsafe")
        permutation_material = MaterialIdentity(
            permutation_path,
            permutation_path.stat().st_size,
            hashlib.sha256(permutation_path.read_bytes()).hexdigest(),
        )
        with np.load(permutation_path, allow_pickle=False) as archive:
            if set(archive.files) != {"slot0", "slot1"}:
                raise ValueError("permutation archive keys changed")
            persisted_permutations = tuple(
                np.ascontiguousarray(archive[f"slot{slot}"]) for slot in range(2)
            )
        if any(
            not np.array_equal(observed, expected)
            for observed, expected in zip(
                persisted_permutations, frozen_permutations(), strict=True
            )
        ):
            raise ValueError("persisted PCG64 permutations changed")
        persisted_hashes = tuple(array_sha256(value) for value in persisted_permutations)
        records, manifests, sources = {}, {}, {}
        permutation_hashes = None
        for manifest_path in sorted(split_root.glob("group_*/MANIFEST.json")):
            group_root = manifest_path.parent
            if (
                group_root.is_symlink()
                or group_root.resolve().parent != split_root.resolve()
                or manifest_path.is_symlink()
                or not manifest_path.is_file()
                or {entry.name for entry in group_root.iterdir()}
                != {"MANIFEST.json", "features.npz"}
            ):
                raise ValueError("temporal-moment cache material path changed")
            raw = manifest_path.read_bytes()
            value = json.loads(raw)
            group_id = value.get("group_id")
            hashes = value.get("permutation_sha256")
            if (
                value.get("schema") != CACHE_SCHEMA
                or value.get("condition") != condition
                or value.get("split") != "train"
                or value.get("feature_keys") != sorted(FEATURE_KEYS)
                or value.get("targets_opened") is not False
                or not isinstance(group_id, int)
                or group_root.name != f"group_{group_id:03d}"
                or not isinstance(hashes, list)
                or len(hashes) != 2
            ):
                raise ValueError("temporal-moment manifest contract changed")
            if permutation_hashes is None:
                permutation_hashes = tuple(hashes)
            elif permutation_hashes != tuple(hashes):
                raise ValueError("permutation identity differs across groups")
            spec = value["feature"]
            material = MaterialIdentity(group_root / spec["name"], spec["bytes"], spec["sha256"])
            material.verify()
            records[group_id] = material
            manifests[group_id] = MaterialIdentity(
                manifest_path, len(raw), hashlib.sha256(raw).hexdigest()
            )
            sources[group_id] = value["source_feature_sha256"]
        if permutation_hashes != persisted_hashes:
            raise ValueError("manifest and persisted permutation identities differ")
        for group_id, source_digest in sources.items():
            if source_digest != source_cache._records[group_id].sha256:
                raise ValueError(f"temporal-moment source feature changed for group {group_id}")
        return cls(
            root,
            condition,
            records,
            manifests,
            sources,
            permutation_hashes,
            permutation_material,
            source_cache.root,
        )

    @property
    def group_ids(self) -> tuple[int, ...]:
        return tuple(self._records)

    def validate_source(self, corpus: ShardedMemoryCorpus) -> None:
        if corpus.split != "train" or corpus.group_ids != self.group_ids:
            raise ValueError("temporal-moment corpus identity changed")

    def load_group(self, group_id: int) -> FeatureGroup:
        material = self._records[group_id]
        material.verify()
        with np.load(material.path, allow_pickle=False) as archive:
            if set(archive.files) != FEATURE_KEYS:
                raise ValueError("temporal-moment feature keys changed")
            arrays = {key: np.ascontiguousarray(archive[key]) for key in sorted(FEATURE_KEYS)}
        return FeatureGroup(group_id, arrays)

    def verify_receipt(self, result: Mapping[str, Any], source_cache: FeatureCacheCorpus) -> None:
        if source_cache.root.resolve() != self.source_cache_root:
            raise ValueError("temporal-moment source cache root changed")
        if (
            result.get("status") != "passed"
            or result.get("mode") != "temporal_moment_cache"
            or result.get("cache_root") != str(self.root)
            or result.get("conditions") != list(CONDITIONS)
            or result.get("targets_opened") is not False
            or result.get("original_dev_opened") is not False
            or result.get("confirmation_opened") is not False
        ):
            raise ValueError("temporal-moment receipt root/boundary changed")
        if tuple(result.get("permutation_sha256", ())) != self.permutation_sha256:
            raise ValueError("temporal-moment permutation receipt changed")
        expected_permutation = {
            "path": str(self.permutation_material.path.resolve()),
            "bytes": self.permutation_material.bytes,
            "sha256": self.permutation_material.sha256,
        }
        if result.get("permutation_identity") != expected_permutation:
            raise ValueError("temporal-moment permutation material receipt changed")
        condition_records = result.get("condition_records")
        if not isinstance(condition_records, dict) or set(condition_records) != set(CONDITIONS):
            raise ValueError("temporal-moment condition receipt set changed")
        by_group = {item["group_id"]: item for item in condition_records[self.condition]}
        if tuple(sorted(by_group)) != self.group_ids:
            raise ValueError("temporal-moment group receipt set changed")
        for group_id in self.group_ids:
            feature, manifest = self._records[group_id], self._manifests[group_id]
            record = by_group[group_id]
            if record["source_feature_sha256"] != source_cache._records[group_id].sha256:
                raise ValueError("temporal-moment source feature receipt changed")
            for name, material in (("feature", feature), ("manifest", manifest)):
                expected = {
                    "path": str(material.path.resolve()),
                    "bytes": material.bytes,
                    "sha256": material.sha256,
                }
                if record[name] != expected:
                    raise ValueError("temporal-moment material receipt changed")
