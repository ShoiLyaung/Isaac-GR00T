"""Data, metric, and artifact contracts for MIKASA native-RGB memory runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
from typing import Any, Final

import numpy as np
import torch

from gr00t.experimental.mikasa_native_rgb_memory import TARGET_DIM, TARGET_SHAPE, MemoryBatch


MODEL_INPUT_KEYS: Final = frozenset(
    {
        "cue_base_rgb",
        "cue_hand_rgb",
        "cue_proprio",
        "cue_elapsed_steps",
        "cue_valid_mask",
        "post_base_rgb",
        "post_hand_rgb",
        "post_proprio",
        "post_elapsed_steps",
        "post_valid_mask",
    }
)
PRIVILEGED_TARGET_KEYS: Final = frozenset(
    {
        "group_id",
        "condition_code",
        "ordered_shape_ids",
        "center_xy",
        "radius",
        "rotation",
        "ordered_waypoints",
        "ordered_checkpoints",
    }
)
ORDERED_CONDITION_CODES: Final = (0, 1, 2, 3, 4, 5)
ORDERED_SHAPE_IDS: Final = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
ROWS_PER_GROUP: Final = 6
GROUPS_PER_BATCH: Final = 4
BATCH_SIZE: Final = ROWS_PER_GROUP * GROUPS_PER_BATCH
SHARD_SCHEMA: Final = "g1-mikasa-scaled-memory-corpus-r1/shard-v1"
COHORT_SHA256: Final = "5c4704492deec3b9a2082a6c08b3d6bfa4ee463434e53f4a84cfa26065a83161"
DATA_SCHEMA_SHA256: Final = "81f843a3df8943e4851636445c4b9a1625536ea092695d1d20282f21945d338e"
EXPECTED_SPLIT_GROUPS: Final = {
    "train": (
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
    ),
    "dev": (9, 12, 17, 25, 27, 29),
    "confirmation": (8, 13, 15, 21, 30, 32),
}
EXPECTED_SENTINEL_GROUPS: Final = {"train": 0, "dev": 9, "confirmation": 8}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "bytes": len(value), "sha256": sha256_bytes(value)}


def write_new_json(path: Path, value: Any) -> dict[str, Any]:
    return write_new_bytes(path, canonical_json_bytes(value))


@dataclass(frozen=True)
class MaterialIdentity:
    path: Path
    bytes: int
    sha256: str

    def verify(self) -> None:
        if self.path.is_symlink() or not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.stat().st_size != self.bytes:
            raise ValueError(f"byte identity changed: {self.path}")
        if sha256_file(self.path) != self.sha256:
            raise ValueError(f"SHA-256 identity changed: {self.path}")


def _material_from_manifest(
    directory: Path,
    manifest: Mapping[str, Any],
    label: str,
    default_name: str,
) -> MaterialIdentity:
    nested = manifest.get(label)
    if isinstance(nested, Mapping):
        name = nested.get("file", nested.get("name", nested.get("path")))
        size = nested.get("bytes")
        digest = nested.get("sha256")
    else:
        name = manifest.get(f"{label}_file", default_name)
        size = manifest.get(f"{label}_bytes")
        digest = manifest.get(f"{label}_sha256")
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} manifest filename must be a basename")
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"{label} manifest bytes missing or invalid")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} manifest SHA-256 missing or invalid")
    return MaterialIdentity(directory / name, size, digest)


@dataclass(frozen=True)
class ShardInfo:
    group_id: int
    split: str
    manifest: MaterialIdentity
    model_input: MaterialIdentity
    privileged_target: MaterialIdentity


@dataclass(frozen=True)
class GroupArrays:
    group_id: int
    model: dict[str, np.ndarray]
    target: dict[str, np.ndarray]

    def flattened_targets(self) -> torch.Tensor:
        return torch.from_numpy(
            self.target["ordered_checkpoints"].reshape(ROWS_PER_GROUP, TARGET_DIM)
        )


def _load_npz_exact(
    material: MaterialIdentity, expected_keys: frozenset[str]
) -> dict[str, np.ndarray]:
    material.verify()
    with np.load(material.path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"archive keys changed for {material.path}: "
                f"extra={sorted(set(archive.files) - expected_keys)}, "
                f"missing={sorted(expected_keys - set(archive.files))}"
            )
        return {key: np.ascontiguousarray(archive[key]) for key in sorted(expected_keys)}


def _validate_group_arrays(
    group_id: int, model: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]
) -> None:
    expected_model = {
        "cue_base_rgb": ((6, 130, 128, 128, 3), np.uint8),
        "cue_hand_rgb": ((6, 130, 128, 128, 3), np.uint8),
        "cue_proprio": ((6, 130, 7), "floating"),
        "cue_elapsed_steps": ((6, 130), "integer"),
        "cue_valid_mask": ((6, 130), np.bool_),
        "post_base_rgb": ((6, 128, 128, 3), np.uint8),
        "post_hand_rgb": ((6, 128, 128, 3), np.uint8),
        "post_proprio": ((6, 7), "floating"),
        "post_elapsed_steps": ((6,), "integer"),
        "post_valid_mask": ((6,), np.bool_),
    }
    expected_target = {
        "group_id": ((6,), "integer"),
        "condition_code": ((6,), "integer"),
        "ordered_shape_ids": ((6, 2), "integer"),
        "center_xy": ((6, 2), "floating"),
        "radius": ((6,), "floating"),
        "rotation": ((6,), "floating"),
        "ordered_waypoints": ((6, 2, 64, 2), "floating"),
        "ordered_checkpoints": ((6, *TARGET_SHAPE), "floating"),
    }
    for values, expected, label in (
        (model, expected_model, "model"),
        (target, expected_target, "target"),
    ):
        for key, (shape, dtype) in expected.items():
            value = values[key]
            dtype_ok = (
                value.dtype == dtype
                if not isinstance(dtype, str)
                else (
                    np.issubdtype(value.dtype, np.floating)
                    if dtype == "floating"
                    else np.issubdtype(value.dtype, np.integer)
                )
            )
            if value.shape != shape or not dtype_ok:
                raise ValueError(
                    f"group {group_id} {label} {key} is {value.shape}/{value.dtype}, "
                    f"expected {shape}/{dtype}"
                )
            if (
                dtype == "floating"
                or not isinstance(dtype, str)
                and np.issubdtype(dtype, np.floating)
            ) and not bool(np.isfinite(value).all()):
                raise ValueError(f"group {group_id} {label} {key} contains nonfinite values")
    expected_elapsed = np.tile(np.arange(1, 131, dtype=np.int64), (ROWS_PER_GROUP, 1))
    if not np.array_equal(model["cue_elapsed_steps"], expected_elapsed):
        raise ValueError(f"group {group_id} cue elapsed contract changed")
    if not bool(model["cue_valid_mask"].all()) or not bool(model["post_valid_mask"].all()):
        raise ValueError(f"group {group_id} contains invalid sensor frames")
    if not np.array_equal(target["group_id"], np.full(ROWS_PER_GROUP, group_id, np.int64)):
        raise ValueError(f"group {group_id} target group IDs changed")
    if tuple(int(value) for value in target["condition_code"]) != ORDERED_CONDITION_CODES:
        raise ValueError(f"group {group_id} ordered condition codes changed")
    if (
        tuple(tuple(int(item) for item in row) for row in target["ordered_shape_ids"])
        != ORDERED_SHAPE_IDS
    ):
        raise ValueError(f"group {group_id} ordered shape row contract changed")
    if not np.array_equal(model["post_elapsed_steps"], np.full(ROWS_PER_GROUP, 132)):
        raise ValueError(f"group {group_id} post elapsed contract changed")


def _validate_target_group(group_id: int, target: Mapping[str, np.ndarray]) -> None:
    expected = {
        "group_id": ((6,), "integer"),
        "condition_code": ((6,), "integer"),
        "ordered_shape_ids": ((6, 2), "integer"),
        "center_xy": ((6, 2), "floating"),
        "radius": ((6,), "floating"),
        "rotation": ((6,), "floating"),
        "ordered_waypoints": ((6, 2, 64, 2), "floating"),
        "ordered_checkpoints": ((6, *TARGET_SHAPE), "floating"),
    }
    for key, (shape, kind) in expected.items():
        value = target[key]
        kind_ok = (
            np.issubdtype(value.dtype, np.floating)
            if kind == "floating"
            else np.issubdtype(value.dtype, np.integer)
        )
        if value.shape != shape or not kind_ok:
            raise ValueError(f"group {group_id} target {key} contract changed")
        if kind == "floating" and not bool(np.isfinite(value).all()):
            raise ValueError(f"group {group_id} target {key} contains nonfinite values")
    if not np.array_equal(target["group_id"], np.full(ROWS_PER_GROUP, group_id)):
        raise ValueError(f"group {group_id} target group IDs changed")
    if tuple(int(value) for value in target["condition_code"]) != ORDERED_CONDITION_CODES:
        raise ValueError(f"group {group_id} ordered condition codes changed")
    if (
        tuple(tuple(int(item) for item in row) for row in target["ordered_shape_ids"])
        != ORDERED_SHAPE_IDS
    ):
        raise ValueError(f"group {group_id} ordered shape row contract changed")


class ShardedMemoryCorpus:
    """Manifest-first corpus view that opens one geometry shard at a time."""

    def __init__(self, root: Path, split: str, shards: Sequence[ShardInfo]):
        self.root = root.resolve()
        self.split = split
        self.shards = tuple(sorted(shards, key=lambda item: item.group_id))
        self._by_group = {item.group_id: item for item in self.shards}
        if len(self._by_group) != len(self.shards):
            raise ValueError("duplicate group shard")
        if tuple(self._by_group) != EXPECTED_SPLIT_GROUPS[split]:
            raise ValueError(
                f"{split} group IDs changed: {tuple(self._by_group)} != {EXPECTED_SPLIT_GROUPS[split]}"
            )

    @classmethod
    def discover(
        cls,
        root: Path,
        split: str,
        *,
        confirmation_authorized: bool = False,
        verify_privileged_targets: bool = True,
    ) -> "ShardedMemoryCorpus":
        if split not in {"train", "dev", "confirmation"}:
            raise ValueError(f"unsupported split: {split}")
        # This check intentionally occurs before stat/glob/open under the root.
        if split == "confirmation" and not confirmation_authorized:
            raise PermissionError("confirmation is sealed until dev and freeze receipts pass")
        if root.is_symlink():
            raise ValueError(f"corpus root may not be a symlink: {root}")
        split_root = root.resolve() / split
        if split_root.is_symlink() or not split_root.is_dir():
            raise ValueError(f"split root must be a regular directory: {split_root}")
        manifests = sorted(split_root.glob("group_*/MANIFEST.json"))
        if not manifests:
            raise FileNotFoundError(f"no group manifests under {split_root}")
        shards: list[ShardInfo] = []
        for manifest_path in manifests:
            group_root = manifest_path.parent
            if group_root.is_symlink() or group_root.resolve().parent != split_root.resolve():
                raise ValueError(f"group directory escapes the split root: {group_root}")
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(f"manifest must be a regular non-symlink file: {manifest_path}")
            raw = manifest_path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"manifest is not an object: {manifest_path}")
            group_id = value.get("group_id")
            if (
                not isinstance(group_id, int)
                or manifest_path.parent.name != f"group_{group_id:03d}"
            ):
                raise ValueError(f"group manifest path/ID mismatch: {manifest_path}")
            if value.get("split") != split:
                raise ValueError(f"group manifest split mismatch: {manifest_path}")
            if value.get("model_input_keys") != sorted(MODEL_INPUT_KEYS):
                raise ValueError(f"model key declaration changed: {manifest_path}")
            if value.get("privileged_target_keys") != sorted(PRIVILEGED_TARGET_KEYS):
                raise ValueError(f"target key declaration changed: {manifest_path}")
            if value.get("camera_order") != ["base_camera", "hand_camera"]:
                raise ValueError(f"camera order changed: {manifest_path}")
            boundaries = value.get("boundaries")
            if boundaries not in (
                {
                    "cue_elapsed_inclusive": [1, 130],
                    "action_onset_elapsed": 131,
                    "post_capture_elapsed": 132,
                },
                None,
            ):
                raise ValueError(f"phase boundaries changed: {manifest_path}")
            manifest = MaterialIdentity(manifest_path, len(raw), sha256_bytes(raw))
            if value.get("schema") != SHARD_SCHEMA or value.get("version") != 1:
                raise ValueError(f"shard schema/version changed: {manifest_path}")
            if value.get("frozen_cohort_sha256") != COHORT_SHA256:
                raise ValueError(f"cohort identity changed: {manifest_path}")
            if value.get("data_schema_sha256") != DATA_SCHEMA_SHA256:
                raise ValueError(f"data schema identity changed: {manifest_path}")
            if value.get("ordered_pair_rows") != [list(pair) for pair in ORDERED_SHAPE_IDS]:
                raise ValueError(f"ordered pair row contract changed: {manifest_path}")
            if value.get("cue_elapsed_inclusive") != [1, 130]:
                raise ValueError(f"cue boundary changed: {manifest_path}")
            if value.get("action_onset_elapsed") != 131 or value.get("post_capture_elapsed") != 132:
                raise ValueError(f"action/post boundary changed: {manifest_path}")
            sixpack = value.get("sixpack")
            if not isinstance(sixpack, dict) or "sentinel_repeat" not in value:
                raise ValueError(f"sixpack/sentinel evidence missing: {manifest_path}")
            sixpack_gates = sixpack.get("gates")
            if (
                not isinstance(sixpack_gates, dict)
                or not sixpack_gates
                or not all(gate is True for gate in sixpack_gates.values())
            ):
                raise ValueError(f"sixpack gates are not all passed: {manifest_path}")
            sentinel = value["sentinel_repeat"]
            if group_id == EXPECTED_SENTINEL_GROUPS[split]:
                if (
                    not isinstance(sentinel, dict)
                    or sentinel.get("performed") is not True
                    or sentinel.get("all_exact") is not True
                    or not isinstance(sentinel.get("sample_hash_parity"), dict)
                    or not all(item is True for item in sentinel["sample_hash_parity"].values())
                    or not isinstance(sentinel.get("repeat_sixpack_gates"), dict)
                    or not all(item is True for item in sentinel["repeat_sixpack_gates"].values())
                ):
                    raise ValueError(f"sentinel repeat gates are not all passed: {manifest_path}")
            elif sentinel is not None:
                raise ValueError(f"non-sentinel group contains repeat evidence: {manifest_path}")
            expected_files = {"MANIFEST.json", "model_inputs.npz", "privileged_targets.npz"}
            actual_files = {
                entry.name for entry in manifest_path.parent.iterdir() if entry.is_file()
            }
            if actual_files != expected_files:
                raise ValueError(f"group shard file set changed: {manifest_path.parent}")
            model = _material_from_manifest(
                manifest_path.parent, value, "model_inputs", "model_inputs.npz"
            )
            target = _material_from_manifest(
                manifest_path.parent, value, "privileged_targets", "privileged_targets.npz"
            )
            model.verify()
            if verify_privileged_targets:
                target.verify()
            shards.append(ShardInfo(group_id, split, manifest, model, target))
        return cls(root, split, shards)

    @property
    def group_ids(self) -> tuple[int, ...]:
        return tuple(item.group_id for item in self.shards)

    def load_group(self, group_id: int) -> GroupArrays:
        try:
            shard = self._by_group[group_id]
        except KeyError as error:
            raise KeyError(f"group {group_id} is not in {self.split}") from error
        model = _load_npz_exact(shard.model_input, MODEL_INPUT_KEYS)
        target = _load_npz_exact(shard.privileged_target, PRIVILEGED_TARGET_KEYS)
        _validate_group_arrays(group_id, model, target)
        return GroupArrays(group_id, model, target)

    def load_model_group(self, group_id: int) -> dict[str, np.ndarray]:
        """Load policy-visible arrays without opening the privileged target file."""

        try:
            shard = self._by_group[group_id]
        except KeyError as error:
            raise KeyError(f"group {group_id} is not in {self.split}") from error
        model = _load_npz_exact(shard.model_input, MODEL_INPUT_KEYS)
        # Validate against a shape-only dummy target without reading the actual
        # target shard. This path is used by the target-blind feature builder.
        expected_elapsed = np.tile(np.arange(1, 131, dtype=np.int64), (ROWS_PER_GROUP, 1))
        if model["cue_base_rgb"].shape != (6, 130, 128, 128, 3):
            raise ValueError(f"group {group_id} cue base shape changed")
        if model["cue_hand_rgb"].shape != (6, 130, 128, 128, 3):
            raise ValueError(f"group {group_id} cue hand shape changed")
        if model["post_base_rgb"].shape != (6, 128, 128, 3):
            raise ValueError(f"group {group_id} post base shape changed")
        if model["post_hand_rgb"].shape != (6, 128, 128, 3):
            raise ValueError(f"group {group_id} post hand shape changed")
        if any(
            model[key].dtype != np.uint8
            for key in ("cue_base_rgb", "cue_hand_rgb", "post_base_rgb", "post_hand_rgb")
        ):
            raise ValueError(f"group {group_id} native RGB dtype changed")
        if not np.array_equal(model["cue_elapsed_steps"], expected_elapsed):
            raise ValueError(f"group {group_id} cue elapsed contract changed")
        if not bool(model["cue_valid_mask"].all()) or not bool(model["post_valid_mask"].all()):
            raise ValueError(f"group {group_id} contains invalid sensor frames")
        if not np.array_equal(model["post_elapsed_steps"], np.full(ROWS_PER_GROUP, 132)):
            raise ValueError(f"group {group_id} post elapsed contract changed")
        if model["post_proprio"].shape != (6, 7) or not np.issubdtype(
            model["post_proprio"].dtype, np.floating
        ):
            raise ValueError(f"group {group_id} post proprio contract changed")
        if not bool(np.isfinite(model["post_proprio"]).all()):
            raise ValueError(f"group {group_id} post proprio contains nonfinite values")
        return model

    def load_target_group(self, group_id: int) -> GroupArrays:
        """Load labels/association without opening the large native-RGB shard."""

        try:
            shard = self._by_group[group_id]
        except KeyError as error:
            raise KeyError(f"group {group_id} is not in {self.split}") from error
        target = _load_npz_exact(shard.privileged_target, PRIVILEGED_TARGET_KEYS)
        _validate_target_group(group_id, target)
        return GroupArrays(group_id, {}, target)

    def iter_groups(self) -> Iterator[GroupArrays]:
        for group_id in self.group_ids:
            yield self.load_group(group_id)

    def iter_target_groups(self) -> Iterator[GroupArrays]:
        for group_id in self.group_ids:
            yield self.load_target_group(group_id)


def concatenate_groups(groups: Sequence[GroupArrays]) -> GroupArrays:
    if not groups:
        raise ValueError("cannot concatenate zero groups")
    model = {
        key: np.ascontiguousarray(np.concatenate([group.model[key] for group in groups], axis=0))
        for key in MODEL_INPUT_KEYS
    }
    target = {
        key: np.ascontiguousarray(np.concatenate([group.target[key] for group in groups], axis=0))
        for key in PRIVILEGED_TARGET_KEYS
    }
    return GroupArrays(-1, model, target)


FEATURE_KEYS: Final = frozenset({"temporal_grid", "post_rgb_grid", "post_proprio"})


def _block_mean_16(value: np.ndarray) -> np.ndarray:
    """Target-blind 128x128 RGB to 8x8 RGB block means in [0,1]."""

    if value.shape[-3:] != (128, 128, 3) or value.dtype != np.uint8:
        raise ValueError("block mean requires uint8 [...,128,128,3] native RGB")
    prefix = value.shape[:-3]
    normalized = value.astype(np.float32) / np.float32(255.0)
    blocked = normalized.reshape(*prefix, 8, 16, 8, 16, 3)
    return np.ascontiguousarray(blocked.mean(axis=(-4, -2), dtype=np.float32))


def build_native_rgb_features(model: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct the frozen 8x8 teacher grids from policy-visible arrays only."""

    if set(model) != MODEL_INPUT_KEYS:
        raise ValueError("feature builder received a non-policy model view")
    cue_base = _block_mean_16(model["cue_base_rgb"])
    cue_hand = _block_mean_16(model["cue_hand_rgb"])
    post_base = _block_mean_16(model["post_base_rgb"])
    post_hand = _block_mean_16(model["post_hand_rgb"])
    cue = np.stack((cue_base, cue_hand), axis=2)
    post = np.stack((post_base, post_hand), axis=1)
    expanded_post = np.broadcast_to(post[:, None], cue.shape)
    temporal = np.concatenate((cue, expanded_post, cue - expanded_post), axis=-1)
    result = {
        "temporal_grid": np.ascontiguousarray(temporal, dtype=np.float32),
        "post_rgb_grid": np.ascontiguousarray(post, dtype=np.float32),
        "post_proprio": np.ascontiguousarray(model["post_proprio"], dtype=np.float32),
    }
    validate_feature_arrays(result)
    return result


def validate_feature_arrays(value: Mapping[str, np.ndarray]) -> None:
    if set(value) != FEATURE_KEYS:
        raise ValueError("feature-cache key contract changed")
    expected = {
        "temporal_grid": (6, 130, 2, 8, 8, 9),
        "post_rgb_grid": (6, 2, 8, 8, 3),
        "post_proprio": (6, 7),
    }
    for key, shape in expected.items():
        array = value[key]
        if array.shape != shape or array.dtype != np.float32 or not bool(np.isfinite(array).all()):
            raise ValueError(f"feature {key} violates {shape}/float32/finite contract")
    if not bool((value["post_rgb_grid"] >= 0).all() and (value["post_rgb_grid"] <= 1).all()):
        raise ValueError("post RGB grid must remain in [0,1]")
    cue, post, difference = np.split(value["temporal_grid"], 3, axis=-1)
    expected_post = np.broadcast_to(value["post_rgb_grid"][:, None], post.shape)
    if not np.array_equal(post, expected_post) or not np.allclose(
        difference, cue - post, rtol=0.0, atol=1e-7
    ):
        raise ValueError("temporal grid channel contract changed")


@dataclass(frozen=True)
class FeatureGroup:
    group_id: int
    arrays: dict[str, np.ndarray]

    def memory_batch(self) -> MemoryBatch:
        return MemoryBatch(
            temporal_grid=torch.from_numpy(self.arrays["temporal_grid"]),
            post_rgb_grid=torch.from_numpy(self.arrays["post_rgb_grid"]),
            post_proprio=torch.from_numpy(self.arrays["post_proprio"]),
        )


def concatenate_feature_groups(groups: Sequence[FeatureGroup]) -> FeatureGroup:
    if not groups:
        raise ValueError("cannot concatenate zero feature groups")
    arrays = {
        key: np.ascontiguousarray(np.concatenate([group.arrays[key] for group in groups], axis=0))
        for key in FEATURE_KEYS
    }
    return FeatureGroup(-1, arrays)


def _serialize_npz(value: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **{key: value[key] for key in sorted(value)})
    return stream.getvalue()


def build_feature_cache(corpus: ShardedMemoryCorpus, output_root: Path) -> list[dict[str, Any]]:
    """Write one target-blind train/dev feature shard per geometry."""

    if corpus.split not in {"train", "dev"}:
        raise PermissionError("persistent feature caches are allowed only for train/dev")
    split_root = output_root.resolve() / corpus.split
    if split_root.exists():
        raise FileExistsError(split_root)
    records: list[dict[str, Any]] = []
    for group_id in corpus.group_ids:
        group_root = output_root.resolve() / corpus.split / f"group_{group_id:03d}"
        if group_root.exists():
            raise FileExistsError(group_root)
        model = corpus.load_model_group(group_id)
        features = build_native_rgb_features(model)
        feature_bytes = _serialize_npz(features)
        feature_identity = write_new_bytes(group_root / "native_rgb_features.npz", feature_bytes)
        manifest = {
            "schema": "g1-mikasa-native-rgb-memory-backbone-r1/feature-cache-v1",
            "split": corpus.split,
            "group_id": group_id,
            "source_model_input_sha256": corpus._by_group[group_id].model_input.sha256,
            "feature_keys": sorted(FEATURE_KEYS),
            "feature": {
                "file": "native_rgb_features.npz",
                "bytes": feature_identity["bytes"],
                "sha256": feature_identity["sha256"],
            },
            "transform": "uint8_div255_block_mean_16x16_cue_post_signed_diff",
            "targets_opened": False,
        }
        manifest_identity = write_new_json(group_root / "MANIFEST.json", manifest)
        records.append(
            {"group_id": group_id, "feature": feature_identity, "manifest": manifest_identity}
        )
    return records


class FeatureCacheCorpus:
    """Train/dev-only view over write-once target-blind feature grids."""

    def __init__(
        self,
        root: Path,
        split: str,
        records: Mapping[int, MaterialIdentity],
        source_model_sha256: Mapping[int, str],
    ):
        if split not in {"train", "dev"}:
            raise PermissionError("confirmation feature caches are forbidden")
        self.root = root.resolve()
        self.split = split
        self._records = dict(sorted(records.items()))
        self.source_model_sha256 = dict(sorted(source_model_sha256.items()))
        if tuple(self._records) != EXPECTED_SPLIT_GROUPS[split]:
            raise ValueError(f"{split} feature-cache group IDs changed")
        if self._records.keys() != self.source_model_sha256.keys():
            raise ValueError("feature-cache source identity keys changed")

    @classmethod
    def discover(cls, root: Path, split: str) -> "FeatureCacheCorpus":
        if split not in {"train", "dev"}:
            raise PermissionError("confirmation feature caches are forbidden")
        if root.is_symlink():
            raise ValueError(f"feature-cache root may not be a symlink: {root}")
        records: dict[int, MaterialIdentity] = {}
        source_model_sha256: dict[int, str] = {}
        split_root = root.resolve() / split
        if split_root.is_symlink() or not split_root.is_dir():
            raise ValueError(f"feature-cache split must be a regular directory: {split_root}")
        for path in sorted(split_root.glob("group_*/MANIFEST.json")):
            if path.parent.is_symlink() or path.parent.resolve().parent != split_root.resolve():
                raise ValueError("feature-cache group directory escapes split root")
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"feature manifest is not a regular file: {path}")
            actual_files = {entry.name for entry in path.parent.iterdir() if entry.is_file()}
            if actual_files != {"MANIFEST.json", "native_rgb_features.npz"}:
                raise ValueError("feature-cache group file set changed")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("schema") != "g1-mikasa-native-rgb-memory-backbone-r1/feature-cache-v1":
                raise ValueError("feature-cache schema changed")
            group_id = manifest.get("group_id")
            if not isinstance(group_id, int) or path.parent.name != f"group_{group_id:03d}":
                raise ValueError("feature group path/ID mismatch")
            if manifest.get("split") != split or manifest.get("feature_keys") != sorted(
                FEATURE_KEYS
            ):
                raise ValueError("feature manifest split/key contract changed")
            if manifest.get("targets_opened") is not False:
                raise ValueError("feature cache is not target blind")
            if manifest.get("transform") != "uint8_div255_block_mean_16x16_cue_post_signed_diff":
                raise ValueError("feature-cache transform changed")
            source_digest = manifest.get("source_model_input_sha256")
            if not isinstance(source_digest, str) or len(source_digest) != 64:
                raise ValueError("feature-cache source model identity missing")
            source_model_sha256[group_id] = source_digest
            records[group_id] = _material_from_manifest(
                path.parent, manifest, "feature", "native_rgb_features.npz"
            )
            records[group_id].verify()
        if not records:
            raise FileNotFoundError(f"no feature cache manifests for {split}")
        return cls(root, split, records, source_model_sha256)

    @property
    def group_ids(self) -> tuple[int, ...]:
        return tuple(self._records)

    def validate_source(self, corpus: ShardedMemoryCorpus) -> None:
        if self.split != corpus.split or self.group_ids != corpus.group_ids:
            raise ValueError("feature cache and source corpus split/group IDs differ")
        for group_id in corpus.group_ids:
            if self.source_model_sha256[group_id] != corpus._by_group[group_id].model_input.sha256:
                raise ValueError(f"feature cache source identity changed for group {group_id}")

    def load_group(self, group_id: int) -> FeatureGroup:
        try:
            material = self._records[group_id]
        except KeyError as error:
            raise KeyError(group_id) from error
        arrays = _load_npz_exact(material, FEATURE_KEYS)
        validate_feature_arrays(arrays)
        return FeatureGroup(group_id, arrays)


@dataclass(frozen=True)
class TargetStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, corpus: ShardedMemoryCorpus) -> "TargetStats":
        if corpus.split != "train":
            raise ValueError("target statistics may only be fitted on the train split")
        count = 0
        total = np.zeros(TARGET_DIM, dtype=np.float64)
        square = np.zeros(TARGET_DIM, dtype=np.float64)
        for group in corpus.iter_target_groups():
            target = (
                group.target["ordered_checkpoints"]
                .reshape(ROWS_PER_GROUP, TARGET_DIM)
                .astype(np.float64)
            )
            count += target.shape[0]
            total += target.sum(axis=0)
            square += np.square(target).sum(axis=0)
        if count < 1:
            raise ValueError("cannot fit target statistics on an empty corpus")
        mean = total / count
        variance = np.maximum(square / count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        if bool((std < 1e-8).any()):
            raise ValueError("train-only target standard deviation is degenerate")
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.std, device=value.device, dtype=value.dtype)
        return (value - mean) / std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.std, device=value.device, dtype=value.dtype)
        return value * std + mean

    def as_json(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "source_split": "train"}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "TargetStats":
        if value.get("source_split") != "train":
            raise ValueError("checkpoint target statistics are not train-only")
        mean = np.asarray(value.get("mean"), dtype=np.float32)
        std = np.asarray(value.get("std"), dtype=np.float32)
        if mean.shape != (TARGET_DIM,) or std.shape != (TARGET_DIM,):
            raise ValueError("checkpoint target statistic shape changed")
        if not bool(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
            raise ValueError("checkpoint target statistics are invalid")
        return cls(mean, std)


class GroupBalancedSchedule:
    """Deterministic four-sixpack schedule producing exactly 24 rows per step."""

    def __init__(self, group_ids: Sequence[int], *, seed: int):
        if len(group_ids) < GROUPS_PER_BATCH or len(group_ids) % GROUPS_PER_BATCH:
            raise ValueError("group count must be positive and divisible by four")
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("group schedule contains duplicates")
        self.group_ids = tuple(int(value) for value in group_ids)
        self.seed = int(seed)

    def batch(self, step: int) -> tuple[int, ...]:
        if step < 0:
            raise ValueError("step must be nonnegative")
        batches_per_epoch = len(self.group_ids) // GROUPS_PER_BATCH
        epoch, offset = divmod(step, batches_per_epoch)
        ordered = list(self.group_ids)
        random.Random(self.seed + epoch).shuffle(ordered)
        selected = ordered[offset * GROUPS_PER_BATCH : (offset + 1) * GROUPS_PER_BATCH]
        if len(selected) != GROUPS_PER_BATCH:
            raise RuntimeError("group-balanced schedule truncated a batch")
        return tuple(selected)


def serialize_checkpoint(payload: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def physical_metrics(
    prediction_normalized: torch.Tensor,
    target_physical: torch.Tensor,
    stats: TargetStats,
    candidates_physical: torch.Tensor,
    target_candidate_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction_normalized.ndim != 2 or prediction_normalized.shape[-1] != TARGET_DIM:
        raise ValueError("prediction must have shape [rows,48]")
    if target_physical.shape != prediction_normalized.shape:
        raise ValueError("target/prediction shape mismatch")
    prediction = stats.denormalize(prediction_normalized).reshape(-1, *TARGET_SHAPE)
    target = target_physical.reshape(-1, *TARGET_SHAPE)
    error = (prediction - target).abs()
    candidate = candidates_physical.reshape(-1, ROWS_PER_GROUP, TARGET_DIM)
    if candidate.shape[0] != prediction.shape[0]:
        raise ValueError("candidate batch mismatch")
    distance = torch.square(prediction.reshape(-1, 1, TARGET_DIM) - candidate).mean(dim=-1)
    return {
        "first_mae_m": error[:, 0].mean(dim=(1, 2)),
        "second_mae_m": error[:, 1].mean(dim=(1, 2)),
        "complete_mae_m": error.mean(dim=(1, 2, 3)),
        "normalized_mse": torch.square(
            prediction_normalized - stats.normalize(target_physical)
        ).mean(dim=1),
        "order_correct": distance.argmin(dim=1).eq(target_candidate_index).to(torch.float32),
        "predicted_candidate_index": distance.argmin(dim=1),
    }


def group_equal_mean(records: Sequence[Mapping[str, Any]], metric: str) -> float:
    by_group: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_group[int(record["group_id"])].append(float(record[metric]))
    if not by_group:
        raise ValueError("cannot aggregate empty metric records")
    return float(np.mean([np.mean(values) for _, values in sorted(by_group.items())]))


def seed_equal_mean(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]], metric: str
) -> float:
    if not records_by_seed:
        raise ValueError("cannot aggregate empty seed records")
    return float(
        np.mean(
            [group_equal_mean(records_by_seed[seed], metric) for seed in sorted(records_by_seed)]
        )
    )


def assert_checkpoint_set(checkpoints: Mapping[str, Mapping[int, Path]]) -> None:
    expected_conditions = {
        "mamba_current",
        "mamba_one_frame",
        "mamba_last_segment",
        "mamba_full",
        "gru_full",
        "transformer_full",
    }
    if set(checkpoints) != expected_conditions:
        raise ValueError("checkpoint condition set changed")
    for condition, by_seed in checkpoints.items():
        if set(by_seed) != {230923, 230924, 230925}:
            raise ValueError(f"checkpoint seeds changed for {condition}")
        for path in by_seed.values():
            if not path.is_file():
                raise FileNotFoundError(path)


def confirmation_spec_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("confirmation spec must be a regular file")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != (
        "g1-mikasa-native-rgb-memory-backbone-r1/confirmation-spec-v1"
    ):
        raise ValueError("confirmation spec schema changed")
    argv = value.get("generator_argv")
    materials = value.get("generator_materials")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        raise ValueError("confirmation spec generator_argv is invalid")
    if (
        not isinstance(materials, list)
        or not materials
        or len(set(materials)) != len(materials)
        or not all(isinstance(item, str) and Path(item).is_absolute() for item in materials)
    ):
        raise ValueError("confirmation spec generator_materials is invalid")
    identities: dict[str, str] = {}
    for item in materials:
        material = Path(item)
        if material.is_symlink() or not material.is_file():
            raise ValueError(f"confirmation generator material is not regular: {material}")
        identities[str(material.resolve())] = sha256_file(material)
    if not any(item in argv for item in materials):
        raise ValueError("confirmation argv does not reference a frozen generator material")
    return value, {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "generator_material_sha256": identities,
    }


def verify_confirmation_receipts(
    dev_receipt_path: Path,
    freeze_receipt_path: Path,
    checkpoints: Mapping[str, Mapping[int, Path]],
    source_paths: Mapping[str, Path],
    confirmation_spec_path: Path,
) -> dict[str, Any]:
    """Authorize confirmation without touching any confirmation corpus path."""

    assert_checkpoint_set(checkpoints)
    dev = json.loads(dev_receipt_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_receipt_path.read_text(encoding="utf-8"))
    if dev.get("status") != "passed" or dev.get("dev_gate_passed") is not True:
        raise PermissionError("dev receipt does not open confirmation")
    if freeze.get("status") != "frozen":
        raise PermissionError("checkpoint/evaluator freeze receipt is absent")
    observed = {
        condition: {str(seed): sha256_file(path) for seed, path in sorted(by_seed.items())}
        for condition, by_seed in sorted(checkpoints.items())
    }
    if freeze.get("checkpoint_sha256") != observed:
        raise PermissionError("checkpoint freeze identities changed")
    if any(path.is_symlink() or not path.is_file() for path in source_paths.values()):
        raise PermissionError("model/evaluator source material is not a regular file")
    observed_sources = {name: sha256_file(path) for name, path in sorted(source_paths.items())}
    if freeze.get("source_sha256") != observed_sources:
        raise PermissionError("model/evaluator source freeze identity changed")
    _, spec_identity = confirmation_spec_identity(confirmation_spec_path)
    if freeze.get("confirmation_spec") != spec_identity:
        raise PermissionError("confirmation generator/spec freeze identity changed")
    return {
        "dev_receipt_sha256": sha256_file(dev_receipt_path),
        "freeze_receipt_sha256": sha256_file(freeze_receipt_path),
        "checkpoint_sha256": observed,
        "source_sha256": observed_sources,
        "confirmation_spec": spec_identity,
    }
