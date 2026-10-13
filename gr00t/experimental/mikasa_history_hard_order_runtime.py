"""Evidence, loader, target-blind cache, and schedule contracts for native order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import random
import stat
from typing import Any, Final

import numpy as np
import torch

from gr00t.experimental.mikasa_history_hard_order import OrderBatch


EXPERIMENT_ID: Final = "g1-mikasa-history-hard-native-order-backbones-r1"
DECISION_AUDIT_ID: Final = "g1-mikasa-history-hard-decision-anchor-audit-r1"
PAIR_CODES: Final = ("01", "02", "10", "12", "20", "21")
TRAIN_GROUPS: Final = (0, 1, 2, 3, 4, 5)
DEV_GROUPS: Final = (6, 7)
ROWS_PER_GROUP: Final = 6
BATCH_GROUPS: Final = 4
BATCH_SIZE: Final = 24

CORPUS_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/"
    "g1-mikasa-history-hard-native-corpus-anchor-first-r1/native-corpus-gpu0-render-r1"
)
CORPUS_RESULT: Final = CORPUS_ROOT / "RESULT.json"
CORPUS_RESULT_BYTES: Final = 308_525
CORPUS_RESULT_SHA256: Final = "422981f6581b23f0476378967b48a43db3861f89b5a76231574f799183f7a84b"

AUDIT_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/"
    "g1-mikasa-history-hard-decision-anchor-audit-r1/decision-anchor-audit-cpu-r1"
)
INPUT_MANIFEST: Final = AUDIT_ROOT / "INPUT_MANIFEST.json"
INPUT_MANIFEST_BYTES: Final = 37_114
INPUT_MANIFEST_SHA256: Final = "9c6fe42299886738b00e3168140d024b2b609ea14bae7631a5b20235af063563"
TARGET_MANIFEST: Final = AUDIT_ROOT / "TARGET_MANIFEST.json"
TARGET_MANIFEST_BYTES: Final = 30_702
TARGET_MANIFEST_SHA256: Final = "8e414d9663ae32dcc3cb88af8624ac508aebac5bea03071420531d4d6e8e7aa9"
AUDIT_RESULT: Final = AUDIT_ROOT / "RESULT.json"
AUDIT_RESULT_BYTES: Final = 13_987
AUDIT_RESULT_SHA256: Final = "2fdaf51bda7ba0b0cb4caa2eae6796bd6354417b8afdb5f2f8c2c052f30834d0"

FRESH_PURE_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/"
    "g1-mikasa-history-hard-fresh-geometry-pure-r1/source-acceptance-cpu-r1"
)
FRESH_PURE_RESULT: Final = FRESH_PURE_ROOT / "RESULT.json"
FRESH_PURE_RESULT_BYTES: Final = 4_670
FRESH_PURE_RESULT_SHA256: Final = "031be8e09d4105e9de6f8b2c72a7487068856ec470900a11924b9d38789d72ac"

MODEL_FIELDS: Final = frozenset(
    {
        "rgb",
        "proprio",
        "action",
        "reward",
        "success",
        "done",
        "elapsed_steps",
        "valid_mask",
        "language_instruction",
        "camera_order",
        "action_controller",
        "action_translation_frame",
        "action_rotation_frame",
        "action_rotation_parameterization",
        "action_normalized_range",
    }
)
TARGET_FIELDS: Final = frozenset(
    {
        "row_id",
        "group_id",
        "pair_code",
        "seed",
        "ordered_path_ids",
        "ordered_waypoints",
        "ordered_checkpoints",
        "template_bank_sha256",
    }
)
CACHE_FIELDS: Final = frozenset({"temporal_grid", "current_rgb_grid", "current_proprio"})


class RuntimeContractError(RuntimeError):
    """A frozen evidence, data, cache, or schedule contract changed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def array_identity(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    descriptor = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(descriptor))
    digest.update(memoryview(array).cast("B"))
    return {**descriptor, "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class MaterialIdentity:
    path: Path
    bytes: int
    sha256: str

    def verify(self, *, root: Path | None = None, mode: int = 0o444) -> None:
        if root is not None:
            try:
                self.path.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (FileNotFoundError, ValueError) as exc:
                raise RuntimeContractError(f"material escapes root: {self.path}") from exc
        try:
            info = self.path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeContractError(f"material missing: {self.path}") from exc
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != mode:
            raise RuntimeContractError(f"material must be regular mode {mode:04o}: {self.path}")
        if info.st_size != self.bytes or sha256_file(self.path) != self.sha256:
            raise RuntimeContractError(f"material identity changed: {self.path}")

    def as_json(self) -> dict[str, Any]:
        return {"path": str(self.path), "bytes": self.bytes, "sha256": self.sha256}


def _bound_json(
    path: Path, size: int, digest: str, label: str
) -> tuple[dict[str, Any], MaterialIdentity]:
    identity = MaterialIdentity(path, size, digest)
    identity.verify()
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeContractError(f"{label} must be a JSON object")
    return value, identity


def validate_frozen_evidence(*, include_target: bool) -> dict[str, Any]:
    corpus, corpus_identity = _bound_json(
        CORPUS_RESULT,
        CORPUS_RESULT_BYTES,
        CORPUS_RESULT_SHA256,
        "corpus result",
    )
    if (
        corpus.get("status") != "passed"
        or corpus.get("rows_completed") != 48
        or corpus.get("split_counts") != {"train": 36, "dev": 12, "confirmation": 0}
    ):
        raise RuntimeContractError("qualified corpus result changed")
    audit, audit_identity = _bound_json(
        AUDIT_RESULT,
        AUDIT_RESULT_BYTES,
        AUDIT_RESULT_SHA256,
        "decision audit result",
    )
    if (
        audit.get("status") != "passed"
        or audit.get("audit", {}).get("rows_passed") != 48
        or audit.get("decision", {}).get(
            "six_condition_two_slot_native_history_classifier_permitted"
        )
        is not True
    ):
        raise RuntimeContractError("decision audit did not open native order classification")
    input_manifest, input_identity = _bound_json(
        INPUT_MANIFEST,
        INPUT_MANIFEST_BYTES,
        INPUT_MANIFEST_SHA256,
        "input manifest",
    )
    if (
        input_manifest.get("contains_labels_or_targets") is not False
        or input_manifest.get("core", {}).get("row_count") != 48
        or input_manifest.get("core", {}).get("corpus_result_sha256") != CORPUS_RESULT_SHA256
        or input_manifest.get("peer_target_core_sha256")
        != "b3055d5f836523fd167c4d30540d4f06f76b42b13909000d7e7a0c8776762467"
    ):
        raise RuntimeContractError("target-blind input manifest changed")
    fresh, fresh_identity = _bound_json(
        FRESH_PURE_RESULT,
        FRESH_PURE_RESULT_BYTES,
        FRESH_PURE_RESULT_SHA256,
        "fresh pure result",
    )
    if (
        fresh.get("status") != "passed"
        or fresh.get("fresh_data_rows") != 0
        or fresh.get("model_outputs_read") != 0
        or fresh.get("decision", {}).get("fresh_expert_or_corpus_generation_permitted") is not False
    ):
        raise RuntimeContractError("fresh pure boundary changed")
    evidence = {
        "corpus_result": corpus_identity.as_json(),
        "decision_audit_result": audit_identity.as_json(),
        "input_manifest": input_identity.as_json(),
        "fresh_pure_result": fresh_identity.as_json(),
        "target_manifest_opened": include_target,
        "fresh_data_opened": False,
    }
    if include_target:
        target, target_identity = _bound_json(
            TARGET_MANIFEST,
            TARGET_MANIFEST_BYTES,
            TARGET_MANIFEST_SHA256,
            "target manifest",
        )
        if (
            target.get("privileged") is not True
            or target.get("core", {}).get("row_count") != 48
            or target.get("peer_input_core_sha256") != input_manifest.get("core_sha256")
            or target.get("core_sha256") != input_manifest.get("peer_target_core_sha256")
        ):
            raise RuntimeContractError("privileged target manifest changed")
        evidence["target_manifest"] = target_identity.as_json()
    return evidence


def _identity_from_record(value: Mapping[str, Any], label: str) -> MaterialIdentity:
    if set(value) != {"path", "bytes", "sha256"}:
        raise RuntimeContractError(f"{label} identity fields changed")
    return MaterialIdentity(Path(str(value["path"])), int(value["bytes"]), str(value["sha256"]))


@dataclass(frozen=True)
class RowSource:
    row_id: int
    model: MaterialIdentity
    target: MaterialIdentity
    receipt: MaterialIdentity
    sample_id: str


class HistoryHardSources:
    """Receipt-first source index; targets remain unopened until explicitly requested."""

    def __init__(self, rows: Sequence[RowSource], input_samples: Sequence[Mapping[str, Any]]):
        self.rows = tuple(rows)
        self.input_samples = tuple(dict(value) for value in input_samples)
        if tuple(row.row_id for row in self.rows) != tuple(range(48)):
            raise RuntimeContractError("source row IDs must be exactly 0..47")
        if tuple(row.sample_id for row in self.rows) != tuple(
            str(value["sample_id"]) for value in self.input_samples
        ):
            raise RuntimeContractError("source sample order differs from input manifest")

    @classmethod
    def discover(cls) -> "HistoryHardSources":
        validate_frozen_evidence(include_target=False)
        corpus = json.loads(CORPUS_RESULT.read_bytes())
        input_manifest = json.loads(INPUT_MANIFEST.read_bytes())
        cohort = corpus.get("cohort")
        records = corpus.get("row_records")
        samples = input_manifest.get("core", {}).get("samples")
        if (
            not isinstance(cohort, list)
            or not isinstance(records, list)
            or not isinstance(samples, list)
        ):
            raise RuntimeContractError("corpus/input row records are missing")
        by_row = {int(record["row"]["row_id"]): record for record in records}
        rows = []
        for expected_row, sample in zip(cohort, samples, strict=True):
            row_id = int(expected_row["row_id"])
            if row_id not in by_row or row_id not in range(48):
                raise RuntimeContractError("corpus row record set changed")
            record = by_row[row_id]
            receipt = _identity_from_record(record["receipt"], f"row {row_id} receipt")
            receipt.verify(root=CORPUS_ROOT)
            receipt_value = json.loads(receipt.path.read_bytes())
            if receipt_value.get("status") != "passed" or receipt_value.get("row") != expected_row:
                raise RuntimeContractError(f"row {row_id} receipt identity changed")
            artifacts = receipt_value.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise RuntimeContractError(f"row {row_id} artifacts missing")
            model = _identity_from_record(artifacts["model_npz"], f"row {row_id} model")
            target = _identity_from_record(
                artifacts["privileged_target_npz"], f"row {row_id} target"
            )
            model.verify(root=CORPUS_ROOT)
            # The target path/hash is discovered from the receipt but is not
            # opened or hashed again during target-blind cache construction.
            expected_sample_id = hashlib.sha256(
                f"{DECISION_AUDIT_ID}\0{model.sha256}".encode()
            ).hexdigest()
            if sample.get("sample_id") != expected_sample_id:
                raise RuntimeContractError(f"row {row_id} sample identity changed")
            rows.append(RowSource(row_id, model, target, receipt, expected_sample_id))
        return cls(rows, samples)


def _elapsed_indices(elapsed: np.ndarray, required: Sequence[int]) -> dict[int, int]:
    if elapsed.ndim != 1 or elapsed.dtype != np.int64:
        raise RuntimeContractError("elapsed_steps must be int64 vector")
    result = {}
    for value in required:
        found = np.flatnonzero(elapsed == value)
        if found.shape != (1,):
            raise RuntimeContractError(f"elapsed {value} is not unique")
        result[value] = int(found[0])
    return result


def _block_mean_16(value: np.ndarray) -> np.ndarray:
    if value.shape[-3:] != (128, 128, 3) or value.dtype != np.uint8:
        raise RuntimeContractError("block mean requires uint8 [...,128,128,3]")
    prefix = value.shape[:-3]
    normalized = value.astype(np.float32) / np.float32(255.0)
    blocked = normalized.reshape(*prefix, 8, 16, 8, 16, 3)
    return np.ascontiguousarray(blocked.mean(axis=(-4, -2), dtype=np.float32))


def extract_row_features(row: RowSource, sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
    row.model.verify(root=CORPUS_ROOT)
    with np.load(row.model.path, allow_pickle=False) as archive:
        if set(archive.files) != set(MODEL_FIELDS):
            raise RuntimeContractError(f"row {row.row_id} model field allowlist changed")
        elapsed = np.asarray(archive["elapsed_steps"])
        indices = _elapsed_indices(elapsed, (*range(1, 131), 132))
        rgb = np.asarray(archive["rgb"])
        proprio = np.asarray(archive["proprio"])
        valid = np.asarray(archive["valid_mask"])
        camera_order = tuple(str(value) for value in np.asarray(archive["camera_order"]).tolist())
        length = len(elapsed)
        if (
            rgb.shape != (length, 128, 128, 6)
            or rgb.dtype != np.uint8
            or proprio.shape != (length, 7)
            or proprio.dtype != np.float32
            or valid.shape != (length,)
            or valid.dtype != np.bool_
            or camera_order != ("base_camera", "hand_camera")
        ):
            raise RuntimeContractError(f"row {row.row_id} model array contract changed")
        selected = np.asarray([indices[value] for value in (*range(1, 131), 132)])
        if not bool(valid[selected].all()):
            raise RuntimeContractError(f"row {row.row_id} selected frame is invalid")
        cue_rgb = np.ascontiguousarray(rgb[indices[1] : indices[130] + 1])
        current_rgb = np.ascontiguousarray(rgb[indices[132]])
        current_proprio = np.ascontiguousarray(proprio[indices[132]])
    identities = {
        "temporal_rgb_1_130": array_identity(cue_rgb),
        "current_rgb_132": array_identity(current_rgb),
        "current_proprio_132": array_identity(current_proprio),
    }
    for key, identity in identities.items():
        if sample.get(key) != identity:
            raise RuntimeContractError(f"row {row.row_id} input manifest {key} changed")
    cue_grid = np.stack(
        (_block_mean_16(cue_rgb[..., :3]), _block_mean_16(cue_rgb[..., 3:])),
        axis=1,
    )
    current_grid = np.stack(
        (_block_mean_16(current_rgb[..., :3]), _block_mean_16(current_rgb[..., 3:])),
        axis=0,
    )
    repeated = np.broadcast_to(current_grid[None], cue_grid.shape)
    temporal = np.concatenate((cue_grid, repeated, cue_grid - repeated), axis=-1)
    return {
        "temporal_grid": np.ascontiguousarray(temporal, dtype=np.float32),
        "current_rgb_grid": np.ascontiguousarray(current_grid, dtype=np.float32),
        "current_proprio": np.ascontiguousarray(current_proprio, dtype=np.float32),
    }


def validate_cache_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(CACHE_FIELDS):
        raise RuntimeContractError("cache field set changed")
    expected = {
        "temporal_grid": (48, 130, 2, 8, 8, 9),
        "current_rgb_grid": (48, 2, 8, 8, 3),
        "current_proprio": (48, 7),
    }
    for key, shape in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeContractError(f"cache {key} violates {shape}/float32/finite")
    cue, current, delta = np.split(arrays["temporal_grid"], 3, axis=-1)
    repeated = np.broadcast_to(arrays["current_rgb_grid"][:, None], current.shape)
    if not np.array_equal(current, repeated) or not np.allclose(
        delta, cue - current, rtol=0.0, atol=1e-7
    ):
        raise RuntimeContractError("cache cue/current/delta channel contract changed")


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **{key: arrays[key] for key in sorted(arrays)})
    return stream.getvalue()


def write_once(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise FileExistsError(path) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    path.chmod(0o444)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "mode": "0444",
    }


def freeze_directory(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build_target_blind_cache(
    output_root: Path, *, source_acceptance: Mapping[str, Any]
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    sources = HistoryHardSources.discover()
    values = [
        extract_row_features(row, sample)
        for row, sample in zip(sources.rows, sources.input_samples, strict=True)
    ]
    arrays = {
        key: np.ascontiguousarray(np.stack([value[key] for value in values]))
        for key in CACHE_FIELDS
    }
    validate_cache_arrays(arrays)
    output_root.mkdir(parents=True)
    feature = write_once(output_root / "FEATURES.npz", _npz_bytes(arrays))
    result = {
        "schema": f"{EXPERIMENT_ID}/cache-v1",
        "status": "passed",
        "mode": "target_blind_cache",
        "rows": 48,
        "cache_fields": sorted(CACHE_FIELDS),
        "feature": feature,
        "sample_ids": [row.sample_id for row in sources.rows],
        "source_model_sha256": [row.model.sha256 for row in sources.rows],
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "corpus_result_sha256": CORPUS_RESULT_SHA256,
        "targets_opened": False,
        "target_manifest_opened": False,
        "fresh_data_opened": False,
        "dev_model_outputs": 0,
        "confirmation_model_outputs": 0,
        "retry_count": 0,
        "source_acceptance": dict(source_acceptance),
        "evidence": validate_frozen_evidence(include_target=False),
    }
    result_identity = write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return {**result, "result_identity": result_identity}


@dataclass(frozen=True)
class CachedOrderData:
    arrays: dict[str, np.ndarray]
    labels: np.ndarray
    groups: np.ndarray
    splits: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def indices_for_groups(self, group_ids: Sequence[int]) -> np.ndarray:
        requested = tuple(int(value) for value in group_ids)
        if any(value not in (*TRAIN_GROUPS, *DEV_GROUPS) for value in requested):
            raise PermissionError("fresh/unregistered groups are forbidden")
        indices = np.flatnonzero(np.isin(self.groups, requested))
        if len(indices) != ROWS_PER_GROUP * len(requested):
            raise RuntimeContractError("group row count changed")
        return indices

    def batch_for_groups(self, group_ids: Sequence[int]) -> tuple[OrderBatch, torch.Tensor]:
        indices = self.indices_for_groups(group_ids)
        batch = OrderBatch(
            temporal_grid=torch.from_numpy(
                np.ascontiguousarray(self.arrays["temporal_grid"][indices])
            ),
            current_rgb_grid=torch.from_numpy(
                np.ascontiguousarray(self.arrays["current_rgb_grid"][indices])
            ),
            current_proprio=torch.from_numpy(
                np.ascontiguousarray(self.arrays["current_proprio"][indices])
            ),
        )
        labels = torch.from_numpy(np.ascontiguousarray(self.labels[indices], dtype=np.int64))
        return batch, labels


def _load_cache_result(cache_root: Path) -> tuple[dict[str, Any], MaterialIdentity]:
    result_path = cache_root.resolve() / "RESULT.json"
    if cache_root.is_symlink() or not result_path.is_file() or result_path.is_symlink():
        raise RuntimeContractError("cache root/result is invalid")
    raw = result_path.read_bytes()
    identity = MaterialIdentity(result_path, len(raw), sha256_bytes(raw))
    identity.verify(root=cache_root)
    result = json.loads(raw)
    if (
        result.get("status") != "passed"
        or result.get("mode") != "target_blind_cache"
        or result.get("targets_opened") is not False
        or result.get("target_manifest_opened") is not False
        or result.get("fresh_data_opened") is not False
        or result.get("rows") != 48
        or result.get("cache_fields") != sorted(CACHE_FIELDS)
    ):
        raise RuntimeContractError("cache result boundary changed")
    return result, identity


def load_order_data(cache_root: Path) -> tuple[CachedOrderData, dict[str, Any]]:
    """Open privileged labels only after the immutable target-blind cache passes."""

    result, result_identity = _load_cache_result(cache_root)
    feature_spec = result.get("feature")
    if not isinstance(feature_spec, Mapping):
        raise RuntimeContractError("cache feature identity missing")
    feature = MaterialIdentity(
        Path(str(feature_spec["path"])),
        int(feature_spec["bytes"]),
        str(feature_spec["sha256"]),
    )
    expected_feature = cache_root.resolve() / "FEATURES.npz"
    if feature.path.resolve() != expected_feature:
        raise RuntimeContractError("cache feature escapes root")
    feature.verify(root=cache_root)
    with np.load(feature.path, allow_pickle=False) as archive:
        if set(archive.files) != set(CACHE_FIELDS):
            raise RuntimeContractError("cache NPZ fields changed")
        arrays = {key: np.ascontiguousarray(archive[key]) for key in CACHE_FIELDS}
    validate_cache_arrays(arrays)

    validate_frozen_evidence(include_target=True)
    sources = HistoryHardSources.discover()
    target_manifest = json.loads(TARGET_MANIFEST.read_bytes())
    target_samples = target_manifest.get("core", {}).get("samples")
    if not isinstance(target_samples, list) or len(target_samples) != 48:
        raise RuntimeContractError("target sample list changed")
    if tuple(result.get("sample_ids", ())) != tuple(row.sample_id for row in sources.rows):
        raise RuntimeContractError("cache/source sample association changed")
    labels, groups, splits = [], [], []
    for row, target_record in zip(sources.rows, target_samples, strict=True):
        if target_record.get("sample_id") != row.sample_id:
            raise RuntimeContractError(f"row {row.row_id} target sample ID changed")
        row.target.verify(root=CORPUS_ROOT)
        if row.target.sha256 != target_record.get("source_target_npz_sha256"):
            raise RuntimeContractError(f"row {row.row_id} target hash changed")
        with np.load(row.target.path, allow_pickle=False) as archive:
            if set(archive.files) != set(TARGET_FIELDS):
                raise RuntimeContractError(f"row {row.row_id} target fields changed")
            ordered = np.asarray(archive["ordered_path_ids"])
            group_id = int(np.asarray(archive["group_id"]).item())
            pair_code = str(np.asarray(archive["pair_code"]).item())
            target_row = int(np.asarray(archive["row_id"]).item())
        if (
            ordered.shape != (2,)
            or ordered.dtype != np.int64
            or ordered.tolist() != target_record.get("ordered_path_ids")
            or group_id != target_record.get("group_id")
            or pair_code != target_record.get("ordered_pair_code")
            or target_row != row.row_id
            or pair_code not in PAIR_CODES
        ):
            raise RuntimeContractError(f"row {row.row_id} privileged label contract changed")
        split = str(target_record.get("split"))
        if split != ("train" if group_id in TRAIN_GROUPS else "dev"):
            raise RuntimeContractError(f"row {row.row_id} split/group contract changed")
        labels.append(ordered)
        groups.append(group_id)
        splits.append(split)
    data = CachedOrderData(
        arrays=arrays,
        labels=np.ascontiguousarray(np.stack(labels), dtype=np.int64),
        groups=np.ascontiguousarray(groups, dtype=np.int64),
        splits=tuple(splits),
        sample_ids=tuple(row.sample_id for row in sources.rows),
    )
    for group_id in (*TRAIN_GROUPS, *DEV_GROUPS):
        indices = data.indices_for_groups((group_id,))
        if not np.array_equal(
            data.labels[indices], np.asarray([[int(c[0]), int(c[1])] for c in PAIR_CODES])
        ):
            raise RuntimeContractError(f"group {group_id} ordered sixpack changed")
    return data, {
        "cache_result": result_identity.as_json(),
        "target_manifest_sha256": TARGET_MANIFEST_SHA256,
        "targets_opened_after_cache": True,
        "fresh_data_opened": False,
    }


class GroupBalancedSchedule:
    """Cycle all 15 four-of-six train group combinations in seeded order."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.combinations = tuple(itertools.combinations(TRAIN_GROUPS, BATCH_GROUPS))
        if len(self.combinations) != 15:
            raise RuntimeError("four-of-six combination contract changed")

    def batch(self, step: int) -> tuple[int, ...]:
        if step < 0:
            raise ValueError("step must be nonnegative")
        epoch, offset = divmod(step, len(self.combinations))
        ordered = list(self.combinations)
        random.Random(self.seed + epoch).shuffle(ordered)
        return tuple(ordered[offset])
