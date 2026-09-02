"""Dataset composition that attaches codec evidence before native processing."""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .config import CodecEvidenceConfig, parse_codec_evidence_config
from .contract import CodecEvidenceKey
from .episode_bundle import (
    BUNDLE_SCHEMA_VERSION,
    MAX_EPISODE_BATCH_SIZE,
    SOURCE_KINDS,
    EpisodeBundle,
    EpisodeBundleKey,
    EpisodeBundleStore,
)
from .processor import CodecEvidenceProcessorCore
from .store import FixtureManifestCodecEvidenceStore


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def episode_bundle_store_from_pilot_manifest(
    root: str | Path, *, pilot_manifest_sha256: str
) -> EpisodeBundleStore:
    """Create a store only after closing the pinned six-bundle pilot index."""

    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError("Episode-bundle fixture_root must be a directory")
    manifest_path = root_path / "pilot_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Pinned pilot_manifest.json is missing or a symlink")
    if _sha256_path(manifest_path) != pilot_manifest_sha256:
        raise ValueError("Pilot manifest SHA-256 mismatch")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    required = {
        "schema_version",
        "complete",
        "pilot_generation_attempt",
        "source_kinds",
        "camera_id",
        "rows",
        "bundles",
        "bundle_count",
        "episode_uids",
        "determinism_encode_repetitions_per_bundle",
        "total_internal_encode_executions",
        "output_policy",
        "dataset_mutations",
        "gpu_used",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Pilot manifest keys do not match the R5 contract")
    if (
        manifest["schema_version"] != "codec-evidence-r5-pilot-v1"
        or manifest["complete"] is not True
        or manifest["dataset_mutations"] != 0
        or manifest["gpu_used"] is not False
        or manifest["rows"] != [0, 59]
        or set(manifest["source_kinds"]) != set(SOURCE_KINDS)
    ):
        raise ValueError("Pilot manifest violates the accepted CPU fixture contract")
    entries = manifest["bundles"]
    if (
        not isinstance(entries, list)
        or manifest["bundle_count"] != len(entries)
        or len(entries) != 6
        or not isinstance(manifest["episode_uids"], list)
        or len(manifest["episode_uids"]) != 3
        or len(set(manifest["episode_uids"])) != 3
    ):
        raise ValueError("R5 pilot must contain exactly three episodes and six bundles")
    pins: dict[EpisodeBundleKey, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Pilot bundle entries must be objects")
        identity_fields = {
            "episode_uid",
            "camera_id",
            "source_kind",
            "relative_path",
            "sha256",
            "bytes",
            "frames",
        }
        if not identity_fields.issubset(entry):
            raise ValueError("Pilot bundle entry lacks immutable identity fields")
        key = EpisodeBundleKey(entry["episode_uid"], entry["camera_id"], entry["source_kind"])
        if key in pins:
            raise ValueError(f"Duplicate pilot bundle key: {key.canonical}")
        expected_relative = f"{key.episode_uid}/{key.camera_id}/{key.source_kind}/bundle.json"
        if entry["relative_path"] != expected_relative:
            raise ValueError("Pilot bundle relative path is not canonical")
        bundle_path = root_path / expected_relative
        if bundle_path.is_symlink() or not bundle_path.is_file():
            raise ValueError("Pilot bundle manifest artifact is missing or a symlink")
        if type(entry["bytes"]) is not int or bundle_path.stat().st_size != entry["bytes"]:
            raise ValueError("Pilot bundle byte count mismatch")
        if _sha256_path(bundle_path) != entry["sha256"]:
            raise ValueError("Pilot bundle SHA-256 mismatch")
        if entry["frames"] != 60:
            raise ValueError("R5 pilot bundles must cover exact rows 0..59")
        pins[key] = entry["sha256"]
    if set(manifest["episode_uids"]) != {key.episode_uid for key in pins}:
        raise ValueError("Pilot episode UID index disagrees with its bundles")
    expected_keys = {
        (uid, manifest["camera_id"], source_kind)
        for uid in manifest["episode_uids"]
        for source_kind in SOURCE_KINDS
    }
    if {(key.episode_uid, key.camera_id, key.source_kind) for key in pins} != expected_keys:
        raise ValueError("Pilot must contain both source kinds for every episode")
    return EpisodeBundleStore(root_path, manifest_sha256_by_key=pins)


def _live_snapshot_inputs(base_dataset: Any) -> tuple[Any, dict[str, Any], list[str]]:
    """Resolve the warehouse binding from the DatasetFactory object actually in use."""

    guard = getattr(base_dataset, "dataset_generation_guard", None)
    if guard is None:
        raise RuntimeError("Production codec snapshots require a dataset-generation shared lease")
    guard.ensure_current_process()
    guard.verify_frozen_identity()
    identity = guard.identity
    with identity.split_manifest_path.open("r", encoding="utf-8") as stream:
        split_manifest = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    train_uids = sorted(
        row["episode_uid"]
        for row in split_manifest.get("assignments", [])
        if isinstance(row, dict) and row.get("eligible") and row.get("split") == "train"
    )
    if not train_uids or len(train_uids) != len(set(train_uids)):
        raise ValueError("Live split does not resolve one unique train episode UID set")

    episode_loader = getattr(base_dataset, "episode_loader", None)
    adapter_config = getattr(episode_loader, "adapter_config", None)
    if not isinstance(adapter_config, dict) or adapter_config.get("split") != "train":
        raise ValueError("Production codec snapshots may only augment the frozen train split")
    loaders = getattr(episode_loader, "loaders", None)
    if not isinstance(loaders, list) or not loaders:
        raise ValueError("Manifest collection loaders are unavailable for UID cross-checking")
    selected_uids = sorted(
        uid for loader in loaders for uid in getattr(loader, "selected_episode_uids", ())
    )
    if selected_uids != train_uids:
        raise ValueError("DatasetFactory episode UIDs differ from the frozen train manifest")

    binding = {
        "dataset_root": str(identity.dataset_root),
        "catalog_database_path": str(identity.catalog_path),
        "catalog_database_sha256": identity.catalog_sha256,
        "split_manifest_path": str(identity.split_manifest_path),
        "split_manifest_sha256": identity.split_manifest_sha256,
        "payload_manifest_path": str(identity.payload_manifest_path),
        "payload_manifest_sha256": identity.payload_manifest_sha256,
        "train_statistics_path": str(identity.train_statistics_path),
        "train_statistics_sha256": identity.train_statistics_sha256,
        "train_episode_uid_digest": identity.train_episode_uid_digest,
        "resolved_train_episode_count": len(train_uids),
    }
    return guard, binding, train_uids


def episode_bundle_store_from_snapshot_manifest(
    base_dataset: Any,
    config: CodecEvidenceConfig,
    *,
    verifier_only: bool = False,
) -> Any:
    """Open one explicitly pinned production snapshot through the warehouse oracle."""

    if config.fixture_root is None or config.fixture_manifest_sha256 is None:
        raise ValueError("Enabled codec snapshot config lacks its root or manifest pin")
    guard, binding, train_uids = _live_snapshot_inputs(base_dataset)
    from .warehouse import open_fixture_snapshot

    store = open_fixture_snapshot(
        config.fixture_root,
        snapshot_manifest_sha256=config.fixture_manifest_sha256,
        expected_dataset_binding=binding,
        expected_train_episode_uids=train_uids,
        camera_id=config.camera_id,
        source_kind=config.source_kind,
        verifier_only=verifier_only,
    )
    guard.verify_frozen_identity()
    return store


def _bounded_episode_shards(shards: Any) -> list[list[tuple[int, Any]]]:
    """Preserve sample order while preventing a 1024-view materialization."""

    bounded: list[list[tuple[int, Any]]] = []
    for shard in shards:
        current: list[tuple[int, Any]] = []
        current_size = 0
        for episode_index, step_indices in shard:
            offset = 0
            while offset < len(step_indices):
                remaining = MAX_EPISODE_BATCH_SIZE - current_size
                take = min(remaining, len(step_indices) - offset)
                current.append((episode_index, step_indices[offset : offset + take]))
                current_size += take
                offset += take
                if current_size == MAX_EPISODE_BATCH_SIZE:
                    bounded.append(current)
                    current = []
                    current_size = 0
        if current:
            bounded.append(current)
    return bounded


def _episode_uid_for_base_dataset(base_dataset: Any, episode_index: int) -> str:
    episode_loader = getattr(base_dataset, "episode_loader", None)
    candidates = [episode_loader, base_dataset]
    for candidate in candidates:
        if candidate is None:
            continue
        selected = getattr(candidate, "selected_episode_uids", None)
        if selected is not None:
            return selected[episode_index]
    mapping = getattr(episode_loader, "mapping", None)
    loaders = getattr(episode_loader, "loaders", None)
    if mapping is not None and loaders is not None:
        loader_index, local_index = mapping[episode_index]
        selected = getattr(loaders[loader_index], "selected_episode_uids", None)
        if selected is not None:
            return selected[local_index]
    raise RuntimeError("Codec evidence requires manifest-filtered episode UID provenance")


class CodecEvidenceAugmentedDataset:
    """Composition wrapper preserving the native dataset and processor contract."""

    def __init__(
        self,
        base_dataset: Any,
        config: CodecEvidenceConfig,
        *,
        store: Any | None = None,
        processor: CodecEvidenceProcessorCore | None = None,
        verifier_only: bool = False,
    ):
        if not config.enabled:
            raise ValueError("The augmented dataset requires enabled codec evidence")
        self.base_dataset = base_dataset
        self.config = config
        self.verifier_only = bool(verifier_only)
        self.dataset_generation_guard = getattr(base_dataset, "dataset_generation_guard", None)
        self._episode_bundle_mode = config.schema_version == BUNDLE_SCHEMA_VERSION
        if self._episode_bundle_mode and config.source_kind not in SOURCE_KINDS:
            raise ValueError("Episode-bundle datasets require an explicit source_kind")
        self.store = store
        if self.store is None:
            if self._episode_bundle_mode:
                from .warehouse import SNAPSHOT_FILENAME

                snapshot_manifest = Path(config.fixture_root) / SNAPSHOT_FILENAME
                if snapshot_manifest.is_file() or snapshot_manifest.is_symlink():
                    self.store = episode_bundle_store_from_snapshot_manifest(
                        base_dataset,
                        config,
                        verifier_only=self.verifier_only,
                    )
                else:
                    # Legacy R5 pilot compatibility remains explicit.  New C0
                    # canaries/full-split fixtures always publish SNAPSHOT_FILENAME.
                    self.store = episode_bundle_store_from_pilot_manifest(
                        config.fixture_root,
                        pilot_manifest_sha256=config.fixture_manifest_sha256,
                    )
            else:
                self.store = FixtureManifestCodecEvidenceStore(
                    config.fixture_root,
                    fixture_manifest_sha256=config.fixture_manifest_sha256,
                    record_schema_sha256=config.record_schema_sha256,
                )
        self.codec_processor = processor or CodecEvidenceProcessorCore()
        self._bundle_cache: OrderedDict[EpisodeBundleKey, EpisodeBundle] = OrderedDict()
        source_shards = base_dataset.sharded_episodes
        if self.verifier_only:
            from .warehouse import CANARY_COVERAGE

            manifest = getattr(self.store, "manifest", None)
            if not isinstance(manifest, dict) or manifest.get("coverage_kind") != CANARY_COVERAGE:
                raise ValueError("Verifier-only wrapper requires a validated canary snapshot")
            covered_uids = set(manifest["covered_episode_uids"])
            source_shards = [
                [
                    (episode_index, step_indices)
                    for episode_index, step_indices in shard
                    if _episode_uid_for_base_dataset(base_dataset, episode_index) in covered_uids
                ]
                for shard in source_shards
            ]
            source_shards = [shard for shard in source_shards if shard]
            observed = {
                _episode_uid_for_base_dataset(base_dataset, episode_index)
                for shard in source_shards
                for episode_index, _step_indices in shard
            }
            if observed != covered_uids:
                raise ValueError(
                    "Canary snapshot UIDs are not all present in DatasetFactory shards"
                )
        self.sharded_episodes = (
            _bounded_episode_shards(source_shards) if self._episode_bundle_mode else source_shards
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)

    def __len__(self) -> int:
        self._ensure_generation_guard()
        return len(self.sharded_episodes)

    def get_shard_length(self, index: int) -> int:
        self._ensure_generation_guard()
        return sum(len(step_indices) for _, step_indices in self.sharded_episodes[index])

    def _ensure_generation_guard(self) -> None:
        if self.dataset_generation_guard is not None:
            self.dataset_generation_guard.ensure_current_process()

    def _bundle(self, key: EpisodeBundleKey) -> EpisodeBundle:
        self._ensure_generation_guard()
        bundle = self._bundle_cache.get(key)
        if bundle is None:
            bundle = self.store.get(key)
            self._bundle_cache[key] = bundle
            if len(self._bundle_cache) > 2:
                self._bundle_cache.popitem(last=False)
        else:
            self._bundle_cache.move_to_end(key)
        return bundle

    def _episode_uid(self, shard_index: int) -> str:
        return _episode_uid_for_base_dataset(self.base_dataset, shard_index)

    def get_shard(self, index: int) -> list[dict[str, Any]]:
        self._ensure_generation_guard()
        datapoints = []
        for episode_index, step_indices in self.sharded_episodes[index]:
            episode_data = self.base_dataset.episode_loader[episode_index]
            episode_uid = self._episode_uid(episode_index)
            for step_index in step_indices:
                if self._episode_bundle_mode:
                    bundle_key = EpisodeBundleKey(
                        episode_uid,
                        self.config.camera_id,
                        self.config.source_kind,
                    )
                    sample = self.codec_processor.process(
                        self._bundle(bundle_key).view(int(step_index))
                    )
                else:
                    key = CodecEvidenceKey(
                        episode_uid=episode_uid,
                        observation_row=int(step_index),
                        camera_id=self.config.camera_id,
                    )
                    sample = self.codec_processor.process(self.store.get(key))
                datapoint = self.base_dataset.get_datapoint(episode_data, int(step_index))
                if "codec_evidence_sample" in datapoint or "codec_evidence" in datapoint:
                    raise ValueError("Native dataset unexpectedly emitted a codec-evidence key")
                datapoint["codec_evidence_sample"] = sample
                datapoints.append(datapoint)
        return datapoints


def codec_config_from_adapter_path(path: str | Path) -> CodecEvidenceConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return parse_codec_evidence_config(value.get("codec_evidence"))


def maybe_wrap_codec_evidence_dataset(dataset: Any, adapter_config_path: str | Path | None) -> Any:
    if adapter_config_path is None:
        return dataset
    config = codec_config_from_adapter_path(adapter_config_path)
    if not config.enabled:
        return dataset
    return CodecEvidenceAugmentedDataset(dataset, config)


def wrap_codec_evidence_dataset_for_verification(
    dataset: Any, adapter_config_path: str | Path
) -> CodecEvidenceAugmentedDataset:
    """Explicit verifier-only entrance for a non-training canary snapshot."""

    config = codec_config_from_adapter_path(adapter_config_path)
    if not config.enabled:
        raise ValueError("Verifier-only codec wrapper requires enabled codec evidence")
    return CodecEvidenceAugmentedDataset(dataset, config, verifier_only=True)
