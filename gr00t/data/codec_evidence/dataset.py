"""Dataset composition for optional, immutable codec evidence.

The adapter JSON is the only configuration entrance.  Missing configuration
and ``enabled=false`` return the original dataset object before resolving a
fixture root, allocating a store, or reading codec artifacts.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .codec_contract import codec_contract_sha256, validate_codec_contract
from .config import CodecEvidenceConfig, parse_codec_evidence_config
from .contract import REASON_CODES, SHA256_RE, CodecEvidenceKey
from .episode_bundle import (
    BUNDLE_SCHEMA_VERSION,
    SOURCE_KINDS,
    EpisodeBundle,
    EpisodeBundleKey,
    EpisodeBundleStore,
)
from .identity import EPISODE_IDENTITY_SCHEMA_VERSION
from .processor import CodecEvidenceProcessorCore
from .store import FixtureManifestCodecEvidenceStore


EPISODE_BUNDLE_INDEX_SCHEMA_VERSION = "codec-evidence-episode-bundle-index-v1"
FIXTURE_MANIFEST_FILENAME = "fixture_manifest.json"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_regular_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Fixture artifact paths must be non-empty root-relative paths")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Symlinks are forbidden in fixture artifact paths")
    resolved = (root / relative).resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ValueError("Fixture artifact path escapes its root")
    if not resolved.is_file():
        raise ValueError("Fixture artifact must be a regular file")
    return resolved


def episode_bundle_store_from_index_manifest(
    root: str | Path,
    *,
    fixture_manifest_sha256: str,
    camera_id: str,
    source_kind: str,
) -> Any:
    """Open a hash-pinned bundle index without generating any fixture data."""

    if not isinstance(fixture_manifest_sha256, str) or not SHA256_RE.fullmatch(
        fixture_manifest_sha256
    ):
        raise ValueError("fixture_manifest_sha256 must be a lowercase SHA-256")
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("Episode-bundle fixture root must not be a symlink")
    root_path = root_path.resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError("Episode-bundle fixture root must be a directory")
    manifest_path = _safe_regular_file(root_path, FIXTURE_MANIFEST_FILENAME)
    if _sha256_path(manifest_path) != fixture_manifest_sha256:
        raise ValueError("Episode-bundle index SHA-256 mismatch")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    from .warehouse import FULL_FIXTURE_MANIFEST_SCHEMA_VERSION

    if (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == FULL_FIXTURE_MANIFEST_SCHEMA_VERSION
    ):
        from .warehouse import WarehouseEpisodeBundleStore

        store = WarehouseEpisodeBundleStore(
            root_path,
            fixture_manifest_sha256=fixture_manifest_sha256,
        )
        if store.manifest["camera_id"] != camera_id or store.manifest["source_kind"] != source_kind:
            raise ValueError("Full fixture manifest identity differs from adapter config")
        return store
    required = {
        "schema_version",
        "complete",
        "camera_id",
        "source_kind",
        "identity",
        "codec_contract",
        "bundles",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Episode-bundle index fields do not match the frozen contract")
    if (
        manifest["schema_version"] != EPISODE_BUNDLE_INDEX_SCHEMA_VERSION
        or manifest["complete"] is not True
        or manifest["camera_id"] != camera_id
        or manifest["source_kind"] != source_kind
    ):
        raise ValueError("Episode-bundle index identity does not match adapter config")
    identity = manifest["identity"]
    if not isinstance(identity, Mapping) or set(identity) != {
        "episode_identity_schema_version",
        "codec_contract_sha256",
    }:
        raise ValueError("Episode-bundle index identity block is malformed")
    if identity["episode_identity_schema_version"] != EPISODE_IDENTITY_SCHEMA_VERSION:
        raise ValueError("Episode-bundle index episode identity schema differs")
    codec_contract = validate_codec_contract(manifest["codec_contract"])
    contract_sha256 = codec_contract_sha256(codec_contract)
    if identity["codec_contract_sha256"] != contract_sha256:
        raise ValueError("Episode-bundle index codec_contract SHA-256 mismatch")
    if codec_contract["camera_id"] != camera_id or codec_contract["source_kind"] != source_kind:
        raise ValueError("Episode-bundle index codec_contract differs from adapter config")
    entries = manifest["bundles"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Episode-bundle index must contain at least one bundle")
    pins: dict[EpisodeBundleKey, str] = {}
    for entry in entries:
        fields = {"episode_uid", "camera_id", "source_kind", "relative_path", "sha256", "bytes"}
        if not isinstance(entry, Mapping) or set(entry) != fields:
            raise ValueError("Episode-bundle index entry fields do not match the contract")
        key = EpisodeBundleKey(
            episode_uid=entry["episode_uid"],
            camera_id=entry["camera_id"],
            source_kind=entry["source_kind"],
        )
        if key.camera_id != camera_id or key.source_kind != source_kind:
            raise ValueError("Episode-bundle index mixes camera or source-kind identities")
        if key in pins:
            raise ValueError(f"Duplicate episode bundle key: {key.canonical}")
        expected_relative = f"{key.episode_uid}/{key.camera_id}/{key.source_kind}/bundle.json"
        if entry["relative_path"] != expected_relative:
            raise ValueError("Episode-bundle index relative path is not canonical")
        bundle_manifest = _safe_regular_file(root_path, expected_relative)
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ValueError("Episode-bundle index byte count must be non-negative integer")
        if bundle_manifest.stat().st_size != entry["bytes"]:
            raise ValueError("Episode-bundle manifest byte count mismatch")
        if not isinstance(entry["sha256"], str) or not SHA256_RE.fullmatch(entry["sha256"]):
            raise ValueError("Episode-bundle manifest SHA-256 is malformed")
        if _sha256_path(bundle_manifest) != entry["sha256"]:
            raise ValueError("Episode-bundle manifest SHA-256 mismatch")
        pins[key] = entry["sha256"]
    store = EpisodeBundleStore(root_path, manifest_sha256_by_key=pins)
    store.coverage_kind = "LEGACY_SUBSET_NONTRAINING"
    store.fixture_manifest_schema_version = EPISODE_BUNDLE_INDEX_SCHEMA_VERSION
    for key in store.keys:
        bundle_identity = store.identity(key)
        if (
            bundle_identity["codec_contract_sha256"] != contract_sha256
            or bundle_identity["codec_contract"] != codec_contract
        ):
            raise ValueError("Episode-bundle index contains cross-bundle codec contract drift")
    return store


def _episode_uid_for_base_dataset(base_dataset: Any, episode_index: int) -> str:
    """Resolve the stable content-derived UID exposed by an adapter loader."""

    episode_loader = getattr(base_dataset, "episode_loader", None)
    for candidate in (episode_loader, base_dataset):
        selected = getattr(candidate, "selected_episode_uids", None)
        if selected is not None:
            try:
                value = selected[episode_index]
            except (IndexError, KeyError, TypeError) as error:
                raise RuntimeError("Episode UID index does not cover the dataset") from error
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise ValueError("Adapter episode UID must be a lowercase SHA-256")
            return value
    mapping = getattr(episode_loader, "mapping", None)
    loaders = getattr(episode_loader, "loaders", None)
    if mapping is not None and loaders is not None:
        loader_index, local_index = mapping[episode_index]
        selected = getattr(loaders[loader_index], "selected_episode_uids", None)
        if selected is not None:
            value = selected[local_index]
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise ValueError("Adapter episode UID must be a lowercase SHA-256")
            return value
    raise RuntimeError("Codec evidence requires stable episode UID provenance")


class CodecEvidenceAugmentedDataset:
    """Attach one validated evidence view before the native processor runs."""

    def __init__(
        self,
        base_dataset: Any,
        config: CodecEvidenceConfig,
        *,
        store: Any | None = None,
        processor: CodecEvidenceProcessorCore | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("The augmented dataset requires enabled codec evidence")
        self.base_dataset = base_dataset
        self.config = config
        self._episode_bundle_mode = config.schema_version == BUNDLE_SCHEMA_VERSION
        if self._episode_bundle_mode and config.source_kind not in SOURCE_KINDS:
            raise ValueError("Episode-bundle mode requires an explicit source_kind")
        if store is None:
            if self._episode_bundle_mode:
                store = episode_bundle_store_from_index_manifest(
                    config.fixture_root,
                    fixture_manifest_sha256=config.fixture_manifest_sha256,
                    camera_id=config.camera_id,
                    source_kind=config.source_kind,
                )
            else:
                store = FixtureManifestCodecEvidenceStore(
                    config.fixture_root,
                    fixture_manifest_sha256=config.fixture_manifest_sha256,
                    record_schema_sha256=config.record_schema_sha256,
                )
        self.store = store
        self.codec_processor = processor or CodecEvidenceProcessorCore()
        self.sharded_episodes = base_dataset.sharded_episodes
        self._bundle_cache: OrderedDict[EpisodeBundleKey, EpisodeBundle] = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)

    def __len__(self) -> int:
        return len(self.sharded_episodes)

    def get_shard_length(self, index: int) -> int:
        return sum(len(step_indices) for _, step_indices in self.sharded_episodes[index])

    def _bundle(self, key: EpisodeBundleKey) -> EpisodeBundle:
        bundle = self._bundle_cache.get(key)
        if bundle is None:
            bundle = self.store.get(key)
            self._bundle_cache[key] = bundle
            if len(self._bundle_cache) > 2:
                self._bundle_cache.popitem(last=False)
        else:
            self._bundle_cache.move_to_end(key)
        return bundle

    def get_shard(self, index: int) -> list[dict[str, Any]]:
        datapoints: list[dict[str, Any]] = []
        for episode_index, step_indices in self.sharded_episodes[index]:
            episode_data = self.base_dataset.episode_loader[episode_index]
            episode_uid = _episode_uid_for_base_dataset(self.base_dataset, episode_index)
            for step_index in step_indices:
                row = int(step_index)
                if self._episode_bundle_mode:
                    key = EpisodeBundleKey(
                        episode_uid=episode_uid,
                        camera_id=self.config.camera_id,
                        source_kind=self.config.source_kind,
                    )
                    sample = self.codec_processor.process(self._bundle(key).view(row))
                else:
                    key = CodecEvidenceKey(episode_uid, row, self.config.camera_id)
                    sample = self.codec_processor.process(self.store.get(key))
                if (
                    not bool(sample.sample_valid)
                    or int(sample.sample_reason_code) != REASON_CODES["NONE"]
                ):
                    raise ValueError(
                        "Enabled training Codec evidence must be valid with reason NONE"
                    )
                datapoint = self.base_dataset.get_datapoint(episode_data, row)
                if "codec_evidence_sample" in datapoint or "codec_evidence" in datapoint:
                    raise ValueError("Native dataset unexpectedly emitted codec evidence")
                datapoint["codec_evidence_sample"] = sample
                datapoints.append(datapoint)
        return datapoints


def codec_config_from_adapter_path(path: str | Path) -> CodecEvidenceConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError("Adapter config must contain one JSON object")
    return parse_codec_evidence_config(value.get("codec_evidence"))


def maybe_wrap_codec_evidence_dataset(
    dataset: Any,
    adapter_config_path: str | Path | None,
    *,
    store: Any | None = None,
    config: CodecEvidenceConfig | None = None,
) -> Any:
    """Return ``dataset`` unchanged unless its explicit codec block is enabled."""

    if config is None:
        if adapter_config_path is None:
            return dataset
        config = codec_config_from_adapter_path(adapter_config_path)
    if not config.enabled:
        return dataset
    return CodecEvidenceAugmentedDataset(dataset, config, store=store)


__all__ = [
    "EPISODE_BUNDLE_INDEX_SCHEMA_VERSION",
    "FIXTURE_MANIFEST_FILENAME",
    "CodecEvidenceAugmentedDataset",
    "codec_config_from_adapter_path",
    "episode_bundle_store_from_index_manifest",
    "maybe_wrap_codec_evidence_dataset",
]
