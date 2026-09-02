"""Optional read-only causal Codec evidence composition for H40 datasets."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

from .config import CodecEvidenceConfig, parse_codec_evidence_config
from .contract import REASON_CODES, SHA256_RE
from .episode_bundle import BUNDLE_SCHEMA_VERSION, SOURCE_KINDS, EpisodeBundle, EpisodeBundleKey
from .processor import CodecEvidenceProcessorCore
from .warehouse import WarehouseEpisodeBundleStore


FIXTURE_MANIFEST_FILENAME = "fixture_manifest.json"
EPISODE_BUNDLE_INDEX_SCHEMA_VERSION = "codec-evidence-episode-bundle-index-v1"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def episode_bundle_store_from_index_manifest(
    root: str | Path,
    *,
    fixture_manifest_sha256: str,
    camera_id: str,
    source_kind: str,
) -> WarehouseEpisodeBundleStore:
    """Open one hash-pinned complete causal fixture without writing it."""

    if not isinstance(fixture_manifest_sha256, str) or not SHA256_RE.fullmatch(
        fixture_manifest_sha256
    ):
        raise ValueError("fixture_manifest_sha256 must be a lowercase SHA-256")
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(SOURCE_KINDS)}")
    store = WarehouseEpisodeBundleStore(
        root,
        fixture_manifest_sha256=fixture_manifest_sha256,
    )
    if store.manifest["camera_id"] != camera_id:
        raise ValueError("Full fixture manifest camera identity differs")
    if store.manifest["source_kind"] != source_kind:
        raise ValueError("Full fixture manifest source identity differs")
    return store


def _episode_uid_for_base_dataset(base_dataset: Any, episode_index: int) -> str:
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
    """Attach one validated causal T8 view before the native processor runs."""

    def __init__(
        self,
        base_dataset: Any,
        config: CodecEvidenceConfig,
        *,
        store: Any | None = None,
        processor: CodecEvidenceProcessorCore | None = None,
    ) -> None:
        if not config.enabled or config.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError("The augmented dataset requires enabled episode-bundle evidence")
        if config.source_kind not in SOURCE_KINDS:
            raise ValueError("Episode-bundle mode requires an explicit source_kind")
        self.base_dataset = base_dataset
        self.config = config
        self.store = store or episode_bundle_store_from_index_manifest(
            config.fixture_root,
            fixture_manifest_sha256=config.fixture_manifest_sha256,
            camera_id=config.camera_id,
            source_kind=config.source_kind,
        )
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
            key = EpisodeBundleKey(
                episode_uid=episode_uid,
                camera_id=self.config.camera_id,
                source_kind=self.config.source_kind,
            )
            bundle = self._bundle(key)
            for step_index in step_indices:
                sample = self.codec_processor.process(bundle.view(int(step_index)))
                if (
                    not bool(sample.sample_valid)
                    or int(sample.sample_reason_code) != REASON_CODES["NONE"]
                ):
                    raise ValueError("Enabled training Codec evidence must be valid")
                datapoint = self.base_dataset.get_datapoint(episode_data, int(step_index))
                if "codec_evidence_sample" in datapoint or "codec_evidence" in datapoint:
                    raise ValueError("Native dataset unexpectedly emitted Codec evidence")
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
    """Return the native dataset unchanged unless Codec evidence is enabled."""

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
