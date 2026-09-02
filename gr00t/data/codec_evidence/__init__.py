"""Baseline-neutral causal codec-evidence contracts.

The modules imported here depend only on Python and NumPy. Framework-specific
conversion lives in gr00t.data.codec_evidence.torch_backend, and the GR00T
model adapter lives under gr00t.model.
"""

from .config import CodecEvidenceConfig, parse_codec_evidence_config
from .contract import (
    FRAME_TYPE_CODES,
    HISTORY,
    MAX_CODEC_BATCH_SIZE,
    MAX_CODEC_TRANSPORT_BYTES,
    MAX_MV_PER_FRAME,
    MAX_PADDED_MV_ELEMENTS,
    MV_FIELD_DTYPES,
    REASON_CODES,
    CodecEvidenceBatch,
    CodecEvidenceKey,
    CodecEvidenceRecord,
    CodecEvidenceSample,
    assert_batch_conforms,
    collate_codec_evidence_numpy,
    preflight_codec_batch_allocation,
)
from .dataset import (
    CodecEvidenceAugmentedDataset,
    codec_config_from_adapter_path,
    maybe_wrap_codec_evidence_dataset,
)
from .episode_bundle import (
    BUNDLE_SCHEMA_VERSION,
    FRAME_VALIDITY_NAMES,
    K_MAX,
    MAX_EPISODE_BATCH_SIZE,
    MAX_EPISODE_TRANSPORT_BYTES,
    METADATA_FEATURE_NAMES,
    METADATA_WIDTH,
    SOURCE_KINDS,
    EpisodeBundle,
    EpisodeBundleKey,
    EpisodeBundleStore,
    EpisodeT8View,
    InMemoryEpisodeBundleStore,
    RawMotionVectorAudit,
    assert_episode_batch_conforms,
    collate_episode_views_numpy,
    make_synthetic_episode_bundle,
    validate_episode_view,
    write_episode_bundle,
)
from .processor import CodecEvidenceProcessorCore
from .store import CodecEvidenceStore, FixtureManifestCodecEvidenceStore, InMemoryCodecEvidenceStore


__all__ = [
    "FRAME_TYPE_CODES",
    "FRAME_VALIDITY_NAMES",
    "HISTORY",
    "K_MAX",
    "MAX_CODEC_BATCH_SIZE",
    "MAX_CODEC_TRANSPORT_BYTES",
    "MAX_MV_PER_FRAME",
    "MAX_PADDED_MV_ELEMENTS",
    "MAX_EPISODE_BATCH_SIZE",
    "MAX_EPISODE_TRANSPORT_BYTES",
    "METADATA_FEATURE_NAMES",
    "METADATA_WIDTH",
    "MV_FIELD_DTYPES",
    "REASON_CODES",
    "CodecEvidenceConfig",
    "CodecEvidenceBatch",
    "CodecEvidenceKey",
    "CodecEvidenceAugmentedDataset",
    "CodecEvidenceProcessorCore",
    "CodecEvidenceRecord",
    "CodecEvidenceSample",
    "CodecEvidenceStore",
    "EpisodeBundle",
    "EpisodeBundleKey",
    "EpisodeBundleStore",
    "EpisodeT8View",
    "FixtureManifestCodecEvidenceStore",
    "InMemoryCodecEvidenceStore",
    "InMemoryEpisodeBundleStore",
    "RawMotionVectorAudit",
    "SOURCE_KINDS",
    "BUNDLE_SCHEMA_VERSION",
    "assert_batch_conforms",
    "assert_episode_batch_conforms",
    "collate_codec_evidence_numpy",
    "collate_episode_views_numpy",
    "preflight_codec_batch_allocation",
    "codec_config_from_adapter_path",
    "maybe_wrap_codec_evidence_dataset",
    "make_synthetic_episode_bundle",
    "parse_codec_evidence_config",
    "validate_episode_view",
    "write_episode_bundle",
]
