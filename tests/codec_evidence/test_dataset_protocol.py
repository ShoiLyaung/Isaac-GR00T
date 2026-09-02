from __future__ import annotations

import json
from pathlib import Path

from gr00t.data.codec_evidence.config import parse_codec_evidence_config
from gr00t.data.codec_evidence.dataset import (
    CodecEvidenceAugmentedDataset,
    maybe_wrap_codec_evidence_dataset,
)
from gr00t.data.codec_evidence.episode_bundle import (
    SOURCE_KIND_NATIVE_CANONICAL,
    InMemoryEpisodeBundleStore,
    make_synthetic_episode_bundle,
)
from gr00t.data.codec_evidence.schemas import schema_path
from gr00t.data.codec_evidence.torch_backend import (
    EPISODE_AUDIT_IDENTITY_KEYS,
    EPISODE_MODEL_TENSOR_KEYS,
    TorchCodecEvidenceCollator,
)
import numpy as np
import pytest


def _config(tmp_path: Path):
    return parse_codec_evidence_config(
        {
            "enabled": True,
            "fixture_root": str(tmp_path.resolve()),
            "fixture_manifest_sha256": "a" * 64,
            "modalities": ["mv", "rpe", "i_frame", "metadata"],
            "gate_init": 0.0,
            "schema_version": "codec-evidence-episode-bundle-v1",
            "camera_id": "cam_left_high",
            "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
            "d_codec": 64,
            "conditioner_heads": 8,
        }
    )


class _BaseDataset:
    def __init__(self, episode_uid: str):
        self.selected_episode_uids = (episode_uid,)
        self.sharded_episodes = [[(0, np.asarray([0, 7], dtype=np.int64))]]
        self.episode_loader = [object()]

    def get_datapoint(self, episode_data, row):
        del episode_data
        return {"row": row}


def test_missing_and_disabled_configs_preserve_native_object(tmp_path, monkeypatch):
    dataset = object()
    assert maybe_wrap_codec_evidence_dataset(dataset, None) is dataset
    adapter = tmp_path / "adapter.json"
    adapter.write_text(
        json.dumps(
            {
                "codec_evidence": {
                    "enabled": False,
                    "fixture_root": "/must/not/be/read",
                }
            }
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled Codec config attempted a fixture read")

    monkeypatch.setattr(
        "gr00t.data.codec_evidence.dataset.episode_bundle_store_from_index_manifest",
        forbidden,
    )
    assert maybe_wrap_codec_evidence_dataset(dataset, adapter) is dataset


def test_record_v2_is_not_an_active_configuration(tmp_path):
    with pytest.raises(ValueError, match="Unsupported codec-evidence schema"):
        parse_codec_evidence_config(
            {
                "enabled": True,
                "fixture_root": str(tmp_path.resolve()),
                "fixture_manifest_sha256": "a" * 64,
                "schema_version": "codec-evidence-record-v2",
            }
        )


def test_episode_bundle_dataset_attaches_valid_views(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=40)
    store = InMemoryEpisodeBundleStore([bundle])
    dataset = CodecEvidenceAugmentedDataset(
        _BaseDataset(bundle.key.episode_uid),
        _config(tmp_path),
        store=store,
    )
    shard = dataset.get_shard(0)
    assert [row["row"] for row in shard] == [0, 7]
    assert all(row["codec_evidence_sample"].key == bundle.key for row in shard)


def test_torch_boundary_drops_audit_identity():
    bundle = make_synthetic_episode_bundle(num_frames=8)
    batch = TorchCodecEvidenceCollator()([bundle.view(7)])
    assert set(batch) == EPISODE_MODEL_TENSOR_KEYS
    assert not EPISODE_AUDIT_IDENTITY_KEYS.intersection(batch)


def test_active_schemas_are_local_documents():
    names = (
        "codec-evidence-episode-bundle-v1.schema.json",
        "codec-evidence-episode-bundle-index-v1.schema.json",
        "codec-evidence-episode-identity-v1.schema.json",
        "codec-evidence-full-fixture-manifest-v1.schema.json",
        "codec-evidence-global-contract-v2.schema.json",
    )
    for name in names:
        path = schema_path(name)
        assert path.is_file()
        assert isinstance(json.loads(path.read_text()), dict)
