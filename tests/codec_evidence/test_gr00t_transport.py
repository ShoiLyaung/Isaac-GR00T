from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from gr00t.data.codec_evidence.config import CodecEvidenceConfig
from gr00t.data.codec_evidence.dataset import (
    CodecEvidenceAugmentedDataset,
    episode_bundle_store_from_pilot_manifest,
)
from gr00t.data.codec_evidence.episode_bundle import (
    SOURCE_KIND_NATIVE_CANONICAL,
    SOURCE_KIND_RAW_JPEG,
    EpisodeBundleStore,
    InMemoryEpisodeBundleStore,
    make_synthetic_episode_bundle,
)
from gr00t.data.codec_evidence.processor import CodecEvidenceProcessorCore
from gr00t.data.codec_evidence.store import InMemoryCodecEvidenceStore
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator, Gr00tN1d7Processor
import numpy as np
import pytest


def test_dataset_wrapper_loads_evidence_before_native_processing(synthetic_codec_record):
    trace = []

    class TracedStore(InMemoryCodecEvidenceStore):
        def get(self, key):
            trace.append("store")
            return super().get(key)

    class CohortLoader:
        selected_episode_uids = [synthetic_codec_record.key.episode_uid]

    class EpisodeLoader:
        mapping = [(0, 0)]
        loaders = [CohortLoader()]

        def __getitem__(self, index):
            return object()

    class BaseDataset:
        sharded_episodes = [[(0, np.array([7]))]]
        episode_loader = EpisodeLoader()
        embodiment_tag = SimpleNamespace(value="new_embodiment")

        def __len__(self):
            return 1

        def get_shard_length(self, index):
            return 1

        def set_processor(self, processor):
            self.processor = processor

        def get_dataset_statistics(self):
            return {"state": {}, "action": {}}

        def get_datapoint(self, episode_data, step_index):
            trace.append("native")
            return {"state": np.zeros((1, 132), dtype=np.float32)}

    config = CodecEvidenceConfig(
        enabled=True,
        fixture_root=None,
        fixture_manifest_sha256="b" * 64,
        record_schema_sha256="c" * 64,
    )
    wrapped = CodecEvidenceAugmentedDataset(
        BaseDataset(),
        config,
        store=TracedStore([synthetic_codec_record]),
        processor=CodecEvidenceProcessorCore(),
    )
    result = wrapped.get_shard(0)
    assert trace == ["store", "native"]
    assert "codec_evidence_sample" in result[0]
    assert result[0]["codec_evidence_sample"].key.episode_uid == (
        synthetic_codec_record.key.episode_uid
    )


def test_real_collator_places_single_action_side_key(synthetic_codec_record):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    collator = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    collator.codec_evidence_enabled = True
    from gr00t.data.codec_evidence.torch_backend import TorchCodecEvidenceCollator

    collator.codec_evidence_collator = TorchCodecEvidenceCollator()
    collator.model_type = "qwen"
    collator.model_name = "synthetic"
    output = collator(
        [
            {
                "state": np.zeros((1, 132), dtype=np.float32),
                "codec_evidence_sample": sample,
            }
        ]
    )["inputs"]
    assert "codec_evidence" in output
    assert "codec_evidence_sample" not in output
    assert tuple(output["codec_evidence"]["rpe"].shape) == (1, 8, 2, 2, 4)


def test_collator_fails_closed_on_missing_or_disabled(synthetic_codec_record):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    collator = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    collator.codec_evidence_enabled = False
    collator.codec_evidence_collator = None
    with pytest.raises(ValueError, match="disabled"):
        collator([{"codec_evidence_sample": sample}])

    collator.codec_evidence_enabled = True
    from gr00t.data.codec_evidence.torch_backend import TorchCodecEvidenceCollator

    collator.codec_evidence_collator = TorchCodecEvidenceCollator()
    with pytest.raises(ValueError, match="missing"):
        collator(
            [
                {"state": np.zeros((1, 132)), "codec_evidence_sample": sample},
                {"state": np.zeros((1, 132))},
            ]
        )


def test_online_processor_codec_transport_is_symmetric_and_fail_closed(
    synthetic_codec_record,
):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    from gr00t.data.codec_evidence.torch_backend import TorchCodecEvidenceCollator

    processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)
    processor.codec_evidence_enabled = True
    processor._collator = SimpleNamespace(codec_evidence_collator=TorchCodecEvidenceCollator())

    batch = processor._collate_codec_evidence_observation({"codec_evidence_sample": sample}, 1)
    assert tuple(batch["rpe"].shape) == (1, 8, 2, 2, 4)
    with pytest.raises(KeyError):
        processor._collate_codec_evidence_observation({}, 1)
    with pytest.raises(ValueError):
        processor._collate_codec_evidence_observation({"codec_evidence_sample": [sample]}, 2)


def test_episode_bundle_view_reaches_the_single_action_side_transport():
    bundle = make_synthetic_episode_bundle(num_frames=60)
    view = CodecEvidenceProcessorCore().process(bundle.view(30))
    collator = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    collator.codec_evidence_enabled = True
    from gr00t.data.codec_evidence.torch_backend import TorchCodecEvidenceCollator

    collator.codec_evidence_collator = TorchCodecEvidenceCollator()
    collator.model_type = "qwen"
    collator.model_name = "synthetic"
    output = collator(
        [
            {
                "state": np.zeros((1, 132), dtype=np.float32),
                "codec_evidence_sample": view,
            }
        ]
    )["inputs"]
    evidence = output["codec_evidence"]
    assert tuple(evidence["mv_cell_features"].shape) == (1, 8, 2, 2, 11)
    assert tuple(evidence["metadata_features"].shape) == (1, 8, 21)
    assert tuple(evidence["anchor_appearance"].shape) == (1, 2, 3, 256, 256)
    assert "motion_vector_fields" not in evidence


def test_episode_bundle_dataset_wrapper_loads_once_and_bounds_shards():
    bundle = make_synthetic_episode_bundle(num_frames=130)

    class CountingStore(InMemoryEpisodeBundleStore):
        def __init__(self, bundles):
            super().__init__(bundles)
            self.loads = 0

        def get(self, key):
            self.loads += 1
            return super().get(key)

    class EpisodeLoader:
        selected_episode_uids = [bundle.key.episode_uid]

        def __getitem__(self, index):
            return object()

    class BaseDataset:
        sharded_episodes = [[(0, np.asarray([14, 30], dtype=np.int64))]]
        episode_loader = EpisodeLoader()

        def get_datapoint(self, episode_data, step_index):
            return {"row": int(step_index)}

    config = CodecEvidenceConfig(
        enabled=True,
        schema_version="codec-evidence-episode-bundle-v1",
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
        camera_id="cam_left_high",
    )
    store = CountingStore([bundle])
    wrapped = CodecEvidenceAugmentedDataset(BaseDataset(), config, store=store)
    assert [wrapped.get_shard_length(index) for index in range(len(wrapped))] == [2]
    first = wrapped.get_shard(0)
    assert len(first) == 2
    assert store.loads == 1
    assert first[0]["codec_evidence_sample"].observation_row == 14
    assert first[-1]["codec_evidence_sample"].observation_row == 30

    class LargeShardBase(BaseDataset):
        sharded_episodes = [[(0, np.arange(130, dtype=np.int64))]]

    bounded = CodecEvidenceAugmentedDataset(LargeShardBase(), config, store=store)
    assert [bounded.get_shard_length(index) for index in range(len(bounded))] == [64, 64, 2]


def _snapshot_dispatch_fixture(tmp_path: Path):
    uid = "a" * 64
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"assignments": [{"episode_uid": uid, "eligible": True, "split": "train"}]}),
        encoding="utf-8",
    )
    identity = SimpleNamespace(
        dataset_root=tmp_path / "dataset",
        catalog_path=tmp_path / "dataset/catalog/current.sqlite3",
        catalog_sha256="b" * 64,
        split_manifest_path=split_path,
        split_manifest_sha256="c" * 64,
        payload_manifest_path=tmp_path / "payload.json",
        payload_manifest_sha256="d" * 64,
        train_statistics_path=tmp_path / "stats.json",
        train_statistics_sha256="e" * 64,
        train_episode_uid_digest=sha256((uid + "\n").encode("utf-8")).hexdigest(),
    )

    class Guard:
        def __init__(self):
            self.identity = identity
            self.ensure_calls = 0
            self.verify_calls = 0

        def ensure_current_process(self):
            self.ensure_calls += 1

        def verify_frozen_identity(self):
            self.verify_calls += 1

    guard = Guard()
    cohort = SimpleNamespace(selected_episode_uids=[uid])
    episode_loader = SimpleNamespace(
        adapter_config={"split": "train"}, loaders=[cohort], mapping=[(0, 0)]
    )
    base = SimpleNamespace(
        dataset_generation_guard=guard,
        episode_loader=episode_loader,
        sharded_episodes=[[(0, np.asarray([0], dtype=np.int64))]],
    )
    snapshot_root = tmp_path / "warehouse/canaries/canary-r1"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "fixture_snapshot_manifest.json").write_text("{}\n", encoding="utf-8")
    config = CodecEvidenceConfig(
        enabled=True,
        fixture_root=snapshot_root,
        fixture_manifest_sha256="f" * 64,
        schema_version="codec-evidence-episode-bundle-v1",
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
        camera_id="cam_left_high",
    )
    return base, config, guard, uid


def test_training_wrapper_dispatches_snapshot_with_verifier_only_false(tmp_path, monkeypatch):
    base, config, guard, uid = _snapshot_dispatch_fixture(tmp_path)
    observed = {}

    def reject_canary(_root, **kwargs):
        observed.update(kwargs)
        raise ValueError("CANARY_SUBSET_NONTRAINING is verifier-only")

    monkeypatch.setattr("gr00t.data.codec_evidence.warehouse.open_fixture_snapshot", reject_canary)
    with pytest.raises(ValueError, match="verifier-only"):
        CodecEvidenceAugmentedDataset(base, config)
    assert observed["verifier_only"] is False
    assert observed["expected_train_episode_uids"] == [uid]
    assert observed["expected_dataset_binding"]["catalog_database_sha256"] == "b" * 64
    assert guard.ensure_calls == 1
    assert guard.verify_calls >= 1


def test_explicit_verifier_wrapper_allows_canary_snapshot_dispatch(tmp_path, monkeypatch):
    base, config, guard, uid = _snapshot_dispatch_fixture(tmp_path)
    sentinel_store = SimpleNamespace(
        manifest={
            "coverage_kind": "CANARY_SUBSET_NONTRAINING",
            "covered_episode_uids": [uid],
        }
    )
    observed = {}

    def accept_canary(_root, **kwargs):
        observed.update(kwargs)
        return sentinel_store

    monkeypatch.setattr("gr00t.data.codec_evidence.warehouse.open_fixture_snapshot", accept_canary)
    wrapped = CodecEvidenceAugmentedDataset(base, config, verifier_only=True)
    assert wrapped.store is sentinel_store
    assert observed["verifier_only"] is True
    assert observed["expected_train_episode_uids"] == [uid]
    assert guard.ensure_calls == 1
    assert guard.verify_calls >= 2


def test_verifier_wrapper_filters_real_base_shards_to_exact_canary_uids(tmp_path, monkeypatch):
    base, config, _guard, uid = _snapshot_dispatch_fixture(tmp_path)
    other_uid = "0" * 64
    base.episode_loader.loaders[0].selected_episode_uids.append(other_uid)
    second = SimpleNamespace(selected_episode_uids=[])
    base.episode_loader.loaders.append(second)
    base.episode_loader.mapping = [(0, 0), (0, 1)]
    base.sharded_episodes = [
        [
            (0, np.asarray([0], dtype=np.int64)),
            (1, np.asarray([0], dtype=np.int64)),
        ]
    ]

    # Keep the live train contract exact while the canary covers one member.
    split = json.loads(base.dataset_generation_guard.identity.split_manifest_path.read_text())
    split["assignments"].append({"episode_uid": other_uid, "eligible": True, "split": "train"})
    base.dataset_generation_guard.identity.split_manifest_path.write_text(
        json.dumps(split), encoding="utf-8"
    )
    base.dataset_generation_guard.identity.train_episode_uid_digest = sha256(
        (
            (other_uid + "\n" + uid + "\n") if other_uid < uid else (uid + "\n" + other_uid + "\n")
        ).encode()
    ).hexdigest()
    store = SimpleNamespace(
        manifest={
            "coverage_kind": "CANARY_SUBSET_NONTRAINING",
            "covered_episode_uids": [uid],
        }
    )
    monkeypatch.setattr(
        "gr00t.data.codec_evidence.warehouse.open_fixture_snapshot",
        lambda *_args, **_kwargs: store,
    )
    wrapped = CodecEvidenceAugmentedDataset(base, config, verifier_only=True)
    assert len(wrapped.sharded_episodes) == 1
    assert [episode_index for episode_index, _ in wrapped.sharded_episodes[0]] == [0]


def test_pilot_manifest_is_pinned_and_closes_both_sources(tmp_path):
    uids = [character * 64 for character in "abc"]
    bundles = []
    for uid in uids:
        for source_kind in (
            SOURCE_KIND_RAW_JPEG,
            SOURCE_KIND_NATIVE_CANONICAL,
        ):
            relative = f"{uid}/cam_left_high/{source_kind}/bundle.json"
            path = tmp_path / relative
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
            bundles.append(
                {
                    "episode_uid": uid,
                    "camera_id": "cam_left_high",
                    "source_kind": source_kind,
                    "relative_path": relative,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "frames": 60,
                }
            )
    manifest = {
        "schema_version": "codec-evidence-r5-pilot-v1",
        "complete": True,
        "pilot_generation_attempt": 1,
        "source_kinds": [SOURCE_KIND_RAW_JPEG, SOURCE_KIND_NATIVE_CANONICAL],
        "camera_id": "cam_left_high",
        "rows": [0, 59],
        "bundles": bundles,
        "bundle_count": 6,
        "episode_uids": uids,
        "determinism_encode_repetitions_per_bundle": 2,
        "total_internal_encode_executions": 12,
        "output_policy": "CREATE_NEW_ROOT_AND_WRITER_NO_CLOBBER",
        "dataset_mutations": 0,
        "gpu_used": False,
    }
    manifest_path = tmp_path / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pinned_sha = sha256(manifest_path.read_bytes()).hexdigest()
    store = episode_bundle_store_from_pilot_manifest(tmp_path, pilot_manifest_sha256=pinned_sha)
    assert isinstance(store, EpisodeBundleStore)
    manifest["gpu_used"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        episode_bundle_store_from_pilot_manifest(tmp_path, pilot_manifest_sha256=pinned_sha)
