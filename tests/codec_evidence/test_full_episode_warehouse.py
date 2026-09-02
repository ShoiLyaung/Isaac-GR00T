from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.episode_bundle import make_synthetic_episode_bundle
from gr00t.data.codec_evidence.implementation_manifest import (
    implementation_file_identities,
    verify_implementation_manifest,
    write_implementation_manifest_exclusive,
)
from gr00t.data.codec_evidence.warehouse import (
    CANARY_COVERAGE,
    FULL_SPLIT_COVERAGE,
    SOURCE_KIND,
    WarehouseExclusiveLease,
    derivation_key_sha256,
    episode_uid_digest,
    frozen_train_episode_uid_digest,
    open_fixture_snapshot,
    publish_object,
    validate_snapshot_manifest,
    verify_object,
    write_snapshot_manifest,
)
import numpy as np
import pytest


def _restore_permissions(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.exists():
            path.chmod(0o644)


@pytest.fixture
def warehouse(tmp_path: Path):
    root = tmp_path / "warehouse"
    yield root
    _restore_permissions(root)


def _publish(root: Path, tmp_path: Path, *, uid: str, bitstream: bytes = b"mp4"):
    bundle = make_synthetic_episode_bundle(num_frames=35, episode_uid=uid)
    derivation = {
        "schema_version": "codec-evidence-canonical-derivation-v1",
        "episode_uid": uid,
        "camera_id": "cam_left_high",
        "source_kind": SOURCE_KIND,
        "episode": {
            "cohort_id": "synthetic",
            "dataset_name": "synthetic",
            "source_episode_index": 0,
            "num_frames": bundle.num_frames,
        },
        "canonical_video": {"sha256": "1" * 64, "bytes": 3},
        "canonical_parquet": {"sha256": "2" * 64, "bytes": 4},
        "codec_recipe": {"codec": "h264", "gop": 30},
        "toolchain_identity": {"producer": {"sha256": "3" * 64}},
    }
    identity = {"name": "synthetic", "sha256": "4" * 64}
    manifest = {
        "schema_version": "codec-evidence-warehouse-object-v1",
        "derivation_payload_sha256": derivation_key_sha256(derivation),
        "episode_uid": uid,
        "camera_id": "cam_left_high",
        "source_kind": SOURCE_KIND,
        "num_frames": bundle.num_frames,
        "episode": {
            "cohort_id": "synthetic",
            "dataset_name": "synthetic",
            "source_episode_index": 0,
        },
        "width": 640,
        "height": 480,
        "time_base_num": 1,
        "time_base_den": 30_000,
        "gop_size": 30,
        "input_identity": {"canonical_video_sha256": "1" * 64},
        "frame_hashes": {
            "source": list(bundle.source_frame_sha256),
            "canonical": list(bundle.canonical_frame_sha256),
            "fixture": list(bundle.fixture_frame_sha256),
        },
        "identities": {
            "tool": identity,
            "encoder": identity,
            "decoder": identity,
            "extractor": identity,
        },
    }
    counts = np.asarray(
        [0 if row % 30 == 0 else 1 for row in range(bundle.num_frames)], dtype=np.int64
    )
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    fields = {
        name: np.zeros((int(offsets[-1]),), dtype=dtype) for name, dtype in MV_FIELD_DTYPES.items()
    }
    accepted = tmp_path / f"{uid[:8]}-{bitstream.hex()}.mp4"
    accepted.write_bytes(bitstream)
    return publish_object(
        root,
        derivation_payload=derivation,
        manifest_payload=manifest,
        arrays=bundle.arrays_dict(),
        raw_mv_offsets=offsets,
        raw_mv_fields=fields,
        accepted_bitstream=accepted,
    )


def _binding(train_uids: list[str]) -> dict[str, object]:
    return {
        "dataset_root": "/frozen/dataset",
        "catalog_database_path": "/frozen/dataset/catalog/current.sqlite3",
        "catalog_database_sha256": "5" * 64,
        "split_manifest_path": "/frozen/split.json",
        "split_manifest_sha256": "6" * 64,
        "payload_manifest_path": "/frozen/payload.json",
        "payload_manifest_sha256": "7" * 64,
        "train_statistics_path": "/frozen/stats.json",
        "train_statistics_sha256": "8" * 64,
        "train_episode_uid_digest": frozen_train_episode_uid_digest(train_uids),
        "resolved_train_episode_count": len(train_uids),
    }


def test_publish_reuses_exact_object_and_requires_derivation_binding(
    warehouse: Path, tmp_path: Path
):
    uid = "a" * 64
    first = _publish(warehouse, tmp_path, uid=uid)
    second = _publish(warehouse, tmp_path, uid=uid)
    assert first.object_content_sha256 == second.object_content_sha256
    assert not first.reused
    assert second.reused
    verify_object(warehouse, first.object_content_sha256)

    binding = warehouse / "derivations" / f"{first.derivation_key_sha256}.json"
    binding.unlink()
    with pytest.raises(ValueError, match="JSON artifact is missing"):
        verify_object(warehouse, first.object_content_sha256)
    recovered = _publish(warehouse, tmp_path, uid=uid)
    assert recovered.reused is True
    verify_object(warehouse, recovered.object_content_sha256)


def test_same_derivation_cannot_map_to_different_content(warehouse: Path, tmp_path: Path):
    uid = "b" * 64
    _publish(warehouse, tmp_path, uid=uid, bitstream=b"first")
    with pytest.raises(ValueError, match="different content digest"):
        _publish(warehouse, tmp_path, uid=uid, bitstream=b"second")
    assert len(list((warehouse / "objects").iterdir())) == 1


def test_derivation_rejects_nested_dataset_identity_leak():
    payload = {
        "schema_version": "codec-evidence-canonical-derivation-v1",
        "episode_uid": "c" * 64,
        "camera_id": "cam_left_high",
        "source_kind": SOURCE_KIND,
        "episode": {},
        "canonical_video": {},
        "canonical_parquet": {},
        "codec_recipe": {},
        "toolchain_identity": {"nested": {"split_manifest_sha256": "0" * 64}},
    }
    with pytest.raises(ValueError, match="split-neutral"):
        derivation_key_sha256(payload)


def test_canary_digest_is_distinct_and_open_is_train_subset_only(warehouse: Path, tmp_path: Path):
    train_uids = ["a" * 64, "b" * 64]
    assert episode_uid_digest(train_uids) != frozen_train_episode_uid_digest(train_uids)
    obj = _publish(warehouse, tmp_path, uid=train_uids[0])
    binding = _binding(train_uids)
    manifest = write_snapshot_manifest(
        warehouse,
        snapshot_id="valid-canary",
        coverage_kind=CANARY_COVERAGE,
        dataset_binding=binding,
        objects=[obj],
        producer_identity={"test": True},
    )
    manifest_sha = sha256(manifest.read_bytes()).hexdigest()
    store = open_fixture_snapshot(
        manifest.parent,
        snapshot_manifest_sha256=manifest_sha,
        expected_dataset_binding=binding,
        expected_train_episode_uids=train_uids,
        camera_id="cam_left_high",
        source_kind=SOURCE_KIND,
        verifier_only=True,
    )
    assert store.manifest["covered_episode_uids"] == [train_uids[0]]
    with pytest.raises(ValueError, match="verifier-only"):
        open_fixture_snapshot(
            manifest.parent,
            snapshot_manifest_sha256=manifest_sha,
            expected_dataset_binding=binding,
            expected_train_episode_uids=train_uids,
            camera_id="cam_left_high",
            source_kind=SOURCE_KIND,
        )

    outside = _publish(warehouse, tmp_path, uid="c" * 64)
    outside_manifest = write_snapshot_manifest(
        warehouse,
        snapshot_id="outside-train",
        coverage_kind=CANARY_COVERAGE,
        dataset_binding=binding,
        objects=[outside],
        producer_identity={"test": True},
    )
    outside_sha = sha256(outside_manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="outside the live train split"):
        open_fixture_snapshot(
            outside_manifest.parent,
            snapshot_manifest_sha256=outside_sha,
            expected_dataset_binding=binding,
            expected_train_episode_uids=train_uids,
            camera_id="cam_left_high",
            source_kind=SOURCE_KIND,
            verifier_only=True,
        )


def test_snapshot_publish_cross_checks_object_entry(warehouse: Path, tmp_path: Path):
    obj = _publish(warehouse, tmp_path, uid="d" * 64)
    entry = obj.snapshot_entry()
    entry["num_frames"] += 1
    with pytest.raises(ValueError, match="differs from its object payload"):
        write_snapshot_manifest(
            warehouse,
            snapshot_id="bad-entry",
            coverage_kind=CANARY_COVERAGE,
            dataset_binding=_binding(["d" * 64]),
            objects=[entry],
            producer_identity={"test": True},
        )


def test_full_split_requires_and_opens_exact_frozen_train_coverage(warehouse: Path, tmp_path: Path):
    train_uids = ["e" * 64, "f" * 64]
    objects = [_publish(warehouse, tmp_path, uid=uid) for uid in train_uids]
    binding = _binding(train_uids)
    manifest = write_snapshot_manifest(
        warehouse,
        snapshot_id="full-train",
        coverage_kind=FULL_SPLIT_COVERAGE,
        dataset_binding=binding,
        objects=objects,
        producer_identity={"test": True},
    )
    manifest_sha = sha256(manifest.read_bytes()).hexdigest()
    store = open_fixture_snapshot(
        manifest.parent,
        snapshot_manifest_sha256=manifest_sha,
        expected_dataset_binding=binding,
        expected_train_episode_uids=train_uids,
        camera_id="cam_left_high",
        source_kind=SOURCE_KIND,
    )
    assert store.manifest["training_eligible"] is True
    assert store.manifest["covered_episode_uids"] == train_uids


def test_object_and_snapshot_top_level_tampering_fail_exact_validation(
    warehouse: Path, tmp_path: Path
):
    uid = "9" * 64
    obj = _publish(warehouse, tmp_path, uid=uid)
    object_manifest = obj.object_path / "object_manifest.json"
    object_manifest.chmod(0o644)
    object_payload = json.loads(object_manifest.read_text(encoding="utf-8"))
    object_payload["manifest_payload"]["split"] = "train"
    object_manifest.write_text(json.dumps(object_payload), encoding="utf-8")
    object_manifest.chmod(0o444)
    with pytest.raises(ValueError, match="fields/schema mismatch"):
        verify_object(warehouse, obj.object_content_sha256)

    # Publish a separate intact object because the first is deliberately corrupt.
    intact = _publish(warehouse, tmp_path, uid="8" * 64)
    snapshot = write_snapshot_manifest(
        warehouse,
        snapshot_id="tamper-id",
        coverage_kind=CANARY_COVERAGE,
        dataset_binding=_binding(["8" * 64]),
        objects=[intact],
        producer_identity={"test": True},
    )
    snapshot.chmod(0o644)
    snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
    snapshot_payload["snapshot_id"] = "different-directory"
    snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    snapshot.chmod(0o444)
    with pytest.raises(ValueError, match="ID differs from its directory"):
        validate_snapshot_manifest(
            warehouse,
            snapshot,
            manifest_sha256=sha256(snapshot.read_bytes()).hexdigest(),
            verifier_only=True,
        )


def test_exclusive_warehouse_lease_fails_closed_on_conflict(warehouse: Path):
    with WarehouseExclusiveLease(warehouse):
        with pytest.raises(RuntimeError, match="owns the exclusive lease"):
            with WarehouseExclusiveLease(warehouse):
                pass


def test_implementation_manifest_rehashes_exact_file_closure(tmp_path: Path):
    first = tmp_path / "first.py"
    second = tmp_path / "tool"
    first.write_text("one\n", encoding="utf-8")
    second.write_bytes(b"two")
    files = implementation_file_identities({"source": first, "tool": second})
    manifest = write_implementation_manifest_exclusive(
        tmp_path / "implementation.json", run_id="RUN", files=files
    )
    verified = verify_implementation_manifest(manifest, expected_run_id="RUN", expected_files=files)
    assert verified["file_count"] == 2
    with pytest.raises(FileExistsError):
        write_implementation_manifest_exclusive(manifest, run_id="RUN", files=files)
    first.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted after freeze"):
        verify_implementation_manifest(manifest, expected_run_id="RUN", expected_files=files)
