from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.dataset import episode_bundle_store_from_index_manifest
from gr00t.data.codec_evidence.episode_bundle import make_synthetic_episode_bundle
from gr00t.data.codec_evidence.schemas import schema_path
from gr00t.data.codec_evidence.warehouse import (
    FINAL_MANIFEST_FILENAME,
    FULL_COVERAGE_KIND,
    OBJECT_COMMIT_FILENAME,
    AttemptJournal,
    WarehouseEpisodeBundleStore,
    commit_full_fixture_manifest,
    ensure_warehouse_layout,
    publish_episode_object,
    resolve_derivation,
    reusable_derivation,
    validate_full_fixture_manifest,
    verify_object,
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
    root.chmod(0o755)


@pytest.fixture
def warehouse(tmp_path: Path):
    root = tmp_path / "warehouse"
    yield root
    _restore_permissions(root)


def _raw_motion_vectors(num_frames: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    counts = np.asarray([0 if row % 30 == 0 else 1 for row in range(num_frames)])
    offsets = np.concatenate((np.zeros((1,), dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    fields = {
        name: np.arange(int(offsets[-1]), dtype=np.int64).astype(dtype)
        for name, dtype in MV_FIELD_DTYPES.items()
    }
    fields["block_width"][:] = 16
    fields["block_height"][:] = 16
    fields["motion_scale"][:] = 4
    return offsets, fields


def _publish(
    warehouse: Path,
    tmp_path: Path,
    *,
    seed: str = "a",
    derivation_variant: str = "same",
    bitstream: bytes = b"synthetic-mp4",
    prepublish_guard=None,
):
    bundle = make_synthetic_episode_bundle(
        num_frames=40,
        episode_uid=seed * 64,
    )
    production_contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind=bundle.key.source_kind,
    )
    bundle = replace(
        bundle,
        codec_contract=production_contract,
        codec_contract_sha256=codec_contract_sha256(production_contract),
    )
    offsets, fields = _raw_motion_vectors(bundle.num_frames)
    bitstream_path = tmp_path / f"{seed}-{sha256(bitstream).hexdigest()}.mp4"
    bitstream_path.write_bytes(bitstream)
    derivation = {
        "schema_version": "codec-evidence-test-derivation-v1",
        "episode_uid": bundle.key.episode_uid,
        "canonical_video_sha256": "1" * 64,
        "canonical_parquet_sha256": "2" * 64,
        "codec_contract_sha256": bundle.codec_contract_sha256,
        "variant": derivation_variant,
    }
    published = publish_episode_object(
        warehouse,
        derivation_payload=derivation,
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
        raw_mv_offsets=offsets,
        raw_mv_fields=fields,
        accepted_bitstream=bitstream_path,
        prepublish_guard=prepublish_guard,
    )
    return bundle, derivation, published


def test_prepublish_budget_guard_runs_before_binding_or_object_mutation(
    warehouse: Path, tmp_path: Path
):
    observed = []

    def reject(stage_bytes: int) -> None:
        observed.append(stage_bytes)
        raise RuntimeError("focused budget exceeded")

    with pytest.raises(RuntimeError, match="focused budget exceeded"):
        _publish(warehouse, tmp_path, prepublish_guard=reject)
    assert observed and observed[0] > 0
    assert not list((warehouse / "objects").iterdir())
    assert not list((warehouse / "derivations").iterdir())
    assert not list((warehouse / ".staging").iterdir())


def test_publish_reuses_only_a_fully_verified_current_episode_bundle(
    warehouse: Path, tmp_path: Path
):
    bundle, derivation, first = _publish(warehouse, tmp_path)
    _other_bundle, _other_derivation, second = _publish(warehouse, tmp_path)

    assert first.object_content_sha256 == second.object_content_sha256
    assert first.derivation_sha256 == second.derivation_sha256
    assert first.reused is False
    assert second.reused is True
    assert (first.object_path / OBJECT_COMMIT_FILENAME).is_file()
    verified = verify_object(warehouse, first.object_content_sha256)
    assert verified["bundle_manifest"]["sha256"] == first.bundle_manifest_sha256

    resolved = resolve_derivation(
        warehouse,
        derivation,
        expected_key=bundle.key,
    )
    assert resolved.reused is True
    assert resolved.object_content_sha256 == first.object_content_sha256


def test_same_derivation_cannot_publish_different_content(warehouse: Path, tmp_path: Path):
    _bundle, _derivation, first = _publish(
        warehouse,
        tmp_path,
        bitstream=b"first-bitstream",
    )
    with pytest.raises(ValueError, match="different content digest"):
        _publish(
            warehouse,
            tmp_path,
            bitstream=b"second-bitstream",
        )

    object_paths = [path for path in (warehouse / "objects").iterdir() if path.is_dir()]
    assert object_paths == [first.object_path]


def test_exact_uncommitted_object_gets_marker_but_corrupt_partial_is_never_overwritten(
    warehouse: Path, tmp_path: Path
):
    _bundle, _derivation, first = _publish(warehouse, tmp_path)
    first.object_path.chmod(0o755)
    marker = first.object_path / OBJECT_COMMIT_FILENAME
    marker.unlink()

    _bundle, _derivation, recovered = _publish(warehouse, tmp_path)
    assert recovered.reused is True
    assert marker.is_file()
    verify_object(warehouse, recovered.object_content_sha256)

    first.object_path.chmod(0o755)
    marker.unlink()
    victim = next((first.object_path / "payload" / "bundle").rglob("frame_index.npy"))
    victim.parent.chmod(0o755)
    victim.chmod(0o644)
    original = victim.read_bytes()
    victim.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _publish(warehouse, tmp_path)
    assert victim.read_bytes() != original
    assert not marker.exists()


def test_empty_reserved_object_path_is_completed_without_replacing_its_inode(
    warehouse: Path, tmp_path: Path
):
    source = tmp_path / "source-warehouse"
    _bundle, _derivation, source_object = _publish(source, tmp_path)
    _restore_permissions(source)

    root = ensure_warehouse_layout(warehouse)
    reservation = root / "objects" / source_object.object_content_sha256
    reservation.mkdir()
    inode = reservation.stat().st_ino
    _bundle, _derivation, recovered = _publish(warehouse, tmp_path)

    assert recovered.reused is True
    assert recovered.object_path.stat().st_ino == inode
    assert (recovered.object_path / OBJECT_COMMIT_FILENAME).is_file()
    verify_object(warehouse, recovered.object_content_sha256)


def test_committed_object_corruption_blocks_resolve_and_reuse(warehouse: Path, tmp_path: Path):
    _bundle, derivation, first = _publish(warehouse, tmp_path)
    manifest = first.object_path / "payload" / "object_manifest.json"
    manifest.chmod(0o644)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["num_frames"] += 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o444)

    with pytest.raises(ValueError, match="content digest mismatch"):
        resolve_derivation(warehouse, derivation, expected_key=first.key)
    with pytest.raises(ValueError, match="content digest mismatch"):
        _publish(warehouse, tmp_path)


def test_manual_resume_requires_explicit_prior_attempt_and_exact_plan(warehouse: Path):
    plan = {
        "schema_version": "c3-test-plan-v1",
        "episode_count": 806,
        "input_snapshot_sha256": "a" * 64,
        "budget_sha256": "b" * 64,
    }
    first = AttemptJournal.start(warehouse, attempt_id="attempt-1", plan=plan)
    first.finish("FAILED", {"reason": "synthetic interruption"})
    resumed = AttemptJournal.manual_resume(
        warehouse,
        attempt_id="attempt-2",
        resume_from_attempt_id="attempt-1",
        plan=dict(plan),
    )
    started = json.loads((resumed.path / "ATTEMPT_STARTED.json").read_text(encoding="utf-8"))
    assert started["mode"] == "MANUAL_RESUME"
    assert started["resume_from_attempt_id"] == "attempt-1"
    assert started["plan_sha256"] == first.plan_sha256

    drifted = dict(plan)
    drifted["episode_count"] = 805
    with pytest.raises(ValueError, match="exact prior plan"):
        AttemptJournal.manual_resume(
            warehouse,
            attempt_id="attempt-drift",
            resume_from_attempt_id="attempt-1",
            plan=drifted,
        )
    assert not (warehouse / "attempts" / "attempt-drift").exists()

    with pytest.raises(ValueError, match="JSON artifact is missing"):
        AttemptJournal.manual_resume(
            warehouse,
            attempt_id="attempt-no-latest-fallback",
            resume_from_attempt_id="explicitly-missing",
            plan=plan,
        )


def test_attempt_has_one_exclusive_terminal_record(warehouse: Path):
    plan = {"schema_version": "c3-test-plan-v1", "episode_count": 806}
    attempt = AttemptJournal.start(warehouse, attempt_id="attempt-1", plan=plan)
    terminal = attempt.finish("FAILED", {"reason": "synthetic interruption"})

    with pytest.raises(FileExistsError):
        attempt.finish("READY", {"reason": "conflicting terminal"})

    payload = json.loads(terminal.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["result"] == {"reason": "synthetic interruption"}


def test_manual_resume_rejects_ready_attempt(warehouse: Path):
    plan = {"schema_version": "c3-test-plan-v1", "episode_count": 806}
    first = AttemptJournal.start(warehouse, attempt_id="attempt-1", plan=plan)
    first.finish("READY", {"reason": "complete"})

    with pytest.raises(ValueError, match="FAILED or STOPPED"):
        AttemptJournal.manual_resume(
            warehouse,
            attempt_id="attempt-2",
            resume_from_attempt_id="attempt-1",
            plan=plan,
        )
    assert not (warehouse / "attempts" / "attempt-2").exists()


def test_manual_resume_lookup_treats_reserved_missing_object_as_pending(
    warehouse: Path, tmp_path: Path
):
    bundle, derivation, published = _publish(warehouse, tmp_path)
    _restore_permissions(published.object_path)
    shutil.rmtree(published.object_path)

    assert reusable_derivation(warehouse, derivation, expected_key=bundle.key) is None
    _bundle, _derivation, recovered = _publish(warehouse, tmp_path)
    assert recovered.reused is False
    reusable = reusable_derivation(warehouse, derivation, expected_key=bundle.key)
    assert reusable is not None
    assert reusable.object_content_sha256 == recovered.object_content_sha256


def test_final_manifest_is_full_coverage_atomic_no_replace_and_store_roundtrips(
    warehouse: Path, tmp_path: Path
):
    bundle, _derivation, published = _publish(warehouse, tmp_path)
    manifest = commit_full_fixture_manifest(
        warehouse,
        fixture_id="full-test",
        expected_episode_count=1,
        expected_episode_uids=[bundle.key.episode_uid],
        objects=[published],
        input_snapshot_sha256="c" * 64,
        budget_sha256="d" * 64,
        codec_contract=bundle.metadata_dict()["codec_contract"],
        producer_identity={"test": True},
    )
    assert manifest == warehouse / FINAL_MANIFEST_FILENAME
    original = manifest.read_bytes()
    manifest_sha256 = sha256(original).hexdigest()
    validated = validate_full_fixture_manifest(
        warehouse,
        fixture_manifest_sha256=manifest_sha256,
    )
    assert validated["complete"] is True
    assert validated["coverage_kind"] == "FULL_EPISODE_SET"
    assert validated["coverage_kind"] == FULL_COVERAGE_KIND
    schema = json.loads(
        schema_path("codec-evidence-full-fixture-manifest-v1.schema.json").read_text()
    )
    assert set(schema["required"]) == set(validated)
    assert validated["expected_episode_count"] == 1
    assert validated["covered_episode_count"] == 1

    store = WarehouseEpisodeBundleStore(
        warehouse,
        fixture_manifest_sha256=manifest_sha256,
    )
    assert store.keys == frozenset({bundle.key})
    assert store.get(bundle.key).num_frames == bundle.num_frames
    assert store.identity(bundle.key)["codec_contract_sha256"] == bundle.codec_contract_sha256
    assert int(store.audit_raw_motion_vectors(bundle.key).offsets[-1]) == bundle.num_frames - 2
    assert store.audit_accepted_bitstream(bundle.key).read_bytes() == b"synthetic-mp4"
    dispatched = episode_bundle_store_from_index_manifest(
        warehouse,
        fixture_manifest_sha256=manifest_sha256,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
    )
    assert isinstance(dispatched, WarehouseEpisodeBundleStore)
    assert dispatched.get(bundle.key).num_frames == bundle.num_frames

    with pytest.raises(FileExistsError):
        commit_full_fixture_manifest(
            warehouse,
            fixture_id="full-test",
            expected_episode_count=1,
            expected_episode_uids=[bundle.key.episode_uid],
            objects=[published],
            input_snapshot_sha256="c" * 64,
            budget_sha256="d" * 64,
            codec_contract=bundle.metadata_dict()["codec_contract"],
            producer_identity={"test": True},
        )
    assert manifest.read_bytes() == original


def test_final_manifest_rejects_incomplete_coverage_before_commit(warehouse: Path, tmp_path: Path):
    bundle, _derivation, published = _publish(warehouse, tmp_path)
    with pytest.raises(ValueError, match="expected/covered episode counts differ"):
        commit_full_fixture_manifest(
            warehouse,
            fixture_id="incomplete",
            expected_episode_count=2,
            expected_episode_uids=[bundle.key.episode_uid, "f" * 64],
            objects=[published],
            input_snapshot_sha256="c" * 64,
            budget_sha256="d" * 64,
            codec_contract=bundle.metadata_dict()["codec_contract"],
            producer_identity={"test": True},
        )
    assert not (warehouse / FINAL_MANIFEST_FILENAME).exists()


def test_final_manifest_collision_never_unlinks_or_replaces_existing_file(
    warehouse: Path, tmp_path: Path
):
    source_bundle, _derivation, published = _publish(warehouse, tmp_path, seed="e")
    collision = warehouse / FINAL_MANIFEST_FILENAME
    collision.write_bytes(b"preexisting-owner-data\n")
    before = collision.stat()

    with pytest.raises(FileExistsError):
        commit_full_fixture_manifest(
            warehouse,
            fixture_id="collision",
            expected_episode_count=1,
            expected_episode_uids=[source_bundle.key.episode_uid],
            objects=[published],
            input_snapshot_sha256="c" * 64,
            budget_sha256="d" * 64,
            codec_contract=source_bundle.metadata_dict()["codec_contract"],
            producer_identity={"test": True},
        )
    after = collision.stat()
    assert collision.read_bytes() == b"preexisting-owner-data\n"
    assert after.st_ino == before.st_ino
