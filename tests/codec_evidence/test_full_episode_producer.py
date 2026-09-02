from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
import shutil

from gr00t.data.codec_evidence import full_episode as producer
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.episode_bundle import (
    FRAME_TYPE_IDR,
    FRAME_TYPE_P,
    make_synthetic_episode_bundle,
)
from gr00t.data.codec_evidence.full_episode import (
    CanonicalEpisodeInput,
    DatasetFactoryCanonicalDecoder,
    assert_torchcodec_pyav_exact,
    build_canonical_object,
    build_episode_arrays,
    encode_h264_twice,
    extract_mvs_and_decoded_rgb,
    install_forbidden_upstream_jpeg_audit,
    probe_packets,
    publish_verified_canonical_object,
    trace_h264_headers,
    verify_canonical_input_identity,
)
from gr00t.data.codec_evidence.warehouse import (
    ARRAY_NAMES,
    CANARY_COVERAGE,
    SOURCE_KIND,
    derivation_key_sha256,
    frozen_train_episode_uid_digest,
    open_fixture_snapshot,
    publish_object,
    sha256_path,
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


def _moving_rgb(num_frames: int, *, height: int = 64, width: int = 96) -> np.ndarray:
    """Small deterministic raster sequence with enough motion for exported P-frame MVs."""

    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = (rows + columns) % 256
    base[..., 1] = (2 * rows + columns) % 256
    base[..., 2] = (rows + 2 * columns) % 256
    frames: list[np.ndarray] = []
    for row in range(num_frames):
        frame = np.roll(base, shift=(row % 7, row % 11), axis=(0, 1)).copy()
        left = (row * 2) % (width - 16)
        frame[16:32, left : left + 16] = np.asarray([255, 20, 30], dtype=np.uint8)
        frames.append(frame)
    return np.stack(frames)


@pytest.mark.parametrize("num_frames", [35, 44, 60, 65])
def test_real_cpu_pipeline_preserves_dynamic_n_partial_gop_and_exact_decode_parity(
    tmp_path: Path, num_frames: int
):
    scratch = tmp_path / f"n-{num_frames}"
    scratch.mkdir()
    encoded = encode_h264_twice(_moving_rgb(num_frames), scratch)
    assert encoded.attempt_sha256 == (encoded.bitstream_sha256, encoded.bitstream_sha256)

    packets = probe_packets(encoded.bitstream_path, num_frames=num_frames)
    headers = trace_h264_headers(encoded.bitstream_path, num_frames=num_frames)
    extracted = extract_mvs_and_decoded_rgb(encoded.bitstream_path, num_frames=num_frames)
    arrays = build_episode_arrays(extracted, packets, headers)

    rows = np.arange(num_frames, dtype=np.int64)
    gop_starts = rows[rows % producer.GOP == 0]
    expected_frame_type = np.where(rows % producer.GOP == 0, FRAME_TYPE_IDR, FRAME_TYPE_P).astype(
        np.uint8
    )
    expected_references = rows - 1
    expected_references[gop_starts] = -1
    expected_clock = rows * producer.PTS_STEP

    assert np.array_equal(packets.pts, expected_clock)
    assert np.array_equal(packets.dts, expected_clock)
    assert np.array_equal(packets.duration, np.full(num_frames, producer.PTS_STEP))
    assert np.array_equal(np.flatnonzero(packets.key_frame), gop_starts)
    assert np.array_equal(headers.frame_type, expected_frame_type)
    assert np.array_equal(headers.reference_index, expected_references)
    assert all(headers.proof_claims.values())

    counts = np.diff(extracted.mv_offsets)
    assert extracted.mv_offsets.dtype == np.int64
    assert extracted.mv_offsets.shape == (num_frames + 1,)
    assert np.array_equal(counts[gop_starts], np.zeros(len(gop_starts), dtype=np.int64))
    assert np.all(counts[expected_frame_type == FRAME_TYPE_P] > 0)
    for name, dtype in MV_FIELD_DTYPES.items():
        assert extracted.mv_fields[name].dtype == dtype
        assert extracted.mv_fields[name].shape == (int(extracted.mv_offsets[-1]),)

    assert set(arrays) == set(ARRAY_NAMES)
    assert np.array_equal(arrays["frame_index"], rows)
    assert np.array_equal(arrays["gop_id"], rows // producer.GOP)
    assert np.array_equal(arrays["gop_position"], rows % producer.GOP)
    assert np.array_equal(arrays["reference_index"], expected_references)
    assert np.array_equal(arrays["anchor_frame_index"], gop_starts)
    assert np.array_equal(arrays["anchor_index_per_frame"], (rows // producer.GOP) * producer.GOP)
    assert arrays["anchor_appearance"].shape == (len(gop_starts), 3, 256, 256)
    assert arrays["frame_validity"].shape == (num_frames, 7)
    assert arrays["mv_cell_features"].shape == (num_frames, 2, 2, 11)
    assert arrays["rpe"].shape == (num_frames, 2, 2, 4)
    assert int(arrays["gop_position"][-1]) == (num_frames - 1) % producer.GOP

    parity = assert_torchcodec_pyav_exact(encoded.bitstream_path, extracted.rgb)
    assert parity["all_rows_array_equal"] is True
    assert parity["max_abs_diff"] == 0
    assert parity["num_frames"] == num_frames
    assert parity["shape"] == [num_frames, 64, 96, 3]
    assert parity["per_row_hash_equality_count"] == num_frames
    assert (
        parity["per_row_dataset_factory_torchcodec_rgb_sha256"] == parity["per_row_pyav_rgb_sha256"]
    )


def test_input_drift_fails_before_publish_and_creates_no_warehouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    video = tmp_path / "canonical.mp4"
    parquet = tmp_path / "episode.parquet"
    video.write_bytes(b"canonical-video-before")
    parquet.write_bytes(b"canonical-parquet-before")
    episode = CanonicalEpisodeInput(
        episode_uid="a" * 64,
        cohort_id="synthetic",
        dataset_name="synthetic",
        source_episode_index=0,
        num_frames=1,
        parquet_path=parquet,
        parquet_sha256=sha256_path(parquet),
        parquet_bytes=parquet.stat().st_size,
        canonical_video_path=video,
        canonical_video_sha256=sha256_path(video),
        canonical_video_bytes=video.stat().st_size,
    )
    video.write_bytes(b"canonical-video-after-drift")
    publish_called = False

    def forbidden_publish(*_args, **_kwargs):
        nonlocal publish_called
        publish_called = True
        raise AssertionError("publish_object must not run after input drift")

    monkeypatch.setattr(producer, "publish_object", forbidden_publish)
    root = tmp_path / "warehouse"
    with pytest.raises(ValueError, match="BEFORE_PUBLISH: canonical_video identity drifted"):
        publish_verified_canonical_object(
            episode,
            root,
            derivation_payload={},
            manifest_payload={},
            arrays={},
            raw_mv_offsets=np.empty(0, dtype=np.int64),
            raw_mv_fields={},
            accepted_bitstream=video,
        )
    assert publish_called is False
    assert not root.exists()


def _canonical_derivation(uid: str) -> dict[str, object]:
    return {
        "schema_version": "codec-evidence-canonical-derivation-v1",
        "episode_uid": uid,
        "camera_id": "cam_left_high",
        "source_kind": SOURCE_KIND,
        "episode": {
            "cohort_id": "synthetic",
            "dataset_name": "synthetic",
            "source_episode_index": 0,
            "num_frames": 35,
        },
        "canonical_video": {"sha256": "1" * 64, "bytes": 3},
        "canonical_parquet": {"sha256": "2" * 64, "bytes": 4},
        "codec_recipe": {"codec": "h264", "gop": 30},
        "toolchain_identity": {"producer": {"sha256": "3" * 64}},
    }


def test_canonical_producer_api_has_no_upstream_raw_source_parameter():
    assert [field.name for field in fields(CanonicalEpisodeInput)] == [
        "episode_uid",
        "cohort_id",
        "dataset_name",
        "source_episode_index",
        "num_frames",
        "parquet_path",
        "parquet_sha256",
        "parquet_bytes",
        "canonical_video_path",
        "canonical_video_sha256",
        "canonical_video_bytes",
    ]
    forbidden_parameters = {
        "raw_root",
        "raw_source",
        "raw_jpeg",
        "raw_jpeg_root",
        "upstream_raw",
        "upstream_jpeg",
        "source_jpeg",
    }
    for callable_object in (
        build_canonical_object,
        DatasetFactoryCanonicalDecoder.decode_episode,
        publish_verified_canonical_object,
    ):
        assert forbidden_parameters.isdisjoint(signature(callable_object).parameters)


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "raw_root",
        "raw_source_path",
        "raw_jpeg_sha256",
        "source_jpeg_sha256",
        "upstream_raw_sha256",
        "upstream_jpeg_sha256",
        "dual_source_identity",
    ],
)
def test_canonical_derivation_rejects_any_upstream_raw_source_identity(forbidden_key: str):
    payload = _canonical_derivation("9" * 64)
    payload["toolchain_identity"] = {
        "producer": {"sha256": "3" * 64},
        "forbidden_nested_identity": {forbidden_key: "4" * 64},
    }
    with pytest.raises(ValueError, match="split-neutral"):
        derivation_key_sha256(payload)


def _publish_synthetic(root: Path, tmp_path: Path, *, uid: str):
    bundle = make_synthetic_episode_bundle(num_frames=35, episode_uid=uid)
    derivation = _canonical_derivation(uid)
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
    accepted = tmp_path / f"{uid[:8]}.mp4"
    accepted.write_bytes(b"mp4")
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


def test_object_artifact_tamper_is_rejected(warehouse: Path, tmp_path: Path):
    published = _publish_synthetic(warehouse, tmp_path, uid="b" * 64)
    accepted = published.object_path / "accepted.mp4"
    accepted.chmod(0o644)
    accepted.write_bytes(b"tampered-object-artifact")
    with pytest.raises(ValueError, match=r"Artifact byte-count mismatch: accepted\.mp4"):
        verify_object(warehouse, published.object_content_sha256)


def test_derivation_binding_tamper_is_rejected(warehouse: Path, tmp_path: Path):
    published = _publish_synthetic(warehouse, tmp_path, uid="c" * 64)
    binding = warehouse / "derivations" / f"{published.derivation_key_sha256}.json"
    binding.chmod(0o644)
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["object_content_sha256"] = "f" * 64
    binding.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Derivation binding is missing, corrupt"):
        verify_object(warehouse, published.object_content_sha256)


def test_snapshot_manifest_tamper_is_rejected_even_with_rehashed_file(
    warehouse: Path, tmp_path: Path
):
    uid = "d" * 64
    published = _publish_synthetic(warehouse, tmp_path, uid=uid)
    binding = _binding([uid])
    manifest = write_snapshot_manifest(
        warehouse,
        snapshot_id="tampered-canary",
        coverage_kind=CANARY_COVERAGE,
        dataset_binding=binding,
        objects=[published],
        producer_identity={"test": True},
    )
    manifest.parent.chmod(0o755)
    manifest.chmod(0o644)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["covered_episode_uid_digest"] = sha256(b"tampered").hexdigest()
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot covered UID closure mismatch"):
        open_fixture_snapshot(
            manifest.parent,
            snapshot_manifest_sha256=sha256_path(manifest),
            expected_dataset_binding=binding,
            expected_train_episode_uids=[uid],
            camera_id="cam_left_high",
            source_kind=SOURCE_KIND,
            verifier_only=True,
        )


def test_raw_root_absence_and_byte_changes_cannot_change_derivation_or_content_identity(
    warehouse: Path, tmp_path: Path
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    raw_root = dataset_root / "raw"
    assert not raw_root.exists()

    uid = "e" * 64
    absent = _publish_synthetic(warehouse, tmp_path, uid=uid)
    raw_root.mkdir()
    upstream = raw_root / "legacy-frame.jpg"
    upstream.write_bytes(b"upstream-state-one")
    first_bytes = _publish_synthetic(warehouse, tmp_path, uid=uid)
    upstream.write_bytes(b"upstream-state-two-with-different-bytes")
    second_bytes = _publish_synthetic(warehouse, tmp_path, uid=uid)

    assert absent.reused is False
    assert first_bytes.reused is True
    assert second_bytes.reused is True
    assert {
        absent.derivation_key_sha256,
        first_bytes.derivation_key_sha256,
        second_bytes.derivation_key_sha256,
    } == {absent.derivation_key_sha256}
    assert {
        absent.object_content_sha256,
        first_bytes.object_content_sha256,
        second_bytes.object_content_sha256,
    } == {absent.object_content_sha256}

    manifest = verify_object(warehouse, absent.object_content_sha256)
    serialized = json.dumps(manifest, sort_keys=True)
    for forbidden_identity in (
        "raw_root",
        "raw_jpeg",
        "raw_source",
        "source_jpeg",
        "upstream_raw",
        "upstream_jpeg",
        "dual_source",
    ):
        assert forbidden_identity not in serialized


def test_canonical_positive_path_has_zero_forbidden_raw_opens(warehouse: Path, tmp_path: Path):
    dataset_root = tmp_path / "audited-dataset"
    canonical_root = dataset_root / "lerobot"
    raw_root = dataset_root / "raw"
    canonical_root.mkdir(parents=True)
    raw_root.mkdir()
    (raw_root / "must-not-open.jpg").write_bytes(b"forbidden-upstream")
    video = canonical_root / "canonical.mp4"
    parquet = canonical_root / "episode.parquet"
    video.write_bytes(b"canonical-video")
    parquet.write_bytes(b"canonical-parquet")
    episode = CanonicalEpisodeInput(
        episode_uid="f" * 64,
        cohort_id="synthetic",
        dataset_name="synthetic",
        source_episode_index=0,
        num_frames=1,
        parquet_path=parquet,
        parquet_sha256=sha256_path(parquet),
        parquet_bytes=parquet.stat().st_size,
        canonical_video_path=video,
        canonical_video_sha256=sha256_path(video),
        canonical_video_bytes=video.stat().st_size,
    )

    audit = install_forbidden_upstream_jpeg_audit(dataset_root)
    identity = verify_canonical_input_identity(episode, stage="CANONICAL_POSITIVE_TEST")
    published = _publish_synthetic(warehouse, tmp_path, uid=episode.episode_uid)
    verify_object(warehouse, published.object_content_sha256)

    assert identity["canonical_video"]["sha256"] == episode.canonical_video_sha256
    assert identity["canonical_parquet"]["sha256"] == episode.parquet_sha256
    assert audit.forbidden_open_count == 0
    assert audit.observed_forbidden_paths == ()


class _ExactCanonicalDatasetFactoryDecoder:
    def __init__(self, rgb: np.ndarray) -> None:
        self.rgb = np.array(rgb, copy=True)

    def decode_episode(
        self, episode: CanonicalEpisodeInput
    ) -> tuple[np.ndarray, dict[str, object]]:
        return np.array(self.rgb, copy=True), {
            "dataset_factory_constructed": True,
            "claim_scope": "TEMP_ONLY_EXACT_CANONICAL_DATASET_FACTORY_TEST_DOUBLE",
            "episode_uid": episode.episode_uid,
            "num_frames": episode.num_frames,
            "dtype": "uint8",
            "shape": list(self.rgb.shape),
        }


def test_real_build_object_is_invariant_to_raw_absence_and_two_raw_byte_states(
    tmp_path: Path,
):
    import pandas as pd

    num_frames = 35
    source_root = tmp_path / "canonical-source"
    source_root.mkdir()
    canonical_encode = encode_h264_twice(_moving_rgb(num_frames), source_root)
    canonical_parquet = source_root / "episode.parquet"
    rows = np.arange(num_frames, dtype=np.int64)
    pd.DataFrame(
        {
            "timestamp": rows.astype(np.float64) / producer.FPS,
            "frame_index": rows,
            "episode_index": np.full(num_frames, 7, dtype=np.int64),
            "task_index": np.zeros(num_frames, dtype=np.int64),
            "annotation.human.validity": np.ones(num_frames, dtype=np.int64),
        }
    ).to_parquet(canonical_parquet, index=False)
    fixed_tool_identity = {
        "test_fixture": {
            "identity": "C0_TEMP_ONLY_FIXED_TOOLCHAIN",
            "execution": "CPU_ONLY",
        }
    }

    results = []
    manifests = []
    audit_counts = []
    warehouses: list[Path] = []
    try:
        for state_index, raw_bytes in enumerate((None, b"raw-state-one", b"raw-state-two")):
            state_root = tmp_path / f"state-{state_index}"
            dataset_root = state_root / "dataset"
            canonical_root = dataset_root / "lerobot"
            canonical_root.mkdir(parents=True)
            video = canonical_root / "canonical.mp4"
            parquet = canonical_root / "episode.parquet"
            shutil.copyfile(canonical_encode.bitstream_path, video)
            shutil.copyfile(canonical_parquet, parquet)
            if raw_bytes is not None:
                raw_root = dataset_root / "raw"
                raw_root.mkdir()
                (raw_root / "legacy-frame.jpg").write_bytes(raw_bytes)

            episode = CanonicalEpisodeInput(
                episode_uid="8" * 64,
                cohort_id="synthetic",
                dataset_name="synthetic",
                source_episode_index=7,
                num_frames=num_frames,
                parquet_path=parquet,
                parquet_sha256=sha256_path(parquet),
                parquet_bytes=parquet.stat().st_size,
                canonical_video_path=video,
                canonical_video_sha256=sha256_path(video),
                canonical_video_bytes=video.stat().st_size,
            )
            canonical_rgb, _ = producer.decode_canonical_pyav(
                episode, rows.astype(np.float64) / producer.FPS
            )
            audit = install_forbidden_upstream_jpeg_audit(dataset_root)
            warehouse = state_root / "warehouse"
            scratch = state_root / "scratch"
            scratch.mkdir()
            warehouses.append(warehouse)
            published, evidence = build_canonical_object(
                warehouse,
                episode,
                scratch_root=scratch,
                tool_identity=fixed_tool_identity,
                dataset_factory_decoder=_ExactCanonicalDatasetFactoryDecoder(canonical_rgb),
            )
            results.append(published)
            manifests.append(verify_object(warehouse, published.object_content_sha256))
            audit_counts.append(audit.forbidden_open_count)
            assert evidence["input_sha256_unchanged"] is True
            assert evidence["torchcodec_pyav_exact"] is True
            assert audit.observed_forbidden_paths == ()
    finally:
        for warehouse in warehouses:
            _restore_permissions(warehouse)

    assert audit_counts == [0, 0, 0]
    assert len({result.derivation_key_sha256 for result in results}) == 1
    assert len({result.object_content_sha256 for result in results}) == 1
    assert manifests[0] == manifests[1] == manifests[2]
