from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

from gr00t.data.codec_evidence.episode_bundle import (
    SOURCE_KIND_RAW_JPEG,
    EpisodeBundleKey,
    EpisodeBundleStore,
    write_episode_bundle,
)
import numpy as np
import pytest


_PRODUCER_PATH = (
    Path(__file__).resolve().parents[2] / "examples/G1_ID10_H40_NATIVE30/build_codec_r5_pilot.py"
)
_SPEC = importlib.util.spec_from_file_location("r5_pilot_producer_under_test", _PRODUCER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pilot
_SPEC.loader.exec_module(pilot)


def _synthetic_frames() -> np.ndarray:
    height, width = 64, 96
    y, x = np.mgrid[:height, :width]
    frames = []
    for row in range(pilot.FRAME_COUNT):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (x * 3 + row * 2) % 256
        frame[..., 1] = (y * 4 + row * 3) % 256
        frame[..., 2] = ((x + y) * 2 + row * 5) % 256
        left = row % 40
        frame[16:32, left : left + 20] = np.asarray([255, 20, 100], dtype=np.uint8)
        frames.append(frame)
    return np.stack(frames, axis=0)


@pytest.fixture(scope="module")
def synthetic_codec(tmp_path_factory):
    if not Path("/usr/bin/ffmpeg").is_file() or not Path("/usr/bin/ffprobe").is_file():
        pytest.skip("system ffmpeg/ffprobe are unavailable")
    pytest.importorskip("av")
    root = tmp_path_factory.mktemp("r5-pilot-codec")
    frames = _synthetic_frames()
    encoded = pilot.encode_h264_twice(frames, root)
    packet = pilot.probe_packets(encoded.bitstream_path)
    headers = pilot.trace_h264_headers(encoded.bitstream_path)
    extracted = pilot.extract_mvs_and_decoded_rgb(encoded.bitstream_path)
    arrays = pilot.build_episode_arrays(extracted, packet, headers)
    return {
        "root": root,
        "frames": frames,
        "encoded": encoded,
        "packet": packet,
        "headers": headers,
        "extracted": extracted,
        "arrays": arrays,
    }


def test_deterministic_encoding_packet_clock_and_trace_proof(synthetic_codec):
    encoded = synthetic_codec["encoded"]
    packet = synthetic_codec["packet"]
    headers = synthetic_codec["headers"]
    assert encoded.attempts == 2
    assert encoded.attempt_sha256[0] == encoded.attempt_sha256[1]
    assert encoded.bitstream_sha256 == encoded.attempt_sha256[0]
    expected_time = np.arange(60, dtype=np.int64) * 1000
    np.testing.assert_array_equal(packet.pts, expected_time)
    np.testing.assert_array_equal(packet.dts, expected_time)
    np.testing.assert_array_equal(np.flatnonzero(packet.key_frame), [0, 30])

    assert np.count_nonzero(headers.frame_type == 1) == 2
    assert np.count_nonzero(headers.frame_type == 3) == 58
    expected_reference = np.arange(60, dtype=np.int64) - 1
    expected_reference[[0, 30]] = -1
    np.testing.assert_array_equal(headers.reference_index, expected_reference)
    assert np.all(headers.nal_ref_idc[headers.frame_type == 3] > 0)
    assert np.all(headers.num_ref_idx_active_override_flag[headers.frame_type == 3] == 0)
    assert np.all(headers.ref_pic_list_modification_flag_l0[headers.frame_type == 3] == 0)
    assert np.all(headers.adaptive_ref_pic_marking_mode_flag[headers.frame_type == 3] == 0)
    assert headers.max_num_ref_frames == 1
    assert headers.num_ref_idx_l0_default_active_minus1 == 0
    assert headers.gaps_in_frame_num_allowed_flag == 0
    assert headers.max_frame_num == 16
    assert all(headers.proof_claims.values())
    mv_counts = np.diff(synthetic_codec["extracted"].mv_offsets)
    assert np.all(mv_counts[headers.frame_type == 3] > 0)


def test_reference_identity_is_independent_of_mv_source(synthetic_codec):
    extracted = synthetic_codec["extracted"]
    headers = synthetic_codec["headers"]
    changed_fields = dict(extracted.mv_fields)
    changed_fields["source"] = np.full_like(changed_fields["source"], 123)
    changed = replace(extracted, mv_fields=changed_fields)
    arrays = pilot.build_episode_arrays(changed, synthetic_codec["packet"], headers)
    np.testing.assert_array_equal(arrays["reference_index"], headers.reference_index)
    assert headers.proof_claims["reference_identity_from_bitstream_not_mv_source"]


def test_alignment_gate_records_rows_and_rejects_sequence_shift():
    height, width = 64, 96
    y, x = np.mgrid[:height, :width]
    base = np.stack(((x * 2) % 256, (y * 3) % 256, ((x + y) * 2) % 256), axis=-1).astype(np.uint8)
    raw = []
    for row in range(60):
        frame = base.copy()
        frame[20:28, 10 + row : 18 + row] = np.asarray([255, 10, 10], dtype=np.uint8)
        raw.append(frame)
    raw = np.stack(raw, axis=0)
    canonical = raw.copy()
    alignment = pilot.prove_alignment(raw, canonical)
    assert alignment.per_row_ssim.shape == (60,)
    assert alignment.offset_mean_ssim.shape == (7,)
    assert alignment.median_same_row_ssim == pytest.approx(1.0)
    assert alignment.offsets[np.argmax(alignment.offset_mean_ssim)] == 0
    shifted = np.concatenate([canonical[1:], canonical[-1:]], axis=0)
    with pytest.raises(ValueError, match="Offset zero"):
        pilot.prove_alignment(raw, shifted)


def test_cpu_mv_aggregation_matches_codec_dynamics_core():
    torch = pytest.importorskip("torch")
    from gr00t.model.gr00t_n1d7.codec_dynamics import CodecDynamicsCore

    vectors_per_frame = 8
    offsets = np.arange(0, (60 + 1) * vectors_per_frame, vectors_per_frame, dtype=np.int64)
    one_frame = {
        "source": np.asarray([-1, -1, 1, 1, -2, 2, -3, 3], dtype=np.int32),
        "block_width": np.asarray([16, 8, 32, 64, 16, 8, 32, 64], dtype=np.uint8),
        "block_height": np.asarray([8, 16, 64, 32, 8, 16, 64, 32], dtype=np.uint8),
        "source_x": np.asarray([0, 100, 319, 500, 20, 200, 400, 639], dtype=np.int16),
        "source_y": np.asarray([0, 100, 239, 400, 20, 200, 300, 479], dtype=np.int16),
        "destination_x": np.asarray([10, 319, 320, 500, 10, 319, 320, 500], dtype=np.int16),
        "destination_y": np.asarray([10, 100, 10, 100, 240, 479, 240, 479], dtype=np.int16),
        "flags": np.asarray([0, 1, 2, 3, 4, 8, 16, 32], dtype=np.uint64),
        "motion_x": np.asarray([-8, 4, 12, -16, 20, -24, 28, -32], dtype=np.int32),
        "motion_y": np.asarray([4, -8, 16, 12, -20, 24, -28, 32], dtype=np.int32),
        "motion_scale": np.asarray([0, 1, 2, 4, 8, 16, 2, 4], dtype=np.uint16),
    }
    fields = {name: np.tile(value, 60) for name, value in one_frame.items()}
    cpu, cpu_mask, _, _ = pilot.aggregate_motion_vectors_2x2(offsets, fields, width=640, height=480)
    padded_fields = {
        name: torch.from_numpy(value.reshape(1, 60, vectors_per_frame))
        for name, value in fields.items()
    }
    batch = {
        "motion_vector_fields": padded_fields,
        "mv_mask": torch.ones((1, 60, vectors_per_frame), dtype=torch.bool),
        "frame_validity": {
            "frame_valid": torch.ones((1, 60), dtype=torch.bool),
            "mv_valid": torch.ones((1, 60), dtype=torch.bool),
        },
    }
    expected = CodecDynamicsCore._aggregate_mv(batch, torch.float32)
    expected = expected.reshape(60, 2, 2, 11).cpu().numpy()
    np.testing.assert_array_equal(cpu, expected)
    assert cpu_mask.all()


def _metadata(synthetic_codec) -> dict:
    source = [sha256(frame.tobytes()).hexdigest() for frame in synthetic_codec["frames"]]
    fixture = [sha256(frame.tobytes()).hexdigest() for frame in synthetic_codec["extracted"].rgb]
    return {
        "split": "train",
        "cohort_id": "synthetic",
        "dataset_name": "synthetic",
        "source_episode_index": 0,
        "width": 96,
        "height": 64,
        "time_base_num": 1,
        "time_base_den": 30000,
        "gop_size": 30,
        "source_frame_sha256": source,
        "canonical_frame_sha256": source,
        "fixture_frame_sha256": fixture,
        "tool_identity": {"cpu_only": True},
        "encoder_identity": {"attempts": 2},
        "decoder_identity": {"name": "PyAV"},
        "extractor_identity": {"reference": "trace_headers_not_mv_source"},
    }


def test_writer_roundtrip_retains_bitstream_and_never_overwrites(tmp_path, synthetic_codec):
    uid = "d" * 64
    metadata = _metadata(synthetic_codec)
    path = write_episode_bundle(
        tmp_path,
        episode_uid=uid,
        camera_id=pilot.CAMERA_ID,
        source_kind=SOURCE_KIND_RAW_JPEG,
        metadata=metadata,
        arrays=synthetic_codec["arrays"],
        raw_mv_offsets=synthetic_codec["extracted"].mv_offsets,
        raw_mv_fields=synthetic_codec["extracted"].mv_fields,
        accepted_bitstream=synthetic_codec["encoded"].bitstream_path,
    )
    key = EpisodeBundleKey(uid, pilot.CAMERA_ID, SOURCE_KIND_RAW_JPEG)
    store = EpisodeBundleStore(tmp_path)
    loaded = store.get(key)
    assert path.name == "bundle.json"
    assert set(np.unique(loaded.frame_type)) == {1, 3}
    np.testing.assert_array_equal(
        loaded.reference_index, synthetic_codec["headers"].reference_index
    )
    audit = store.audit_raw_motion_vectors(key)
    np.testing.assert_array_equal(audit.offsets, synthetic_codec["extracted"].mv_offsets)
    retained = store.audit_accepted_bitstream(key)
    assert sha256(retained.read_bytes()).hexdigest() == synthetic_codec["encoded"].bitstream_sha256

    with pytest.raises(FileExistsError, match="already exists"):
        write_episode_bundle(
            tmp_path,
            episode_uid=uid,
            camera_id=pilot.CAMERA_ID,
            source_kind=SOURCE_KIND_RAW_JPEG,
            metadata=metadata,
            arrays=synthetic_codec["arrays"],
        )


def test_output_root_reservation_is_no_clobber(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        pilot.reserve_output_root(existing, dataset_root=dataset_root)
    assert marker.read_text(encoding="utf-8") == "user data"
    with pytest.raises(ValueError, match="dataset_root"):
        pilot.reserve_output_root(dataset_root / "forbidden", dataset_root=dataset_root)


def _write_manifest_identity_fixture(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    catalog_path = dataset_root / "catalog/current.sqlite3"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_bytes(b"synthetic catalog")
    catalog = {
        "database_path": str(catalog_path),
        "database_sha256": sha256(catalog_path.read_bytes()).hexdigest(),
    }
    dataset_name = "synthetic manifest dataset"
    episode_index = 7
    parquet_sha = sha256(b"parquet identity").hexdigest()
    video_sha = sha256(b"video identity").hexdigest()
    uid_material = f"icra27-episode-v1\0{dataset_name}\0{episode_index}\0{parquet_sha}\0{video_sha}"
    uid = sha256(uid_material.encode("utf-8")).hexdigest()
    episode = pilot.PilotEpisode(
        episode_uid=uid,
        cohort_id="synthetic_cohort",
        dataset_name=dataset_name,
        source_episode_index=episode_index,
        expected_rows=60,
        expected_global_index_start=0,
        raw_relative_path="raw/synthetic/episode_0007",
        parquet_relative_path="lerobot/synthetic/episode_000007.parquet",
        video_relative_path="lerobot/synthetic/episode_000007.mp4",
    )
    monkeypatch.setattr(pilot, "PILOT_EPISODES", (episode,))
    payload_row = {
        "episode_uid": uid,
        "cohort_id": episode.cohort_id,
        "dataset_name": dataset_name,
        "episode_index": episode_index,
        "rows": episode.expected_rows,
        "parquet_sha256": parquet_sha,
        "cam_left_high_sha256": video_sha,
        "parquet_relative_path": episode.parquet_relative_path,
        "cam_left_high_relative_path": episode.video_relative_path,
    }
    payload = {
        "schema_version": pilot.PAYLOAD_SCHEMA_VERSION,
        "contract_id": pilot.DATASET_CONTRACT_ID,
        "selected_camera_key": "observation.images.cam_left_high",
        "contract_instruction": pilot.CONTRACT_INSTRUCTION,
        "catalog": catalog,
        "dataset_root": str(dataset_root),
        "episodes": [payload_row],
    }
    contract_root = tmp_path / "contract"
    contract_root.mkdir()
    payload_path = contract_root / "payload_manifest.json"
    payload_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    assignment = {
        "episode_uid": uid,
        "cohort_id": episode.cohort_id,
        "dataset_name": dataset_name,
        "episode_index": episode_index,
        "rows": episode.expected_rows,
        "split": "train",
        "eligible": True,
    }
    split = {
        "schema_version": pilot.SPLIT_SCHEMA_VERSION,
        "contract_id": pilot.DATASET_CONTRACT_ID,
        "split_manifest_id": pilot.SPLIT_MANIFEST_ID,
        "split_algorithm_id": pilot.SPLIT_ALGORITHM_ID,
        "split_seed": pilot.SPLIT_SEED,
        "contract_instruction": pilot.CONTRACT_INSTRUCTION,
        "selected_camera_key": "observation.images.cam_left_high",
        "dataset_root": str(dataset_root),
        "catalog": catalog,
        "payload_manifest_path": str(payload_path),
        "payload_manifest_sha256": sha256(payload_path.read_bytes()).hexdigest(),
        "assignments": [assignment],
        "train_episode_uid_digest": sha256(f"{uid}\n".encode()).hexdigest(),
    }
    split_path = contract_root / "split_manifest.json"
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    return split_path, payload_path


def test_manifest_identity_rejects_duplicate_uid(tmp_path, monkeypatch):
    split_path, _ = _write_manifest_identity_fixture(tmp_path, monkeypatch)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["assignments"].append(dict(split["assignments"][0]))
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated or incomplete"):
        pilot._resolve_manifest_inputs(split_path)


def test_manifest_identity_rejects_changed_payload_with_old_uid(tmp_path, monkeypatch):
    split_path, payload_path = _write_manifest_identity_fixture(tmp_path, monkeypatch)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["episodes"][0]["parquet_sha256"] = sha256(b"changed parquet").hexdigest()
    payload_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["payload_manifest_sha256"] = sha256(payload_path.read_bytes()).hexdigest()
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="UID no longer derives"):
        pilot._resolve_manifest_inputs(split_path)


def test_episode_dispatches_both_frozen_source_kinds(tmp_path, monkeypatch):
    episode = pilot.PilotEpisode(
        episode_uid="e" * 64,
        cohort_id="synthetic",
        dataset_name="synthetic",
        source_episode_index=0,
        expected_rows=60,
        expected_global_index_start=0,
        raw_relative_path="raw/episode",
        parquet_relative_path="lerobot/episode.parquet",
        video_relative_path="lerobot/episode.mp4",
    )
    dataset_root = tmp_path / "dataset"
    raw_root = dataset_root / episode.raw_relative_path
    raw_root.mkdir(parents=True)
    (raw_root / "data.json").write_text("{}", encoding="utf-8")
    parquet = dataset_root / episode.parquet_relative_path
    video = dataset_root / episode.video_relative_path
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"parquet")
    video.write_bytes(b"video")
    payload_row = {
        "parquet_sha256": sha256(parquet.read_bytes()).hexdigest(),
        "parquet_bytes": parquet.stat().st_size,
        "cam_left_high_sha256": sha256(video.read_bytes()).hexdigest(),
        "cam_left_high_bytes": video.stat().st_size,
    }
    raw = np.zeros((60, 64, 96, 3), dtype=np.uint8)
    canonical = np.ones_like(raw)
    ledger = [
        {"relative_path": f"raw/episode/colors/{row:06d}.jpg", "sha256": "f" * 64, "bytes": 1}
        for row in range(60)
    ]
    monkeypatch.setattr(
        pilot,
        "_read_raw_episode",
        lambda *args, **kwargs: (raw, [{}] * 60, ledger),
    )
    monkeypatch.setattr(
        pilot,
        "_read_canonical_parquet",
        lambda *args, **kwargs: np.arange(60, dtype=np.float64) / 30.0,
    )
    monkeypatch.setattr(
        pilot,
        "_decode_canonical_video",
        lambda *args, **kwargs: (canonical, {"codec": "sniffed"}),
    )
    alignment = pilot.AlignmentEvidence(
        per_row_ssim=np.ones((60,), dtype=np.float64),
        per_row_psnr_db=np.full((60,), 40.0, dtype=np.float64),
        offsets=np.arange(-3, 4, dtype=np.int8),
        offset_mean_ssim=np.asarray([0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7]),
        median_same_row_ssim=1.0,
    )
    monkeypatch.setattr(pilot, "prove_alignment", lambda *args, **kwargs: alignment)
    dispatched = []

    def fake_encode_bundle(
        output_root,
        selected_episode,
        source_rgb,
        canonical_rgb,
        source_kind,
        *args,
        **kwargs,
    ):
        dispatched.append((source_kind, source_rgb))
        return {"episode_uid": selected_episode.episode_uid, "source_kind": source_kind}

    monkeypatch.setattr(pilot, "_encode_source_bundle", fake_encode_bundle)
    output = tmp_path / "output"
    output.mkdir()
    summaries = pilot._build_one_episode(
        output,
        dataset_root,
        episode,
        payload_row,
        {"manifest": "synthetic"},
        ffmpeg=Path("/usr/bin/ffmpeg"),
        ffprobe=Path("/usr/bin/ffprobe"),
        common_tool_identity={"cpu": True},
    )
    assert [row["source_kind"] for row in summaries] == [
        "RAW_JPEG_DERIVED",
        "NATIVE_CANONICAL_DECODED_DERIVED",
    ]
    assert dispatched[0][1] is raw
    assert dispatched[1][1] is canonical
