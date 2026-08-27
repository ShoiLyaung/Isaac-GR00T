from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.episode_bundle import (
    FRAME_TYPE_IDR,
    FRAME_TYPE_P,
    make_synthetic_episode_bundle,
)
from gr00t.data.codec_evidence.production import (
    GOP,
    PTS_STEP,
    CanonicalEpisodeInput,
    assert_torchcodec_pyav_exact,
    build_episode_arrays,
    decode_canonical_pyav,
    encode_h264_twice,
    extract_mvs_and_decoded_rgb,
    probe_packets,
    raster_sha256,
    sha256_path,
    toolchain_identity,
    trace_h264_headers,
    validate_canonical_parquet,
    verify_canonical_input_identity,
)
import numpy as np
import pandas as pd
import pytest


def _moving_rgb(num_frames: int, *, height: int = 480, width: int = 640) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = (rows + columns) % 256
    base[..., 1] = (2 * rows + columns) % 256
    base[..., 2] = (rows + 2 * columns) % 256
    frames: list[np.ndarray] = []
    for row in range(num_frames):
        frame = np.roll(base, shift=(row % 7, row % 11), axis=(0, 1)).copy()
        left = (row * 5) % (width - 32)
        frame[32:64, left : left + 32] = np.asarray([255, 20, 30], dtype=np.uint8)
        frames.append(frame)
    return np.stack(frames)


def _canonical_episode(tmp_path: Path, bitstream: Path, num_frames: int) -> CanonicalEpisodeInput:
    parquet = tmp_path / "episode_000000.parquet"
    rows = np.arange(num_frames, dtype=np.int64)
    pd.DataFrame(
        {
            "timestamp": rows.astype(np.float64) / 30.0,
            "frame_index": rows,
            "episode_index": np.zeros(num_frames, dtype=np.int64),
            "task_index": np.zeros(num_frames, dtype=np.int64),
            "annotation.human.validity": np.ones(num_frames, dtype=np.int64),
        }
    ).to_parquet(parquet, index=False)
    return CanonicalEpisodeInput(
        episode_uid="a" * 64,
        cohort_id="focused-production-test",
        dataset_name="focused-production-test",
        source_episode_index=0,
        num_frames=num_frames,
        parquet_path=parquet,
        parquet_sha256=sha256_path(parquet),
        parquet_bytes=parquet.stat().st_size,
        canonical_video_path=bitstream,
        canonical_video_sha256=sha256_path(bitstream),
        canonical_video_bytes=bitstream.stat().st_size,
    )


@pytest.mark.parametrize("num_frames", [40, 65])
def test_real_cpu_dynamic_pipeline_is_deterministic_causal_and_bundle_compatible(
    tmp_path, num_frames
):
    source_rgb = _moving_rgb(num_frames)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    encoded = encode_h264_twice(source_rgb, scratch)
    assert encoded.attempt_sha256 == (encoded.bitstream_sha256,) * 2
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        encode_h264_twice(source_rgb, scratch)

    packets = probe_packets(encoded.bitstream_path, num_frames=num_frames)
    headers = trace_h264_headers(encoded.bitstream_path, num_frames=num_frames)
    extracted = extract_mvs_and_decoded_rgb(encoded.bitstream_path, num_frames=num_frames)
    arrays = build_episode_arrays(extracted, packets, headers)

    rows = np.arange(num_frames, dtype=np.int64)
    gop_starts = rows[rows % GOP == 0]
    expected_type = np.full(num_frames, FRAME_TYPE_P, dtype=np.uint8)
    expected_type[gop_starts] = FRAME_TYPE_IDR
    expected_reference = rows - 1
    expected_reference[gop_starts] = -1
    expected_clock = rows * PTS_STEP

    assert np.array_equal(packets.pts, expected_clock)
    assert np.array_equal(packets.dts, expected_clock)
    assert np.array_equal(packets.duration, np.full(num_frames, PTS_STEP))
    assert np.array_equal(np.flatnonzero(packets.key_frame), gop_starts)
    assert np.array_equal(headers.frame_type, expected_type)
    assert np.array_equal(headers.reference_index, expected_reference)
    assert all(headers.proof_claims.values())

    assert extracted.mv_offsets.dtype == np.dtype("int64")
    assert extracted.mv_offsets.shape == (num_frames + 1,)
    counts = np.diff(extracted.mv_offsets)
    assert np.array_equal(counts[gop_starts], np.zeros(len(gop_starts), dtype=np.int64))
    assert np.all(counts[expected_type == FRAME_TYPE_P] > 0)
    assert set(extracted.mv_fields) == set(MV_FIELD_DTYPES)
    for name, dtype in MV_FIELD_DTYPES.items():
        assert extracted.mv_fields[name].dtype == dtype
        assert extracted.mv_fields[name].shape == (int(extracted.mv_offsets[-1]),)

    assert np.array_equal(arrays["frame_index"], rows)
    assert np.array_equal(arrays["gop_id"], rows // GOP)
    assert np.array_equal(arrays["gop_position"], rows % GOP)
    assert np.array_equal(arrays["reference_index"], expected_reference)
    assert np.array_equal(arrays["anchor_frame_index"], gop_starts)
    assert np.array_equal(arrays["anchor_index_per_frame"], (rows // GOP) * GOP)
    assert arrays["mv_cell_features"].shape == (num_frames, 2, 2, 11)
    assert arrays["rpe"].shape == (num_frames, 2, 2, 4)
    assert arrays["anchor_appearance"].shape == (len(gop_starts), 3, 256, 256)

    episode = _canonical_episode(tmp_path, encoded.bitstream_path, num_frames)
    before = verify_canonical_input_identity(episode, stage="BEFORE_READ")
    timestamps = validate_canonical_parquet(episode)
    canonical_rgb, canonical_identity = decode_canonical_pyav(episode, timestamps)
    assert np.array_equal(canonical_rgb, extracted.rgb)
    assert canonical_identity["decode_policy"] == "SNIFF_ACTUAL_CODEC_THEN_RGB24_CPU"
    parity = assert_torchcodec_pyav_exact(episode.canonical_video_path, canonical_rgb)
    assert parity["all_rows_array_equal"] is True
    assert parity["max_abs_diff"] == 0
    assert parity["per_row_hash_equality_count"] == num_frames
    assert (
        verify_canonical_input_identity(episode, stage="AFTER_READ")["canonical_video"]
        == (before["canonical_video"])
    )

    identity = {"name": "focused-test", "version": "1", "sha256": "f" * 64}
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind="NATIVE_CANONICAL_DECODED_DERIVED",
        identity=identity,
    )
    synthetic = make_synthetic_episode_bundle(num_frames=num_frames)
    production = replace(
        synthetic,
        source_frame_sha256=tuple(raster_sha256(frame) for frame in canonical_rgb),
        canonical_frame_sha256=tuple(raster_sha256(frame) for frame in canonical_rgb),
        fixture_frame_sha256=tuple(raster_sha256(frame) for frame in extracted.rgb),
        tool_identity=identity,
        encoder_identity=identity,
        decoder_identity=identity,
        extractor_identity=identity,
        codec_contract=contract,
        codec_contract_sha256=codec_contract_sha256(contract),
        **arrays,
    )
    assert production.codec_contract["fixture_mode"] == PRODUCTION_FIXTURE_MODE
    assert production.num_frames == num_frames
    assert production.view(30).anchor_mask.tolist() == [True, True]


def test_toolchain_identity_freezes_binaries_encoder_extractor_and_loaded_decoder_libraries():
    identity = toolchain_identity(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe")
    repeated = toolchain_identity(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe")
    assert repeated == identity
    assert identity["schema_version"] == "codec-evidence-production-toolchain-v1"
    assert len(identity["identity_sha256"]) == 64
    assert identity["ffmpeg"]["sha256"] == sha256_path(identity["ffmpeg"]["path"])
    assert identity["ffprobe"]["sha256"] == sha256_path(identity["ffprobe"]["path"])
    assert identity["libx264"]["sha256"] == sha256_path(identity["libx264"]["path"])
    system_sonames = {item["soname"] for item in identity["system_ffmpeg_codec_libraries"]}
    assert any(name.startswith("libavcodec") for name in system_sonames)
    assert any(name.startswith("libx264") for name in system_sonames)
    for library in identity["system_ffmpeg_codec_libraries"]:
        assert library["sha256"] == sha256_path(library["path"])
    extractor = identity["motion_vector_extractor"]
    assert extractor["production_module"]["symbol"] == "extract_mvs_and_decoded_rgb"
    assert extractor["pyav_motionvectors_module"]["path"].endswith(".so")
    assert any(
        item["name"] == "av.sidedata.motionvectors"
        for item in identity["pyav"]["loaded_extension_modules"]
    )
    for extension in identity["pyav"]["loaded_extension_modules"]:
        assert extension["sha256"] == sha256_path(extension["path"])
    assert identity["torchcodec"]["actually_loaded_shared_libraries"]
    for library in identity["torchcodec"]["actually_loaded_shared_libraries"]:
        assert library["sha256"] == sha256_path(library["path"])


def test_tool_paths_must_be_absolute(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="ffmpeg must be an absolute path"):
        encode_h264_twice(_moving_rgb(1), scratch, ffmpeg="ffmpeg")
    with pytest.raises(ValueError, match="ffprobe must be an absolute path"):
        toolchain_identity(ffmpeg="/usr/bin/ffmpeg", ffprobe="ffprobe")


def test_legal_intra_coded_p_picture_may_have_empty_mv_csr(tmp_path):
    frames = np.random.default_rng(7).integers(0, 256, size=(5, 480, 640, 3), dtype=np.uint8)
    scratch = tmp_path / "scratch-empty-mv"
    scratch.mkdir()
    encoded = encode_h264_twice(frames, scratch)
    packets = probe_packets(encoded.bitstream_path, num_frames=5)
    headers = trace_h264_headers(encoded.bitstream_path, num_frames=5)
    extracted = extract_mvs_and_decoded_rgb(encoded.bitstream_path, num_frames=5)
    assert np.all(headers.frame_type[1:] == FRAME_TYPE_P)
    assert np.any(np.diff(extracted.mv_offsets)[1:] == 0)
    arrays = build_episode_arrays(extracted, packets, headers)
    assert arrays["mv_cell_features"].shape == (5, 2, 2, 11)
    assert np.isfinite(arrays["mv_cell_features"]).all()
