from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

from gr00t.data.codec_evidence import prefix_materializer as factory
from gr00t.data.codec_evidence.config import parse_codec_model_config
from gr00t.data.codec_evidence.episode_bundle import (
    FRAME_TYPE_IDR,
    FRAME_TYPE_P,
    SOURCE_KIND_NATIVE_CANONICAL,
    EpisodeBundleKey,
    validate_episode_view,
)
from gr00t.data.codec_evidence.production import (
    X264_CAUSAL_PARAMETERS,
    build_episode_arrays,
    encode_h264_twice,
    extract_mvs_and_decoded_rgb,
    probe_packets,
    probe_x264_encoder_options,
    trace_h264_headers,
)
import numpy as np
import pytest


FACTORY_PATH = Path(factory.__file__).resolve(strict=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(*, expanded: bool = True) -> dict:
    digest = _sha256(FACTORY_PATH)
    value = {
        "schema_version": factory.TRAINING_CONTRACT_SCHEMA_VERSION,
        "runtime_ready": True,
        "online_prefix_equivalent": True,
        "checkpoint_index_sha256": "a" * 64,
        "training_source_commit": "b" * 40,
        "runtime_source_commit": "c" * 40,
        "materializer_id": factory.EXPECTED_MATERIALIZER_ID,
        "materializer_sha256": digest,
        "source_kind": factory.EXPECTED_SOURCE_KIND,
        "camera_id": "cam_left_high",
        "simulator_hz": 50,
        "evidence_hz": 30,
        "history_length": 8,
        "future_frame_fallback_allowed": False,
    }
    if expanded:
        value.update(
            {
                "experiment_id": "causal-experiment",
                "run_id": "causal-run-attempt-1",
                "checkpoint_identity": "d" * 64,
                "factory_module": str(FACTORY_PATH),
                "factory_module_sha256": digest,
            }
        )
    return value


def _frames(count: int, *, offset: int = 0) -> tuple[np.ndarray, ...]:
    rows, columns = np.indices((480, 640), dtype=np.uint16)
    base = np.stack(
        (
            (columns + offset) % 256,
            (rows * 2 + offset) % 256,
            (columns // 2 + rows // 3 + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return tuple(np.ascontiguousarray(np.roll(base, index, axis=1)) for index in range(count))


def _production_arrays(frames: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
    rgb = np.stack(frames)
    with tempfile.TemporaryDirectory(prefix="prefix-materializer-expected-") as temporary:
        encoded = encode_h264_twice(rgb, Path(temporary))
        probe_x264_encoder_options(encoded.bitstream_path)
        packets = probe_packets(encoded.bitstream_path, num_frames=len(frames))
        headers = trace_h264_headers(encoded.bitstream_path, num_frames=len(frames))
        extracted = extract_mvs_and_decoded_rgb(encoded.bitstream_path, num_frames=len(frames))
        return build_episode_arrays(extracted, packets, headers)


def _assert_views_equal(left, right) -> None:
    assert left.key == right.key
    for name in left.__dataclass_fields__:
        if name == "key":
            continue
        assert np.array_equal(getattr(left, name), getattr(right, name)), name


def test_factory_is_self_hash_bound_and_names_exact_production_dependencies() -> None:
    materialize = factory.build_prefix_materializer(_contract())
    identity = materialize.factory_identity

    assert callable(materialize)
    assert materialize.materializer_id == factory.EXPECTED_MATERIALIZER_ID
    assert materialize.materializer_sha256 == _sha256(FACTORY_PATH)
    assert identity["factory"]["path"] == str(FACTORY_PATH)
    assert identity["factory"]["sha256"] == _sha256(FACTORY_PATH)
    assert identity["x264_parameters"] == list(X264_CAUSAL_PARAMETERS)
    assert identity["stateless_reset_isolation"] is True
    assert identity["future_rows_accepted"] is False
    assert identity["retained_bitstream"] is False

    drifted = _contract()
    drifted["materializer_sha256"] = "0" * 64
    with pytest.raises(factory.PrefixMaterializerError, match="differs from this factory"):
        factory.build_prefix_materializer(drifted)


def test_factory_view_model_and_processor_share_native_canonical_source_kind() -> None:
    codec_config = {
        "enabled": True,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        "d_codec": 64,
        "conditioner_heads": 8,
    }
    model_config = parse_codec_model_config(codec_config)
    processor_config = parse_codec_model_config(codec_config)

    assert factory.EXPECTED_SOURCE_KIND == "NATIVE_CANONICAL_DECODED_DERIVED"
    assert model_config["source_kind"] == factory.EXPECTED_SOURCE_KIND
    assert processor_config["source_kind"] == factory.EXPECTED_SOURCE_KIND
    assert _contract()["source_kind"] == factory.EXPECTED_SOURCE_KIND

    drifted = _contract()
    drifted["source_kind"] = "NATIVE_CAUSAL_PREFIX_DERIVED"
    with pytest.raises(factory.PrefixMaterializerError, match="source_kind is not exact"):
        factory.build_prefix_materializer(drifted)
    with pytest.raises(ValueError, match="source_kind"):
        EpisodeBundleKey("f" * 64, "cam_left_high", "NATIVE_CAUSAL_PREFIX_DERIVED")


def test_factory_prefix_equals_same_rows_from_full_production_encode(monkeypatch, tmp_path) -> None:
    frames = _frames(12)
    prefix = frames[:9]
    episode_uid = "1" * 64
    expected_arrays = _production_arrays(frames)
    expected = factory.episode_arrays_to_view(
        expected_arrays,
        episode_uid=episode_uid,
        source_kind=factory.EXPECTED_SOURCE_KIND,
        observation_row=8,
    )
    monkeypatch.setattr(factory.tempfile, "tempdir", str(tmp_path))

    observed = factory.build_prefix_materializer(_contract())(prefix, episode_uid, 8)

    validate_episode_view(observed)
    _assert_views_equal(observed, expected)
    assert list(tmp_path.iterdir()) == []


def test_factory_is_stateless_across_episode_reset() -> None:
    materialize = factory.build_prefix_materializer(_contract(expanded=False))
    first = materialize(_frames(5), "2" * 64, 4)
    after_reset = materialize(_frames(1, offset=77), "3" * 64, 0)
    fresh = factory.build_prefix_materializer(_contract(expanded=False))(
        _frames(1, offset=77), "3" * 64, 0
    )

    assert first.key.episode_uid == "2" * 64
    assert after_reset.key.episode_uid == "3" * 64
    assert int(after_reset.frame_mask.sum()) == 1
    _assert_views_equal(after_reset, fresh)


def test_factory_gop30_view_has_old_and_new_anchor_without_future_rows() -> None:
    view = factory.build_prefix_materializer(_contract())(_frames(31), "4" * 64, 30)
    active = view.frame_mask

    assert np.array_equal(view.frame_index[active], np.arange(23, 31, dtype=np.int64))
    assert np.array_equal(
        view.frame_type[active],
        np.asarray([FRAME_TYPE_P] * 7 + [FRAME_TYPE_IDR], dtype=np.uint8),
    )
    assert np.array_equal(view.anchor_frame_index[view.anchor_mask], [0, 30])
    assert np.array_equal(view.frame_anchor_slot[active], [0] * 7 + [1])
    assert int(view.observation_row) == 30


def test_factory_rejects_future_or_missing_rows_before_encoding(monkeypatch) -> None:
    materialize = factory.build_prefix_materializer(_contract())
    calls = []

    def unexpected_encode(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("encoder must not run")

    monkeypatch.setattr(factory, "encode_h264_twice", unexpected_encode)
    with pytest.raises(factory.PrefixMaterializerError, match="future rows are forbidden"):
        materialize(_frames(2), "5" * 64, 0)
    with pytest.raises(factory.PrefixMaterializerError, match="future rows are forbidden"):
        materialize(_frames(1), "5" * 64, 1)
    with pytest.raises(factory.PrefixMaterializerError, match="episode_uid"):
        materialize(_frames(1), "not-a-sha", 0)
    assert calls == []
