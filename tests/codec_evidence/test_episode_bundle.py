from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from gr00t.data.codec_evidence import episode_bundle as episode_bundle_module
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.episode_bundle import (
    FRAME_TYPE_IDR,
    K_MAX,
    MAX_EPISODE_BATCH_SIZE,
    METADATA_WIDTH,
    MV_CELL_WIDTH,
    SOURCE_KIND_NATIVE_CANONICAL,
    SOURCE_KIND_RAW_JPEG,
    EpisodeBundleKey,
    EpisodeBundleStore,
    InMemoryEpisodeBundleStore,
    assert_episode_batch_conforms,
    collate_episode_views_numpy,
    make_synthetic_episode_bundle,
    write_episode_bundle,
)
import numpy as np
import pytest


def _raw_motion_vectors(num_frames: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    offsets = np.arange(num_frames + 1, dtype=np.int64)
    fields = {
        name: np.arange(num_frames, dtype=np.int64).astype(dtype)
        for name, dtype in MV_FIELD_DTYPES.items()
    }
    fields["block_width"][:] = 16
    fields["block_height"][:] = 16
    fields["motion_scale"][:] = 4
    return offsets, fields


def _write(
    root: Path,
    bundle,
    *,
    raw: bool = False,
    accepted_bitstream: Path | None = None,
) -> Path:
    offsets, fields = _raw_motion_vectors(bundle.num_frames)
    return write_episode_bundle(
        root,
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
        raw_mv_offsets=offsets if raw else None,
        raw_mv_fields=fields if raw else None,
        accepted_bitstream=accepted_bitstream,
    )


def test_synthetic_bundle_is_immutable_and_t8_padding_is_numeric_zero():
    bundle = make_synthetic_episode_bundle()
    with pytest.raises(ValueError, match="read-only"):
        bundle.frame_index[0] = 9
    with pytest.raises(TypeError):
        bundle.tool_identity["name"] = "changed"

    view = bundle.view(3)
    assert view.frame_mask.tolist() == [False] * 4 + [True] * 4
    for name in (
        "frame_index",
        "frame_type",
        "frame_validity",
        "frame_reason_code",
        "mv_cell_features",
        "mv_cell_mask",
        "rpe",
        "rpe_mask",
        "metadata_features",
    ):
        assert not np.asarray(getattr(view, name))[:4].any(), name
    assert np.array_equal(view.frame_anchor_slot[:4], np.full(4, -1, dtype=np.int64))
    assert view.metadata_features.shape == (8, METADATA_WIDTH)
    assert view.metadata_features[4, 0] == 1.0
    assert view.metadata_features[4, 3] == pytest.approx(-3.0 / 7.0)
    assert view.metadata_features[4, 4] == pytest.approx(-0.1)
    assert view.metadata_features[-1, 3] == 0.0
    assert view.metadata_features[-1, 4] == 0.0
    assert view.metadata_features[-1, 6] == pytest.approx(3.0 / 29.0)
    assert view.metadata_features[-1, 7] == pytest.approx(1.0 / 30.0)


def test_episode_global_predecessor_and_cross_gop_anchor_closure():
    bundle = make_synthetic_episode_bundle()

    # The first frame of row 14's T8 view is frame 7.  Its predecessor frame 6
    # lies outside the view but was validated against the episode-global graph.
    mid_gop = bundle.view(14)
    assert mid_gop.frame_index.tolist() == list(range(7, 15))
    assert bundle.reference_index[7] == 6
    assert mid_gop.anchor_mask.tolist() == [True, False]
    assert np.all(mid_gop.frame_anchor_slot == 0)

    boundary = bundle.view(30)
    assert boundary.frame_index.tolist() == list(range(23, 31))
    assert boundary.anchor_mask.tolist() == [True, True]
    assert boundary.anchor_frame_index.tolist() == [0, 30]
    assert boundary.frame_anchor_slot.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert boundary.frame_type[-1] == FRAME_TYPE_IDR

    bad_reference = bundle.reference_index.copy()
    bad_reference[7] = 5
    with pytest.raises(ValueError, match="episode-global predecessor"):
        replace(bundle, reference_index=bad_reference)

    bad_anchor = bundle.anchor_index_per_frame.copy()
    bad_anchor[30] = 0
    with pytest.raises(ValueError, match="nearest available past IDR"):
        replace(bundle, anchor_index_per_frame=bad_anchor)


def test_k2_anchor_bank_fails_instead_of_truncating():
    bundle = make_synthetic_episode_bundle(num_frames=8, gop_size=3)
    assert bundle.anchor_frame_index.tolist() == [0, 3, 6]
    with pytest.raises(ValueError, match="K_MAX anchors"):
        bundle.view(7)


def test_bounded_numpy_collator_and_preallocation_guards(monkeypatch):
    bundle = make_synthetic_episode_bundle()
    early = bundle.view(3)
    boundary = bundle.view(30)
    batch = collate_episode_views_numpy([early, boundary])
    assert_episode_batch_conforms(batch)
    assert batch["mv_cell_features"].shape == (2, 8, 2, 2, MV_CELL_WIDTH)
    assert batch["rpe"].shape == (2, 8, 2, 2, 4)
    assert batch["metadata_features"].shape == (2, 8, METADATA_WIDTH)
    assert batch["anchor_appearance"].shape == (2, K_MAX, 3, 256, 256)
    assert "motion_vector_fields" not in batch
    assert "raw_mv_offsets" not in batch

    extra = dict(batch)
    extra["unbounded"] = np.empty(0)
    with pytest.raises(ValueError, match="batch keys"):
        assert_episode_batch_conforms(extra)

    called = False

    def forbidden_stack(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("np.stack ran before the transport guard")

    monkeypatch.setattr(episode_bundle_module, "MAX_EPISODE_TRANSPORT_BYTES", 1)
    monkeypatch.setattr(episode_bundle_module.np, "stack", forbidden_stack)
    with pytest.raises(ValueError, match="TRANSPORT_BYTES"):
        collate_episode_views_numpy([early])
    assert called is False


def test_batch_count_guard_precedes_allocation(monkeypatch):
    view = make_synthetic_episode_bundle().view(7)
    called = False

    def forbidden_stack(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("np.stack ran before the count guard")

    monkeypatch.setattr(episode_bundle_module.np, "stack", forbidden_stack)
    with pytest.raises(ValueError, match="MAX_EPISODE_BATCH_SIZE"):
        collate_episode_views_numpy([view] * (MAX_EPISODE_BATCH_SIZE + 1))
    assert called is False


def test_write_roundtrip_is_non_overwritable_and_raw_mv_is_audit_only(tmp_path):
    bundle = make_synthetic_episode_bundle()
    bitstream = tmp_path / "accepted.mp4"
    bitstream.write_bytes(b"synthetic accepted bitstream")
    manifest_path = _write(tmp_path / "bundles", bundle, raw=True, accepted_bitstream=bitstream)
    expected_path = (
        tmp_path
        / "bundles"
        / bundle.key.episode_uid
        / bundle.key.camera_id
        / bundle.key.source_kind
        / "bundle.json"
    )
    assert manifest_path == expected_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_kind"] == SOURCE_KIND_NATIVE_CANONICAL
    assert manifest["num_frames"] == bundle.num_frames
    assert manifest["k_max"] == 2
    assert len(manifest["frame_hashes"]["source"]) == bundle.num_frames
    assert manifest["accepted_bitstream"]["sha256"] == sha256(bitstream.read_bytes()).hexdigest()
    assert manifest["raw_motion_vector_audit"] is not None

    with pytest.raises(FileExistsError, match="already exists"):
        _write(tmp_path / "bundles", bundle, raw=True)

    store = EpisodeBundleStore(tmp_path / "bundles")
    loaded = store.get(bundle.key)
    assert loaded.key == bundle.key
    assert np.array_equal(loaded.mv_cell_features, bundle.mv_cell_features)
    assert store.get_view(bundle.key, 30).anchor_mask.all()
    audit = store.audit_raw_motion_vectors(bundle.key, frame_index=7)
    assert audit.offsets.tolist() == [0, 1]
    assert audit.fields["motion_x"].tolist() == [7]
    with pytest.raises(ValueError, match="read-only"):
        audit.fields["motion_x"][0] = 0
    assert store.audit_accepted_bitstream(bundle.key).read_bytes() == bitstream.read_bytes()


def test_array_and_pinned_manifest_hashes_fail_closed(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    manifest_path = _write(tmp_path / "array-tamper", bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mv_path = manifest_path.parent / manifest["arrays"]["mv_cell_features"]["relative_path"]
    mv_path.chmod(0o644)
    with mv_path.open("r+b") as stream:
        stream.seek(-1, 2)
        final = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([final[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        EpisodeBundleStore(tmp_path / "array-tamper").get(bundle.key)

    other = make_synthetic_episode_bundle(
        num_frames=8,
        episode_uid="b" * 64,
    )
    other_manifest = _write(tmp_path / "manifest-tamper", other)
    pin = sha256(other_manifest.read_bytes()).hexdigest()
    other_manifest.chmod(0o644)
    data = json.loads(other_manifest.read_text(encoding="utf-8"))
    data["split"] = "test"
    other_manifest.write_text(json.dumps(data), encoding="utf-8")
    pinned = EpisodeBundleStore(
        tmp_path / "manifest-tamper",
        manifest_sha256_by_key={other.key: pin},
    )
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        pinned.get(other.key)


def test_source_kind_is_part_of_identity_and_path(tmp_path):
    native = make_synthetic_episode_bundle(num_frames=8)
    raw = make_synthetic_episode_bundle(
        num_frames=8,
        source_kind=SOURCE_KIND_RAW_JPEG,
    )
    native_manifest = _write(tmp_path, native)
    raw_manifest = _write(tmp_path, raw)
    assert native_manifest != raw_manifest
    assert native_manifest.parent.name == SOURCE_KIND_NATIVE_CANONICAL
    assert raw_manifest.parent.name == SOURCE_KIND_RAW_JPEG
    store = EpisodeBundleStore(tmp_path)
    assert store.get(native.key).key != store.get(raw.key).key
    with pytest.raises(ValueError, match="source_kind"):
        EpisodeBundleKey(native.key.episode_uid, native.key.camera_id, "UNKNOWN")


def test_strict_masks_hash_lengths_and_in_memory_store():
    bundle = make_synthetic_episode_bundle(num_frames=8)
    store = InMemoryEpisodeBundleStore([bundle])
    assert store.get_view(bundle.key, 7).frame_mask.all()
    with pytest.raises(ValueError, match="Duplicate"):
        InMemoryEpisodeBundleStore([bundle, bundle])

    features = bundle.mv_cell_features.copy()
    mask = bundle.mv_cell_mask.copy()
    mask[2, 0, 0] = False
    with pytest.raises(ValueError, match="Masked MV cell features"):
        replace(bundle, mv_cell_features=features, mv_cell_mask=mask)

    with pytest.raises(ValueError, match="exactly N hashes"):
        replace(bundle, fixture_frame_sha256=bundle.fixture_frame_sha256[:-1])


def test_raw_audit_absence_and_writer_schema_fail_closed(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    _write(tmp_path, bundle)
    store = EpisodeBundleStore(tmp_path)
    with pytest.raises(KeyError, match="no raw MV audit"):
        store.audit_raw_motion_vectors(bundle.key)

    metadata = bundle.metadata_dict()
    metadata["unexpected"] = True
    with pytest.raises(ValueError, match="metadata keys"):
        write_episode_bundle(
            tmp_path / "bad",
            episode_uid=bundle.key.episode_uid,
            camera_id=bundle.key.camera_id,
            source_kind=bundle.key.source_kind,
            metadata=metadata,
            arrays=bundle.arrays_dict(),
        )

    with pytest.raises(ValueError, match="provided together"):
        write_episode_bundle(
            tmp_path / "bad-raw",
            episode_uid=bundle.key.episode_uid,
            camera_id=bundle.key.camera_id,
            source_kind=bundle.key.source_kind,
            metadata=bundle.metadata_dict(),
            arrays=bundle.arrays_dict(),
            raw_mv_offsets=np.zeros(bundle.num_frames + 1, dtype=np.int64),
        )
