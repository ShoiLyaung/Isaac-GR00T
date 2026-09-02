from __future__ import annotations

from dataclasses import replace

from gr00t.data.codec_evidence import episode_bundle as episode_bundle_module
from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    SYNTHETIC_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
from gr00t.data.codec_evidence.contract import REASON_CODES
from gr00t.data.codec_evidence.episode_bundle import (
    FRAME_TYPE_IDR,
    K_MAX,
    MAX_EPISODE_BATCH_SIZE,
    METADATA_WIDTH,
    MV_CELL_WIDTH,
    SOURCE_KIND_NATIVE_CANONICAL,
    InMemoryEpisodeBundleStore,
    assert_episode_batch_conforms,
    collate_episode_views_numpy,
    make_synthetic_episode_bundle,
    validate_episode_view,
)
import numpy as np
import pytest


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


def test_all_invalid_with_none_reason_is_rejected_by_bundle_constructor():
    bundle = make_synthetic_episode_bundle(num_frames=8)
    with pytest.raises(ValueError, match="validity priority.*PACKET_INVALID"):
        replace(bundle, frame_validity=np.zeros_like(bundle.frame_validity))


def test_reason_priority_is_accepted_and_rechecked_on_episode_view():
    bundle = make_synthetic_episode_bundle(num_frames=8)
    validity = bundle.frame_validity.copy()
    # Packet precedes reference in the frozen record-v2 priority.
    validity[2, 1] = False
    validity[2, 2] = False
    reasons = bundle.frame_reason_code.copy()
    reasons[2] = REASON_CODES["PACKET_INVALID"]
    accepted = replace(bundle, frame_validity=validity, frame_reason_code=reasons)
    view = accepted.view(7)
    validate_episode_view(view)

    wrong_view_reasons = view.frame_reason_code.copy()
    wrong_view_reasons[2] = REASON_CODES["NONE"]
    with pytest.raises(ValueError, match="validity priority.*PACKET_INVALID"):
        validate_episode_view(replace(view, frame_reason_code=wrong_view_reasons))


def test_missing_anchor_reason_is_explicit_and_wrong_anchor_reason_is_rejected():
    bundle = make_synthetic_episode_bundle(num_frames=8)
    missing_reason = np.full_like(bundle.frame_reason_code, REASON_CODES["I_FRAME_UNAVAILABLE"])
    missing = replace(
        bundle,
        anchor_index_per_frame=np.full_like(bundle.anchor_index_per_frame, -1),
        anchor_frame_index=np.empty((0,), dtype=np.int64),
        anchor_appearance=np.empty((0, 3, 256, 256), dtype=np.float32),
        frame_reason_code=missing_reason,
    )
    view = missing.view(7)
    assert not view.anchor_mask.any()
    assert np.all(view.frame_reason_code == REASON_CODES["I_FRAME_UNAVAILABLE"])
    validate_episode_view(view)

    with pytest.raises(ValueError, match="expected I_FRAME_UNAVAILABLE"):
        replace(
            bundle,
            anchor_index_per_frame=np.full_like(bundle.anchor_index_per_frame, -1),
            anchor_frame_index=np.empty((0,), dtype=np.int64),
            anchor_appearance=np.empty((0, 3, 256, 256), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="expected NONE"):
        replace(
            bundle,
            frame_reason_code=np.full_like(
                bundle.frame_reason_code, REASON_CODES["I_FRAME_UNAVAILABLE"]
            ),
        )


def test_episode_timestamps_are_exact_row_times_1000():
    bundle = make_synthetic_episode_bundle(num_frames=8)
    pts = bundle.pts.copy()
    pts[-1] += 1
    with pytest.raises(ValueError, match="frame_index \\* 1000"):
        replace(bundle, pts=pts)

    dts = bundle.dts.copy()
    dts[-1] += 1
    with pytest.raises(ValueError, match="frame_index \\* 1000"):
        replace(bundle, dts=dts)


def test_production_contract_rejects_early_idr_intra_and_non_gop30():
    synthetic = make_synthetic_episode_bundle(num_frames=40)
    identity = dict(synthetic.tool_identity)
    production_contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
        gop_size=30,
        identity=identity,
    )
    production = replace(
        synthetic,
        codec_contract=production_contract,
        codec_contract_sha256=codec_contract_sha256(production_contract),
    )
    assert production.codec_contract["fixture_mode"] == PRODUCTION_FIXTURE_MODE

    early_type = production.frame_type.copy()
    early_reference = production.reference_index.copy()
    early_type[5] = FRAME_TYPE_IDR
    early_reference[5] = -1
    with pytest.raises(ValueError, match="IDR iff row%30"):
        replace(production, frame_type=early_type, reference_index=early_reference)

    intra_type = production.frame_type.copy()
    intra_reference = production.reference_index.copy()
    intra_type[5] = 2
    intra_reference[5] = -1
    with pytest.raises(ValueError, match="IDR iff row%30"):
        replace(production, frame_type=intra_type, reference_index=intra_reference)

    gop_id = production.gop_id.copy()
    gop_id[5] = 1
    with pytest.raises(ValueError, match="IDR iff row%30"):
        replace(production, gop_id=gop_id)


def test_non_gop30_is_only_allowed_in_explicit_synthetic_mode():
    bundle = make_synthetic_episode_bundle(num_frames=8, gop_size=3)
    assert bundle.codec_contract["fixture_mode"] == SYNTHETIC_FIXTURE_MODE
    assert bundle.codec_contract["gop_size"] == 3
