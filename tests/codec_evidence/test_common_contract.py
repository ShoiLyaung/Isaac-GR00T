from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from gr00t.data.codec_evidence.config import parse_codec_evidence_config
from gr00t.data.codec_evidence.contract import (
    MAX_CODEC_BATCH_SIZE,
    MAX_PADDED_MV_ELEMENTS,
    CodecEvidenceKey,
    assert_batch_conforms,
    collate_codec_evidence_numpy,
    preflight_codec_batch_allocation,
)
from gr00t.data.codec_evidence.processor import CodecEvidenceProcessorCore
from gr00t.data.codec_evidence.store import (
    FixtureManifestCodecEvidenceStore,
    InMemoryCodecEvidenceStore,
)
import numpy as np
import pytest


def test_disabled_config_does_not_touch_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled config touched filesystem")

    monkeypatch.setattr(Path, "exists", forbidden)
    config = parse_codec_evidence_config({"enabled": False, "fixture_root": "/does/not/exist"})
    assert config.enabled is False
    assert config.fixture_root is None


def test_enabled_config_requires_pinned_absolute_fixture():
    with pytest.raises(ValueError, match="fixture_root"):
        parse_codec_evidence_config({"enabled": True})
    with pytest.raises(ValueError, match="absolute"):
        parse_codec_evidence_config(
            {
                "enabled": True,
                "fixture_root": "relative",
                "fixture_manifest_sha256": "a" * 64,
                "record_schema_sha256": "b" * 64,
            }
        )


def test_episode_bundle_config_requires_explicit_source_and_only_root_pin():
    value = {
        "enabled": True,
        "fixture_root": "/does/not/need/to/exist/during-parse",
        "fixture_manifest_sha256": "a" * 64,
        "schema_version": "codec-evidence-episode-bundle-v1",
    }
    with pytest.raises(ValueError, match="requires source_kind"):
        parse_codec_evidence_config(value)
    value["source_kind"] = "RAW_JPEG_DERIVED"
    config = parse_codec_evidence_config(value)
    assert config.schema_version == "codec-evidence-episode-bundle-v1"
    assert config.source_kind == "RAW_JPEG_DERIVED"
    assert config.record_schema_sha256 is None


def test_processor_collation_and_store(synthetic_codec_record):
    processor = CodecEvidenceProcessorCore()
    store = InMemoryCodecEvidenceStore([synthetic_codec_record])
    sample = processor.process(store.get(synthetic_codec_record.key))
    batch = collate_codec_evidence_numpy([sample, sample])
    assert_batch_conforms(batch)
    assert batch["frame_type"].shape == (2, 8)
    assert batch["motion_vector_fields"]["flags"].dtype == np.uint64
    assert batch["mv_mask"].shape == (2, 8, 2)
    assert batch["mv_mask"].all()
    assert batch["i_frame_appearance"].shape == (2, 3, 256, 256)


def test_processor_rejects_future_and_noncausal_reference(synthetic_codec_record):
    processor = CodecEvidenceProcessorCore()
    payload = deepcopy(synthetic_codec_record.payload)
    payload["history"]["frames"][7]["frame_index"] = 8
    future = replace(synthetic_codec_record, payload=payload)
    with pytest.raises(ValueError, match="future"):
        processor.process(future)

    payload = deepcopy(synthetic_codec_record.payload)
    payload["history"]["frames"][7]["reference_index"] = 0
    wrong_reference = replace(synthetic_codec_record, payload=payload)
    with pytest.raises(ValueError, match="immediately preceding"):
        processor.process(wrong_reference)


def test_processor_rejects_time_gop_and_reason_drift(synthetic_codec_record):
    processor = CodecEvidenceProcessorCore()
    payload = deepcopy(synthetic_codec_record.payload)
    payload["history"]["frames"][7]["pts"] = payload["history"]["frames"][6]["pts"]
    with pytest.raises(ValueError, match="PTS and DTS"):
        processor.process(replace(synthetic_codec_record, payload=payload))

    payload = deepcopy(synthetic_codec_record.payload)
    payload["history"]["frames"][7]["gop_position"] = 12
    with pytest.raises(ValueError, match="GOP identity"):
        processor.process(replace(synthetic_codec_record, payload=payload))

    payload = deepcopy(synthetic_codec_record.payload)
    payload["history"]["frames"][7]["validity"]["rpe"] = False
    payload["sample_valid"] = False
    payload["sample_reason_code"] = "RPE_INVALID"
    with pytest.raises(ValueError, match="Frame reason code"):
        processor.process(replace(synthetic_codec_record, payload=payload))


def test_left_padding_and_gop_boundary_are_valid(synthetic_codec_record):
    processor = CodecEvidenceProcessorCore()
    payload = deepcopy(synthetic_codec_record.payload)
    for slot in range(4):
        frame = payload["history"]["frames"][slot]
        frame.update(
            {
                "frame_index": -1,
                "pts": -1,
                "dts": -1,
                "frame_type": "PAD",
                "gop_id": -1,
                "gop_position": -1,
                "reference_index": -1,
                "reason_code": "LEFT_PAD",
                "source_sha256": None,
                "decoded_sha256": None,
                "mv_offset_begin": 0,
                "mv_offset_end": 0,
                "mv_count": 0,
            }
        )
        frame["validity"] = {name: False for name in frame["validity"]}
    for slot in range(4, 8):
        frame = payload["history"]["frames"][slot]
        frame_index = slot - 4
        frame.update(
            {
                "frame_index": frame_index,
                "pts": frame_index * 1000,
                "dts": frame_index * 1000,
                "frame_type": "I" if frame_index == 0 else "P",
                "gop_id": 0,
                "gop_position": frame_index,
                "reference_index": -1 if frame_index == 0 else frame_index - 1,
            }
        )
        frame["mv_offset_begin"] = (slot - 4) * 2
        frame["mv_offset_end"] = (slot - 3) * 2
        frame["mv_count"] = 2
    offsets = np.asarray([0, 0, 0, 0, 0, 2, 4, 6, 8], dtype=np.int64)
    fields = {name: values[8:].copy() for name, values in synthetic_codec_record.mv_fields.items()}
    padded_key = replace(synthetic_codec_record.key, observation_row=3)
    payload["observation_row"] = 3
    payload["i_frame"]["frame_index"] = 0
    padded = replace(
        synthetic_codec_record,
        key=padded_key,
        payload=payload,
        mv_offsets=offsets,
        mv_fields=fields,
        rpe=np.concatenate(
            [
                np.zeros((4, 2, 2, 4), dtype=np.float32),
                synthetic_codec_record.rpe[4:],
            ]
        ),
        camera_motion_summary=np.concatenate(
            [
                np.zeros((4, 4), dtype=np.float32),
                synthetic_codec_record.camera_motion_summary[4:],
            ]
        ),
    )
    sample = processor.process(padded)
    batch = collate_codec_evidence_numpy([sample])
    assert not batch["frame_validity"]["frame_valid"][0, :4].any()
    assert not batch["mv_mask"][0, :4].any()
    assert not batch["rpe"][0, :4].any()
    assert not batch["camera_motion_summary"][0, :4].any()

    payload = deepcopy(synthetic_codec_record.payload)
    boundary_key = replace(synthetic_codec_record.key, observation_row=32)
    payload["observation_row"] = 32
    for slot, frame_index in enumerate(range(25, 33)):
        frame = payload["history"]["frames"][slot]
        gop_id, gop_position = divmod(frame_index, 30)
        frame.update(
            {
                "frame_index": frame_index,
                "pts": frame_index * 1000,
                "dts": frame_index * 1000,
                "frame_type": "I" if gop_position == 0 else "P",
                "gop_id": gop_id,
                "gop_position": gop_position,
                "reference_index": -1 if gop_position == 0 else frame_index - 1,
            }
        )
    payload["i_frame"]["frame_index"] = 30
    processor.process(replace(synthetic_codec_record, key=boundary_key, payload=payload))


def test_padding_and_i_frame_identity_fail_closed(synthetic_codec_record):
    processor = CodecEvidenceProcessorCore()
    payload = deepcopy(synthetic_codec_record.payload)
    frame = payload["history"]["frames"][0]
    frame.update(
        {
            "frame_index": -1,
            "pts": -1,
            "dts": -1,
            "frame_type": "PAD",
            "gop_id": -1,
            "gop_position": -1,
            "reference_index": -1,
            "reason_code": "LEFT_PAD",
            "mv_offset_begin": 0,
            "mv_offset_end": 2,
            "mv_count": 2,
        }
    )
    frame["validity"] = {name: False for name in frame["validity"]}
    with pytest.raises(ValueError, match="null source and decoded hashes"):
        processor.process(replace(synthetic_codec_record, payload=payload))

    frame["source_sha256"] = None
    frame["decoded_sha256"] = None
    with pytest.raises(ValueError, match="zero MV descriptors"):
        processor.process(replace(synthetic_codec_record, payload=payload))

    payload = deepcopy(synthetic_codec_record.payload)
    payload["i_frame"]["frame_index"] = 7
    with pytest.raises(ValueError, match="nearest legal past GOP30 I-frame"):
        processor.process(replace(synthetic_codec_record, payload=payload))

    payload = deepcopy(synthetic_codec_record.payload)
    payload["i_frame"]["source_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="hashes disagree"):
        processor.process(replace(synthetic_codec_record, payload=payload))


def test_padding_numeric_evidence_must_be_zero(synthetic_codec_record):
    processor = CodecEvidenceProcessorCore()
    payload = deepcopy(synthetic_codec_record.payload)
    frame = payload["history"]["frames"][0]
    frame.update(
        {
            "frame_index": -1,
            "pts": -1,
            "dts": -1,
            "frame_type": "PAD",
            "gop_id": -1,
            "gop_position": -1,
            "reference_index": -1,
            "reason_code": "LEFT_PAD",
            "source_sha256": None,
            "decoded_sha256": None,
            "mv_offset_begin": 0,
            "mv_offset_end": 0,
            "mv_count": 0,
        }
    )
    frame["validity"] = {name: False for name in frame["validity"]}
    for nonpad in payload["history"]["frames"][1:]:
        nonpad["mv_offset_begin"] -= 2
        nonpad["mv_offset_end"] -= 2
    offsets = synthetic_codec_record.mv_offsets - 2
    offsets[0] = 0
    fields = {name: values[2:].copy() for name, values in synthetic_codec_record.mv_fields.items()}
    record = replace(
        synthetic_codec_record,
        payload=payload,
        mv_offsets=offsets,
        mv_fields=fields,
        rpe=synthetic_codec_record.rpe.copy(),
        camera_motion_summary=synthetic_codec_record.camera_motion_summary.copy(),
    )
    with pytest.raises(ValueError, match="zero RPE"):
        processor.process(record)
    record.rpe[0] = 0
    with pytest.raises(ValueError, match="zero camera motion"):
        processor.process(record)


def test_record_v2_rejects_outside_t8_anchor_without_episode_closure(
    synthetic_codec_record,
):
    payload = deepcopy(synthetic_codec_record.payload)
    key = replace(synthetic_codec_record.key, observation_row=14)
    payload["observation_row"] = 14
    for slot, frame_index in enumerate(range(7, 15)):
        frame = payload["history"]["frames"][slot]
        frame.update(
            {
                "frame_index": frame_index,
                "pts": frame_index * 1000,
                "dts": frame_index * 1000,
                "frame_type": "P",
                "gop_id": 0,
                "gop_position": frame_index,
                "reference_index": frame_index - 1,
            }
        )
    with pytest.raises(ValueError, match="outside T8"):
        CodecEvidenceProcessorCore().process(
            replace(synthetic_codec_record, key=key, payload=payload)
        )


def test_zero_motion_vectors_produce_exact_m_batch_zero(synthetic_codec_record):
    payload = deepcopy(synthetic_codec_record.payload)
    for frame in payload["history"]["frames"]:
        frame["mv_offset_begin"] = 0
        frame["mv_offset_end"] = 0
        frame["mv_count"] = 0
    empty = replace(
        synthetic_codec_record,
        payload=payload,
        mv_offsets=np.zeros(9, dtype=np.int64),
        mv_fields={
            name: np.empty((0,), dtype=value.dtype)
            for name, value in synthetic_codec_record.mv_fields.items()
        },
    )
    sample = CodecEvidenceProcessorCore().process(empty)
    batch = collate_codec_evidence_numpy([sample])
    assert batch["mv_mask"].shape == (1, 8, 0)
    assert batch["mv_count"].sum() == 0


def test_legacy_ragged_batch_is_preflight_bounded_before_allocation():
    assert preflight_codec_batch_allocation(1, 0) > 0
    with pytest.raises(ValueError, match="batch_size"):
        preflight_codec_batch_allocation(MAX_CODEC_BATCH_SIZE + 1, 0)
    minimum_count_over_limit = MAX_PADDED_MV_ELEMENTS // (14 * 8) + 1
    with pytest.raises(ValueError, match="padded MV elements"):
        preflight_codec_batch_allocation(14, minimum_count_over_limit)


def test_store_rejects_duplicate_and_missing(synthetic_codec_record):
    with pytest.raises(ValueError, match="Duplicate"):
        InMemoryCodecEvidenceStore([synthetic_codec_record, synthetic_codec_record])
    store = InMemoryCodecEvidenceStore([synthetic_codec_record])
    with pytest.raises(KeyError, match="Missing"):
        store.get(CodecEvidenceKey("d" * 64, 7, "cam_left_high"))


def test_framework_neutral_modules_do_not_import_torch_jax_or_gr00t():
    root = Path(__file__).parents[2] / "gr00t/data/codec_evidence"
    for name in ("config.py", "contract.py", "processor.py", "store.py", "dataset.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "import torch" not in source
        assert "import jax" not in source
        assert "from gr00t." not in source


def _artifact(root: Path, relative: str, value: np.ndarray) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    data = path.read_bytes()
    shape = ["N_total"] if value.ndim == 1 and value.shape != (9,) else list(value.shape)
    return {
        "relative_path": relative,
        "sha256": sha256(data).hexdigest(),
        "bytes": len(data),
        "dtype": str(value.dtype),
        "shape": shape,
    }


def test_fixture_manifest_has_non_circular_hash_closure(tmp_path, synthetic_codec_record):
    record_schema_source = Path(
        "/home/sl/agent-team-workspaces/icra27-control/schemas/codec-evidence-record-v2.schema.json"
    )
    schema_bytes = record_schema_source.read_bytes()
    schema_path = tmp_path / "record_schema.json"
    schema_path.write_bytes(schema_bytes)
    schema_sha = sha256(schema_bytes).hexdigest()
    encoder = {
        "container": "mp4",
        "codec": "h264",
        "encoder": "libx264",
        "encoder_binary_sha256": "d" * 64,
        "encoder_build_identity": "synthetic-test",
        "extractor_identity": "synthetic-test",
        "extractor_sha256": "e" * 64,
        "bframes": 0,
        "references": 1,
        "gop": 30,
        "scene_cut": 0,
        "open_gop": 0,
        "time_base_num": 1,
        "time_base_den": 30000,
    }
    binding_payload = {
        "fixture_id": "synthetic-conformance",
        "record_schema_sha256": schema_sha,
        "encoder_extractor_identity": encoder,
    }
    binding = sha256(
        json.dumps(
            binding_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    offsets = _artifact(tmp_path, "arrays/offsets.npy", synthetic_codec_record.mv_offsets)
    fields = {
        name: _artifact(tmp_path, f"arrays/{name}.npy", value)
        for name, value in synthetic_codec_record.mv_fields.items()
    }
    rpe = _artifact(tmp_path, "arrays/rpe.npy", synthetic_codec_record.rpe)
    appearance = _artifact(
        tmp_path, "arrays/appearance.npy", synthetic_codec_record.i_frame_appearance
    )
    camera = _artifact(tmp_path, "arrays/camera.npy", synthetic_codec_record.camera_motion_summary)
    payload = deepcopy(synthetic_codec_record.payload)
    payload.update(
        {
            "synthetic_only": True,
            "fixture_manifest_sha256": binding,
            "fixture_schema_sha256": schema_sha,
            "encoder_extractor_identity": encoder,
            "view_claim": "HEAD_FRONT_EGOCENTRIC_CANDIDATE_VIEW_NOT_CALIBRATION_PROVEN",
            "terminology": {
                "motion_vectors": "CODEC_ENCODER_DECISIONS_NOT_OPTICAL_FLOW",
                "rpe": "DECODED_LUMA_MV_WARP_PROXY_NOT_BITSTREAM_RESIDUAL",
                "row_grid": "SYNTHETIC_30_HZ_REPRESENTATION_NOT_PROVEN_HARDWARE_CAPTURE_CLOCK",
            },
            "mv_storage": {
                "layout": "CSR_RAGGED_PER_FRAME_NO_TRUNCATION",
                "pickle_allowed": False,
                "max_per_frame": 19200,
                "offsets": offsets,
                "fields": fields,
            },
            "rpe": rpe,
            "camera_motion_summary": camera,
        }
    )
    payload["history"]["padding"] = (
        "LEFT_ZERO_NEGATIVE_ONE_TEMPORAL_IDS_ALL_VALIDITY_FALSE_NO_REPEAT"
    )
    payload["i_frame"].update(
        {
            "selection": "NEAREST_VERIFIED_PAST_IDR_AT_OR_BEFORE_OBSERVATION",
            "appearance": appearance,
        }
    )
    record_path = tmp_path / "records/record.json"
    record_path.parent.mkdir()
    record_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    record_bytes = record_path.read_bytes()
    manifest = {
        "schema_version": "codec-evidence-fixture-manifest-v1",
        "fixture_id": "synthetic-conformance",
        "identity": {
            "record_schema_sha256": schema_sha,
            "encoder_extractor_identity": encoder,
        },
        "fixture_binding_sha256": binding,
        "record_schema": {
            "relative_path": "record_schema.json",
            "sha256": schema_sha,
            "bytes": len(schema_bytes),
            "dtype": "json",
            "shape": [],
        },
        "records": [
            {
                "episode_uid": synthetic_codec_record.key.episode_uid,
                "observation_row": synthetic_codec_record.key.observation_row,
                "camera_id": synthetic_codec_record.key.camera_id,
                "record": {
                    "relative_path": "records/record.json",
                    "sha256": sha256(record_bytes).hexdigest(),
                    "bytes": len(record_bytes),
                    "dtype": "json",
                    "shape": [],
                },
            }
        ],
    }
    manifest_path = tmp_path / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha = sha256(manifest_path.read_bytes()).hexdigest()
    store = FixtureManifestCodecEvidenceStore(
        tmp_path,
        fixture_manifest_sha256=manifest_sha,
        record_schema_sha256=schema_sha,
    )
    loaded = store.get(synthetic_codec_record.key)
    assert loaded.key == synthetic_codec_record.key
    assert np.array_equal(loaded.mv_offsets, synthetic_codec_record.mv_offsets)
    record_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Record (bytes|hash) mismatch"):
        store.get(synthetic_codec_record.key)
