from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from gr00t.data.codec_evidence import schemas as codec_schemas
from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    SYNTHETIC_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
from gr00t.data.codec_evidence.config import parse_codec_evidence_config
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES, REASON_CODES
from gr00t.data.codec_evidence.dataset import (
    EPISODE_BUNDLE_INDEX_SCHEMA_VERSION,
    CodecEvidenceAugmentedDataset,
    episode_bundle_store_from_index_manifest,
    maybe_wrap_codec_evidence_dataset,
)
from gr00t.data.codec_evidence.episode_bundle import (
    SOURCE_KIND_NATIVE_CANONICAL,
    EpisodeBundleKey,
    InMemoryEpisodeBundleStore,
    make_neutral_episode_view,
    make_synthetic_episode_bundle,
    write_episode_bundle,
)
from gr00t.data.codec_evidence.identity import EPISODE_IDENTITY_SCHEMA_VERSION
from gr00t.data.codec_evidence.schemas import schema_path
from gr00t.data.codec_evidence.torch_backend import (
    EPISODE_AUDIT_IDENTITY_KEYS,
    EPISODE_MODEL_TENSOR_KEYS,
    TorchCodecEvidenceCollator,
)
import numpy as np
import pytest
import torch


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _validate_jsonschema_cases(
    tmp_path: Path,
    schemas: dict[str, object],
    cases: list[tuple[str, object]],
) -> subprocess.CompletedProcess[str]:
    validation_input = tmp_path / f"schema-validation-{len(list(tmp_path.iterdir()))}.json"
    _write_json(validation_input, {"schemas": schemas, "cases": cases})
    candidate_interpreters = tuple(
        dict.fromkeys(
            candidate
            for candidate in (os.environ.get("CONDA_PYTHON_EXE"), sys.executable)
            if candidate
        )
    )
    jsonschema_python = None
    for candidate in candidate_interpreters:
        probe = subprocess.run(
            [candidate, "-c", "import jsonschema"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            jsonschema_python = candidate
            break
    if jsonschema_python is None:
        pytest.fail("No local Python interpreter provides jsonschema")
    script = """
import json
import sys
import jsonschema
value = json.load(open(sys.argv[1], encoding='utf-8'))
store = {schema['$id']: schema for schema in value['schemas'].values()}
for name, payload in value['cases']:
    schema = value['schemas'][name]
    resolver = jsonschema.RefResolver.from_schema(schema, store=store)
    jsonschema.Draft202012Validator(schema, resolver=resolver).validate(payload)
"""
    return subprocess.run(
        [jsonschema_python, "-c", script, str(validation_input)],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_bundle_index_many(root: Path, manifests: list[Path]) -> Path:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    first = payloads[0]
    index = {
        "schema_version": EPISODE_BUNDLE_INDEX_SCHEMA_VERSION,
        "complete": True,
        "camera_id": first["camera_id"],
        "source_kind": first["source_kind"],
        "identity": {
            "episode_identity_schema_version": EPISODE_IDENTITY_SCHEMA_VERSION,
            "codec_contract_sha256": first["codec_contract_sha256"],
        },
        "codec_contract": first["codec_contract"],
        "bundles": [],
    }
    for path, payload in zip(manifests, payloads, strict=True):
        index["bundles"].append(
            {
                "episode_uid": payload["episode_uid"],
                "camera_id": payload["camera_id"],
                "source_kind": payload["source_kind"],
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )
    path = root / "fixture_manifest.json"
    _write_json(path, index)
    return path


def _write_bundle_index(root: Path, bundle_manifest: Path, key: EpisodeBundleKey) -> Path:
    assert json.loads(bundle_manifest.read_text(encoding="utf-8"))["episode_uid"] == key.episode_uid
    return _write_bundle_index_many(root, [bundle_manifest])


class _Loader:
    def __init__(self, episode_uid: str):
        self.selected_episode_uids = (episode_uid,)

    def __getitem__(self, index: int) -> dict[str, int]:
        assert index == 0
        return {"episode": index}


class _BaseDataset:
    def __init__(self, episode_uid: str):
        self.episode_loader = _Loader(episode_uid)
        self.sharded_episodes = [[(0, np.asarray([14, 31], dtype=np.int64))]]

    def get_datapoint(self, episode_data: object, step_index: int) -> dict[str, object]:
        return {"native": (episode_data, step_index)}


def test_missing_and_disabled_configs_return_the_same_object(tmp_path, monkeypatch):
    dataset = object()
    assert maybe_wrap_codec_evidence_dataset(dataset, None) is dataset

    adapter = tmp_path / "adapter.json"
    _write_json(
        adapter,
        {
            "codec_evidence": {
                "enabled": False,
                "fixture_root": "/path/that/must/not/be-resolved",
            }
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled codec config attempted to construct a fixture store")

    monkeypatch.setattr(
        "gr00t.data.codec_evidence.dataset.episode_bundle_store_from_index_manifest",
        forbidden,
    )
    monkeypatch.setattr(
        "gr00t.data.codec_evidence.dataset.FixtureManifestCodecEvidenceStore",
        forbidden,
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    assert maybe_wrap_codec_evidence_dataset(dataset, adapter) is dataset
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_adapter_config_rejects_duplicate_json_keys(tmp_path):
    adapter = tmp_path / "adapter.json"
    adapter.write_text(
        '{"codec_evidence":{"enabled":false},"codec_evidence":{"enabled":false}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        maybe_wrap_codec_evidence_dataset(object(), adapter)


def test_episode_bundle_index_binds_out_of_t8_anchor_and_dataset_transport(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=40)
    bundle_manifest = write_episode_bundle(
        tmp_path,
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
    )
    index = _write_bundle_index(tmp_path, bundle_manifest, bundle.key)
    config = parse_codec_evidence_config(
        {
            "enabled": True,
            "fixture_root": str(tmp_path),
            "fixture_manifest_sha256": _sha256_path(index),
            "schema_version": "codec-evidence-episode-bundle-v1",
            "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        }
    )
    dataset = CodecEvidenceAugmentedDataset(_BaseDataset(bundle.key.episode_uid), config)
    datapoints = dataset.get_shard(0)
    assert [int(item["codec_evidence_sample"].observation_row) for item in datapoints] == [14, 31]

    row14 = datapoints[0]["codec_evidence_sample"]
    assert row14.frame_mask.all()
    # Row 14's nearest IDR is row 0, outside T8 rows 7..14.  The bundle
    # manifest hash and its array hashes bind the K2 appearance identity.
    assert row14.anchor_mask.tolist() == [True, False]
    assert row14.anchor_frame_index.tolist() == [0, 0]
    assert np.all(row14.frame_anchor_slot == 0)


def test_episode_bundle_index_rejects_wrong_bundle_hash(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    bundle_manifest = write_episode_bundle(
        tmp_path,
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
    )
    index = _write_bundle_index(tmp_path, bundle_manifest, bundle.key)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["bundles"][0]["sha256"] = "f" * 64
    _write_json(index, payload)
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        episode_bundle_store_from_index_manifest(
            tmp_path,
            fixture_manifest_sha256=_sha256_path(index),
            camera_id=bundle.key.camera_id,
            source_kind=bundle.key.source_kind,
        )


def test_index_store_keys_and_identity_are_immutable_without_array_load(tmp_path, monkeypatch):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    manifest = write_episode_bundle(
        tmp_path,
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
    )
    index = _write_bundle_index(tmp_path, manifest, bundle.key)
    store = episode_bundle_store_from_index_manifest(
        tmp_path,
        fixture_manifest_sha256=_sha256_path(index),
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
    )
    assert store.keys == frozenset({bundle.key})
    with pytest.raises(AttributeError):
        store.keys.add(bundle.key)

    def forbidden(*args, **kwargs):
        raise AssertionError("identity() loaded episode arrays")

    monkeypatch.setattr(store, "_load_array", forbidden)
    identity = store.identity(bundle.key)
    assert identity["codec_contract_sha256"] == bundle.codec_contract_sha256
    with pytest.raises(TypeError):
        identity["codec_contract_sha256"] = "0" * 64


def test_bundle_identity_relabel_attack_is_rejected_even_if_index_is_rehashed(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    manifest = write_episode_bundle(
        tmp_path,
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
    )
    manifest.chmod(0o644)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["episode_identity"]["source_id"] = "relabelled-source"
    _write_json(manifest, payload)
    index = _write_bundle_index(tmp_path, manifest, bundle.key)
    with pytest.raises(ValueError, match="preimage does not derive"):
        episode_bundle_store_from_index_manifest(
            tmp_path,
            fixture_manifest_sha256=_sha256_path(index),
            camera_id=bundle.key.camera_id,
            source_kind=bundle.key.source_kind,
        )


def test_index_rejects_cross_bundle_global_contract_drift(tmp_path):
    first = make_synthetic_episode_bundle(num_frames=8)
    second = make_synthetic_episode_bundle(num_frames=8, episode_uid="b" * 64)
    different_identity = {"name": "synthetic-alt", "version": "1", "sha256": "9" * 64}
    second_contract = make_codec_contract(
        fixture_mode=SYNTHETIC_FIXTURE_MODE,
        source_kind=second.key.source_kind,
        gop_size=second.gop_size,
        identity=different_identity,
    )
    second = replace(
        second,
        tool_identity=different_identity,
        encoder_identity=different_identity,
        decoder_identity=different_identity,
        extractor_identity=different_identity,
        codec_contract=second_contract,
        codec_contract_sha256=codec_contract_sha256(second_contract),
    )
    manifests = []
    for bundle in (first, second):
        manifests.append(
            write_episode_bundle(
                tmp_path,
                episode_uid=bundle.key.episode_uid,
                camera_id=bundle.key.camera_id,
                source_kind=bundle.key.source_kind,
                metadata=bundle.metadata_dict(),
                arrays=bundle.arrays_dict(),
            )
        )
    index = _write_bundle_index_many(tmp_path, manifests)
    with pytest.raises(ValueError, match="cross-bundle codec contract drift"):
        episode_bundle_store_from_index_manifest(
            tmp_path,
            fixture_manifest_sha256=_sha256_path(index),
            camera_id=first.key.camera_id,
            source_kind=first.key.source_kind,
        )


def test_episode_bundle_in_memory_enabled_path_and_torch_transport():
    bundle = make_synthetic_episode_bundle(num_frames=40)
    config = parse_codec_evidence_config(
        {
            "enabled": True,
            "fixture_root": "/synthetic/not-read/when-store-injected",
            "fixture_manifest_sha256": "a" * 64,
            "schema_version": "codec-evidence-episode-bundle-v1",
            "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        }
    )
    wrapped = CodecEvidenceAugmentedDataset(
        _BaseDataset(bundle.key.episode_uid),
        config,
        store=InMemoryEpisodeBundleStore([bundle]),
    )
    views = [item["codec_evidence_sample"] for item in wrapped.get_shard(0)]
    torch_batch = TorchCodecEvidenceCollator()(views)
    assert set(torch_batch) == EPISODE_MODEL_TENSOR_KEYS
    assert not EPISODE_AUDIT_IDENTITY_KEYS.intersection(torch_batch)
    assert torch_batch["mv_cell_features"].shape == (2, 8, 2, 2, 11)
    assert torch_batch["anchor_appearance"].shape == (2, 2, 3, 256, 256)
    assert all(isinstance(value, torch.Tensor) for value in torch_batch.values())


def test_episode_bundle_all_invalid_has_no_active_model_frame():
    bundle = make_synthetic_episode_bundle(num_frames=40)
    invalid = replace(
        bundle,
        frame_validity=np.zeros_like(bundle.frame_validity),
        frame_reason_code=np.full_like(bundle.frame_reason_code, REASON_CODES["PACKET_INVALID"]),
        mv_cell_features=np.zeros_like(bundle.mv_cell_features),
        mv_cell_mask=np.zeros_like(bundle.mv_cell_mask),
        rpe=np.zeros_like(bundle.rpe),
        rpe_mask=np.zeros_like(bundle.rpe_mask),
        camera_motion_summary=np.zeros_like(bundle.camera_motion_summary),
        camera_motion_mask=np.zeros_like(bundle.camera_motion_mask),
    )
    view = invalid.view(7)
    batch = TorchCodecEvidenceCollator()([view])
    assert batch["frame_mask"].all()
    assert not batch["frame_validity"][..., 0].any()

    config = parse_codec_evidence_config(
        {
            "enabled": True,
            "fixture_root": "/synthetic/not-read/when-store-injected",
            "fixture_manifest_sha256": "a" * 64,
            "schema_version": "codec-evidence-episode-bundle-v1",
            "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        }
    )
    wrapped = CodecEvidenceAugmentedDataset(
        _BaseDataset(invalid.key.episode_uid),
        config,
        store=InMemoryEpisodeBundleStore([invalid]),
    )
    with pytest.raises(ValueError, match="Enabled training Codec evidence must be valid"):
        wrapped.get_shard(0)


@pytest.mark.parametrize("reason_name", ["MISSING_RECORD", "SCHEMA_INVALID", "HASH_MISMATCH"])
def test_episode_transport_neutral_reasons_are_zero_and_fail_closed_on_drift(reason_name):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    view = make_neutral_episode_view(bundle.key, 7, REASON_CODES[reason_name])
    batch = TorchCodecEvidenceCollator()([view])
    assert not batch["sample_valid"].item()
    assert batch["sample_reason_code"].item() == REASON_CODES[reason_name]
    assert not batch["frame_mask"].any()
    assert not batch["mv_cell_features"].any()

    nonzero = view.mv_cell_features.copy()
    nonzero[0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="Transport-neutral episode evidence"):
        TorchCodecEvidenceCollator()([replace(view, mv_cell_features=nonzero)])
    mask = view.frame_mask.copy()
    mask[0] = True
    with pytest.raises(ValueError, match="Transport-neutral episode evidence"):
        TorchCodecEvidenceCollator()([replace(view, frame_mask=mask)])


def test_episode_transport_neutral_rejects_negative_observation_row():
    bundle = make_synthetic_episode_bundle(num_frames=8)
    with pytest.raises(ValueError, match="observation_row"):
        make_neutral_episode_view(bundle.key, -1, REASON_CODES["MISSING_RECORD"])


def test_branch_owned_schemas_are_local_json_documents():
    names = (
        "codec-evidence-record-v2.schema.json",
        "codec-evidence-fixture-manifest-v1.schema.json",
        "codec-evidence-batch-v1.schema.json",
        "codec-evidence-episode-batch-v1.schema.json",
        "codec-evidence-episode-identity-v1.schema.json",
        "codec-evidence-global-contract-v2.schema.json",
        "codec-evidence-episode-bundle-v1.schema.json",
        "codec-evidence-episode-bundle-index-v1.schema.json",
        "codec-evidence-full-fixture-manifest-v1.schema.json",
    )
    package_schema_dir = Path(codec_schemas.__file__).resolve().parent
    for name in names:
        path = schema_path(name)
        assert path.resolve().parent == package_schema_dir
        assert path.is_file()
        assert not path.is_symlink()
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["$schema"].startswith("https://json-schema.org/")
        if name == "codec-evidence-episode-batch-v1.schema.json":
            padding = value["properties"]["padding_policy"]["const"]
            assert padding["ordinary_numeric_and_frame_index"] == 0
            assert padding["frame_anchor_slot"] == -1
            assert padding["frame_mask"] is False


def test_synthetic_bundle_and_index_validate_against_branch_owned_schemas(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    manifest_path = write_episode_bundle(
        tmp_path,
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
    )
    index_path = _write_bundle_index(tmp_path, manifest_path, bundle.key)
    names = (
        "codec-evidence-episode-identity-v1.schema.json",
        "codec-evidence-global-contract-v2.schema.json",
        "codec-evidence-episode-bundle-v1.schema.json",
        "codec-evidence-episode-bundle-index-v1.schema.json",
    )
    schemas = {name: json.loads(schema_path(name).read_text(encoding="utf-8")) for name in names}
    bundle_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    cases = [
        ("codec-evidence-episode-identity-v1.schema.json", bundle_payload["episode_identity"]),
        ("codec-evidence-global-contract-v2.schema.json", bundle_payload["codec_contract"]),
        ("codec-evidence-episode-bundle-v1.schema.json", bundle_payload),
        ("codec-evidence-episode-bundle-index-v1.schema.json", index_payload),
    ]
    completed = _validate_jsonschema_cases(tmp_path, schemas, cases)
    assert completed.returncode == 0, completed.stderr


def test_bundle_schema_closes_accepted_bitstream_and_exact_raw_mv_fields(tmp_path):
    bundle = make_synthetic_episode_bundle(num_frames=8)
    offsets = np.arange(9, dtype=np.int64)
    fields = {name: np.zeros((8,), dtype=dtype) for name, dtype in MV_FIELD_DTYPES.items()}
    fields["block_width"][:] = 16
    fields["block_height"][:] = 16
    fields["motion_scale"][:] = 1
    bitstream = tmp_path / "accepted.mp4"
    bitstream.write_bytes(b"non-empty-production-shape-probe")
    manifest_path = write_episode_bundle(
        tmp_path / "bundles",
        episode_uid=bundle.key.episode_uid,
        camera_id=bundle.key.camera_id,
        source_kind=bundle.key.source_kind,
        metadata=bundle.metadata_dict(),
        arrays=bundle.arrays_dict(),
        raw_mv_offsets=offsets,
        raw_mv_fields=fields,
        accepted_bitstream=bitstream,
    )
    names = (
        "codec-evidence-episode-identity-v1.schema.json",
        "codec-evidence-global-contract-v2.schema.json",
        "codec-evidence-episode-bundle-v1.schema.json",
    )
    schemas = {name: json.loads(schema_path(name).read_text(encoding="utf-8")) for name in names}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid = _validate_jsonschema_cases(
        tmp_path,
        schemas,
        [("codec-evidence-episode-bundle-v1.schema.json", payload)],
    )
    assert valid.returncode == 0, valid.stderr

    missing_field = json.loads(json.dumps(payload))
    missing_field["raw_motion_vector_audit"]["fields"].pop("motion_scale")
    invalid = _validate_jsonschema_cases(
        tmp_path,
        schemas,
        [("codec-evidence-episode-bundle-v1.schema.json", missing_field)],
    )
    assert invalid.returncode != 0

    wrong_bitstream = json.loads(json.dumps(payload))
    wrong_bitstream["accepted_bitstream"]["dtype"] = "uint8"
    invalid = _validate_jsonschema_cases(
        tmp_path,
        schemas,
        [("codec-evidence-episode-bundle-v1.schema.json", wrong_bitstream)],
    )
    assert invalid.returncode != 0

    missing_production_audit = json.loads(json.dumps(payload))
    missing_production_audit["codec_contract"]["fixture_mode"] = "PRODUCTION_NATIVE_CANONICAL"
    missing_production_audit["codec_contract"]["frame_type_rule"] = (
        "IDR_IFF_ROW_MOD_30_ZERO_ELSE_P_REF_ROW_MINUS_1"
    )
    missing_production_audit["accepted_bitstream"] = None
    missing_production_audit["raw_motion_vector_audit"] = None
    invalid = _validate_jsonschema_cases(
        tmp_path,
        schemas,
        [("codec-evidence-episode-bundle-v1.schema.json", missing_production_audit)],
    )
    assert invalid.returncode != 0


def test_full_fixture_schema_validates_instance_and_rejects_synthetic_contract(tmp_path):
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
    )
    uid = "a" * 64
    payload = {
        "schema_version": "codec-evidence-full-fixture-manifest-v1",
        "fixture_id": "full-r1",
        "complete": True,
        "coverage_kind": "FULL_EPISODE_SET",
        "camera_id": "cam_left_high",
        "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        "expected_episode_count": 1,
        "covered_episode_count": 1,
        "covered_episode_uids": [uid],
        "covered_episode_uid_digest": "b" * 64,
        "input_snapshot_sha256": "c" * 64,
        "budget_sha256": "d" * 64,
        "codec_contract": contract,
        "codec_contract_sha256": codec_contract_sha256(contract),
        "objects": [
            {
                "episode_uid": uid,
                "camera_id": "cam_left_high",
                "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
                "num_frames": 40,
                "derivation_sha256": "e" * 64,
                "object_content_sha256": "f" * 64,
                "bundle_manifest_sha256": "1" * 64,
                "bundle_manifest_bytes": 1,
            }
        ],
        "producer_identity": {"implementation": "focused"},
    }
    names = (
        "codec-evidence-global-contract-v2.schema.json",
        "codec-evidence-full-fixture-manifest-v1.schema.json",
    )
    schemas = {name: json.loads(schema_path(name).read_text()) for name in names}
    valid = _validate_jsonschema_cases(
        tmp_path,
        schemas,
        [("codec-evidence-full-fixture-manifest-v1.schema.json", payload)],
    )
    assert valid.returncode == 0, valid.stderr

    synthetic = make_codec_contract(
        fixture_mode=SYNTHETIC_FIXTURE_MODE,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
    )
    payload["codec_contract"] = synthetic
    payload["codec_contract_sha256"] = codec_contract_sha256(synthetic)
    invalid = _validate_jsonschema_cases(
        tmp_path,
        schemas,
        [("codec-evidence-full-fixture-manifest-v1.schema.json", payload)],
    )
    assert invalid.returncode != 0


def test_episode_bundle_config_is_global_and_hash_pinned():
    value = {
        "enabled": True,
        "fixture_root": "/tmp/fixture-does-not-need-to-exist-during-parse",
        "fixture_manifest_sha256": "a" * 64,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        "camera_id": "cam_left_high",
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "d_codec": 64,
        "conditioner_heads": 8,
    }
    assert parse_codec_evidence_config(value) == parse_codec_evidence_config(dict(value))
