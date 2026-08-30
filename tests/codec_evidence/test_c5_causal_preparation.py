from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/G1_ALL_LEROBOT_CODEC/prepare_c5_causal_codec_only.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_c5_causal_codec_only", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
causal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(causal)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture_manifest() -> dict:
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind="NATIVE_CANONICAL_DECODED_DERIVED",
    )
    uids = [_sha(f"episode-{index}") for index in range(causal.EXPECTED_EPISODES)]
    uids.sort()
    objects = []
    for index, uid in enumerate(uids):
        objects.append(
            {
                "episode_uid": uid,
                "camera_id": "cam_left_high",
                "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
                "num_frames": 1 if index else causal.EXPECTED_TOTAL_FRAMES - 805,
                "derivation_sha256": _sha(f"derivation-{index}"),
                "object_content_sha256": _sha(f"object-{index}"),
                "bundle_manifest_sha256": _sha(f"bundle-{index}"),
                "bundle_manifest_bytes": 1,
            }
        )
    uid_digest = causal._canonical_json_sha256(sorted(uids))
    return {
        "schema_version": "codec-evidence-full-fixture-manifest-v1",
        "fixture_id": causal.EXPECTED_FIXTURE_ID,
        "complete": True,
        "coverage_kind": "FULL_EPISODE_SET",
        "camera_id": "cam_left_high",
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "expected_episode_count": causal.EXPECTED_EPISODES,
        "covered_episode_count": causal.EXPECTED_EPISODES,
        "covered_episode_uids": uids,
        "covered_episode_uid_digest": uid_digest,
        "input_snapshot_sha256": causal.EXPECTED_INPUT_SNAPSHOT_SHA256,
        "budget_sha256": causal.EXPECTED_BUDGET_SHA256,
        "codec_contract": contract,
        "codec_contract_sha256": codec_contract_sha256(contract),
        "objects": objects,
        "producer_identity": {"placeholder": True},
    }


def _arrange_inputs(tmp_path: Path, monkeypatch):
    native_root = tmp_path / "native-preparation"
    fixture_root = tmp_path / "causal-fixture"
    entries = []
    for source_id in causal.SOURCE_IDS:
        source = native_root / "sources" / source_id
        view = source / "LOADER_VIEW"
        view.mkdir(parents=True)
        canonical = tmp_path / "canonical" / source_id
        (canonical / "data").mkdir(parents=True)
        (canonical / "videos").mkdir()
        (view / "data").symlink_to(canonical / "data")
        (view / "videos").symlink_to(canonical / "videos")
        adapter = source / "ADAPTER_CONFIG.json"
        _write_json(
            adapter,
            {
                "source_id": source_id,
                "baseline_id": "native",
                "canonical_root": str(canonical),
            },
        )
        entries.append(
            {
                "dataset_path": str(view),
                "mix_ratio": 0.2,
                "adapter_config_path": str(adapter),
            }
        )
    native_manifest = native_root / "DATASET_MANIFEST.json"
    native_closure = native_root / "CPU_MIXTURE_CLOSURE.json"
    fixture_manifest = fixture_root / "fixture_manifest.json"
    _write_json(
        native_manifest,
        {"schema_version": "gr00t-finetune-dataset-manifest-v1", "datasets": entries},
    )
    _write_json(native_closure, {"status": "PASS", "total_episodes": 806})
    fixture = _fixture_manifest()
    _write_json(fixture_manifest, fixture)

    monkeypatch.setattr(causal, "NATIVE_PREPARATION", native_root)
    monkeypatch.setattr(causal, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(causal, "NATIVE_MANIFEST_SHA256", causal.sha256_file(native_manifest))
    monkeypatch.setattr(causal, "NATIVE_CLOSURE_SHA256", causal.sha256_file(native_closure))
    monkeypatch.setattr(
        causal,
        "EXPECTED_EPISODE_UID_DIGEST",
        fixture["covered_episode_uid_digest"],
    )
    monkeypatch.setattr(
        causal,
        "NATIVE_ADAPTER_SHA256",
        {
            source_id: causal.sha256_file(
                native_root / "sources" / source_id / "ADAPTER_CONFIG.json"
            )
            for source_id in causal.SOURCE_IDS
        },
    )
    authority_calls = []

    def authoritative(root, *, fixture_manifest_sha256, verify_objects):
        authority_calls.append((root, fixture_manifest_sha256, verify_objects))
        return causal.load_json(fixture_manifest)

    monkeypatch.setattr(causal, "validate_full_fixture_manifest", authoritative)
    monkeypatch.setattr(
        causal,
        "validate_producer_and_proof",
        lambda _root, _fixture: {
            "input_snapshot_sha256": causal.EXPECTED_INPUT_SNAPSHOT_SHA256,
            "budget_sha256": causal.EXPECTED_BUDGET_SHA256,
            "producer_identity": {"test": True},
            "producer_attempt_plan_sha256": "1" * 64,
            "execution_plan_sha256": "2" * 64,
            "causal_prefix_equivalence_proof": {
                "path": str(causal.CAUSAL_PROOF_PATH),
                "sha256": causal.CAUSAL_PROOF_SHA256,
            },
        },
    )
    return entries, fixture_manifest, authority_calls


def test_causal_preparation_requires_explicit_manifest_sha_and_fixed_root():
    assert causal.FIXTURE_ROOT == Path(
        "/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r2"
    )
    with pytest.raises(SystemExit):
        causal._parser().parse_args(["--output-root", "/tmp/output"])
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        causal.codec_evidence_config("not-a-sha")


def test_causal_preparation_materializes_only_immutable_overlay(tmp_path, monkeypatch):
    entries, fixture_manifest, authority_calls = _arrange_inputs(tmp_path, monkeypatch)
    manifest_sha256 = causal.sha256_file(fixture_manifest)
    output = tmp_path / "causal-preparation"

    summary = causal.materialize(output, manifest_sha256)

    assert summary["schema_version"] == causal.PREPARATION_SUMMARY_SCHEMA_VERSION
    assert summary["input_fixture_manifest_sha256"] == manifest_sha256
    assert summary["plan"]["codec_contract"]["rc_lookahead"] == 0
    assert summary["plan"]["codec_contract"]["mbtree"] == 0
    assert summary["plan"]["codec_contract"]["min_keyint"] == 16
    assert authority_calls == [(causal.FIXTURE_ROOT, manifest_sha256, True)]
    manifest = causal.load_json(output / "DATASET_MANIFEST.json")
    assert [Path(entry["dataset_path"]) for entry in manifest["datasets"]] == [
        Path(entry["dataset_path"]).resolve() for entry in entries
    ]
    expected_evidence = causal.codec_evidence_config(manifest_sha256)
    for source_id, entry in zip(causal.SOURCE_IDS, manifest["datasets"]):
        adapter = causal.load_json(Path(entry["adapter_config_path"]))
        assert adapter["source_id"] == source_id
        assert adapter["codec_evidence"] == expected_evidence
        assert list((output / "sources" / source_id).iterdir()) == [
            output / "sources" / source_id / "ADAPTER_CONFIG.json"
        ]
    with pytest.raises(FileExistsError):
        causal.materialize(output, manifest_sha256)


def test_causal_preparation_rejects_contract_or_uid_drift(tmp_path, monkeypatch):
    _entries, fixture_manifest, _calls = _arrange_inputs(tmp_path, monkeypatch)
    fixture = causal.load_json(fixture_manifest)
    manifest_sha256 = causal.sha256_file(fixture_manifest)
    causal.validate_inputs(manifest_sha256)

    fixture["codec_contract"]["rc_lookahead"] = 1
    _write_json(fixture_manifest, fixture)
    drifted_sha256 = causal.sha256_file(fixture_manifest)
    with pytest.raises(RuntimeError, match="global codec contract"):
        causal.validate_inputs(drifted_sha256)

    fixture = _fixture_manifest()
    fixture["covered_episode_uids"][0] = _sha("wrong-episode")
    _write_json(fixture_manifest, fixture)
    drifted_sha256 = causal.sha256_file(fixture_manifest)
    with pytest.raises(RuntimeError, match="UID digest"):
        causal.validate_inputs(drifted_sha256)


def test_causal_preparation_propagates_authoritative_cas_rejection(tmp_path, monkeypatch):
    _entries, fixture_manifest, _calls = _arrange_inputs(tmp_path, monkeypatch)
    manifest_sha256 = causal.sha256_file(fixture_manifest)

    def reject_truncated_object(*_args, **_kwargs):
        raise ValueError("Full fixture object entry is malformed")

    monkeypatch.setattr(causal, "validate_full_fixture_manifest", reject_truncated_object)
    with pytest.raises(RuntimeError, match="authoritative causal full-fixture validation"):
        causal.validate_inputs(manifest_sha256)


def test_producer_and_proof_binding_rejects_identity_drift(tmp_path, monkeypatch):
    fixture_root = tmp_path / "fixture"
    fixture = _fixture_manifest()
    uid_digest = fixture["covered_episode_uid_digest"]
    monkeypatch.setattr(causal, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(causal, "EXPECTED_EPISODE_UID_DIGEST", uid_digest)
    proof_identity = {
        "path": str(causal.CAUSAL_PROOF_PATH),
        "bytes": 123,
        "sha256": causal.CAUSAL_PROOF_SHA256,
    }
    monkeypatch.setattr(causal, "validate_causal_proof", lambda: dict(proof_identity))
    lengths = sorted(
        (
            {"episode_uid": item["episode_uid"], "num_frames": item["num_frames"]}
            for item in fixture["objects"]
        ),
        key=lambda item: item["episode_uid"],
    )
    derivations = sorted(
        (
            {
                "episode_uid": item["episode_uid"],
                "derivation_sha256": item["derivation_sha256"],
            }
            for item in fixture["objects"]
        ),
        key=lambda item: item["episode_uid"],
    )
    plan = {
        "fixture_id": causal.EXPECTED_FIXTURE_ID,
        "output_root": str(fixture_root),
        "input_snapshot_sha256": causal.EXPECTED_INPUT_SNAPSHOT_SHA256,
        "budget_sha256": causal.EXPECTED_BUDGET_SHA256,
        "episode_freeze_core_sha256": causal.EXPECTED_EPISODE_FREEZE_CORE_SHA256,
        "expected_episode_count": causal.EXPECTED_EPISODES,
        "expected_total_frames": causal.EXPECTED_TOTAL_FRAMES,
        "expected_episode_uid_digest": uid_digest,
        "expected_episode_lengths_sha256": causal.canonical_json_sha256(lengths),
        "toolchain": {"git": {"head": "a" * 40}},
        "toolchain_identity_sha256": "b" * 64,
        "implementation_identity_sha256": "c" * 64,
        "codec_contract": fixture["codec_contract"],
        "codec_contract_sha256": fixture["codec_contract_sha256"],
        "causal_prefix_equivalence_proof": proof_identity,
    }
    plan["plan_sha256"] = causal.canonical_json_sha256(plan)
    attempt_id = "G1-ALL-LEROBOT-CAUSAL-FULL-R1-ATTEMPT1"
    started = {
        "schema_version": causal.ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "mode": "INITIAL",
        "resume_from_attempt_id": None,
        "plan_sha256": causal.canonical_json_sha256(plan),
        "plan": plan,
    }
    producer = {
        "schema_version": "g1-all-lerobot-c3-producer-identity-v1",
        "attempt_id": attempt_id,
        "resume_chain": [],
        "attempt_plan_sha256": started["plan_sha256"],
        "execution_plan_sha256": plan["plan_sha256"],
        "git_head": "a" * 40,
        "toolchain_identity_sha256": "b" * 64,
        "implementation_identity_sha256": "c" * 64,
        "input_freeze_file_sha256": causal.EXPECTED_INPUT_FREEZE_FILE_SHA256,
        "input_snapshot_sha256": causal.EXPECTED_INPUT_SNAPSHOT_SHA256,
        "episode_freeze_core_sha256": causal.EXPECTED_EPISODE_FREEZE_CORE_SHA256,
        "expected_episode_lengths_sha256": causal.canonical_json_sha256(lengths),
        "expected_derivations_sha256": causal.canonical_json_sha256(derivations),
        "automatic_retries": 0,
    }
    fixture["producer_identity"] = producer
    _write_json(
        fixture_root / "attempts" / attempt_id / "ATTEMPT_STARTED.json",
        started,
    )

    identity = causal.validate_producer_and_proof(fixture_root, fixture)
    assert identity["producer_identity"] == producer
    assert identity["causal_prefix_equivalence_proof"] == proof_identity

    drifted = json.loads(json.dumps(fixture))
    drifted["producer_identity"]["expected_derivations_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="producer identity differs"):
        causal.validate_producer_and_proof(fixture_root, drifted)

    monkeypatch.setattr(
        causal,
        "validate_causal_proof",
        lambda: {**proof_identity, "sha256": "f" * 64},
    )
    with pytest.raises(RuntimeError, match="plan/proof binding"):
        causal.validate_producer_and_proof(fixture_root, fixture)
