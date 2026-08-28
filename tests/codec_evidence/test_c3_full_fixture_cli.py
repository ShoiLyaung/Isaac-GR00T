from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from gr00t.data.codec_evidence.episode_bundle import SOURCE_KIND_NATIVE_CANONICAL, EpisodeBundleKey
from gr00t.data.codec_evidence.warehouse import (
    ATTEMPT_SCHEMA_VERSION,
    AttemptJournal,
    canonical_json_sha256,
    ensure_warehouse_layout,
)
import numpy as np
import pytest


def _load_cli():
    path = Path(__file__).parents[2] / "examples/G1_ALL_LEROBOT_CODEC/build_c3_full_fixture.py"
    spec = importlib.util.spec_from_file_location("c3_full_fixture_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attempt_plan() -> dict[str, object]:
    return {
        "schema_version": "focused-c3-plan-v1",
        "input_snapshot_sha256": "a" * 64,
        "budget_sha256": "b" * 64,
    }


def test_output_filesystem_identity_excludes_volatile_free_space(tmp_path, monkeypatch):
    module = _load_cli()
    output = tmp_path / "candidate"
    monkeypatch.setattr(module, "EXPECTED_OUTPUT_ROOT", output)
    monkeypatch.setattr(module, "MAX_NEW_BYTES", 100)
    free_values = iter((1_000, 900))

    def disk_usage(_path):
        return SimpleNamespace(total=2_000, used=1_000, free=next(free_values))

    monkeypatch.setattr(module.shutil, "disk_usage", disk_usage)
    first = module._output_filesystem_identity()
    second = module._output_filesystem_identity()
    assert (
        first
        == second
        == {
            "output_parent": str(tmp_path.resolve()),
            "required_free_bytes": 100,
        }
    )


def test_full_input_verifier_visits_every_metadata_and_episode(monkeypatch):
    module = _load_cli()
    sources = [{"source_id": "a"}, {"source_id": "b"}]
    episodes = [
        {"source_id": "a", "num_frames": 3},
        {"source_id": "a", "num_frames": 5},
        {"source_id": "b", "num_frames": 7},
    ]
    frozen = {"sources": sources, "episodes": episodes}
    metadata_seen: list[str] = []
    episodes_seen: list[int] = []
    marker_checks: list[bool] = []
    monkeypatch.setattr(module, "EXPECTED_FRAMES", 15)
    monkeypatch.setattr(module, "_check_dataset_markers", lambda: marker_checks.append(True))
    monkeypatch.setattr(
        module,
        "_verify_source_metadata",
        lambda source: metadata_seen.append(source["source_id"]),
    )
    monkeypatch.setattr(
        module,
        "_episode_input",
        lambda episode, _source: SimpleNamespace(num_frames=episode["num_frames"]),
    )
    monkeypatch.setattr(
        module,
        "verify_canonical_input_identity",
        lambda episode, *, stage: episodes_seen.append(episode.num_frames),
    )

    module._verify_all_frozen_inputs(frozen)

    assert metadata_seen == ["a", "b"]
    assert episodes_seen == [3, 5, 7]
    assert len(marker_checks) == 2


def test_initial_attempt_reserves_exact_root_without_adopting_collision(tmp_path, monkeypatch):
    module = _load_cli()
    output = tmp_path / "warehouse"
    monkeypatch.setattr(module, "EXPECTED_OUTPUT_ROOT", output)
    monkeypatch.setattr(module, "INITIAL_ATTEMPT_ID", "attempt-1")
    plan = _attempt_plan()

    journal, prior_wall, chain = module._create_attempt(
        plan,
        attempt_id="attempt-1",
        resume_from_attempt_id=None,
    )
    assert journal.attempt_id == "attempt-1"
    assert prior_wall == 0.0
    assert chain == []

    with pytest.raises(FileExistsError):
        module._create_attempt(
            plan,
            attempt_id="attempt-1",
            resume_from_attempt_id=None,
        )


def test_single_terminal_is_warehouse_compatible_and_manual_resume_is_exact(tmp_path, monkeypatch):
    module = _load_cli()
    output = tmp_path / "warehouse"
    monkeypatch.setattr(module, "EXPECTED_OUTPUT_ROOT", output)
    monkeypatch.setattr(module, "INITIAL_ATTEMPT_ID", "attempt-1")
    plan = _attempt_plan()
    first, _prior_wall, _chain = module._create_attempt(
        plan,
        attempt_id="attempt-1",
        resume_from_attempt_id=None,
    )
    terminal_path = module._finish_attempt_once(
        first,
        status="FAILED",
        result={"resources": {"cumulative_wall_seconds": 12.5}},
        resume_from_attempt_id=None,
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["schema_version"] == ATTEMPT_SCHEMA_VERSION
    assert terminal["status"] == "FAILED"

    with pytest.raises(FileExistsError):
        module._finish_attempt_once(
            first,
            status="READY",
            result={},
            resume_from_attempt_id=None,
        )

    resumed, prior_wall, chain = module._create_attempt(
        plan,
        attempt_id="attempt-2",
        resume_from_attempt_id="attempt-1",
    )
    assert resumed.attempt_id == "attempt-2"
    assert prior_wall == 12.5
    assert chain == ["attempt-1"]


def test_plan_mismatch_rejects_before_manual_resume_scratch_cleanup(tmp_path, monkeypatch):
    module = _load_cli()
    output = tmp_path / "warehouse"
    monkeypatch.setattr(module, "EXPECTED_OUTPUT_ROOT", output)
    monkeypatch.setattr(module, "INITIAL_ATTEMPT_ID", "attempt-1")
    first, _prior_wall, _chain = module._create_attempt(
        _attempt_plan(),
        attempt_id="attempt-1",
        resume_from_attempt_id=None,
    )
    module._finish_attempt_once(
        first,
        status="STOPPED",
        result={"resources": {"cumulative_wall_seconds": 4.0}},
        resume_from_attempt_id=None,
    )
    sentinel = output / ".scratch" / "attempt-1" / "preserved.bin"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"preserve until an exact manual resume")
    drifted = {**_attempt_plan(), "budget_sha256": "c" * 64}

    with pytest.raises(ValueError, match="exact current plan"):
        module._create_attempt(
            drifted,
            attempt_id="attempt-2",
            resume_from_attempt_id="attempt-1",
        )
    assert sentinel.read_bytes() == b"preserve until an exact manual resume"
    assert not (output / "attempts" / "attempt-2").exists()


def test_prepublish_guard_rejects_scratch_and_storage_before_publish(tmp_path, monkeypatch):
    module = _load_cli()
    root = ensure_warehouse_layout(tmp_path / "warehouse")
    scratch = root / ".scratch" / "attempt"
    scratch.mkdir(parents=True)
    (scratch / "encoded.bin").write_bytes(b"123")
    monitor = module.BudgetMonitor(
        root=root,
        started=module.time.monotonic(),
        prior_cumulative_wall_seconds=0.0,
        persistent_bytes=0,
    )
    monkeypatch.setattr(module, "MAX_SCRATCH_BYTES", 6)
    monkeypatch.setattr(module, "PUBLISH_OVERHEAD_RESERVE_BYTES", 0)
    monkeypatch.setattr(module, "FINAL_COMMIT_RESERVE_BYTES", 0)

    with pytest.raises(RuntimeError, match="scratch quota"):
        monitor.prepublish_guard(4)

    (scratch / "encoded.bin").unlink()
    persistent = root / "attempts" / "receipt.bin"
    persistent.write_bytes(b"x" * 90)
    monkeypatch.setattr(module, "MAX_SCRATCH_BYTES", 100)
    monkeypatch.setattr(module, "MAX_NEW_BYTES", 100)
    with pytest.raises(RuntimeError, match="storage quota"):
        monitor.prepublish_guard(11)


def test_final_bundle_facts_require_every_gop30_anchor(monkeypatch, tmp_path):
    module = _load_cli()
    uid = "1" * 64
    key = EpisodeBundleKey(uid, module.CAMERA_ID, SOURCE_KIND_NATIVE_CANONICAL)
    published = SimpleNamespace(
        key=key,
        object_path=tmp_path / "object",
        bundle_manifest_sha256="a" * 64,
    )
    bundle = SimpleNamespace(
        num_frames=40,
        num_anchors=2,
        anchor_frame_index=np.asarray([0, 30], dtype=np.int64),
    )

    class FakeStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def get(self, _key):
            return bundle

    monkeypatch.setattr(module, "EpisodeBundleStore", FakeStore)
    monkeypatch.setattr(module, "EXPECTED_FRAMES", 40)
    monkeypatch.setattr(module, "EXPECTED_CODEC_ANCHORS", 2)
    frozen = {"episodes": [{"episode_uid": uid, "num_frames": 40}]}
    assert module._verify_final_bundle_facts([published], frozen) == (40, 2)

    bundle.anchor_frame_index = np.asarray([0], dtype=np.int64)
    bundle.num_anchors = 1
    with pytest.raises(RuntimeError, match="length/anchor closure"):
        module._verify_final_bundle_facts([published], frozen)


def test_supervisor_recovers_one_plan_bound_ready_terminal_from_atomic_manifest(
    tmp_path, monkeypatch
):
    module = _load_cli()
    root = ensure_warehouse_layout(tmp_path / "warehouse")
    plan = {
        "plan_sha256": "unused-until-normalized",
        "toolchain": {"git": {"head": "d" * 40}},
        "toolchain_identity_sha256": "1" * 64,
        "implementation_identity_sha256": "2" * 64,
        "input_freeze": {"sha256": "3" * 64},
        "input_snapshot_sha256": "4" * 64,
        "episode_freeze_core_sha256": "5" * 64,
        "expected_episode_lengths_sha256": "placeholder",
        "expected_episode_uid_digest": "placeholder",
        "budget_sha256": "6" * 64,
        "codec_contract": {"focused": True},
        "codec_contract_sha256": "7" * 64,
    }
    plan_without_digest = dict(plan)
    plan_without_digest.pop("plan_sha256")
    plan["plan_sha256"] = canonical_json_sha256(plan_without_digest)
    journal = AttemptJournal.start(root, attempt_id="attempt-1", plan=plan)
    uid = "8" * 64
    derivation = "9" * 64
    object_digest = "a" * 64
    lengths = [{"episode_uid": uid, "num_frames": 40}]
    derivations = [{"episode_uid": uid, "derivation_sha256": derivation}]
    plan["expected_episode_lengths_sha256"] = canonical_json_sha256(lengths)
    plan["expected_episode_uid_digest"] = canonical_json_sha256([uid])
    # Recreate the attempt with the final exact plan used by the manifest.
    root = tmp_path / "warehouse-exact"
    root = ensure_warehouse_layout(root)
    plan_without_digest = dict(plan)
    plan_without_digest.pop("plan_sha256")
    plan["plan_sha256"] = canonical_json_sha256(plan_without_digest)
    journal = AttemptJournal.start(root, attempt_id="attempt-1", plan=plan)
    producer = {
        "schema_version": "g1-all-lerobot-c3-producer-identity-v1",
        "attempt_id": journal.attempt_id,
        "resume_chain": [],
        "attempt_plan_sha256": journal.plan_sha256,
        "execution_plan_sha256": plan["plan_sha256"],
        "git_head": plan["toolchain"]["git"]["head"],
        "toolchain_identity_sha256": plan["toolchain_identity_sha256"],
        "implementation_identity_sha256": plan["implementation_identity_sha256"],
        "input_freeze_file_sha256": plan["input_freeze"]["sha256"],
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "episode_freeze_core_sha256": plan["episode_freeze_core_sha256"],
        "expected_episode_lengths_sha256": plan["expected_episode_lengths_sha256"],
        "expected_derivations_sha256": canonical_json_sha256(derivations),
        "automatic_retries": 0,
    }
    object_root = root / "objects" / object_digest
    object_root.mkdir()
    module._write_json_new(
        object_root / "OBJECT_COMMITTED.json",
        {
            "schema_version": "codec-evidence-warehouse-object-commit-v1",
            "object_content_sha256": object_digest,
            "object_manifest_sha256": "b" * 64,
        },
    )
    module._write_json_new(
        root / "derivations" / f"{derivation}.json",
        {
            "schema_version": "codec-evidence-derivation-binding-v1",
            "derivation_sha256": derivation,
            "object_content_sha256": object_digest,
            "episode_uid": uid,
            "camera_id": module.CAMERA_ID,
            "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        },
    )
    manifest = {
        "schema_version": "codec-evidence-full-fixture-manifest-v1",
        "fixture_id": module.FIXTURE_ID,
        "complete": True,
        "coverage_kind": "FULL_EPISODE_SET",
        "camera_id": module.CAMERA_ID,
        "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        "expected_episode_count": 1,
        "covered_episode_count": 1,
        "covered_episode_uids": [uid],
        "covered_episode_uid_digest": canonical_json_sha256([uid]),
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "budget_sha256": plan["budget_sha256"],
        "codec_contract": plan["codec_contract"],
        "codec_contract_sha256": plan["codec_contract_sha256"],
        "objects": [
            {
                "episode_uid": uid,
                "camera_id": module.CAMERA_ID,
                "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
                "num_frames": 40,
                "derivation_sha256": derivation,
                "object_content_sha256": object_digest,
                "bundle_manifest_sha256": "c" * 64,
                "bundle_manifest_bytes": 1,
            }
        ],
        "producer_identity": producer,
    }
    module._write_json_new(root / "fixture_manifest.json", manifest)
    monkeypatch.setattr(module, "EXPECTED_EPISODES", 1)
    monkeypatch.setattr(module, "EXPECTED_FRAMES", 40)
    monkeypatch.setattr(module, "EXPECTED_CODEC_ANCHORS", 2)

    result = module._recover_ready_from_manifest(
        journal,
        plan,
        resume_from_attempt_id=None,
        prior_wall=0.0,
        supervisor_started=module.time.monotonic(),
    )
    assert result["status"] == "READY"
    terminal = json.loads((journal.path / "ATTEMPT_TERMINAL.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "READY"


def test_execute_requires_frozen_outer_timeout_supervisor(monkeypatch):
    module = _load_cli()
    monkeypatch.delenv("C3_OUTER_TIMEOUT_SECONDS", raising=False)
    with pytest.raises(RuntimeError, match="requires the frozen outer"):
        module._verify_outer_timeout_supervisor()
