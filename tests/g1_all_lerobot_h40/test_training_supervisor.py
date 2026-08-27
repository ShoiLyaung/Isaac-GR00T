from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "examples/G1_ALL_LEROBOT_H40/launch_joint_training.py"
SPEC = importlib.util.spec_from_file_location("joint_training_supervisor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)


def test_profile_and_formal_commands_differ_only_at_bounded_training_fields(tmp_path):
    manifest = tmp_path / "DATASET_MANIFEST.json"
    profile = supervisor.build_train_command(
        mode=supervisor.MODES["profile"],
        dataset_manifest=manifest,
        train_output=tmp_path / "profile",
        master_port=12345,
    )
    formal = supervisor.build_train_command(
        mode=supervisor.MODES["formal"],
        dataset_manifest=manifest,
        train_output=tmp_path / "formal",
        master_port=12345,
    )

    for command in (profile, formal):
        assert "--dataset-manifest-path" in command
        assert "--dataset-adapter" in command
        assert "g1_all_lerobot_h40_v1" in command
        assert "--action-horizon" in command
        assert "40" in command
        assert "--tune-visual" in command
        assert "--no-tune-llm" in command
        assert "--nproc_per_node=2" in command
        assert "--adapter-config-path" not in command
    assert profile[profile.index("--max-steps") + 1] == "2"
    assert formal[formal.index("--max-steps") + 1] == "20000"
    assert "--save-only-model" in profile
    assert "--no-save-only-model" in formal


def test_isolation_is_offline_and_writes_only_attempt_directory(tmp_path):
    command = supervisor.isolation_command(["uv", "run", "true"], tmp_path)
    assert command[:3] == ["bwrap", "--die-with-parent", "--new-session"]
    assert "--unshare-net" in command
    bind = command.index("--bind")
    assert command[bind + 1 : bind + 3] == [str(tmp_path), str(tmp_path)]
    assert command[-3:] == ["uv", "run", "true"]


def test_gpu_gate_rejects_existing_compute_and_low_memory():
    snapshot = {
        "gpus": [
            {"index": 0, "free_mib": 48000, "uncorrected_ecc": "[N/A]"},
            {"index": 1, "free_mib": 48000, "uncorrected_ecc": "[N/A]"},
        ],
        "compute_apps": [
            {
                "gpu_uuid": "GPU-x",
                "pid": 123,
                "used_mib": 1,
                "process_name": "python",
            }
        ],
    }
    try:
        supervisor.gate_gpus(snapshot, 40000)
    except RuntimeError as exc:
        assert "active compute processes" in str(exc)
    else:
        raise AssertionError("active compute process was accepted")

    snapshot["compute_apps"] = []
    snapshot["gpus"][1]["free_mib"] = 39999
    try:
        supervisor.gate_gpus(snapshot, 40000)
    except RuntimeError as exc:
        assert "below 40000" in str(exc)
    else:
        raise AssertionError("low-memory GPU was accepted")


def test_posttrain_evidence_closes_finite_nccl_and_sampling_contract(tmp_path):
    evidence_dir = tmp_path / "distributed_evidence"
    evidence_dir.mkdir()
    source_ids = ["a", "b"]
    for rank, counts in enumerate(([10, 6], [8, 8])):
        value = {
            "trainer_global_step": 1,
            "training_monitor": {
                "complete": True,
                "raw_loss_count": 1,
                "finite_loss_count": 1,
                "gradient_norm_count": 1,
                "finite_gradient_norm_count": 1,
                "skipped_optimizer_steps": 0,
                "overflow_events": 0,
            },
            "nccl_flight_recorder": {
                "required": True,
                "complete": True,
                "no_circular_buffer_wrap": True,
                "operation_sequence_sha256": "same",
            },
            "dataset_sampling": {
                "rank_local_counts_exact": True,
                "evidence": {
                    "total_samples_yielded": 16,
                    "datasets": [
                        {"samples_yielded": counts[0]},
                        {"samples_yielded": counts[1]},
                    ],
                },
            },
        }
        (evidence_dir / f"rank_{rank}_posttrain.json").write_text(
            json.dumps(value), encoding="utf-8"
        )

    result = supervisor._posttrain_evidence(tmp_path, 1, source_ids)

    assert result["complete"] is True
    assert result["actual_total_samples"] == 32
    assert result["source_sample_totals"] == {"a": 18, "b": 14}


def test_proc_identity_rejects_pid_reuse_identity():
    identity = supervisor.proc_identity(os.getpid())
    assert identity is not None
    assert supervisor.identity_matches(identity) is True
    identity["start_time"] += 1
    assert supervisor.identity_matches(identity) is False
