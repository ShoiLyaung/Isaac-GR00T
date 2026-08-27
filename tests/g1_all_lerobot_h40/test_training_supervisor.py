from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


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

    result = supervisor._posttrain_evidence(
        tmp_path, final_global_step=1, steps_this_attempt=1, source_ids=source_ids
    )

    assert result["complete"] is True
    assert result["actual_total_samples"] == 32
    assert result["source_sample_totals"] == {"a": 18, "b": 14}


def test_proc_identity_rejects_pid_reuse_identity():
    identity = supervisor.proc_identity(os.getpid())
    assert identity is not None
    assert supervisor.identity_matches(identity) is True
    identity["start_time"] += 1
    assert supervisor.identity_matches(identity) is False


def _fake_resume_checkpoint(root: Path, step: int = 20000) -> tuple[Path, Path]:
    checkpoint = root / f"checkpoint-{step}"
    global_step = checkpoint / f"global_step{step}"
    global_step.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": step}), encoding="utf-8"
    )
    (checkpoint / "latest").write_text(f"global_step{step}\n", encoding="utf-8")
    for name in (
        "scheduler.pt",
        "rng_state_0.pth",
        "rng_state_1.pth",
        "training_args.bin",
    ):
        (checkpoint / name).write_bytes(name.encode())
    for rank in (0, 1):
        (global_step / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(
            f"optimizer-{rank}".encode()
        )
    (global_step / "mp_rank_00_model_states.pt").write_bytes(b"model-state")
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    for name in ("config.json", "embodiment_id.json", "processor_config.json", "statistics.json"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    analysis = root / "CHECKPOINT_ANALYSIS.json"
    analysis.write_text(
        json.dumps(
            {
                "analysis_succeeded": True,
                "checkpoint": {
                    "complete": True,
                    "global_step": step,
                    "path": str(checkpoint.resolve()),
                },
            }
        ),
        encoding="utf-8",
    )
    files = {
        path.relative_to(checkpoint).as_posix(): supervisor.sha256_file(path)
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    contract = root / "resume-contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "g1-all-lerobot-h40-resume-contract-v1",
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_analysis": {
                    "path": str(analysis.resolve()),
                    "sha256": supervisor.sha256_file(analysis),
                },
                "resume_global_step": step,
                "target_global_step": 40000,
                "continuation_training": {
                    "dataset_manifest_sha256": "a" * 64,
                    "preparation_summary_sha256": "b" * 64,
                    "cpu_mixture_closure_sha256": "c" * 64,
                    "zero2_config_sha256": "d" * 64,
                    "global_batch_size": 32,
                    "action_horizon": 40,
                },
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, contract


def test_continuation_targets_absolute_40000_with_20000_step_delta(tmp_path):
    mode = supervisor.MODES["continue_40k"]
    command = supervisor.build_train_command(
        mode=mode,
        dataset_manifest=tmp_path / "manifest.json",
        train_output=tmp_path / "train",
        master_port=12345,
    )

    assert mode.resume_step == 20000
    assert mode.max_steps - mode.resume_step == 20000
    assert command[command.index("--max-steps") + 1] == "40000"
    assert "--no-save-only-model" in command
    assert command[command.index("--save-steps") + 1] == "1000"


def test_resume_checkpoint_requires_full_state_and_command_uses_explicit_path(tmp_path):
    source, contract = _fake_resume_checkpoint(tmp_path / "source")
    identity = supervisor.validate_resume_checkpoint(source, 20000, 40000, contract)
    command = supervisor.build_train_command(
        mode=supervisor.MODES["continue_40k"],
        dataset_manifest=tmp_path / "manifest.json",
        train_output=tmp_path / "new-output",
        master_port=12345,
        resume_checkpoint=source,
    )

    assert identity["global_step"] == 20000
    assert command[command.index("--resume-from-checkpoint") + 1] == str(source)
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "new-output")
    assert str(source) != str(tmp_path / "new-output")

    (source / "scheduler.pt").unlink()
    with pytest.raises(RuntimeError, match="regular and contained"):
        supervisor.validate_resume_checkpoint(source, 20000, 40000, contract)


def test_resume_output_is_disjoint_and_explicitly_read_only(tmp_path):
    source, _contract = _fake_resume_checkpoint(tmp_path / "source")
    run_dir = tmp_path / "run"
    supervisor.validate_disjoint_output(run_dir, source)
    isolated = supervisor.isolation_command(["true"], run_dir, source)
    resume_mount = isolated.index("--ro-bind", isolated.index("--bind") + 1)
    assert isolated[resume_mount + 1 : resume_mount + 3] == [str(source), str(source)]

    with pytest.raises(RuntimeError, match="must be disjoint"):
        supervisor.validate_disjoint_output(source / "nested-run", source)


def test_terminal_collection_error_fails_closed_without_raising(tmp_path):
    evidence = tmp_path / "distributed_evidence"
    evidence.mkdir()
    (evidence / "rank_0_posttrain.json").write_text("{", encoding="utf-8")

    posttrain, optimizer_complete, error = supervisor.collect_terminal_evidence(
        tmp_path, supervisor.MODES["continue_40k"], ["source"]
    )

    assert posttrain["complete"] is False
    assert optimizer_complete is False
    assert "JSONDecodeError" in error


def test_supervisor_signal_latch_preserves_first_signal():
    latch = supervisor.SupervisorSignals()
    latch._handle(supervisor.signal.SIGTERM, None)
    latch._handle(supervisor.signal.SIGINT, None)
    assert latch.signum == supervisor.signal.SIGTERM


def test_resume_evidence_uses_attempt_delta_but_closes_final_global_step(tmp_path):
    evidence_dir = tmp_path / "distributed_evidence"
    evidence_dir.mkdir()
    for rank in (0, 1):
        value = {
            "trainer_global_step": 40000,
            "training_monitor": {
                "complete": True,
                "raw_loss_count": 20000,
                "finite_loss_count": 20000,
                "gradient_norm_count": 20000,
                "finite_gradient_norm_count": 20000,
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
                    "total_samples_yielded": 320000,
                    "datasets": [{"samples_yielded": 320000}],
                },
            },
        }
        (evidence_dir / f"rank_{rank}_posttrain.json").write_text(
            json.dumps(value), encoding="utf-8"
        )

    result = supervisor._posttrain_evidence(
        tmp_path,
        final_global_step=40000,
        steps_this_attempt=20000,
        source_ids=["all"],
    )

    assert result["complete"] is True
    assert result["global_steps"] == [40000, 40000]
    assert result["steps_this_attempt"] == 20000
    assert result["expected_total_samples"] == 640000
