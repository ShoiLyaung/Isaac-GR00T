from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import (
    _CODEC_ONLY_PARAMETER_NAMES,
    _CODEC_ONLY_TRAINABLE_NUMEL,
)
from safetensors.torch import save_file
import torch
from transformers import AutoConfig, AutoProcessor


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "examples/G1_ALL_LEROBOT_CODEC/launch_c5_codec_only.py"
SPEC = importlib.util.spec_from_file_location("c5_training_supervisor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
c5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = c5
SPEC.loader.exec_module(c5)


def test_launcher_bootstraps_source_root_without_pythonpath(tmp_path):
    code = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('isolated_c5_launcher', {str(MODULE_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert str(module.SOURCE_ROOT) in sys.path
import gr00t.model
print(module.SOURCE_ROOT)
"""
    environment = {
        **os.environ,
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == str(SOURCE_ROOT)


def test_c5_commands_freeze_only_codec_and_profile_warmup(tmp_path):
    manifest = tmp_path / "DATASET_MANIFEST.json"
    profile = c5.build_train_command(
        mode=c5.MODES["profile"],
        dataset_manifest=manifest,
        train_output=tmp_path / "profile",
        master_port=12345,
    )
    pilot = c5.build_train_command(
        mode=c5.MODES["pilot"],
        dataset_manifest=manifest,
        train_output=tmp_path / "pilot",
        master_port=12345,
    )

    for command in (profile, pilot):
        assert command[command.index("--base-model-path") + 1] == str(c5.RETAINED_CHECKPOINT)
        assert command[command.index("--output-dir") + 1] in {
            str(tmp_path / "profile"),
            str(tmp_path / "pilot"),
        }
        assert "--experiment-name" not in command
        assert command[command.index("--codec-trainability") + 1] == "codec_only"
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--state-dropout-prob") + 1] == "0.2"
        assert "--no-tune-llm" in command
        assert "--no-tune-visual" in command
        assert "--no-tune-projector" in command
        assert "--no-tune-diffusion-model" in command
        assert "--no-tune-vlln" in command
        assert "--nproc_per_node=2" in command
        assert "--master_port=12345" in command
    assert profile[profile.index("--warmup-ratio") + 1] == "0.0"
    assert pilot[pilot.index("--warmup-ratio") + 1] == "0.05"
    assert "--save-only-model" in profile
    assert "--no-save-only-model" in pilot


def test_supervisor_trainable_identity_matches_model_contract():
    assert c5.EXPECTED_PARAMETER_NAMES == _CODEC_ONLY_PARAMETER_NAMES
    assert c5.EXPECTED_TRAINABLE_NUMEL == _CODEC_ONLY_TRAINABLE_NUMEL


def test_codec_posttrain_closes_optimizer_gradient_and_rank_state(tmp_path):
    evidence_dir = tmp_path / "distributed_evidence"
    evidence_dir.mkdir()
    source_ids = list(c5.SOURCE_IDS)
    optimizer = {
        "mode": "codec_only",
        "parameter_count": len(c5.EXPECTED_PARAMETER_NAMES),
        "trainable_numel": c5.EXPECTED_TRAINABLE_NUMEL,
        "parameter_names": list(c5.EXPECTED_PARAMETER_NAMES),
        "parameter_names_sha256": c5.EXPECTED_PARAMETER_NAMES_SHA256,
    }
    gate_name = "action_head.codec_conditioning.conditioner.gate"
    non_gate = [name for name in c5.EXPECTED_PARAMETER_NAMES if name != gate_name]
    rank_values = []
    for rank in range(2):
        datasets = [{"samples_yielded": 0} for _ in source_ids]
        datasets[rank]["samples_yielded"] = 32
        partition = non_gate
        gradient_steps = []
        for step_index in range(2):
            nonzero_names = partition[:1] if step_index == 1 else []
            gradient_steps.append(
                {
                    "step_index": step_index,
                    "gate": {
                        "present": True,
                        "finite": True,
                        "nonzero": True,
                        "call_count": 1,
                    },
                    "non_gate_present_names": partition,
                    "non_gate_nonzero_names": nonzero_names,
                    "non_gate_nonfinite_names": [],
                    "non_gate_invalid_call_names": [],
                }
            )
        value = {
            "trainer_global_step": 2,
            "optimizer_trainability": optimizer,
            "training_monitor": {
                "complete": True,
                "raw_loss_count": 2,
                "finite_loss_count": 2,
                "gradient_norm_count": 2,
                "finite_gradient_norm_count": 2,
                "skipped_optimizer_steps": 0,
                "overflow_events": 0,
                "trainability": {
                    "complete": True,
                    "initial_state_sha256": "a" * 64,
                    "final_state_sha256": "b" * 64,
                    "initial_gate": 0.0,
                    "final_gate": 0.001,
                    "first_gradient_steps": gradient_steps,
                },
            },
            "nccl_flight_recorder": {
                "required": True,
                "complete": True,
                "no_circular_buffer_wrap": True,
                "operation_sequence_sha256": "same",
            },
            "dataset_sampling": {
                "rank_local_counts_exact": True,
                "evidence": {"total_samples_yielded": 32, "datasets": datasets},
            },
        }
        rank_values.append(value)
        (evidence_dir / f"rank_{rank}_posttrain.json").write_text(json.dumps(value))

    evidence = c5._codec_posttrain_evidence(tmp_path, 2, source_ids)

    assert evidence["complete"] is True
    assert evidence["codec_state_closure_complete"] is True
    assert len(evidence["codec_ranks"]) == 2

    def rejects(mutator):
        values = copy.deepcopy(rank_values)
        mutator(values)
        for rank, value in enumerate(values):
            (evidence_dir / f"rank_{rank}_posttrain.json").write_text(json.dumps(value))
        result = c5._codec_posttrain_evidence(tmp_path, 2, source_ids)
        assert result["distributed_gradients_complete"] is False
        assert result["complete"] is False

    rejects(
        lambda values: values[1]["training_monitor"]["trainability"]["first_gradient_steps"][0][
            "non_gate_present_names"
        ].pop()
    )
    rejects(
        lambda values: [
            rank["training_monitor"]["trainability"]["first_gradient_steps"][0]["gate"].update(
                nonzero=False
            )
            for rank in values
        ]
    )
    rejects(
        lambda values: [
            step.update(non_gate_nonzero_names=[])
            for rank in values
            for step in rank["training_monitor"]["trainability"]["first_gradient_steps"][1:]
        ]
    )
    rejects(
        lambda values: values[0]["training_monitor"]["trainability"]["first_gradient_steps"][
            0
        ].update(non_gate_nonfinite_names=[non_gate[0]])
    )
    rejects(
        lambda values: values[0]["training_monitor"]["trainability"]["first_gradient_steps"][0][
            "gate"
        ].update(call_count=2)
    )
    rejects(
        lambda values: values[0]["training_monitor"]["trainability"]["first_gradient_steps"][
            0
        ].update(non_gate_invalid_call_names=[non_gate[0]])
    )


def test_saved_codec_checkpoint_reloads_exact_declared_tensors(tmp_path, monkeypatch):
    names = (
        "action_head.codec_conditioning.conditioner.gate",
        "action_head.codec_conditioning.core.mv_stem.bias",
    )
    monkeypatch.setattr(c5, "EXPECTED_PARAMETER_NAMES", names)
    monkeypatch.setattr(c5, "EXPECTED_TRAINABLE_NUMEL", 2)
    retained = tmp_path / "retained"
    retained.mkdir()
    (retained / "statistics.json").write_text("{}")
    retained_mapping = {"new_embodiment": 10, "another_embodiment": 3}
    (retained / "embodiment_id.json").write_text(json.dumps(retained_mapping))
    monkeypatch.setattr(c5, "RETAINED_CHECKPOINT", retained)
    tensors = {names[0]: torch.tensor(0.25), names[1]: torch.tensor([1.0])}
    shard = "model.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in names}})
    )
    config = {
        "codec_evidence": c5._model_codec_contract(),
        "codec_trainability": "codec_only",
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": False,
        "tune_diffusion_model": False,
        "tune_vlln": False,
        "tune_top_llm_layers": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"codec_evidence": c5._model_codec_contract()}})
    )
    (tmp_path / "statistics.json").write_text("{}")
    (tmp_path / "embodiment_id.json").write_text(json.dumps(retained_mapping))
    config_calls = []
    processor_calls = []
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda path, **kwargs: (
            config_calls.append((path, kwargs))
            or type(
                "LoadedConfig",
                (),
                {
                    "codec_evidence": c5._model_codec_contract(),
                    "codec_trainability": "codec_only",
                },
            )()
        ),
    )
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        lambda path, **kwargs: (
            processor_calls.append((path, kwargs))
            or type(
                "LoadedProcessor",
                (),
                {
                    "codec_evidence": c5._model_codec_contract(),
                    "embodiment_id_mapping": retained_mapping,
                },
            )()
        ),
    )

    first = c5._saved_codec_checkpoint(tmp_path, None)
    second = c5._saved_codec_checkpoint(tmp_path, first["state_sha256"])

    assert first["complete"] is False
    assert second["complete"] is True
    assert second["parameter_count"] == 2
    assert second["gate"] == 0.25
    assert second["standalone_reload_complete"] is True
    assert config_calls[-1][1]["local_files_only"] is True
    assert processor_calls[-1][1]["local_files_only"] is True
    assert processor_calls[-1][1]["transformers_loading_kwargs"] == {
        "trust_remote_code": True,
        "local_files_only": True,
    }
