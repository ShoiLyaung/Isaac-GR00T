from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import gr00t.experimental.mikasa_native_rgb_memory_experiment as experiment
import gr00t.experimental.mikasa_native_rgb_memory_runtime as runtime
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_MIKASA_MEMORY"
sys.path.insert(0, str(EXAMPLE))
runner = importlib.import_module("native_rgb_memory_backbone")


def _checkpoint_tree(root: Path) -> dict[str, dict[int, Path]]:
    result = {}
    for condition in experiment.CONDITIONS:
        result[condition] = {}
        for seed in experiment.TRAIN_SEEDS:
            path = root / "checkpoints" / condition / f"seed_{seed}" / "checkpoint_final.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{condition}:{seed}".encode())
            result[condition][seed] = path
    return result


def test_cpu_synthetic_smoke_is_vla_free_and_finite() -> None:
    result = runner.synthetic_cpu_smoke()
    assert result["status"] == "passed"
    assert result["prediction_shape"] == [2, 48]
    assert result["finite"] is True
    assert result["native_vla_imported"] is False
    assert result["temporal_identity"]["parameters"] == 790_528


def test_joint_confirmation_fails_before_generator_or_corpus_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = _checkpoint_tree(tmp_path)
    dev = tmp_path / "DEV_PASS_RECEIPT.json"
    freeze = tmp_path / "FREEZE_RECEIPT.json"
    dev.write_text(json.dumps({"status": "gate_failed", "dev_gate_passed": False}))
    freeze.write_text(json.dumps({"status": "frozen"}))
    touched = []

    def forbidden_run(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("confirmation generator was launched")

    def forbidden_discover(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("confirmation corpus was opened")

    monkeypatch.setattr(experiment.subprocess, "run", forbidden_run)
    monkeypatch.setattr(experiment.ShardedMemoryCorpus, "discover", forbidden_discover)
    with pytest.raises(PermissionError, match="dev receipt"):
        experiment.run_joint_confirmation(
            dev_receipt_path=dev,
            freeze_receipt_path=freeze,
            checkpoint_root=tmp_path,
            source_paths={"test": Path(__file__)},
            confirmation_spec_path=tmp_path / "sealed-confirmation-spec.json",
            confirmation_corpus_root=tmp_path / "sealed-confirmation",
            output_root=tmp_path / "result",
            device=runner._device("cpu"),
        )
    assert touched == []
    assert not (tmp_path / "sealed-confirmation").exists()
    assert set(checkpoints) == set(experiment.CONDITIONS)


def test_confirmation_freeze_covers_sources_spec_generator_and_checkpoints(
    tmp_path: Path,
) -> None:
    checkpoints = _checkpoint_tree(tmp_path)
    source = tmp_path / "evaluator.py"
    generator = tmp_path / "generate.py"
    source.write_text("# evaluator\n")
    generator.write_text("# generator\n")
    spec = tmp_path / "CONFIRMATION_SPEC.json"
    spec.write_text(
        json.dumps(
            {
                "schema": "g1-mikasa-native-rgb-memory-backbone-r1/confirmation-spec-v1",
                "generator_argv": ["python", str(generator)],
                "generator_materials": [str(generator)],
            }
        )
    )
    _, spec_identity = runtime.confirmation_spec_identity(spec)
    dev = tmp_path / "DEV_PASS_RECEIPT.json"
    freeze = tmp_path / "FREEZE_RECEIPT.json"
    dev.write_text(json.dumps({"status": "passed", "dev_gate_passed": True}))
    freeze.write_text(
        json.dumps(
            {
                "status": "frozen",
                "checkpoint_sha256": {
                    condition: {
                        str(seed): runtime.sha256_file(path)
                        for seed, path in sorted(by_seed.items())
                    }
                    for condition, by_seed in sorted(checkpoints.items())
                },
                "source_sha256": {"evaluator": runtime.sha256_file(source)},
                "confirmation_spec": spec_identity,
            }
        )
    )
    evidence = runtime.verify_confirmation_receipts(
        dev,
        freeze,
        checkpoints,
        {"evaluator": source},
        spec,
    )
    assert evidence["confirmation_spec"] == spec_identity
    generator.write_text("# changed generator\n")
    with pytest.raises(PermissionError, match="generator/spec"):
        runtime.verify_confirmation_receipts(
            dev,
            freeze,
            checkpoints,
            {"evaluator": source},
            spec,
        )


def test_runner_declares_all_frozen_stages_and_no_native_vla_import() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    for flag in (
        "--build-feature-cache",
        "--cost",
        "--preflight",
        "--overfit",
        "--matched-train",
        "--eval-dev",
        "--joint-confirmation",
    ):
        assert flag in source
    assert "gr00t.model.gr00t_n1d7" not in source.split("def synthetic_cpu_smoke", 1)[0]
    assert "gr00t.model.modules.dit" not in source.split("def synthetic_cpu_smoke", 1)[0]
    assert "causal_conv1d" in source
    assert "finite_by_block" in source
    assert str(runner.OVERLAY_ROOT).endswith("dependencies/python-overlay")


def test_matched_training_requires_all_three_passed_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "cost": {
            "status": "passed",
            "mode": "cost",
            "three_blocks_passed": True,
            "block_passes": [True, True, True],
            "overlay": {"verified": True},
        },
        "batch24_preflight": {
            "status": "passed",
            "mode": "batch24_preflight",
            "steps": 20,
            "batch_size": 24,
            "finite": True,
            "formal_checkpoint_written": False,
            "overlay": {"verified": True},
        },
        "overfit": {
            "status": "passed",
            "mode": "overfit",
            "gates": {"loss": True, "order": True},
            "mamba_overlay": {"verified": True},
        },
    }
    paths = {}
    for name, value in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    monkeypatch.setattr(runner, "verify_mamba_overlay", lambda: {"verified": True})
    evidence = runner.verify_training_prerequisites(
        cost_path=paths["cost"],
        cost_sha256=runtime.sha256_file(paths["cost"]),
        preflight_path=paths["batch24_preflight"],
        preflight_sha256=runtime.sha256_file(paths["batch24_preflight"]),
        overfit_path=paths["overfit"],
        overfit_sha256=runtime.sha256_file(paths["overfit"]),
    )
    assert set(evidence) == {"cost", "batch24_preflight", "overfit", "mamba_overlay"}
    payloads["overfit"]["status"] = "gate_failed"
    paths["overfit"].write_text(json.dumps(payloads["overfit"]))
    with pytest.raises(PermissionError, match="identity changed"):
        runner.verify_training_prerequisites(
            cost_path=paths["cost"],
            cost_sha256=runtime.sha256_file(paths["cost"]),
            preflight_path=paths["batch24_preflight"],
            preflight_sha256=runtime.sha256_file(paths["batch24_preflight"]),
            overfit_path=paths["overfit"],
            overfit_sha256="0" * 64,
        )


def test_formal_training_writes_only_final_step_checkpoint_name() -> None:
    source = Path(experiment.__file__).read_text(encoding="utf-8")
    assert '"checkpoint_final.pt"' in source
    assert "checkpoint-{step}" not in source
    assert "save_steps" not in source
    assert experiment.CONDITIONS == (
        "mamba_current",
        "mamba_one_frame",
        "mamba_last_segment",
        "mamba_full",
        "gru_full",
        "transformer_full",
    )
    assert experiment.TRAIN_SEEDS == (230923, 230924, 230925)


def test_dev_gate_uses_all_controls_groups_seeds_and_interventions() -> None:
    values = {
        "mamba_current": (1.0, 0.55),
        "mamba_one_frame": (1.0, 0.60),
        "mamba_last_segment": (1.0, 0.65),
        "mamba_full": (0.70, 0.90),
        "gru_full": (0.69, 0.91),
        "transformer_full": (0.72, 0.88),
    }
    records = {}
    aggregates = {}
    for condition, (mae, accuracy) in values.items():
        records[condition] = {}
        by_seed = {}
        for seed in experiment.TRAIN_SEEDS:
            records[condition][seed] = [
                {
                    "group_id": group_id,
                    "first_mae_m": mae,
                    "complete_mae_m": mae,
                }
                for group_id in range(6)
            ]
            by_seed[str(seed)] = {
                "first_mae_m": mae,
                "complete_mae_m": mae,
                "order_correct": accuracy,
            }
        aggregates[condition] = {
            "by_seed": by_seed,
            "seed_equal": {
                "first_mae_m": mae,
                "complete_mae_m": mae,
                "order_correct": accuracy,
            },
        }
    interventions = {
        str(seed): {
            "substitute": {"order_correct": 0.90, "first_mae_m": 0.70},
            "zero": {"order_correct": 0.20, "first_mae_m": 0.80},
            "shuffle": {"order_correct": 0.20, "first_mae_m": 0.80},
        }
        for seed in experiment.TRAIN_SEEDS
    }
    evidence = {
        "records": records,
        "aggregates": aggregates,
        "interventions": interventions,
    }
    cost = {
        "three_blocks_passed": True,
        "mamba_transformer_p95_ratio_t1040": 0.70,
        "mamba_transformer_peak_allocated_ratio_t1040": 0.80,
    }
    gates = experiment.evaluate_dev_gates(evidence, cost)
    assert gates and all(gates.values())
    evidence["aggregates"]["mamba_full"]["seed_equal"]["order_correct"] = 0.79
    failed = experiment.evaluate_dev_gates(evidence, cost)
    assert failed["mamba_full_order_accuracy_at_least_80pct"] is False
