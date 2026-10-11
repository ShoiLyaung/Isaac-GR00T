from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (
    EXPECTED_SPLIT_GROUPS,
    canonical_json_bytes,
    write_new_json,
)
from gr00t.experimental.mikasa_temporal_moment_cache import CONDITIONS
import gr00t.experimental.mikasa_temporal_moment_experiment as experiment
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_MIKASA_MEMORY"))
runner = importlib.import_module("temporal_moment_sufficiency")


def _evidence(mae: float, order: float) -> dict:
    groups = (1, 5, 16, 22, 23, 28)
    records = {
        seed: [
            {
                "group_id": group,
                "first_mae_m": mae,
                "complete_mae_m": mae,
            }
            for group in groups
        ]
        for seed in (230923, 230924, 230925)
    }
    aggregate = {
        "first_shape_accuracy": 0.95,
        "second_shape_accuracy": 0.95,
        "ordered_pair_accuracy": order,
        "first_mae_m": mae,
        "complete_mae_m": mae,
        "swap_shape_exact": 1.0,
        "swap_checkpoint_exact": 1.0,
        "swap_checkpoint_max_abs": 0.0,
    }
    return {"records": records, "aggregates": {"seed_equal": aggregate}}


def test_smoke_and_relative_gates_are_serializable() -> None:
    smoke = runner.smoke()
    assert smoke["status"] == "passed"
    assert smoke["mamba_loaded"] is False
    ordered, control = _evidence(0.02, 0.9), _evidence(0.03, 0.6)
    gates = experiment.relative_gates(ordered, control, "control")
    assert gates and all(gates.values())
    assert json.loads(canonical_json_bytes(gates)) == gates
    prior = {"aggregates": {"seed_equal": dict(ordered["aggregates"]["seed_equal"])}}
    assert all(experiment.reproduction_gates(ordered, prior).values())
    prior["aggregates"]["seed_equal"].pop("swap_checkpoint_exact")
    with pytest.raises(ValueError, match="metric set"):
        experiment.reproduction_gates(ordered, prior)


def test_nine_train_calls_finish_before_first_target_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    class FakeCorpus:
        group_ids = EXPECTED_SPLIT_GROUPS["train"]

    monkeypatch.setattr(experiment, "validate_frozen_split", lambda groups: {"ok": True})
    monkeypatch.setattr(experiment.TargetStats, "fit", lambda corpus: object())
    monkeypatch.setattr(experiment, "FitTargetCorpus", lambda corpus: object())
    monkeypatch.setattr(experiment, "fit_target_mean", lambda corpus: np.zeros(48, np.float32))
    monkeypatch.setattr(experiment, "FitOnlyData", lambda corpus, cache: object())

    predict_count = 0

    def fake_predict(checkpoints, cache, *, device):
        nonlocal predict_count
        predict_count += 1
        events.append(("predict", predict_count))
        return {"prediction": predict_count}

    def fake_train(data, stats, *, seed, checkpoint_path, **kwargs):
        condition = checkpoint_path.parts[-3]
        events.append(("train", condition, seed))
        return {
            "steps": 1500,
            "schedule_sha256": f"schedule-{seed}",
            "curve": [{"step": step, "lr": float(step) / 1500} for step in range(1, 1501)],
        }

    def fake_score(predictions, corpus, target_mean):
        events.append(("score", predictions["prediction"]))
        return _evidence(0.02, 0.9)

    monkeypatch.setattr(experiment, "predict_calibration", fake_predict)
    monkeypatch.setattr(experiment, "train_fit_seed", fake_train)
    monkeypatch.setattr(experiment, "score_predictions", fake_score)
    monkeypatch.setattr(experiment, "evaluate_calibration_gates", lambda evidence: {"abs": True})
    monkeypatch.setattr(experiment, "relative_gates", lambda *args: {"rel": True})
    monkeypatch.setattr(experiment, "reproduction_gates", lambda *args: {"repro": True})
    monkeypatch.setattr(
        experiment, "verify_historical_checkpoints", lambda checkpoints, prior: {"ok": True}
    )
    result = experiment.run_temporal_moment(
        FakeCorpus(),
        object(),
        {condition: object() for condition in CONDITIONS},
        {seed: Path(f"seed_{seed}/checkpoint_final.pt") for seed in (230923, 230924, 230925)},
        {},
        tmp_path / "run",
        device=torch.device("cpu"),
        provenance={},
    )
    train_indices = [index for index, event in enumerate(events) if event[0] == "train"]
    score_indices = [index for index, event in enumerate(events) if event[0] == "score"]
    assert len(train_indices) == 9
    assert max(train_indices) < min(score_indices)
    assert result["calibration_opened_after_optimizer_steps"] == 13500


def test_write_once_and_dev_confirmation_paths_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    write_new_json(path, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"status": "changed"})
    for name in ("dev", "confirmation"):
        with pytest.raises(PermissionError):
            runner.reject_path(tmp_path / name / "secret")


def test_historical_checkpoints_bind_path_bytes_and_sha(tmp_path: Path) -> None:
    checkpoints, training = {}, []
    for seed in (230923, 230924, 230925):
        path = tmp_path / f"seed_{seed}" / "checkpoint_final.pt"
        path.parent.mkdir()
        body = f"checkpoint-{seed}".encode()
        path.write_bytes(body)
        checkpoints[seed] = path
        training.append(
            {
                "seed": seed,
                "checkpoint": {
                    "path": str(path.resolve()),
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                },
            }
        )
    verified = experiment.verify_historical_checkpoints(checkpoints, {"training": training})
    assert set(verified) == {"230923", "230924", "230925"}
    checkpoints[230923].write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt changed"):
        experiment.verify_historical_checkpoints(checkpoints, {"training": training})


def test_learning_rate_hash_uses_actual_curve_values() -> None:
    first = [{"step": step, "lr": step / 1500} for step in range(1, 1501)]
    second = [dict(record) for record in first]
    assert experiment.learning_rate_curve_sha256(first) == experiment.learning_rate_curve_sha256(
        second
    )
    second[700]["lr"] += 1e-9
    assert experiment.learning_rate_curve_sha256(first) != experiment.learning_rate_curve_sha256(
        second
    )


def test_build_cli_disables_privileged_target_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discover_calls = []

    class FakeSourceCache:
        root = tmp_path / "source-cache"

    def fake_corpus_discover(root, split, **kwargs):
        discover_calls.append(kwargs)
        return object()

    monkeypatch.setattr(runner.ShardedMemoryCorpus, "discover", fake_corpus_discover)
    monkeypatch.setattr(
        runner.FeatureCacheCorpus,
        "discover",
        lambda root, split: FakeSourceCache(),
    )
    monkeypatch.setattr(
        runner,
        "_read",
        lambda path, digest, status: (
            {"status": "passed", "targets_opened": False},
            {"path": str(path.resolve()), "bytes": 1, "sha256": digest},
        ),
    )
    monkeypatch.setattr(runner, "verify_source_cache_receipt", lambda *args: {"ok": True})
    monkeypatch.setattr(
        runner,
        "build_temporal_moment_caches",
        lambda source, corpus, output: {
            "records": {condition: [] for condition in CONDITIONS},
            "permutation_identity": {},
            "permutation_sha256": [],
        },
    )
    monkeypatch.setattr(runner, "write_new_json", lambda path, result: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal_moment_sufficiency.py",
            "--build-cache",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--source-cache-root",
            str(tmp_path / "source-cache"),
            "--corpus-result",
            str(tmp_path / "corpus-result.json"),
            "--source-cache-result",
            str(tmp_path / "source-result.json"),
            "--output-root",
            str(tmp_path / "output"),
        ],
    )
    assert runner.main() == 0
    assert discover_calls == [{"verify_privileged_targets": False}]


def test_source_cache_receipt_rejects_unrelated_root(tmp_path: Path) -> None:
    source_cache = type("Source", (), {"root": tmp_path / "actual"})()
    with pytest.raises(ValueError, match="root"):
        runner.verify_source_cache_receipt(
            source_cache,
            {},
            {"path": str(tmp_path / "other/RESULT.json")},
        )
