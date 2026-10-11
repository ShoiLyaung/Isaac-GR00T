from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import gr00t.experimental.mikasa_event_factorized_mapping as factorized
from gr00t.experimental.mikasa_event_trajectory import CONDITIONS, group_permutations
from gr00t.experimental.mikasa_native_rgb_memory_runtime import write_new_json
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_MIKASA_MEMORY"))
runner = importlib.import_module("event_factorized_mapping_diagnosis")


def _sealed_mapping(root: Path, provenance: dict) -> dict:
    materials = {}
    for condition in CONDITIONS:
        payload = {
            "cal_group": np.repeat(factorized.CALIBRATION_GROUPS, 6).astype(np.int64),
            "cal_row": np.tile(np.arange(6), 6).astype(np.int64),
        }
        for prefix, dimension in (
            ("shape", factorized.SHAPE_DIM),
            ("checkpoint", factorized.CHECKPOINT_DIM),
        ):
            payload[f"{prefix}_mean"] = np.zeros(dimension, np.float32)
            payload[f"{prefix}_std"] = np.ones(dimension, np.float32)
            payload[f"{prefix}_neighbor_group"] = np.full(36, factorized.FIT_GROUPS[0], np.int64)
            payload[f"{prefix}_neighbor_row"] = np.zeros(36, np.int64)
            payload[f"{prefix}_distance"] = np.zeros(36, np.float32)
        if condition.startswith("shuffle_"):
            seed = int(condition.split("_")[1])
            for group in factorized.FIT_GROUPS + factorized.CALIBRATION_GROUPS:
                payload[f"permutation_{group}"] = group_permutations(seed, group)
        path = root / f"{condition}.npz"
        body = factorized._npz_bytes(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        materials[condition] = {
            "path": str(path.resolve()),
            "bytes": len(body),
            "sha256": __import__("hashlib").sha256(body).hexdigest(),
        }
    result = {
        "status": "passed",
        "mode": "factorized_event_mapping",
        "schema": factorized.SCHEMA,
        "mapping_root": str(root.resolve()),
        "materials": materials,
        "all_descriptors_distances_finite": True,
        "provenance": provenance,
        "targets_opened": False,
        "dev_opened": False,
        "confirmation_opened": False,
    }
    (root / "RESULT.json").write_text(json.dumps(result))
    return result


def test_cpu_smoke_reports_exact_factorized_dimensions(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(sys, "argv", ["event_factorized_mapping_diagnosis.py", "--smoke"])
    assert runner.main() == 0
    output = capsys.readouterr().out
    assert '"shape_dim": 780' in output
    assert '"checkpoint_dim": 1047' in output


def test_repeat_rejected_before_receipt_or_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    calls = {"read": 0, "target": 0}

    def blocked_read(*args, **kwargs):
        calls["read"] += 1
        raise AssertionError("receipt opened")

    monkeypatch.setattr(runner, "_read", blocked_read)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event_factorized_mapping_diagnosis.py",
            "--score",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--corpus-result",
            str(tmp_path / "corpus-result"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--cache-result",
            str(tmp_path / "cache-result"),
            "--fused-map-result",
            str(tmp_path / "fused-map"),
            "--fused-score-result",
            str(tmp_path / "fused-score"),
            "--mapping-root",
            str(tmp_path / "mapping"),
            "--mapping-result",
            str(tmp_path / "mapping-result"),
            "--expected-mapping-sha256",
            "1" * 64,
            "--output-root",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError):
        runner.main()
    assert calls == {"read": 0, "target": 0}


def test_write_once_and_dev_confirmation_rejection(tmp_path: Path) -> None:
    result = tmp_path / "RESULT.json"
    write_new_json(result, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_new_json(result, {"status": "changed"})
    for name in ("dev", "confirmation"):
        with pytest.raises(PermissionError):
            runner.reject_path(tmp_path / name / "secret")


def test_runner_exposes_only_map_and_score_formal_stages() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'mode.add_argument("--map"' in source
    assert 'mode.add_argument("--score"' in source
    assert "verify_privileged_targets=False" in source


def test_mapping_receipt_rejects_provenance_boundary_and_extra_file_before_score(
    tmp_path: Path,
) -> None:
    provenance = {"corpus": {"sha256": "a" * 64}, "cache": {"sha256": "b" * 64}}
    root = tmp_path / "mapping"
    result = _sealed_mapping(root, provenance)
    factorized.FactorizedMapping.discover(root, result, provenance)
    changed = json.loads(json.dumps(result))
    changed["provenance"]["cache"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt"):
        factorized.FactorizedMapping.discover(root, changed, provenance)
    changed = json.loads(json.dumps(result))
    changed["dev_opened"] = True
    with pytest.raises(ValueError, match="receipt"):
        factorized.FactorizedMapping.discover(root, changed, provenance)
    (root / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="entry set"):
        factorized.FactorizedMapping.discover(root, result, provenance)


def test_score_runner_provenance_mutation_causes_zero_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "corpus": {"path": "corpus", "bytes": 1, "sha256": "a" * 64},
        "cache": {"path": "cache", "bytes": 1, "sha256": "b" * 64},
        "fused_map": {"path": "map", "bytes": 1, "sha256": "c" * 64},
        "fused_score": {"path": "score", "bytes": 1, "sha256": "d" * 64},
    }
    mapping_root = tmp_path / "mapping"
    mapping_result = _sealed_mapping(mapping_root, expected)
    mapping_result["provenance"]["cache"]["sha256"] = "0" * 64
    score_calls = []

    def fake_read(path, digest, status):
        name = path.name
        if name == "mapping-result":
            return mapping_result, {"path": str(path), "bytes": 1, "sha256": digest}
        key = {
            "corpus-result": "corpus",
            "cache-result": "cache",
            "fused-map": "fused_map",
            "fused-score": "fused_score",
        }[name]
        return {"status": status}, expected[key]

    monkeypatch.setattr(runner, "_read", fake_read)
    monkeypatch.setattr(runner.ShardedMemoryCorpus, "discover", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner.EventCache, "discover", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner,
        "score_factorized",
        lambda *args, **kwargs: score_calls.append(True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event_factorized_mapping_diagnosis.py",
            "--score",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--corpus-result",
            str(tmp_path / "corpus-result"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--cache-result",
            str(tmp_path / "cache-result"),
            "--fused-map-result",
            str(tmp_path / "fused-map"),
            "--fused-score-result",
            str(tmp_path / "fused-score"),
            "--mapping-root",
            str(mapping_root),
            "--mapping-result",
            str(tmp_path / "mapping-result"),
            "--expected-mapping-sha256",
            "1" * 64,
            "--output-root",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(ValueError, match="cache provenance"):
        runner.main()
    assert score_calls == []
