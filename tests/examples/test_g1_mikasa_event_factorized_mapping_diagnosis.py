from __future__ import annotations

import importlib
from pathlib import Path
import sys

from gr00t.experimental.mikasa_native_rgb_memory_runtime import write_new_json
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_MIKASA_MEMORY"))
runner = importlib.import_module("event_factorized_mapping_diagnosis")


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
