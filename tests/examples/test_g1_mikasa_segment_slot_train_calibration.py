from __future__ import annotations

import importlib
from pathlib import Path
import sys

from gr00t.experimental.mikasa_native_rgb_memory_runtime import write_new_json
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_MIKASA_MEMORY"
sys.path.insert(0, str(EXAMPLE))
runner = importlib.import_module("segment_slot_train_calibration")


def test_cpu_smoke_preserves_model_loss_and_split_contract() -> None:
    result = runner.synthetic_cpu_smoke()
    assert result["status"] == "passed"
    assert result["shape"] == [2, 2, 3]
    assert result["checkpoint_shape"] == [2, 2, 24]
    assert result["finite"] is True
    assert result["split"]["canonical_split_sha256"] == (
        "c9fd3d603d02a73182e5649c6a9a9477af4f6758d398d9126acdc844b33246d2"
    )
    assert result["mamba_loaded"] is False
    assert result["native_vla_loaded"] is False


def test_immutable_model_loss_and_runtime_sources_match_base() -> None:
    assert runner.verify_immutable_slot_sources() == runner.FROZEN_SOURCE_SHA256


@pytest.mark.parametrize("name", ("dev", "confirmation"))
def test_original_dev_and_confirmation_paths_are_rejected(name: str, tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="forbidden"):
        runner.reject_non_train_path(tmp_path / name / "secret")
    runner.reject_non_train_path(tmp_path / "train-dev-corpus-gpu0-r1")


def test_output_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    write_new_json(path, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"status": "changed"})


def test_runner_exposes_one_formal_mode_and_final_checkpoints_only() -> None:
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    module_source = (
        ROOT / "gr00t/experimental/mikasa_segment_slot_train_calibration.py"
    ).read_text(encoding="utf-8")
    assert 'mode.add_argument("--train-calibration"' in runner_source
    assert 'mode.add_argument("--dev"' not in runner_source
    assert 'mode.add_argument("--confirmation"' not in runner_source
    assert '"checkpoint_final.pt"' in module_source
    assert "checkpoint-{step}" not in module_source
    assert "MikasaSegmentSlotModel" in module_source
    assert "segment_slot_loss" in module_source
