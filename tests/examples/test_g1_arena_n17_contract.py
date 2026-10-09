from __future__ import annotations

from pathlib import Path
import sys

from gr00t.data.types import ActionRepresentation, ActionType
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_ARENA_N17"
sys.path.insert(0, str(EXAMPLE))
import arena_config as config  # noqa: E402
import cpu_contract_probe as probe  # noqa: E402


def test_config_h40_and_absolute_actions():
    assert config.STATE_KEYS == ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
    assert config.ACTION_KEYS[-2:] == ("base_height_command", "navigate_command")
    action = config.ARENA_G1_N17_CONFIG["action"]
    assert action.delta_indices == list(range(40))
    assert len(action.action_configs) == 7
    assert all(item.rep == ActionRepresentation.ABSOLUTE for item in action.action_configs)
    assert all(item.type == ActionType.NON_EEF for item in action.action_configs)
    assert config.EXECUTE_HORIZON == 16


def test_whitelist_rejects_privileged_columns():
    expected = [
        *(f"state.{key}" for key in config.STATE_KEYS),
        *(f"action.{key}" for key in config.ACTION_KEYS),
        "language.annotation.human.task_description",
    ]
    probe.verify_whitelist(expected)
    with pytest.raises(ValueError, match="leakage"):
        probe.verify_whitelist(expected + ["state.object_pose"])


def test_schedule_and_roundtrip_helper():
    assert probe.FRAMES // 16 == 57
    assert probe.FRAMES % 16 == 13
    first = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    second = {"x": first["x"] + np.asarray([[1e-6, -1e-6]], dtype=np.float32)}
    assert probe.max_roundtrip_error(first, second) == pytest.approx(1e-6, rel=0, abs=1e-7)
