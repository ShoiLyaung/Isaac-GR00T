"""Simulator-facing mapping must exclude velocity/torque and preserve q units."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[3] / "examples/G1_DIT_MAMBA_BASELINE/insertion_bridge.py"
SPEC = importlib.util.spec_from_file_location("dit_mamba_qonly_bridge", PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def test_only_q_can_affect_body_input():
    qdq = np.arange(58, dtype=np.float32)
    original = bridge.combine_body_telemetry(qdq, np.ones(29, dtype=np.float32))
    altered = qdq.copy()
    altered[29:] = np.nan
    ignored_tau = np.full(29, np.nan, dtype=np.float32)
    assert np.array_equal(original, bridge.combine_body_telemetry(altered, ignored_tau))
    assert original.shape == (29,)


def test_observation_contains_exact_q_groups_and_both_hands():
    q = np.arange(29, dtype=np.float32)
    hand = np.concatenate((bridge.HAND_MIN, bridge.HAND_MAX))
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    observation = bridge.observation_from_sim(q, hand, rgb)
    state = observation["state"]
    assert set(state) == {
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    assert np.array_equal(state["left_arm"][0, 0], q[15:22])
    assert np.array_equal(state["right_arm"][0, 0], q[22:29])
    assert np.array_equal(state["left_hand"], np.zeros((1, 1, 6), dtype=np.float32))
    assert np.array_equal(state["right_hand"], np.ones((1, 1, 6), dtype=np.float32))
    assert all(value.dtype == np.float32 for value in state.values())
    assert observation["video"]["ego_view"].shape == (1, 1, 480, 640, 3)
    with pytest.raises(ValueError, match=r"q\[29\]"):
        bridge.observation_from_sim(np.zeros(87), hand, rgb)


def test_left_action_is_absolute_and_other_joint_offsets_stay_zero():
    names = (
        list(bridge.ARM_NAMES)
        + list(bridge.HAND_NAMES)
        + list(bridge.HAND_DEPENDENCIES)
        + ["right_arm_joint", "waist_joint"]
    )
    default = np.linspace(0.01, 0.2, len(names), dtype=np.float32)
    arm = np.linspace(-0.4, 0.5, 7, dtype=np.float32)
    hand = np.linspace(0, 1, 6, dtype=np.float32)
    result = bridge.action_to_sim(arm, hand, names, default)
    assert np.allclose(result[:7] + default[:7], arm)
    assert np.allclose(result[7:13] + default[7:13], bridge.hand_u_to_radians(hand))
    assert np.array_equal(result[-2:], np.zeros(2, dtype=np.float32))


def test_frozen_r3_compatibility_entrypoint_matches_qonly():
    path = PATH.parents[1] / "G1_NATIVE_BASELINE/insertion_bridge.py"
    spec = importlib.util.spec_from_file_location("frozen_r3_qonly_bridge", path)
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    assert compatibility.PROTOCOL_ID == bridge.PROTOCOL_ID
    assert compatibility.ARM_NAMES == bridge.ARM_NAMES
    assert compatibility.HAND_NAMES == bridge.HAND_NAMES
    assert np.array_equal(
        compatibility.combine_body_telemetry(np.arange(58), np.ones(29)), np.arange(29)
    )


def test_policy_reset_does_not_change_global_rng():
    import torch

    path = PATH.parent / "serve_checkpoint.py"
    spec = importlib.util.spec_from_file_location("test_compact_qonly_server", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    policy = object.__new__(server.CompactHeadPolicy)
    policy.policy_seed = 42
    policy.calls = 17
    before = torch.random.get_rng_state().clone()
    result = policy.reset({"seed": 999})
    assert torch.equal(before, torch.random.get_rng_state())
    assert not result["rng_reseeded"]
    assert policy.calls == 0
