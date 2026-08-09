from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gr00t.data.dataset.g1_id25_h50_adapter import (
    ACTION_GROUPS,
    EMBODIMENT_TAG,
    HORIZON,
    LANGUAGE,
    STATE_GROUPS,
)
from gr00t.eval.g1_id25_h50_open_loop import (
    evaluate_g1_id25_h50_anchor,
    load_frozen_modality_config,
    validate_loaded_policy_contract,
)
from gr00t.eval.sim.UnitreeIsaacLab.g1_id25_h50_insertion import (
    EvaluatorConfig,
    TerminationCause,
    build_full_joint_target,
    classify_termination,
    evaluate_insertion,
    sim_observation_to_policy,
    validate_policy_modality_config,
)
from gr00t.policy import BasePolicy


CONFIG = Path(__file__).resolve().parents[2] / "examples/G1_ID25_H50/g1_id25_h50_config.py"


def _decoded_chunk(*, ramp: bool = False) -> dict[str, np.ndarray]:
    left_arm = np.zeros((1, HORIZON, 7), dtype=np.float32)
    if ramp:
        left_arm[0, :, 0] = np.arange(HORIZON, dtype=np.float32)
    return {
        "left_arm": left_arm,
        "right_arm": np.zeros((1, HORIZON, 7), dtype=np.float32),
        "left_hand": np.ones((1, HORIZON, 6), dtype=np.float32),
        "right_hand": np.ones((1, HORIZON, 6), dtype=np.float32),
        "waist": np.zeros((1, HORIZON, 3), dtype=np.float32),
        "base_height_command": np.zeros((1, HORIZON, 1), dtype=np.float32),
        "navigate_command": np.zeros((1, HORIZON, 3), dtype=np.float32),
    }


class FakePolicy(BasePolicy):
    def __init__(self, *, ramp: bool = False, error: BaseException | None = None):
        super().__init__(strict=False)
        self.ramp = ramp
        self.error = error

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def _get_action(self, observation, options=None):
        if self.error is not None:
            raise self.error
        return _decoded_chunk(ramp=self.ramp), {"fixture": True}

    def reset(self, options=None):
        return {}


def _sim_observation() -> dict:
    return {
        "policy": {
            "robot_joint_state": np.zeros((1, 58), dtype=np.float32),
            "robot_inspire_state": np.zeros((1, 12), dtype=np.float32),
            "head_rgb": np.zeros((1, 8, 10, 4), dtype=np.uint8),
        }
    }


class FakeRuntime:
    def __init__(self, *, success_at: int | None = None):
        self.success_at = success_at
        self.targets = []
        self.closed = False

    def reset(self, *, seed: int):
        return _sim_observation(), {"seed": seed}

    def step(self, full_joint_target):
        self.targets.append(np.asarray(full_joint_target).copy())
        success = self.success_at is not None and len(self.targets) == self.success_at
        info = {"termination_terms": {"success": success}}
        return _sim_observation(), 0.0, success, False, info

    def close(self):
        self.closed = True


class FrameLoader:
    def __init__(self):
        self.modality_configs = load_frozen_modality_config(CONFIG)
        rows = HORIZON
        data = {
            "video.ego_view": [np.zeros((8, 10, 3), dtype=np.uint8) for _ in range(rows)],
            "language.annotation.human.task_description": [LANGUAGE] * rows,
        }
        dims = {
            "left_leg": 6,
            "right_leg": 6,
            "waist": 3,
            "left_arm": 7,
            "right_arm": 7,
            "left_hand": 6,
            "right_hand": 6,
        }
        for key in STATE_GROUPS:
            data[f"state.{key}"] = [np.zeros(dims[key], dtype=np.float32) for _ in range(rows)]
        action_dims = dict(
            left_arm=7,
            right_arm=7,
            left_hand=6,
            right_hand=6,
            waist=3,
            base_height_command=1,
            navigate_command=3,
        )
        for key in ACTION_GROUPS:
            fill = 1.0 if key.endswith("hand") else 0.0
            data[f"action.{key}"] = [
                np.full(action_dims[key], fill, dtype=np.float32) for _ in range(rows)
            ]
        self.frame = pd.DataFrame(data)

    def __getitem__(self, index):
        assert index == 0
        return self.frame


def test_open_loop_uses_h50_adapter_semantics() -> None:
    result = evaluate_g1_id25_h50_anchor(FakePolicy(), FrameLoader())
    assert result["status"] == "OPEN_LOOP_CONNECTIVITY_PASS_NO_QUALITY_CLAIM"
    assert result["prediction_shape"] == [50, 26]
    assert result["mse"] == 0.0
    assert result["source_offsets"][:6] == [0, 0, 1, 1, 2, 3]
    assert result["true_50hz_capture"] is False


def test_reload_and_server_modality_guards() -> None:
    modality = load_frozen_modality_config(CONFIG)
    validate_policy_modality_config(modality)

    class Value:
        pass

    policy = Value()
    policy.model = Value()
    policy.model.config = Value()
    policy.model.config.action_horizon = 50
    policy.processor = Value()
    policy.processor.max_action_horizon = 50
    policy.processor.embodiment_id_mapping = {EMBODIMENT_TAG: 25}
    policy.modality_configs = modality
    validate_loaded_policy_contract(policy)

    policy.model.config.action_horizon = 40
    with pytest.raises(RuntimeError):
        validate_loaded_policy_contract(policy)
    bad_modality = dict(modality)
    bad_modality.pop("language")
    with pytest.raises(RuntimeError):
        validate_policy_modality_config(bad_modality)


def test_sim_observation_and_ingress_contract() -> None:
    observation = sim_observation_to_policy(_sim_observation())
    assert observation["video"]["ego_view"].shape == (1, 1, 8, 10, 3)
    assert observation["state"]["left_arm"].shape == (1, 1, 7)
    assert observation["language"]["annotation.human.task_description"] == [[LANGUAGE]]
    target = build_full_joint_target(np.r_[np.arange(14), np.ones(12)].astype(np.float32))
    assert target.shape == (53,)
    assert target[11] == 0.0
    assert target[12] == 7.0


def test_evaluator_executes_each_h50_tick_once_without_second_zoh() -> None:
    runtime = FakeRuntime(success_at=3)
    result = evaluate_insertion(
        FakePolicy(ramp=True),
        runtime,
        EvaluatorConfig(n_episodes=1, max_episode_steps=10, n_action_steps=HORIZON),
    )
    assert runtime.closed
    assert [target[11] for target in runtime.targets] == [0.0, 1.0, 2.0]
    assert result["runtime_action_index_formula"] == "chunk_index=k"
    assert result["runtime_second_zoh_applied"] is False
    assert result["episodes"][0]["termination_cause"] == "success"


@pytest.mark.parametrize(
    ("terminated", "truncated", "terms", "expected"),
    [
        (True, False, {"success": True}, TerminationCause.SUCCESS),
        (True, False, {"shaft_out_of_bound": True}, TerminationCause.SHAFT_OUT_OF_BOUNDS),
        (True, False, {"housing_out_of_bound": True}, TerminationCause.HOUSING_OUT_OF_BOUNDS),
        (False, True, {}, TerminationCause.SIMULATOR_TIME_LIMIT),
        (True, False, {}, TerminationCause.UNKNOWN_TERMINATION),
        (
            True,
            False,
            {"success": True, "shaft_out_of_bound": True},
            TerminationCause.AMBIGUOUS_TERMINATION,
        ),
    ],
)
def test_typed_termination(terminated, truncated, terms, expected) -> None:
    assert classify_termination(terminated, truncated, {"termination_terms": terms}) is expected


def test_policy_timeout_is_typed_and_runtime_closes() -> None:
    runtime = FakeRuntime()
    result = evaluate_insertion(FakePolicy(error=TimeoutError()), runtime, EvaluatorConfig())
    assert runtime.closed
    assert result["episodes"][0]["termination_cause"] == "policy_timeout"


def test_config_rejects_non_contract_values() -> None:
    with pytest.raises(ValueError):
        EvaluatorConfig(task="wrong").validate()
    with pytest.raises(ValueError):
        EvaluatorConfig(n_action_steps=51).validate()
    with pytest.raises(ValueError):
        EvaluatorConfig(n_episodes=True).validate()
