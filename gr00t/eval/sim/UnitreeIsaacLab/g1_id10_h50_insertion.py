# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named G1 ID10/H50 insertion evaluator and 26-D -> 53-D ingress.

The core is simulator-package agnostic and CPU testable. The runtime adapter
is supplied by the authorized launcher. Training-data ZOH is not applied a
second time here: one decoded H50 target is consumed per 50 Hz simulator tick.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Any, Protocol

import numpy as np

from gr00t.data.dataset.g1_id10_h50_adapter import (
    ACTION_GROUPS,
    ACTIVE_ACTION_DIM,
    BASELINE_ID,
    EMBODIMENT_TAG,
    HORIZON,
    LANGUAGE,
    STATE_GROUPS,
    TEMPORAL_OPTION,
    h50_zoh_offsets,
    inspire_rad_to_u,
    resolve_decoded_action_26d,
    scatter_inspire_u_to_sim_joint_targets,
)
from gr00t.policy import BasePolicy


TASK_NAME = "Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"
RESULT_SCHEMA = "g1-id10-h50-insertion-evaluator-result-v1"
FULL_SIM_ACTION_DIM = 53
SIM_CONTROL_RATE_HZ = 50
SIM_DT_SECONDS = 0.005
SIM_DECIMATION = 4
SUCCESS_DISTANCE_METERS = 0.05

LEFT_ARM_SIM_INDICES = (11, 15, 19, 21, 23, 25, 27)
RIGHT_ARM_SIM_INDICES = (12, 16, 20, 22, 24, 26, 28)
LEFT_ARM_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_NAMES = tuple(name.replace("left_", "right_", 1) for name in LEFT_ARM_NAMES)


class TerminationCause(str, Enum):
    SUCCESS = "success"
    SHAFT_OUT_OF_BOUNDS = "shaft_out_of_bounds"
    HOUSING_OUT_OF_BOUNDS = "housing_out_of_bounds"
    SIMULATOR_TIME_LIMIT = "simulator_time_limit"
    EVALUATOR_STEP_LIMIT = "evaluator_step_limit"
    POLICY_TIMEOUT = "policy_timeout"
    EVALUATOR_EXCEPTION = "evaluator_exception"
    AMBIGUOUS_TERMINATION = "ambiguous_termination"
    UNKNOWN_TERMINATION = "unknown_termination"


@dataclass(frozen=True)
class EvaluatorConfig:
    task: str = TASK_NAME
    n_episodes: int = 1
    max_episode_steps: int = 1000
    n_action_steps: int = HORIZON
    policy_timeout_ms: int = 15000
    seed: int = 42

    def validate(self) -> None:
        if self.task != TASK_NAME:
            raise ValueError(f"Only the frozen task is accepted: {self.task}")
        for name, value in (
            ("n_episodes", self.n_episodes),
            ("max_episode_steps", self.max_episode_steps),
            ("n_action_steps", self.n_action_steps),
            ("policy_timeout_ms", self.policy_timeout_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_action_steps > HORIZON:
            raise ValueError("n_action_steps cannot exceed the native H50 chunk")


class SingleEnvRuntime(Protocol):
    """Minimal clean boundary implemented by the authorized IsaacLab launcher."""

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self, full_joint_target: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


def validate_policy_modality_config(modality: dict[str, Any]) -> None:
    """Validate the policy-server processor contract before simulator reset."""

    required = {"video", "state", "action", "language"}
    if set(modality) != required:
        raise RuntimeError(f"Policy modality keys differ from G1 contract: {sorted(modality)}")
    if modality["video"].modality_keys != ["ego_view"]:
        raise RuntimeError("Policy camera is not the single accepted ego_view")
    if modality["state"].modality_keys != STATE_GROUPS:
        raise RuntimeError("Policy state-group order differs from G1 contract")
    if modality["action"].modality_keys != ACTION_GROUPS:
        raise RuntimeError("Policy action-group order differs from G1 contract")
    if modality["action"].delta_indices != h50_zoh_offsets():
        raise RuntimeError("Policy action indices differ from frozen H50 ZOH contract")
    if modality["language"].modality_keys != ["annotation.human.task_description"]:
        raise RuntimeError("Policy language key differs from G1 contract")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_env(value: Any, *, name: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == 0:
        raise ValueError(f"{name} must include an environment dimension")
    if array.shape[0] != 1:
        raise ValueError(f"{name} requires exactly one environment, got {array.shape}")
    return array[0]


def sim_observation_to_policy(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert the exact Play-task observation to the frozen policy surface."""

    policy_obs = observation.get("policy", observation)
    required = {"robot_joint_state", "robot_inspire_state", "head_rgb"}
    missing = required.difference(policy_obs)
    if missing:
        raise KeyError(f"Simulator observation missing: {sorted(missing)}")

    body = _single_env(policy_obs["robot_joint_state"], name="robot_joint_state")
    if body.ndim != 1 or body.shape[0] < 29 or not np.isfinite(body[:29]).all():
        raise ValueError("robot_joint_state must contain finite 29-D positions")
    body = body[:29].astype(np.float32, copy=False)

    hands_rad = _single_env(policy_obs["robot_inspire_state"], name="robot_inspire_state")
    if hands_rad.shape != (12,) or not np.isfinite(hands_rad).all():
        raise ValueError("robot_inspire_state must be finite [1,12]")
    # Simulator observation order is right-hand first, then left-hand.
    right_u = inspire_rad_to_u(hands_rad[:6])
    left_u = inspire_rad_to_u(hands_rad[6:])
    eps = 8 * np.finfo(np.float32).eps
    for name, value in (("left_hand", left_u), ("right_hand", right_u)):
        if np.any(value < -eps) or np.any(value > 1.0 + eps):
            raise ValueError(f"Simulator {name} radians map outside Inspire u range")
    left_u = np.clip(left_u, 0.0, 1.0)
    right_u = np.clip(right_u, 0.0, 1.0)

    image = _single_env(policy_obs["head_rgb"], name="head_rgb")
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"head_rgb must be HWC RGB/RGBA, got {image.shape}")
    image = image[..., :3]
    if image.dtype != np.uint8:
        if not np.isfinite(image).all():
            raise ValueError("head_rgb contains non-finite values")
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    state = {
        "left_leg": body[0:6],
        "right_leg": body[6:12],
        "waist": body[12:15],
        "left_arm": body[15:22],
        "right_arm": body[22:29],
        "left_hand": left_u.astype(np.float32),
        "right_hand": right_u.astype(np.float32),
    }
    return {
        "video": {"ego_view": image[None, None, ...]},
        "state": {key: value[None, None, ...] for key, value in state.items()},
        "language": {"annotation.human.task_description": [[LANGUAGE]]},
    }


def build_full_joint_target(active_action_26d: np.ndarray) -> np.ndarray:
    """Map one processor-decoded absolute 26-D action to the 53-D task surface."""

    active = np.asarray(active_action_26d, dtype=np.float32)
    if active.shape != (ACTIVE_ACTION_DIM,) or not np.isfinite(active).all():
        raise ValueError("Active action must be finite [26]")
    result = np.zeros(FULL_SIM_ACTION_DIM, dtype=np.float32)
    result[list(LEFT_ARM_SIM_INDICES)] = active[:7]
    result[list(RIGHT_ARM_SIM_INDICES)] = active[7:14]
    result = scatter_inspire_u_to_sim_joint_targets(result, active[14:20], active[20:26])
    if result.shape != (FULL_SIM_ACTION_DIM,) or not np.isfinite(result).all():
        raise AssertionError("Invalid 53-D simulator ingress")
    return result


def _flag(value: Any) -> bool:
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar done flag, got {array.shape}")
    return bool(array.reshape(-1)[0])


def classify_termination(
    terminated: Any, truncated: Any, info: dict[str, Any]
) -> TerminationCause | None:
    """Classify exact task terms without collapsing failure causes to bool."""

    is_terminated = _flag(terminated)
    is_truncated = _flag(truncated)
    if not is_terminated and not is_truncated:
        return None
    if is_truncated and not is_terminated:
        return TerminationCause.SIMULATOR_TIME_LIMIT

    terms = info.get("termination_terms", {})
    if not isinstance(terms, dict):
        raise ValueError("termination_terms must be a mapping")
    active = {name for name, value in terms.items() if bool(value)}
    known = {
        "success": TerminationCause.SUCCESS,
        "shaft_out_of_bound": TerminationCause.SHAFT_OUT_OF_BOUNDS,
        "housing_out_of_bound": TerminationCause.HOUSING_OUT_OF_BOUNDS,
    }
    resolved = [known[name] for name in known if name in active]
    if is_truncated:
        resolved.append(TerminationCause.SIMULATOR_TIME_LIMIT)
    if len(resolved) > 1:
        return TerminationCause.AMBIGUOUS_TERMINATION
    if len(resolved) == 1:
        return resolved[0]
    return TerminationCause.UNKNOWN_TERMINATION


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or exc.__class__.__name__ == "Again"


def _unbatch_decoded(action: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for key, value in action.items():
        array = np.asarray(value)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError(f"Decoded {key} requires batch size one, got {array.shape}")
            array = array[0]
        result[key] = array
    return result


def evaluate_insertion(
    policy: BasePolicy,
    runtime: SingleEnvRuntime,
    config: EvaluatorConfig,
) -> dict[str, Any]:
    """Run the named evaluator through a one-environment runtime boundary."""

    config.validate()
    episodes = []
    try:
        for episode_index in range(config.n_episodes):
            observation, reset_info = runtime.reset(seed=config.seed + episode_index)
            try:
                policy.reset()
            except NotImplementedError:
                pass
            env_steps = 0
            policy_calls = 0
            clamp_count = 0
            policy_latencies = []
            cause: TerminationCause | None = None
            error_type = None
            while env_steps < config.max_episode_steps and cause is None:
                started = time.perf_counter()
                try:
                    decoded, _ = policy.get_action(sim_observation_to_policy(observation))
                except Exception as exc:
                    policy_latencies.append(time.perf_counter() - started)
                    cause = (
                        TerminationCause.POLICY_TIMEOUT
                        if _is_timeout(exc)
                        else TerminationCause.EVALUATOR_EXCEPTION
                    )
                    error_type = exc.__class__.__name__
                    break
                policy_latencies.append(time.perf_counter() - started)
                policy_calls += 1
                active_chunk, chunk_clamps = resolve_decoded_action_26d(
                    _unbatch_decoded(decoded)
                )
                clamp_count += chunk_clamps

                # The dataset adapter already produced a nominal 50 Hz H50
                # target grid. Execute k=0,1,... exactly once at 50 Hz; do not
                # apply n+floor(3*k/5) again at runtime.
                for chunk_index in range(config.n_action_steps):
                    if env_steps >= config.max_episode_steps:
                        break
                    target = build_full_joint_target(active_chunk[chunk_index])
                    observation, _, terminated, truncated, info = runtime.step(target)
                    env_steps += 1
                    cause = classify_termination(terminated, truncated, info)
                    if cause is not None:
                        break
            if cause is None:
                cause = TerminationCause.EVALUATOR_STEP_LIMIT
            episodes.append(
                {
                    "episode_index": episode_index,
                    "seed": config.seed + episode_index,
                    "termination_cause": cause.value,
                    "success": cause is TerminationCause.SUCCESS,
                    "env_steps": env_steps,
                    "policy_calls": policy_calls,
                    "clamp_count": clamp_count,
                    "policy_latency_seconds": {
                        "count": len(policy_latencies),
                        "mean": float(np.mean(policy_latencies)) if policy_latencies else None,
                        "max": float(np.max(policy_latencies)) if policy_latencies else None,
                    },
                    "reset_info_keys": sorted(reset_info),
                    "error_type": error_type,
                }
            )
    finally:
        runtime.close()

    success_count = sum(item["success"] for item in episodes)
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "EVALUATOR_COMPLETE",
        "baseline_id": BASELINE_ID,
        "embodiment_tag": EMBODIMENT_TAG,
        "task": TASK_NAME,
        "success_distance_meters": SUCCESS_DISTANCE_METERS,
        "action_horizon": HORIZON,
        "active_action_dim": ACTIVE_ACTION_DIM,
        "full_sim_action_dim": FULL_SIM_ACTION_DIM,
        "sim_control_rate_hz": SIM_CONTROL_RATE_HZ,
        "temporal_option": TEMPORAL_OPTION,
        "training_source_index_formula": "n+floor(3*k/5)",
        "runtime_action_index_formula": "chunk_index=k",
        "runtime_second_zoh_applied": False,
        "true_50hz_capture": False,
        "config": asdict(config),
        "episodes": episodes,
        "summary": {
            "episode_count": len(episodes),
            "success_count": success_count,
            "success_rate": success_count / len(episodes),
            "termination_counts": {
                cause.value: sum(item["termination_cause"] == cause.value for item in episodes)
                for cause in TerminationCause
            },
        },
    }


def validate_isaaclab_runtime_surface(env: Any) -> None:
    """Validate the task surface before any policy action is applied."""

    if int(env.num_envs) != 1:
        raise RuntimeError("Named evaluator requires exactly one simulator environment")
    if not math.isclose(float(env.cfg.sim.dt), SIM_DT_SECONDS, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Simulator dt differs from frozen task contract")
    if int(env.cfg.decimation) != SIM_DECIMATION:
        raise RuntimeError("Simulator decimation differs from frozen task contract")
    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    if len(joint_names) != FULL_SIM_ACTION_DIM:
        raise RuntimeError("Simulator articulation is not the expected 53-D surface")
    expected_arms = {
        **dict(zip(LEFT_ARM_SIM_INDICES, LEFT_ARM_NAMES, strict=True)),
        **dict(zip(RIGHT_ARM_SIM_INDICES, RIGHT_ARM_NAMES, strict=True)),
    }
    observed_arms = {index: joint_names[index] for index in expected_arms}
    if observed_arms != expected_arms:
        raise RuntimeError(f"Simulator arm indices/names changed: {observed_arms}")
    action_term = env.action_manager.get_term("joint_pos")
    if (
        env.action_manager.active_terms != ["joint_pos"]
        or action_term.action_dim != FULL_SIM_ACTION_DIM
    ):
        raise RuntimeError(
            "Simulator action manager differs from the frozen 53-D joint-position surface"
        )
    if list(action_term._joint_names) != joint_names:
        raise RuntimeError("Action-term joint order differs from articulation order")


class IsaacLabSingleEnvRuntime:
    """Thin adapter around an already-created authorized IsaacLab environment."""

    def __init__(self, env: Any) -> None:
        validate_isaaclab_runtime_surface(env)
        self.env = env

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        value = self.env.reset(seed=seed)
        if isinstance(value, tuple) and len(value) == 2:
            return value
        return value, {}

    def step(
        self, full_joint_target: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import torch

        target = np.asarray(full_joint_target, dtype=np.float32)
        if target.shape != (FULL_SIM_ACTION_DIM,) or not np.isfinite(target).all():
            raise ValueError("IsaacLab ingress must be finite [53]")
        tensor = torch.as_tensor(target, dtype=torch.float32, device=self.env.device).reshape(1, -1)
        observation, reward, terminated, truncated, info = self.env.step(tensor)
        normalized_info = dict(info)
        log = normalized_info.get("log", {})
        normalized_info["termination_terms"] = {
            name: bool(log.get(f"Episode_Termination/{name}", 0.0))
            for name in ("success", "shaft_out_of_bound", "housing_out_of_bound")
        }
        reward_value = float(_numpy(reward).reshape(-1)[0])
        return observation, reward_value, _flag(terminated), _flag(truncated), normalized_info

    def close(self) -> None:
        self.env.close()
