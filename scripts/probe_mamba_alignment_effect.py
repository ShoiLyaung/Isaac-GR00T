from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t_wbc.control.utils.n1_utils import concat_action

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"


def _json_default(obj: Any):
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_env(n_action_steps: int, max_episode_steps: int):
    return gym.vector.SyncVectorEnv(
        [
            lambda: create_eval_env(
                env_name=ENV_NAME,
                env_idx=0,
                total_n_envs=1,
                wrapper_configs=WrapperConfigs(
                    video=VideoConfig(video_dir=None, max_episode_steps=max_episode_steps),
                    multistep=MultiStepConfig(
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps,
                        terminate_on_success=True,
                    ),
                ),
            )
        ]
    )


def _make_policy(model_path: str) -> Gr00tSimPolicyWrapper:
    return Gr00tSimPolicyWrapper(
        Gr00tPolicy(
            embodiment_tag=EmbodimentTag.UNITREE_G1,
            model_path=model_path,
            device=0,
        )
    )


def _first_step_action(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[0, 0], dtype=np.float32) for key, value in action.items()}


def _collect_group_stats(robot_model, action: dict[str, np.ndarray]) -> dict[str, Any]:
    wbc_goal = concat_action(robot_model, action)
    return {
        "navigate_command": action["action.navigate_command"].tolist(),
        "left_arm_mean_abs": float(np.mean(np.abs(action["action.left_arm"]))),
        "right_arm_mean_abs": float(np.mean(np.abs(action["action.right_arm"]))),
        "right_hand_mean_abs": float(np.mean(np.abs(action["action.right_hand"]))),
        "target_upper_body_pose_abs_max": float(
            np.max(np.abs(np.asarray(wbc_goal["target_upper_body_pose"], dtype=np.float32)))
        ),
    }


def ratio(num: float, den: float) -> float | None:
    if den == 0.0:
        return None
    return num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-model-path", required=True)
    parser.add_argument("--mamba-model-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-prefix", default="reinit_20260326")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=1440)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args.n_action_steps, args.max_episode_steps)
    obs, _ = env.reset(seed=args.seed)
    robot_model = env.envs[0].env.robot_model

    dit_policy = _make_policy(args.dit_model_path)
    mamba_policy = _make_policy(args.mamba_model_path)

    steps = []
    for outer_step in range(args.outer_steps):
        dit_action, _ = dit_policy.get_action(obs)

        mamba_policy.policy.model.config.mamba_inference_align_actions = False
        mamba_raw_action, _ = mamba_policy.get_action(obs)

        mamba_policy.policy.model.config.mamba_inference_align_actions = True
        mamba_aligned_action, _ = mamba_policy.get_action(obs)

        dit_first = _first_step_action(dit_action)
        mamba_raw_first = _first_step_action(mamba_raw_action)
        mamba_aligned_first = _first_step_action(mamba_aligned_action)

        dit_stats = _collect_group_stats(robot_model, dit_first)
        mamba_raw_stats = _collect_group_stats(robot_model, mamba_raw_first)
        mamba_aligned_stats = _collect_group_stats(robot_model, mamba_aligned_first)

        steps.append(
            {
                "outer_step": outer_step,
                "dit": dit_stats,
                "mamba_raw": mamba_raw_stats,
                "mamba_aligned": mamba_aligned_stats,
                "ratios": {
                    "raw_over_dit_left_arm": ratio(
                        mamba_raw_stats["left_arm_mean_abs"], dit_stats["left_arm_mean_abs"]
                    ),
                    "aligned_over_raw_left_arm": ratio(
                        mamba_aligned_stats["left_arm_mean_abs"], mamba_raw_stats["left_arm_mean_abs"]
                    ),
                    "raw_over_dit_right_arm": ratio(
                        mamba_raw_stats["right_arm_mean_abs"], dit_stats["right_arm_mean_abs"]
                    ),
                    "aligned_over_raw_right_arm": ratio(
                        mamba_aligned_stats["right_arm_mean_abs"], mamba_raw_stats["right_arm_mean_abs"]
                    ),
                    "raw_over_dit_right_hand": ratio(
                        mamba_raw_stats["right_hand_mean_abs"], dit_stats["right_hand_mean_abs"]
                    ),
                    "aligned_over_raw_right_hand": ratio(
                        mamba_aligned_stats["right_hand_mean_abs"], mamba_raw_stats["right_hand_mean_abs"]
                    ),
                    "raw_over_dit_target_upper_body_pose": ratio(
                        mamba_raw_stats["target_upper_body_pose_abs_max"],
                        dit_stats["target_upper_body_pose_abs_max"],
                    ),
                    "aligned_over_raw_target_upper_body_pose": ratio(
                        mamba_aligned_stats["target_upper_body_pose_abs_max"],
                        mamba_raw_stats["target_upper_body_pose_abs_max"],
                    ),
                    "raw_over_dit_nav_x": ratio(
                        abs(mamba_raw_stats["navigate_command"][0]),
                        abs(dit_stats["navigate_command"][0]),
                    ),
                    "aligned_over_raw_nav_x": ratio(
                        abs(mamba_aligned_stats["navigate_command"][0]),
                        abs(mamba_raw_stats["navigate_command"][0]),
                    ),
                },
            }
        )

        obs, rewards, terminations, truncations, infos = env.step(dit_action)
        if bool(terminations[0]) or bool(truncations[0]):
            break

    env.close()

    def mean_ratio(key: str) -> float:
        values = [step["ratios"][key] for step in steps if step["ratios"][key] is not None]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "n_steps": len(steps),
        "mean_ratios": {
            "raw_over_dit_left_arm": mean_ratio("raw_over_dit_left_arm"),
            "aligned_over_raw_left_arm": mean_ratio("aligned_over_raw_left_arm"),
            "raw_over_dit_right_arm": mean_ratio("raw_over_dit_right_arm"),
            "aligned_over_raw_right_arm": mean_ratio("aligned_over_raw_right_arm"),
            "raw_over_dit_right_hand": mean_ratio("raw_over_dit_right_hand"),
            "aligned_over_raw_right_hand": mean_ratio("aligned_over_raw_right_hand"),
            "raw_over_dit_target_upper_body_pose": mean_ratio(
                "raw_over_dit_target_upper_body_pose"
            ),
            "aligned_over_raw_target_upper_body_pose": mean_ratio(
                "aligned_over_raw_target_upper_body_pose"
            ),
            "raw_over_dit_nav_x": mean_ratio("raw_over_dit_nav_x"),
            "aligned_over_raw_nav_x": mean_ratio("aligned_over_raw_nav_x"),
        },
        "step0": steps[0] if steps else None,
    }

    json_path = output_dir / f"mamba_alignment_probe_{args.output_prefix}.json"
    md_path = output_dir / f"mamba_alignment_probe_{args.output_prefix}.md"
    json_path.write_text(json.dumps({"steps": steps, "summary": summary}, indent=2, default=_json_default))

    lines = ["# Mamba Alignment Probe", "## Mean Ratios"]
    for key, value in summary["mean_ratios"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Step 0")
    for key, value in summary["step0"].items():
        lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
