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
    create_gr00t_sim_policy,
)
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


def run_trace(
    *,
    label: str,
    model_path: str,
    seed: int,
    outer_steps: int,
    n_action_steps: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    env = _make_env(n_action_steps=n_action_steps, max_episode_steps=max_episode_steps)
    obs, _ = env.reset(seed=seed)
    policy = create_gr00t_sim_policy(model_path, EmbodimentTag.UNITREE_G1)

    multistep_env = env.envs[0]
    wbc_env = multistep_env.env
    robot_model = wbc_env.robot_model
    raw_env = multistep_env.unwrapped

    left_arm_indices = robot_model.get_joint_group_indices("left_arm")
    right_arm_indices = robot_model.get_joint_group_indices("right_arm")
    right_hand_indices = robot_model.get_joint_group_indices("right_hand")
    upper_body_indices = robot_model.get_joint_group_indices("upper_body")

    left_arm_joint_names = [robot_model.joint_names[i] for i in left_arm_indices]
    right_arm_joint_names = [robot_model.joint_names[i] for i in right_arm_indices]
    right_hand_joint_names = [robot_model.joint_names[i] for i in right_hand_indices]
    upper_body_joint_names = [robot_model.joint_names[i] for i in upper_body_indices]

    steps: list[dict[str, Any]] = []
    for outer_step in range(outer_steps):
        action_chunk, _ = policy.get_action(obs)
        first_step_action = {
            key: np.asarray(value[0, 0], dtype=np.float32) for key, value in action_chunk.items()
        }
        first_step_wbc_goal = concat_action(robot_model, first_step_action)

        next_obs, rewards, terminations, truncations, infos = env.step(action_chunk)
        dq = np.asarray(next_obs["dq"][0, 0], dtype=np.float32)

        step_record = {
            "outer_step": outer_step,
            "reward": float(rewards[0]),
            "done": bool(terminations[0]),
            "truncated": bool(truncations[0]),
            "last_safety_ok": bool(getattr(raw_env, "last_safety_ok", True)),
            "sim_safety_warning_emitted": bool(
                getattr(raw_env, "_sim_safety_warning_emitted", False)
            ),
            "navigate_command_first_step": first_step_action["action.navigate_command"].tolist(),
            "left_arm_action_first_step": first_step_action["action.left_arm"].tolist(),
            "right_arm_action_first_step": first_step_action["action.right_arm"].tolist(),
            "right_hand_action_first_step": first_step_action["action.right_hand"].tolist(),
            "target_upper_body_pose_first_step": np.asarray(
                first_step_wbc_goal["target_upper_body_pose"], dtype=np.float32
            ).tolist(),
            "left_arm_dq": dq[left_arm_indices].tolist(),
            "right_arm_dq": dq[right_arm_indices].tolist(),
            "right_hand_dq": dq[right_hand_indices].tolist(),
            "left_arm_dq_abs_max": float(np.abs(dq[left_arm_indices]).max()),
            "right_arm_dq_abs_max": float(np.abs(dq[right_arm_indices]).max()),
            "right_hand_dq_abs_max": float(np.abs(dq[right_hand_indices]).max()),
            "obs_dq_abs_max": float(np.abs(dq).max()),
            "success_any": bool(np.any(infos["success"][0])) if "success" in infos else False,
        }
        steps.append(step_record)
        obs = next_obs
        if step_record["done"] or step_record["truncated"]:
            break

    env.close()
    return {
        "label": label,
        "seed": seed,
        "left_arm_joint_names": left_arm_joint_names,
        "right_arm_joint_names": right_arm_joint_names,
        "right_hand_joint_names": right_hand_joint_names,
        "upper_body_joint_names": upper_body_joint_names,
        "steps": steps,
    }


def _sign_match_count(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.sign(a) == np.sign(b)))


def summarize_pair(dit_trace: dict[str, Any], mamba_trace: dict[str, Any]) -> dict[str, Any]:
    dit_steps = dit_trace["steps"]
    mamba_steps = mamba_trace["steps"]
    n_steps = min(len(dit_steps), len(mamba_steps))
    if n_steps == 0:
        raise RuntimeError("No rollout steps collected")

    def stack(step_key: str, steps: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([step[step_key] for step in steps[:n_steps]], dtype=np.float32)

    summary = {}
    for group_name, joint_names in [
        ("left_arm", dit_trace["left_arm_joint_names"]),
        ("right_arm", dit_trace["right_arm_joint_names"]),
    ]:
        dit_action = stack(f"{group_name}_action_first_step", dit_steps)
        mamba_action = stack(f"{group_name}_action_first_step", mamba_steps)
        dit_dq = stack(f"{group_name}_dq", dit_steps)
        mamba_dq = stack(f"{group_name}_dq", mamba_steps)
        group_summary = []
        for idx, joint_name in enumerate(joint_names):
            dit_action_joint = dit_action[:, idx]
            mamba_action_joint = mamba_action[:, idx]
            dit_dq_joint = dit_dq[:, idx]
            mamba_dq_joint = mamba_dq[:, idx]
            dit_action_mean_abs = float(np.mean(np.abs(dit_action_joint)))
            mamba_action_mean_abs = float(np.mean(np.abs(mamba_action_joint)))
            dit_dq_mean_abs = float(np.mean(np.abs(dit_dq_joint)))
            mamba_dq_mean_abs = float(np.mean(np.abs(mamba_dq_joint)))
            group_summary.append(
                {
                    "joint": joint_name,
                    "action_sign_match_steps": _sign_match_count(
                        dit_action_joint, mamba_action_joint
                    ),
                    "action_mean_abs_dit": dit_action_mean_abs,
                    "action_mean_abs_mamba": mamba_action_mean_abs,
                    "action_abs_ratio_mamba_over_dit": (
                        None if dit_action_mean_abs == 0.0 else mamba_action_mean_abs / dit_action_mean_abs
                    ),
                    "dq_mean_abs_dit": dit_dq_mean_abs,
                    "dq_mean_abs_mamba": mamba_dq_mean_abs,
                    "dq_abs_ratio_mamba_over_dit": (
                        None if dit_dq_mean_abs == 0.0 else mamba_dq_mean_abs / dit_dq_mean_abs
                    ),
                }
            )
        summary[group_name] = group_summary

    dit_nav = stack("navigate_command_first_step", dit_steps)
    mamba_nav = stack("navigate_command_first_step", mamba_steps)
    dit_right_hand = stack("right_hand_action_first_step", dit_steps)
    mamba_right_hand = stack("right_hand_action_first_step", mamba_steps)
    dit_target = stack("target_upper_body_pose_first_step", dit_steps)
    mamba_target = stack("target_upper_body_pose_first_step", mamba_steps)

    global_summary = {
        "dit_obs_dq_abs_max_max": float(max(step["obs_dq_abs_max"] for step in dit_steps[:n_steps])),
        "mamba_obs_dq_abs_max_max": float(max(step["obs_dq_abs_max"] for step in mamba_steps[:n_steps])),
        "dit_left_arm_dq_abs_max_max": float(
            max(step["left_arm_dq_abs_max"] for step in dit_steps[:n_steps])
        ),
        "mamba_left_arm_dq_abs_max_max": float(
            max(step["left_arm_dq_abs_max"] for step in mamba_steps[:n_steps])
        ),
        "dit_right_arm_dq_abs_max_max": float(
            max(step["right_arm_dq_abs_max"] for step in dit_steps[:n_steps])
        ),
        "mamba_right_arm_dq_abs_max_max": float(
            max(step["right_arm_dq_abs_max"] for step in mamba_steps[:n_steps])
        ),
        "dit_all_safety_ok": bool(all(step["last_safety_ok"] for step in dit_steps[:n_steps])),
        "mamba_all_safety_ok": bool(all(step["last_safety_ok"] for step in mamba_steps[:n_steps])),
        "dit_any_safety_warning": bool(
            any(step["sim_safety_warning_emitted"] for step in dit_steps[:n_steps])
        ),
        "mamba_any_safety_warning": bool(
            any(step["sim_safety_warning_emitted"] for step in mamba_steps[:n_steps])
        ),
        "n_compared_outer_steps": n_steps,
    }

    return {
        "summary": summary,
        "navigate_command": {
            "dit_mean_abs": np.mean(np.abs(dit_nav), axis=0).tolist(),
            "mamba_mean_abs": np.mean(np.abs(mamba_nav), axis=0).tolist(),
            "mamba_over_dit_ratio": (
                np.mean(np.abs(mamba_nav), axis=0) / np.maximum(np.mean(np.abs(dit_nav), axis=0), 1e-8)
            ).tolist(),
        },
        "right_hand": {
            "dit_mean_abs_per_dim": np.mean(np.abs(dit_right_hand), axis=0).tolist(),
            "mamba_mean_abs_per_dim": np.mean(np.abs(mamba_right_hand), axis=0).tolist(),
            "mamba_over_dit_ratio_per_dim": (
                np.mean(np.abs(mamba_right_hand), axis=0)
                / np.maximum(np.mean(np.abs(dit_right_hand), axis=0), 1e-8)
            ).tolist(),
        },
        "target_upper_body_pose": {
            "dit_mean_abs_max": float(np.mean(np.max(np.abs(dit_target), axis=1))),
            "mamba_mean_abs_max": float(np.mean(np.max(np.abs(mamba_target), axis=1))),
            "mamba_over_dit_ratio": float(
                np.mean(np.max(np.abs(mamba_target), axis=1))
                / max(np.mean(np.max(np.abs(dit_target), axis=1)), 1e-8)
            ),
        },
        "global": global_summary,
    }


def write_markdown(summary: dict[str, Any], path: Path):
    lines = ["# Rollout Joint Diff Summary"]
    for group_name in ["left_arm", "right_arm"]:
        lines.append(f"## {group_name}")
        lines.append("| joint | sign-match steps | action abs ratio | dq abs ratio |")
        lines.append("|---|---:|---:|---:|")
        for row in summary["summary"][group_name]:
            action_ratio = row["action_abs_ratio_mamba_over_dit"]
            dq_ratio = row["dq_abs_ratio_mamba_over_dit"]
            lines.append(
                f"| {row['joint']} | {row['action_sign_match_steps']} / {summary['global']['n_compared_outer_steps']} | "
                f"{action_ratio:.2f} | {dq_ratio:.2f} |"
            )
        lines.append("")

    lines.append("## navigate_command")
    lines.append(f"- dit_mean_abs: {summary['navigate_command']['dit_mean_abs']}")
    lines.append(f"- mamba_mean_abs: {summary['navigate_command']['mamba_mean_abs']}")
    lines.append(
        f"- mamba_over_dit_ratio: {summary['navigate_command']['mamba_over_dit_ratio']}"
    )
    lines.append("")

    lines.append("## right_hand")
    lines.append(f"- dit_mean_abs_per_dim: {summary['right_hand']['dit_mean_abs_per_dim']}")
    lines.append(f"- mamba_mean_abs_per_dim: {summary['right_hand']['mamba_mean_abs_per_dim']}")
    lines.append(
        f"- mamba_over_dit_ratio_per_dim: {summary['right_hand']['mamba_over_dit_ratio_per_dim']}"
    )
    lines.append("")

    lines.append("## target_upper_body_pose")
    lines.append(
        f"- dit_mean_abs_max: {summary['target_upper_body_pose']['dit_mean_abs_max']}"
    )
    lines.append(
        f"- mamba_mean_abs_max: {summary['target_upper_body_pose']['mamba_mean_abs_max']}"
    )
    lines.append(
        f"- mamba_over_dit_ratio: {summary['target_upper_body_pose']['mamba_over_dit_ratio']}"
    )
    lines.append("")

    lines.append("## global")
    for key, value in summary["global"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-model-path", required=True)
    parser.add_argument("--mamba-model-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-prefix", default="reinit")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=1440)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dit_trace = run_trace(
        label="dit",
        model_path=args.dit_model_path,
        seed=args.seed,
        outer_steps=args.outer_steps,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )
    mamba_trace = run_trace(
        label="mamba_reinit",
        model_path=args.mamba_model_path,
        seed=args.seed,
        outer_steps=args.outer_steps,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )
    summary = summarize_pair(dit_trace, mamba_trace)

    dit_path = output_dir / f"joint_rollout_trace_dit_{args.output_prefix}.json"
    mamba_path = output_dir / f"joint_rollout_trace_mamba_{args.output_prefix}.json"
    summary_json_path = output_dir / f"joint_rollout_trace_compare_summary_{args.output_prefix}.json"
    summary_md_path = output_dir / f"joint_rollout_trace_compare_summary_{args.output_prefix}.md"

    dit_path.write_text(json.dumps(dit_trace, indent=2, default=_json_default))
    mamba_path.write_text(json.dumps(mamba_trace, indent=2, default=_json_default))
    summary_json_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    write_markdown(summary, summary_md_path)

    print(f"Wrote {dit_path}")
    print(f"Wrote {mamba_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")


if __name__ == "__main__":
    main()
