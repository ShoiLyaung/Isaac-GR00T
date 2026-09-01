from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.rollout_policy import MultiStepConfig, VideoConfig, WrapperConfigs, create_eval_env
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper, _rec_to_dtype

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"


def _json_default(obj: Any):
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_env(n_action_steps: int, max_episode_steps: int):
    return gym.vector.SyncVectorEnv([
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
    ])


def _flat_to_nested_obs(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    for modality in ["video", "state", "language"]:
        for key in policy.modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
                if key == "task" and "annotation.human.coarse_action" in observation:
                    parsed_key = "annotation.human.coarse_action"
                elif parsed_key not in observation:
                    parsed_key = "annotation.human.task_description"
                nested[modality][key] = [[str(item)] for item in observation[parsed_key]]
            else:
                nested[modality][key] = observation[f"{modality}.{key}"]
    return nested


def _prepare_inputs(policy: Gr00tPolicy, observation: dict[str, Any]):
    nested_obs = _flat_to_nested_obs(policy, observation)
    unbatched_observations = policy._unbatch_observation(nested_obs)
    processed_inputs = []
    states = []
    for obs in unbatched_observations:
        vla_step_data = policy._to_vla_step_data(obs)
        states.append(vla_step_data.states)
        processed_inputs.append(policy.processor([{"type": "episode_step", "content": vla_step_data}]))
    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
    batched_states = {
        k: np.stack([s[k] for s in states], axis=0) for k in policy.modality_configs["state"].modality_keys
    }
    return collated_inputs, batched_states


def _sample_right_hand(policy: Gr00tPolicy, observation: dict[str, Any], n_samples: int):
    collated_inputs, batched_states = _prepare_inputs(policy, observation)
    decoded_samples = []
    norm_samples = []
    start = 0
    for key in policy.modality_configs["action"].modality_keys:
        dim = int(policy.processor.state_action_processor.norm_params[policy.embodiment_tag.value]["action"][key]["dim"].item())
        end = start + dim
        if key == "right_hand":
            right_hand_slice = (start, end)
            break
        start = end
    else:
        raise RuntimeError('right_hand not found')

    for _ in range(n_samples):
        with torch.inference_mode():
            model_pred = policy.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float().cpu().numpy()
        decoded_action = policy.processor.decode_action(
            normalized_action, policy.embodiment_tag, batched_states
        )
        norm_samples.append(normalized_action[0, 0, right_hand_slice[0]:right_hand_slice[1]])
        decoded_samples.append(decoded_action["right_hand"][0, 0])

    norm_samples = np.asarray(norm_samples, dtype=np.float32)
    decoded_samples = np.asarray(decoded_samples, dtype=np.float32)
    return {
        "n_samples": int(n_samples),
        "norm_mean": norm_samples.mean(axis=0),
        "norm_std": norm_samples.std(axis=0),
        "norm_abs_mean": np.abs(norm_samples).mean(axis=0),
        "decoded_mean": decoded_samples.mean(axis=0),
        "decoded_std": decoded_samples.std(axis=0),
        "decoded_abs_mean": np.abs(decoded_samples).mean(axis=0),
        "decoded_min": decoded_samples.min(axis=0),
        "decoded_max": decoded_samples.max(axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-model-path", required=True)
    parser.add_argument("--mamba-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--samples-per-step", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=1440)
    args = parser.parse_args()

    env = _make_env(args.n_action_steps, args.max_episode_steps)
    obs, _ = env.reset(seed=args.seed)

    dit = Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1, model_path=args.dit_model_path, device=0)
    mamba = Gr00tPolicy(embodiment_tag=EmbodimentTag.UNITREE_G1, model_path=args.mamba_model_path, device=0)
    mamba.model.config.mamba_inference_align_actions = False
    dit_sim = Gr00tSimPolicyWrapper(dit)

    records = []
    for outer_step in range(args.outer_steps):
        dit_stats = _sample_right_hand(dit, obs, args.samples_per_step)
        mamba_stats = _sample_right_hand(mamba, obs, args.samples_per_step)
        records.append({
            "outer_step": outer_step,
            "dit": dit_stats,
            "mamba": mamba_stats,
        })
        dit_action, _ = dit_sim.get_action(obs)
        obs, *_ = env.step(dit_action)

    env.close()
    output = {"seed": args.seed, "samples_per_step": args.samples_per_step, "records": records}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=_json_default))
    print(output_path)


if __name__ == "__main__":
    main()
