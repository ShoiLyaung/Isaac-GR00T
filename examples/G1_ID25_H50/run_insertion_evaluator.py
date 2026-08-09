"""Authorized-only launcher for the named G1 ID25/H50 insertion evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from isaaclab.app import AppLauncher


SIM_ROOT = Path("/home/sl/humanoid-lfd/unitree_sim_isaaclab")
SIM_COMMIT = "71d529ed7e45b6f95ab72056d37f7e00605de8ae"
TASK = "Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"
SIM_RELEVANT = (
    "tasks/g1_tasks/insertion_shaft_g1_29dof_inspire/__init__.py",
    "tasks/g1_tasks/insertion_shaft_g1_29dof_inspire/insertion_shaft_g1_29dof_inspire_env_cfg.py",
    "tasks/common_observations/g1_29dof_state.py",
    "tasks/common_observations/inspire_state.py",
    "tasks/common_observations/camera_state.py",
    "tasks/common_termination/base_termination_insertion_shaft.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SIM_ROOT), *args], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--execute-simulator", action="store_true")
parser.add_argument("--policy-host", default="127.0.0.1")
parser.add_argument("--policy-port", type=int, required=True)
parser.add_argument("--policy-timeout-ms", type=int, default=15000)
parser.add_argument("--n-episodes", type=int, default=1)
parser.add_argument("--max-episode-steps", type=int, default=1000)
parser.add_argument("--n-action-steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.execute_simulator:
    raise RuntimeError("Actual simulator execution requires explicit --execute-simulator")
args.headless = True
if hasattr(args, "enable_cameras"):
    args.enable_cameras = True

output = args.output.resolve()
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> dict:
    import gymnasium as gym

    sys.path.insert(0, str(SIM_ROOT))
    import tasks  # noqa: F401, PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    from gr00t.eval.sim.UnitreeIsaacLab.g1_id25_h50_insertion import (
        EvaluatorConfig,
        IsaacLabSingleEnvRuntime,
        evaluate_insertion,
        validate_policy_modality_config,
    )
    from gr00t.policy.server_client import PolicyClient

    if _git("rev-parse", "HEAD") != SIM_COMMIT:
        raise RuntimeError("Simulator commit mismatch")
    dirty = _git("status", "--porcelain", "--", *SIM_RELEVANT)
    if dirty:
        raise RuntimeError(f"Targeted simulator source is dirty: {dirty}")

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    policy = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
    )
    try:
        if not policy.ping():
            raise RuntimeError("Policy server preflight failed")
        validate_policy_modality_config(policy.get_modality_config())
        result = evaluate_insertion(
            policy,
            IsaacLabSingleEnvRuntime(env),
            EvaluatorConfig(
                n_episodes=args.n_episodes,
                max_episode_steps=args.max_episode_steps,
                n_action_steps=args.n_action_steps,
                policy_timeout_ms=args.policy_timeout_ms,
                seed=args.seed,
            ),
        )
        result["input_identity"] = {
            "simulator_commit": SIM_COMMIT,
            "simulator_files_sha256": {
                path: _sha256(SIM_ROOT / path) for path in SIM_RELEVANT
            },
            "policy_endpoint": f"tcp://{args.policy_host}:{args.policy_port}",
        }
        return result
    finally:
        policy.close()


try:
    result = main()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
finally:
    simulation_app.close()
