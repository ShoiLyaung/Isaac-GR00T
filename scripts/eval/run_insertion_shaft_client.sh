#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

UNITREE_SIM_ROOT="${UNITREE_SIM_ROOT:-/home/sl/humanoid-lfd/unitree_sim_isaaclab}"
UNITREE_SIM_PYTHON="${UNITREE_SIM_PYTHON:-/home/sl/miniconda3/envs/env_isaaclab/bin/python}"
UNITREE_SIM_TASK="${UNITREE_SIM_TASK:-Isaac-Insertion-Shaft-G129-Inspire-Joint-Play}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
N_EPISODES="${N_EPISODES:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
N_ACTION_STEPS="${N_ACTION_STEPS:-20}"
N_ENVS="${N_ENVS:-1}"
ENV_NAME="${ENV_NAME:-unitree_g1_isaaclab/${UNITREE_SIM_TASK}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
VIDEO_DIR="${VIDEO_DIR:-$REPO_ROOT/outputs/eval_videos/insertion_shaft/${RUN_TAG}}"

if [[ ! -d "$UNITREE_SIM_ROOT" ]]; then
    echo "Unitree simulation root does not exist: $UNITREE_SIM_ROOT"
    echo "Set UNITREE_SIM_ROOT to your unitree_sim_isaaclab path."
    exit 1
fi

if [[ ! -x "$UNITREE_SIM_PYTHON" ]]; then
    echo "Python executable not found: $UNITREE_SIM_PYTHON"
    echo "Set UNITREE_SIM_PYTHON to your env_isaaclab python path."
    exit 1
fi

export PROJECT_ROOT="$UNITREE_SIM_ROOT"
export PYTHONPATH="$REPO_ROOT:$UNITREE_SIM_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export UNITREE_SIM_ROOT
export UNITREE_SIM_DEVICE="${UNITREE_SIM_DEVICE:-cuda:0}"
export UNITREE_SIM_HEADLESS="${UNITREE_SIM_HEADLESS:-1}"
export UNITREE_SIM_ENABLE_CAMERAS="${UNITREE_SIM_ENABLE_CAMERAS:-1}"

mkdir -p "$VIDEO_DIR"
echo "Running insertion eval:"
echo "  ENV_NAME=$ENV_NAME"
echo "  N_EPISODES=$N_EPISODES"
echo "  MAX_EPISODE_STEPS=$MAX_EPISODE_STEPS"
echo "  N_ACTION_STEPS=$N_ACTION_STEPS"
echo "  VIDEO_DIR=$VIDEO_DIR"

"$UNITREE_SIM_PYTHON" gr00t/eval/rollout_policy.py \
    --n_episodes "$N_EPISODES" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --env_name "$ENV_NAME" \
    --n_action_steps "$N_ACTION_STEPS" \
    --n_envs "$N_ENVS" \
    --video_dir "$VIDEO_DIR" \
    --policy_client_host "$HOST" \
    --policy_client_port "$PORT"
