#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GROOT_WBC_PYTHON="${GROOT_WBC_PYTHON:-$REPO_ROOT/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
ENV_NAME="${ENV_NAME:-gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc}"
N_EPISODES="${N_EPISODES:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1440}"
N_ACTION_STEPS="${N_ACTION_STEPS:-20}"
N_ENVS="${N_ENVS:-1}"

if [[ ! -x "$GROOT_WBC_PYTHON" ]]; then
    echo "Python executable not found: $GROOT_WBC_PYTHON"
    echo "Set GROOT_WBC_PYTHON to your GR00T-WholeBodyControl sim python path."
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

"$GROOT_WBC_PYTHON" gr00t/eval/rollout_policy.py \
    --n_episodes "$N_EPISODES" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --env_name "$ENV_NAME" \
    --n_action_steps "$N_ACTION_STEPS" \
    --n_envs "$N_ENVS" \
    --policy_client_host "$HOST" \
    --policy_client_port "$PORT"

