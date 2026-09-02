#!/usr/bin/env bash

# M0 eval: spin up a ZMQ policy server + run the rollout client.
#
# Two-stage pattern:
#   1. Start `gr00t/eval/run_gr00t_server.py` in the background. It loads
#      the fine-tuned checkpoint (NEW_EMBODIMENT modality config) and
#      waits on a TCP socket for action requests.
#   2. Start `gr00t/eval/rollout_policy.py` as the client. It builds the
#      Unitree-G1 IsaacLab env, wraps it for GR00T IO, and replays
#      N_EPISODES episodes. Each step it asks the server for an action.
#
# Both stages log to ${LOG_DIR} and on client exit the server is killed
# via an EXIT trap so we never leave a dangling Python process.

set -euo pipefail

# ----- 1. Parameters (override via env-vars) -----
# MODEL_PATH defaults to the latest M0 checkpoint matching the M0
# finetune.sh output-dir pattern. The literal `*` is expanded by the
# glob below.
MODEL_PATH="${MODEL_PATH:-outputs/insertion_shaft_m0_finetune_*/checkpoint-10000}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-examples/insertion_shaft_m0/modality_config_insertion_shaft_m0.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# Network — default 5555 matches `DEFAULT_MODEL_SERVER_PORT` in
# run_gr00t_server.py. If yuqi's eval is already using 5555, set PORT=5556.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda:0}"

# Rollout parameters (mirrors RolloutConfig defaults except n_episodes,
# max_episode_steps, n_action_steps).
N_EPISODES="${N_EPISODES:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
N_ENVS="${N_ENVS:-1}"

# Hardcoded env name — registered by Phase 0-C E1+E2 (env_utils.py +
# rollout_policy.py elif branch + UnitreeG1/unitree_g1_env.py).
ENV_NAME="${ENV_NAME:-unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play}"

# Outputs — unique timestamps so we never overwrite a previous eval.
LOG_DIR="${LOG_DIR:-outputs/logs/m0_eval_$(date +%Y%m%d_%H%M%S)}"
VIDEO_DIR="${VIDEO_DIR:-outputs/eval_videos/m0_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

# ----- 2. Resolve MODEL_PATH (support glob) -----
if [[ "${MODEL_PATH}" == *"*"* ]]; then
    # shellcheck disable=SC2086
    MODEL_PATH="$(ls -d ${MODEL_PATH} 2>/dev/null | sort | tail -1 || true)"
fi
if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH '${MODEL_PATH}' not found." >&2
    echo "Set MODEL_PATH explicitly, or ensure outputs/insertion_shaft_m0_finetune_*/checkpoint-10000 exists." >&2
    exit 1
fi

# ----- 3. Start server in background -----
SERVER_LOG="${LOG_DIR}/server.log"
SERVER_PID_FILE="${LOG_DIR}/server.pid"

echo "=== M0 eval server ==="
echo "MODEL_PATH:          ${MODEL_PATH}"
echo "EMBODIMENT_TAG:      ${EMBODIMENT_TAG}"
echo "MODALITY_CONFIG:     ${MODALITY_CONFIG_PATH}"
echo "HOST:PORT:           ${HOST}:${PORT}"
echo "DEVICE:              ${DEVICE}"
echo "SERVER_LOG:          ${SERVER_LOG}"
echo "===================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ----- 3. Start server in background (uses .venv = m0-clean venv) -----
# Server runs GR00T model inference. Needs the right torch 2.7.1 +
# transformers 4.57.3 + flash-attn 2.7.4 that the model was trained with.
SERVER_LOG="${LOG_DIR}/server.log"
SERVER_PID_FILE="${LOG_DIR}/server.pid"

echo "=== M0 eval server ==="
echo "MODEL_PATH:          ${MODEL_PATH}"
echo "EMBODIMENT_TAG:      ${EMBODIMENT_TAG}"
echo "MODALITY_CONFIG:     ${MODALITY_CONFIG_PATH}"
echo "HOST:PORT:           ${HOST}:${PORT}"
echo "DEVICE:              ${DEVICE}"
echo "SERVER_LOG:          ${SERVER_LOG}"
echo "===================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# If we're not already in the m0-clean venv, activate it for the server.
# (When this script is sourced, .venv is already active; when run via
#  nohup from a different shell, we activate here.)
if [ -z "${VIRTUAL_ENV}" ] || [ "$(basename "${VIRTUAL_ENV}")" != ".venv" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi
cd "${REPO_ROOT}"

# PYTHONUNBUFFERED=1 prevents stdout buffering so the "Server is ready"
# log line lands in SERVER_LOG immediately (we poll it below).
# --use_sim_policy_wrapper: env outputs flat keys (video.ego_view,
# state.left_arm, ...) but Gr00tPolicy expects nested dict
# ({video: {ego_view: ...}, state: {left_arm: ...}}). The wrapper
# converts flat→nested. See gr00t/policy/gr00t_policy.py:485+.
PYTHONUNBUFFERED=1 python gr00t/eval/run_gr00t_server.py \
    --model_path "${MODEL_PATH}" \
    --embodiment_tag "${EMBODIMENT_TAG}" \
    --modality_config_path "${MODALITY_CONFIG_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --device "${DEVICE}" \
    --use_sim_policy_wrapper \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${SERVER_PID_FILE}"
echo "server PID: ${SERVER_PID}"

# Ensure server dies if this script exits (success, error, or signal).
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

# Wait for server readiness — both `run_gr00t_server.py:157` and
# `server_client.py:134` print a "ready" / "listening" line. We accept
# either.
for i in $(seq 1 180); do
    if grep -qE "Server is ready|✓ Server ready" "${SERVER_LOG}" 2>/dev/null; then
        echo "server ready after ${i}s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server died before becoming ready. Last 40 lines of ${SERVER_LOG}:" >&2
        tail -n 40 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 1
done

if ! grep -qE "Server is ready|✓ Server ready" "${SERVER_LOG}" 2>/dev/null; then
    echo "ERROR: server did not become ready within 180s." >&2
    tail -n 40 "${SERVER_LOG}" >&2 || true
    exit 1
fi

# ----- 4. Run rollout client (uses env_isaaclab — has Isaac Sim + isaaclab) -----
# The client only needs:
#   - isaaclab (AppLauncher) + Isaac Sim (omni.*)
#   - rollout_policy.py / MultiStepWrapper / unitree_g1_env.py
# The client does NOT load the GR00T model directly; it uses PolicyClient
# (ZMQ) to talk to the server. The ZMQ protocol is language-agnostic, so
# we can run client in env_isaaclab (Python 3.11) while server runs in
# m0-clean venv (Python 3.10). This is the only way to bridge the torch
# version conflict between the GR00T model (needs 2.7.1) and Isaac Sim
# (needs 3.11 / torch 2.6).
CLIENT_LOG="${LOG_DIR}/client.log"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/sl/miniconda3/envs/env_isaaclab/bin/python}"

if [ ! -x "${ISAACLAB_PYTHON}" ]; then
    echo "ERROR: env_isaaclab python not found at ${ISAACLAB_PYTHON}" >&2
    kill "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi

# env_isaaclab doesn't have the m0-clean package installed editable; the
# simplest reliable way is to add the repo root to PYTHONPATH so its
# gr00t/ package is importable from that Python. Isaac Sim's own
# site-packages (omni.*) is already on env_isaaclab's sys.path.
# Also prepend the M0 flash_attn stub (env_isaaclab's installed
# flash_attn .so is built against a different torch ABI, but the client
# never actually uses it; the stub satisfies the import chain triggered
# by `from transformers import AutoModel, AutoProcessor`).
M0_STUBS_DIR="${M0_STUBS_DIR:-/home/sl/.m0_stubs}"
export PYTHONPATH="${M0_STUBS_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== M0 eval client ==="
echo "PYTHON:              ${ISAACLAB_PYTHON}"
echo "ENV_NAME:            ${ENV_NAME}"
echo "N_EPISODES:          ${N_EPISODES}"
echo "MAX_EPISODE_STEPS:   ${MAX_EPISODE_STEPS}"
echo "N_ACTION_STEPS:      ${N_ACTION_STEPS}"
echo "N_ENVS:              ${N_ENVS}"
echo "VIDEO_DIR:           ${VIDEO_DIR}"
echo "CLIENT_LOG:          ${CLIENT_LOG}"
echo "PYTHONPATH:          ${PYTHONPATH}"
echo "===================="

# PYTHONUNBUFFERED=1 again so success-rate stdout is line-buffered into
# the tee'd log.
PYTHONUNBUFFERED=1 "${ISAACLAB_PYTHON}" gr00t/eval/rollout_policy.py \
    --n_episodes "${N_EPISODES}" \
    --policy_client_host "${HOST}" \
    --policy_client_port "${PORT}" \
    --max_episode_steps "${MAX_EPISODE_STEPS}" \
    --env_name "${ENV_NAME}" \
    --n_action_steps "${N_ACTION_STEPS}" \
    --n_envs "${N_ENVS}" \
    --video_dir "${VIDEO_DIR}" \
    2>&1 | tee "${CLIENT_LOG}"

# Trap (set above) will kill the server on exit. Client's final stdout
# already includes the success-rate line printed by rollout_policy.py.
