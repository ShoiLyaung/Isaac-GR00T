#!/usr/bin/env bash
# End-to-end eval orchestration for the N1.7-DiT baseline:
#   1) start the GR00T-N1.7 server in background (model_path = $MODEL_PATH)
#   2) poll TCP port until the server accepts connections
#   3) run the IsaacLab client for N_EPISODES
#   4) shut down the server cleanly
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint \
#   bash scripts/baselines/run_eval_n1d7_main_300ep.sh
#
# Override defaults via env vars:
#   MODEL_PATH=/path/to/checkpoint       (REQUIRED — finetuned ckpt)
#   EMBODIMENT_TAG=UNITREE_G1            (server default — slim policy zero-fills waist/base/nav)
#   HOST=127.0.0.1  PORT=5556
#   N_EPISODES=10  MAX_EPISODE_STEPS=800  N_ACTION_STEPS=20  N_ENVS=1
#   ENV_TASK=Isaac-Insertion-Shaft-G129-Inspire-Joint-Play
#   SERVER_READY_TIMEOUT_S=180           (seconds to wait for server port)
#   CLIENT_TIMEOUT_S=3600                (overall client wall clock)
#   RUN_TAG=...                          (defaults to MODEL_PATH basename + timestamp)
#   VIDEO_DIR=...                        (defaults to outputs/eval_videos/...)
#
# All logs go to outputs/logs/baseline_<date>/eval/<RUN_TAG>/{server.log,client.log}

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/sl/Isaac-GR00T}"

# --- Model & policy ---------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH is required (e.g. /home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_<date>/checkpoint-5000)" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

EMBODIMENT_TAG="${EMBODIMENT_TAG:-UNITREE_G1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5556}"

# --- Eval knobs -------------------------------------------------------------
N_EPISODES="${N_EPISODES:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
N_ACTION_STEPS="${N_ACTION_STEPS:-20}"
N_ENVS="${N_ENVS:-1}"
ENV_TASK="${ENV_TASK:-Isaac-Insertion-Shaft-G129-Inspire-Joint-Play}"
ENV_NAME="${ENV_NAME:-unitree_g1_isaaclab/${ENV_TASK}}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-180}"
CLIENT_TIMEOUT_S="${CLIENT_TIMEOUT_S:-3600}"

RUN_TAG="${RUN_TAG:-$(basename "$MODEL_PATH")_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/outputs/logs/baseline_$(date +%Y%m%d)/eval/${RUN_TAG}}"
VIDEO_DIR="${VIDEO_DIR:-${ROOT_DIR}/outputs/eval_videos/insertion_shaft/${RUN_TAG}}"

# --- Paths to the existing scripts ------------------------------------------
SERVER_SH="${ROOT_DIR}/scripts/eval/run_insertion_shaft_server_n1d7.sh"
CLIENT_SH="${ROOT_DIR}/scripts/eval/run_insertion_shaft_client.sh"

CONDA_PY_SERVER="/home/sl/miniconda3/envs/gr00t_n1d7/bin/python"
UNITREE_SIM_ROOT="${UNITREE_SIM_ROOT:-/home/sl/humanoid-lfd/unitree_sim_isaaclab}"
UNITREE_SIM_PYTHON="${UNITREE_SIM_PYTHON:-/home/sl/miniconda3/envs/env_isaaclab/bin/python}"

# --- Pre-flight checks ------------------------------------------------------
echo "[run_eval_n1d7_main_300ep] MODEL_PATH=$MODEL_PATH"
echo "[run_eval_n1d7_main_300ep] EMBODIMENT_TAG=$EMBODIMENT_TAG HOST=$HOST PORT=$PORT"
echo "[run_eval_n1d7_main_300ep] N_EPISODES=$N_EPISODES MAX_EPISODE_STEPS=$MAX_EPISODE_STEPS N_ACTION_STEPS=$N_ACTION_STEPS"
echo "[run_eval_n1d7_main_300ep] ENV_NAME=$ENV_NAME"
echo "[run_eval_n1d7_main_300ep] LOG_ROOT=$LOG_ROOT"
echo "[run_eval_n1d7_main_300ep] VIDEO_DIR=$VIDEO_DIR"

if [[ ! -f "$SERVER_SH" ]]; then
  echo "[ERROR] server script missing: $SERVER_SH" >&2
  exit 1
fi
if [[ ! -f "$CLIENT_SH" ]]; then
  echo "[ERROR] client script missing: $CLIENT_SH" >&2
  exit 1
fi
if [[ ! -d "$UNITREE_SIM_ROOT" ]]; then
  echo "[ERROR] UNITREE_SIM_ROOT missing: $UNITREE_SIM_ROOT" >&2
  exit 1
fi
if [[ ! -x "$UNITREE_SIM_PYTHON" ]]; then
  echo "[ERROR] UNITREE_SIM_PYTHON not executable: $UNITREE_SIM_PYTHON" >&2
  exit 1
fi

# Refuse if port already in use (likely a leftover server)
if command -v ss >/dev/null 2>&1; then
  if ss -lnt "sport = :$PORT" | grep -q LISTEN; then
    echo "[ERROR] port $PORT already LISTENing — kill the previous server first" >&2
    ss -lntp "sport = :$PORT" || true
    exit 1
  fi
fi

mkdir -p "$LOG_ROOT" "$VIDEO_DIR"
SERVER_LOG="${LOG_ROOT}/server.log"
CLIENT_LOG="${LOG_ROOT}/client.log"
SUMMARY="${LOG_ROOT}/summary.json"

# --- Start server (background) ----------------------------------------------
echo "[run_eval_n1d7_main_300ep] Starting server (background)..."
SERVER_PID_FILE="${LOG_ROOT}/server.pid"
MODEL_PATH="$MODEL_PATH" EMBODIMENT_TAG="$EMBODIMENT_TAG" HOST="$HOST" PORT="$PORT" \
    nohup bash "$SERVER_SH" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$SERVER_PID_FILE"
echo "[run_eval_n1d7_main_300ep] server PID=$SERVER_PID  log=$SERVER_LOG"

# Trap to ensure server cleanup on any exit (success/failure/interrupt)
cleanup() {
  local exit_code=$?
  echo "[run_eval_n1d7_main_300ep] cleanup (exit=$exit_code)..."
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run_eval_n1d7_main_300ep] killing server PID=$SERVER_PID"
    kill -INT "$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    # Give it 10s to shut down cleanly
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  # Also kill any orphaned python children
  if [[ -n "${SERVER_PID:-}" ]]; then
    pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --- Poll server port until ready -------------------------------------------
echo "[run_eval_n1d7_main_300ep] Waiting for server on $HOST:$PORT (timeout=${SERVER_READY_TIMEOUT_S}s)..."
DEADLINE=$(( $(date +%s) + SERVER_READY_TIMEOUT_S ))
READY=0
while [[ $(date +%s) -lt $DEADLINE ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] server process $SERVER_PID exited prematurely — see $SERVER_LOG" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 2
  fi
  # Use Python to check TCP connection (handles both ss/netstat absence)
  if /home/sl/miniconda3/envs/env_isaaclab/bin/python - <<PY 2>/dev/null
import socket
s = socket.socket(); s.settimeout(0.5)
try:
    s.connect(("$HOST", $PORT)); s.close(); raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
  then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "[ERROR] server did not open port $HOST:$PORT within ${SERVER_READY_TIMEOUT_S}s" >&2
  tail -n 120 "$SERVER_LOG" >&2 || true
  exit 3
fi
echo "[run_eval_n1d7_main_300ep] server READY on $HOST:$PORT"

# --- Run client (foreground, with timeout) ----------------------------------
echo "[run_eval_n1d7_main_300ep] Launching client (timeout=${CLIENT_TIMEOUT_S}s)..."
set +e
UNITREE_SIM_ROOT="$UNITREE_SIM_ROOT" \
UNITREE_SIM_PYTHON="$UNITREE_SIM_PYTHON" \
UNITREE_SIM_TASK="$ENV_TASK" \
HOST="$HOST" PORT="$PORT" \
N_EPISODES="$N_EPISODES" \
MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" \
N_ACTION_STEPS="$N_ACTION_STEPS" \
N_ENVS="$N_ENVS" \
ENV_NAME="$ENV_NAME" \
VIDEO_DIR="$VIDEO_DIR" \
RUN_TAG="$RUN_TAG" \
timeout "${CLIENT_TIMEOUT_S}s" \
    bash "$CLIENT_SH" >"$CLIENT_LOG" 2>&1
CLIENT_EXIT=$?
set -e
echo "[run_eval_n1d7_main_300ep] client exited with code $CLIENT_EXIT"

# --- Summary ----------------------------------------------------------------
cat > "$SUMMARY" <<JSON
{
  "model_path": "$MODEL_PATH",
  "embodiment_tag": "$EMBODIMENT_TAG",
  "host": "$HOST",
  "port": $PORT,
  "n_episodes": $N_EPISODES,
  "max_episode_steps": $MAX_EPISODE_STEPS,
  "n_action_steps": $N_ACTION_STEPS,
  "n_envs": $N_ENVS,
  "env_name": "$ENV_NAME",
  "run_tag": "$RUN_TAG",
  "video_dir": "$VIDEO_DIR",
  "client_exit_code": $CLIENT_EXIT,
  "server_log": "$SERVER_LOG",
  "client_log": "$CLIENT_LOG"
}
JSON
echo "[run_eval_n1d7_main_300ep] summary: $SUMMARY"
echo "[run_eval_n1d7_main_300ep] video_dir: $VIDEO_DIR"

# cleanup() will tear down the server via the EXIT trap.
exit $CLIENT_EXIT