#!/usr/bin/env bash
# Launch the GR00T-N1.7-3B policy server for the G1 IsaacLab shaft-insertion task.
#
# Runs in the dedicated ``gr00t_n1d7`` conda env (transformers >= 4.57) so
# the main env's transformers 4.51.3 pin (required for the N1.6 Eagle backbone)
# is left untouched.
#
# Usage:
#   bash scripts/eval/run_insertion_shaft_server_n1d7.sh
#
# Override defaults via env vars:
#   MODEL_PATH=/path/to/checkpoint HOST=... PORT=... EMBODIMENT_TAG=UNITREE_G1 \
#     bash scripts/eval/run_insertion_shaft_server_n1d7.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sl/Isaac-GR00T}"
MODEL_PATH="${MODEL_PATH:-/home/sl/checkpoints/GR00T-N1.7-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-UNITREE_G1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5556}"  # 5556 to avoid collision with N1.6 server on 5555

# Substitute backbone for the gated Cosmos-Reason2-2B. The processor config
# and image preprocessor config are identical between Cosmos-Reason2-2B and
# Qwen3-VL-2B-Instruct (both are Qwen3-VL arch), so this lets us sidestep the
# gated HF repo while still loading the local N1.7-3B weights correctly.
BACKBONE_SUBSTITUTE="${BACKBONE_SUBSTITUTE:-Qwen/Qwen3-VL-2B-Instruct}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-n1d7-$$}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
mkdir -p "$MPLCONFIGDIR"

echo "[run_insertion_shaft_server_n1d7] MODEL_PATH=$MODEL_PATH"
echo "[run_insertion_shaft_server_n1d7] EMBODIMENT_TAG=$EMBODIMENT_TAG"
echo "[run_insertion_shaft_server_n1d7] HOST=$HOST PORT=$PORT"
echo "[run_insertion_shaft_server_n1d7] BACKBONE_SUBSTITUTE=$BACKBONE_SUBSTITUTE"
echo "[run_insertion_shaft_server_n1d7] Using conda env: gr00t_n1d7"

# Run the server inside the gr00t_n1d7 conda env.
exec /home/sl/miniconda3/envs/gr00t_n1d7/bin/python \
    "${REPO_ROOT}/gr00t/model/gr00t_n1d7/run_server.py" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --backbone-substitute "$BACKBONE_SUBSTITUTE" \
    --host "$HOST" \
    --port "$PORT" \
    --device "${DEVICE:-cuda:0}"