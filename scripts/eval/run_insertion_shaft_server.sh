#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-G1-PnPAppleToPlate}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-UNITREE_G1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

uv --preview-features extra-build-dependencies run python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --use-sim-policy-wrapper \
    --host "$HOST" \
    --port "$PORT"

