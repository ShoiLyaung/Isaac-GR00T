#!/usr/bin/env bash

# M0 finetune: clean N1.7 baseline on insertion_shaft 300 ep dataset.
#
# This is the Milestone 0 (M0) clean-baseline finetune launcher for the
# insertion-shaft dual-arm task. It is intentionally a thin, parameter-free
# wrapper around the upstream `examples/finetune.sh` recipe:
#
#   * Base model: local cache at /home/sl/checkpoints/GR00T-N1.7-3B/
#     (avoids any HF Hub access — important because Cosmos-Reason2-2B is
#     gated and we have no HF token in this environment).
#   * Dataset:    LeRobot v2.1 300-episode insertion-shaft shard produced by
#                scripts/lerobot_conversion/m0/run_all.sh.
#   * Embodiment: NEW_EMBODIMENT (4 streams + 3 cameras; see
#                modality_config_insertion_shaft_m0.py).
#
# NO architecture / hyperparameter changes from `examples/finetune.sh`
# defaults. Do not add new flags here — if you need to change a value,
# pass it via env-var (MAX_STEPS, GLOBAL_BATCH_SIZE, ...) or open an
# upstream-recipe PR first.

set -euo pipefail

# ----- 1. Paths (override via env-vars, but defaults match the M0 layout) -----
DATASET_PATH="${DATASET_PATH:-/home/sl/groot_datasets/lerobot/insertion_shaft_300ep_m0}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/sl/checkpoints/GR00T-N1.7-3B/}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-examples/insertion_shaft_m0/modality_config_insertion_shaft_m0.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/insertion_shaft_m0_finetune_$(date +%Y%m%d_%H%M%S)}"

# ----- 2. Hyperparameters (mirror examples/finetune.sh defaults) -----
# Defaults below are the official `examples/finetune.sh` values — do not
# deviate without an approved proposal.
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-10000}"
USE_WANDB="${USE_WANDB:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"

# ----- 3. Launcher selection (single-GPU vs torchrun) -----
# Single-GPU mode runs `python` directly so HF Trainer does NOT wrap the
# model in DataParallel (which crashes with StopIteration in the model's
# device property). Multi-GPU mode uses torchrun. This matches the
# behaviour of `examples/finetune.sh` lines 151-158.
if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    LAUNCHER=(python)
else
    LAUNCHER=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}")
fi

WANDB_FLAG=()
if [ "$USE_WANDB" -eq 1 ]; then
    WANDB_FLAG+=(--use_wandb)
fi

# ----- 4. Launch finetune -----
# Repository root = parent of this script's parent (= repo root for
# examples/insertion_shaft_m0/finetune.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== M0 finetune ==="
echo "DATASET_PATH:         ${DATASET_PATH}"
echo "BASE_MODEL_PATH:      ${BASE_MODEL_PATH}"
echo "MODALITY_CONFIG:      ${MODALITY_CONFIG_PATH}"
echo "EMBODIMENT_TAG:       ${EMBODIMENT_TAG}"
echo "OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "NUM_GPUS:             ${NUM_GPUS}"
echo "MAX_STEPS:            ${MAX_STEPS}"
echo "GLOBAL_BATCH_SIZE:    ${GLOBAL_BATCH_SIZE}"
echo "LEARNING_RATE:        1e-4 (official default)"
echo "WARMUP_RATIO:         0.05 (official default)"
echo "WEIGHT_DECAY:         1e-5 (official default)"
echo "SAVE_TOTAL_LIMIT:     5 (official default)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "===================="

# Flag name rationale:
#   --base_model_path / --dataset_path / --modality_config_path /
#   --embodiment_tag / --output_dir / --save_steps / --max_steps /
#   --global_batch_size / --dataloader_num_workers / --shard_size /
#   --num_shards_per_epoch / --episode_sampling_rate
#     → direct fields on `FinetuneConfig`
#       (gr00t/configs/finetune_config.py).
#   --save_total_limit / --warmup_ratio / --weight_decay /
#   --learning_rate
#     → other direct fields on `FinetuneConfig`; values hardcoded to the
#       official defaults per §3.3 of the M0 GAP_ANALYSIS.
#   --color_jitter_params <k1> <v1> <k2> <v2> ...
#     → `FinetuneConfig.color_jitter_params: dict[str, float]`. tyro
#       accepts repeated key/value pairs; underscores and hyphens both
#       work (we use underscores to match `examples/finetune.sh:123`).
#
# transformers_local_files_only is in TrainingConfig (not exposed via tyro.cli(FinetuneConfig)),
# so we set it via env vars HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 (see below).
# Commented out after HF gated access to nvidia/Cosmos-Reason2-2B was approved (2026-06-28).
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# HF_TOKEN must be set in env before invoking this script, e.g.:
#   export HF_TOKEN="hf_xxxxx"
exec "${LAUNCHER[@]}" gr00t/experiment/launch_finetune.py \
    --base_model_path "${BASE_MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --modality_config_path "${MODALITY_CONFIG_PATH}" \
    --embodiment_tag "${EMBODIMENT_TAG}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus "${NUM_GPUS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 5 \
    --max_steps "${MAX_STEPS}" \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    "${WANDB_FLAG[@]}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --shard_size "${SHARD_SIZE}" \
    --num_shards_per_epoch "${NUM_SHARDS_PER_EPOCH}" \
    --episode_sampling_rate "${EPISODE_SAMPLING_RATE}" \
