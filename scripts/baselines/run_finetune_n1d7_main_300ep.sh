#!/usr/bin/env bash
# Finetune GR00T-N1.7-3B (DiT system1) on the insertion-shaft 300-episode
# GR00T-flavored LeRobot dataset produced by Stage 2.
#
# Run inside the gr00t_n1d7 conda env (transformers >= 4.57).
# Reads an existing launch script + the slim N1.7 launcher; this wrapper
# only fixes the dataset path, output dir and run knobs.
#
# Usage:
#   bash scripts/baselines/run_finetune_n1d7_main_300ep.sh
#
# Override defaults via env vars (all optional):
#   DATASET_PATH=/path/to/dataset \
#   OUTPUT_DIR_ROOT=/home/sl/Isaac-GR00T/outputs \
#   MAX_STEPS=5000 \
#   GLOBAL_BATCH_SIZE=4 \
#   LEARNING_RATE=1e-4 \
#   SAVE_STEPS=500 \
#   NUM_GPUS=1 \
#   MODALITY_CONFIG_PATH=/path/to/modality_config.py \
#   EXTRA_FLAGS="--tune-visual --tune-llm" \
#   bash scripts/baselines/run_finetune_n1d7_main_300ep.sh

set -euo pipefail

# --- Paths & knobs ----------------------------------------------------------
ROOT_DIR="${ROOT_DIR:-/home/sl/Isaac-GR00T}"
DATASET_PATH="${DATASET_PATH:-/home/sl/datasets/insertion_shaft_main_300ep_groot}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/sl/checkpoints/GR00T-N1.7-3B}"
OUTPUT_DIR_ROOT="${OUTPUT_DIR_ROOT:-${ROOT_DIR}/outputs}"
RUN_TAG="${RUN_TAG:-n1d7_main_300ep_baseline_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_ROOT}/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/logs/baseline_$(date +%Y%m%d)/finetune}"

MAX_STEPS="${MAX_STEPS:-5000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29515}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"  # 300 ep dataset; sample all
SHARD_SIZE="${SHARD_SIZE:-1024}"  # 2**10 = 1024 from launch_finetune_n1d7 defaults
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"  # from launch_finetune_n1d7 defaults

EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
# Server-side default is UNITREE_G1 but the slim N1.7 policy on the train
# side reads MODALITY_CONFIGS["new_embodiment"], so we register + tag it
# NEW_EMBODIMENT here (matches scripts/train/finetune_insertion_shaft_n1d7.sh).

MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${ROOT_DIR}/examples/GR00T-WholeBodyControl/modality_config_insertion_shaft_n1d7.py}"

CONDA_PY="${CONDA_PY:-/home/sl/miniconda3/envs/gr00t_n1d7/bin/python}"
LAUNCHER="${ROOT_DIR}/gr00t/experiment/launch_finetune_n1d7.py"

# --- Pre-flight checks ------------------------------------------------------
echo "[run_finetune_n1d7_main_300ep] DATASET_PATH=$DATASET_PATH"
echo "[run_finetune_n1d7_main_300ep] BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "[run_finetune_n1d7_main_300ep] OUTPUT_DIR=$OUTPUT_DIR"
echo "[run_finetune_n1d7_main_300ep] MAX_STEPS=$MAX_STEPS GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "[run_finetune_n1d7_main_300ep] NUM_GPUS=$NUM_GPUS MODALITY_CONFIG_PATH=$MODALITY_CONFIG_PATH"

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "[ERROR] DATASET_PATH does not exist: $DATASET_PATH" >&2
  echo "        (Stage 2 conversion may not have completed.)" >&2
  exit 1
fi
if [[ ! -d "$DATASET_PATH/data" ]]; then
  echo "[ERROR] $DATASET_PATH/data missing — not a LeRobot dataset?" >&2
  exit 1
fi
if [[ ! -d "$DATASET_PATH/videos" ]]; then
  echo "[WARN] $DATASET_PATH/videos missing — video-less dataset?" >&2
fi
EP_COUNT=$(find "$DATASET_PATH/data" -name "*.parquet" -type f 2>/dev/null | wc -l)
echo "[run_finetune_n1d7_main_300ep] Detected $EP_COUNT parquet episode files"
if [[ "$EP_COUNT" -lt 300 ]]; then
  echo "[WARN] Expected ~300 episodes (Stage 2 may not be complete)." >&2
fi
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
  echo "[ERROR] BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$BASE_MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "[ERROR] BASE_MODEL_PATH missing model.safetensors.index.json: $BASE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$CONDA_PY" ]]; then
  echo "[ERROR] conda python not executable: $CONDA_PY" >&2
  exit 1
fi
if [[ ! -f "$MODALITY_CONFIG_PATH" ]]; then
  echo "[ERROR] MODALITY_CONFIG_PATH missing: $MODALITY_CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$LAUNCHER" ]]; then
  echo "[ERROR] launcher missing: $LAUNCHER" >&2
  exit 1
fi
# Refuse to clobber an existing run dir
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[ERROR] OUTPUT_DIR already exists: $OUTPUT_DIR" >&2
  echo "        Set RUN_TAG=... to pick a new name." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# --- Env --------------------------------------------------------------------
export LOGURU_LEVEL=INFO
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-n1d7-train-$$}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # Use local Qwen3-VL substitute only
mkdir -p "$MPLCONFIGDIR"

# --- Run --------------------------------------------------------------------
cd "$ROOT_DIR"

TRAIN_LOG="${LOG_DIR}/train.log"
echo "[run_finetune_n1d7_main_300ep] Logging to: $TRAIN_LOG"

"$CONDA_PY" -u "$LAUNCHER" \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --modality-config-path "$MODALITY_CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus "$NUM_GPUS" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "$WARMUP_RATIO" \
    --weight-decay "$WEIGHT_DECAY" \
    --dataloader-num-workers "$DATALOADER_WORKERS" \
    --episode-sampling-rate "$EPISODE_SAMPLING_RATE" \
    --shard-size "$SHARD_SIZE" \
    --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH" \
    --bf16 \
    $( [[ "${USE_WANDB:-false}" == "true" ]] && echo "--use-wandb" || echo "--no-use-wandb" ) \
    ${EXTRA_FLAGS:-} \
    2>&1 | tee "$TRAIN_LOG"

echo "[run_finetune_n1d7_main_300ep] Done. Checkpoints under: $OUTPUT_DIR"