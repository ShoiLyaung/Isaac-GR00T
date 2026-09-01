#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/sl/Isaac-GR00T"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29510}"

# IMPORTANT:
# launch_finetune.py requires LeRobot dataset format (meta/info.json + parquet + videos).
# Set DATASET_PATH to your converted LeRobot directory.
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/groot_datasets/100_with_tau_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/g1_dit_100_with_tau_fulltrain_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$DATASET_PATH/meta/info.json" ]]; then
  echo "[ERROR] DATASET_PATH is not a LeRobot dataset: $DATASET_PATH"
  echo "        Missing: $DATASET_PATH/meta/info.json"
  echo "        Current raw dataset path is likely: $ROOT_DIR/groot_datasets/100_with_tau"
  echo "        Please convert it to LeRobot format first, then rerun."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

cd "$ROOT_DIR"

uv run torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/GR00T-WholeBodyControl/modality_config_100_with_tau.py \
  --system1-type dit \
  --num-gpus "$NUM_GPUS" \
  --max-steps 10000 \
  --global-batch-size 8 \
  --save-steps 1000 \
  --save-total-limit 5 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --dataloader-num-workers 6 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
