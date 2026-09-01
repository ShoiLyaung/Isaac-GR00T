#!/usr/bin/env bash
# Finetune GR00T-N1.7-3B on the insertion-shaft dataset (33 episodes).
#
# Dataset: /home/sl/Isaac-GR00T/groot_datasets/insertion_with_torque_groot
# Modality config: examples/GR00T-WholeBodyControl/modality_config_insertion_shaft_n1d7.py
# Modality config registers "new_embodiment" with 4 state/action keys
# (left_arm, right_arm, left_hand, right_hand) matching the dataset's 26-dim schema.
# At eval time, the slim policy zero-fills the 3 env-required keys (waist,
# base_height_command, navigate_command) that aren't in the dataset.
#
# Run inside the gr00t_n1d7 conda env (transformers >= 4.57).

set -euo pipefail

ROOT_DIR="/home/sl/Isaac-GR00T"
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/groot_datasets/insertion_with_torque_groot}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/g1_n1d7_insertion_shaft_$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29515}"

if [[ ! -f "$DATASET_PATH/meta/info.json" ]]; then
  echo "[ERROR] DATASET_PATH is not a LeRobot dataset: $DATASET_PATH"
  echo "        Missing: $DATASET_PATH/meta/info.json"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export LOGURU_LEVEL=INFO
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-n1d7-train}"

cd "$ROOT_DIR"

echo "[finetune_insertion_shaft_n1d7] DATASET_PATH=$DATASET_PATH"
echo "[finetune_insertion_shaft_n1d7] OUTPUT_DIR=$OUTPUT_DIR"
echo "[finetune_insertion_shaft_n1d7] NUM_GPUS=$NUM_GPUS"
echo "[finetune_insertion_shaft_n1d7] Using conda env: gr00t_n1d7"

/home/sl/miniconda3/envs/gr00t_n1d7/bin/python -u \
    "$ROOT_DIR/gr00t/experiment/launch_finetune_n1d7.py" \
    --base-model-path /home/sl/checkpoints/GR00T-N1.7-3B \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$ROOT_DIR/examples/GR00T-WholeBodyControl/modality_config_insertion_shaft_n1d7.py" \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus "$NUM_GPUS" \
    --max-steps "${MAX_STEPS:-2000}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE:-4}" \
    --save-steps "${SAVE_STEPS:-500}" \
    --save-total-limit 3 \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --warmup-ratio 0.05 \
    --weight-decay 1e-5 \
    --dataloader-num-workers "${DATALOADER_WORKERS:-2}" \
    --bf16 \
    --use-wandb "${USE_WANDB:-false}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"