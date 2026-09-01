#!/usr/bin/env bash
# Reproduction commands for n1d7_main_300ep_baseline_20260627_151214
# Run from /home/sl/Isaac-GR00T
# All metadata frozen in this directory.

set -euo pipefail

# 0. Pin to the exact commit (the working tree was dirty at run-time but the
#    code that mattered for the run was HEAD = ae17468).
git checkout ae17468d611fb46934d4e9353354f9952dc8e2d9

# 1. Verify preflight
nvidia-smi --query-gpu=memory.free --format=csv,noheader -i 0  # expect ≥ 12 GB
ls /home/sl/datasets/insertion_shaft_main_300ep_groot/meta/info.json
ss -lnt 'sport = :5755 or sport = :5756'  # ports should be free

# 2. Stage 1 (raw → LeRobot v3) — output already cached at
#    /home/sl/.cache/huggingface/lerobot/yuqi/insertion_shaft_main_300ep/
#    rerun via:
UNITREE_LEROBOT_ROOT=/home/yuqi/unitree_lerobot \
bash scripts/lerobot_conversion/convert_insertion_shaft_to_lerobot.sh

# 3. Stage 2 (LeRobot v3 → GR00T-flavored) — output at
#    /home/sl/datasets/insertion_shaft_main_300ep_groot/
UNITREE_LEROBOT_ROOT=/home/yuqi/unitree_lerobot \
bash scripts/lerobot_conversion/convert_lerobot_to_groot.sh

# 4. Finetune (5000 steps)
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH=/home/sl/datasets/insertion_shaft_main_300ep_groot \
OUTPUT_DIR=/home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_$(date +%Y%m%d_%H%M%S) \
LOG_DIR=/home/sl/Isaac-GR00T/outputs/logs/baseline_$(date +%Y%m%d)/finetune \
MAX_STEPS=5000 GLOBAL_BATCH_SIZE=4 SAVE_STEPS=500 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup bash scripts/baselines/run_finetune_n1d7_main_300ep.sh \
  > outputs/logs/baseline_$(date +%Y%m%d)/finetune.log 2>&1 &

# 5. Eval (10 episodes, hand_dim=6 to match dataset)
MODEL_PATH=/home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_<DATE>/checkpoint-5000 \
N_EPISODES=10 MAX_EPISODE_STEPS=800 N_ACTION_STEPS=20 \
HOST=127.0.0.1 PORT=5756 \
UNITREE_SIM_HAND_DIM=6 \
nohup bash scripts/baselines/run_eval_n1d7_main_300ep.sh \
  > outputs/logs/baseline_<DATE>/eval.log 2>&1 &
