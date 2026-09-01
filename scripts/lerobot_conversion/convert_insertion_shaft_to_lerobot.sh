#!/usr/bin/env bash
# Convert `groot_datasets/raw/insertion_shaft/<batch>/` → basic LeRobot v3 dataset.
#
# Stage 1 of the conversion pipeline (Stage 2 is `convert_lerobot_to_groot.sh`).
#
# Usage:
#   ./convert_insertion_shaft_to_lerobot.sh <batch_dir> <repo_id> [flags...]
#
# Example:
#   ./convert_insertion_shaft_to_lerobot.sh \
#     /home/sl/Isaac-GR00T/groot_datasets/raw/insertion_shaft/2026_06_02_main_300ep_NEWEST \
#     yuqi/insertion_shaft_threecam_keep_effort \
#     --keep-cameras cam_left_high,cam_left_wrist,cam_right_wrist
#
# The script runs from the unitree_lerobot repo root and uses its bundled venv.
# Outputs to ~/.cache/huggingface/lerobot/<repo_id> (LeRobot default cache).

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <batch_dir> <repo_id> [extra flags...]"
    echo "  batch_dir: e.g. .../groot_datasets/raw/insertion_shaft/2026_06_02_main_300ep_NEWEST"
    echo "  repo_id:  e.g. yuqi/insertion_shaft_threecam_keep_effort"
    echo "  extra flags: passed to convert_unitree_json_to_lerobot_multicam_keep_effort.py"
    exit 1
fi

BATCH_DIR="$1"
REPO_ID="$2"
shift 2

# Resolve the parent of the batch (the script wants the task's parent dir,
# i.e. where --raw-dir points to scan for batch/task/episode dirs).
# Per the upstream docs:
#   --raw-dir should be $RAW_PARENT/task_name (or its parent, depending on layout).
#
# Our layout is: .../groot_datasets/raw/insertion_shaft/<batch>/episode_NNNN/data.json
# We pass the batch dir directly; the multicam script also supports
# --raw-dir pointing at a dir containing data.json (single-episode mode).

UNITREE_LEROBOT_ROOT="/home/sl/Isaac-GR00T/external_dependencies/unitree_lerobot"

if [ ! -d "$BATCH_DIR" ]; then
    echo "ERROR: batch_dir not found: $BATCH_DIR"
    exit 1
fi

# Use the vendored lerobot fork at unitree_lerobot/unitree_lerobot/lerobot/src
# (lerobot 0.4.1 — exposes lerobot.datasets.lerobot_dataset required by the
# multicam converter). env_isaaclab's lerobot 0.1.0 is too old for the new API.
# Also add UNITREE_LEROBOT_ROOT itself so `from unitree_lerobot.utils.constants
# import ROBOT_CONFIGS` resolves inside the converter script.
export PYTHONPATH="${UNITREE_LEROBOT_ROOT}/unitree_lerobot/lerobot/src:${UNITREE_LEROBOT_ROOT}:${PYTHONPATH:-}"

# Step 0: rename episodes to contiguous episode_0000, episode_0001, ...
echo "[1/2] Renaming episode folders to be contiguous..."
python3 "${UNITREE_LEROBOT_ROOT}/unitree_lerobot/utils/sort_and_rename_folders.py" \
    --data_dir "$BATCH_DIR"

# Step 1: raw JSON → basic LeRobot v3 (multi-cam, with torque)
echo "[2/2] Converting to basic LeRobot v3 (multi-cam, with torque)..."
python3 "${UNITREE_LEROBOT_ROOT}/unitree_lerobot/utils/convert_unitree_json_to_lerobot_multicam_keep_effort.py" \
    --raw-dir "$BATCH_DIR" \
    --repo-id "$REPO_ID" \
    --robot-type Unitree_G1_Inspire_Sim_OneCam \
    --camera-map color_0=cam_left_high,color_1=cam_left_wrist,color_2=cam_right_wrist \
    "$@"

echo
echo "Done. Output at: ~/.cache/huggingface/lerobot/${REPO_ID}"
echo "Next: run convert_lerobot_to_groot.sh to produce the GR00T-flavored dataset."
