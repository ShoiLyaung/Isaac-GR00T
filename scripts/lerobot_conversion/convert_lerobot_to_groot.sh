#!/usr/bin/env bash
# Convert basic LeRobot v3 dataset → GR00T-flavored LeRobot layout.
#
# Stage 2 of the conversion pipeline (Stage 1 is `convert_insertion_shaft_to_lerobot.sh`).
#
# Usage:
#   ./convert_lerobot_to_groot.sh <src_repo_id> <dst_dir> [flags...]
#
# Example:
#   ./convert_lerobot_to_groot.sh \
#     yuqi/insertion_shaft_threecam_keep_effort \
#     /home/sl/datasets/insertion_shaft_threecam_groot \
#     --keep-videos cam_left_high,cam_left_wrist,cam_right_wrist
#
# Required: ffmpeg + ffprobe on PATH (the script uses them to re-encode videos).

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <src_repo_id> <dst_dir> [extra flags...]"
    echo "  src_repo_id: e.g. yuqi/insertion_shaft_threecam_keep_effort"
    echo "  dst_dir:     e.g. /home/sl/datasets/insertion_shaft_threecam_groot"
    echo "  extra flags: passed to convert_unitree_inspire_multicam_to_groot_keep_effort.py"
    exit 1
fi

SRC_REPO_ID="$1"
DST_DIR="$2"
shift 2

SRC_DIR="$HOME/.cache/huggingface/lerobot/${SRC_REPO_ID}"
UNITREE_LEROBOT_ROOT="/home/sl/Isaac-GR00T/external_dependencies/unitree_lerobot"

# Use the vendored lerobot fork at unitree_lerobot/unitree_lerobot/lerobot/src
# (lerobot 0.4.1 — has lerobot.datasets.lerobot_dataset required by the
# multicam converter). env_isaaclab's lerobot 0.1.0 is too old for the new API.
# Also add UNITREE_LEROBOT_ROOT itself so the converter script's
# `from unitree_lerobot...` imports resolve.
export PYTHONPATH="${UNITREE_LEROBOT_ROOT}/unitree_lerobot/lerobot/src:${UNITREE_LEROBOT_ROOT}:${PYTHONPATH:-}"

# Check dependencies
for tool in ffmpeg ffprobe; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not on PATH. Install with: sudo apt-get install ffmpeg"
        exit 1
    fi
done

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source LeRobot dataset not found: $SRC_DIR"
    echo "       (run convert_insertion_shaft_to_lerobot.sh first)"
    exit 1
fi

echo "Converting $SRC_DIR → $DST_DIR"
python3 "${UNITREE_LEROBOT_ROOT}/convert_unitree_inspire_multicam_to_groot_keep_effort.py" \
    --src "$SRC_DIR" \
    --dst "$DST_DIR" \
    --force \
    "$@"

echo
echo "Done. Output at: $DST_DIR"
echo "Verify with: cat ${DST_DIR}/meta/info.json | python3 -m json.tool | head -30"
