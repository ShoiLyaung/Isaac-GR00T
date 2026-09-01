#!/usr/bin/env bash
# End-to-end conversion of all insertion_shaft raw batches → GR00T-flavored LeRobot.
#
# This wraps both Stage 1 and Stage 2 for every batch in
# groot_datasets/raw/insertion_shaft/.
#
# Edit the REPO_IDS / DST_DIRS arrays to customize.

set -euo pipefail

# Require explicit confirmation — this script does a lot of work
if [ "${CONFIRM:-}" != "yes" ]; then
    echo "This script will convert ALL insertion_shaft batches. Set CONFIRM=yes to proceed."
    echo "Usage: CONFIRM=yes $0"
    echo
    echo "Will convert:"
    for entry in "2026_06_02_main_300ep_NEWEST:yuqi/insertion_shaft_main_300ep" \
                 "2026_04_to_05_main_344ep:yuqi/insertion_shaft_main_344ep" \
                 "2026_04_09_bihand_19ep:yuqi/insertion_shaft_bihand_19ep"; do
        BATCH="${entry%%:*}"
        echo "  ${RAW_ROOT:-/home/sl/Isaac-GR00T/groot_datasets/raw/insertion_shaft}/${BATCH}"
    done
    exit 1
fi

# Resolve script dir (this file's dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source dataset location (matches the cleaned-up layout from 2026-06-23)
RAW_ROOT="/home/sl/Isaac-GR00T/groot_datasets/raw/insertion_shaft"
DST_ROOT="${HOME}/datasets/insertion_shaft_groot"

# Map each batch to its repo_id and DST subdir
# Format: "batch_dir_suffix:repo_id"
BATCHES=(
    "2026_06_02_main_300ep_NEWEST:yuqi/insertion_shaft_main_300ep"
    "2026_04_to_05_main_344ep:yuqi/insertion_shaft_main_344ep"
    "2026_04_09_bihand_19ep:yuqi/insertion_shaft_bihand_19ep"
)

# Cameras to keep (3-cam canonical mapping)
KEEP_CAMERAS="cam_left_high,cam_left_wrist,cam_right_wrist"

for entry in "${BATCHES[@]}"; do
    BATCH="${entry%%:*}"
    REPO_ID="${entry##*:}"

    BATCH_DIR="${RAW_ROOT}/${BATCH}"
    DST_DIR="${DST_ROOT}/${BATCH}_groot"

    if [ ! -d "$BATCH_DIR" ]; then
        echo "[skip] $BATCH_DIR not found"
        continue
    fi

    echo
    echo "============================================"
    echo "Batch: $BATCH"
    echo "  raw:  $BATCH_DIR"
    echo "  repo: $REPO_ID"
    echo "  dst:  $DST_DIR"
    echo "============================================"

    # Stage 1: raw → basic LeRobot
    "$SCRIPT_DIR/convert_insertion_shaft_to_lerobot.sh" \
        "$BATCH_DIR" \
        "$REPO_ID" \
        --keep-cameras "$KEEP_CAMERAS"

    # Stage 2: basic LeRobot → GR00T-flavored
    "$SCRIPT_DIR/convert_lerobot_to_groot.sh" \
        "$REPO_ID" \
        "$DST_DIR" \
        --keep-videos "$KEEP_CAMERAS"
done

echo
echo "All batches converted. Output under $DST_ROOT"
