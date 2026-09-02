#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# End-to-end orchestrator for the m0 Unitree-G1 LeRobot v2.1 conversion pipeline.
#
# Steps:
#   01_enumerate_episodes.py   - scan raw, write episodes.jsonl + scratch inv.
#   02_pick_camera_mapping.py - hard-coded user-approved color_0/1/2 mapping.
#   03_encode_videos.py        - ffmpeg libx264 mp4 per (episode, camera).
#   04_write_parquets.py       - flat 26-dim state/action + scalar columns.
#   05_write_meta.py           - info.json / modality.json / tasks.jsonl.
#
# After this script finishes, run `gr00t/data/stats.py` (stats + relative_stats)
# followed by 99_smoke_test.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

raw_root=""
out_root=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-root)
            raw_root="$2"; shift 2;;
        --out-root)
            out_root="$2"; shift 2;;
        --max-episodes)
            export MAX_EPISODES="$2"; shift 2;;
        -h|--help)
            cat <<USAGE
Usage: $0 --raw-root <path> --out-root <path> [--max-episodes N]

  --raw-root      Directory containing episode_NNNN/ sub-directories.
  --out-root      Target LeRobot v2.1 dataset root (created if absent).
  --max-episodes  Optional cap on episode count for smoke testing.
USAGE
            exit 0;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2;;
    esac
done

if [[ -z "$raw_root" || -z "$out_root" ]]; then
    echo "ERROR: --raw-root and --out-root are required" >&2
    exit 2
fi

mkdir -p "$out_root"

max_eps_args=()
if [[ -n "${MAX_EPISODES:-}" ]]; then
    max_eps_args=(--max-episodes "$MAX_EPISODES")
fi

step() {
    local script="$1"; shift
    echo "==== [run_all] $script $* ===="
    python3 "$SCRIPT_DIR/$script" "$@"
}

step 01_enumerate_episodes.py --raw-root "$raw_root" --out-root "$out_root"
step 02_pick_camera_mapping.py --out-root "$out_root"
step 03_encode_videos.py --raw-root "$raw_root" --out-root "$out_root" "${max_eps_args[@]}"
step 04_write_parquets.py --raw-root "$raw_root" --out-root "$out_root" "${max_eps_args[@]}"
step 05_write_meta.py --raw-root "$raw_root" --out-root "$out_root"

echo "==== [run_all] DONE ===="
echo "Next: run gr00t/data/stats.py to generate stats + relative_stats."
echo "Then: python3 $SCRIPT_DIR/99_smoke_test.py --out-root $out_root"
