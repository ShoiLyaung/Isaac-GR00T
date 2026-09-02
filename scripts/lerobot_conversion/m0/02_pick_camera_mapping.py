#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 2 — Pick the canonical color-index → semantic-name camera mapping.

Our raw dataset has three cameras addressed by raw color indices (`color_0`,
`color_1`, `color_2`) but no metadata naming the physical location of each
index. We adopt the user-approved convention:

    color_0 → cam_left_high       (left arm, third-person overview)
    color_1 → cam_left_wrist      (left arm wrist-mounted egocentric view)
    color_2 → cam_right_wrist     (right arm wrist-mounted egocentric view)

The mapping is written as a small JSON file consumed by steps 3 and 4 and is
mirrored into `<out-root>/meta/modality.json`'s `video` section (also written
by step 5).

Usage:
    python 02_pick_camera_mapping.py --out-root /path/to/insertion_shaft_300ep_m0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# User-approved mapping; see M0 gap-analysis §3.1 D2.
CAMERA_MAPPING: dict[str, dict[str, str]] = {
    # raw_index -> {semantic_name, parquet_video_key}
    "color_0": {
        "semantic_name": "cam_left_high",
        "parquet_video_key": "observation.images.cam_left_high",
    },
    "color_1": {
        "semantic_name": "cam_left_wrist",
        "parquet_video_key": "observation.images.cam_left_wrist",
    },
    "color_2": {
        "semantic_name": "cam_right_wrist",
        "parquet_video_key": "observation.images.cam_right_wrist",
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", required=True, type=Path,
                   help="Target LeRobot v2.1 dataset root.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_root: Path = args.out_root
    scratch_dir = out_root / ".scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = scratch_dir / "02_camera_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(CAMERA_MAPPING, f, indent=2)
    print(f"[02] Wrote {mapping_path}", flush=True)
    for raw_idx, info in CAMERA_MAPPING.items():
        print(f"[02]   {raw_idx:>8s} -> {info['semantic_name']:<17s} "
              f"({info['parquet_video_key']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
