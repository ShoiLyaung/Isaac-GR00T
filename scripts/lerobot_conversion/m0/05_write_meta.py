#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 5 - Emit LeRobot v2.1 meta files (info / modality / tasks).

Produces three files under ``<out-root>/meta/``:

- ``info.json``         : dataset-wide schema, v2.1 chunk templates, fps,
                          total_episodes, total_frames, total_tasks, etc.
- ``tasks.jsonl``       : ``{"task_index": int, "task": str}`` per line.
- ``modality.json``     : state / action slice indices plus video / annotation
                          ``original_key`` mappings used by the loader's
                          ``_extract_joint_groups`` and video decoder.

These mirror ``examples/SO100/modality.json`` so the m0-clean loader can
treat our 4-stream / 3-cam dataset exactly like the SO100 reference.

Usage:
    python 05_write_meta.py \
        --raw-root /path/to/raw \
        --out-root /path/to/insertion_shaft_300ep_m0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 26-dim state/action layout (left_arm | right_arm | left_ee | right_ee).
STATE_SLICES = {
    "left_arm":  {"start": 0,  "end": 7},
    "right_arm": {"start": 7,  "end": 14},
    "left_hand": {"start": 14, "end": 20},
    "right_hand": {"start": 20, "end": 26},
}

ACTION_SLICES = {
    "left_arm":  {"start": 0,  "end": 7},
    "right_arm": {"start": 7,  "end": 14},
    "left_hand": {"start": 14, "end": 20},
    "right_hand": {"start": 20, "end": 26},
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    return p.parse_args()


def _load_inventory(out_root):
    path = out_root / ".scratch" / "01_episode_inventory.json"
    if not path.exists():
        raise FileNotFoundError(str(path) + " missing - run step 1 first")
    with open(path, "r") as f:
        return json.load(f)


def _load_camera_mapping(out_root):
    path = out_root / ".scratch" / "02_camera_mapping.json"
    if not path.exists():
        raise FileNotFoundError(str(path) + " missing - run step 2 first")
    with open(path, "r") as f:
        return json.load(f)


def _build_info(inventory, cam_mapping, fps=30.0):
    """Build the LeRobot v2.1 info.json document."""
    total_episodes = len(inventory)
    total_frames = sum(item["length"] for item in inventory)
    # Single-task dataset: collect unique task names.
    task_names = sorted({item["task_name"] for item in inventory if item["task_name"]})
    if not task_names:
        task_names = ["Isaac-Insertion-Shaft-G129-Inspire-Joint"]

    # State / action are 26-dim float32; per-frame column names match a
    # human-friendly naming. The semantic grouping is captured in
    # modality.json so the loader can slice accordingly.
    state_names = (
        ["left_arm_{}".format(i) for i in range(7)] +
        ["right_arm_{}".format(i) for i in range(7)] +
        ["left_hand_{}".format(i) for i in range(6)] +
        ["right_hand_{}".format(i) for i in range(6)]
    )
    video_keys = [info["parquet_video_key"] for info in cam_mapping.values()]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [26],
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [26],
            "names": state_names,  # state and action share the joint layout.
        },
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
    }
    for video_key in video_keys:
        features[video_key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_Inspire_Sim_DualArm",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_names),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:{}".format(total_episodes)},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": (total_episodes + 999) // 1000,
        "total_videos": total_episodes * len(video_keys),
    }


def _build_modality(cam_mapping):
    """Build meta/modality.json with state / action slices + video keys."""
    video_section = {}
    for info in cam_mapping.values():
        semantic = info["semantic_name"]
        video_section[semantic] = {
            "original_key": info["parquet_video_key"],
        }
    return {
        "state": STATE_SLICES,
        "action": ACTION_SLICES,
        "video": video_section,
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            },
        },
    }


def _build_tasks(inventory):
    """Return list of {task_index, task} dicts, one per unique task name."""
    seen = []
    task_index_for = {}
    for item in inventory:
        name = item["task_name"] or "Isaac-Insertion-Shaft-G129-Inspire-Joint"
        if name not in task_index_for:
            task_index_for[name] = len(task_index_for)
            seen.append(name)
    return [
        {"task_index": task_index_for[name], "task": name}
        for name in seen
    ]


def main():
    args = _parse_args()
    out_root = args.out_root
    inventory = _load_inventory(out_root)
    cam_mapping = _load_camera_mapping(out_root)

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = _build_info(inventory, cam_mapping)
    modality = _build_modality(cam_mapping)
    tasks = _build_tasks(inventory)

    info_path = meta_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print("[05] Wrote", info_path, flush=True)

    modality_path = meta_dir / "modality.json"
    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=4)
    print("[05] Wrote", modality_path, flush=True)

    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for rec in tasks:
            f.write(json.dumps(rec) + "\n")
    print("[05] Wrote", tasks_path, "(", len(tasks), "tasks)", flush=True)

    print("[05] OK: meta/ written, {} episodes, {} total frames".format(
        info["total_episodes"], info["total_frames"]
    ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
