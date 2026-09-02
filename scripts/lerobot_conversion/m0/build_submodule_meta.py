"""Build the missing meta/ files for the submodule-converted Stage 2 dataset
so that GR00T's LeRobot loader can read it.

Inputs (read-only):
  /home/sl/groot_datasets/lerobot/insertion_shaft_300ep_submodule_groot/
    - data/chunk-000/episode_*.parquet  (300 files, 12 columns each)
    - videos/chunk-000/{video_key}/episode_*.mp4  (3 cam dirs)

Outputs (writes):
  {root}/meta/
    - info.json
    - modality.json
    - stats.json           (via gr00t/data/stats.py)
    - tasks.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import statistics
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path("/home/sl/groot_datasets/lerobot/insertion_shaft_300ep_submodule_groot")
META = ROOT / "meta"
DATA = ROOT / "data" / "chunk-000"
VIDEO = ROOT / "videos" / "chunk-000"

CHUNK_SIZE = 1000
FPS = 30.0
JOINT_NAMES = [
    # 26 dims, PascalCase (matching Stage 1 info.json)
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandPinky", "kLeftHandRing", "kLeftHandMiddle", "kLeftHandIndex",
    "kLeftHandThumbBend", "kLeftHandThumbRotation",
    "kRightHandPinky", "kRightHandRing", "kRightHandMiddle", "kRightHandIndex",
    "kRightHandThumbBend", "kRightHandThumbRotation",
]
STATE_SLICES = {
    "left_arm":  (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 20),
    "right_hand": (20, 26),
}


def main():
    META.mkdir(parents=True, exist_ok=True)

    # 1. Inventory: count episodes + frames
    ep_files = sorted(DATA.glob("episode_*.parquet"))
    n_episodes = len(ep_files)
    total_frames = 0
    for ep_path in ep_files:
        # Use cheap parquet metadata to avoid full read
        md = pq.read_metadata(ep_path)
        total_frames += md.num_rows
    print(f"Found {n_episodes} episodes, {total_frames} total frames")

    # 2. Verify video paths
    cam_dirs = sorted([p.name for p in VIDEO.glob("observation.images.*")])
    print(f"Camera dirs: {cam_dirs}")

    # 3. info.json (LeRobot v2.1 layout)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_Inspire_Sim_OneCam",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{n_episodes}"},
        # v2.1 layout templates — m0-clean loader uses these.
        # The loader only formats with `{episode_chunk}`, `{video_key}`, `{episode_index}` —
        # any other placeholder (e.g. `{file_index}`) will raise KeyError.
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [26], "names": JOINT_NAMES},
            "action": {"dtype": "float32", "shape": [26], "names": JOINT_NAMES},
            "observation.effort": {"dtype": "float32", "shape": [26], "names": JOINT_NAMES},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "annotation.human.action.task_description": {"dtype": "string", "shape": [1], "names": None},
            "annotation.human.validity": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        },
    }
    for cam in cam_dirs:
        info["features"][cam] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "info": {
                "video.fps": int(FPS),
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    with open(META / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote info.json ({len(info['features'])} features)")

    # 4. modality.json (state/action/video/annotation slicing)
    modality = {
        "state": {k: {"start": v[0], "end": v[1]} for k, v in STATE_SLICES.items()},
        "action": {k: {"start": v[0], "end": v[1]} for k, v in STATE_SLICES.items()},
        "video": {cam.replace("observation.images.", ""): {"original_key": cam} for cam in cam_dirs},
        "annotation": {
            "human.action.task_description": {"original_key": "annotation.human.action.task_description"},
            "human.validity": {"original_key": "annotation.human.validity"},
        },
    }
    with open(META / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)
    print(f"Wrote modality.json ({len(cam_dirs)} video streams)")

    # 5. tasks.jsonl
    with open(META / "tasks.jsonl", "w") as f:
        # Read first episode's task desc to be authoritative
        first_t = pq.read_table(ep_files[0]).column("annotation.human.action.task_description").to_pylist()
        task_desc = first_t[0] if first_t else "task"
        f.write(json.dumps({"task_index": 0, "task": task_desc}) + "\n")
    print(f"Wrote tasks.jsonl (1 task: '{task_desc}')")

    print("\n=== meta/ files generated ===")
    for p in sorted(META.glob("*")):
        print(f"  {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
