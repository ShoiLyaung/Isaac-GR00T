#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 3 - Encode raw per-frame jpgs into per-episode mp4 videos.

For each raw episode, this script locates ``colors/NNNNNN_color_X.jpg`` for
each camera in the mapping emitted by step 2, then encodes the sequence into
a libx264 / yuv420p mp4 placed under

    <out-root>/videos/chunk-{episode_chunk:03d}/<video_key>/episode_{episode_index:06d}.mp4

The encoding uses a single ``ffmpeg`` subprocess per (episode, camera) pair.
libx264 is preferred over h265 / av1 to maximise torchcodec decoding
compatibility.

Usage:
    python 03_encode_videos.py \
        --raw-root /path/to/raw \
        --out-root /path/to/insertion_shaft_300ep_m0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", type=str, default="medium")
    p.add_argument("--max-episodes", type=int, default=-1)
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


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH; install via apt or conda.")


def _encode_one(ep_dir, raw_index, out_path, num_frames, fps, crf, preset):
    colors_dir = ep_dir / "colors"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = out_path.parent / (".__stage_" + raw_index)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    try:
        for frame_idx in range(num_frames):
            src = colors_dir / ("{:06d}_{}.jpg".format(frame_idx, raw_index))
            if not src.exists():
                raise FileNotFoundError(
                    "Missing frame " + str(src) +
                    " (expected {} frames)".format(num_frames)
                )
            dst = staging_dir / ("{:06d}.jpg".format(frame_idx))
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", str(staging_dir / "%06d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", preset,
            "-crf", str(crf),
            "-r", str(fps),
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed for {} (rc={}): {}".format(
                    out_path, proc.returncode, proc.stderr
                )
            )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def main():
    args = _parse_args()
    _check_ffmpeg()

    raw_root = args.raw_root
    out_root = args.out_root
    fps = args.fps
    crf = args.crf
    preset = args.preset
    max_eps = args.max_episodes

    inventory = _load_inventory(out_root)
    cam_mapping = _load_camera_mapping(out_root)
    videos_root = out_root / "videos"

    if max_eps > 0:
        inventory = inventory[:max_eps]

    total_videos = 0
    for ep_idx, item in enumerate(inventory):
        ep_dir = raw_root / item["episode_dir"]
        if not ep_dir.is_dir():
            raise FileNotFoundError("Missing raw episode dir: " + str(ep_dir))
        chunk_idx = ep_idx // 1000
        num_frames = item["length"]
        for raw_index, info in cam_mapping.items():
            video_key = info["parquet_video_key"]
            out_path = (
                videos_root
                / ("chunk-{:03d}".format(chunk_idx))
                / video_key
                / ("episode_{:06d}.mp4".format(ep_idx))
            )
            _encode_one(
                ep_dir=ep_dir,
                raw_index=raw_index,
                out_path=out_path,
                num_frames=num_frames,
                fps=fps,
                crf=crf,
                preset=preset,
            )
            total_videos += 1
        if (ep_idx + 1) % 25 == 0 or ep_idx == len(inventory) - 1:
            print(
                "[03] Encoded {}/{} episodes ({} videos so far)".format(
                    ep_idx + 1, len(inventory), total_videos
                ),
                flush=True,
            )

    print(
        "[03] OK: {} videos encoded under {}".format(
            total_videos, videos_root
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
