#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 1 — Enumerate raw episodes and emit per-episode metadata.

Scans `<raw-root>/episode_NNNN/data.json` files, validates each episode has the
expected structure (frame count, state/action shapes, cameras), and writes:

- `<out-root>/meta/episodes.jsonl` (one line per episode, LeRobot v2.1 schema).
- `<out-root>/.scratch/01_episode_inventory.json` (per-episode debug info).

Usage:
    python 01_enumerate_episodes.py \
        --raw-root /path/to/2026_06_02_main_300ep_NEWEST \
        --out-root /path/to/insertion_shaft_300ep_m0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, type=Path,
                   help="Directory containing episode_NNNN/ sub-directories.")
    p.add_argument("--out-root", required=True, type=Path,
                   help="Target LeRobot v2.1 dataset root.")
    return p.parse_args()


def _validate_episode(ep_dir: Path) -> dict[str, Any]:
    """Read data.json for a single episode and return its validated summary."""
    data_path = ep_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.json in {ep_dir}")
    with open(data_path, "r") as f:
        payload = json.load(f)

    info = payload.get("info", {})
    frames = payload.get("data", [])
    if not frames:
        raise ValueError(f"Episode {ep_dir.name} has zero frames")

    img_meta = info.get("image", {})
    if img_meta:
        if img_meta.get("width") != 640 or img_meta.get("height") != 480:
            raise ValueError(
                f"Unexpected image dims {img_meta.get('width')}x{img_meta.get('height')} "
                f"in {ep_dir.name}; expected 640x480"
            )
        if abs(float(img_meta.get("fps", 30.0)) - 30.0) > 1e-3:
            raise ValueError(
                f"Unexpected fps {img_meta.get('fps')} in {ep_dir.name}; expected 30.0"
            )

    first_frame = frames[0]
    states = first_frame.get("states", {})
    actions = first_frame.get("actions", {})
    colors = first_frame.get("colors", {})

    # Validate that state vector dimensions match the documented 26-dim layout.
    state_dims = {
        "left_arm": len(states["left_arm"].get("qpos", [])),
        "right_arm": len(states["right_arm"].get("qpos", [])),
        "left_ee": len(states["left_ee"].get("qpos", [])),
        "right_ee": len(states["right_ee"].get("qpos", [])),
        "body": len(states["body"].get("qpos", [])),
    }
    total_state_dim = sum(state_dims.values())
    if total_state_dim != 26:
        raise ValueError(
            f"Episode {ep_dir.name} has state dim {total_state_dim} "
            f"(breakdown: {state_dims}); expected 26"
        )

    action_dims = {
        "left_arm": len(actions["left_arm"].get("qpos", [])),
        "right_arm": len(actions["right_arm"].get("qpos", [])),
        "left_ee": len(actions["left_ee"].get("qpos", [])),
        "right_ee": len(actions["right_ee"].get("qpos", [])),
        "body": len(actions["body"].get("qpos", [])),
    }
    total_action_dim = sum(action_dims.values())
    if total_action_dim != 26:
        raise ValueError(
            f"Episode {ep_dir.name} has action dim {total_action_dim} "
            f"(breakdown: {action_dims}); expected 26"
        )

    expected_cams = {"color_0", "color_1", "color_2"}
    if set(colors.keys()) != expected_cams:
        raise ValueError(
            f"Episode {ep_dir.name} has unexpected cameras {sorted(colors.keys())}; "
            f"expected {sorted(expected_cams)}"
        )

    return {
        "episode_dir": ep_dir.name,
        "length": len(frames),
        "state_dims": state_dims,
        "action_dims": action_dims,
        "task_name": first_frame.get("sim_state", {}).get("task_name", ""),
        "fps": float(img_meta.get("fps", 30.0)) if img_meta else 30.0,
    }


def _enumerate_episode_dirs(raw_root: Path) -> list[Path]:
    """Return sorted list of episode directories matching episode_NNNN pattern."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    eps = sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not eps:
        raise FileNotFoundError(f"No episode_NNNN directories under {raw_root}")
    return eps


def main() -> int:
    args = _parse_args()
    raw_root: Path = args.raw_root
    out_root: Path = args.out_root
    meta_dir = out_root / "meta"
    scratch_dir = out_root / ".scratch"
    meta_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    ep_dirs = _enumerate_episode_dirs(raw_root)
    print(f"[01] Found {len(ep_dirs)} raw episode directories under {raw_root}", flush=True)

    inventory: list[dict[str, Any]] = []
    for ep_dir in ep_dirs:
        summary = _validate_episode(ep_dir)
        inventory.append(summary)
        print(
            f"[01] {ep_dir.name}: {summary['length']} frames, "
            f"task={summary['task_name']!r}",
            flush=True,
        )

    # Write episodes.jsonl in LeRobot v2.1 schema.
    ep_jsonl_path = meta_dir / "episodes.jsonl"
    with open(ep_jsonl_path, "w") as f:
        for idx, item in enumerate(inventory):
            rec = {
                "episode_index": idx,
                "tasks": [item["task_name"]],
                "length": item["length"],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[01] Wrote {ep_jsonl_path} ({len(inventory)} episodes)", flush=True)

    # Persist scratch inventory for downstream steps and debugging.
    inv_path = scratch_dir / "01_episode_inventory.json"
    with open(inv_path, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"[01] Wrote {inv_path}", flush=True)

    total_frames = sum(item["length"] for item in inventory)
    print(
        f"[01] OK: {len(inventory)} episodes, {total_frames} total frames",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
