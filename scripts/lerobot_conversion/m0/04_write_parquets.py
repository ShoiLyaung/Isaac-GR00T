#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 4 - Emit LeRobot v2.1 parquet files from raw per-episode data.json.

Each raw episode directory contains a single ``data.json`` whose ``data`` field
is a list of frames. This script flattens the 26-dim state / 26-dim action
arrays (left_arm | right_arm | left_ee | right_ee), aligns each row with the
scalar parquet columns expected by LeRobot v2.1 (timestamp, frame_index,
episode_index, index, task_index), and writes one parquet per episode to:

    <out-root>/data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet

The output schema is identical to what `gr00t/data/stats.py` expects:

    observation.state : list[float32] of length 26
    action            : list[float32] of length 26
    timestamp         : float32
    frame_index       : int64
    episode_index     : int64
    index             : int64   (global monotonic frame counter)
    task_index        : int64

Usage:
    python 04_write_parquets.py \
        --raw-root /path/to/raw \
        --out-root /path/to/insertion_shaft_300ep_m0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STATE_GROUPS = ("left_arm", "right_arm", "left_ee", "right_ee")
STATE_DIMS = {"left_arm": 7, "right_arm": 7, "left_ee": 6, "right_ee": 6}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--task-index", type=int, default=0,
                   help="Single-task dataset: hard-coded task index (default 0).")
    p.add_argument("--max-episodes", type=int, default=-1)
    return p.parse_args()


def _load_inventory(out_root):
    path = out_root / ".scratch" / "01_episode_inventory.json"
    if not path.exists():
        raise FileNotFoundError(str(path) + " missing - run step 1 first")
    with open(path, "r") as f:
        return json.load(f)


def _flatten_state(state_dict, field):
    """Concatenate the qpos arrays of left_arm, right_arm, left_ee, right_ee."""
    parts = []
    for grp in STATE_GROUPS:
        qpos = state_dict[grp].get(field, [])
        if len(qpos) != STATE_DIMS[grp]:
            raise ValueError(
                "Expected {} dim for {}.{} got {}".format(
                    STATE_DIMS[grp], grp, field, len(qpos)
                )
            )
        parts.append(qpos)
    return np.concatenate(parts).astype(np.float32)


def _build_episode_table(raw_ep_dir, ep_index, length, task_index):
    """Load data.json for one raw episode and return a pandas DataFrame."""
    data_path = raw_ep_dir / "data.json"
    with open(data_path, "r") as f:
        payload = json.load(f)
    frames = payload["data"]
    if len(frames) != length:
        raise ValueError(
            "Frame count mismatch in {}: inventory says {}, data says {}".format(
                raw_ep_dir.name, length, len(frames)
            )
        )

    state_rows = []
    action_rows = []
    for fr in frames:
        state_rows.append(_flatten_state(fr["states"], "qpos"))
        action_rows.append(_flatten_state(fr["actions"], "qpos"))

    state_arr = np.stack(state_rows).astype(np.float32)
    action_arr = np.stack(action_rows).astype(np.float32)

    timestamp = np.arange(length, dtype=np.float32) / 30.0
    frame_index = np.arange(length, dtype=np.int64)
    episode_index_arr = np.full(length, ep_index, dtype=np.int64)

    base_index = ep_index * 100000  # monotonic, large spacing to avoid clashes.
    index = base_index + frame_index

    task_index_arr = np.full(length, task_index, dtype=np.int64)

    return pd.DataFrame({
        "observation.state": [row.tolist() for row in state_arr],
        "action": [row.tolist() for row in action_arr],
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index_arr,
        "index": index,
        "task_index": task_index_arr,
    })


def main():
    args = _parse_args()
    raw_root = args.raw_root
    out_root = args.out_root
    task_index = args.task_index
    max_eps = args.max_episodes

    inventory = _load_inventory(out_root)
    if max_eps > 0:
        inventory = inventory[:max_eps]

    data_root = out_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    for ep_idx, item in enumerate(inventory):
        ep_dir = raw_root / item["episode_dir"]
        df = _build_episode_table(
            raw_ep_dir=ep_dir,
            ep_index=ep_idx,
            length=item["length"],
            task_index=task_index,
        )
        chunk_idx = ep_idx // 1000
        chunk_dir = data_root / ("chunk-{:03d}".format(chunk_idx))
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out_path = chunk_dir / ("episode_{:06d}.parquet".format(ep_idx))
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_path)
        if (ep_idx + 1) % 25 == 0 or ep_idx == len(inventory) - 1:
            print(
                "[04] Wrote {}/{} parquet files -> {}".format(
                    ep_idx + 1, len(inventory), out_path
                ),
                flush=True,
            )

    print(
        "[04] OK: {} parquet files written under {}".format(
            len(inventory), data_root
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
