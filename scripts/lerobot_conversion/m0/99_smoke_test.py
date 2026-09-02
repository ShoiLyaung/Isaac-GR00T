#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 99 - Smoke-test the converted LeRobot v2.1 dataset via m0-clean loader.

This script:
1. Imports the m0-clean ``LeRobotEpisodeLoader`` and instantiates it on the
   converted dataset.
2. Loads episode 0, prints per-modality keys + shapes.
3. Sanity-checks that:
       observation.state      -> (length, 26)
       action                 -> (16, 26)  at every reachable step
       observation.images.<*> -> 1 frame per camera (current timestep only)
       language               -> str
4. Prints the recorded task name from episodes.jsonl / tasks.jsonl.

If anything is wrong, the script exits non-zero with a descriptive message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", required=True, type=Path,
                   help="Converted LeRobot v2.1 dataset root.")
    p.add_argument("--modality-config-path", required=True, type=Path,
                   help=".py file that calls register_modality_config(...).")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--episode-index", type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    out_root = args.out_root
    modality_cfg = args.modality_config_path
    embodiment_tag = args.embodiment_tag

    if not (out_root / "meta" / "info.json").exists():
        print("ERROR: meta/info.json missing under", out_root, flush=True)
        return 1
    if not (out_root / "meta" / "modality.json").exists():
        print("ERROR: meta/modality.json missing under", out_root, flush=True)
        return 1
    if not (out_root / "meta" / "stats.json").exists():
        print("ERROR: meta/stats.json missing - run gr00t/data/stats.py first",
              flush=True)
        return 1

    # Import m0-clean modules fresh.
    repo_root = Path("/home/sl/Isaac-GR00T-m0-clean")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # 1. Load custom modality config (side-effect: register it).
    cfg_path = modality_cfg.resolve()
    if not cfg_path.exists():
        print("ERROR: modality config .py not found:", cfg_path, flush=True)
        return 1
    import importlib.util
    spec = importlib.util.spec_from_file_location(cfg_path.stem, str(cfg_path))
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    print("[99] Loaded modality config:", cfg_path, flush=True)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    et = EmbodimentTag(embodiment_tag)
    modality_configs = MODALITY_CONFIGS[et.value]
    print("[99] Embodiment tag:", et.value, flush=True)
    print("[99] Modality config keys:", list(modality_configs.keys()), flush=True)

    loader = LeRobotEpisodeLoader(
        dataset_path=out_root,
        modality_configs=modality_configs,
        video_backend="torchcodec",
    )
    print("[99] Loader len (episodes):", len(loader), flush=True)

    ep_idx = args.episode_index
    if ep_idx < 0 or ep_idx >= len(loader):
        print("ERROR: episode_index out of range", flush=True)
        return 1
    df = loader[ep_idx]
    print("[99] Episode", ep_idx, "DataFrame columns:", list(df.columns), flush=True)
    print("[99] Episode", ep_idx, "row count:", len(df), flush=True)

    expected_cols = {
        "state.left_arm", "state.right_arm",
        "state.left_hand", "state.right_hand",
        "action.left_arm", "action.right_arm",
        "action.left_hand", "action.right_hand",
        "video.cam_left_high", "video.cam_left_wrist", "video.cam_right_wrist",
        "language.annotation.human.action.task_description",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        print("ERROR: missing columns:", sorted(missing), flush=True)
        return 1
    print("[99] All expected columns present.", flush=True)

    # Each row holds a single per-frame slice. The 16-step horizon is assembled
    # downstream by `extract_step_data` (used by ShardedSingleStepDataset).
    # We re-implement that stacking inline to avoid importing the heavy
    # `transformers` + `flash_attn` chain that pulls in the m0 model code.

    # Inspect per-row shapes.
    s_larm = np.asarray(df["state.left_arm"].iloc[0])
    s_rarm = np.asarray(df["state.right_arm"].iloc[0])
    s_lhand = np.asarray(df["state.left_hand"].iloc[0])
    s_rhand = np.asarray(df["state.right_hand"].iloc[0])
    print("[99] state.left_arm[0]  shape={} dtype={}".format(s_larm.shape, s_larm.dtype), flush=True)
    print("[99] state.right_arm[0] shape={} dtype={}".format(s_rarm.shape, s_rarm.dtype), flush=True)
    print("[99] state.left_hand[0] shape={} dtype={}".format(s_lhand.shape, s_lhand.dtype), flush=True)
    print("[99] state.right_hand[0]shape={} dtype={}".format(s_rhand.shape, s_rhand.dtype), flush=True)
    state_total = s_larm.size + s_rarm.size + s_lhand.size + s_rhand.size
    if state_total != 26:
        print("ERROR: state total dim is {}, expected 26".format(state_total), flush=True)
        return 1
    print("[99] state total dim OK: 26", flush=True)

    a_larm = np.asarray(df["action.left_arm"].iloc[0])
    a_rarm = np.asarray(df["action.right_arm"].iloc[0])
    a_lhand = np.asarray(df["action.left_hand"].iloc[0])
    a_rhand = np.asarray(df["action.right_hand"].iloc[0])
    print("[99] action.left_arm[0]  shape={} dtype={}".format(a_larm.shape, a_larm.dtype), flush=True)
    print("[99] action.right_arm[0] shape={} dtype={}".format(a_rarm.shape, a_rarm.dtype), flush=True)
    print("[99] action.left_hand[0] shape={} dtype={}".format(a_lhand.shape, a_lhand.dtype), flush=True)
    print("[99] action.right_hand[0]shape={} dtype={}".format(a_rhand.shape, a_rhand.dtype), flush=True)
    expected_action_dims = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6}
    actual = {
        "left_arm": a_larm.size,
        "right_arm": a_rarm.size,
        "left_hand": a_lhand.size,
        "right_hand": a_rhand.size,
    }
    if actual != expected_action_dims:
        print("ERROR: action dims {} expected {}".format(actual, expected_action_dims), flush=True)
        return 1
    print("[99] action joint dims OK (7/7/6/6 per row, total 26).", flush=True)

    # Video column: each row of the loader DataFrame holds ONE decoded frame
    # of shape (H, W, C) = (480, 640, 3). The loader decodes the entire video
    # once and aligns each frame to its row. Per-step single-frame indexing
    # is done by extract_step_data below.
    frame_high = np.asarray(df["video.cam_left_high"].iloc[0])
    frame_left = np.asarray(df["video.cam_left_wrist"].iloc[0])
    frame_right = np.asarray(df["video.cam_right_wrist"].iloc[0])
    print("[99] cam_left_high  frame[0] shape={} dtype={}".format(frame_high.shape, frame_high.dtype), flush=True)
    print("[99] cam_left_wrist frame[0] shape={} dtype={}".format(frame_left.shape, frame_left.dtype), flush=True)
    print("[99] cam_right_wristframe[0] shape={} dtype={}".format(frame_right.shape, frame_right.dtype), flush=True)
    expected_video_shape = (480, 640, 3)
    for name, arr in [("cam_left_high", frame_high),
                      ("cam_left_wrist", frame_left),
                      ("cam_right_wrist", frame_right)]:
        if arr.shape != expected_video_shape:
            print("ERROR: {} frame[0] shape {} expected {}".format(
                name, arr.shape, expected_video_shape
            ), flush=True)
            return 1
        if arr.dtype != np.uint8:
            print("ERROR: {} dtype {}".format(name, arr.dtype), flush=True)
            return 1
    print("[99] video frames OK (480x640x3 uint8, all 3 cameras).", flush=True)

    lang = df["language.annotation.human.action.task_description"].iloc[0]
    print("[99] language[0] =", repr(lang), flush=True)
    if not isinstance(lang, str) or not lang:
        print("ERROR: language should be a non-empty string", flush=True)
        return 1
    print("[99] language OK.", flush=True)

    # Now run the same logic that extract_step_data uses, inline, to validate
    # the 16-step action stacking works against our dataset. Inlined to avoid
    # importing the heavy `transformers` + `flash_attn` chain pulled in by
    # `sharded_single_step_dataset.py` (env_isaaclab has a flash-attn symbol
    # conflict that is unrelated to data correctness).
    print("[99] Running inline 16-step stacking at step 0 ...", flush=True)
    step_data = {"state": {}, "action": {}, "video": {}, "language": {}}
    for modality, cfg in modality_configs.items():
        indices_to_load = [0 + d for d in cfg.delta_indices]  # step_index = 0
        for key in cfg.modality_keys:
            col = "{}.{}".format(modality, key)
            if col not in df.columns:
                print("ERROR: column missing", col, flush=True)
                return 1
            sub = df[col].iloc[indices_to_load]
            if modality in ("state", "action"):
                step_data[modality][key] = np.vstack(
                    [np.array(sub.iloc[i]).astype(np.float32) for i in range(len(sub))]
                )
            elif modality == "video":
                # Each row holds one (H, W, C) frame; stack into (N, H, W, C).
                step_data["video"][key] = np.stack(
                    [np.asarray(sub.iloc[i]).astype(np.uint8) for i in range(len(sub))],
                    axis=0,
                )
            else:
                step_data[modality][key] = sub.tolist()

    text = step_data["language"][list(step_data["language"].keys())[0]][0]
    print("[99] inline text:", repr(text), flush=True)
    for img_key, arr in step_data["video"].items():
        print("[99]   image[{}] shape={} dtype={}".format(img_key, arr.shape, arr.dtype), flush=True)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            print("ERROR: image[{}] shape unexpected".format(img_key), flush=True)
            return 1
    for k, v in step_data["state"].items():
        print("[99]   state[{}] shape={} dtype={}".format(k, v.shape, v.dtype), flush=True)
    for k, v in step_data["action"].items():
        print("[99]   action[{}] shape={} dtype={}".format(k, v.shape, v.dtype), flush=True)
        if v.shape != (16, expected_action_dims[k]):
            print("ERROR: action[{}] shape {} expected ({},{})".format(
                k, v.shape, 16, expected_action_dims[k]
            ), flush=True)
            return 1
    print("[99] inline 16-step stacking OK.", flush=True)

    # Print stats summary.
    with open(out_root / "meta" / "stats.json") as f:
        stats = json.load(f)
    print("[99] stats.json keys:", list(stats.keys()), flush=True)
    if "observation.state" in stats and "action" in stats:
        s_dim = len(stats["observation.state"]["mean"])
        a_dim = len(stats["action"]["mean"])
        print("[99] stats['observation.state'] mean dim={}".format(s_dim), flush=True)
        print("[99] stats['action'] mean dim={}".format(a_dim), flush=True)
        if s_dim != 26 or a_dim != 26:
            print("ERROR: stats dims mismatch", flush=True)
            return 1
        print("[99] stats dims OK (26 each).", flush=True)

    rel_path = out_root / "meta" / "relative_stats.json"
    if rel_path.exists():
        with open(rel_path) as f:
            rel = json.load(f)
        print("[99] relative_stats.json keys:", list(rel.keys()), flush=True)
    else:
        print("[99] (relative_stats.json absent - OK if no RELATIVE action configs)",
              flush=True)

    print("[99] SMOKE TEST PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
