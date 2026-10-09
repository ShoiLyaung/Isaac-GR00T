#!/usr/bin/env python3
"""Validate the frozen Arena G1 H40/execute16 contract without a VLA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from arena_config import (
    ACTION_HORIZON,
    ACTION_KEYS,
    ARENA_G1_N17_CONFIG,
    EXECUTE_HORIZON,
    STATE_KEYS,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.state_action_processor import StateActionProcessor
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "g1-arena-n17-contract-port-r1"
BASE_SOURCE_COMMIT = "5615f267c990c661a8f116c16165b2b23867d6a1"
DOWNLOAD_RESULT_SHA256 = "4e4c957a0365c8e698da3f64a276b7fec7b30f301b23e9b1be6dda7694b9e5e3"
DATASET_REVISION = "07de42fdade76102fc09d921c7b6200d0b31ec01"
EPISODE = 45
FRAMES = 925
SOURCE_IDENTITY_PATHS = (
    "examples/G1_ARENA_N17/arena_config.py",
    "examples/G1_ARENA_N17/cpu_contract_probe.py",
    "tests/examples/test_g1_arena_n17_contract.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(expected_commit: str) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != expected_commit:
        raise ValueError("Arena N1.7 contract source commit changed")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, commit],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("Arena N1.7 contract source is not based on frozen N1.7 base")
    dirty = set()
    for extra in ((), ("--cached",)):
        value = subprocess.check_output(
            ["git", "-C", str(REPO), "diff", *extra, "--name-only", "--", *SOURCE_IDENTITY_PATHS],
            text=True,
        )
        dirty.update(value.splitlines())
    untracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_IDENTITY_PATHS,
        ],
        text=True,
    )
    dirty.update(untracked.splitlines())
    if dirty:
        raise ValueError(f"Arena N1.7 contract source is dirty: {sorted(dirty)}")
    return commit


def verify_download(path: Path, dataset_root: Path) -> dict[str, Any]:
    if sha256(path) != DOWNLOAD_RESULT_SHA256:
        raise ValueError("Arena episode45 download result changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "passed"
        or result.get("revision") != DATASET_REVISION
        or result.get("episode") != EPISODE
    ):
        raise ValueError("wrong Arena episode45 download identity")
    for row in result["files"]:
        source = Path(row["path"])
        relative = Path(row["relative_path"])
        target = dataset_root.parent / relative
        if target.resolve() != source.resolve():
            raise ValueError(f"dataset root does not match download manifest: {relative}")
        if source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]:
            raise ValueError(f"Arena input changed: {source}")
    return result


def nested_finite(values: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(value).all() for value in values.values())


def max_roundtrip_error(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> float:
    return max(float(np.max(np.abs(first[key] - second[key]))) for key in first)


def verify_whitelist(columns: list[str]) -> None:
    expected = {
        *(f"state.{key}" for key in STATE_KEYS),
        *(f"action.{key}" for key in ACTION_KEYS),
        "language.annotation.human.task_description",
    }
    if set(columns) != expected:
        raise ValueError(f"loader field leakage or omission: {sorted(set(columns) ^ expected)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--download-result", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    source_commit = verify_source(args.expected_source_commit)
    verify_download(args.download_result.resolve(), args.dataset_root.resolve())
    loader = LeRobotEpisodeLoader(args.dataset_root, ARENA_G1_N17_CONFIG)
    lowdim = loader._load_parquet_data(EPISODE)
    if len(lowdim) != FRAMES:
        raise ValueError("Arena episode45 frame count changed")
    verify_whitelist(lowdim.columns.tolist())
    anchor = (FRAMES - ACTION_HORIZON) // 2
    decoded = loader._load_video_data(EPISODE, np.asarray([anchor], dtype=np.int64))["ego_view"][0]
    lowdim["video.ego_view"] = [None] * len(lowdim)
    lowdim.at[anchor, "video.ego_view"] = decoded
    step = extract_step_data(
        lowdim, anchor, ARENA_G1_N17_CONFIG, EmbodimentTag.NEW_EMBODIMENT, allow_padding=False
    )
    statistics = loader.get_dataset_statistics()
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    processor = StateActionProcessor(
        modality_configs={tag: ARENA_G1_N17_CONFIG},
        statistics={tag: statistics},
        use_percentiles=False,
        clip_outliers=False,
        apply_sincos_state_encoding=False,
        use_relative_action=False,
    )
    processor.eval()
    normalized_state, normalized_action = processor.apply(step.states, step.actions, tag)
    recovered_state, recovered_action = processor.unapply(normalized_state, normalized_action, tag)
    state_error = max_roundtrip_error(step.states, recovered_state)
    action_error = max_roundtrip_error(step.actions, recovered_action)
    state_shapes = {key: list(value.shape) for key, value in step.states.items()}
    action_shapes = {key: list(value.shape) for key, value in step.actions.items()}
    expected_state = {
        "left_arm": [1, 7],
        "right_arm": [1, 7],
        "left_hand": [1, 7],
        "right_hand": [1, 7],
        "waist": [1, 3],
    }
    expected_action = {
        "left_arm": [40, 7],
        "right_arm": [40, 7],
        "left_hand": [40, 7],
        "right_hand": [40, 7],
        "waist": [40, 3],
        "base_height_command": [40, 1],
        "navigate_command": [40, 3],
    }
    forbidden = {"left_leg", "right_leg", "left_wrist_pose", "right_wrist_pose"}
    gates = {
        "source_commit": source_commit == args.expected_source_commit,
        "frames_925": len(lowdim) == FRAMES,
        "image_shape": list(np.asarray(step.images["ego_view"][0]).shape) == [480, 640, 3],
        "state_shapes": state_shapes == expected_state,
        "action_shapes": action_shapes == expected_action,
        "finite": nested_finite(step.states)
        and nested_finite(step.actions)
        and nested_finite(normalized_state)
        and nested_finite(normalized_action),
        "roundtrip": state_error <= 1e-5 and action_error <= 1e-5,
        "language": "brown box" in step.text.lower() and "blue bin" in step.text.lower(),
        "no_privileged_fields": not (set(step.states) | set(step.actions)) & forbidden,
        "h40": len(ARENA_G1_N17_CONFIG["action"].delta_indices) == 40,
        "execute16_schedule": FRAMES // EXECUTE_HORIZON == 57 and FRAMES % EXECUTE_HORIZON == 13,
    }
    result = {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "experiment_id": EXPERIMENT,
        "source_commit": source_commit,
        "dataset_revision": DATASET_REVISION,
        "download_result": {
            "path": str(args.download_result.resolve()),
            "bytes": args.download_result.stat().st_size,
            "sha256": sha256(args.download_result),
        },
        "episode": EPISODE,
        "anchor": anchor,
        "visible_fields": {
            "video": list(step.images),
            "state": list(step.states),
            "action": list(step.actions),
            "language": step.text,
        },
        "shapes": {"state": state_shapes, "action": action_shapes, "image": list(decoded.shape)},
        "dimensions": {"state": 31, "action": 35},
        "roundtrip_max_abs": {"state": state_error, "action": action_error},
        "schedule": {
            "fps": 50,
            "action_horizon": ACTION_HORIZON,
            "execute_horizon": EXECUTE_HORIZON,
            "full_execute_windows": FRAMES // EXECUTE_HORIZON,
            "tail_frames": FRAMES % EXECUTE_HORIZON,
            "codec_mamba_updates": FRAMES,
        },
        "gates": gates,
        "optimizer_steps": 0,
        "vla_model_instantiated": False,
        "wall_seconds": time.monotonic() - started,
        "claim_boundary": "single-episode CPU N1.7 data and scheduler contract only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
