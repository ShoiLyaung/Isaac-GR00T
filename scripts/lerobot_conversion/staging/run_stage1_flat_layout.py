#!/usr/bin/env python3
"""Wrapper that runs the submodule's Stage 1 conversion on a flat raw layout.

The submodule's `unitree_lerobot/utils/convert_unitree_json_to_lerobot_multicam_keep_effort.py`
expects a `raw_dir/<task>/<episode>/data.json` two-level layout, but the M0 raw
dataset uses a single-level `raw_dir/<episode>/data.json` layout (no task
wrapper). We monkey-patch `iter_episode_paths` to accept the flat layout
before delegating to the original `main()`.

This is a workaround wrapper, not a modification of the submodule.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Submodule paths — keep in sync with M0 spec.
SUBMODULE_ROOT = Path("/home/sl/Isaac-GR00T/external_dependencies/unitree_lerobot")
STAGE1_SCRIPT = (
    SUBMODULE_ROOT
    / "unitree_lerobot"
    / "utils"
    / "convert_unitree_json_to_lerobot_multicam_keep_effort.py"
)

PYTHONPATH_PREPEND = [
    str(SUBMODULE_ROOT / "unitree_lerobot" / "lerobot" / "src"),
    str(SUBMODULE_ROOT),
]
for entry in PYTHONPATH_PREPEND:
    if entry not in sys.path:
        sys.path.insert(0, entry)


def load_stage1_module():
    spec = importlib.util.spec_from_file_location("stage1_submodule", STAGE1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage 1 module from {STAGE1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage1_submodule"] = module
    spec.loader.exec_module(module)
    return module


def flat_iter_episode_paths(raw_dir: Path) -> list[Path]:
    """Episode discovery that accepts `raw_dir/<episode>/data.json` (flat layout).

    Falls back to submodule's two-level layout when nothing matches, so this
    wrapper remains backward compatible.
    """
    episode_paths: list[Path] = []
    # Flat layout: each immediate subdir is an episode, must contain data.json.
    for child in sorted(raw_dir.glob("*")):
        if child.is_dir() and (child / "data.json").exists():
            episode_paths.append(child)
    if episode_paths:
        return episode_paths

    # Fall back to submodule's two-level layout (task/episode/data.json).
    for task_path in sorted(raw_dir.glob("*")):
        if not task_path.is_dir():
            continue
        for episode_path in sorted(task_path.glob("*")):
            if (episode_path / "data.json").exists():
                episode_paths.append(episode_path)
    if not episode_paths and (raw_dir / "data.json").exists():
        episode_paths.append(raw_dir)
    if not episode_paths:
        raise FileNotFoundError(f"No episode data.json files found under {raw_dir}")
    return episode_paths


def main() -> None:
    # Ensure flat-layout discovery is used before main() runs.
    mod = load_stage1_module()
    mod.iter_episode_paths = flat_iter_episode_paths
    mod.main()


if __name__ == "__main__":
    main()
