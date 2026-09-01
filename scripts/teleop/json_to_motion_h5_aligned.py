#!/usr/bin/env python3
"""Convert Unitree teleop JSON to a StandG1-style H5 aligned with the H5 replay provider.

Key alignment rule
------------------
The target H5 uses 29 columns, but the *first 17 columns* must be:
    waist(3) + left arm(7) + right arm(7)

That matches the user's H5 replay provider, which slices the first 17 columns as the
upper-body command sequence.

This converter therefore writes columns as:
    0:3   waist
    3:10  left arm
    10:17 right arm
    17:29 lower body (12 joints from sim_state)

Data sources
------------
- Upper body: by default from episode["actions"], because these are the teleop arm commands.
- Lower body/root pose: from sim_state.init_state.
- Hand DOFs are NOT written into the 29-D body motion clip.

Notes
-----
- If body.qpos is empty (common in XR teleop data), waist is filled with zeros.
- action/state qvel is often empty, so upper-body velocity is estimated by finite difference.
- local_rot/global_rot/joint_pos/contact remain structure-compatible placeholders unless
  you later add FK/contact reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

UPPER_DIM = 17
LOWER_DIM = 12
TOTAL_DIM = 29


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_sim_state(frame: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sim_blob = frame.get("sim_state", {}).get("init_state")
    if not sim_blob:
        raise ValueError("Frame is missing sim_state.init_state")
    sim_state = json.loads(sim_blob)
    robot = sim_state["articulation"]["robot"]
    root_pose = np.asarray(robot["root_pose"][0], dtype=np.float32)      # [x,y,z,w,x,y,z]
    joint_pos = np.asarray(robot["joint_position"][0], dtype=np.float32)
    joint_vel = np.asarray(robot["joint_velocity"][0], dtype=np.float32)
    return root_pose[:3], root_pose[3:7], joint_pos, joint_vel


def _safe_qpos(container: Dict, key: str) -> np.ndarray:
    entry = container.get(key, {})
    qpos = entry.get("qpos", []) if isinstance(entry, dict) else []
    return np.asarray(qpos, dtype=np.float32).reshape(-1)


def extract_upper_from_frame(frame: Dict, source: str) -> np.ndarray:
    """Return aligned upper body [waist(3), left_arm(7), right_arm(7)]."""
    if source not in {"actions", "states"}:
        raise ValueError(f"upper source must be 'actions' or 'states', got {source}")

    blob = frame.get(source, {})
    waist = _safe_qpos(blob, "body")
    left_arm = _safe_qpos(blob, "left_arm")
    right_arm = _safe_qpos(blob, "right_arm")

    if left_arm.size != 7 or right_arm.size != 7:
        raise ValueError(
            f"Expected 7 arm joints per side in {source}; got left={left_arm.size}, right={right_arm.size}"
        )

    waist3 = np.zeros(3, dtype=np.float32)
    if waist.size >= 3:
        waist3[:] = waist[:3]

    return np.concatenate([waist3, left_arm, right_arm], axis=0).astype(np.float32)


def finite_difference(arr: np.ndarray, fps: float) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[0] <= 1:
        return out
    out[0] = (arr[1] - arr[0]) * fps
    out[-1] = (arr[-1] - arr[-2]) * fps
    if arr.shape[0] > 2:
        out[1:-1] = (arr[2:] - arr[:-2]) * (0.5 * fps)
    return out


def convert_json_to_motion_h5(
    input_json: Path,
    output_h5: Path,
    fps: float = 100.0,
    group_name: str = "group1",
    upper_source: str = "actions",
    normalize_xy_to_first_frame: bool = True,
    compress: bool = True,
) -> None:
    data = load_json(input_json)
    frames: List[Dict] = data["data"]
    if not frames:
        raise ValueError("Input JSON contains zero frames")

    n = len(frames)
    joint_angle = np.zeros((n, TOTAL_DIM), dtype=np.float32)
    joint_vel = np.zeros((n, TOTAL_DIM), dtype=np.float32)
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_rot = np.zeros((n, 3), dtype=np.float32)

    local_rot = np.tile(np.eye(3, dtype=np.float32), (n, TOTAL_DIM, 1, 1))
    global_rot = np.tile(np.eye(3, dtype=np.float32), (n, TOTAL_DIM, 1, 1))
    joint_pos_out = np.zeros((n, TOTAL_DIM, 3), dtype=np.float32)
    contact = np.zeros((max(n - 1, 0), 2), dtype=bool)

    upper_angle = np.zeros((n, UPPER_DIM), dtype=np.float32)
    lower_angle = np.zeros((n, LOWER_DIM), dtype=np.float32)
    lower_vel = np.zeros((n, LOWER_DIM), dtype=np.float32)

    first_root_xy = None

    for i, frame in enumerate(frames):
        root_xyz, root_quat, jp, jv = parse_sim_state(frame)
        if jp.size < LOWER_DIM or jv.size < LOWER_DIM:
            raise ValueError(
                f"Frame {i}: sim_state joint vector shorter than {LOWER_DIM}. jp={jp.size}, jv={jv.size}"
            )

        upper_angle[i] = extract_upper_from_frame(frame, upper_source)
        lower_angle[i] = jp[:LOWER_DIM]      # keep lower body from sim_state
        lower_vel[i] = jv[:LOWER_DIM]

        if first_root_xy is None:
            first_root_xy = root_xyz[:2].copy()

        if normalize_xy_to_first_frame:
            root_pos[i, :2] = root_xyz[:2] - first_root_xy
            root_pos[i, 2] = root_xyz[2]
        else:
            root_pos[i] = root_xyz

        root_rot[i] = quat_wxyz_to_euler_xyz(root_quat)
        joint_pos_out[i, 0] = root_pos[i]
        global_rot[i, 0] = quat_wxyz_to_rotmat(root_quat)

    upper_vel = finite_difference(upper_angle, fps)

    joint_angle[:, :UPPER_DIM] = upper_angle
    joint_angle[:, UPPER_DIM:] = lower_angle
    joint_vel[:, :UPPER_DIM] = upper_vel
    joint_vel[:, UPPER_DIM:] = lower_vel

    smooth_ang = joint_angle.astype(np.float64)

    compression = "gzip" if compress else None
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group(group_name)
        g.create_dataset("joint_angle", data=joint_angle, compression=compression)
        g.create_dataset("joint_vel", data=joint_vel, compression=compression)
        g.create_dataset("root_pos", data=root_pos, compression=compression)
        g.create_dataset("root_rot", data=root_rot, compression=compression)
        g.create_dataset("local_rot", data=local_rot, compression=compression)
        g.create_dataset("global_rot", data=global_rot, compression=compression)
        g.create_dataset("joint_pos", data=joint_pos_out, compression=compression)
        g.create_dataset("contact", data=contact, compression=compression)
        g.create_dataset("smooth_ang", data=smooth_ang, compression=compression)

        g.attrs["fps"] = float(fps)
        g.attrs["dt"] = float(1.0 / fps)
        g.attrs["source_json"] = str(input_json)
        g.attrs["upper_source"] = upper_source
        g.attrs["joint_layout"] = (
            "0:3 waist, 3:10 left_arm, 10:17 right_arm, 17:29 lower_body_from_sim_state"
        )
        g.attrs["note"] = (
            "Aligned for H5UpperBodyPlayer first-17-column convention; "
            "local_rot/global_rot/joint_pos/contact are structure-compatible placeholders."
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Unitree teleop JSON to aligned StandG1-style H5")
    p.add_argument("input_json", type=Path, help="Path to input data.json")
    p.add_argument("output_h5", type=Path, help="Path to output .h5")
    p.add_argument("--fps", type=float, default=100.0, help="Sampling frequency metadata (default: 100)")
    p.add_argument(
        "--upper-source",
        choices=["actions", "states"],
        default="actions",
        help="Where to take waist/arm trajectories from (default: actions)",
    )
    p.add_argument("--group-name", type=str, default="group1", help="Output H5 group name")
    p.add_argument("--keep-world-xy", action="store_true", help="Do not subtract the first-frame XY root position")
    p.add_argument("--no-compress", action="store_true", help="Disable HDF5 gzip compression")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    convert_json_to_motion_h5(
        input_json=args.input_json,
        output_h5=args.output_h5,
        fps=args.fps,
        group_name=args.group_name,
        upper_source=args.upper_source,
        normalize_xy_to_first_frame=not args.keep_world_xy,
        compress=not args.no_compress,
    )
    print(f"Saved aligned H5: {args.output_h5}")


if __name__ == "__main__":
    main()
