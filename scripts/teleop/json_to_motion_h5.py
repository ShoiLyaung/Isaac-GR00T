#!/usr/bin/env python3
"""Convert Unitree teleop JSON data into a motion-library-like HDF5 file.

This script is designed for JSON files recorded by xr_teleoperate / unitree_sim_isaaclab
and writes an H5 file with the same top-level dataset names as a typical StandG1 motion
library clip:
    /group1/joint_angle
    /group1/joint_vel
    /group1/root_pos
    /group1/root_rot
    /group1/local_rot
    /group1/global_rot
    /group1/joint_pos
    /group1/contact
    /group1/smooth_ang

Important note
--------------
The source JSON contains enough information to reconstruct:
    - root pose / root velocity
    - full robot joint position / velocity
But it does NOT directly contain per-link FK outputs such as joint_pos, local_rot,
global_rot, or foot contact flags in the same form as the motion-library H5.

Therefore this script produces a *structure-compatible* H5:
    - joint_angle / joint_vel / root_pos / root_rot are populated from real data
    - smooth_ang duplicates joint_angle
    - local_rot / global_rot / joint_pos / contact are placeholder fields
      with valid shapes, so downstream code that only reads joint_angle will work.

If you later need physically accurate local_rot / global_rot / joint_pos / contact,
you must add URDF-based forward kinematics or export those values directly from sim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
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
    """Convert quaternion [w, x, y, z] to XYZ Euler angles in radians."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)



def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def extract_frame_robot_state(frame: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return root_xyz, root_quat_wxyz, joint_pos_53, joint_vel_53 from one frame."""
    sim_blob = frame.get("sim_state", {}).get("init_state")
    if not sim_blob:
        raise ValueError("Frame is missing sim_state.init_state")

    sim_state = json.loads(sim_blob)
    robot = sim_state["articulation"]["robot"]

    root_pose = np.asarray(robot["root_pose"][0], dtype=np.float32)         # [x,y,z,w,x,y,z]
    joint_pos = np.asarray(robot["joint_position"][0], dtype=np.float32)    # length 53 in current file
    joint_vel = np.asarray(robot["joint_velocity"][0], dtype=np.float32)    # length 53 in current file

    root_xyz = root_pose[:3]
    root_quat = root_pose[3:7]
    return root_xyz, root_quat, joint_pos, joint_vel



def convert_json_to_motion_h5(
    input_json: Path,
    output_h5: Path,
    fps: float = 100.0,
    group_name: str = "group1",
    joint_count: int = 29,
    normalize_xy_to_first_frame: bool = True,
) -> None:
    data = load_json(input_json)
    frames: List[Dict] = data["data"]
    if len(frames) == 0:
        raise ValueError("Input JSON contains zero frames")

    n = len(frames)

    joint_angle = np.zeros((n, joint_count), dtype=np.float32)
    joint_vel = np.zeros((n, joint_count), dtype=np.float32)
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_rot = np.zeros((n, 3), dtype=np.float32)

    # Placeholder fields for structure compatibility.
    local_rot = np.tile(np.eye(3, dtype=np.float32), (n, joint_count, 1, 1))
    global_rot = np.tile(np.eye(3, dtype=np.float32), (n, joint_count, 1, 1))
    joint_pos_out = np.zeros((n, joint_count, 3), dtype=np.float32)
    contact = np.ones((max(n - 1, 0), 2), dtype=bool)

    first_root_xy = None

    for i, frame in enumerate(frames):
        xyz, quat_wxyz, jp53, jv53 = extract_frame_robot_state(frame)
        if jp53.shape[0] < joint_count or jv53.shape[0] < joint_count:
            raise ValueError(
                f"Frame {i}: joint vector shorter than {joint_count}. "
                f"joint_position={jp53.shape[0]}, joint_velocity={jv53.shape[0]}"
            )

        # Use the first 29 entries:
        # 12 lower-body + 3 waist + 14 arm joints.
        joint_angle[i] = jp53[:joint_count]
        joint_vel[i] = jv53[:joint_count]

        if first_root_xy is None:
            first_root_xy = xyz[:2].copy()

        if normalize_xy_to_first_frame:
            root_pos[i, 0:2] = xyz[:2] - first_root_xy
            root_pos[i, 2] = xyz[2]
        else:
            root_pos[i] = xyz

        root_rot[i] = quat_wxyz_to_euler_xyz(quat_wxyz)

        # Placeholders with at least some meaningful content:
        joint_pos_out[i, 0] = root_pos[i]
        global_rot[i, 0] = quat_wxyz_to_rotmat(quat_wxyz)

    smooth_ang = joint_angle.astype(np.float64)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group(group_name)
        g.create_dataset("joint_angle", data=joint_angle, compression="gzip")
        g.create_dataset("joint_vel", data=joint_vel, compression="gzip")
        g.create_dataset("root_pos", data=root_pos, compression="gzip")
        g.create_dataset("root_rot", data=root_rot, compression="gzip")
        g.create_dataset("local_rot", data=local_rot, compression="gzip")
        g.create_dataset("global_rot", data=global_rot, compression="gzip")
        g.create_dataset("joint_pos", data=joint_pos_out, compression="gzip")
        g.create_dataset("contact", data=contact, compression="gzip")
        g.create_dataset("smooth_ang", data=smooth_ang, compression="gzip")

        g.attrs["fps"] = float(fps)
        g.attrs["dt"] = float(1.0 / fps)
        g.attrs["source_json"] = str(input_json)
        g.attrs["joint_count"] = int(joint_count)
        g.attrs["note"] = (
            "joint_angle/joint_vel/root_pos/root_rot come from sim_state; "
            "local_rot/global_rot/joint_pos/contact are placeholders unless you add FK/contact export."
        )



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Unitree teleop JSON to motion-like H5")
    parser.add_argument("input_json", type=Path, help="Path to input data.json")
    parser.add_argument("output_h5", type=Path, help="Path to output .h5")
    parser.add_argument("--fps", type=float, default=100.0, help="Sampling frequency to write into attrs (default: 100)")
    parser.add_argument("--group-name", type=str, default="group1", help="Output H5 group name (default: group1)")
    parser.add_argument("--joint-count", type=int, default=29, help="Number of whole-body joints to export from sim_state joint vectors")
    parser.add_argument(
        "--keep-world-xy",
        action="store_true",
        help="Keep root x/y in world frame instead of subtracting first frame x/y",
    )
    return parser



def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    convert_json_to_motion_h5(
        input_json=args.input_json,
        output_h5=args.output_h5,
        fps=args.fps,
        group_name=args.group_name,
        joint_count=args.joint_count,
        normalize_xy_to_first_frame=not args.keep_world_xy,
    )
    print(f"Saved: {args.output_h5}")



if __name__ == "__main__":
    main()
