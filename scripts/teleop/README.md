# scripts/teleop/

Data-collection and conversion scripts for Unitree XR-teleop demonstrations.
Copied verbatim from `~/teleop/` so they live next to the rest of the Isaac-GR00T
scripts tree (`scripts/train/`, `scripts/eval/`, `scripts/lerobot_conversion/`,
`scripts/deployment/`) and are discoverable by Claude Code without having to
remember an external path.

The canonical source of truth is still `~/teleop/`. If you update a script
there, please re-copy it here (or replace this directory with a sync).

---

## Layout

```
scripts/teleop/
├── README.md                         (this file)
│
├── json_to_motion_h5.py              ← standalone, run anywhere with h5py
├── json_to_motion_h5_aligned.py      ← standalone, run anywhere with h5py
│
├── teleop_hand_and_arm.py            ← CANONICAL — use this one
├── teleop_hand_and_arm_print_tau.py  ← [DEPRECATED] older debug variant
├── teleop_hand_and_arm_old.py        ← [DEPRECATED] original 1.0 version
│
└── teleop/                           ← Python package, mirrors ~/teleop/*
    ├── __init__.py
    ├── robot_control/                ← arm/hand controllers + IK + retargeting
    │   ├── robot_arm.py
    │   ├── robot_arm_ik.py
    │   ├── robot_arm_original.py
    │   ├── robot_hand_unitree.py
    │   ├── robot_hand_inspire.py
    │   ├── robot_hand_brainco.py
    │   ├── hand_retargeting.py
    │   └── dex-retargeting/          ← third-party submodule (MIT, see its README)
    │
    ├── televuer/                     ← XR (Quest/Pico) teleop server
    │   ├── src/televuer/
    │   │   ├── televuer.py
    │   │   └── tv_wrapper.py
    │   └── example/test_televuer.py
    │
    ├── teleimager/                   ← camera streaming (USB / MIPI)
    │   ├── src/teleimager/
    │   │   ├── image_server.py
    │   │   └── image_client.py
    │   └── cam_config_*.yaml
    │
    └── utils/                        ← shared utilities
        ├── episode_writer.py         ← writes data.json + colors + depths per episode
        ├── ipc.py
        ├── motion_switcher.py        ← H5 motion-library integration
        ├── rerun_visualizer.py       ← Rerun SDK visualization (requires `rerun-sdk`)
        ├── sim_state_topic.py
        └── weighted_moving_filter.py
```

**Excluded** on purpose: `~/teleop/utils/data/` (≈ 4.1 GB of recorded episodes).
It stays in `~/teleop/utils/data/` because it's a raw data sink, not code.

---

## What each script does

### `json_to_motion_h5.py` — basic JSON → H5

Convert a single episode's `data.json` (recorded by `teleop_hand_and_arm.py`)
into a StandG1 motion-library-style H5 file with the 9 canonical groups
(`joint_angle`, `joint_vel`, `root_pos`, `root_rot`, `local_rot`, `global_rot`,
`joint_pos`, `contact`, `smooth_ang`). FK-derived fields are placeholders.

```bash
python scripts/teleop/json_to_motion_h5.py episode_0000/data.json out.h5 \
    --fps 100 --group-name group1
```

### `json_to_motion_h5_aligned.py` — aligned JSON → H5

Same as above, but **the first 17 columns** of `joint_angle` are guaranteed to be
`waist(3) + left_arm(7) + right_arm(7)`. Use this if the H5 will be consumed by
the H5 replay provider that slices `[0:17]` as the upper-body command.

```bash
python scripts/teleop/json_to_motion_h5_aligned.py episode_0000/data.json out.h5 \
    --fps 100 --upper-source actions --keep-world-xy
```

### `teleop_hand_and_arm.py` — **CANONICAL** teleop + recording entry point

Live XR teleop loop with sim/real-robot dispatch, optional motion control, IPC
keyboard input, optional `--record` mode that writes one episode folder per
session to `--task-dir`.

Differences from the deprecated variants:
- Adds `arm_ctrl.get_current_dual_arm_tau_est()` + 1-Hz print loop
- Falls back to `getattr(head_img, "bgr", head_img)` if the camera frame has
  no `.bgr` attribute (some teleimager versions return raw RGB)
- Populates `qvel` and `torque` in the saved JSON instead of empty lists

```bash
# Real robot, XR hand tracking, with recording
python scripts/teleop/teleop_hand_and_arm.py \
    --arm G1_29 --ee inspire_ftp \
    --img-server-ip 192.168.123.164 \
    --record --task-name "insert_shaft" \
    --task-dir /home/sl/teleop/utils/data/

# Sim mode (IsaacLab)
python scripts/teleop/teleop_hand_and_arm.py \
    --sim --headless --arm G1_29 --ee inspire_ftp \
    --img-server-ip 127.0.0.1 --record \
    --task-name "pick_cube"
```

### `teleop_hand_and_arm_print_tau.py` — **[DEPRECATED]**

Same as canonical but only adds the tau-print loop; does NOT have the
`head_img.bgr` fallback. Kept for bisecting older recordings.

### `teleop_hand_and_arm_old.py` — **[DEPRECATED]**

Original 1.0 version. Saves empty `qvel` and `torque` lists in the JSON, has no
tau print and no `.bgr` fallback. Kept for historical reference only.

---

## Environment

The `teleop_hand_and_arm*.py` scripts need:

| Package | Where it normally comes from |
|---|---|
| `unitree_sdk2py` | `pip install unitree-sdk2py` (or vendored) |
| `sshkeyboard` | `pip install sshkeyboard` |
| `numpy`, `h5py` | conda / pip |
| `rerun-sdk` | `pip install rerun-sdk` (only for `utils/rerun_visualizer.py`) |

The `logging_mp` calls in `episode_writer.py` use the **0.x API** (`getLogger`,
`basicConfig`); the **1.x API** is `get_logger` / `basic_config`. Make sure you
run the scripts in the conda env that has the 0.x API — `env_isaaclab` and
`gr00t_n1d7` here ship 1.x and will fail on import of `episode_writer`. The
expected env is one built from the original `~/teleop/setup` (e.g. with
`pip install unitree-sdk2py==<0.x>`).

The `json_to_motion_h5*.py` scripts only need `h5py` + `numpy` and run in any
Python env.

---

## How to keep this in sync with `~/teleop/`

When you change a script in `~/teleop/`:

```bash
# Re-copy a single file
cp -v ~/teleop/json_to_motion_h5.py scripts/teleop/
cp -v ~/teleop/teleop_hand_and_arm.py scripts/teleop/

# Or re-sync the whole tree (excludes ~/teleop/utils/data which is 4.1 GB)
rsync -av --exclude='utils/data' --exclude='*.egg-info' --exclude='__pycache__' \
    ~/teleop/ scripts/teleop/
```

After re-syncing, sanity-check:

```bash
# Confirm all scripts still compile
for f in scripts/teleop/*.py; do
  python -m py_compile "$f" && echo "OK: $f"
done

# Confirm the package still imports (best-effort; requires the right env)
python -c "
import sys; sys.path.insert(0, 'scripts/teleop')
import teleop
from teleop.utils import episode_writer, ipc, motion_switcher, sim_state_topic, weighted_moving_filter, rerun_visualizer
print('package import OK')
"
```

---

## See also

- `scripts/train/` — how to finetune N1.6 / N1.7 on the recorded episodes
- `scripts/lerobot_conversion/` — convert LeRobot v3 datasets to v2
- `scripts/eval/` — rollout evaluation against IsaacLab envs
- `~/teleop/utils/data/` — raw recorded episodes (4.1 GB; not in git)