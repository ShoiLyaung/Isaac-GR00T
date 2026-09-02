"""UnitreeG1Env: thin wrapper that bridges Unitree IsaacLab ManagerBasedRLEnv
to GR00T's flat-key observation/action contract.

Observation contract (flat keys):
  - video.ego_view:           np.uint8 (H, W, 3)
  - state.left_leg:           np.float32 (6,)
  - state.right_leg:          np.float32 (6,)
  - state.waist:              np.float32 (3,)
  - state.left_arm:           np.float32 (7,)
  - state.right_arm:          np.float32 (7,)
  - state.left_hand:          np.float32 (6,)
  - state.right_hand:         np.float32 (6,)
  - annotation.human.action.task_description: str

Action contract (flat dict):
  - action.left_arm:           np.float32 (7,)   RELATIVE delta
  - action.right_arm:          np.float32 (7,)   RELATIVE delta
  - action.left_hand:          np.float32 (6,)   ABSOLUTE set-point
  - action.right_hand:         np.float32 (6,)   ABSOLUTE set-point
  - action.waist:              np.float32 (3,)   ABSOLUTE set-point
  - action.base_height_command: np.float32 (1,)  absolute
  - action.navigate_command:    np.float32 (3,)  absolute (zero)

The wrapper maps these flat dicts into a single full-joint-dim joint-position
target that the underlying IsaacLab env accepts via its configured
``JointPositionAction`` term. Joint ID resolution is performed at runtime
by querying physx's actual ``joint_names`` list — we never hardcode indices,
because the underlying articulation ordering can shift with USD revisions.

Arms use RELATIVE deltas from the cached target; hands and waist use
ABSOLUTE replacement; legs/commands keep their last value (the policy is
not expected to drive them).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


# ---------- path bootstrap ----------

# The Unitree IsaacLab tasks package is not pip-installed; it lives at a
# fixed location on this machine and must be importable before isaaclab.*
# or task registration modules are touched.
_UNITREE_SIM_ROOT = "/home/sl/humanoid-lfd/unitree_sim_isaaclab"
os.environ.setdefault("PROJECT_ROOT", _UNITREE_SIM_ROOT)
if _UNITREE_SIM_ROOT not in sys.path:
    sys.path.insert(0, _UNITREE_SIM_ROOT)


# Boy-joint count used by the obs chunk (29 DoF body without inspire fingers).
# This is the size of ``robot_joint_state``'s first 29 dims, not the size of
# the full robot articulation tensor (which includes hand joints and is
# larger — 43 or 49 depending on the USD). The action term uses the *full*
# robot joint count, resolved at runtime via ``joint_names``.
BOY_NUM_JOINTS = 29


class UnitreeG1Env(gym.Env):
    """Bridge from Unitree IsaacLab ManagerBasedRLEnv to GR00T flat-key contract."""

    # Class-level guard so that AppLauncher is only spun up once per process.
    # ``gym.make`` for vectorized envs can call into us repeatedly.
    _app_started: bool = False

    def __init__(self, task_name: str, **kwargs: Any):
        super().__init__()

        # Lazy import / first-time AppLauncher bootstrap.
        if not UnitreeG1Env._app_started:
            from isaaclab.app import AppLauncher

            self._app_launcher = AppLauncher(
                headless=True,
                enable_cameras=True,
                device="cuda:0",
            )
            # Touching .app triggers actual startup.
            self._app_launcher.app
            UnitreeG1Env._app_started = True

        # Importing the tasks package populates gym registry with the
        # insertion-shaft and other Unitree G1 envs.
        import tasks  # noqa: F401

        # Isaac Lab's `ManagerBasedRLEnv.__init__` requires a real
        # `cfg: ManagerBasedEnvCfg` object — not the `env_cfg_entry_point`
        # string that gym registered. We resolve the entry point through
        # the standard Isaac Lab helper, then pass `cfg=...` to gym.make.
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

        self._env = gym.make(task_name, cfg=env_cfg).unwrapped
        self._task_name = task_name

        # Cache device for action-tensor placement.
        self._device = self._env.scene["robot"].data.default_joint_pos.device

        # Cached joint-position target. Arms use RELATIVE on top of this;
        # hands/waist overwrite their slots. Legs stay at default.
        self._joint_target: np.ndarray | None = None

        # Initialize joint-id mappings. We need a reset() to populate
        # ``self._env.scene["robot"].data.joint_names`` with the actual
        # physx articulation ordering — that ordering depends on the USD
        # asset revision and must be queried, not hardcoded.
        self._resolve_joint_ids()

        # Resolve the ``success`` termination term once so step() can read
        # it cheaply. We accept either a TerminationTerm object (with
        # ``.func``/``.params``) or a plain callable.
        self._success_func = None
        self._success_params: dict[str, Any] = {}
        term_mgr = getattr(self._env, "termination_manager", None)
        if term_mgr is not None and hasattr(term_mgr, "terms"):
            term = term_mgr.terms.get("success")
            if term is not None:
                if hasattr(term, "func"):
                    self._success_func = term.func
                    self._success_params = getattr(term, "params", {}) or {}
                elif callable(term):
                    self._success_func = term

        # Spaces: obs keys are matched against the unitree_g1_full_body_*
        # pre-registered modality config (state.* / video.* / annotation.*).
        # The trained model's modality config declares 3 video streams
        # (cam_left_high / cam_left_wrist / cam_right_wrist) but the env
        # only provides 1 camera (head_rgb / g1_front_camera). We replicate
        # the same image into all 3 video keys so the model can be served.
        self.observation_space = spaces.Dict(
            {
                "video.cam_left_high": spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
                "video.cam_left_wrist": spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
                "video.cam_right_wrist": spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
                "state.left_leg": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "state.right_leg": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "state.waist": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "state.left_arm": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "state.right_arm": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "state.left_hand": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "state.right_hand": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "annotation.human.action.task_description": spaces.Text(max_length=512),
            }
        )
        # The trained model only outputs 4 action keys (left_arm,
        # right_arm, left_hand, right_hand). The 3 untrained lower-body /
        # base / nav keys are zero-filled in _map_action below. The
        # action_space must match what the policy outputs to keep
        # gymnasium.vector.SyncVectorEnv happy.
        # M0-R-left experiment: only left_arm + left_hand (the active 13-dim
        # signal). right_arm / right_hand / waist / base_height / nav are
        # NOT in action_space — env keeps them at default (0). This is the
        # controlled test for "data has 14 dead channels → train only on
        # 2 active streams".
        self.action_space = spaces.Dict(
            {
                "action.left_arm": spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
                "action.left_hand": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
            }
        )

    # ----- joint-id resolution -----

    def _ids_by_name(self, joint_names, patterns):
        """Find joint IDs whose names contain any of the given substrings.

        Returns a list of (id, name) pairs in the order the patterns appear,
        filtered to the first match per pattern. This is the building block
        used by ``_resolve_joint_ids`` to find specific joints by name.
        """
        out = []
        used = set()
        for pat in patterns:
            for i, name in enumerate(joint_names):
                if i in used:
                    continue
                if pat in name:
                    out.append((i, name))
                    used.add(i)
                    break
        return out

    def _resolve_joint_ids(self):
        """Query physx for the actual joint name list, then return ordered slices.

        The order matches the dataset's ``boy_joint_indices`` for the 29-DoF
        body chunk (legs / waist / arms), plus the inspire hand IDs that
        are documented in ``tasks/common_observations/inspire_state.py``.

        The actual physx joint order depends on the USD asset; we never
        hardcode IDs because that order can shift with USD revisions.
        """
        # The articulation may not have joint_names populated until after
        # the first reset(). Call reset() here defensively (it is
        # idempotent in IsaacLab).
        try:
            joint_names = list(self._env.scene["robot"].data.joint_names)
        except Exception as exc:
            raise RuntimeError(
                f"Could not query joint_names before reset; got: {exc}"
            ) from exc

        if not joint_names:
            # Force a reset and try again.
            try:
                self._env.reset()
            except Exception as exc:
                raise RuntimeError(
                    f"joint_names empty and reset() failed: {exc}"
                ) from exc
            joint_names = list(self._env.scene["robot"].data.joint_names)

        self.num_joints = len(joint_names)
        self._joint_name_list = list(joint_names)

        # Arm: 7 joints each. Order matters because the policy's action
        # vector is positionally aligned with the dataset's boy_joint_indices.
        # left arm order:
        left_arm_pat = [
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "left_wrist_roll",
            "left_wrist_pitch",
            "left_wrist_yaw",
        ]
        right_arm_pat = [
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
            "right_wrist_roll",
            "right_wrist_pitch",
            "right_wrist_yaw",
        ]
        left_arm_pairs = self._ids_by_name(joint_names, left_arm_pat)
        right_arm_pairs = self._ids_by_name(joint_names, right_arm_pat)

        # Waist: 3 joints in yaw/roll/pitch order.
        waist_pairs = self._ids_by_name(
            joint_names, ["waist_yaw", "waist_roll", "waist_pitch"]
        )

        # Inspire hand: in this USD the hand joints use the L_/R_ prefix
        # (e.g. ``L_index_proximal_joint``). The dataset's inspire_joint_indices
        # already documents the canonical 6-dim left/right hand selection.
        # The pattern set below picks the 6 most-important hand joints
        # per side, in the same order as the dataset modality config
        # (index_proximal/intermediate, middle_proximal/intermediate,
        # thumb_proximal_yaw/pitch). We drop thumb_intermediate / thumb_distal
        # / pinky / ring to keep the action stream at 6 dim.
        left_hand_pat = [
            "L_index_proximal",
            "L_index_intermediate",
            "L_middle_proximal",
            "L_middle_intermediate",
            "L_thumb_proximal_yaw",
            "L_thumb_proximal_pitch",
        ]
        right_hand_pat = [
            "R_index_proximal",
            "R_index_intermediate",
            "R_middle_proximal",
            "R_middle_intermediate",
            "R_thumb_proximal_yaw",
            "R_thumb_proximal_pitch",
        ]
        left_hand_pairs = self._ids_by_name(joint_names, left_hand_pat)
        right_hand_pairs = self._ids_by_name(joint_names, right_hand_pat)

        # Validate counts — surface a clear diagnostic on failure.
        if len(left_arm_pairs) != 7:
            raise RuntimeError(
                f"LEFT_ARM joints: expected 7, got {len(left_arm_pairs)} -> "
                f"{left_arm_pairs}"
            )
        if len(right_arm_pairs) != 7:
            raise RuntimeError(
                f"RIGHT_ARM joints: expected 7, got {len(right_arm_pairs)} -> "
                f"{right_arm_pairs}"
            )
        if len(waist_pairs) != 3:
            raise RuntimeError(
                f"WAIST joints: expected 3, got {len(waist_pairs)} -> {waist_pairs}"
            )
        if len(left_hand_pairs) != 6:
            raise RuntimeError(
                f"LEFT_HAND joints: expected 6, got {len(left_hand_pairs)} -> "
                f"{left_hand_pairs}"
            )
        if len(right_hand_pairs) != 6:
            raise RuntimeError(
                f"RIGHT_HAND joints: expected 6, got {len(right_hand_pairs)} -> "
                f"{right_hand_pairs}"
            )

        # Strip names to leave just IDs.
        self.LEFT_ARM_IDS = [i for i, _ in left_arm_pairs]
        self.RIGHT_ARM_IDS = [i for i, _ in right_arm_pairs]
        self.WAIST_IDS = [i for i, _ in waist_pairs]
        self.LEFT_HAND_IDS = [i for i, _ in left_hand_pairs]
        self.RIGHT_HAND_IDS = [i for i, _ in right_hand_pairs]

        # Joint-id -> readable name, useful for debugging.
        self.LEFT_ARM_NAMES = [n for _, n in left_arm_pairs]
        self.RIGHT_ARM_NAMES = [n for _, n in right_arm_pairs]
        self.WAIST_NAMES = [n for _, n in waist_pairs]
        self.LEFT_HAND_NAMES = [n for _, n in left_hand_pairs]
        self.RIGHT_HAND_NAMES = [n for _, n in right_hand_pairs]

        # Sanity: no duplicates, all in range.
        all_ids = (
            self.LEFT_ARM_IDS
            + self.RIGHT_ARM_IDS
            + self.WAIST_IDS
            + self.LEFT_HAND_IDS
            + self.RIGHT_HAND_IDS
        )
        if len(set(all_ids)) != len(all_ids):
            dupes = {x for x in all_ids if all_ids.count(x) > 1}
            raise RuntimeError(f"duplicate joint ids in action mapping: {dupes}")
        if any(i < 0 or i >= self.num_joints for i in all_ids):
            raise RuntimeError(
                f"joint id out of range (num_joints={self.num_joints}): "
                f"{[(i, joint_names[i]) for i in all_ids if i < 0 or i >= self.num_joints]}"
            )

    # ----- helpers -----

    def _convert_obs(self, obs_dict: dict) -> dict:
        """Map IsaacLab nested obs -> GR00T flat-key dict."""
        policy_obs = obs_dict["policy"]

        # robot_joint_state: shape (num_envs, 58) = 29 pos + 29 vel
        joint_state = policy_obs["robot_joint_state"]
        if hasattr(joint_state, "cpu"):
            joint_state = joint_state.cpu()
        # obs ordering follows boy_joint_indices from g1_29dof_state.py
        # which already permutes leg/waist/arm into a canonical layout.
        joint_pos = np.asarray(joint_state[0, :BOY_NUM_JOINTS], dtype=np.float32)

        # robot_inspire_state: shape (num_envs, 12) in the inspire_joint_indices order
        inspire_state = policy_obs["robot_inspire_state"]
        if hasattr(inspire_state, "cpu"):
            inspire_state = inspire_state.cpu()
        inspire_pos = np.asarray(inspire_state[0], dtype=np.float32)
        right_hand_6 = inspire_pos[:6].astype(np.float32)
        left_hand_6 = inspire_pos[6:12].astype(np.float32)

        # head_rgb: shape (num_envs, H, W, 3) uint8
        rgb = policy_obs["head_rgb"][0]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb, dtype=np.uint8)
        # The obs chunk layout for boy_joint_indices is
        #   leg[12] | waist[3] | arm[14]
        # We split it accordingly.
        obs_leg_left = joint_pos[0:6]
        obs_leg_right = joint_pos[6:12]
        obs_waist = joint_pos[12:15]
        obs_arm = joint_pos[15:29]  # 14 dims -> left(7) + right(7) in obs order

        return {
            # The trained model's modality config declares 3 video streams
            # (cam_left_high / cam_left_wrist / cam_right_wrist) but the
            # Unitree IsaacLab env only provides 1 camera (head_rgb /
            # g1_front_camera). We replicate the same image into all 3
            # video keys so the model can be served by this env without
            # changing the saved modality config.
            "video.cam_left_high": rgb,
            "video.cam_left_wrist": rgb,
            "video.cam_right_wrist": rgb,
            "state.left_leg": obs_leg_left,
            "state.right_leg": obs_leg_right,
            "state.waist": obs_waist,
            "state.left_arm": obs_arm[0:7],
            "state.right_arm": obs_arm[7:14],
            "state.left_hand": left_hand_6,
            "state.right_hand": right_hand_6,
            "annotation.human.action.task_description": self._task_name,
        }

    def _map_action(self, action: dict) -> np.ndarray:
        """Compose a full-joint-dim target vector from the 4-stream flat dict.

        Returns a 1D ``np.float32`` array of length ``self.num_joints``.
        The caller (``step``) is responsible for subtracting
        ``default_joint_pos`` to undo the ``JointPositionAction`` offset.
        """
        asset = self._env.scene["robot"]

        # Use the robot's current joint positions as the anchor. This
        # preserves whatever the simulator's PD controller was tracking
        # last step, instead of forcing a reset back to default.
        if self._joint_target is None or len(self._joint_target) != self.num_joints:
            current_joint_pos = asset.data.joint_pos
            if hasattr(current_joint_pos, "cpu"):
                current_joint_pos = current_joint_pos[0].cpu().numpy().astype(np.float32)
            else:
                current_joint_pos = np.asarray(current_joint_pos[0], dtype=np.float32)
            self._joint_target = current_joint_pos.copy()

        target = self._joint_target.copy()

        # arms: RELATIVE delta (only update the slots the policy actually
        # wrote — never touch legs / waist / hands if the policy is silent).
        if "action.left_arm" in action:
            left_arm_delta = np.asarray(action["action.left_arm"], dtype=np.float32)
            for i, idx in enumerate(self.LEFT_ARM_IDS):
                target[idx] = target[idx] + left_arm_delta[i]
        if "action.right_arm" in action:
            right_arm_delta = np.asarray(action["action.right_arm"], dtype=np.float32)
            for i, idx in enumerate(self.RIGHT_ARM_IDS):
                target[idx] = target[idx] + right_arm_delta[i]

        # hands: ABSOLUTE set-point.
        if "action.left_hand" in action:
            left_hand_abs = np.asarray(action["action.left_hand"], dtype=np.float32)
            for i, idx in enumerate(self.LEFT_HAND_IDS):
                target[idx] = left_hand_abs[i]
        if "action.right_hand" in action:
            right_hand_abs = np.asarray(action["action.right_hand"], dtype=np.float32)
            for i, idx in enumerate(self.RIGHT_HAND_IDS):
                target[idx] = right_hand_abs[i]

        # waist: ABSOLUTE set-point. The trained model doesn't emit waist /
        # base_height / nav commands; we keep the env target at its
        # current value for those slots rather than letting them
        # silently zero.
        if "action.waist" in action and len(action["action.waist"]) == len(self.WAIST_IDS):
            waist_abs = np.asarray(action["action.waist"], dtype=np.float32)
            for i, idx in enumerate(self.WAIST_IDS):
                target[idx] = waist_abs[i]

        self._joint_target = target
        return target

    def _compute_success(self) -> bool:
        if self._success_func is None:
            return False
        try:
            result = self._success_func(self._env, **self._success_params)
            if hasattr(result, "cpu"):
                result = result.cpu().numpy()
            arr = np.asarray(result)
            if arr.ndim == 0:
                return bool(arr.item())
            return bool(arr[0])
        except Exception:
            return False

    def _compute_shaft_housing_distance(self) -> float:
        """Compute the current shaft-housing distance (in meters).

        Used to track per-step progress for diagnostic purposes. Returns
        ``float("nan")`` if the env scene does not expose the required
        objects (e.g. a different task) — never raises.
        """
        try:
            shaft_pos = self._env.scene["object"].data.root_pos_w[0].cpu().numpy()
            housing_pos = self._env.scene["housing"].data.root_pos_w[0].cpu().numpy()
            return float(np.linalg.norm(shaft_pos - housing_pos))
        except Exception:
            return float("nan")

    # ----- gym.Env API -----

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._env.reset(seed=seed)
        # Refresh joint-id resolution in case the underlying articulation
        # was re-instantiated by reset (it normally isn't, but this is
        # cheap and defensive).
        if not hasattr(self, "LEFT_ARM_IDS"):
            self._resolve_joint_ids()
        self._joint_target = None
        # Diagnostic: clear per-episode shaft-housing distance history.
        self._dist_history = []
        info = dict(info) if info else {}
        info.setdefault("success", False)
        return self._convert_obs(obs), info

    def step(self, action: dict):
        target_np = self._map_action(action)

        # IsaacLab's ``JointPositionAction`` computes:
        #   final_target = raw_action * scale + default_offset
        # We want the simulated joints to track ``target_np`` (which is in
        # the same units as ``joint_pos``). For the G1 29-DoF env the
        # ``JointPositionAction`` config uses ``use_default_offset=True``
        # and ``scale=1.0``, so:
        #   raw_action = target_np - default_joint_pos
        # to undo the offset. This is the only correct way to drive
        # joints at non-default positions.
        asset = self._env.scene["robot"]
        default_pos = asset.data.default_joint_pos
        if hasattr(default_pos, "cpu"):
            default_np = default_pos[0].cpu().numpy().astype(np.float32)
        else:
            default_np = np.asarray(default_pos[0], dtype=np.float32)

        # Defensive: if the resolved target dim doesn't match default dim,
        # something is wrong. Surface it clearly.
        if target_np.shape != default_np.shape:
            raise RuntimeError(
                f"target dim {target_np.shape} != default dim {default_np.shape}; "
                f"num_joints={self.num_joints}"
            )

        raw_action = (target_np - default_np).astype(np.float32)

        import torch

        action_tensor = torch.from_numpy(raw_action).to(self._device).unsqueeze(0)
        # Make sure it is contiguous / float32 (matches action term dtype).
        action_tensor = action_tensor.contiguous().to(dtype=torch.float32)

        obs, reward, terminated, truncated, info = self._env.step(action_tensor)

        info = dict(info) if info else {}
        info["success"] = self._compute_success()
        # Diagnostic: track shaft-housing distance per step (and keep the
        # full per-step history available to the rollout caller for
        # post-hoc analysis of failure mode A/B/C).
        if not hasattr(self, "_dist_history"):
            self._dist_history = []
        self._dist_history.append(self._compute_shaft_housing_distance())
        info["shaft_housing_distance"] = self._dist_history[-1]
        info["shaft_housing_min"] = min(self._dist_history)
        info["shaft_housing_distance_history"] = list(self._dist_history)
        return self._convert_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ---------- registration ----------

_REGISTERED = False


def register_unitree_g1_envs() -> None:
    """Register the UnitreeG1 GR00T envs with gym (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return

    register(
        id="unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play",
        entry_point="gr00t.eval.sim.UnitreeG1.unitree_g1_env:UnitreeG1Env",
        kwargs={"task_name": "Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"},
        disable_env_checker=True,
    )

    _REGISTERED = True
