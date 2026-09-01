import os
import sys
import atexit
from multiprocessing import shared_memory
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


_ISAACLAB_APP = None
_ISAACLAB_APP_CLOSE_REGISTERED = False
_KNOWN_SHARED_MEMORY_NAMES = (
    "isaac_multi_image_shm",
    "isaac_multi_depth_image_shm",
)


def _cleanup_known_shared_memory() -> None:
    for name in _KNOWN_SHARED_MEMORY_NAMES:
        shm = None
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass


def _shutdown_isaaclab_app() -> None:
    global _ISAACLAB_APP
    _cleanup_known_shared_memory()
    if _ISAACLAB_APP is not None:
        try:
            _ISAACLAB_APP.close()
        except Exception:
            pass
        _ISAACLAB_APP = None


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _ensure_isaaclab_app(headless: bool, enable_cameras: bool, device: str):
    global _ISAACLAB_APP, _ISAACLAB_APP_CLOSE_REGISTERED
    if _ISAACLAB_APP is None:
        # Keep import order consistent with unitree_sim_isaaclab scripts.
        try:
            import pinocchio  # noqa: F401
        except Exception:
            pass
        from isaaclab.app import AppLauncher

        _ISAACLAB_APP = AppLauncher(
            headless=headless,
            enable_cameras=enable_cameras,
            device=device,
        ).app
        if not _ISAACLAB_APP_CLOSE_REGISTERED:
            atexit.register(_shutdown_isaaclab_app)
            _ISAACLAB_APP_CLOSE_REGISTERED = True
    return _ISAACLAB_APP


class UnitreeIsaacLabInsertionEnv(gym.Env):
    """Gym adapter that exposes Unitree IsaacLab insertion task with GR00T flat obs/action keys."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    # Mapping used by unitree_sim_isaaclab/robomimic/play.py for insertion tasks.
    LEFT_ARM_ACTION_IDS = [11, 15, 19, 21, 23, 25, 27]
    RIGHT_ARM_ACTION_IDS = [12, 16, 20, 22, 24, 26, 28]
    LEFT_HAND_ACTION_IDS = [31, 32, 30, 29, 43, 33]
    RIGHT_HAND_ACTION_IDS = [36, 37, 35, 34, 48, 38]
    WAIST_ACTION_IDS = [2, 5, 8]

    def __init__(
        self,
        task_name: str,
        *,
        unitree_sim_root: str,
        device: str = "cuda:0",
        headless: bool = True,
        enable_cameras: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.task_name = task_name
        self.unitree_sim_root = unitree_sim_root
        self.device = device
        self.debug = debug
        self._logged_shapes = False
        self._step_idx = 0
        self._last_frame: np.ndarray | None = None
        self._joint_target: np.ndarray | None = None
        hand_dim_env = int(os.environ.get("UNITREE_SIM_HAND_DIM", "7"))
        self.hand_dim = 6 if hand_dim_env <= 6 else 7

        self._bootstrap_pythonpath()
        _ensure_isaaclab_app(headless=headless, enable_cameras=enable_cameras, device=device)

        self._env, self._success_term = self._build_env()
        self._action_dim, self._action_low, self._action_high = self._extract_action_spec()

        sample_obs, _ = self._env.reset()
        flat_sample_obs = self._convert_obs(sample_obs)
        self._last_frame = flat_sample_obs["video.ego_view"]
        self._joint_target = self._read_full_joint_positions()

        self.observation_space = self._build_observation_space(flat_sample_obs)
        self.action_space = self._build_action_space()

    def _bootstrap_pythonpath(self) -> None:
        if not os.path.isdir(self.unitree_sim_root):
            raise FileNotFoundError(
                f"Unitree simulation root not found: {self.unitree_sim_root}"
            )

        os.environ.setdefault("PROJECT_ROOT", self.unitree_sim_root)
        if self.unitree_sim_root not in sys.path:
            sys.path.insert(0, self.unitree_sim_root)

    def _build_env(self):
        import tasks  # noqa: F401  # Registers custom gym tasks.
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(self.task_name, device=self.device, num_envs=1)
        env_cfg.env_name = self.task_name
        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.recorders = None

        success_term = getattr(env_cfg.terminations, "success", None)
        env = gym.make(self.task_name, cfg=env_cfg).unwrapped
        return env, success_term

    def _extract_action_spec(self) -> tuple[int, np.ndarray, np.ndarray]:
        raw_shape = tuple(self._env.action_space.shape)
        if len(raw_shape) == 2:
            action_dim = int(raw_shape[1])
        elif len(raw_shape) == 1:
            action_dim = int(raw_shape[0])
        else:
            raise ValueError(f"Unsupported IsaacLab action space shape: {raw_shape}")

        low = _to_numpy(self._env.action_space.low).reshape(-1)
        high = _to_numpy(self._env.action_space.high).reshape(-1)
        return action_dim, low.astype(np.float32), high.astype(np.float32)

    def _build_action_space(self) -> spaces.Dict:
        inf = np.float32(np.inf)
        return spaces.Dict(
            {
                "action.left_arm": spaces.Box(low=-inf, high=inf, shape=(7,), dtype=np.float32),
                "action.right_arm": spaces.Box(low=-inf, high=inf, shape=(7,), dtype=np.float32),
                "action.left_hand": spaces.Box(
                    low=-inf, high=inf, shape=(self.hand_dim,), dtype=np.float32
                ),
                "action.right_hand": spaces.Box(
                    low=-inf, high=inf, shape=(self.hand_dim,), dtype=np.float32
                ),
                "action.waist": spaces.Box(low=-inf, high=inf, shape=(3,), dtype=np.float32),
                "action.base_height_command": spaces.Box(
                    low=-inf, high=inf, shape=(1,), dtype=np.float32
                ),
                "action.navigate_command": spaces.Box(
                    low=-inf, high=inf, shape=(3,), dtype=np.float32
                ),
            }
        )

    def _build_observation_space(self, sample_obs: dict[str, Any]) -> spaces.Dict:
        observation_space = {}
        for key, value in sample_obs.items():
            if key.startswith("video."):
                observation_space[key] = spaces.Box(
                    low=0,
                    high=255,
                    shape=value.shape,
                    dtype=np.uint8,
                )
            elif key.startswith("state."):
                dim = int(np.asarray(value).shape[-1])
                observation_space[key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(dim,),
                    dtype=np.float32,
                )
            elif key.startswith("annotation."):
                observation_space[key] = spaces.Text(max_length=512)
            else:
                dim = int(np.asarray(value).shape[-1])
                observation_space[key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(dim,),
                    dtype=np.float32,
                )
        return spaces.Dict(observation_space)

    def _read_full_joint_positions(self) -> np.ndarray:
        joint_pos = _to_numpy(self._env.scene["robot"].data.joint_pos)
        if joint_pos.ndim == 2:
            joint_pos = joint_pos[0]
        return joint_pos.astype(np.float32)

    def _get_policy_obs(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        if "policy" in obs_dict:
            return obs_dict["policy"]
        return obs_dict

    def _extract_head_rgb(self, policy_obs: dict[str, Any]) -> np.ndarray:
        image = None
        for key in ("head_rgb", "front_camera", "rgb"):
            if key in policy_obs:
                image = policy_obs[key]
                break
        if image is None:
            for value in policy_obs.values():
                arr = _to_numpy(value)
                if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
                    image = value
                    break
        if image is None:
            raise KeyError("Unable to locate RGB image in IsaacLab policy observation.")

        image_arr = _to_numpy(image)
        if image_arr.ndim == 4:
            image_arr = image_arr[0]
        if image_arr.shape[-1] == 4:
            image_arr = image_arr[..., :3]
        if image_arr.dtype != np.uint8:
            if image_arr.max() <= 1.0:
                image_arr = image_arr * 255.0
            image_arr = np.clip(image_arr, 0.0, 255.0).astype(np.uint8)
        return image_arr

    def _convert_obs(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        policy_obs = self._get_policy_obs(obs_dict)

        robot_joint_state = _to_numpy(policy_obs["robot_joint_state"])
        if robot_joint_state.ndim == 2:
            robot_joint_state = robot_joint_state[0]
        robot_joint_state = robot_joint_state.astype(np.float32)
        if robot_joint_state.shape[0] < 29:
            raise ValueError(
                f"robot_joint_state is too short: {robot_joint_state.shape}. Expected >= 29."
            )
        joint_pos_29 = robot_joint_state[:29]

        inspire_state = _to_numpy(policy_obs.get("robot_inspire_state", np.zeros((1, 12))))
        if inspire_state.ndim == 2:
            inspire_state = inspire_state[0]
        inspire_state = inspire_state.astype(np.float32)
        if inspire_state.shape[0] < 12:
            padded = np.zeros(12, dtype=np.float32)
            padded[: inspire_state.shape[0]] = inspire_state
            inspire_state = padded

        # In this task's observation ordering: first 6 are right hand, last 6 are left hand.
        right_hand_6 = inspire_state[:6]
        left_hand_6 = inspire_state[6:12]

        if self.hand_dim == 7:
            left_hand = np.concatenate([left_hand_6, np.zeros(1, dtype=np.float32)], axis=0)
            right_hand = np.concatenate([right_hand_6, np.zeros(1, dtype=np.float32)], axis=0)
        else:
            left_hand = left_hand_6
            right_hand = right_hand_6

        head_rgb = self._extract_head_rgb(policy_obs)
        self._last_frame = head_rgb

        task_text = (
            str(policy_obs.get("annotation.human.task_description", "")).strip()
            or str(policy_obs.get("task", "")).strip()
            or self.task_name.replace("-", " ")
        )

        flat_obs = {
            "video.ego_view": head_rgb,
            # Backward-compatible alias used by some GR1-trained checkpoints.
            "video.ego_view_bg_crop_pad_res256_freq20": head_rgb,
            "state.left_leg": joint_pos_29[0:6],
            "state.right_leg": joint_pos_29[6:12],
            "state.waist": joint_pos_29[12:15],
            "state.left_arm": joint_pos_29[15:22],
            "state.right_arm": joint_pos_29[22:29],
            "state.left_hand": left_hand,
            "state.right_hand": right_hand,
            "annotation.human.task_description": task_text,
        }

        if self.debug and not self._logged_shapes:
            message = " | ".join(
                [
                    f"{key}:{np.asarray(value).shape}"
                    for key, value in flat_obs.items()
                    if key != "annotation.human.task_description"
                ]
            )
            print(
                f"[UnitreeIsaacLabInsertionEnv] obs sample | {message} | "
                f"task={flat_obs['annotation.human.task_description']}",
                flush=True,
            )
            self._logged_shapes = True

        return flat_obs

    def _set_joint_values(
        self,
        target: np.ndarray,
        current: np.ndarray,
        indices: list[int],
        values: np.ndarray,
        *,
        relative: bool,
    ) -> None:
        valid_indices = [idx for idx in indices if idx < target.shape[0]]
        if not valid_indices:
            return
        n = min(len(valid_indices), values.shape[0])
        dst = np.array(valid_indices[:n], dtype=np.int64)
        src_values = values[:n].astype(np.float32, copy=False)
        if relative:
            target[dst] = current[dst] + src_values
        else:
            target[dst] = src_values

    def _map_action(self, action: dict[str, Any]) -> np.ndarray:
        current = self._read_full_joint_positions()
        if self._joint_target is None:
            self._joint_target = current.copy()

        target = self._joint_target.copy()
        left_arm = np.asarray(action["action.left_arm"], dtype=np.float32).reshape(-1)
        right_arm = np.asarray(action["action.right_arm"], dtype=np.float32).reshape(-1)
        left_hand = np.asarray(action["action.left_hand"], dtype=np.float32).reshape(-1)
        right_hand = np.asarray(action["action.right_hand"], dtype=np.float32).reshape(-1)
        waist = np.asarray(action["action.waist"], dtype=np.float32).reshape(-1)

        # GR00T Unitree_G1 convention: arm actions are relative deltas.
        self._set_joint_values(
            target, current, self.LEFT_ARM_ACTION_IDS, left_arm, relative=True
        )
        self._set_joint_values(
            target, current, self.RIGHT_ARM_ACTION_IDS, right_arm, relative=True
        )
        # Hand and waist are treated as absolute set-points in this wrapper.
        self._set_joint_values(
            target, current, self.LEFT_HAND_ACTION_IDS, left_hand[:6], relative=False
        )
        self._set_joint_values(
            target, current, self.RIGHT_HAND_ACTION_IDS, right_hand[:6], relative=False
        )
        self._set_joint_values(target, current, self.WAIST_ACTION_IDS, waist, relative=False)

        clip_n = min(target.shape[0], self._action_dim, self._action_low.shape[0], self._action_high.shape[0])
        finite_mask = np.isfinite(self._action_low[:clip_n]) & np.isfinite(self._action_high[:clip_n])
        if np.any(finite_mask):
            clipped = target[:clip_n]
            clipped[finite_mask] = np.clip(
                clipped[finite_mask],
                self._action_low[:clip_n][finite_mask],
                self._action_high[:clip_n][finite_mask],
            )
            target[:clip_n] = clipped

        self._joint_target = target
        action_vec = target[: self._action_dim].astype(np.float32, copy=False).reshape(1, -1)
        return action_vec

    def _compute_success(self) -> bool:
        if self._success_term is None:
            return False
        try:
            success = self._success_term.func(self._env, **self._success_term.params)
            success_np = _to_numpy(success).reshape(-1)
            return bool(success_np[0])
        except Exception:
            return False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            try:
                self._env.seed(seed)
            except Exception:
                pass
        obs_dict, _ = self._env.reset()
        flat_obs = self._convert_obs(obs_dict)
        self._joint_target = self._read_full_joint_positions()
        self._step_idx = 0
        return flat_obs, {}

    def step(self, action: dict[str, Any]):
        mapped_action = self._map_action(action)
        try:
            import torch

            env_action = torch.from_numpy(mapped_action).to(device=self.device)
        except Exception:
            env_action = mapped_action

        obs_dict, reward, terminated, truncated, info = self._env.step(env_action)
        flat_obs = self._convert_obs(obs_dict)
        self._step_idx += 1

        reward_scalar = float(_to_numpy(reward).reshape(-1)[0])
        terminated_scalar = bool(_to_numpy(terminated).reshape(-1)[0])
        truncated_scalar = bool(_to_numpy(truncated).reshape(-1)[0])
        success = self._compute_success()

        _ = info
        return flat_obs, reward_scalar, terminated_scalar, truncated_scalar, {"success": success}

    def render(self):
        if self._last_frame is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return self._last_frame

    def close(self):
        self._env.close()
        _cleanup_known_shared_memory()


def create_unitree_isaaclab_insertion_env(task_name: str) -> UnitreeIsaacLabInsertionEnv:
    unitree_sim_root = os.environ.get(
        "UNITREE_SIM_ROOT", "/home/sl/humanoid-lfd/unitree_sim_isaaclab"
    )
    device = os.environ.get("UNITREE_SIM_DEVICE", "cuda:0")
    headless = os.environ.get("UNITREE_SIM_HEADLESS", "1") != "0"
    enable_cameras = os.environ.get("UNITREE_SIM_ENABLE_CAMERAS", "1") != "0"
    debug = os.environ.get("UNITREE_SIM_DEBUG", "0") == "1"

    return UnitreeIsaacLabInsertionEnv(
        task_name=task_name,
        unitree_sim_root=unitree_sim_root,
        device=device,
        headless=headless,
        enable_cameras=enable_cameras,
        debug=debug,
    )
