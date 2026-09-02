"""Modality config for LEFT-ONLY finetune (M0-R-left experiment).

This config removes `right_arm` and `right_hand` from both state and
action streams, based on M3 vision analysis (2026-06-30) which confirmed
that M0-R is fully static (0% pixel change t=0→8s) because the model
output is dominated by 14/26 dead channels in the original data.

By training the model on 2 streams (left_arm + left_hand = 13 dim) instead
of 4 streams (26 dim), we:
- Eliminate 14 dead channels that were trivial-pulling loss to 0.027
- Force the model to learn the only active signal: left arm + left hand
- Expected improvement: from "static robot" to "robot moves left arm"

Wrapper side (UnitreeG1Env):
- action_space has only 2 keys (left_arm, left_hand) to keep
  gymnasium.vector.SyncVectorEnv happy.
- joint_ids are computed dynamically: only left arm (15-21) + left hand
  (29-35, picked 6 of 7 hand joints to match dataset).
- Other joints (right_arm, right_hand, waist) stay at their default
  values in `_joint_target` — no driving, no zero-filling. Env's
  JointPositionAction receives a 1D numpy array where these slots are
  physically written as 0 (default) and the env's use_default_offset
  keeps them at 0.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

inseration_shaft_left_only_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_left_high"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        # 13 dims: left_arm(7) + left_hand(6) — only active signal.
        modality_keys=["left_arm", "left_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        # 13 dims: left_arm(7) RELATIVE + left_hand(6) ABSOLUTE.
        modality_keys=["left_arm", "left_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),  # left_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),  # left_hand
        ],
    ),
    # NOTE: `language` modality is **required** by the upstream GR00T
    # loader (`sharded_single_step_dataset.py:69` asserts
    # `len(language_data) == 1`). We re-add it here. The M0 dataset's
    # `modality.json` maps `annotation.human.action.task_description` to
    # the parquet's `task_index` column, and `tasks.jsonl` translates
    # `task_index=0` to the task string — so the loader can synthesize
    # the language field at read time.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}
register_modality_config(
    inseration_shaft_left_only_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)