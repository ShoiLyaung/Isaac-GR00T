"""Modality config for finetuning GR00T-N1.7 on the insertion-with-torque dataset.

The dataset (``groot_datasets/insertion_with_torque_groot``) has 33 episodes
with 26-dim state/action: left_arm(7), right_arm(7), left_hand(6), right_hand(6).
At eval time the env expects 7 action keys (the missing waist/base_height/
navigate are zero-filled by the slim policy wrapper — see
``gr00t/model/gr00t_n1d7/policy.py:_get_action``).
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


MODALITY_CONFIGS["new_embodiment"] = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],  # maps from observation.images.cam_left_high via dataset loader alias
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),  # N1.7 horizon
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            # left_arm: relative joint delta
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm: relative joint delta
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand: absolute (gripper-like binary signals)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}
