"""External GR00T N1.7 modality contract for the Arena G1 task."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


STATE_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
ACTION_KEYS = STATE_KEYS + ("base_height_command", "navigate_command")
ACTION_HORIZON = 40
EXECUTE_HORIZON = 16


def absolute_action_config() -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


ARENA_G1_N17_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_KEYS)),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=list(ACTION_KEYS),
        action_configs=[absolute_action_config() for _ in ACTION_KEYS],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(ARENA_G1_N17_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
