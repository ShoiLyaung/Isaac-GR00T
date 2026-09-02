"""Modality registration for ID10_INSPIRE6_TASK_BASELINE_H40_NATIVE30."""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.g1_id10_h40_native30_adapter import h40_native30_offsets
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


TAG = "new_embodiment"

MODALITY_CONFIGS[TAG] = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=h40_native30_offsets(),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
            "base_height_command",
            "navigate_command",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm",
            ),
            *[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in range(5)
            ],
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}
