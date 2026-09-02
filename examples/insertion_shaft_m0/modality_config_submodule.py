"""Modality config for submodule-converted insertion_shaft dataset.

This dataset is produced by `/home/sl/Isaac-GR00T/external_dependencies/unitree_lerobot/`
Stage 2 (`convert_unitree_inspire_multicam_to_groot_keep_effort.py`).

Key differences from `modality_config_insertion_shaft_m0.py`:
- Only 1 camera (`cam_left_high`) — the submodule dropped cam_left_wrist and cam_right_wrist.
- Effort is in dataset but we don't use it as input (kept for compatibility / future use).
- **`language` modality is required** (upstream loader asserts `len(language_data) == 1`).
  We work around the loader's `tasks_map = {task_index (int): task_desc (str)}`
  bug by writing `tasks.jsonl` with a **string** task_index matching the
  annotation.human.action.task_description column value (`insertion_shaft_300ep_submodule`).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

inseration_shaft_submodule_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_left_high"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),  # left_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),  # right_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),  # left_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),  # right_hand
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
    # NOTE: No `language` modality (omitted to avoid upstream loader bug;
    # see docstring).
}
register_modality_config(
    inseration_shaft_submodule_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)