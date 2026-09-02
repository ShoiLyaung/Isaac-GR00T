# SPDX-FileCopyrightText: Copyright (c) 2026 Isaac-GR00T m0 project. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modality config for the m0 Unitree-G1 dual-arm insertion-shaft dataset.
#
# Structure mirrors ``examples/SO100/so100_config.py`` exactly so that any
# downstream tooling that already understands SO100 will work on our dataset
# without changes. Values are aligned with the per-stream contract defined in
# ``examples/insertion_shaft_m0/modality.json``:
#
#   state / action slices:
#       left_arm  :  0..7   (7-DoF left arm joints)
#       right_arm :  7..14  (7-DoF right arm joints)
#       left_hand : 14..20  (6-DoF Inspire left hand)
#       right_hand: 20..26  (6-DoF Inspire right hand)
#
#   video keys (must match meta/modality.json "video" section):
#       cam_left_high     (observation.images.cam_left_high)
#       cam_left_wrist    (observation.images.cam_left_wrist)
#       cam_right_wrist   (observation.images.cam_right_wrist)
#
#   language key:
#       annotation.human.action.task_description

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


inseration_shaft_m0_config = {
    # Video: current frame only; keys must match "video" entries in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_high",    # left arm third-person overview
            "cam_left_wrist",   # left arm wrist-mounted egocentric view
            "cam_right_wrist",  # right arm wrist-mounted egocentric view
        ],
    ),
    # State: current proprioceptive reading; keys must match "state" entries
    # in meta/modality.json.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",   # 7-DoF left arm joint positions
            "right_arm",  # 7-DoF right arm joint positions
            "left_hand",  # 6-DoF Inspire left hand joint positions
            "right_hand",  # 6-DoF Inspire right hand joint positions
        ],
    ),
    # Action: 16-step prediction horizon; one ActionConfig per modality key.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            # left_arm: RELATIVE = delta from current state (better generalisation).
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # right_arm: same convention as left_arm.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand: ABSOLUTE = target position; Inspire hands behave like
            # a binary gripper so absolute target positions are well-defined.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: same convention as left_hand.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(inseration_shaft_m0_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
