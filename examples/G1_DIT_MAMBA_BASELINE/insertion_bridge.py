"""Model-side conversion to the existing insertion simulator joint targets.

Inspire units and dependent-finger targets match humanoid-lfd's recording
path: dds/inspire_dds.py and action_provider/action_provider_dds.py.
"""

import numpy as np


PROTOCOL_ID = "g1_qonly26_observation_left13_execution_r1"
ARM_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
HAND_NAMES = (
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
)
HAND_MIN = np.array([0, 0, 0, 0, 0, -0.1], dtype=np.float32)
HAND_MAX = np.array([1.7, 1.7, 1.7, 1.7, 0.5, 1.3], dtype=np.float32)
HAND_DEPENDENCIES = {
    "L_pinky_intermediate_joint": (0, 1.0),
    "L_ring_intermediate_joint": (1, 1.0),
    "L_middle_intermediate_joint": (2, 1.0),
    "L_index_intermediate_joint": (3, 1.0),
    "L_thumb_intermediate_joint": (4, 1.5),
    "L_thumb_distal_joint": (4, 2.4),
}


def hand_radians_to_u(q):
    return np.clip((HAND_MAX - q) / (HAND_MAX - HAND_MIN), 0.0, 1.0)


def hand_u_to_radians(u):
    return (1.0 - np.clip(u, 0.0, 1.0)) * (HAND_MAX - HAND_MIN) + HAND_MIN


def combine_body_telemetry(qdq, tau):
    qdq = np.asarray(qdq, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)
    if qdq.shape != (58,) or tau.shape != (29,):
        raise ValueError("Simulator body helper provides q/dq[58]; applied torque must provide[29]")
    # The evaluator obtains these fields, but only q enters this baseline.
    return qdq[:29].copy()


def observation_from_sim(body_q, inspire_right_then_left, rgb):
    body = np.asarray(body_q, dtype=np.float32)
    hand = np.asarray(inspire_right_then_left, dtype=np.float32)
    image = np.asarray(rgb)
    if (
        body.shape != (29,)
        or hand.shape != (12,)
        or not np.isfinite(body).all()
        or not np.isfinite(hand).all()
    ):
        raise ValueError("Expected finite simulator q[29] and Inspire angles[12]")
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("Expected native head RGB uint8[480,640,3]")
    values = {
        "left_leg": np.zeros(6),
        "right_leg": np.zeros(6),
        "waist": np.zeros(3),
        "left_arm": body[15:22],
        "right_arm": body[22:29],
        "left_hand": hand_radians_to_u(hand[6:12]),
        "right_hand": hand_radians_to_u(hand[:6]),
    }
    return {
        "video": {"ego_view": image[None, None]},
        "state": {
            key: np.asarray(value, dtype=np.float32)[None, None] for key, value in values.items()
        },
        "language": {"annotation.human.task_description": [["insert the shaft into the housing"]]},
    }


def action_to_sim(left_arm_q, left_hand_u, joint_names, default_q):
    arm = np.asarray(left_arm_q, dtype=np.float32)
    hand = np.asarray(left_hand_u, dtype=np.float32)
    default_q = np.asarray(default_q, dtype=np.float32)
    if (
        arm.shape != (7,)
        or hand.shape != (6,)
        or not np.isfinite(arm).all()
        or not np.isfinite(hand).all()
    ):
        raise ValueError("Expected finite decoded left arm[7] and Inspire command[6]")
    if default_q.shape != (len(joint_names),):
        raise ValueError("Simulator joint names/default target mismatch")
    angles = hand_u_to_radians(hand)
    targets = dict(zip(ARM_NAMES, arm, strict=True))
    targets.update(zip(HAND_NAMES, angles, strict=True))
    targets.update(
        {key: angles[index] * scale for key, (index, scale) in HAND_DEPENDENCIES.items()}
    )
    result = np.zeros_like(default_q)
    for key, q in targets.items():
        index = joint_names.index(key)
        # R2 JointPositionAction uses default offsets. The decoded arm target
        # is already absolute; do not add the current arm state again.
        result[index] = q - default_q[index]
    return result
