"""Fork-local slim Gr00tN1d7Processor for the G1 IsaacLab shaft-insertion task.

This subclass bypasses the upstream ``AutoProcessor.from_pretrained`` path
(which would load the checkpoint's baked-in ``real_g1_relative_eef_relative_joints``
EEF config — incompatible with our joint-delta env) and instead constructs the
processor with our local joint-delta config keyed on the fork's existing
``EmbodimentTag.UNITREE_G1`` value (``"unitree_g1"``).

Critical: this file lives under ``gr00t_n1d7/`` and is only imported when an
N1.7 server explicitly opts in. The N1.6 path is unaffected.
"""

from __future__ import annotations

from typing import Any

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from gr00t.model.gr00t_n1d7.policy import (
    ENV_ACTION_DIMS,
    ENV_STATE_DIMS,
)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7_upstream import Gr00tN1d7Processor


# Fork-local joint-delta G1 modality config, keyed on the fork's existing
# ``EmbodimentTag.UNITREE_G1`` value (``"unitree_g1"``). Mirrors the upstream
# ``unitree_g1_full_body_with_waist_height_nav_cmd`` config's semantics
# (RELATIVE/NON_EEF arms, ABSOLUTE/NON_EEF hands/waist/base/nav) but with
# action horizon = 40 (matching the N1.7-3B model's ``action_horizon=40``).
#
# State streams include ``left_leg``/``right_leg`` because our env's
# ``UnitreeIsaacLabInsertionEnv._convert_obs`` emits them (see
# ``insertion_env.py:300-304``).
N1D7_G1_SHAFT_MODALITY_CONFIG: dict[str, ModalityConfig] = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
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
        delta_indices=list(range(40)),
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
            # left_arm: relative joint delta (joint-delta convention)
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
            # left_hand: absolute (gripper signals)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: absolute (gripper signals)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # waist: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # base_height_command: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # navigate_command: absolute
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


# Per the upstream EMBODIMENT_TAG_TO_PROJECTOR_INDEX, the Unitree G1
# joint-delta slot uses projector id 25 (same as the upstream's
# ``unitree_g1_full_body_with_waist_height_nav_cmd`` and
# ``real_g1_relative_eef_relative_joints`` tags).
N1D7_G1_SHAFT_EMBODIMENT_ID: int = 25


class Gr00tN1d7G1ShaftProcessor(Gr00tN1d7Processor):
    """Fork-local slim processor for the G1 IsaacLab shaft-insertion task.

    Accepts the upstream constructor but pre-fills the modality config and
    embodiment id mapping with the joint-delta G1 schema. Bypasses the
    upstream ``AutoProcessor.from_pretrained`` path entirely — callers should
    construct this class directly, not via ``AutoProcessor``.
    """

    def __init__(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        embodiment_id_mapping: dict[str, int] | None = None,
        **overrides: Any,
    ):
        modality_configs = {
            "unitree_g1": N1D7_G1_SHAFT_MODALITY_CONFIG,
        }
        if embodiment_id_mapping is None:
            embodiment_id_mapping = {
                "unitree_g1": N1D7_G1_SHAFT_EMBODIMENT_ID,
            }
        # The upstream statistics.json may contain entries for many
        # embodiments (e.g. xdof, real_g1, oxe_droid). Our slim
        # modality_configs only knows "unitree_g1". The state_action_processor
        # iterates over ALL statistics keys and looks up each in
        # modality_configs — so we must filter statistics to keep only the
        # keys present in modality_configs.
        #
        # For N1.7-3B specifically, the statistics.json has entries under
        # "real_g1_relative_eef_relative_joints" (the EEF config N1.7-3B was
        # trained with). We map this to our "unitree_g1" key so the
        # state_action_processor finds it. Then we fill in synthetic stats
        # (mean=0, std=1, min=-1, max=1) for any state/action keys not
        # covered by the source stats (e.g. left_leg/right_leg, which are
        # not in the EEF config).
        if statistics is not None:
            mapped = {}
            # First try direct match (no mapping needed)
            for k, v in statistics.items():
                if k in modality_configs:
                    mapped[k] = v
            # Fallback: use real_g1 stats for unitree_g1 (closest G1 embodiment)
            if "unitree_g1" not in mapped:
                for candidate in (
                    "real_g1_relative_eef_relative_joints",
                    "real_g1",
                    "unitree_g1_full_body_with_waist_height_nav_cmd",
                ):
                    if candidate in statistics:
                        mapped["unitree_g1"] = statistics[candidate]
                        break
            # Fill in synthetic stats for any state/action keys our config
            # expects but the source stats don't have. The mean=0/std=1
            # normalization is essentially identity (z = (x - 0) / 1 = x).
            # min=-1/max=1 gives range 2 so range_vals = 2 - (-1) = 2 ≠ 0.
            #
            # CRITICAL: For known env keys (state and action), ALWAYS
            # overwrite the source stats with the env-correct dim. The
            # source's "left_hand" may be 7-dim but our env's "left_hand"
            # is 6-dim — using source stats would cause a dim mismatch
            # during normalization.
            if "unitree_g1" in mapped:
                cfg = mapped["unitree_g1"]
                # Determine per-key dim from the env (preferred) for state
                # and action keys, otherwise use 1.
                from gr00t.configs.data.embodiment_configs import (
                    MODALITY_CONFIGS as _MC,
                )

                mod_cfg = _MC.get(
                    "unitree_g1", modality_configs["unitree_g1"]
                )
                for mod_name in ("state", "action"):
                    if mod_name not in mod_cfg:
                        continue
                    keys = mod_cfg[mod_name].modality_keys
                    for k in keys:
                        if mod_name == "state":
                            dim = ENV_STATE_DIMS.get(k, 1)
                        elif mod_name == "action":
                            dim = ENV_ACTION_DIMS.get(k, 1)
                        else:
                            dim = 1
                        cfg.setdefault(mod_name, {})[k] = {
                            "mean": [0.0] * dim,
                            "std": [1.0] * dim,
                            "min": [-1.0] * dim,
                            "max": [1.0] * dim,
                            "q01": [-1.0] * dim,
                            "q99": [1.0] * dim,
                        }
            statistics = mapped

        # N1.7-3B defaults (from its config.json) for image preprocessing.
        # The upstream's __init__ uses None defaults, which causes image
        # augmentation builders to fail. Override with the N1.7-3B values.
        defaults = dict(
            image_crop_size=[230, 230],
            image_target_size=[256, 256],
            shortest_image_edge=256,
            crop_fraction=0.95,
            use_albumentations=True,
            use_percentiles=True,
            formalize_language=True,
        )
        for k, v in defaults.items():
            overrides.setdefault(k, v)
        super().__init__(
            modality_configs=modality_configs,
            statistics=statistics,
            embodiment_id_mapping=embodiment_id_mapping,
            max_state_dim=132,
            max_action_dim=132,
            max_action_horizon=40,
            **overrides,
        )

    def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        """Return the slim modality configs dict.

        Returns:
            ``{"unitree_g1": N1D7_G1_SHAFT_MODALITY_CONFIG}`` so that callers
            passing ``EmbodimentTag.UNITREE_G1`` (whose value is ``"unitree_g1"``)
            get the correct config.
        """
        return {
            "unitree_g1": N1D7_G1_SHAFT_MODALITY_CONFIG,
        }


# Re-export for clarity
__all__ = [
    "N1D7_G1_SHAFT_MODALITY_CONFIG",
    "N1D7_G1_SHAFT_EMBODIMENT_ID",
    "Gr00tN1d7G1ShaftProcessor",
]