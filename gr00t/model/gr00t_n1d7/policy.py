"""Fork-local Gr00tN1d7 policy adapter for the G1 IsaacLab shaft-insertion task.

Bridges the gap between:
  - The env's joint-delta, 7-key action space (per ``insertion_env.py:165-185``)
  - N1.7-3B's 132-dim-per-stream, 40-step action chunk

Inherits from ``Gr00tPolicy`` and overrides only what differs:
  - Loads the model by:
      1. Loading the N1.7-3B config (local path) and overriding
         ``model_name`` to the public ``Qwen/Qwen3-VL-2B-Instruct`` (same
         Qwen3-VL architecture; the gated ``nvidia/Cosmos-Reason2-2B`` is
         the same model with a different repo name).
      2. Constructing an empty ``Qwen3VLForConditionalGeneration`` from the
         public Qwen3-VL config.
      3. Loading the N1.7-3B backbone weights from local safetensors into
         the Qwen3-VL shell (via the vendored ``qwen3_backbone``'s
         ``__inject_backbone__`` hook).
      4. Constructing ``Gr00tN1d7`` with the modified config + inject hook.
      5. Loading the N1.7-3B action-head weights from local safetensors.
  - Constructs the slim processor directly (NOT via ``AutoProcessor``) with
    the fork's joint-delta G1 modality config and the public
    ``Qwen/Qwen3-VL-2B-Instruct`` as a processor substitute.
  - Pads env state from real widths (6/6/3/7/7/6/6 across 7 streams) to
    132 dims per stream (N1.7's ``max_state_dim``).
  - Slices N1.7's 132-dim action outputs back to env-expected dims (7/7/6/3/1/3).
  - Truncates N1.7's 40-step horizon to env's ``n_action_steps`` (default 20).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as _load_safetensors
from transformers import AutoConfig

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


# Env-side per-stream action dimensions (must match ``insertion_env.py:165-185``).
# Sum: 7+7+6+6+3+1+3 = 33 dims total.
#
# Hands are 6-dim here (NOT 7) to match the actual dataset produced by
# ``scripts/lerobot_conversion/convert_insertion_shaft_to_lerobot.sh`` +
# Stage 2 — the conversion hardcodes left_hand/right_hand as 6-dim
# (``modality.json``: start=14, end=20 and start=20, end=26). Eval must run
# with ``UNITREE_SIM_HAND_DIM=6`` to match (default is 7).
ENV_ACTION_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
    "waist": 3,
    "base_height_command": 1,
    "navigate_command": 3,
}

# Env-side per-stream state dimensions (must match
# ``insertion_env.py:300-306``).
# Sum: 6+6+3+7+7+6+6 = 41 dims total.
ENV_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 6,
    "right_hand": 6,
}

# N1.7 model dims (from config.json)
N1D7_MAX_STATE_DIM = 132
N1D7_MAX_ACTION_DIM = 132
N1D7_ACTION_HORIZON = 40

# Public substitute for the gated Cosmos-Reason2-2B (same Qwen3-VL arch).
DEFAULT_BACKBONE_SUBSTITUTE = "Qwen/Qwen3-VL-2B-Instruct"


def _build_qwen3vl_from_local_weights(
    substitute_path: str,
    n1d7_model_path: str,
    backbone_weight_map: dict[str, str],
):
    """Build ``Qwen3VLForConditionalGeneration`` from the public Qwen3-VL
    config, then load the N1.7-3B backbone weights into it.

    Args:
        substitute_path: Local path or HF id of the public Qwen3-VL config
            (used for architecture only — its weights are NOT loaded).
        n1d7_model_path: Local path to the N1.7-3B checkpoint
            (used to locate the safetensors files).
        backbone_weight_map: ``{tensor_name: shard_filename}`` mapping for
            N1.7-3B's ``backbone.*`` tensors.

    Returns:
        ``Qwen3VLForConditionalGeneration`` instance with N1.7-3B backbone
        weights loaded.
    """
    from transformers import (
        AutoConfig,
        Qwen3VLForConditionalGeneration,
    )

    # 1. Load public Qwen3-VL config (for architecture).
    config = AutoConfig.from_pretrained(substitute_path, trust_remote_code=True)

    # 2. Construct empty model on CPU (no meta — we want real tensors).
    model = Qwen3VLForConditionalGeneration(config)

    # 3. Load each shard, rename `backbone.model.X` → `X` and assign.
    #    N1.7-3B wraps the Qwen3VLForConditionalGeneration under
    #    `backbone.model` and (for non-lm_head tensors) has an extra
    #    `model.` between the wrapper and the inner Qwen3VLModel. After
    #    stripping `backbone.model.`, the remaining key (e.g.
    #    `lm_head.weight` or `model.language_model.X`) matches the
    #    current transformers 4.57.x Qwen3VLForConditionalGeneration state
    #    dict format.
    flat = {}
    loaded_shards = set()
    for tensor_name, shard_filename in backbone_weight_map.items():
        # Strip "backbone.model." prefix.
        # tensor_name like "backbone.model.lm_head.weight"
        #             or  "backbone.model.model.language_model.X"
        if tensor_name.startswith("backbone.model."):
            new_key = tensor_name[len("backbone.model."):]
        elif tensor_name.startswith("backbone."):
            new_key = tensor_name[len("backbone."):]
        else:
            new_key = tensor_name
        if shard_filename not in loaded_shards:
            shard_state = _load_safetensors(
                f"{n1d7_model_path}/{shard_filename}"
            )
            for k, v in shard_state.items():
                if k.startswith("backbone.model."):
                    flat[k[len("backbone.model."):]] = v
                elif k.startswith("backbone."):
                    flat[k[len("backbone."):]] = v
            loaded_shards.add(shard_filename)
            del shard_state
    missing, unexpected = model.load_state_dict(flat, strict=False)
    if missing:
        print(f"[_build_qwen3vl_from_local_weights] backbone missing keys "
              f"(first 5): {missing[:5]} (total {len(missing)})")
    if unexpected:
        print(f"[_build_qwen3vl_from_local_weights] backbone unexpected keys "
              f"(first 5): {unexpected[:5]} (total {len(unexpected)})")
    return model


class Gr00tN1d7G1ShaftPolicy(Gr00tPolicy):
    """Gr00tPolicy subclass for N1.7-3B on the G1 IsaacLab shaft-insertion task."""

    def __init__(
        self,
        model_path: str,
        embodiment_tag: EmbodimentTag = EmbodimentTag.UNITREE_G1,
        *,
        device: str | int = "cuda:0",
        backbone_substitute: str = DEFAULT_BACKBONE_SUBSTITUTE,
        n_action_steps: int = 20,
        strict: bool = True,
    ):
        # Initialize parent (BasePolicy) without calling Gr00tPolicy.__init__
        # — we override the model + processor loading entirely.
        from gr00t.policy.policy import BasePolicy

        BasePolicy.__init__(self, strict=strict)

        self.embodiment_tag = embodiment_tag
        self.device = device
        self.n_action_steps = min(n_action_steps, N1D7_ACTION_HORIZON)

        # 1. Load the model. Bypass gated Cosmos-Reason2-2B by:
        #    a) Loading the N1.7-3B config.json from local path
        #    b) Overriding config.model_name to the public Qwen3-VL substitute
        #    c) Pre-building a Qwen3VLForConditionalGeneration from the
        #       Qwen3-VL substitute's config + local N1.7-3B backbone weights
        #    d) Passing the pre-built model via the __inject_backbone__ hook
        #    e) Letting Gr00tN1d7 instantiate everything else (action head,
        #       collator) with the substitute model_name
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.model_name = backbone_substitute

        # Read backbone weight map from local safetensors index.
        index_path = Path(model_path) / "model.safetensors.index.json"
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        backbone_weight_map = {
            k: v for k, v in weight_map.items() if k.startswith("backbone.")
        }
        # Pre-build the backbone model with N1.7-3B weights loaded.
        _inject = lambda name, ek, tlk: _build_qwen3vl_from_local_weights(
            substitute_path=backbone_substitute,
            n1d7_model_path=model_path,
            backbone_weight_map=backbone_weight_map,
        )

        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

        # IMPORTANT: do NOT use meta-device here. The vendored qwen3_backbone
        # calls Qwen3VLForConditionalGeneration.from_pretrained UNLESS we set
        # __inject_backbone__ in transformers_loading_kwargs (which we do).
        model = Gr00tN1d7(
            config,
            transformers_loading_kwargs={
                "trust_remote_code": True,
                "__inject_backbone__": _inject,
            },
        )

        # Load action_head.* weights from local safetensors into the model.
        flat_action_head = {}
        for tensor_name, shard_filename in weight_map.items():
            if tensor_name.startswith("action_head."):
                new_key = tensor_name[len("action_head."):]
                if shard_filename not in flat_action_head:
                    shard_state = _load_safetensors(
                        f"{model_path}/{shard_filename}"
                    )
                    for k, v in shard_state.items():
                        if k.startswith("action_head."):
                            flat_action_head[k[len("action_head."):]] = v
                    del shard_state
        ah_missing, ah_unexpected = model.action_head.load_state_dict(
            flat_action_head, strict=False
        )
        if ah_missing:
            print(f"[Gr00tN1d7G1ShaftPolicy] action_head missing keys "
                  f"(first 5): {ah_missing[:5]} (total {len(ah_missing)})")
        if ah_unexpected:
            print(f"[Gr00tN1d7G1ShaftPolicy] action_head unexpected keys "
                  f"(first 5): {ah_unexpected[:5]} (total {len(ah_unexpected)})")

        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # 2. Build the slim processor directly (NOT AutoProcessor).
        from gr00t.model.gr00t_n1d7.processing import Gr00tN1d7G1ShaftProcessor

        statistics_path = Path(model_path) / "statistics.json"
        statistics = None
        if statistics_path.exists():
            with open(statistics_path) as f:
                statistics = json.load(f)

        self.processor = Gr00tN1d7G1ShaftProcessor(
            statistics=statistics,
            model_name=backbone_substitute,
        )
        self.processor.eval()

        # 3. Cache modality config + collator (matches Gr00tPolicy attributes).
        self.modality_configs = self.processor.get_modality_configs()[
            self.embodiment_tag.value
        ]
        self.collate_fn = self.processor.collator
        self.system1_type = getattr(self.model.config, "system1_type", "dit")
        # N1.7 has no Mamba system1; always dit.

        # Cache language key for _get_action.
        language_keys = self.modality_configs["language"].modality_keys
        assert len(language_keys) == 1, "Only one language key is supported"
        self.language_key = language_keys[0]

        # Per-key state dimensions (used to slice the model's 132-dim output
        # back to env-expected widths).
        self._env_state_real_dims = {k: v for k, v in ENV_STATE_DIMS.items()}
        self._env_action_real_dims = {k: v for k, v in ENV_ACTION_DIMS.items()}

    def _to_vla_step_data(self, observation: dict):
        """Convert one unbatched observation into a VLAStepData for the
        N1.7 processor.

        Overrides ``Gr00tPolicy._to_vla_step_data`` to attach ``masks=None``,
        which N1.7's VLAStepData requires but N1.6's class lacks.
        """
        from gr00t.data.types import VLAStepData

        vla = VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )
        # Attach masks=None for N1.7 compatibility (set as attribute since
        # the fork's VLAStepData doesn't have it as a dataclass field).
        vla.masks = None
        return vla

    def _pad_state_to_n1d7(self, observation: dict) -> dict:
        """Identity passthrough. The slim processor's ``process_observation``
        pads each state stream to ``max_state_dim=132`` internally, AFTER
        per-key normalization. Pre-padding here would cause a dim mismatch
        between the raw state and the per-key statistics.
        """
        return observation

    def _extract_trained_state_keys(
        self, observation: dict
    ) -> dict:
        """Extract only the state keys the model was trained on.

        The env provides 7 state keys (left_leg, right_leg, waist, left_arm,
        right_arm, left_hand, right_hand). The finetuned model was trained
        on only 4 keys (left_arm, right_arm, left_hand, right_hand). This
        method filters the observation to only those 4 keys (zero-filling
        any that are missing). The slim processor pads each stream to
        max_state_dim=132 internally.
        """
        out = dict(observation)
        if "state" not in observation:
            return out
        trained_keys = self.modality_configs["state"].modality_keys
        env_state = observation["state"]
        new_state = {}
        for k in trained_keys:
            if k in env_state:
                new_state[k] = env_state[k]
            else:
                # Zero-fill if env doesn't have this key
                env_dim = ENV_STATE_DIMS.get(k, 1)
                shape = (1, 1, env_dim)  # (B, T, D) placeholder
                new_state[k] = np.zeros(shape, dtype=np.float32)
        out["state"] = new_state
        return out

    def _fill_untrained_action_keys(
        self, actions: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Add the 3 untrained action keys (waist, base_height_command,
        navigate_command) with zero values.

        The model outputs 4 action keys (left_arm, right_arm, left_hand,
        right_hand). The env expects 7 keys. For the 3 missing keys,
        output zeros — this keeps the env's lower-body controller
        (WBC sim) doing its default behavior (stand in place).
        """
        out = dict(actions)
        for key in ("waist", "base_height_command", "navigate_command"):
            if key in out:
                continue
            env_dim = ENV_ACTION_DIMS.get(key, 1)
            # Match the time/batch shape of the model's outputs
            sample = next(iter(actions.values())) if actions else None
            if sample is None:
                shape = (1, self.n_action_steps, env_dim)
            else:
                shape = sample.shape[:-1] + (env_dim,)
            out[key] = np.zeros(shape, dtype=np.float32)
        return out

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute actions from Gr00t-sim-format observations."""
        # Step 0a: Extract only the trained state keys from env's 7-key state.
        filtered_observation = self._extract_trained_state_keys(observation)

        # Step 0b: Identity pad passthrough.
        padded_observation = self._pad_state_to_n1d7(filtered_observation)

        # Step 1: Split batched observation into individual observations.
        unbatched_observations = self._unbatch_observation(padded_observation)
        processed_inputs = []
        states = []

        # Step 2: Process each observation through the slim processor.
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [
                {"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}
            ]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate into model input batch.
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference.
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions back to physical units.
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Step 6: Slice N1.7's 132-dim outputs back to env-expected widths.
        # We return the FULL 40-step horizon (matching the model's
        # delta_indices). The downstream MultiStepWrapper (in
        # rollout_policy.py) handles truncation to ``n_action_steps``.
        casted_action = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in unnormalized_action.items()
        }
        truncated = {}
        for key, arr in casted_action.items():
            env_dim = self._env_action_real_dims.get(key, arr.shape[-1])
            if arr.shape[-1] >= env_dim:
                sliced = arr[..., :env_dim]
            else:
                pad_shape = arr.shape[:-1] + (env_dim - arr.shape[-1],)
                sliced = np.concatenate(
                    [arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=-1
                )
            truncated[key] = sliced

        # Step 7: Add the 3 untrained action keys (waist, base_height_command,
        # navigate_command) as zeros. The env expects 7 keys but the model
        # only learned 4.
        truncated = self._fill_untrained_action_keys(truncated)
        return truncated, {}


__all__ = [
    "Gr00tN1d7G1ShaftPolicy",
    "ENV_ACTION_DIMS",
    "ENV_STATE_DIMS",
    "N1D7_MAX_STATE_DIM",
    "N1D7_MAX_ACTION_DIM",
    "N1D7_ACTION_HORIZON",
    "DEFAULT_BACKBONE_SUBSTITUTE",
]