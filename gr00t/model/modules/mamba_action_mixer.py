"""Stateless bidirectional Mamba2 replacement for action self-attention only.

The official Mamba2 CUDA implementation is required. There is no CPU, simple,
or reference fallback. Both directions share parameters; no inference cache is
created or carried across denoising steps or policy calls.
"""

from collections.abc import Mapping
import os
from typing import Any

from torch import nn


def _mamba2_class():
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
    except ImportError as exc:
        raise RuntimeError(
            "action_self_mixer=mamba2 requires the official mamba_ssm Mamba2"
        ) from exc
    return Mamba2


class BidirectionalMamba2SelfMixer(nn.Module):
    """Attention-compatible self mixer with a noncausal action-chunk receptive field."""

    def __init__(self, dim: int, config: Mapping[str, Any], *, device=None, dtype=None):
        super().__init__()
        self.mamba2 = _mamba2_class()(d_model=dim, **dict(config), device=device, dtype=dtype)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("Mamba2 self mixer accepts only the native unmasked action sequence")
        forward = self.mamba2(hidden_states.contiguous())
        reverse = self.mamba2(hidden_states.flip(1).contiguous()).flip(1)
        return (forward + reverse) * 0.5


def replace_self_mixers(diffusion_model, config: Mapping[str, Any]) -> list[int]:
    """Replace odd self-attention modules; leave every other tensor untouched."""
    if not diffusion_model.config.interleave_self_attention:
        raise ValueError("Mamba2 replacement requires native interleaved self-attention")
    # Validate dependency before changing any module.
    _mamba2_class()
    replacements = []
    for index, block in enumerate(diffusion_model.transformer_blocks):
        if index % 2 == 0:
            continue
        if block.cross_attention_dim is not None:
            raise ValueError(f"Block {index} is not native self-attention")
        if isinstance(block.attn1, BidirectionalMamba2SelfMixer):
            raise ValueError("Self mixer has already been replaced")
        parameter = next(block.attn1.parameters())
        trainability = {p.requires_grad for p in block.attn1.parameters()}
        if len(trainability) != 1:
            raise ValueError("Cannot preserve mixed trainability within a self-attention module")
        replacement = BidirectionalMamba2SelfMixer(
            block.dim, config, device=parameter.device, dtype=parameter.dtype
        )
        replacement.requires_grad_(trainability.pop())
        replacement.train(block.attn1.training)
        replacements.append((index, replacement))
    if not replacements:
        raise ValueError("No action self-attention blocks found")
    # Commit only after every candidate module was constructed successfully.
    for index, replacement in replacements:
        diffusion_model.transformer_blocks[index].attn1 = replacement
    return [index for index, _ in replacements]


def convert_native_model(model, *, mamba2_config=None) -> dict[str, Any]:
    """Convert a strictly loaded native model and audit unchanged tensor identity.

    Keeping the same object AND version counter verifies no untouched tensor was
    replaced or modified by conversion. State dict values themselves are not
    cloned, avoiding an extra full VLM copy in host/device memory.
    """
    if model.config.action_self_mixer != "dit":
        raise ValueError("Conversion accepts only a native DiT model")
    before = {
        name: (tensor, tensor._version, tensor.numel())
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
    }
    config = dict(model.config.action_mamba2_cfg if mamba2_config is None else mamba2_config)
    indices = replace_self_mixers(model.action_head.model, config)
    prefixes = tuple(f"action_head.model.transformer_blocks.{i}.attn1." for i in indices)
    after = dict(list(model.named_parameters()) + list(model.named_buffers()))
    preserved = [name for name in before if not name.startswith(prefixes)]
    if set(after) - {name for name in after if name.startswith(prefixes)} != set(preserved):
        raise RuntimeError("Conversion changed keys outside the declared self mixers")
    if any(
        after[name] is not before[name][0] or after[name]._version != before[name][1]
        for name in preserved
    ):
        raise RuntimeError("Conversion changed a tensor outside the declared self mixers")
    model.config.action_self_mixer = "mamba2"
    model.config.action_mamba2_cfg = config
    model.action_head.config.action_self_mixer = "mamba2"
    model.action_head.config.action_mamba2_cfg = config
    return {
        "replacement_indices": indices,
        "replacement_prefixes": list(prefixes),
        "unchanged_tensor_count": len(preserved),
        "unchanged_tensor_identity_and_version": True,
        "removed_parameter_count": sum(
            n for name, (_, _, n) in before.items() if name.startswith(prefixes)
        ),
        "replacement_parameter_count": sum(
            t.numel() for name, t in after.items() if name.startswith(prefixes)
        ),
        "bidirectional_shared_weights": True,
        "persistent_ssm_state": False,
        "action_self_mixer": "mamba2",
        "mamba2_config": config,
    }


def load_native_then_convert(checkpoint, *, mamba2_config=None, **loading_kwargs):
    """Load native weights strictly first; then initialize only new Mamba tensors."""
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise RuntimeError("Strict checkpoint loading forbids GROOT_SKIP_HF_MODEL_WEIGHTS=1")
    if "output_loading_info" in loading_kwargs or "ignore_mismatched_sizes" in loading_kwargs:
        raise ValueError("Loading-info and mismatch policy are fixed by this factory")
    model, loading_info = Gr00tN1d7.from_pretrained(
        checkpoint, output_loading_info=True, **loading_kwargs
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            raise RuntimeError(f"Native checkpoint failed strict load: {key}={loading_info[key]}")
    report = convert_native_model(model, mamba2_config=mamba2_config)
    report["native_checkpoint"] = str(checkpoint)
    report["native_loading_info"] = loading_info
    return model, report


def load_candidate_strict(checkpoint, **loading_kwargs):
    """Reload a saved candidate with the same strict policy as the native source."""
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise RuntimeError("Strict checkpoint loading forbids GROOT_SKIP_HF_MODEL_WEIGHTS=1")
    if "output_loading_info" in loading_kwargs or "ignore_mismatched_sizes" in loading_kwargs:
        raise ValueError("Loading-info and mismatch policy are fixed by this factory")
    model, loading_info = Gr00tN1d7.from_pretrained(
        checkpoint, output_loading_info=True, **loading_kwargs
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            raise RuntimeError(
                f"Candidate checkpoint failed strict load: {key}={loading_info[key]}"
            )
    if model.config.action_self_mixer != "mamba2":
        raise ValueError("Candidate checkpoint does not declare Mamba2 self mixers")
    return model, loading_info
