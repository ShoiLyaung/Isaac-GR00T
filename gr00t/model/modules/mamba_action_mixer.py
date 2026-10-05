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
            "action_backbone=conditional_mamba2 requires the official mamba_ssm Mamba2"
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


def convert_native_model(model, *, mamba2_config=None) -> dict[str, Any]:
    """Replace the entire DiT after strict native load; audit all outside tensors."""
    from gr00t.model.modules.conditional_mamba import ConditionalMamba2

    if model.config.action_backbone != "dit":
        raise ValueError("Conversion accepts only a native DiT model")
    before = {
        name: (tensor, tensor._version, tensor.numel())
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
    }
    config = dict(model.config.action_mamba2_cfg if mamba2_config is None else mamba2_config)
    old = model.action_head.model
    parameter = next(old.parameters())
    candidate = ConditionalMamba2(
        model.config.diffusion_model_cfg, model.config.backbone_embedding_dim, config
    ).to(device=parameter.device, dtype=parameter.dtype)
    candidate.requires_grad_(model.action_head.tune_diffusion_model)
    candidate.train(old.training)
    model.action_head.model = candidate
    prefix = "action_head.model."
    after = dict(list(model.named_parameters()) + list(model.named_buffers()))
    preserved = [name for name in before if not name.startswith(prefix)]
    if {name for name in after if not name.startswith(prefix)} != set(preserved):
        raise RuntimeError("Conversion changed keys outside the declared backbone")
    if any(
        after[name] is not before[name][0] or after[name]._version != before[name][1]
        for name in preserved
    ):
        raise RuntimeError("Conversion changed a tensor outside the declared backbone")
    model.config.action_backbone = "conditional_mamba2"
    model.config.action_mamba2_cfg = config
    model.action_head.config.action_backbone = "conditional_mamba2"
    model.action_head.config.action_mamba2_cfg = config
    return {
        "replacement_prefix": prefix,
        "unchanged_tensor_count": len(preserved),
        "unchanged_tensor_identity_and_version": True,
        "removed_parameter_count": sum(
            n for name, (_, _, n) in before.items() if name.startswith(prefix)
        ),
        "replacement_parameter_count": sum(
            t.numel() for name, t in after.items() if name.startswith(prefix)
        ),
        "bidirectional_shared_weights": True,
        "persistent_ssm_state": False,
        "action_backbone": "conditional_mamba2",
        "mamba2_config": config,
        "time_and_output_projection_pretrained_transfer": False,
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
    if model.config.action_backbone != "conditional_mamba2":
        raise ValueError("Candidate checkpoint does not declare conditional Mamba2")
    return model, loading_info
