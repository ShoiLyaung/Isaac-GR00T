"""Attention-free conditional Mamba2 action backbone for native flow matching.

Valid VLM tokens retain their original order and precede native state/action
embeddings. Padding is removed before the SSM, not merely zeroed. Every forward
is a fresh bidirectional scan with shared directional weights and no saved state.
"""

from types import SimpleNamespace

from diffusers.models.attention import FeedForward
import torch
from torch import nn
from torch.nn import functional as F

from gr00t.model.modules.dit import AdaLayerNorm, TimestepEncoder
from gr00t.model.modules.mamba_action_mixer import BidirectionalMamba2SelfMixer


class ConditionalMambaBlock(nn.Module):
    def __init__(self, dim, model_config, mamba_config):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.mixer = BidirectionalMamba2SelfMixer(dim, mamba_config)
        self.final_dropout = nn.Dropout(model_config.get("dropout", 0.0))
        self.norm3 = nn.LayerNorm(
            dim,
            eps=model_config.get("norm_eps", 1e-5),
            elementwise_affine=model_config.get("norm_elementwise_affine", False),
        )
        self.ff = FeedForward(
            dim,
            dropout=model_config.get("dropout", 0.0),
            activation_fn=model_config.get("activation_fn", "gelu-approximate"),
            final_dropout=model_config.get("final_dropout", True),
        )

    def forward(self, hidden, temb):
        hidden = hidden + self.final_dropout(self.mixer(self.norm1(hidden, temb)))
        return hidden + self.ff(self.norm3(hidden))


class ConditionalMamba2(nn.Module):
    """Drop-in native DiT forward interface, without any attention module."""

    def __init__(self, diffusion_config, cross_attention_dim, mamba_config):
        super().__init__()
        model_config = dict(diffusion_config)
        self.config = SimpleNamespace(**model_config)
        self.inner_dim = model_config["num_attention_heads"] * model_config["attention_head_dim"]
        if model_config.get("norm_type", "ada_norm") != "ada_norm":
            raise ValueError("Conditional Mamba requires the native AdaLN time contract")
        if model_config.get("positional_embeddings") is not None:
            raise ValueError("Native checkpoint must use external action positions")
        self.condition_projection = nn.Linear(cross_attention_dim, self.inner_dim)
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.blocks = nn.ModuleList(
            [
                ConditionalMambaBlock(self.inner_dim, model_config, mamba_config)
                for _ in range(model_config["num_layers"])
            ]
        )
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, model_config["output_dim"])

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        del image_mask  # Native ordering is retained; no image/text attention alternation exists.
        mask = (
            backbone_attention_mask
            if backbone_attention_mask is not None
            else encoder_attention_mask
        )
        if mask is None:
            raise ValueError("Conditional Mamba requires a valid VLM token mask")
        if mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("VLM token mask shape mismatch")
        if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("VLM token mask must be boolean or binary")
        mask = mask.bool()
        if hidden_states.ndim != 3 or hidden_states.shape[0] != mask.shape[0]:
            raise ValueError("Action sequence batch shape mismatch")
        counts = mask.sum(1)
        if torch.any(counts == 0):
            raise ValueError("Every sample needs at least one conditioning token")
        condition = self.condition_projection(encoder_hidden_states)
        temb = self.timestep_encoder(timestep)
        output_rows = [None] * hidden_states.shape[0]
        # Per-length batches avoid letting any padding tokens enter a state scan.
        for count in sorted(set(counts.tolist())):
            indices = torch.where(counts == count)[0]
            prefix = torch.stack([condition[i][mask[i]] for i in indices])
            hidden = torch.cat((prefix, hidden_states[indices]), dim=1).contiguous()
            time_group = temb[indices]
            for block in self.blocks:
                hidden = block(hidden, time_group)
            hidden = hidden[:, -hidden_states.shape[1] :]
            shift, scale = self.proj_out_1(F.silu(time_group)).chunk(2, dim=1)
            hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
            result = self.proj_out_2(hidden)
            for i, row in zip(indices.tolist(), result, strict=True):
                output_rows[i] = row
        output = torch.stack(output_rows)
        # The native caller discards intermediate states; preserve the tuple ABI.
        return (output, [output]) if return_all_hidden_states else output
