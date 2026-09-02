"""GR00T-only codec dynamics and zero-gated action-side conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CodecModuleSizing:
    d_codec: int
    action_width: int
    conditioner_heads: int
    parameters: int
    encode_flops_batch1: int
    condition_flops_per_call_batch1: int
    training_forward_flops_batch1: int
    policy_query_four_euler_flops_batch1: int
    status: str = "IMPLEMENTATION_SIZED_NOT_RUNTIME_BENCHMARKED"


class CodecDynamicsCore(nn.Module):
    """Encode T8 codec evidence as four spatial tokens per frame."""

    def __init__(
        self, d_codec: int = 64, modalities: tuple[str, ...] = ("mv", "rpe", "i_frame", "metadata")
    ):
        super().__init__()
        self.d_codec = d_codec
        self.modalities = frozenset(modalities)
        self.mv_stem = nn.Linear(11, d_codec)
        self.rpe_stem = nn.Linear(4, d_codec)
        self.i_frame_stem = nn.Linear(3, d_codec)
        self.metadata_stem = nn.Linear(21, d_codec)
        self.norm = nn.LayerNorm(d_codec)
        self.mlp = nn.Sequential(
            nn.Linear(d_codec, 2 * d_codec),
            nn.SiLU(),
            nn.Linear(2 * d_codec, d_codec),
        )

    @staticmethod
    def _aggregate_mv(batch: Mapping[str, Any], dtype: torch.dtype) -> torch.Tensor:
        fields = batch["motion_vector_fields"]
        mask = (
            batch["mv_mask"]
            & batch["frame_validity"]["frame_valid"][..., None].to(torch.bool)
            & batch["frame_validity"]["mv_valid"][..., None].to(torch.bool)
        )
        batch_size, history, count = mask.shape
        device = mask.device
        if count == 0:
            return torch.zeros((batch_size, history, 4, 11), device=device, dtype=dtype)
        scale = fields["motion_scale"].to(torch.float32).clamp_min(1.0)
        features = torch.stack(
            (
                fields["source"].to(torch.float32) / 4.0,
                fields["block_width"].to(torch.float32) / 64.0,
                fields["block_height"].to(torch.float32) / 64.0,
                fields["source_x"].to(torch.float32) / 640.0,
                fields["source_y"].to(torch.float32) / 480.0,
                fields["destination_x"].to(torch.float32) / 640.0,
                fields["destination_y"].to(torch.float32) / 480.0,
                fields["flags"].to(torch.float32) / 16.0,
                fields["motion_x"].to(torch.float32) / scale,
                fields["motion_y"].to(torch.float32) / scale,
                torch.log2(scale) / 16.0,
            ),
            dim=-1,
        )
        destination_x = fields["destination_x"].to(torch.float32)
        destination_y = fields["destination_y"].to(torch.float32)
        cells = (destination_y >= 240).to(torch.long) * 2 + (destination_x >= 320).to(torch.long)
        one_hot = F.one_hot(cells, num_classes=4).to(features.dtype)
        weights = one_hot * mask[..., None].to(features.dtype)
        sums = torch.einsum("btmc,btmf->btcf", weights, features)
        counts = weights.sum(dim=2).unsqueeze(-1).clamp_min(1.0)
        return (sums / counts).to(dtype)

    def forward(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        episode_bundle = "mv_cell_features" in batch
        if episode_bundle:
            frame_validity = batch["frame_validity"]
            if frame_validity.ndim != 3 or frame_validity.shape[-1] != 7:
                raise ValueError("Episode-bundle frame_validity must be [B,8,7]")
            frame_valid = frame_validity[..., 0].to(torch.bool)
            sample_valid = torch.ones(
                (frame_valid.shape[0],), device=frame_valid.device, dtype=torch.bool
            )
        else:
            frame_valid = batch["frame_validity"]["frame_valid"].to(torch.bool)
            sample_valid = batch["sample_valid"].to(torch.bool)
        token_mask = (frame_valid & sample_valid[:, None]).repeat_interleave(4, dim=1)
        reference = batch["rpe"]
        combined = torch.zeros(
            (reference.shape[0], 8, 4, self.d_codec),
            device=reference.device,
            dtype=self.mv_stem.weight.dtype,
        )
        if episode_bundle:
            mv_features = batch["mv_cell_features"].to(self.mv_stem.weight.dtype)
            if mv_features.shape[1:] != (8, 2, 2, 11):
                raise ValueError("Episode-bundle MV cells must be [B,8,2,2,11]")
            mv_mask = batch["mv_cell_mask"].to(torch.bool) & frame_valid[..., None, None]
            mv_proposal = self.mv_stem(
                mv_features.reshape(mv_features.shape[0], 8, 4, 11)
            ) * mv_mask.reshape(mv_mask.shape[0], 8, 4, 1)
        else:
            mv_proposal = self.mv_stem(self._aggregate_mv(batch, self.mv_stem.weight.dtype))
        rpe = reference.to(self.rpe_stem.weight.dtype).reshape(reference.shape[0], 8, 4, 4)
        rpe_mask = batch["rpe_mask"].reshape(reference.shape[0], 8, 4, 1)
        rpe_proposal = self.rpe_stem(rpe) * rpe_mask
        if episode_bundle:
            anchor_bank = batch["anchor_appearance"].to(self.i_frame_stem.weight.dtype)
            if anchor_bank.shape[1:] != (2, 3, 256, 256):
                raise ValueError("Episode-bundle anchor bank must be [B,2,3,256,256]")
            slots = batch["frame_anchor_slot"].to(torch.long)
            safe_slots = slots.clamp_min(0)
            batch_index = torch.arange(anchor_bank.shape[0], device=anchor_bank.device)[:, None]
            selected = anchor_bank[batch_index, safe_slots]
            appearance = F.adaptive_avg_pool2d(selected.flatten(0, 1), (2, 2))
            appearance = appearance.reshape(reference.shape[0], 8, 3, 2, 2)
            appearance = appearance.permute(0, 1, 3, 4, 2).reshape(reference.shape[0], 8, 4, 3)
            selected_anchor_valid = batch["anchor_mask"].to(torch.bool).gather(1, safe_slots)
            appearance_frame_mask = (
                (slots >= 0) & selected_anchor_valid & frame_valid & sample_valid[:, None]
            )
            appearance_mask = appearance_frame_mask[..., None, None]
        else:
            appearance = F.adaptive_avg_pool2d(
                batch["i_frame_appearance"].to(self.i_frame_stem.weight.dtype),
                (2, 2),
            )
            appearance = appearance.permute(0, 2, 3, 1)[:, None].expand(-1, 8, -1, -1, -1)
            appearance = appearance.reshape(reference.shape[0], 8, 4, 3)
            appearance_mask = (batch["i_frame_mask"] & sample_valid)[:, None, None, None]
        appearance_proposal = self.i_frame_stem(appearance) * appearance_mask
        if episode_bundle:
            metadata = batch["metadata_features"].to(torch.float32)
            if metadata.shape[1:] != (8, 21):
                raise ValueError("Episode-bundle metadata must be [B,8,21]")
        else:
            temporal = batch["time_and_reference"]
            frame_index = temporal["frame_index"].to(torch.float32)
            observation_row = batch["observation_row"].to(torch.float32)[:, None]
            pts = temporal["pts"].to(torch.float32)
            dts = temporal["dts"].to(torch.float32)
            current_pts = pts[:, -1:]
            current_dts = dts[:, -1:]
            reference_index = temporal["reference_index"].to(torch.float32)
            relative_reference = torch.where(
                reference_index >= 0,
                (frame_index - reference_index) / 30.0,
                torch.zeros_like(reference_index),
            )
            legacy_type = batch["frame_type"].to(torch.long)
            type_features = torch.stack(
                (
                    legacy_type == 1,
                    torch.zeros_like(legacy_type, dtype=torch.bool),
                    legacy_type == 2,
                ),
                dim=-1,
            ).to(torch.float32)
            relative = torch.stack(
                (
                    (frame_index - observation_row) / 7.0,
                    (pts - current_pts) / 30_000.0,
                    (dts - current_dts) / 30_000.0,
                    temporal["gop_position"].to(torch.float32) / 29.0,
                    relative_reference,
                ),
                dim=-1,
            )
            validity = torch.stack(
                [
                    batch["frame_validity"][name].to(torch.float32)
                    for name in (
                        "frame_valid",
                        "packet_valid",
                        "reference_valid",
                        "dependency_valid",
                        "alignment_valid",
                        "mv_valid",
                        "rpe_valid",
                    )
                ],
                dim=-1,
            )
            camera_mask = batch["camera_motion_mask"].to(torch.bool)
            camera = batch["camera_motion_summary"].to(torch.float32)
            camera = camera * camera_mask[..., None].to(camera.dtype)
            anchor_valid = batch["i_frame_mask"].to(torch.bool)[:, None] & frame_valid
            metadata = torch.cat(
                (
                    type_features,
                    relative,
                    validity,
                    camera_mask.to(torch.float32).unsqueeze(-1),
                    anchor_valid.to(torch.float32).unsqueeze(-1),
                    camera,
                ),
                dim=-1,
            )
            metadata = metadata * frame_valid[..., None].to(metadata.dtype)
        metadata = metadata[:, :, None, :].expand(-1, -1, 4, -1)
        metadata_proposal = self.metadata_stem(metadata.to(self.metadata_stem.weight.dtype))
        proposals = {
            "mv": mv_proposal,
            "rpe": rpe_proposal,
            "i_frame": appearance_proposal,
            "metadata": metadata_proposal,
        }
        if not all(torch.isfinite(value).all() for value in proposals.values()):
            raise FloatingPointError("Codec modality stem produced NaN or Inf")
        for name, proposal in proposals.items():
            combined = combined + proposal * int(name in self.modalities)
        tokens = self.mlp(self.norm(combined)).reshape(reference.shape[0], 32, self.d_codec)
        tokens = tokens * token_mask[..., None].to(tokens.dtype)
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("CodecDynamicsCore produced NaN or Inf")
        return tokens, token_mask


class Gr00tCodecProjector(nn.Module):
    def __init__(self, d_codec: int, action_width: int):
        super().__init__()
        self.projection = nn.Linear(d_codec, action_width)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected = self.projection(tokens)
        return projected * mask[..., None].to(projected.dtype)


class ZeroGatedActionConditioner(nn.Module):
    """Deterministic cross-attention residual with a direct exact-zero gate."""

    def __init__(self, action_width: int, num_heads: int, gate_init: float = 0.0):
        super().__init__()
        if gate_init != 0.0:
            raise ValueError("The parity contract requires gate_init == 0.0")
        if action_width % num_heads:
            raise ValueError("action_width must be divisible by conditioner heads")
        self.query_norm = nn.LayerNorm(action_width)
        self.context_norm = nn.LayerNorm(action_width)
        self.attention = nn.MultiheadAttention(
            action_width, num_heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(action_width)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        action_features: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.shape[:2] != context_mask.shape:
            raise ValueError("Codec context and mask shapes disagree")
        any_valid = context_mask.any(dim=1)
        safe_context = context.clone()
        safe_mask = context_mask.clone()
        if (~any_valid).any():
            safe_context[~any_valid, 0] = 0
            safe_mask[~any_valid, 0] = True
        proposal, _ = self.attention(
            self.query_norm(action_features),
            self.context_norm(safe_context),
            self.context_norm(safe_context),
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        proposal = self.output_norm(proposal)
        if not torch.isfinite(proposal).all():
            raise FloatingPointError("Codec conditioner proposal contains NaN or Inf")
        proposal = proposal * any_valid[:, None, None].to(proposal.dtype)
        conditioned = action_features + self.gate.to(action_features.dtype) * proposal
        return torch.where(any_valid[:, None, None], conditioned, action_features)


class Gr00tCodecConditioningStack(nn.Module):
    """GR00T-specific adapter from the common batch to H40 action features."""

    def __init__(
        self,
        *,
        action_width: int,
        d_codec: int = 64,
        conditioner_heads: int = 8,
        gate_init: float = 0.0,
        modalities: tuple[str, ...] = ("mv", "rpe", "i_frame", "metadata"),
    ):
        super().__init__()
        self.core = CodecDynamicsCore(d_codec=d_codec, modalities=modalities)
        self.projector = Gr00tCodecProjector(d_codec, action_width)
        self.conditioner = ZeroGatedActionConditioner(action_width, conditioner_heads, gate_init)

    def encode(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.core(batch)
        projected = self.projector(tokens, mask)
        return projected, mask

    def condition(
        self,
        action_features: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = context.to(device=action_features.device, dtype=action_features.dtype)
        context_mask = context_mask.to(device=action_features.device)
        return self.conditioner(action_features, context, context_mask)


def size_codec_modules(
    *, action_width: int = 1536, d_codec: int = 64, conditioner_heads: int = 8
) -> CodecModuleSizing:
    stack = Gr00tCodecConditioningStack(
        action_width=action_width,
        d_codec=d_codec,
        conditioner_heads=conditioner_heads,
    )
    parameters = sum(parameter.numel() for parameter in stack.parameters())
    encode_flops = (
        8 * 4 * (11 + 4 + 3 + 21) * d_codec * 2
        + 8 * 4 * (d_codec * 2 * d_codec + 2 * d_codec * d_codec) * 2
        + 32 * d_codec * action_width * 2
    )
    condition_flops = (
        (40 + 2 * 32) * action_width * action_width * 2
        + 2 * 40 * 32 * action_width
        + 40 * action_width * action_width * 2
    )
    return CodecModuleSizing(
        d_codec=d_codec,
        action_width=action_width,
        conditioner_heads=conditioner_heads,
        parameters=parameters,
        encode_flops_batch1=encode_flops,
        condition_flops_per_call_batch1=condition_flops,
        training_forward_flops_batch1=encode_flops + condition_flops,
        policy_query_four_euler_flops_batch1=encode_flops + 4 * condition_flops,
    )
