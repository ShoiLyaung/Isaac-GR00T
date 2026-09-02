"""Codec dynamics and zero-gated action-side conditioning for GR00T N1.7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


_HISTORY = 8
_SPATIAL_TOKENS = 4
_CONTEXT_TOKENS = _HISTORY * _SPATIAL_TOKENS
_MV_FEATURES = 11
_METADATA_FEATURES = 18
_EPISODE_METADATA_FEATURES = 21


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
    """Encode a causal T8 evidence window as four spatial tokens per frame."""

    def __init__(
        self,
        d_codec: int = 64,
        modalities: tuple[str, ...] = ("mv", "rpe", "i_frame", "metadata"),
    ):
        super().__init__()
        allowed_modalities = frozenset(("mv", "rpe", "i_frame", "metadata"))
        unknown_modalities = set(modalities) - allowed_modalities
        if unknown_modalities:
            raise ValueError(f"Unknown codec modalities: {sorted(unknown_modalities)}")
        if d_codec <= 0:
            raise ValueError("d_codec must be positive")

        self.d_codec = d_codec
        self.modalities = frozenset(modalities)
        self.mv_stem = nn.Linear(_MV_FEATURES, d_codec)
        self.rpe_stem = nn.Linear(4, d_codec)
        self.i_frame_stem = nn.Linear(3, d_codec)
        self.metadata_stem = nn.Linear(_METADATA_FEATURES, d_codec)
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
            batch["mv_mask"].to(torch.bool)
            & batch["frame_validity"]["frame_valid"][..., None].to(torch.bool)
            & batch["frame_validity"]["mv_valid"][..., None].to(torch.bool)
            & batch["sample_valid"][:, None, None].to(torch.bool)
        )
        batch_size, history, count = mask.shape
        if history != _HISTORY:
            raise ValueError("Codec MV history must have length 8")
        device = mask.device
        if count == 0:
            return torch.zeros(
                (batch_size, history, _SPATIAL_TOKENS, _MV_FEATURES),
                device=device,
                dtype=dtype,
            )

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
        if features.shape != (batch_size, history, count, _MV_FEATURES):
            raise ValueError("Codec MV field shapes disagree")

        destination_x = fields["destination_x"].to(torch.float32)
        destination_y = fields["destination_y"].to(torch.float32)
        cells = (destination_y >= 240).to(torch.long) * 2 + (destination_x >= 320).to(torch.long)
        one_hot = F.one_hot(cells, num_classes=_SPATIAL_TOKENS).to(features.dtype)
        weights = one_hot * mask[..., None].to(features.dtype)
        sums = torch.einsum("btmc,btmf->btcf", weights, features)
        counts = weights.sum(dim=2).unsqueeze(-1).clamp_min(1.0)
        return (sums / counts).to(dtype)

    @staticmethod
    def _legacy_metadata(batch: Mapping[str, Any], frame_active: torch.Tensor) -> torch.Tensor:
        temporal = batch["time_and_reference"]
        frame_index = temporal["frame_index"].to(torch.float32)
        observation_row = batch["observation_row"].to(torch.float32)[:, None]
        pts = temporal["pts"].to(torch.float32)
        dts = temporal["dts"].to(torch.float32)
        reference_index = temporal["reference_index"].to(torch.float32)
        relative_reference = torch.where(
            reference_index >= 0,
            (frame_index - reference_index) / 30.0,
            torch.zeros_like(reference_index),
        )
        relative = torch.stack(
            (
                (frame_index - observation_row) / 7.0,
                (pts - pts[:, -1:]) / 30_000.0,
                (dts - dts[:, -1:]) / 30_000.0,
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
        camera_mask = batch["camera_motion_mask"].to(torch.bool) & frame_active
        camera = batch["camera_motion_summary"].to(torch.float32)
        camera = camera * camera_mask[..., None].to(camera.dtype)
        metadata = torch.cat(
            (
                batch["frame_type"].to(torch.float32).unsqueeze(-1) / 2.0,
                relative,
                validity,
                camera_mask.to(torch.float32).unsqueeze(-1),
                camera,
            ),
            dim=-1,
        )
        return metadata * frame_active[..., None].to(metadata.dtype)

    @staticmethod
    def _episode_metadata(batch: Mapping[str, Any], frame_active: torch.Tensor) -> torch.Tensor:
        metadata = batch["metadata_features"].to(torch.float32)
        if metadata.shape[1:] == (_HISTORY, _METADATA_FEATURES):
            return metadata * frame_active[..., None].to(metadata.dtype)
        if metadata.shape[1:] != (_HISTORY, _EPISODE_METADATA_FEATURES):
            raise ValueError("Episode-bundle metadata must be [B,8,18] or [B,8,21]")

        # Episode transport carries IDR/intra-non-IDR/P one-hot metadata and an
        # explicit anchor-valid bit. Preserve the frozen R4 18D core capacity by
        # folding those three types to one scalar and omitting anchor-valid; the
        # appearance branch already applies the authoritative slots and mask.
        frame_type_scalar = (
            metadata[..., 0:1] * 0.5 + metadata[..., 1:2] * 0.75 + metadata[..., 2:3]
        )
        metadata = torch.cat(
            (
                frame_type_scalar,
                metadata[..., 3:16],
                metadata[..., 17:21],
            ),
            dim=-1,
        )
        return metadata * frame_active[..., None].to(metadata.dtype)

    def forward(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        episode_bundle = "mv_cell_features" in batch
        if episode_bundle:
            frame_validity = batch["frame_validity"]
            frame_mask = batch["frame_mask"].to(torch.bool)
            if frame_validity.ndim != 3 or frame_validity.shape[1:] != (_HISTORY, 7):
                raise ValueError("Episode-bundle frame_validity must be [B,8,7]")
            if frame_mask.shape != frame_validity.shape[:2]:
                raise ValueError("Episode-bundle frame_mask must be [B,8]")
            sample_valid = batch["sample_valid"].to(torch.bool)
            if sample_valid.shape != frame_mask.shape[:1]:
                raise ValueError("Episode-bundle sample_valid must be [B]")
            sample_reason_code = batch["sample_reason_code"]
            if sample_reason_code.shape != frame_mask.shape[:1]:
                raise ValueError("Episode-bundle sample_reason_code must be [B]")
            frame_active = (
                frame_mask & frame_validity[..., 0].to(torch.bool) & sample_valid[:, None]
            )
        else:
            frame_valid = batch["frame_validity"]["frame_valid"].to(torch.bool)
            if frame_valid.ndim != 2 or frame_valid.shape[1] != _HISTORY:
                raise ValueError("Codec frame validity must be [B,8]")
            sample_valid = batch["sample_valid"].to(torch.bool)
            if sample_valid.shape != frame_valid.shape[:1]:
                raise ValueError("Codec sample validity must be [B]")
            frame_active = frame_valid & sample_valid[:, None]

        token_mask = frame_active.repeat_interleave(_SPATIAL_TOKENS, dim=1)
        reference = batch["rpe"]
        if reference.ndim != 5 or reference.shape[1:] != (_HISTORY, 2, 2, 4):
            raise ValueError("Codec RPE must be [B,8,2,2,4]")
        batch_size = reference.shape[0]
        combined = torch.zeros(
            (batch_size, _HISTORY, _SPATIAL_TOKENS, self.d_codec),
            device=reference.device,
            dtype=self.mv_stem.weight.dtype,
        )

        if episode_bundle:
            mv_features = batch["mv_cell_features"].to(self.mv_stem.weight.dtype)
            if mv_features.shape[1:] != (_HISTORY, 2, 2, _MV_FEATURES):
                raise ValueError("Episode-bundle MV cells must be [B,8,2,2,11]")
            mv_mask = batch["mv_cell_mask"].to(torch.bool)
            if mv_mask.shape != mv_features.shape[:-1]:
                raise ValueError("Episode-bundle MV cell mask shape disagrees")
            mv_mask = mv_mask & frame_active[..., None, None]
            mv_proposal = self.mv_stem(
                mv_features.reshape(batch_size, _HISTORY, _SPATIAL_TOKENS, _MV_FEATURES)
            ) * mv_mask.reshape(batch_size, _HISTORY, _SPATIAL_TOKENS, 1)
        else:
            mv_proposal = self.mv_stem(self._aggregate_mv(batch, self.mv_stem.weight.dtype))

        rpe = reference.to(self.rpe_stem.weight.dtype).reshape(
            batch_size, _HISTORY, _SPATIAL_TOKENS, 4
        )
        rpe_mask = batch["rpe_mask"].to(torch.bool)
        if rpe_mask.shape != reference.shape[:-1]:
            raise ValueError("Codec RPE mask shape disagrees")
        rpe_mask = rpe_mask & frame_active[..., None, None]
        rpe_proposal = self.rpe_stem(rpe) * rpe_mask.reshape(
            batch_size, _HISTORY, _SPATIAL_TOKENS, 1
        )

        if episode_bundle:
            anchor_bank = batch["anchor_appearance"].to(self.i_frame_stem.weight.dtype)
            if anchor_bank.ndim != 5 or anchor_bank.shape[1:] != (2, 3, 256, 256):
                raise ValueError("Episode-bundle anchor bank must be [B,2,3,256,256]")
            anchor_mask = batch["anchor_mask"].to(torch.bool)
            if anchor_mask.shape != (batch_size, 2):
                raise ValueError("Episode-bundle anchor mask must be [B,2]")
            slots = batch["frame_anchor_slot"].to(torch.long)
            if slots.shape != (batch_size, _HISTORY):
                raise ValueError("Episode-bundle frame anchor slots must be [B,8]")
            if ((slots < -1) | (slots >= 2)).any():
                raise ValueError("Episode-bundle frame anchor slots must be -1, 0, or 1")
            safe_slots = slots.clamp_min(0)
            batch_index = torch.arange(batch_size, device=anchor_bank.device)[:, None]
            selected = anchor_bank[batch_index, safe_slots]
            appearance = F.adaptive_avg_pool2d(selected.flatten(0, 1), (2, 2))
            appearance = appearance.reshape(batch_size, _HISTORY, 3, 2, 2)
            appearance = appearance.permute(0, 1, 3, 4, 2).reshape(
                batch_size, _HISTORY, _SPATIAL_TOKENS, 3
            )
            selected_anchor_valid = anchor_mask.gather(1, safe_slots)
            appearance_frame_mask = (slots >= 0) & selected_anchor_valid & frame_active
            appearance_mask = appearance_frame_mask[..., None, None]
        else:
            appearance_input = batch["i_frame_appearance"]
            if appearance_input.shape[1:] != (3, 256, 256):
                raise ValueError("Codec I-frame appearance must be [B,3,256,256]")
            appearance = F.adaptive_avg_pool2d(
                appearance_input.to(self.i_frame_stem.weight.dtype), (2, 2)
            )
            appearance = appearance.permute(0, 2, 3, 1)[:, None].expand(-1, _HISTORY, -1, -1, -1)
            appearance = appearance.reshape(batch_size, _HISTORY, _SPATIAL_TOKENS, 3)
            appearance_mask = (
                batch["i_frame_mask"].to(torch.bool)[:, None, None, None]
                & frame_active[..., None, None]
            )
        appearance_proposal = self.i_frame_stem(appearance) * appearance_mask

        if episode_bundle:
            metadata = self._episode_metadata(batch, frame_active)
        else:
            metadata = self._legacy_metadata(batch, frame_active)
        metadata = metadata[:, :, None, :].expand(-1, -1, _SPATIAL_TOKENS, -1)
        metadata_proposal = self.metadata_stem(metadata.to(self.metadata_stem.weight.dtype))

        proposals = {
            "mv": mv_proposal,
            "rpe": rpe_proposal,
            "i_frame": appearance_proposal,
            "metadata": metadata_proposal,
        }
        for name, proposal in proposals.items():
            if not torch.isfinite(proposal).all():
                raise FloatingPointError(f"Codec {name} stem produced NaN or Inf")
            combined = combined + proposal * int(name in self.modalities)

        tokens = self.mlp(self.norm(combined)).reshape(batch_size, _CONTEXT_TOKENS, self.d_codec)
        tokens = tokens * token_mask[..., None].to(tokens.dtype)
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("CodecDynamicsCore produced NaN or Inf")
        return tokens, token_mask


class Gr00tCodecProjector(nn.Module):
    """Project 32 codec-core tokens to the native action feature width."""

    def __init__(self, d_codec: int, action_width: int):
        super().__init__()
        self.projection = nn.Linear(d_codec, action_width)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != _CONTEXT_TOKENS:
            raise ValueError("Codec core tokens must be [B,32,d_codec]")
        if mask.shape != tokens.shape[:2]:
            raise ValueError("Codec token mask shape disagrees")
        projected = self.projection(tokens)
        return projected * mask[..., None].to(projected.dtype)


class ZeroGatedActionConditioner(nn.Module):
    """Deterministic cross-attention residual with a direct scalar gate."""

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
        if action_features.ndim != 3:
            raise ValueError("Native action features must be [B,H,D]")
        if context.ndim != 3 or context.shape[:2] != context_mask.shape:
            raise ValueError("Codec context and mask shapes disagree")
        if context.shape[0] != action_features.shape[0]:
            raise ValueError("Codec context batch size disagrees with action features")
        if context.shape[2] != action_features.shape[2]:
            raise ValueError("Codec context width disagrees with action features")

        context_mask = context_mask.to(torch.bool)
        any_valid = context_mask.any(dim=1)
        safe_context = context.clone()
        safe_mask = context_mask.clone()
        if (~any_valid).any():
            safe_context[~any_valid, 0] = 0
            safe_mask[~any_valid, 0] = True

        normalized_context = self.context_norm(safe_context)
        proposal, _ = self.attention(
            self.query_norm(action_features),
            normalized_context,
            normalized_context,
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
    """Map the sole codec evidence batch to an H40 action-side residual."""

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
    """Return the frozen analytic sizing contract; this is not a benchmark."""

    stack = Gr00tCodecConditioningStack(
        action_width=action_width,
        d_codec=d_codec,
        conditioner_heads=conditioner_heads,
    )
    parameters = sum(parameter.numel() for parameter in stack.parameters())
    encode_flops = (
        _HISTORY * _SPATIAL_TOKENS * (_MV_FEATURES + 4 + 3 + _METADATA_FEATURES) * d_codec * 2
        + _HISTORY * _SPATIAL_TOKENS * (d_codec * 2 * d_codec + 2 * d_codec * d_codec) * 2
        + _CONTEXT_TOKENS * d_codec * action_width * 2
    )
    condition_flops = (
        (40 + 2 * _CONTEXT_TOKENS) * action_width * action_width * 2
        + 2 * 40 * _CONTEXT_TOKENS * action_width
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
