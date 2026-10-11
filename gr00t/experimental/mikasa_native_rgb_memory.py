"""Native-RGB ordered-memory models for the controlled MIKASA experiment.

The target-blind CPU feature step reduces each native 128x128 camera frame to
an 8x8 block-mean grid. Trainable modules see native-RGB teacher grids, never
codec fields, and remain independent from the GR00T VLM, DiT, action head,
checkpoint, and policy-serving code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from gr00t.experimental.arena_codec_streaming_backbone import (
    WIDTH,
    CausalGRUBackbone,
    CausalMamba2Backbone,
    CausalTransformerBackbone,
    trainable_parameter_count,
)


SEQUENCE_LENGTH: Final = 130
GRID_HEIGHT: Final = 8
GRID_WIDTH: Final = 8
CAMERAS: Final = 2
RGB_CHANNELS: Final = 3
TEMPORAL_CHANNELS: Final = 9
PROPRIO_DIM: Final = 7
TARGET_SHAPE: Final = (2, 12, 2)
TARGET_DIM: Final = 48
CAMERA_DESCRIPTOR_WIDTH: Final = 128

EXPECTED_TEMPORAL_PARAMETERS: Final = {
    "mamba": 864_560,
    "gru": 790_528,
    "transformer": 857_216,
}

CONDITION_TO_BACKBONE: Final = {
    "mamba_current": "mamba",
    "mamba_one_frame": "mamba",
    "mamba_last_segment": "mamba",
    "mamba_full": "mamba",
    "gru_full": "gru",
    "transformer_full": "transformer",
}


def _require_grid(value: torch.Tensor, label: str, *, temporal: bool) -> None:
    expected = (
        (SEQUENCE_LENGTH, CAMERAS, GRID_HEIGHT, GRID_WIDTH, TEMPORAL_CHANNELS)
        if temporal
        else (CAMERAS, GRID_HEIGHT, GRID_WIDTH, RGB_CHANNELS)
    )
    if value.ndim != len(expected) + 1 or tuple(value.shape[1:]) != expected:
        raise ValueError(f"{label} has an invalid shape")
    if value.shape[0] < 1 or not value.is_floating_point():
        raise ValueError(f"{label} must be nonempty floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains nonfinite values")


def _require_proprio(value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != PROPRIO_DIM:
        raise ValueError("post proprio must have shape [batch,7]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("post proprio must be finite floating point")


def post_as_temporal_grid(post_rgb_grid: torch.Tensor) -> torch.Tensor:
    """Build the cue=post, signed-difference=0 temporal representation."""

    _require_grid(post_rgb_grid, "post RGB grid", temporal=False)
    return torch.cat((post_rgb_grid, post_rgb_grid, torch.zeros_like(post_rgb_grid)), dim=-1)


def apply_history_condition(
    temporal_grid: torch.Tensor,
    post_rgb_grid: torch.Tensor,
    condition: str,
) -> torch.Tensor:
    """Return one fixed-length grid view without changing shape or cadence."""

    if condition not in CONDITION_TO_BACKBONE:
        raise ValueError(f"unknown condition: {condition}")
    _require_grid(temporal_grid, "temporal RGB grid", temporal=True)
    _require_grid(post_rgb_grid, "post RGB grid", temporal=False)
    if temporal_grid.shape[0] != post_rgb_grid.shape[0]:
        raise ValueError("temporal/post grid batches must match")
    value = temporal_grid.clone()
    replacement = post_as_temporal_grid(post_rgb_grid).unsqueeze(1).expand_as(value)
    if condition == "mamba_current":
        value.copy_(replacement)
    elif condition == "mamba_one_frame":
        value[:, : SEQUENCE_LENGTH - 1].copy_(replacement[:, : SEQUENCE_LENGTH - 1])
    elif condition == "mamba_last_segment":
        value[:, :66].copy_(replacement[:, :66])
    return value


class GridCameraEncoder(nn.Module):
    """Shared per-camera encoder retaining a 2x2 spatial layout."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.projection = nn.Linear(128 * 2 * 2, CAMERA_DESCRIPTOR_WIDTH)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[1:]) != (
            self.in_channels,
            GRID_HEIGHT,
            GRID_WIDTH,
        ):
            raise ValueError(f"camera encoder expects [batch,{self.in_channels},8,8]")
        encoded = self.convs(value)
        if tuple(encoded.shape[-2:]) != (2, 2):
            raise RuntimeError("native RGB grid encoder lost the frozen 2x2 spatial layout")
        return self.projection(encoded.flatten(start_dim=1))


class NativeRGBTemporalFrontend(nn.Module):
    """Shared 9-channel grid encoder with ordered base/hand fusion."""

    def __init__(self):
        super().__init__()
        self.camera_encoder = GridCameraEncoder(TEMPORAL_CHANNELS)
        self.camera_fusion = nn.Sequential(
            nn.Linear(2 * CAMERA_DESCRIPTOR_WIDTH, WIDTH),
            nn.SiLU(),
            nn.LayerNorm(WIDTH),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or tuple(value.shape[1:]) != (
            CAMERAS,
            GRID_HEIGHT,
            GRID_WIDTH,
            TEMPORAL_CHANNELS,
        ):
            raise ValueError("temporal frontend expects [batch,2,8,8,9]")
        base = value[:, 0].permute(0, 3, 1, 2)
        hand = value[:, 1].permute(0, 3, 1, 2)
        base_descriptor = self.camera_encoder(base)
        hand_descriptor = self.camera_encoder(hand)
        return self.camera_fusion(torch.cat((base_descriptor, hand_descriptor), dim=-1))


class NativeRGBCurrentBranch(nn.Module):
    """Separate post-grid and post-proprio current descriptor branch."""

    def __init__(self):
        super().__init__()
        self.camera_encoder = GridCameraEncoder(RGB_CHANNELS)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(PROPRIO_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * CAMERA_DESCRIPTOR_WIDTH + 64, WIDTH),
            nn.SiLU(),
            nn.LayerNorm(WIDTH),
        )

    def forward(self, post_rgb_grid: torch.Tensor, post_proprio: torch.Tensor) -> torch.Tensor:
        _require_grid(post_rgb_grid, "post RGB grid", temporal=False)
        _require_proprio(post_proprio)
        base = self.camera_encoder(post_rgb_grid[:, 0].permute(0, 3, 1, 2))
        hand = self.camera_encoder(post_rgb_grid[:, 1].permute(0, 3, 1, 2))
        proprio = self.proprio_encoder(post_proprio.to(dtype=torch.float32))
        return self.fusion(torch.cat((base, hand, proprio), dim=-1))


def build_temporal_backbone(kind: str, *, seed: int) -> nn.Module:
    if kind == "mamba":
        module: nn.Module = CausalMamba2Backbone(seed=seed)
    elif kind == "gru":
        module = CausalGRUBackbone(seed=seed)
    elif kind == "transformer":
        module = CausalTransformerBackbone(seed=seed)
    else:
        raise ValueError(f"unknown temporal backbone: {kind}")
    observed = trainable_parameter_count(module)
    expected = EXPECTED_TEMPORAL_PARAMETERS[kind]
    if observed != expected:
        raise RuntimeError(
            f"{kind} temporal parameter contract changed: observed={observed}, expected={expected}"
        )
    return module


@dataclass(frozen=True)
class MemoryBatch:
    temporal_grid: torch.Tensor
    post_rgb_grid: torch.Tensor
    post_proprio: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.temporal_grid.shape[0])


class MikasaNativeRGBMemoryModel(nn.Module):
    """Standalone fixed-length ordered-memory predictor with hard per-call reset."""

    def __init__(self, condition: str, *, seed: int = 230923, frame_chunk_size: int = 32):
        super().__init__()
        if condition not in CONDITION_TO_BACKBONE:
            raise ValueError(f"unknown condition: {condition}")
        if frame_chunk_size < 1 or frame_chunk_size > SEQUENCE_LENGTH:
            raise ValueError("frame_chunk_size must be in [1,130]")
        self.condition = condition
        self.backbone_kind = CONDITION_TO_BACKBONE[condition]
        self.seed = int(seed)
        self.frame_chunk_size = int(frame_chunk_size)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.temporal_frontend = NativeRGBTemporalFrontend()
            self.current_branch = NativeRGBCurrentBranch()
            self.readout = nn.Sequential(
                nn.Linear(2 * WIDTH, WIDTH),
                nn.SiLU(),
                nn.LayerNorm(WIDTH),
                nn.Linear(WIDTH, TARGET_DIM),
            )
        self.temporal = build_temporal_backbone(self.backbone_kind, seed=self.seed)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_descriptors(self, batch: MemoryBatch) -> torch.Tensor:
        value = apply_history_condition(batch.temporal_grid, batch.post_rgb_grid, self.condition)
        device = self._device()
        descriptors: list[torch.Tensor] = []
        for start in range(0, SEQUENCE_LENGTH, self.frame_chunk_size):
            stop = min(start + self.frame_chunk_size, SEQUENCE_LENGTH)
            count = stop - start
            frames = value[:, start:stop].reshape(
                -1, CAMERAS, GRID_HEIGHT, GRID_WIDTH, TEMPORAL_CHANNELS
            )
            descriptor = self.temporal_frontend(frames.to(device=device, dtype=torch.float32))
            descriptors.append(descriptor.reshape(batch.batch_size, count, WIDTH))
        sequence = torch.cat(descriptors, dim=1)
        if sequence.shape != (batch.batch_size, SEQUENCE_LENGTH, WIDTH):
            raise RuntimeError("history descriptor shape changed")
        return sequence

    def encode_history(self, batch: MemoryBatch) -> torch.Tensor:
        sequence = self.encode_descriptors(batch)
        # The inherited full-sequence APIs create state locally. No recurrent
        # state is stored on this model or allowed to cross a forward call.
        return self.temporal(sequence)

    def forward(self, batch: MemoryBatch) -> torch.Tensor:
        history = self.encode_history(batch)[:, -1]
        device = self._device()
        current = self.current_branch(
            batch.post_rgb_grid.to(device=device, dtype=torch.float32),
            batch.post_proprio.to(device=device, dtype=torch.float32),
        )
        prediction = self.readout(torch.cat((history, current), dim=-1))
        if prediction.shape != (batch.batch_size, TARGET_DIM):
            raise RuntimeError("checkpoint head must return [batch,48]")
        return prediction


def temporal_parameter_identity(model: MikasaNativeRGBMemoryModel) -> dict[str, int | str]:
    return {
        "kind": model.backbone_kind,
        "parameters": trainable_parameter_count(model.temporal),
        "expected_parameters": EXPECTED_TEMPORAL_PARAMETERS[model.backbone_kind],
    }
