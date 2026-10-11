"""Two-slot upper-bound model for the controlled MIKASA memory diagnosis.

The model consumes only the frozen target-blind native-RGB feature grid. It is
independent from Mamba, codec inputs, the GR00T VLM, DiT, and action head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


SEQUENCE_LENGTH: Final = 130
SLOT_BOUNDARIES: Final = ((0, 66), (66, 130))
SLOTS: Final = 2
CAMERAS: Final = 2
GRID_HEIGHT: Final = 8
GRID_WIDTH: Final = 8
INPUT_CHANNELS: Final = 9
POOLED_CHANNELS: Final = 18
CAMERA_WIDTH: Final = 128
SLOT_WIDTH: Final = 256
SHAPE_CLASSES: Final = 3
CHECKPOINTS_PER_SLOT: Final = 12
CHECKPOINT_DIM_PER_SLOT: Final = 24


def _validate_temporal_grid(value: torch.Tensor) -> None:
    if value.ndim != 6 or tuple(value.shape[1:]) != (
        SEQUENCE_LENGTH,
        CAMERAS,
        GRID_HEIGHT,
        GRID_WIDTH,
        INPUT_CHANNELS,
    ):
        raise ValueError("temporal grid must have shape [batch,130,2,8,8,9]")
    if value.shape[0] < 1 or not value.is_floating_point():
        raise ValueError("temporal grid must be nonempty floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("temporal grid contains nonfinite values")


def _validate_slot_maps(value: torch.Tensor) -> None:
    if value.ndim != 6 or tuple(value.shape[1:]) != (
        SLOTS,
        CAMERAS,
        GRID_HEIGHT,
        GRID_WIDTH,
        POOLED_CHANNELS,
    ):
        raise ValueError("slot maps must have shape [batch,2,2,8,8,18]")
    if value.shape[0] < 1 or not value.is_floating_point():
        raise ValueError("slot maps must be nonempty floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("slot maps contain nonfinite values")


@dataclass(frozen=True)
class SegmentSlotBatch:
    """The complete model-input ABI; no target or association metadata is allowed."""

    temporal_grid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.temporal_grid.shape[0])


@dataclass(frozen=True)
class SegmentSlotOutput:
    shape_logits: torch.Tensor
    normalized_checkpoints: torch.Tensor


class SharedSlotCameraEncoder(nn.Module):
    """One camera CNN reused by both cameras and both temporal slots."""

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(POOLED_CHANNELS, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.projection = nn.Linear(128 * 2 * 2, CAMERA_WIDTH)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[1:]) != (
            POOLED_CHANNELS,
            GRID_HEIGHT,
            GRID_WIDTH,
        ):
            raise ValueError("shared camera encoder expects [batch,18,8,8]")
        encoded = self.convs(value)
        if tuple(encoded.shape[-2:]) != (2, 2):
            raise RuntimeError("slot camera encoder lost the frozen 2x2 layout")
        return self.projection(encoded.flatten(start_dim=1))


class MikasaSegmentSlotModel(nn.Module):
    """Shared two-slot shape and checkpoint decoder with exact swap equivariance."""

    def __init__(self, *, seed: int = 230923):
        super().__init__()
        self.seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.camera_encoder = SharedSlotCameraEncoder()
            self.camera_fusion = nn.Sequential(
                nn.Linear(2 * CAMERA_WIDTH, SLOT_WIDTH),
                nn.SiLU(),
                nn.LayerNorm(SLOT_WIDTH),
            )
            self.shape_head = nn.Linear(SLOT_WIDTH, SHAPE_CLASSES)
            self.checkpoint_head = nn.Sequential(
                nn.Linear(SLOT_WIDTH, SLOT_WIDTH),
                nn.SiLU(),
                nn.Linear(SLOT_WIDTH, CHECKPOINT_DIM_PER_SLOT),
            )

    @staticmethod
    def pool_segments(temporal_grid: torch.Tensor) -> torch.Tensor:
        _validate_temporal_grid(temporal_grid)
        slots = []
        for start, stop in SLOT_BOUNDARIES:
            segment = temporal_grid[:, start:stop]
            slots.append(torch.cat((segment.mean(dim=1), segment.amax(dim=1)), dim=-1))
        value = torch.stack(slots, dim=1)
        _validate_slot_maps(value)
        return value

    def encode_slot_maps(self, slot_maps: torch.Tensor) -> torch.Tensor:
        _validate_slot_maps(slot_maps)
        batch = int(slot_maps.shape[0])
        flattened = slot_maps.reshape(
            batch * SLOTS * CAMERAS,
            GRID_HEIGHT,
            GRID_WIDTH,
            POOLED_CHANNELS,
        ).permute(0, 3, 1, 2)
        camera = self.camera_encoder(flattened).reshape(
            batch,
            SLOTS,
            CAMERAS,
            CAMERA_WIDTH,
        )
        # Concatenation order is the base/hand camera-identity contract.
        return self.camera_fusion(torch.cat((camera[:, :, 0], camera[:, :, 1]), dim=-1))

    def forward_slot_maps(self, slot_maps: torch.Tensor) -> SegmentSlotOutput:
        slots = self.encode_slot_maps(slot_maps)
        shape_logits = self.shape_head(slots)
        checkpoints = self.checkpoint_head(slots)
        if shape_logits.shape != (slots.shape[0], SLOTS, SHAPE_CLASSES):
            raise RuntimeError("shape head output contract changed")
        if checkpoints.shape != (slots.shape[0], SLOTS, CHECKPOINT_DIM_PER_SLOT):
            raise RuntimeError("checkpoint head output contract changed")
        return SegmentSlotOutput(shape_logits, checkpoints)

    def forward(self, batch: SegmentSlotBatch) -> SegmentSlotOutput:
        return self.forward_slot_maps(self.pool_segments(batch.temporal_grid))


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
