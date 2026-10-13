"""Matched causal backbones for two-slot MIKASA history-hard order prediction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.experimental.arena_codec_streaming_backbone import WIDTH
from gr00t.experimental.mikasa_native_rgb_memory import (
    CAMERAS,
    GRID_HEIGHT,
    GRID_WIDTH,
    PROPRIO_DIM,
    RGB_CHANNELS,
    SEQUENCE_LENGTH,
    TEMPORAL_CHANNELS,
    NativeRGBCurrentBranch,
    NativeRGBTemporalFrontend,
    build_temporal_backbone,
    post_as_temporal_grid,
)


CONDITIONS: Final = (
    "mamba_current",
    "mamba_zero",
    "mamba_shuffle",
    "gru_full",
    "transformer_full",
    "mamba_full",
)
CONDITION_TO_BACKBONE: Final = {
    "mamba_current": "mamba",
    "mamba_zero": "mamba",
    "mamba_shuffle": "mamba",
    "gru_full": "gru",
    "transformer_full": "transformer",
    "mamba_full": "mamba",
}
BOUNDARY_INDICES: Final = (65, 129)
SLOTS: Final = 2
CLASSES: Final = 3
SHUFFLE_DOMAIN: Final = "history-shuffle-v1"
DEV_SHUFFLE_SEEDS: Final = (230941, 230942, 230943, 230944)


def history_shuffle_permutation(seed: int, step: int) -> torch.Tensor:
    """One batch-global permutation derived only from the frozen seed and step."""

    if step < 0:
        raise ValueError("shuffle step must be nonnegative")
    payload = f"{SHUFFLE_DOMAIN}|{int(seed)}|{int(step)}".encode()
    generator_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    return torch.randperm(SEQUENCE_LENGTH, generator=generator, dtype=torch.int64)


@dataclass(frozen=True)
class OrderBatch:
    temporal_grid: torch.Tensor
    current_rgb_grid: torch.Tensor
    current_proprio: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.temporal_grid.shape[0])

    def validate(self) -> None:
        expected_temporal = (
            SEQUENCE_LENGTH,
            CAMERAS,
            GRID_HEIGHT,
            GRID_WIDTH,
            TEMPORAL_CHANNELS,
        )
        expected_current = (CAMERAS, GRID_HEIGHT, GRID_WIDTH, RGB_CHANNELS)
        if self.temporal_grid.ndim != 6 or tuple(self.temporal_grid.shape[1:]) != expected_temporal:
            raise ValueError("temporal grid must be [batch,130,2,8,8,9]")
        if (
            self.current_rgb_grid.ndim != 5
            or tuple(self.current_rgb_grid.shape[1:]) != expected_current
        ):
            raise ValueError("current RGB grid must be [batch,2,8,8,3]")
        if self.current_proprio.shape != (self.batch_size, PROPRIO_DIM):
            raise ValueError("current proprio must be [batch,7]")
        if self.current_rgb_grid.shape[0] != self.batch_size:
            raise ValueError("current RGB batch does not match temporal batch")
        for label, value in (
            ("temporal", self.temporal_grid),
            ("current RGB", self.current_rgb_grid),
            ("current proprio", self.current_proprio),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{label} must be finite floating point")


@dataclass(frozen=True)
class OrderOutput:
    logits: torch.Tensor
    boundary_states: torch.Tensor
    current_descriptor: torch.Tensor

    @property
    def predictions(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)


def apply_history_condition(
    temporal_grid: torch.Tensor,
    current_rgb_grid: torch.Tensor,
    condition: str,
    *,
    shuffle_seed: int | None = None,
    shuffle_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply exactly one evidence intervention while preserving all 130 steps."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown order condition: {condition}")
    value = temporal_grid.clone()
    permutation = None
    if condition == "mamba_current":
        current = post_as_temporal_grid(current_rgb_grid)
        value.copy_(current.unsqueeze(1).expand_as(value))
    elif condition == "mamba_zero":
        value.zero_()
    elif condition == "mamba_shuffle":
        if shuffle_seed is None or shuffle_step is None:
            raise ValueError("mamba_shuffle requires an explicit global seed and step")
        permutation = history_shuffle_permutation(shuffle_seed, shuffle_step)
        value = value.index_select(1, permutation.to(device=value.device))
    return value, permutation


class MikasaHistoryHardOrderModel(nn.Module):
    """Shared frontend/current/head with a reset-per-call causal temporal backbone."""

    def __init__(self, condition: str, *, seed: int, frame_chunk_size: int = 32):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown order condition: {condition}")
        if frame_chunk_size < 1 or frame_chunk_size > SEQUENCE_LENGTH:
            raise ValueError("frame chunk size must be in [1,130]")
        self.condition = condition
        self.seed = int(seed)
        self.frame_chunk_size = int(frame_chunk_size)
        self.backbone_kind = CONDITION_TO_BACKBONE[condition]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.temporal_frontend = NativeRGBTemporalFrontend()
            self.current_branch = NativeRGBCurrentBranch()
            self.shared_head = nn.Sequential(
                nn.Linear(2 * WIDTH, WIDTH),
                nn.SiLU(),
                nn.LayerNorm(WIDTH),
                nn.Linear(WIDTH, CLASSES),
            )
        self.temporal = build_temporal_backbone(self.backbone_kind, seed=self.seed)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_descriptors(self, temporal_grid: torch.Tensor) -> torch.Tensor:
        batch = int(temporal_grid.shape[0])
        device = self._device()
        descriptors: list[torch.Tensor] = []
        for start in range(0, SEQUENCE_LENGTH, self.frame_chunk_size):
            stop = min(start + self.frame_chunk_size, SEQUENCE_LENGTH)
            count = stop - start
            frames = temporal_grid[:, start:stop].reshape(
                -1,
                CAMERAS,
                GRID_HEIGHT,
                GRID_WIDTH,
                TEMPORAL_CHANNELS,
            )
            encoded = self.temporal_frontend(frames.to(device=device, dtype=torch.float32))
            descriptors.append(encoded.reshape(batch, count, WIDTH))
        result = torch.cat(descriptors, dim=1)
        if result.shape != (batch, SEQUENCE_LENGTH, WIDTH):
            raise RuntimeError("temporal descriptor shape changed")
        return result

    def forward(
        self,
        batch: OrderBatch,
        *,
        shuffle_seed: int | None = None,
        shuffle_step: int | None = None,
        condition_override: str | None = None,
    ) -> OrderOutput:
        batch.validate()
        condition = self.condition if condition_override is None else condition_override
        if condition_override is not None and condition_override not in {
            "mamba_zero",
            "mamba_shuffle",
            "mamba_full",
        }:
            raise ValueError("inference condition override must be a Mamba intervention")
        temporal, _ = apply_history_condition(
            batch.temporal_grid,
            batch.current_rgb_grid,
            condition,
            shuffle_seed=shuffle_seed,
            shuffle_step=shuffle_step,
        )
        descriptors = self.encode_descriptors(temporal)
        # Every forward invokes the inherited stateless full-sequence API. No
        # recurrent cache is stored, so state cannot cross a row or batch.
        states = self.temporal(descriptors)
        boundary = states[:, BOUNDARY_INDICES]
        device = self._device()
        current = self.current_branch(
            batch.current_rgb_grid.to(device=device, dtype=torch.float32),
            batch.current_proprio.to(device=device, dtype=torch.float32),
        )
        paired = torch.cat(
            (boundary, current.unsqueeze(1).expand(-1, SLOTS, -1)),
            dim=-1,
        )
        logits = self.shared_head(paired.reshape(-1, 2 * WIDTH)).reshape(-1, SLOTS, CLASSES)
        if logits.shape != (batch.batch_size, SLOTS, CLASSES):
            raise RuntimeError("shared order head must return [batch,2,3]")
        return OrderOutput(logits=logits, boundary_states=boundary, current_descriptor=current)


def order_cross_entropy(output: OrderOutput, labels: torch.Tensor) -> torch.Tensor:
    if labels.shape != (output.logits.shape[0], SLOTS) or labels.dtype != torch.int64:
        raise ValueError("order labels must be int64 [batch,2]")
    if bool(((labels < 0) | (labels >= CLASSES)).any()):
        raise ValueError("order labels must be in [0,2]")
    return F.cross_entropy(output.logits.reshape(-1, CLASSES), labels.reshape(-1), reduction="mean")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
