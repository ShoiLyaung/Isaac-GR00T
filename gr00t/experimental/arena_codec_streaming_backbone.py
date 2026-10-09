"""VLA-free spatial and causal temporal backbones for Arena Codec IR v1.

The module intentionally consumes only already-extracted codec grids.  It has no
dependency on the native VLM, DiT, or action head.  ``mamba_ssm`` is imported
lazily so the repository's ordinary CPU test environment remains usable; an
official Mamba2 installation is required to construct ``CausalMamba2Backbone``.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math

import torch
from torch import nn


CODEC_FIELDS = 18
GRID_HEIGHT = 8
GRID_WIDTH = 8
WIDTH = 256
LAYERS = 2
SEED = 230923

MAMBA2_CONFIG = {
    "d_model": WIDTH,
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "ngroups": 1,
    "chunk_size": 16,
    "use_mem_eff_path": False,
}


def _require_floating_finite(value: torch.Tensor, label: str) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{label} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains nonfinite values")


def _validate_codec_sequence(value: torch.Tensor) -> tuple[int, int, int]:
    if value.ndim != 6 or tuple(value.shape[3:]) != (
        GRID_HEIGHT,
        GRID_WIDTH,
        CODEC_FIELDS,
    ):
        raise ValueError("expected codec sequence [batch,time,cameras,8,8,18]")
    if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] < 1:
        raise ValueError("codec sequence batch, time, and cameras must be nonempty")
    _require_floating_finite(value, "codec sequence")
    return int(value.shape[0]), int(value.shape[1]), int(value.shape[2])


def _validate_codec_frame(value: torch.Tensor) -> tuple[int, int]:
    if value.ndim != 5 or tuple(value.shape[2:]) != (
        GRID_HEIGHT,
        GRID_WIDTH,
        CODEC_FIELDS,
    ):
        raise ValueError("expected codec frame [batch,cameras,8,8,18]")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("codec frame batch and cameras must be nonempty")
    _require_floating_finite(value, "codec frame")
    return int(value.shape[0]), int(value.shape[1])


def _validate_descriptor_sequence(value: torch.Tensor) -> tuple[int, int]:
    if value.ndim != 3 or value.shape[-1] != WIDTH:
        raise ValueError("expected frame descriptors [batch,time,256]")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("descriptor batch and time must be nonempty")
    _require_floating_finite(value, "frame descriptors")
    return int(value.shape[0]), int(value.shape[1])


def _validate_descriptor(value: torch.Tensor) -> int:
    if value.ndim != 2 or value.shape[-1] != WIDTH:
        raise ValueError("expected frame descriptor [batch,256]")
    if value.shape[0] < 1:
        raise ValueError("descriptor batch must be nonempty")
    _require_floating_finite(value, "frame descriptor")
    return int(value.shape[0])


class ArenaCodecSpatialTokenizer(nn.Module):
    """Shared per-frame tokenizer preserving a 2x2 spatial layout before projection.

    The same convolutional weights are applied independently to every camera.
    Per-camera projected descriptors are averaged only after the 2x2 layout has
    been flattened and projected, which keeps the public interface camera-count
    agnostic while producing one descriptor per frame.
    """

    def __init__(self, *, seed: int = SEED):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.conv1 = nn.Conv2d(
                CODEC_FIELDS,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                device="cpu",
                dtype=torch.float32,
            )
            self.conv2 = nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                device="cpu",
                dtype=torch.float32,
            )
            self.projection = nn.Linear(
                128 * 2 * 2,
                WIDTH,
                device="cpu",
                dtype=torch.float32,
            )
        self.activation = nn.SiLU()

    def _encode_cameras(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 3, 1, 2)
        value = self.activation(self.conv1(value))
        value = self.activation(self.conv2(value))
        if tuple(value.shape[-2:]) != (2, 2):
            raise RuntimeError("codec spatial tokenizer must preserve a 2x2 layout")
        return self.projection(value.flatten(start_dim=1))

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        batch, cameras = _validate_codec_frame(frame)
        encoded = self._encode_cameras(frame.reshape(batch * cameras, 8, 8, CODEC_FIELDS))
        return encoded.reshape(batch, cameras, WIDTH).mean(dim=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, time, cameras = _validate_codec_sequence(sequence)
        encoded = self._encode_cameras(sequence.reshape(batch * time * cameras, 8, 8, CODEC_FIELDS))
        return encoded.reshape(batch, time, cameras, WIDTH).mean(dim=2)


@dataclass
class Mamba2State:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    batch_size: int
    observations: int = 0

    def clone(self) -> Mamba2State:
        return Mamba2State(
            [tuple(tensor.clone() for tensor in pair) for pair in self.layers],
            self.batch_size,
            self.observations,
        )

    @torch.no_grad()
    def reset(self) -> None:
        for pair in self.layers:
            for tensor in pair:
                tensor.zero_()
        self.observations = 0

    def tensor_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for pair in self.layers for tensor in pair
        )

    def serialized_nbytes(self) -> int:
        payload = {
            "batch_size": self.batch_size,
            "observations": self.observations,
            "layers": [tuple(tensor.detach().cpu() for tensor in pair) for pair in self.layers],
        }
        stream = BytesIO()
        torch.save(payload, stream)
        return stream.tell()

    def describe(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "observations": self.observations,
            "tensor_bytes": self.tensor_nbytes(),
            "serialized_bytes": self.serialized_nbytes(),
            "layers": [
                [
                    {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                    }
                    for tensor in pair
                ]
                for pair in self.layers
            ],
        }


class _Mamba2ResidualLayer(nn.Module):
    def __init__(self, index: int, *, chunk_size: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except Exception as error:
            raise RuntimeError(
                "official mamba-ssm Mamba2 is required for CausalMamba2Backbone"
            ) from error

        self.norm = nn.LayerNorm(WIDTH, device="cpu", dtype=torch.float32)
        config = {**MAMBA2_CONFIG, "chunk_size": chunk_size}
        self.mamba = Mamba2(
            **config,
            layer_idx=index,
            device="cpu",
            dtype=torch.float32,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mamba(self.norm(value))

    def step(
        self,
        value: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        update, conv_state, ssm_state = self.mamba.step(self.norm(value), *cache)
        if conv_state is not cache[0] or ssm_state is not cache[1]:
            raise RuntimeError("official Mamba2.step must update supplied caches in place")
        return value + update


class CausalMamba2Backbone(nn.Module):
    """Two-layer unidirectional Mamba2 with fixed-size persistent state."""

    def __init__(self, *, seed: int = SEED, chunk_size: int = MAMBA2_CONFIG["chunk_size"]):
        super().__init__()
        if chunk_size not in (1, MAMBA2_CONFIG["chunk_size"]):
            raise ValueError("Arena Mamba2 chunk size must be 1 or the frozen current value 16")
        self.chunk_size = chunk_size
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.layers = nn.ModuleList(
                [_Mamba2ResidualLayer(index, chunk_size=chunk_size) for index in range(LAYERS)]
            )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        _validate_descriptor_sequence(descriptors)
        value = descriptors
        for layer in self.layers:
            value = layer(value)
        return value

    def new_state(self, batch_size: int) -> Mamba2State:
        if batch_size < 1:
            raise ValueError("Mamba2 state batch must be positive")
        dtype = next(self.parameters()).dtype
        return Mamba2State(
            [
                layer.mamba.allocate_inference_cache(
                    batch_size,
                    max_seqlen=1,
                    dtype=dtype,
                )
                for layer in self.layers
            ],
            batch_size,
        )

    @torch.no_grad()
    def step(self, descriptor: torch.Tensor, state: Mamba2State) -> torch.Tensor:
        batch = _validate_descriptor(descriptor)
        if batch != state.batch_size or len(state.layers) != len(self.layers):
            raise ValueError("Mamba2 state does not match descriptor batch or layer count")
        value = descriptor.unsqueeze(1)
        for layer, cache in zip(self.layers, state.layers, strict=True):
            value = layer.step(value, cache)
        state.observations += 1
        return value.squeeze(1)

    @torch.no_grad()
    def stream(
        self,
        descriptors: torch.Tensor,
        state: Mamba2State | None = None,
    ) -> tuple[torch.Tensor, Mamba2State]:
        batch, _ = _validate_descriptor_sequence(descriptors)
        current = self.new_state(batch) if state is None else state
        outputs = [self.step(frame, current) for frame in descriptors.unbind(dim=1)]
        return torch.stack(outputs, dim=1), current


@dataclass
class GRUState:
    hidden: list[torch.Tensor]
    batch_size: int
    observations: int = 0

    def clone(self) -> GRUState:
        return GRUState(
            [tensor.clone() for tensor in self.hidden],
            self.batch_size,
            self.observations,
        )

    @torch.no_grad()
    def reset(self) -> None:
        for tensor in self.hidden:
            tensor.zero_()
        self.observations = 0

    def tensor_nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.hidden)


class _GRUResidualLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH, device="cpu", dtype=torch.float32)
        self.gru = nn.GRU(
            WIDTH,
            WIDTH,
            num_layers=1,
            batch_first=True,
            device="cpu",
            dtype=torch.float32,
        )


class CausalGRUBackbone(nn.Module):
    """Two-layer residual GRU with matching full and persistent-state APIs."""

    def __init__(self, *, seed: int = SEED):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.layers = nn.ModuleList([_GRUResidualLayer() for _ in range(LAYERS)])

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        batch, _ = _validate_descriptor_sequence(descriptors)
        value = descriptors
        for layer in self.layers:
            hidden = torch.zeros(
                1,
                batch,
                WIDTH,
                device=value.device,
                dtype=value.dtype,
            )
            update, _ = layer.gru(layer.norm(value), hidden)
            value = value + update
        return value

    def forward_explicit(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Run the same GRU weights one frame at a time while preserving autograd."""
        batch, _ = _validate_descriptor_sequence(descriptors)
        value = descriptors
        for layer in self.layers:
            hidden = torch.zeros(
                1,
                batch,
                WIDTH,
                device=value.device,
                dtype=value.dtype,
            )
            updates = []
            for frame in value.unbind(dim=1):
                update, hidden = layer.gru(layer.norm(frame).unsqueeze(1), hidden)
                updates.append(update)
            value = value + torch.cat(updates, dim=1)
        return value

    def new_state(self, batch_size: int) -> GRUState:
        if batch_size < 1:
            raise ValueError("GRU state batch must be positive")
        parameter = next(self.parameters())
        return GRUState(
            [
                torch.zeros(
                    1,
                    batch_size,
                    WIDTH,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                for _ in self.layers
            ],
            batch_size,
        )

    @torch.no_grad()
    def step(self, descriptor: torch.Tensor, state: GRUState) -> torch.Tensor:
        batch = _validate_descriptor(descriptor)
        if batch != state.batch_size or len(state.hidden) != len(self.layers):
            raise ValueError("GRU state does not match descriptor batch or layer count")
        value = descriptor.unsqueeze(1)
        for layer, hidden in zip(self.layers, state.hidden, strict=True):
            update, updated_hidden = layer.gru(layer.norm(value), hidden)
            hidden.copy_(updated_hidden)
            value = value + update
        state.observations += 1
        return value.squeeze(1)

    @torch.no_grad()
    def stream(
        self,
        descriptors: torch.Tensor,
        state: GRUState | None = None,
    ) -> tuple[torch.Tensor, GRUState]:
        batch, _ = _validate_descriptor_sequence(descriptors)
        current = self.new_state(batch) if state is None else state
        outputs = [self.step(frame, current) for frame in descriptors.unbind(dim=1)]
        return torch.stack(outputs, dim=1), current


class _SinusoidalPosition(nn.Module):
    def forward(
        self,
        length: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / width)
        )
        value = torch.zeros(length, width, device=device, dtype=torch.float32)
        value[:, 0::2] = torch.sin(position * divisor)
        value[:, 1::2] = torch.cos(position * divisor)
        return value.to(dtype=dtype)


class CausalTransformerBackbone(nn.Module):
    """Two-layer causal TransformerEncoder parameter-matched to Mamba2."""

    def __init__(self, *, seed: int = SEED):
        super().__init__()
        self.position = _SinusoidalPosition()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            layer = nn.TransformerEncoderLayer(
                d_model=WIDTH,
                nhead=8,
                dim_feedforward=320,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                device="cpu",
                dtype=torch.float32,
            )
            self.layers = nn.TransformerEncoder(
                layer,
                num_layers=LAYERS,
                enable_nested_tensor=False,
            )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        _, time = _validate_descriptor_sequence(descriptors)
        value = descriptors + self.position(
            time,
            WIDTH,
            descriptors.device,
            descriptors.dtype,
        )
        mask = torch.triu(
            torch.ones(time, time, device=value.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.layers(value, mask=mask, is_causal=True)


class CurrentDescriptorBackbone(nn.Module):
    """Parameter-free cost lower bound; excluded from parameter matching."""

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        _validate_descriptor_sequence(descriptors)
        return descriptors


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
