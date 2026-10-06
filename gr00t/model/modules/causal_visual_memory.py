"""Small observation-time memory over 192 independent camera/spatial slots.

No model installation, action-time recurrence, global cache, or attention is
hidden here. The caller owns/reset/clones MemoryState and injects the readout at
the original post-VL image positions. step is inference-only; full_sequence is
the official differentiable SSD path from a zero initial state.
"""

from dataclasses import dataclass

import torch
from torch import nn


SLOTS, INPUT_DIM, WIDTH, LAYERS = 192, 2048, 256, 2
CACHE_ELEMENTS = 13_565_952
MAMBA_CONFIG = {
    "d_model": WIDTH,
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "ngroups": 1,
    "chunk_size": 16,
    "use_mem_eff_path": False,
}


@dataclass
class MemoryState:
    """One separately owned official (conv, SSM) cache pair per layer."""

    layers: list[tuple[torch.Tensor, torch.Tensor]]
    observations: int = 0

    def clone(self):
        return MemoryState(
            [
                tuple(t.clone(memory_format=torch.preserve_format) for t in pair)
                for pair in self.layers
            ],
            self.observations,
        )

    def to(self, device):
        # A transfer also owns its storage when the requested device is unchanged.
        return MemoryState(
            [tuple(t.to(device=device, copy=True) for t in pair) for pair in self.layers],
            self.observations,
        )

    @torch.no_grad()
    def reset(self):
        for pair in self.layers:
            for value in pair:
                value.zero_()
        self.observations = 0

    def describe(self):
        return {
            "observations": self.observations,
            "elements": sum(t.numel() for pair in self.layers for t in pair),
            "bytes": sum(t.numel() * t.element_size() for pair in self.layers for t in pair),
            "layers": [
                [
                    {
                        "shape": list(t.shape),
                        "stride": list(t.stride()),
                        "dtype": str(t.dtype),
                        "device": str(t.device),
                    }
                    for t in pair
                ]
                for pair in self.layers
            ],
        }


class MemoryLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        from mamba_ssm import Mamba2

        self.norm = nn.LayerNorm(WIDTH, device="cpu", dtype=torch.float32)
        self.mamba = Mamba2(**MAMBA_CONFIG, layer_idx=index, device="cpu", dtype=torch.float32)

    def forward(self, value):
        return value + self.mamba(self.norm(value))

    def step(self, value, cache):
        update, conv, ssm = self.mamba.step(self.norm(value), *cache)
        if conv is not cache[0] or ssm is not cache[1]:
            raise RuntimeError("Official step must update the supplied caches in place")
        return value + update


class CausalVisualMemory(nn.Module):
    """FP32 master weights; outer BF16 autocast controls arithmetic, not storage."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.input_projection = nn.Linear(INPUT_DIM, WIDTH, device="cpu", dtype=torch.float32)
            self.layers = nn.ModuleList([MemoryLayer(index) for index in range(LAYERS)])
            self.output_projection = nn.Linear(WIDTH, INPUT_DIM, device="cpu", dtype=torch.float32)
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def new_state(self):
        state = MemoryState(
            [
                layer.mamba.allocate_inference_cache(SLOTS, max_seqlen=1, dtype=torch.float32)
                for layer in self.layers
            ]
        )
        for conv, ssm in state.layers:
            if tuple(conv.shape) != (SLOTS, 640, 4) or tuple(ssm.shape) != (SLOTS, 8, 64, 64):
                raise ValueError("Official cache shapes differ from the frozen memory contract")
            if (
                conv.stride() != (2560, 1, 640)
                or conv.dtype != torch.float32
                or ssm.dtype != torch.float32
            ):
                raise ValueError("Expected official transposed FP32 conv cache and FP32 SSM cache")
        if state.describe()["elements"] != CACHE_ELEMENTS:
            raise ValueError("Unexpected memory cache element count")
        return state

    @torch.no_grad()
    def step(self, image_slots, state):
        if tuple(image_slots.shape) != (SLOTS, INPUT_DIM) or len(state.layers) != LAYERS:
            raise ValueError("Expected one current observation with 192 independent image slots")
        latent = self.input_projection(image_slots).unsqueeze(1)
        for layer, cache in zip(self.layers, state.layers):
            latent = layer.step(latent, cache)
        state.observations += 1
        latent = latent.squeeze(1)
        return latent, self.output_projection(latent)

    def full_sequence(self, observations):
        """[time, slot, feature] -> [time, slot, latent], preserving each slot's identity."""
        if observations.ndim != 3 or tuple(observations.shape[1:]) != (SLOTS, INPUT_DIM):
            raise ValueError("Expected [observation_time,192,2048] image features")
        # A slot is batch, observation time is sequence; camera/spatial slots never mix.
        value = self.input_projection(observations.transpose(0, 1))
        for layer in self.layers:
            value = layer(value)
        return value.transpose(0, 1)


def inject_image_residual(backbone_features, image_positions, residual):
    """Keep token count/order, all non-image positions, and the current-Z direct path."""
    if tuple(backbone_features.shape) != (1, 209, INPUT_DIM):
        raise ValueError("Expected native [1,209,2048] post-VL features")
    if tuple(image_positions.shape) != (SLOTS,) or tuple(residual.shape) != (SLOTS, INPUT_DIM):
        raise ValueError("Expected exactly the original 192 image positions and residuals")
    result = backbone_features.clone()
    current = backbone_features[0].index_select(0, image_positions)
    result[0].index_copy_(0, image_positions, current + residual.to(backbone_features.dtype))
    return result
