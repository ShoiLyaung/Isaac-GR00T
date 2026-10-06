"""Small recurrent history control with common projection/norm initialization.

GRU parameters/state/compute are reported, not made artificially equal to Mamba.
Time is the sequence axis; each of the 192 image slots is an independent stream.
"""

from dataclasses import dataclass

import torch
from torch import nn

from gr00t.model.modules.causal_visual_memory import INPUT_DIM, LAYERS, SLOTS, WIDTH


@dataclass
class GRUState:
    hidden: list[torch.Tensor]
    observations: int = 0

    def clone(self):
        return GRUState([v.clone() for v in self.hidden], self.observations)

    @torch.no_grad()
    def reset(self):
        for value in self.hidden:
            value.zero_()
        self.observations = 0

    def describe(self):
        return {
            "observations": self.observations,
            "elements": sum(v.numel() for v in self.hidden),
            "bytes": sum(v.numel() * v.element_size() for v in self.hidden),
            "layers": [
                {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
                for v in self.hidden
            ],
        }


class GRULayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH, device="cpu", dtype=torch.float32)
        self.cell = nn.GRUCell(WIDTH, WIDTH, device="cpu", dtype=torch.float32)


class CausalVisualGRU(nn.Module):
    def __init__(self, common_initialization):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Linear(INPUT_DIM, WIDTH, device="cpu", dtype=torch.float32)
            self.output_projection = nn.Linear(WIDTH, INPUT_DIM, device="cpu", dtype=torch.float32)
            # Recurrence has its own seed42 stream, independent of common projections.
            torch.random.default_generator.manual_seed(42)
            self.layers = nn.ModuleList([GRULayer() for _ in range(LAYERS)])
        with torch.no_grad():
            for name, value in self.state_dict().items():
                if ".cell." not in name:
                    value.copy_(common_initialization[name])

    def new_state(self):
        device = self.input_projection.weight.device
        return GRUState(
            [torch.zeros(SLOTS, WIDTH, device=device, dtype=torch.float32) for _ in self.layers]
        )

    @torch.no_grad()
    def step(self, image_slots, state):
        if tuple(image_slots.shape) != (SLOTS, INPUT_DIM):
            raise ValueError("Expected one observation of 192 visual slots")
        value = self.input_projection(image_slots)
        for layer, hidden in zip(self.layers, state.hidden):
            updated = layer.cell(layer.norm(value), hidden).float()
            hidden.copy_(updated)
            value = value + updated.to(value.dtype)
        state.observations += 1
        return value, self.output_projection(value)

    def full_sequence(self, observations):
        if observations.ndim != 3 or tuple(observations.shape[1:]) != (SLOTS, INPUT_DIM):
            raise ValueError("Expected [observation_time,192,2048]")
        value = self.input_projection(observations)
        for layer in self.layers:
            hidden = torch.zeros(SLOTS, WIDTH, device=value.device, dtype=torch.float32)
            outputs = []
            for current in value:
                # Match deployment's FP32 hidden storage; cast is differentiable.
                hidden = layer.cell(layer.norm(current), hidden).float()
                outputs.append(current + hidden.to(current.dtype))
            value = torch.stack(outputs)
        return value
