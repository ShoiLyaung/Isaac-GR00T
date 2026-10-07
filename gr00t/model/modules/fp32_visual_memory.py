"""Local FP32 execution of an unchanged causal visual memory module."""

from contextlib import contextmanager

import torch


@contextmanager
def _fp32_memory_region():
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


def explicit_cache_values(state):
    """Support only the two explicit cache layouts used by this comparison."""
    if hasattr(state, "layers") and not hasattr(state, "hidden"):
        return [value for pair in state.layers for value in pair]
    if hasattr(state, "hidden") and not hasattr(state, "layers"):
        return list(state.hidden)
    raise ValueError("Expected explicit Mamba (conv, SSM) or GRU hidden caches")


class FP32VisualMemory:
    """Thin executor; weights and explicit state remain owned by the original module.

    full includes the learned readout and preserves its autograd graph. step
    delegates to the original inference-only method and never owns hidden state.
    """

    def __init__(self, memory):
        self.memory = memory
        self._check_weights()

    def _check_weights(self):
        if any(parameter.dtype != torch.float32 for parameter in self.memory.parameters()):
            raise ValueError("FP32 memory execution requires FP32 master weights")

    def full(self, observations):
        self._check_weights()
        if observations.dtype != torch.float32:
            raise ValueError("FP32 memory execution requires FP32 observations")
        with _fp32_memory_region():
            latent = self.memory.full_sequence(observations)
            residual = self.memory.output_projection(latent[-1])
        return latent, residual

    def step(self, observation, state):
        self._check_weights()
        if observation.dtype != torch.float32:
            raise ValueError("FP32 memory execution requires FP32 observations")
        if any(value.dtype != torch.float32 for value in explicit_cache_values(state)):
            raise ValueError("FP32 memory execution requires FP32 explicit caches")
        with _fp32_memory_region():
            return self.memory.step(observation, state)
