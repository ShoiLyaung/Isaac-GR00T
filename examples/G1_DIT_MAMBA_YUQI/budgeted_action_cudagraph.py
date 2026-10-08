"""Tensor-only action-core bridges and explicitly owned CUDA Graph lifetimes."""

from contextlib import contextmanager
import gc
import time

from gr00t.model.modules.budgeted_action_mamba import ActionCondition
import torch
from torch import nn


TENSOR_KEYS = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "image_mask",
    "backbone_attention_mask",
)


def storage_identity(core):
    return [
        (name, value.data_ptr(), str(value.dtype), str(value.device))
        for name, value in list(core.named_parameters()) + list(core.named_buffers())
    ]


def require_hook_free(module):
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in module.modules()):
        raise ValueError("All module hooks must be removed before graph capture")


class TensorCore(nn.Module):
    """New wrapper only; core.forward and all native algorithms are unchanged."""

    def __init__(self, core, *, student, backward):
        super().__init__()
        self.core = core
        self.student = student
        self.backward_surface = backward
        self.register_buffer(
            "unused_code",
            torch.zeros(1, 256, device=next(core.parameters()).device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.eval()

    def forward(self, hidden, features, timestep, image_mask, backbone_mask, *modulation):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            kwargs = dict(
                hidden_states=hidden,
                encoder_hidden_states=features,
                timestep=timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_mask,
                return_all_hidden_states=False,
            )
            if self.student:
                if self.backward_surface:
                    if modulation:
                        raise ValueError("F+B surface must encode its own condition")
                    condition = self.core.encode_condition(features)
                else:
                    if len(modulation) != 1:
                        raise ValueError("Inference surface needs explicit dynamic modulation")
                    condition = ActionCondition(self.unused_code, modulation[0])
                kwargs["condition_cache"] = condition
            elif modulation:
                raise ValueError("Native surface has no condition modulation")
            return self.core(**kwargs)


class ForwardGraph:
    """One forward-only capture; dynamic input copies remain outside the graph."""

    def __init__(self, surface, sample_args):
        require_hook_free(surface)
        if any(value.requires_grad for value in sample_args):
            raise ValueError("Inference graph inputs must not require gradients")
        self.surface = surface
        self.storage = storage_identity(surface.core)
        self.calls = 0
        self.closed = False
        self.static = tuple(value.detach().clone() for value in sample_args)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                surface(*self.static)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        warmup = time.perf_counter() - tick
        self.graph = torch.cuda.CUDAGraph()
        tick = time.perf_counter()
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
            self.output = surface(*self.static)
        torch.cuda.synchronize()
        self.setup = {
            "side_stream_warmup_seconds": warmup,
            "capture_seconds": time.perf_counter() - tick,
            "warmup_calls": 3,
        }

    def __call__(self, *args):
        if self.closed:
            raise RuntimeError("Graph was already released")
        if len(args) != len(self.static):
            raise ValueError("Static input count mismatch")
        for source, target in zip(args, self.static):
            if (
                source.shape != target.shape
                or source.dtype != target.dtype
                or source.device != target.device
            ):
                raise ValueError("Dynamic graph input shape/dtype/device changed")
            target.copy_(source)
        self.graph.replay()
        self.calls += 1
        # Borrowed until the next replay; the eager native decoder consumes it
        # immediately on the same stream. Persisted comparisons must clone/CPU-copy.
        return self.output

    def assert_storage(self):
        if storage_identity(self.surface.core) != self.storage:
            raise ValueError("Captured parameter/buffer storage changed")

    def close(self):
        if not self.closed:
            self.assert_storage()
            torch.cuda.synchronize()
            self.graph.reset()
            self.graph = None
            self.output = None
            self.static = ()
            self.surface = None
            self.closed = True
            gc.collect()
            torch.cuda.empty_cache()


class BackwardGraph:
    """Factory captures only a new wrapper, including student condition parameters."""

    def __init__(self, surface, sample_args):
        require_hook_free(surface)
        if torch.is_autocast_enabled("cuda"):
            raise ValueError("Factory must run outside autocast")
        self.surface = surface
        self.storage = storage_identity(surface.core)
        self.closed = False
        self.original_forward = surface.forward
        self.calls = 0
        torch.cuda.synchronize()
        tick = time.perf_counter()
        self.callable = torch.cuda.make_graphed_callables(
            surface, tuple(sample_args), num_warmup_iters=3, allow_unused_input=True
        )
        torch.cuda.synchronize()
        self.setup = {
            "factory_warmup_and_capture_seconds": time.perf_counter() - tick,
            "warmup_calls": 3,
            "scope": "Factory forward/backward graphs; autocast cacheFalse only inside tensor surface; allow-unused preserves native None gradients",
        }

    def __call__(self, *args):
        if self.closed:
            raise RuntimeError("Graph was already released")
        self.calls += 1
        return self.callable(*args)

    def assert_storage(self):
        if storage_identity(self.surface.core) != self.storage:
            raise ValueError("Captured parameter/buffer storage changed")

    def close(self):
        if not self.closed:
            self.assert_storage()
            self.surface.core.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            self.surface.forward = self.original_forward
            self.callable = None
            self.original_forward = None
            self.surface = None
            self.closed = True
            gc.collect()
            torch.cuda.empty_cache()


class CoreDispatcher(nn.Module):
    def __init__(self, surface, callable_surface=None):
        super().__init__()
        self.surface = surface
        self.backend = callable_surface
        self.calls = 0

    def encode_condition(self, features):
        return self.surface.core.encode_condition(features)

    def forward(self, *args, **kwargs):
        if args:
            raise ValueError("Native core bridge requires keyword inputs")
        tensors = tuple(kwargs[key] for key in TENSOR_KEYS)
        if self.surface.student and not self.surface.backward_surface:
            condition = kwargs.get("condition_cache")
            if not isinstance(condition, ActionCondition):
                raise ValueError("Missing explicit inference condition")
            tensors += (condition.modulation,)
        self.calls += 1
        output = self.surface(*tensors) if self.backend is None else self.backend(*tensors)
        return (output, [output]) if kwargs.get("return_all_hidden_states", False) else output


@contextmanager
def replace_core(model, core):
    original = model.action_head.model
    try:
        model.action_head.model = core
        yield
    finally:
        model.action_head.model = original


def tensor_args(kwargs, *, student, backward):
    values = tuple(kwargs[key].detach().clone() for key in TENSOR_KEYS)
    if student and not backward:
        values += (kwargs["condition_cache"].modulation.detach().clone(),)
    return values
