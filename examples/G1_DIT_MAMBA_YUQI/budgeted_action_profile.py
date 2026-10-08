"""Independent profiler scopes for the frozen action execution diagnosis.

All reported module ranges are inclusive and overlap. They are not additive
clean-wall latency attribution. No production model or kernel is rewritten.
"""

from collections import Counter, defaultdict
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch.profiler import record_function


def tensor_metadata(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (list, tuple)):
        return [tensor_metadata(item) for item in value]
    if isinstance(value, dict):
        return {
            key: tensor_metadata(item)
            for key, item in value.items()
            if isinstance(item, (torch.Tensor, list, tuple, dict))
        }
    return None


class ExecutionScopes:
    """Read-only forward hooks plus one scoped official scan-call wrapper."""

    def __init__(self, model, core, *, student, official_module):
        self.model, self.core, self.student, self.official = model, core, student, official_module
        self.stack = ExitStack()
        self.active = defaultdict(list)
        self.metadata = {}
        self.counts = Counter()

    def module(self, name, module):
        def before(current, args, kwargs):
            scope = record_function("execution/" + name)
            scope.__enter__()
            self.active[name].append(scope)
            self.counts[name] += 1
            if name not in self.metadata:
                self.metadata[name] = {
                    "args": tensor_metadata(args),
                    "kwargs": tensor_metadata(kwargs),
                    "direct_parameters": {
                        n: {"shape": list(p.shape), "dtype": str(p.dtype)}
                        for n, p in current.named_parameters(recurse=False)
                    },
                }

        def after(_module, _args, _kwargs, _output):
            if self.active[name]:
                self.active[name].pop().__exit__(None, None, None)

        self.stack.callback(module.register_forward_pre_hook(before, with_kwargs=True).remove)
        self.stack.callback(
            module.register_forward_hook(after, with_kwargs=True, always_call=True).remove
        )

    def __enter__(self):
        try:
            head = self.model.action_head
            self.module("VLM", self.model.backbone)
            for name in (
                "vlln",
                "vl_self_attention",
                "state_encoder",
                "action_encoder",
                "action_decoder",
            ):
                self.module("native/" + name, getattr(head, name))
            original_encode = head._encode_features

            def encode(*args, **kwargs):
                with record_function("execution/VL_q"):
                    return original_encode(*args, **kwargs)

            self.stack.enter_context(patch.object(head, "_encode_features", encode))
            self.module("action_core", self.core)
            if self.student:
                self.module("condition", self.core.condition_adapter)
                for index, block in enumerate(self.core.blocks):
                    base = f"layer{index}"
                    self.module(base + "/AdaLN", block.norm1)
                    self.module(base + "/FFN", block.ff)
                    self.module(base + "/down", block.mixer.down)
                    self.module(base + "/up", block.mixer.up)
                    self.module(base + "/Mamba", block.mixer.mamba)
                    for name, label in (
                        ("in_proj", "in_proj"),
                        ("conv1d", "conv"),
                        ("norm", "RMS"),
                        ("out_proj", "out_proj"),
                    ):
                        self.module(base + "/Mamba/" + label, getattr(block.mixer.mamba, name))
            else:
                for index, block in enumerate(self.core.transformer_blocks):
                    self.module(f"layer{index}/attention", block.attn1)
                    self.module(f"layer{index}/FFN", block.ff)
                    self.module(f"layer{index}/AdaLN", block.norm1)
            original_scan = self.official.mamba_chunk_scan_combined

            def scan(*args, **kwargs):
                self.counts["official_scan"] += 1
                if "official_scan" not in self.metadata:
                    self.metadata["official_scan"] = {
                        "args": tensor_metadata(args),
                        "kwargs": tensor_metadata(kwargs),
                    }
                with record_function("execution/official_mamba_chunk_scan_combined"):
                    return original_scan(*args, **kwargs)

            self.stack.enter_context(patch.object(self.official, "mamba_chunk_scan_combined", scan))
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        self.stack.close()
        for scopes in self.active.values():
            while scopes:
                scopes.pop().__exit__(*exc)


def event_row(event):
    return {
        "name": event.key,
        "count": event.count,
        "self_cpu_us": float(event.self_cpu_time_total),
        "inclusive_cpu_us": float(event.cpu_time_total),
        "inclusive_device_us": float(getattr(event, "device_time_total", 0.0)),
        "self_device_us": float(getattr(event, "self_device_time_total", 0.0)),
    }


def profiler_summary(profiler, scopes):
    averages = list(profiler.key_averages())
    rows = [event_row(event) for event in averages]
    kernels = {}
    for event in profiler.events():
        if str(event.device_type).endswith("CUDA"):
            row = kernels.setdefault(
                event.name, {"name": event.name, "count": 0, "summed_device_event_us": 0.0}
            )
            row["count"] += 1
            row["summed_device_event_us"] += float(event.time_range.elapsed_us())
    copy_names = {
        "aten::to",
        "aten::_to_copy",
        "aten::copy_",
        "aten::clone",
        "aten::contiguous",
        "aten::type_as",
    }
    return {
        "cpu_self_top30": sorted(rows, key=lambda row: row["self_cpu_us"], reverse=True)[:30],
        "cuda_device_event_top40": sorted(
            kernels.values(), key=lambda row: row["summed_device_event_us"], reverse=True
        )[:40],
        "cuda_device_events": sum(row["count"] for row in kernels.values()),
        "cast_copy_ops": [row for row in rows if row["name"] in copy_names],
        "inclusive_scopes": [row for row in rows if row["name"].startswith("execution/")],
        "module_call_counts": dict(scopes.counts),
        "observed_module_dtypes": scopes.metadata,
        "scope_warning": "Module/record_function CPU and device intervals are inclusive/overlapping. CUDA device events may include Memcpy/Memset, not only arithmetic kernels; their sums may overlap. Do not sum ranges or subtract from clean policy wall time. Dtypes describe observed module inputs/direct parameters, not every internal cast target.",
    }
