"""Read-only scoped observations of the existing memory production methods."""

from contextlib import ExitStack, contextmanager
import inspect
from unittest.mock import patch

import numpy as np
import torch


@contextmanager
def memory_precision(precision):
    if precision not in ("bf16", "fp32"):
        raise ValueError(precision)
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = precision == "bf16"
        torch.backends.cudnn.allow_tf32 = precision == "bf16"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


def normalize_activation(value, path, length, slots, *, conv=False, readout=False):
    """Return [time,slot,feature]; trim only full Conv1d's unused right padding."""
    if value.shape[0] != slots:
        raise ValueError("Trace slot axis changed")
    if path == "step" or readout:
        return value.reshape(1, slots, -1)
    if conv:
        value = value.transpose(1, 2)
    if value.ndim < 3 or value.shape[1] < length:
        raise ValueError("Trace time axis changed")
    return value[:, :length].transpose(0, 1).flatten(2)


def tensor_metadata(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride())}


class MemoryTrace:
    """One fixed model/prefix observer; all captured values own CPU storage."""

    def __init__(self, memory, official, path, length, slots=192):
        self.memory, self.official, self.path = memory, official, path
        self.length, self.slots = length, slots
        self.time = None
        self.layer = None
        self.tensors = {}
        self.metadata = {}
        self.states = []
        self.context = ExitStack()

    def capture(self, key, value, *, conv=False, readout=False):
        if not isinstance(value, torch.Tensor):
            return
        native = tensor_metadata(value)
        canonical = normalize_activation(
            value, self.path, self.length, self.slots, conv=conv, readout=readout
        )
        snapshot = canonical.detach().cpu().clone()
        if not bool(torch.isfinite(snapshot).all()):
            raise ValueError(f"Nonfinite trace stage: {key}")
        times = (
            [self.time if self.time is not None else self.length - 1]
            if (self.path == "step" or readout)
            else list(range(self.length))
        )
        entries = self.tensors.setdefault(key, {})
        for index, time_index in enumerate(times):
            if time_index in entries and not torch.equal(entries[time_index], snapshot[index]):
                raise ValueError(f"Repeated snapshot read changed: {key}")
            entries[time_index] = snapshot[index].clone()
        self.metadata.setdefault(key, []).append(
            {"time": self.time, "native": native, "times": times}
        )

    def hook(self, module, key, *, conv=False, readout=False):
        def record(_module, args, output):
            self.capture(key, output, conv=conv, readout=readout)
            self.metadata[key][-1]["input_dtypes"] = [
                str(v.dtype) for v in args if isinstance(v, torch.Tensor)
            ]

        self.context.callback(module.register_forward_hook(record).remove)

    def observe_layer(self, index, original):
        def call(*args, **kwargs):
            previous = self.layer
            self.layer = index
            try:
                output = original(*args, **kwargs)
                self.capture(f"layer{index}.residual", output)
                return output
            finally:
                self.layer = previous

        return call

    def observe_step(self, original):
        def call(value, state):
            previous = self.time
            self.time = state.observations
            try:
                result = original(value, state)
                if state.observations != self.time + 1:
                    raise ValueError("Observation counter changed unexpectedly")
                summary = []
                for pair in state.layers:
                    layer = []
                    for cache in pair:
                        if cache.dtype != torch.float32 or not bool(torch.isfinite(cache).all()):
                            raise ValueError("Expected finite FP32 step cache")
                        layer.append(
                            {
                                **tensor_metadata(cache),
                                "finite": True,
                                "mean": float(cache.mean()),
                                "rms": float(cache.square().mean().sqrt()),
                                "maxabs": float(cache.abs().max()),
                            }
                        )
                    summary.append(layer)
                self.states.append(
                    {"time": self.time, "observations": state.observations, "layers": summary}
                )
                return result
            finally:
                self.time = previous

        return call

    def observe_ssm(self, original):
        signature = inspect.signature(original)

        def call(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            if self.layer is None:
                raise ValueError("SSM call occurred outside a traced memory layer")
            key = f"layer{self.layer}.ssm"
            metadata = {
                name: tensor_metadata(v) for name, v in bound.items() if isinstance(v, torch.Tensor)
            }
            self.metadata.setdefault(key + ".arguments", []).append(
                {"time": self.time, "tensors": metadata}
            )
            for name in ("x", "B", "C", "dt"):
                value = bound.get(name)
                if value is None:
                    continue
                if name == "dt" and self.path == "step" and value.ndim == 3:
                    # Official step broadcasts a head's dt across headdim.
                    if not torch.equal(value, value[:, :, :1].expand_as(value)):
                        raise ValueError("Unexpected non-broadcast step dt")
                    value = value[:, :, 0]
                self.capture(key + "." + name, value)
            output = original(*args, **kwargs)
            self.capture(key + ".output", output[0] if isinstance(output, tuple) else output)
            return output

        return call

    def __enter__(self):
        try:
            self.hook(self.memory.input_projection, "input_projection")
            self.hook(self.memory.output_projection, "output_projection", readout=True)
            for index, layer in enumerate(self.memory.layers):
                prefix = f"layer{index}"
                for name, module in [
                    ("norm", layer.norm),
                    ("in_proj", layer.mamba.in_proj),
                    ("gated_norm", layer.mamba.norm),
                    ("out_proj", layer.mamba.out_proj),
                ]:
                    self.hook(module, prefix + "." + name)
                if self.path == "full":
                    self.hook(layer.mamba.conv1d, prefix + ".conv_module", conv=True)
                self.hook(layer.mamba.act, prefix + ".act")

                def before_act(_module, args, key=prefix + ".conv_preact"):
                    self.capture(key, args[0])

                self.context.callback(layer.mamba.act.register_forward_pre_hook(before_act).remove)
                method = "forward" if self.path == "full" else "step"
                self.context.enter_context(
                    patch.object(layer, method, self.observe_layer(index, getattr(layer, method)))
                )
            function = (
                "mamba_chunk_scan_combined" if self.path == "full" else "selective_state_update"
            )
            original = getattr(self.official, function)
            if original is None:
                raise ValueError(f"Frozen official SSM backend unavailable: {function}")
            self.context.enter_context(
                patch.object(self.official, function, self.observe_ssm(original))
            )
            if self.path == "step":
                self.context.enter_context(
                    patch.object(self.memory, "step", self.observe_step(self.memory.step))
                )
            return self
        except Exception:
            self.context.close()
            raise

    def __exit__(self, *exception):
        return self.context.__exit__(*exception)

    def export(self):
        stages = {
            key: {
                "times": sorted(values),
                "values": torch.stack([values[t] for t in sorted(values)]),
            }
            for key, values in self.tensors.items()
        }
        state = (
            {"available": True, "observations": self.states}
            if self.path == "step"
            else {
                "available": False,
                "reason": "Production full_sequence exposes no explicit state; internal state not reconstructed",
            }
        )
        return {
            "stages": stages,
            "metadata": self.metadata,
            "state": state,
            "layout": "[observation_time, camera_spatial_slot, flattened_feature]",
            "padding_note": "Full Conv1d/act right-padding-only positions excluded from causal-time comparison",
        }


def error_summary(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("Comparison shape mismatch")
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    delta = a - b
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("Nonfinite diagnostic comparison")
    absolute = delta.abs()
    flat = absolute.flatten()
    rms = float(delta.square().mean().sqrt())
    relative = rms / max(float(b.square().mean().sqrt()), 1e-12)
    position = [int(v) for v in np.unravel_index(int(flat.argmax()), a.shape)]
    index = tuple(position)
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    any_difference = bool(torch.any(flat != 0))
    first = (
        [int(v) for v in np.unravel_index(int((flat != 0).to(torch.uint8).argmax()), a.shape)]
        if any_difference
        else None
    )
    feature = position[-1]
    local = tuple(position[:-1]) + (slice(max(0, feature - 2), feature + 3),)
    return {
        "maxabs": float(flat.max()),
        "rms": rms,
        "relative_rms": relative,
        "abs_quantiles": dict(zip(("p50", "p90", "p95", "p99"), map(float, quantiles))),
        "first_nonzero_position": first,
        "max_position": position,
        "max_actual": float(a[index]),
        "max_reference": float(b[index]),
        "local_actual": a[local].tolist(),
        "local_reference": b[local].tolist(),
    }


def stage_comparisons(step, full, limits):
    rows = []
    for name, reference in full["stages"].items():
        if name not in step["stages"]:
            rows.append(
                {
                    "stage": name,
                    "available": False,
                    "reason": "Stage has no corresponding module call in step",
                }
            )
            continue
        actual = step["stages"][name]
        times = sorted(set(reference["times"]) & set(actual["times"]))
        a = torch.stack([actual["values"][actual["times"].index(t)] for t in times])
        b = torch.stack([reference["values"][reference["times"].index(t)] for t in times])
        per_time = [
            {"time": t, **error_summary(a[i : i + 1], b[i : i + 1])} for i, t in enumerate(times)
        ]
        crossing = [
            r["time"]
            for r in per_time
            if r["maxabs"] > limits["maxabs"] or r["relative_rms"] > limits["relative_rms"]
        ]
        nonzero = [r["time"] for r in per_time if r["first_nonzero_position"] is not None]
        rows.append(
            {
                "stage": name,
                "available": True,
                "times": times,
                "summary": error_summary(a, b),
                "per_time": per_time,
                "first_nonzero_time": min(nonzero) if nonzero else None,
                "first_old_threshold_crossing_time": min(crossing) if crossing else None,
            }
        )
    return rows


def interpretation(bf16, fp32, limits, ratio):
    fp32_pass = (
        fp32["maxabs"] <= limits["maxabs"] and fp32["relative_rms"] <= limits["relative_rms"]
    )
    strong = (
        fp32["rms"] <= bf16["rms"] * ratio
        and fp32["maxabs"] <= bf16["maxabs"] * ratio
        and fp32_pass
    )
    partial = fp32["rms"] < bf16["rms"] or fp32["maxabs"] < bf16["maxabs"]
    return {
        "outcome": "precision_sensitive_strong_evidence"
        if strong
        else ("partial_convergence" if partial else "unresolved"),
        "fp32_latent_old_gate": fp32_pass,
        "rms_ratio": fp32["rms"] / max(bf16["rms"], 1e-30),
        "maxabs_ratio": fp32["maxabs"] / max(bf16["maxabs"], 1e-30),
        "old_failure_unchanged": True,
        "claim_boundary": "Fixed weight/prefix precision sensitivity only; not unique root cause, mathematical golden reference, or deployment/training authorization",
    }
