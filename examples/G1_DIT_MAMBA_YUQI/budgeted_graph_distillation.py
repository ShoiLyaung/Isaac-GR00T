"""Graph KD orchestration with unchanged native sampling/objective and eager teacher."""

from contextlib import ExitStack
from unittest.mock import patch

from budgeted_action_cudagraph import replace_core
from budgeted_action_support import masked_velocity_kd
from contract import tensor_identity
from memory_training_support import fresh
import torch
from torch import nn


class CoreDelegate(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.kwargs = None
        self.calls = 0

    def forward(self, *args, **kwargs):
        if args:
            raise ValueError("Expected native keyword core inputs")
        self.calls += 1
        self.kwargs = dict(kwargs)
        return self.target(**kwargs)


class DecoderDelegate(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.output = None
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        self.output = self.target(*args, **kwargs)
        return self.output


class EncoderDelegate(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.inputs = None
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        self.inputs = args
        return self.target(*args, **kwargs)


def clean_graph_kd(model, teacher, route, frame):
    """One true native draw. No hooks, seed reset, hashes or CPU evidence copies.

    Small delegates retain references needed for KD/receipts. All native module
    bindings restore before teacher execution, so teacher cannot overwrite the
    saved student decoder output. Caller owns AMP, backward, clipping and update.
    """
    head = model.action_head
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Teacher must remain frozen eval")
    core = CoreDelegate(route)
    decoder = DecoderDelegate(head.action_decoder)
    encoder = EncoderDelegate(head.action_encoder)
    original_decoder = head.action_decoder
    sample_time = head.sample_time
    times = []

    def time_delegate(*args, **kwargs):
        value = sample_time(*args, **kwargs)
        times.append(value)
        return value

    with ExitStack() as scope:
        scope.enter_context(patch.object(head, "model", core))
        scope.enter_context(patch.object(head, "action_decoder", decoder))
        scope.enter_context(patch.object(head, "action_encoder", encoder))
        scope.enter_context(patch.object(head, "sample_time", time_delegate))
        result = head(fresh(frame["raw"], model.device), fresh(frame["action_input"], model.device))
        if (core.calls, decoder.calls, encoder.calls, len(times)) != (1, 1, 1, 1):
            raise ValueError("Native KD delegate call cadence changed")
        student_velocity = decoder.output[:, -40:]
        kwargs = {
            k: value.detach() if isinstance(value, torch.Tensor) else value
            for k, value in core.kwargs.items()
        }
        evidence = {
            "continuous_time": times[0].detach(),
            "discrete_time": encoder.inputs[1].detach(),
            "noisy_action": encoder.inputs[0].detach(),
            "post_condition": kwargs["encoder_hidden_states"],
            "state_features": result["state_features"].detach(),
            "target": frame["action_input"]["action"],
            "mask": result["action_mask"].detach(),
            "core_inputs": kwargs,
        }
    with torch.no_grad():
        target = teacher(**kwargs)
        if isinstance(target, tuple):
            target = target[0]
        teacher_velocity = original_decoder(
            target, frame["action_input"]["embodiment_id"].to(model.device)
        )[:, -40:].detach()
    kd = masked_velocity_kd(student_velocity, teacher_velocity, result["action_mask"])
    return {
        "native": result,
        "kd": kd,
        "student_velocity": student_velocity,
        "teacher_velocity": teacher_velocity,
        "evidence": evidence,
    }


def evidence_receipt(values, seed):
    """Invoked only outside clean step timing."""
    data = values["evidence"]
    return {
        "forward_seed": seed,
        **{
            key: tensor_identity(value)
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
        },
        "continuous_time_values": data["continuous_time"].float().cpu().tolist(),
        "discrete_time_values": data["discrete_time"].cpu().tolist(),
        "core_inputs": {
            key: tensor_identity(value) if isinstance(value, torch.Tensor) else value
            for key, value in data["core_inputs"].items()
        },
    }


def eager_receipt(values, frame, seed):
    """Normalize the existing independent eager helper's captured evidence."""
    old = values["receipt"]
    native = values["native"]
    return {
        "forward_seed": seed,
        "continuous_time": old["continuous_time"],
        "discrete_time": old["discrete_time"],
        "noisy_action": old["noisy_action"],
        "post_condition": old["preinjection_post"],
        "state_features": tensor_identity(native["state_features"]),
        "target": tensor_identity(frame["action_input"]["action"]),
        "mask": tensor_identity(native["action_mask"]),
        "continuous_time_values": old["continuous_time_values"],
        "discrete_time_values": old["discrete_time_values"],
        "core_inputs": values["core_input_identity"],
    }


def graph_sample(model, frame, seed, route):
    # The existing F+B graph includes the condition adapter. Thus this parity-only
    # sampling recomputes that same observation condition at each of four Euler
    # steps. It establishes numerical parity, not inference deployment latency.
    from budgeted_action_support import cached_action

    before = route.calls
    with replace_core(model, route):
        value = cached_action(model, frame, seed, student=None)
    if route.calls - before != 4:
        raise ValueError("Graph parity sampling must preserve four Euler calls")
    return value


def dropout_inventory(student):
    rows = {
        name: {
            "p": module.p,
            "training": module.training,
            "effective_dropout": module.p if module.training else 0.0,
        }
        for name, module in student.named_modules()
        if isinstance(module, nn.Dropout)
    }
    if student.training or any(row["effective_dropout"] != 0 for row in rows.values()):
        raise ValueError("Student must use eval-mode effective dropout zero")
    return rows
