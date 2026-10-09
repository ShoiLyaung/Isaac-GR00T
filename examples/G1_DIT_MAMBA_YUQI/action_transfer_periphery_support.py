"""Private native-periphery views, live-input graphs and composite continuation state."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import math
from types import SimpleNamespace
from unittest.mock import patch

from action_transfer_matched_support import StrictBatchDispatcher
from action_transfer_program_support import optimizer_names
from budgeted_action_cudagraph import TENSOR_KEYS, BackwardGraph, CoreDispatcher, TensorCore
from budgeted_action_cudagraph_probe import group_mapping
from budgeted_graph_distillation import CoreDelegate, DecoderDelegate, EncoderDelegate
from contract import tensor_identity
from memory_training import optimizer_for
from memory_training_support import fresh
import torch
from torch import nn
from torch.func import functional_call


CONDITIONS = ("core_only_continuation", "joint_periphery")
PERIPHERY = (
    "vlln",
    "vl_self_attention",
    "state_encoder",
    "action_encoder",
    "action_decoder",
    "position_embedding",
)
PARENT_SEMANTIC = "567b48c8ddc18909ef9bf228920778a503fe0d943b910deb7434bb470aaf61a1"


class BF16View(nn.Module):
    """FP32 optimizer ownership with original differentiable BF16 parameter semantics.

    Casting the native bias as well as its weight is essential: a FP32 bias in
    CategorySpecificLinear promotes its BF16 bmm result and changes B0 outputs.
    No cast is cached across forwards or optimizer updates.
    """

    def __init__(self, master):
        super().__init__()
        self.master = master

    def forward(self, *args, **kwargs):
        views = {name: p.to(torch.bfloat16) for name, p in self.master.named_parameters()}
        views.update(
            {
                name: b.to(torch.bfloat16) if b.is_floating_point() else b
                for name, b in self.master.named_buffers()
            }
        )
        return functional_call(self.master, views, args, kwargs, strict=True)


class NativeStudent(nn.Module):
    """Canonical checkpoint names are core.* and periphery.<original-name>.*."""

    def __init__(self, teacher_head, core, condition):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("Unknown periphery adaptation condition")
        # Never deepcopy the 1.09B native core, and never share a decoder/module.
        head = deepcopy(teacher_head, {id(teacher_head.model): nn.Identity()})
        self.core = core.float().eval().requires_grad_(True)
        self.periphery = nn.ModuleDict()
        for name in PERIPHERY:
            if hasattr(head, name):
                module = getattr(head, name).float().eval()
                module.requires_grad_(condition == "joint_periphery")
                self.periphery[name] = module
                setattr(head, name, BF16View(module))
        # head is only a native algorithm shell, not another checkpoint owner.
        head.model = self.core
        head.eval()
        object.__setattr__(self, "head", head)
        self.condition = condition
        self.eval()

    def model_view(self):
        return SimpleNamespace(action_head=self.head, device=next(self.core.parameters()).device)

    @contextmanager
    def materialized(self):
        """Deployment uses native BF16 modules; transient copies are never saved."""
        with ExitStack() as scope:
            for name, master in self.periphery.items():
                module = deepcopy(master).to(torch.bfloat16).eval().requires_grad_(False)
                scope.enter_context(patch.object(self.head, name, module))
            yield self.model_view()


def live_tensor_args(kwargs):
    """Capture leaf samples with the live graph's gradient flags, never receipts."""
    return tuple(
        kwargs[key].detach().clone().requires_grad_(kwargs[key].requires_grad)
        for key in TENSOR_KEYS
    )


def task_forward(student, route, frame, *, retain_inputs=False):
    head = student.head
    core = CoreDelegate(route)
    decoder = DecoderDelegate(head.action_decoder)
    encoder = EncoderDelegate(head.action_encoder)
    original_time = head.sample_time
    times = []

    def sample_time(*args, **kwargs):
        value = original_time(*args, **kwargs)
        times.append(value)
        return value

    with ExitStack() as scope:
        scope.enter_context(patch.object(head, "model", core))
        scope.enter_context(patch.object(head, "action_decoder", decoder))
        scope.enter_context(patch.object(head, "action_encoder", encoder))
        scope.enter_context(patch.object(head, "sample_time", sample_time))
        device = next(student.core.parameters()).device
        result = head(fresh(frame["raw"], device), fresh(frame["action_input"], device))
    if (core.calls, decoder.calls, encoder.calls, len(times)) != (1, 1, 1, 1):
        raise ValueError("Native task call cadence changed")
    inputs = {key: core.kwargs[key] for key in TENSOR_KEYS}
    if retain_inputs:
        for value in inputs.values():
            if value.requires_grad:
                value.retain_grad()
    evidence = {
        "continuous_time": times[0].detach(),
        "discrete_time": encoder.inputs[1].detach(),
        "noisy_action": encoder.inputs[0].detach(),
        "target": frame["action_input"]["action"],
        "mask": result["action_mask"].detach(),
    }
    return {
        "native": result,
        "velocity": decoder.output[:, -40:],
        "live_inputs": inputs,
        "evidence": evidence,
    }


def receipt(values, seed):
    return {"forward_seed": seed, **{k: tensor_identity(v) for k, v in values["evidence"].items()}}


def eager_route(student):
    return CoreDispatcher(TensorCore(student.core, student=True, backward=True))


def capture_graph(student, frame):
    # Gradients MUST be enabled here: joint hidden/VL inputs are differentiable.
    if not torch.is_grad_enabled():
        raise ValueError("Live graph capture requires autograd enabled")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = task_forward(student, eager_route(student), frame)
    samples = live_tensor_args(values["live_inputs"])
    del values
    expected = student.condition == "joint_periphery"
    if any(samples[i].requires_grad != expected for i in (0, 1)):
        raise ValueError("Live hidden/VL input gradient path is missing")
    surface = TensorCore(student.core, student=True, backward=True)
    lease = BackwardGraph(surface, samples)
    return lease, StrictBatchDispatcher(surface, lease, samples), samples


def adaptation_lr(step):
    if not 50001 <= step <= 70000:
        raise ValueError("Adaptation LR is defined only for steps50001..70000")
    return 1e-6 + 9e-6 * (1 + math.cos(math.pi * (step - 50001) / 19999)) / 2


def make_optimizer(student, plan, parent_payload):
    optimizer = optimizer_for(student.core, plan)
    if optimizer_names(student.core, optimizer) != parent_payload["optimizer_parameter_names"]:
        raise ValueError("Parent ordered core optimizer inventory differs")
    optimizer.load_state_dict(parent_payload["optimizer"])
    if student.condition == "joint_periphery":
        for decay in (True, False):
            params = [
                p
                for name, p in student.periphery.named_parameters()
                if (
                    p.ndim >= 2
                    and name.rsplit(".", 1)[-1] not in ("b", "bias")
                    and not getattr(p, "_no_weight_decay", False)
                )
                == decay
            ]
            if params:
                optimizer.add_param_group(
                    {
                        "params": params,
                        "weight_decay": plan["optimizer"][
                            "matrix_weight_decay" if decay else "bias_norm_no_weight_decay"
                        ],
                    }
                )
    optimizer_steps(student, optimizer, 0)
    return optimizer


def optimizer_steps(student, optimizer, added_steps):
    counts = {"core": 0, "periphery": 0}
    for name, parameter in student.named_parameters():
        part = name.split(".")[0]
        expected = 40000 + added_steps if part == "core" else added_steps
        state = optimizer.state.get(parameter)
        needs_state = part == "core" or (student.condition == "joint_periphery" and added_steps > 0)
        if not needs_state:
            if state:
                raise ValueError(f"Unexpected optimizer state: {name}")
            continue
        if not state or int(state["step"]) != expected:
            raise ValueError(f"AdamW no-reset step mismatch: {name}")
        if any(not bool(torch.isfinite(state[k]).all()) for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError(f"Nonfinite AdamW state: {name}")
        counts[part] += 1
    return {"counts": counts, "core_step": 40000 + added_steps, "periphery_step": added_steps}


def gradient_mapping(student):
    result = {"core." + k: "core." + v for k, v in group_mapping(student.core, "student").items()}
    result.update(
        {
            "periphery." + k: "periphery." + k.split(".")[0]
            for k, _ in student.periphery.named_parameters()
        }
    )
    return result


def gradient_norms(student):
    groups = {"core": list(student.core.parameters())}
    groups.update({"periphery." + k: list(v.parameters()) for k, v in student.periphery.items()})
    result = {}
    for name, parameters in groups.items():
        gradients = [p.grad.detach() for p in parameters if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError(f"Nonfinite gradients: {name}")
        result[name] = math.sqrt(sum(float(g.float().square().sum()) for g in gradients))
    return result


def storage_inventory(student, teacher):
    def inventory(module):
        return {
            name: {
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "requires_grad": p.requires_grad,
                "parameters": p.numel(),
                "device": str(p.device),
                "storage_ptr": p.untyped_storage().data_ptr(),
                "storage_bytes": p.untyped_storage().nbytes(),
            }
            for name, p in module.named_parameters()
        }

    s, t = inventory(student), inventory(teacher)
    teacher_ranges = {
        (v["device"], v["storage_ptr"], v["storage_ptr"] + v["storage_bytes"]) for v in t.values()
    }
    for name, v in s.items():
        lo, hi = v["storage_ptr"], v["storage_ptr"] + v["storage_bytes"]
        if any(
            v["device"] == device and max(lo, a) < min(hi, b) for device, a, b in teacher_ranges
        ):
            raise ValueError(f"Student aliases frozen teacher: {name}")
    core = sum(p.numel() for p in student.core.parameters())
    periphery = sum(p.numel() for p in student.periphery.parameters())
    return {
        "student": s,
        "teacher": t,
        "disjoint": True,
        "core_parameters": core,
        "periphery_parameters": periphery,
        "final_checkpoint_payload_bytes": 12 * core
        + (12 if student.condition == "joint_periphery" else 4) * periphery,
        "two_condition_payload_bytes": 24 * core + 16 * periphery,
    }
