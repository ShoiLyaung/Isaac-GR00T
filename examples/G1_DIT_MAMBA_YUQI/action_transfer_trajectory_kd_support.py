"""Matched mixed KD queries; only discrete x_t source differs between variants."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from action_transfer_matched_support import batch_forward, masked_batch_kd
from action_transfer_program_trajectory import TraceCore, TraceDecoder, TraceEncoder
from budgeted_action_cudagraph import TENSOR_KEYS
from budgeted_graph_distillation import evidence_receipt
from contract import tensor_identity
import memory_prototype as proto
from memory_training_support import sampling_packet
import torch


VARIANTS = ("data_line_mixed", "teacher_trajectory_mixed")


class CloneTraceCore(TraceCore):
    def forward(self, *args, **kwargs):
        if args:
            raise ValueError("Native teacher core requires keyword inputs")
        self.kwargs.append(
            {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        )
        return self.target(**kwargs)


class CloneTraceEncoder(TraceEncoder):
    def forward(self, *args, **kwargs):
        self.inputs.append(
            tuple(v.detach().clone() if isinstance(v, torch.Tensor) else v for v in args)
        )
        return self.target(*args, **kwargs)


class CloneTraceDecoder(TraceDecoder):
    def forward(self, *args, **kwargs):
        value = self.target(*args, **kwargs)
        self.outputs.append(value.detach().clone())
        return value


@torch.no_grad()
def capture_teacher(model, frame):
    """Original B4 four-Euler sampler, with no noise/time replacement or second integrator."""
    head = model.action_head
    teacher = head.model
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Frozen eval teacher required")
    packet = sampling_packet(model, frame)
    if packet["features"].backbone_features.dtype != torch.float32:
        raise ValueError("Inference post-VL/noise contract requires FP32")
    core, encoder, decoder = (
        CloneTraceCore(teacher),
        CloneTraceEncoder(head.action_encoder),
        CloneTraceDecoder(head.action_decoder),
    )
    with ExitStack() as scope:
        scope.enter_context(patch.object(head, "model", core))
        scope.enter_context(patch.object(head, "action_encoder", encoder))
        scope.enter_context(patch.object(head, "action_decoder", decoder))
        final = proto.sample_packet(model, packet).detach().clone()
    if (len(core.kwargs), len(encoder.inputs), len(decoder.outputs)) != (4, 4, 4):
        raise ValueError("Teacher sampler must retain four Euler calls")
    steps = []
    for index, (kwargs, inputs, output) in enumerate(
        zip(core.kwargs, encoder.inputs, decoder.outputs)
    ):
        x_t, timestep = inputs[:2]
        if (
            x_t.shape != (4, 40, 13)
            or x_t.dtype != torch.float32
            or not torch.equal(timestep, torch.full_like(timestep, index * 250))
        ):
            raise ValueError("Teacher trajectory B4/FP32/current-bucket contract changed")
        velocity = output[:, -40:].detach().clone()
        if not torch.isfinite(x_t).all() or not torch.isfinite(velocity).all():
            raise ValueError("Nonfinite teacher trajectory")
        steps.append({"x_t": x_t, "timestep": timestep, "kwargs": kwargs, "velocity": velocity})
    return {
        "steps": steps,
        "epsilon": steps[0]["x_t"].detach().clone(),
        "final_action": final,
        "state_features": packet["features"].state_features.detach().clone(),
        "post_condition": packet["features"].backbone_features.detach().clone(),
        "embodiment_id": packet["action"].embodiment_id.detach().clone(),
    }


def _requested_shape(args, kwargs):
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(args[0])
    if args:
        return tuple(args)
    return tuple(kwargs["size"])


@contextmanager
def observe_native_noise(expected_shape, expected_dtype):
    """Observe the one original continuous Gaussian draw without replacing its math."""
    original = torch.randn
    record = {"calls": 0}

    def draw(*args, **kwargs):
        record["calls"] += 1
        if (
            record["calls"] != 1
            or _requested_shape(args, kwargs) != tuple(expected_shape)
            or kwargs.get("dtype") != expected_dtype
        ):
            raise ValueError("Unexpected continuous native noise draw")
        value = original(*args, **kwargs)
        record["epsilon"] = value.detach().clone()
        return value

    with patch.object(torch, "randn", draw):
        yield record
    if record["calls"] != 1:
        raise ValueError("Missing continuous native noise draw")


@contextmanager
def discrete_draws(head, epsilon, bucket):
    """Only the scoped native head.forward draw changes; native interpolation remains."""
    if (
        epsilon.shape != (4, 40, 13)
        or epsilon.dtype != torch.float32
        or bucket not in (0, 250, 500, 750)
    ):
        raise ValueError(
            "Discrete source requires full FP32 B4 epsilon and an exact inference bucket"
        )
    counts = {"noise": 0, "time": 0}

    def draw(*args, **kwargs):
        counts["noise"] += 1
        if (
            counts["noise"] != 1
            or _requested_shape(args, kwargs) != tuple(epsilon.shape)
            or kwargs.get("dtype") != torch.float32
            or torch.device(kwargs.get("device")) != epsilon.device
            or set(kwargs) - {"size", "dtype", "device"}
        ):
            raise ValueError("Discrete native noise override shape/dtype/device/count mismatch")
        return epsilon.detach().clone()

    def sample_time(batch_size, *, device, dtype):
        counts["time"] += 1
        if (
            counts["time"] != 1
            or batch_size != 4
            or dtype != torch.float32
            or torch.device(device) != epsilon.device
        ):
            raise ValueError("Discrete native time override shape/dtype/device/count mismatch")
        return torch.full((4,), bucket / 1000, device=device, dtype=torch.float32)

    with ExitStack() as scope:
        scope.enter_context(patch.object(torch, "randn", draw))
        scope.enter_context(patch.object(head, "sample_time", sample_time))
        yield counts
    if counts != {"noise": 1, "time": 1}:
        raise ValueError("Native discrete forward must make exactly one noise and time draw")


def continuous_query(model, teacher, route, frame):
    if frame["action_input"]["action"].dtype != torch.bfloat16:
        raise ValueError("Common native continuous branch must preserve cached BF16 action")
    with observe_native_noise((4, 40, 13), torch.bfloat16) as draw:
        values = batch_forward(model, teacher, route, frame)
    values.update(
        epsilon=draw["epsilon"],
        branch="continuous",
        bucket=None,
        teacher_trace=None,
        native_loss_diagnostic=values["native"]["loss"],
    )
    return values


def discrete_query(model, teacher, route, frame, trace, bucket, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown mixed-KD variant")
    selected = trace["steps"][bucket // 250]
    if variant == "data_line_mixed":
        inputs = {
            **frame,
            "action_input": {
                **frame["action_input"],
                "action": frame["action_input"]["action"].float(),
            },
        }
        with discrete_draws(model.action_head, trace["epsilon"], bucket) as counts:
            values = batch_forward(model, teacher, route, inputs)
        if not torch.equal(
            values["evidence"]["discrete_time"],
            torch.full_like(values["evidence"]["discrete_time"], bucket),
        ):
            raise ValueError("FP32 discrete data-line time rounded to a different bucket")
        native_loss = None  # Native head computes it, but this intervention does not optimize/report it as a task objective.
        values["draw_counts"] = dict(counts)
    else:
        kwargs = selected["kwargs"]
        prediction = route(**kwargs)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = model.action_head.action_decoder(prediction, trace["embodiment_id"])[:, -40:]
        with torch.no_grad():
            target = teacher(**kwargs)
            if isinstance(target, tuple):
                target = target[0]
            target = model.action_head.action_decoder(target, trace["embodiment_id"])[
                :, -40:
            ].detach()
        mask = frame["action_input"]["action_mask"].to(model.device).float()
        evidence = {
            "continuous_time": torch.full(
                (4,), bucket / 1000, device=model.device, dtype=torch.float32
            ),
            "discrete_time": selected["timestep"],
            "noisy_action": selected["x_t"],
            "post_condition": trace["post_condition"],
            "state_features": trace["state_features"],
            "target": frame["action_input"]["action"].float(),
            "mask": mask,
            "core_inputs": kwargs,
        }
        values = {
            "kd": masked_batch_kd(prediction, target, mask),
            "student_velocity": prediction,
            "teacher_velocity": target,
            "evidence": evidence,
            "native": {"action_mask": mask},
            "draw_counts": {"noise": 0, "time": 0},
        }
        native_loss = None
    for key in ("post_condition", "state_features"):
        if not torch.equal(values["evidence"][key], trace[key]):
            raise ValueError("Discrete state sources changed fixed q/Z")
    values.update(
        epsilon=trace["epsilon"],
        branch="discrete",
        bucket=bucket,
        teacher_trace=trace,
        native_loss_diagnostic=native_loss,
    )
    if bucket == 0 and not torch.equal(values["evidence"]["noisy_action"], trace["epsilon"]):
        raise ValueError("t0 selected state must equal original teacher epsilon")
    return values


def mixed_query(model, teacher, route, frame, planned, variant):
    if planned["branch"] == "continuous":
        return continuous_query(model, teacher, route, frame)
    if planned["branch"] != "discrete":
        raise ValueError("Unknown frozen KD distribution branch")
    trace = capture_teacher(model, frame)
    return discrete_query(model, teacher, route, frame, trace, planned["bucket"], variant)


def query_receipts(values, frame, forward_seed):
    evidence = values["evidence"]
    shared = {
        "branch": values["branch"],
        "bucket": values["bucket"],
        "forward_seed": forward_seed,
        "epsilon": tensor_identity(values["epsilon"]),
        "continuous_time": tensor_identity(evidence["continuous_time"]),
        "discrete_time": tensor_identity(evidence["discrete_time"]),
        "post_condition": tensor_identity(evidence["post_condition"]),
        "state_features": tensor_identity(evidence["state_features"]),
        "current_q": tensor_identity(frame["action_input"]["state"]),
        "target_cached": tensor_identity(frame["action_input"]["action"]),
        "target_query": tensor_identity(evidence["target"]),
        "mask": tensor_identity(evidence["mask"]),
    }
    if values["teacher_trace"] is not None:
        shared["teacher_rollout"] = [
            {"x_t": tensor_identity(s["x_t"]), "velocity": tensor_identity(s["velocity"])}
            for s in values["teacher_trace"]["steps"]
        ]
    selected = {
        "x_t": tensor_identity(evidence["noisy_action"]),
        "teacher_velocity": tensor_identity(values["teacher_velocity"]),
        "core_inputs": {k: tensor_identity(evidence["core_inputs"][k]) for k in TENSOR_KEYS},
    }
    return shared, selected, evidence_receipt(values, forward_seed)


def assert_t0_equal(left, right):
    tensors = ("student_velocity", "teacher_velocity", "kd")
    if any(not torch.equal(left[key], right[key]) for key in tensors):
        raise ValueError("Same-weight t0 KD target/output mismatch")
    for key in (
        "noisy_action",
        "continuous_time",
        "discrete_time",
        "post_condition",
        "state_features",
        "target",
        "mask",
    ):
        a, b = left["evidence"][key], right["evidence"][key]
        if a.dtype != b.dtype or not torch.equal(a.to(b.device), b):
            raise ValueError(f"Same-weight t0 input mismatch: {key}")
    for key in TENSOR_KEYS:
        if not torch.equal(
            left["evidence"]["core_inputs"][key], right["evidence"]["core_inputs"][key]
        ):
            raise ValueError(f"Same-weight t0 core input mismatch: {key}")
    return {"passed": True, "same_weight": True, "selected_state_target_student_output_exact": True}
