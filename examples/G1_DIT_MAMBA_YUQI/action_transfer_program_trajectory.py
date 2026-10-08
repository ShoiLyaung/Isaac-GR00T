"""Observe the existing four-step sampler and query both cores on shared states."""

from contextlib import ExitStack
from unittest.mock import patch

from budgeted_action_support import cached_action, masked_velocity_kd
from contract import tensor_identity, write_new
from gr00t.model.modules.budgeted_action_mamba import ConditionedActionRequest
from memory_numerical_trace import error_summary
import memory_prototype as proto
from memory_training_support import sampling_packet
from runner_core import set_seed
import torch
from torch import nn


class TraceCore(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.kwargs = []

    def forward(self, *args, **kwargs):
        if args:
            raise ValueError("Expected native sampler keyword core call")
        self.kwargs.append(
            {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        )
        return self.target(**kwargs)


class TraceEncoder(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.inputs = []

    def forward(self, *args, **kwargs):
        self.inputs.append(tuple(v.detach() if isinstance(v, torch.Tensor) else v for v in args))
        return self.target(*args, **kwargs)


class TraceDecoder(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.outputs = []

    def forward(self, *args, **kwargs):
        value = self.target(*args, **kwargs)
        self.outputs.append(value.detach())
        return value


def capture_shared_states(model, student, frame, seed, driver):
    """No Euler update is implemented here: all x_t come from the native sampler."""
    if driver not in ("teacher", "student"):
        raise ValueError("Unknown trajectory driver")
    teacher = model.action_head.model
    head = model.action_head
    original_decoder = head.action_decoder
    reference = cached_action(model, frame, seed, student=student if driver == "student" else None)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        packet = sampling_packet(model, frame)
        condition = student.encode_condition(packet["features"].backbone_features)
        target = teacher if driver == "teacher" else ConditionedActionRequest(student, condition)
        core = TraceCore(target)
        encoder = TraceEncoder(head.action_encoder)
        decoder = TraceDecoder(original_decoder)
        with ExitStack() as scope:
            scope.enter_context(patch.object(head, "model", core))
            scope.enter_context(patch.object(head, "action_encoder", encoder))
            scope.enter_context(patch.object(head, "action_decoder", decoder))
            set_seed(seed, cuda=True)
            final = proto.sample_packet(model, packet).detach().cpu()
        if len(core.kwargs) != 4 or len(encoder.inputs) != 4 or len(decoder.outputs) != 4:
            raise ValueError("Original four-step sampler cadence changed")
        if not torch.equal(final, reference):
            raise ValueError("Trajectory observation changed native final action")
        rows = []
        tensor_rows = []
        for index, (kwargs, args, driver_output) in enumerate(
            zip(core.kwargs, encoder.inputs, decoder.outputs)
        ):
            teacher_output = teacher(**kwargs)
            if isinstance(teacher_output, tuple):
                teacher_output = teacher_output[0]
            teacher_velocity = original_decoder(teacher_output, packet["action"].embodiment_id)[
                :, -40:
            ]
            student_output = student(**kwargs, condition_cache=condition)
            if isinstance(student_output, tuple):
                student_output = student_output[0]
            student_velocity = original_decoder(student_output, packet["action"].embodiment_id)[
                :, -40:
            ]
            mask = frame["action_input"]["action_mask"].to(model.device)
            kd = masked_velocity_kd(student_velocity, teacher_velocity, mask)
            actual_driver = driver_output[:, -40:]
            queried_driver = teacher_velocity if driver == "teacher" else student_velocity
            replay_exact = torch.equal(actual_driver, queried_driver)
            replay_error = error_summary(queried_driver[:, :16], actual_driver[:, :16])
            bucket = int(args[1].detach().cpu()[0])
            if bucket != index * 250:
                raise ValueError("Original Euler time buckets changed")
            tensors = {
                "x_t": args[0].detach().cpu(),
                "time_bucket": args[1].detach().cpu(),
                "driver_velocity": actual_driver.detach().cpu(),
                "teacher_velocity": teacher_velocity.detach().cpu(),
                "student_velocity": student_velocity.detach().cpu(),
                "action_mask": mask.detach().cpu(),
            }
            if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
                raise ValueError("Nonfinite trajectory evidence")
            rows.append(
                {
                    "step": index,
                    "time_bucket": bucket,
                    "shared_state_velocity_kd": float(kd),
                    "core_kwargs_identity": {
                        k: tensor_identity(v) if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items()
                    },
                    "x_t_identity": tensor_identity(args[0]),
                    "driver_velocity_replay_exact": replay_exact,
                    "driver_velocity_replay_error": replay_error,
                }
            )
            tensor_rows.append(tensors)
    return (
        {
            "driver": driver,
            "seed": seed,
            "final_action_exact": True,
            "steps": rows,
            "scope": "Teacher/student velocities queried on the same actual driver state; separate driver trajectories are not treated as shared states",
        },
        {"final_action": final, "steps": tensor_rows},
    )


def diagnose_probe(context, step, output):
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for case in context.plan["trajectory_cases"]:
        episode, anchor, seed = case["episode"], case["anchor"], case["seed"]
        context.budget()
        frame = context.cache.frame(episode, anchor)
        for driver in context.plan["trajectory_contract"]["driver_paths"]:
            report, tensors = capture_shared_states(
                context.model, context.student, frame, seed, driver
            )
            from contract import sha

            path = output / f"{episode}-{anchor}-{seed}-{driver}.pt"
            torch.save(tensors, path)
            report.update(
                episode=episode,
                anchor=anchor,
                optimizer_step=step,
                tensor_path=str(path),
                tensor_sha256=sha(path),
            )
            reports.append(report)
    result = {
        "optimizer_step": step,
        "cases": reports,
        "scope": "Fixed-checkpoint diagnostic only; no trajectory optimizer loss or sampler rewrite",
    }
    write_new(output / "RESULT.json", result)
    return result
