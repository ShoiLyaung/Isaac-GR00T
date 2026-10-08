"""Fixed teacher/student metrics and isolated cost probes; no run scheduling."""

import gc
import json
import time

from budgeted_action_support import cached_action, capture_velocity, current_policy
from contract import sha, write_new
import memory_prototype as proto
from memory_training import raw_record
import numpy as np
from runner_core import aggregate_metrics, deterministic_seed, set_seed
import torch
from torch import nn


METRICS = (
    "flow_loss",
    "combined_mae_mixed_units",
    "arm_mae_rad",
    "hand_mae_u",
    "teacher_velocity_kd",
    "teacher_action_physical_mae",
    "teacher_action_normalized_mae",
)


def activate(context, mode, *, backward=False):
    """Keep one action core on GPU for cost, with identical frozen outer modules."""
    if context.model.action_head.model is not context.teacher:
        raise ValueError("A previous request leaked its model binding")
    for core in (context.teacher, context.student):
        core.zero_grad(set_to_none=True)
        core.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    if mode == "native":
        context.teacher.to(device="cuda", dtype=torch.float32 if backward else torch.bfloat16)
        context.teacher.requires_grad_(backward).eval()
    else:
        context.student.to(device="cuda", dtype=torch.float32).requires_grad_(True).eval()
    torch.cuda.synchronize()


def restore_teacher(context):
    context.teacher.zero_grad(set_to_none=True)
    context.student.zero_grad(set_to_none=True)
    context.teacher.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False).eval()
    context.student.cuda().float().eval()


def linear_call_audit(context, observation, mode):
    core = context.teacher if mode == "native" else context.student
    linears = []
    calls = {"condition": 0, "mamba": 0, "core": 0}
    shapes = []
    handles = []

    def linear(name):
        def hook(module, args):
            rows = int(np.prod(args[0].shape[:-1]))
            linears.append(
                {
                    "name": name,
                    "rows": rows,
                    "in": module.in_features,
                    "out": module.out_features,
                    "macs": rows * module.in_features * module.out_features,
                }
            )

        return hook

    def count(label):
        def hook(module, args):
            calls[label] += 1
            if label == "mamba":
                shapes.append(list(args[0].shape))

        return hook

    set_seed(42, cuda=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = (
            current_policy(context.model, observation, None if mode == "native" else core)
            .detach()
            .cpu()
        )
    try:
        for name, module in core.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_pre_hook(linear(name)))
        handles.append(core.register_forward_pre_hook(count("core")))
        if mode == "student":
            handles.append(core.condition_adapter.register_forward_pre_hook(count("condition")))
            for block in core.blocks:
                handles.append(block.mixer.mamba.register_forward_pre_hook(count("mamba")))
        set_seed(42, cuda=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            observed = (
                current_policy(context.model, observation, None if mode == "native" else core)
                .detach()
                .cpu()
            )
    finally:
        for handle in handles:
            handle.remove()
    if not torch.equal(clean, observed):
        raise ValueError("Call audit changed fixed-seed policy output")
    expected = (
        {"condition": 1, "mamba": 64, "core": 4}
        if mode == "student"
        else {"condition": 0, "mamba": 0, "core": 4}
    )
    if calls != expected or (mode == "student" and shapes != [[1, 41, 512]] * 64):
        raise ValueError("Actual one-condition/41-token/bidirectional call contract changed")
    return {
        "counts": calls,
        "mamba_shapes": shapes,
        "linear_macs": sum(r["macs"] for r in linears),
        "linear_calls": linears,
        "instrumented_action_exact": True,
        "instrumentation_removed": True,
        "scope": "All action-core nn.Linear only, including one student condition adapter; excludes bias, normalization, attention QK/AV, convolution and scan arithmetic",
    }


def policy_costs(context, cache):
    output = context.args.output / "policy-cost"
    output.mkdir(exist_ok=False)
    reports = []
    audit = {}
    for index, episode in enumerate(context.plan["dev"]):
        anchor = context.plan["grids"][str(episode)][-1]
        batch, raw = context.source.batch([(episode, anchor)])
        observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
        proto.validate_input_layout(observation, context.source.modality["video"].modality_keys)
        order = ("native", "student") if index % 2 == 0 else ("student", "native")
        timings = {}
        for mode in order:
            activate(context, mode)
            if index == 0:
                audit[mode] = linear_call_audit(context, observation, mode)
            rows = []
            actions = []
            with (output / f"{episode}-{mode}.jsonl").open("x") as stream:
                for iteration in range(55):
                    context.budget()
                    set_seed(42, cuda=True)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        predicted = current_policy(
                            context.model,
                            observation,
                            None if mode == "native" else context.student,
                        )
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - tick) * 1000
                    peak = torch.cuda.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    value, identity = proto.action_record(
                        predicted, raw[0], context.source.processor
                    )
                    if actions and not torch.equal(value, actions[0]):
                        raise ValueError("Fixed-seed current policy repeat was not exact")
                    actions.append(value)
                    row = {
                        "episode": episode,
                        "anchor": anchor,
                        "mode": mode,
                        "iteration": iteration,
                        "warmup": iteration < 5,
                        "wall_ms": elapsed,
                        "peak_allocated_bytes": peak,
                        "peak_reserved_bytes": reserved,
                        **identity,
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    del predicted, value
                context.budget()
            torch.save(torch.stack(actions), output / f"{episode}-{mode}-actions.pt")
            timings[mode] = proto.timing_stats(rows)
        native, student = timings["native"], timings["student"]
        gates = {
            "p95_10pct_faster": student["p95_ms"] <= native["p95_ms"] * 0.9,
            "peak_not_above_native": student["peak_allocated_bytes"]
            <= native["peak_allocated_bytes"],
            "absolute_peak": max(native["peak_allocated_bytes"], student["peak_allocated_bytes"])
            <= context.plan["max_allocated_bytes"],
        }
        report = {
            "episode": episode,
            "anchor": anchor,
            "order": list(order),
            "timings": timings,
            "gates": gates,
            "passed": all(gates.values()),
            "p95_relative_difference": student["p95_ms"] / native["p95_ms"] - 1,
            "core_mode": "eval; no dropout",
            "boundary": "prepared CPU input H2D/VLM/VL/state/action/Euler4; includes student condition once; excludes decode/processor/seed/checks",
        }
        reports.append(report)
        write_new(output / f"COST-{episode}.json", report)
    restore_teacher(context)
    write_new(output / "LINEAR_CALL_AUDIT.json", audit)
    return reports, audit


def backward_cost(context, cache):
    """Both cores FP32/AMPBF16, native task loss only; no optimizer or KD replay."""
    output = context.args.output / "backward-cost"
    output.mkdir(exist_ok=False)
    frame = cache.frame(context.plan["fb_input"]["episode"], context.plan["fb_input"]["anchor"])
    reports = {}
    for mode in ("native", "student"):
        activate(context, mode, backward=True)
        core = context.teacher if mode == "native" else context.student
        rows = []
        with (output / f"{mode}.jsonl").open("x") as stream:
            for iteration in range(25):
                context.budget()
                core.zero_grad(set_to_none=True)
                set_seed(42, cuda=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = capture_velocity(
                        context.model, frame, 42, student=None if mode == "native" else core
                    )
                    loss = result["native"]["loss"]
                loss.backward()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - tick) * 1000
                peak = torch.cuda.max_memory_allocated()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in core.parameters()
                ):
                    raise ValueError("Nonfinite task forward/backward cost probe")
                row = {
                    "mode": mode,
                    "iteration": iteration,
                    "warmup": iteration < 5,
                    "wall_ms": wall_ms,
                    "peak_allocated_bytes": peak,
                    "native_loss": float(loss.detach()),
                    "optimizer_steps": 0,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                del result, loss
        core.zero_grad(set_to_none=True)
        values = [r["wall_ms"] for r in rows if not r["warmup"]]
        reports[mode] = {
            "count": 20,
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        }
    restore_teacher(context)
    gates = {
        "p50_10pct_faster": reports["student"]["p50_ms"] <= reports["native"]["p50_ms"] * 0.9,
        "peak_not_above_native": reports["student"]["peak_allocated_bytes"]
        <= reports["native"]["peak_allocated_bytes"],
    }
    report = {
        "timings": reports,
        "gates": gates,
        "passed": all(gates.values()),
        "optimizer_steps": 0,
        "core_mode": "eval, gradient enabled; no dropout; not a complete training step",
        "scope": "Both action cores FP32 master/AMPBF16; original task loss backward outside autocast, frozen outer modules; cached rawVL/state/target through native postprocess/encoder/decoder plus student condition; no VLM/RGB/KD replay/optimizer",
    }
    write_new(output / "RESULT.json", report)
    return report


def evaluate_student(context, cache, episode, output):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from run_train_eval import physical_metrics

    output.mkdir(exist_ok=False)
    context.student.eval()
    rows = []
    actions = []
    teacher_actions = []
    for anchor in context.plan["eval_anchors"][str(episode)]:
        frame = cache.frame(episode, anchor)
        for nominal in context.plan["noise_seeds"]:
            context.budget()
            seed = deterministic_seed("diagnostic", episode, anchor, nominal)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                values = capture_velocity(
                    context.model,
                    frame,
                    seed,
                    student=context.student,
                    teacher=context.teacher,
                    replay_teacher=True,
                )
            actual = cached_action(context.model, frame, seed, student=context.student)
            teacher = cached_action(context.model, frame, seed)
            _, identity = proto.action_record(actual, raw_record(frame), context.source.processor)
            prediction = actual[0].float().numpy()
            target_prediction = teacher[0].float().numpy()
            metrics = physical_metrics(prediction, raw_record(frame), context.source.processor)
            decoded = context.source.processor.decode_action(
                prediction, EmbodimentTag.NEW_EMBODIMENT, state=raw_record(frame).states
            )
            decoded_teacher = context.source.processor.decode_action(
                target_prediction, EmbodimentTag.NEW_EMBODIMENT, state=raw_record(frame).states
            )
            physical = np.concatenate([decoded[k] for k in ("left_arm", "left_hand")], -1)
            physical_teacher = np.concatenate(
                [decoded_teacher[k] for k in ("left_arm", "left_hand")], -1
            )
            row = {
                "episode": episode,
                "anchor": anchor,
                "seed": nominal,
                "paired_rng_seed": seed,
                "flow_loss": float(values["native"]["loss"]),
                "teacher_velocity_kd": float(values["kd"]),
                "teacher_action_physical_mae": float(np.abs(physical - physical_teacher).mean()),
                "teacher_action_normalized_mae": float(
                    np.abs(prediction[:16] - target_prediction[:16]).mean()
                ),
                **metrics,
                "noise_receipt": values["receipt"],
                **identity,
            }
            if not all(np.isfinite(row[k]) for k in METRICS):
                raise ValueError("Nonfinite teacher/student metric")
            rows.append(row)
            actions.append(prediction)
            teacher_actions.append(target_prediction)
    if len(rows) != 64:
        raise ValueError("Expected16 anchors x4 seeds for one-episode evidence")
    per_episode, metrics = aggregate_metrics(rows, METRICS)
    write_new(output / "details.json", rows)
    np.save(output / "student_actions.npy", np.stack(actions))
    np.save(output / "teacher_actions.npy", np.stack(teacher_actions))
    report = {
        "status": "passed",
        "rows": 64,
        "metrics": metrics,
        "per_episode": per_episode,
        "student_actions_sha256": sha(output / "student_actions.npy"),
        "teacher_actions_sha256": sha(output / "teacher_actions.npy"),
        "scope": "same-noise native Euler4 sampled action; physical mixed-unit activeH16x13 teacher MAE is the overfit gate metric",
    }
    write_new(output / "RESULT.json", report)
    return report
