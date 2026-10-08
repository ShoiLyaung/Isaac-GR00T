"""One zero-update fixed-weight action execution diagnosis, with no adoption retry."""

import argparse
from copy import deepcopy
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from budgeted_action_profile import ExecutionScopes, profiler_summary
from budgeted_action_support import bind_request, clean_task_loss, current_policy
from contract import sha, write_new
from fp32_pilot_support import PilotCache
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
from memory_numerical_trace import error_summary
import memory_prototype as proto
from memory_training import raw_record
import numpy as np
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-budgeted-mamba-execution-diagnosis-r1"
)
VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "cb845d84e7e6a02707a86a634d4589e80450ea1877b18903f8090a7e87cb9eb7"
EXECUTION_RUN = "joint-diagnosis-gpu0-r1"
MODES = ("native", "student_original", "student_bf16_storage")
SENSITIVE_SUFFIXES = (".mamba.A_log", ".mamba.D", ".mamba.dt_bias")
FILES = {
    "examples/G1_DIT_MAMBA_YUQI/budgeted_action_execution_probe.py",
    "examples/G1_DIT_MAMBA_YUQI/budgeted_action_profile.py",
    "examples/G1_DIT_MAMBA_YUQI/README-execution-diagnosis.md",
    "tests/examples/test_budgeted_execution.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def derived_bf16(original):
    """Independent storage intervention; the original master's bytes are untouched."""
    before = proto.content_hash(original.state_dict())
    candidate = deepcopy(original)
    preserved = {
        name: value.detach().cpu().clone()
        for name, value in original.named_parameters()
        if name.endswith(SENSITIVE_SUFFIXES)
    }
    if len(preserved) != 24 or any(value.dtype != torch.float32 for value in preserved.values()):
        raise ValueError("Expected24 original FP32 A_log/D/dt_bias parameters")
    candidate.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for name, value in preserved.items():
            candidate.get_parameter(name).data = value.clone()
    for name, value in candidate.named_parameters():
        expected = original.get_parameter(name).detach().cpu()
        expected = expected if name in preserved else expected.to(torch.bfloat16)
        if value.dtype != expected.dtype or not torch.equal(value.detach().cpu(), expected):
            raise ValueError("Derived storage cast violated its per-parameter contract")
    if proto.content_hash(original.state_dict()) != before:
        raise ValueError("Casting derived storage mutated original master weights")
    return candidate


def module_identity(module):
    return {
        "state_content_sha256": proto.content_hash(module.state_dict()),
        "parameters": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "requires_grad": value.requires_grad,
            }
            for name, value in module.named_parameters()
        },
        "buffers": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in module.named_buffers()
        },
        "parameter_count": sum(p.numel() for p in module.parameters()),
    }


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen execution diagnosis PLAN changed")
    plan = read(args.plan)
    if args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve() or REPO != Path(
        plan["source_worktree"]
    ):
        raise ValueError("Frozen diagnostic source/artifact root mismatch")
    if tuple(plan["modes"]) != MODES or plan["decision_limits"]["training_authorized"]:
        raise ValueError("Only the frozen zero-training three-mode diagnosis is allowed")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen source/input changed: {path}")
    for relative in plan["immutable_source_relative_paths"]:
        expected = {
            row["sha256"] for row in plan["records"] if row["path"].endswith("/" + relative)
        }
        if len(expected) != 1 or sha(REPO / relative) not in expected:
            raise ValueError(f"Frozen existing implementation changed: {relative}")
    return plan


def environment(plan):
    import budgeted_action_profile
    import budgeted_action_support
    import gr00t
    from gr00t.model.gr00t_n1d7 import gr00t_n1d7
    from gr00t.model.modules import budgeted_action_mamba
    import mamba_ssm.modules.mamba2 as official
    import triton.language.core as core

    modules = (
        gr00t,
        gr00t_n1d7,
        budgeted_action_mamba,
        budgeted_action_support,
        budgeted_action_profile,
        proto,
    )
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("Import escaped the execution-diagnosis worktree")
    if (
        sys.version_info[:2] != (3, 10)
        or Path(sys.prefix) != VENV
        or torch.__version__.split("+")[0] != "2.7.1"
    ):
        raise ValueError("Original Python3.10/torch2.7.1 environment required")
    kernels = [official, core] + [
        importlib.import_module("mamba_ssm.ops.triton." + name)
        for name in (
            "ssd_combined",
            "ssd_chunk_scan",
            "ssd_chunk_state",
            "selective_state_update",
            "layernorm_gated",
        )
    ]
    frozen = {Path(row["path"]).resolve(): row["sha256"] for row in plan["records"]}
    if any(
        Path(m.__file__).resolve() not in frozen
        or sha(m.__file__) != frozen[Path(m.__file__).resolve()]
        for m in kernels
    ):
        raise ValueError("Official kernel source differs from frozen inputs")
    if (
        importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
        or official.causal_conv1d_fn is not None
        or os.environ.get("TRITON_F32_DEFAULT") != "ieee"
    ):
        raise ValueError("Original kernel/precision environment changed")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Skipping base weights is forbidden")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(FILES):
        raise ValueError("Only committed new diagnostic files are permitted")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != str(plan["physical_gpu"])
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose only the single physical GPU frozen in this execution plan")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "imports": {m.__name__: m.__file__ for m in (*modules, *kernels)},
        "python": sys.version,
        "torch": torch.__version__,
        "mamba": importlib.metadata.version("mamba-ssm"),
        "triton": importlib.metadata.version("triton"),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "pythonpath": os.environ.get("PYTHONPATH"),
        "gpu_name": torch.cuda.get_device_name(0),
    }


def context_for(args, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    plan = verify_plan(args)
    source = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    context = SimpleNamespace(args=args, plan=plan, deadline=deadline, stage_peak=0)

    def budget():
        context.stage_peak = max(context.stage_peak, torch.cuda.max_memory_allocated())
        if time.monotonic() > deadline:
            raise TimeoutError("Execution diagnosis wall budget exceeded")
        total = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if total > plan["max_artifact_bytes"] or context.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Execution diagnosis storage/allocated hard limit exceeded")
        return total

    context.budget = budget
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    context.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    proto.assert_frozen(model)
    context.teacher = model.action_head.model
    context.base_before = proto.content_hash(model.state_dict())
    if context.base_before != plan["base_content_sha256"]:
        raise ValueError("Frozen base content mismatch")
    initial = torch.load(
        plan["student_initialization"]["path"], map_location="cpu", weights_only=True
    )
    if proto.content_hash(initial) != plan["student_initialization"]["content_sha256"]:
        raise ValueError("Frozen original student content mismatch")
    original = BudgetedActionMamba(context.teacher)
    original.load_state_dict(initial, strict=True)
    original.eval().requires_grad_(True)
    context.students = {
        "student_original": original,
        "student_bf16_storage": derived_bf16(original).eval().requires_grad_(True),
    }
    del initial
    context.source = NativeAnchors(Path(plan["prepared"]))
    context.source.processor.eval()
    context.cache = PilotCache(
        plan["cache_root"], plan, expected_report_sha=plan["cache_report_sha256"]
    )
    if context.cache.report["integrity"]["base_before"] != context.base_before:
        raise ValueError("Frozen cache/base identity mismatch")
    context.observations = {}
    context.raw = {}
    for row in plan["cost_inputs"]:
        ep, anchor = row["episode"], row["anchor"]
        batch, raw = context.source.batch([(ep, anchor)])
        observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
        proto.validate_input_layout(observation, context.source.modality["video"].modality_keys)
        context.observations[ep] = observation
        context.raw[ep] = raw[0]
        context.cache.frame(ep, anchor)
    context.initial_identities = {
        "native": module_identity(context.teacher),
        **{name: module_identity(core) for name, core in context.students.items()},
    }
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "source": source,
            "strict_loading": loading,
            "base_content_sha256": context.base_before,
            "initial_modes": context.initial_identities,
            "student_original_file_sha256": sha(plan["student_initialization"]["path"]),
            "cast_semantics": "Derived floating storage cast; only24 A_log/D/dt_bias retain exact original FP32. Derived weights are not bitwise-identical to checkpoint.",
            "optimizer_steps": 0,
        },
    )
    context.budget()
    return context


def activate(context, mode, *, backward=False):
    if context.model.action_head.model is not context.teacher:
        raise ValueError("Request binding leaked")
    context.budget()
    for core in (context.teacher, *context.students.values()):
        core.zero_grad(set_to_none=True)
        core.cpu()
    context.teacher.requires_grad_(False).eval()
    for core in context.students.values():
        core.eval().requires_grad_(True)
    gc.collect()
    torch.cuda.empty_cache()
    core = context.teacher if mode == "native" else context.students[mode]
    if mode == "native":
        core.to(device="cuda", dtype=torch.float32 if backward else torch.bfloat16).requires_grad_(
            backward
        )
    else:
        core.cuda()
    torch.cuda.synchronize()
    context.budget()
    return core


def restore_inference_native(context):
    context.teacher.zero_grad(set_to_none=True)
    context.teacher.to(dtype=torch.bfloat16).requires_grad_(False).eval()


def action_value(context, predicted, episode):
    value, identity = proto.action_record(predicted, context.raw[episode], context.source.processor)
    return value, identity


def summary(rows, key="wall_ms"):
    measured = [row[key] for row in rows if not row["warmup"]]
    if not measured or not np.isfinite(measured).all() or min(measured) <= 0:
        raise ValueError("Invalid positive finite timing series")
    return {
        "count": len(measured),
        "mean_ms": float(np.mean(measured)),
        "p50_ms": float(np.percentile(measured, 50)),
        "p95_ms": float(np.percentile(measured, 95)),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
        "peak_reserved_bytes": max(row.get("peak_reserved_bytes", 0) for row in rows),
    }


def order_for(index):
    shift = index % len(MODES)
    return MODES[shift:] + MODES[:shift]


def reproduce_and_numerical(context):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from run_train_eval import physical_metrics

    output = context.args.output / "numerical"
    output.mkdir(exist_ok=False)
    reference_rows = []
    measurements = {}
    tensors = {}
    # Both original paths must reproduce old output before the derived intervention.
    for row in context.plan["cost_inputs"]:
        ep = row["episode"]
        observation = context.observations[ep]
        for mode in ("native", "student_original"):
            core = activate(context, mode)
            set_seed(42, cuda=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = current_policy(
                    context.model, observation, None if mode == "native" else core
                )
            actual, identity = action_value(context, predicted, ep)
            expected = torch.load(
                context.plan["reference_actions"][str(ep)][mode],
                map_location="cpu",
                weights_only=True,
            )[0]
            if not torch.equal(actual, expected):
                raise ValueError(
                    "Original checkpoint execution did not exactly reproduce old seed42 action"
                )
            reference_rows.append(
                {"episode": ep, "mode": mode, "old_action_exact": True, **identity}
            )
            del predicted
    write_new(output / "OLD_ACTION_PARITY.json", reference_rows)
    for row in context.plan["cost_inputs"]:
        ep, anchor = row["episode"], row["anchor"]
        frame = context.cache.frame(ep, anchor)
        measurements[ep] = {}
        tensors[ep] = {}
        for mode in MODES:
            core = activate(context, mode)
            measurements[ep][mode] = {}
            tensors[ep][mode] = {}
            for seed in context.plan["numerical_seeds"]:
                context.budget()
                set_seed(seed, cuda=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = current_policy(
                        context.model, context.observations[ep], None if mode == "native" else core
                    )
                value, identity = action_value(context, predicted, ep)
                set_seed(seed, cuda=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = clean_task_loss(
                        context.model, frame, student=None if mode == "native" else core
                    )
                metrics = physical_metrics(
                    value[0].float().numpy(), raw_record(frame), context.source.processor
                )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite original native flow objective")
                measurements[ep][mode][seed] = {"flow_loss": float(loss), **metrics, **identity}
                tensors[ep][mode][seed] = value
                del predicted, loss
        torch.save(tensors[ep], output / f"{ep}-actions.pt")
    pairs = []
    for row in context.plan["cost_inputs"]:
        ep = row["episode"]
        raw = context.raw[ep]
        for seed in context.plan["numerical_seeds"]:
            original = tensors[ep]["student_original"][seed]
            candidate = tensors[ep]["student_bf16_storage"][seed]
            errors = error_summary(candidate[:, :16], original[:, :16])
            physical = []
            for value in (original, candidate):
                decoded = context.source.processor.decode_action(
                    value[0].float().numpy(), EmbodimentTag.NEW_EMBODIMENT, state=raw.states
                )
                physical.append(np.concatenate([decoded[k] for k in ("left_arm", "left_hand")], -1))
            delta = np.abs(physical[1] - physical[0])
            pairs.append(
                {
                    "episode": ep,
                    "anchor": row["anchor"],
                    "seed": seed,
                    "normalized_action_error": errors,
                    "numeric_gate": all(
                        errors[k] <= v for k, v in context.plan["action_limits"].items()
                    ),
                    "physical_delta": {
                        "combined_mae_mixed_units": float(delta.mean()),
                        "arm_mae_rad": float(delta[:, :7].mean()),
                        "hand_mae_u": float(delta[:, 7:].mean()),
                    },
                    "metrics": {mode: measurements[ep][mode][seed] for mode in MODES},
                }
            )
    report = {
        "old_parity_rows": len(reference_rows),
        "paired_seeds": len(pairs),
        "numeric_gate": all(row["numeric_gate"] for row in pairs),
        "pairs": pairs,
        "scope": "Derived storage-only precision intervention; native loss is original head.forward flow objective, not imitation loss; false numerical gate forbids adoption but does not suppress remaining diagnosis",
    }
    write_new(output / "RESULT.json", report)
    return report


def clean_cost(context):
    output = context.args.output / "clean"
    output.mkdir(exist_ok=False)
    results = {}
    first_actions = {}
    timing = context.plan["timing"]
    for index, row in enumerate(context.plan["cost_inputs"]):
        ep = row["episode"]
        results[ep] = {}
        first_actions[ep] = {}
        for mode in order_for(index):
            core = activate(context, mode)
            rows = []
            actions = []
            with (output / f"{ep}-{mode}.jsonl").open("x") as stream:
                for iteration in range(timing["clean_warmup"] + timing["clean_measured"]):
                    context.budget()
                    set_seed(context.plan["cost_seed"], cuda=True)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        predicted = current_policy(
                            context.model,
                            context.observations[ep],
                            None if mode == "native" else core,
                        )
                        torch.cuda.synchronize()
                        wall = (time.perf_counter() - tick) * 1000
                    allocated, reserved = (
                        torch.cuda.max_memory_allocated(),
                        torch.cuda.max_memory_reserved(),
                    )
                    value, identity = action_value(context, predicted, ep)
                    if actions and not torch.equal(value, actions[0]):
                        raise ValueError("Clean repeated action is not bitwise stable")
                    actions.append(value)
                    record = {
                        "episode": ep,
                        "anchor": row["anchor"],
                        "mode": mode,
                        "iteration": iteration,
                        "warmup": iteration < timing["clean_warmup"],
                        "wall_ms": wall,
                        "peak_allocated_bytes": allocated,
                        "peak_reserved_bytes": reserved,
                        **identity,
                    }
                    rows.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    del predicted, value
            context.budget()
            torch.save(torch.stack(actions), output / f"{ep}-{mode}-actions.pt")
            first_actions[ep][mode] = actions[0]
            results[ep][mode] = summary(rows)
        write_new(
            output / f"{ep}-SUMMARY.json", {"order": list(order_for(index)), "modes": results[ep]}
        )
    return results, first_actions


def segmented_policy(model, observation, student=None):
    """Five sequential non-nested host/CUDA-event ranges, independent of clean timing."""
    names = ("H2D", "VLM", "VL_q", "condition", "Euler4")
    events = {
        name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for name in names
    }
    stages = {}

    def stage(name, function):
        start, end = events[name]
        tick = time.perf_counter()
        start.record()
        result = function()
        end.record()
        end.synchronize()
        stages[name] = {
            "wall_ms": (time.perf_counter() - tick) * 1000,
            "cuda_event_ms": start.elapsed_time(end),
        }
        return result

    torch.cuda.synchronize()
    tick = time.perf_counter()
    backbone_input, action_input = stage("H2D", lambda: model.prepare_input(dict(observation)))
    backbone = stage("VLM", lambda: model.backbone(backbone_input))
    features = stage("VL_q", lambda: model.action_head._encode_features(backbone, action_input))
    if student is None:
        condition = None
        stages["condition"] = {"wall_ms": 0.0, "cuda_event_ms": 0.0}
    else:
        condition = stage("condition", lambda: student.encode_condition(features.backbone_features))
    packet = {"features": features, "backbone": backbone, "action": action_input}

    def sample():
        if student is None:
            return proto.sample_packet(model, packet)
        with bind_request(model, student, condition):
            return proto.sample_packet(model, packet)

    value = stage("Euler4", sample)
    total = (time.perf_counter() - tick) * 1000
    return value, total, stages


def segmented_cost(context, clean, first_actions):
    output = context.args.output / "segmented"
    output.mkdir(exist_ok=False)
    reports = []
    timing = context.plan["timing"]
    for index, item in enumerate(context.plan["cost_inputs"]):
        ep = item["episode"]
        for mode in order_for(index):
            core = activate(context, mode)
            rows = []
            with (output / f"{ep}-{mode}.jsonl").open("x") as stream:
                for iteration in range(timing["segmented_warmup"] + timing["segmented_measured"]):
                    context.budget()
                    set_seed(context.plan["cost_seed"], cuda=True)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        predicted, wall, stages = segmented_policy(
                            context.model,
                            context.observations[ep],
                            None if mode == "native" else core,
                        )
                    value, identity = action_value(context, predicted, ep)
                    if not torch.equal(value, first_actions[ep][mode]):
                        raise ValueError("Segmented action differs from same-mode clean output")
                    row = {
                        "episode": ep,
                        "mode": mode,
                        "iteration": iteration,
                        "warmup": iteration < timing["segmented_warmup"],
                        "wall_ms": wall,
                        "stages": stages,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        **identity,
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    del predicted, value
            totals = summary(rows)
            baseline = clean[ep][mode]
            relative = {key: totals[key] / baseline[key] - 1 for key in ("p50_ms", "p95_ms")}
            valid = all(
                abs(value) <= timing["max_segmented_relative_perturbation"]
                for value in relative.values()
            )
            measured = [r for r in rows if not r["warmup"]]
            means = {
                name: {
                    kind: float(np.mean([r["stages"][name][kind] for r in measured]))
                    for kind in ("wall_ms", "cuda_event_ms")
                }
                for name in measured[0]["stages"]
            }
            shares = {name: row["wall_ms"] / totals["mean_ms"] for name, row in means.items()}
            report = {
                "episode": ep,
                "mode": mode,
                "segmented_total": totals,
                "clean_total": baseline,
                "relative_perturbation": relative,
                "undisturbed_share_interpretation_allowed": valid,
                "mean_stages": means,
                "mean_stage_wall_over_mean_segmented_total": shares,
                "interpretation": "eligible only within this measured input/mode"
                if valid
                else "qualified: synchronization perturbs clean distribution by over5%; no clean deployment-share claim",
                "scope": "Sequential non-nested stage walls and CUDA event intervals; no summing stage p95; event intervals are not pure kernel-busy time; native condition is defined as zero and has no timed CUDA-event/sync range",
            }
            reports.append(report)
            write_new(output / f"{ep}-{mode}-SUMMARY.json", report)
    return reports


def backward_pass(context, mode, frame):
    core = context.teacher if mode == "native" else context.students[mode]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = clean_task_loss(context.model, frame, student=None if mode == "native" else core)
    loss.backward()
    return loss


def check_backward(core, loss):
    if not bool(torch.isfinite(loss)) or any(
        p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in core.parameters()
    ):
        raise ValueError("Nonfinite native-task forward/backward evidence")


def backward_cost(context):
    output = context.args.output / "backward"
    output.mkdir(exist_ok=False)
    item = context.plan["fb_input"]
    frame = context.cache.frame(item["episode"], item["anchor"])
    reports = {}
    timing = context.plan["timing"]
    for mode in ("native", "student_original"):
        core = activate(context, mode, backward=True)
        rows = []
        with (output / f"{mode}.jsonl").open("x") as stream:
            for iteration in range(timing["fb_warmup"] + timing["fb_measured"]):
                context.budget()
                core.zero_grad(set_to_none=True)
                set_seed(context.plan["cost_seed"], cuda=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                loss = backward_pass(context, mode, frame)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - tick) * 1000
                allocated, reserved = (
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                check_backward(core, loss)
                row = {
                    "mode": mode,
                    "iteration": iteration,
                    "warmup": iteration < timing["fb_warmup"],
                    "wall_ms": elapsed,
                    "native_loss": float(loss.detach()),
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "optimizer_steps": 0,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                del loss
        core.zero_grad(set_to_none=True)
        context.budget()
        reports[mode] = summary(rows)
        restore_inference_native(context)
    write_new(
        output / "RESULT.json",
        {
            "modes": reports,
            "scope": "FP32 action-core masters, eval/no-dropout, native task loss F+B outside-autocast backward, AMPBF16 forward; cached rawVL H2D/VL_q/action/decoder and student condition included; no VLM/teacher replay/optimizer/hooks/hash inside timer",
        },
    )
    return reports


def profile_calls(context, first_actions):
    import mamba_ssm.modules.mamba2 as official
    from torch.profiler import ProfilerActivity, profile, record_function

    output = context.args.output / "profiles"
    output.mkdir(exist_ok=False)
    item = context.plan["profiling"]["input"]
    ep = item["episode"]
    frame = context.cache.frame(ep, item["anchor"])
    reports = []
    for kind, modes in (("policy", MODES), ("fb", ("native", "student_original"))):
        for mode in modes:
            core = activate(context, mode, backward=kind == "fb")
            if kind == "policy":
                expected = first_actions[ep][mode]
            else:
                core.zero_grad(set_to_none=True)
                set_seed(context.plan["cost_seed"], cuda=True)
                expected = backward_pass(context, mode, frame).detach().cpu()
                core.zero_grad(set_to_none=True)
            set_seed(context.plan["cost_seed"], cuda=True)
            with ExecutionScopes(
                context.model, core, student=mode != "native", official_module=official
            ) as scopes:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                ) as observed:
                    if kind == "policy":
                        with (
                            record_function("execution/profiled_policy"),
                            torch.no_grad(),
                            torch.autocast("cuda", dtype=torch.bfloat16),
                        ):
                            value = current_policy(
                                context.model,
                                context.observations[ep],
                                None if mode == "native" else core,
                            )
                    else:
                        with (
                            record_function("execution/profiled_forward"),
                            torch.autocast("cuda", dtype=torch.bfloat16),
                        ):
                            value = clean_task_loss(
                                context.model, frame, student=None if mode == "native" else core
                            )
                        with record_function("execution/profiled_backward"):
                            value.backward()
                    torch.cuda.synchronize()
            actual = value.detach().cpu()
            if not torch.equal(actual, expected):
                raise ValueError("Profiler/scopes changed the corresponding clean output")
            if kind == "fb":
                check_backward(core, value)
            del value
            trace = output / f"{kind}-{mode}.json"
            observed.export_chrome_trace(str(trace))
            context.budget()
            evidence = profiler_summary(observed, scopes)
            core.zero_grad(set_to_none=True)
            set_seed(context.plan["cost_seed"], cuda=True)
            if kind == "policy":
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    after = current_policy(
                        context.model, context.observations[ep], None if mode == "native" else core
                    )
            else:
                after = backward_pass(context, mode, frame)
            if not torch.equal(after.detach().cpu(), expected):
                raise ValueError("Uninstrumented output changed after profiler cleanup")
            del after
            core.zero_grad(set_to_none=True)
            if any(scopes.active.values()):
                raise ValueError("Profile scope remained active")
            report = {
                "kind": kind,
                "mode": mode,
                "input": item,
                "trace_path": str(trace),
                "trace_bytes": trace.stat().st_size,
                "trace_sha256": sha(trace),
                "profiled_output_exact": True,
                "unhooked_output_exact": True,
                "optimizer_steps": 0,
                "output_comparand": "full normalized1x40x13 action"
                if kind == "policy"
                else "native task loss scalar; gradients finite, not a gradient-equality claim",
                **evidence,
            }
            reports.append(report)
            write_new(output / f"{kind}-{mode}-SUMMARY.json", report)
            del observed, scopes, actual, expected
            gc.collect()
            restore_inference_native(context)
            context.budget()
    return reports


def decisions(context, numerical, clean):
    rows = []
    limits = context.plan["decision_limits"]
    for item in context.plan["cost_inputs"]:
        ep = item["episode"]
        m = clean[ep]
        derived = m["student_bf16_storage"]
        original = m["student_original"]
        native = m["native"]
        rows.append(
            {
                "episode": ep,
                "cast_p95_ratio_vs_original": derived["p95_ms"] / original["p95_ms"],
                "cast_p95_ratio_vs_native": derived["p95_ms"] / native["p95_ms"],
                "cast_10pct_faster_than_original": derived["p95_ms"]
                <= original["p95_ms"] * limits["cast_max_p95_ratio_vs_original"],
                "cast_original_native_speed_gate": derived["p95_ms"]
                <= native["p95_ms"] * limits["cast_max_p95_ratio_vs_native"],
                "cast_original_native_allocated_gate": derived["peak_allocated_bytes"]
                <= native["peak_allocated_bytes"] * limits["cast_max_allocated_ratio_vs_native"],
            }
        )
    return {
        "per_input": rows,
        "dtype_numeric_gate": numerical["numeric_gate"],
        "cast_10pct_faster_all_inputs": all(r["cast_10pct_faster_than_original"] for r in rows),
        "cast_native_cost_gate_all_inputs": all(
            r["cast_original_native_speed_gate"] and r["cast_original_native_allocated_gate"]
            for r in rows
        ),
        "dtype_adoption_eligible": numerical["numeric_gate"]
        and all(
            r["cast_10pct_faster_than_original"]
            and r["cast_original_native_speed_gate"]
            and r["cast_original_native_allocated_gate"]
            for r in rows
        ),
        "old_training_gate_opened": False,
        "optimizer_steps": 0,
        "scope": "Inference-storage eligibility is reported separately. No outcome authorizes old training, whose original F+B cost gate remains unmet.",
    }


def final_integrity(context):
    restore_inference_native(context)
    if context.model.action_head.model is not context.teacher:
        raise ValueError("Request binding remained installed")
    proto.assert_frozen(context.model)
    after = proto.content_hash(context.model.state_dict())
    if after != context.base_before:
        raise ValueError("Native base changed during diagnosis")
    identities = {
        "native": module_identity(context.teacher),
        **{mode: module_identity(core) for mode, core in context.students.items()},
    }
    for mode in MODES:
        if identities[mode] != context.initial_identities[mode]:
            raise ValueError(f"Fixed mode weights/dtypes/grad flags changed: {mode}")
    original = context.plan["student_initialization"]
    if (
        sha(original["path"]) != original["sha256"]
        or identities["student_original"]["state_content_sha256"] != original["content_sha256"]
    ):
        raise ValueError("Original student checkpoint file/master content changed")
    context.cache.assert_unchanged()
    size = context.budget()
    return {
        "base_before": context.base_before,
        "base_after": after,
        "final_mode_identities": identities,
        "original_checkpoint_file_unchanged": True,
        "original_master_unchanged": True,
        "derived_cast_unchanged": True,
        "cache_unchanged": True,
        "stage_peak_allocated_bytes": context.stage_peak,
        "artifact_bytes": size,
    }


def run(args, deadline):
    context = context_for(args, deadline)
    numerical = reproduce_and_numerical(context)
    clean, first_actions = clean_cost(context)
    segmented = segmented_cost(context, clean, first_actions)
    backward = backward_cost(context)
    profiles = profile_calls(context, first_actions)
    decision = decisions(context, numerical, clean)
    integrity = final_integrity(context)
    return {
        "status": "completed",
        "technical_validity": "passed",
        "measurements_completed": True,
        "plan_sha256": PLAN_SHA,
        "optimizer_steps": 0,
        "numerical": numerical,
        "clean_policy": clean,
        "segmented": segmented,
        "backward": backward,
        "profiles": profiles,
        "decision": decision,
        "integrity": integrity,
        "old_results_unchanged": True,
        "training_authorized": False,
    }


def execute(args):
    if args.output.resolve() != (args.artifact_root / EXECUTION_RUN / "output").resolve():
        raise ValueError("One canonical diagnosis output is required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = run(args, started + args.wall_seconds)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        result["elapsed_seconds"] = time.monotonic() - started
        result["gpu_after_release"] = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        write_new(args.output / "RESULT.json", result)
    except Exception as exc:
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "technical_validity": "not_established",
                "plan_sha256": PLAN_SHA,
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
                "optimizer_steps": 0,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=1750)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1750:
        parser.error("Frozen diagnosis wall bound exceeded")
    execute(args)


if __name__ == "__main__":
    main()
