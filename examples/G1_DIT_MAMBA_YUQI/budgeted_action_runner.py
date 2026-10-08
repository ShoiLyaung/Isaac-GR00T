"""Budgeted action-Mamba cost/preflight and bounded fresh KD overfit.

The first reviewed source milestone intentionally does not implement the two
1000-step pilot commands. Their CLI names fail before outputs or model loads.
"""

import argparse
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

from budgeted_action_evaluation import backward_cost, evaluate_student, policy_costs
from budgeted_action_support import (
    cached_action,
    capture_velocity,
    gradient_groups,
    parameter_changes,
    transfer_audit,
)
from contract import sha, write_new
from fp32_memory_pilot import cache_native_parity
from fp32_pilot_support import PilotCache
from gr00t.model.modules.budgeted_action_mamba import (
    EXPECTED_LINEAR_MACS,
    EXPECTED_PARAMETERS,
    BudgetedActionMamba,
)
import memory_prototype as proto
from memory_training import optimizer_for, raw_record
from memory_training_support import native_flow
import numpy as np
from runner_core import deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-budgeted-mamba-action-distillation-r1"
)
VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "2562f74a0cc0475577a76a622a685cfa2f4b04bf8bc306c3a0a4ff6394736f99"
FILES = {
    "gr00t/model/modules/budgeted_action_mamba.py",
    "examples/G1_DIT_MAMBA_YUQI/budgeted_action_runner.py",
    "examples/G1_DIT_MAMBA_YUQI/budgeted_action_support.py",
    "examples/G1_DIT_MAMBA_YUQI/budgeted_action_evaluation.py",
    "examples/G1_DIT_MAMBA_YUQI/README-budgeted-action.md",
    "tests/examples/test_budgeted_action.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen action-distillation PLAN changed")
    plan = read(args.plan)
    if args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve() or REPO != Path(
        plan["source_worktree"]
    ):
        raise ValueError("Frozen source/artifact root identity mismatch")
    if plan["confirmation_allowed"] or set(plan["train"] + plan["dev"]) != {
        125,
        141,
        76,
        132,
        203,
        9,
        99,
        120,
        150,
        147,
        152,
        112,
    }:
        raise ValueError("Only the frozen train8/dev4 are permitted")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
    for relative in plan["immutable_source_relative_paths"]:
        matches = {row["sha256"] for row in plan["records"] if row["path"].endswith("/" + relative)}
        if len(matches) != 1 or sha(REPO / relative) not in matches:
            raise ValueError(f"Native source/helper changed: {relative}")
    for stage in ("overfit", "pilot"):
        if sha(plan[stage + "_schedule_file"]) != plan["schedule_sha256"][stage]:
            raise ValueError("Frozen optimizer schedule identity changed")
    for row in plan["native_parity_inputs"]:
        if (
            deterministic_seed("diagnostic", row["episode"], row["anchor"], row["nominal_seed"])
            != row["actual_seed"]
        ):
            raise ValueError("Native parity seed mapping changed")
    return plan


def environment(plan):
    import budgeted_action_evaluation
    import budgeted_action_support
    import gr00t
    from gr00t.model.gr00t_n1d7 import gr00t_n1d7
    from gr00t.model.modules import budgeted_action_mamba
    import mamba_ssm.modules.mamba2 as official
    import triton.language.core as triton_core

    modules = (
        gr00t,
        gr00t_n1d7,
        budgeted_action_mamba,
        budgeted_action_support,
        budgeted_action_evaluation,
        proto,
    )
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("A source import escaped the budgeted-action worktree")
    if (
        sys.version_info[:2] != (3, 10)
        or Path(sys.prefix) != VENV
        or torch.__version__.split("+")[0] != "2.7.1"
    ):
        raise ValueError("Original Python3.10/torch2.7.1 environment required")
    kernels = [official, triton_core] + [
        importlib.import_module("mamba_ssm.ops.triton." + name)
        for name in (
            "ssd_combined",
            "ssd_chunk_scan",
            "ssd_chunk_state",
            "selective_state_update",
            "layernorm_gated",
        )
    ]
    frozen = {Path(r["path"]).resolve(): r["sha256"] for r in plan["records"]}
    if any(
        Path(m.__file__).resolve() not in frozen
        or sha(m.__file__) != frozen[Path(m.__file__).resolve()]
        for m in kernels
    ):
        raise ValueError("Active official kernel source changed")
    if (
        importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
        or official.causal_conv1d_fn is not None
        or os.environ.get("TRITON_F32_DEFAULT") != "ieee"
    ):
        raise ValueError("Frozen Mamba/no-causal-conv/IEEE default contract changed")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Pretrained weights cannot be skipped")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(FILES):
        raise ValueError("Expected clean scoped action-student source")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose only authorized physicalGPU1")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "imports": {m.__name__: m.__file__ for m in (*modules, *kernels)},
        "python": sys.version,
        "torch": torch.__version__,
        "mamba": importlib.metadata.version("mamba-ssm"),
        "triton": importlib.metadata.version("triton"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
    }


def require_preflight(args):
    root = args.artifact_root / "preflight-r1/output"
    result = read(root / "RESULT.json")
    if (
        result.get("status") != "passed"
        or not result.get("continuation_gate")
        or result.get("plan_sha256") != PLAN_SHA
    ):
        raise ValueError("Passing matching preflight required before optimizer step0")
    identity = result["initialization"]
    if sha(identity["path"]) != identity["sha256"]:
        raise ValueError("Frozen preflight initialization changed")
    return identity


def context_for(args, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    plan = verify_plan(args)
    initialization = require_preflight(args) if args.command == "overfit" else None
    source = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    context = SimpleNamespace(args=args, plan=plan, deadline=deadline, stage_peak_allocated_bytes=0)

    def budget():
        context.stage_peak_allocated_bytes = max(
            context.stage_peak_allocated_bytes, torch.cuda.max_memory_allocated()
        )
        if time.monotonic() > deadline:
            raise TimeoutError("Budgeted action stage exceeded frozen wall bound")
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if (
            size > plan["max_artifact_bytes"]
            or context.stage_peak_allocated_bytes > plan["max_allocated_bytes"]
        ):
            raise RuntimeError("Budgeted action storage/allocated-memory hard bound exceeded")
        return size

    context.budget = budget
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    context.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    proto.assert_frozen(model)
    context.teacher = model.action_head.model
    context.base_before = proto.content_hash(model.state_dict())
    if context.base_before != plan["base_content_sha256"]:
        raise ValueError("Loaded teacher differs from frozen B0 base")
    context.student = BudgetedActionMamba(context.teacher)
    context.transfer = transfer_audit(context.student, context.teacher)
    context.initial = {k: v.detach().cpu().clone() for k, v in context.student.state_dict().items()}
    initial_hash = proto.content_hash(context.initial)
    if initialization is not None:
        values = torch.load(initialization["path"], map_location="cpu", weights_only=True)
        if (
            proto.content_hash(values) != initialization["content_sha256"]
            or initial_hash != initialization["content_sha256"]
        ):
            raise ValueError(
                "Overfit must begin from fresh common initialization, never warm-start"
            )
        context.student.load_state_dict(values, strict=True)
    context.student.cuda().float().eval()
    context.source = NativeAnchors(Path(plan["prepared"]))
    context.source.processor.eval()
    context.cache = PilotCache(
        plan["cache_root"], plan, expected_report_sha=plan["cache_report_sha256"]
    )
    if context.cache.report["integrity"]["base_before"] != context.base_before:
        raise ValueError("Cache/base weight identity mismatch")
    for ep in plan["train"] + plan["dev"]:
        context.cache.episode(ep)
    context.budget()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "environment": source,
            "strict_loading": loading,
            "base_before": context.base_before,
            "student_initial_content_sha256": initial_hash,
            "transfer": context.transfer,
            "fresh_only": True,
            "optimizer_resume": False,
            "initialization": initialization,
        },
    )
    return context


def integrity(context):
    if context.model.action_head.model is not context.teacher:
        raise ValueError("Student request binding leaked into base")
    proto.assert_frozen(context.model)
    base_after = proto.content_hash(context.model.state_dict())
    if base_after != context.base_before:
        raise ValueError("Frozen pretrained base content changed")
    context.cache.assert_unchanged()
    if any(p.dtype != torch.float32 for p in context.student.parameters()):
        raise ValueError("Student master precision changed")
    return {
        "base_before": context.base_before,
        "base_after": base_after,
        "base_frozen_eval_no_grad": True,
        "cache_unchanged": True,
        "student_fp32_master": True,
        "artifact_bytes": context.budget(),
        "peak_allocated_bytes": context.stage_peak_allocated_bytes,
        "peak_scope": "stage high-water collected before timing resets and at budget boundaries",
    }


def historical_parity(context):
    root = Path(context.plan["native_reference_root"])
    details = read(root / "details.json")
    actions = np.load(root / "normalized_predictions.npy")
    index = {(r["episode"], r["anchor"], r["seed"]): i for i, r in enumerate(details)}
    rows = []
    for row in context.plan["native_parity_inputs"]:
        context.budget()
        ep, a, s = row["episode"], row["anchor"], row["nominal_seed"]
        i = index[(ep, a, s)]
        frame = context.cache.frame(ep, a)
        actual = cached_action(context.model, frame, row["actual_seed"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            flow, receipt = native_flow(context.model, frame, None, row["actual_seed"])
        if (
            not np.array_equal(actual[0].float().numpy(), actions[i])
            or float(flow["loss"]) != details[i]["flow_loss"]
        ):
            raise ValueError("Historical native action/metric exact parity failed")
        rows.append({**row, "action_exact": True, "flow_exact": True, "noise_receipt": receipt})
    if len(rows) != 64:
        raise ValueError("Expected64 fixed historical native comparisons")
    write_new(context.args.output / "NATIVE_HISTORICAL_PARITY.json", rows)
    return rows


def condition_contract(context):
    from budgeted_action_support import bind_request
    from memory_training_support import sampling_packet

    frame = context.cache.frame(
        context.plan["fb_input"]["episode"], context.plan["fb_input"]["anchor"]
    )
    context.student.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        packet = sampling_packet(context.model, frame)
        features = packet["features"].backbone_features
        condition = context.student.encode_condition(features)
        snapshots = (condition.code.clone(), condition.modulation.clone())
        observed = []

        def input_hook(_module, args, kwargs):
            observed.append(
                (
                    kwargs["hidden_states"].clone(),
                    kwargs["timestep"].clone(),
                    kwargs["condition_cache"],
                )
            )

        handle = context.student.register_forward_pre_hook(input_hook, with_kwargs=True)
        try:
            with bind_request(context.model, context.student, condition):
                set_seed(42, cuda=True)
                predicted = proto.sample_packet(context.model, packet)
        finally:
            handle.remove()
        if len(observed) != 4 or any(cache is not condition for _, _, cache in observed):
            raise ValueError("One immutable observation condition must span exactly4 Euler calls")
        again = context.student.encode_condition(features)
        for tick, delta in ((0, 0.0), (750, 0.01)):
            context.student(
                hidden_states=observed[0][0] + delta,
                encoder_hidden_states=features,
                timestep=torch.tensor([tick], device="cuda"),
                condition_cache=condition,
            )
        changed = features.clone()
        changed[:, 4:68] = features[:, 70:134]
        swapped = context.student.encode_condition(changed)
    if not torch.equal(condition.code, again.code) or not torch.equal(
        condition.modulation, again.modulation
    ):
        raise ValueError("Same observation failed exact condition repeat")
    if not all(
        torch.equal(a, b) for a, b in zip(snapshots, (condition.code, condition.modulation))
    ):
        raise ValueError("Denoising mutated observation condition")
    if torch.equal(condition.code, swapped.code):
        raise ValueError("Camera-position intervention did not affect condition code")
    actual = cached_action(context.model, frame, 42, student=context.student)
    changed_action = cached_action(
        context.model, frame, 42, student=context.student, modulation_zero=True
    )
    if not torch.equal(predicted.detach().cpu(), actual) or torch.equal(actual, changed_action):
        raise ValueError("Request restoration or condition-action sensitivity failed")
    proto.action_record(actual, raw_record(frame), context.source.processor)
    report = {
        "passed": True,
        "euler_calls": 4,
        "same_request_cache_object": True,
        "condition_independent_of_time_noisy_action": True,
        "camera_swap_changes_code": True,
        "zero_modulation_changes_action": True,
        "request_restored_exact": True,
    }
    write_new(context.args.output / "CONDITION_CONTRACT.json", report)
    torch.save(
        {
            "condition_code": condition.code.cpu(),
            "condition_modulation": condition.modulation.cpu(),
            "normal_action": actual,
            "zero_modulation_action": changed_action,
        },
        context.args.output / "condition-evidence.pt",
    )
    return report


def zero_step_gradients(context):
    frame = context.cache.frame(
        context.plan["fb_input"]["episode"], context.plan["fb_input"]["anchor"]
    )
    context.student.train()
    context.student.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = capture_velocity(
            context.model,
            frame,
            42,
            student=context.student,
            teacher=context.teacher,
            replay_teacher=True,
        )
        objective = result["kd"]
    objective.backward()
    groups = gradient_groups(context.student)
    passed = all(row["finite_nonzero_parameters"] > 0 for row in groups.values())
    if any(p.grad is not None for p in context.teacher.parameters()):
        raise ValueError("Teacher received gradients")
    report = {
        "passed": passed,
        "objective": "native-mask teacher velocity KD",
        "optimizer_steps": 0,
        "kd": float(objective.detach()),
        "native_loss": float(result["native"]["loss"].detach()),
        "groups": groups,
        "teacher_detached": not result["teacher_velocity"].requires_grad,
        "receipt": result["receipt"],
        "core_input_identity": result["core_input_identity"],
    }
    context.student.zero_grad(set_to_none=True)
    context.student.eval()
    write_new(context.args.output / "GRADIENT_PREFLIGHT.json", report)
    return report


def preflight(context):
    before = proto.content_hash(context.student.state_dict())
    initialization = context.args.output / "student-initialization.pt"
    torch.save(context.initial, initialization)
    init = {"path": str(initialization), "sha256": sha(initialization), "content_sha256": before}
    parity = cache_native_parity(
        context, context.cache, context.plan["train"] + context.plan["dev"]
    )
    write_new(context.args.output / "CACHE_NATIVE_PARITY.json", parity)
    historical = historical_parity(context)
    condition = condition_contract(context)
    gradients = zero_step_gradients(context)
    # False cost gates aggregate after all preregistered measurements; exceptions stop.
    policy, audit = policy_costs(context, context.cache)
    fb = backward_cost(context, context.cache)
    student_count = sum(p.numel() for p in context.student.parameters())
    native_count = sum(p.numel() for p in context.teacher.parameters())
    limits = context.plan["gates"]["cost"]
    static = {
        "student_parameters": student_count,
        "native_parameters": native_count,
        "parameter_ratio": student_count / native_count,
        "student_linear_macs": audit["student"]["linear_macs"],
        "native_linear_macs": audit["native"]["linear_macs"],
        "linear_mac_ratio": audit["student"]["linear_macs"] / audit["native"]["linear_macs"],
    }
    if (
        student_count != EXPECTED_PARAMETERS
        or audit["student"]["linear_macs"] != EXPECTED_LINEAR_MACS
    ):
        raise ValueError(
            "Actual student parameter/linear-work inventory differs from frozen architecture"
        )
    gates = {
        "parameters": static["parameter_ratio"] <= limits["max_parameter_ratio"],
        "linear_macs": static["student_linear_macs"] <= limits["max_linear_macs"]
        and static["linear_mac_ratio"] <= limits["max_linear_mac_ratio"],
        "policy_cost": all(r["passed"] for r in policy),
        "backward_cost": fb["passed"],
        "zero_step_gradients": gradients["passed"],
        "condition_contract": condition["passed"],
        "native_parity": len(historical) == 64 and len(parity) == 12,
    }
    after = proto.content_hash(context.student.state_dict())
    if before != after:
        raise ValueError("Zero-step preflight changed student initialization")
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "continuation_gate": all(gates.values()),
        "optimizer_steps": 0,
        "gates": gates,
        "initialization": init,
        "transfer": context.transfer,
        "static_cost": static,
        "policy_cost": policy,
        "backward_cost": fb,
        "gradient_preflight": gradients,
        "condition_contract": condition,
        "integrity": integrity(context),
    }


def overfit(context):
    before = evaluate_student(context, context.cache, 125, context.args.output / "step0")
    schedule = read(context.plan["overfit_schedule_file"])
    if len(schedule) != 200:
        raise ValueError("Expected exactly200 frozen overfit steps")
    optimizer = optimizer_for(context.student, context.plan)
    seen = {}
    completed = 0
    with (context.args.output / "training_steps.jsonl").open("x") as stream:
        for planned in schedule:
            context.budget()
            if planned["step"] != completed + 1 or planned["episode"] != 125:
                raise ValueError("Overfit schedule changed")
            for group in optimizer.param_groups:
                group["lr"] = planned["lr"]
            context.student.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = capture_velocity(
                    context.model,
                    context.cache.frame(125, planned["anchor"]),
                    planned["forward_seed"],
                    student=context.student,
                    teacher=context.teacher,
                    replay_teacher=True,
                )
                loss = result["kd"]
            loss.backward()
            groups = gradient_groups(context.student)
            for name, value in groups.items():
                seen[name] = seen.get(name, False) or value["finite_nonzero_parameters"] > 0
            norm = torch.nn.utils.clip_grad_norm_(
                context.student.parameters(),
                context.plan["optimizer"]["gradient_clip"],
                error_if_nonfinite=True,
            )
            proto.assert_frozen(context.model)
            optimizer.step()
            completed += 1
            row = {
                **planned,
                "optimizer_steps_completed": completed,
                "objective": "teacher_velocity_kd",
                "teacher_velocity_kd": float(loss.detach()),
                "native_loss": float(result["native"]["loss"].detach()),
                "gradient_norm_before_clip": float(norm),
                "gradient_groups": groups,
                "noise_receipt": result["receipt"],
                "core_input_identity": result["core_input_identity"],
                "elapsed_step_seconds": time.monotonic() - tick,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            del result, loss
            if completed % 25 == 0:
                print(json.dumps({"overfit_steps": completed}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    context.student.eval()
    checkpoint = context.args.output / "student-step200.pt"
    torch.save({k: v.detach().cpu() for k, v in context.student.state_dict().items()}, checkpoint)
    changes = parameter_changes(context.student, context.initial, seen)
    after = evaluate_student(context, context.cache, 125, context.args.output / "step200")
    condition = condition_contract(context)
    gate = {
        "kd_reduction_30pct": after["metrics"]["teacher_velocity_kd"]
        <= before["metrics"]["teacher_velocity_kd"] * 0.7,
        "teacher_physical_action_mae_reduction_10pct": after["metrics"][
            "teacher_action_physical_mae"
        ]
        <= before["metrics"]["teacher_action_physical_mae"] * 0.9,
        "gradient_parameter_changes": changes["passed"],
        "condition_sensitivity": condition["passed"],
    }
    return {
        "status": "passed" if all(gate.values()) else "failed",
        "continuation_gate": all(gate.values()),
        "optimizer_steps": completed,
        "gates": gate,
        "step0": before,
        "step200": after,
        "changes": changes,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha(checkpoint),
            "content_sha256": proto.content_hash(context.student.state_dict()),
        },
        "integrity": integrity(context),
        "pilot_implemented_in_this_commit": False,
    }


def execute(args):
    if args.command not in ("preflight", "overfit"):
        raise NotImplementedError(
            "This reviewed milestone implements preflight/overfit only; no pilot is launched"
        )
    expected = args.artifact_root / (args.command + "-r1") / "output"
    if args.output.resolve() != expected.resolve():
        raise ValueError("One canonical output root per stage required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        context = context_for(args, started + args.wall_seconds)
        result = preflight(context) if args.command == "preflight" else overfit(context)
        result.update(
            plan_sha256=PLAN_SHA,
            elapsed_seconds=time.monotonic() - started,
            automatic_retry=False,
            old_results_unchanged=True,
        )
        del context
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
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
                "continuation_gate": False,
                "plan_sha256": PLAN_SHA,
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
                "optimizer_resume_authorized": False,
            },
        )
        raise
    if not result["continuation_gate"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "overfit", "train", "evaluate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--condition", choices=("task_only", "distill_then_task"))
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1750 and args.command in ("preflight", "overfit"):
        parser.error("Frozen30-minute stage bound exceeded")
    execute(args)


if __name__ == "__main__":
    main()
