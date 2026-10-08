"""Phase A: one long single-episode data-line KD attempt plus fixed trajectory probes."""

import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

from action_transfer_program_support import (
    PLAN_SHA,
    checkpoint,
    load_resume,
    read,
    restore_rng,
    restore_training,
    validate_run_id,
)
from action_transfer_program_trajectory import diagnose_probe
from budgeted_action_support import parameter_changes
from budgeted_graph_distillation import dropout_inventory
from budgeted_graph_distillation_runner import (
    close_graph,
    create_graph,
    environment,
    evaluate,
    final_integrity,
    graph_step,
    probes,
)
from contract import sha, write_new
from fp32_pilot_support import PilotCache
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
import memory_prototype as proto
from memory_training import optimizer_for
import numpy as np
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen Phase A PLAN changed")
    plan = read(args.plan)
    validate_run_id(args, plan)
    if (
        REPO != Path(plan["source_worktree"])
        or args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve()
    ):
        raise ValueError("Frozen source/artifact identity mismatch")
    if plan["training_steps"] != 5000 or plan["quality_failure_stops_training"]:
        raise ValueError("Expected explicit5000-step non-stopping quality protocol")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen source/input changed: {path}")
    for relative in plan["immutable_source_relative_paths"]:
        expected = {r["sha256"] for r in plan["records"] if r["path"].endswith("/" + relative)}
        if len(expected) != 1 or sha(REPO / relative) not in expected:
            raise ValueError(f"Existing implementation changed: {relative}")
    if sha(plan["overfit_schedule_file"]) != plan["schedule_sha256"]["overfit"]:
        raise ValueError("Long schedule identity mismatch")
    prior = read(plan["compatibility_prerequisite"])
    if (
        sha(plan["compatibility_prerequisite"]) != plan["compatibility_prerequisite_sha256"]
        or prior["status"] != "passed"
        or not prior["continuation_gate"]
    ):
        raise ValueError("Passed graph-update compatibility required")
    return plan


def context_for(args, plan, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    source = environment(plan)
    parent, state = load_resume(args, plan, source["commit"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    ctx = SimpleNamespace(
        args=args,
        plan=plan,
        source_commit=source["commit"],
        deadline=deadline,
        stage_peak=0,
        lease=None,
        optimizer_steps_completed=0,
    )

    def budget():
        ctx.stage_peak = max(ctx.stage_peak, torch.cuda.max_memory_allocated())
        total = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if time.monotonic() > deadline:
            raise TimeoutError("Phase A attempt wall bound exceeded")
        if total > plan["max_artifact_bytes"] or ctx.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Phase A resource bound exceeded")
        return total

    ctx.budget = budget
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    ctx.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    ctx.teacher = model.action_head.model
    ctx.base_before = proto.content_hash(model.state_dict())
    if ctx.base_before != plan["base_content_sha256"]:
        raise ValueError("Frozen B0 changed")
    ctx.initial = torch.load(
        plan["student_initialization"]["path"], map_location="cpu", weights_only=True
    )
    if proto.content_hash(ctx.initial) != plan["student_initialization"]["content_sha256"]:
        raise ValueError("Frozen fresh student initialization changed")
    ctx.student = BudgetedActionMamba(ctx.teacher)
    ctx.student.load_state_dict(ctx.initial, strict=True)
    ctx.student.cuda().float().eval().requires_grad_(True)
    optimizer = optimizer_for(ctx.student, plan)
    pending_rng = None
    if parent is not None:
        pending_rng = restore_training(ctx.student, optimizer, parent, state)
        ctx.optimizer_steps_completed = parent["completed_step"]
    ctx.cache = PilotCache(
        plan["cache_root"], plan, expected_report_sha=plan["cache_report_sha256"]
    )
    ctx.cache.episode(125)
    if ctx.cache.report["integrity"]["base_before"] != ctx.base_before:
        raise ValueError("Frozen cache/base mismatch")
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "source": source,
            "strict_loading": loading,
            "run_id": args.run_id,
            "parent_resume_index": None
            if parent is None
            else {"path": str(args.resume_index.resolve()), "sha256": args.resume_index_sha256},
            "starting_step": ctx.optimizer_steps_completed,
            "base_content_sha256": ctx.base_before,
            "starting_student_content_sha256": proto.content_hash(ctx.student.state_dict()),
            "dropout": dropout_inventory(ctx.student),
            "resume_rng_restored_after_capture": parent is not None,
        },
    )
    ctx.budget()
    return ctx, optimizer, parent, pending_rng


def quality_report(baseline, current, limits):
    old, new = baseline["metrics"], current["metrics"]
    return {
        "kd_reduction_30pct": new["teacher_velocity_kd"]
        <= old["teacher_velocity_kd"] * limits["kd_max_final_initial_ratio"],
        "teacher_physical_action_mae_reduction_10pct": new["teacher_action_physical_mae"]
        <= old["teacher_action_physical_mae"]
        * limits["teacher_physical_action_mae_max_final_initial_ratio"],
        "kd_final_initial_ratio": new["teacher_velocity_kd"] / old["teacher_velocity_kd"],
        "teacher_action_final_initial_ratio": new["teacher_action_physical_mae"]
        / old["teacher_action_physical_mae"],
        "scope": "Historical200-step quality predicates reported only; they do not terminate authorized long training",
    }


def old_initial_parity(ctx, output):
    old = Path(ctx.plan["old_step0_reference"])
    old_arrays = read(old / "RESULT.json")["array_sha256"]
    for filename in (
        "student_actions.npy",
        "teacher_actions.npy",
        "student_velocity.npy",
        "teacher_velocity.npy",
        "action_masks.npy",
    ):
        if sha(old / filename) != old_arrays[Path(filename).stem]:
            raise ValueError(f"Anchored old step0 array identity changed: {filename}")
        if not np.array_equal(np.load(output / filename), np.load(old / filename)):
            raise ValueError(f"Fresh old step0 tensor parity failed: {filename}")
    return {"old_step0_reference": str(old), "compared_arrays": 5, "exact": True}


def record_probe(ctx, step, baseline, precomputed=None):
    directory = ctx.args.output / "probes" / f"step{step}"
    directory.mkdir(parents=True, exist_ok=False)
    if precomputed is None:
        quality = evaluate(ctx, directory / "quality")
    else:
        quality = precomputed
    numerical, _ = probes(ctx, step, directory / "graph-parity")
    trajectory = diagnose_probe(ctx, step, directory / "trajectory")
    report = {
        "step": step,
        "quality": quality,
        "quality_directory": str(directory / "quality")
        if precomputed is None
        else str(ctx.args.output / "initial-quality"),
        "historical_quality_predicates": quality_report(
            baseline, quality, ctx.plan["overfit_gates"]
        ),
        "numerical": numerical,
        "trajectory": trajectory,
        "student_content_sha256": proto.content_hash(ctx.student.state_dict()),
        "quality_failure_stops_training": False,
    }
    write_new(directory / "RESULT.json", report)
    if not numerical["passed"]:
        raise ValueError("Graph/eager numerical integrity failed during long training")
    return report


def run(ctx, optimizer, parent, pending_rng):
    schedule = read(ctx.plan["overfit_schedule_file"])
    if len(schedule) != ctx.plan["training_steps"]:
        raise ValueError("Expected frozen5000 schedule rows")
    start = ctx.optimizer_steps_completed
    records = {} if parent is None else parent["probe_records"]
    seen = {} if parent is None else parent["gradient_groups_seen"]
    indexes = []
    if parent is None:
        baseline = evaluate(ctx, ctx.args.output / "initial-quality")
        parity = old_initial_parity(ctx, ctx.args.output / "initial-quality")
        write_new(ctx.args.output / "OLD_INITIAL_PARITY.json", parity)
    else:
        baseline = records["0"]["quality"]
    create_graph(ctx)
    if parent is None:
        records["0"] = record_probe(ctx, 0, baseline, precomputed=baseline)
    else:
        numeric, _ = probes(ctx, start, ctx.args.output / "resume-parity")
        if not numeric["passed"]:
            raise ValueError("Resumed graph/eager numerical integrity failed")
        restore_rng(pending_rng)
        write_new(
            ctx.args.output / "RESUME_RESTORED.json",
            {
                "completed_step": start,
                "next_step": start + 1,
                "rng_restored_after_capture_and_checks": True,
                "optimizer_parameter_state_restored": True,
                "parent_index_sha256": ctx.args.resume_index_sha256,
            },
        )
    with (ctx.args.output / "training_steps.jsonl").open("x") as stream:
        for planned in schedule[start:]:
            ctx.budget()
            if planned["step"] != ctx.optimizer_steps_completed + 1 or planned["episode"] != 125:
                raise ValueError("Original global schedule progression changed")
            row = graph_step(ctx, optimizer, planned)
            for group, value in row["gradient_groups"].items():
                seen[group] = seen.get(group, False) or value["finite_nonzero_parameters"] > 0
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            step = ctx.optimizer_steps_completed
            if step in ctx.plan["probe_steps"]:
                records[str(step)] = record_probe(ctx, step, baseline)
            if step in ctx.plan["weight_checkpoint_steps"]:
                ctx.budget()
                indexes.append(checkpoint(ctx, optimizer, step, records, seen))
                ctx.budget()
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "run_id": ctx.args.run_id,
                            "completed_step": step,
                            "teacher_velocity_kd": row["teacher_velocity_kd"],
                        }
                    ),
                    flush=True,
                )
    optimizer.zero_grad(set_to_none=True)
    changes = parameter_changes(ctx.student, ctx.initial, seen)
    close_graph(ctx)
    return {
        "status": "completed",
        "technical_validity": "passed",
        "training_completed": ctx.optimizer_steps_completed == 5000,
        "optimizer_steps": ctx.optimizer_steps_completed,
        "attempt_optimizer_steps": ctx.optimizer_steps_completed - start,
        "starting_step": start,
        "run_id": ctx.args.run_id,
        "source_commit": ctx.source_commit,
        "probes": records,
        "checkpoints": indexes,
        "changes": changes,
        "final_quality": records["5000"]["historical_quality_predicates"],
        "quality_failure_stops_training": False,
        "integrity": final_integrity(ctx),
        "trajectory_training_started": False,
        "full_vla_training_started": False,
        "scope": "Long single-episode data-line velocity KD and original-sampler diagnostics only; matched multi-episode conditions require their own frozen protocol",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("long-kd",))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=3550)
    parser.add_argument("--resume-index", type=Path)
    parser.add_argument("--resume-index-sha256")
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 3550:
        parser.error("Frozen Phase A wall bound exceeded")
    plan = verify_plan(args)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ctx = None
    completed = 0
    try:
        ctx, optimizer, parent, pending_rng = context_for(args, plan, started + args.wall_seconds)
        result = run(ctx, optimizer, parent, pending_rng)
        result.update(
            plan_sha256=PLAN_SHA, elapsed_seconds=time.monotonic() - started, automatic_retry=False
        )
        completed = ctx.optimizer_steps_completed
        ctx = None
        del optimizer, parent, pending_rng
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
                "technical_validity": "not_established",
                "run_id": args.run_id,
                "plan_sha256": PLAN_SHA,
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "optimizer_steps_completed": completed
                if ctx is None
                else ctx.optimizer_steps_completed,
                "automatic_retry": False,
                "available_checkpoint_indexes": [
                    str(p)
                    for p in sorted((args.output / "checkpoints").glob("checkpoint-step*.json"))
                ],
            },
        )
        raise


if __name__ == "__main__":
    main()
