"""Explicit phase-2 AdamW continuation from frozen matched 20k terminals to 50k."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import action_transfer_matched_runner as matched
from action_transfer_matched_support import (
    MMapCache,
    assemble_batch,
    checkpoint_log_rows,
    log_prefix,
)
from action_transfer_program_support import (
    cpu_tree,
    optimizer_names,
    restore_rng,
    rng_state,
    save_tensor_new,
)
from budgeted_action_support import parameter_changes
from budgeted_graph_distillation import dropout_inventory
from budgeted_graph_distillation_runner import environment
from contract import sha, write_new
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
import memory_prototype as proto
from memory_training import optimizer_for
import numpy as np
from run_train_eval import NativeAnchors, strict_reload
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def continuation_lr(step):
    if not 20001 <= step <= 50000:
        raise ValueError("Continuation LR is defined only for global steps20001..50000")
    u = (step - 20001) / 29999
    return 1e-6 + 9e-6 * (1 + math.cos(math.pi * u)) / 2


def verify_plan(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Explicit continuation PLAN SHA mismatch")
    plan = read(args.plan)
    if (
        Path(plan["source_worktree"]).resolve() != REPO
        or args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve()
        or (plan["starting_step"], plan["training_steps"]) != (20000, 50000)
    ):
        raise ValueError("Continuation source/cohort/budget identity changed")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen continuation material changed: {path}")
    identities = {Path(r["path"]).resolve(): r["sha256"] for r in plan["records"]}
    for relative in plan["immutable_source_relative_paths"]:
        path = (REPO / relative).resolve()
        if identities.get(path) != sha(path):
            raise ValueError(f"Original source changed: {relative}")
    if sha(plan["schedule_file"]) != plan["schedule_sha256"]:
        raise ValueError("Continuation schedule changed")
    schedule = read(plan["schedule_file"])
    if len(schedule) != 50000:
        raise ValueError("Expected complete50000-row schedule")
    for parent in plan["parents"].values():
        if sha(parent["schedule_path"]) != parent["schedule_sha256"] or schedule[:20000] != read(
            parent["schedule_path"]
        ):
            raise ValueError("Historical20000-row schedule prefix changed")
    for row in schedule[20000:]:
        if (
            row["phase"] != 2
            or row["phase_step"] != row["step"] - 10000
            or row["lr"] != continuation_lr(row["step"])
        ):
            raise ValueError("Frozen no-reset phase-2 continuation schedule differs")
    prior = read(plan["prior_preflight"])
    if (
        sha(plan["prior_preflight"]) != plan["prior_preflight_sha256"]
        or not prior["continuation_gate"]
    ):
        raise ValueError("Historical interface preflight evidence changed")
    if (args.resume_index is None) != (args.resume_index_sha256 is None):
        raise ValueError("Explicit resume index and SHA required together")
    expected = plan["run_ids"][args.condition] if args.command == "train" else "joint-evaluation-r1"
    if args.run_id is None:
        args.run_id = expected
    if args.resume_index is None and args.run_id != expected:
        raise ValueError("Fresh continuation run identity changed")
    if args.resume_index is not None and (args.command != "train" or args.run_id == expected):
        raise ValueError("Resume requires a new immutable training attempt")
    if (
        any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_id)
        or args.output.resolve() != args.artifact_root.resolve() / args.run_id / "output"
    ):
        raise ValueError("Continuation output identity changed")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(plan["allowed_source_files"]):
        raise ValueError("Only clean committed continuation source is permitted")
    args.cache_report_sha256 = plan["cache_report_sha256"]
    return plan


def file_record_valid(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"] or (
        "bytes" in record and path.stat().st_size != record["bytes"]
    ):
        raise ValueError("Checkpoint payload SHA/size changed")


def parent_record(plan, condition):
    value = plan["parents"][condition]
    return {"path": value["index_path"], "sha256": value["index_sha256"]}


def expected_identity(plan, condition, *, current_sha=None, source_commit=None):
    parent = plan["parents"][condition]
    return {
        "condition": condition,
        "plan_sha256": parent["plan_sha256"] if current_sha is None else current_sha,
        "source_commit": parent["source_commit"] if source_commit is None else source_commit,
        "cache_report_sha256": plan["cache_report_sha256"],
        "schedule_sha256": parent["schedule_sha256"]
        if current_sha is None
        else plan["schedule_sha256"],
    }


def import_checkpoint(args, plan, source_commit):
    original = plan["parents"][args.condition]
    for path_key, sha_key in (("terminal_path", "terminal_sha256"), ("plan_path", "plan_sha256")):
        if sha(original[path_key]) != original[sha_key]:
            raise ValueError("Frozen original parent changed")
    terminal = read(original["terminal_path"])
    if (
        terminal["condition"] != args.condition
        or not terminal["training_completed"]
        or terminal["optimizer_steps"] != 20000
    ):
        raise ValueError("Must continue the fixed terminal20k, not a selected best checkpoint")
    record = (
        parent_record(plan, args.condition)
        if args.resume_index is None
        else {"path": str(args.resume_index.resolve()), "sha256": args.resume_index_sha256}
    )
    file_record_valid(record)
    index = read(record["path"])
    expected = (
        expected_identity(plan, args.condition)
        if args.resume_index is None
        else expected_identity(
            plan, args.condition, current_sha=args.plan_sha256, source_commit=source_commit
        )
    )
    if any(index.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint source/PLAN/cache/schedule identity changed")
    step = index["completed_step"]
    allowed = [20000] if args.resume_index is None else plan["resume_checkpoint_steps"]
    if (
        step not in allowed
        or step >= 50000
        or index["next_step"] != step + 1
        or index["phase"] != 2
        or index["next_phase_reinitialize_optimizer"]
        or index["run_id"] == args.run_id
    ):
        raise ValueError("Invalid phase-2 restore boundary or optimizer reset")
    for key in ("weights", "optimizer_rng"):
        file_record_valid(index[key])
    payload = torch.load(index["optimizer_rng"]["path"], map_location="cpu", weights_only=True)
    for key in ("completed_step", "phase", "condition", "plan_sha256", "source_commit"):
        if payload[key] != index[key]:
            raise ValueError("Optimizer payload/header mismatch")
    if payload["phase_optimizer_steps"] != step - 10000:
        raise ValueError("Phase-2 optimizer step count changed")
    return record, index, payload


def optimizer_steps(optimizer, expected_count=158, expected_step=None):
    values = [int(state["step"]) for state in optimizer.state.values()]
    if (
        len(values) != expected_count
        or not values
        or (expected_step is not None and set(values) != {expected_step})
    ):
        raise ValueError("AdamW state count or phase-2 step changed")
    for state in optimizer.state.values():
        if any(not bool(torch.isfinite(state[key]).all()) for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Nonfinite restored AdamW moments")
    return {"states": len(values), "min_step": min(values), "max_step": max(values)}


def restore_optimizer(core, optimizer, payload, expected_step, expected_count=158):
    if optimizer_names(core, optimizer) != payload["optimizer_parameter_names"]:
        raise ValueError("Ordered optimizer parameters differ from the parent")
    optimizer.load_state_dict(payload["optimizer"])
    return optimizer_steps(optimizer, expected_count, expected_step)


def context_for(args, plan, deadline):
    source = environment(plan)
    record, index, payload = import_checkpoint(args, plan, source["commit"])
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
        optimizer_steps_completed=index["completed_step"],
        starting_step=index["completed_step"],
        parent_index=record,
        parent_gradient_groups_seen=index["gradient_groups_seen"].copy(),
    )

    def budget():
        ctx.stage_peak = max(ctx.stage_peak, torch.cuda.max_memory_allocated())
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if time.monotonic() > deadline:
            raise TimeoutError("Continuation wall bound exceeded")
        if size > plan["max_artifact_bytes"] or ctx.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Continuation resource bound exceeded")
        return size

    ctx.budget = budget
    ctx.cache = MMapCache(plan["cache_root"], plan, args.cache_report_sha256, budget)
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    ctx.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    ctx.teacher = model.action_head.model
    ctx.base_before = proto.content_hash(model.state_dict())
    if (
        ctx.base_before != plan["base_content_sha256"]
        or ctx.cache.report["integrity"]["base_before"] != ctx.base_before
    ):
        raise ValueError("Frozen base/cache identity mismatch")
    ctx.initial = torch.load(index["weights"]["path"], map_location="cpu", weights_only=True)
    if proto.content_hash(ctx.initial) != index["student_content_sha256"]:
        raise ValueError("Parent student semantic identity changed")
    if any(
        v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in ctx.initial.values()
    ):
        raise ValueError("Parent student requires finite FP32 masters")
    ctx.student = BudgetedActionMamba(ctx.teacher)
    ctx.student.load_state_dict(ctx.initial, strict=True)
    ctx.student.cuda().float().eval().requires_grad_(True)
    optimizer = optimizer_for(ctx.student, plan)
    state_report = restore_optimizer(
        ctx.student, optimizer, payload, index["completed_step"] - 10000
    )
    ctx.optimizer = optimizer
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": args.plan_sha256,
            "source": source,
            "strict_loading": loading,
            "parent_index": record,
            "starting_step": ctx.starting_step,
            "optimizer": state_report,
            "student_content_sha256": index["student_content_sha256"],
            "cache_report_sha256": args.cache_report_sha256,
            "base_content_sha256": ctx.base_before,
            "dropout": dropout_inventory(ctx.student),
            "new_optimizer_reset": False,
        },
    )
    return ctx, index, payload["rng"]


def compare_reproduction(reference, actual):
    file_record_valid(reference)
    file_record_valid(actual)
    left, right = read(reference["path"]), read(actual["path"])
    if left["rows"] != 512 or right["rows"] != 512:
        raise ValueError("Full512 parent development reproduction required")
    for key in (
        "metrics",
        "per_episode",
        "teacher_metrics",
        "teacher_per_episode",
        "checkpoint_content_sha256",
        "cache_report_sha256",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Restored parent development differs: {key}")
    for name in left["array_sha256"]:
        a, b = (
            Path(reference["path"]).parent / f"{name}.npy",
            Path(actual["path"]).parent / f"{name}.npy",
        )
        if sha(a) != left["array_sha256"][name] or sha(b) != right["array_sha256"][name]:
            raise ValueError("Reproduction array identity changed")
        if not np.array_equal(np.load(a), np.load(b)):
            raise ValueError(f"Restored parent array is not exact: {name}")
    for filename, key in (
        ("details.json", "details_sha256"),
        ("NORMALIZATION.json", "normalization_sha256"),
    ):
        a, b = Path(reference["path"]).parent / filename, Path(actual["path"]).parent / filename
        if sha(a) != left[key] or sha(b) != right[key] or read(a) != read(b):
            raise ValueError(f"Restored parent input/metric details differ: {filename}")
    return {
        "passed": True,
        "rows": 512,
        "arrays_exact": list(left["array_sha256"]),
        "metrics_exact": True,
        "all_input_receipts_exact": True,
        "reference": {"path": reference["path"], "sha256": reference["sha256"]},
        "reproduced": {"path": actual["path"], "sha256": actual["sha256"]},
        "tolerance_fallback": False,
    }


def restoration_gate(ctx, index, schedule):
    frame = assemble_batch(ctx.cache, schedule[ctx.starting_step]["batch"])
    matched.create_batch_graph(ctx, frame, schedule[ctx.starting_step]["forward_seed"])
    reproduction = matched.evaluate_pairs(
        ctx, matched.dev_pairs(ctx.plan), ctx.args.output / "restoration-dev"
    )
    reference = index["evaluations"][f"dev-{ctx.starting_step}"]
    report = compare_reproduction(reference, reproduction)
    # Only indices are adapted for the new frozen gate; original loss and
    # all28 gradient-group comparison mathematics remain untouched.
    old_indices = ctx.plan["preflight_batch_indices"]
    ctx.plan["preflight_batch_indices"] = ctx.plan["restoration_batch_indices"]
    try:
        gradients = matched.batch_checks(
            ctx, schedule, ctx.starting_step, ctx.args.output / "restoration-batch"
        )
    finally:
        ctx.plan["preflight_batch_indices"] = old_indices
    if not gradients["passed"]:
        raise ValueError("Restored B4 eager/Graph task and KD gradients differ")
    if ctx.optimizer_steps_completed != ctx.starting_step:
        raise ValueError("Restoration gate must perform zero optimizer updates")
    result = {
        "passed": True,
        "optimizer_updates": 0,
        "starting_step": ctx.starting_step,
        "development": report,
        "batch": gradients,
    }
    write_new(ctx.args.output / "RESTORATION_RESULT.json", result)
    return result


def save_checkpoint(ctx, optimizer, step, evaluations, seen):
    state_report = optimizer_steps(optimizer, expected_step=step - 10000)
    root = ctx.args.output / "checkpoints"
    root.mkdir(exist_ok=True)
    weights = save_tensor_new(
        root / f"student-step{step}.pt",
        {k: v.detach().cpu() for k, v in ctx.student.state_dict().items()},
    )
    payload = {
        "optimizer": cpu_tree(optimizer.state_dict()),
        "optimizer_parameter_names": optimizer_names(ctx.student, optimizer),
        "rng": rng_state(),
        "completed_step": step,
        "phase": 2,
        "phase_optimizer_steps": step - 10000,
        "plan_sha256": ctx.args.plan_sha256,
        "source_commit": ctx.source_commit,
        "condition": ctx.args.condition,
    }
    opt = save_tensor_new(root / f"optimizer-rng-step{step}.pt", payload)
    index = {
        "run_id": ctx.args.run_id,
        "condition": ctx.args.condition,
        "source_commit": ctx.source_commit,
        "plan_sha256": ctx.args.plan_sha256,
        "cache_report_sha256": ctx.args.cache_report_sha256,
        "schedule_sha256": ctx.plan["schedule_sha256"],
        "completed_step": step,
        "next_step": step + 1,
        "phase": 2,
        "next_phase": 2,
        "next_phase_reinitialize_optimizer": False,
        "weights": weights,
        "optimizer_rng": opt,
        "student_content_sha256": proto.content_hash(ctx.student.state_dict()),
        "evaluations": evaluations,
        "gradient_groups_seen": {
            key: ctx.parent_gradient_groups_seen.get(key, False) or value
            for key, value in seen.items()
        },
        "attempt_gradient_groups_seen": seen,
        "parent_index": ctx.parent_index,
        "starting_step": ctx.starting_step,
        "training_log": log_prefix(ctx.args.output / "training_steps.jsonl"),
        "optimizer_state": state_report,
    }
    path = root / f"checkpoint-step{step}.json"
    write_new(path, index)
    return {"path": str(path.resolve()), "sha256": sha(path), "step": step}


def train(ctx, index, pending_rng):
    optimizer = ctx.optimizer
    ctx.optimizer = None
    schedule = read(ctx.plan["schedule_file"])
    restoration = restoration_gate(ctx, index, schedule)
    # Graph capture and full restoration evidence can consume RNG. Restore only
    # now, immediately before the first real continuation update.
    restore_rng(pending_rng)
    records = index["evaluations"]
    seen = {key: False for key in index["gradient_groups_seen"]}
    checkpoints = []
    first_state = None
    with (ctx.args.output / "training_steps.jsonl").open("x") as stream:
        for planned in schedule[ctx.starting_step :]:
            ctx.budget()
            if (
                planned["step"] != ctx.optimizer_steps_completed + 1
                or planned["phase"] != 2
                or planned["phase_step"] != planned["step"] - 10000
                or planned["lr"] != continuation_lr(planned["step"])
            ):
                raise ValueError("Frozen continuation progression/LR changed")
            # Both suffixes are task-only. Reuse the existing exact update body,
            # then record the originating matched condition without changing math.
            row = matched.update(ctx, optimizer, planned, "task_only")
            row["condition"] = ctx.args.condition
            row["continuation"] = True
            if planned["step"] == ctx.starting_step + 1:
                first_state = optimizer_steps(optimizer, expected_step=planned["step"] - 10000)
                row["restored_optimizer_first_update"] = first_state
            for group, value in row["gradient_groups"].items():
                seen[group] = seen.get(group, False) or value["finite_nonzero_parameters"] > 0
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            step = ctx.optimizer_steps_completed
            if step in ctx.plan["dev_probe_steps"]:
                records[f"dev-{step}"] = matched.evaluate_pairs(
                    ctx, matched.dev_pairs(ctx.plan), ctx.args.output / f"dev-step{step}"
                )
            if step in ctx.plan["train_probe_steps"]:
                records[f"train-{step}"] = matched.evaluate_pairs(
                    ctx, ctx.plan["train_probe_pairs"], ctx.args.output / f"train-probe-step{step}"
                )
            if step in ctx.plan["weight_checkpoint_steps"]:
                checkpoints.append(save_checkpoint(ctx, optimizer, step, records, seen))
                ctx.budget()
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "condition": ctx.args.condition,
                            "step": step,
                            "native_loss": row["native_loss"],
                            "kd": row["teacher_velocity_kd"],
                            "lr": row["lr"],
                        }
                    ),
                    flush=True,
                )
    final_state = optimizer_steps(optimizer, expected_step=40000)
    optimizer.zero_grad(set_to_none=True)
    changes = parameter_changes(ctx.student, ctx.initial, seen)
    matched.release_batch(ctx)
    return {
        "status": "completed",
        "technical_validity": "passed",
        "training_completed": ctx.optimizer_steps_completed == 50000,
        "condition": ctx.args.condition,
        "optimizer_steps": ctx.optimizer_steps_completed,
        "attempt_optimizer_steps": ctx.optimizer_steps_completed - ctx.starting_step,
        "new_contract_optimizer_steps": 30000,
        "cumulative_samples": 200000,
        "new_contract_samples": 120000,
        "starting_step": ctx.starting_step,
        "parent_index": ctx.parent_index,
        "restoration": restoration,
        "optimizer_first_update": first_state,
        "optimizer_final": final_state,
        "evaluations": records,
        "checkpoints": checkpoints,
        "changes_since_import": changes,
        "attempt_gradient_groups_seen": seen,
        "parent_gradient_groups_seen": ctx.parent_gradient_groups_seen,
        "quality_failure_stops_training": False,
        "new_optimizer_reset": False,
        "integrity": matched.integrity(ctx),
    }


def lineage_rows(record, plan, condition, current_sha, source_commit, seen=None):
    import hashlib

    seen = set() if seen is None else seen
    file_record_valid(record)
    if record["sha256"] in seen:
        raise ValueError("Continuation checkpoint lineage cycle")
    seen.add(record["sha256"])
    if record == parent_record(plan, condition):
        return checkpoint_log_rows(record, expected_identity(plan, condition))
    index = read(record["path"])
    expected = expected_identity(
        plan, condition, current_sha=current_sha, source_commit=source_commit
    )
    if any(index.get(k) != v for k, v in expected.items()):
        raise ValueError("Continuation lineage identity mismatch")
    rows = lineage_rows(index["parent_index"], plan, condition, current_sha, source_commit, seen)
    if len(rows) != index["starting_step"]:
        raise ValueError("Continuation log starting step differs")
    value = index["training_log"]
    with Path(value["path"]).open("rb") as stream:
        data = stream.read(value["bytes"])
    if hashlib.sha256(data).hexdigest() != value["sha256"]:
        raise ValueError("Continuation log prefix changed")
    suffix = [json.loads(line) for line in data.splitlines()]
    if len(suffix) != value["rows"]:
        raise ValueError("Continuation prefix row count changed")
    rows.extend(suffix)
    if [r["step"] for r in rows] != list(range(1, index["completed_step"] + 1)):
        raise ValueError("Continuation global optimizer steps missing/duplicated")
    return rows


def evaluate(args, plan):
    results = {}
    logs = {}
    terminals = {}
    for condition in plan["conditions"]:
        path = getattr(args, condition + "_terminal")
        digest = getattr(args, condition + "_terminal_sha256")
        if (path is None) != (digest is None):
            raise ValueError("Explicit terminal override needs its SHA")
        if path is None:
            path = args.artifact_root / plan["run_ids"][condition] / "output/RESULT.json"
        if digest is not None and sha(path) != digest:
            raise ValueError("Explicit continuation terminal changed")
        terminal = read(path)
        if (
            not terminal.get("training_completed")
            or terminal["optimizer_steps"] != 50000
            or terminal["condition"] != condition
            or terminal["plan_sha256"] != args.plan_sha256
            or terminal["cache_report_sha256"] != plan["cache_report_sha256"]
        ):
            raise ValueError("Two fixed final50000 continuation terminals required")
        record = terminal["evaluations"]["dev-50000"]
        file_record_valid(record)
        value = read(record["path"])
        root = Path(record["path"]).parent
        for name, digest in value["array_sha256"].items():
            if sha(root / f"{name}.npy") != digest:
                raise ValueError("Final continuation evaluation array changed")
        if sha(root / "details.json") != value["details_sha256"]:
            raise ValueError("Final continuation details changed")
        final = terminal["checkpoints"][-1]
        file_record_valid(final)
        idx = read(final["path"])
        file_record_valid(idx["weights"])
        if (
            idx["completed_step"] != 50000
            or idx["student_content_sha256"] != value["checkpoint_content_sha256"]
        ):
            raise ValueError("Final continuation checkpoint/evaluation mismatch")
        results[condition] = value
        logs[condition] = lineage_rows(
            final, plan, condition, args.plan_sha256, terminal["source_commit"]
        )
        terminals[condition] = {
            "path": str(path.resolve()),
            "sha256": sha(path),
            "source_commit": terminal["source_commit"],
            "teacher_actions_sha256": value["array_sha256"]["teacher_actions"],
        }
    if (
        len({v["source_commit"] for v in terminals.values()}) != 1
        or len({v["teacher_actions_sha256"] for v in terminals.values()}) != 1
    ):
        raise ValueError("Matched continuation source/teacher references differ")
    schedule = read(plan["schedule_file"])
    left, right = (logs[c] for c in plan["conditions"])
    if len(left) != 50000 or len(right) != 50000:
        raise ValueError("Complete50000 paired logs required")
    for planned, a, b in zip(schedule, left, right):
        if (
            any(a.get(k) != v or b.get(k) != v for k, v in planned.items())
            or a["noise_receipt"] != b["noise_receipt"]
        ):
            raise ValueError("Matched cumulative schedule/noise/time/condition receipts differ")
        if planned["step"] > 20000 and (a["objective"] != "task" or b["objective"] != "task"):
            raise ValueError("Continuation objective changed")
    task, distilled = (results[c] for c in plan["conditions"])
    gates = {}
    limits = plan["dev_gates"]
    for key, bound in (
        ("flow_loss", limits["max_distilled_task_flow_ratio"]),
        ("combined_mae_mixed_units", limits["max_distilled_task_combined_ratio"]),
    ):
        ratio = distilled["metrics"][key] / task["metrics"][key]
        better = sum(
            distilled["per_episode"][str(ep)][key] < task["per_episode"][str(ep)][key]
            for ep in plan["dev"]
        )
        gates[key] = {
            "ratio": ratio,
            "episodes_better": better,
            "passed": ratio <= bound and better >= limits["min_episodes_better"],
        }
    retention = {
        k: distilled["metrics"][k]
        <= distilled["teacher_metrics"][k] * limits["max_native_metric_ratio"]
        for k in limits["native_metrics"]
    }
    return {
        "status": "completed",
        "technical_validity": "passed",
        "results": results,
        "terminals": terminals,
        "matched_gates": gates,
        "native_retention": retention,
        "quality_gate": all(v["passed"] for v in gates.values()) and all(retention.values()),
        "paired_new_steps": 30000,
        "paired_new_samples": 120000,
        "paired_cumulative_steps": 50000,
        "paired_cumulative_samples": 200000,
        "historical_prefix_steps": 20000,
        "optimizer_steps": 0,
        "scope": "CPU-only fixed50000 development summary; no new final/confirmation or deployment claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--condition", choices=("task_only", "distill_then_task"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume-index", type=Path)
    parser.add_argument("--resume-index-sha256")
    for condition in ("task_only", "distill_then_task"):
        parser.add_argument("--" + condition.replace("_", "-") + "-terminal", type=Path)
        parser.add_argument("--" + condition.replace("_", "-") + "-terminal-sha256")
    args = parser.parse_args()
    if args.command == "train" and args.condition is None:
        parser.error("--condition is required for training")
    plan = verify_plan(args)
    if not 0 < args.wall_seconds <= plan["train_internal_wall_seconds"]:
        parser.error("Frozen wall bound exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ctx = None
    try:
        if args.command == "train":
            ctx, index, pending_rng = context_for(args, plan, started + args.wall_seconds)
            result = train(ctx, index, pending_rng)
            result["source_commit"] = ctx.source_commit
        else:
            result = evaluate(args, plan)
    except Exception as exc:
        result = {
            "status": "failed",
            "technical_validity": "not_established",
            "exception": type(exc).__name__,
            "message": str(exc),
            "optimizer_steps_completed": 0 if ctx is None else ctx.optimizer_steps_completed,
            "automatic_retry": False,
        }
    result.update(
        plan_sha256=args.plan_sha256,
        cache_report_sha256=args.cache_report_sha256,
        run_id=args.run_id,
        elapsed_seconds=time.monotonic() - started,
    )
    write_new(args.output / "RESULT.json", result)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
