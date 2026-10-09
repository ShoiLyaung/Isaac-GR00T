"""Bounded fresh-parent integrity matrix; first base change stops all remaining cases."""

import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import action_transfer_continuation as continuation
import action_transfer_matched_runner as matched
from action_transfer_matched_support import MMapCache, assemble_batch
from action_transfer_program_support import cpu_tree, restore_rng
import action_transfer_trajectory_kd as original
from action_transfer_trajectory_kd_support import mixed_query, query_receipts
from budgeted_action_cudagraph import CoreDispatcher, TensorCore
from budgeted_graph_distillation_runner import environment
from contract import sha, write_new
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
from integrity_diagnosis_support import BaseAudit, BaseMutation
from memory_training import optimizer_for
from run_train_eval import NativeAnchors, strict_reload
from runner_core import set_seed
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def verify_plan(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Explicit integrity PLAN SHA mismatch")
    plan = read(args.plan)
    if (
        Path(plan["source_worktree"]).resolve() != REPO
        or Path(plan["artifact_root"]).resolve() != args.artifact_root.resolve()
        or args.output.resolve() != args.artifact_root.resolve() / plan["run_id"] / "output"
        or plan["formal_training_authorized"]
        or not plan["first_mutation_stops_matrix"]
        or len(plan["diagnostic_matrix"]) * len(plan["diagnostic_cases"])
        > plan["diagnostic_optimizer_steps_max"]
    ):
        raise ValueError("Frozen integrity matrix/source/budget contract changed")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen diagnostic material changed: {path}")
    identities = {Path(row["path"]).resolve(): row["sha256"] for row in plan["records"]}
    for rel in plan["immutable_source_relative_paths"]:
        path = (REPO / rel).resolve()
        if identities.get(path) != sha(path):
            raise ValueError("Original implementation changed")
    return plan


def extra_tensors(ctx):
    values = {}
    if hasattr(ctx, "student"):
        values.update({"student." + k: v for k, v in ctx.student.named_parameters()})
        values.update({"student_buffer." + k: v for k, v in ctx.student.named_buffers()})
        values.update(
            {
                "gradient." + k: v.grad
                for k, v in ctx.student.named_parameters()
                if v.grad is not None
            }
        )
    if hasattr(ctx, "optimizer"):
        for i, state in enumerate(ctx.optimizer.state.values()):
            values.update(
                {f"optimizer.{i}.{k}": v for k, v in state.items() if isinstance(v, torch.Tensor)}
            )
    values.update(
        {f"graph_input.{i}": v for i, v in enumerate(getattr(ctx, "samples", None) or ())}
    )
    values.update(ctx.observed)
    return values


class AuditRoute(nn.Module):
    def __init__(self, target, ctx):
        super().__init__()
        self.target = target
        self.ctx = ctx

    def forward(self, *args, **kwargs):
        value = self.target(*args, **kwargs)
        self.ctx.observed = {
            f"student_input.{k}": v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        }
        self.ctx.observed["student_output"] = value[0] if isinstance(value, tuple) else value
        try:
            self.ctx.audit.check("student_forward", stable=False)
        finally:
            self.ctx.observed = {}
        return value


def attach_teacher_check(ctx):
    def after(_module, _args, kwargs, output):
        ctx.observed = {
            f"teacher_input.{k}": v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        }
        ctx.observed["teacher_output"] = output[0] if isinstance(output, tuple) else output
        try:
            ctx.audit.check("teacher_query", stable=False)
        finally:
            ctx.observed = {}

    return ctx.teacher.register_forward_hook(after, with_kwargs=True)


def query_checked(ctx, route, frame, planned, variant):
    set_seed(planned["forward_seed"], cuda=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = mixed_query(ctx.model, ctx.teacher, route, frame, planned, variant)
    ctx.audit.check("query_scopes_restored", stable=True)
    return values


def run_one(args, plan, entry, parent, source, deadline, totals):
    out = args.output / entry["name"]
    out.mkdir()
    ctx = SimpleNamespace(args=args, plan=plan, lease=None, observed={}, samples=(), stage_peak=0)

    def budget():
        ctx.stage_peak = max(ctx.stage_peak, torch.cuda.max_memory_allocated())
        if time.monotonic() > deadline:
            raise TimeoutError("Integrity diagnostic wall bound exceeded")
        if (
            ctx.stage_peak > plan["max_allocated_bytes"]
            or sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
            > plan["max_artifact_bytes"]
        ):
            raise RuntimeError("Integrity diagnostic resource bound exceeded")

    ctx.budget = budget
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    ctx.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    ctx.teacher = model.action_head.model
    ctx.audit = BaseAudit(
        ctx.model,
        out,
        plan["base_content_sha256"],
        plan["diff_chunk_bytes"],
        plan["diff_sample_elements"],
        budget=budget,
    )
    ctx.audit.extra = lambda: extra_tensors(ctx)
    ctx.audit.check("base_loaded", stable=True)
    ctx.student = BudgetedActionMamba(ctx.teacher)
    ctx.audit.check("student_constructed", stable=True)
    index, payload = parent
    initial = torch.load(index["weights"]["path"], map_location="cpu", weights_only=True)
    ctx.student.load_state_dict(initial, strict=True)
    ctx.student.cuda().float().eval().requires_grad_(True)
    del initial
    ctx.audit.check("student_loaded", stable=True)
    ctx.optimizer = optimizer_for(ctx.student, plan)
    # AdamW's noncapturable CPU step tensors may alias load_state_dict inputs.
    # Clone payload state so each fixed matrix entry genuinely restarts at20k.
    continuation.restore_optimizer(ctx.student, ctx.optimizer, cpu_tree(payload), 10000)
    ctx.audit.check("optimizer_restored", stable=True)
    write_new(out / "INITIAL_ALIASES.json", ctx.audit.aliases())
    ctx.cache = MMapCache(plan["cache_root"], plan, plan["cache_report_sha256"], budget)
    row = read(plan["schedule_file"])[plan["diagnostic_schedule_index"]]
    frame = assemble_batch(ctx.cache, row["batch"])
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    hook = None
    checked = None
    try:
        ctx.audit.phase = "setup"
        if entry["execution"] == "graph":
            # The original helper creates one graph. No audit hook exists during capture.
            saved_output = args.output
            args.output = out
            try:
                matched.create_batch_graph(ctx, frame, row["forward_seed"])
            finally:
                args.output = saved_output
            ctx.audit.check("graph_constructed", stable=True)
        elif entry["execution"] == "eager":
            ctx.route = CoreDispatcher(TensorCore(ctx.student, student=True, backward=True))
            ctx.audit.check("eager_surface_constructed", stable=True)
        else:
            raise ValueError("Unknown fixed diagnostic execution")
        checked = AuditRoute(ctx.route, ctx)
        hook = attach_teacher_check(ctx)
        write_new(out / "POST_SETUP_ALIASES.json", ctx.audit.aliases())
        proofs = []
        for case in plan["diagnostic_cases"]:
            ctx.audit.phase = "proof." + case["name"]
            ctx.student.zero_grad(set_to_none=True)
            values = query_checked(ctx, checked, frame, {**row, **case}, entry["variant"])
            values["kd"].backward()
            ctx.audit.check("backward", stable=True)
            proofs.append({"case": case, "kd": float(values["kd"].detach())})
            del values
            ctx.student.zero_grad(set_to_none=True)
            ctx.audit.check("proof_gradients_cleared", stable=True)
        write_new(out / "PROOF.json", {"cases": proofs, "optimizer_updates": 0})
        restore_rng(payload["rng"])
        steps = []
        for local_step, case in enumerate(plan["diagnostic_cases"], 1):
            if totals["optimizer_steps"] >= plan["diagnostic_optimizer_steps_max"]:
                raise ValueError("Diagnostic update bound reached")
            ctx.audit.phase = f"update{local_step}.{case['name']}"
            ctx.optimizer.zero_grad(set_to_none=True)
            for group in ctx.optimizer.param_groups:
                group["lr"] = plan["diagnostic_lr"]
            values = query_checked(ctx, checked, frame, {**row, **case}, entry["variant"])
            values["kd"].backward()
            ctx.audit.check("backward", stable=True)
            norm = torch.nn.utils.clip_grad_norm_(
                ctx.student.parameters(),
                plan["optimizer"]["gradient_clip"],
                error_if_nonfinite=True,
            )
            ctx.audit.check("gradient_clip", stable=True)
            ctx.optimizer.step()
            totals["optimizer_steps"] += 1
            args.diagnostic_optimizer_steps = totals["optimizer_steps"]
            ctx.audit.check("optimizer_update", stable=True)
            shared, selected, _ = query_receipts(values, frame, row["forward_seed"])
            steps.append(
                {
                    "diagnostic_update": local_step,
                    "matrix_total_optimizer_updates": totals["optimizer_steps"],
                    "parent_global_step": 20000,
                    "parent_optimizer_step": 10000,
                    "actual_optimizer_step": 10000 + local_step,
                    "case": case,
                    "kd": float(values["kd"].detach()),
                    "grad_norm": float(norm),
                    "shared": shared,
                    "selected": selected,
                }
            )
            del values
        write_new(out / "DIAGNOSTIC_UPDATES.json", steps)
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.audit.phase = "fixed_evaluation"
        ctx.audit.check("before_fixed_evaluation", stable=True)
        # Existing native evaluator only; no new RGB/data or scientific quality gate.
        result = matched.evaluate_pairs(
            ctx, plan["diagnostic_evaluation_pairs"], out / "fixed-evaluation"
        )
        ctx.audit.check("after_fixed_evaluation", stable=True)
        hook.remove()
        hook = None
        checked.target = None
        checked = None
        ctx.audit.phase = "release"
        ctx.audit.check("before_release", stable=True)
        if ctx.lease is not None:
            matched.release_batch(ctx)
        else:
            ctx.route = None
            ctx.samples = ()
            gc.collect()
            torch.cuda.empty_cache()
        ctx.audit.check("after_release", stable=True)
        ctx.cache.assert_readonly()
        value = {
            "status": "not_reproduced",
            "matrix_entry": entry,
            "checks": ctx.audit.checks,
            "diagnostic_optimizer_steps": len(steps),
            "fixed_evaluation": {"path": result["path"], "sha256": result["sha256"]},
            "initial_base_hash": ctx.audit.full_hash,
            "stage_peak_allocated_bytes": ctx.stage_peak,
            "source": source,
            "strict_loading": loading,
            "limitations": "Checks synchronize CUDA and copy base tensors to CPU; a short non-reproduction does not validate historical r1",
        }
        write_new(out / "RESULT.json", value)
        return value
    finally:
        if hook is not None:
            hook.remove()
        if checked is not None:
            checked.target = None
        if ctx.lease is not None:
            ctx.student.zero_grad(set_to_none=True)
            ctx.lease.close()
            ctx.route.backend = None
            ctx.route.surface = None
            ctx.lease = None
        ctx.audit.extra = lambda: {}


def run_matrix(args, plan, deadline, entry_runner=run_one):
    source = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    args.resume_index = args.resume_index_sha256 = None
    args.cache_report_sha256 = plan["cache_report_sha256"]
    args.run_id = plan["run_id"]
    args.condition = plan["diagnostic_matrix"][0]["variant"]
    _, index, payload = original.import_parent(args, plan, source["commit"])
    if index["completed_step"] != 20000:
        raise ValueError("Only the frozen20k parent may seed the integrity diagnosis")
    totals = {"optimizer_steps": 0}
    args.diagnostic_optimizer_steps = 0
    rows = []
    try:
        for entry in plan["diagnostic_matrix"]:
            rows.append(entry_runner(args, plan, entry, (index, payload), source, deadline, totals))
            gc.collect()
            torch.cuda.empty_cache()
    except BaseMutation as exc:
        return {
            "status": "mutation_reproduced",
            "technical_validity": "diagnostic_failure_located",
            "matrix_completed": rows,
            "first_mutation": str(exc),
            "diagnostic_optimizer_steps": totals["optimizer_steps"],
            "formal_optimizer_steps": 0,
            "historical_r1_requalified": False,
        }
    return {
        "status": "not_reproduced",
        "matrix_completed": rows,
        "diagnostic_optimizer_steps": totals["optimizer_steps"],
        "formal_optimizer_steps": 0,
        "historical_r1_requalified": False,
        "scope": "Only fixed short fresh-parent coverage; no retry or longer run was started",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    plan = verify_plan(args)
    if not 0 < args.wall_seconds <= plan["internal_wall_seconds"]:
        parser.error("Frozen diagnostic wall bound exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        result = run_matrix(args, plan, start + args.wall_seconds)
    except Exception as exc:
        result = {
            "status": "diagnostic_error",
            "exception": type(exc).__name__,
            "message": str(exc),
            "formal_optimizer_steps": 0,
            "diagnostic_optimizer_steps": getattr(args, "diagnostic_optimizer_steps", 0),
            "automatic_retry": False,
        }
    result.update(plan_sha256=args.plan_sha256, elapsed_seconds=time.monotonic() - start)
    write_new(args.output / "RESULT.json", result)
    if result["status"] == "diagnostic_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
