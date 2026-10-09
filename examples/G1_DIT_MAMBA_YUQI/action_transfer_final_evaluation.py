"""One SHA-bound final16 cache-and-evaluate command; never selects or retries a candidate."""

import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

from action_transfer_checkpoint import (
    load_structure_template,
    load_student,
    semantic_identity,
    verify_candidate,
)
from action_transfer_final_support import (
    FinalCache,
    assert_teacher_equal,
    evaluate_final_candidate,
    final_comparisons,
    final_pairs,
    materialize_native_reference,
    verify_final_contract,
)
from action_transfer_full_cache import encode_frame, save_episode
from budgeted_graph_distillation_runner import environment
from contract import sha, write_new
from integrity_diagnosis_support import BaseAudit
import memory_prototype as proto
from run_train_eval import NativeAnchors, strict_reload
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def verify_configuration_records(plan, identities):
    paths = {
        "base_config": Path(plan["base_checkpoint"]) / "config.json",
        "processor_config": Path(plan["prepared"]) / "processor/processor_config.json",
        "processor_statistics": Path(plan["prepared"]) / "processor/statistics.json",
    }
    records = {}
    for name, path in paths.items():
        digest = identities.get(path.resolve())
        if digest is None or digest != sha(path):
            raise ValueError(f"Required non-weight configuration is not frozen: {name}")
        records[name] = {"path": str(path.resolve()), "sha256": digest}
    expected = {
        "action_horizon": 40,
        "max_state_dim": 13,
        "max_action_dim": 13,
        "num_inference_timesteps": 4,
    }
    config = read(paths["base_config"])
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen B0 config violates the H40/13D/four-Euler contract")
    return records


def runtime_contract(model, processor):
    expected = {
        "action_horizon": 40,
        "max_state_dim": 13,
        "max_action_dim": 13,
        "num_inference_timesteps": 4,
    }
    actual = {key: getattr(model.config, key) for key in expected}
    head = {
        "action_horizon": model.action_head.action_horizon,
        "action_dim": model.action_head.action_dim,
        "num_inference_timesteps": model.action_head.num_inference_timesteps,
    }
    processing = {
        "max_state_dim": processor.max_state_dim,
        "max_action_dim": processor.max_action_dim,
        "max_action_horizon": processor.max_action_horizon,
    }
    if (
        actual != expected
        or head != {"action_horizon": 40, "action_dim": 13, "num_inference_timesteps": 4}
        or processing != {"max_state_dim": 13, "max_action_dim": 13, "max_action_horizon": 40}
    ):
        raise ValueError("Loaded model/head/processor runtime contract differs before final cache")
    return {
        "model_config": actual,
        "action_head": head,
        "processor": processing,
        "checked_before_cache_started": True,
    }


def verify_plan(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Explicit final PLAN SHA required")
    plan = read(args.plan)
    if (
        Path(plan["source_worktree"]).resolve() != REPO
        or Path(plan["artifact_root"]).resolve() != args.artifact_root.resolve()
        or args.output.resolve() != args.artifact_root.resolve() / plan["run_id"] / "output"
        or plan["optimizer_steps"] != 0
    ):
        raise ValueError("Final source/output/zero-update identity changed")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen final input changed: {path}")
    identities = {Path(r["path"]).resolve(): r["sha256"] for r in plan["records"]}
    args.configuration_records = verify_configuration_records(plan, identities)
    for relative in plan["immutable_source_relative_paths"]:
        path = (REPO / relative).resolve()
        if identities.get(path) != sha(path):
            raise ValueError(f"Original final evaluator mathematics changed: {relative}")
    if sha(plan["program_cohort_file"]) != plan["program_cohort_sha256"]:
        raise ValueError("Program cohort identity changed")
    verify_final_contract(plan, read(plan["program_cohort_file"]))
    # Verify every frozen candidate, including structure and actual tensor content,
    # before any final RGB decode or VLM call. No future hash is synthesized.
    torch.set_num_threads(4)
    signatures = {
        kind: load_structure_template(record)
        for kind, record in plan["student_structure_templates"].items()
    }
    args.candidate_identities = [
        verify_candidate(candidate, signatures[candidate["kind"]])
        for candidate in plan["candidates"]
    ]
    return plan


def build_final_cache(ctx, output):
    output.mkdir(parents=True, exist_ok=False)
    files = []
    encoded = 0
    for ep in ctx.plan["final"]:
        ctx.progress.update(stage="cache", episode=ep, anchor=None)
        ctx.budget()
        if ctx.source.metadata[ep]["length"] != ctx.plan["lengths"][str(ep)]:
            raise ValueError("Final episode metadata length changed")
        frames = {}
        started = time.monotonic()
        for anchor in ctx.plan["eval_anchors"][str(ep)]:
            ctx.progress["anchor"] = anchor
            ctx.budget()
            frames[anchor] = encode_frame(ctx.model, ctx.source, ep, anchor)
            encoded += 1
            ctx.progress["encoded_observations"] = encoded
        record = {"episode": ep, **save_episode(output / f"episode-{ep:06d}.pt", frames)}
        files.append(record)
        ctx.event(
            "cache_episode_completed",
            episode=ep,
            observations=len(frames),
            elapsed_seconds=time.monotonic() - started,
            **{"file_sha256": record["sha256"]},
        )
    expected = sum(len(ctx.plan["eval_anchors"][str(ep)]) for ep in ctx.plan["final"])
    if encoded != expected:
        raise ValueError("Final cache is incomplete")
    proto.assert_frozen(ctx.model)
    if proto.content_hash(ctx.model.state_dict()) != ctx.base_before:
        raise ValueError("VLM cache build changed the base")
    result = {
        "status": "passed",
        "cohort": "final",
        "episodes": ctx.plan["final"],
        "observations": encoded,
        "files": files,
        "plan_sha256": ctx.args.plan_sha256,
        "program_cohort_sha256": ctx.plan["program_cohort_sha256"],
        "anchors": ctx.plan["eval_anchors"],
        "base_content_sha256": ctx.base_before,
        "raw_post_separate": True,
        "scope": "Only the512 explicitly frozen final observations; no train/dev or reserve aliases",
        "optimizer_steps": 0,
    }
    path = output / "CACHE_RESULT.json"
    write_new(path, result)
    return {"path": str(path.resolve()), "sha256": sha(path), "observations": encoded}


def run(args, plan, deadline):
    identity = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    ctx = SimpleNamespace(args=args, plan=plan, stage_peak=0, progress=args.progress)

    def budget():
        ctx.stage_peak = max(ctx.stage_peak, torch.cuda.max_memory_allocated())
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if time.monotonic() > deadline:
            raise TimeoutError("Joint final evaluation wall bound exceeded")
        if size > plan["max_artifact_bytes"] or ctx.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Joint final evaluation resource bound exceeded")
        return size

    ctx.budget = budget
    events_path = args.output / "EXECUTION.jsonl"

    def event(name, **fields):
        value = {"event": name, "elapsed_seconds": time.monotonic() - args.started, **fields}
        with events_path.open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    ctx.event = event
    ctx.progress["stage"] = "load_base"
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    ctx.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    ctx.teacher = model.action_head.model
    ctx.base_before = proto.content_hash(model.state_dict())
    if ctx.base_before != plan["base_content_sha256"]:
        raise ValueError("Frozen final B0 identity mismatch")
    ctx.audit = BaseAudit(ctx.model, args.output, ctx.base_before, budget=budget)
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    actual_contract = runtime_contract(ctx.model, ctx.source.processor)
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": args.plan_sha256,
            "source": identity,
            "strict_loading": loading,
            "base_content_sha256": ctx.base_before,
            "non_weight_configuration_records": args.configuration_records,
            "runtime_contract": actual_contract,
            "candidate_order": [c["name"] for c in plan["candidates"]],
            "candidate_identities": args.candidate_identities,
            "student_structure_templates": plan["student_structure_templates"],
            "cohort": "final",
            "episodes": plan["final"],
            "expected_observations": 512,
            "expected_outputs_per_model": 2048,
            "optimizer_steps": 0,
        },
    )
    event("cache_started")
    cache_record = build_final_cache(ctx, args.output / "cache")
    args.cache_report_sha256 = cache_record["sha256"]
    ctx.cache = FinalCache(args.output / "cache", plan, args.plan_sha256, args.cache_report_sha256)
    event("cache_completed", **cache_record)
    pairs = final_pairs(plan)
    records = {}
    candidate_results = {}
    first = None
    for candidate in plan["candidates"]:
        name = candidate["name"]
        ctx.progress.update(stage="candidate", candidate=name, episode=None, anchor=None)
        budget()
        event(
            "candidate_started",
            candidate=name,
            native_reference_computed_concurrently=first is None,
        )
        directory = args.output / "candidates" / name
        terminal_path = args.output / "candidate-terminals" / f"{name}.json"
        record = None
        try:
            ctx.candidate = candidate
            ctx.student = load_student(ctx.model.action_head, candidate, device="cuda")
            ctx.audit.check("before-candidate:" + name, stable=True)
            record = evaluate_final_candidate(ctx, pairs, directory)
            if (
                record["result"]["rows"] != 2048
                or record["result"]["checkpoint_content_sha256"] != candidate["content_sha256"]
            ):
                raise ValueError("Final candidate did not finish all2048 frozen outputs")
            if first is None:
                first = record
                native = materialize_native_reference(
                    first, args.output / "native-reference", ctx.base_before, pairs
                )
                records["native_reference"] = native
                candidate_results["native_reference"] = native["result"]
                event(
                    "native_reference_materialized",
                    from_candidate=name,
                    rows=2048,
                    result_sha256=native["sha256"],
                )
            else:
                assert_teacher_equal(first, record, pairs)
            if (
                proto.content_hash(ctx.student.state_dict()) != candidate["content_sha256"]
                or sha(candidate["path"]) != candidate["sha256"]
            ):
                raise ValueError("Final scoring changed candidate weights")
            ctx.cache.assert_readonly()
            ctx.audit.check("after-candidate:" + name, stable=True)
            terminal = {
                "status": "completed",
                "cohort": "final",
                "candidate": name,
                "rows": 2048,
                "result": {"path": record["path"], "sha256": record["sha256"]},
                "checkpoint": candidate,
                "semantic_identity": semantic_identity(ctx.student.state_dict(), candidate["kind"]),
                "teacher_reference_exact": True,
                "optimizer_steps": 0,
            }
            write_new(terminal_path, terminal)
            records[name] = record
            candidate_results[name] = record["result"]
            ctx.progress["completed_candidates"].append(name)
            event("candidate_completed", candidate=name, rows=2048, result_sha256=record["sha256"])
        except Exception as exc:
            if not terminal_path.exists():
                write_new(
                    terminal_path,
                    {
                        "status": "failed",
                        "cohort": "final",
                        "candidate": name,
                        "completed_rows": None if record is None else record["result"]["rows"],
                        "expected_rows": 2048,
                        "exception": type(exc).__name__,
                        "message": str(exc),
                        "optimizer_steps": 0,
                    },
                )
            raise
        finally:
            if hasattr(ctx, "student"):
                del ctx.student
            gc.collect()
            torch.cuda.empty_cache()
    ctx.progress.update(stage="aggregate", candidate=None)
    if len(ctx.progress["completed_candidates"]) != len(plan["candidates"]):
        raise ValueError("Joint final candidate set incomplete")
    ctx.cache.assert_readonly()
    proto.assert_frozen(ctx.model)
    if proto.content_hash(ctx.model.state_dict()) != ctx.base_before:
        raise ValueError("Final evaluation changed B0")
    analysis = final_comparisons(candidate_results, plan)
    budget()
    event("joint_completed", outputs_per_model=2048)
    return {
        "status": "completed",
        "technical_validity": "passed",
        "cohort": "final",
        "complete": True,
        "source": identity,
        "cache": cache_record,
        "records": {
            name: {"path": r["path"], "sha256": r["sha256"]} for name, r in records.items()
        },
        "results": candidate_results,
        **analysis,
        "completed_candidates": ctx.progress["completed_candidates"],
        "expected_candidates": [c["name"] for c in plan["candidates"]],
        "outputs_per_model": 2048,
        "teacher_references_exact": True,
        "base_cache_checkpoints_unchanged": True,
        "optimizer_steps": 0,
        "execution_log_sha256": sha(events_path),
        "peak_allocated_bytes": ctx.stage_peak,
        "scope": "One frozen final16 joint evaluation; no checkpoint, anchor, cohort or rerun selection from results; reserve untouched",
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
        parser.error("Frozen final wall bound exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    args.started = time.monotonic()
    args.progress = {
        "stage": "initialization",
        "encoded_observations": 0,
        "completed_candidates": [],
        "candidate": None,
        "episode": None,
        "anchor": None,
    }
    try:
        result = run(args, plan, args.started + args.wall_seconds)
    except Exception as exc:
        result = {
            "status": "failed",
            "technical_validity": "not_established",
            "cohort": "final",
            "complete": False,
            "exception": type(exc).__name__,
            "message": str(exc),
            "progress": args.progress,
            "expected_candidates": [c["name"] for c in plan["candidates"]],
            "optimizer_steps": 0,
            "automatic_retry": False,
        }
    result.update(plan_sha256=args.plan_sha256, elapsed_seconds=time.monotonic() - args.started)
    write_new(args.output / "RESULT.json", result)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
