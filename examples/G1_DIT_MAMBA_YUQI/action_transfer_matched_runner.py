"""Matched B4 action transfer: preflight, two frozen20000-step conditions, CPU joint review."""

import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

from action_transfer_matched_support import (
    PLAN_SHA,
    MMapCache,
    StrictBatchDispatcher,
    assemble_batch,
    batch_forward,
    checkpoint_log_rows,
    objective_for,
    phase_for_step,
    read_resume,
    restore_checkpoint,
    save_checkpoint,
)
from action_transfer_program_support import restore_rng
from budgeted_action_cudagraph import (
    BackwardGraph,
    CoreDispatcher,
    ForwardGraph,
    TensorCore,
    tensor_args,
)
from budgeted_action_cudagraph_probe import (
    assert_gradients_cleared,
    gradient_comparison,
    group_mapping,
    snapshot_gradients,
)
from budgeted_action_support import (
    cached_action,
    capture_velocity,
    gradient_groups,
    parameter_changes,
)
from budgeted_graph_distillation import dropout_inventory, evidence_receipt
from budgeted_graph_distillation_runner import environment
from contract import sha, write_new
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
from memory_numerical_trace import error_summary
import memory_prototype as proto
from memory_training import optimizer_for, raw_record
from memory_training_support import native_flow
import numpy as np
from runner_core import aggregate_metrics, deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
METRICS = (
    "flow_loss",
    "combined_mae_mixed_units",
    "arm_mae_rad",
    "hand_mae_u",
    "teacher_velocity_kd",
    "teacher_action_physical_mae",
)


def read(path):
    return json.loads(Path(path).read_text())


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen matched PLAN changed")
    plan = read(args.plan)
    if (
        REPO != Path(plan["source_worktree"])
        or args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve()
    ):
        raise ValueError("Matched source/artifact identity mismatch")
    if (
        not args.cache_report_sha256
        or sha(Path(plan["cache_root"]) / "CACHE_RESULT.json") != args.cache_report_sha256
    ):
        raise ValueError("Explicit completed cache-report SHA required before model load")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen matched material changed: {path}")
    for relative in plan["immutable_source_relative_paths"]:
        expected = {r["sha256"] for r in plan["records"] if r["path"].endswith("/" + relative)}
        if len(expected) != 1 or sha(REPO / relative) not in expected:
            raise ValueError(f"Existing implementation changed: {relative}")
    if sha(plan["schedule_file"]) != plan["schedule_sha256"]:
        raise ValueError("Matched schedule changed")
    if args.command == "preflight":
        expected = plan["preflight_run_id"]
    elif args.command == "train":
        if args.condition not in plan["conditions"]:
            raise ValueError("Training condition required")
        expected = plan["train_run_ids"][args.condition]
    else:
        expected = "joint-evaluation-r1"
    if (args.resume_index is None) != (args.resume_index_sha256 is None):
        raise ValueError("Parent resume index and SHA required together")
    if args.resume_index is None and args.run_id != expected:
        raise ValueError("Fresh canonical run_id mismatch")
    if args.resume_index is not None and (args.command != "train" or args.run_id == expected):
        raise ValueError("Resume requires new training-attempt identity")
    if (
        any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_id)
        or args.output.resolve() != (args.artifact_root / args.run_id / "output").resolve()
    ):
        raise ValueError("Output/run identity mismatch")
    return plan


def preflight_required(args, plan, commit):
    path = args.artifact_root / plan["preflight_run_id"] / "output/RESULT.json"
    r = read(path)
    if (
        r["status"] != "passed"
        or not r["continuation_gate"]
        or r["plan_sha256"] != PLAN_SHA
        or r["source_commit"] != commit
        or r["cache_report_sha256"] != args.cache_report_sha256
    ):
        raise ValueError("Matching B4/B1 preflight required")
    initial = validate_initial_dev(r["dev_initial"], plan, args.cache_report_sha256)
    return {"path": str(path), "sha256": sha(path), "dev_initial": initial}


def validate_initial_dev(record, plan, cache_sha):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError("Fresh shared dev baseline changed")
    value = read(path)
    if (
        value["status"] != "passed"
        or value["rows"] != len(dev_pairs(plan))
        or value["checkpoint_content_sha256"] != plan["student_initialization"]["content_sha256"]
        or value["cache_report_sha256"] != cache_sha
    ):
        raise ValueError("Shared dev baseline is not the frozen fresh initialization")
    for name, digest in value["array_sha256"].items():
        if sha(path.parent / f"{name}.npy") != digest:
            raise ValueError("Shared dev baseline arrays changed")
    if sha(path.parent / "details.json") != value["details_sha256"]:
        raise ValueError("Shared dev baseline details changed")
    return record


def initial_dev_baseline(ctx):
    if ctx.optimizer_steps_completed != 0 or ctx.lease is not None:
        raise ValueError("Shared dev baseline must precede graph capture and all updates")
    if (
        proto.content_hash(ctx.student.state_dict())
        != ctx.plan["student_initialization"]["content_sha256"]
    ):
        raise ValueError("Shared baseline requires original fresh student weights")
    return evaluate_pairs(ctx, dev_pairs(ctx.plan), ctx.args.output / "dev-step0")


def context_for(args, plan, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    source = environment(plan)
    prerequisite = (
        preflight_required(args, plan, source["commit"]) if args.command == "train" else None
    )
    parent, payload = read_resume(args, plan, source["commit"])
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
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if time.monotonic() > deadline:
            raise TimeoutError("Matched stage wall bound exceeded")
        if size > plan["max_artifact_bytes"] or ctx.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Matched storage/GPU bound exceeded")
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
        raise ValueError("Cache/base frozen identity mismatch")
    ctx.initial = torch.load(
        plan["student_initialization"]["path"], map_location="cpu", weights_only=True
    )
    if proto.content_hash(ctx.initial) != plan["student_initialization"]["content_sha256"]:
        raise ValueError("Fresh original student initialization required")
    ctx.student = BudgetedActionMamba(ctx.teacher)
    ctx.student.load_state_dict(ctx.initial, strict=True)
    ctx.student.cuda().float().eval().requires_grad_(True)
    optimizer = optimizer_for(ctx.student, plan)
    pending_rng = None
    if parent is not None:
        pending_rng = restore_checkpoint(ctx.student, optimizer, parent, payload)
        ctx.optimizer_steps_completed = parent["completed_step"]
    ctx.prerequisite = prerequisite
    ctx.starting_step = ctx.optimizer_steps_completed
    ctx.parent_index = (
        None
        if parent is None
        else {"path": str(args.resume_index.resolve()), "sha256": args.resume_index_sha256}
    )
    ctx.optimizer = optimizer
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "source": source,
            "strict_loading": loading,
            "cache_report_sha256": args.cache_report_sha256,
            "prerequisite": prerequisite,
            "parent_index": None
            if parent is None
            else {"path": str(args.resume_index), "sha256": args.resume_index_sha256},
            "starting_step": ctx.optimizer_steps_completed,
            "base_content_sha256": ctx.base_before,
            "starting_student_content_sha256": proto.content_hash(ctx.student.state_dict()),
            "dropout": dropout_inventory(ctx.student),
            "batch_scope": "Concatenated immutable B1 raw VLM cache; actual B4 native head postprocessing, no B1-post substitution",
        },
    )
    return ctx, parent, pending_rng


def create_batch_graph(ctx, frame, seed):
    if ctx.lease is not None:
        raise ValueError("A graph is already live")
    eager = CoreDispatcher(TensorCore(ctx.student, student=True, backward=True))
    set_seed(seed, cuda=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        values = batch_forward(ctx.model, ctx.teacher, eager, frame)
    samples = tensor_args(values["evidence"]["core_inputs"], student=True, backward=True)
    del values, eager
    surface = TensorCore(ctx.student, student=True, backward=True)
    ctx.lease = BackwardGraph(surface, samples)
    ctx.route = StrictBatchDispatcher(surface, ctx.lease, samples)
    ctx.samples = samples
    write_new(
        ctx.args.output / "BATCH_CAPTURE.json",
        {
            "capture_count": 1,
            "batch_size": 4,
            **ctx.lease.setup,
            "input_signature": [
                {"shape": list(v.shape), "dtype": str(v.dtype), "requires_grad": v.requires_grad}
                for v in samples
            ],
        },
    )


def release_batch(ctx):
    ctx.student.zero_grad(set_to_none=True)
    ctx.lease.assert_storage()
    ctx.lease.close()
    ctx.route.backend = None
    ctx.route.surface = None
    ctx.lease = None
    ctx.route = None
    ctx.samples = None
    gc.collect()
    torch.cuda.empty_cache()


def batch_checks(ctx, schedule, update, output):
    output.mkdir(exist_ok=False)
    mapping = group_mapping(ctx.student, "student")
    reports = []
    eager = StrictBatchDispatcher(
        TensorCore(ctx.student, student=True, backward=True), None, ctx.samples
    )
    for index in ctx.plan["preflight_batch_indices"]:
        planned = schedule[index]
        frame = assemble_batch(ctx.cache, planned["batch"])
        for objective in ("task", "kd"):
            data = {}
            for path, route in (("eager", eager), ("graph", ctx.route)):
                ctx.budget()
                ctx.student.zero_grad(set_to_none=True)
                assert_gradients_cleared(ctx.student)
                set_seed(planned["forward_seed"], cuda=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values = batch_forward(ctx.model, ctx.teacher, route, frame)
                loss = values["native"]["loss"] if objective == "task" else values["kd"]
                loss.backward()
                data[path] = {
                    "loss": float(loss.detach()),
                    "receipt": evidence_receipt(values, planned["forward_seed"]),
                    "gradients": snapshot_gradients(ctx.student),
                    "velocity": values["student_velocity"].detach().cpu(),
                }
                del values, loss
            if data["eager"]["receipt"] != data["graph"]["receipt"]:
                raise ValueError("Actual B4 noise/time/condition/target/mask differs")
            stats = gradient_comparison(
                data["eager"]["gradients"],
                data["graph"]["gradients"],
                mapping,
                ctx.plan["gradient_limits"],
            )
            path = output / f"gradients-batch{index}-{objective}.json"
            write_new(path, stats)
            difference = abs(data["graph"]["loss"] - data["eager"]["loss"])
            limit = ctx.plan["loss_limits"]
            loss_gate = difference <= limit["atol"] + limit["rtol"] * abs(data["eager"]["loss"])
            reports.append(
                {
                    "update": update,
                    "batch_index": index,
                    "objective": objective,
                    "loss_gate": loss_gate,
                    "reference_loss": data["eager"]["loss"],
                    "graph_loss": data["graph"]["loss"],
                    "gradient_gate": stats["passed"],
                    "gradient_path": str(path),
                    "gradient_sha256": sha(path),
                    "receipt": data["eager"]["receipt"],
                    "passed": loss_gate and stats["passed"],
                }
            )
            del data
            ctx.student.zero_grad(set_to_none=True)
            gc.collect()
            ctx.lease.assert_storage()
    report = {
        "update": update,
        "passed": all(row["passed"] for row in reports),
        "comparisons": reports,
    }
    write_new(output / "RESULT.json", report)
    return report


def b1_deployment(ctx):
    if ctx.lease is not None:
        raise ValueError("Release B4 graph before B1 deployment graph")
    case = ctx.plan["inference_probes"][0]
    frame = ctx.cache.frame(case["episode"], case["anchor"])
    seen = []

    def capture(_module, args, kwargs):
        seen.append(tensor_args(kwargs, student=True, backward=False))

    handle = ctx.student.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        cached_action(ctx.model, frame, case["seed"], student=ctx.student)
    finally:
        handle.remove()
    if len(seen) != 4:
        raise ValueError("Original B1 sampler must have4core calls")
    surface = TensorCore(ctx.student, student=True, backward=False)
    lease = ForwardGraph(surface, seen[0])
    route = CoreDispatcher(surface, lease)
    rows = []
    tensors = {}
    for case in ctx.plan["inference_probes"]:
        frame = ctx.cache.frame(case["episode"], case["anchor"])
        expected = cached_action(ctx.model, frame, case["seed"], student=ctx.student)
        before = route.calls
        actual = cached_action(ctx.model, frame, case["seed"], student=route)
        if route.calls - before != 4:
            raise ValueError("B1 Graph sampler changed native4Euler cadence")
        stats = error_summary(actual[:, :16], expected[:, :16])
        passed = all(stats[k] <= v for k, v in ctx.plan["action_limits"].items())
        rows.append({**case, "error": stats, "passed": passed})
        tensors[str(case["episode"])] = {"eager": expected, "graph": actual}
    lease.close()
    route.backend = None
    route.surface = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.save(tensors, ctx.args.output / "B1_DEPLOYMENT_ACTIONS.pt")
    return {
        "passed": all(r["passed"] for r in rows),
        "cases": rows,
        "separate_b1_capture": True,
        "batch4_graph_released_before_capture": True,
    }


def native_batch_check(ctx, frame, seed):
    set_seed(seed, cuda=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected, receipt = native_flow(ctx.model, frame, None, seed)
        set_seed(seed, cuda=True)
        actual = batch_forward(ctx.model, ctx.teacher, ctx.teacher, frame)
    for key in ("loss", "action_mask", "state_features", "backbone_features"):
        if not torch.equal(expected[key], actual["native"][key]):
            raise ValueError("Original B4 native objective parity failed")
    if torch.count_nonzero(actual["student_velocity"] - actual["teacher_velocity"]):
        raise ValueError("Same-state native teacher replay changed velocity")
    return {
        "native_head_b4_exact": True,
        "batch_size": 4,
        "mask_sum": float(actual["native"]["action_mask"].sum()),
        "native_flow_loss": float(expected["loss"]),
        "receipt": receipt,
        "scope": "Native head on concatenated B1 raw-cache features, not end-to-end B4 VLM equality",
    }


def update(ctx, optimizer, planned, condition):
    start = time.monotonic()
    frame = assemble_batch(ctx.cache, planned["batch"])
    preparation = time.monotonic() - start
    for group in optimizer.param_groups:
        group["lr"] = planned["lr"]
    optimizer.zero_grad(set_to_none=True)
    assert_gradients_cleared(ctx.student)
    set_seed(planned["forward_seed"], cuda=True)
    ctx.lease.assert_storage()
    dropout_inventory(ctx.student)
    objective = objective_for(condition, planned["step"])
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = batch_forward(ctx.model, ctx.teacher, ctx.route, frame)
    loss = values["kd"] if objective == "kd" else values["native"]["loss"]
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        ctx.student.parameters(), ctx.plan["optimizer"]["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    ctx.optimizer_steps_completed += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - tick
    groups = gradient_groups(ctx.student)
    receipt = evidence_receipt(values, planned["forward_seed"])
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite matched objective")
    ctx.lease.assert_storage()
    proto.assert_frozen(ctx.model)
    return {
        **planned,
        "condition": condition,
        "objective": objective,
        "optimizer_steps_completed": ctx.optimizer_steps_completed,
        "phase_optimizer_steps": planned["phase_step"],
        "teacher_velocity_kd": float(values["kd"].detach()),
        "native_loss": float(values["native"]["loss"].detach()),
        "objective_loss": float(loss.detach()),
        "gradient_norm_before_clip": float(norm),
        "gradient_groups": groups,
        "noise_receipt": receipt,
        "frame_content_sha256": frame["frame_content_sha256"],
        "batch_preparation_seconds": preparation,
        "clean_step_wall_seconds": wall,
        "elapsed_step_seconds": time.monotonic() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gradient_scope": "postclip groups outside clean timer",
        "cost_scope": "batch/cache preparation separately; clean includes actual B4 VL/q/action, Graph student, eager teacher for both conditions, backward/clip/AdamW; total includes receipts/statistics",
    }


def integrity(ctx):
    proto.assert_frozen(ctx.model)
    if proto.content_hash(ctx.model.state_dict()) != ctx.base_before:
        raise ValueError("Frozen base/periphery changed")
    ctx.cache.assert_readonly()
    size = ctx.budget()
    return {
        "base_before": ctx.base_before,
        "base_after": ctx.base_before,
        "cache_report_sha256": ctx.args.cache_report_sha256,
        "cache_readonly": True,
        "student_content_sha256": proto.content_hash(ctx.student.state_dict()),
        "stage_peak_allocated_bytes": ctx.stage_peak,
        "artifact_bytes": size,
    }


def preflight(ctx, optimizer):
    dev_initial = initial_dev_baseline(ctx)
    schedule = read(ctx.plan["schedule_file"])
    frame = assemble_batch(ctx.cache, schedule[0]["batch"])
    native = native_batch_check(ctx, frame, schedule[0]["forward_seed"])
    write_new(ctx.args.output / "NATIVE_B4_PARITY.json", native)
    create_batch_graph(ctx, frame, schedule[0]["forward_seed"])
    checks = []
    before = proto.content_hash(ctx.student.state_dict())
    with (ctx.args.output / "training_steps.jsonl").open("x") as stream:
        for completed in range(4):
            if completed in ctx.plan["preflight_update_checks"]:
                check = batch_checks(
                    ctx, schedule, completed, ctx.args.output / f"update-{completed}"
                )
                checks.append(check)
                if not check["passed"]:
                    release_batch(ctx)
                    return {
                        "status": "failed",
                        "continuation_gate": False,
                        "optimizer_steps": ctx.optimizer_steps_completed,
                        "checks": checks,
                        "dev_initial": dev_initial,
                        "integrity": integrity(ctx),
                    }
            if completed < 3:
                row = update(ctx, optimizer, schedule[completed], "distill_then_task")
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    changed = proto.content_hash(ctx.student.state_dict()) != before
    optimizer.zero_grad(set_to_none=True)
    release_batch(ctx)
    deployment = b1_deployment(ctx)
    passed = changed and deployment["passed"] and all(r["passed"] for r in checks)
    return {
        "status": "passed" if passed else "failed",
        "continuation_gate": passed,
        "optimizer_steps": 3,
        "checks": checks,
        "dev_initial": dev_initial,
        "b1_deployment": deployment,
        "weight_content_changed": changed,
        "preflight_weights_discarded": True,
        "integrity": integrity(ctx),
    }


def evaluate_pairs(ctx, pairs, output):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from run_train_eval import physical_metrics

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    arrays = {
        key: []
        for key in (
            "student_actions",
            "teacher_actions",
            "student_physical",
            "teacher_physical",
            "physical_targets",
            "student_velocity",
            "teacher_velocity",
            "action_masks",
        )
    }
    processor = ctx.source.processor
    params = processor.state_action_processor.norm_params["new_embodiment"]["action"]
    write_new(
        output / "NORMALIZATION.json",
        {
            "representation": "ABSOLUTE",
            "normalization": "minmax",
            "order": ["left_arm7", "left_hand6"],
            "min": np.concatenate(
                [np.asarray(params[k]["min"]) for k in ("left_arm", "left_hand")]
            ).tolist(),
            "max": np.concatenate(
                [np.asarray(params[k]["max"]) for k in ("left_arm", "left_hand")]
            ).tolist(),
            "active_horizon": 16,
        },
    )
    for pair in pairs:
        ctx.budget()
        ep, a, nominal = pair["episode"], pair["anchor"], pair["nominal_seed"]
        seed = deterministic_seed("diagnostic", ep, a, nominal)
        frame = ctx.cache.frame(ep, a)
        raw = raw_record(frame)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            values = capture_velocity(
                ctx.model,
                frame,
                seed,
                student=ctx.student,
                teacher=ctx.teacher,
                replay_teacher=True,
            )
            teacher_native, teacher_receipt = native_flow(ctx.model, frame, None, seed)
        if values["receipt"] != teacher_receipt:
            raise ValueError("B1 student/teacher actual flow inputs differ")
        actual = cached_action(ctx.model, frame, seed, student=ctx.student)
        teacher = cached_action(ctx.model, frame, seed)
        _, identity = proto.action_record(actual, raw, processor)
        proto.action_record(teacher, raw, processor)
        prediction = actual[0].float().numpy()
        teacher_prediction = teacher[0].float().numpy()
        physical = []
        for value in (prediction, teacher_prediction):
            decoded = processor.decode_action(value, EmbodimentTag.NEW_EMBODIMENT, state=raw.states)
            physical.append(np.concatenate([decoded[k] for k in ("left_arm", "left_hand")], -1))
        metrics = physical_metrics(prediction, raw, processor)
        teacher_metrics = {
            "flow_loss": float(teacher_native["loss"]),
            **physical_metrics(teacher_prediction, raw, processor),
        }
        row = {
            "episode": ep,
            "anchor": a,
            "seed": nominal,
            "paired_rng_seed": seed,
            "flow_loss": float(values["native"]["loss"]),
            "teacher_velocity_kd": float(values["kd"]),
            "teacher_action_physical_mae": float(np.abs(physical[0] - physical[1]).mean()),
            **metrics,
            "teacher_metrics": teacher_metrics,
            "noise_receipt": values["receipt"],
            "teacher_noise_receipt": teacher_receipt,
            **identity,
        }
        if not all(np.isfinite(row[k]) for k in METRICS):
            raise ValueError("Nonfinite B1 evaluation metric")
        rows.append(row)
        for key, value in [
            ("student_actions", prediction),
            ("teacher_actions", teacher_prediction),
            ("student_physical", physical[0]),
            ("teacher_physical", physical[1]),
            (
                "physical_targets",
                np.concatenate([raw.actions[k] for k in ("left_arm", "left_hand")], -1),
            ),
            ("student_velocity", values["velocity"][0].float().cpu().numpy()),
            ("teacher_velocity", values["teacher_velocity"][0].float().cpu().numpy()),
            ("action_masks", values["native"]["action_mask"][0].float().cpu().numpy()),
        ]:
            arrays[key].append(value)
    episode, metrics = aggregate_metrics(rows, METRICS)
    teacher_rows = [
        {"episode": r["episode"], "anchor": r["anchor"], "seed": r["seed"], **r["teacher_metrics"]}
        for r in rows
    ]
    teacher_episode, teacher_metrics = aggregate_metrics(teacher_rows, METRICS[:4])
    for name, value in arrays.items():
        np.save(output / f"{name}.npy", np.stack(value))
    write_new(output / "details.json", rows)
    result = {
        "status": "passed",
        "rows": len(rows),
        "metrics": metrics,
        "per_episode": episode,
        "teacher_metrics": teacher_metrics,
        "teacher_per_episode": teacher_episode,
        "array_sha256": {name: sha(output / f"{name}.npy") for name in arrays},
        "details_sha256": sha(output / "details.json"),
        "normalization_sha256": sha(output / "NORMALIZATION.json"),
        "checkpoint_content_sha256": proto.content_hash(ctx.student.state_dict()),
        "cache_report_sha256": ctx.args.cache_report_sha256,
        "scope": "B1 original eager sampling; same pretraining-domain train/dev cohort, no final/reserve outputs",
    }
    write_new(output / "RESULT.json", result)
    return {
        "path": str((output / "RESULT.json").resolve()),
        "sha256": sha(output / "RESULT.json"),
        "result": result,
    }


def dev_pairs(plan):
    return [
        {"episode": ep, "anchor": a, "nominal_seed": seed}
        for ep in plan["dev"]
        for a in plan["eval_anchors"][str(ep)]
        for seed in plan["noise_seeds"]
    ]


def train(ctx, parent, pending_rng):
    optimizer = ctx.optimizer
    ctx.optimizer = None
    schedule = read(ctx.plan["schedule_file"])
    start = ctx.optimizer_steps_completed
    records = (
        {"dev-0": ctx.prerequisite["dev_initial"]} if parent is None else parent["evaluations"]
    )
    if records["dev-0"] != ctx.prerequisite["dev_initial"]:
        raise ValueError("Resume changed the shared fresh development baseline")
    seen = {} if parent is None else parent["gradient_groups_seen"]
    indexes = []
    if len(schedule) != 20000:
        raise ValueError("Exactly20000 matched schedule rows required")
    frame = assemble_batch(ctx.cache, schedule[start]["batch"])
    create_batch_graph(ctx, frame, schedule[start]["forward_seed"])
    if parent is None:
        records["train-0"] = evaluate_pairs(
            ctx, ctx.plan["train_probe_pairs"], ctx.args.output / "train-probe-step0"
        )
    else:
        restore_rng(pending_rng)
    active_phase = phase_for_step(start + 1)
    with (ctx.args.output / "training_steps.jsonl").open("x") as stream:
        for planned in schedule[start:]:
            ctx.budget()
            if (
                planned["step"] != ctx.optimizer_steps_completed + 1
                or planned["phase"] != phase_for_step(planned["step"])
                or planned["phase_step"] != planned["step"] - (planned["phase"] - 1) * 10000
            ):
                raise ValueError("Matched global/phase schedule progression changed")
            if planned["phase"] != active_phase:
                optimizer.zero_grad(set_to_none=True)
                del optimizer
                optimizer = optimizer_for(ctx.student, ctx.plan)
                active_phase = planned["phase"]
                write_new(
                    ctx.args.output / f"PHASE-{active_phase}-OPTIMIZER.json",
                    {
                        "phase": active_phase,
                        "fresh_state": not bool(optimizer.state),
                        "next_global_step": planned["step"],
                        "same_for_both_conditions": True,
                    },
                )
            row = update(ctx, optimizer, planned, ctx.args.condition)
            for group, value in row["gradient_groups"].items():
                seen[group] = seen.get(group, False) or value["finite_nonzero_parameters"] > 0
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            step = ctx.optimizer_steps_completed
            if step in ctx.plan["dev_probe_steps"]:
                records[f"dev-{step}"] = evaluate_pairs(
                    ctx, dev_pairs(ctx.plan), ctx.args.output / f"dev-step{step}"
                )
            if step in ctx.plan["train_probe_steps"]:
                records[f"train-{step}"] = evaluate_pairs(
                    ctx, ctx.plan["train_probe_pairs"], ctx.args.output / f"train-probe-step{step}"
                )
            if step in ctx.plan["weight_checkpoint_steps"]:
                indexes.append(save_checkpoint(ctx, optimizer, step, records, seen))
                ctx.budget()
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "condition": ctx.args.condition,
                            "step": step,
                            "phase": planned["phase"],
                            "objective": row["objective"],
                            "native_loss": row["native_loss"],
                            "kd": row["teacher_velocity_kd"],
                        }
                    ),
                    flush=True,
                )
    optimizer.zero_grad(set_to_none=True)
    changes = parameter_changes(ctx.student, ctx.initial, seen)
    release_batch(ctx)
    return {
        "status": "completed",
        "technical_validity": "passed",
        "training_completed": ctx.optimizer_steps_completed == 20000,
        "condition": ctx.args.condition,
        "optimizer_steps": ctx.optimizer_steps_completed,
        "attempt_optimizer_steps": ctx.optimizer_steps_completed - start,
        "starting_step": start,
        "parent_index": ctx.parent_index,
        "evaluations": records,
        "checkpoints": indexes,
        "changes": changes,
        "quality_failure_stops_training": False,
        "integrity": integrity(ctx),
    }


def joint_evaluate(args, plan):
    results = {}
    details = {}
    teacher_arrays = []
    step_logs = {}
    terminal_identities = {}
    for condition in plan["conditions"]:
        override = getattr(args, condition + "_terminal", None)
        override_sha = getattr(args, condition + "_terminal_sha256", None)
        if (override is None) != (override_sha is None):
            raise ValueError("Explicit terminal path and SHA required together")
        terminal_path = (
            Path(override)
            if override
            else args.artifact_root / plan["train_run_ids"][condition] / "output/RESULT.json"
        )
        if override_sha is not None and sha(terminal_path) != override_sha:
            raise ValueError("Explicit training terminal SHA mismatch")
        terminal = read(terminal_path)
        if (
            terminal.get("condition") != condition
            or not terminal.get("training_completed")
            or terminal["optimizer_steps"] != 20000
            or terminal["plan_sha256"] != PLAN_SHA
            or terminal["cache_report_sha256"] != args.cache_report_sha256
        ):
            raise ValueError("Two complete frozen matched conditions required")
        record = terminal["evaluations"]["dev-20000"]
        path = Path(record["path"])
        if sha(path) != record["sha256"]:
            raise ValueError("Final development result changed")
        value = read(path)
        final_checkpoint = terminal["checkpoints"][-1]
        if sha(final_checkpoint["path"]) != final_checkpoint["sha256"]:
            raise ValueError("Final checkpoint index changed")
        final_index = read(final_checkpoint["path"])
        if (
            final_index["completed_step"] != 20000
            or value["checkpoint_content_sha256"] != final_index["student_content_sha256"]
            or sha(final_index["weights"]["path"]) != final_index["weights"]["sha256"]
        ):
            raise ValueError("Final development/checkpoint identity mismatch")
        terminal_identities[condition] = {
            "path": str(terminal_path.resolve()),
            "sha256": sha(terminal_path),
            "source_commit": terminal["source_commit"],
        }
        results[condition] = value
        if (
            sha(path.parent / "details.json") != value["details_sha256"]
            or sha(path.parent / "NORMALIZATION.json") != value["normalization_sha256"]
        ):
            raise ValueError("Final evaluation details/normalization changed")
        details[condition] = read(path.parent / "details.json")
        for name, digest in value["array_sha256"].items():
            if sha(path.parent / f"{name}.npy") != digest:
                raise ValueError("Final evaluation array changed")
        teacher_arrays.append(np.load(path.parent / "teacher_actions.npy"))
        step_logs[condition] = checkpoint_log_rows(
            terminal["checkpoints"][-1],
            {
                "condition": condition,
                "source_commit": terminal["source_commit"],
                "plan_sha256": PLAN_SHA,
                "cache_report_sha256": args.cache_report_sha256,
                "schedule_sha256": plan["schedule_sha256"],
            },
        )
    if len({v["source_commit"] for v in terminal_identities.values()}) != 1:
        raise ValueError("Matched conditions used different implementation commits")
    if not np.array_equal(*teacher_arrays):
        raise ValueError("Matched conditions have different frozen teacher sample reference")
    expected_pairs = dev_pairs(plan)
    left_details, right_details = (details[c] for c in plan["conditions"])
    if len(left_details) != len(expected_pairs) or len(right_details) != len(expected_pairs):
        raise ValueError("Incomplete final B1 development outputs")
    for pair, a, b in zip(expected_pairs, left_details, right_details):
        expected = (pair["episode"], pair["anchor"], pair["nominal_seed"])
        if any((v["episode"], v["anchor"], v["seed"]) != expected for v in (a, b)):
            raise ValueError("Final development input order changed")
        if a["teacher_metrics"] != b["teacher_metrics"] or a["noise_receipt"] != b["noise_receipt"]:
            raise ValueError("Final development teacher/native draw reference differs")
    schedule = read(plan["schedule_file"])
    left, right = (step_logs[c] for c in plan["conditions"])
    if len(left) != 20000 or len(right) != 20000:
        raise ValueError("Joint evaluator requires complete paired global-step logs")
    paired_keys = (
        "continuous_time",
        "discrete_time",
        "noisy_action",
        "post_condition",
        "state_features",
        "target",
        "mask",
        "core_inputs",
    )
    for planned, a, b in zip(schedule, left, right):
        if any(a.get(k) != v or b.get(k) != v for k, v in planned.items()):
            raise ValueError("Matched schedule/LR/batch pairing changed")
        if any(a["noise_receipt"][k] != b["noise_receipt"][k] for k in paired_keys):
            raise ValueError(
                "Actual native batch noise/time/input receipts differ across conditions"
            )
    task, distilled = (results[c] for c in plan["conditions"])
    gates = {}
    limits = plan["dev_gates"]
    for key in ("flow_loss", "combined_mae_mixed_units"):
        ratio = distilled["metrics"][key] / task["metrics"][key]
        better = sum(
            distilled["per_episode"][str(ep)][key] < task["per_episode"][str(ep)][key]
            for ep in plan["dev"]
        )
        bound = (
            limits["max_distilled_task_flow_ratio"]
            if key == "flow_loss"
            else limits["max_distilled_task_combined_ratio"]
        )
        gates[key] = {
            "ratio": ratio,
            "episodes_better": better,
            "passed": ratio <= bound and better >= limits["min_episodes_better"],
        }
    retention = {
        key: distilled["metrics"][key]
        <= distilled["teacher_metrics"][key] * limits["max_native_metric_ratio"]
        for key in limits["native_metrics"]
    }
    return {
        "status": "completed",
        "technical_validity": "passed",
        "results": results,
        "terminals": terminal_identities,
        "matched_gates": gates,
        "native_retention": retention,
        "quality_gate": all(row["passed"] for row in gates.values()) and all(retention.values()),
        "paired_steps": 20000,
        "paired_samples": 80000,
        "teacher_actions_exact": True,
        "optimizer_steps": 0,
        "scope": "Zero-GPU joint review of final B1 dev evidence; no final/reserve or deployment-cost claims",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--condition", choices=("task_only", "distill_then_task"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-report-sha256", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--resume-index", type=Path)
    parser.add_argument("--resume-index-sha256")
    for condition in ("task_only", "distill_then_task"):
        parser.add_argument("--" + condition.replace("_", "-") + "-terminal", type=Path)
        parser.add_argument("--" + condition.replace("_", "-") + "-terminal-sha256")
    args = parser.parse_args()
    plan = verify_plan(args)
    limit = (
        plan["train_internal_wall_seconds"]
        if args.command == "train"
        else plan["preflight_internal_wall_seconds"]
    )
    if not 0 < args.wall_seconds <= limit:
        parser.error("Frozen stage wall bound exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ctx = None
    completed = 0
    try:
        if args.command == "evaluate":
            result = joint_evaluate(args, plan)
        else:
            ctx, parent, pending_rng = context_for(args, plan, started + args.wall_seconds)
            source_commit = ctx.source_commit
            result = (
                preflight(ctx, ctx.optimizer)
                if args.command == "preflight"
                else train(ctx, parent, pending_rng)
            )
            result["source_commit"] = source_commit
            completed = ctx.optimizer_steps_completed
            ctx = None
            del parent, pending_rng
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            result["gpu_after_release"] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
        result.update(
            plan_sha256=PLAN_SHA,
            cache_report_sha256=args.cache_report_sha256,
            run_id=args.run_id,
            elapsed_seconds=time.monotonic() - started,
        )
        write_new(args.output / "RESULT.json", result)
    except Exception as exc:
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "technical_validity": "not_established",
                "plan_sha256": PLAN_SHA,
                "cache_report_sha256": args.cache_report_sha256,
                "exception": type(exc).__name__,
                "message": str(exc),
                "optimizer_steps_completed": completed
                if ctx is None
                else ctx.optimizer_steps_completed,
                "automatic_retry": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    if args.command == "preflight" and not result["continuation_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
