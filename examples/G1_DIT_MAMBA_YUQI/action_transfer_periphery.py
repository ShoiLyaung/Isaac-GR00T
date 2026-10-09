"""Bounded native-periphery adaptation from the valid task-only 50k terminal."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

from action_transfer_continuation import compare_reproduction, file_record_valid
import action_transfer_matched_runner as matched
from action_transfer_matched_support import MMapCache, assemble_batch, log_prefix, masked_batch_kd
from action_transfer_periphery_support import (
    CONDITIONS,
    PARENT_SEMANTIC,
    NativeStudent,
    adaptation_lr,
    capture_graph,
    eager_route,
    gradient_mapping,
    gradient_norms,
    make_optimizer,
    optimizer_steps,
    receipt,
    storage_inventory,
    task_forward,
)
from action_transfer_program_support import (
    cpu_tree,
    optimizer_names,
    restore_rng,
    rng_state,
    save_tensor_new,
)
from budgeted_action_cudagraph import TENSOR_KEYS
from budgeted_action_cudagraph_probe import gradient_comparison, snapshot_gradients
from budgeted_action_support import cached_action
from budgeted_graph_distillation import dropout_inventory
from budgeted_graph_distillation_runner import environment
from contract import sha, write_new
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
from integrity_diagnosis_support import BaseAudit
import memory_prototype as proto
from memory_training import raw_record
from memory_training_support import native_flow
import numpy as np
from run_train_eval import NativeAnchors, strict_reload
from runner_core import aggregate_metrics, deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
METRICS = matched.METRICS


def read(path):
    return json.loads(Path(path).read_text())


def verify_plan(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Explicit periphery PLAN SHA mismatch")
    plan = read(args.plan)
    if (
        Path(plan["source_worktree"]).resolve() != REPO
        or Path(plan["artifact_root"]).resolve() != args.artifact_root.resolve()
        or (plan["starting_step"], plan["training_steps"], plan["batch_size"]) != (50000, 70000, 4)
        or plan["conditions"] != list(CONDITIONS)
        or plan["dev_probe_steps"] != [55000, 60000, 65000, 70000]
        or plan["train_probe_steps"] != [60000, 70000]
        or plan["weight_checkpoint_steps"] != [70000]
        or plan["sample_offset"] != 200000
    ):
        raise ValueError("Periphery protocol identity changed")
    options = plan["optimizer"]
    if (
        options["betas"] != [0.9, 0.95]
        or options["eps"] != 1e-8
        or options["matrix_weight_decay"] != 1e-4
        or options["bias_norm_no_weight_decay"] != 0
        or options["gradient_clip"] != 1
        or options["lr"] != 1e-5
        or options["cosine_min_lr"] != 1e-6
        or options["warmup_steps"] != 0
    ):
        raise ValueError("Frozen optimizer contract changed")
    if plan["max_artifact_bytes"] > 16 * 2**30 or plan["max_allocated_bytes"] > 44 * 2**30:
        raise ValueError("Periphery resource cap expanded")
    for row in plan["records"]:
        file_record_valid(row)
    identities = {Path(row["path"]).resolve(): row["sha256"] for row in plan["records"]}
    for relative in plan["immutable_source_relative_paths"]:
        path = (REPO / relative).resolve()
        if identities.get(path) != sha(path):
            raise ValueError(f"Original source changed: {relative}")
    if sha(plan["schedule_file"]) != plan["schedule_sha256"]:
        raise ValueError("Shared adaptation schedule changed")
    schedule = read(plan["schedule_file"])
    if len(schedule) != 20000:
        raise ValueError("Expected exact20000-row adaptation suffix")
    for i, row in enumerate(schedule):
        if (
            row["step"] != 50001 + i
            or row["lr"] != adaptation_lr(row["step"])
            or len(row["batch"]) != 4
        ):
            raise ValueError("Frozen adaptation progression/LR differs")
        if any(
            p["episode"] not in plan["train"] or p["anchor"] not in plan["grids"][str(p["episode"])]
            for p in row["batch"]
        ):
            raise ValueError("Schedule escaped training cohort")
    expected_run = (
        plan["run_ids"][args.command]
        if args.command != "train"
        else plan["run_ids"]["train"][args.condition]
    )
    if (
        args.run_id != expected_run
        or args.output.resolve() != args.artifact_root.resolve() / args.run_id / "output"
    ):
        raise ValueError("Immutable run/output identity changed")
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_id):
        raise ValueError("Expected simple run identity")
    if args.command == "train":
        record = {"path": str(args.preflight_result), "sha256": args.preflight_sha256}
        file_record_valid(record)
        prior = read(record["path"])
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if (
            prior["status"] != "passed"
            or not prior["continuation_gate"]
            or prior["plan_sha256"] != args.plan_sha256
            or prior["source_commit"] != commit
        ):
            raise ValueError("Passing same-source/PLAN periphery preflight required")
    args.cache_report_sha256 = plan["cache_report_sha256"]
    return plan


def load_parent(plan):
    parent = plan["parent"]
    for key in ("index", "terminal"):
        file_record_valid(parent[key])
    index, terminal = read(parent["index"]["path"]), read(parent["terminal"]["path"])
    if (
        index["completed_step"] != 50000
        or index["condition"] != "task_only"
        or index["student_content_sha256"] != PARENT_SEMANTIC
        or index["source_commit"] != "fca9b2006850d3b9b7823ff3cd0bf3849b08ba4e"
        or not terminal["training_completed"]
        or terminal["status"] != "completed"
        or terminal["technical_validity"] != "passed"
        or index["cache_report_sha256"] != plan["cache_report_sha256"]
    ):
        raise ValueError("Only the valid task-only50k terminal is a parent")
    for key in ("weights", "optimizer_rng"):
        file_record_valid(index[key])
    payload = torch.load(
        index["optimizer_rng"]["path"], map_location="cpu", weights_only=True, mmap=True
    )
    for key in ("completed_step", "condition", "source_commit", "plan_sha256"):
        if payload[key] != index[key]:
            raise ValueError("Parent optimizer/index metadata mismatch")
    if payload["phase_optimizer_steps"] != 40000:
        raise ValueError("Parent AdamW must continue from40000")
    weights = torch.load(index["weights"]["path"], map_location="cpu", weights_only=True, mmap=True)
    if proto.content_hash(weights) != PARENT_SEMANTIC:
        raise ValueError("Parent semantic identity mismatch")
    return index, payload, weights


def context_for(args, plan, deadline, condition):
    source = environment(plan)
    index, payload, weights = load_parent(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    ctx = SimpleNamespace(
        args=args,
        plan=plan,
        condition=condition,
        source_commit=source["commit"],
        deadline=deadline,
        lease=None,
        route=None,
        samples=None,
        completed_step=50000,
        stage_peak=0,
        parent_index=index,
        parent_payload=payload,
        initial_core=weights,
    )

    def budget():
        ctx.stage_peak = max(ctx.stage_peak, torch.cuda.max_memory_allocated())
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if time.monotonic() > deadline:
            raise TimeoutError("Periphery stage wall bound exceeded")
        if size > plan["max_artifact_bytes"] or ctx.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Periphery resource bound exceeded")
        return size

    ctx.budget = budget
    ctx.output = args.output / condition if args.command == "preflight" else args.output
    if args.command == "preflight":
        ctx.output.mkdir(exist_ok=False)
    ctx.cache = MMapCache(plan["cache_root"], plan, plan["cache_report_sha256"], budget)
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    ctx.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    ctx.audit = BaseAudit(model, ctx.output, plan["base_content_sha256"], budget=budget)
    if ctx.cache.report["integrity"]["base_before"] != ctx.audit.full_hash:
        raise ValueError("Raw cache/base identity differs")
    core = BudgetedActionMamba(model.action_head.model)
    core.load_state_dict(weights, strict=True)
    ctx.student = NativeStudent(model.action_head, core.cuda(), condition)
    ctx.optimizer = make_optimizer(ctx.student, plan, payload)
    ctx.initial_periphery = cpu_tree(ctx.student.periphery.state_dict())
    inventory = storage_inventory(ctx.student, model)
    if (
        inventory["core_parameters"],
        inventory["periphery_parameters"],
        len(list(ctx.student.core.parameters())),
        len(list(ctx.student.periphery.parameters())),
    ) != (230785600, 515142048, 158, 81):
        raise ValueError("Frozen core/periphery parameter inventory changed")
    if inventory["two_condition_payload_bytes"] > plan["max_artifact_bytes"]:
        raise ValueError("Final checkpoints exceed storage cap before proof")
    ctx.audit.extra = lambda: dict(ctx.student.named_parameters())
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    write_new(
        ctx.output / "IDENTITY.json",
        {
            "source": source,
            "strict_loading": loading,
            "parent": plan["parent"],
            "core_semantic_sha256": proto.content_hash(ctx.student.core.state_dict()),
            "periphery_semantic_sha256": proto.content_hash(ctx.student.periphery.state_dict()),
            "base_semantic_sha256": ctx.audit.full_hash,
            "inventory": inventory,
            "dropout": dropout_inventory(ctx.student),
        },
    )
    return ctx


def create_graph(ctx, planned):
    frame = assemble_batch(ctx.cache, planned["batch"])
    set_seed(planned["forward_seed"], cuda=True)
    ctx.lease, ctx.route, ctx.samples = capture_graph(ctx.student, frame)
    write_new(
        ctx.output / "BATCH_CAPTURE.json",
        {
            **ctx.lease.setup,
            "input_signature": [
                {
                    "key": k,
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "requires_grad": v.requires_grad,
                }
                for k, v in zip(TENSOR_KEYS, ctx.samples)
            ],
        },
    )
    ctx.student.zero_grad(set_to_none=True)
    ctx.audit.check("after-capture", stable=True)


def release_graph(ctx):
    ctx.student.zero_grad(set_to_none=True)
    if ctx.lease is not None:
        ctx.lease.close()
        ctx.lease = ctx.route = ctx.samples = None
    gc.collect()
    torch.cuda.empty_cache()
    ctx.audit.check("after-graph-release", stable=True)


def batch_check(ctx, planned, label):
    frame = assemble_batch(ctx.cache, planned["batch"])
    data = {}
    for name, route in (("eager", eager_route(ctx.student)), ("graph", ctx.route)):
        ctx.student.zero_grad(set_to_none=True)
        set_seed(planned["forward_seed"], cuda=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = task_forward(ctx.student, route, frame, retain_inputs=True)
        loss = values["native"]["loss"]
        loss.backward()
        data[name] = {
            "loss": float(loss.detach()),
            "receipt": receipt(values, planned["forward_seed"]),
            "velocity": values["velocity"].detach().cpu(),
            "parameters": snapshot_gradients(ctx.student),
            "inputs": {
                k: None if not v.requires_grad or v.grad is None else v.grad.detach().cpu().clone()
                for k, v in values["live_inputs"].items()
            },
        }
        del values, loss
    if data["eager"]["receipt"] != data["graph"]["receipt"]:
        raise ValueError("Eager/Graph actual native draws differ")
    gradients = gradient_comparison(
        data["eager"]["parameters"],
        data["graph"]["parameters"],
        gradient_mapping(ctx.student),
        ctx.plan["gradient_limits"],
    )
    inputs = gradient_comparison(
        data["eager"]["inputs"],
        data["graph"]["inputs"],
        {k: k for k in TENSOR_KEYS},
        ctx.plan["gradient_limits"],
    )
    if ctx.condition == "joint_periphery" and any(
        data["graph"]["inputs"][k] is None or not bool(data["graph"]["inputs"][k].abs().sum() > 0)
        for k in TENSOR_KEYS[:2]
    ):
        raise ValueError("Joint graph did not return nonzero live hidden/VL gradients")
    limit = ctx.plan["loss_limits"]
    loss_ok = abs(data["eager"]["loss"] - data["graph"]["loss"]) <= limit["atol"] + limit[
        "rtol"
    ] * abs(data["eager"]["loss"])
    norms = gradient_norms(ctx.student)
    expected = [k for k, m in ctx.student.periphery.items() if any(True for _ in m.parameters())]
    if ctx.condition == "joint_periphery" and any(norms["periphery." + k] <= 0 for k in expected):
        raise ValueError("A reachable joint periphery module has no gradient")
    result = {
        "passed": gradients["passed"] and inputs["passed"] and loss_ok,
        "parameters": gradients,
        "live_inputs": inputs,
        "norms": norms,
        "losses": {k: v["loss"] for k, v in data.items()},
        "receipt": data["eager"]["receipt"],
    }
    write_new(ctx.output / f"BATCH_CHECK_{label}.json", result)
    ctx.student.zero_grad(set_to_none=True)
    del data
    ctx.budget()
    if not result["passed"]:
        raise ValueError("Periphery eager/Graph compatibility failed")
    return result


def flow_capture(model, frame, seed, route=None):
    captured = []
    with ExitStack() as scope:
        if route is not None:
            scope.enter_context(patch.object(model.action_head, "model", route))
        scope.callback(
            model.action_head.action_decoder.register_forward_hook(
                lambda _m, _a, out: captured.append(out[:, -40:])
            ).remove
        )
        native, draw = native_flow(model, frame, None, seed)
    if len(captured) != 1:
        raise ValueError("Native decoder cadence changed")
    return native, draw, captured[0]


def evaluation_values(ctx, frame, seed):
    view = ctx.student.model_view()
    native, draw, velocity = flow_capture(view, frame, seed, eager_route(ctx.student))
    teacher, teacher_draw, teacher_velocity = flow_capture(ctx.model, frame, seed)
    for key in ("continuous_time", "discrete_time", "noisy_action", "forward_seed"):
        if draw[key] != teacher_draw[key]:
            raise ValueError("Independent native teacher/student random inputs differ")
    values = {
        "native": native,
        "receipt": draw,
        "velocity": velocity,
        "teacher_velocity": teacher_velocity,
        "kd": masked_batch_kd(velocity, teacher_velocity, native["action_mask"]),
    }
    return values, teacher, teacher_draw


def evaluate_pairs(ctx, pairs, output):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from run_train_eval import physical_metrics

    ctx.audit.check("before-probe:" + output.name, stable=True)
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
            values, teacher_native, teacher_receipt = evaluation_values(ctx, frame, seed)
        actual = cached_action(ctx.student.model_view(), frame, seed, student=ctx.student.core)
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
        "checkpoint_content_sha256": proto.content_hash(ctx.student.core.state_dict()),
        "cache_report_sha256": ctx.args.cache_report_sha256,
        "scope": "B1 original eager sampling; same pretraining-domain train/dev cohort, no final/reserve outputs",
    }
    result["periphery_content_sha256"] = proto.content_hash(ctx.student.periphery.state_dict())
    ctx.audit.check("after-probe:" + output.name, stable=True)
    write_new(output / "RESULT.json", result)
    return {
        "path": str((output / "RESULT.json").resolve()),
        "sha256": sha(output / "RESULT.json"),
        "result": result,
    }


def deployment_parity(ctx, label):
    rows = []
    for pair in ctx.plan["inference_probes"]:
        frame = ctx.cache.frame(pair["episode"], pair["anchor"])
        seed = pair["seed"]
        actual = cached_action(ctx.student.model_view(), frame, seed, student=ctx.student.core)
        with ctx.student.materialized() as model:
            deployed = cached_action(model, frame, seed, student=ctx.student.core)
        passed = torch.equal(actual, deployed)
        rows.append({**pair, "exact": passed})
    result = {"passed": all(r["exact"] for r in rows), "rows": rows}
    write_new(ctx.output / f"DEPLOYMENT_{label}.json", result)
    if not result["passed"]:
        raise ValueError("BF16 deployment materialization differs from differentiable views")
    return result


def update(ctx, planned):
    ctx.budget()
    if planned["step"] != ctx.completed_step + 1:
        raise ValueError("Optimizer progression changed")
    frame = assemble_batch(ctx.cache, planned["batch"])
    for group in ctx.optimizer.param_groups:
        group["lr"] = planned["lr"]
    ctx.optimizer.zero_grad(set_to_none=True)
    set_seed(planned["forward_seed"], cuda=True)
    ctx.lease.assert_storage()
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = task_forward(ctx.student, ctx.route, frame)
    loss = values["native"]["loss"]
    loss.backward()
    norms = gradient_norms(ctx.student)
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in ctx.student.parameters() if p.requires_grad], 1, error_if_nonfinite=True
    )
    ctx.optimizer.step()
    ctx.completed_step += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - tick
    ctx.lease.assert_storage()
    # Content hashes run only at declared boundaries, never on every forward.
    proto.assert_frozen(ctx.model)
    return {
        **planned,
        "condition": ctx.condition,
        "objective": "task",
        "native_loss": float(loss.detach()),
        "core_optimizer_step": ctx.completed_step - 10000,
        "periphery_optimizer_step": ctx.completed_step - 50000
        if ctx.condition == "joint_periphery"
        else 0,
        "gradient_norm_before_clip": float(norm),
        "clip_applied": float(norm) > 1,
        "gradient_norms_before_clip": norms,
        "noise_receipt": receipt(values, planned["forward_seed"]),
        "frame_content_sha256": frame["frame_content_sha256"],
        "step_seconds": wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def parameter_changes(ctx):
    changed = {}
    for name, module in ctx.student.periphery.items():
        prefix = name + "."
        changed[name] = any(
            not torch.equal(v.detach().cpu(), ctx.initial_periphery[prefix + k])
            for k, v in module.state_dict().items()
        )
    active = [
        name
        for name, module in ctx.student.periphery.items()
        if any(True for _ in module.parameters())
    ]
    expected = ctx.condition == "joint_periphery"
    if any(changed[name] != expected for name in active):
        raise ValueError("Frozen or trainable periphery update witness failed")
    return changed


def save_checkpoint(ctx, root, evaluations):
    root.mkdir(exist_ok=False)
    added = ctx.completed_step - 50000
    ctx.audit.check("before-checkpoint", stable=True)
    state_report = optimizer_steps(ctx.student, ctx.optimizer, added)
    weights = save_tensor_new(root / "student.pt", cpu_tree(ctx.student.state_dict()))
    ctx.budget()
    opt = save_tensor_new(
        root / "optimizer-rng.pt",
        {
            "optimizer": cpu_tree(ctx.optimizer.state_dict()),
            "optimizer_parameter_names": optimizer_names(ctx.student, ctx.optimizer),
            "rng": rng_state(),
            "completed_step": ctx.completed_step,
            "added_steps": added,
            "condition": ctx.condition,
            "plan_sha256": ctx.args.plan_sha256,
            "source_commit": ctx.source_commit,
        },
    )
    index = {
        "condition": ctx.condition,
        "completed_step": ctx.completed_step,
        "added_steps": added,
        "weights": weights,
        "optimizer_rng": opt,
        "student_content_sha256": proto.content_hash(ctx.student.state_dict()),
        "core_content_sha256": proto.content_hash(ctx.student.core.state_dict()),
        "periphery_content_sha256": proto.content_hash(ctx.student.periphery.state_dict()),
        "optimizer_state": state_report,
        "evaluations": evaluations,
        "source_commit": ctx.source_commit,
        "plan_sha256": ctx.args.plan_sha256,
        "cache_report_sha256": ctx.plan["cache_report_sha256"],
        "schedule_sha256": ctx.plan["schedule_sha256"],
        "parent": ctx.plan["parent"],
    }
    write_new(root / "checkpoint.json", index)
    ctx.budget()
    ctx.audit.check("after-checkpoint", stable=True)
    return {
        "path": str((root / "checkpoint.json").resolve()),
        "sha256": sha(root / "checkpoint.json"),
    }


def reload_checkpoint(ctx, record, planned, label):
    """Exact CPU identity, names/step/RNG restoration and a fresh GPU eager replay."""
    file_record_valid(record)
    index = read(record["path"])
    for key in ("weights", "optimizer_rng"):
        file_record_valid(index[key])
    weights = torch.load(index["weights"]["path"], map_location="cpu", weights_only=True, mmap=True)
    payload = torch.load(
        index["optimizer_rng"]["path"], map_location="cpu", weights_only=True, mmap=True
    )
    if (
        proto.content_hash(weights) != index["student_content_sha256"]
        or set(weights) != set(ctx.student.state_dict())
        or payload["optimizer_parameter_names"] != optimizer_names(ctx.student, ctx.optimizer)
        or any(
            payload[k] != index[k]
            for k in ("completed_step", "added_steps", "condition", "source_commit", "plan_sha256")
        )
    ):
        raise ValueError("Composite CPU checkpoint restore identity mismatch")
    frame = assemble_batch(ctx.cache, planned["batch"])

    def replay():
        set_seed(planned["forward_seed"], cuda=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            values = task_forward(ctx.student, eager_route(ctx.student), frame)
        return (
            values["velocity"].detach().cpu(),
            float(values["native"]["loss"]),
            receipt(values, planned["forward_seed"]),
        )

    before = replay()
    # Overwrite both core and periphery in place from CPU. Existing graph has
    # already been released; checkpoint contains every trainable/frozen owner.
    ctx.student.load_state_dict(weights, strict=True)
    ctx.optimizer.load_state_dict(payload["optimizer"])
    after = replay()
    restore_rng(payload["rng"])
    restored_rng = rng_state()
    rng_ok = (
        restored_rng["python"] == payload["rng"]["python"]
        and restored_rng["numpy"] == payload["rng"]["numpy"]
        and torch.equal(restored_rng["torch_cpu"], payload["rng"]["torch_cpu"])
        and all(
            torch.equal(a, b)
            for a, b in zip(restored_rng["torch_cuda"], payload["rng"]["torch_cuda"])
        )
    )
    passed = torch.equal(before[0], after[0]) and before[1:] == after[1:] and rng_ok
    result = {
        "passed": passed,
        "cpu_semantic_exact": True,
        "gpu_velocity_exact": torch.equal(before[0], after[0]),
        "rng_exact": rng_ok,
        "optimizer": optimizer_steps(ctx.student, ctx.optimizer, index["added_steps"]),
        "checkpoint": record,
    }
    write_new(ctx.output / f"RELOAD_{label}.json", result)
    ctx.audit.check("after-checkpoint-reload", stable=True)
    if not passed:
        raise ValueError("Composite GPU checkpoint reproduction failed")
    return result


def reset_parent(ctx):
    ctx.student.load_state_dict(
        {
            **{"core." + k: v for k, v in ctx.initial_core.items()},
            **{"periphery." + k: v for k, v in ctx.initial_periphery.items()},
        },
        strict=True,
    )
    ctx.optimizer = make_optimizer(ctx.student, ctx.plan, ctx.parent_payload)
    ctx.completed_step = 50000
    restore_rng(ctx.parent_payload["rng"])
    ctx.audit.check("restored-parent-after-proof", stable=True)
    return {
        "core_sha256": proto.content_hash(ctx.student.core.state_dict()),
        "periphery_sha256": proto.content_hash(ctx.student.periphery.state_dict()),
        "optimizer": optimizer_steps(ctx.student, ctx.optimizer, 0),
    }


def preflight_one(ctx, schedule):
    initial = evaluate_pairs(ctx, matched.dev_pairs(ctx.plan), ctx.output / "parent-dev")
    reproduction = compare_reproduction(ctx.parent_index["evaluations"]["dev-50000"], initial)
    create_graph(ctx, schedule[0])
    before = batch_check(ctx, schedule[0], "before")
    proof = [update(ctx, planned) for planned in schedule[:3]]
    after = batch_check(ctx, schedule[1], "after3")
    changes = parameter_changes(ctx)
    ctx.audit.check("after-discarded-proof-updates", stable=True)
    deployment = deployment_parity(ctx, "proof")
    release_graph(ctx)
    scratch = ctx.output / "proof-checkpoint-scratch"
    record = save_checkpoint(ctx, scratch, {})
    reload = reload_checkpoint(ctx, record, schedule[0], "proof")
    # Only the private scratch created immediately above is removed. Its exact
    # immutable file hashes/reload statistics remain in RELOAD_proof.json.
    shutil.rmtree(scratch)
    reset = reset_parent(ctx)
    if (
        reset["core_sha256"] != PARENT_SEMANTIC
        or reset["periphery_sha256"] != initial["result"]["periphery_content_sha256"]
    ):
        raise ValueError("Proof updates did not reset exactly to the common parent")
    ctx.cache.assert_readonly()
    step_seconds = float(np.median([row["step_seconds"] for row in proof[1:]]))
    # Whole stage bound also covers full dev/probes, load/capture and audit. A
    # measured projection reports infeasibility rather than changing the task.
    projection = step_seconds * 20000
    if projection >= ctx.plan["train_internal_wall_seconds"]:
        raise ValueError("Measured20000-update projection exceeds frozen training wall cap")
    return {
        "passed": True,
        "reproduction": reproduction,
        "batch_before": before["passed"],
        "batch_after": after["passed"],
        "proof_updates_discarded": 3,
        "proof_rows": proof,
        "changes": changes,
        "deployment": deployment,
        "reload": reload,
        "restored_parent": reset,
        "measured_step_seconds": step_seconds,
        "projected_update_seconds": projection,
        "peak_allocated_bytes": ctx.stage_peak,
        "frozen_base_checks": ctx.audit.checks,
    }


def train(ctx, schedule):
    initial = evaluate_pairs(ctx, matched.dev_pairs(ctx.plan), ctx.output / "parent-dev")
    compare_reproduction(ctx.parent_index["evaluations"]["dev-50000"], initial)
    create_graph(ctx, schedule[0])
    batch_check(ctx, schedule[0], "restored-parent")
    restore_rng(ctx.parent_payload["rng"])
    evaluations = {}
    clipped = 0
    tick = time.monotonic()
    with (ctx.output / "training_steps.jsonl").open("x") as stream:
        for planned in schedule:
            row = update(ctx, planned)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            clipped += int(row["clip_applied"])
            step = ctx.completed_step
            if step % 1000 == 0:
                ctx.audit.check(f"step{step}", stable=True)
            if step in ctx.plan["train_probe_steps"]:
                evaluations[f"train-{step}"] = evaluate_pairs(
                    ctx, ctx.plan["train_probe_pairs"], ctx.output / f"train-probe-step{step}"
                )
            if step in ctx.plan["dev_probe_steps"]:
                evaluations[f"dev-{step}"] = evaluate_pairs(
                    ctx, matched.dev_pairs(ctx.plan), ctx.output / f"dev-step{step}"
                )
    training_seconds = time.monotonic() - tick
    changes = parameter_changes(ctx)
    batch_check(ctx, schedule[-1], "terminal")
    deployment = deployment_parity(ctx, "terminal")
    release_graph(ctx)
    checkpoint = save_checkpoint(ctx, ctx.output / "checkpoint-step70000", evaluations)
    reload = reload_checkpoint(ctx, checkpoint, schedule[-1], "terminal")
    ctx.cache.assert_readonly()
    ctx.audit.check("terminal-release", stable=True)
    return {
        "status": "passed",
        "technical_validity": "passed",
        "training_completed": True,
        "condition": ctx.condition,
        "optimizer_steps": 70000,
        "added_steps": 20000,
        "evaluations": evaluations,
        "checkpoint": checkpoint,
        "reload": reload,
        "deployment": deployment,
        "periphery_changes": changes,
        "training_log": log_prefix(ctx.output / "training_steps.jsonl"),
        "clip_count": clipped,
        "clip_frequency": clipped / 20000,
        "training_seconds": training_seconds,
        "frozen_base_checks": ctx.audit.checks,
        "peak_allocated_bytes": ctx.stage_peak,
        "artifact_bytes": ctx.budget(),
    }


def paired_bootstrap(left, right, episodes):
    rng = np.random.default_rng(20261009)
    indices = rng.integers(0, len(episodes), size=(10000, len(episodes)))
    result = {}
    for metric in METRICS:
        a = np.asarray([left[str(ep)][metric] for ep in episodes])
        b = np.asarray([right[str(ep)][metric] for ep in episodes])
        differences = b - a
        samples = differences[indices].mean(axis=1)
        result[metric] = {
            "core_only_mean": float(a.mean()),
            "joint_mean": float(b.mean()),
            "joint_over_core_only": float(b.mean() / a.mean()),
            "episodes_better": int((differences < 0).sum()),
            "paired_episode_differences": {
                str(ep): float(v) for ep, v in zip(episodes, differences)
            },
            "mean_difference_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return result


def joint_evaluate(args, plan):
    terminals = []
    for condition, path, digest in (
        (CONDITIONS[0], args.core_only_result, args.core_only_sha256),
        (CONDITIONS[1], args.joint_result, args.joint_sha256),
    ):
        file_record_valid({"path": str(path), "sha256": digest})
        terminal = read(path)
        if (
            terminal["status"] != "passed"
            or terminal["technical_validity"] != "passed"
            or terminal["condition"] != condition
            or terminal["optimizer_steps"] != 70000
            or terminal["added_steps"] != 20000
            or terminal["plan_sha256"] != args.plan_sha256
        ):
            raise ValueError("Both valid fixed70k terminals required")
        file_record_valid(terminal["training_log"])
        file_record_valid(terminal["checkpoint"])
        index = read(terminal["checkpoint"]["path"])
        for key in ("weights", "optimizer_rng"):
            file_record_valid(index[key])
        for record in terminal["evaluations"].values():
            file_record_valid(record)
            result = read(record["path"])
            root = Path(record["path"]).parent
            for name, digest in result["array_sha256"].items():
                if sha(root / f"{name}.npy") != digest:
                    raise ValueError("Fixed70k evaluation arrays changed")
            for name, key in (
                ("details.json", "details_sha256"),
                ("NORMALIZATION.json", "normalization_sha256"),
            ):
                if sha(root / name) != result[key]:
                    raise ValueError("Evaluation input/normalization receipts changed")
        terminals.append(terminal)
    if terminals[0]["source_commit"] != terminals[1]["source_commit"]:
        raise ValueError("Matched periphery conditions used different sources")
    paired_keys = ("step", "batch", "forward_seed", "lr", "noise_receipt", "frame_content_sha256")
    with (
        open(terminals[0]["training_log"]["path"]) as a,
        open(terminals[1]["training_log"]["path"]) as b,
    ):
        count = 0
        while True:
            left, right = a.readline(), b.readline()
            if not left or not right:
                if left != right:
                    raise ValueError("Matched update count differs")
                break
            x, y = json.loads(left), json.loads(right)
            if any(x[k] != y[k] for k in paired_keys) or x["step"] != 50001 + count:
                raise ValueError("Actual paired schedule/noise/time/target receipts differ")
            count += 1
    if count != 20000:
        raise ValueError("Expected all20000 matched task updates")
    quality = [read(t["evaluations"]["dev-70000"]["path"]) for t in terminals]
    if any(q["rows"] != 512 for q in quality):
        raise ValueError("Expected full512 fixed terminal outputs")
    for key in (
        "teacher_actions",
        "teacher_velocity",
        "teacher_physical",
        "physical_targets",
        "action_masks",
    ):
        if quality[0]["array_sha256"][key] != quality[1]["array_sha256"][key]:
            raise ValueError("Independent B0 reference drifted across conditions")
    statistics = paired_bootstrap(quality[0]["per_episode"], quality[1]["per_episode"], plan["dev"])
    gate = all(
        statistics[k]["joint_over_core_only"] <= 0.98 and statistics[k]["episodes_better"] >= 6
        for k in ("flow_loss", "combined_mae_mixed_units")
    )
    return {
        "status": "passed",
        "technical_validity": "passed",
        "paired_updates": count,
        "statistics": statistics,
        "retain_joint_periphery_development_gate": gate,
        "bootstrap": {"unit": "paired episode", "replicates": 10000, "seed": 20261009},
        "source_commit": terminals[0]["source_commit"],
        "scope": "Development selection only; final/reserve and final100 simulation remain sealed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, choices=("preflight", "train", "evaluate"))
    parser.add_argument("--condition", choices=CONDITIONS)
    for name in (
        "plan",
        "artifact-root",
        "output",
        "preflight-result",
        "core-only-result",
        "joint-result",
    ):
        parser.add_argument(
            "--" + name, type=Path, required=name in ("plan", "artifact-root", "output")
        )
    for name in ("plan-sha256", "run-id", "preflight-sha256", "core-only-sha256", "joint-sha256"):
        parser.add_argument("--" + name, required=name in ("plan-sha256", "run-id"))
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    if args.command == "train" and (
        args.condition is None or args.preflight_result is None or args.preflight_sha256 is None
    ):
        parser.error("train requires condition and explicit passing preflight identity")
    if args.command == "evaluate" and any(
        v is None
        for v in (
            args.core_only_result,
            args.core_only_sha256,
            args.joint_result,
            args.joint_sha256,
        )
    ):
        parser.error("evaluate requires both explicit terminal identities")
    plan = verify_plan(args)
    limit = (
        plan["train_internal_wall_seconds"]
        if args.command == "train"
        else plan["preflight_internal_wall_seconds"]
    )
    if (
        not 0 < args.wall_seconds <= limit
        or (args.command == "preflight" and args.wall_seconds > 3600)
        or (args.command == "train" and args.wall_seconds > 9000)
    ):
        parser.error("Frozen stage wall budget exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ctx = None
    try:
        if args.command == "evaluate":
            result = joint_evaluate(args, plan)
        elif args.command == "preflight":
            result = {"status": "passed", "continuation_gate": True, "conditions": {}}
            schedule = read(plan["schedule_file"])
            for condition in CONDITIONS:
                ctx = context_for(args, plan, started + args.wall_seconds, condition)
                result["source_commit"] = ctx.source_commit
                result["conditions"][condition] = preflight_one(ctx, schedule)
                ctx.audit.check("preflight-final-release", stable=True)
                ctx = None
                gc.collect()
                torch.cuda.empty_cache()
        else:
            ctx = context_for(args, plan, started + args.wall_seconds, args.condition)
            result = train(ctx, read(plan["schedule_file"]))
            result["source_commit"] = ctx.source_commit
        result.update(
            plan_sha256=args.plan_sha256,
            run_id=args.run_id,
            cache_report_sha256=plan["cache_report_sha256"],
            elapsed_seconds=time.monotonic() - started,
        )
        write_new(args.output / "RESULT.json", result)
    except Exception as exc:
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "technical_validity": "not_established",
                "plan_sha256": args.plan_sha256,
                "run_id": args.run_id,
                "exception": type(exc).__name__,
                "message": str(exc),
                "optimizer_steps_completed": None if ctx is None else ctx.completed_step,
                "automatic_retry": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
