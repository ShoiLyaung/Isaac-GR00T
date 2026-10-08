"""Graph update compatibility followed by one fresh episode125 KD overfit."""

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

from budgeted_action_cudagraph import (
    BackwardGraph,
    CoreDispatcher,
    TensorCore,
    storage_identity,
    tensor_args,
)
from budgeted_action_cudagraph_probe import (
    assert_gradients_cleared,
    gradient_comparison,
    group_mapping,
    snapshot_gradients,
)
from budgeted_action_runner import condition_contract
from budgeted_action_support import (
    cached_action,
    capture_velocity,
    gradient_groups,
    parameter_changes,
)
from budgeted_graph_distillation import (
    clean_graph_kd,
    dropout_inventory,
    eager_receipt,
    evidence_receipt,
    graph_sample,
)
from contract import sha, write_new
from fp32_pilot_support import PilotCache
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
from memory_numerical_trace import error_summary
import memory_prototype as proto
from memory_training import optimizer_for, raw_record
import numpy as np
from runner_core import aggregate_metrics, deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "359c1ff545424edb78b8e35d20c4cac1b6d68d84fafa8de2739243d213ba5422"


def read(path):
    return json.loads(Path(path).read_text())


def verify_inputs(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen Graph distillation PLAN changed")
    plan = read(args.plan)
    if (
        REPO != Path(plan["source_worktree"])
        or args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve()
    ):
        raise ValueError("Frozen source/artifact identity mismatch")
    for row in plan["records"]:
        source = Path(row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError(f"Frozen material changed: {source}")
    for relative in plan["immutable_source_relative_paths"]:
        expected = {r["sha256"] for r in plan["records"] if r["path"].endswith("/" + relative)}
        if len(expected) != 1 or sha(REPO / relative) not in expected:
            raise ValueError(f"Existing implementation changed: {relative}")
    prior = read(plan["graph_prerequisite"])
    if (
        sha(plan["graph_prerequisite"]) != plan["graph_prerequisite_sha256"]
        or prior["technical_validity"] != "passed"
        or not prior["decision"]["candidate_execution_gate"]
    ):
        raise ValueError("Frozen graph engineering prerequisite missing")
    if sha(plan["overfit_schedule_file"]) != plan["schedule_sha256"]["overfit"]:
        raise ValueError("Frozen200-row schedule changed")
    return plan


def environment(plan):
    import budgeted_action_cudagraph
    import budgeted_graph_distillation
    import gr00t
    from gr00t.model.modules import budgeted_action_mamba
    import mamba_ssm.modules.mamba2 as official
    import triton.language.core as triton_core

    local = (
        gr00t,
        budgeted_action_mamba,
        budgeted_action_cudagraph,
        budgeted_graph_distillation,
        proto,
    )
    if any(REPO not in Path(m.__file__).resolve().parents for m in local):
        raise ValueError("Import escaped new graph-distillation worktree")
    if (
        Path(sys.prefix) != VENV
        or sys.version_info[:2] != (3, 10)
        or torch.__version__.split("+")[0] != "2.7.1"
    ):
        raise ValueError("Original runtime environment required")
    modules = [official, triton_core, torch.cuda.graphs] + [
        importlib.import_module("mamba_ssm.ops.triton." + n)
        for n in (
            "ssd_combined",
            "ssd_chunk_scan",
            "ssd_chunk_state",
            "selective_state_update",
            "layernorm_gated",
        )
    ]
    hashes = {Path(r["path"]).resolve(): r["sha256"] for r in plan["records"]}
    if any(
        Path(m.__file__).resolve() not in hashes
        or sha(m.__file__) != hashes[Path(m.__file__).resolve()]
        for m in modules
    ):
        raise ValueError("Kernel or graph helper dependency changed")
    if (
        importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
        or official.causal_conv1d_fn is not None
        or os.environ.get("TRITON_F32_DEFAULT") != "ieee"
    ):
        raise ValueError("Original Mamba/precision environment required")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(plan["allowed_source_files"]):
        raise ValueError("Only committed new stage files are permitted")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Pretrained weights cannot be skipped")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != str(plan["physical_gpu"])
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Only frozen physical GPU allowed")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "imports": {m.__name__: m.__file__ for m in (*local, *modules)},
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": torch.cuda.get_device_name(0),
    }


def compatibility_required(plan, source_commit):
    result = read(plan["compatibility_result"])
    if (
        result.get("status") != "passed"
        or not result.get("continuation_gate")
        or result.get("plan_sha256") != PLAN_SHA
        or result.get("source_commit") != source_commit
        or result.get("optimizer_steps") != 3
    ):
        raise ValueError("Matching passed3-update compatibility required before overfit")
    return {"path": plan["compatibility_result"], "sha256": sha(plan["compatibility_result"])}


def context_for(args, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    plan = verify_inputs(args)
    source = environment(plan)
    prerequisite = (
        compatibility_required(plan, source["commit"]) if args.command == "overfit" else None
    )
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
            raise TimeoutError("Graph distillation stage wall bound exceeded")
        if total > plan["max_artifact_bytes"] or ctx.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Graph distillation resource bound exceeded")
        return total

    ctx.budget = budget
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    ctx.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    ctx.teacher = model.action_head.model
    proto.assert_frozen(model)
    ctx.base_before = proto.content_hash(model.state_dict())
    if ctx.base_before != plan["base_content_sha256"]:
        raise ValueError("Frozen base mismatch")
    ctx.initial = torch.load(
        plan["student_initialization"]["path"], map_location="cpu", weights_only=True
    )
    if proto.content_hash(ctx.initial) != plan["student_initialization"]["content_sha256"]:
        raise ValueError("Fresh initialization content mismatch")
    ctx.student = BudgetedActionMamba(ctx.teacher)
    ctx.student.load_state_dict(ctx.initial, strict=True)
    ctx.student.cuda().float().eval().requires_grad_(True)
    ctx.cache = PilotCache(
        plan["cache_root"], plan, expected_report_sha=plan["cache_report_sha256"]
    )
    if ctx.cache.report["integrity"]["base_before"] != ctx.base_before:
        raise ValueError("Cache/base identities differ")
    ctx.cache.episode(125)
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "environment": source,
            "strict_loading": loading,
            "prerequisite": prerequisite,
            "base_before": ctx.base_before,
            "fresh_student_content_sha256": proto.content_hash(ctx.student.state_dict()),
            "fresh_optimizer": True,
            "warm_start": False,
            "dropout": dropout_inventory(ctx.student),
        },
    )
    ctx.budget()
    return ctx


def create_graph(ctx):
    item = ctx.plan["fb_input"]
    frame = ctx.cache.frame(item["episode"], item["anchor"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sample = capture_velocity(ctx.model, frame, 42, student=ctx.student)
    args = tensor_args(sample["core_kwargs"], student=True, backward=True)
    del sample
    surface = TensorCore(ctx.student, student=True, backward=True)
    lease = BackwardGraph(surface, args)
    ctx.lease = lease
    ctx.route = CoreDispatcher(surface, lease)
    ctx.storage = storage_identity(ctx.student)
    write_new(
        ctx.args.output / "CAPTURE.json",
        {
            "capture_count": 1,
            **lease.setup,
            "parameter_storage": ctx.storage,
            "optimizer_inside_graph": False,
            "teacher_inside_graph": False,
            "condition_inside_graph": True,
        },
    )
    ctx.budget()


def close_graph(ctx):
    ctx.student.zero_grad(set_to_none=True)
    ctx.lease.assert_storage()
    ctx.lease.close()
    ctx.route.backend = None
    ctx.route.surface = None
    ctx.lease = None
    ctx.route = None
    gc.collect()
    torch.cuda.empty_cache()


def final_integrity(ctx):
    proto.assert_frozen(ctx.model)
    if proto.content_hash(ctx.model.state_dict()) != ctx.base_before:
        raise ValueError("Frozen teacher/periphery changed")
    ctx.cache.assert_unchanged()
    if (
        sha(ctx.plan["student_initialization"]["path"])
        != ctx.plan["student_initialization"]["sha256"]
    ):
        raise ValueError("Original initialization file changed")
    artifact_bytes = ctx.budget()
    return {
        "base_before": ctx.base_before,
        "base_after": ctx.base_before,
        "cache_unchanged": True,
        "initialization_file_unchanged": True,
        "student_content_sha256": proto.content_hash(ctx.student.state_dict()),
        "dropout": dropout_inventory(ctx.student),
        "stage_peak_allocated_bytes": ctx.stage_peak,
        "artifact_bytes": artifact_bytes,
    }


def probes(ctx, update, output):
    output.mkdir(exist_ok=False)
    mapping = group_mapping(ctx.student, "student")
    rows = []
    tensors = {}
    for case in ctx.plan["compatibility_probes"]:
        ctx.budget()
        a, seed = case["anchor"], case["seed"]
        frame = ctx.cache.frame(125, a)
        data = {}
        for label in ("eager", "graph"):
            ctx.student.zero_grad(set_to_none=True)
            assert_gradients_cleared(ctx.student)
            set_seed(seed, cuda=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = (
                    capture_velocity(
                        ctx.model,
                        frame,
                        seed,
                        student=ctx.student,
                        teacher=ctx.teacher,
                        replay_teacher=True,
                    )
                    if label == "eager"
                    else clean_graph_kd(ctx.model, ctx.teacher, ctx.route, frame)
                )
                loss = values["kd"]
            loss.backward()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite KD probe")
            student_velocity = (
                values["velocity"] if label == "eager" else values["student_velocity"]
            )
            receipt = (
                eager_receipt(values, frame, seed)
                if label == "eager"
                else evidence_receipt(values, seed)
            )
            data[label] = {
                "kd": float(loss.detach()),
                "native_loss": float(values["native"]["loss"].detach()),
                "receipt": receipt,
                "gradients": snapshot_gradients(ctx.student),
                "velocity": student_velocity.detach().cpu(),
            }
            del values, loss
        if data["eager"]["receipt"] != data["graph"]["receipt"]:
            raise ValueError("Actual eager/Graph input receipts differ")
        stats = gradient_comparison(
            data["eager"]["gradients"],
            data["graph"]["gradients"],
            mapping,
            ctx.plan["gradient_limits"],
        )
        grad_path = output / f"gradients-{a}-{seed}.json"
        write_new(grad_path, stats)
        loss_flags = {
            key: abs(data["graph"][key] - data["eager"][key])
            <= ctx.plan["loss_limits"]["atol"]
            + ctx.plan["loss_limits"]["rtol"] * abs(data["eager"][key])
            for key in ("kd", "native_loss")
        }
        eager_action = cached_action(ctx.model, frame, seed, student=ctx.student)
        graph_action = graph_sample(ctx.model, frame, seed, ctx.route)
        proto.action_record(eager_action, raw_record(frame), ctx.source.processor)
        proto.action_record(graph_action, raw_record(frame), ctx.source.processor)
        errors = error_summary(graph_action[:, :16], eager_action[:, :16])
        action_gate = all(errors[k] <= v for k, v in ctx.plan["action_limits"].items())
        tensors[f"{a}:{seed}"] = {
            "eager_velocity": data["eager"]["velocity"],
            "graph_velocity": data["graph"]["velocity"],
            "eager_action": eager_action,
            "graph_action": graph_action,
        }
        rows.append(
            {
                **case,
                "update": update,
                "eager_kd": data["eager"]["kd"],
                "graph_kd": data["graph"]["kd"],
                "eager_native_loss": data["eager"]["native_loss"],
                "graph_native_loss": data["graph"]["native_loss"],
                "loss_gates": loss_flags,
                "gradient_gate": stats["passed"],
                "gradient_path": str(grad_path),
                "gradient_sha256": sha(grad_path),
                "action_error": errors,
                "action_gate": action_gate,
                "receipt": data["eager"]["receipt"],
                "passed": all(loss_flags.values()) and stats["passed"] and action_gate,
            }
        )
        ctx.student.zero_grad(set_to_none=True)
        del data
        gc.collect()
        ctx.lease.assert_storage()
        ctx.budget()
    path = output / "actions-velocities.pt"
    torch.save(tensors, path)
    report = {
        "update": update,
        "passed": all(row["passed"] for row in rows),
        "cases": rows,
        "tensor_path": str(path),
        "tensor_sha256": sha(path),
        "student_content_sha256": proto.content_hash(ctx.student.state_dict()),
        "storage_unchanged": storage_identity(ctx.student) == ctx.storage,
        "graph_sampling_scope": "Parity only: original4Euler, condition recomputed inside each core replay; no deployment-cost claim",
    }
    write_new(output / "RESULT.json", report)
    return report, tensors


def apply_optimizer_step(ctx, optimizer):
    optimizer.step()
    # Count before synchronization, evidence conversion or disk logging can fail.
    ctx.optimizer_steps_completed += 1


def graph_step(ctx, optimizer, planned):
    frame = ctx.cache.frame(125, planned["anchor"])
    for group in optimizer.param_groups:
        group["lr"] = planned["lr"]
    optimizer.zero_grad(set_to_none=True)
    assert_gradients_cleared(ctx.student)
    set_seed(planned["forward_seed"], cuda=True)
    ctx.lease.assert_storage()
    dropout_inventory(ctx.student)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = clean_graph_kd(ctx.model, ctx.teacher, ctx.route, frame)
    values["kd"].backward()
    norm = torch.nn.utils.clip_grad_norm_(
        ctx.student.parameters(), ctx.plan["optimizer"]["gradient_clip"], error_if_nonfinite=True
    )
    apply_optimizer_step(ctx, optimizer)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - tick
    groups = gradient_groups(ctx.student)
    receipt = evidence_receipt(values, planned["forward_seed"])
    if not bool(torch.isfinite(values["kd"])) or not bool(torch.isfinite(values["native"]["loss"])):
        raise ValueError("Nonfinite update loss")
    ctx.lease.assert_storage()
    proto.assert_frozen(ctx.model)
    row = {
        **planned,
        "optimizer_steps_completed": ctx.optimizer_steps_completed,
        "objective": "teacher_velocity_kd",
        "teacher_velocity_kd": float(values["kd"].detach()),
        "native_loss": float(values["native"]["loss"].detach()),
        "gradient_norm_before_clip": float(norm),
        "gradient_groups": groups,
        "gradient_group_norm_scope": "post-clip; captured outside timer",
        "noise_receipt": receipt,
        "elapsed_step_seconds": elapsed,
        "clean_step_wall_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gradients_cleared_before_forward": True,
        "storage_unchanged": True,
        "step_cost_scope": "cached input H2D/native VL-state-action, graph student, eager teacher replay, KD backward, clip and AdamW; excludes cache read/seed/CPU hashes/statistics; no VLM/RGB",
    }
    return row


def compatibility(ctx):
    create_graph(ctx)
    optimizer = optimizer_for(ctx.student, ctx.plan)
    schedule = read(ctx.plan["overfit_schedule_file"])[:3]
    reports = []
    initial_values = None
    last_values = None
    completed = 0
    initial_hash = proto.content_hash(ctx.student.state_dict())
    with (ctx.args.output / "training_steps.jsonl").open("x") as stream:
        for update in range(4):
            if update in ctx.plan["compatibility_check_updates"]:
                report, tensors = probes(ctx, update, ctx.args.output / f"update-{update}")
                reports.append(report)
                if update == 0:
                    initial_values = tensors
                if update == 3:
                    last_values = tensors
                if not report["passed"]:
                    close_graph(ctx)
                    return {
                        "status": "failed",
                        "continuation_gate": False,
                        "optimizer_steps": completed,
                        "checks": reports,
                        "failed_gate": "update_numeric",
                        "integrity": final_integrity(ctx),
                    }
            if update < 3:
                ctx.budget()
                before = proto.content_hash(ctx.student.state_dict())
                row = graph_step(ctx, optimizer, schedule[update])
                completed += 1
                row["content_changed"] = proto.content_hash(ctx.student.state_dict()) != before
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if not row["content_changed"]:
                    raise ValueError("Actual Graph optimizer update changed no student content")
    witness = {
        "velocity_changed": any(
            not torch.equal(initial_values[k]["graph_velocity"], last_values[k]["graph_velocity"])
            for k in initial_values
        ),
        "action_changed": any(
            not torch.equal(initial_values[k]["graph_action"], last_values[k]["graph_action"])
            for k in initial_values
        ),
        "same_probe_changed": any(
            not torch.equal(initial_values[k]["graph_velocity"], last_values[k]["graph_velocity"])
            and not torch.equal(initial_values[k]["graph_action"], last_values[k]["graph_action"])
            for k in initial_values
        ),
        "weight_content_changed": proto.content_hash(ctx.student.state_dict()) != initial_hash,
        "storage_unchanged": storage_identity(ctx.student) == ctx.storage,
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    close_graph(ctx)
    passed = all(witness.values()) and all(row["passed"] for row in reports)
    return {
        "status": "passed" if passed else "failed",
        "continuation_gate": passed,
        "optimizer_steps": completed,
        "checks": reports,
        "update_witness": witness,
        "compatibility_weights_discarded": True,
        "checkpoint_saved": False,
        "integrity": final_integrity(ctx),
    }


def evaluate(ctx, output):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from run_train_eval import physical_metrics

    output.mkdir(exist_ok=False)
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
    lo = np.concatenate([np.asarray(params[k]["min"]) for k in ("left_arm", "left_hand")])
    hi = np.concatenate([np.asarray(params[k]["max"]) for k in ("left_arm", "left_hand")])
    write_new(
        output / "NORMALIZATION.json",
        {
            "representation": "ABSOLUTE",
            "normalization": "minmax",
            "order": ["left_arm7", "left_hand6"],
            "min": lo.tolist(),
            "max": hi.tolist(),
            "active_horizon": 16,
            "model_horizon": 40,
            "native_decode_clipping": True,
        },
    )
    for a in ctx.plan["eval_anchors"]["125"]:
        frame = ctx.cache.frame(125, a)
        raw = raw_record(frame)
        for nominal in ctx.plan["noise_seeds"]:
            ctx.budget()
            seed = deterministic_seed("diagnostic", 125, a, nominal)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                values = capture_velocity(
                    ctx.model,
                    frame,
                    seed,
                    student=ctx.student,
                    teacher=ctx.teacher,
                    replay_teacher=True,
                )
            actual = cached_action(ctx.model, frame, seed, student=ctx.student)
            teacher = cached_action(ctx.model, frame, seed)
            _, identity = proto.action_record(actual, raw, processor)
            proto.action_record(teacher, raw, processor)
            pred = actual[0].float().numpy()
            target_pred = teacher[0].float().numpy()
            physical = []
            for value in (pred, target_pred):
                decoded = processor.decode_action(
                    value, EmbodimentTag.NEW_EMBODIMENT, state=raw.states
                )
                physical.append(
                    np.concatenate([decoded[k] for k in ("left_arm", "left_hand")], axis=-1)
                )
            metric = physical_metrics(pred, raw, processor)
            row = {
                "episode": 125,
                "anchor": a,
                "seed": nominal,
                "paired_rng_seed": seed,
                "flow_loss": float(values["native"]["loss"]),
                "teacher_velocity_kd": float(values["kd"]),
                "teacher_action_physical_mae": float(np.abs(physical[0] - physical[1]).mean()),
                "teacher_action_normalized_mae": float(np.abs(pred[:16] - target_pred[:16]).mean()),
                **metric,
                "noise_receipt": eager_receipt(values, frame, seed),
                **identity,
            }
            rows.append(row)
            for key, value in [
                ("student_actions", pred),
                ("teacher_actions", target_pred),
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
    keys = (
        "flow_loss",
        "teacher_velocity_kd",
        "teacher_action_physical_mae",
        "teacher_action_normalized_mae",
        "combined_mae_mixed_units",
        "arm_mae_rad",
        "hand_mae_u",
    )
    if len(rows) != 64 or not all(np.isfinite(row[k]) for row in rows for k in keys):
        raise ValueError("Expected64 finite evaluation outputs")
    per_episode, metrics = aggregate_metrics(rows, keys)
    for name, values in arrays.items():
        np.save(output / f"{name}.npy", np.stack(values))
    write_new(output / "details.json", rows)
    result = {
        "status": "passed",
        "rows": 64,
        "metrics": metrics,
        "per_episode": per_episode,
        "array_sha256": {name: sha(output / f"{name}.npy") for name in arrays},
        "sampling": "Original eager once-per-observation condition; no graph deployment-speed claim",
    }
    write_new(output / "RESULT.json", result)
    return result


def overfit(ctx):
    before = evaluate(ctx, ctx.args.output / "step0")
    create_graph(ctx)
    optimizer = optimizer_for(ctx.student, ctx.plan)
    seen = {}
    completed = 0
    schedule = read(ctx.plan["overfit_schedule_file"])
    if len(schedule) != 200:
        raise ValueError("Exactly200 old schedule rows required")
    with (ctx.args.output / "training_steps.jsonl").open("x") as stream:
        for planned in schedule:
            ctx.budget()
            if planned["step"] != completed + 1 or planned["episode"] != 125:
                raise ValueError("Frozen schedule ordering changed")
            row = graph_step(ctx, optimizer, planned)
            completed += 1
            for key, value in row["gradient_groups"].items():
                seen[key] = seen.get(key, False) or value["finite_nonzero_parameters"] > 0
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if completed % 25 == 0:
                print(json.dumps({"optimizer_steps": completed}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    checkpoint = ctx.args.output / "student-step200.pt"
    torch.save({k: v.detach().cpu() for k, v in ctx.student.state_dict().items()}, checkpoint)
    terminal, _ = probes(ctx, 200, ctx.args.output / "terminal")
    after = evaluate(ctx, ctx.args.output / "step200")
    intervention = condition_contract(ctx)
    changes = parameter_changes(ctx.student, ctx.initial, seen)
    limits = ctx.plan["overfit_gates"]
    gates = {
        "kd_reduction_30pct": after["metrics"]["teacher_velocity_kd"]
        <= before["metrics"]["teacher_velocity_kd"] * limits["kd_max_final_initial_ratio"],
        "teacher_physical_action_mae_reduction_10pct": after["metrics"][
            "teacher_action_physical_mae"
        ]
        <= before["metrics"]["teacher_action_physical_mae"]
        * limits["teacher_physical_action_mae_max_final_initial_ratio"],
        "all28_groups_update": changes["passed"],
        "terminal_graph_eager_parity": terminal["passed"],
        "condition_intervention": intervention["passed"],
    }
    close_graph(ctx)
    passed = all(gates.values())
    return {
        "status": "passed" if passed else "failed",
        "continuation_gate": passed,
        "optimizer_steps": completed,
        "gates": gates,
        "step0": before,
        "step200": after,
        "terminal": terminal,
        "changes": changes,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha(checkpoint),
            "content_sha256": proto.content_hash(ctx.student.state_dict()),
        },
        "integrity": final_integrity(ctx),
        "pilot_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compatibility", "overfit"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= (850 if args.command == "compatibility" else 1750):
        parser.error("Frozen stage wall bound exceeded")
    if args.output.resolve() != (args.artifact_root / (args.command + "-r1") / "output").resolve():
        parser.error("One canonical output per stage required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ctx = None
    completed_count = 0
    try:
        ctx = context_for(args, started + args.wall_seconds)
        source_commit = ctx.source_commit
        result = compatibility(ctx) if args.command == "compatibility" else overfit(ctx)
        result.update(
            plan_sha256=PLAN_SHA,
            source_commit=source_commit,
            elapsed_seconds=time.monotonic() - started,
            old_results_unchanged=True,
        )
        completed_count = ctx.optimizer_steps_completed
        ctx = None
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
                "optimizer_steps_completed": completed_count
                if ctx is None
                else ctx.optimizer_steps_completed,
                "optimizer_steps": completed_count
                if ctx is None
                else ctx.optimizer_steps_completed,
                "step_count_scope": "successful optimizer.step returns; counted before subsequent synchronization/evidence/logging",
            },
        )
        raise
    if not result["continuation_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
