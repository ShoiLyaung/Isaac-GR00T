"""Two explicit mixed-KD variants continuing one frozen20k parent to50k."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace

import action_transfer_continuation as continuation
import action_transfer_matched_runner as matched
from action_transfer_matched_support import (
    MMapCache,
    StrictBatchDispatcher,
    assemble_batch,
    checkpoint_log_rows,
)
from action_transfer_program_support import restore_rng, rng_state
from action_transfer_trajectory_kd_support import (
    VARIANTS,
    assert_t0_equal,
    capture_teacher,
    discrete_query,
    mixed_query,
    query_receipts,
)
from budgeted_action_cudagraph import TensorCore
from budgeted_action_cudagraph_probe import (
    assert_gradients_cleared,
    gradient_comparison,
    group_mapping,
    snapshot_gradients,
)
from budgeted_action_support import gradient_groups, parameter_changes
from budgeted_graph_distillation_runner import environment
from contract import sha, write_new
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
import memory_prototype as proto
from memory_training import optimizer_for
from memory_training_support import sampling_packet
import numpy as np
from run_train_eval import NativeAnchors, strict_reload
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def verify_mixture(plan):
    if tuple(plan["conditions"]) != VARIANTS or plan["parent_condition"] != "distill_then_task":
        raise ValueError("Expected two mixed-KD variants from the same KD-task20k parent")
    parents = plan["parents"]
    if len({(v["index_path"], v["index_sha256"]) for v in parents.values()}) != 1:
        raise ValueError("Mixed variants must share one frozen parent checkpoint")
    rows = read(plan["schedule_file"])[20000:]
    counts = Counter()
    for row in rows:
        branch = "continuous" if row["step"] % 2 else "discrete"
        if row["branch"] != branch or (branch == "continuous" and row["bucket"] is not None):
            raise ValueError("Frozen odd/even mixture changed")
        if branch == "discrete":
            counts[row["bucket"]] += 1
    if counts != {0: 3750, 250: 3750, 500: 3750, 750: 3750}:
        raise ValueError("Exactly3750 discrete updates per inference bucket required")
    return counts


def import_parent(args, plan, source_commit):
    original = plan["parents"][args.condition]
    for p, s in (("terminal_path", "terminal_sha256"), ("plan_path", "plan_sha256")):
        if sha(original[p]) != original[s]:
            raise ValueError("Explicit frozen parent identity changed")
    terminal = read(original["terminal_path"])
    if (
        not terminal["training_completed"]
        or terminal["optimizer_steps"] != 20000
        or terminal["condition"] != plan["parent_condition"]
    ):
        raise ValueError("Fixed KD-task20k terminal required")
    record = (
        continuation.parent_record(plan, args.condition)
        if args.resume_index is None
        else {"path": str(args.resume_index.resolve()), "sha256": args.resume_index_sha256}
    )
    continuation.file_record_valid(record)
    index = read(record["path"])
    expected = {
        "condition": plan["parent_condition"] if args.resume_index is None else args.condition,
        "plan_sha256": original["plan_sha256"] if args.resume_index is None else args.plan_sha256,
        "source_commit": original["source_commit"] if args.resume_index is None else source_commit,
        "cache_report_sha256": plan["cache_report_sha256"],
        "schedule_sha256": original["schedule_sha256"]
        if args.resume_index is None
        else plan["schedule_sha256"],
    }
    if any(index.get(k) != v for k, v in expected.items()):
        raise ValueError("Mixed-KD import source/parent/variant identity changed")
    step = index["completed_step"]
    if (
        step not in ([20000] if args.resume_index is None else plan["resume_checkpoint_steps"])
        or step >= 50000
        or index["next_step"] != step + 1
        or index["phase"] != 2
        or index["next_phase_reinitialize_optimizer"]
        or index["run_id"] == args.run_id
    ):
        raise ValueError("Invalid mixed-KD no-reset restore boundary")
    for key in ("weights", "optimizer_rng"):
        continuation.file_record_valid(index[key])
    payload = torch.load(index["optimizer_rng"]["path"], map_location="cpu", weights_only=True)
    if (
        any(
            payload[k] != index[k]
            for k in ("condition", "completed_step", "phase", "plan_sha256", "source_commit")
        )
        or payload["phase_optimizer_steps"] != step - 10000
    ):
        raise ValueError("Mixed-KD optimizer payload/header mismatch")
    return record, index, payload


def context_for(args, plan, deadline):
    identity = environment(plan)
    record, index, payload = import_parent(args, plan, identity["commit"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    ctx = SimpleNamespace(
        args=args,
        plan=plan,
        source_commit=identity["commit"],
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
            raise TimeoutError("Mixed-KD wall bound exceeded")
        if size > plan["max_artifact_bytes"] or ctx.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Mixed-KD resource bound exceeded")
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
        raise ValueError("Base/cache identity changed")
    ctx.initial = torch.load(index["weights"]["path"], map_location="cpu", weights_only=True)
    if proto.content_hash(ctx.initial) != index["student_content_sha256"]:
        raise ValueError("Parent student semantic changed")
    if any(
        v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in ctx.initial.values()
    ):
        raise ValueError("Finite FP32 student masters required")
    ctx.student = BudgetedActionMamba(ctx.teacher)
    ctx.student.load_state_dict(ctx.initial, strict=True)
    ctx.student.cuda().float().eval().requires_grad_(True)
    ctx.optimizer = optimizer_for(ctx.student, plan)
    restored = continuation.restore_optimizer(
        ctx.student, ctx.optimizer, payload, ctx.starting_step - 10000
    )
    ctx.source = NativeAnchors(Path(plan["prepared"]))
    ctx.source.processor.eval()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": args.plan_sha256,
            "source": identity,
            "strict_loading": loading,
            "parent_index": record,
            "parent_condition": plan["parent_condition"],
            "variant": args.condition,
            "starting_step": ctx.starting_step,
            "student_content_sha256": index["student_content_sha256"],
            "optimizer": restored,
            "new_optimizer_reset": False,
            "cache_report_sha256": args.cache_report_sha256,
            "precision_scope": "common original BF16 data-line; discrete FP32 epsilon/time/xt; BF16 AMP and FP32 student masters",
        },
    )
    return ctx, index, payload["rng"]


def proof(ctx, planned, output):
    output.mkdir(parents=True, exist_ok=False)
    frame = assemble_batch(ctx.cache, planned["batch"])
    mapping = group_mapping(ctx.student, "student")
    eager = StrictBatchDispatcher(
        TensorCore(ctx.student, student=True, backward=True), None, ctx.samples
    )
    before = proto.content_hash(ctx.student.state_dict())
    reports = []
    for spec in ctx.plan["restoration_branches"]:
        row = {**planned, **spec}
        data = {}
        for name, route in (("eager", eager), ("graph", ctx.route)):
            ctx.budget()
            ctx.student.zero_grad(set_to_none=True)
            assert_gradients_cleared(ctx.student)
            set_seed(row["forward_seed"], cuda=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = mixed_query(ctx.model, ctx.teacher, route, frame, row, ctx.args.condition)
            values["kd"].backward()
            shared, selected, _ = query_receipts(values, frame, row["forward_seed"])
            data[name] = {
                "loss": float(values["kd"].detach()),
                "shared": shared,
                "selected": selected,
                "gradients": snapshot_gradients(ctx.student),
            }
            del values
        if (
            data["eager"]["shared"] != data["graph"]["shared"]
            or data["eager"]["selected"] != data["graph"]["selected"]
        ):
            raise ValueError("Eager/Graph mixed-KD query inputs/teacher target changed")
        stats = gradient_comparison(
            data["eager"]["gradients"],
            data["graph"]["gradients"],
            mapping,
            ctx.plan["gradient_limits"],
        )
        delta = abs(data["eager"]["loss"] - data["graph"]["loss"])
        limits = ctx.plan["loss_limits"]
        gate = delta <= limits["atol"] + limits["rtol"] * abs(data["eager"]["loss"])
        label = f"{spec['branch']}-{spec['bucket']}"
        path = output / f"gradients-{label}.json"
        write_new(path, stats)
        reports.append(
            {
                **spec,
                "loss_reference": data["eager"]["loss"],
                "loss_graph": data["graph"]["loss"],
                "loss_gate": gate,
                "gradient_gate": stats["passed"],
                "gradient_path": str(path),
                "gradient_sha256": sha(path),
                "shared": data["eager"]["shared"],
                "selected": data["eager"]["selected"],
            }
        )
        del data
        ctx.student.zero_grad(set_to_none=True)
        ctx.lease.assert_storage()
    # Same weights, one captured teacher rollout and two explicitly selected x_t sources.
    set_seed(planned["forward_seed"], cuda=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        trace = capture_teacher(ctx.model, frame)
        a = discrete_query(ctx.model, ctx.teacher, eager, frame, trace, 0, "data_line_mixed")
        b = discrete_query(
            ctx.model, ctx.teacher, eager, frame, trace, 0, "teacher_trajectory_mixed"
        )
        t0 = assert_t0_equal(a, b)
        set_seed(planned["forward_seed"], cuda=True)
        packet = sampling_packet(ctx.model, frame)
        normal = proto.sample_packet(ctx.model, packet)
        if not torch.equal(trace["final_action"], normal):
            raise ValueError("Teacher capture changed original four-Euler trajectory")
    torch.save(
        {
            "epsilon": trace["epsilon"].cpu(),
            "x_t_data_line": a["evidence"]["noisy_action"].cpu(),
            "x_t_trajectory": b["evidence"]["noisy_action"].cpu(),
            "teacher_velocity": a["teacher_velocity"].cpu(),
            "student_velocity": a["student_velocity"].cpu(),
            "mask": a["evidence"]["mask"].cpu(),
        },
        output / "T0_PARITY.pt",
    )
    ctx.student.zero_grad(set_to_none=True)
    if proto.content_hash(ctx.student.state_dict()) != before:
        raise ValueError("Zero-update mixed-KD proof changed weights")
    result = {
        "passed": all(r["loss_gate"] and r["gradient_gate"] for r in reports),
        "branches": reports,
        "t0": t0,
        "teacher_capture_final_exact": True,
        "optimizer_updates": 0,
        "student_content_sha256": before,
        "t0_arrays_sha256": sha(output / "T0_PARITY.pt"),
    }
    write_new(output / "RESULT.json", result)
    if not result["passed"]:
        raise ValueError("Mixed-KD branch Graph/gradient proof failed")
    return result


def update(ctx, optimizer, planned):
    started = time.monotonic()
    frame = assemble_batch(ctx.cache, planned["batch"])
    preparation = time.monotonic() - started
    for group in optimizer.param_groups:
        group["lr"] = planned["lr"]
    optimizer.zero_grad(set_to_none=True)
    assert_gradients_cleared(ctx.student)
    ctx.lease.assert_storage()
    set_seed(planned["forward_seed"], cuda=True)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = mixed_query(ctx.model, ctx.teacher, ctx.route, frame, planned, ctx.args.condition)
    loss = values["kd"]
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        ctx.student.parameters(), ctx.plan["optimizer"]["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    ctx.optimizer_steps_completed += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - tick
    shared, selected, receipt = query_receipts(values, frame, planned["forward_seed"])
    groups = gradient_groups(ctx.student)
    proto.assert_frozen(ctx.model)
    ctx.lease.assert_storage()
    return {
        **planned,
        "condition": ctx.args.condition,
        "objective": "active16_teacher_velocity_kd",
        "optimizer_steps_completed": ctx.optimizer_steps_completed,
        "phase_optimizer_steps": planned["phase_step"],
        "teacher_velocity_kd": float(loss.detach()),
        "native_loss": None
        if values["native_loss_diagnostic"] is None
        else float(values["native_loss_diagnostic"].detach()),
        "gradient_norm_before_clip": float(norm),
        "gradient_groups": groups,
        "shared_receipt": shared,
        "selected_receipt": selected,
        "noise_receipt": receipt,
        "frame_content_sha256": frame["frame_content_sha256"],
        "batch_preparation_seconds": preparation,
        "clean_step_wall_seconds": wall,
        "elapsed_step_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cost_scope": "Same teacher4Euler plus selected teacher query/student core on discrete steps; peripheral extra data-line encoding differs, not FLOPs matched",
    }


def train(ctx, index, pending_rng):
    optimizer = ctx.optimizer
    ctx.optimizer = None
    schedule = read(ctx.plan["schedule_file"])
    gate_row = schedule[ctx.plan["restoration_batch_indices"][0]]
    gate_frame = assemble_batch(ctx.cache, gate_row["batch"])
    matched.create_batch_graph(ctx, gate_frame, gate_row["forward_seed"])
    actual = matched.evaluate_pairs(
        ctx, matched.dev_pairs(ctx.plan), ctx.args.output / "restoration-dev"
    )
    restored = continuation.compare_reproduction(
        index["evaluations"][f"dev-{ctx.starting_step}"], actual
    )
    initial_proof = proof(ctx, gate_row, ctx.args.output / "restoration-branches")
    restore_rng(pending_rng)
    records = index["evaluations"]
    seen = {k: False for k in index["gradient_groups_seen"]}
    checkpoints = []
    first_proof = None
    with (ctx.args.output / "training_steps.jsonl").open("x") as stream:
        for planned in schedule[ctx.starting_step :]:
            ctx.budget()
            if (
                planned["step"] != ctx.optimizer_steps_completed + 1
                or planned["phase_step"] != planned["step"] - 10000
                or planned["lr"] != continuation.continuation_lr(planned["step"])
            ):
                raise ValueError("Mixed-KD frozen progression changed")
            row = update(ctx, optimizer, planned)
            for key, value in row["gradient_groups"].items():
                seen[key] = seen.get(key, False) or value["finite_nonzero_parameters"] > 0
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            step = ctx.optimizer_steps_completed
            if step == ctx.starting_step + 1:
                after_rng = rng_state()
                state = continuation.optimizer_steps(optimizer, expected_step=step - 10000)
                first_proof = proof(ctx, gate_row, ctx.args.output / "after-first-update-proof")
                first_proof["optimizer_state"] = state
                restore_rng(after_rng)
            if step in ctx.plan["dev_probe_steps"]:
                records[f"dev-{step}"] = matched.evaluate_pairs(
                    ctx, matched.dev_pairs(ctx.plan), ctx.args.output / f"dev-step{step}"
                )
            if step in ctx.plan["train_probe_steps"]:
                records[f"train-{step}"] = matched.evaluate_pairs(
                    ctx, ctx.plan["train_probe_pairs"], ctx.args.output / f"train-probe-step{step}"
                )
            if step in ctx.plan["weight_checkpoint_steps"]:
                checkpoints.append(
                    continuation.save_checkpoint(ctx, optimizer, step, records, seen)
                )
                ctx.budget()
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "variant": ctx.args.condition,
                            "step": step,
                            "branch": planned["branch"],
                            "bucket": planned["bucket"],
                            "kd": row["teacher_velocity_kd"],
                            "lr": row["lr"],
                        }
                    ),
                    flush=True,
                )
    final_optimizer = continuation.optimizer_steps(optimizer, expected_step=40000)
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
        "starting_step": ctx.starting_step,
        "parent_index": ctx.parent_index,
        "restoration_dev": restored,
        "restoration_branches": initial_proof,
        "after_first_update_proof": first_proof,
        "optimizer_final": final_optimizer,
        "evaluations": records,
        "checkpoints": checkpoints,
        "changes_since_import": changes,
        "attempt_gradient_groups_seen": seen,
        "parent_gradient_groups_seen": ctx.parent_gradient_groups_seen,
        "quality_failure_stops_training": False,
        "new_optimizer_reset": False,
        "new_contract_optimizer_steps": 30000,
        "new_contract_samples": 120000,
        "cumulative_samples": 200000,
        "integrity": matched.integrity(ctx),
    }


def lineage_rows(record, plan, condition, plan_sha, source_commit, seen=None):
    import hashlib

    seen = set() if seen is None else seen
    continuation.file_record_valid(record)
    if record["sha256"] in seen:
        raise ValueError("Mixed-KD checkpoint lineage cycle")
    seen.add(record["sha256"])
    if record == continuation.parent_record(plan, condition):
        expected = continuation.expected_identity(plan, condition)
        expected["condition"] = plan["parent_condition"]
        return checkpoint_log_rows(record, expected)
    index = read(record["path"])
    expected = continuation.expected_identity(
        plan, condition, current_sha=plan_sha, source_commit=source_commit
    )
    if any(index.get(k) != v for k, v in expected.items()):
        raise ValueError("Mixed-KD lineage source/variant identity changed")
    rows = lineage_rows(index["parent_index"], plan, condition, plan_sha, source_commit, seen)
    if len(rows) != index["starting_step"]:
        raise ValueError("Mixed-KD lineage starting step differs")
    meta = index["training_log"]
    with Path(meta["path"]).open("rb") as stream:
        data = stream.read(meta["bytes"])
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise ValueError("Mixed-KD log prefix changed")
    suffix = [json.loads(line) for line in data.splitlines()]
    if len(suffix) != meta["rows"]:
        raise ValueError("Mixed-KD log prefix count differs")
    rows.extend(suffix)
    if [r["step"] for r in rows] != list(range(1, index["completed_step"] + 1)):
        raise ValueError("Mixed-KD global log steps missing/duplicated")
    return rows


def read_final(path, digest, plan_sha, condition):
    if not digest or sha(path) != digest:
        raise ValueError("Explicit terminal SHA required")
    terminal = read(path)
    if (
        terminal["condition"] != condition
        or terminal["plan_sha256"] != plan_sha
        or not terminal["training_completed"]
        or terminal["optimizer_steps"] != 50000
    ):
        raise ValueError("Fixed final50000 terminal required")
    record = terminal["evaluations"]["dev-50000"]
    continuation.file_record_valid(record)
    value = read(record["path"])
    root = Path(record["path"]).parent
    for name, digest in value["array_sha256"].items():
        if sha(root / f"{name}.npy") != digest:
            raise ValueError("Final dev array changed")
    if (
        sha(root / "details.json") != value["details_sha256"]
        or sha(root / "NORMALIZATION.json") != value["normalization_sha256"]
    ):
        raise ValueError("Final dev detail/normalization changed")
    checkpoint = terminal["checkpoints"][-1]
    continuation.file_record_valid(checkpoint)
    index = read(checkpoint["path"])
    continuation.file_record_valid(index["weights"])
    if (
        index["completed_step"] != 50000
        or index["student_content_sha256"] != value["checkpoint_content_sha256"]
    ):
        raise ValueError("Final checkpoint/dev identity mismatch")
    return terminal, value, read(root / "details.json")


def paired_bootstrap(candidate, control, episodes, spec):
    if spec["unit"] != "episode":
        raise ValueError("Bootstrap unit must remain episode")
    indices = np.random.default_rng(spec["seed"]).integers(
        0, len(episodes), size=(spec["resamples"], len(episodes))
    )
    alpha = (1 - spec["confidence"]) / 2
    output = {}
    for key in ("flow_loss", "combined_mae_mixed_units", "teacher_action_physical_mae"):
        a = np.array([candidate["per_episode"][str(ep)][key] for ep in episodes], np.float64)
        b = np.array([control["per_episode"][str(ep)][key] for ep in episodes], np.float64)
        delta = a - b
        means = delta[indices].mean(1)
        candidate_means, control_means = a[indices].mean(1), b[indices].mean(1)
        invalid_denominators = int(np.sum(control_means <= 0))
        gains = None if invalid_denominators else 1 - candidate_means / control_means
        output[key] = {
            "episode_differences": dict(zip(map(str, episodes), delta.tolist())),
            "mean_difference": float(delta.mean()),
            "paired_mean_difference_ci95": np.quantile(means, [alpha, 1 - alpha]).tolist(),
            "ratio_of_episode_equal_means": float(a.mean() / b.mean()) if b.mean() > 0 else None,
            "relative_gain": float(1 - a.mean() / b.mean()) if b.mean() > 0 else None,
            "paired_relative_gain_ci95": None
            if gains is None
            else np.quantile(gains, [alpha, 1 - alpha]).tolist(),
            "nonpositive_bootstrap_denominators": invalid_denominators,
            "relative_gain_definition": "1 - ratio of resampled episode-equal means; undefined for nonpositive control mean; no resamples silently dropped",
            "episodes_better": int(np.sum(delta < 0)),
            "direction": "candidate minus control; negative favors candidate",
        }
    return {
        "metrics": output,
        "resamples": spec["resamples"],
        "seed": spec["seed"],
        "unit": "episode",
        "confidence": spec["confidence"],
        "scope": "Descriptive uncertainty across8development episodes; sampling seeds are not independent training repeats",
    }


def evaluate(args, plan):
    results = {}
    terminals = {}
    details = {}
    logs = {}
    for variant in plan["conditions"]:
        override = getattr(args, variant + "_terminal")
        digest = getattr(args, variant + "_terminal_sha256")
        if (override is None) != (digest is None):
            raise ValueError("Explicit resumed terminal path and SHA required together")
        path = override or args.artifact_root / plan["run_ids"][variant] / "output/RESULT.json"
        if digest is None:
            digest = sha(path)  # Canonical completed local run; final identity is recorded below.
        terminal, value, rows = read_final(path, digest, args.plan_sha256, variant)
        terminals[variant] = {
            "path": str(path),
            "sha256": digest,
            "source_commit": terminal["source_commit"],
        }
        results[variant] = value
        details[variant] = rows
        logs[variant] = lineage_rows(
            terminal["checkpoints"][-1], plan, variant, args.plan_sha256, terminal["source_commit"]
        )
    if len({terminals[v]["source_commit"] for v in plan["conditions"]}) != 1:
        raise ValueError("New mixed-KD variants used different source commits")
    for condition, control in plan["external_task_controls"].items():
        digest = getattr(args, condition + "_control_sha256")
        terminal, value, rows = read_final(
            Path(control["terminal_path"]), digest, control["plan_sha256"], condition
        )
        if (
            terminal["source_commit"] != control["source_commit"]
            or sha(control["plan_path"]) != control["plan_sha256"]
        ):
            raise ValueError("External task control source/PLAN identity changed")
        terminals[condition] = {
            "path": control["terminal_path"],
            "sha256": digest,
            "source_commit": terminal["source_commit"],
        }
        results[condition] = value
        details[condition] = rows
    expected_pairs = matched.dev_pairs(plan)
    baseline = details[plan["conditions"][0]]
    for name, rows in details.items():
        if len(rows) != 512:
            raise ValueError("All comparisons require full512 output development evaluations")
        for pair, a, b in zip(expected_pairs, baseline, rows):
            if (
                any(
                    (r["episode"], r["anchor"], r["seed"])
                    != (pair["episode"], pair["anchor"], pair["nominal_seed"])
                    for r in (a, b)
                )
                or a["noise_receipt"] != b["noise_receipt"]
            ):
                raise ValueError("Cross-run evaluation inputs/seeds/noise/targets changed")
    for key in ("teacher_actions", "physical_targets"):
        if len({v["array_sha256"][key] for v in results.values()}) != 1:
            raise ValueError("Frozen teacher sample or physical target references differ")
    if (
        len({v["normalization_sha256"] for v in results.values()}) != 1
        or len({v["cache_report_sha256"] for v in results.values()}) != 1
    ):
        raise ValueError("Cross-run normalization/cache identity differs")
    schedule = read(plan["schedule_file"])
    a, b = (logs[k] for k in plan["conditions"])
    if len(a) != 50000 or len(b) != 50000:
        raise ValueError("Full matched cumulative50000 logs required")
    for row, left, right in zip(schedule, a, b):
        if any(left.get(k) != v or right.get(k) != v for k, v in row.items()):
            raise ValueError("Matched mixed-KD schedule differs")
        if row["step"] <= 20000:
            if left != right:
                raise ValueError("Shared parent log prefix differs")
        elif left["shared_receipt"] != right["shared_receipt"]:
            raise ValueError(
                "Mixed variants changed shared noise/time/q/Z/target/mask/teacher rollout"
            )
        elif (
            row["branch"] == "continuous" and left["selected_receipt"] != right["selected_receipt"]
        ):
            raise ValueError("Common continuous query or teacher target differs")
    primary = results["teacher_trajectory_mixed"]
    comparisons = {}
    gates = {}
    limits = plan["dev_gates"]
    for name in ("data_line_mixed", "task_only", "distill_then_task"):
        comparisons[name] = paired_bootstrap(
            primary, results[name], plan["dev"], plan["paired_bootstrap"]
        )
        gates[name] = {}
        for key, bound in (
            ("flow_loss", limits["max_distilled_task_flow_ratio"]),
            ("combined_mae_mixed_units", limits["max_distilled_task_combined_ratio"]),
        ):
            value = comparisons[name]["metrics"][key]
            gates[name][key] = (
                value["ratio_of_episode_equal_means"] is not None
                and value["ratio_of_episode_equal_means"] <= bound
                and value["episodes_better"] >= limits["min_episodes_better"]
            )
    retention = {
        k: primary["metrics"][k]
        <= primary["teacher_metrics"][k] * limits["max_native_metric_ratio"]
        for k in limits["native_metrics"]
    }
    return {
        "status": "completed",
        "technical_validity": "passed",
        "terminals": terminals,
        "results": results,
        "paired_comparisons": comparisons,
        "quality_gates": gates,
        "native_retention": retention,
        "quality_gate": all(all(v.values()) for v in gates.values()) and all(retention.values()),
        "teacher_imitation_improved_vs_data_line": comparisons["data_line_mixed"]["metrics"][
            "teacher_action_physical_mae"
        ]["mean_difference"]
        < 0,
        "paired_new_steps": 30000,
        "paired_new_samples": 120000,
        "paired_cumulative_steps": 50000,
        "optimizer_steps": 0,
        "scope": "37.5percent of suffix steps intervene on state source; posthoc8dev engineering comparison, no independent train repeats or confirmation claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--condition", choices=VARIANTS)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume-index", type=Path)
    parser.add_argument("--resume-index-sha256")
    for variant in VARIANTS:
        parser.add_argument("--" + variant.replace("_", "-") + "-terminal", type=Path)
        parser.add_argument("--" + variant.replace("_", "-") + "-terminal-sha256")
    parser.add_argument("--task-only-control-sha256")
    parser.add_argument("--distill-then-task-control-sha256")
    args = parser.parse_args()
    if args.command == "train" and args.condition is None:
        parser.error("--condition required")
    plan = continuation.verify_plan(args)
    verify_mixture(plan)
    if not 0 < args.wall_seconds <= plan["train_internal_wall_seconds"]:
        parser.error("Frozen mixed-KD wall bound exceeded")
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
