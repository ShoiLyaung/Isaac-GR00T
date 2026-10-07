"""One preregistered FP32 four-condition pilot with gated joint evaluation."""

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

from contract import sha, tensor_identity, write_new
from fp32_memory_overfit import checked_initialization, objective_backward, sample_cached
from fp32_pilot_support import (
    CONDITIONS,
    GRU_INITIAL_SHA,
    INITIAL_SHA,
    SCHEDULE_SHA,
    PilotCache,
    compare_receipts,
    make_memory,
    numerical_probe,
    parameter_witness,
    read,
)
from gr00t.model.modules.fp32_visual_memory import FP32VisualMemory
import memory_prototype as proto
from memory_training import group_reports, optimizer_for
from memory_training_support import (
    ACTION_KEYS,
    RAW_KEYS,
    frame_hash,
    fresh,
    native_flow,
    selected_prefix,
)
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-causal-mamba-fp32-matched-pilot-r1"
)
VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "e21de910e1f5ccd95f9fda62ca539b8f85690c89398b7d69c4cc3dd633835650"
FILES = {
    "gr00t/model/modules/fp32_visual_memory.py",
    "examples/G1_DIT_MAMBA_YUQI/fp32_memory_pilot.py",
    "examples/G1_DIT_MAMBA_YUQI/fp32_pilot_support.py",
    "examples/G1_DIT_MAMBA_YUQI/fp32_pilot_evaluation.py",
    "examples/G1_DIT_MAMBA_YUQI/README-fp32-memory-pilot.md",
    "tests/examples/test_fp32_memory_pilot.py",
}
RESULT_NAMES = {
    "preflight": "PREFLIGHT_RESULT.json",
    "train": "TRAIN_RESULT.json",
    "cache-confirmation": "CACHE_RESULT.json",
    "evaluate": "EVALUATION_RESULT.json",
}


def result_root(args, run):
    return args.artifact_root / run / "output"


def require_result(path):
    result = read(path)
    if (
        result.get("status") != "passed"
        or result.get("plan_sha256") != PLAN_SHA
        or result.get("continuation_gate") is not True
    ):
        raise ValueError(f"Missing passing prerequisite: {path}")
    return result


def prerequisites(args):
    paths = []
    if args.command in ("train", "evaluate", "cache-confirmation"):
        paths.append(result_root(args, "preflight-r1") / "RESULT.json")
    if args.command == "train":
        if args.condition not in CONDITIONS:
            raise ValueError("Unknown training condition")
        paths.extend(
            result_root(args, f"train-{c}-r1") / "RESULT.json"
            for c in CONDITIONS[: CONDITIONS.index(args.condition)]
        )
    if args.command in ("evaluate", "cache-confirmation"):
        paths.extend(result_root(args, f"train-{c}-r1") / "RESULT.json" for c in CONDITIONS)
    if args.command == "cache-confirmation" or (
        args.command == "evaluate" and args.cohort == "confirmation"
    ):
        paths.append(result_root(args, "dev-joint-r1") / "RESULT.json")
    if args.command == "evaluate" and args.cohort == "confirmation":
        paths.append(result_root(args, "cache-confirmation-r1") / "RESULT.json")
    return {str(path): {"sha256": sha(path), "result": require_result(path)} for path in paths}


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen FP32 pilot PLAN changed")
    plan = read(args.plan)
    if (
        plan["conditions"] != list(CONDITIONS)
        or plan["train"] != [125, 141, 76, 132, 203, 9, 99, 120]
        or plan["dev"] != [150, 147, 152, 112]
        or plan["confirmation"] != [52, 22, 35, 41]
    ):
        raise ValueError("Frozen cohort/condition contract changed")
    if sha(plan["pilot_schedule_file"]) != SCHEDULE_SHA:
        raise ValueError("Original500-step pilot schedule changed")
    for row in plan["records"]:
        source = Path(row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError(f"Frozen input changed: {source}")
    for relative in plan["immutable_source_relative_paths"]:
        rows = [r for r in plan["records"] if r["path"].endswith("/" + relative)]
        if (
            not rows
            or len({r["sha256"] for r in rows}) != 1
            or sha(REPO / relative) != rows[0]["sha256"]
        ):
            raise ValueError(f"Immutable native/math/helper source changed: {relative}")
    previous = read(plan["prior_overfit_result"])
    if (
        previous["status"] != "passed"
        or previous["continuation_gate"] is not True
        or previous["optimizer_steps"] != 200
    ):
        raise ValueError("Fresh FP32 overfit gate is not passed")
    return plan


def environment(plan):
    import fp32_memory_overfit
    import fp32_pilot_evaluation
    import fp32_pilot_support
    import gr00t
    from gr00t.model.gr00t_n1d7 import gr00t_n1d7
    from gr00t.model.modules import causal_visual_gru, causal_visual_memory, fp32_visual_memory
    import mamba_ssm.modules.mamba2 as official
    import memory_training
    import memory_training_support
    import run_train_eval
    import triton.language.core as core

    modules = (
        gr00t,
        gr00t_n1d7,
        causal_visual_memory,
        causal_visual_gru,
        fp32_visual_memory,
        fp32_pilot_evaluation,
        fp32_pilot_support,
        fp32_memory_overfit,
        memory_training,
        memory_training_support,
        run_train_eval,
        proto,
    )
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("Import escaped the isolated FP32 pilot worktree")
    if (
        sys.version_info[:2] != (3, 10)
        or Path(sys.prefix) != VENV
        or torch.__version__.split("+")[0] != "2.7.1"
        or VENV not in Path(torch.__file__).resolve().parents
    ):
        raise ValueError("Original Python3.10/torch2.7.1 environment required")
    if (
        importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
        or official.causal_conv1d_fn is not None
        or os.environ.get("TRITON_F32_DEFAULT") != "ieee"
    ):
        raise ValueError("Original Mamba/no-causal-conv/IEEE default contract changed")
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
    frozen = {Path(r["path"]).resolve(): r["sha256"] for r in plan["records"]}
    if any(
        Path(m.__file__).resolve() not in frozen
        or sha(m.__file__) != frozen[Path(m.__file__).resolve()]
        for m in kernels
    ):
        raise ValueError("Active kernel source differs from frozen identity")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Pretrained weights cannot be skipped")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changes = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changes.issubset(FILES):
        raise ValueError("Expected clean scoped pilot source")
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
        "dirty": dirty,
        "imports": {m.__name__: m.__file__ for m in (*modules, *kernels)},
        "python": sys.version,
        "torch": torch.__version__,
        "mamba": importlib.metadata.version("mamba-ssm"),
        "triton": importlib.metadata.version("triton"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
    }


def context_for(args, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    plan = verify_plan(args)
    if args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve():
        raise ValueError("Artifact root differs from the frozen one-attempt pilot identity")
    prior = prerequisites(args)  # No model or confirmation decoding before these gates.
    source_identity = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    context = SimpleNamespace(args=args, deadline=deadline, plan=plan, prerequisites=prior)

    def budget():
        if time.monotonic() > deadline:
            raise TimeoutError("FP32 pilot stage wall budget exceeded")
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if (
            size > plan["max_artifact_bytes"]
            or torch.cuda.max_memory_allocated() > plan["max_allocated_bytes"]
        ):
            raise RuntimeError("FP32 pilot storage/GPU hard budget exceeded")
        return size

    context.budget = budget
    context.initial = checked_initialization(plan["prototype_initialization"])
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    context.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    proto.assert_frozen(context.model)
    context.base_before = proto.content_hash(context.model.state_dict())
    context.source = NativeAnchors(Path(plan["prepared"]))
    context.source.processor.eval()
    context.budget()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "environment": source_identity,
            "strict_loading": loading,
            "base_before": context.base_before,
            "initial_content_sha256": INITIAL_SHA,
            "gru_initial_content_sha256": GRU_INITIAL_SHA,
            "schedule_sha256": SCHEDULE_SHA,
            "prerequisites": {p: {"sha256": v["sha256"]} for p, v in prior.items()},
            "warm_start": False,
            "optimizer_resume": False,
            "grad_scaler": False,
        },
    )
    return context


def cache_for(context, confirmation=False):
    if confirmation:
        root = result_root(context.args, "cache-confirmation-r1")
        report = require_result(root / "RESULT.json")
        cache = PilotCache(
            root, context.plan, confirmation=True, expected_report_sha=report["cache_result_sha256"]
        )
    else:
        cache = PilotCache(
            context.plan["cache_root"],
            context.plan,
            expected_report_sha=context.plan["cache_report_sha256"],
        )
    if cache.report["integrity"]["base_before"] != context.base_before:
        raise ValueError("Cache and live base parameter identities differ")
    return cache


def integrity(context, cache=None):
    proto.assert_frozen(context.model)
    after = proto.content_hash(context.model.state_dict())
    if context.base_before != after:
        raise ValueError("Frozen pretrained weights changed")
    if cache is not None:
        cache.assert_unchanged()
    return {
        "base_before": context.base_before,
        "base_after": after,
        "base_frozen_eval_no_grad": True,
        "cache_unchanged": cache is not None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "artifact_bytes": context.budget(),
    }


def cache_native_parity(context, cache, episodes):
    rows = []
    for episode in episodes:
        context.budget()
        anchor = context.plan["grids"][str(episode)][-1]
        frame = cache.frame(episode, anchor)
        batch, raw = context.source.batch([(episode, anchor)])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            full, receipt = native_flow(context.model, frame, None, 42, full_batch=batch)
            cached, receipt_cached = native_flow(context.model, frame, None, 42)
            zero = torch.zeros(192, 2048, device="cuda", dtype=torch.float32)
            injected, receipt_zero = native_flow(context.model, frame, zero, 42)
            for key in (
                "loss",
                "action_loss",
                "action_mask",
                "state_features",
                "backbone_features",
            ):
                if not torch.equal(full[key], cached[key]) or not torch.equal(
                    full[key], injected[key]
                ):
                    raise ValueError(f"Native/cache/zero exact flow parity failed: {key}")
            if (
                receipt != receipt_cached
                or receipt != receipt_zero
                or tensor_identity(full["backbone_features"])
                != tensor_identity(frame["post_vl_features"])
            ):
                raise ValueError("Native/cache effective condition/noise mismatch")
            set_seed(42, cuda=True)
            native = context.model.get_action(
                {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
            )["action_pred"].cpu()
            predicted = sample_cached(context.model, frame, zero, 42)
            proto.action_record(predicted, raw[0], context.source.processor, native)
        rows.append(
            {"episode": episode, "anchor": anchor, "flow_action_exact": True, "receipt": receipt}
        )
    return rows


def preflight_command(context):
    cache = cache_for(context)
    # Load all334 observations and validate their tensor identity, not merely file hashes.
    for episode in context.plan["train"] + context.plan["dev"]:
        cache.episode(episode)
    parity = cache_native_parity(context, cache, context.plan["train"] + context.plan["dev"])
    write_new(context.args.output / "CACHE_NATIVE_PARITY.json", parity)
    episode, anchor = 120, 736
    sequence, indices = cache.prefix(episode, anchor)
    if len(indices) != 47:
        raise ValueError("Longest frozen prefix changed")
    reports = []
    for condition in CONDITIONS:
        context.budget()
        module = make_memory(context.initial, condition).cuda().eval()
        initial_hash = proto.content_hash(module.state_dict())
        path = context.args.output / f"{condition}-initialization.pt"
        torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, path)
        shadow = proto.make_shadow(module)
        numerical = numerical_probe(
            context, cache, shadow, condition, episode, anchor, "preflight-shadow"
        )
        gradients = {}
        selected = selected_prefix(sequence, condition)
        for label, candidate in (("zero_initialization", module), ("nonzero_shadow", shadow)):
            candidate.zero_grad(set_to_none=True)
            evidence = objective_backward(
                context.model,
                FP32VisualMemory(candidate),
                selected,
                cache.frame(episode, anchor),
                42,
                context.plan["feature_residual_l2_coefficient"],
            )
            groups = group_reports(candidate)
            required = list(groups) if label == "nonzero_shadow" else ["output_projection"]
            gradients[label] = {
                **evidence,
                "groups": groups,
                "required_groups": required,
                "passed": all(groups[g]["finite_nonzero_parameters"] > 0 for g in required),
            }
            candidate.zero_grad(set_to_none=True)
            context.budget()
        if proto.content_hash(module.state_dict()) != initial_hash:
            raise ValueError("Shadow or backward mutated fresh initialization")
        state = module.new_state()
        reports.append(
            {
                "condition": condition,
                "initialization": {
                    "path": str(path),
                    "sha256": sha(path),
                    "content_sha256": initial_hash,
                },
                "parameters": sum(p.numel() for p in module.parameters()),
                "state": state.describe(),
                "prefix": indices,
                "effective_length": len(selected),
                "numerical": numerical,
                "gradients": gradients,
                "current_reset_A_log_L1_unreachable_allowed": condition == "current_reset_mamba",
                "passed": numerical["passed"] and all(v["passed"] for v in gradients.values()),
            }
        )
        del module, shadow, state, selected
        gc.collect()
        torch.cuda.empty_cache()
    passed = all(r["passed"] for r in reports)
    return {
        "status": "passed" if passed else "failed",
        "continuation_gate": passed,
        "optimizer_steps": 0,
        "conditions": reports,
        "cache_native_parity_rows": len(parity),
        "integrity": integrity(context, cache),
    }


def load_fresh(context, condition):
    result = require_result(result_root(context.args, "preflight-r1") / "RESULT.json")
    row = next(r for r in result["conditions"] if r["condition"] == condition)
    identity = row["initialization"]
    if sha(identity["path"]) != identity["sha256"]:
        raise ValueError("Frozen preflight initialization file changed")
    values = torch.load(identity["path"], map_location="cpu", weights_only=True)
    if proto.content_hash(values) != identity["content_sha256"]:
        raise ValueError("Frozen preflight initialization content changed")
    module = make_memory(context.initial, condition)
    module.load_state_dict(values, strict=True)
    return module.cuda(), identity


def train_command(context):
    condition = context.args.condition
    cache = cache_for(context)
    module, identity = load_fresh(context, condition)
    before = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    schedule = read(context.plan["pilot_schedule_file"])
    if len(schedule) != 500:
        raise ValueError("Exactly500 frozen optimizer steps required")
    optimizer = optimizer_for(module, context.plan)
    completed, seen = 0, {}
    try:
        with (context.args.output / "training_steps.jsonl").open("x") as stream:
            for planned in schedule:
                context.budget()
                if planned["step"] != completed + 1:
                    raise ValueError("Optimizer schedule order changed")
                sequence, indices = cache.prefix(planned["episode"], planned["anchor"])
                if indices != planned["prefix"]:
                    raise ValueError("Scheduled causal prefix mismatch")
                selected = selected_prefix(sequence, condition)
                for group in optimizer.param_groups:
                    group["lr"] = planned["lr"]
                module.train()
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                evidence = objective_backward(
                    context.model,
                    FP32VisualMemory(module),
                    selected,
                    cache.frame(planned["episode"], planned["anchor"]),
                    planned["forward_seed"],
                    context.plan["feature_residual_l2_coefficient"],
                )
                gradients = group_reports(module)
                for name, value in gradients.items():
                    seen[name] = seen.get(name, False) or value["finite_nonzero_parameters"] > 0
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    context.plan["optimizer"]["gradient_clip"],
                    error_if_nonfinite=True,
                )
                proto.assert_frozen(context.model)
                optimizer.step()
                completed += 1
                row = {
                    **planned,
                    **evidence,
                    "condition": condition,
                    "effective_length": len(selected),
                    "effective_prefix": indices[-1:]
                    if condition == "current_reset_mamba"
                    else indices,
                    "past_content_zeroed": condition == "zero_history_mamba",
                    "gradient_groups": gradients,
                    "gradient_norm_before_clip": float(gradient_norm),
                    "optimizer_steps_completed": completed,
                    "elapsed_step_seconds": time.monotonic() - tick,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                del sequence, selected
                context.budget()
                if completed % 25 == 0:
                    print(
                        json.dumps({"condition": condition, "optimizer_steps": completed}),
                        flush=True,
                    )
    except Exception:
        torch.save(
            {
                "module": {k: v.detach().cpu() for k, v in module.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "completed_steps": completed,
                "condition": condition,
                "plan_sha256": PLAN_SHA,
                "resume_authorized": False,
            },
            context.args.output / "INTERRUPTED_CHECKPOINT.pt",
        )
        raise
    optimizer.zero_grad(set_to_none=True)
    module.eval()
    checkpoint = context.args.output / "checkpoint-step500.pt"
    torch.save(
        {
            "module": {k: v.detach().cpu() for k, v in module.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "completed_steps": completed,
            "condition": condition,
            "plan_sha256": PLAN_SHA,
            "schedule_sha256": SCHEDULE_SHA,
            "initial_content_sha256": identity["content_sha256"],
            "base_content_sha256": context.base_before,
            "resume_authorized": False,
        },
        checkpoint,
    )
    final_hash = proto.content_hash(module.state_dict())
    changes = parameter_witness(module, before, seen)
    write_new(context.args.output / "GRADIENT_PARAMETER_CHANGES.json", changes)
    del optimizer
    numerical = numerical_probe(context, cache, module, condition, 120, 736, "step500")
    cache_parity = cache_native_parity(context, cache, [120])
    if proto.content_hash(module.state_dict()) != final_hash:
        raise ValueError("Terminal checks mutated frozen step500 weights")
    passed = changes["passed"] and numerical["passed"]
    return {
        "status": "passed" if passed else "failed",
        "continuation_gate": passed,
        "condition": condition,
        "optimizer_steps": completed,
        "initialization": identity,
        "schedule_sha256": SCHEDULE_SHA,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha(checkpoint),
            "module_content_sha256": final_hash,
        },
        "changes": changes,
        "numerical": numerical,
        "disabled_parity": cache_parity,
        "integrity": integrity(context, cache),
    }


def frozen_checkpoints(context):
    checkpoints, rows = {}, {}
    for condition in CONDITIONS:
        root = result_root(context.args, f"train-{condition}-r1")
        result = require_result(root / "RESULT.json")
        identity = result["checkpoint"]
        if (
            result["optimizer_steps"] != 500
            or result["condition"] != condition
            or sha(identity["path"]) != identity["sha256"]
        ):
            raise ValueError("Terminal checkpoint identity mismatch")
        checkpoints[condition] = identity
        rows[condition] = [
            json.loads(line) for line in (root / "training_steps.jsonl").read_text().splitlines()
        ]
    pairing = compare_receipts(rows, read(context.plan["pilot_schedule_file"]))
    if context.args.command == "cache-confirmation" or (
        context.args.command == "evaluate" and context.args.cohort == "confirmation"
    ):
        dev = require_result(result_root(context.args, "dev-joint-r1") / "RESULT.json")
        if checkpoints != dev["checkpoints"]:
            raise ValueError(
                "Confirmation must use exactly the checkpoint inventory that passed dev"
            )
    write_new(context.args.output / "FROZEN_CHECKPOINTS.json", checkpoints)
    write_new(context.args.output / "TRAINING_PAIRING.json", pairing)
    return checkpoints


def cache_confirmation_command(context):
    files = []
    checkpoints = frozen_checkpoints(context)
    for episode in context.plan["confirmation"]:
        values = {}
        for anchor in context.plan["grids"][str(episode)]:
            context.budget()
            batch, raw = context.source.batch([(episode, anchor)])
            observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
            proto.validate_input_layout(observation, context.source.modality["video"].modality_keys)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                backbone_input, action_input = context.model.prepare_input(dict(batch))
                backbone = context.model.backbone(backbone_input)
                raw_cpu = {k: backbone[k].detach().cpu().clone() for k in RAW_KEYS}
                clean_raw = fresh({k: backbone[k] for k in RAW_KEYS}, context.model.device)
                clean_action = fresh(
                    {k: action_input[k] for k in ACTION_KEYS}, context.model.device
                )
                features = context.model.action_head._encode_features(clean_raw, clean_action)
                proto.validate_packet({"features": features, "backbone": clean_raw}, observation)
            frame = {
                "episode": episode,
                "anchor": anchor,
                "raw": raw_cpu,
                "post_vl_features": features.backbone_features.detach().cpu().clone(),
                "action_input": {k: action_input[k].detach().cpu().clone() for k in ACTION_KEYS},
                "physical_states": {
                    k: torch.as_tensor(v).clone() for k, v in raw[0].states.items()
                },
                "physical_actions": {
                    k: torch.as_tensor(v).clone() for k, v in raw[0].actions.items()
                },
                "input_identity": {k: tensor_identity(v) for k, v in batch.items()},
            }
            if any(
                not bool(torch.isfinite(v).all())
                for v in (
                    *frame["raw"].values(),
                    frame["post_vl_features"],
                    *frame["action_input"].values(),
                )
            ):
                raise ValueError("Nonfinite confirmation cache tensor")
            frame["content_sha256"] = frame_hash(frame)
            values[anchor] = frame
        path = context.args.output / f"episode-{episode:06d}.pt"
        torch.save(values, path)
        files.append(
            {
                "episode": episode,
                "file": path.name,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                "observations": len(values),
            }
        )
    if sum(r["observations"] for r in files) != 119:
        raise ValueError("Confirmation grid count changed")
    result = {
        "status": "passed",
        "continuation_gate": True,
        "plan_sha256": PLAN_SHA,
        "cohort": "confirmation",
        "episodes": context.plan["confirmation"],
        "files": files,
        "observations": 119,
        "optimizer_steps": 0,
        "raw_post_separate": True,
        "checkpoints": checkpoints,
        "integrity": integrity(context),
    }
    write_new(context.args.output / "CACHE_RESULT.json", result)
    return {**result, "cache_result_sha256": sha(context.args.output / "CACHE_RESULT.json")}


def expected_output(args):
    if args.command == "preflight":
        run = "preflight-r1"
    elif args.command == "train":
        run = f"train-{args.condition}-r1"
    elif args.command == "cache-confirmation":
        run = "cache-confirmation-r1"
    else:
        run = f"{args.cohort}-joint-r1"
    return result_root(args, run).resolve()


def execute(args):
    if args.output.resolve() != expected_output(args):
        raise ValueError("Only one canonical output root per frozen stage/condition is allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        context = context_for(args, started + args.wall_seconds)
        if args.command == "preflight":
            report = preflight_command(context)
        elif args.command == "train":
            report = train_command(context)
        elif args.command == "cache-confirmation":
            report = cache_confirmation_command(context)
        else:
            from fp32_pilot_evaluation import evaluate_command

            report = evaluate_command(context)
        report.update(
            plan_sha256=PLAN_SHA,
            elapsed_seconds=time.monotonic() - started,
            old_failures_unchanged=True,
            automatic_retry=False,
        )
        del context
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        report["gpu_after_release"] = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        alias = args.output / RESULT_NAMES[args.command]
        if not alias.exists():
            write_new(alias, report)
        write_new(args.output / "RESULT.json", report)
    except Exception as exc:
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "technical_validity": "not_established",
                "continuation_gate": False,
                "plan_sha256": PLAN_SHA,
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
                "optimizer_resume_authorized": False,
                "old_failures_unchanged": True,
            },
        )
        raise
    if not report["continuation_gate"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=RESULT_NAMES)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--cohort", choices=("dev", "confirmation"))
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= (7150 if args.command == "train" else 1750):
        parser.error("Frozen wall budget exceeded")
    if args.command == "train" and args.condition is None:
        parser.error("Train requires --condition")
    if args.command == "evaluate" and args.cohort is None:
        parser.error("Evaluate requires --cohort")
    if args.artifact_root.resolve() not in args.output.resolve().parents:
        parser.error("Output must belong to the isolated artifact root")
    execute(args)


if __name__ == "__main__":
    main()
