"""Frozen-cache, zero-update preflight and one bounded memory overfit attempt.

This first handoff deliberately exposes only cache/preflight/overfit. Pilot and
joint evaluation commands require a later reviewed implementation after the
overfit gate. No command retries, resumes, changes the plan, or trains the base.
"""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from contract import sha, tensor_identity, write_new
from gr00t.model.modules.causal_visual_gru import CausalVisualGRU
from gr00t.model.modules.causal_visual_memory import CausalVisualMemory
import memory_prototype as prototype
from memory_training_support import (
    ACTION_KEYS,
    CONDITIONS,
    RAW_KEYS,
    FeatureCache,
    authorize_cache_cohort,
    frame_hash,
    fresh,
    native_flow,
    overfit_gate,
    prepare_residual,
    sampling_packet,
)
import numpy as np
from runner_core import aggregate_metrics, deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-pretrained-causal-mamba-memory-training-r1"
)
EXPECTED_VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "2cd77859437a88640d6906d48d388717521f0b3a27bbe46dd3c578b63264370d"
EVALUATION_SHA = "102116be60a56dcfd87c7811b25e5f176c166b4101c0f5b5a486d931894c7acd"
NUMERICAL_SHA = "07ce2b7e1af6cb5e8294d4ae0e6a5a58e573d920b1173c38892ae7ecd02741e3"
SCHEDULE_SHAS = {
    "OVERFIT_SCHEDULE.json": "3b845ac1497b8912c41ec4833f5695caf59d1d911e2564db56ee187e52f25e02",
    "PILOT_SCHEDULE.json": "30eb1b694947f5f56f070f540dd6b85461841b8d836820471c62613f4db96cc2",
}
MAX_STORAGE = 8 * 2**30
MAX_ALLOCATED = 44 * 2**30
RESULT_NAMES = {
    "cache": "CACHE_RESULT.json",
    "preflight": "PREFLIGHT_RESULT.json",
    "overfit": "OVERFIT_RESULT.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def budget(context):
    if time.monotonic() > context.deadline:
        raise TimeoutError("Memory training stage exceeded its frozen wall budget")
    total = sum(p.stat().st_size for p in context.args.artifact_root.rglob("*") if p.is_file())
    if total > MAX_STORAGE:
        raise RuntimeError("Memory training artifacts exceeded total 8 GiB budget")
    if torch.cuda.max_memory_allocated() > MAX_ALLOCATED:
        raise RuntimeError("Memory training exceeded 44 GiB allocated GPU memory")
    return total


def require_result(path, plan_hash=PLAN_SHA):
    value = read(path)
    if value.get("status") != "passed" or value.get("plan_sha256") != plan_hash:
        raise ValueError(f"Missing matching prerequisite gate: {path}")
    return value


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen PLAN identity changed")
    plan = read(args.plan)
    if plan["status"] != "frozen_before_optimizer_step0" or args.checkpoint.resolve() != Path(
        plan["base_checkpoint"]
    ):
        raise ValueError("Expected frozen original Yuqi17000 training plan")
    if plan["conditions"] != list(CONDITIONS) or set(plan["train"] + plan["dev"]) & set(
        plan["confirmation"]
    ):
        raise ValueError("Condition order or episode partition changed")
    evaluation_path = args.plan.with_name("EVALUATION_CONTRACT.json")
    if sha(evaluation_path) != EVALUATION_SHA:
        raise ValueError("Frozen evaluation RNG contract changed")
    numerical_path = args.plan.with_name("NUMERICAL_CONTRACT.json")
    if sha(numerical_path) != NUMERICAL_SHA:
        raise ValueError("Frozen full/step numerical contract changed")
    for row in read(numerical_path)["sources"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Frozen numerical predicate source changed")
        if (
            Path(row["path"]).name == "memory_prototype.py"
            and sha(prototype.__file__) != row["sha256"]
        ):
            raise ValueError("Active numerical helper differs from the frozen comparator")
    evaluation = read(evaluation_path)
    for row in evaluation["sources"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"Evaluation contract source changed: {row['path']}")
    seeds = {}
    for row in evaluation["evaluation_seeds"]:
        key = (row["episode"], row["anchor"], row["nominal_seed"])
        if row["actual_seed"] != deterministic_seed("diagnostic", *key) or key in seeds:
            raise ValueError("Frozen evaluation seed mapping changed")
        seeds[key] = row["actual_seed"]
    for name, expected in SCHEDULE_SHAS.items():
        if sha(args.plan.with_name(name)) != expected:
            raise ValueError(f"Frozen shared schedule changed: {name}")
    # Confirmation is only hashed here, never decoded or fed to a model.
    for row in plan["source_files"] + plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen plan material changed: {path}")
    gate = read(plan["prototype_result"])
    if not (
        gate["status"] == gate["technical_validity"] == gate["cost_gate"] == "passed"
        and gate["optimizer_steps"] == 0
    ):
        raise ValueError("The original zero-step prototype must have passed all gates")
    prototype_identity = prototype.verify_frozen_inputs(SimpleNamespace(checkpoint=args.checkpoint))
    return plan, seeds, prototype_identity


def verify_environment():
    import gr00t
    from gr00t.model.gr00t_n1d7 import gr00t_n1d7
    from gr00t.model.modules import causal_visual_gru, causal_visual_memory
    import mamba_ssm.modules.mamba2 as official
    import memory_training_support
    import run_train_eval

    modules = (
        gr00t,
        gr00t_n1d7,
        causal_visual_gru,
        causal_visual_memory,
        run_train_eval,
        memory_training_support,
        prototype,
    )
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("A source import escaped the frozen training worktree")
    if sys.version_info[:2] != (3, 10) or Path(sys.prefix) != EXPECTED_VENV:
        raise ValueError("Expected original Python 3.10 environment")
    if (
        torch.__version__.split("+")[0] != "2.7.1"
        or EXPECTED_VENV not in Path(torch.__file__).resolve().parents
    ):
        raise ValueError("Expected original torch 2.7.1")
    overlay = prototype.ARTIFACTS / "g1-dit-mamba-yuqi-5000step-r1/dependencies/python-overlay"
    if (
        overlay not in Path(official.__file__).resolve().parents
        or official.causal_conv1d_fn is not None
    ):
        raise ValueError("Expected original Mamba overlay without causal-conv1d")
    if importlib.metadata.version("mamba-ssm") != "2.3.2.post1":
        raise ValueError("Mamba package version changed")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Skipping pretrained weights is forbidden")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty:
        raise ValueError(f"Expected committed clean source: {dirty}")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "dirty": dirty,
        "imports": {m.__name__: m.__file__ for m in modules},
        "python": sys.version,
        "torch": torch.__version__,
        "mamba_file": official.__file__,
        "pythonpath": os.environ.get("PYTHONPATH"),
    }


def context_for(args, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    # Fail prerequisites before any model output or confirmation data decoding.
    plan, seeds, ancestor_identity = verify_plan(args)
    if args.command == "cache":
        if args.cohort != "train_dev":
            raise ValueError("Confirmation cache is outside this first-stage implementation")
        authorize_cache_cohort(plan, args.cohort)
    else:
        require_result(args.artifact_root / "cache-train-dev-r1/output/CACHE_RESULT.json")
        if args.command == "overfit":
            require_result(
                args.artifact_root / "training-preflight-r1/output/PREFLIGHT_RESULT.json"
            )
    environment = verify_environment()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose only the authorized GPU0")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    context = SimpleNamespace(args=args, deadline=deadline, plan=plan, seeds=seeds)
    budget(context)
    model, loading = strict_reload(args.checkpoint)
    context.model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    context.model.requires_grad_(False)
    prototype.assert_frozen(context.model)
    context.base_before = prototype.content_hash(context.model.state_dict())
    context.source = NativeAnchors(Path(plan["prepared"]))
    context.source.processor.eval()
    context.initial = torch.load(
        plan["prototype_initialization"], map_location="cpu", weights_only=True
    )
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "evaluation_contract_sha256": EVALUATION_SHA,
            "numerical_contract_sha256": NUMERICAL_SHA,
            "schedule_sha256": SCHEDULE_SHAS,
            "source": environment,
            "ancestor_inputs": ancestor_identity,
            "strict_loading": loading,
            "base_content_sha256": context.base_before,
            "initial_memory_content_sha256": prototype.content_hash(context.initial),
            "scope": plan["confirmation_scope"],
            "base_mode": "eval; state dropout disabled, distinct from prior full fine-tuning",
        },
    )
    return context


def make_adapter(context, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "gru_history":
        module = CausalVisualGRU(context.initial)
    else:
        module = CausalVisualMemory()
        module.load_state_dict(context.initial, strict=True)
    if any(p.dtype != torch.float32 for p in module.parameters()):
        raise ValueError("New adapter master parameters must remain FP32")
    return module.to("cuda")


def close_integrity(context, cache=None):
    prototype.assert_frozen(context.model)
    after = prototype.content_hash(context.model.state_dict())
    if after != context.base_before:
        raise ValueError("Frozen pretrained parameter content changed")
    if cache is not None:
        cache.assert_unchanged()
    return {
        "base_before": context.base_before,
        "base_after": after,
        "base_frozen_eval_no_grad": True,
        "cache_content_unchanged": cache is not None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "artifact_bytes": budget(context),
    }


def cache_command(context):
    model, source, plan = context.model, context.source, context.plan
    files = []
    for episode in authorize_cache_cohort(plan, "train_dev"):
        values = {}
        for anchor in plan["grids"][str(episode)]:
            budget(context)
            batch, raw = source.batch([(episode, anchor)])
            observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
            prototype.validate_input_layout(observation, source.modality["video"].modality_keys)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                backbone_input, action_input = model.prepare_input(dict(batch))
                backbone = model.backbone(backbone_input)
                # Save raw BEFORE native _encode_features mutates its container.
                raw_cpu = {k: backbone[k].detach().cpu().clone() for k in RAW_KEYS}
                clean_raw = fresh({k: backbone[k] for k in RAW_KEYS}, model.device)
                clean_action = fresh({k: action_input[k] for k in ACTION_KEYS}, model.device)
                features = model.action_head._encode_features(clean_raw, clean_action)
                prototype.validate_packet(
                    {"features": features, "backbone": clean_raw}, observation
                )
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
            for value in [
                *frame["raw"].values(),
                frame["post_vl_features"],
                *frame["action_input"].values(),
            ]:
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite feature-cache tensor")
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
        print(json.dumps({"cached_episode": episode, "observations": len(values)}), flush=True)
    return {
        "status": "passed",
        "plan_sha256": PLAN_SHA,
        "cohort": "train_dev",
        "episodes": plan["train"] + plan["dev"],
        "files": files,
        "observations": sum(r["observations"] for r in files),
        "optimizer_steps": 0,
        "raw_post_separate": True,
        "integrity": close_integrity(context),
    }


def cache_for(context):
    cache = FeatureCache(
        context.args.artifact_root / "cache-train-dev-r1/output", context.plan, PLAN_SHA
    )
    if cache.report["integrity"]["base_before"] != context.base_before:
        raise ValueError("Cached features and current live base weights differ")
    return cache


def raw_record(frame):
    return SimpleNamespace(
        states={k: v.numpy() for k, v in frame["physical_states"].items()},
        actions={k: v.numpy() for k, v in frame["physical_actions"].items()},
    )


def group_reports(module):
    groups = {
        "input_projection": module.input_projection,
        "output_projection": module.output_projection,
    }
    for index, layer in enumerate(module.layers):
        groups[f"norm{index}"] = layer.norm
        groups[f"recurrence{index}"] = layer.mamba if hasattr(layer, "mamba") else layer.cell
    return {name: prototype.gradient_group(value) for name, value in groups.items()}


def sample_cached(context, frame, residual, seed):
    packet = sampling_packet(context.model, frame)
    set_seed(seed, cuda=True)
    return prototype.sample_packet(context.model, packet, residual)


def numerical_probe(context, cache, module, condition, episode, anchor, label):
    sequence, indices = cache.prefix(episode, anchor, "cuda")
    frame = cache.frame(episode, anchor)
    module.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full, full_residual, effective_length = prepare_residual(module, sequence, condition)
        step, step_residual, _ = prepare_residual(module, sequence, condition, deployment=True)
        latent = prototype.differences(step, full)
        prototype.check_tolerance(latent)
        actual = sample_cached(context, frame, step_residual, 42)
        expected = sample_cached(context, frame, full_residual, 42)
        action = prototype.differences(actual[:, :16], expected[:, :16])
        prototype.check_tolerance(action, action=True)
        prototype.action_record(actual, raw_record(frame), context.source.processor)
        prototype.action_record(expected, raw_record(frame), context.source.processor)
    return {
        "condition": condition,
        "label": label,
        "episode": episode,
        "anchor": anchor,
        "prefix": indices,
        "effective_length": effective_length,
        "latent": latent,
        "active_action": action,
        "status": "passed",
    }


def preflight_command(context):
    cache = cache_for(context)
    parity = []
    for episode in context.plan["train"] + context.plan["dev"]:
        anchor = context.plan["grids"][str(episode)][-1]
        frame = cache.frame(episode, anchor)
        batch, raw = context.source.batch([(episode, anchor)])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            full, full_noise = native_flow(context.model, frame, None, 42, full_batch=batch)
            cached, cached_noise = native_flow(context.model, frame, None, 42)
            zero = torch.zeros(192, 2048, device="cuda", dtype=frame["post_vl_features"].dtype)
            hooked, hook_noise = native_flow(context.model, frame, zero, 42)
            for name in (
                "loss",
                "action_loss",
                "action_mask",
                "state_features",
                "backbone_features",
            ):
                if not torch.equal(full[name], cached[name]) or not torch.equal(
                    full[name], hooked[name]
                ):
                    raise ValueError(f"Native/raw-cache/zero-hook exact flow parity failed: {name}")
            if full_noise != cached_noise or full_noise != hook_noise:
                raise ValueError(
                    "Native flow draws or effective conditioning changed at zero injection"
                )
            if tensor_identity(full["backbone_features"]) != tensor_identity(
                frame["post_vl_features"]
            ):
                raise ValueError("Cached post-VL tensor is not the native postprocess result")
            set_seed(42, cuda=True)
            native = context.model.get_action(
                {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
            )["action_pred"]
            predicted = sample_cached(context, frame, zero, 42)
            _, action = prototype.action_record(
                predicted, raw[0], context.source.processor, native.cpu()
            )
        parity.append(
            {
                "episode": episode,
                "anchor": anchor,
                "flow_exact": True,
                "action_exact": True,
                "noise_receipt": full_noise,
                **action,
            }
        )
    write_new(
        context.args.output / "CACHE_NATIVE_PARITY.json", {"status": "passed", "rows": parity}
    )
    longest = max(
        context.plan["train"] + context.plan["dev"],
        key=lambda ep: len(context.plan["grids"][str(ep)]),
    )
    anchor = context.plan["grids"][str(longest)][-1]
    frame = cache.frame(longest, anchor)
    reports = []
    for condition in CONDITIONS:
        budget(context)
        module = make_adapter(context, condition).eval()
        before = prototype.content_hash(module.state_dict())
        initial_path = context.args.output / f"{condition}-initialization.pt"
        torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, initial_path)
        state = module.new_state()
        state_description = state.describe()
        del state
        shadow = prototype.make_shadow(module)
        numerical = numerical_probe(
            context, cache, shadow, condition, longest, anchor, "nonzero-shadow-initialization"
        )
        sequence, indices = cache.prefix(longest, anchor, "cuda")
        gradient_evidence = {}
        for label, candidate in [("zero_initialization", module), ("nonzero_shadow", shadow)]:
            candidate.train()
            candidate.zero_grad(set_to_none=True)
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, residual, effective_length = prepare_residual(candidate, sequence, condition)
                result, receipt = native_flow(context.model, frame, residual, 42)
                objective = result["loss"] + 0.01 * residual.float().square().mean()
                objective.backward()
            gradients = group_reports(candidate)
            required = list(gradients) if label == "nonzero_shadow" else ["output_projection"]
            if any(gradients[name]["finite_nonzero_parameters"] == 0 for name in required):
                raise ValueError(
                    f"Longest-prefix reachable gradient group failed: {condition} {label}"
                )
            gradient_evidence[label] = gradients
            candidate.zero_grad(set_to_none=True)
        prototype.assert_frozen(context.model)
        if prototype.content_hash(module.state_dict()) != before:
            raise ValueError("Shadow preflight mutated initial module")
        reports.append(
            {
                "condition": condition,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                ),
                "master_dtypes": sorted({str(p.dtype) for p in module.parameters()}),
                "state": state_description,
                "initialization_sha256": sha(initial_path),
                "initial_content_sha256": before,
                "prefix": indices,
                "effective_length": effective_length,
                "numerical": numerical,
                "gradients": gradient_evidence,
                "noise_receipt": receipt,
                "native_loss": float(result["loss"].detach()),
                "objective": float(objective.detach()),
                "current_reset_A_log_L1_unreachable_allowed": condition == "current_reset_mamba",
            }
        )
        del module, shadow, sequence, residual, result, objective
        gc.collect()
        torch.cuda.empty_cache()
        budget(context)
    write_new(
        context.args.output / "LONG_PREFIX_PREFLIGHT.json",
        {"status": "passed", "conditions": reports, "optimizer_steps": 0},
    )
    return {
        "status": "passed",
        "plan_sha256": PLAN_SHA,
        "optimizer_steps": 0,
        "cache_native_parity_rows": len(parity),
        "conditions": reports,
        "integrity": close_integrity(context, cache),
    }


def evaluate_adapter(context, cache, module, condition, episode, output):
    from run_train_eval import METRICS, physical_metrics

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    predictions = []
    module.eval()
    for anchor in context.plan["eval_anchors"][str(episode)]:
        budget(context)
        frame = cache.frame(episode, anchor)
        sequence, indices = cache.prefix(episode, anchor, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, residual, effective_length = prepare_residual(
                module, sequence, condition, deployment=True
            )
            for nominal in context.plan["noise_seeds"]:
                seed = context.seeds[(episode, anchor, nominal)]
                result, receipt = native_flow(context.model, frame, residual, seed)
                predicted = sample_cached(context, frame, residual, seed)
                value, action = prototype.action_record(
                    predicted, raw_record(frame), context.source.processor
                )
                metrics = physical_metrics(
                    value[0].float().numpy(), raw_record(frame), context.source.processor
                )
                from gr00t.data.embodiment_tags import EmbodimentTag

                decoded = context.source.processor.decode_action(
                    value[0].float().numpy(),
                    EmbodimentTag.NEW_EMBODIMENT,
                    state=raw_record(frame).states,
                )
                hand = np.abs(decoded["left_hand"] - raw_record(frame).actions["left_hand"]).mean(
                    axis=0
                )
                row = {
                    "episode": episode,
                    "anchor": anchor,
                    "seed": nominal,
                    "paired_rng_seed": seed,
                    "prefix": indices,
                    "effective_length": effective_length,
                    "flow_loss": float(result["loss"]),
                    **metrics,
                    **{f"hand_mae_u_{i}": float(v) for i, v in enumerate(hand)},
                    "noise_receipt": receipt,
                    **action,
                }
                if not all(
                    np.isfinite(row[k]) for k in (*METRICS, *[f"hand_mae_u_{i}" for i in range(6)])
                ):
                    raise ValueError("Nonfinite evaluation metric")
                rows.append(row)
                predictions.append(value[0].float().numpy())
    per_episode, metrics = aggregate_metrics(
        rows, (*METRICS, *[f"hand_mae_u_{i}" for i in range(6)])
    )
    write_new(output / "details.json", rows)
    np.save(output / "normalized_predictions.npy", np.stack(predictions))
    report = {
        "status": "passed",
        "condition": condition,
        "rows": len(rows),
        "per_episode": per_episode,
        "metrics": metrics,
        "aggregation": "seed -> anchor -> episode equal weighting",
        "prediction_sha256": sha(output / "normalized_predictions.npy"),
        "flow_objective": "native head.forward loss including mask.sum()+1e-6",
        "deployment_path": "memory.step",
    }
    write_new(output / "EVALUATION_RESULT.json", report)
    return report


def optimizer_for(module, plan):
    decay = []
    no_decay = []
    for name, parameter in module.named_parameters():
        target = (
            decay
            if parameter.ndim >= 2 and not getattr(parameter, "_no_weight_decay", False)
            else no_decay
        )
        target.append(parameter)
    options = plan["optimizer"]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": options["matrix_weight_decay"]},
            {"params": no_decay, "weight_decay": options["bias_norm_no_weight_decay"]},
        ],
        lr=options["lr"],
        betas=tuple(options["betas"]),
        eps=options["eps"],
    )


def learned_change_witness(module, before, seen_gradients):
    changes = {
        name: not torch.equal(value.detach().cpu(), before[name])
        for name, value in module.state_dict().items()
    }
    groups = {
        "input_projection": "input_projection.",
        "output_projection": "output_projection.",
        "norm0": "layers.0.norm.",
        "norm1": "layers.1.norm.",
        "recurrence0": "layers.0.mamba.",
        "recurrence1": "layers.1.mamba.",
    }
    report = {
        group: {
            "finite_nonzero_gradient_seen": seen_gradients.get(group, False),
            "changed_parameters": [
                name for name, changed in changes.items() if changed and name.startswith(prefix)
            ],
        }
        for group, prefix in groups.items()
    }
    if any(
        not row["finite_nonzero_gradient_seen"] or not row["changed_parameters"]
        for row in report.values()
    ):
        raise ValueError("A reachable overfit module lacks gradient or parameter-change evidence")
    return report


def overfit_command(context):
    cache = cache_for(context)
    module = make_adapter(context, "mamba_history")
    before = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    initial_hash = prototype.content_hash(before)
    episode = context.plan["overfit_episode"]
    baseline = evaluate_adapter(
        context, cache, module, "mamba_history", episode, context.args.output / "step0"
    )
    rows = read(context.args.plan.with_name("OVERFIT_SCHEDULE.json"))
    if len(rows) != 200 or context.plan["overfit_steps"] != 200:
        raise ValueError("Expected exactly 200 frozen overfit steps")
    optimizer = optimizer_for(module, context.plan)
    seen = {}
    completed = 0
    try:
        with (context.args.output / "training_steps.jsonl").open("x") as stream:
            for schedule in rows:
                budget(context)
                if schedule["step"] != completed + 1 or schedule["episode"] != episode:
                    raise ValueError("Frozen overfit schedule order changed")
                sequence, indices = cache.prefix(episode, schedule["anchor"], "cuda")
                if indices != schedule["prefix"]:
                    raise ValueError("Training prefix differs from the frozen schedule")
                for group in optimizer.param_groups:
                    group["lr"] = schedule["lr"]
                module.train()
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, residual, effective_length = prepare_residual(
                        module, sequence, "mamba_history"
                    )
                    result, receipt = native_flow(
                        context.model,
                        cache.frame(episode, schedule["anchor"]),
                        residual,
                        schedule["forward_seed"],
                    )
                    feature_penalty = residual.float().square().mean()
                    objective = (
                        result["loss"]
                        + context.plan["feature_residual_l2_coefficient"] * feature_penalty
                    )
                objective.backward()
                gradients = group_reports(module)
                for name, value in gradients.items():
                    seen[name] = seen.get(name, False) or value["finite_nonzero_parameters"] > 0
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    context.plan["optimizer"]["gradient_clip"],
                    error_if_nonfinite=True,
                )
                prototype.assert_frozen(context.model)
                optimizer.step()
                completed += 1
                row = {
                    **schedule,
                    "condition": "mamba_history",
                    "effective_length": effective_length,
                    "native_loss": float(result["loss"].detach()),
                    "feature_l2": float(feature_penalty.detach()),
                    "objective": float(objective.detach()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "gradient_groups": gradients,
                    "noise_receipt": receipt,
                    "elapsed_step_seconds": time.monotonic() - tick,
                    "optimizer_steps_completed": completed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                del sequence, residual, result, objective, feature_penalty
                budget(context)
                if completed % 25 == 0:
                    print(json.dumps({"overfit_steps": completed}), flush=True)
    except Exception:
        torch.save(
            {
                "module": {k: v.detach().cpu() for k, v in module.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "completed_steps": completed,
                "condition": "mamba_history",
                "plan_sha256": PLAN_SHA,
                "resume_authorized": False,
            },
            context.args.output / "INTERRUPTED_CHECKPOINT.pt",
        )
        raise
    optimizer.zero_grad(set_to_none=True)
    changed = learned_change_witness(module, before, seen)
    checkpoint = context.args.output / "checkpoint-step200.pt"
    torch.save(
        {
            "module": {k: v.detach().cpu() for k, v in module.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "completed_steps": completed,
            "condition": "mamba_history",
            "plan_sha256": PLAN_SHA,
            "schedule_sha256": SCHEDULE_SHAS["OVERFIT_SCHEDULE.json"],
            "initial_content_sha256": initial_hash,
            "base_content_sha256": context.base_before,
        },
        checkpoint,
    )
    after = evaluate_adapter(
        context, cache, module, "mamba_history", episode, context.args.output / "step200"
    )
    last = context.plan["grids"][str(episode)][-1]
    numerical = numerical_probe(
        context, cache, module, "mamba_history", episode, last, "learned-step200"
    )
    # Disabled branch is the untouched full native get_action, checked against raw cache sampling.
    frame = cache.frame(episode, last)
    batch, raw = context.source.batch([(episode, last)])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        set_seed(42, cuda=True)
        disabled = context.model.get_action(
            {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
        )["action_pred"]
        reference = sample_cached(context, frame, None, 42)
        prototype.action_record(disabled, raw[0], context.source.processor, reference.cpu())
    gate = overfit_gate(baseline, after)
    evidence = {
        "before": baseline,
        "after": after,
        "gate": gate,
        "gradient_and_changes": changed,
        "numerical": numerical,
        "disabled_native_exact": True,
        "checkpoint_sha256": sha(checkpoint),
        "initial_content_sha256": initial_hash,
        "final_content_sha256": prototype.content_hash(module.state_dict()),
        "integrity": close_integrity(context, cache),
    }
    write_new(context.args.output / "OVERFIT_EVIDENCE.json", evidence)
    if not all(gate.values()):
        raise ValueError(f"Frozen overfit continuation gate failed: {gate}")
    return {
        "status": "passed",
        "plan_sha256": PLAN_SHA,
        "optimizer_steps": completed,
        "continuation_gate": True,
        "overfit_not_pilot_initialization": True,
        "evidence": evidence,
        "scope": "One-episode adapter trainability; not memory quality/generalization or pilot success",
    }


def execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        context = context_for(args, started + args.wall_seconds)
        report = {
            "cache": cache_command,
            "preflight": preflight_command,
            "overfit": overfit_command,
        }[args.command](context)
        report["elapsed_seconds"] = time.monotonic() - started
        write_new(args.output / RESULT_NAMES[args.command], report)
    except Exception as exc:
        write_new(
            args.output / RESULT_NAMES[args.command],
            {
                "status": "failed",
                "plan_sha256": PLAN_SHA,
                "continuation_gate": False,
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(RESULT_NAMES))
    for name in ("plan", "checkpoint", "artifact-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=1750)
    parser.add_argument("--cohort", default="train_dev")
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1750:
        parser.error("This stage requires wall budget in (0,1750] seconds")
    execute(args)


if __name__ == "__main__":
    main()
