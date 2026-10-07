"""One fresh-initialization, single-episode FP32 memory overfit; no pilot."""

import argparse
from contextlib import ExitStack
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from contract import sha, write_new
from gr00t.model.modules.causal_visual_memory import CausalVisualMemory
from gr00t.model.modules.fp32_visual_memory import FP32VisualMemory
from memory_numerical_diagnosis import base_parity, check_action, old_gate
from memory_numerical_trace import error_summary
import memory_prototype as proto
from memory_training import group_reports, learned_change_witness, optimizer_for, raw_record
from memory_training_support import frame_hash, native_flow, overfit_gate, sampling_packet
import numpy as np
from runner_core import aggregate_metrics, deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-causal-mamba-fp32-overfit-r1"
)
VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "a54edc0e382fec015d85ead08fa948754726e2d7bafd1f6cd132b1fb844d138c"
INITIAL_SHA = "08469967ccb48df87bd2094a9aa7a6a9fb6a144bac8b5d7a2752eeaaf90b9c55"
SCHEDULE_SHA = "3b845ac1497b8912c41ec4833f5695caf59d1d911e2564db56ee187e52f25e02"
FILES = {
    "examples/G1_DIT_MAMBA_YUQI/fp32_memory_overfit.py",
    "examples/G1_DIT_MAMBA_YUQI/README-fp32-memory-overfit.md",
    "tests/examples/test_fp32_memory_overfit.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def precision_flags():
    return {
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def checked_initialization(path):
    values = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(values, dict)
        or not values
        or not all(isinstance(v, torch.Tensor) for v in values.values())
    ):
        raise ValueError(
            "Expected plain prototype state_dict; checkpoint/optimizer resume forbidden"
        )
    if proto.content_hash(values) != INITIAL_SHA:
        raise ValueError("Only the frozen fresh prototype initialization is permitted")
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError("Expected finite FP32 prototype parameters")
    if torch.count_nonzero(values["output_projection.weight"]) or torch.count_nonzero(
        values["output_projection.bias"]
    ):
        raise ValueError("Fresh output projection must be exactly zero")
    return values


def sequence_for(cache, anchor, device):
    indices = list(range(0, anchor + 1, 16))
    if any(a not in cache for a in indices):
        raise ValueError("Missing causal episode125 prefix")
    return torch.stack(
        [cache[a]["post_vl_features"][0, proto.IMAGE_POSITIONS] for a in indices]
    ).to(device), indices


def objective_backward(model, executor, sequence, frame, seed, coefficient):
    """Memory forward FP32; base forward BF16; backward OUTSIDE autocast."""
    forward_memory = []
    with ExitStack() as scope:
        for label, module in [
            ("input_projection", executor.memory.input_projection),
            ("readout", executor.memory.output_projection),
        ]:

            def capture(_module, args, label=label):
                forward_memory.append({"label": label, **precision_flags()})

            scope.callback(module.register_forward_pre_hook(capture).remove)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent, residual = executor.full(sequence)
            base_forward = precision_flags()
            result, receipt = native_flow(model, frame, residual, seed)
            penalty = residual.float().square().mean()
            objective = result["loss"] + coefficient * penalty
            if not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(objective)):
                raise ValueError("Nonfinite FP32 training latent or objective")
        backward = precision_flags()
        if (
            backward["cuda_autocast_enabled"]
            or not backward["matmul_allow_tf32"]
            or not backward["cudnn_allow_tf32"]
        ):
            raise ValueError("Backward must be outside autocast with original ambient TF32=true")
        objective.backward()
    return {
        "native_loss": float(result["loss"].detach()),
        "feature_l2": float(penalty.detach()),
        "objective": float(objective.detach()),
        "noise_receipt": receipt,
        "precision": {
            "forward_memory": forward_memory,
            "base_forward": base_forward,
            "backward_ambient": backward,
            "scope": "Local FP32 memory forward; ambient-TF32 backward, no all-IEEE gradient claim",
        },
    }


def step_prefix(memory, executor, sequence):
    state = memory.new_state()
    with torch.no_grad():
        outputs = []
        for current in sequence:
            latent, residual = executor.step(current, state)
            outputs.append(latent)
    if state.observations != len(sequence):
        raise ValueError("FP32 deployment prefix observation count changed")
    latent = torch.stack(outputs)
    if not all(
        bool(torch.isfinite(value).all())
        for value in (latent, residual, *[t for pair in state.layers for t in pair])
    ):
        raise ValueError("Nonfinite FP32 deployment latent/residual/cache")
    return latent, residual, state


def sample_cached(model, frame, residual, seed):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        packet = sampling_packet(model, frame)
        set_seed(seed, cuda=True)
        return check_action(proto.sample_packet(model, packet, residual))


def budget(plan, output, deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("FP32 overfit wall budget exceeded")
    size = sum(p.stat().st_size for p in output.parent.parent.rglob("*") if p.is_file())
    if size > plan["max_artifact_bytes"]:
        raise RuntimeError("FP32 overfit exceeded 1 GiB artifacts")
    if torch.cuda.max_memory_allocated() > plan["max_allocated_bytes"]:
        raise RuntimeError("FP32 overfit exceeded 12 GiB allocated GPU memory")
    return size


def verify_plan(path):
    if sha(path) != PLAN_SHA:
        raise ValueError("Frozen fresh FP32 overfit PLAN changed")
    plan = read(path)
    for row in plan["records"]:
        source = Path(row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError(f"Frozen FP32 overfit input changed: {source}")
    for relative in plan["immutable_source_relative_paths"]:
        rows = [r for r in plan["records"] if r["path"].endswith("/" + relative)]
        if len(rows) != 1 or sha(REPO / relative) != rows[0]["sha256"]:
            raise ValueError(f"Production source changed: {relative}")
    return plan


def environment(plan):
    import gr00t
    from gr00t.model.gr00t_n1d7 import gr00t_n1d7
    from gr00t.model.modules import causal_visual_memory, fp32_visual_memory
    import mamba_ssm.modules.mamba2 as official
    import memory_training
    import memory_training_support
    import run_train_eval
    import triton.language.core as core

    modules = (
        gr00t,
        gr00t_n1d7,
        causal_visual_memory,
        fp32_visual_memory,
        proto,
        memory_training,
        memory_training_support,
        run_train_eval,
    )
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("Import escaped the fresh FP32 overfit worktree")
    if (
        sys.version_info[:2] != (3, 10)
        or Path(sys.prefix) != VENV
        or torch.__version__.split("+")[0] != "2.7.1"
    ):
        raise ValueError("Expected original Python3.10/torch2.7.1 environment")
    if (
        VENV not in Path(torch.__file__).resolve().parents
        or importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
    ):
        raise ValueError("Unexpected torch/Mamba installation")
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
        raise ValueError("Active kernel source differs from the frozen inputs")
    if official.causal_conv1d_fn is not None or os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise ValueError("Expected original overlay/IEEE default")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Pretrained weights may not be skipped")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(FILES):
        raise ValueError("Expected clean source with only the fresh-overfit files")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose only the authorized GPU1")
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
        "triton_f32_default": os.environ["TRITON_F32_DEFAULT"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
    }


def preflight(plan, model, memory, cache, processor, output, deadline):
    before = proto.content_hash(memory.state_dict())
    sequence, indices = sequence_for(cache, plan["preflight_anchor"], "cuda")
    frame = cache[plan["preflight_anchor"]]
    executor = FP32VisualMemory(memory)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, residual = executor.full(sequence)
        if torch.count_nonzero(residual):
            raise ValueError("Fresh FP32 readout does not exactly restore zero residual")
        native, native_receipt = native_flow(model, frame, None, 42)
        candidate, candidate_receipt = native_flow(model, frame, residual, 42)
        for key in ("loss", "action_loss", "action_mask", "state_features", "backbone_features"):
            if not torch.equal(native[key], candidate[key]):
                raise ValueError(f"Fresh native/zero-readout flow parity failed: {key}")
        if native_receipt != candidate_receipt:
            raise ValueError("Fresh zero-readout changed native conditioning/noise")
        reference = sample_cached(model, frame, None, 42)
        predicted = sample_cached(model, frame, residual, 42)
        proto.action_record(predicted, raw_record(frame), processor, reference)
    reports = {}
    shadow = proto.make_shadow(memory)
    for label, current in [("zero_initialization", memory), ("nonzero_shadow", shadow)]:
        current.zero_grad(set_to_none=True)
        evidence = objective_backward(
            model,
            FP32VisualMemory(current),
            sequence,
            frame,
            42,
            plan["feature_residual_l2_coefficient"],
        )
        groups = group_reports(current)
        required = list(groups) if label == "nonzero_shadow" else ["output_projection"]
        passed = all(groups[name]["finite_nonzero_parameters"] > 0 for name in required)
        reports[label] = {
            **evidence,
            "groups": groups,
            "required_groups": required,
            "passed": passed,
        }
        current.zero_grad(set_to_none=True)
        proto.assert_frozen(model)
        budget(plan, output, deadline)
    if proto.content_hash(memory.state_dict()) != before:
        raise ValueError("Preflight or shadow changed the fresh training initialization")
    result = {
        "status": "passed" if all(r["passed"] for r in reports.values()) else "failed",
        "optimizer_steps": 0,
        "anchor": plan["preflight_anchor"],
        "prefix": indices,
        "native_zero_flow_action_exact": True,
        "reports": reports,
        "initial_content_sha256_before": before,
        "initial_content_sha256_after": before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_new(output / "PREFLIGHT_RESULT.json", result)
    if result["status"] != "passed":
        raise ValueError("Fresh FP32 zero-step preflight gradient gate failed")
    return result


def evaluate(plan, model, memory, cache, processor, output, deadline):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from run_train_eval import METRICS, physical_metrics

    output.mkdir(parents=True, exist_ok=False)
    executor = FP32VisualMemory(memory)
    memory.eval()
    rows, predictions = [], []
    for anchor in plan["eval_anchors"]:
        budget(plan, output, deadline)
        sequence, indices = sequence_for(cache, anchor, "cuda")
        _, residual, state = step_prefix(memory, executor, sequence)
        del state
        frame = cache[anchor]
        for nominal in plan["nominal_seeds"]:
            seed = deterministic_seed("diagnostic", 125, anchor, nominal)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                result, receipt = native_flow(model, frame, residual, seed)
            predicted = sample_cached(model, frame, residual, seed)
            _, action_identity = proto.action_record(predicted, raw_record(frame), processor)
            prediction = predicted[0].float().numpy()
            metrics = physical_metrics(prediction, raw_record(frame), processor)
            decoded = processor.decode_action(
                prediction, EmbodimentTag.NEW_EMBODIMENT, state=raw_record(frame).states
            )
            hand = np.abs(decoded["left_hand"] - raw_record(frame).actions["left_hand"]).mean(
                axis=0
            )
            row = {
                "episode": 125,
                "anchor": anchor,
                "seed": nominal,
                "paired_rng_seed": seed,
                "prefix": indices,
                "flow_loss": float(result["loss"]),
                **metrics,
                **{f"hand_mae_u_{index}": float(value) for index, value in enumerate(hand)},
                "noise_receipt": receipt,
                **action_identity,
            }
            if not all(
                np.isfinite(row[key]) for key in (*METRICS, *[f"hand_mae_u_{i}" for i in range(6)])
            ):
                raise ValueError("Nonfinite FP32-step evaluation metric")
            rows.append(row)
            predictions.append(prediction)
    if len(rows) != 64:
        raise ValueError("Expected exactly16 anchors x4 seeds")
    per_episode, metrics = aggregate_metrics(
        rows, (*METRICS, *[f"hand_mae_u_{i}" for i in range(6)])
    )
    write_new(output / "details.json", rows)
    np.save(output / "normalized_predictions.npy", np.stack(predictions))
    report = {
        "status": "passed",
        "condition": "fresh_fp32_mamba_history",
        "deployment_path": "FP32VisualMemory.step",
        "rows": 64,
        "metrics": metrics,
        "per_episode": per_episode,
        "aggregation": "seed -> anchor -> episode equal weighting",
        "flow_objective": "original native head.forward loss including mask.sum()+1e-6",
        "prediction_sha256": sha(output / "normalized_predictions.npy"),
    }
    write_new(output / "EVALUATION_RESULT.json", report)
    return report


def terminal_checks(plan, model, memory, cache, processor, output, deadline):
    executor = FP32VisualMemory(memory)
    memory.eval()
    reports, disabled_reports = [], []
    for anchor in plan["numerical_anchors"]:
        budget(plan, output, deadline)
        sequence, indices = sequence_for(cache, anchor, "cuda")
        with torch.no_grad():
            full, full_residual = executor.full(sequence)
            step, step_residual, state = step_prefix(memory, executor, sequence)
        if not all(torch.isfinite(v).all() for v in (full, full_residual, step, step_residual)):
            raise ValueError("Nonfinite terminal FP32 memory tensor")
        latent = error_summary(step.cpu(), full.cpu())
        seeds = [
            (f"nominal_{s}", deterministic_seed("diagnostic", 125, anchor, s))
            for s in plan["nominal_seeds"]
        ]
        seeds.append(("direct_42", plan["additional_direct_seed"]))
        actions, tensors = (
            [],
            {
                "full_latent": full.cpu(),
                "step_latent": step.cpu(),
                "full_residual": full_residual.cpu(),
                "step_residual": step_residual.cpu(),
                "actions": {},
            },
        )
        for label, seed in seeds:
            actual = sample_cached(model, cache[anchor], step_residual, seed)
            reference = sample_cached(model, cache[anchor], full_residual, seed)
            proto.action_record(actual, raw_record(cache[anchor]), processor)
            proto.action_record(reference, raw_record(cache[anchor]), processor)
            difference = error_summary(actual[:, :16], reference[:, :16])
            actions.append(
                {
                    "seed_label": label,
                    "actual_seed": seed,
                    **difference,
                    "old_action_gate": old_gate(difference, plan["action_limits"]),
                }
            )
            tensors["actions"][label] = {"full": reference, "step": actual}
        torch.save(tensors, output / f"terminal-{anchor}.pt")
        reports.append(
            {
                "anchor": anchor,
                "prefix": indices,
                "latent": latent,
                "old_latent_gate": old_gate(latent, plan["latent_limits"]),
                "active_actions": actions,
                "step_state": state.describe(),
                "step_state_content_sha256": proto.state_hash(state),
            }
        )
        # A false numerical flag never prevents these remaining terminal checks.
        parity, disabled = base_parity(model, cache[anchor], seeds)
        disabled_reports.append({"anchor": anchor, **parity})
        torch.save(disabled, output / f"disabled-{anchor}.pt")
        del state
    write_new(output / "TERMINAL_NUMERICAL.json", reports)
    write_new(output / "DISABLED_PARITY.json", disabled_reports)
    return reports, disabled_reports


def train_200(plan, model, memory, cache, schedule, output, deadline):
    optimizer = optimizer_for(memory, plan)
    executor = FP32VisualMemory(memory)
    before = {name: value.detach().cpu().clone() for name, value in memory.state_dict().items()}
    if proto.content_hash(before) != INITIAL_SHA:
        raise ValueError("Training must begin at the fresh prototype, never overfit weights")
    seen, completed = {}, 0
    try:
        with (output / "training_steps.jsonl").open("x") as stream:
            for item in schedule:
                budget(plan, output, deadline)
                if item["step"] != completed + 1 or item["episode"] != 125 or completed >= 200:
                    raise ValueError("Frozen200-step schedule order changed")
                sequence, indices = sequence_for(cache, item["anchor"], "cuda")
                if indices != item["prefix"]:
                    raise ValueError("Training prefix differs from the frozen schedule")
                for group in optimizer.param_groups:
                    group["lr"] = item["lr"]
                memory.train()
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                evidence = objective_backward(
                    model,
                    executor,
                    sequence,
                    cache[item["anchor"]],
                    item["forward_seed"],
                    plan["feature_residual_l2_coefficient"],
                )
                gradients = group_reports(memory)
                for name, group in gradients.items():
                    seen[name] = seen.get(name, False) or group["finite_nonzero_parameters"] > 0
                norm = torch.nn.utils.clip_grad_norm_(
                    memory.parameters(), plan["optimizer"]["gradient_clip"], error_if_nonfinite=True
                )
                proto.assert_frozen(model)
                optimizer.step()
                completed += 1
                if any(not bool(torch.isfinite(p).all()) for p in memory.parameters()):
                    raise ValueError("Optimizer produced nonfinite memory parameters")
                row = {
                    **item,
                    **evidence,
                    "condition": "fresh_fp32_mamba_history",
                    "gradient_groups": gradients,
                    "gradient_norm_before_clip": float(norm),
                    "optimizer_steps_completed": completed,
                    "elapsed_step_seconds": time.monotonic() - tick,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                del sequence
                budget(plan, output, deadline)
                if completed % 25 == 0:
                    print(json.dumps({"fp32_overfit_steps": completed}), flush=True)
    except Exception:
        torch.save(
            {
                "module": {k: v.detach().cpu() for k, v in memory.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "completed_steps": completed,
                "plan_sha256": PLAN_SHA,
                "resume_authorized": False,
            },
            output / "INTERRUPTED_CHECKPOINT.pt",
        )
        raise
    optimizer.zero_grad(set_to_none=True)
    checkpoint = output / "checkpoint-step200.pt"
    torch.save(
        {
            "module": {k: v.detach().cpu() for k, v in memory.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "completed_steps": completed,
            "plan_sha256": PLAN_SHA,
            "condition": "fresh_fp32_mamba_history",
            "schedule_sha256": SCHEDULE_SHA,
            "initial_content_sha256": INITIAL_SHA,
            "optimizer_resume_used": False,
        },
        checkpoint,
    )
    parameters = {
        name: {
            "changed": not torch.equal(value.detach().cpu(), before[name]),
            "maxabs_change": float((value.detach().cpu() - before[name]).abs().max()),
        }
        for name, value in memory.state_dict().items()
    }
    try:
        groups = learned_change_witness(memory, before, seen)
        changes = {"passed": True, "groups": groups, "parameters": parameters}
    except ValueError as exc:
        changes = {
            "passed": False,
            "error": str(exc),
            "seen_gradient_groups": seen,
            "parameters": parameters,
        }
    write_new(output / "GRADIENT_PARAMETER_CHANGES.json", changes)
    return (
        completed,
        changes,
        {
            "path": str(checkpoint.resolve()),
            "sha256": sha(checkpoint),
            "module_content_sha256": proto.content_hash(memory.state_dict()),
        },
    )


def adoption(quality, numerical, changes):
    numerical_pass = all(
        row["old_latent_gate"] and all(v["old_action_gate"] for v in row["active_actions"])
        for row in numerical
    )
    return {
        "quality": all(quality.values()),
        "terminal_numerical": numerical_pass,
        "gradient_and_parameter_changes": changes["passed"],
    }


def run(args, deadline):
    from run_train_eval import processor_from, strict_reload

    plan = verify_plan(args.plan)
    source = environment(plan)
    if plan["initial_content_sha256"] != INITIAL_SHA or plan["optimizer_steps"] != 200:
        raise ValueError("Fresh initialization/200step contract changed")
    if sha(plan["schedule_file"]) != SCHEDULE_SHA or plan["schedule_sha256"] != SCHEDULE_SHA:
        raise ValueError("Frozen original overfit schedule changed")
    schedule = read(plan["schedule_file"])
    if len(schedule) != 200:
        raise ValueError("Expected exactly200 planned optimizer steps")
    for row in plan["evaluation_seeds"]:
        if row["actual_seed"] != deterministic_seed(
            "diagnostic", 125, row["anchor"], row["nominal_seed"]
        ):
            raise ValueError("Frozen evaluation RNG mapping changed")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    initial = checked_initialization(plan["prototype_initialization"])
    if sum(v.numel() for v in initial.values()) != plan["expected_parameter_count"]:
        raise ValueError("Unexpected fresh memory parameter count")
    cache = torch.load(plan["cache_file"], map_location="cpu", weights_only=True)
    if list(cache) != plan["grid"] or any(v["episode"] != 125 for v in cache.values()):
        raise ValueError("Only the frozen episode125 cache is allowed")
    cache_before = {anchor: frame_hash(frame) for anchor, frame in cache.items()}
    if any(frame["content_sha256"] != cache_before[anchor] for anchor, frame in cache.items()):
        raise ValueError("Frozen cache tensor identity changed")
    memory = CausalVisualMemory()
    memory.load_state_dict(initial, strict=True)
    memory.cuda().eval()
    if proto.content_hash(memory.state_dict()) != INITIAL_SHA:
        raise ValueError("Fresh initialization changed on device transfer")
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    proto.assert_frozen(model)
    base_before = proto.content_hash(model.state_dict())
    processor = processor_from(Path(plan["prepared"]) / "processor")
    processor.eval()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "environment": source,
            "strict_loading": loading,
            "base_before": base_before,
            "cache_before": cache_before,
            "fresh_initial_content_sha256": INITIAL_SHA,
            "prototype_initialization_sha256": sha(plan["prototype_initialization"]),
            "schedule_sha256": SCHEDULE_SHA,
            "warm_start": False,
            "optimizer_resume": False,
            "grad_scaler": False,
        },
    )
    budget(plan, args.output, deadline)
    initial_gate = preflight(plan, model, memory, cache, processor, args.output, deadline)
    before = evaluate(plan, model, memory, cache, processor, args.output / "step0", deadline)
    completed, changes, checkpoint = train_200(
        plan, model, memory, cache, schedule, args.output, deadline
    )
    after = evaluate(plan, model, memory, cache, processor, args.output / "step200", deadline)
    numerical, disabled = terminal_checks(
        plan, model, memory, cache, processor, args.output, deadline
    )
    proto.assert_frozen(model)
    base_after = proto.content_hash(model.state_dict())
    cache_after = {anchor: frame_hash(frame) for anchor, frame in cache.items()}
    if base_before != base_after or cache_before != cache_after:
        raise ValueError("Frozen base or episode125 cache changed")
    if any(p.grad is not None or p.dtype != torch.float32 for p in memory.parameters()):
        raise ValueError("Expected cleared gradients and FP32 memory master parameters")
    final_memory_hash = proto.content_hash(memory.state_dict())
    if final_memory_hash != checkpoint["module_content_sha256"]:
        raise ValueError("Terminal evaluation changed the step200 memory weights")
    integrity = {
        "step200_memory_content_sha256_before_evaluation": checkpoint["module_content_sha256"],
        "base_before": base_before,
        "base_after": base_after,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "final_memory_content_sha256": final_memory_hash,
        "base_frozen_eval_no_grad": True,
        "memory_gradients_cleared": True,
    }
    write_new(args.output / "FINAL_INTEGRITY.json", integrity)
    quality = overfit_gate(before, after)
    gates = adoption(quality, numerical, changes)
    report = {
        "status": "passed" if all(gates.values()) else "failed",
        "technical_validity": "passed",
        "measurements_completed": True,
        "continuation_gate": all(gates.values()),
        "adoption_gates": gates,
        "optimizer_steps": completed,
        "quality_gate": quality,
        "step0": before,
        "step200": after,
        "preflight": initial_gate,
        "terminal_numerical": numerical,
        "disabled_parity": disabled,
        "gradient_parameter_changes": changes,
        "checkpoint": checkpoint,
        "integrity": integrity,
        "prior_fixed_weight_cost_evidence": {
            "path": plan["prior_repair_result"],
            "sha256": sha(plan["prior_repair_result"]),
            "new_terminal_checkpoint_cost_measured": False,
        },
        "old_failures_unchanged": True,
        "pilot_started": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "artifact_bytes_before_result": budget(plan, args.output, deadline),
        "records": [
            {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        ],
        "scope": "Single seen training episode trainability only; no matched history controls, generalization, closed-loop or new-checkpoint cost claim",
    }
    return report


def execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        report = run(args, started + args.wall_seconds)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        report["elapsed_seconds"] = time.monotonic() - started
        report["gpu_after_tensor_release"] = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        write_new(args.output / "RESULT.json", report)
    except Exception as exc:
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "technical_validity": "not_established",
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
                "optimizer_resume_authorized": False,
                "old_failures_unchanged": True,
                "completed_step_evidence": [
                    str(p.resolve())
                    for name in (
                        "training_steps.jsonl",
                        "INTERRUPTED_CHECKPOINT.pt",
                        "checkpoint-step200.pt",
                    )
                    if (p := args.output / name).exists()
                ],
            },
        )
        raise
    if not report["continuation_gate"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=1750)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1750:
        parser.error("Wall budget must be in (0,1750] seconds")
    execute(args)


if __name__ == "__main__":
    main()
