"""One fixed-weight full/step x BF16/FP32 numerical diagnosis; no training."""

import argparse
import gc
import importlib
import importlib.metadata
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from contract import sha, tensor_identity, write_new
from gr00t.model.modules.causal_visual_memory import CausalVisualMemory, inject_image_residual
from memory_numerical_trace import (
    MemoryTrace,
    error_summary,
    interpretation,
    memory_precision,
    stage_comparisons,
)
import memory_prototype as prototype
from memory_training_support import (
    frame_hash,
    fresh,
    native_flow,
    prepare_residual,
    sampling_packet,
)
import numpy as np
from runner_core import deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-causal-mamba-numerical-diagnosis-r2"
)
EXPECTED_VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "7a8280f54013a013bf81ef04ebd186c57253c3fe170e7e2d3e58031368b95afb"
NEW_FILES = {
    "examples/G1_DIT_MAMBA_YUQI/memory_numerical_diagnosis.py",
    "examples/G1_DIT_MAMBA_YUQI/memory_numerical_trace.py",
    "examples/G1_DIT_MAMBA_YUQI/README-memory-numerical-diagnosis.md",
    "tests/examples/test_memory_numerical_diagnosis.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def budget(plan, output, deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Numerical diagnosis exceeded its internal wall budget")
    size = sum(p.stat().st_size for p in output.parent.parent.rglob("*") if p.is_file())
    if size > plan["max_artifact_bytes"]:
        raise RuntimeError("Numerical diagnosis exceeded 1 GiB artifact budget")
    if torch.cuda.max_memory_allocated() > plan["max_allocated_bytes"]:
        raise RuntimeError("Numerical diagnosis exceeded 12 GiB allocated GPU memory")
    return size


def frozen_inputs(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen numerical diagnosis PLAN changed")
    plan = read(args.plan)
    if (
        plan["episode"] != 125
        or plan["prefix"] != list(range(0, 321, 16))
        or plan["optimizer_steps"] != 0
    ):
        raise ValueError("Only the frozen episode125 prefix and zero updates are allowed")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen diagnosis input changed: {path}")
    for relative in plan["immutable_source_relative_paths"]:
        rows = [r for r in plan["records"] if r["path"].endswith("/" + relative)]
        if len(rows) != 1 or sha(REPO / relative) != rows[0]["sha256"]:
            raise ValueError(f"Active production source differs from frozen source: {relative}")
    return plan


def source_environment(plan):
    import gr00t
    from gr00t.model.gr00t_n1d7 import gr00t_n1d7
    from gr00t.model.modules import causal_visual_memory
    import mamba_ssm.modules.mamba2 as official
    import memory_training_support
    import run_train_eval
    import triton.language.core as triton_core

    modules = (
        gr00t,
        gr00t_n1d7,
        causal_visual_memory,
        prototype,
        memory_training_support,
        run_train_eval,
    )
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("A production source import escaped the diagnosis worktree")
    if sys.version_info[:2] != (3, 10) or Path(sys.prefix) != EXPECTED_VENV:
        raise ValueError("Expected original Python3.10 environment")
    if (
        torch.__version__.split("+")[0] != "2.7.1"
        or EXPECTED_VENV not in Path(torch.__file__).resolve().parents
    ):
        raise ValueError("Expected original torch2.7.1 environment")
    if (
        importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
        or official.causal_conv1d_fn is not None
    ):
        raise ValueError("Expected frozen Mamba2 overlay without causal-conv1d")
    if os.environ.get("TRITON_F32_DEFAULT") != plan["triton_f32_default"]:
        raise ValueError("TRITON_F32_DEFAULT differs from the frozen environment")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Skipping pretrained weights is forbidden")
    overlay_modules = [official, triton_core]
    for name in (
        "ssd_combined",
        "ssd_chunk_scan",
        "ssd_chunk_state",
        "selective_state_update",
        "layernorm_gated",
    ):
        overlay_modules.append(importlib.import_module("mamba_ssm.ops.triton." + name))
    recorded = {Path(row["path"]).resolve(): row["sha256"] for row in plan["records"]}
    if any(
        Path(m.__file__).resolve() not in recorded
        or sha(m.__file__) != recorded[Path(m.__file__).resolve()]
        for m in overlay_modules
    ):
        raise ValueError("Active official kernel/Triton source differs from frozen identity")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(NEW_FILES):
        raise ValueError("Diagnosis requires clean source and only the authorized new files")
    return official, {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "dirty": dirty,
        "changed_files": sorted(changed),
        "imports": {m.__name__: m.__file__ for m in (*modules, *overlay_modules)},
        "python": sys.version,
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "mamba": importlib.metadata.version("mamba-ssm"),
        "triton_f32_default": os.environ["TRITON_F32_DEFAULT"],
    }


def reproduction(actual, plan):
    expected = plan["expected_bf16_latent"]
    if not all(
        math.isfinite(values[key])
        for values in (actual, expected)
        for key in ("maxabs", "rms", "relative_rms")
    ):
        raise ValueError("Nonfinite historical statistic input")
    passed = actual["maxabs"] == expected["maxabs"] and all(
        abs(actual[k] - expected[k]) <= plan["statistics_reproduction_atol"]
        for k in ("rms", "relative_rms")
    )
    return {
        "passed": passed,
        "role": "historical_compatibility_only",
        "reduction": "gpu_float32",
        "difference": {key: actual[key] - expected[key] for key in expected},
        "expected": expected,
        "actual": actual,
        "statistics_atol": plan["statistics_reproduction_atol"],
    }


def old_gate(stats, limits):
    return all(stats[key] <= maximum for key, maximum in limits.items())


def check_frozen(module):
    if any(m.training for m in module.modules()) or any(
        p.requires_grad or p.grad is not None for p in module.parameters()
    ):
        raise ValueError("Every base/memory parameter must stay frozen/eval without gradients")


def save_tensors(path, value, plan, output, deadline):
    torch.save(value, path)
    budget(plan, output, deadline)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def run_memory(memory, sequence, mode):
    precision, path = mode.split("_")
    with torch.no_grad(), memory_precision(precision):
        latent, residual, length = prepare_residual(
            memory, sequence, "mamba_history", deployment=path == "step"
        )
    if (
        length != 21
        or not bool(torch.isfinite(latent).all())
        or not bool(torch.isfinite(residual).all())
    ):
        raise ValueError("Expected finite production output for all21 observations")
    if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise ValueError("Memory precision context did not restore original base TF32 flags")
    return {"latent": latent, "residual": residual}


def check_action(value):
    if tuple(value.shape) != (1, 40, 13) or not bool(torch.isfinite(value).all()):
        raise ValueError("Expected finite native (1,40,13) normalized actions")
    return value.detach().cpu()


def base_parity(model, frame, seeds):
    records = []
    predictions = {}
    observation = {
        k: v for k, v in frame["action_input"].items() if k not in ("action", "action_mask")
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, seed in seeds:
            set_seed(seed, cuda=True)
            native = model.action_head.get_action(
                fresh(frame["raw"], model.device), fresh(observation, model.device), options=None
            )["action_pred"]
            packet = sampling_packet(model, frame)
            if not torch.equal(
                packet["features"].backbone_features.cpu(), frame["post_vl_features"]
            ):
                raise ValueError("Native raw-cache postprocess no longer matches frozen post-Z")
            set_seed(seed, cuda=True)
            cached = prototype.sample_packet(model, packet)
            native, cached = check_action(native), check_action(cached)
            if not torch.equal(native, cached):
                raise ValueError("Disabled native-head/raw-cache sampling parity failed")
            predictions[name] = native
            records.append(
                {
                    "seed_label": name,
                    "actual_seed": seed,
                    "exact": True,
                    "action_identity": tensor_identity(native),
                }
            )
    return {
        "status": "passed",
        "scope": "Native head.get_action at the frozen raw-VLM cache boundary; no new RGB/VLM forward",
        "records": records,
    }, predictions


def historical_action_compatibility(predicted, flow, seed, row, old_prediction):
    reference = torch.as_tensor(old_prediction).unsqueeze(0)
    reference = check_action(reference)
    if not math.isfinite(flow) or not math.isfinite(row["flow_loss"]):
        raise ValueError("Nonfinite historical flow input")
    seed_equal = row["paired_rng_seed"] == seed
    action_exact = torch.equal(predicted, reference)
    flow_exact = flow == row["flow_loss"]
    return {
        "role": "historical_compatibility_only",
        "seed_equal": seed_equal,
        "action_exact": action_exact,
        "flow_exact": flow_exact,
        "exact": seed_equal and action_exact and flow_exact,
        "flow_difference": flow - row["flow_loss"],
        "action_difference": error_summary(predicted, reference),
    }


def mode_actions(model, frame, residual, mode, seeds, old_rows, old_predictions):
    records = []
    tensors = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for label, seed in seeds:
            result, receipt = native_flow(model, frame, residual, seed)
            packet = sampling_packet(model, frame)
            set_seed(seed, cuda=True)
            predicted = check_action(prototype.sample_packet(model, packet, residual))
            flow = float(result["loss"])
            old_exact = None
            compatibility = None
            if mode == "bf16_step" and label.startswith("nominal_"):
                nominal = int(label.split("_")[1])
                index, row = old_rows[nominal]
                compatibility = historical_action_compatibility(
                    predicted, flow, seed, row, old_predictions[index]
                )
                old_exact = compatibility["exact"]
            tensors[label] = predicted
            records.append(
                {
                    "mode": mode,
                    "seed_label": label,
                    "actual_seed": seed,
                    "flow_loss": flow,
                    "noise_receipt": receipt,
                    "action_identity": tensor_identity(predicted),
                    "historical_exact": old_exact,
                    "historical_compatibility": compatibility,
                }
            )
    return records, tensors


def run(args, deadline):
    from run_train_eval import strict_reload

    plan = frozen_inputs(args)
    official, environment = source_environment(plan)
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose only the authorized GPU1")
    environment["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    environment["logical_cuda_device"] = 0
    environment["gpu_name"] = torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    budget(plan, args.output, deadline)
    cache = torch.load(plan["cache_file"], map_location="cpu", weights_only=True)
    if list(cache) != plan["prefix"]:
        raise ValueError("Only episode125 anchors0..320 may be consumed")
    before_cache = {anchor: frame_hash(frame) for anchor, frame in cache.items()}
    if any(
        frame["content_sha256"] != before_cache[anchor] or frame["episode"] != 125
        for anchor, frame in cache.items()
    ):
        raise ValueError("Frozen cache content/episode identity changed")
    sequence = torch.stack(
        [cache[a]["post_vl_features"][0, prototype.IMAGE_POSITIONS] for a in plan["prefix"]]
    ).cuda()
    if not bool(torch.isfinite(sequence).all()) or sequence.dtype != torch.float32:
        raise ValueError("Expected finite native FP32 cached post-VL input")
    frame = cache[320]
    checkpoint = torch.load(plan["memory_checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["completed_steps"] != 200 or checkpoint["condition"] != "mamba_history":
        raise ValueError("Expected fixed completed200-step Mamba history checkpoint")
    memory = CausalVisualMemory()
    memory.load_state_dict(checkpoint["module"], strict=True)
    memory = memory.cuda().eval().requires_grad_(False)
    if any(p.dtype != torch.float32 for p in memory.parameters()):
        raise ValueError("Memory master weights must remain FP32")
    memory_before = prototype.content_hash(memory.state_dict())
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    base_before = prototype.content_hash(model.state_dict())
    if base_before != checkpoint["base_content_sha256"]:
        raise ValueError("Frozen base content differs from the training-start identity")
    del checkpoint
    check_frozen(model)
    check_frozen(memory)
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "environment": environment,
            "strict_loading": loading,
            "base_before": base_before,
            "memory_before": memory_before,
            "cache_before": before_cache,
            "precision_scope": "Only memory changes precision/TF32; base always BF16 autocast and original TF32=true",
            "optimizer_steps": 0,
            "historical_training_status": "failed; unchanged",
        },
    )
    input_file = save_tensors(
        args.output / "INPUT_SEQUENCE.pt",
        {"post_image_slots": sequence.cpu(), "anchors": plan["prefix"]},
        plan,
        args.output,
        deadline,
    )
    gpu_modes = {}
    saved_modes = {}
    artifacts = {"input_sequence": input_file}
    for mode in plan["modes"]:
        budget(plan, args.output, deadline)
        gpu_modes[mode] = run_memory(memory, sequence, mode)
        saved_modes[mode] = {k: v.detach().cpu().clone() for k, v in gpu_modes[mode].items()}
        artifacts[mode] = save_tensors(
            args.output / f"{mode}-untraced.pt", saved_modes[mode], plan, args.output, deadline
        )
        if mode == "bf16_step":
            stats = prototype.differences(
                gpu_modes["bf16_step"]["latent"], gpu_modes["bf16_full"]["latent"]
            )
            reproduced = reproduction(stats, plan)
            write_new(args.output / "BF16_HISTORICAL_COMPATIBILITY.json", reproduced)
        print(json.dumps({"untraced_complete": mode}), flush=True)
    traces = {}
    for mode in plan["modes"]:
        budget(plan, args.output, deadline)
        path = mode.split("_")[1]
        with MemoryTrace(memory, official, path, len(plan["prefix"])) as observer:
            traced = run_memory(memory, sequence, mode)
        if any(not torch.equal(traced[k], gpu_modes[mode][k]) for k in ("latent", "residual")):
            raise ValueError(f"Trace disturbed production memory outputs: {mode}")
        traces[mode] = observer.export()
        artifacts[mode + "_trace"] = save_tensors(
            args.output / f"{mode}-trace.pt", traces[mode], plan, args.output, deadline
        )
        write_new(
            args.output / f"{mode}-TRACE.json",
            {
                "status": "passed",
                "exact_untraced": True,
                "state": traces[mode]["state"],
                "stage_metadata": traces[mode]["metadata"],
                "tensor_artifact": artifacts[mode + "_trace"],
            },
        )
        del traced, observer
    for precision in ("bf16", "fp32"):
        comparison = stage_comparisons(
            traces[precision + "_step"], traces[precision + "_full"], plan["latent_limits"]
        )
        write_new(
            args.output / f"{precision}-STAGE_COMPARISONS.json",
            {
                "reference": precision + "_full",
                "actual": precision + "_step",
                "rows": comparison,
                "threshold_scope": "Intermediate threshold crossings are localization diagnostics, not new model success gates",
            },
        )
    del traces
    seeds = [
        (f"nominal_{s}", deterministic_seed("diagnostic", 125, 320, s))
        for s in plan["nominal_seeds"]
    ] + [("direct_42", plan["additional_direct_seed"])]
    old_details = read(plan["historical_details"])
    old_rows = {
        row["seed"]: (i, row)
        for i, row in enumerate(old_details)
        if row["episode"] == 125 and row["anchor"] == 320
    }
    if set(old_rows) != set(plan["nominal_seeds"]):
        raise ValueError("Missing fixed historical anchor320 seed outputs")
    old_predictions = np.load(plan["historical_predictions"])
    if (
        old_predictions.shape != (len(old_details), 40, 13)
        or not np.isfinite(old_predictions).all()
    ):
        raise ValueError("Malformed or nonfinite historical action tensor")
    parity, disabled_actions = base_parity(model, frame, seeds)
    write_new(args.output / "BASE_DISABLED_PARITY.json", parity)
    artifacts["disabled_actions"] = save_tensors(
        args.output / "disabled-actions.pt", disabled_actions, plan, args.output, deadline
    )
    all_records = []
    all_actions = {}
    conditions = {}
    for mode in plan["modes"]:
        budget(plan, args.output, deadline)
        residual = gpu_modes[mode]["residual"]
        condition = inject_image_residual(
            frame["post_vl_features"].cuda(),
            torch.tensor(prototype.IMAGE_POSITIONS, device="cuda"),
            residual,
        )
        conditions[mode] = condition.cpu()
        records, actions = mode_actions(
            model, frame, residual, mode, seeds, old_rows, old_predictions
        )
        if any(
            row["noise_receipt"]["effective_condition"] != tensor_identity(condition)
            for row in records
        ):
            raise ValueError("Native flow effective condition differs from saved Z+residual")
        all_records.extend(records)
        all_actions[mode] = actions
        artifacts[mode + "_actions"] = save_tensors(
            args.output / f"{mode}-actions-condition.pt",
            {
                "actions": actions,
                "condition": conditions[mode],
                "residual": saved_modes[mode]["residual"],
            },
            plan,
            args.output,
            deadline,
        )
    for label, seed in seeds:
        rows = [r for r in all_records if r["seed_label"] == label]
        for key in ("continuous_time", "discrete_time", "noisy_action", "preinjection_post"):
            if len({json.dumps(r["noise_receipt"][key], sort_keys=True) for r in rows}) != 1:
                raise ValueError(
                    f"Paired native inputs/noise differ between memory modes: {label} {key}"
                )
    write_new(args.output / "ACTION_FLOW_DETAILS.json", all_records)
    cross = []
    for reference, actual in combinations(plan["modes"], 2):
        lat = error_summary(saved_modes[actual]["latent"], saved_modes[reference]["latent"])
        actions = []
        for label, seed in seeds:
            error = error_summary(
                all_actions[actual][label][:, :16], all_actions[reference][label][:, :16]
            )
            actions.append(
                {
                    "seed_label": label,
                    "actual_seed": seed,
                    **error,
                    "old_action_gate": old_gate(error, plan["action_limits"]),
                }
            )
        cross.append(
            {
                "reference": reference,
                "actual": actual,
                "latent": lat,
                "latent_old_gate": old_gate(lat, plan["latent_limits"]),
                "residual": error_summary(
                    saved_modes[actual]["residual"][None], saved_modes[reference]["residual"][None]
                ),
                "condition": error_summary(conditions[actual], conditions[reference]),
                "active_actions": actions,
            }
        )
    write_new(args.output / "CROSS_MODE_COMPARISONS.json", cross)
    gpu_fp32_report = {
        "reduction": "gpu_float32",
        "role": "report_only",
        "bf16": prototype.differences(
            gpu_modes["bf16_step"]["latent"], gpu_modes["bf16_full"]["latent"]
        ),
        "fp32": prototype.differences(
            gpu_modes["fp32_step"]["latent"], gpu_modes["fp32_full"]["latent"]
        ),
    }
    write_new(args.output / "GPU_FP32_LATENT_REPORT.json", gpu_fp32_report)
    bf16 = error_summary(saved_modes["bf16_step"]["latent"], saved_modes["bf16_full"]["latent"])
    fp32 = error_summary(saved_modes["fp32_step"]["latent"], saved_modes["fp32_full"]["latent"])
    conclusion = interpretation(
        bf16, fp32, plan["latent_limits"], plan["precision_convergence_ratio"]
    )
    check_frozen(model)
    check_frozen(memory)
    base_after = prototype.content_hash(model.state_dict())
    memory_after = prototype.content_hash(memory.state_dict())
    after_cache = {anchor: frame_hash(value) for anchor, value in cache.items()}
    if base_before != base_after or memory_before != memory_after or before_cache != after_cache:
        raise ValueError("Fixed base/memory/cache content changed during diagnosis")
    if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise ValueError("Original base precision settings were not restored")
    integrity = {
        "status": "passed",
        "base_before": base_before,
        "base_after": base_after,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "cache_before": before_cache,
        "cache_after": after_cache,
        "frozen_eval_no_grad": True,
    }
    write_new(args.output / "FINAL_INTEGRITY.json", integrity)
    return {
        "status": "passed",
        "technical_validity": "passed",
        "optimizer_steps": 0,
        "old_training_failure_unchanged": True,
        "historical_bf16_step_outputs_exact": sum(
            row["historical_exact"] is True for row in all_records
        ),
        "historical_bf16_step_outputs_compared": sum(
            row["historical_exact"] is not None for row in all_records
        ),
        "historical_bf16_latent_compatibility": reproduced,
        "gpu_fp32_latent_report": gpu_fp32_report,
        "primary_reduction": "cpu_float64",
        "trace_untraced_exact_modes": plan["modes"],
        "base_disabled_parity": parity,
        "bf16_latent": bf16,
        "fp32_latent": fp32,
        "interpretation": conclusion,
        "cross_mode_comparisons": cross,
        "integrity": integrity,
        "artifacts": artifacts,
        "artifact_bytes_before_result": budget(plan, args.output, deadline),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "claim_scope": "One fixed learned checkpoint and one21-observation prefix; no training, repair, deployment permission or memory-quality claim",
    }


def execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        report = run(args, started + args.wall_seconds)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        report["gpu_after_tensor_release"] = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        report["elapsed_seconds"] = time.monotonic() - started
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
                "optimizer_steps": 0,
                "old_training_failure_unchanged": True,
                "automatic_retry": False,
            },
        )
        raise


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
