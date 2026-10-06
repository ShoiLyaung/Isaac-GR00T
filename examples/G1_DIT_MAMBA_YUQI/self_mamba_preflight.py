"""No-training native reuse parity and actual self-Mamba token/cost preflight."""

import argparse
from collections import defaultdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from contract import sha, tensor_identity, write_new
import numpy as np
from preflight import full_model
from run_train_eval import (
    METRICS,
    NativeAnchors,
    evaluate_pairs,
    load_plan,
    physical_metrics,
    source_identity,
    strict_reload,
)
from runner_core import deterministic_seed, set_seed
import torch


def exact_historical_parity(current, historical):
    rows = json.loads((current / "details.json").read_text())
    old = json.loads((historical / "details.json").read_text())
    index = {(r["episode"], r["anchor"], r["seed"]): i for i, r in enumerate(old)}
    predictions = np.load(current / "normalized_predictions.npy")
    expected = np.load(historical / "normalized_predictions.npy", mmap_mode="r")
    max_delta = 0.0
    for i, row in enumerate(rows):
        j = index[(row["episode"], row["anchor"], row["seed"])]
        if not np.array_equal(predictions[i], expected[j]):
            raise ValueError(f"Historical action parity failed: row {i}")
        for metric in METRICS:
            delta = abs(row[metric] - old[j][metric])
            max_delta = max(max_delta, delta)
            if delta != 0.0:
                raise ValueError(f"Historical metric parity failed: row {i} {metric} {delta}")
    if len(rows) != 64:
        raise ValueError("Expected exactly 64 historical outputs per native checkpoint")
    return {
        "status": "passed",
        "outputs": 64,
        "action_max_abs": 0.0,
        "metric_max_abs": max_delta,
        "historical_details_sha256": sha(historical / "details.json"),
        "historical_predictions_sha256": sha(historical / "normalized_predictions.npy"),
    }


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Self-Mamba preflight exceeded its wall budget")


def profile(model, source, pair, condition, output, deadline):
    model.eval()
    source.processor.eval()
    episode, anchor = map(int, pair)
    batch, raw = source.batch([(episode, anchor)])
    observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
    seed = deterministic_seed("diagnostic", episode, anchor, 42)

    def call():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.get_action(dict(observation))["action_pred"]

    timings = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(55):
        check_deadline(deadline)
        set_seed(seed, cuda=True)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        predicted = call()
        torch.cuda.synchronize()
        if i >= 5:
            timings.append((time.perf_counter() - tick) * 1000)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if tuple(predicted.shape) != (1, 40, 13) or not bool(torch.isfinite(predicted).all()):
        raise ValueError("Expected finite (1,40,13) actions")
    metrics = physical_metrics(predicted[0].float().cpu().numpy(), raw[0], source.processor)
    counts = defaultdict(int)
    shapes = defaultdict(list)
    handles = []

    def hook(group):
        def record(module, args):
            counts[group] += 1
            shape = list(args[0].shape)
            shapes[group].append(shape)
            if group == "ffn":
                counts["ffn_rows"] += int(np.prod(shape[:-1]))

        return record

    try:
        for i, block in enumerate(model.action_head.model.transformer_blocks):
            handles.append(block.ff.register_forward_pre_hook(hook("ffn")))
            if i % 2 == 0:
                handles.append(block.attn1.register_forward_pre_hook(hook("cross")))
            elif condition == "dit_self_mamba2":
                handles.append(block.attn1.mamba2.register_forward_pre_hook(hook("mamba")))
            else:
                handles.append(block.attn1.register_forward_pre_hook(hook("self_attention")))
        set_seed(seed, cuda=True)
        instrumented = call()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if not torch.equal(predicted, instrumented):
        raise ValueError("Counting hooks changed output")
    if counts["cross"] != 64 or counts["ffn"] != 128 or counts["ffn_rows"] != 5248:
        raise ValueError(f"Native FFN/cross contract drift: {dict(counts)}")
    expected_mamba = 128 if condition == "dit_self_mamba2" else 0
    if counts["mamba"] != expected_mamba or any(s[1] != 41 for s in shapes["mamba"]):
        raise ValueError("Mamba must execute 128 full forwards on 41 tokens per policy")
    result = {
        "status": "passed",
        "condition": condition,
        "episode": episode,
        "anchor": anchor,
        "paired_rng_seed": seed,
        "warmup": 5,
        "measured": 50,
        "policy_service_ms": {
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        },
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "counts": dict(counts),
        "input_shapes": dict(shapes),
        "prediction": tensor_identity(predicted),
        "physical_shape": [16, 13],
        "metrics": metrics,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "action_backbone_parameter_count": sum(
            p.numel() for p in model.action_head.model.parameters()
        ),
        "boundary": "host wall full native get_action including H2D/VLM/4-step sampling; excludes data decode/processor/RPC",
        "cost_gate": "descriptive only; no latency threshold",
    }
    output.mkdir()
    write_new(output / "PROFILE.json", result)
    write_new(output / "timings_ms.json", timings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-trained-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1800:
        parser.error("Wall budget must be <=1800 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.wall_seconds
    try:
        if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
            raise ValueError("Preflight forbids skipped checkpoint weights")
        import gr00t
        import mamba_ssm.modules.mamba2 as official

        root = Path(__file__).resolve().parents[2]
        if root not in Path(gr00t.__file__).resolve().parents:
            raise ValueError("GR00T import is outside this experiment worktree")
        if official.causal_conv1d_fn is not None:
            raise ValueError("Use the original overlay without causal-conv1d")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        plan, prepared, _, anchors = load_plan(args.plan)
        source = NativeAnchors(prepared)
        source.processor.eval()
        identity = {
            "source": source_identity(),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "gr00t_file": gr00t.__file__,
            "mamba_file": official.__file__,
            "dependencies": {
                key: importlib.metadata.version(key) for key in ("torch", "mamba-ssm", "triton")
            },
            "schedule_sha256": plan["schedule_sha256"],
            "gpu": torch.cuda.get_device_name(0),
            "base_checkpoint": str(args.checkpoint.resolve()),
            "native_trained_checkpoint": str(args.native_trained_checkpoint.resolve()),
        }
        write_new(args.output / "IDENTITY.json", identity)
        parities, profiles, audits = {}, {}, {}
        for label, checkpoint in (
            ("frozen_reference", args.checkpoint),
            ("dit_pretrained", args.native_trained_checkpoint),
        ):
            check_deadline(deadline)
            model, loading = strict_reload(checkpoint)
            model = model.to(device="cuda", dtype=torch.bfloat16).eval()
            current = args.output / f"parity-{label}"
            evaluate_pairs(
                model,
                source,
                anchors["seen_probe"],
                current,
                wall_seconds=max(1, int(deadline - time.monotonic())),
            )
            historical = args.historical_root / f"evaluate-{label}-r1" / "evaluate"
            parities[label] = exact_historical_parity(current, historical)
            parities[label]["strict_loading"] = loading
            if label == "frozen_reference":
                profiles[label] = profile(
                    model,
                    source,
                    anchors["seen_probe"][0],
                    label,
                    args.output / f"profile-{label}",
                    deadline,
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_new(args.output / "NATIVE_PARITY.json", {"status": "passed", "conditions": parities})
        for condition in ("dit_reinit_self_attention", "dit_self_mamba2"):
            check_deadline(deadline)
            model, audit = full_model(
                SimpleNamespace(checkpoint=args.checkpoint, condition=condition)
            )
            audits[condition] = audit
            write_new(args.output / f"AUDIT-{condition}.json", audit)
            model = model.to(device="cuda", dtype=torch.bfloat16).eval()
            profiles[condition] = profile(
                model,
                source,
                anchors["seen_probe"][0],
                condition,
                args.output / f"profile-{condition}",
                deadline,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        # Both new conditions start from exactly the same retained checkpoint tensors.
        hashes = {v["retained_content_sha256_before"] for v in audits.values()}
        if len(hashes) != 1:
            raise ValueError("New conditions do not share identical retained initialization")
        check_deadline(deadline)
        write_new(
            args.output / "RESULT.json",
            {
                "status": "passed",
                "native_parity": parities,
                "profiles": profiles,
                "initialization_audits": audits,
                "optimizer_steps": 0,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    except Exception as exc:
        write_new(
            args.output / "FAILURE.json",
            {
                "status": "failed",
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
