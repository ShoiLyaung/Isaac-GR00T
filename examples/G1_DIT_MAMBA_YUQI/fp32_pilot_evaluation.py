"""Joint fixed-checkpoint evaluation and current-RGB policy costs for FP32 pilot."""

import gc
import json
import time

from contract import sha, tensor_identity, write_new
from fp32_memory_overfit import sample_cached
from fp32_memory_validation import call_audit, policy_action, prefill_snapshot
from fp32_pilot_support import (
    ALL_METRICS,
    CONDITIONS,
    NOISE_KEYS,
    make_memory,
    memory_outputs,
    numerical_probe,
    quality_gates,
)
from gr00t.model.modules.fp32_visual_memory import FP32VisualMemory
import memory_prototype as proto
from memory_training import raw_record
from memory_training_support import native_flow
import numpy as np
from runner_core import aggregate_metrics, deterministic_seed, set_seed
import torch


def load_learned(context, condition, identity):
    if sha(identity["path"]) != identity["sha256"]:
        raise ValueError("Frozen checkpoint changed after joint inventory")
    values = torch.load(identity["path"], map_location="cpu", weights_only=True)
    if (
        values["condition"] != condition
        or values["completed_steps"] != 500
        or values["plan_sha256"] != context.plan_sha
        or values["base_content_sha256"] != context.base_before
    ):
        raise ValueError("Wrong final checkpoint condition/step/plan")
    if proto.content_hash(values["module"]) != identity["module_content_sha256"]:
        raise ValueError("Learned module content identity mismatch")
    module = make_memory(context.initial, condition)
    module.load_state_dict(values["module"], strict=True)
    return module.cuda().eval()


def evaluate_condition(context, cache, module, condition, episodes):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from run_train_eval import physical_metrics

    output = context.args.output / condition
    output.mkdir(exist_ok=False)
    rows, predictions = [], []
    for episode in episodes:
        for anchor in context.plan["eval_anchors"][str(episode)]:
            context.budget()
            frame = cache.frame(episode, anchor)
            if module is None:
                residual, indices, length = None, [], 0
            else:
                sequence, indices = cache.prefix(episode, anchor)
                _, residual, length, state = memory_outputs(module, sequence, condition, True)
                del sequence, state
            for nominal in context.plan["noise_seeds"]:
                seed = deterministic_seed("diagnostic", episode, anchor, nominal)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result, receipt = native_flow(context.model, frame, residual, seed)
                predicted = sample_cached(context.model, frame, residual, seed)
                _, identity = proto.action_record(
                    predicted, raw_record(frame), context.source.processor
                )
                prediction = predicted[0].float().numpy()
                metrics = physical_metrics(prediction, raw_record(frame), context.source.processor)
                decoded = context.source.processor.decode_action(
                    prediction, EmbodimentTag.NEW_EMBODIMENT, state=raw_record(frame).states
                )
                hand = np.abs(decoded["left_hand"] - raw_record(frame).actions["left_hand"]).mean(
                    axis=0
                )
                row = {
                    "condition": condition,
                    "episode": episode,
                    "anchor": anchor,
                    "seed": nominal,
                    "paired_rng_seed": seed,
                    "prefix": indices,
                    "effective_length": length,
                    "flow_loss": float(result["loss"]),
                    **metrics,
                    **{f"hand_mae_u_{i}": float(v) for i, v in enumerate(hand)},
                    "noise_receipt": receipt,
                    **identity,
                }
                if not all(np.isfinite(row[key]) for key in ALL_METRICS):
                    raise ValueError("Nonfinite joint evaluation metric")
                rows.append(row)
                predictions.append(prediction)
            del residual
    if len(rows) != 256 or len({(r["episode"], r["anchor"], r["seed"]) for r in rows}) != 256:
        raise ValueError("Expected four episodes x16 unique anchors x4 seeds")
    per_episode, metrics = aggregate_metrics(rows, ALL_METRICS)
    write_new(output / "details.json", rows)
    np.save(output / "normalized_predictions.npy", np.stack(predictions))
    report = {
        "condition": condition,
        "status": "passed",
        "rows": 256,
        "metrics": metrics,
        "per_episode": per_episode,
        "aggregation": "seed -> anchor -> episode equal weight",
        "prediction_sha256": sha(output / "normalized_predictions.npy"),
        "deployment_path": "native" if module is None else "FP32VisualMemory.step",
        "scope": "Pretrained-domain adapter-held-out engineering evidence only",
    }
    write_new(output / "EVALUATION_RESULT.json", report)
    return report, rows


def check_evaluation_pairing(all_rows):
    reference = all_rows["native"]
    for condition, rows in all_rows.items():
        if len(rows) != len(reference):
            raise ValueError("Joint condition row count changed")
        for a, b in zip(rows, reference):
            if any(a[key] != b[key] for key in ("episode", "anchor", "seed", "paired_rng_seed")):
                raise ValueError("Joint anchor/noise pairing changed")
            if any(a["noise_receipt"][key] != b["noise_receipt"][key] for key in NOISE_KEYS):
                raise ValueError("Joint actual native noise/time/state conditioning mismatch")
    return {
        "passed": True,
        "rows_per_condition": 256,
        "conditions": list(all_rows),
        "noise_receipt_keys": list(NOISE_KEYS),
    }


def cost_probes(context, cache, module, episodes):
    """Only native versus carry; decode, seed and state restore outside host timer."""
    output = context.args.output / "cost"
    output.mkdir(exist_ok=False)
    module.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    executor = FP32VisualMemory(module)
    reports = []
    for index, episode in enumerate(episodes):
        anchor = context.plan["grids"][str(episode)][-1]
        indices = context.plan["grids"][str(episode)]
        batch, raw = context.source.batch([(episode, anchor)])
        observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
        proto.validate_input_layout(observation, context.source.modality["video"].modality_keys)
        frame = cache.frame(episode, anchor)
        for key in (
            "state",
            "action",
            "action_mask",
            "embodiment_id",
            "input_ids",
            "attention_mask",
            "image_grid_thw",
        ):
            if tensor_identity(batch[key]) != frame["input_identity"][key]:
                raise ValueError("Cost current input/target identity mismatch")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            packet = proto.encode_observation(context.model, observation)
        if not torch.equal(packet["features"].backbone_features.cpu(), frame["post_vl_features"]):
            raise ValueError("Cost current RGB features differ from the frozen cache")
        del packet
        order = ["native", "fp32_step"] if index % 2 == 0 else ["fp32_step", "native"]
        records, prefill, audit = [], None, None
        for mode in order:
            module.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            prefix = state = None
            if mode == "fp32_step":
                module.cuda()
                snapshot, prefill = prefill_snapshot(
                    module, executor, cache.episode(episode), indices[:-1], mode
                )
                prefix = snapshot.to("cuda")
                prefix_hash = proto.state_hash(prefix)
                audit = call_audit(context.model, module, executor, observation, prefix, mode)
            elif any(p.is_cuda for p in module.parameters()):
                raise ValueError("Native timer retains unused GPU memory module")
            actions = []
            with (output / f"{episode}-{mode}.jsonl").open("x") as stream:
                for iteration in range(55):
                    context.budget()
                    if prefix is not None:
                        state = prefix.clone()
                    set_seed(42, cuda=True)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        predicted = policy_action(
                            context.model, module, executor, state, observation, mode
                        )
                        torch.cuda.synchronize()
                        wall_ms = (time.perf_counter() - tick) * 1000
                    allocated, reserved = (
                        torch.cuda.max_memory_allocated(),
                        torch.cuda.max_memory_reserved(),
                    )
                    value, identity = proto.action_record(
                        predicted, raw[0], context.source.processor
                    )
                    if actions and not torch.equal(value, actions[0]):
                        raise ValueError("Fixed-seed current-RGB policy repeat was not exact")
                    if state is not None and state.observations != len(indices):
                        raise ValueError(
                            "Current observation advanced the cached prefix incorrectly"
                        )
                    row = {
                        "episode": episode,
                        "anchor": anchor,
                        "mode": mode,
                        "iteration": iteration,
                        "warmup": iteration < 5,
                        "wall_ms": wall_ms,
                        "peak_allocated_bytes": allocated,
                        "peak_reserved_bytes": reserved,
                        "gpu_state_copies": 0 if prefix is None else 2,
                        "actual_seed": 42,
                        **identity,
                    }
                    records.append(row)
                    actions.append(value)
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    del predicted, value
                    state = None
                    context.budget()
            torch.save(torch.stack(actions), output / f"{episode}-{mode}-actions.pt")
            if prefix is not None and proto.state_hash(prefix) != prefix_hash:
                raise ValueError("Cost working copy mutated frozen prefix state")
            prefix = state = None
        module.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        timings = {
            mode: proto.timing_stats([r for r in records if r["mode"] == mode]) for mode in order
        }
        native, candidate = timings["native"], timings["fp32_step"]
        extra = candidate["peak_allocated_bytes"] - native["peak_allocated_bytes"]
        gate = {
            "latency_passed": candidate["p95_ms"] <= native["p95_ms"] * 1.10,
            "extra_peak_allocated_passed": extra <= 512 * 2**20,
            "absolute_allocated_passed": max(
                native["peak_allocated_bytes"], candidate["peak_allocated_bytes"]
            )
            <= 44 * 2**30,
            "p95_relative_difference": candidate["p95_ms"] / native["p95_ms"] - 1,
            "extra_peak_allocated_bytes": extra,
        }
        report = {
            "episode": episode,
            "anchor": anchor,
            "order": order,
            "timings": timings,
            "gate": gate,
            "passed": all(
                gate[k]
                for k in (
                    "latency_passed",
                    "extra_peak_allocated_passed",
                    "absolute_allocated_passed",
                )
            ),
            "memory_only_prefill": prefill,
            "independent_call_audit": audit,
            "boundary": "currentCPU observation/H2D/VLM/VL/q/memory/Euler4 inside; seed/prefill/state-restore/decode/checks outside; no hooks during measured calls",
        }
        reports.append(report)
        write_new(output / f"COST-{episode}.json", report)
    return reports


def evaluate_command(context):
    from fp32_memory_pilot import (
        PLAN_SHA,
        cache_for,
        cache_native_parity,
        frozen_checkpoints,
        integrity,
    )

    context.plan_sha = PLAN_SHA
    checkpoints = frozen_checkpoints(context)  # Freeze all four before any dev outputs.
    episodes = context.plan[context.args.cohort]
    cache = cache_for(context, confirmation=context.args.cohort == "confirmation")
    parity = cache_native_parity(context, cache, episodes)
    write_new(context.args.output / "DISABLED_CACHE_PARITY.json", parity)
    evaluations, all_rows, numerical = {}, {}, []
    for condition in ("native", *CONDITIONS):
        context.budget()
        module = (
            None
            if condition == "native"
            else load_learned(context, condition, checkpoints[condition])
        )
        evaluations[condition], all_rows[condition] = evaluate_condition(
            context, cache, module, condition, episodes
        )
        if module is not None:
            for episode in episodes:
                numerical.append(
                    numerical_probe(
                        context,
                        cache,
                        module,
                        condition,
                        episode,
                        context.plan["grids"][str(episode)][-1],
                        context.args.cohort,
                    )
                )
            if (
                proto.content_hash(module.state_dict())
                != checkpoints[condition]["module_content_sha256"]
            ):
                raise ValueError("Evaluation mutated a frozen checkpoint")
            del module
            gc.collect()
            torch.cuda.empty_cache()
        print(
            json.dumps({"evaluated": condition, "cohort": context.args.cohort, "rows": 256}),
            flush=True,
        )
    pairing = check_evaluation_pairing(all_rows)
    write_new(context.args.output / "EVALUATION_PAIRING.json", pairing)
    quality = quality_gates(evaluations, episodes)
    # Record quality and finish the preregistered cost even when this gate is false.
    write_new(context.args.output / "QUALITY_GATES.json", quality)
    module = load_learned(context, "mamba_history", checkpoints["mamba_history"])
    costs = cost_probes(context, cache, module, episodes)
    if (
        proto.content_hash(module.state_dict())
        != checkpoints["mamba_history"]["module_content_sha256"]
    ):
        raise ValueError("Cost probe changed trained carry weights")
    for identity in checkpoints.values():
        if sha(identity["path"]) != identity["sha256"]:
            raise ValueError("Frozen checkpoint file changed during joint evaluation")
    final_integrity = integrity(context, cache)
    gates = {
        "quality": quality["passed"],
        "learned_full_step": all(r["passed"] for r in numerical),
        "current_policy_cost": all(r["passed"] for r in costs),
        "pairing": pairing["passed"],
        "native_disabled_cache_parity": True,
        "base_cache_checkpoint_integrity": True,
    }
    passed = all(gates.values())
    return {
        "status": "passed" if passed else "failed",
        "technical_validity": "passed" if gates["learned_full_step"] else "failed",
        "measurements_completed": True,
        "continuation_gate": passed,
        "cohort": context.args.cohort,
        "optimizer_steps": 0,
        "checkpoints": checkpoints,
        "gates": gates,
        "quality": quality,
        "evaluations": evaluations,
        "numerical": numerical,
        "cost": costs,
        "integrity": final_integrity,
        "scope": context.plan["confirmation_scope"],
        "scientific_scope": "Offline pretrained-domain adapter comparison; no closed-loop/generalization or general Mamba superiority claim",
    }
