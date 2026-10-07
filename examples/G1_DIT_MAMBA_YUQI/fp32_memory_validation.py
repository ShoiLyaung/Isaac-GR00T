"""Fixed-weight local FP32 memory validation; no optimizer or training updates."""

import argparse
from contextlib import ExitStack
import gc
import importlib
import importlib.metadata
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from contract import sha, tensor_identity, write_new
from gr00t.model.modules.causal_visual_memory import CausalVisualMemory, inject_image_residual
from gr00t.model.modules.fp32_visual_memory import FP32VisualMemory
from memory_numerical_diagnosis import check_action, historical_action_compatibility, old_gate
from memory_numerical_trace import error_summary, memory_precision
import memory_prototype as proto
from memory_training_support import frame_hash, native_flow, prepare_residual, sampling_packet
import numpy as np
from runner_core import deterministic_seed, set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-causal-mamba-fp32-repair-validation-r1"
)
VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "0bef98e851f826705d40e12343c0f58fcd3daf4fa8ff7cf364a76596773ba048"
FILES = {
    "gr00t/model/modules/fp32_visual_memory.py",
    "examples/G1_DIT_MAMBA_YUQI/fp32_memory_validation.py",
    "examples/G1_DIT_MAMBA_YUQI/README-fp32-memory-validation.md",
    "tests/examples/test_fp32_memory_validation.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def budget(plan, output, deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("FP32 repair validation wall budget exceeded")
    size = sum(p.stat().st_size for p in output.parent.parent.rglob("*") if p.is_file())
    if size > plan["max_artifact_bytes"]:
        raise RuntimeError("FP32 repair validation exceeded 1 GiB artifacts")
    if torch.cuda.max_memory_allocated() > plan["max_allocated_bytes"]:
        raise RuntimeError("FP32 repair validation exceeded 12 GiB allocated memory")
    return size


def verify_inputs(plan_path):
    if sha(plan_path) != PLAN_SHA:
        raise ValueError("Frozen FP32 validation PLAN changed")
    plan = read(plan_path)
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen validation input changed: {path}")
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
    import memory_training_support
    import run_train_eval
    import triton.language.core as core

    modules = (
        gr00t,
        gr00t_n1d7,
        causal_visual_memory,
        fp32_visual_memory,
        proto,
        memory_training_support,
        run_train_eval,
    )
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("Import escaped the FP32 repair worktree")
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
    hashes = {Path(r["path"]).resolve(): r["sha256"] for r in plan["records"]}
    if any(
        Path(m.__file__).resolve() not in hashes
        or sha(m.__file__) != hashes[Path(m.__file__).resolve()]
        for m in kernels
    ):
        raise ValueError("Active kernel source differs from frozen inputs")
    if official.causal_conv1d_fn is not None or os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise ValueError("Expected original no-causal-conv1d/IEEE-default overlay")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Pretrained weights may not be skipped")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(FILES):
        raise ValueError("Expected clean source with only the authorized validation files")
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


def prefix_sequence(cache, indices, device):
    return torch.stack(
        [cache[a]["post_vl_features"][0, proto.IMAGE_POSITIONS] for a in indices]
    ).to(device)


def restore_base_precision():
    if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise ValueError("Memory scope leaked precision settings into the base")


def sample_context(packet, residual):
    features, backbone, action = packet["features"], packet["backbone"], packet["action"]
    if "action" in action or "action_mask" in action:
        raise ValueError("Sampling context may not enable RTC")
    condition = inject_image_residual(
        features.backbone_features,
        torch.tensor(proto.IMAGE_POSITIONS, device=features.backbone_features.device),
        residual,
    )
    return {
        "backbone_features": condition,
        "state_features": features.state_features,
        "embodiment_id": action.embodiment_id,
        "backbone_output": backbone,
        "action_input": action,
        "options": None,
    }


def sampling_identity(kwargs):
    tensors = {
        name: kwargs[name] for name in ("backbone_features", "state_features", "embodiment_id")
    }
    tensors.update({"backbone." + k: v for k, v in kwargs["backbone_output"].items()})
    tensors.update({"action." + k: v for k, v in kwargs["action_input"].items()})
    return {
        "context_sha256": proto.content_hash(tensors),
        "condition": tensor_identity(kwargs["backbone_features"]),
        "state_features": tensor_identity(kwargs["state_features"]),
        "image_mask": tensor_identity(kwargs["backbone_output"].image_mask),
        "attention_mask": tensor_identity(kwargs["backbone_output"].backbone_attention_mask),
    }


def repeated_actions(model, kwargs, seed, repeats):
    identity = sampling_identity(kwargs)
    predictions, records = [], []
    for repeat in range(repeats):
        captured = {}
        calls = []

        def capture(_module, args):
            calls.append(1)
            if len(calls) == 1:
                captured["initial_noise"] = args[0].detach().clone()

        handle = model.action_head.action_encoder.register_forward_pre_hook(capture)
        try:
            set_seed(seed, cuda=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model.action_head.get_action_with_features(**kwargs)["action_pred"]
        finally:
            handle.remove()
        if len(calls) != 4 or sampling_identity(kwargs) != identity:
            raise ValueError("Sampling changed fixed context or native Euler4 call count")
        value = check_action(predicted)
        noise = tensor_identity(captured["initial_noise"])
        if records and noise != records[0]["initial_noise"]:
            raise ValueError("Same-condition sampling initial noise changed after RNG reset")
        predictions.append(value)
        records.append(
            {
                "repeat": repeat,
                "actual_seed": seed,
                "initial_noise": noise,
                "context": identity,
                "action_identity": tensor_identity(value),
            }
        )
    floor = [
        {
            "left_repeat": i,
            "right_repeat": j,
            **error_summary(predictions[j][:, :16], predictions[i][:, :16]),
        }
        for i, j in combinations(range(repeats), 2)
    ]
    return torch.stack(predictions), records, floor


def memory_output(memory, executor, sequence, mode):
    if mode == "fp32_full":
        latent, residual = executor.full(sequence)
        return latent, residual, None
    state = None
    if mode.endswith("step"):
        state = memory.new_state()
        latent = []
        for current in sequence:
            if mode == "fp32_step":
                value, residual = executor.step(current, state)
            else:
                with memory_precision("bf16"):
                    value, residual = memory.step(current, state)
            latent.append(value)
        return torch.stack(latent), residual, state
    with memory_precision("bf16"):
        latent, residual, _ = prepare_residual(memory, sequence, "mamba_history")
    return latent, residual, None


def save(path, values):
    torch.save(values, path)
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def numerical_actions(plan, model, memory, cache, output, deadline):
    executor = FP32VisualMemory(memory)
    numerical, action_rows, floors, flow_rows, lifecycle = [], [], [], [], []
    old_details = read(plan["historical_details"])
    old_predictions = np.load(plan["historical_predictions"])
    old_index = {
        (r["anchor"], r["seed"]): (i, r) for i, r in enumerate(old_details) if r["episode"] == 125
    }
    primary = {tuple(p) for p in plan["primary_pairs"]}
    for anchor in plan["anchors"]:
        indices = plan["prefixes"][str(anchor)]
        sequence = prefix_sequence(cache, indices, "cuda")
        frame = cache[anchor]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base_packet = sampling_packet(model, frame)
        states, tensors, actions = {}, {}, {}
        for mode in plan["modes"]:
            budget(plan, output, deadline)
            with torch.no_grad():
                latent, residual, state = memory_output(memory, executor, sequence, mode)
                if tuple(latent.shape) != (len(indices), 192, 256) or tuple(residual.shape) != (
                    192,
                    2048,
                ):
                    raise ValueError("Unexpected memory latent/readout shape")
                if not torch.isfinite(latent).all() or not torch.isfinite(residual).all():
                    raise ValueError("Nonfinite memory output")
                if mode.startswith("fp32"):
                    with memory_precision("fp32"):
                        oracle_latent, oracle_residual, _ = prepare_residual(
                            memory, sequence, "mamba_history", deployment=mode.endswith("step")
                        )
                    if not torch.equal(latent, oracle_latent) or not torch.equal(
                        residual, oracle_residual
                    ):
                        raise ValueError(
                            f"FP32 wrapper/direct-oracle exact parity failed: {anchor} {mode}"
                        )
            restore_base_precision()
            tensors[mode] = {"latent": latent.cpu(), "residual": residual.cpu()}
            if state is not None:
                if any(not bool(torch.isfinite(v).all()) for pair in state.layers for v in pair):
                    raise ValueError("Nonfinite explicit memory state")
                state_before = proto.state_hash(state)
                cpu_snapshot = state.to("cpu")
                if (
                    state.observations != len(indices)
                    or proto.state_hash(cpu_snapshot) != state_before
                ):
                    raise ValueError("Step state snapshot/counter changed")
                states[mode] = {"state": state, "hash": state_before}
                lifecycle.append(
                    {
                        "anchor": anchor,
                        "mode": mode,
                        "observations": state.observations,
                        "state": cpu_snapshot.describe(),
                        "state_sha256": state_before,
                    }
                )
            seeds = [
                (f"nominal_{s}", deterministic_seed("diagnostic", 125, anchor, s))
                for s in plan["nominal_seeds"]
            ]
            seeds.append(("direct_42", plan["additional_direct_seed"]))
            actions[mode] = {}
            for label, seed in seeds:
                kwargs = sample_context(base_packet, residual)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    flow, receipt = native_flow(model, frame, residual, seed)
                if receipt["effective_condition"] != tensor_identity(kwargs["backbone_features"]):
                    raise ValueError(
                        "Flow and repeated sampler do not share the same effective condition"
                    )
                sampled, rows, floor = repeated_actions(model, kwargs, seed, plan["action_repeats"])
                actions[mode][label] = sampled
                historical = []
                if (
                    anchor == plan["historical_anchor_available"]
                    and mode == "bf16_step"
                    and label.startswith("nominal_")
                ):
                    nominal = int(label.split("_")[1])
                    index, old = old_index[(anchor, nominal)]
                    historical = [
                        historical_action_compatibility(
                            value, float(flow["loss"]), seed, old, old_predictions[index]
                        )
                        for value in sampled
                    ]
                action_rows.append(
                    {
                        "anchor": anchor,
                        "mode": mode,
                        "seed_label": label,
                        "repeats": rows,
                        "historical_compatibility": historical,
                    }
                )
                floors.append(
                    {"anchor": anchor, "mode": mode, "seed_label": label, "comparisons": floor}
                )
                flow_rows.append(
                    {
                        "anchor": anchor,
                        "mode": mode,
                        "seed_label": label,
                        "flow_loss": float(flow["loss"]),
                        "actual_seed": seed,
                        "noise_receipt": receipt,
                    }
                )
                tensors[mode].setdefault("contexts", {})[label] = {
                    "condition": kwargs["backbone_features"].cpu(),
                    "state_features": kwargs["state_features"].cpu(),
                    "image_mask": kwargs["backbone_output"].image_mask.cpu(),
                    "attention_mask": kwargs["backbone_output"].backbone_attention_mask.cpu(),
                }
            if state is not None and proto.state_hash(state) != states[mode]["hash"]:
                raise ValueError("Sampling reads changed the memory state")
            save(output / f"anchor-{anchor}-{mode}.pt", {**tensors[mode], "actions": actions[mode]})
        for ref, actual in combinations(plan["modes"], 2):
            lat = error_summary(tensors[actual]["latent"], tensors[ref]["latent"])
            action_errors = []
            for label in actions[ref]:
                for repeat in range(plan["action_repeats"]):
                    difference = error_summary(
                        actions[actual][label][repeat, :, :16], actions[ref][label][repeat, :, :16]
                    )
                    action_errors.append(
                        {
                            "seed_label": label,
                            "repeat": repeat,
                            **difference,
                            "old_action_gate": old_gate(difference, plan["action_limits"]),
                        }
                    )
            numerical.append(
                {
                    "anchor": anchor,
                    "reference": ref,
                    "actual": actual,
                    "primary_pair": (ref, actual) in primary,
                    "latent": lat,
                    "old_latent_gate": old_gate(lat, plan["latent_limits"]),
                    "residual": error_summary(
                        tensors[actual]["residual"][None], tensors[ref]["residual"][None]
                    ),
                    "active_actions": action_errors,
                }
            )
        for label in actions[plan["modes"][0]]:
            flow_selected = [
                r for r in flow_rows if r["anchor"] == anchor and r["seed_label"] == label
            ]
            sampled_selected = [
                r for r in action_rows if r["anchor"] == anchor and r["seed_label"] == label
            ]
            for key in ("continuous_time", "discrete_time", "noisy_action", "preinjection_post"):
                if (
                    len(
                        {json.dumps(r["noise_receipt"][key], sort_keys=True) for r in flow_selected}
                    )
                    != 1
                ):
                    raise ValueError("Current flow inputs/noise are not paired across modes")
            if (
                len(
                    {
                        json.dumps(r["repeats"][0]["initial_noise"], sort_keys=True)
                        for r in sampled_selected
                    }
                )
                != 1
            ):
                raise ValueError("Sampling initial noise is not paired across modes")
    write_new(output / "NUMERICAL_COMPARISONS.json", numerical)
    write_new(output / "ACTION_REPEATS.json", action_rows)
    write_new(output / "REPEAT_FLOOR.json", floors)
    write_new(output / "FLOW_DETAILS.json", flow_rows)
    write_new(output / "STATE_LIFECYCLE.json", lifecycle)
    return {
        "comparisons": numerical,
        "action_count": sum(len(r["repeats"]) for r in action_rows),
        "flow_count": len(flow_rows),
        "wrapper_oracle_exact_checks": len(plan["anchors"]) * 2,
        "primary_numerical_passed": all(
            r["old_latent_gate"] and all(a["old_action_gate"] for a in r["active_actions"])
            for r in numerical
            if r["primary_pair"]
        ),
    }


def gradient_probe(plan, model, memory, cache, output, deadline):
    before = proto.content_hash(memory.state_dict())
    memory.requires_grad_(True)
    executor = FP32VisualMemory(memory)
    records = []
    try:
        for anchor in plan["gradient_anchors"]:
            budget(plan, output, deadline)
            memory.zero_grad(set_to_none=True)
            sequence = prefix_sequence(cache, plan["prefixes"][str(anchor)], "cuda")
            forward_flags = []

            def precision_flags(label):
                return {
                    "label": label,
                    "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
                    "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                }

            with ExitStack() as scope:
                for label, part in [
                    ("input_projection", memory.input_projection),
                    ("readout", memory.output_projection),
                ]:

                    def capture(_module, args, label=label):
                        forward_flags.append(precision_flags(label))

                    scope.callback(part.register_forward_pre_hook(capture).remove)
                with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    latent, residual = executor.full(sequence)
                    result, receipt = native_flow(
                        model, cache[anchor], residual, plan["gradient_seed"]
                    )
                    backward_flags = precision_flags("backward_ambient")
                    result["loss"].backward()
            groups = {
                "input_projection": memory.input_projection,
                "output_projection": memory.output_projection,
            }
            for index, layer in enumerate(memory.layers):
                groups[f"norm{index}"] = layer.norm
                groups[f"recurrence{index}"] = layer.mamba
            reports = {name: proto.gradient_group(module) for name, module in groups.items()}
            records.append(
                {
                    "anchor": anchor,
                    "loss": float(result["loss"].detach()),
                    "noise_receipt": receipt,
                    "groups": reports,
                    "forward_memory_flags": forward_flags,
                    "backward_ambient_flags": backward_flags,
                    "gradient_claim": "Reachability after FP32 memory forward only; not all-FP32/IEEE backward or gradient-equivalence proof",
                    "reachable_groups_passed": all(
                        v["finite_nonzero_parameters"] > 0 for v in reports.values()
                    ),
                }
            )
            proto.assert_frozen(model)
            del sequence, latent, residual, result
    finally:
        memory.zero_grad(set_to_none=True)
        memory.requires_grad_(False)
    if proto.content_hash(memory.state_dict()) != before:
        raise ValueError("Zero-step gradient dry run changed memory weights")
    report = {
        "status": "completed",
        "optimizer_steps": 0,
        "records": records,
        "reachable_groups_passed": all(r["reachable_groups_passed"] for r in records),
        "weight_content_unchanged": True,
    }
    write_new(output / "GRADIENT_DRY_RUN.json", report)
    return report


def policy_action(model, memory, executor, state, observation, mode):
    if mode == "native":
        return model.get_action(dict(observation))["action_pred"]
    packet = proto.encode_observation(model, observation)
    if mode == "fp32_step":
        _, residual = executor.step(proto.slots(packet), state)
    else:
        _, residual = memory.step(proto.slots(packet), state)
    return proto.sample_packet(model, packet, residual)


def prefill_snapshot(memory, executor, cache, indices, mode):
    state = memory.new_state()
    with torch.no_grad():
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for index in indices:
            current = cache[index]["post_vl_features"][0, proto.IMAGE_POSITIONS].cuda()
            if mode == "fp32_step":
                executor.step(current, state)
            else:
                with memory_precision("bf16"):
                    memory.step(current, state)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - tick) * 1000
    snapshot = state.to("cpu")
    if proto.state_hash(state) != proto.state_hash(snapshot) or state.observations != len(indices):
        raise ValueError("Prefill snapshot did not preserve state")
    return snapshot, {
        "wall_ms": wall_ms,
        "observations": len(indices),
        "state": snapshot.describe(),
        "sha256": proto.state_hash(snapshot),
        "boundary": "Frozen cached post-VL H2D plus memory-only prefill; not RGB cold start",
    }


def call_audit(model, memory, executor, observation, prefix, mode):
    order = []

    def counted(name, method):
        def call(*args, **kwargs):
            order.append(name)
            return method(*args, **kwargs)

        return call

    def dit(*args):
        order.append("dit")

    state = prefix.clone()
    with ExitStack() as scope:
        for index, layer in enumerate(memory.layers):
            scope.enter_context(
                patch.object(layer.mamba, "step", counted(f"mamba{index}", layer.mamba.step))
            )
        scope.callback(model.action_head.model.register_forward_pre_hook(dit).remove)
        set_seed(42, cuda=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            value = policy_action(model, memory, executor, state, observation, mode)
    check_action(value)
    if (
        order != ["mamba0", "mamba1", "dit", "dit", "dit", "dit"]
        or state.observations != prefix.observations + 1
    ):
        raise ValueError("Actual production call-count/order changed")
    return {
        "order": order,
        "state_observations": state.observations,
        "instrumentation_removed": True,
    }


def cost_gate(native, candidate, plan):
    return {
        "p95_passed": candidate["p95_ms"]
        <= native["p95_ms"] * (1 + plan["max_extra_policy_p95_fraction"]),
        "extra_peak_passed": candidate["peak_allocated_bytes"] - native["peak_allocated_bytes"]
        <= plan["max_extra_peak_allocated_bytes"],
        "p95_relative_difference": candidate["p95_ms"] / native["p95_ms"] - 1,
        "extra_peak_allocated_bytes": candidate["peak_allocated_bytes"]
        - native["peak_allocated_bytes"],
    }


def cost_probes(plan, model, memory, cache, output, deadline):
    from run_train_eval import NativeAnchors

    # All diagnostics have returned only CPU records before this function begins.
    memory.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    source = NativeAnchors(Path(plan["prepared"]))
    source.processor.eval()
    executor = FP32VisualMemory(memory)
    reports = []
    for anchor in plan["anchors"]:
        batch, raw = source.batch([(125, anchor)])
        observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
        proto.validate_input_layout(observation, source.modality["video"].modality_keys)
        for key in (
            "state",
            "action",
            "action_mask",
            "embodiment_id",
            "input_ids",
            "attention_mask",
            "image_grid_thw",
        ):
            if tensor_identity(batch[key]) != cache[anchor]["input_identity"][key]:
                raise ValueError(
                    f"Current state/target/mask/layout identity changed: {anchor} {key}"
                )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            packet = proto.encode_observation(model, observation)
        rgb_compat = error_summary(
            packet["features"].backbone_features.cpu(), cache[anchor]["post_vl_features"]
        )
        del packet
        records, prefill_reports, audits = [], {}, {}
        for mode in plan["cost_order"][str(anchor)]:
            memory.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            prefix = state = None
            if mode != "native":
                memory.cuda()
                snapshot, prefill = prefill_snapshot(
                    memory, executor, cache, plan["prefixes"][str(anchor)][:-1], mode
                )
                prefix = snapshot.to("cuda")
                prefix_hash = proto.state_hash(prefix)
                prefill_reports[mode] = prefill
                audits[mode] = call_audit(model, memory, executor, observation, prefix, mode)
            elif any(p.is_cuda for p in memory.parameters()):
                raise ValueError("Native timing retains unused memory weights on GPU")
            mode_outputs = []
            with (output / f"cost-{anchor}-{mode}.jsonl").open("x") as stream:
                for iteration in range(plan["cost_warmup"] + plan["cost_measured"]):
                    budget(plan, output, deadline)
                    if prefix is not None:
                        state = prefix.clone()
                    set_seed(plan["cost_seed"], cuda=True)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        predicted = policy_action(model, memory, executor, state, observation, mode)
                        torch.cuda.synchronize()
                        wall_ms = (time.perf_counter() - tick) * 1000
                    allocated, reserved = (
                        torch.cuda.max_memory_allocated(),
                        torch.cuda.max_memory_reserved(),
                    )
                    value, action = proto.action_record(predicted, raw[0], source.processor)
                    if state is not None and state.observations != len(
                        plan["prefixes"][str(anchor)]
                    ):
                        raise ValueError("Current observation advanced state incorrectly")
                    row = {
                        "anchor": anchor,
                        "mode": mode,
                        "iteration": iteration,
                        "warmup": iteration < plan["cost_warmup"],
                        "wall_ms": wall_ms,
                        "peak_allocated_bytes": allocated,
                        "peak_reserved_bytes": reserved,
                        "gpu_state_copies": 0 if prefix is None else 2,
                        "actual_seed": plan["cost_seed"],
                        **action,
                    }
                    records.append(row)
                    mode_outputs.append(value)
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    del predicted, value
                    state = None
                    budget(plan, output, deadline)
            save(output / f"cost-{anchor}-{mode}-actions.pt", torch.stack(mode_outputs))
            if prefix is not None and proto.state_hash(prefix) != prefix_hash:
                raise ValueError("Cost working clone mutated shared prefix state")
            prefix = state = None
            write_new(
                output / f"cost-{anchor}-{mode}-repeat.json",
                {
                    "scope": "Descriptive variation with recomputed current RGB policy; not same frozen condition sampling floor",
                    "against_first": [
                        error_summary(v[:, :16], mode_outputs[0][:, :16]) for v in mode_outputs
                    ],
                },
            )
        memory.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        timings = {
            mode: proto.timing_stats([r for r in records if r["mode"] == mode])
            for mode in plan["cost_modes"]
        }
        gate = cost_gate(timings["native"], timings["fp32_step"], plan)
        report = {
            "anchor": anchor,
            "order": plan["cost_order"][str(anchor)],
            "timings": timings,
            "candidate_gate": gate,
            "memory_only_prefill": prefill_reports,
            "independent_call_audit": audits,
            "current_rgb_vs_cache_compatibility": rgb_compat,
            "boundary": "Current CPU observation/H2D/VLM/VL/q/memory/Euler4 included; seed/restore/decode/checks excluded",
        }
        reports.append(report)
        write_new(output / f"COST-{anchor}.json", report)
    return reports


def run(args, deadline):
    from run_train_eval import strict_reload

    plan = verify_inputs(args.plan)
    source = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cache = torch.load(plan["cache_file"], map_location="cpu", weights_only=True)
    cache_before = {a: frame_hash(v) for a, v in cache.items()}
    if list(cache) != plan["prefixes"]["320"] or any(
        v["content_sha256"] != cache_before[a] for a, v in cache.items()
    ):
        raise ValueError("Frozen episode125 cache identity changed")
    checkpoint = torch.load(plan["memory_checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["completed_steps"] != 200 or checkpoint["condition"] != "mamba_history":
        raise ValueError("Expected unchanged step200 memory")
    memory = CausalVisualMemory()
    memory.load_state_dict(checkpoint["module"], strict=True)
    memory.cuda().eval().requires_grad_(False)
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    base_before = proto.content_hash(model.state_dict())
    memory_before = proto.content_hash(memory.state_dict())
    if base_before != checkpoint["base_content_sha256"]:
        raise ValueError("Original base weight identity differs")
    del checkpoint
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "environment": source,
            "strict_loading": loading,
            "base_before": base_before,
            "memory_before": memory_before,
            "cache_before": cache_before,
            "optimizer_steps": 0,
        },
    )
    budget(plan, args.output, deadline)
    numerical = numerical_actions(plan, model, memory, cache, args.output, deadline)
    gradients = gradient_probe(plan, model, memory, cache, args.output, deadline)
    costs = cost_probes(plan, model, memory, cache, args.output, deadline)
    proto.assert_frozen(model)
    proto.assert_frozen(memory)
    restore_base_precision()
    base_after = proto.content_hash(model.state_dict())
    memory_after = proto.content_hash(memory.state_dict())
    cache_after = {a: frame_hash(v) for a, v in cache.items()}
    if base_after != base_before or memory_after != memory_before or cache_after != cache_before:
        raise ValueError("Fixed base/memory/cache contents changed")
    if numerical["action_count"] != 120 or numerical["flow_count"] != 40:
        raise ValueError("Incomplete prescribed repeated-action/flow samples")
    gates = {
        "primary_numerical": numerical["primary_numerical_passed"],
        "gradient_reachability": gradients["reachable_groups_passed"],
        "candidate_cost": all(
            r["candidate_gate"]["p95_passed"] and r["candidate_gate"]["extra_peak_passed"]
            for r in costs
        ),
    }
    supported = all(gates.values())
    return {
        "status": "passed" if supported else "failed",
        "technical_validity": "passed",
        "measurements_completed": True,
        "limited_repair_supported": supported,
        "adoption_gates": gates,
        "optimizer_steps": 0,
        "old_failures_unchanged": True,
        "numerical": numerical,
        "gradient": gradients,
        "costs": costs,
        "action_count": 120,
        "flow_count": 40,
        "cost_calls": 330,
        "integrity": {
            "base_before": base_before,
            "base_after": base_after,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "cache_before": cache_before,
            "cache_after": cache_after,
        },
        "artifact_bytes_before_result": budget(plan, args.output, deadline),
        "claim_scope": "Two fixed prefixes and one learned checkpoint; local execution candidate only, no new training or generalization/deployment claim",
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
                "old_failures_unchanged": True,
                "automatic_retry": False,
            },
        )
        raise
    if not report["limited_repair_supported"]:
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
