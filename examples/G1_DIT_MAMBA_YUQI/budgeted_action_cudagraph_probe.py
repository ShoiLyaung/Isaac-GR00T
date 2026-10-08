"""Frozen-weight three-path action-core CUDA Graph comparison; zero optimizer."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from budgeted_action_cudagraph import (
    BackwardGraph,
    CoreDispatcher,
    ForwardGraph,
    TensorCore,
    replace_core,
    tensor_args,
)
from budgeted_action_support import bind_request, clean_task_loss, current_policy, group_modules
from contract import sha, tensor_identity, write_new
from fp32_pilot_support import PilotCache
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
from memory_numerical_trace import error_summary
import memory_prototype as proto
from memory_training_support import fresh, native_flow
import numpy as np
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
PLAN_SHA = "889617e9da4976bac04ea017f6b7801638daa0d64edf2471569ee9784d5ed93d"
FILES = {
    "examples/G1_DIT_MAMBA_YUQI/budgeted_action_cudagraph.py",
    "examples/G1_DIT_MAMBA_YUQI/budgeted_action_cudagraph_probe.py",
    "examples/G1_DIT_MAMBA_YUQI/README-cudagraph.md",
    "tests/examples/test_budgeted_cudagraph.py",
}
PATHS = ("production_eager", "bridge_eager", "graph")


def read(path):
    return json.loads(Path(path).read_text())


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen graph PLAN changed")
    plan = read(args.plan)
    if (
        REPO != Path(plan["source_worktree"])
        or args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve()
    ):
        raise ValueError("Frozen graph source/artifact identity mismatch")
    if plan["training_authorized"] or plan["optimizer_steps"] != 0:
        raise ValueError("Only zero-update execution comparison allowed")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
    for relative in plan["immutable_source_relative_paths"]:
        expected = {r["sha256"] for r in plan["records"] if r["path"].endswith("/" + relative)}
        if len(expected) != 1 or sha(REPO / relative) not in expected:
            raise ValueError(f"Existing source changed: {relative}")
    return plan


def environment(plan):
    import budgeted_action_cudagraph
    import budgeted_action_support
    import gr00t
    from gr00t.model.modules import budgeted_action_mamba
    import mamba_ssm.modules.mamba2 as official
    import triton.language.core as core

    modules = (
        gr00t,
        budgeted_action_mamba,
        budgeted_action_support,
        budgeted_action_cudagraph,
        proto,
    )
    if any(REPO not in Path(m.__file__).resolve().parents for m in modules):
        raise ValueError("Import escaped graph worktree")
    if (
        sys.version_info[:2] != (3, 10)
        or Path(sys.prefix) != VENV
        or torch.__version__.split("+")[0] != "2.7.1"
    ):
        raise ValueError("Original Python/torch environment required")
    kernels = [official, core, torch.cuda.graphs] + [
        importlib.import_module("mamba_ssm.ops.triton." + name)
        for name in (
            "ssd_combined",
            "ssd_chunk_scan",
            "ssd_chunk_state",
            "selective_state_update",
            "layernorm_gated",
        )
    ]
    frozen = {Path(row["path"]).resolve(): row["sha256"] for row in plan["records"]}
    if any(
        Path(m.__file__).resolve() not in frozen
        or sha(m.__file__) != frozen[Path(m.__file__).resolve()]
        for m in kernels
    ):
        raise ValueError("Official graph/kernel source identity changed")
    if (
        importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
        or official.causal_conv1d_fn is not None
        or os.environ.get("TRITON_F32_DEFAULT") != "ieee"
    ):
        raise ValueError("Frozen kernel/precision environment changed")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Pretrained weights cannot be skipped")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(FILES):
        raise ValueError("Expected clean graph-only source additions")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != str(plan["physical_gpu"])
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose only frozen physicalGPU")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "imports": {m.__name__: m.__file__ for m in (*modules, *kernels)},
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(0),
    }


def context_for(args, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    plan = verify_plan(args)
    source = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    context = SimpleNamespace(args=args, plan=plan, deadline=deadline, lease=None, stage_peak=0)

    def budget():
        context.stage_peak = max(context.stage_peak, torch.cuda.max_memory_allocated())
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if time.monotonic() > deadline:
            raise TimeoutError("Graph diagnosis wall bound exceeded")
        if size > plan["max_artifact_bytes"] or context.stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Graph diagnosis resource bound exceeded")
        return size

    context.budget = budget
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    context.model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    context.teacher = model.action_head.model
    proto.assert_frozen(model)
    context.base_before = proto.content_hash(model.state_dict())
    if context.base_before != plan["base_content_sha256"]:
        raise ValueError("Frozen B0 content differs")
    values = torch.load(
        plan["student_initialization"]["path"], map_location="cpu", weights_only=True
    )
    if proto.content_hash(values) != plan["student_initialization"]["content_sha256"]:
        raise ValueError("Frozen student initialization differs")
    context.student = BudgetedActionMamba(context.teacher)
    context.student.load_state_dict(values, strict=True)
    context.student.eval().requires_grad_(True)
    del values
    context.source = NativeAnchors(Path(plan["prepared"]))
    context.source.processor.eval()
    context.cache = PilotCache(
        plan["cache_root"], plan, expected_report_sha=plan["cache_report_sha256"]
    )
    context.observations = {}
    context.raw = {}
    for row in plan["cost_inputs"]:
        ep, a = row["episode"], row["anchor"]
        batch, raw = context.source.batch([(ep, a)])
        obs = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
        proto.validate_input_layout(obs, context.source.modality["video"].modality_keys)
        context.observations[ep] = obs
        context.raw[ep] = raw[0]
        context.cache.frame(ep, a)
    context.budget()
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "source": source,
            "strict_loading": loading,
            "base_content_sha256": context.base_before,
            "student_content_sha256": proto.content_hash(context.student.state_dict()),
            "optimizer_steps": 0,
        },
    )
    return context


def activate(context, name, phase):
    if context.lease is not None:
        raise ValueError("Release graph before moving any captured core")
    context.budget()
    for core in (context.teacher, context.student):
        core.zero_grad(set_to_none=True)
        core.cpu()
        core.eval()
    context.teacher.to(dtype=torch.bfloat16).requires_grad_(False)
    context.student.float().requires_grad_(True)
    gc.collect()
    torch.cuda.empty_cache()
    core = context.teacher if name == "native" else context.student
    core.to(
        device="cuda", dtype=torch.float32 if phase == "fb" or name == "student" else torch.bfloat16
    )
    if phase == "fb":
        core.requires_grad_(True)
    core.eval()
    torch.cuda.synchronize()
    context.budget()
    return core


@contextmanager
def route_context(context, name, route):
    if name == "native" and route is not None:
        with replace_core(context.model, route):
            yield
    else:
        yield


def policy(context, name, episode, route=None):
    with route_context(context, name, route):
        return current_policy(
            context.model,
            context.observations[episode],
            None if name == "native" else (route or context.student),
        )


def task_loss(context, name, frame, route=None):
    if route is None:
        return clean_task_loss(
            context.model, frame, student=None if name == "native" else context.student
        )
    with replace_core(context.model, route):
        return context.model.action_head(
            fresh(frame["raw"], context.model.device),
            fresh(frame["action_input"], context.model.device),
        )["loss"]


def validation_flow(context, name, frame, seed, route=None):
    if route is not None:
        with replace_core(context.model, route):
            result, receipt = native_flow(context.model, frame, None, seed)
    elif name == "native":
        result, receipt = native_flow(context.model, frame, None, seed)
    else:
        condition = context.student.encode_condition(
            frame["post_vl_features"].to(context.model.device)
        )
        with bind_request(context.model, context.student, condition):
            result, receipt = native_flow(context.model, frame, None, seed)
    receipt["state_features"] = tensor_identity(result["state_features"])
    receipt["target_action"] = tensor_identity(frame["action_input"]["action"])
    receipt["action_mask"] = tensor_identity(result["action_mask"])
    return result["loss"], receipt


def sample_surface(context, name, phase):
    core = context.teacher if name == "native" else context.student
    seen = []

    def capture(_module, args, kwargs):
        seen.append(tensor_args(kwargs, student=name == "student", backward=phase == "fb"))

    handle = core.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        set_seed(42, cuda=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if phase == "inference":
                value = policy(context, name, context.plan["cost_inputs"][0]["episode"])
            else:
                item = context.plan["fb_input"]
                value = task_loss(
                    context, name, context.cache.frame(item["episode"], item["anchor"])
                )
        del value
    finally:
        handle.remove()
    if len(seen) != (4 if phase == "inference" else 1):
        raise ValueError("Native sampler/core cadence changed")
    return seen[0]


def make_route(context, name, phase, samples, path):
    if path == "production_eager":
        return None, None
    core = context.teacher if name == "native" else context.student
    surface = TensorCore(core, student=name == "student", backward=phase == "fb")
    if path == "bridge_eager":
        return CoreDispatcher(surface), None
    if context.lease is not None:
        raise ValueError("Only one graph lifetime is permitted")
    lease = (
        ForwardGraph(surface, samples) if phase == "inference" else BackwardGraph(surface, samples)
    )
    context.lease = lease
    return CoreDispatcher(surface, lease), lease


def release_route(context, route, lease):
    if lease is not None:
        lease.close()
        context.lease = None
    if route is not None:
        route.backend = None
        route.surface = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def sampled(context, name, ep, seed, route=None):
    set_seed(seed, cuda=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = policy(context, name, ep, route)
    result, identity = proto.action_record(value, context.raw[ep], context.source.processor)
    return result, identity


def summarize(rows):
    values = [r["wall_ms"] for r in rows if not r["warmup"]]
    if not values or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Invalid timing series")
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
    }


def policy_references(context, name, samples, output):
    refs = {}
    rows = []
    for path in ("production_eager", "bridge_eager"):
        route, lease = make_route(context, name, "inference", samples, path)
        refs[path] = {}
        for item in context.plan["cost_inputs"]:
            ep = item["episode"]
            for seed in context.plan["numerical_seeds"]:
                context.budget()
                value, identity = sampled(context, name, ep, seed, route)
                refs[path][(ep, seed)] = value
                if path == "production_eager" and seed == 42:
                    old = torch.load(
                        context.plan["reference_actions"][str(ep)][name],
                        map_location="cpu",
                        weights_only=True,
                    )[0]
                    if not torch.equal(value, old):
                        raise ValueError("Old production seed42 action did not reproduce bitwise")
                rows.append({"path": path, "episode": ep, "seed": seed, **identity})
        release_route(context, route, lease)
        del route
    torch.save(
        {
            path: {f"{ep}:{seed}": value for (ep, seed), value in values.items()}
            for path, values in refs.items()
        },
        output / "eager-reference-actions.pt",
    )
    write_new(output / "EAGER_REFERENCES.json", rows)
    return refs


def dynamic_refresh(context, name, route, lease, refs):
    episodes = [context.plan["cost_inputs"][i]["episode"] for i in (0, 1, 0)]
    times = []
    equal = []
    actions = []

    def inspect(_module, args, kwargs, output):
        current = tensor_args(kwargs, student=name == "student", backward=False)
        equal.append(all(torch.equal(a, b) for a, b in zip(current, lease.static)))
        times.append(int(kwargs["timestep"].detach().cpu()[0]))

    handle = route.register_forward_hook(inspect, with_kwargs=True)
    try:
        for ep in episodes:
            actions.append(sampled(context, name, ep, 42, route)[0])
    finally:
        handle.remove()
    if not all(equal) or times != [0, 250, 500, 750] * 3 or not torch.equal(actions[0], actions[2]):
        raise ValueError("Dynamic A-B-A/static-input refresh or Euler time order failed")
    return {
        "episodes": episodes,
        "time_buckets": times,
        "all_static_input_copies_equal": True,
        "a_repeat_bitwise": True,
        "core_calls": len(equal),
    }


def inference_phase(context, name):
    output = context.args.output / f"inference-{name}"
    output.mkdir(exist_ok=False)
    activate(context, name, "inference")
    samples = sample_surface(context, name, "inference")
    refs = policy_references(context, name, samples, output)
    costs = {}
    numeric = []
    setup = {}
    dynamic = None
    for path in context.plan["path_order"][name]:
        if path != "graph" and context.lease is not None:
            raise ValueError("Eager timing cannot coexist with graph pool")
        route, lease = make_route(context, name, "inference", samples, path)
        if lease is not None:
            setup = {
                **lease.setup,
                "allocated_after_capture": torch.cuda.memory_allocated(),
                "reserved_after_capture": torch.cuda.memory_reserved(),
            }
            dynamic = dynamic_refresh(context, name, route, lease, refs)
            graph_values = {}
            for item in context.plan["cost_inputs"]:
                ep = item["episode"]
                for seed in context.plan["numerical_seeds"]:
                    value, identity = sampled(context, name, ep, seed, route)
                    stats = error_summary(value[:, :16], refs["bridge_eager"][(ep, seed)][:, :16])
                    production_error = error_summary(
                        value[:, :16], refs["production_eager"][(ep, seed)][:, :16]
                    )
                    numeric.append(
                        {
                            "episode": ep,
                            "seed": seed,
                            "bridge_production_exact": torch.equal(
                                refs["production_eager"][(ep, seed)],
                                refs["bridge_eager"][(ep, seed)],
                            ),
                            "graph_bridge_error": stats,
                            "graph_production_error": production_error,
                            "graph_production_gate": all(
                                production_error[k] <= v
                                for k, v in context.plan["action_limits"].items()
                            ),
                            "graph_action_gate": all(
                                stats[k] <= v for k, v in context.plan["action_limits"].items()
                            ),
                            **identity,
                        }
                    )
                    graph_values[(ep, seed)] = value
            torch.save(
                {f"{ep}:{seed}": value for (ep, seed), value in graph_values.items()},
                output / "graph-numerical-actions.pt",
            )
        costs[path] = {}
        for item in context.plan["cost_inputs"]:
            ep = item["episode"]
            rows = []
            actions = []
            with (output / f"{path}-{ep}.jsonl").open("x") as stream:
                for index in range(55):
                    context.budget()
                    set_seed(42, cuda=True)
                    torch.cuda.reset_peak_memory_stats()
                    before_calls = route.calls if route is not None else None
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        value = policy(context, name, ep, route)
                        torch.cuda.synchronize()
                        wall = (time.perf_counter() - tick) * 1000
                    peak, reserved = (
                        torch.cuda.max_memory_allocated(),
                        torch.cuda.max_memory_reserved(),
                    )
                    result, identity = proto.action_record(
                        value, context.raw[ep], context.source.processor
                    )
                    expected = graph_values[(ep, 42)] if path == "graph" else refs[path][(ep, 42)]
                    if not torch.equal(result, expected):
                        raise ValueError("Clean same-path action changed")
                    if route is not None and route.calls - before_calls != 4:
                        raise ValueError("Policy must execute exactly4 core dispatches")
                    record = {
                        "model": name,
                        "path": path,
                        "episode": ep,
                        "iteration": index,
                        "warmup": index < 5,
                        "wall_ms": wall,
                        "peak_allocated_bytes": peak,
                        "peak_reserved_bytes": reserved,
                        "core_dispatches": 4,
                        **identity,
                    }
                    rows.append(record)
                    actions.append(result)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    del value, result
            torch.save(torch.stack(actions), output / f"{path}-{ep}-actions.pt")
            costs[path][ep] = summarize(rows)
        if lease is not None:
            lease.assert_storage()
        release_route(context, route, lease)
        del route, lease
        context.budget()
    setup_seconds = setup["side_stream_warmup_seconds"] + setup["capture_seconds"]
    break_even = {
        str(ep): {
            path: (
                math.ceil(
                    setup_seconds
                    * 1000
                    / (costs[path][ep]["mean_ms"] - costs["graph"][ep]["mean_ms"])
                )
                if costs[path][ep]["mean_ms"] > costs["graph"][ep]["mean_ms"]
                else None
            )
            for path in ("production_eager", "bridge_eager")
        }
        for ep in costs["graph"]
    }
    report = {
        "model": name,
        "costs": costs,
        "numeric": numeric,
        "numeric_gate": all(
            r["bridge_production_exact"] and r["graph_action_gate"] and r["graph_production_gate"]
            for r in numeric
        ),
        "dynamic_refresh": dynamic,
        "capture": setup,
        "break_even_policy_invocations": break_even,
        "capture_count": 1,
        "graph_pool_released": context.lease is None,
    }
    write_new(output / "RESULT.json", report)
    return report


def group_mapping(core, name):
    if name == "student":
        lookup = {
            id(parameter): label
            for label, module in group_modules(core).items()
            for parameter in module.parameters()
        }
        return {key: lookup[id(parameter)] for key, parameter in core.named_parameters()}
    result = {}
    for key, _ in core.named_parameters():
        pieces = key.split(".")
        result[key] = "block" + pieces[1] if pieces[0] == "transformer_blocks" else pieces[0]
    if set(result.values()) != {
        *[f"block{i}" for i in range(32)],
        "timestep_encoder",
        "proj_out_1",
        "proj_out_2",
    }:
        raise ValueError("Native gradient groups changed")
    return result


def assert_gradients_cleared(core):
    if any(parameter.grad is not None for parameter in core.parameters()):
        raise ValueError("Gradient accumulation: zero_grad did not clear every parameter")


def gradient_fingerprint(core):
    """Small descriptive sample, without an extra exactness gate."""
    schema, samples = [], []
    for name, parameter in core.named_parameters():
        gradient = parameter.grad
        schema.append([name, None if gradient is None else list(gradient.shape)])
        if gradient is not None:
            flat = gradient.detach().flatten()
            indices = torch.tensor(
                np.linspace(0, flat.numel() - 1, min(8, flat.numel()), dtype=np.int64),
                device=flat.device,
            )
            samples.append(flat.index_select(0, indices).float())
    digest = hashlib.sha256(json.dumps(schema).encode())
    if samples:
        digest.update(torch.cat(samples).cpu().numpy().tobytes())
    return digest.hexdigest()


def snapshot_gradients(core):
    values = {}
    for name, parameter in core.named_parameters():
        if parameter.grad is None:
            values[name] = None
        else:
            value = parameter.grad.detach().to(device="cpu", copy=True)
            if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite or malformed gradient")
            values[name] = value
    return values


def metric_from_sums(sums, limits):
    ref = math.sqrt(max(sums["reference_sq"], 0))
    candidate = math.sqrt(max(sums["candidate_sq"], 0))
    error = math.sqrt(max(sums["error_sq"], 0))
    if ref <= limits["zero_reference_norm"]:
        return {
            "reference_norm": ref,
            "candidate_norm": candidate,
            "relative_l2": None,
            "cosine": None,
            "passed": candidate <= limits["zero_reference_norm"],
        }
    cosine = sums["dot"] / (ref * candidate) if candidate > 0 else 0.0
    relative = error / ref
    return {
        "reference_norm": ref,
        "candidate_norm": candidate,
        "relative_l2": relative,
        "cosine": cosine,
        "passed": relative <= limits["relative_l2"] and cosine >= limits["cosine_min"],
    }


def gradient_comparison(reference, candidate, mapping, limits):
    if reference.keys() != candidate.keys():
        raise ValueError("Gradient parameter inventory changed")
    records = []
    sums = {}
    pattern = True
    for name, a in reference.items():
        b = candidate[name]
        matching = (a is None) == (b is None)
        pattern = pattern and matching
        row = {
            "parameter": name,
            "group": mapping[name],
            "reference_none": a is None,
            "candidate_none": b is None,
            "none_pattern_match": matching,
        }
        values = {"reference_sq": 0.0, "candidate_sq": 0.0, "dot": 0.0, "error_sq": 0.0}
        if a is not None and b is not None:
            if a.shape != b.shape:
                raise ValueError("Gradient shape mismatch")
            av, bv = a.double().flatten(), b.double().flatten()
            delta = bv - av
            values = {
                "reference_sq": float(av.square().sum()),
                "candidate_sq": float(bv.square().sum()),
                "dot": float((av * bv).sum()),
                "error_sq": float(delta.square().sum()),
            }
            indices = np.linspace(0, len(av) - 1, min(8, len(av)), dtype=np.int64).tolist()
            row.update(
                shape=list(a.shape),
                dtype=str(a.dtype),
                maxabs=float(delta.abs().max()),
                sample_indices=indices,
                reference_sample=av[indices].tolist(),
                candidate_sample=bv[indices].tolist(),
            )
        elif not matching:
            value = a if a is not None else b
            sq = float(value.double().square().sum())
            values["reference_sq" if a is not None else "candidate_sq"] = sq
            values["error_sq"] = sq
        row.update(values)
        records.append(row)
        for group in (mapping[name], "GLOBAL"):
            total = sums.setdefault(group, {key: 0.0 for key in values})
            for key, value in values.items():
                total[key] += value
    metrics = {group: {**value, **metric_from_sums(value, limits)} for group, value in sums.items()}
    return {
        "passed": pattern and all(row["passed"] for row in metrics.values()),
        "none_pattern_match": pattern,
        "parameters": records,
        "groups": metrics,
        "reduction": "CPU FP64 sufficient statistics; small fixed coordinate samples only, no gradient files",
    }


def fb_numeric(context, name, route, output):
    core = context.teacher if name == "native" else context.student
    mapping = group_mapping(core, name)
    reports = []
    bridge = CoreDispatcher(TensorCore(core, student=name == "student", backward=True))
    for item in context.plan["cost_inputs"]:
        ep, a = item["episode"], item["anchor"]
        frame = context.cache.frame(ep, a)
        for seed in context.plan["fb_numerical_seeds"]:
            context.budget()
            data = {}
            for path, current in (
                ("production_eager", None),
                ("bridge_eager", bridge),
                ("graph", route),
            ):
                core.zero_grad(set_to_none=True)
                assert_gradients_cleared(core)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, receipt = validation_flow(context, name, frame, seed, current)
                loss.backward()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite native loss")
                data[path] = {
                    "loss": float(loss.detach()),
                    "receipt": receipt,
                    "gradients": snapshot_gradients(core),
                }
                del loss
            if not all(
                data[path]["receipt"] == data["production_eager"]["receipt"] for path in PATHS
            ):
                raise ValueError("Actual noisy action/time/condition/state receipts differ")
            pairs = []
            for ref, candidate in context.plan["gradient_comparisons"]:
                lhs, rhs = data[ref], data[candidate]
                difference = abs(rhs["loss"] - lhs["loss"])
                limits = context.plan["loss_limits"]
                loss_gate = difference <= limits["atol"] + limits["rtol"] * abs(lhs["loss"])
                stats = gradient_comparison(
                    lhs["gradients"], rhs["gradients"], mapping, context.plan["gradient_limits"]
                )
                details = {
                    "reference": ref,
                    "candidate": candidate,
                    "reference_loss": lhs["loss"],
                    "candidate_loss": rhs["loss"],
                    "loss_abs_difference": difference,
                    "loss_gate": loss_gate,
                    **stats,
                }
                path = output / f"gradients-{ep}-{seed}-{ref}-vs-{candidate}.json"
                write_new(path, details)
                pairs.append(
                    {
                        "reference": ref,
                        "candidate": candidate,
                        "loss_gate": loss_gate,
                        "gradient_gate": stats["passed"],
                        "sufficient_statistics_path": str(path),
                        "sha256": sha(path),
                    }
                )
            reports.append(
                {
                    "episode": ep,
                    "anchor": a,
                    "seed": seed,
                    "actual_receipts_equal": True,
                    "receipt": data["production_eager"]["receipt"],
                    "comparisons": pairs,
                    "passed": all(p["loss_gate"] and p["gradient_gate"] for p in pairs),
                }
            )
            core.zero_grad(set_to_none=True)
            del data
            gc.collect()
            context.budget()
    del bridge
    return reports


def fb_phase(context, name):
    output = context.args.output / f"fb-{name}"
    output.mkdir(exist_ok=False)
    core = activate(context, name, "fb")
    samples = sample_surface(context, name, "fb")
    costs = {}
    numeric = []
    setup = {}
    item = context.plan["fb_input"]
    frame = context.cache.frame(item["episode"], item["anchor"])
    for path in context.plan["path_order"][name]:
        if path != "graph" and context.lease is not None:
            raise ValueError("Eager F+B cannot retain graph pool")
        route, lease = make_route(context, name, "fb", samples, path)
        if lease is not None:
            setup = {
                **lease.setup,
                "allocated_after_capture": torch.cuda.memory_allocated(),
                "reserved_after_capture": torch.cuda.memory_reserved(),
            }
            numeric = fb_numeric(context, name, route, output)
        rows = []
        with (output / f"{path}.jsonl").open("x") as stream:
            for index in range(25):
                context.budget()
                core.zero_grad(set_to_none=True)
                assert_gradients_cleared(core)
                set_seed(42, cuda=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = task_loss(context, name, frame, route)
                loss.backward()
                torch.cuda.synchronize()
                wall = (time.perf_counter() - tick) * 1000
                peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in core.parameters()
                ):
                    raise ValueError("Nonfinite F+B cost evidence")
                row = {
                    "model": name,
                    "path": path,
                    "iteration": index,
                    "warmup": index < 5,
                    "wall_ms": wall,
                    "native_loss": float(loss.detach()),
                    "gradient_sample_sha256": gradient_fingerprint(core),
                    "gradients_cleared_before_forward": True,
                    "peak_allocated_bytes": peak,
                    "peak_reserved_bytes": reserved,
                    "optimizer_steps": 0,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                del loss
        core.zero_grad(set_to_none=True)
        costs[path] = summarize(rows)
        costs[path]["gradient_sample_repeat_exact"] = (
            len({r["gradient_sample_sha256"] for r in rows}) == 1
        )
        costs[path]["gradient_sample_repeat_is_descriptive"] = True
        if lease is not None:
            lease.assert_storage()
        release_route(context, route, lease)
        del route, lease
        context.budget()
    break_even = {
        path: (
            math.ceil(
                setup["factory_warmup_and_capture_seconds"]
                * 1000
                / (costs[path]["mean_ms"] - costs["graph"]["mean_ms"])
            )
            if costs[path]["mean_ms"] > costs["graph"]["mean_ms"]
            else None
        )
        for path in ("production_eager", "bridge_eager")
    }
    report = {
        "model": name,
        "costs": costs,
        "numeric": numeric,
        "numeric_gate": all(row["passed"] for row in numeric),
        "capture": setup,
        "capture_count": 1,
        "break_even_fb_invocations": break_even,
        "graph_pool_released": context.lease is None,
        "scope": "Original cached-raw native task F+B, frozen outer modules, eval cores with FP32 masters, backward outside AMP; no optimizer",
    }
    write_new(output / "RESULT.json", report)
    context.teacher.zero_grad(set_to_none=True)
    context.teacher.to(dtype=torch.bfloat16).requires_grad_(False).eval()
    return report


def decide(context, inference, fb):
    speed = {}
    limits = context.plan["decision_limits"]
    for name in ("native", "student"):
        speed[name] = []
        for item in context.plan["cost_inputs"]:
            ep = item["episode"]
            cost = inference[name]["costs"]
            graph = cost["graph"][ep]
            speed[name].append(
                {
                    "episode": ep,
                    "graph_production_p95_ratio": graph["p95_ms"]
                    / cost["production_eager"][ep]["p95_ms"],
                    "graph_bridge_p95_ratio": graph["p95_ms"] / cost["bridge_eager"][ep]["p95_ms"],
                    "production_speed_gate": graph["p95_ms"]
                    <= cost["production_eager"][ep]["p95_ms"]
                    * limits["max_graph_production_p95_ratio"],
                    "bridge_speed_gate": graph["p95_ms"]
                    <= cost["bridge_eager"][ep]["p95_ms"] * limits["max_graph_bridge_p95_ratio"],
                }
            )
    final = []
    for item in context.plan["cost_inputs"]:
        ep = item["episode"]
        n = inference["native"]["costs"]["graph"][ep]
        s = inference["student"]["costs"]["graph"][ep]
        final.append(
            {
                "episode": ep,
                "student_native_p95_ratio": s["p95_ms"] / n["p95_ms"],
                "speed_gate": s["p95_ms"]
                <= n["p95_ms"] * limits["max_student_native_graph_p95_ratio"],
                "allocated_gate": s["peak_allocated_bytes"]
                <= n["peak_allocated_bytes"] * limits["max_student_native_graph_allocated_ratio"],
            }
        )
    n, s = fb["native"]["costs"]["graph"], fb["student"]["costs"]["graph"]
    fb_gate = (
        s["p50_ms"] <= n["p50_ms"] * limits["max_student_native_graph_fb_p50_ratio"]
        and s["peak_allocated_bytes"] <= n["peak_allocated_bytes"]
    )
    numerical = all(row["numeric_gate"] for row in (*inference.values(), *fb.values()))
    return {
        "per_model_graph_speed": speed,
        "student_vs_native_graph_policy": final,
        "student_native_graph_fb_p50_ratio": s["p50_ms"] / n["p50_ms"],
        "student_native_graph_fb_gate": fb_gate,
        "numerical_gate": numerical,
        "candidate_execution_gate": numerical
        and fb_gate
        and all(r["speed_gate"] and r["allocated_gate"] for r in final),
        "all_graph_vs_eager_gates": all(
            r["production_speed_gate"] and r["bridge_speed_gate"]
            for rows in speed.values()
            for r in rows
        ),
        "old_training_gate_opened": False,
        "optimizer_steps": 0,
    }


def run(args, deadline):
    context = context_for(args, deadline)
    inference = {}
    fb = {}
    for name in context.plan["models"]:
        inference[name] = inference_phase(context, name)
    for name in context.plan["models"]:
        fb[name] = fb_phase(context, name)
    if context.lease is not None:
        raise ValueError("Graph lifecycle leaked into final integrity")
    context.teacher.zero_grad(set_to_none=True)
    context.teacher.to(dtype=torch.bfloat16).requires_grad_(False).eval()
    context.student.zero_grad(set_to_none=True)
    proto.assert_frozen(context.model)
    base_after = proto.content_hash(context.model.state_dict())
    student_after = proto.content_hash(context.student.state_dict())
    if (
        base_after != context.base_before
        or student_after != context.plan["student_initialization"]["content_sha256"]
    ):
        raise ValueError("Fixed weights changed during graph diagnosis")
    if (
        sha(context.plan["student_initialization"]["path"])
        != context.plan["student_initialization"]["sha256"]
    ):
        raise ValueError("Original initialization file changed")
    context.cache.assert_unchanged()
    context.budget()
    return {
        "status": "completed",
        "technical_validity": "passed",
        "measurements_completed": True,
        "plan_sha256": PLAN_SHA,
        "inference": inference,
        "fb": fb,
        "decision": decide(context, inference, fb),
        "optimizer_steps": 0,
        "training_authorized": False,
        "integrity": {
            "base_before": context.base_before,
            "base_after": base_after,
            "student_after": student_after,
            "cache_unchanged": True,
            "all_graphs_released": True,
            "stage_peak_allocated_bytes": context.stage_peak,
            "artifact_bytes": context.budget(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=1750)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1750:
        parser.error("Frozen wall bound exceeded")
    if args.output.resolve() != (args.artifact_root / "joint-graph-r1/output").resolve():
        parser.error("One canonical graph-run output required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = run(args, started + args.wall_seconds)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        result["elapsed_seconds"] = time.monotonic() - started
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
                "technical_validity": "not_established",
                "plan_sha256": PLAN_SHA,
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "optimizer_steps": 0,
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
