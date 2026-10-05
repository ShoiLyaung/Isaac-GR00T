"""Independent Yuqi condition-sensitivity diagnostics and one-shot GPU profile.

This does not alter training, checkpoint weights, or model architecture. Raw VLM
features are cached on CPU only while diagnosing a frozen eval-mode checkpoint.
Interventions act on mixed image/text features *before* action-head vlln and
vl_self_attention. They do not identify a purely visual causal effect.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time
from types import SimpleNamespace

from contract import sha, tensor_identity, write_new
import numpy as np
from run_train_eval import (
    METRICS,
    NativeAnchors,
    load_plan,
    physical_metrics,
    read,
    source_identity,
    strict_reload,
)
from runner_core import (
    NOISE_SEEDS,
    aggregate_metrics,
    deterministic_seed,
    masked_flow,
    preserve_eval_state,
    set_seed,
)
import torch
from transformers.feature_extraction_utils import BatchFeature


COHORTS = ("train_probe", "seen_probe")
VARIANTS = ("normal", "zero_mixed_vl", "cyclic_mixed_vl")
ACTION_KEYS = ("state", "action", "action_mask", "embodiment_id")
RAW_KEYS = ("backbone_features", "backbone_attention_mask", "image_mask")


def clone_features(values, device):
    # Native action_head mutates BatchFeature entries; every invocation owns its container.
    return BatchFeature(data={key: value.to(device).clone() for key, value in values.items()})


def condition_view(records, index, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    donor = (index + 1) % len(records) if variant == "cyclic_mixed_vl" else index
    values = {key: value.clone() for key, value in records[donor]["condition"].items()}
    if variant == "zero_mixed_vl":
        values["backbone_features"].zero_()
    if values["backbone_features"].shape[:2] != values["backbone_attention_mask"].shape:
        raise ValueError("Condition features and mask must move together")
    if values["image_mask"].shape != values["backbone_attention_mask"].shape:
        raise ValueError("Image/text mask must move with its condition")
    return values, donor


def reference_rows(root):
    output = {}
    identities = {}
    for cohort in COHORTS:
        directory = root / f"step5000_{cohort}"
        rows = read(directory / "details.json")
        predictions = np.load(directory / "normalized_predictions.npy")
        if len(rows) != 64 or predictions.shape != (64, 40, 13):
            raise ValueError(f"Expected final fixed16 x four noise seeds: {directory}")
        for row, prediction in zip(rows, predictions):
            key = (cohort, row["episode"], row["anchor"], row["seed"])
            if key in output:
                raise ValueError(f"Duplicate reference key: {key}")
            output[key] = (row, prediction)
        identities[cohort] = {
            "directory": str(directory.resolve()),
            "details_sha256": sha(directory / "details.json"),
            "prediction_sha256": sha(directory / "normalized_predictions.npy"),
        }
    return output, identities


def freeze_conditions(model, source, pairs):
    records = []
    for episode, anchor in pairs:
        batch, raw = source.batch([(episode, anchor)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_inputs, action_inputs = model.prepare_input(dict(batch))
            condition = model.backbone(backbone_inputs)
        records.append(
            {
                "episode": int(episode),
                "anchor": int(anchor),
                "physical": raw[0],
                "condition": {key: condition[key].detach().cpu() for key in RAW_KEYS},
                "action": {key: action_inputs[key].detach().cpu() for key in ACTION_KEYS},
            }
        )
    return records


def run_head(model, condition, action, paired_seed):
    set_seed(paired_seed, cuda=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.action_head(
            clone_features(condition, model.device), clone_features(action, model.device)
        )
        flow = float(masked_flow(result["action_loss"], result["action_mask"])[0])
    del result
    observation = {
        key: value for key, value in action.items() if key not in ("action", "action_mask")
    }
    set_seed(paired_seed, cuda=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.action_head.get_action(
            clone_features(condition, model.device), clone_features(observation, model.device)
        )["action_pred"]
    if tuple(prediction.shape) != (1, 40, 13) or not torch.isfinite(prediction).all():
        raise ValueError("Expected finite compact13 H40 diagnostic action")
    return flow, prediction[0].float().cpu().numpy()


def interventions(args, model, source, anchors):
    references, reference_identity = reference_rows(args.reference_probes)
    rows, predictions = [], []
    started = time.monotonic()
    normal_parity = {"checked_outputs": 0, "max_abs_prediction": 0.0, "max_abs_flow": 0.0}
    with preserve_eval_state(model, source.processor), torch.no_grad():
        for cohort in COHORTS:
            records = freeze_conditions(model, source, anchors[cohort])
            for index, record in enumerate(records):
                for noise_seed in NOISE_SEEDS:
                    if time.monotonic() - started > args.wall_seconds:
                        raise TimeoutError("Condition diagnostic wall budget exceeded")
                    episode, anchor = record["episode"], record["anchor"]
                    paired_seed = deterministic_seed("diagnostic", episode, anchor, noise_seed)
                    normal = None
                    for variant in VARIANTS:
                        condition, donor_index = condition_view(records, index, variant)
                        flow, prediction = run_head(model, condition, record["action"], paired_seed)
                        if variant == "normal":
                            normal = prediction
                            reference, reference_prediction = references[
                                (cohort, episode, anchor, noise_seed)
                            ]
                            action_delta = float(np.max(np.abs(prediction - reference_prediction)))
                            flow_delta = abs(flow - reference["flow_loss"])
                            normal_parity["checked_outputs"] += 1
                            normal_parity["max_abs_prediction"] = max(
                                normal_parity["max_abs_prediction"], action_delta
                            )
                            normal_parity["max_abs_flow"] = max(
                                normal_parity["max_abs_flow"], flow_delta
                            )
                            if action_delta != 0 or flow_delta != 0:
                                raise ValueError(
                                    f"Normal cached-condition/main-eval parity failed: {(cohort, episode, anchor, noise_seed)}, action={action_delta}, flow={flow_delta}"
                                )
                        donor = records[donor_index]
                        row = {
                            "cohort": cohort,
                            "variant": variant,
                            "episode": episode,
                            "anchor": anchor,
                            "seed": noise_seed,
                            "paired_rng_seed": paired_seed,
                            "flow_loss": flow,
                            "active_mask_sum": 208,
                            "condition_episode": donor["episode"],
                            "condition_anchor": donor["anchor"],
                            "effective_vl_tokens": int(condition["backbone_attention_mask"].sum()),
                            "action_rms_change_active16_normalized": float(
                                np.sqrt(
                                    np.mean((prediction[:16].astype(np.float64) - normal[:16]) ** 2)
                                )
                            ),
                            **physical_metrics(prediction, record["physical"], source.processor),
                        }
                        rows.append(row)
                        predictions.append(prediction)
                print(json.dumps({"cohort": cohort, "anchors_done": index + 1}), flush=True)
    summaries = {}
    metric_names = (*METRICS, "action_rms_change_active16_normalized")
    for cohort in COHORTS:
        summaries[cohort] = {}
        for variant in VARIANTS:
            selected = [
                row for row in rows if row["cohort"] == cohort and row["variant"] == variant
            ]
            per_episode, metrics = aggregate_metrics(selected, metric_names)
            summaries[cohort][variant] = {"metrics": metrics, "per_episode": per_episode}
    write_new(args.output / "condition_details.json", rows)
    np.save(args.output / "condition_normalized_predictions.npy", np.stack(predictions))
    write_new(
        args.output / "CONDITION_DIAGNOSTIC.json",
        {
            "status": "completed",
            "rows": len(rows),
            "cohorts": summaries,
            "normal_parity": normal_parity,
            "reference_final_probes": reference_identity,
            "elapsed_seconds": time.monotonic() - started,
            "intervention_boundary": "raw mixed image/text VLM output, before vlln and vl_self_attention",
            "zero_semantics": "all raw feature values zero; original masks retained; downstream learned biases retained",
            "cyclic_semantics": "donor=(anchor_index+1)%16 within each frozen cohort; features and both masks move together",
            "fixed": "target anchor state, physical/normalized action target, embodiment, noise/time seed; eval mode",
            "aggregation": "noise seed -> anchor -> episode, episodes equal-weight within cohort",
            "claim_scope": "mixed multimodal-condition sensitivity only; not pure-vision causality or proof of an SSM bottleneck",
            "predictions_sha256": sha(args.output / "condition_normalized_predictions.npy"),
        },
    )


@contextmanager
def module_profile_hooks(model):
    """Measure observed calls without modifying implementation or architecture."""
    handles, records = [], []
    pending = defaultdict(list)
    modules = dict(model.named_modules())
    targets = {"backbone": "vl_backbone", "action_head.model": "action_backbone"}
    for name in modules:
        if name.startswith("action_head.model.") and name.endswith(".ff"):
            targets[name] = "head_ffn"
        if name.startswith("action_head.model.") and name.endswith(".mixer.mamba2"):
            targets[name] = "mamba2_forward_scan"
    ffn_names = [name for name, group in targets.items() if group == "head_ffn"]
    macs = {"observed_linear_macs": 0}

    def make_pre(name, group):
        def pre(module, inputs):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            annotation = torch.profiler.record_function(f"yuqi::{group}::{name}")
            annotation.__enter__()
            start.record()
            shape = (
                list(inputs[0].shape) if inputs and isinstance(inputs[0], torch.Tensor) else None
            )
            pending[name].append((start, stop, annotation, time.perf_counter(), shape))

        return pre

    def make_post(name, group):
        def post(module, inputs, output):
            start, stop, annotation, tick, shape = pending[name].pop()
            stop.record()
            cpu_ms = (time.perf_counter() - tick) * 1000
            annotation.__exit__(None, None, None)
            records.append(
                {
                    "module": name,
                    "group": group,
                    "input_shape": shape,
                    "host_enqueue_ms": cpu_ms,
                    "start": start,
                    "stop": stop,
                }
            )

        return post

    def linear_work(module, inputs, output):
        shape = inputs[0].shape
        macs["observed_linear_macs"] += (
            int(np.prod(shape[:-1])) * module.in_features * module.out_features
        )

    try:
        for name, group in targets.items():
            handles.append(modules[name].register_forward_pre_hook(make_pre(name, group)))
            handles.append(modules[name].register_forward_hook(make_post(name, group)))
        for name, module in modules.items():
            if isinstance(module, torch.nn.Linear) and any(
                name.startswith(prefix + ".") for prefix in ffn_names
            ):
                handles.append(module.register_forward_hook(linear_work))
        yield records, macs
    finally:
        for handle in handles:
            handle.remove()


def inside_policy(event):
    cursor = event
    while cursor is not None:
        if cursor.name == "yuqi::native_policy_inference":
            return True
        cursor = getattr(cursor, "cpu_parent", None)
    return False


def host_sync_evidence(profiler):
    groups = defaultdict(lambda: {"count": 0, "self_cpu_ms": 0.0, "inclusive_cpu_ms": 0.0})
    for event in profiler.events():
        if not inside_policy(event):
            continue
        if "Synchronize" not in event.name and event.name not in (
            "aten::item",
            "aten::_local_scalar_dense",
            "aten::tolist",
        ):
            continue
        value = groups[event.name]
        value["count"] += 1
        value["self_cpu_ms"] += event.self_cpu_time_total / 1000
        value["inclusive_cpu_ms"] += event.cpu_time_total / 1000
    return dict(groups)


def profile(args, model, source, anchors):
    deadline = time.monotonic() + args.wall_seconds
    episode, anchor = map(int, anchors["seen_probe"][0])
    paired_seed = deterministic_seed("diagnostic", episode, anchor, 42)
    with preserve_eval_state(model, source.processor), torch.no_grad():
        batch, _ = source.batch([(episode, anchor)])
        observation = {
            key: value for key, value in batch.items() if key not in ("action", "action_mask")
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_inputs, _ = model.prepare_input(dict(observation))
            raw = model.backbone(backbone_inputs)
        vl_tokens = int(raw["backbone_attention_mask"].sum())
        image_tokens = int((raw["image_mask"] & raw["backbone_attention_mask"]).sum())
        del raw, backbone_inputs

        def call():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.get_action(dict(observation))["action_pred"]

        # Isolated wall latency: no profiler, module hooks, or module-level CUDA events.
        timings = []
        for index in range(13):
            if time.monotonic() > deadline:
                raise TimeoutError("Profile diagnostic wall budget exceeded")
            set_seed(paired_seed, cuda=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            clean_prediction = call()
            torch.cuda.synchronize()
            if index >= 3:
                timings.append((time.perf_counter() - tick) * 1000)
        clean_identity = tensor_identity(clean_prediction)
        set_seed(paired_seed, cuda=True)
        torch.cuda.synchronize()
        with module_profile_hooks(model) as (records, macs):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                with torch.profiler.record_function("yuqi::native_policy_inference"):
                    instrumented_prediction = call()
        torch.cuda.synchronize()
        if not torch.equal(clean_prediction, instrumented_prediction):
            raise ValueError("Profiler/hooks changed the native action output")
    summaries = defaultdict(
        lambda: {
            "calls": 0,
            "cuda_interval_ms": 0.0,
            "host_enqueue_ms": 0.0,
            "token_rows": 0,
            "token_channels": 0,
        }
    )
    for record in records:
        record["cuda_interval_ms"] = record.pop("start").elapsed_time(record.pop("stop"))
        summary = summaries[record["group"]]
        summary["calls"] += 1
        summary["cuda_interval_ms"] += record["cuda_interval_ms"]
        summary["host_enqueue_ms"] += record["host_enqueue_ms"]
        if record["input_shape"] and record["group"] in ("head_ffn", "mamba2_forward_scan"):
            shape = record["input_shape"]
            summary["token_rows"] += int(np.prod(shape[:-1]))
            summary["token_channels"] += int(np.prod(shape))
    layers = model.config.diffusion_model_cfg["num_layers"]
    denoising_steps = model.config.num_inference_timesteps
    is_mamba = model.config.action_backbone == "conditional_mamba2"
    expected_ffn_rows = layers * denoising_steps * (vl_tokens + 41 if is_mamba else 41)
    if summaries["head_ffn"]["token_rows"] != expected_ffn_rows:
        raise ValueError(
            "Observed head FFN token work differs from the actual architecture contract"
        )
    scans = summaries.get("mamba2_forward_scan", {}).get("calls", 0)
    if scans != (layers * denoising_steps * 2 if is_mamba else 0):
        raise ValueError("Observed bidirectional official Mamba2 call count changed")
    sync_events = host_sync_evidence(prof)
    prof.export_chrome_trace(str(args.output / "native_policy_trace.json"))
    write_new(args.output / "profile_module_calls.json", records)
    write_new(
        args.output / "PROFILE.json",
        {
            "status": "completed",
            "episode": episode,
            "anchor": anchor,
            "seed": 42,
            "paired_rng_seed": paired_seed,
            "batch": 1,
            "cameras": 3,
            "effective_vl_tokens": vl_tokens,
            "effective_image_tokens": image_tokens,
            "effective_nonimage_tokens": vl_tokens - image_tokens,
            "state_plus_action_tokens": 41,
            "action_tokens": 40,
            "supervised_action_tokens": 16,
            "layers": layers,
            "denoising_steps": denoising_steps,
            "module_measurements": dict(summaries),
            "ffn_token_rows_expected": expected_ffn_rows,
            "ffn_linear_work": macs,
            "ffn_linear_flops_convention": "2 * observed Linear MACs; excludes activation, normalization and other operators",
            "official_mamba2_forward_scan_calls": scans,
            "scan_count_semantics": "official Mamba2 forward calls: one fresh scan per direction, includes its projection/convolution; not a pure SSM-kernel timer",
            "host_sync_events_inside_policy": sync_events,
            "runtime_sync_visibility": any("Synchronize" in name for name in sync_events),
            "runtime_sync_visibility_caution": "Missing runtime events do not establish absence of synchronization; inspect ATen events and the retained trace",
            "runtime_host_sync_self_cpu_ms": sum(
                row["self_cpu_ms"] for name, row in sync_events.items() if "Synchronize" in name
            ),
            "host_sync_scope": "events nested within native policy region only; excludes diagnostic/profiler final synchronization; nested ATen inclusive times are not added",
            "isolated_policy_latency_without_hooks_or_profiler": {
                "warmup_calls": 3,
                "measured_calls": 10,
                "milliseconds": timings,
                "p50_ms": float(np.median(timings)),
                "p95_ms": float(np.percentile(timings, 95)),
                "scope": "CPU prepared real B1 input through native get_action, including device input transfer, VLM and four-step denoising; excludes decode/processor",
            },
            "instrumented_output_exactly_equal": True,
            "action_identity": clean_identity,
            "timing_caution": "instrumented intervals are diagnostic, include launch/idle gaps, and nested module totals must not be added; use separate clean latency for service comparisons",
        },
    )


def selftest():
    # Different lengths make a mask/feature mismatch visible; original targets remain untouched.
    records = [
        {
            "condition": {
                "backbone_features": torch.full((1, n, 3), float(n)),
                "backbone_attention_mask": torch.ones(1, n, dtype=torch.bool),
                "image_mask": torch.arange(n)[None, :] % 2 == 0,
            },
            "action": {"state": torch.tensor([float(n)])},
        }
        for n in (2, 5)
    ]
    value, donor = condition_view(records, 0, "cyclic_mixed_vl")
    assert donor == 1 and value["backbone_features"].shape == (1, 5, 3)
    assert torch.equal(value["image_mask"], records[1]["condition"]["image_mask"])
    zero, _ = condition_view(records, 0, "zero_mixed_vl")
    assert torch.count_nonzero(zero["backbone_features"]) == 0
    assert torch.equal(zero["image_mask"], records[0]["condition"]["image_mask"])
    assert torch.all(records[0]["condition"]["backbone_features"] == 2)
    assert records[0]["action"]["state"].item() == 2
    region = SimpleNamespace(name="yuqi::native_policy_inference", cpu_parent=None)
    inside = SimpleNamespace(
        name="cudaStreamSynchronize",
        cpu_parent=region,
        self_cpu_time_total=4000,
        cpu_time_total=4000,
    )
    outside = SimpleNamespace(
        name="cudaDeviceSynchronize",
        cpu_parent=None,
        self_cpu_time_total=99000,
        cpu_time_total=99000,
    )
    evidence = host_sync_evidence(SimpleNamespace(events=lambda: [inside, outside]))
    assert set(evidence) == {"cudaStreamSynchronize"}
    assert evidence["cudaStreamSynchronize"]["self_cpu_ms"] == 4
    print(
        json.dumps(
            {
                "status": "passed",
                "cpu_intervention_contract": True,
                "host_sync_excludes_profiler_barrier": True,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("interventions", "profile", "selftest"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--reference-probes",
        type=Path,
        help="Training output's probes/ containing both step5000 cohorts",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wall-seconds", type=int, default=5400)
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
        return
    if args.checkpoint is None or args.plan is None or args.output is None:
        parser.error("--checkpoint, --plan and --output are required")
    if args.command == "interventions" and args.reference_probes is None:
        parser.error("interventions requires --reference-probes for strict normal-view parity")
    if not 0 < args.wall_seconds <= 5400:
        parser.error("Use a positive diagnostic budget no greater than 90 minutes")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        manifest, prepared, _, anchors = load_plan(args.plan)
        model, loading = strict_reload(args.checkpoint)
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        source = NativeAnchors(prepared)
        write_new(
            args.output / "IDENTITY.json",
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "config_sha256": sha(args.checkpoint / "config.json"),
                "strict_loading": loading,
                "source": source_identity(),
                "diagnostic_source_sha256": sha(__file__),
                "schedule_sha256": manifest["schedule_sha256"],
                "anchors_sha256": manifest["anchors_sha256"],
                "command": args.command,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(),
                "bf16": True,
                "tf32": True,
            },
        )
        {"interventions": interventions, "profile": profile}[args.command](
            args, model, source, anchors
        )
    except Exception as exc:
        write_new(
            args.output / "FAILURE.json",
            {"status": "failed", "exception": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
