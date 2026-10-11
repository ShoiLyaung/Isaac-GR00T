#!/usr/bin/env python3
"""Run the frozen MIKASA native-RGB ordered-memory experiment stages."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OVERLAY_ROOT = Path(
    "/home/sl/research/icra27-artifacts/g1-arena-codec-streaming-backbone-r1/"
    "dependencies/python-overlay"
)
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))
if str(REPO) not in sys.path:
    sys.path.insert(1, str(REPO))

from gr00t.experimental.mikasa_native_rgb_memory import (  # noqa: E402
    MemoryBatch,
    MikasaNativeRGBMemoryModel,
    build_temporal_backbone,
    temporal_parameter_identity,
)
from gr00t.experimental.mikasa_native_rgb_memory_experiment import (  # noqa: E402
    LoadedTrainData,
    run_dev_evaluation,
    run_joint_confirmation,
    run_matched_training,
    run_overfit,
)
from gr00t.experimental.mikasa_native_rgb_memory_runtime import (  # noqa: E402
    FeatureCacheCorpus,
    GroupBalancedSchedule,
    ShardedMemoryCorpus,
    TargetStats,
    build_feature_cache,
    write_new_json,
)
import numpy as np  # noqa: E402
import torch  # noqa: E402


EXPERIMENT = "g1-mikasa-native-rgb-memory-backbone-r1"
MAMBA_VERSION = "2.3.2.post1"
COST_LENGTHS = (130, 260, 520, 1040)
FROZEN_SOURCE_PATHS = {
    relative: REPO / relative
    for relative in (
        "gr00t/experimental/arena_codec_streaming_backbone.py",
        "gr00t/experimental/mikasa_native_rgb_memory.py",
        "gr00t/experimental/mikasa_native_rgb_memory_runtime.py",
        "gr00t/experimental/mikasa_native_rgb_memory_experiment.py",
        "examples/G1_MIKASA_MEMORY/native_rgb_memory_backbone.py",
    )
}
FROZEN_SOURCE_PATHS.update(
    {
        "overlay/mamba2.py": OVERLAY_ROOT / "mamba_ssm/modules/mamba2.py",
        "overlay/selective_scan_cuda.so": (
            OVERLAY_ROOT / "selective_scan_cuda.cpython-312-x86_64-linux-gnu.so"
        ),
        "overlay/RECORD": OVERLAY_ROOT / "mamba_ssm-2.3.2.post1.dist-info/RECORD",
    }
)


def verify_mamba_overlay() -> dict[str, Any]:
    if not OVERLAY_ROOT.is_dir():
        raise FileNotFoundError(OVERLAY_ROOT)
    package = importlib.import_module("mamba_ssm")
    package_path = Path(package.__file__).resolve()
    if not package_path.is_relative_to(OVERLAY_ROOT.resolve()):
        raise RuntimeError(f"mamba_ssm did not resolve from the frozen overlay: {package_path}")
    if importlib.metadata.version("mamba-ssm") != MAMBA_VERSION:
        raise RuntimeError("mamba_ssm version changed")
    if importlib.util.find_spec("causal_conv1d") is not None or "causal_conv1d" in sys.modules:
        raise RuntimeError("causal_conv1d must be absent for this experiment")
    materials = {
        "mamba2.py": OVERLAY_ROOT / "mamba_ssm/modules/mamba2.py",
        "selective_scan_cuda.so": (
            OVERLAY_ROOT / "selective_scan_cuda.cpython-312-x86_64-linux-gnu.so"
        ),
        "RECORD": OVERLAY_ROOT / "mamba_ssm-2.3.2.post1.dist-info/RECORD",
    }
    if any(path.is_symlink() or not path.is_file() for path in materials.values()):
        raise RuntimeError("frozen Mamba overlay material is missing or not regular")
    return {
        "overlay_root": str(OVERLAY_ROOT.resolve()),
        "mamba_ssm_file": str(package_path),
        "mamba_ssm_version": MAMBA_VERSION,
        "causal_conv1d_absent": True,
        "material_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(materials.items())
        },
    }


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _verify_result_receipt(
    path: Path, expected_sha256: str, mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if len(expected_sha256) != 64 or observed != expected_sha256:
        raise PermissionError(f"{mode} result identity changed")
    value = json.loads(raw)
    if value.get("status") != "passed" or value.get("mode") != mode:
        raise PermissionError(f"{mode} result did not pass")
    return {"path": str(path.resolve()), "sha256": expected_sha256}, value


def verify_training_prerequisites(
    *,
    cost_path: Path,
    cost_sha256: str,
    preflight_path: Path,
    preflight_sha256: str,
    overfit_path: Path,
    overfit_sha256: str,
) -> dict[str, Any]:
    cost_identity, cost = _verify_result_receipt(cost_path, cost_sha256, "cost")
    preflight_identity, preflight = _verify_result_receipt(
        preflight_path, preflight_sha256, "batch24_preflight"
    )
    overfit_identity, overfit = _verify_result_receipt(overfit_path, overfit_sha256, "overfit")
    current_overlay = verify_mamba_overlay()
    identities = {
        "cost": cost_identity,
        "batch24_preflight": preflight_identity,
        "overfit": overfit_identity,
        "mamba_overlay": current_overlay,
    }
    if cost.get("three_blocks_passed") is not True or not all(cost.get("block_passes", [])):
        raise PermissionError("cost result did not pass all three resource blocks")
    if cost.get("overlay") != current_overlay:
        raise PermissionError("cost result Mamba overlay identity changed")
    if (
        preflight.get("steps") != 20
        or preflight.get("batch_size") != 24
        or preflight.get("finite") is not True
        or preflight.get("formal_checkpoint_written") is not False
    ):
        raise PermissionError("batch24 preflight contract changed")
    if preflight.get("overlay") != current_overlay:
        raise PermissionError("batch24 preflight Mamba overlay identity changed")
    if not isinstance(overfit.get("gates"), dict) or not all(overfit["gates"].values()):
        raise PermissionError("overfit gates did not all pass")
    if overfit.get("mamba_overlay") != current_overlay:
        raise PermissionError("overfit Mamba overlay identity changed")
    return identities


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


@torch.inference_mode()
def _measure_temporal(
    module: torch.nn.Module,
    descriptors: torch.Tensor,
    *,
    warmup: int,
    measurements: int,
) -> dict[str, Any]:
    if descriptors.device.type != "cuda":
        raise ValueError("formal cost measurement requires CUDA")
    for _ in range(warmup):
        output = module(descriptors)
    torch.cuda.synchronize(descriptors.device)
    host_ms: list[float] = []
    device_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats(descriptors.device)
    for _ in range(measurements):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter_ns()
        start_event.record()
        output = module(descriptors)
        end_event.record()
        end_event.synchronize()
        host_ms.append((time.perf_counter_ns() - started) / 1e6)
        device_ms.append(float(start_event.elapsed_time(end_event)))
    return {
        "host_p50_ms": statistics.median(host_ms),
        "host_p95_ms": _percentile(host_ms, 95),
        "device_p50_ms": statistics.median(device_ms),
        "device_p95_ms": _percentile(device_ms, 95),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(descriptors.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(descriptors.device),
        "finite": bool(torch.isfinite(output).all()),
        "shape": list(output.shape),
    }


def run_cost_probe(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    output_root: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if device.type != "cuda":
        raise ValueError("cost probe requires CUDA")
    overlay = verify_mamba_overlay()
    cache.validate_source(corpus)
    feature = cache.load_group(cache.group_ids[0])
    row = {key: value[:1] for key, value in feature.arrays.items()}
    base_model = MikasaNativeRGBMemoryModel("mamba_full", seed=230923).to(device=device)
    descriptors = base_model.encode_descriptors(
        MemoryBatch(
            temporal_grid=torch.from_numpy(row["temporal_grid"]),
            post_rgb_grid=torch.from_numpy(row["post_rgb_grid"]),
            post_proprio=torch.from_numpy(row["post_proprio"]),
        )
    ).detach()
    del base_model
    measurements: dict[str, dict[str, list[dict[str, Any]]]] = {
        "mamba": {},
        "gru": {},
        "transformer": {},
    }
    state_bytes: dict[str, int] = {}
    for kind in measurements:
        module = (
            build_temporal_backbone(kind, seed=230923).to(device=device, dtype=torch.float32).eval()
        )
        if kind in {"mamba", "gru"}:
            state_bytes[kind] = int(module.new_state(1).tensor_nbytes())
        for length in COST_LENGTHS:
            repeats = math.ceil(length / descriptors.shape[1])
            value = descriptors.repeat(1, repeats, 1)[:, :length].contiguous()
            blocks = []
            for _ in range(3):
                blocks.append(_measure_temporal(module, value, warmup=10, measurements=30))
            measurements[kind][str(length)] = blocks
        del module
        torch.cuda.empty_cache()
    mamba_p95 = [block["device_p95_ms"] for block in measurements["mamba"]["1040"]]
    transformer_p95 = [block["device_p95_ms"] for block in measurements["transformer"]["1040"]]
    mamba_peak = [block["peak_allocated_bytes"] for block in measurements["mamba"]["1040"]]
    transformer_peak = [
        block["peak_allocated_bytes"] for block in measurements["transformer"]["1040"]
    ]
    p95_ratios = [left / right for left, right in zip(mamba_p95, transformer_p95, strict=True)]
    peak_ratios = [left / right for left, right in zip(mamba_peak, transformer_peak, strict=True)]
    finite_by_block = [
        all(
            measurements[kind][str(length)][block]["finite"]
            for kind in measurements
            for length in COST_LENGTHS
        )
        for block in range(3)
    ]
    block_passes = [
        finite and (latency <= 0.80 or memory <= 0.67)
        for finite, latency, memory in zip(finite_by_block, p95_ratios, peak_ratios, strict=True)
    ]
    result = {
        "status": "passed" if all(block_passes) else "gate_failed",
        "mode": "cost",
        "overlay": overlay,
        "lengths": list(COST_LENGTHS),
        "measurements": measurements,
        "state_bytes": state_bytes,
        "mamba_transformer_p95_ratio_t1040": statistics.median(p95_ratios),
        "mamba_transformer_peak_allocated_ratio_t1040": statistics.median(peak_ratios),
        "three_blocks_passed": all(block_passes),
        "block_passes": block_passes,
        "finite_by_block": finite_by_block,
    }
    write_new_json(output_root / "RESULT.json", result)
    return result


def run_batch24_preflight(
    corpus: ShardedMemoryCorpus,
    cache: FeatureCacheCorpus,
    output_root: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    overlay = verify_mamba_overlay()
    data = LoadedTrainData(corpus, cache)
    stats = TargetStats.fit(corpus)
    started = time.monotonic()
    model = MikasaNativeRGBMemoryModel("mamba_full", seed=230923).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    schedule = GroupBalancedSchedule(data.group_ids, seed=230923)
    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(20):
        batch, targets = data.batch(schedule.batch(step))
        target = torch.from_numpy(
            np.concatenate(
                [group.target["ordered_checkpoints"].reshape(6, 48) for group in targets], axis=0
            ).astype(np.float32)
        ).to(device=device)
        prediction = model(batch)
        loss = torch.square(prediction - stats.normalize(target)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    result = {
        "status": "passed" if bool(np.isfinite(losses).all()) else "gate_failed",
        "mode": "batch24_preflight",
        "steps": 20,
        "batch_size": 24,
        "losses": losses,
        "finite": bool(np.isfinite(losses).all()),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "wall_seconds": time.monotonic() - started,
        "formal_checkpoint_written": False,
        "overlay": overlay,
    }
    write_new_json(output_root / "RESULT.json", result)
    return result


def synthetic_cpu_smoke() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(230923)
    temporal = torch.rand((2, 130, 2, 8, 8, 9), generator=generator)
    post = torch.rand((2, 2, 8, 8, 3), generator=generator)
    proprio = torch.rand((2, 7), generator=generator)
    model = MikasaNativeRGBMemoryModel("gru_full", seed=230923, frame_chunk_size=16)
    prediction = model(MemoryBatch(temporal, post, proprio))
    loss = prediction.square().mean()
    loss.backward()
    return {
        "status": "passed",
        "prediction_shape": list(prediction.shape),
        "finite": bool(torch.isfinite(prediction).all()),
        "temporal_identity": temporal_parameter_identity(model),
        "native_vla_imported": any(
            name.startswith("gr00t.model.gr00t_n1d7") or name.startswith("gr00t.model.modules.dit")
            for name in sys.modules
        ),
    }


def _print_result(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-smoke", action="store_true")
    mode.add_argument("--build-feature-cache", action="store_true")
    mode.add_argument("--cost", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--overfit", action="store_true")
    mode.add_argument("--matched-train", action="store_true")
    mode.add_argument("--eval-dev", action="store_true")
    mode.add_argument("--joint-confirmation", action="store_true")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--cost-result", type=Path)
    parser.add_argument("--expected-cost-sha256")
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--expected-preflight-sha256")
    parser.add_argument("--overfit-result", type=Path)
    parser.add_argument("--expected-overfit-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dev-receipt", type=Path)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--confirmation-spec", type=Path)
    args = parser.parse_args()
    if args.synthetic_smoke:
        _print_result(synthetic_cpu_smoke())
        return 0
    if args.build_feature_cache:
        if args.corpus_root is None or args.feature_cache_root is None:
            parser.error("feature-cache build requires --corpus-root and --feature-cache-root")
        records = {}
        for split in ("train", "dev"):
            corpus = ShardedMemoryCorpus.discover(
                args.corpus_root,
                split,
                verify_privileged_targets=False,
            )
            records[split] = build_feature_cache(corpus, args.feature_cache_root)
        _print_result({"status": "passed", "records": records})
        return 0
    if args.corpus_root is None or args.output_root is None:
        parser.error("selected mode requires --corpus-root and --output-root")
    device = _device(args.device)
    if device != torch.device("cuda:0"):
        parser.error("formal cost/training/evaluation modes are frozen to cuda:0")
    if args.joint_confirmation:
        if any(
            value is None
            for value in (
                args.checkpoint_root,
                args.dev_receipt,
                args.freeze_receipt,
                args.confirmation_spec,
            )
        ):
            parser.error("joint confirmation requires checkpoint/receipt/spec arguments")
        verify_mamba_overlay()
        result = run_joint_confirmation(
            dev_receipt_path=args.dev_receipt,
            freeze_receipt_path=args.freeze_receipt,
            checkpoint_root=args.checkpoint_root,
            source_paths=FROZEN_SOURCE_PATHS,
            confirmation_spec_path=args.confirmation_spec,
            confirmation_corpus_root=args.corpus_root,
            output_root=args.output_root,
            device=device,
        )
        _print_result({"status": result["status"]})
        return 0 if result["status"] == "passed" else 2
    if args.feature_cache_root is None:
        parser.error("selected mode requires --feature-cache-root")
    split = "dev" if args.eval_dev else "train"
    corpus = ShardedMemoryCorpus.discover(args.corpus_root, split)
    cache = FeatureCacheCorpus.discover(args.feature_cache_root, split)
    if args.cost:
        result = run_cost_probe(corpus, cache, args.output_root, device=device)
    elif args.preflight:
        result = run_batch24_preflight(corpus, cache, args.output_root, device=device)
    elif args.overfit:
        overlay = verify_mamba_overlay()
        result = run_overfit(
            corpus,
            cache,
            args.output_root,
            device=device,
            mamba_overlay=overlay,
        )
    elif args.matched_train:
        prerequisite_values = (
            args.cost_result,
            args.expected_cost_sha256,
            args.preflight_result,
            args.expected_preflight_sha256,
            args.overfit_result,
            args.expected_overfit_sha256,
        )
        if any(value is None for value in prerequisite_values):
            parser.error("matched training requires cost, preflight and overfit result paths/SHAs")
        prerequisites = verify_training_prerequisites(
            cost_path=args.cost_result,
            cost_sha256=args.expected_cost_sha256,
            preflight_path=args.preflight_result,
            preflight_sha256=args.expected_preflight_sha256,
            overfit_path=args.overfit_result,
            overfit_sha256=args.expected_overfit_sha256,
        )
        result = run_matched_training(
            corpus,
            cache,
            args.output_root,
            device=device,
            prerequisites=prerequisites,
        )
    elif args.eval_dev:
        if (
            args.checkpoint_root is None
            or args.cost_result is None
            or args.expected_cost_sha256 is None
            or args.confirmation_spec is None
        ):
            parser.error("dev evaluation requires --checkpoint-root and cost result path/SHA")
        verify_mamba_overlay()
        _verify_result_receipt(args.cost_result, args.expected_cost_sha256, "cost")
        result = run_dev_evaluation(
            corpus,
            cache,
            args.checkpoint_root,
            args.cost_result,
            args.output_root,
            device=device,
            source_paths=FROZEN_SOURCE_PATHS,
            confirmation_spec_path=args.confirmation_spec,
        )
    else:
        raise AssertionError("unreachable mode")
    _print_result({"status": result["status"], "mode": result["mode"]})
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
