#!/usr/bin/env python3
"""Run the frozen MIKASA segment-slot bottleneck diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (  # noqa: E402
    FeatureCacheCorpus,
    ShardedMemoryCorpus,
    TargetStats,
)
from gr00t.experimental.mikasa_segment_slot_bottleneck import MikasaSegmentSlotModel  # noqa: E402
from gr00t.experimental.mikasa_segment_slot_experiment import (  # noqa: E402
    run_matched_dev,
    run_overfit,
    segment_slot_loss,
)
import numpy as np  # noqa: E402
import torch  # noqa: E402


EXPERIMENT = "g1-mikasa-segment-slot-bottleneck-diagnosis-r1"
CORPUS_RESULT_SHA256 = "3ed061ffdace8f9189cef272b283e582106481458c2088ad1ec0b824b443af99"
FEATURE_CACHE_RESULT_SHA256 = "4d3db7d3a3310092887c94f7092b66858858df5c5ff68899f0f10c0af99f1f26"
DENSE_DEV_RESULT_SHA256 = "cbf022aed26497bb1da89fad13287b1784418a52ed80e0c602504b2ac3ffa3af"


def _read_frozen_result(
    path: Path,
    expected_sha256: str,
    *,
    expected_status: str,
    expected_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"frozen result identity changed: {path}")
    value = json.loads(raw)
    if value.get("status") != expected_status:
        raise ValueError(f"frozen result status changed: {path}")
    if expected_mode is not None and value.get("mode") != expected_mode:
        raise ValueError(f"frozen result mode changed: {path}")
    return value, {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": observed,
    }


def _reject_confirmation_path(path: Path | None) -> None:
    if path is not None and any("confirmation" in part.lower() for part in path.parts):
        raise PermissionError("confirmation paths are forbidden in this dev-only diagnosis")


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device != torch.device("cuda:0"):
        raise ValueError("formal segment-slot runs are frozen to cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def frozen_provenance(
    corpus_result_path: Path,
    cache_result_path: Path,
    dense_result_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _, corpus_identity = _read_frozen_result(
        corpus_result_path,
        CORPUS_RESULT_SHA256,
        expected_status="passed",
    )
    cache, cache_identity = _read_frozen_result(
        cache_result_path,
        FEATURE_CACHE_RESULT_SHA256,
        expected_status="passed",
    )
    if (
        cache.get("targets_opened") is not False
        or cache.get("confirmation_materialized") is not False
    ):
        raise ValueError("frozen feature cache crossed its target/confirmation boundary")
    provenance = {
        "corpus_result": corpus_identity,
        "feature_cache_result": cache_identity,
    }
    if dense_result_path is None:
        return provenance, None
    dense, dense_identity = _read_frozen_result(
        dense_result_path,
        DENSE_DEV_RESULT_SHA256,
        expected_status="gate_failed",
        expected_mode="dev_eval",
    )
    provenance["dense_dev_result"] = dense_identity
    return provenance, dense


def verify_overfit_receipt(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, identity = _read_frozen_result(
        path,
        expected_sha256,
        expected_status="passed",
        expected_mode="segment_slot_overfit",
    )
    if not isinstance(value.get("gates"), dict) or not all(value["gates"].values()):
        raise PermissionError("segment-slot overfit gates did not all pass")
    return value, identity


def synthetic_cpu_smoke() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(230923)
    temporal = torch.rand((2, 130, 2, 8, 8, 9), generator=generator)
    shapes = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
    checkpoints = torch.randn((2, 2, 12, 2), generator=generator)
    stats = TargetStats(np.zeros(48, np.float32), np.ones(48, np.float32))
    model = MikasaSegmentSlotModel(seed=230923)
    slot_maps = model.pool_segments(temporal)
    output = model.forward_slot_maps(slot_maps)
    swapped = model.forward_slot_maps(slot_maps.flip(1))
    total, mse, ce = segment_slot_loss(output, shapes, checkpoints, stats)
    total.backward()
    return {
        "status": "passed",
        "shape": list(output.shape_logits.shape),
        "checkpoint_shape": list(output.normalized_checkpoints.shape),
        "finite": bool(
            torch.isfinite(output.shape_logits).all()
            and torch.isfinite(output.normalized_checkpoints).all()
            and torch.isfinite(total)
        ),
        "loss": {
            "total": float(total.detach()),
            "checkpoint_mse": float(mse.detach()),
            "shape_ce": float(ce.detach()),
        },
        "swap_shape_exact": torch.equal(swapped.shape_logits, output.shape_logits.flip(1)),
        "swap_checkpoint_exact": torch.equal(
            swapped.normalized_checkpoints,
            output.normalized_checkpoints.flip(1),
        ),
        "mamba_loaded": "mamba_ssm" in sys.modules,
        "native_vla_loaded": any(
            name.startswith("gr00t.model.gr00t_n1d7") or name.startswith("gr00t.model.modules.dit")
            for name in sys.modules
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-smoke", action="store_true")
    mode.add_argument("--overfit", action="store_true")
    mode.add_argument("--matched-dev", action="store_true")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--corpus-result", type=Path)
    parser.add_argument("--feature-cache-result", type=Path)
    parser.add_argument("--dense-dev-result", type=Path)
    parser.add_argument("--overfit-result", type=Path)
    parser.add_argument("--expected-overfit-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.synthetic_smoke:
        print(json.dumps(synthetic_cpu_smoke(), indent=2, sort_keys=True))
        return 0
    required = (
        args.corpus_root,
        args.feature_cache_root,
        args.corpus_result,
        args.feature_cache_result,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("formal modes require corpus/cache roots, result receipts and output root")
    for path in (
        args.corpus_root,
        args.feature_cache_root,
        args.corpus_result,
        args.feature_cache_result,
        args.dense_dev_result,
        args.overfit_result,
        args.output_root,
    ):
        _reject_confirmation_path(path)
    device = _device(args.device)
    if args.overfit:
        provenance, _ = frozen_provenance(
            args.corpus_result,
            args.feature_cache_result,
            None,
        )
        corpus = ShardedMemoryCorpus.discover(args.corpus_root, "train")
        cache = FeatureCacheCorpus.discover(args.feature_cache_root, "train")
        result = run_overfit(
            corpus,
            cache,
            args.output_root,
            device=device,
            provenance=provenance,
        )
    else:
        if (
            args.dense_dev_result is None
            or args.overfit_result is None
            or args.expected_overfit_sha256 is None
        ):
            parser.error("matched-dev requires dense and overfit result identities")
        provenance, dense = frozen_provenance(
            args.corpus_result,
            args.feature_cache_result,
            args.dense_dev_result,
        )
        overfit, overfit_identity = verify_overfit_receipt(
            args.overfit_result,
            args.expected_overfit_sha256,
        )
        provenance["overfit_result"] = overfit_identity
        result = run_matched_dev(
            ShardedMemoryCorpus.discover(args.corpus_root, "train"),
            FeatureCacheCorpus.discover(args.feature_cache_root, "train"),
            ShardedMemoryCorpus.discover(args.corpus_root, "dev"),
            FeatureCacheCorpus.discover(args.feature_cache_root, "dev"),
            dense,
            overfit,
            args.output_root,
            device=device,
            provenance=provenance,
        )
    print(json.dumps({"status": result["status"], "mode": result["mode"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
