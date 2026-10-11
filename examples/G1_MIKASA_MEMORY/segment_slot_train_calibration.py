#!/usr/bin/env python3
"""Run the frozen fit18/cal6 segment-slot train calibration."""

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
from gr00t.experimental.mikasa_segment_slot_experiment import segment_slot_loss  # noqa: E402
from gr00t.experimental.mikasa_segment_slot_train_calibration import (  # noqa: E402
    run_train_calibration,
    validate_frozen_split,
)
import numpy as np  # noqa: E402
import torch  # noqa: E402


EXPERIMENT = "g1-mikasa-segment-slot-train-calibration-r1"
CORPUS_RESULT_SHA256 = "3ed061ffdace8f9189cef272b283e582106481458c2088ad1ec0b824b443af99"
FEATURE_CACHE_RESULT_SHA256 = "4d3db7d3a3310092887c94f7092b66858858df5c5ff68899f0f10c0af99f1f26"
FROZEN_SOURCE_SHA256 = {
    "gr00t/experimental/mikasa_native_rgb_memory_runtime.py": (
        "134d6ac78d65b76a6f1bda2c7eef54b2260c4aad3780c3ecd3a08de43bdbbc87"
    ),
    "gr00t/experimental/mikasa_segment_slot_bottleneck.py": (
        "1cc259d06be74957a55f00f93dcbb99ad927b15b82f404e64052609f321f6042"
    ),
    "gr00t/experimental/mikasa_segment_slot_experiment.py": (
        "5ccdbd3cf42c0b4e963927c3a30f2e77770f050ae05d76f63197342ca44d8102"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_immutable_slot_sources() -> dict[str, str]:
    observed = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = REPO / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"immutable slot source changed: {relative}")
        observed[relative] = digest
    return observed


def _read_result(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"frozen evidence identity changed: {path}")
    value = json.loads(raw)
    if value.get("status") != "passed":
        raise ValueError(f"frozen evidence did not pass: {path}")
    return value, {"path": str(path.resolve()), "bytes": len(raw), "sha256": digest}


def verify_frozen_evidence(
    corpus_result_path: Path,
    feature_cache_result_path: Path,
) -> dict[str, Any]:
    _, corpus = _read_result(corpus_result_path, CORPUS_RESULT_SHA256)
    cache_value, cache = _read_result(
        feature_cache_result_path,
        FEATURE_CACHE_RESULT_SHA256,
    )
    if (
        cache_value.get("targets_opened") is not False
        or cache_value.get("confirmation_materialized") is not False
    ):
        raise ValueError("feature cache crossed its frozen target/confirmation boundary")
    return {
        "corpus_result": corpus,
        "feature_cache_result": cache,
        "immutable_source_sha256": verify_immutable_slot_sources(),
    }


def reject_non_train_path(path: Path | None) -> None:
    if path is None:
        return
    forbidden = {"dev", "confirmation"}
    if any(part.lower() in forbidden for part in path.parts):
        raise PermissionError("original dev and confirmation paths are forbidden")


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device != torch.device("cuda:0"):
        raise ValueError("formal train-calibration is frozen to cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def synthetic_cpu_smoke() -> dict[str, Any]:
    temporal = torch.rand(
        2,
        130,
        2,
        8,
        8,
        9,
        generator=torch.Generator().manual_seed(230923),
    )
    model = MikasaSegmentSlotModel(seed=230923)
    output = model.forward_slot_maps(model.pool_segments(temporal))
    shapes = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
    checkpoints = torch.randn(2, 2, 12, 2, generator=torch.Generator().manual_seed(230923))
    stats = TargetStats(np.zeros(48, np.float32), np.ones(48, np.float32))
    total, checkpoint_mse, shape_ce = segment_slot_loss(output, shapes, checkpoints, stats)
    total.backward()
    return {
        "status": "passed",
        "split": validate_frozen_split(
            (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 16, 18, 19, 20, 22, 23, 24, 26, 28, 31, 33, 34, 35)
        ),
        "shape": list(output.shape_logits.shape),
        "checkpoint_shape": list(output.normalized_checkpoints.shape),
        "loss": {
            "total": float(total.detach()),
            "checkpoint_mse": float(checkpoint_mse.detach()),
            "shape_ce": float(shape_ce.detach()),
        },
        "finite": bool(torch.isfinite(total)),
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
    mode.add_argument("--train-calibration", action="store_true")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--corpus-result", type=Path)
    parser.add_argument("--feature-cache-result", type=Path)
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
        parser.error("formal run requires corpus/cache roots, evidence receipts and output root")
    for path in required:
        reject_non_train_path(path)
    provenance = verify_frozen_evidence(args.corpus_result, args.feature_cache_result)
    device = _device(args.device)
    result = run_train_calibration(
        ShardedMemoryCorpus.discover(args.corpus_root, "train"),
        FeatureCacheCorpus.discover(args.feature_cache_root, "train"),
        args.output_root,
        device=device,
        provenance=provenance,
    )
    print(json.dumps({"status": result["status"], "mode": result["mode"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
