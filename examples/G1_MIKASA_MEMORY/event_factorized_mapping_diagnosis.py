#!/usr/bin/env python3
"""Freeze and score factorized event-trajectory 1-NN mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gr00t.experimental.mikasa_event_factorized_mapping import (  # noqa: E402
    SCHEMA,
    FactorizedMapping,
    factorized_views,
    freeze_factorized_mapping,
    score_factorized,
    unchanged_gates,
)
from gr00t.experimental.mikasa_event_trajectory import EventCache  # noqa: E402
from gr00t.experimental.mikasa_native_rgb_memory_runtime import (  # noqa: E402
    ShardedMemoryCorpus,
    canonical_json_bytes,
    write_new_json,
)
import numpy as np  # noqa: E402


CORPUS_SHA = "3ed061ffdace8f9189cef272b283e582106481458c2088ad1ec0b824b443af99"
CACHE_SHA = "487d83a1c3f5b5ede8e0d3feaf1275cba64cbb9cd4f36453b007307c415f6898"
FUSED_MAP_SHA = "bf3b6da4ddb63cfadba995711bc45f7eb99e01ffbbffdf9fb41202d0d2fbe52f"
FUSED_SCORE_SHA = "b08b237c93d6a3f619823e84186fbf327f22f67e6e166bb9d5d16c0b8a2dd8ed"


def _read(path: Path, digest: str, status: str) -> tuple[dict, dict]:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if path.is_symlink() or observed != digest:
        raise ValueError("frozen result identity changed")
    value = json.loads(raw)
    if value.get("status") != status:
        raise ValueError("frozen result status changed")
    return value, {"path": str(path.resolve()), "bytes": len(raw), "sha256": observed}


def reject_path(path: Path | None) -> None:
    if path is not None and any(part.lower() in {"dev", "confirmation"} for part in path.parts):
        raise PermissionError("dev and confirmation paths are forbidden")


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--map", action="store_true")
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--corpus-result", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-result", type=Path)
    parser.add_argument("--fused-map-result", type=Path)
    parser.add_argument("--fused-score-result", type=Path)
    parser.add_argument("--mapping-root", type=Path)
    parser.add_argument("--mapping-result", type=Path)
    parser.add_argument("--expected-mapping-sha256")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.smoke:
        arrays = {
            "raw": np.zeros((6, 2, 2, 65, 3), np.float32),
            "canonical": np.zeros((6, 2, 2, 65, 2), np.float32),
            "valid": np.ones((6, 2, 2, 65), np.bool_),
            "post_proprio": np.zeros((6, 7), np.float32),
        }
        shape, checkpoint, _ = factorized_views(arrays, "ordered", 0)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "shape_dim": shape.shape[1],
                    "checkpoint_dim": checkpoint.shape[1],
                },
                sort_keys=True,
            )
        )
        return 0
    required = (
        args.corpus_root,
        args.corpus_result,
        args.cache_root,
        args.cache_result,
        args.fused_map_result,
        args.fused_score_result,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("formal stages require corpus/cache/prior receipts and output root")
    for value in vars(args).values():
        if isinstance(value, Path):
            reject_path(value)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    corpus_result, corpus_id = _read(args.corpus_result, CORPUS_SHA, "passed")
    cache_result, cache_id = _read(args.cache_result, CACHE_SHA, "passed")
    _, fused_map_id = _read(args.fused_map_result, FUSED_MAP_SHA, "passed")
    _, fused_score_id = _read(args.fused_score_result, FUSED_SCORE_SHA, "gate_failed")
    corpus = ShardedMemoryCorpus.discover(
        args.corpus_root, "train", verify_privileged_targets=False
    )
    cache = EventCache.discover(args.cache_root, cache_result, corpus)
    provenance = {
        "corpus": corpus_id,
        "cache": cache_id,
        "fused_map": fused_map_id,
        "fused_score": fused_score_id,
    }
    if args.map:
        built = freeze_factorized_mapping(cache, args.output_root, started + 600)
        result = {
            "status": "passed",
            "mode": "factorized_event_mapping",
            "schema": SCHEMA,
            "mapping_root": str(args.output_root.resolve()),
            "materials": built["materials"],
            "all_descriptors_distances_finite": built["all_descriptors_distances_finite"],
            "cache": cache_id,
            "provenance": provenance,
            "targets_opened": False,
            "dev_opened": False,
            "confirmation_opened": False,
        }
    else:
        if (
            args.mapping_root is None
            or args.mapping_result is None
            or args.expected_mapping_sha256 is None
        ):
            parser.error("score requires sealed factorized mapping")
        mapping_result, mapping_id = _read(
            args.mapping_result, args.expected_mapping_sha256, "passed"
        )
        if mapping_result.get("cache") != cache_id:
            raise ValueError("factorized mapping cache provenance changed")
        mapping = FactorizedMapping.discover(args.mapping_root, mapping_result)
        evidence = score_factorized(mapping, corpus)
        gate_values = unchanged_gates(evidence)
        result = {
            "status": "passed" if all(gate_values.values()) else "gate_failed",
            "mode": "factorized_event_score",
            "mapping": mapping_id,
            "provenance": provenance,
            "evidence": evidence,
            "gates": gate_values,
            "target_open_order": ["fit_mean", "fit_neighbors", "calibration_once"],
            "dev_opened": False,
            "confirmation_opened": False,
        }
    projected = sum(p.stat().st_size for p in args.output_root.rglob("*") if p.is_file()) + len(
        canonical_json_bytes(result)
    )
    if projected > 256 * 1024 * 1024:
        raise RuntimeError("factorized stage exceeds 256 MiB")
    if time.monotonic() > started + 600:
        raise TimeoutError("factorized stage exceeds 600 seconds")
    write_new_json(args.output_root / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "mode": result["mode"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
