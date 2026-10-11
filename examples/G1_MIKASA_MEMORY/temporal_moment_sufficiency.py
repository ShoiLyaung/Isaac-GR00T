#!/usr/bin/env python3
"""Build temporal-moment caches and run the matched sufficiency experiment."""

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

from gr00t.experimental.mikasa_native_rgb_memory_runtime import (  # noqa: E402
    FeatureCacheCorpus,
    ShardedMemoryCorpus,
    canonical_json_bytes,
    write_new_json,
)
from gr00t.experimental.mikasa_segment_slot_experiment import TRAIN_SEEDS  # noqa: E402
from gr00t.experimental.mikasa_temporal_moment_cache import (  # noqa: E402
    CONDITIONS,
    TemporalMomentCache,
    build_temporal_moment_caches,
    frozen_permutations,
    time_weights,
)
from gr00t.experimental.mikasa_temporal_moment_experiment import run_temporal_moment  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


CORPUS_SHA = "3ed061ffdace8f9189cef272b283e582106481458c2088ad1ec0b824b443af99"
CACHE_SHA = "4d3db7d3a3310092887c94f7092b66858858df5c5ff68899f0f10c0af99f1f26"
PRIOR_SHA = "c94ad05a2aeabb2f902d36a0b8b765099521fcdc871ead3eee553a7c4df5fb44"


def _read(path: Path, digest: str, status: str) -> tuple[dict, dict]:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if path.is_symlink() or observed != digest:
        raise ValueError("frozen evidence identity changed")
    value = json.loads(raw)
    if value.get("status") != status:
        raise ValueError("frozen evidence status changed")
    return value, {"path": str(path.resolve()), "bytes": len(raw), "sha256": observed}


def verify_source_cache_receipt(
    source_cache: FeatureCacheCorpus,
    result: dict,
    result_identity: dict,
) -> dict:
    result_root = Path(result_identity["path"]).parent.resolve()
    if source_cache.root.resolve() != result_root:
        raise ValueError("source cache root does not match frozen RESULT root")
    if (
        result.get("status") != "passed"
        or result.get("schema") != "g1-mikasa-native-rgb-memory-backbone-r1/feature-cache-result-v1"
        or result.get("targets_opened") is not False
        or result.get("confirmation_materialized") is not False
        or result.get("train_groups") != 24
    ):
        raise ValueError("source cache RESULT contract changed")
    train_records = [
        record for record in result.get("group_records", []) if record.get("split") == "train"
    ]
    by_group = {record.get("group_id"): record for record in train_records}
    if tuple(sorted(by_group)) != source_cache.group_ids:
        raise ValueError("source cache RESULT train group set changed")
    for group_id in source_cache.group_ids:
        record = by_group[group_id]
        feature = source_cache._records[group_id]
        manifest_path = source_cache.root / "train" / f"group_{group_id:03d}" / "MANIFEST.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("source cache manifest is missing or unsafe")
        manifest_raw = manifest_path.read_bytes()
        if record.get("feature") != {
            "file": feature.path.name,
            "bytes": feature.bytes,
            "sha256": feature.sha256,
        }:
            raise ValueError(f"source cache feature receipt changed for group {group_id}")
        if record.get("manifest") != {
            "bytes": len(manifest_raw),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }:
            raise ValueError(f"source cache manifest receipt changed for group {group_id}")
        if record.get("source_model_input_sha256") != source_cache.source_model_sha256[group_id]:
            raise ValueError(f"source cache source identity changed for group {group_id}")
    return {
        "root": str(result_root),
        "groups_verified": len(source_cache.group_ids),
        "materials_verified": 2 * len(source_cache.group_ids),
    }


def verify_corpus_receipt(
    corpus: ShardedMemoryCorpus,
    result: dict,
    result_identity: dict,
) -> dict:
    result_root = Path(result_identity["path"]).parent.resolve()
    if corpus.root.resolve() != result_root:
        raise ValueError("corpus root does not match frozen RESULT root")
    train_records = [
        record for record in result.get("group_records", []) if record.get("split") == "train"
    ]
    by_group = {record.get("group_id"): record for record in train_records}
    if tuple(sorted(by_group)) != corpus.group_ids:
        raise ValueError("corpus RESULT train group set changed")
    for group_id in corpus.group_ids:
        record = by_group[group_id]
        shard = corpus._by_group[group_id]
        if record.get("directory") != f"train/group_{group_id:03d}":
            raise ValueError(f"corpus directory receipt changed for group {group_id}")
        expected = {
            "manifest": {
                "name": shard.manifest.path.name,
                "bytes": shard.manifest.bytes,
                "sha256": shard.manifest.sha256,
            },
            "model_inputs": {
                "name": shard.model_input.path.name,
                "bytes": shard.model_input.bytes,
                "sha256": shard.model_input.sha256,
            },
            "privileged_targets": {
                "name": shard.privileged_target.path.name,
                "bytes": shard.privileged_target.bytes,
                "sha256": shard.privileged_target.sha256,
            },
        }
        for name, identity in expected.items():
            if record.get(name) != identity:
                raise ValueError(f"corpus {name} receipt changed for group {group_id}")
        target_path = shard.privileged_target.path
        if target_path.is_symlink() or not target_path.is_file():
            raise ValueError("declared privileged target path is missing or unsafe")
        if target_path.resolve().parent != shard.manifest.path.resolve().parent:
            raise ValueError("declared privileged target escapes group root")
    return {
        "root": str(result_root),
        "groups_verified": len(corpus.group_ids),
        "target_bytes_opened": False,
    }


def reject_path(path: Path | None) -> None:
    if path is not None and any(part.lower() in {"dev", "confirmation"} for part in path.parts):
        raise PermissionError("dev and confirmation paths are forbidden")


def smoke() -> dict:
    permutations = frozen_permutations()
    return {
        "status": "passed",
        "weights": [
            {
                "mean": float(time_weights(length).mean()),
                "rms": float(np.sqrt(np.mean(time_weights(length) ** 2))),
            }
            for length in (66, 64)
        ],
        "permutation_heads": [value[:8].tolist() for value in permutations],
        "mamba_loaded": "mamba_ssm" in sys.modules,
    }


def main() -> int:
    command_started = time.monotonic()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--build-cache", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--moment-cache-root", type=Path)
    parser.add_argument("--corpus-result", type=Path)
    parser.add_argument("--source-cache-result", type=Path)
    parser.add_argument("--moment-cache-result", type=Path)
    parser.add_argument("--expected-moment-cache-sha256")
    parser.add_argument("--prior-result", type=Path)
    parser.add_argument("--historical-checkpoint-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return 0
    for value in vars(args).values():
        if isinstance(value, Path):
            reject_path(value)
    if any(
        value is None
        for value in (
            args.corpus_root,
            args.source_cache_root,
            args.corpus_result,
            args.source_cache_result,
            args.output_root,
        )
    ):
        parser.error("formal modes require corpus/cache roots and receipts")
    corpus_value, corpus_id = _read(args.corpus_result, CORPUS_SHA, "passed")
    source_value, source_id = _read(args.source_cache_result, CACHE_SHA, "passed")
    if source_value.get("targets_opened") is not False:
        raise ValueError("source cache is not target blind")
    source_cache = FeatureCacheCorpus.discover(args.source_cache_root, "train")
    source_cache_binding = verify_source_cache_receipt(source_cache, source_value, source_id)
    if args.build_cache:
        corpus = ShardedMemoryCorpus.discover(
            args.corpus_root,
            "train",
            verify_privileged_targets=False,
        )
        corpus_binding = verify_corpus_receipt(corpus, corpus_value, corpus_id)
        built = build_temporal_moment_caches(source_cache, corpus, args.output_root)
        result = {
            "status": "passed",
            "mode": "temporal_moment_cache",
            "cache_root": str(args.output_root.resolve()),
            "conditions": list(CONDITIONS),
            "condition_records": built["records"],
            "permutation_identity": built["permutation_identity"],
            "permutation_sha256": built["permutation_sha256"],
            "corpus_result": corpus_id,
            "corpus_binding": corpus_binding,
            "source_cache_result": source_id,
            "source_cache_binding": source_cache_binding,
            "targets_opened": False,
            "original_dev_opened": False,
            "confirmation_opened": False,
        }
        projected = sum(p.stat().st_size for p in args.output_root.rglob("*") if p.is_file()) + len(
            canonical_json_bytes(result)
        )
        if projected > 512 * 1024 * 1024:
            raise RuntimeError("temporal-moment cache exceeds 512 MiB")
        if time.monotonic() > command_started + 600:
            raise TimeoutError("temporal-moment cache exceeded 600 seconds")
        write_new_json(args.output_root / "RESULT.json", result)
    else:
        corpus = ShardedMemoryCorpus.discover(
            args.corpus_root,
            "train",
            verify_privileged_targets=False,
        )
        corpus_binding = verify_corpus_receipt(corpus, corpus_value, corpus_id)
        if any(
            value is None
            for value in (
                args.moment_cache_root,
                args.moment_cache_result,
                args.expected_moment_cache_sha256,
                args.prior_result,
                args.historical_checkpoint_root,
            )
        ):
            parser.error("run mode requires moment cache, prior, and historical checkpoints")
        moment_result, moment_id = _read(
            args.moment_cache_result, args.expected_moment_cache_sha256, "passed"
        )
        prior, prior_id = _read(args.prior_result, PRIOR_SHA, "gate_failed")
        caches = {
            condition: TemporalMomentCache.discover(args.moment_cache_root, condition, source_cache)
            for condition in CONDITIONS
        }
        for cache in caches.values():
            cache.verify_receipt(moment_result, source_cache)
        checkpoints = {
            seed: args.historical_checkpoint_root / f"seed_{seed}" / "checkpoint_final.pt"
            for seed in TRAIN_SEEDS
        }
        result = run_temporal_moment(
            corpus,
            source_cache,
            caches,
            checkpoints,
            prior,
            args.output_root,
            device=torch.device("cuda:0"),
            deadline=command_started + 60 * 60,
            provenance={
                "corpus": corpus_id,
                "corpus_binding": corpus_binding,
                "source_cache": source_id,
                "source_cache_binding": source_cache_binding,
                "moment_cache": moment_id,
                "historical_result": prior_id,
            },
        )
    print(json.dumps({"status": result["status"], "mode": result["mode"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
