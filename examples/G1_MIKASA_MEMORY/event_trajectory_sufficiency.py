#!/usr/bin/env python3
"""Three-stage CPU runner for the frozen event-trajectory sufficiency audit."""

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

from gr00t.experimental.mikasa_event_trajectory import (  # noqa: E402
    EventCache,
    FrozenMapping,
    build_cache,
    freeze_mapping,
    gates,
    score_mapping,
    stable_top64_token,
)
from gr00t.experimental.mikasa_native_rgb_memory_runtime import (  # noqa: E402
    ShardedMemoryCorpus,
    canonical_json_bytes,
    write_new_json,
)
import numpy as np  # noqa: E402


CORPUS_SHA = "3ed061ffdace8f9189cef272b283e582106481458c2088ad1ec0b824b443af99"


def _read(path: Path, expected_sha: str) -> tuple[dict, dict]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path.is_symlink() or digest != expected_sha:
        raise ValueError("frozen result identity changed")
    value = json.loads(raw)
    if value.get("status") != "passed":
        raise ValueError("frozen result did not pass")
    return value, {"path": str(path.resolve()), "bytes": len(raw), "sha256": digest}


def reject_path(path: Path | None) -> None:
    if path is not None and any(part.lower() in {"dev", "confirmation"} for part in path.parts):
        raise PermissionError("dev and confirmation paths are forbidden")


def verify_corpus(corpus: ShardedMemoryCorpus, result: dict, identity: dict) -> dict:
    root = Path(identity["path"]).parent.resolve()
    if corpus.root.resolve() != root:
        raise ValueError("corpus root changed")
    records = {
        record["group_id"]: record
        for record in result["group_records"]
        if record["split"] == "train"
    }
    if tuple(sorted(records)) != corpus.group_ids:
        raise ValueError("corpus train group set changed")
    for group_id, shard in corpus._by_group.items():
        record = records[group_id]
        for key, material in (
            ("manifest", shard.manifest),
            ("model_inputs", shard.model_input),
            ("privileged_targets", shard.privileged_target),
        ):
            if record[key] != {
                "name": material.path.name,
                "bytes": material.bytes,
                "sha256": material.sha256,
            }:
                raise ValueError(f"corpus {key} identity changed")
    return {"root": str(root), "groups": 24, "target_bytes_opened": False}


def event_cache_from_result(root: Path, result: dict, corpus: ShardedMemoryCorpus) -> EventCache:
    return EventCache.discover(root, result, corpus)


def smoke() -> dict:
    previous = np.zeros((128, 128, 3), np.uint8)
    current = previous.copy()
    current[10, 20] = 255
    token, valid = stable_top64_token(previous, current)
    return {"status": "passed", "token": token.tolist(), "valid": valid}


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--cache", action="store_true")
    mode.add_argument("--map", action="store_true")
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--corpus-result", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-result", type=Path)
    parser.add_argument("--expected-cache-sha256")
    parser.add_argument("--mapping-root", type=Path)
    parser.add_argument("--mapping-result", type=Path)
    parser.add_argument("--expected-mapping-sha256")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return 0
    for value in vars(args).values():
        if isinstance(value, Path):
            reject_path(value)
    if args.corpus_root is None or args.corpus_result is None or args.output_root is None:
        parser.error("formal stages require corpus root/result and output root")
    corpus_result, corpus_identity = _read(args.corpus_result, CORPUS_SHA)
    corpus = ShardedMemoryCorpus.discover(
        args.corpus_root, "train", verify_privileged_targets=False
    )
    corpus_binding = verify_corpus(corpus, corpus_result, corpus_identity)
    if args.cache:
        deadline = started + 1800
        built = build_cache(corpus, args.output_root, deadline)
        result = {
            "status": "passed",
            "mode": "event_trajectory_cache",
            "cache_root": str(args.output_root.resolve()),
            "group_records": built["records"],
            "corpus": corpus_identity,
            "corpus_binding": corpus_binding,
            "targets_opened": False,
            "dev_opened": False,
            "confirmation_opened": False,
        }
        limit = 512 * 1024 * 1024
    elif args.map:
        deadline = started + 600
        if (
            args.cache_root is None
            or args.cache_result is None
            or args.expected_cache_sha256 is None
        ):
            parser.error("mapping requires sealed cache identity")
        cache_result, cache_identity = _read(args.cache_result, args.expected_cache_sha256)
        cache = event_cache_from_result(args.cache_root, cache_result, corpus)
        built = freeze_mapping(cache, args.output_root, deadline)
        result = {
            "status": "passed",
            "mode": "event_trajectory_mapping",
            "mapping_root": str(args.output_root.resolve()),
            "materials": built["materials"],
            "cache": cache_identity,
            "targets_opened": False,
            "dev_opened": False,
            "confirmation_opened": False,
        }
        limit = 256 * 1024 * 1024
    else:
        deadline = started + 600
        if (
            args.mapping_root is None
            or args.mapping_result is None
            or args.expected_mapping_sha256 is None
        ):
            parser.error("scoring requires sealed mapping identity")
        mapping_result, mapping_identity = _read(args.mapping_result, args.expected_mapping_sha256)
        mapping = FrozenMapping.discover(args.mapping_root, mapping_result, corpus)
        evidence = score_mapping(mapping, corpus)
        gate_values = gates(evidence)
        result = {
            "status": "passed" if all(gate_values.values()) else "gate_failed",
            "mode": "event_trajectory_score",
            "mapping": mapping_identity,
            "evidence": evidence,
            "gates": gate_values,
            "target_open_order": ["fit_mean", "fit_neighbors", "calibration_once"],
            "dev_opened": False,
            "confirmation_opened": False,
        }
        limit = 256 * 1024 * 1024
    projected = sum(p.stat().st_size for p in args.output_root.rglob("*") if p.is_file()) + len(
        canonical_json_bytes(result)
    )
    if projected > limit:
        raise RuntimeError("stage artifact exceeds frozen limit")
    if time.monotonic() > deadline:
        raise TimeoutError("event-trajectory stage exceeded frozen deadline")
    write_new_json(args.output_root / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "mode": result["mode"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
