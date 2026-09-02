#!/usr/bin/env python3
"""Analyze a completed checkpoint after a terminal training evidence failure.

The analysis is intentionally conservative: it verifies the retained checkpoint
and reconstructs exact source sample counts from deterministic shard lengths and
the rank-local shard activation log.  It does not recreate in-memory NCCL or
per-step monitor evidence and therefore never marks the original Run technically
valid.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
from typing import Any

import numpy as np
from safetensors import safe_open


ACTIVATION = re.compile(r"Rank ([01]), Worker 0: Wait for shard (\d+) in dataset ([0-4])")
SOURCE_IDS = (
    "no_torque_100",
    "aug1p2mm_220",
    "aug5p4mm_153",
    "plain_300",
    "with_torque_33",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def shard_lengths(
    episode_lengths: list[int],
    *,
    action_horizon: int = 40,
    shard_size: int = 1024,
    episode_sampling_rate: float = 0.1,
    seed: int = 42,
) -> list[int]:
    """Reproduce ``ShardedSingleStepDataset.shard_dataset`` lengths."""
    rng = np.random.default_rng(seed)
    episode_order = rng.permutation(len(episode_lengths))
    effective = [max(0, length - action_horizon + 1) for length in episode_lengths]
    number_of_shards = int(np.ceil(sum(effective) / shard_size))
    result = np.zeros(number_of_shards, dtype=int)
    number_of_splits = int(1 / episode_sampling_rate)
    for episode_index in episode_order:
        step_indices = np.arange(effective[episode_index])
        rng.shuffle(step_indices)
        for split_index in range(number_of_splits):
            count = len(step_indices[split_index::number_of_splits])
            target = int(np.argmin(result))
            result[target] += count
    if not all(value > 0 for value in result):
        raise RuntimeError("reconstructed shard lengths contain an empty shard")
    return [int(value) for value in result]


def reconstruct_source_samples(
    training_log: str,
    lengths_by_dataset: list[list[int]],
    *,
    samples_per_rank: int,
    source_ids: tuple[str, ...] = SOURCE_IDS,
) -> dict[str, Any]:
    records: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for rank, shard, dataset in ACTIVATION.findall(training_log):
        records[int(rank)].append((int(dataset), int(shard)))

    source_totals = {source_id: 0 for source_id in source_ids}
    rank_summaries: dict[str, Any] = {}
    for rank in (0, 1):
        activations = records[rank]
        if not activations:
            raise RuntimeError(f"rank {rank} has no shard activation evidence")
        counts = [0] * len(source_ids)
        consumed = 0
        for dataset_index, shard_index in activations[:-1]:
            count = lengths_by_dataset[dataset_index][shard_index]
            counts[dataset_index] += count
            consumed += count
        final_dataset, final_shard = activations[-1]
        remainder = samples_per_rank - consumed
        final_capacity = lengths_by_dataset[final_dataset][final_shard]
        if not 0 < remainder <= final_capacity:
            raise RuntimeError(
                f"rank {rank} final shard remainder is invalid: "
                f"remainder={remainder}, capacity={final_capacity}"
            )
        counts[final_dataset] += remainder
        rank_summaries[str(rank)] = {
            "activated_shards": len(activations),
            "samples": dict(zip(source_ids, counts)),
            "total_samples": sum(counts),
            "final_shard": {
                "source_id": source_ids[final_dataset],
                "shard_index": final_shard,
                "consumed_samples": remainder,
                "capacity": final_capacity,
            },
        }
        for source_id, count in zip(source_ids, counts):
            source_totals[source_id] += count

    total = sum(source_totals.values())
    return {
        "method": (
            "completed logged shards use deterministic seed-42 shard lengths; "
            "the final active shard receives the rank remainder"
        ),
        "rank_counts": rank_summaries,
        "source_totals": source_totals,
        "source_proportions": {
            source_id: count / total for source_id, count in source_totals.items()
        },
        "total_samples": total,
        "all_sources_nonzero": all(count > 0 for count in source_totals.values()),
    }


def inspect_model(checkpoint: Path, index: dict[str, Any]) -> dict[str, Any]:
    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in index["weight_map"].items():
        expected_by_shard[shard].add(key)
    shard_summaries: dict[str, Any] = {}
    total_numel = 0
    all_finite = True
    all_keys_match = True
    for shard_name, expected_keys in sorted(expected_by_shard.items()):
        shard_path = checkpoint / shard_name
        observed_keys: set[str] = set()
        shard_numel = 0
        shard_finite = True
        dtypes: set[str] = set()
        with safe_open(shard_path, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                observed_keys.add(key)
                tensor = stream.get_tensor(key)
                shard_numel += tensor.numel()
                dtypes.add(str(tensor.dtype))
                if tensor.is_floating_point() and not bool(tensor.isfinite().all().item()):
                    shard_finite = False
        keys_match = observed_keys == expected_keys
        total_numel += shard_numel
        all_finite = all_finite and shard_finite
        all_keys_match = all_keys_match and keys_match
        shard_summaries[shard_name] = {
            "bytes": shard_path.stat().st_size,
            "tensor_count": len(observed_keys),
            "numel": shard_numel,
            "dtypes": sorted(dtypes),
            "all_finite": shard_finite,
            "index_keys_match": keys_match,
            "sha256": sha256_file(shard_path),
        }
    return {
        "weight_map_entries": len(index["weight_map"]),
        "total_numel": total_numel,
        "all_floating_tensors_finite": all_finite,
        "index_keys_match": all_keys_match,
        "shards": shard_summaries,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest_path = args.dataset_manifest.resolve(strict=True)
    training_stdout = args.training_stdout.resolve(strict=True)
    training_stderr = args.training_stderr.resolve(strict=True)
    supervisor_summary_path = args.supervisor_summary.resolve(strict=True)
    manifest = load_json(manifest_path)
    trainer_state = load_json(checkpoint / "trainer_state.json")
    model_config = load_json(checkpoint / "config.json")
    model_index = load_json(checkpoint / "model.safetensors.index.json")
    supervisor_summary = load_json(supervisor_summary_path)

    if trainer_state.get("global_step") != args.max_steps:
        raise RuntimeError("checkpoint trainer_state did not reach max_steps")
    if trainer_state.get("max_steps") != args.max_steps:
        raise RuntimeError("checkpoint trainer_state max_steps mismatch")
    if model_config.get("action_horizon") != 40:
        raise RuntimeError("checkpoint action_horizon is not 40")
    if supervisor_summary.get("technical_valid") is not False:
        raise RuntimeError("analysis expects a fail-closed supervisor summary")

    log_history = [entry for entry in trainer_state["log_history"] if "loss" in entry]
    logged_values = [
        value
        for entry in log_history
        for key, value in entry.items()
        if key in {"loss", "grad_norm", "learning_rate"} and isinstance(value, (int, float))
    ]
    if not all(math.isfinite(value) for value in logged_values):
        raise RuntimeError("trainer_state contains a non-finite logged value")

    lengths_by_dataset: list[list[int]] = []
    source_ids: list[str] = []
    for entry in manifest["datasets"]:
        dataset_path = Path(entry["dataset_path"])
        episodes = [
            json.loads(line)["length"]
            for line in (dataset_path / "meta/episodes.jsonl").read_text().splitlines()
            if line
        ]
        lengths_by_dataset.append(shard_lengths(episodes))
        source_ids.append(load_json(Path(entry["adapter_config_path"]))["source_id"])
    if tuple(source_ids) != SOURCE_IDS:
        raise RuntimeError(f"unexpected source order: {source_ids}")
    samples = reconstruct_source_samples(
        training_stdout.read_text(encoding="utf-8"),
        lengths_by_dataset,
        samples_per_rank=args.max_steps * args.global_batch_size // args.world_size,
    )
    if samples["total_samples"] != args.max_steps * args.global_batch_size:
        raise RuntimeError("reconstructed sample denominator mismatch")

    model = inspect_model(checkpoint, model_index)
    required_small = (
        "config.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
        "trainer_state.json",
    )
    optimizer_files = sorted(
        (checkpoint / f"global_step{args.max_steps}").glob(
            "bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"
        )
    )
    mp_model = checkpoint / f"global_step{args.max_steps}/mp_rank_00_model_states.pt"
    checkpoint_complete = (
        all((checkpoint / name).is_file() for name in required_small)
        and len(optimizer_files) == args.world_size
        and mp_model.is_file()
        and model["all_floating_tensors_finite"]
        and model["index_keys_match"]
    )

    hashes = {
        "dataset_manifest": sha256_file(manifest_path),
        "supervisor_summary": sha256_file(supervisor_summary_path),
        "training_stdout": sha256_file(training_stdout),
        "training_stderr": sha256_file(training_stderr),
        **{name: sha256_file(checkpoint / name) for name in required_small},
        **{
            path.relative_to(checkpoint).as_posix(): sha256_file(path)
            for path in (*optimizer_files, mp_model)
        },
    }
    return {
        "schema_version": "g1-all-lerobot-h40-checkpoint-analysis-v1",
        "analysis_succeeded": True,
        "original_run_execution": "failed during posttrain evidence collection",
        "original_failure": (
            "the source worktree absolute path moved before record_posttrain reread "
            "gr00t/configs/deepspeed/zero2_config.json"
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "complete": checkpoint_complete,
            "usable_model_root_candidate": checkpoint_complete,
            "global_step": trainer_state["global_step"],
            "max_steps": trainer_state["max_steps"],
            "epoch": trainer_state.get("epoch"),
            "action_horizon": model_config["action_horizon"],
            "logged_loss_entries": len(log_history),
            "logged_loss_grad_lr_all_finite": True,
            "model": model,
            "optimizer_rank_files": [
                path.relative_to(checkpoint).as_posix() for path in optimizer_files
            ],
        },
        "source_sampling": samples,
        "hashes": hashes,
        "technical_valid": False,
        "technical_valid_reason": (
            "rank posttrain evidence is absent; in-memory per-step finite monitor and "
            "NCCL terminal trace cannot be recovered postmortem"
        ),
        "scientific_outcome": "not_evaluated",
        "limitations": [
            "2,000 trainer log points are finite but are not the missing 20,000-point monitor",
            "source counts are reconstructed from deterministic shard lengths and activation logs",
            "no evaluator/native task result is inferred from checkpoint completion",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-stdout", type=Path, required=True)
    parser.add_argument("--training-stderr", type=Path, required=True)
    parser.add_argument("--supervisor-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args(argv)
    for name in (
        "checkpoint",
        "dataset_manifest",
        "training_stdout",
        "training_stderr",
        "supervisor_summary",
        "output",
    ):
        if not getattr(args, name).is_absolute():
            parser.error(f"{name.replace('_', '-')} must be absolute")
    if args.output.exists():
        parser.error("output must not already exist")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = analyze(arguments)
    atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
