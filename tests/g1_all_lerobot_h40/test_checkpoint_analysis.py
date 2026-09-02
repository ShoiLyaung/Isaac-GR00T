from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "examples/G1_ALL_LEROBOT_H40/analyze_completed_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_reconstruct_source_samples_accounts_for_final_partial_shard():
    log = "\n".join(
        [
            "Rank 0, Worker 0: Wait for shard 0 in dataset 0 in 0.00 seconds",
            "Rank 0, Worker 0: Wait for shard 0 in dataset 1 in 0.00 seconds",
            "Rank 1, Worker 0: Wait for shard 0 in dataset 1 in 0.00 seconds",
            "Rank 1, Worker 0: Wait for shard 0 in dataset 0 in 0.00 seconds",
        ]
    )
    result = analysis.reconstruct_source_samples(
        log,
        [[6], [7], [1], [1], [1]],
        samples_per_rank=10,
        source_ids=("a", "b", "c", "d", "e"),
    )

    assert result["rank_counts"]["0"]["samples"] == {
        "a": 6,
        "b": 4,
        "c": 0,
        "d": 0,
        "e": 0,
    }
    assert result["rank_counts"]["1"]["samples"] == {
        "a": 3,
        "b": 7,
        "c": 0,
        "d": 0,
        "e": 0,
    }
    assert result["total_samples"] == 20


def test_shard_lengths_preserve_horizon_anchor_denominator():
    episodes = [45, 51, 63]
    lengths = analysis.shard_lengths(
        episodes,
        action_horizon=40,
        shard_size=8,
        episode_sampling_rate=0.5,
        seed=42,
    )

    assert sum(lengths) == sum(length - 40 + 1 for length in episodes)
    assert all(length > 0 for length in lengths)
