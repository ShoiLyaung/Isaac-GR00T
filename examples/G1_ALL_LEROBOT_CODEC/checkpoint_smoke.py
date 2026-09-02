#!/usr/bin/env python3
"""Load one causal Codec checkpoint through the public policy and request one action."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = "groot-main-causal-checkpoint-smoke-v1"
FIXTURE_MANIFEST_SHA256 = "5e4ec3590dd20fc767957bc1f7b74d8eaf60a766a45be6d3e4a1f8c55412b4b9"
EPISODE_UID = "cb927e3e88cfe78e4bfe72d42c0ea60fc23de5274478dcd9e9937be6b3e538c9"
SOURCE_KIND = "NATIVE_CANONICAL_DECODED_DERIVED"
CAMERA_ID = "cam_left_high"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _observation(step: Any, language_key: str) -> dict[str, Any]:
    return {
        "video": {
            key: np.asarray(value, dtype=np.uint8)[None, ...] for key, value in step.images.items()
        },
        "state": {
            key: np.asarray(value, dtype=np.float32)[None, ...]
            for key, value in step.states.items()
        },
        "language": {language_key: [[str(step.text)]]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--loader-view", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--episode-index", type=int, default=64)
    parser.add_argument("--observation-row", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    source_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        text=True,
    )
    if source_commit != args.expected_source_commit or dirty:
        raise RuntimeError("Checkpoint smoke requires the exact clean source commit")
    checkpoint = args.checkpoint.resolve(strict=True)
    fixture_root = args.fixture_root.resolve(strict=True)
    fixture_manifest = fixture_root / "fixture_manifest.json"
    if _sha256(fixture_manifest) != FIXTURE_MANIFEST_SHA256:
        raise RuntimeError("Causal fixture manifest identity drifted")

    from gr00t.data.codec_evidence.episode_bundle import EpisodeBundleKey
    from gr00t.data.codec_evidence.warehouse import WarehouseEpisodeBundleStore
    from gr00t.data.dataset.g1_all_lerobot_h40_adapter import G1AllLerobotH40EpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    import gr00t.model  # noqa: F401 - register local AutoModel/AutoProcessor classes
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from transformers import AutoProcessor

    tag = EmbodimentTag.NEW_EMBODIMENT
    cpu_processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    modalities = cpu_processor.get_modality_configs()[tag.value]
    loader = G1AllLerobotH40EpisodeLoader(
        args.loader_view,
        modalities,
        args.adapter_config,
        require_runtime_verified_fixed_state=True,
    )
    step = extract_step_data(
        loader[args.episode_index],
        args.observation_row,
        modalities,
        tag,
        allow_padding=False,
    )
    store = WarehouseEpisodeBundleStore(
        fixture_root,
        fixture_manifest_sha256=FIXTURE_MANIFEST_SHA256,
    )
    evidence = store.get(EpisodeBundleKey(EPISODE_UID, CAMERA_ID, SOURCE_KIND)).view(
        args.observation_row
    )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=str(checkpoint),
        device=args.device,
        strict=True,
        strict_checkpoint_loading=True,
    )
    stack = policy.model.action_head.codec_conditioning
    calls = {"encode": 0, "condition": 0}
    original_encode = stack.encode
    original_condition = stack.condition

    def encode(*values, **kwargs):
        calls["encode"] += 1
        return original_encode(*values, **kwargs)

    def condition(*values, **kwargs):
        calls["condition"] += 1
        return original_condition(*values, **kwargs)

    stack.encode = encode
    stack.condition = condition
    action, info = policy.get_action(
        _observation(step, policy.language_key),
        options={"codec_evidence_sample": evidence},
    )
    if calls != {"encode": 1, "condition": 4}:
        raise RuntimeError(f"Codec call counts drifted: {calls}")
    active_groups = ("left_arm", "right_arm", "left_hand", "right_hand")
    active = np.concatenate([np.asarray(action[key]) for key in active_groups], axis=-1)
    if active.shape != (1, 40, 26) or not np.isfinite(active).all():
        raise RuntimeError(f"Decoded active action is invalid: {active.shape}")

    args.output_root.mkdir(parents=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY",
        "technical_valid": True,
        "source_commit": source_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_index_sha256": _sha256(checkpoint / "model.safetensors.index.json"),
        "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "episode_uid": EPISODE_UID,
        "observation_row": args.observation_row,
        "codec_calls": calls,
        "active_action_shape": list(active.shape),
        "active_action_finite": True,
        "active_action_min": float(active.min()),
        "active_action_max": float(active.max()),
        "info_keys": sorted(info),
        "wall_seconds": time.monotonic() - started,
    }
    summary = args.output_root / "CHECKPOINT_SMOKE_SUMMARY.json"
    _write_new(summary, result)
    _write_new(
        args.output_root / "CHECKPOINT_SMOKE_TERMINAL.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "READY",
            "summary_sha256": _sha256(summary),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
