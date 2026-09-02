#!/usr/bin/env python3
"""Verify the immutable CPU-only R5 codec-evidence pilot.

This verifier consumes only the published pilot root.  It does not inspect the
raw or LeRobot datasets, contact a network service, use a GPU, load model
weights, or run training.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


# Prevent optional dependency update checks before importing GR00T/model code.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from gr00t.data.codec_evidence.dataset import episode_bundle_store_from_pilot_manifest
from gr00t.data.codec_evidence.episode_bundle import (
    FRAME_TYPE_P,
    K_MAX,
    METADATA_WIDTH,
    MV_CELL_WIDTH,
    SOURCE_KINDS,
    EpisodeBundle,
    EpisodeBundleKey,
    EpisodeT8View,
    assert_episode_batch_conforms,
    collate_episode_views_numpy,
)
import numpy as np


SCHEMA_VERSION = "codec-evidence-r5-pilot-verification-v1"
EXPECTED_CAMERA_ID = "cam_left_high"
EXPECTED_FRAMES = 60
VERIFICATION_ROWS = (0, 14, 30, 59)
TORCH_SEED = 20260820
ACTION_WIDTH = 64
D_CODEC = 16
CONDITIONER_HEADS = 8
ACTION_HORIZON = 40
FORBIDDEN_DATA_COMPONENTS = frozenset({"raw", "lerobot"})
FORBIDDEN_TRAINING_KEYS = frozenset(
    {
        "motion_vector_fields",
        "mv_offsets",
        "mv_count",
        "mv_mask",
        "raw_mv_offsets",
        "raw_mv_fields",
    }
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _reject_forbidden_path(path: Path, name: str) -> None:
    forbidden = FORBIDDEN_DATA_COMPONENTS.intersection(path.parts)
    if forbidden:
        raise ValueError(f"{name} may not be inside raw/lerobot data: {sorted(forbidden)}")


def _resolve_fixture_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve(strict=True)
    _reject_forbidden_path(root, "fixture_root")
    if not root.is_dir():
        raise ValueError("fixture_root must be a directory")
    return root


def _resolve_new_output(value: str | Path, *, fixture_root: Path) -> Path:
    output = Path(value).expanduser().resolve(strict=False)
    _reject_forbidden_path(output, "output")
    if output == fixture_root or fixture_root in output.parents:
        raise ValueError("Verification output may not mutate the fixture root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite verification output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    return output


def _inventory_fixture_root(root: Path) -> tuple[int, int]:
    """Return exact regular-file count/bytes without entering data or symlinks."""

    file_count = 0
    byte_count = 0
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory_name in FORBIDDEN_DATA_COMPONENTS:
                raise ValueError("fixture_root contains a forbidden raw/lerobot data directory")
            if directory.is_symlink():
                raise ValueError("Symlinks are forbidden in the pilot fixture root")
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Pilot inventory accepts only regular non-symlink files")
            file_count += 1
            byte_count += path.stat().st_size
    return file_count, byte_count


def _pilot_index(root: Path) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    manifest = _load_json(root / "pilot_manifest.json")
    episode_uids = manifest.get("episode_uids")
    source_kinds = manifest.get("source_kinds")
    if not isinstance(episode_uids, list) or len(episode_uids) != 3 or len(set(episode_uids)) != 3:
        raise ValueError("Verifier requires exactly three unique pilot episode UIDs")
    if not isinstance(source_kinds, list) or set(source_kinds) != set(SOURCE_KINDS):
        raise ValueError("Verifier requires exactly both frozen pilot source kinds")
    if manifest.get("bundle_count") != 6 or manifest.get("camera_id") != EXPECTED_CAMERA_ID:
        raise ValueError("Verifier requires six cam_left_high pilot bundles")
    return manifest, tuple(sorted(episode_uids)), tuple(sorted(source_kinds))


def _assert_t8_closure(bundle: EpisodeBundle, row: int, view: EpisodeT8View) -> dict[str, Any]:
    expected_rows = list(range(max(0, row - 7), row + 1))
    active_rows = view.frame_index[view.frame_mask].tolist()
    if active_rows != expected_rows or int(view.observation_row) != row:
        raise ValueError(f"T8 row closure failed at observation row {row}")

    external_predecessors: list[tuple[int, int]] = []
    active_set = set(expected_rows)
    for frame in expected_rows:
        if int(bundle.frame_type[frame]) == FRAME_TYPE_P:
            reference = int(bundle.reference_index[frame])
            if reference != frame - 1:
                raise ValueError("T8 P frame lacks its episode-global predecessor proof")
            if reference not in active_set:
                external_predecessors.append((frame, reference))

    for offset, frame in zip(np.flatnonzero(view.frame_mask), expected_rows):
        expected_anchor = int(bundle.anchor_index_per_frame[frame])
        slot = int(view.frame_anchor_slot[offset])
        if expected_anchor < 0:
            if slot != -1:
                raise ValueError("T8 frame unexpectedly names an anchor")
        elif slot < 0 or int(view.anchor_frame_index[slot]) != expected_anchor:
            raise ValueError("T8 frame-to-anchor closure differs from the episode bundle")

    if row == 14 and (7, 6) not in external_predecessors:
        raise ValueError("Row 14 must prove frame 7's predecessor 6 outside the T8 window")
    if row == 30:
        if (
            view.anchor_mask.tolist() != [True, True]
            or view.anchor_frame_index.tolist() != [0, 30]
            or view.frame_anchor_slot.tolist() != [0, 0, 0, 0, 0, 0, 0, 1]
        ):
            raise ValueError("Row 30 must close the GOP boundary with the exact K2 anchor bank")
    return {
        "row": row,
        "active_rows": active_rows,
        "anchor_frame_index": view.anchor_frame_index[view.anchor_mask].tolist(),
        "external_predecessors": [list(pair) for pair in external_predecessors],
    }


def _torch_from_numpy_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value)
        elif isinstance(value, tuple) and all(isinstance(item, str) for item in value):
            result[key] = value
        else:
            raise TypeError(f"Unsupported fixed-batch value for Torch: {key}")
    return result


def _verify_dynamics(batch: Mapping[str, Any]) -> dict[str, Any]:
    from gr00t.model.gr00t_n1d7.codec_dynamics import Gr00tCodecConditioningStack
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(TORCH_SEED)
    stack = Gr00tCodecConditioningStack(
        action_width=ACTION_WIDTH,
        d_codec=D_CODEC,
        conditioner_heads=CONDITIONER_HEADS,
        gate_init=0.0,
    ).cpu()
    stack.eval()
    torch_batch = _torch_from_numpy_batch(batch)
    if any(
        value.device.type != "cpu"
        for value in torch_batch.values()
        if isinstance(value, torch.Tensor)
    ):
        raise RuntimeError("Verifier tensor transport left the CPU")
    generator = torch.Generator(device="cpu").manual_seed(TORCH_SEED)
    action = torch.randn(
        len(batch["episode_uid"]),
        ACTION_HORIZON,
        ACTION_WIDTH,
        generator=generator,
        device="cpu",
    )
    with torch.no_grad():
        context, context_mask = stack.encode(torch_batch)
        conditioned = stack.condition(action, context, context_mask)
    if not torch.isfinite(context).all() or not torch.isfinite(conditioned).all():
        raise FloatingPointError("Small codec dynamics stack produced NaN or Inf")
    if not torch.equal(conditioned, action):
        raise AssertionError("gate=0 must preserve bit-exact action features")
    if float(stack.conditioner.gate.detach().cpu().item()) != 0.0:
        raise AssertionError("Codec conditioner gate is not exact zero")
    return {
        "torch_seed": TORCH_SEED,
        "device": "cpu",
        "action_width": ACTION_WIDTH,
        "d_codec": D_CODEC,
        "conditioner_heads": CONDITIONER_HEADS,
        "context_shape": list(context.shape),
        "context_mask_shape": list(context_mask.shape),
        "context_finite": True,
        "conditioned_finite": True,
        "gate_exact_zero": True,
        "torch_equal_native_action": True,
        "action_parity_mismatches": 0,
    }


def verify_codec_r5_pilot(
    fixture_root: str | Path, *, pilot_manifest_sha256: str
) -> dict[str, Any]:
    """Verify one pinned six-bundle pilot and return a JSON-serializable result."""

    started = time.perf_counter()
    root = _resolve_fixture_root(fixture_root)
    pinned_sha256 = _require_sha256(pilot_manifest_sha256, "pilot_manifest_sha256")
    store = episode_bundle_store_from_pilot_manifest(root, pilot_manifest_sha256=pinned_sha256)
    manifest, episode_uids, source_kinds = _pilot_index(root)
    views: list[EpisodeT8View] = []
    bundle_results: list[dict[str, Any]] = []
    raw_mv_total = 0
    accepted_bitstream_bytes = 0

    for episode_uid in episode_uids:
        for source_kind in source_kinds:
            key = EpisodeBundleKey(episode_uid, EXPECTED_CAMERA_ID, source_kind)
            bundle = store.get(key)
            if (
                bundle.num_frames != EXPECTED_FRAMES
                or bundle.split != "train"
                or bundle.key.camera_id != EXPECTED_CAMERA_ID
            ):
                raise ValueError("Every pilot bundle must be train/cam_left_high/N60")
            row_results = []
            for row in VERIFICATION_ROWS:
                view = bundle.view(row)
                row_results.append(_assert_t8_closure(bundle, row, view))
                views.append(view)
            raw_audit = store.audit_raw_motion_vectors(key)
            if raw_audit.offsets.shape != (EXPECTED_FRAMES + 1,):
                raise ValueError("Raw MV audit does not cover the exact N60 episode")
            raw_mv_total += int(raw_audit.offsets[-1])
            accepted_bitstream = store.audit_accepted_bitstream(key)
            accepted_bitstream_bytes += accepted_bitstream.stat().st_size
            bundle_results.append(
                {
                    "episode_uid": episode_uid,
                    "source_kind": source_kind,
                    "camera_id": bundle.key.camera_id,
                    "frames": bundle.num_frames,
                    "split": bundle.split,
                    "raw_mv_count": int(raw_audit.offsets[-1]),
                    "accepted_bitstream_bytes": accepted_bitstream.stat().st_size,
                    "rows": row_results,
                }
            )

    if len(bundle_results) != 6 or len(views) != 24:
        raise AssertionError("Verifier did not close exactly six bundles and twenty-four views")
    batch = collate_episode_views_numpy(views)
    assert_episode_batch_conforms(batch)
    leaked_keys = sorted(set(batch).intersection(FORBIDDEN_TRAINING_KEYS))
    if leaked_keys:
        raise ValueError(f"Fixed training batch exposes raw MV transport: {leaked_keys}")
    expected_shapes = {
        "mv_cell_features": (24, 8, 2, 2, MV_CELL_WIDTH),
        "rpe": (24, 8, 2, 2, 4),
        "metadata_features": (24, 8, METADATA_WIDTH),
        "anchor_appearance": (24, K_MAX, 3, 256, 256),
    }
    for name, expected in expected_shapes.items():
        if batch[name].shape != expected:
            raise ValueError(f"Fixed batch {name} shape differs from {expected}")
    dynamics = _verify_dynamics(batch)
    root_files, root_bytes = _inventory_fixture_root(root)
    elapsed_seconds = time.perf_counter() - started
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "fixture_root": str(root),
        "pilot_manifest_sha256": pinned_sha256,
        "pilot_generation_attempt": manifest["pilot_generation_attempt"],
        "episode_uids": list(episode_uids),
        "source_kinds": list(source_kinds),
        "bundle_count": len(bundle_results),
        "view_count": len(views),
        "verification_rows": list(VERIFICATION_ROWS),
        "bundles": bundle_results,
        "fixed_batch": {
            "batch_size": len(views),
            "mv_cell_features_shape": list(batch["mv_cell_features"].shape),
            "rpe_shape": list(batch["rpe"].shape),
            "metadata_features_shape": list(batch["metadata_features"].shape),
            "anchor_appearance_shape": list(batch["anchor_appearance"].shape),
            "raw_mv_transport_keys": leaked_keys,
            "bounded_no_raw_m_axis": True,
        },
        "raw_mv_audit_bundle_count": len(bundle_results),
        "raw_mv_total": raw_mv_total,
        "accepted_bitstream_bundle_count": len(bundle_results),
        "accepted_bitstream_bytes": accepted_bitstream_bytes,
        "dynamics": dynamics,
        "root_files": root_files,
        "root_bytes": root_bytes,
        "root_inventory": {"files": root_files, "bytes": root_bytes},
        "elapsed": elapsed_seconds,
        "elapsed_seconds": elapsed_seconds,
        "P0": 0,
        "P1": 0,
        "priority_findings": {"P0": 0, "P1": 0},
        "gpu_used": False,
        "network_used": False,
        "training_steps": 0,
        "dataset_reads": {"raw": 0, "lerobot": 0},
        "dataset_writes": {"raw": 0, "lerobot": 0},
    }


def verify_and_write(
    fixture_root: str | Path,
    *,
    pilot_manifest_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    root = _resolve_fixture_root(fixture_root)
    output_path = _resolve_new_output(output, fixture_root=root)
    result = verify_codec_r5_pilot(root, pilot_manifest_sha256=pilot_manifest_sha256)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the pinned CPU-only R5 codec pilot")
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--pilot-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = verify_and_write(
        args.fixture_root,
        pilot_manifest_sha256=args.pilot_manifest_sha256,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
