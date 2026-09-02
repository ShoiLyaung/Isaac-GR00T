"""Production-path verifier for a pinned full-episode codec snapshot."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .config import CodecEvidenceConfig
from .contract import MAX_MV_PER_FRAME, MV_FIELD_DTYPES
from .dataset import CodecEvidenceAugmentedDataset
from .episode_bundle import BUNDLE_SCHEMA_VERSION, FRAME_TYPE_IDR, FRAME_TYPE_P, EpisodeBundleKey
from .torch_backend import EPISODE_AUDIT_IDENTITY_KEYS, EPISODE_MODEL_TENSOR_KEYS
from .warehouse import (
    CANARY_COVERAGE,
    SOURCE_KIND,
    WarehouseEpisodeBundleStore,
    canonical_json_sha256,
    open_fixture_snapshot,
    verify_object,
)


VERIFICATION_SCHEMA_VERSION = "codec-evidence-full-episode-canary-verification-v1"


def _tensor_leaves(value: Any) -> list[Any]:
    import torch
    from transformers import BatchFeature

    if isinstance(value, (dict, BatchFeature)):
        mapping = value.data if isinstance(value, BatchFeature) else value
        return [leaf for item in mapping.values() for leaf in _tensor_leaves(item)]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Production model transport contains a non-Tensor leaf")
    return [value]


def _real_prepare_input(codec_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch
    from torch import nn
    from transformers import BatchFeature

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead

    class Backbone:
        @staticmethod
        def prepare_input(batch):
            return BatchFeature(data={"state": batch["state"]})

    action_head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(action_head)
    harness = SimpleNamespace(
        backbone=Backbone(),
        action_head=action_head,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    _, action_input = Gr00tN1d7.prepare_input(
        harness,
        {
            "state": torch.zeros((1, 1, 132), dtype=torch.float64),
            "codec_evidence": dict(codec_payload),
        },
    )
    return action_input["codec_evidence"]


def _verify_object_science(
    store: WarehouseEpisodeBundleStore,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = verify_object(store.root, entry["object_content_sha256"])
    payload = manifest["manifest_payload"]
    num_frames = entry["num_frames"]
    parity = payload["identities"]["decoder"]["torchcodec_exact_parity"]
    source_hashes = payload["frame_hashes"]["source"]
    if (
        parity.get("dataset_factory", {}).get("dataset_factory_constructed") is not True
        or parity.get("all_rows_array_equal") is not True
        or parity.get("max_abs_diff") != 0
        or parity.get("per_row_hash_equality_count") != num_frames
        or parity.get("per_row_pyav_rgb_sha256") != source_hashes
        or parity.get("per_row_dataset_factory_torchcodec_rgb_sha256") != source_hashes
        or parity.get("per_row_ledger_sha256") != canonical_json_sha256(source_hashes)
    ):
        raise ValueError("Per-row DatasetFactory/TorchCodec/PyAV parity closure failed")
    key = EpisodeBundleKey(entry["episode_uid"], entry["camera_id"], entry["source_kind"])
    bundle = store.get(key)
    if bundle.num_frames != num_frames:
        raise ValueError("Loaded bundle frame count differs from snapshot")
    expected_anchors = np.arange(0, num_frames, 30, dtype=np.int64)
    if not np.array_equal(bundle.anchor_frame_index, expected_anchors):
        raise ValueError("Full-episode anchor table differs from GOP30 starts")
    expected_references = np.arange(num_frames, dtype=np.int64) - 1
    expected_references[::30] = -1
    if not np.array_equal(bundle.reference_index, expected_references):
        raise ValueError("Full-episode causal reference graph mismatch")
    frame_rows = np.arange(num_frames, dtype=np.int64)
    expected_types = np.where(frame_rows % 30 == 0, FRAME_TYPE_IDR, FRAME_TYPE_P).astype(np.uint8)
    expected_clock = frame_rows * 1000
    if not np.array_equal(bundle.frame_type, expected_types):
        raise ValueError("I/IDR and P frame types differ from exact GOP30 positions")
    if not np.array_equal(bundle.pts, expected_clock) or not np.array_equal(
        bundle.dts, expected_clock
    ):
        raise ValueError("Object PTS/DTS differ from row*1000 no-reorder clock")
    claims = payload["identities"]["extractor"]["proof_claims"]
    if (
        not isinstance(claims, dict)
        or not claims
        or not all(value is True for value in claims.values())
    ):
        raise ValueError("H.264 causal proof claims are incomplete or false")

    rows = sorted({0, 14, min(30, num_frames - 1), num_frames - 1})
    views = {row: bundle.view(row) for row in rows}
    row_zero = views[0]
    if row_zero.frame_mask.tolist() != [False] * 7 + [True]:
        raise ValueError("Row-zero T8 view does not have exact seven-slot left padding")
    if row_zero.frame_anchor_slot[:7].tolist() != [-1] * 7:
        raise ValueError("Row-zero PAD anchor slots must use the -1 invalid sentinel")
    if row_zero.anchor_mask.tolist() != [True, False] or np.any(row_zero.anchor_appearance[1] != 0):
        raise ValueError("Row-zero unused anchor-bank slot is not masked exact zero")
    for name in (
        "frame_type",
        "frame_validity",
        "frame_reason_code",
        "mv_cell_features",
        "mv_cell_mask",
        "rpe",
        "rpe_mask",
        "metadata_features",
        "anchor_appearance",
        "anchor_mask",
    ):
        value = np.asarray(getattr(row_zero, name))
        if name in {"anchor_appearance", "anchor_mask"}:
            continue
        if np.any(value[:7] != 0):
            raise ValueError(f"Row-zero PAD {name} is not numerically zero/false")
    row_fourteen = views[14]
    if (
        row_fourteen.frame_index[row_fourteen.frame_mask].tolist() != list(range(7, 15))
        or int(bundle.reference_index[7]) != 6
    ):
        raise ValueError("Row14 does not preserve external predecessor proof 7->6")
    if num_frames > 30:
        row_thirty = views[30]
        if row_thirty.anchor_mask.tolist() != [
            True,
            True,
        ] or row_thirty.anchor_frame_index.tolist() != [0, 30]:
            raise ValueError("Row30 does not preserve exact K2 anchor closure [0,30]")
    descriptors = {item["relative_path"] for item in manifest["artifacts"]}
    required_audit = {"raw_motion_vectors/offsets.npy"} | {
        f"raw_motion_vectors/fields/{name}.npy"
        for name in (
            "source",
            "block_width",
            "block_height",
            "source_x",
            "source_y",
            "destination_x",
            "destination_y",
            "flags",
            "motion_x",
            "motion_y",
            "motion_scale",
        )
    }
    if not required_audit.issubset(descriptors) or "accepted.mp4" not in descriptors:
        raise ValueError("Object lacks the complete no-truncation audit closure")
    object_root = store.root / "objects" / entry["object_content_sha256"]
    with (object_root / "raw_motion_vectors/offsets.npy").open("rb") as stream:
        offsets = np.load(stream, allow_pickle=False)
    if (
        offsets.dtype != np.int64
        or offsets.shape != (num_frames + 1,)
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("Raw MV offsets do not close the complete episode")
    counts = np.diff(offsets)
    if (
        np.any(counts > MAX_MV_PER_FRAME)
        or np.any(counts[frame_rows % 30 == 0] != 0)
        or np.any(counts[frame_rows % 30 != 0] <= 0)
    ):
        raise ValueError("Raw MV counts prove truncation or an empty P-frame decision set")
    raw_mv_count = int(offsets[-1])
    for name, dtype in MV_FIELD_DTYPES.items():
        with (object_root / f"raw_motion_vectors/fields/{name}.npy").open("rb") as stream:
            field = np.load(stream, allow_pickle=False)
        if field.dtype != dtype or field.shape != (raw_mv_count,):
            raise ValueError(f"Raw MV field {name} dtype/shape closure failed")
    return {
        "episode_uid": key.episode_uid,
        "frames": num_frames,
        "anchors": len(expected_anchors),
        "tail": num_frames % 30 or 30,
        "verified_view_rows": rows,
        "view_anchor_counts": {
            str(row): int(view.anchor_mask.sum()) for row, view in views.items()
        },
        "per_row_decoder_hashes": num_frames,
        "artifact_count": len(descriptors),
        "raw_mv_count": raw_mv_count,
        "raw_mv_field_count": len(MV_FIELD_DTYPES),
        "pts_dts_exact": True,
        "causal_proof_claims_all_true": True,
    }


def verify_full_episode_snapshot(
    *,
    snapshot_root: str | Path,
    snapshot_manifest_sha256: str,
    expected_dataset_binding: Mapping[str, Any],
    expected_train_episode_uids: Sequence[str],
    expected_canary_episode_uids: Sequence[str],
    base_dataset: Any,
) -> dict[str, Any]:
    """Verify object science and the real wrapper→processor→collator transport."""

    store = open_fixture_snapshot(
        snapshot_root,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        expected_dataset_binding=expected_dataset_binding,
        expected_train_episode_uids=expected_train_episode_uids,
        camera_id="cam_left_high",
        source_kind=SOURCE_KIND,
        verifier_only=True,
    )
    manifest = store.manifest
    expected_canary = sorted(expected_canary_episode_uids)
    if (
        manifest["coverage_kind"] != CANARY_COVERAGE
        or manifest["training_eligible"] is not False
        or manifest["covered_episode_uids"] != expected_canary
    ):
        raise ValueError("Canary manifest does not pin the exact authorized episode subset")
    object_results = [_verify_object_science(store, entry) for entry in manifest["objects"]]

    config = CodecEvidenceConfig(
        enabled=True,
        fixture_root=Path(snapshot_root),
        fixture_manifest_sha256=snapshot_manifest_sha256,
        schema_version=BUNDLE_SCHEMA_VERSION,
        camera_id="cam_left_high",
        source_kind=SOURCE_KIND,
    )
    wrapper = CodecEvidenceAugmentedDataset(
        base_dataset,
        config,
        store=store,
        verifier_only=True,
    )
    if len(wrapper) == 0:
        raise ValueError("Verifier-only wrapper produced no covered shards")
    datapoints = wrapper.get_shard(0)
    if not datapoints or "codec_evidence_sample" not in datapoints[0]:
        raise ValueError("Production wrapper did not emit codec_evidence_sample")
    from .torch_backend import TorchCodecEvidenceCollator

    codec_payload = TorchCodecEvidenceCollator()([datapoints[0]["codec_evidence_sample"]])
    if set(codec_payload) != set(EPISODE_MODEL_TENSOR_KEYS):
        raise ValueError("Production collator tensor keys differ from the frozen transport")
    if set(codec_payload).intersection(EPISODE_AUDIT_IDENTITY_KEYS):
        raise ValueError("CPU audit identities crossed the model/device boundary")
    prepared = _real_prepare_input(codec_payload)
    leaves = _tensor_leaves(prepared)
    if set(prepared) != set(EPISODE_MODEL_TENSOR_KEYS) or not leaves:
        raise ValueError("Real prepare_input changed the codec tensor transport")
    if any(not bool(np.isfinite(leaf.detach().cpu().numpy()).all()) for leaf in leaves):
        raise ValueError("Codec tensor transport contains NaN/Inf")
    expected_shapes = {
        "mv_cell_features": (1, 8, 2, 2, 11),
        "rpe": (1, 8, 2, 2, 4),
        "metadata_features": (1, 8, 21),
        "anchor_appearance": (1, 2, 3, 256, 256),
    }
    for name, shape in expected_shapes.items():
        if tuple(prepared[name].shape) != shape:
            raise ValueError(f"Prepared {name} shape differs from {shape}")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS",
        "coverage_kind": CANARY_COVERAGE,
        "training_eligible": False,
        "episode_uids": expected_canary,
        "objects": object_results,
        "production_transport": {
            "path": "store->dataset wrapper->processor->collator->real prepare_input.action_input",
            "tensor_keys": sorted(prepared),
            "tensor_leaf_count": len(leaves),
            "audit_identity_keys_at_model_boundary": [],
            "shapes": {name: list(shape) for name, shape in expected_shapes.items()},
            "finite": True,
        },
        "gpu_used": False,
        "training_steps": 0,
    }


__all__ = ["VERIFICATION_SCHEMA_VERSION", "verify_full_episode_snapshot"]
