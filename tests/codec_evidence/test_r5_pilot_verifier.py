from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.episode_bundle import (
    SOURCE_KIND_NATIVE_CANONICAL,
    SOURCE_KIND_RAW_JPEG,
    make_synthetic_episode_bundle,
    write_episode_bundle,
)
import numpy as np
import pytest


_VERIFIER_PATH = (
    Path(__file__).resolve().parents[2] / "examples/G1_ID10_H40_NATIVE30/verify_codec_r5_pilot.py"
)
_SPEC = importlib.util.spec_from_file_location("r5_pilot_verifier_under_test", _VERIFIER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)


def _raw_motion_vectors(num_frames: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    offsets = np.arange(num_frames + 1, dtype=np.int64)
    fields = {
        name: np.arange(num_frames, dtype=np.int64).astype(dtype)
        for name, dtype in MV_FIELD_DTYPES.items()
    }
    fields["block_width"][:] = 16
    fields["block_height"][:] = 16
    fields["motion_scale"][:] = 4
    return offsets, fields


@pytest.fixture(scope="module")
def synthetic_pilot(tmp_path_factory):
    base = tmp_path_factory.mktemp("r5-pilot-verifier")
    root = base / "pilot"
    inputs = base / "inputs"
    root.mkdir()
    inputs.mkdir()
    episode_uids = [character * 64 for character in "abc"]
    source_kinds = [SOURCE_KIND_RAW_JPEG, SOURCE_KIND_NATIVE_CANONICAL]
    summaries = []
    for episode_index, episode_uid in enumerate(episode_uids):
        for source_kind in source_kinds:
            bundle = make_synthetic_episode_bundle(
                num_frames=60,
                episode_uid=episode_uid,
                source_kind=source_kind,
            )
            metadata = bundle.metadata_dict()
            metadata.update(
                {
                    "split": "train",
                    "cohort_id": f"synthetic-cohort-{episode_index}",
                    "dataset_name": "synthetic-verifier",
                    "source_episode_index": episode_index,
                }
            )
            raw_offsets, raw_fields = _raw_motion_vectors(bundle.num_frames)
            accepted = inputs / f"{episode_uid[:1]}-{source_kind}.mp4"
            accepted.write_bytes(f"accepted:{episode_uid}:{source_kind}".encode("ascii"))
            manifest_path = write_episode_bundle(
                root,
                episode_uid=episode_uid,
                camera_id="cam_left_high",
                source_kind=source_kind,
                metadata=metadata,
                arrays=bundle.arrays_dict(),
                raw_mv_offsets=raw_offsets,
                raw_mv_fields=raw_fields,
                accepted_bitstream=accepted,
            )
            summaries.append(
                {
                    "episode_uid": episode_uid,
                    "camera_id": "cam_left_high",
                    "source_kind": source_kind,
                    "relative_path": manifest_path.relative_to(root).as_posix(),
                    "sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                    "bytes": manifest_path.stat().st_size,
                    "frames": 60,
                }
            )
    manifest = {
        "schema_version": "codec-evidence-r5-pilot-v1",
        "complete": True,
        "pilot_generation_attempt": 1,
        "source_kinds": source_kinds,
        "camera_id": "cam_left_high",
        "rows": [0, 59],
        "bundles": summaries,
        "bundle_count": 6,
        "episode_uids": episode_uids,
        "determinism_encode_repetitions_per_bundle": 2,
        "total_internal_encode_executions": 12,
        "output_policy": "CREATE_NEW_ROOT_AND_WRITER_NO_CLOBBER",
        "dataset_mutations": 0,
        "gpu_used": False,
    }
    manifest_path = root / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
    }


def test_cli_verifies_six_bundles_t8_k2_audits_fixed_batch_and_parity(
    tmp_path, synthetic_pilot, monkeypatch
):
    output = tmp_path / "verification.json"
    original_open = Path.open

    def forbid_dataset_open(path, *args, **kwargs):
        if {"raw", "lerobot"}.intersection(path.resolve(strict=False).parts):
            raise AssertionError(f"Verifier attempted dataset access: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_dataset_open)
    result_code = verifier.main(
        [
            "--fixture-root",
            str(synthetic_pilot["root"]),
            "--pilot-manifest-sha256",
            synthetic_pilot["manifest_sha256"],
            "--output",
            str(output),
        ]
    )
    assert result_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema_version"] == verifier.SCHEMA_VERSION
    assert result["status"] == "PASS"
    assert result["bundle_count"] == 6
    assert result["view_count"] == 24
    assert result["verification_rows"] == [0, 14, 30, 59]
    assert len(result["episode_uids"]) == 3
    assert set(result["source_kinds"]) == {
        SOURCE_KIND_RAW_JPEG,
        SOURCE_KIND_NATIVE_CANONICAL,
    }
    assert result["raw_mv_audit_bundle_count"] == 6
    assert result["raw_mv_total"] == 6 * 60
    assert result["accepted_bitstream_bundle_count"] == 6
    assert result["accepted_bitstream_bytes"] > 0
    assert result["fixed_batch"] == {
        "batch_size": 24,
        "mv_cell_features_shape": [24, 8, 2, 2, 11],
        "rpe_shape": [24, 8, 2, 2, 4],
        "metadata_features_shape": [24, 8, 21],
        "anchor_appearance_shape": [24, 2, 3, 256, 256],
        "raw_mv_transport_keys": [],
        "bounded_no_raw_m_axis": True,
    }
    assert result["dynamics"]["device"] == "cpu"
    assert result["dynamics"]["context_finite"] is True
    assert result["dynamics"]["conditioned_finite"] is True
    assert result["dynamics"]["gate_exact_zero"] is True
    assert result["dynamics"]["torch_equal_native_action"] is True
    assert result["dynamics"]["action_parity_mismatches"] == 0
    assert result["P0"] == result["P1"] == 0
    assert result["priority_findings"] == {"P0": 0, "P1": 0}
    assert result["gpu_used"] is False
    assert result["network_used"] is False
    assert result["training_steps"] == 0
    assert result["dataset_reads"] == {"raw": 0, "lerobot": 0}
    assert result["dataset_writes"] == {"raw": 0, "lerobot": 0}
    assert result["root_files"] > 0
    assert result["root_bytes"] > 0
    assert result["root_inventory"] == {
        "files": result["root_files"],
        "bytes": result["root_bytes"],
    }
    assert result["elapsed"] == result["elapsed_seconds"]
    assert result["elapsed_seconds"] >= 0

    for bundle_result in result["bundles"]:
        assert bundle_result["frames"] == 60
        assert bundle_result["split"] == "train"
        assert bundle_result["camera_id"] == "cam_left_high"
        by_row = {row["row"]: row for row in bundle_result["rows"]}
        assert [7, 6] in by_row[14]["external_predecessors"]
        assert by_row[30]["anchor_frame_index"] == [0, 30]


def test_output_is_checked_before_verification_and_never_overwritten(
    tmp_path, synthetic_pilot, monkeypatch
):
    output = tmp_path / "existing.json"
    output.write_text("user evidence\n", encoding="utf-8")

    def forbidden_verify(*args, **kwargs):
        raise AssertionError("Verification ran despite an occupied output path")

    monkeypatch.setattr(verifier, "verify_codec_r5_pilot", forbidden_verify)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        verifier.verify_and_write(
            synthetic_pilot["root"],
            pilot_manifest_sha256=synthetic_pilot["manifest_sha256"],
            output=output,
        )
    assert output.read_text(encoding="utf-8") == "user evidence\n"


def test_manifest_pin_and_required_raw_audit_fail_closed(tmp_path, synthetic_pilot):
    with pytest.raises(ValueError, match="Pilot manifest SHA-256 mismatch"):
        verifier.verify_codec_r5_pilot(synthetic_pilot["root"], pilot_manifest_sha256="f" * 64)

    copied = tmp_path / "missing-audit"
    shutil.copytree(synthetic_pilot["root"], copied)
    pilot_manifest_path = copied / "pilot_manifest.json"
    pilot_manifest_path.chmod(0o644)
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    entry = pilot_manifest["bundles"][0]
    bundle_manifest_path = copied / entry["relative_path"]
    bundle_manifest_path.chmod(0o644)
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    bundle_manifest["raw_motion_vector_audit"] = None
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    entry["sha256"] = sha256(bundle_manifest_path.read_bytes()).hexdigest()
    entry["bytes"] = bundle_manifest_path.stat().st_size
    pilot_manifest_path.write_text(
        json.dumps(pilot_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    pilot_sha = sha256(pilot_manifest_path.read_bytes()).hexdigest()
    with pytest.raises(KeyError, match="no raw MV audit"):
        verifier.verify_codec_r5_pilot(copied, pilot_manifest_sha256=pilot_sha)


def test_inventory_rejects_raw_or_lerobot_subtrees_without_entering_them(tmp_path):
    root = tmp_path / "fixture"
    forbidden = root / "raw"
    forbidden.mkdir(parents=True)
    (forbidden / "must-not-read.txt").write_text("payload", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden raw/lerobot"):
        verifier._inventory_fixture_root(root)


def test_verifier_source_has_no_gpu_network_subprocess_or_training_surface():
    source = _VERIFIER_PATH.read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "import requests" not in source
    assert "urlopen" not in source
    assert "torch.cuda" not in source
    assert ".backward(" not in source
    assert "optimizer" not in source.lower()
