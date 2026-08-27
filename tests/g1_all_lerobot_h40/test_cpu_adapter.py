from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.data.dataset.g1_all_lerobot_h40_adapter import (
    ADAPTER_KIND,
    BASELINE_ID,
    CODEC_CAMERA_ID,
    CODEC_EPISODE_UID_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    VIEW_SCHEMA_VERSION,
    G1AllLerobotH40EpisodeLoader,
    build_codec_episode_identity,
    build_metadata_loader_view,
    load_adapter_config,
)
from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTIVE_ACTION_DIM,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    HORIZON,
    TEMPORAL_OPTION,
    compute_h40_relative_statistics,
)
from gr00t.data.dataset.g1_id10_h50_adapter import _sha256
import numpy as np
import pandas as pd
import pytest


TRAINING_LANGUAGE = "insert the shaft into the housing"
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")


def _codec_block(root: Path, *, manifest_sha256: str = "a" * 64) -> dict:
    return {
        "enabled": True,
        "fixture_root": str(root.resolve()),
        "fixture_manifest_sha256": manifest_sha256,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": CODEC_CAMERA_ID,
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "d_codec": 64,
        "conditioner_heads": 8,
    }


def _codec_episode_identity(source_id: str, salt: int = 1) -> dict:
    from gr00t.data.codec_evidence.identity import episode_identity_with_uid

    return episode_identity_with_uid(
        {
            "schema_version": CODEC_EPISODE_UID_SCHEMA_VERSION,
            "source_id": source_id,
            "canonical_dataset_identity": {
                "info_sha256": f"{salt:064x}",
                "modality_sha256": f"{salt + 10:064x}",
            },
            "episode_index": 0,
            "episode_length": 45,
            "camera_id": CODEC_CAMERA_ID,
            "parquet": {
                "relative_path": "data/chunk-000/episode_000000.parquet",
                "bytes": 1,
                "sha256": f"{salt + 20:064x}",
            },
            "video": {
                "relative_path": (
                    "videos/chunk-000/observation.images.cam_left_high/episode_000000.mp4"
                ),
                "bytes": 1,
                "sha256": f"{salt + 30:064x}",
            },
        }
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _stats(dim: int) -> dict[str, list[float]]:
    return {
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "min": [-1.0] * dim,
        "max": [1.0] * dim,
        "q01": [-1.0] * dim,
        "q99": [1.0] * dim,
    }


def _make_canonical(
    root: Path,
    *,
    raw_label: str,
    cameras: tuple[str, ...] = ("cam_left_high",),
    length: int = 45,
) -> Path:
    (root / "data/chunk-000").mkdir(parents=True)
    for camera in cameras:
        camera_root = root / f"videos/chunk-000/observation.images.{camera}"
        camera_root.mkdir(parents=True)
        (camera_root / "episode_000000.mp4").write_bytes(f"synthetic-{camera}".encode("ascii"))
    states, actions = [], []
    for index in range(length):
        arms = np.linspace(-0.2, 0.2, 14, dtype=np.float32) + index * 0.001
        hands = np.linspace(0.1, 0.9, 12, dtype=np.float32)
        states.append(np.concatenate([arms, hands]).astype(np.float32))
        actions.append(np.concatenate([arms + 0.05, hands]).astype(np.float32))
    pd.DataFrame(
        {
            "observation.state": states,
            "action": actions,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.zeros(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            "annotation.human.action.task_description": [raw_label] * length,
            "annotation.human.validity": np.ones(length, dtype=np.int64),
        }
    ).to_parquet(root / "data/chunk-000/episode_000000.parquet", index=False)
    _write_json(
        root / "meta/info.json",
        {
            "chunks_size": 1000,
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [26]},
                "action": {"dtype": "float32", "shape": [26]},
                **{
                    f"observation.images.{camera}": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    }
                    for camera in cameras
                },
            },
        },
    )
    _write_json(
        root / "meta/modality.json",
        {
            "state": {
                "left_arm": {"start": 0, "end": 7, "original_key": "observation.state"},
                "right_arm": {"start": 7, "end": 14, "original_key": "observation.state"},
                "left_hand": {"start": 14, "end": 20, "original_key": "observation.state"},
                "right_hand": {"start": 20, "end": 26, "original_key": "observation.state"},
            },
            "action": {
                "left_arm": {"start": 0, "end": 7, "original_key": "action"},
                "right_arm": {"start": 7, "end": 14, "original_key": "action"},
                "left_hand": {"start": 14, "end": 20, "original_key": "action"},
                "right_hand": {"start": 20, "end": 26, "original_key": "action"},
            },
            "video": {
                camera: {"original_key": f"observation.images.{camera}"} for camera in cameras
            },
            "annotation": {
                "human.action.task_description": {
                    "original_key": "annotation.human.action.task_description"
                }
            },
        },
    )
    _write_json(root / "meta/stats.json", {"observation.state": _stats(26), "action": _stats(26)})
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": 0, "length": length, "tasks": [raw_label]}],
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": raw_label}])
    return root


def _modality(repo: Path):
    config = repo / "examples/G1_ID10_H40_NATIVE30/g1_id10_h40_native30_config.py"
    spec = importlib.util.spec_from_file_location("test_all_lerobot_h40", config)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    return MODALITY_CONFIGS[EMBODIMENT_TAG]


def _adapter_config(
    path: Path,
    *,
    canonical: Path,
    raw_label: str,
    stats_path: Path,
    source_id: str,
    codec_evidence: dict | None = None,
) -> Path:
    value = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "adapter_kind": ADAPTER_KIND,
        "baseline_id": BASELINE_ID,
        "source_id": source_id,
        "canonical_root": str(canonical.resolve()),
        "canonical_info_sha256": _sha256(canonical / "meta/info.json"),
        "canonical_modality_sha256": _sha256(canonical / "meta/modality.json"),
        "raw_task_labels": [raw_label],
        "training_language": TRAINING_LANGUAGE,
        "embodiment_tag": EMBODIMENT_TAG,
        "embodiment_id": EMBODIMENT_ID,
        "temporal_option": TEMPORAL_OPTION,
        "source_index_formula": "n+k",
        "row_grid_rate_hz": 30,
        "true_30hz_hardware_clock_proven": False,
        "true_50hz_capture": False,
        "horizon": HORIZON,
        "active_action_dims": ACTIVE_ACTION_DIM,
        "hand_scope": "PROJECT_INSPIRE6_EXTENSION_NO_DIRECT_OFFICIAL_MATCH",
        "runtime_verified_fixed_state": False,
        "fixed_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
        },
        "fixed_action": {
            "waist": [0.0] * 3,
            "base_height_command": [0.0],
            "navigate_command": [0.0] * 3,
        },
        "derived_relative_stats_path": str(stats_path.resolve()),
    }
    if codec_evidence is not None:
        value["codec_evidence"] = codec_evidence
    _write_json(path, value)
    return path


@pytest.mark.parametrize(
    ("raw_label", "cameras"),
    [
        ("insert the shaft into the housing", ("cam_left_high",)),
        ("100_notau", ("cam_left_high",)),
        (
            "insertion_with_torque",
            ("cam_left_high", "cam_left_wrist", "cam_right_wrist"),
        ),
    ],
)
def test_loader_view_preserves_source_label_and_normalizes_language(
    tmp_path: Path, raw_label: str, cameras: tuple[str, ...]
):
    canonical = _make_canonical(tmp_path / "canonical", raw_label=raw_label, cameras=cameras)
    protected = {
        path: path.read_bytes()
        for path in [
            canonical / "meta/info.json",
            canonical / "meta/modality.json",
            canonical / "meta/tasks.jsonl",
            canonical / "data/chunk-000/episode_000000.parquet",
        ]
    }
    source_id = "source"
    view = tmp_path / "view"
    manifest = build_metadata_loader_view(
        canonical,
        view,
        source_id=source_id,
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=[raw_label],
        training_language=TRAINING_LANGUAGE,
    )
    assert manifest["schema_version"] == VIEW_SCHEMA_VERSION
    assert manifest["raw_task_labels"] == [raw_label]
    assert manifest["training_language"] == TRAINING_LANGUAGE
    assert set(json.loads((view / "meta/modality.json").read_text())["video"]) == {"ego_view"}
    assert json.loads((view / "meta/tasks.jsonl").read_text())["task"] == TRAINING_LANGUAGE
    assert (view / "data").is_symlink() and (view / "data").resolve() == (
        canonical / "data"
    ).resolve()
    assert (view / "videos").is_symlink() and (view / "videos").resolve() == (
        canonical / "videos"
    ).resolve()
    assert all(path.read_bytes() == content for path, content in protected.items())

    stats_path = tmp_path / "relative.json"
    stats = compute_h40_relative_statistics(canonical, stats_path)
    assert stats["anchors"] == 6
    config = _adapter_config(
        tmp_path / "adapter.json",
        canonical=canonical,
        raw_label=raw_label,
        stats_path=stats_path,
        source_id=source_id,
    )
    assert load_adapter_config(config)["raw_task_labels"] == [raw_label]
    loader = G1AllLerobotH40EpisodeLoader(
        view,
        _modality(Path(__file__).parents[2]),
        config,
        require_runtime_verified_fixed_state=False,
    )
    frame = loader._load_parquet_data(0)
    assert frame["language.annotation.human.task_description"].unique().tolist() == [
        TRAINING_LANGUAGE
    ]
    for side in ("left_arm", "right_arm"):
        for name in STAT_NAMES:
            assert np.asarray(
                loader.get_dataset_statistics()["relative_action"][side][name]
            ).shape == (
                40,
                7,
            )


def test_loader_fails_closed_on_raw_label_mismatch(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="100_notau")
    view = tmp_path / "view"
    build_metadata_loader_view(
        canonical,
        view,
        source_id="source",
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=["100_notau"],
        training_language=TRAINING_LANGUAGE,
    )
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    config = _adapter_config(
        tmp_path / "adapter.json",
        canonical=canonical,
        raw_label="100_notau",
        stats_path=stats_path,
        source_id="source",
    )
    parquet = canonical / "data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(parquet)
    frame["annotation.human.action.task_description"] = "wrong"
    frame.to_parquet(parquet, index=False)
    loader = G1AllLerobotH40EpisodeLoader(
        view,
        _modality(Path(__file__).parents[2]),
        config,
        require_runtime_verified_fixed_state=False,
    )
    with pytest.raises(ValueError, match="Raw task label mismatch"):
        loader._load_parquet_data(0)


def test_loader_view_is_non_overwriting_and_external_only(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    kwargs = {
        "source_id": "source",
        "expected_info_sha256": _sha256(canonical / "meta/info.json"),
        "expected_modality_sha256": _sha256(canonical / "meta/modality.json"),
        "raw_task_labels": ["task"],
        "training_language": TRAINING_LANGUAGE,
    }
    with pytest.raises(ValueError, match="must not be inside"):
        build_metadata_loader_view(canonical, canonical / "view", **kwargs)
    view = tmp_path / "view"
    build_metadata_loader_view(canonical, view, **kwargs)
    with pytest.raises(FileExistsError):
        build_metadata_loader_view(canonical, view, **kwargs)


def test_codec_absent_and_disabled_do_not_hash_episode_payloads(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    view = tmp_path / "view"
    build_metadata_loader_view(
        canonical,
        view,
        source_id="source",
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=["task"],
        training_language=TRAINING_LANGUAGE,
    )
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    absent = _adapter_config(
        tmp_path / "absent.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source",
    )
    disabled = _adapter_config(
        tmp_path / "disabled.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source",
        codec_evidence={"enabled": False},
    )
    assert "codec_evidence" not in load_adapter_config(absent)
    for config in (absent, disabled):
        with patch(
            "gr00t.data.dataset.g1_all_lerobot_h40_adapter._sha256",
            side_effect=AssertionError("disabled Codec path must not hash payloads"),
        ):
            loader = G1AllLerobotH40EpisodeLoader(
                view,
                _modality(Path(__file__).parents[2]),
                config,
                require_runtime_verified_fixed_state=False,
            )
        assert not hasattr(loader, "selected_episode_uids")


def test_codec_absent_and_disabled_do_not_import_optional_package():
    script = """
import sys
from gr00t.data.dataset.g1_all_lerobot_h40_adapter import parse_adapter_codec_evidence_config
assert not any(name.startswith('gr00t.data.codec_evidence') for name in sys.modules)
assert parse_adapter_codec_evidence_config({}) is None
assert parse_adapter_codec_evidence_config(
    {'codec_evidence': {'enabled': False, 'fixture_root': '/must/not/be-read'}}
) is None
assert not any(name.startswith('gr00t.data.codec_evidence') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("payload_name", ["data", "videos"])
def test_enabled_codec_loader_rejects_retargeted_payload_symlink(tmp_path: Path, payload_name: str):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    other = _make_canonical(tmp_path / "other", raw_label="task")
    view = tmp_path / "view"
    build_metadata_loader_view(
        canonical,
        view,
        source_id="source",
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=["task"],
        training_language=TRAINING_LANGUAGE,
    )
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    config = _adapter_config(
        tmp_path / "enabled.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source",
        codec_evidence=_codec_block(tmp_path / "warehouse"),
    )
    payload = view / payload_name
    payload.unlink()
    payload.symlink_to(other / payload_name, target_is_directory=True)
    with pytest.raises(ValueError, match="target differs from canonical payload"):
        G1AllLerobotH40EpisodeLoader(
            view,
            _modality(Path(__file__).parents[2]),
            config,
            require_runtime_verified_fixed_state=False,
        )


@pytest.mark.parametrize(
    ("metadata_name", "mutation", "error"),
    [
        ("info.json", "fps", "info.json differs"),
        ("info.json", "path", "info.json differs"),
        ("modality.json", "camera", "modality.json differs"),
        ("stats.json", "stats", "stats.json differs"),
        ("episodes.jsonl", "episodes", "episodes.jsonl differs"),
        (
            "g1_all_lerobot_h40_view_manifest.json",
            "manifest",
            "manifest differs",
        ),
        ("tasks.jsonl", "tasks", "tasks.jsonl differs"),
    ],
)
def test_enabled_codec_loader_rejects_view_metadata_drift(
    tmp_path: Path, metadata_name: str, mutation: str, error: str
):
    canonical = _make_canonical(
        tmp_path / "canonical",
        raw_label="task",
        cameras=("cam_left_high", "cam_left_wrist"),
    )
    view = tmp_path / "view"
    build_metadata_loader_view(
        canonical,
        view,
        source_id="source",
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=["task"],
        training_language=TRAINING_LANGUAGE,
    )
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    config = _adapter_config(
        tmp_path / "enabled.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source",
        codec_evidence=_codec_block(tmp_path / "warehouse"),
    )
    metadata_path = view / "meta" / metadata_name
    if mutation in {"fps", "path"}:
        value = json.loads(metadata_path.read_text())
        if mutation == "fps":
            value["fps"] = 29
        else:
            value["video_path"] = "videos/changed/{episode_index:06d}.mp4"
        _write_json(metadata_path, value)
    elif mutation == "camera":
        value = json.loads(metadata_path.read_text())
        value["video"]["ego_view"]["original_key"] = "observation.images.cam_left_wrist"
        _write_json(metadata_path, value)
    elif mutation == "stats":
        value = json.loads(metadata_path.read_text())
        value["action"]["mean"][0] = 123.0
        _write_json(metadata_path, value)
    elif mutation == "episodes":
        row = json.loads(metadata_path.read_text().splitlines()[0])
        row["length"] += 1
        _write_jsonl(metadata_path, [row])
    elif mutation == "manifest":
        value = json.loads(metadata_path.read_text())
        value["row_grid_rate_hz"] = 29
        _write_json(metadata_path, value)
    else:
        _write_jsonl(metadata_path, [{"task_index": 0, "task": "changed"}])

    with pytest.raises(ValueError, match=error):
        G1AllLerobotH40EpisodeLoader(
            view,
            _modality(Path(__file__).parents[2]),
            config,
            require_runtime_verified_fixed_state=False,
        )


def test_enabled_codec_uid_binds_source_dataset_parquet_and_head_video(tmp_path: Path):
    canonical = _make_canonical(
        tmp_path / "canonical",
        raw_label="task",
        cameras=("cam_left_high", "cam_left_wrist", "cam_right_wrist"),
    )
    view = tmp_path / "view"
    build_metadata_loader_view(
        canonical,
        view,
        source_id="source-a",
        expected_info_sha256=_sha256(canonical / "meta/info.json"),
        expected_modality_sha256=_sha256(canonical / "meta/modality.json"),
        raw_task_labels=["task"],
        training_language=TRAINING_LANGUAGE,
    )
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    config = _adapter_config(
        tmp_path / "enabled.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source-a",
        codec_evidence=_codec_block(tmp_path / "warehouse"),
    )
    loader = G1AllLerobotH40EpisodeLoader(
        view,
        _modality(Path(__file__).parents[2]),
        config,
        require_runtime_verified_fixed_state=False,
    )
    assert len(loader.selected_episode_uids) == len(loader.episodes_metadata) == 1
    identity = loader.codec_episode_identities[0]
    assert identity["schema_version"] == CODEC_EPISODE_UID_SCHEMA_VERSION
    assert identity["camera_id"] == CODEC_CAMERA_ID
    assert identity["video"]["relative_path"].endswith(
        "observation.images.cam_left_high/episode_000000.mp4"
    )
    assert len(identity["episode_uid"]) == 64
    expected = build_codec_episode_identity(
        source_id="source-a",
        canonical_root=canonical,
        canonical_info_sha256=_sha256(canonical / "meta/info.json"),
        canonical_modality_sha256=_sha256(canonical / "meta/modality.json"),
        data_path_pattern=loader.data_path_pattern,
        video_path_pattern=loader.video_path_pattern,
        chunk_size=loader.chunk_size,
        episode_index=0,
        episode_length=45,
    )
    assert identity == expected
    changed_source = build_codec_episode_identity(
        source_id="source-b",
        canonical_root=canonical,
        canonical_info_sha256=_sha256(canonical / "meta/info.json"),
        canonical_modality_sha256=_sha256(canonical / "meta/modality.json"),
        data_path_pattern=loader.data_path_pattern,
        video_path_pattern=loader.video_path_pattern,
        chunk_size=loader.chunk_size,
        episode_index=0,
        episode_length=45,
    )
    assert changed_source["episode_uid"] != identity["episode_uid"]

    parquet = canonical / "data/chunk-000/episode_000000.parquet"
    parquet_bytes = parquet.read_bytes()
    parquet.write_bytes(parquet_bytes + b"synthetic-identity-change")
    changed_parquet = build_codec_episode_identity(
        source_id="source-a",
        canonical_root=canonical,
        canonical_info_sha256=_sha256(canonical / "meta/info.json"),
        canonical_modality_sha256=_sha256(canonical / "meta/modality.json"),
        data_path_pattern=loader.data_path_pattern,
        video_path_pattern=loader.video_path_pattern,
        chunk_size=loader.chunk_size,
        episode_index=0,
        episode_length=45,
    )
    parquet.write_bytes(parquet_bytes)
    assert changed_parquet["episode_uid"] != identity["episode_uid"]

    head_video = canonical / "videos/chunk-000/observation.images.cam_left_high/episode_000000.mp4"
    original_uid = loader.selected_episode_uids[0]
    head_video.write_bytes(b"different-synthetic-head-video")
    changed_loader = G1AllLerobotH40EpisodeLoader(
        view,
        _modality(Path(__file__).parents[2]),
        config,
        require_runtime_verified_fixed_state=False,
    )
    assert changed_loader.selected_episode_uids[0] != original_uid


def test_codec_block_is_strict_and_freezes_head_camera(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    block = _codec_block(tmp_path / "warehouse")
    block["unknown"] = True
    unknown = _adapter_config(
        tmp_path / "unknown.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source",
        codec_evidence=block,
    )
    with pytest.raises(ValueError, match="Unknown codec_evidence fields"):
        load_adapter_config(unknown)

    wrong_camera_block = _codec_block(tmp_path / "warehouse")
    wrong_camera_block["camera_id"] = "cam_left_wrist"
    wrong_camera = _adapter_config(
        tmp_path / "wrong-camera.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source",
        codec_evidence=wrong_camera_block,
    )
    with pytest.raises(ValueError, match="requires cam_left_high"):
        load_adapter_config(wrong_camera)


def test_adapter_config_rejects_duplicate_json_keys(tmp_path: Path):
    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    config_path = _adapter_config(
        tmp_path / "adapter.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source",
    )
    value = config_path.read_text().rstrip()
    config_path.write_text(value[:-1] + ',"source_id":"other"}\n')
    with pytest.raises(ValueError, match="Duplicate JSON key: source_id"):
        load_adapter_config(config_path)


def test_factory_adapter_never_writes_canonical_stats():
    from gr00t.data.dataset.factory import DatasetFactory

    spec = SimpleNamespace(
        dataset_paths=["loader-view"],
        embodiment_tag=EMBODIMENT_TAG,
        mix_ratio=1.0,
        dataset_adapter=ADAPTER_KIND,
        adapter_config_path="adapter.json",
    )
    data = SimpleNamespace(
        datasets=[spec],
        mode="single_turn",
        modality_configs={EMBODIMENT_TAG: {}},
        video_backend="torchcodec",
        shard_size=128,
        episode_sampling_rate=0.1,
        seed=42,
        allow_padding=False,
        num_shards_per_epoch=1,
        override_pretraining_statistics=True,
    )
    config = SimpleNamespace(data=data, training=SimpleNamespace(eval_strategy="no"))
    fake_dataset = MagicMock()
    fake_dataset.__len__.return_value = 1
    mixture = object()
    with (
        patch(
            "gr00t.data.dataset.factory._resolve_all_lerobot_codec_config",
            return_value=None,
        ),
        patch("gr00t.data.dataset.factory.generate_stats") as generate_stats,
        patch("gr00t.data.dataset.factory.generate_rel_stats") as generate_rel_stats,
        patch(
            "gr00t.data.dataset.g1_all_lerobot_h40_adapter.G1AllLerobotH40SingleStepDataset",
            return_value=fake_dataset,
        ) as adapter_dataset,
        patch("gr00t.data.dataset.factory.ShardedMixtureDataset", return_value=mixture),
    ):
        result, evaluation = DatasetFactory(config).build(MagicMock())
    assert result is mixture and evaluation is None
    generate_stats.assert_not_called()
    generate_rel_stats.assert_not_called()
    adapter_dataset.assert_called_once()


def test_factory_rejects_per_leaf_codec_semantic_drift(tmp_path: Path):
    from gr00t.data.dataset.factory import _resolve_all_lerobot_codec_config

    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    specs = []
    for index in range(2):
        block = _codec_block(tmp_path / ("warehouse-a" if index == 0 else "warehouse-b"))
        config = _adapter_config(
            tmp_path / f"adapter-{index}.json",
            canonical=canonical,
            raw_label="task",
            stats_path=stats_path,
            source_id=f"source-{index}",
            codec_evidence=block,
        )
        specs.append(
            SimpleNamespace(
                dataset_adapter=ADAPTER_KIND,
                adapter_config_path=str(config),
            )
        )
    with pytest.raises(ValueError, match="one identical global codec_evidence config"):
        _resolve_all_lerobot_codec_config(specs)


def test_factory_treats_absent_and_disabled_dead_fields_as_native(tmp_path: Path):
    from gr00t.data.dataset.factory import _resolve_all_lerobot_codec_config

    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    configs = [
        _adapter_config(
            tmp_path / "adapter-absent.json",
            canonical=canonical,
            raw_label="task",
            stats_path=stats_path,
            source_id="source-absent",
        ),
        _adapter_config(
            tmp_path / "adapter-disabled.json",
            canonical=canonical,
            raw_label="task",
            stats_path=stats_path,
            source_id="source-disabled",
            codec_evidence={
                "enabled": False,
                "schema_version": "codec-evidence-episode-bundle-v1",
                "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
                "modalities": ["metadata"],
                "d_codec": 32,
                "conditioner_heads": 4,
            },
        ),
    ]
    specs = [
        SimpleNamespace(dataset_adapter=ADAPTER_KIND, adapter_config_path=str(path))
        for path in configs
    ]

    assert _resolve_all_lerobot_codec_config(specs) is None


def test_factory_rejects_record_v2_as_enabled_five_source_training(tmp_path: Path):
    from gr00t.data.dataset.factory import _resolve_all_lerobot_codec_config

    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    config = _adapter_config(
        tmp_path / "adapter.json",
        canonical=canonical,
        raw_label="task",
        stats_path=stats_path,
        source_id="source-record-v2",
        codec_evidence={
            "enabled": True,
            "fixture_root": str(tmp_path / "fixture"),
            "fixture_manifest_sha256": "a" * 64,
            "record_schema_sha256": "b" * 64,
            "schema_version": "codec-evidence-record-v2",
        },
    )
    specs = [SimpleNamespace(dataset_adapter=ADAPTER_KIND, adapter_config_path=str(config))]

    with pytest.raises(ValueError, match="requires codec-evidence-episode-bundle-v1"):
        _resolve_all_lerobot_codec_config(specs)


def test_factory_wraps_five_codec_leaves_under_one_global_config(tmp_path: Path):
    from gr00t.data.codec_evidence.codec_contract import PRODUCTION_FIXTURE_MODE
    from gr00t.data.codec_evidence.episode_bundle import (
        SOURCE_KIND_NATIVE_CANONICAL,
        EpisodeBundleKey,
    )
    from gr00t.data.dataset.factory import DatasetFactory

    canonical = _make_canonical(tmp_path / "canonical", raw_label="task")
    stats_path = tmp_path / "relative.json"
    compute_h40_relative_statistics(canonical, stats_path)
    block = _codec_block(tmp_path / "warehouse")
    specs = []
    fake_datasets = []
    identities_by_uid = {}
    mix_ratios = [0.11, 0.22, 0.27, 0.31, 0.09]
    for index in range(5):
        config_path = _adapter_config(
            tmp_path / f"adapter-{index}.json",
            canonical=canonical,
            raw_label="task",
            stats_path=stats_path,
            source_id=f"source-{index}",
            codec_evidence=block,
        )
        specs.append(
            SimpleNamespace(
                dataset_paths=[f"loader-view-{index}"],
                embodiment_tag=EMBODIMENT_TAG,
                mix_ratio=mix_ratios[index],
                dataset_adapter=ADAPTER_KIND,
                adapter_config_path=str(config_path),
            )
        )
        fake_dataset = MagicMock()
        fake_dataset.__len__.return_value = 1
        identity = _codec_episode_identity(f"source-{index}", index + 1)
        episode_uid = identity["episode_uid"]
        identities_by_uid[episode_uid] = identity
        fake_dataset.episode_loader.selected_episode_uids = (episode_uid,)
        fake_dataset.episode_loader.codec_episode_identities = (identity,)
        fake_datasets.append(fake_dataset)
    data = SimpleNamespace(
        datasets=specs,
        mode="single_turn",
        modality_configs={EMBODIMENT_TAG: {}},
        video_backend="torchcodec",
        shard_size=128,
        episode_sampling_rate=0.1,
        seed=42,
        allow_padding=False,
        num_shards_per_epoch=1,
        override_pretraining_statistics=True,
    )
    config = SimpleNamespace(data=data, training=SimpleNamespace(eval_strategy="no"))
    mixture = object()
    processor = MagicMock()
    fake_store = MagicMock()
    fake_store.keys = frozenset(
        EpisodeBundleKey(uid, CODEC_CAMERA_ID, SOURCE_KIND_NATIVE_CANONICAL)
        for uid in identities_by_uid
    )
    fake_store.identity.side_effect = lambda key: {
        "episode_identity": {
            name: value
            for name, value in identities_by_uid[key.episode_uid].items()
            if name != "episode_uid"
        },
        "codec_contract": {"fixture_mode": PRODUCTION_FIXTURE_MODE},
    }
    with (
        patch(
            "gr00t.data.dataset.g1_all_lerobot_h40_adapter.G1AllLerobotH40SingleStepDataset",
            side_effect=fake_datasets,
        ) as adapter_dataset,
        patch(
            "gr00t.data.codec_evidence.dataset.maybe_wrap_codec_evidence_dataset",
            side_effect=lambda dataset, _path, **_kwargs: dataset,
        ) as codec_wrapper,
        patch(
            "gr00t.data.codec_evidence.dataset.episode_bundle_store_from_index_manifest",
            return_value=fake_store,
        ) as store_constructor,
        patch(
            "gr00t.data.dataset.factory.ShardedMixtureDataset", return_value=mixture
        ) as mixture_constructor,
    ):
        result, evaluation = DatasetFactory(config).build(processor)
    assert result is mixture and evaluation is None
    assert adapter_dataset.call_count == 5
    assert codec_wrapper.call_count == 5
    store_constructor.assert_called_once()
    assert all(call.kwargs["store"] is fake_store for call in codec_wrapper.call_args_list)
    wrapped_configs = [call.kwargs["config"] for call in codec_wrapper.call_args_list]
    assert all(config is wrapped_configs[0] for config in wrapped_configs)
    assert wrapped_configs[0].fixture_manifest_sha256 == block["fixture_manifest_sha256"]
    mixture_constructor.assert_called_once()
    mixture_kwargs = mixture_constructor.call_args.kwargs
    assert mixture_kwargs["datasets"] == fake_datasets
    assert mixture_kwargs["weights"] == pytest.approx(mix_ratios)
    assert mixture_kwargs["processor"] is processor
    assert mixture_kwargs["seed"] == 42
    assert mixture_kwargs["training"] is True
    assert mixture_kwargs["override_pretraining_statistics"] is True


def test_codec_wrapper_delegates_leaf_statistics_and_processor_without_drift(tmp_path: Path):
    from gr00t.data.codec_evidence.config import parse_codec_evidence_config
    from gr00t.data.codec_evidence.dataset import CodecEvidenceAugmentedDataset
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset

    leaf_stats = {
        "state": {"joint": _stats(1)},
        "action": {"joint": _stats(1)},
    }
    leaf = SimpleNamespace(
        embodiment_tag=SimpleNamespace(value=EMBODIMENT_TAG),
        sharded_episodes=[],
        get_dataset_statistics=MagicMock(return_value=leaf_stats),
        set_processor=MagicMock(),
    )
    wrapped = CodecEvidenceAugmentedDataset(
        leaf,
        parse_codec_evidence_config(_codec_block(tmp_path / "warehouse")),
        store=object(),
    )
    processor = MagicMock()
    mixture = ShardedMixtureDataset.__new__(ShardedMixtureDataset)
    mixture.datasets = [wrapped]
    mixture.weights = [1.0]
    mixture.processor = processor
    mixture.override_pretraining_statistics = True

    mixture.merge_statistics()

    leaf.get_dataset_statistics.assert_called_once_with()
    leaf.set_processor.assert_called_once_with(processor)
    processor.set_statistics.assert_called_once_with(
        {EMBODIMENT_TAG: leaf_stats},
        override=True,
    )


def test_factory_rejects_missing_extra_or_wrong_codec_episode_cohort(tmp_path: Path):
    from gr00t.data.codec_evidence.codec_contract import (
        PRODUCTION_FIXTURE_MODE,
        SYNTHETIC_FIXTURE_MODE,
    )
    from gr00t.data.codec_evidence.config import parse_codec_evidence_config
    from gr00t.data.codec_evidence.episode_bundle import EpisodeBundleKey
    from gr00t.data.dataset.factory import _validate_all_lerobot_codec_store_coverage

    config = parse_codec_evidence_config(_codec_block(tmp_path / "warehouse"))
    identity = _codec_episode_identity("selected-source")
    identities = {identity["episode_uid"]: identity}
    expected_key = EpisodeBundleKey(identity["episode_uid"], config.camera_id, config.source_kind)
    extra_identity = _codec_episode_identity("unselected-source", 2)
    extra_key = EpisodeBundleKey(
        extra_identity["episode_uid"], config.camera_id, config.source_kind
    )

    missing_store = SimpleNamespace(keys=frozenset(), identity=MagicMock())
    with pytest.raises(ValueError, match="missing=1, extra=0"):
        _validate_all_lerobot_codec_store_coverage(config, identities, missing_store)

    extra_store = SimpleNamespace(keys=frozenset({expected_key, extra_key}), identity=MagicMock())
    with pytest.raises(ValueError, match="missing=0, extra=1"):
        _validate_all_lerobot_codec_store_coverage(config, identities, extra_store)

    wrong_identity = dict(identity)
    wrong_identity["source_id"] = "wrong-source"
    wrong_identity.pop("episode_uid")
    wrong_store = SimpleNamespace(
        keys=frozenset({expected_key}),
        identity=MagicMock(
            return_value={
                "episode_identity": wrong_identity,
                "codec_contract": {"fixture_mode": PRODUCTION_FIXTURE_MODE},
            }
        ),
    )
    with pytest.raises(ValueError, match="identity differs from adapter"):
        _validate_all_lerobot_codec_store_coverage(config, identities, wrong_store)

    synthetic_store = SimpleNamespace(
        keys=frozenset({expected_key}),
        identity=MagicMock(
            return_value={
                "episode_identity": {
                    name: value for name, value in identity.items() if name != "episode_uid"
                },
                "codec_contract": {"fixture_mode": SYNTHETIC_FIXTURE_MODE},
            }
        ),
    )
    with pytest.raises(ValueError, match="requires production Codec fixture mode"):
        _validate_all_lerobot_codec_store_coverage(config, identities, synthetic_store)
