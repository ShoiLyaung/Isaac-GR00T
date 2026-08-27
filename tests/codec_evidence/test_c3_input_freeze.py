from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gr00t.data.codec_evidence.identity import episode_uid_from_preimage
from gr00t.data.codec_evidence.production import sha256_path
import numpy as np
import pandas as pd
import pytest


def _load_freezer():
    path = Path(__file__).parents[2] / "examples/G1_ALL_LEROBOT_CODEC/freeze_c3_inputs.py"
    spec = importlib.util.spec_from_file_location("c3_input_freezer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_temporary_one_episode_freeze_is_exact_and_non_overwritable(tmp_path, monkeypatch):
    module = _load_freezer()
    repository = tmp_path / "repository"
    dataset_root = tmp_path / "dataset"
    source_root = dataset_root / "source"
    (source_root / "data/chunk-000").mkdir(parents=True)
    (source_root / "videos/chunk-000/observation.images.cam_left_high").mkdir(parents=True)
    rows = np.arange(40, dtype=np.int64)
    parquet = source_root / "data/chunk-000/episode_000000.parquet"
    pd.DataFrame(
        {
            "timestamp": (rows.astype(np.float64) / 30.0).astype(np.float32),
            "frame_index": rows,
            "episode_index": np.zeros(40, dtype=np.int64),
            "task_index": np.zeros(40, dtype=np.int64),
            "annotation.human.validity": np.ones(40, dtype=np.int64),
        }
    ).to_parquet(parquet, index=False)
    video = source_root / "videos/chunk-000/observation.images.cam_left_high/episode_000000.mp4"
    video.write_bytes(b"focused-video-identity")
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
    }
    modality = {"video": {"cam_left_high": {"original_key": "cam_left_high"}}}
    _write_json(source_root / "meta/info.json", info)
    _write_json(source_root / "meta/modality.json", modality)
    (source_root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 40}) + "\n", encoding="utf-8"
    )
    recipe = {
        "dataset_root": str(dataset_root),
        "total_h40_anchors": 1,
        "datasets": [
            {
                "source_id": "source",
                "relative_path": "source",
                "expected_info_sha256": sha256_path(source_root / "meta/info.json"),
                "expected_modality_sha256": sha256_path(source_root / "meta/modality.json"),
                "episodes": 1,
                "frames": 40,
                "h40_anchors": 1,
            }
        ],
    }
    recipe_path = repository / module.RECIPE_RELATIVE
    _write_json(recipe_path, recipe)
    monkeypatch.setattr(module, "RECIPE_SHA256", sha256_path(recipe_path))
    monkeypatch.setattr(module, "EXPECTED_EPISODES", 1)
    monkeypatch.setattr(module, "EXPECTED_FRAMES", 40)
    monkeypatch.setattr(
        module,
        "_probe_video",
        lambda _path, expected: {
            "codec_name": "h264",
            "nb_frames": str(expected),
            "nb_read_frames": str(expected),
        },
    )
    monkeypatch.setattr(
        module,
        "_c2_basis",
        lambda: (
            {
                "peak_rss_bytes": 100,
                "frames": 40,
                "output_bytes_before_terminal_receipts": 1000,
            },
            {
                "source": {
                    "c2_frames": 40,
                    "c2_allocated_output_bytes": 1000,
                    "bytes_per_frame": 25.0,
                    "raw_mv_per_frame": 2.0,
                    "wall_seconds_per_frame": 0.1,
                }
            },
        ),
    )
    frozen = module.build_freeze(repository)
    assert frozen["episode_count"] == 1
    assert frozen["total_frames"] == 40
    assert frozen["total_codec_anchors"] == 2
    assert frozen["episodes"][0]["episode_uid"] == episode_uid_from_preimage(
        frozen["episodes"][0]["episode_identity"]
    )
    content = dict(frozen)
    digest = content.pop("content_sha256")
    assert digest == module.canonical_json_sha256(content)
    output = tmp_path / "freeze.json"
    module._write_new(output, frozen)
    with pytest.raises(FileExistsError):
        module._write_new(output, frozen)


def test_contained_payload_rejects_symlink_and_parent_escape(tmp_path):
    module = _load_freezer()
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (root / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="may not be a symlink"):
        module._contained_payload(root, "linked")
    with pytest.raises(ValueError, match="escapes source root"):
        module._contained_payload(root, "../outside")


def test_freeze_output_is_fixed_inside_repository(tmp_path):
    module = _load_freezer()
    repository = tmp_path / "repository"
    expected = repository / module.OUTPUT_RELATIVE
    expected.parent.mkdir(parents=True)
    assert module._validate_output_path(repository, module.OUTPUT_RELATIVE) == expected
    with pytest.raises(ValueError, match="output is fixed"):
        module._validate_output_path(repository, tmp_path / "dataset/freeze.json")
    symlink_parent = tmp_path / "linked-parent"
    symlink_parent.symlink_to(expected.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="existing real directory"):
        module._write_new(symlink_parent / "freeze.json", {"value": 1})
