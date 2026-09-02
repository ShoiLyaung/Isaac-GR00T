from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.data.codec_evidence.full_episode import DatasetFactoryCanonicalDecoder
import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "examples/G1_ID10_H40_NATIVE30/build_codec_full_episode_canary.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("c1_full_episode_canary_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c1_plan_is_fixed_to_one_cpu_attempt_and_three_authorized_uids():
    runner = _load_runner()
    assert runner.ATTEMPT_ID.endswith("-ATTEMPT1")
    assert len(runner.AUTHORIZED_EPISODES) == 3
    assert len({row["episode_uid"] for row in runner.AUTHORIZED_EPISODES}) == 3
    assert [row["num_frames"] for row in runner.AUTHORIZED_EPISODES] == [1050, 635, 524]
    assert runner.MAX_WALL_SECONDS == 10_800
    assert runner.MAX_PERSISTENT_BYTES == runner.MAX_SCRATCH_BYTES == 4 * 1024**3


def test_failed_generation_guard_preflight_creates_no_attempt_or_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = _load_runner()
    warehouse = tmp_path / "warehouse"
    scratch = tmp_path / "scratch"

    def fail(*_args, **_kwargs):
        raise RuntimeError("guard unavailable")

    monkeypatch.setattr(runner.DatasetGenerationGuard, "acquire_for_manifest", fail)
    args = argparse.Namespace(
        warehouse_root=warehouse,
        scratch_root=scratch,
        split_manifest=tmp_path / "split.json",
        train_statistics=tmp_path / "stats.json",
    )
    with pytest.raises(RuntimeError, match="guard unavailable"):
        runner.run_canary(args, enforce_authorized_paths=False)
    assert not warehouse.exists()
    assert not scratch.exists()


class _FakeGuard:
    def __init__(self):
        self.ensure_calls = 0
        self.verify_calls = 0

    def ensure_current_process(self):
        self.ensure_calls += 1

    def verify_frozen_identity(self):
        self.verify_calls += 1


class _Holder:
    dataset_generation_guard = None


def test_dataset_factory_decoder_binds_same_lifetime_guard_to_loader_and_dataset():
    decoder = DatasetFactoryCanonicalDecoder.__new__(DatasetFactoryCanonicalDecoder)
    decoder.single_dataset = _Holder()
    decoder.collection = _Holder()
    decoder._bound_generation_guard = None
    decoder._prior_dataset_guard = None
    decoder._prior_collection_guard = None
    guard = _FakeGuard()

    decoder.bind_generation_guard(guard)
    assert decoder.single_dataset.dataset_generation_guard is guard
    assert decoder.collection.dataset_generation_guard is guard
    decoder.unbind_generation_guard(guard)
    assert decoder.single_dataset.dataset_generation_guard is None
    assert decoder.collection.dataset_generation_guard is None
    assert guard.ensure_calls == 1
    assert guard.verify_calls == 2


def test_attempt_started_precedes_scratch_and_one_toolchain_is_reused():
    runner = _load_runner()
    source = inspect.getsource(runner.run_canary)
    assert source.index("journal = AttemptJournal") < source.index("scratch.mkdir")
    assert source.index("tools = toolchain_identity") < source.index("journal = AttemptJournal")
    assert "tool_identity=tools" in source
    assert "tool_identity=None" not in source


def test_implementation_closure_includes_resolved_configs_and_regression_tests(
    tmp_path: Path,
):
    runner = _load_runner()
    paths = {}
    for name in ("tool", "libx264", "adapter", "modality", "launch", "ffmpeg", "ffprobe"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    from gr00t.data.codec_evidence.warehouse import sha256_path

    toolchain = {
        "pyav": {"path": str(paths["tool"]), "sha256": sha256_path(paths["tool"])},
        "ffmpeg": {
            "path": str(paths["ffmpeg"]),
            "sha256": sha256_path(paths["ffmpeg"]),
            "loaded_libx264": {
                "path": str(paths["libx264"]),
                "sha256": sha256_path(paths["libx264"]),
            },
        },
    }
    args = argparse.Namespace(
        adapter_config_path=paths["adapter"],
        modality_config_path=paths["modality"],
        launch_finetune=paths["launch"],
        ffmpeg=paths["ffmpeg"],
        ffprobe=paths["ffprobe"],
    )
    files = runner.expected_implementation_files(args, toolchain)
    assert files["adapter_config_input"]["path"] == str(paths["adapter"])
    assert files["modality_config_input"]["path"] == str(paths["modality"])
    assert "test.cpu_adapter" in files
    assert "test.strict_load_contract" in files
    assert "test.codec_evidence.test_full_episode_warehouse" in files
    assert "outer_supervisor" in files
    assert files["outer.python"]["path"] == str(Path(sys.executable).resolve())
    assert files["outer.bwrap"]["path"] == "/usr/bin/bwrap"
    assert files["outer.timeout"]["path"] == "/usr/bin/timeout"
    assert files["outer.time"]["path"] == "/usr/bin/time"
    assert "oracle.legacy_300_compatibility" in files
    assert "oracle.split_loader_cpu" in files
    assert files["toolchain.pyav"]["path"] == str(paths["tool"])
    assert files["toolchain.ffmpeg.loaded_libx264"]["path"] == str(paths["libx264"])


def test_adapter_manifest_path_mismatch_is_rejected_before_dataset_construction(tmp_path: Path):
    runner = _load_runner()
    split = tmp_path / "split.json"
    wrong_split = tmp_path / "wrong-split.json"
    stats = tmp_path / "stats.json"
    for path in (split, wrong_split, stats):
        path.write_text("{}\n", encoding="utf-8")
    adapter = tmp_path / "adapter.json"
    adapter.write_text(
        json.dumps(
            {
                "adapter_kind": "g1_id10_h40_native30_v1",
                "split": "train",
                "embodiment_id": 10,
                "embodiment_tag": "new_embodiment",
                "horizon": 40,
                "active_action_dims": 26,
                "split_manifest_path": str(wrong_split),
                "derived_relative_stats_path": str(stats),
            }
        ),
        encoding="utf-8",
    )
    guard = SimpleNamespace(
        identity=SimpleNamespace(
            split_manifest_path=split.resolve(),
            train_statistics_path=stats.resolve(),
        )
    )
    args = argparse.Namespace(adapter_config_path=adapter)
    with pytest.raises(ValueError, match="split_manifest_path differs"):
        runner.validate_adapter_launch_cross_binding(
            args,
            guard,
            SimpleNamespace(),
            tmp_path,
        )


def test_shifted_but_length_40_action_offsets_are_rejected():
    runner = _load_runner()
    config = SimpleNamespace(
        data=SimpleNamespace(
            modality_configs={
                "new_embodiment": {
                    name: SimpleNamespace(delta_indices=delta)
                    for name, delta in {
                        "video": [0],
                        "state": [0],
                        "action": list(range(5, 45)),
                        "language": [0],
                    }.items()
                }
            }
        ),
        model=SimpleNamespace(max_state_dim=132, max_action_dim=132),
    )
    with pytest.raises(ValueError, match="exact H40/132-D"):
        runner.validate_resolved_h40_modality(config)
