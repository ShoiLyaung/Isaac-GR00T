from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from safetensors.torch import save_file
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "examples/G1_ALL_LEROBOT_CODEC/launch_c5_causal_codec_only.py"
SPEC = importlib.util.spec_from_file_location("c5_causal_training_supervisor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
causal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = causal
SPEC.loader.exec_module(causal)


def test_causal_launcher_requires_manifest_and_preserves_historical_schedule(tmp_path):
    assert causal.FIXTURE_ROOT == Path(
        "/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r2"
    )
    assert causal.RETAINED_CHECKPOINT == causal.history.RETAINED_CHECKPOINT
    assert causal.CHECKPOINT_HASHES == causal.history.CHECKPOINT_HASHES
    assert causal.EXPECTED_PARAMETER_NAMES == causal.history.EXPECTED_PARAMETER_NAMES
    assert causal.EXPECTED_TRAINABLE_NUMEL == 9_571_649
    assert causal.MODES == causal.history.MODES
    with pytest.raises(SystemExit):
        causal.parse_args(
            [
                "plan",
                "--mode",
                "profile",
                "--run-dir",
                str(tmp_path / "run"),
                "--preparation-root",
                str(tmp_path / "prep"),
                "--expected-source-commit",
                "a" * 40,
            ]
        )

    manifest = tmp_path / "DATASET_MANIFEST.json"
    for mode_name in ("profile", "pilot", "formal"):
        mode = causal.MODES[mode_name]
        actual = causal.build_train_command(
            mode=mode,
            dataset_manifest=manifest,
            train_output=tmp_path / mode_name,
            master_port=12345,
        )
        expected = causal.history.build_train_command(
            mode=causal.history.MODES[mode_name],
            dataset_manifest=manifest,
            train_output=tmp_path / mode_name,
            master_port=12345,
        )
        assert actual == expected
        assert actual[actual.index("--base-model-path") + 1] == str(causal.RETAINED_CHECKPOINT)
        assert actual[actual.index("--codec-trainability") + 1] == "codec_only"
        assert actual[actual.index("--seed") + 1] == "42"
        assert "resume" not in " ".join(actual).lower()


def _rank_evidence(initial_state_sha256: str) -> dict:
    return {
        "complete": True,
        "codec_state_closure_complete": True,
        "codec_ranks": [
            {
                "initial_state_sha256": initial_state_sha256,
                "initial_gate": 0.0,
            }
            for _rank in range(2)
        ],
    }


def test_causal_posttrain_requires_frozen_initial_codec_state(monkeypatch, tmp_path):
    accepted = _rank_evidence(causal.EXPECTED_INITIAL_CODEC_STATE_SHA256)
    monkeypatch.setattr(
        causal.history,
        "_codec_posttrain_evidence",
        lambda *_args, **_kwargs: copy.deepcopy(accepted),
    )
    result = causal._codec_posttrain_evidence(tmp_path, 2, list(causal.SOURCE_IDS))
    assert result["complete"] is True
    assert result["initial_codec_state_closure_complete"] is True
    assert all(rank["frozen_initial_state_match"] for rank in result["codec_ranks"])

    drifted = _rank_evidence("0" * 64)
    monkeypatch.setattr(
        causal.history,
        "_codec_posttrain_evidence",
        lambda *_args, **_kwargs: copy.deepcopy(drifted),
    )
    result = causal._codec_posttrain_evidence(tmp_path, 2, list(causal.SOURCE_IDS))
    assert result["complete"] is False
    assert result["codec_state_closure_complete"] is False
    assert result["initial_codec_state_closure_complete"] is False


def test_retained_pretrain_index_requires_1031_native_and_zero_codec_keys(tmp_path):
    retained = tmp_path / "retained"
    retained.mkdir()
    native_names = [f"native.parameter.{index:04d}" for index in range(1_031)]
    index_path = retained / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in native_names}})
    )

    identity = causal.validate_retained_native_index(retained)

    assert identity["complete"] is True
    assert identity["native_tensor_count"] == 1_031
    assert identity["codec_tensor_count"] == 0

    index_path.write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in native_names[:-1]}})
    )
    with pytest.raises(RuntimeError, match="exactly 1031 native keys"):
        causal.validate_retained_native_index(retained)

    contaminated = native_names[:-1] + ["action_head.codec_conditioning.conditioner.gate"]
    index_path.write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in contaminated}})
    )
    with pytest.raises(RuntimeError, match="zero Codec keys"):
        causal.validate_retained_native_index(retained)


def _write_checkpoint(root: Path, tensors: dict[str, torch.Tensor]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    shard = "model.safetensors"
    save_file(tensors, root / shard)
    causal.base.atomic_json(
        root / "model.safetensors.index.json",
        {"weight_map": {name: shard for name in tensors}},
    )


def test_native_tensor_closure_requires_value_and_byte_identity(tmp_path, monkeypatch):
    native_names = ("native.a", "native.b")
    codec_name = "action_head.codec_conditioning.conditioner.gate"
    retained = tmp_path / "retained"
    target = tmp_path / "target"
    native_tensors = {
        native_names[0]: torch.tensor([1.0, -0.0], dtype=torch.float32),
        native_names[1]: torch.tensor([2, 3], dtype=torch.int64),
    }
    _write_checkpoint(retained, native_tensors)
    _write_checkpoint(
        target,
        {
            **native_tensors,
            codec_name: torch.tensor(0.5, dtype=torch.float32),
        },
    )
    monkeypatch.setattr(causal, "RETAINED_CHECKPOINT", retained)
    monkeypatch.setattr(causal, "EXPECTED_NATIVE_TENSOR_COUNT", 2)
    monkeypatch.setattr(causal, "EXPECTED_PARAMETER_NAMES", (codec_name,))

    closure = causal._native_tensor_closure(target)

    assert closure["complete"] is True
    assert closure["key_closure_complete"] is True
    assert closure["torch_equal_complete"] is True
    assert closure["byte_equal_complete"] is True
    assert closure["retained_native_state_sha256"] == closure["target_native_state_sha256"]

    _write_checkpoint(
        tmp_path / "drifted",
        {
            native_names[0]: torch.tensor([1.0, 0.0], dtype=torch.float32),
            native_names[1]: native_tensors[native_names[1]],
            codec_name: torch.tensor(0.5, dtype=torch.float32),
        },
    )
    closure = causal._native_tensor_closure(tmp_path / "drifted")
    assert closure["complete"] is False
    assert closure["torch_equal_complete"] is True
    assert closure["byte_equal_complete"] is False
    assert closure["byte_mismatch_names"] == [native_names[0]]


def test_saved_checkpoint_cannot_pass_without_native_closure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        causal.history,
        "_saved_codec_checkpoint",
        lambda *_args, **_kwargs: {"complete": True},
    )
    monkeypatch.setattr(
        causal,
        "_native_tensor_closure",
        lambda _root: {"complete": False, "byte_equal_complete": False},
    )
    result = causal._saved_codec_checkpoint(tmp_path, "a" * 64)
    assert result["complete"] is False
    assert result["native_tensor_closure"]["complete"] is False
