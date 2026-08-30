from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_module():
    path = Path(__file__).parents[2] / "examples/G1_ALL_LEROBOT_CODEC/run_c4_gpu_parity.py"
    spec = importlib.util.spec_from_file_location("c4_gpu_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_codec_model_config_is_fixture_free_zero_gate_exact_surface():
    module = _load_module()
    assert module.MODEL_CODEC_CONFIG == {
        "enabled": True,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "d_codec": 64,
        "conditioner_heads": 8,
    }
    assert "fixture_root" not in module.MODEL_CODEC_CONFIG
    assert "fixture_manifest_sha256" not in module.MODEL_CODEC_CONFIG


def test_native_batch_comparison_allows_only_nested_codec_evidence():
    module = _load_module()
    native = {"state": torch.arange(4), "action": torch.ones((1, 2))}
    codec = {**native, "codec_evidence": {"sentinel": torch.ones(1)}}
    module._assert_native_batch_equal(native, codec)

    with pytest.raises(RuntimeError, match="beyond codec_evidence"):
        module._assert_native_batch_equal(native, {**codec, "unexpected": torch.ones(1)})
    with pytest.raises(RuntimeError, match="native preprocessed tensor differs"):
        module._assert_native_batch_equal(native, {**codec, "state": torch.arange(4) + 1})


def test_loading_info_accepts_only_explicit_complete_codec_missing_keys():
    module = _load_module()
    allowed = {
        "action_head.codec_conditioning.core.weight",
        "action_head.codec_conditioning.projector.bias",
    }
    result = module._loading_failures(
        {
            "missing_keys": sorted(allowed),
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        allowed,
    )
    assert set(result["allowed_missing"]) == allowed

    with pytest.raises(RuntimeError, match="loading mismatch"):
        module._loading_failures(
            {"missing_keys": ["native.weight"], "unexpected_keys": []},
            allowed,
        )


def test_tensor_tree_identity_is_exact_and_changes_with_value():
    module = _load_module()
    first = module._tensor_tree_identity({"x": torch.tensor([[1.0, 2.0]])})
    second = module._tensor_tree_identity({"x": torch.tensor([[1.0, 3.0]])})
    assert first["x"]["shape"] == [1, 2]
    assert first["x"]["dtype"] == "torch.float32"
    assert first["x"]["sha256"] != second["x"]["sha256"]


def test_gpu_inventory_requires_one_numeric_selector(monkeypatch):
    module = _load_module()

    class Result:
        stdout = "0, GPU-a, RTX, 49140, 100, 49040, 0\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Result())
    assert module._gpu_inventory("0")["memory_free_mib"] == 49040
    with pytest.raises(ValueError, match="one physical numeric GPU"):
        module._gpu_inventory("0,1")
