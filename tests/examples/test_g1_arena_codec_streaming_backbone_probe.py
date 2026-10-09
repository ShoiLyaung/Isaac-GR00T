from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/G1_ARENA_N17"
sys.path.insert(0, str(EXAMPLE))
import codec_streaming_backbone_probe as probe  # noqa: E402


def _write_archive(path: Path, *, extra_field: bool = False) -> None:
    codec = np.zeros(probe.INPUT_SHAPE, dtype=np.float32)
    rgb = np.full((925, 1, 8, 8, 3), np.nan, dtype=np.float32)
    payload = {
        "codec_v0": codec + 1,
        "codec_v1": codec,
        "decoded_rgb_teacher": rgb,
        "reconstruction_v0": rgb,
        "reconstruction_v1": rgb,
    }
    if extra_field:
        payload["action_teacher"] = np.zeros((925, 1), dtype=np.float32)
    np.savez_compressed(path, **payload)


def test_loader_consumes_only_codec_v1_and_reports_excluded_fields(tmp_path: Path) -> None:
    path = tmp_path / "codec.npz"
    _write_archive(path)
    codec, metadata = probe.load_codec_input(
        path,
        expected_bytes=path.stat().st_size,
        expected_sha256=probe.sha256(path),
    )
    assert codec.shape == probe.INPUT_SHAPE
    assert codec.dtype == np.float32
    assert np.count_nonzero(codec) == 0
    assert metadata["fields_consumed_by_backbone"] == ["codec_v1"]
    assert set(metadata["fields_excluded_from_backbone"]) == {
        "codec_v0",
        "decoded_rgb_teacher",
        "reconstruction_v0",
        "reconstruction_v1",
    }


def test_loader_rejects_archive_field_leakage(tmp_path: Path) -> None:
    path = tmp_path / "codec-extra.npz"
    _write_archive(path, extra_field=True)
    with pytest.raises(ValueError, match="field whitelist"):
        probe.load_codec_input(
            path,
            expected_bytes=path.stat().st_size,
            expected_sha256=probe.sha256(path),
        )


def _passing_gate_inputs():
    numerical = {
        "outputs": {
            name: {"shape": [1, 925, 256], "finite": True}
            for name in ("mamba2", "gru", "transformer", "identity")
        },
        "parity": {
            name: {
                "full_vs_step_max_abs": 9e-5,
                "full_vs_chunked_max_abs": 9e-5,
            }
            for name in ("mamba2", "gru")
        },
        "causality": {
            name: {"suffix_edit_prefix_max_abs": 0.0}
            for name in ("mamba2", "gru", "transformer", "identity")
        },
        "reset": {name: {"exact": True} for name in ("mamba2", "gru")},
        "mamba_state": {
            "serialized_growth_fraction": 0.0,
            "allocated_second_half_growth_bytes": 0,
        },
    }
    parameters = {
        "mamba2": 864_560,
        "gru": 790_528,
        "transformer": 857_216,
        "identity": 0,
    }
    mamba_peaks = (100, 200, 300, 400)
    transformer_peaks = (200, 420, 660, 1_000)
    cost = {
        "mamba2": {
            str(length): {"peak_delta_bytes": peak}
            for length, peak in zip(probe.LENGTHS, mamba_peaks, strict=True)
        },
        "transformer": {
            str(length): {"peak_delta_bytes": peak}
            for length, peak in zip(probe.LENGTHS, transformer_peaks, strict=True)
        },
        "mamba2_streaming_step": {"p95_ms": 4.999},
    }
    return numerical, parameters, cost


def _material(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": probe.sha256(path),
    }


def _valid_plan(output_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": probe.EXPERIMENT,
        "run_id": probe.RUN_ID,
        "source": {
            "root": str(probe.REPO),
            "commit": "1" * 40,
            "base_commit": probe.BASE_SOURCE_COMMIT,
        },
        "input": {
            "path": str(probe.INPUT_PATH),
            "bytes": probe.INPUT_BYTES,
            "sha256": probe.INPUT_SHA256,
            "field": probe.INPUT_FIELD,
            "shape": list(probe.INPUT_SHAPE),
            "dtype": "float32",
        },
        "dependency": {
            "manifest": _material(probe.DEPENDENCY_MANIFEST_PATH),
            "mamba_file": _material(probe.MAMBA2_FILE_PATH),
        },
        "runtime": copy.deepcopy(probe.FROZEN_RUNTIME),
        "protocol": copy.deepcopy(probe.FROZEN_PROTOCOL),
        "thresholds": copy.deepcopy(probe.FROZEN_THRESHOLDS),
        "resources": copy.deepcopy(probe.FROZEN_RESOURCES),
        "output": {
            "root": str(output_root / probe.RUN_ID),
            "result_file": "RESULT.json",
        },
    }


def _write_plan(path: Path, plan: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return probe.sha256(path)


def test_gate_evaluation_preserves_frozen_thresholds() -> None:
    numerical, parameters, cost = _passing_gate_inputs()
    gates, derived = probe.evaluate_gates(numerical, parameters, cost)
    assert all(gates.values())
    assert derived["memory_gate_metric"] == "peak_delta_bytes_above_warmed_baseline"
    assert derived["transformer_over_mamba_peak_delta_ratio"]["925"] == 2.5

    boundary = copy.deepcopy(cost)
    boundary["mamba2_streaming_step"]["p95_ms"] = 5.0
    gates, _ = probe.evaluate_gates(numerical, parameters, boundary)
    assert not gates["mamba_stream_step_p95_below_5ms"]

    parity_failure = copy.deepcopy(numerical)
    parity_failure["parity"]["mamba2"]["full_vs_step_max_abs"] = 1.00001e-4
    gates, _ = probe.evaluate_gates(parity_failure, parameters, cost)
    assert not gates["mamba_gru_full_step_chunk_parity"]


def test_plan_binds_runtime_protocol_dependencies_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "ARTIFACT_ROOT", tmp_path)
    plan = _valid_plan(tmp_path)
    plan_path = tmp_path / "resolved-plan-r1/PLAN.json"
    plan_sha256 = _write_plan(plan_path, plan)
    loaded, identity = probe.verify_plan(plan_path, plan_sha256)
    assert loaded == plan
    assert identity["sha256"] == plan_sha256
    assert identity["dependency_manifest_sha256"] == plan["dependency"]["manifest"]["sha256"]

    changed = copy.deepcopy(plan)
    changed["thresholds"]["full_step_chunk_max_abs"] = 2e-4
    changed_path = tmp_path / "changed-plan/PLAN.json"
    changed_sha256 = _write_plan(changed_path, changed)
    with pytest.raises(ValueError, match="thresholds changed"):
        probe.verify_plan(changed_path, changed_sha256)


def test_runner_is_vla_free_and_has_no_optimizer_surface() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "Gr00tN1d7Pipeline" not in source
    assert "from_pretrained" not in source
    assert "torch.optim" not in source
    assert not any(name == "gr00t.model" or name.startswith("gr00t.model.") for name in imported)
    assert probe.MAX_WALL_SECONDS == 1800
    assert probe.MAX_ARTIFACT_BYTES == 1 << 30


def test_kernel_event_filter_excludes_memcpy_and_memset() -> None:
    class Event:
        def __init__(self, device_type: str, name: str):
            self.device_type = device_type
            self.name = name

    assert probe._is_cuda_kernel_event(Event("DeviceType.CUDA", "void causal_conv1d"))
    assert not probe._is_cuda_kernel_event(Event("DeviceType.CUDA", "Memcpy HtoD"))
    assert not probe._is_cuda_kernel_event(Event("DeviceType.CUDA", "Memset (Device)"))
    assert not probe._is_cuda_kernel_event(Event("DeviceType.CPU", "cudaLaunchKernel"))


def test_write_new_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "RESULT.json"
    assert probe.write_new(path, {"status": "passed"}) > 0
    with pytest.raises(FileExistsError):
        probe.write_new(path, {"status": "changed"})
