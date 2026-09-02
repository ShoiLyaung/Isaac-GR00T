from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "examples/G1_ALL_LEROBOT_H40/reload_smoke.py"
SPEC = importlib.util.spec_from_file_location("g1_all_lerobot_reload_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint-20000"
    checkpoint.mkdir()
    _json(
        checkpoint / "config.json",
        {
            "architectures": ["Gr00tN1d7"],
            "action_horizon": 40,
            "tune_visual": True,
        },
    )
    _json(
        checkpoint / "model.safetensors.index.json",
        {"weight_map": {"a": "model-00001-of-00001.safetensors"}},
    )
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    _json(
        checkpoint / "processor_config.json",
        {"processor_kwargs": {"max_action_horizon": 40}},
    )
    _json(checkpoint / "statistics.json", {"new_embodiment": {}})
    _json(checkpoint / "embodiment_id.json", {"new_embodiment": 10})

    view = tmp_path / "view"
    for name in ("data", "videos"):
        (view / name).mkdir(parents=True)
    for name in ("info.json", "modality.json"):
        _json(view / "meta" / name, {})
    (view / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "insert the shaft into the housing"}) + "\n",
        encoding="utf-8",
    )
    shared = {
        "source_id": "plain_300",
        "canonical_root": "/canonical/plain",
        "canonical_info_sha256": "1" * 64,
        "canonical_modality_sha256": "2" * 64,
        "raw_task_labels": ["insert the shaft into the housing"],
        "training_language": "insert the shaft into the housing",
    }
    _json(
        view / "meta/g1_all_lerobot_h40_view_manifest.json",
        {**shared, "camera_mapping": {"model_key": "ego_view"}},
    )
    adapter = tmp_path / "ADAPTER_CONFIG.json"
    _json(
        adapter,
        {
            **shared,
            "adapter_kind": "g1_all_lerobot_h40_v1",
            "embodiment_tag": "new_embodiment",
            "embodiment_id": 10,
            "horizon": 40,
            "active_action_dims": 26,
        },
    )
    output = tmp_path / "output"
    output.mkdir()
    return checkpoint, view, adapter, output


def _argv(inputs: tuple[Path, Path, Path, Path], action: str = "plan") -> list[str]:
    checkpoint, view, adapter, output = inputs
    return [
        action,
        "--checkpoint",
        str(checkpoint),
        "--loader-view",
        str(view),
        "--adapter-config",
        str(adapter),
        "--output-dir",
        str(output),
    ]


def _action(fill: float = 0.25) -> dict[str, np.ndarray]:
    return {
        key: np.full((1, 40, dim), fill, dtype=np.float32) for key, dim in smoke.ACTION_DIMS.items()
    }


def test_plan_writes_dry_run_without_loading_runtime(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(smoke, "execute", lambda plan: pytest.fail("plan touched runtime"))

    assert smoke.main(_argv(inputs)) == 0

    result = json.loads((inputs[3] / smoke.SUMMARY_NAME).read_text())
    assert result["status"] == "DRY_RUN_READY"
    assert result["gpu_or_model_touched"] is False
    assert result["policy_get_action_calls"] == 0
    assert result["checkpoint_inventory"]["action_horizon"] == 40


def test_plan_rejects_checkpoint_without_declared_shard(tmp_path):
    inputs = _inputs(tmp_path)
    (inputs[0] / "model-00001-of-00001.safetensors").unlink()

    assert smoke.main(_argv(inputs)) == 1

    result = json.loads((inputs[3] / smoke.SUMMARY_NAME).read_text())
    assert result["status"] == "FAIL"
    assert result["technical_valid"] is False
    assert result["error_type"] == "FileNotFoundError"


def test_call_policy_once_reaches_public_boundary_exactly_once():
    calls = []

    class Policy:
        def get_action(self, observation):
            calls.append(observation)
            return _action(), {"request": 1}

    step = SimpleNamespace(
        images={"ego_view": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        states={"left_arm": np.zeros((1, 7), dtype=np.float32)},
        text="insert the shaft into the housing",
    )
    action, info = smoke.call_policy_once(Policy(), step)

    assert len(calls) == 1
    assert info == {"request": 1}
    assert calls[0]["video"]["ego_view"].shape == (1, 1, 8, 8, 3)
    assert calls[0]["state"]["left_arm"].shape == (1, 1, 7)
    assert calls[0]["language"] == {
        "annotation.human.task_description": [["insert the shaft into the housing"]]
    }
    assert set(action) == set(smoke.ACTION_DIMS)


def test_action_summary_preserves_oob_hands_without_clamp():
    action = _action()
    action["left_hand"][0, 0, 0] = -0.5
    action["right_hand"][0, 0, 0] = 1.5

    active, summary = smoke.summarize_actions(action)

    assert active.shape == (1, 40, 26)
    assert active[0, 0, 14] == pytest.approx(-0.5)
    assert active[0, 0, 20] == pytest.approx(1.5)
    assert summary["active_26d"]["clamp_applied"] is False
    assert summary["diagnostics"]["hand_unit_interval_oob_elements"] == 2
    assert summary["diagnostics"]["oob_policy"] == "record_only_no_clamp"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.pop("waist"), "groups differ"),
        (
            lambda value: value.__setitem__(
                "left_arm", np.full((1, 40, 7), np.nan, dtype=np.float32)
            ),
            "non-finite",
        ),
        (
            lambda value: value.__setitem__("right_arm", np.zeros((1, 39, 7), dtype=np.float32)),
            "shape",
        ),
        (
            lambda value: value.__setitem__("left_hand", np.zeros((1, 40, 6), dtype=np.float64)),
            "dtype",
        ),
    ],
)
def test_action_summary_fails_closed(mutate, match):
    action = _action()
    mutate(action)
    with pytest.raises(ValueError, match=match):
        smoke.summarize_actions(action)
