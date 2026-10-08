"""CPU cache-contract tests; no native model or GPU execution."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/G1_DIT_MAMBA_YUQI"))
import action_transfer_full_cache as cache  # noqa: E402
from fp32_pilot_support import PilotCache  # noqa: E402


def fake_frame(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(cache.proto, "validate_input_layout", lambda *a, **k: None)
    monkeypatch.setattr(cache.proto, "validate_packet", lambda *a, **k: None)
    raw_features = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    backbone = BatchFeature(
        data={
            "backbone_features": raw_features,
            "backbone_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "image_mask": torch.ones(1, 2, dtype=torch.bool),
        }
    )
    inputs = {
        "state": torch.ones(1, 1, 13, dtype=torch.bfloat16),
        "action": torch.zeros(1, 40, 13, dtype=torch.bfloat16),
        "action_mask": torch.ones(1, 40, 13, dtype=torch.bool),
        "embodiment_id": torch.tensor([10]),
    }

    def mutate(raw, action):
        raw["backbone_features"].add_(2)
        raw["backbone_features"] = raw["backbone_features"].float() + 3
        action["state"].add_(5)
        return BatchFeature(data={"backbone_features": raw["backbone_features"]})

    model = SimpleNamespace(
        device="cpu",
        prepare_input=lambda value: ({}, BatchFeature(data=inputs)),
        backbone=lambda value: backbone,
        action_head=SimpleNamespace(_encode_features=mutate),
    )
    physical = SimpleNamespace(
        states={"left_arm": torch.zeros(1, 7).numpy(), "left_hand": torch.zeros(1, 6).numpy()},
        actions={"left_arm": torch.zeros(16, 7).numpy(), "left_hand": torch.zeros(16, 6).numpy()},
    )
    source = SimpleNamespace(
        batch=lambda pairs: (inputs, [physical]),
        modality={"video": SimpleNamespace(modality_keys=["a", "b", "c"])},
    )
    value = cache.encode_frame(model, source, 0, 0)
    return value, raw_features, inputs


def test_raw_preserved_before_mutating_postprocess_and_action_clone(monkeypatch):
    frame, original, inputs = fake_frame(monkeypatch)
    assert torch.equal(frame["raw"]["backbone_features"], torch.ones_like(original))
    assert frame["raw"]["backbone_features"].dtype == torch.bfloat16
    assert frame["post_vl_features"].dtype == torch.float32
    assert torch.all(frame["post_vl_features"] == 6)
    assert torch.all(original == 1) and torch.all(frame["action_input"]["state"] == 1)
    assert torch.all(inputs["state"] == 1)
    assert frame["content_sha256"] == cache.frame_hash(frame)


def test_overlap_checks_values_dtype_and_input_identity(monkeypatch):
    frame, _, _ = fake_frame(monkeypatch)
    assert cache.compare_frame(frame, deepcopy(frame))["all_tensor_and_input_identities_exact"]
    changed = deepcopy(frame)
    changed["post_vl_features"][0, 0, 0] += 1
    with pytest.raises(ValueError) as error:
        cache.compare_frame(changed, frame)
    assert "episode=0 anchor=0" in str(error.value)
    assert "section=post_vl_features" in str(error.value)
    assert "new_frame_sha256=" in str(error.value) and "old_frame_sha256=" in str(error.value)
    changed = deepcopy(frame)
    changed["input_identity"]["state"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="input_identity.state"):
        cache.compare_frame(changed, frame)


def test_write_new_episode_is_pilotcache_compatible(monkeypatch, tmp_path):
    frame, _, _ = fake_frame(monkeypatch)
    path = tmp_path / "episode-000000.pt"
    identity = {"episode": 0, **cache.save_episode(path, {0: frame})}
    with pytest.raises(FileExistsError):
        cache.save_episode(path, {0: frame})
    (tmp_path / "CACHE_RESULT.json").write_text(
        json.dumps(
            {"status": "passed", "cohort": "train_dev", "episodes": [0], "files": [identity]}
        )
    )
    reader = PilotCache(tmp_path, {"train": [0], "dev": [], "grids": {"0": [0]}})
    assert reader.frame(0, 0)["content_sha256"] == frame["content_sha256"]
    reader.assert_unchanged()


def valid_plan():
    train = list(range(180))
    dev = list(range(180, 188))
    lengths = {str(ep): 16 + 4 * (117 - 1) for ep in train}
    # 180*117=21060; add91 train observations. Dev7*104+109=837.
    lengths["0"] += 4 * 91
    lengths.update({str(ep): 16 + 4 * (104 - 1) for ep in dev})
    lengths["187"] += 4 * 5
    grids = {
        str(ep): list(range(0, length - 15, 4))
        for ep in train + dev
        for length in [lengths[str(ep)]]
    }
    plan = {
        "train": train,
        "dev": dev,
        "lengths": lengths,
        "grids": grids,
        "expected_observations": 21988,
    }
    cohort = {"train": train, "dev": dev, "final": [188], "reserve": [189]}
    return plan, cohort


def test_cohort_rejects_final_reserve_and_unscheduled_tail():
    plan, cohort = valid_plan()
    assert len(cache.verify_cohort(plan, cohort)) == 188
    changed = deepcopy(plan)
    changed["grids"]["0"].append(changed["grids"]["0"][-1] + 1)
    with pytest.raises(ValueError, match="Stride4"):
        cache.verify_cohort(changed, cohort)
    bad = deepcopy(cohort)
    bad["final"].append(plan["train"][0])
    with pytest.raises(ValueError, match="final/reserve"):
        cache.verify_cohort(plan, bad)
