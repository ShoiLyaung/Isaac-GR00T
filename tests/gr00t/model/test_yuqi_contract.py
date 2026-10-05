"""Small CPU checks for Yuqi recipe migration and multimodal batch slicing."""

from pathlib import Path
import sys

import pytest
import torch


DIRECTORY = Path(__file__).resolve().parents[3] / "examples/G1_DIT_MAMBA_YUQI"
sys.path.insert(0, str(DIRECTORY))
import contract  # noqa: E402
import preflight  # noqa: E402


def test_null_only_compatibility_is_explicit_and_immutable():
    raw = {
        "processor_kwargs": {
            "letter_box_transform": False,
            "modality_configs": {"new_embodiment": {"action": {"active_action_keys": None}}},
        }
    }
    migrated, audit = contract.processor_compatibility_config(raw, legacy_letterbox=True)
    assert raw["processor_kwargs"]["letter_box_transform"] is False
    assert (
        "active_action_keys"
        in raw["processor_kwargs"]["modality_configs"]["new_embodiment"]["action"]
    )
    assert migrated["processor_kwargs"]["letter_box_transform"] is True
    assert audit["removed_null_active_action_keys"] == 1
    raw["processor_kwargs"]["modality_configs"]["new_embodiment"]["action"][
        "active_action_keys"
    ] = ["left_arm"]
    with pytest.raises(ValueError, match="nonempty"):
        contract.processor_compatibility_config(raw, legacy_letterbox=True)


def test_multicamera_patch_slices_match_rank_samples():
    grids = torch.tensor([[1, 2, 2], [1, 2, 3], [1, 2, 4], [1, 3, 3], [1, 3, 4], [1, 3, 5]])
    counts = grids.prod(-1)
    pixels = torch.arange(int(counts.sum())).reshape(-1, 1)
    batch = {
        "state": torch.tensor([[[1.0] * 13], [[2.0] * 13]]),
        "action": torch.zeros(2, 40, 13),
        "input_ids": torch.tensor([[10, 11], [20, 21]]),
        "image_grid_thw": grids,
        "pixel_values": pixels,
    }
    result = preflight.slice_batch(batch, 1, 2)
    assert torch.equal(result["image_grid_thw"], grids[3:])
    assert torch.equal(result["pixel_values"], pixels[int(counts[:3].sum()) :])
    assert torch.equal(result["input_ids"], batch["input_ids"][1:])
    assert torch.equal(result["state"], batch["state"][1:])


def test_microbatch_seed_resets_intervening_architecture_draws():
    contract.seed_microbatch(8, 1, initialize_cuda=False)
    a = torch.randn(4, 40, 13)
    torch.randn(1000)
    contract.seed_microbatch(8, 1, initialize_cuda=False)
    b = torch.randn(4, 40, 13)
    assert torch.equal(a, b)
    contract.seed_microbatch(8, 0, initialize_cuda=False)
    assert not torch.equal(a, torch.randn(4, 40, 13))
