"""CPU tests for the frozen Yuqi experiment's decision-relevant contracts."""

from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/G1_DIT_MAMBA_YUQI"))
from contract import preserve_yuqi_language
from runner_core import (
    aggregate_metrics,
    deterministic_seed,
    make_schedule,
    masked_flow,
    preserve_eval_state,
    rank_slots,
    set_seed,
)


def test_epoch_schedule_and_rank_coverage_ignore_ambient_rng():
    lengths = {episode: 64 for episode in (0, 1, 150, 147, 152, 112)}
    schedule, anchors = make_schedule(lengths, steps=20)
    random.seed(987)
    np.random.normal(size=1000)
    torch.rand(177)
    other, other_anchors = make_schedule(lengths, steps=20)
    assert np.array_equal(schedule, other)
    assert anchors == other_anchors
    assert len({tuple(pair) for pair in schedule.reshape(-1, 2)[: 6 * 49]}) == 6 * 49
    assert all(0 <= anchor <= lengths[episode] - 16 for episode, anchor in schedule.reshape(-1, 2))
    for batch, accum in ((16, 1), (8, 2)):
        all_slots = [
            slot for rank in (0, 1) for micro in rank_slots(rank, batch, accum) for slot in micro
        ]
        assert sorted(all_slots) == list(range(32))
    assert len(anchors["diagnostic"]) == 128
    assert len(anchors["train_probe"]) == len(anchors["seen_probe"]) == 16


def test_microbatch_randomness_isolated_from_extra_backbone_draws():
    def next_draws(extra):
        set_seed(deterministic_seed("forward", 0, 0, 0))
        torch.randn(extra)
        random.random()
        np.random.rand(extra)
        set_seed(deterministic_seed("forward", 1, 0, 0))
        return (
            torch.randn(3, 40, 13),
            torch.distributions.Beta(1.5, 1).sample((3,)),
            np.random.rand(3),
        )

    left, right = next_draws(10), next_draws(2222)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[1], right[1])
    assert np.array_equal(left[2], right[2])
    assert deterministic_seed("forward", 1, 0, 0) != deterministic_seed("forward", 1, 0, 1)


def test_mask_is_active16_compact13_and_uses_mask_sum():
    mask = torch.zeros(2, 40, 13)
    mask[:, :16] = 1
    losses = torch.full_like(mask, 999)
    losses[:, :16] = 2
    assert torch.equal(masked_flow(losses, mask), torch.tensor([2.0, 2.0]))
    mask[0, 16, 0] = 1
    with pytest.raises(ValueError, match="16x13"):
        masked_flow(losses, mask)
    with pytest.raises(ValueError, match="compact13"):
        masked_flow(torch.ones(1, 40, 26), torch.ones(1, 40, 26))


def test_probe_restores_heterogeneous_modes_and_rng_on_exception():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[0].eval()

    class Processor:
        training = True
        state_action_processor = SimpleNamespace(training=True)

        def eval(self):
            self.training = self.state_action_processor.training = False

    processor = Processor()
    set_seed(52)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    set_seed(52)
    with pytest.raises(RuntimeError):
        with preserve_eval_state(model, processor):
            assert not any(module.training for module in model.modules())
            assert not processor.training
            set_seed(91)
            torch.rand(10)
            raise RuntimeError("intentional")
    assert [module.training for module in model.modules()] == [True, False, True]
    assert processor.training and processor.state_action_processor.training
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_equal_episode_aggregation_does_not_overweight_more_rows():
    rows = [
        {"episode": 0, "anchor": 0, "metric": 0.0},
        {"episode": 1, "anchor": 0, "metric": 6.0},
        {"episode": 1, "anchor": 0, "metric": 2.0},
        {"episode": 1, "anchor": 1, "metric": 8.0},
    ]
    per_episode, total = aggregate_metrics(rows, ["metric"])
    assert per_episode["1"]["metric"] == 6
    assert total["metric"] == 3


def test_known_yuqi_text_identity_preserves_semantics():
    loader = SimpleNamespace(tasks_map={0: "insert the shaft into the housing"})
    preserve_yuqi_language(loader)
    assert loader.tasks_map[0] == loader.tasks_map["insert the shaft into the housing"]
    with pytest.raises(KeyError):
        _ = loader.tasks_map["unknown task"]
