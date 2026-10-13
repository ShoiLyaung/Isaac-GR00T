from __future__ import annotations

import gr00t.experimental.mikasa_history_hard_order as order
from gr00t.experimental.mikasa_history_hard_order_experiment import shared_initialization
import torch
from torch import nn


class IdentityBackbone(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _batch(rows: int = 2) -> order.OrderBatch:
    generator = torch.Generator().manual_seed(230923)
    return order.OrderBatch(
        temporal_grid=torch.rand((rows, 130, 2, 8, 8, 9), generator=generator),
        current_rgb_grid=torch.rand((rows, 2, 8, 8, 3), generator=generator),
        current_proprio=torch.rand((rows, 7), generator=generator),
    )


def test_shuffle_depends_only_on_seed_step_and_is_batch_global() -> None:
    first = order.history_shuffle_permutation(230923, 17)
    second = order.history_shuffle_permutation(230923, 17)
    assert torch.equal(first, second)
    assert not torch.equal(first, order.history_shuffle_permutation(230923, 18))
    assert not torch.equal(first, order.history_shuffle_permutation(230924, 17))
    assert sorted(first.tolist()) == list(range(130))

    temporal = torch.arange(3 * 130, dtype=torch.float32).reshape(3, 130, 1, 1, 1, 1)
    temporal = temporal.expand(-1, -1, 2, 8, 8, 9).clone()
    current = torch.zeros((3, 2, 8, 8, 3))
    shuffled, observed = order.apply_history_condition(
        temporal,
        current,
        "mamba_shuffle",
        shuffle_seed=230923,
        shuffle_step=17,
    )
    assert torch.equal(observed, first)
    for row in range(3):
        assert torch.equal(shuffled[row], temporal[row, first])


def test_current_zero_and_full_conditions_preserve_shape_and_cadence() -> None:
    batch = _batch()
    full, permutation = order.apply_history_condition(
        batch.temporal_grid, batch.current_rgb_grid, "mamba_full"
    )
    assert permutation is None
    assert torch.equal(full, batch.temporal_grid)
    zero, _ = order.apply_history_condition(
        batch.temporal_grid, batch.current_rgb_grid, "mamba_zero"
    )
    assert zero.shape == batch.temporal_grid.shape and torch.count_nonzero(zero) == 0
    current, _ = order.apply_history_condition(
        batch.temporal_grid, batch.current_rgb_grid, "mamba_current"
    )
    assert current.shape == batch.temporal_grid.shape
    assert torch.equal(current[:, 0], current[:, -1])
    cue, repeated, delta = torch.chunk(current, 3, dim=-1)
    assert torch.equal(cue, repeated)
    assert torch.count_nonzero(delta) == 0


def test_two_boundaries_use_one_shared_head(monkeypatch) -> None:
    monkeypatch.setattr(
        order, "build_temporal_backbone", lambda *_args, **_kwargs: IdentityBackbone()
    )
    model = order.MikasaHistoryHardOrderModel("mamba_full", seed=230923, frame_chunk_size=65)
    captured = []
    hook = model.shared_head.register_forward_pre_hook(
        lambda _module, args: captured.append(tuple(args[0].shape))
    )
    output = model(_batch())
    hook.remove()
    assert output.logits.shape == (2, 2, 3)
    assert output.boundary_states.shape == (2, 2, 256)
    assert captured == [(4, 512)]
    assert order.BOUNDARY_INDICES == (65, 129)
    assert len([module for module in model.modules() if module is model.shared_head]) == 1


def test_gru_forward_resets_state_per_call_and_per_sample() -> None:
    model = order.MikasaHistoryHardOrderModel("gru_full", seed=230923, frame_chunk_size=65).eval()
    batch = _batch(2)
    with torch.inference_mode():
        together = model(batch).logits
        second = model(
            order.OrderBatch(
                batch.temporal_grid[1:2],
                batch.current_rgb_grid[1:2],
                batch.current_proprio[1:2],
            )
        ).logits
        repeated = model(batch).logits
    torch.testing.assert_close(together[1:2], second, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(together, repeated, rtol=0, atol=0)
    assert not any(name in model.__dict__ for name in ("state", "cache", "hidden"))


def test_shared_initialization_is_condition_invariant(monkeypatch) -> None:
    monkeypatch.setattr(
        order, "build_temporal_backbone", lambda *_args, **_kwargs: IdentityBackbone()
    )
    identities = [
        shared_initialization(order.MikasaHistoryHardOrderModel(condition, seed=230923))
        for condition in order.CONDITIONS
    ]
    assert all(value == identities[0] for value in identities[1:])


def test_cross_entropy_is_mean_over_two_slots() -> None:
    logits = torch.tensor([[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
    output = order.OrderOutput(logits, torch.zeros(1, 2, 256), torch.zeros(1, 256))
    labels = torch.tensor([[0, 1]], dtype=torch.int64)
    observed = order.order_cross_entropy(output, labels)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
    torch.testing.assert_close(observed, expected)
