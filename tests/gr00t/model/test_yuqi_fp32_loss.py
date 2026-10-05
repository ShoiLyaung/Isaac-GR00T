"""Exercise the real head.forward loss with lightweight modules and no VLM/GPU."""

from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class PassEncoder(nn.Module):
    def forward(self, values, *unused):
        return values


class PassBackbone(nn.Module):
    def forward(self, *, hidden_states, **unused):
        return hidden_states, []


class PredictionDecoder(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = nn.Parameter(prediction.clone())

    def forward(self, *unused):
        return self.prediction


def forward_fixture(monkeypatch, prediction, target, mask):
    # Use the actual public forward implementation without constructing a 32-layer model.
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(head)
    head.config = SimpleNamespace(
        state_history_length=1, use_alternate_vl_dit=True, add_pos_embed=False
    )
    head.tune_projector = head.tune_diffusion_model = head.tune_vlln = True
    head.state_dropout_prob = 0.0
    head.num_timestep_buckets = 1000
    head.state_encoder = PassEncoder()
    head.action_encoder = PassEncoder()
    head.action_decoder = PredictionDecoder(prediction)
    head.vlln = nn.Identity()
    head.vl_self_attention = nn.Identity()
    head.model = PassBackbone()
    head.sample_time = lambda size, device, dtype: torch.full(
        (size,), 0.25, device=device, dtype=dtype
    )
    monkeypatch.setattr(
        torch, "randn", lambda shape, device, dtype: torch.zeros(shape, device=device, dtype=dtype)
    )
    target = target.clone().requires_grad_()
    action = BatchFeature(
        data={
            "state": torch.zeros(1, 1, 13, dtype=prediction.dtype),
            "action": target,
            "action_mask": mask,
            "embodiment_id": torch.tensor([10]),
        }
    )
    condition = BatchFeature(
        data={
            "backbone_features": torch.zeros(1, 3, 13, dtype=prediction.dtype),
            "backbone_attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "image_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    )
    return head, action, condition, target


def tensors(dtype):
    prediction = torch.linspace(-0.9, 0.7, 40 * 13).reshape(1, 40, 13).to(dtype)
    target = torch.linspace(0.7, -0.1, 40 * 13).reshape(1, 40, 13).to(dtype)
    mask = torch.zeros_like(target)
    mask[:, :16] = 1
    return prediction, target, mask


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_head_loss_and_gradients_match_explicit_fp32_objective(monkeypatch, dtype):
    prediction, target, mask = tensors(dtype)
    head, action, condition, actual_target = forward_fixture(monkeypatch, prediction, target, mask)
    # This call has no autocast, matching the observed DeepSpeed BF16 execution boundary.
    output = head(condition, action)
    assert (
        output["loss"].dtype
        == output["action_loss"].dtype
        == output["action_mask"].dtype
        == torch.float32
    )
    expected_pred = prediction[:, :16].float().clone().requires_grad_()
    expected_target = target[:, :16].float().clone().requires_grad_()
    expected = (expected_pred - expected_target).square().sum() / (208 + 1e-6)
    assert torch.equal(output["loss"], expected)
    assert torch.equal(output["action_loss"][:, :16], (expected_pred - expected_target).square())
    assert torch.count_nonzero(output["action_loss"][:, 16:]) == 0
    expected.backward()
    output["loss"].backward()
    assert head.action_decoder.prediction.grad.dtype == actual_target.grad.dtype == dtype
    assert torch.equal(head.action_decoder.prediction.grad[:, :16], expected_pred.grad.to(dtype))
    assert torch.equal(actual_target.grad[:, :16], expected_target.grad.to(dtype))
    assert torch.count_nonzero(head.action_decoder.prediction.grad[:, 16:]) == 0
    assert torch.count_nonzero(actual_target.grad[:, 16:]) == 0
    if dtype == torch.bfloat16:
        rounded_reduction = ((prediction - target).square() * mask).sum() / mask.sum()
        assert output["loss"].item() != rounded_reduction.float().item()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_inactive_nonfinite_predictions_and_targets_do_not_poison_loss_or_gradients(
    monkeypatch, bad_value
):
    prediction, target, mask = tensors(torch.bfloat16)
    prediction[:, 16:] = bad_value
    target[:, 16:] = bad_value
    head, action, condition, actual_target = forward_fixture(monkeypatch, prediction, target, mask)
    output = head(condition, action)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["action_loss"]).all()
    output["loss"].backward()
    for grad in (head.action_decoder.prediction.grad, actual_target.grad):
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad[:, 16:]) == 0


@pytest.mark.parametrize(
    "side,message",
    [("prediction", "supervised pred_actions"), ("target", "supervised target velocity")],
)
def test_active_nan_fails_closed_before_reduction(monkeypatch, side, message):
    prediction, target, mask = tensors(torch.bfloat16)
    (prediction if side == "prediction" else target)[0, 0, 0] = float("nan")
    head, action, condition, _ = forward_fixture(monkeypatch, prediction, target, mask)
    with pytest.raises(FloatingPointError, match=message):
        head(condition, action)


def test_empty_mask_fails_closed(monkeypatch):
    prediction, target, mask = tensors(torch.bfloat16)
    mask.zero_()
    head, action, condition, _ = forward_fixture(monkeypatch, prediction, target, mask)
    with pytest.raises(ValueError, match="action_mask is empty"):
        head(condition, action)
