from __future__ import annotations

from types import SimpleNamespace

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class _ActionEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, actions, timesteps, embodiment_id):
        del timesteps, embodiment_id
        padding = torch.zeros(
            (*actions.shape[:-1], self.width - actions.shape[-1]),
            device=actions.device,
            dtype=actions.dtype,
        )
        return torch.cat((actions, padding), dim=-1)


class _StateEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, state, embodiment_id):
        del embodiment_id
        return torch.zeros((state.shape[0], 1, self.width), device=state.device, dtype=state.dtype)


class _Decoder(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, hidden_states, embodiment_id):
        del embodiment_id
        return hidden_states[..., : self.action_dim]


class _DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_hidden_states = None

    def forward(self, *, hidden_states, return_all_hidden_states=False, **kwargs):
        del kwargs
        self.last_hidden_states = hidden_states
        if return_all_hidden_states:
            return hidden_states, None
        return hidden_states


class _CountingConditioning:
    def __init__(self, width: int):
        self.width = width
        self.encode_calls = 0
        self.condition_calls = 0
        self.last_native = None

    def encode(self, evidence):
        assert evidence["sentinel"] == 1
        self.encode_calls += 1
        return (
            torch.zeros((1, 32, self.width)),
            torch.ones((1, 32), dtype=torch.bool),
        )

    def condition(self, action, context, mask):
        del context, mask
        self.condition_calls += 1
        self.last_native = action
        return action + 1000.0


def _tiny_head() -> tuple[Gr00tN1d7ActionHead, _CountingConditioning]:
    width = 8
    action_dim = 4
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(head)
    head.config = SimpleNamespace(
        action_horizon=40,
        state_history_length=1,
        add_pos_embed=False,
        use_alternate_vl_dit=False,
    )
    head.action_dim = action_dim
    head.action_horizon = 40
    head.num_inference_timesteps = 4
    head.num_timestep_buckets = 1000
    head.input_embedding_dim = width
    head.state_dropout_prob = 0.0
    head.action_encoder = _ActionEncoder(width)
    head.state_encoder = _StateEncoder(width)
    head.action_decoder = _Decoder(action_dim)
    head.model = _DiT()
    head.vlln = nn.Identity()
    head.vl_self_attention = nn.Identity()
    conditioning = _CountingConditioning(width)
    head.codec_evidence_enabled = True
    head.codec_conditioning = conditioning
    head.set_frozen_modules_to_eval_mode = lambda: None
    head.sample_time = lambda batch_size, device, dtype: torch.full(
        (batch_size,), 0.5, device=device, dtype=dtype
    )
    return head, conditioning


def test_training_inserts_conditioner_after_action_encoding_before_state_concat():
    torch.manual_seed(0)
    head, conditioning = _tiny_head()
    backbone = BatchFeature(
        data={
            "backbone_features": torch.zeros((1, 2, 8)),
            "backbone_attention_mask": torch.ones((1, 2), dtype=torch.bool),
        }
    )
    action_input = BatchFeature(
        data={
            "state": torch.zeros((1, 1, 4)),
            "action": torch.zeros((1, 40, 4)),
            "action_mask": torch.ones((1, 40, 4)),
            "embodiment_id": torch.zeros((1,), dtype=torch.long),
            "codec_evidence": {"sentinel": 1},
        }
    )

    output = head(backbone, action_input)

    assert torch.isfinite(output["loss"])
    assert conditioning.encode_calls == 1
    assert conditioning.condition_calls == 1
    assert head.model.last_hidden_states.shape == (1, 41, 8)
    assert torch.equal(head.model.last_hidden_states[:, :1], torch.zeros((1, 1, 8)))
    assert torch.equal(head.model.last_hidden_states[:, 1:], conditioning.last_native + 1000.0)


def test_four_euler_evaluations_reuse_one_encoded_context_and_one_helper():
    torch.manual_seed(0)
    head, conditioning = _tiny_head()
    backbone = BatchFeature(
        data={
            "backbone_features": torch.zeros((1, 2, 8)),
            "backbone_attention_mask": torch.ones((1, 2), dtype=torch.bool),
        }
    )
    action_input = BatchFeature(
        data={
            "embodiment_id": torch.zeros((1,), dtype=torch.long),
            "codec_evidence": {"sentinel": 1},
        }
    )

    output = head.get_action_with_features(
        backbone_features=backbone.backbone_features,
        state_features=torch.zeros((1, 1, 8)),
        embodiment_id=action_input.embodiment_id,
        backbone_output=backbone,
        action_input=action_input,
    )

    assert output.action_pred.shape == (1, 40, 4)
    assert conditioning.encode_calls == 1
    assert conditioning.condition_calls == 4


def test_zero_gate_is_direct_scalar_dropout_free_and_torch_equal():
    torch.manual_seed(3)
    from gr00t.model.gr00t_n1d7.codec_dynamics import Gr00tCodecConditioningStack

    stack = Gr00tCodecConditioningStack(action_width=64, d_codec=16, conditioner_heads=8)
    action = torch.randn((2, 40, 64))
    context = torch.randn((2, 32, 64))
    mask = torch.ones((2, 32), dtype=torch.bool)

    output = stack.condition(action, context, mask)

    assert stack.conditioner.gate.ndim == 0
    assert stack.conditioner.gate.item() == 0.0
    assert stack.conditioner.attention.dropout == 0.0
    assert not any(isinstance(module, nn.Dropout) for module in stack.modules())
    assert torch.equal(output, action)


def test_absent_config_serialization_is_native_surface():
    assert "codec_evidence" not in Gr00tN1d7Config().to_filtered_dict()
