"""Eight-layer attention-free action backbone with request-local VL conditioning.

This module replaces the action core only. The native VLM, VL postprocessor,
state/action encoders and decoder remain outside. It never retains SSM state.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.model.modules.mamba_action_mixer import _mamba2_class


WIDTH = 1536
MIXER_WIDTH = 512
SELECTED_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
IMAGE_POSITIONS = tuple(range(4, 68)) + tuple(range(70, 134)) + tuple(range(136, 200))
NONIMAGE_POSITIONS = tuple(i for i in range(209) if i not in IMAGE_POSITIONS)
PERMUTATION = IMAGE_POSITIONS + NONIMAGE_POSITIONS
MAMBA_CONFIG = dict(
    d_state=64, d_conv=4, expand=1, headdim=64, ngroups=1, chunk_size=16, use_mem_eff_path=False
)
EXPECTED_PARAMETERS = 230_785_600
EXPECTED_LINEAR_MACS = 29_579_837_440


@dataclass(frozen=True)
class ActionCondition:
    code: torch.Tensor
    modulation: torch.Tensor


class ActionConditionAdapter(nn.Module):
    """Same fixed-layout bottleneck math as the independent-condition prototype."""

    def __init__(self):
        super().__init__()
        self.register_buffer("permutation", torch.tensor(PERMUTATION), persistent=False)
        self.token = nn.Linear(2048, 128, device="cpu", dtype=torch.float32)
        self.code = nn.Linear(209 * 128, 256, device="cpu", dtype=torch.float32)
        self.modulation = nn.Linear(256, 8 * 2 * WIDTH, device="cpu", dtype=torch.float32)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (209, 2048):
            raise ValueError("Expected fixed-layout [batch,209,2048] native VL features")
        tokens = F.silu(self.token(features[:, self.permutation]))
        code = F.silu(self.code(tokens.flatten(1)))
        return ActionCondition(code, self.modulation(code).reshape(-1, 8, 2, WIDTH))


class NarrowBidirectionalMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(WIDTH, MIXER_WIDTH, device="cpu", dtype=torch.float32)
        self.mamba = _mamba2_class()(
            d_model=MIXER_WIDTH, **MAMBA_CONFIG, device="cpu", dtype=torch.float32
        )
        self.up = nn.Linear(MIXER_WIDTH, WIDTH, device="cpu", dtype=torch.float32)

    def forward(self, value):
        projected = self.down(value)
        forward = self.mamba(projected.contiguous())
        reverse = self.mamba(projected.flip(1).contiguous()).flip(1)
        return self.up((forward + reverse) * 0.5)


class BudgetedActionBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ("norm1", "norm3", "ff", "final_dropout"):
            value = getattr(original, name)
            setattr(
                self,
                name,
                deepcopy(value).to(device="cpu", dtype=torch.float32)
                if value is not None
                else nn.Identity(),
            )
        self.mixer = NarrowBidirectionalMixer()

    def forward(self, hidden, temb, modulation):
        gamma, beta = modulation.unbind(1)
        conditioned = self.norm1(hidden, temb) * (1 + gamma[:, None]) + beta[:, None]
        hidden = hidden + self.final_dropout(self.mixer(conditioned))
        return hidden + self.ff(self.norm3(hidden))


class BudgetedActionMamba(nn.Module):
    """Fresh seed42 new mixers/adapter; exact selected BF16 teacher values -> FP32."""

    def __init__(self, native):
        super().__init__()
        if native.inner_dim != WIDTH or len(native.transformer_blocks) != 32:
            raise ValueError("Expected frozen native 32-layer/1536-wide action core")
        self.inner_dim = WIDTH
        # Only new parameters consume this CPU seed; copying native parameters
        # does not initialize or perturb any source tensor.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.blocks = nn.ModuleList(
                [BudgetedActionBlock(native.transformer_blocks[i]) for i in SELECTED_LAYERS]
            )
            self.condition_adapter = ActionConditionAdapter()
        for name in ("timestep_encoder", "norm_out", "proj_out_1", "proj_out_2"):
            setattr(
                self, name, deepcopy(getattr(native, name)).to(device="cpu", dtype=torch.float32)
            )
        if sum(p.numel() for p in self.parameters()) != EXPECTED_PARAMETERS:
            raise ValueError("Budgeted student parameter contract changed")
        self.requires_grad_(True)
        if any("attention" in type(m).__name__.lower() for m in self.modules()):
            raise ValueError("Student action core must contain no attention module")

    def encode_condition(self, features):
        return self.condition_adapter(features)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
        *,
        condition_cache=None,
    ):
        if not isinstance(condition_cache, ActionCondition):
            raise ValueError("Explicit request-local ActionCondition required")
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (41, WIDTH):
            raise ValueError("Action backbone must process exactly41 state/action tokens")
        batch = hidden_states.shape[0]
        if tuple(condition_cache.modulation.shape) != (batch, 8, 2, WIDTH):
            raise ValueError("Condition cache shape mismatch")
        if condition_cache.modulation.device != hidden_states.device:
            raise ValueError("Condition cache device mismatch")
        if tuple(encoder_hidden_states.shape) != (batch, 209, 2048):
            raise ValueError("Frozen VL token layout changed")
        hidden = hidden_states.contiguous()
        temb = self.timestep_encoder(timestep)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, temb, condition_cache.modulation[:, index])
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden)
        return (output, [output]) if return_all_hidden_states else output


class ConditionedActionRequest(nn.Module):
    """A single caller-owned request binding; the shared student owns no cache."""

    def __init__(self, student, condition):
        super().__init__()
        if not isinstance(condition, ActionCondition):
            raise ValueError("Expected explicit ActionCondition")
        self.student = student
        self.condition = condition

    def forward(self, *args, **kwargs):
        if "condition_cache" in kwargs:
            raise ValueError("A request condition cannot be overwritten during denoising")
        return self.student(*args, **kwargs, condition_cache=self.condition)
