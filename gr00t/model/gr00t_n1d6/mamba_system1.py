import logging
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    swish,
)


_MambaCore = None
_MAMBA_IMPORT_ERROR = None
try:
    from mamba_ssm import Mamba2 as _MambaCore
except Exception as exc_mamba2:
    try:
        from mamba_ssm import Mamba as _MambaCore
    except Exception as exc_mamba1:
        _MAMBA_IMPORT_ERROR = f"Mamba2 import failed: {exc_mamba2!r}; Mamba import failed: {exc_mamba1!r}"


def _shape(x):
    if x is None:
        return "None"
    if torch.is_tensor(x):
        return str(tuple(x.shape))
    return str(type(x))


class _StableTemporalMixerCore(nn.Module):
    """Numerically stable causal temporal mixer used when mamba_ssm is unavailable."""

    def __init__(self, hidden_size: int, d_conv: int, expand: int):
        super().__init__()
        expanded_size = hidden_size * expand
        self.window_size = max(2, d_conv * 2 + 1)
        self.in_proj = nn.Linear(hidden_size, expanded_size * 2)
        self.out_proj = nn.Linear(expanded_size, hidden_size)
        self.register_buffer("branch_gain", torch.tensor(0.1))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _causal_window_average(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size <= 1:
            return x

        batch_size, seq_len, hidden_size = x.shape
        prefix = torch.zeros(batch_size, 1, hidden_size, device=x.device, dtype=x.dtype)
        csum = torch.cat((prefix, torch.cumsum(x, dim=1)), dim=1)
        idx = torch.arange(seq_len, device=x.device)
        end = idx + 1
        start = torch.clamp(end - self.window_size, min=0)
        mixed = csum[:, end, :] - csum[:, start, :]
        denom = (end - start).to(dtype=x.dtype).view(1, seq_len, 1)
        return mixed / denom.clamp_min(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        autocast_ctx = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type in ("cuda", "cpu")
            else nullcontext()
        )

        with autocast_ctx:
            compute_dtype = self.in_proj.weight.dtype
            x = torch.nan_to_num(x.to(dtype=compute_dtype), nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e4, 1e4
            )
            value, gate = self.in_proj(x).chunk(2, dim=-1)
            value = F.silu(value)
            mixed = self._causal_window_average(value)
            mixed = mixed * torch.sigmoid(gate)
            mixed = torch.tanh(self.out_proj(mixed))
            mixed = mixed * self.branch_gain.to(device=mixed.device, dtype=mixed.dtype)
            mixed = torch.nan_to_num(mixed, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return mixed.to(dtype=out_dtype)


class _MambaLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        if _MambaCore is None:
            self.norm = nn.Identity()
            self.core = _StableTemporalMixerCore(hidden_size, d_conv, expand)
        else:
            self.norm = nn.LayerNorm(hidden_size)
            try:
                self.core = _MambaCore(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            except TypeError:
                self.core = _MambaCore(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        x = self.norm(residual)
        x = self.core(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        out = residual + self.dropout(x)
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)


class _MambaStack(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _MambaLayer(
                    hidden_size=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.Identity() if _MambaCore is None else nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)


class VideoMambaEncoder(nn.Module):
    """Phase-2 video encoder: video token sequence -> pooled video context."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_size)
        self.video_mamba = _MambaStack(
            hidden_size=hidden_size,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, vl_tokens: torch.Tensor, image_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.in_proj(vl_tokens)
        if image_mask is not None:
            mask = image_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * mask
        x = self.video_mamba(x)
        if image_mask is None:
            return x.mean(dim=1, keepdim=True)

        mask = image_mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask).sum(dim=1, keepdim=True) / denom


class VideoToRobotInjector(nn.Module):
    """Unidirectional injection: Video hidden -> Robot hidden."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.robot_norm = nn.LayerNorm(hidden_size)
        self.video_norm = nn.LayerNorm(hidden_size)
        self.video_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, robot_hidden: torch.Tensor, video_context: torch.Tensor) -> torch.Tensor:
        video_context = self.video_norm(video_context)
        video_context = video_context.expand(-1, robot_hidden.shape[1], -1)

        robot_norm = self.robot_norm(robot_hidden)
        gate = torch.sigmoid(self.gate_proj(torch.cat((robot_norm, video_context), dim=-1)))
        injected = torch.tanh(self.video_proj(video_context))
        return robot_hidden + gate * injected


class _StableConditionedTemporalMixerBlock(nn.Module):
    """Conditioned causal temporal mixer built from stable attention + FFN blocks."""

    def __init__(self, hidden_size: int, expand: int, window_size: int, dropout: float):
        super().__init__()
        del window_size  # Kept in signature for config compatibility.
        expanded_size = hidden_size * max(2, expand)
        num_heads = max(1, min(8, hidden_size // 128))
        while hidden_size % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.pre_norm = nn.LayerNorm(hidden_size)
        self.cond_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff_up = nn.Linear(hidden_size, expanded_size)
        self.ff_down = nn.Linear(expanded_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("attn_gain", torch.tensor(0.1))
        self.register_buffer("ff_gain", torch.tensor(0.1))
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ff_down.weight)
        nn.init.zeros_(self.ff_down.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        autocast_ctx = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type in ("cuda", "cpu")
            else nullcontext()
        )

        with autocast_ctx:
            compute_dtype = self.cond_proj.weight.dtype
            residual = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e4, 1e4
            ).to(dtype=compute_dtype)
            cond = torch.nan_to_num(cond, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e4, 1e4
            ).to(dtype=compute_dtype)
            cond_bias = 0.1 * torch.tanh(self.cond_proj(cond)).unsqueeze(1)

            attn_input = self.pre_norm(residual) + cond_bias.to(dtype=residual.dtype)
            seq_len = attn_input.shape[1]
            causal_mask = torch.ones(seq_len, seq_len, device=attn_input.device, dtype=torch.bool)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attn_out, _ = self.attn(
                attn_input,
                attn_input,
                attn_input,
                attn_mask=causal_mask,
                need_weights=False,
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e3, 1e3
            )
            out = residual + self.dropout(
                attn_out * self.attn_gain.to(device=attn_out.device, dtype=attn_out.dtype)
            )

            ff_input = self.ff_norm(out.to(dtype=self.ff_norm.weight.dtype)).to(dtype=compute_dtype)
            ff_hidden = F.gelu(self.ff_up(ff_input))
            ff_hidden = torch.nan_to_num(ff_hidden, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e3, 1e3
            )
            ff_out = self.ff_down(ff_hidden)
            ff_out = torch.nan_to_num(ff_out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e3, 1e3
            )
            out = out + self.dropout(
                ff_out * self.ff_gain.to(device=ff_out.device, dtype=ff_out.dtype)
            )
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return out.to(dtype=out_dtype)


class _StableConditionedTemporalMixerStack(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _StableConditionedTemporalMixerBlock(
                    hidden_size=hidden_size,
                    expand=expand,
                    window_size=d_conv * 2 + 1,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond)
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)


class MambaSystem1ActionHead(nn.Module):
    """System1 implemented with a Mamba-style sequence mixer."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.num_timestep_buckets = config.num_timestep_buckets
        self.system1_debug_shapes = config.system1_debug_shapes
        self.system1_debug_gradients = getattr(config, "system1_debug_gradients", False)
        self.debug_robot_mamba_mode = getattr(config, "debug_robot_mamba_mode", "full")
        self.mamba_enable_video_injection = config.mamba_enable_video_injection
        self.require_real_mamba_backend = getattr(config, "require_real_mamba_backend", False)

        if self.require_real_mamba_backend and _MambaCore is None:
            raise RuntimeError(
                "Real mamba_ssm backend unavailable; refusing to initialize MambaSystem1ActionHead. "
                f"Import detail: {_MAMBA_IMPORT_ERROR}"
            )

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        self.state_token_proj = nn.Linear(self.input_embedding_dim, self.hidden_size)
        self.action_token_proj = nn.Linear(self.input_embedding_dim, self.hidden_size)
        if _MambaCore is None:
            self.robot_mamba = None
            self.fallback_temporal_mixer = _StableConditionedTemporalMixerStack(
                hidden_size=self.hidden_size,
                num_layers=config.mamba_num_layers,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dropout=config.mamba_dropout,
            )
        else:
            self.robot_mamba = _MambaStack(
                hidden_size=self.hidden_size,
                num_layers=config.mamba_num_layers,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dropout=config.mamba_dropout,
            )
            self.fallback_temporal_mixer = None

        if self.mamba_enable_video_injection:
            self.video_mamba = VideoMambaEncoder(
                input_dim=config.backbone_embedding_dim,
                hidden_size=self.hidden_size,
                num_layers=config.mamba_video_num_layers,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dropout=config.mamba_dropout,
            )
            self.video_to_robot = VideoToRobotInjector(hidden_size=self.hidden_size)
        else:
            self.video_mamba = None
            self.video_to_robot = None

        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self._logged_train_contract = False
        self._logged_infer_contract = False
        self._logged_non_finite_warning = False
        self._finite_debug_logged: set[str] = set()
        self._gradient_debug_logged: set[str] = set()

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )
        self._register_gradient_sanitizers()

        backend = "mamba_ssm" if _MambaCore is not None else "stable_temporal_mixer"
        if _MambaCore is None:
            message = (
                "Real mamba_ssm backend unavailable; using stable_temporal_mixer instead. "
                f"Import detail: {_MAMBA_IMPORT_ERROR}"
            )
            logging.warning(message)
        logging.info(
            "Using MambaSystem1ActionHead (backend=%s, phase2_video_injection=%s, debug_robot_mamba_mode=%s)",
            backend,
            self.mamba_enable_video_injection,
            self.debug_robot_mamba_mode,
        )

    def _register_gradient_sanitizers(self):
        def _sanitize_grad(grad: torch.Tensor) -> torch.Tensor:
            return torch.clamp(
                torch.nan_to_num(grad, nan=0.0, posinf=1e3, neginf=-1e3), min=-1e3, max=1e3
            )

        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(_sanitize_grad)

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True

        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.state_token_proj.requires_grad_(False)
            self.action_token_proj.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)

        if not tune_diffusion_model:
            if self.robot_mamba is not None:
                self.robot_mamba.requires_grad_(False)
            if self.fallback_temporal_mixer is not None:
                self.fallback_temporal_mixer.requires_grad_(False)
            if self.video_mamba is not None:
                self.video_mamba.requires_grad_(False)
                self.video_to_robot.requires_grad_(False)

        if not tune_vlln:
            self.vlln.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.state_token_proj.eval()
                self.action_token_proj.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                if self.robot_mamba is not None:
                    self.robot_mamba.eval()
                if self.fallback_temporal_mixer is not None:
                    self.fallback_temporal_mixer.eval()
                if self.video_mamba is not None:
                    self.video_mamba.eval()
                    self.video_to_robot.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        if isinstance(self.vlln, nn.LayerNorm):
            norm_dtype = self.vlln.weight.dtype
            backbone_features = self.vlln(backbone_features.to(dtype=norm_dtype))
        else:
            backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _assert_contract(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        require_action: bool,
    ):
        required_backbone_keys = ["backbone_features"]
        required_action_keys = ["state", "embodiment_id"]
        if require_action:
            required_action_keys.extend(["action", "action_mask"])

        missing_backbone = [k for k in required_backbone_keys if k not in backbone_output]
        missing_action = [k for k in required_action_keys if k not in action_input]
        if missing_backbone:
            raise KeyError(f"MambaSystem1 missing backbone keys: {missing_backbone}")
        if missing_action:
            raise KeyError(f"MambaSystem1 missing action_input keys: {missing_action}")

        backbone_features = backbone_output["backbone_features"]
        state = action_input["state"]
        embodiment_id = action_input["embodiment_id"]

        assert backbone_features.ndim == 3, (
            f"Expected backbone_features [B,L,C], got {_shape(backbone_features)}"
        )
        assert state.ndim == 3, f"Expected state [B,S,D], got {_shape(state)}"
        assert embodiment_id.ndim == 1, f"Expected embodiment_id [B], got {_shape(embodiment_id)}"
        assert state.shape[0] == backbone_features.shape[0], (
            f"Batch mismatch state/backbone: {_shape(state)} vs {_shape(backbone_features)}"
        )
        assert embodiment_id.shape[0] == state.shape[0], (
            f"Batch mismatch embodiment/state: {_shape(embodiment_id)} vs {_shape(state)}"
        )

        if "backbone_attention_mask" in backbone_output:
            backbone_attention_mask = backbone_output["backbone_attention_mask"]
            assert backbone_attention_mask.ndim == 2, (
                "Expected backbone_attention_mask [B,L], "
                f"got {_shape(backbone_attention_mask)}"
            )
            assert backbone_attention_mask.shape[:2] == backbone_features.shape[:2], (
                f"Mask/backbone mismatch: {_shape(backbone_attention_mask)} "
                f"vs {_shape(backbone_features)}"
            )

        if "image_mask" in backbone_output:
            image_mask = backbone_output["image_mask"]
            assert image_mask.ndim == 2, f"Expected image_mask [B,L], got {_shape(image_mask)}"
            assert image_mask.shape[:2] == backbone_features.shape[:2], (
                f"image_mask/backbone mismatch: {_shape(image_mask)} "
                f"vs {_shape(backbone_features)}"
            )

        if require_action:
            action = action_input["action"]
            action_mask = action_input["action_mask"]
            assert action.ndim == 3, f"Expected action [B,T,D], got {_shape(action)}"
            assert action_mask.ndim == 3, (
                f"Expected action_mask [B,T,D], got {_shape(action_mask)}"
            )
            assert action.shape == action_mask.shape, (
                f"action/action_mask mismatch: {_shape(action)} vs {_shape(action_mask)}"
            )
            assert action.shape[0] == state.shape[0], (
                f"Batch mismatch action/state: {_shape(action)} vs {_shape(state)}"
            )

    def _log_contract_once(
        self,
        mode: str,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ):
        if not self.system1_debug_shapes:
            return
        if mode == "train" and self._logged_train_contract:
            return
        if mode == "infer" and self._logged_infer_contract:
            return

        logging.info(
            "[System1:mamba][%s] action_input keys=%s backbone_output keys=%s",
            mode,
            sorted(action_input.keys()),
            sorted(backbone_output.keys()),
        )
        logging.info(
            "[System1:mamba][%s] shapes: state=%s embodiment_id=%s action=%s action_mask=%s "
            "backbone_features=%s backbone_attention_mask=%s image_mask=%s",
            mode,
            _shape(action_input.get("state")),
            _shape(action_input.get("embodiment_id")),
            _shape(action),
            _shape(action_mask),
            _shape(backbone_output.get("backbone_features")),
            _shape(backbone_output.get("backbone_attention_mask")),
            _shape(backbone_output.get("image_mask")),
        )

        if mode == "train":
            self._logged_train_contract = True
        if mode == "infer":
            self._logged_infer_contract = True

    def _encode_video_context(self, backbone_output: BatchFeature) -> torch.Tensor | None:
        if not self.mamba_enable_video_injection or self.video_mamba is None:
            return None
        vl_tokens = backbone_output["backbone_features"]
        image_mask = backbone_output.get("image_mask")
        return self.video_mamba(vl_tokens, image_mask)

    def _log_non_finite_once(self, name: str, x: torch.Tensor):
        if not self.system1_debug_shapes:
            return
        if name in self._finite_debug_logged:
            return
        if torch.isfinite(x).all():
            return
        self._finite_debug_logged.add(name)
        logging.warning(
            "[System1:mamba] Non-finite tensor at %s: shape=%s nan=%d inf=%d",
            name,
            _shape(x),
            int(torch.isnan(x).sum().item()),
            int(torch.isinf(x).sum().item()),
        )

    def _register_gradient_probe(self, name: str, x: torch.Tensor):
        if not self.system1_debug_gradients or not self.training:
            return
        if not torch.is_tensor(x) or not x.requires_grad:
            return

        def _hook(grad: torch.Tensor):
            if name in self._gradient_debug_logged:
                return grad
            if torch.isfinite(grad).all():
                return grad
            self._gradient_debug_logged.add(name)
            logging.error(
                "[System1:mamba] Non-finite gradient at %s: shape=%s nan=%d inf=%d abs_max=%.4f",
                name,
                _shape(grad),
                int(torch.isnan(grad).sum().item()),
                int(torch.isinf(grad).sum().item()),
                float(torch.nan_to_num(grad).abs().max().item()),
            )
            return grad

        x.register_hook(_hook)

    def _encode_state_features(self, action_input: BatchFeature) -> torch.Tensor:
        state_features = self.state_encoder(action_input["state"], action_input["embodiment_id"])

        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        if self.training and self.state_additive_noise_scale > 0:
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise
        return state_features

    def _safe_category_linear_forward(
        self,
        layer: CategorySpecificMLP | nn.Module,
        x: torch.Tensor,
        cat_ids: torch.Tensor,
        name: str | None = None,
    ) -> torch.Tensor:
        selected_w = layer.W[cat_ids].float()
        selected_b = layer.b[cat_ids].float()
        x = torch.clamp(
            torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4), min=-1e3, max=1e3
        )
        if name is not None:
            self._log_non_finite_once(f"{name}_input", x)
            self._register_gradient_probe(f"{name}_input", x)
        y = torch.bmm(x, selected_w) + selected_b.unsqueeze(1)
        y = torch.clamp(torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)
        if name is not None:
            self._log_non_finite_once(name, y)
            self._register_gradient_probe(name, y)
        return y

    def _safe_action_encoder(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != bsz:
            raise ValueError("Expected `timesteps` to have shape (B,).")
        timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)

        a_emb = self._safe_category_linear_forward(
            self.action_encoder.W1, actions, cat_ids, name="action_encoder_w1"
        )
        tau_emb = self.action_encoder.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        self._log_non_finite_once("action_encoder_tau_emb", tau_emb)
        self._register_gradient_probe("action_encoder_tau_emb", tau_emb)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        self._log_non_finite_once("action_encoder_concat", x)
        self._register_gradient_probe("action_encoder_concat", x)
        x = self._safe_category_linear_forward(
            self.action_encoder.W2, x, cat_ids, name="action_encoder_w2"
        )
        x = swish(x)
        self._log_non_finite_once("action_encoder_swish", x)
        self._register_gradient_probe("action_encoder_swish", x)
        x = self._safe_category_linear_forward(
            self.action_encoder.W3, x, cat_ids, name="action_encoder_w3"
        )
        return x.float()

    def _safe_action_decoder(self, hidden: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        x = self._safe_category_linear_forward(
            self.action_decoder.layer1, hidden, cat_ids, name="action_decoder_layer1"
        )
        x = F.relu(x)
        self._log_non_finite_once("action_decoder_relu", x)
        self._register_gradient_probe("action_decoder_relu", x)
        x = self._safe_category_linear_forward(
            self.action_decoder.layer2, x, cat_ids, name="action_decoder_layer2"
        )
        return x.float()

    def _apply_robot_mamba(self, robot_hidden: torch.Tensor) -> torch.Tensor:
        if self.robot_mamba is None:
            raise RuntimeError("robot_mamba is unavailable for the stable temporal fallback backend")
        mode = self.debug_robot_mamba_mode
        if mode == "skip":
            return torch.nan_to_num(robot_hidden, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e4, 1e4
            )
        if mode == "norm_only":
            x = robot_hidden
            for layer in self.robot_mamba.layers:
                x = layer.norm(
                    torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
                )
            x = self.robot_mamba.final_norm(x)
            return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        if mode == "core_only":
            x = robot_hidden
            for layer in self.robot_mamba.layers:
                residual = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
                y = layer.core(residual)
                y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
                x = torch.nan_to_num(residual + layer.dropout(y), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        if mode != "full":
            raise ValueError(f"Unsupported debug_robot_mamba_mode={mode}")
        return self.robot_mamba(robot_hidden)

    def _apply_fallback_temporal_mixer(
        self,
        action_hidden: torch.Tensor,
        state_hidden: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        if self.fallback_temporal_mixer is None:
            raise RuntimeError("fallback_temporal_mixer is unavailable when mamba_ssm backend is active")
        if self.debug_robot_mamba_mode == "skip":
            return torch.nan_to_num(action_hidden, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                -1e4, 1e4
            )
        if self.debug_robot_mamba_mode != "full":
            raise ValueError(
                "debug_robot_mamba_mode only supports 'full' or 'skip' for the stable temporal fallback path"
            )

        state_condition = state_hidden.mean(dim=1)
        if self.video_to_robot is not None:
            video_context = self._encode_video_context(backbone_output)
            state_condition = state_condition + video_context.squeeze(1).to(dtype=state_condition.dtype)

        action_hidden = self.fallback_temporal_mixer(action_hidden, state_condition)
        self._log_non_finite_once("fallback_action_hidden_after_temporal_mixer", action_hidden)
        self._register_gradient_probe(
            "fallback_action_hidden_after_temporal_mixer", action_hidden
        )
        return action_hidden

    def _predict_velocity(
        self,
        action_trajectory: torch.Tensor,
        t_discretized: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        action_features = self._safe_action_encoder(
            actions=action_trajectory, timesteps=t_discretized, cat_ids=embodiment_id
        )
        self._log_non_finite_once("action_features", action_features)
        self._register_gradient_probe("action_features", action_features)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=action_features.device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            pos_embs = torch.clamp(
                torch.nan_to_num(pos_embs, nan=0.0, posinf=1e4, neginf=-1e4), -1e3, 1e3
            ).to(dtype=action_features.dtype)
            action_features = torch.clamp(
                torch.nan_to_num(action_features + pos_embs, nan=0.0, posinf=1e4, neginf=-1e4),
                -1e3,
                1e3,
            )
            self._log_non_finite_once("action_features_with_pos", action_features)
            self._register_gradient_probe("action_features_with_pos", action_features)

        state_hidden = F.linear(
            torch.clamp(
                torch.nan_to_num(state_features.float(), nan=0.0, posinf=1e4, neginf=-1e4),
                min=-1e3,
                max=1e3,
            ),
            self.state_token_proj.weight.float(),
            None
            if self.state_token_proj.bias is None
            else self.state_token_proj.bias.float(),
        ).float()
        state_hidden = torch.nan_to_num(state_hidden, nan=0.0, posinf=1e4, neginf=-1e4)

        action_hidden = F.linear(
            torch.clamp(
                torch.nan_to_num(action_features.float(), nan=0.0, posinf=1e4, neginf=-1e4),
                min=-1e3,
                max=1e3,
            ),
            self.action_token_proj.weight.float(),
            None
            if self.action_token_proj.bias is None
            else self.action_token_proj.bias.float(),
        ).float()
        action_hidden = torch.nan_to_num(action_hidden, nan=0.0, posinf=1e4, neginf=-1e4)
        self._log_non_finite_once("state_hidden", state_hidden)
        self._log_non_finite_once("action_hidden", action_hidden)
        self._register_gradient_probe("state_hidden", state_hidden)
        self._register_gradient_probe("action_hidden", action_hidden)

        if self.fallback_temporal_mixer is not None:
            action_hidden = self._apply_fallback_temporal_mixer(
                action_hidden=action_hidden,
                state_hidden=state_hidden,
                backbone_output=backbone_output,
            )
            pred = self._safe_action_decoder(action_hidden, embodiment_id)
            self._log_non_finite_once("action_decoder_pred", pred)
            self._register_gradient_probe("action_decoder_pred", pred)
            return pred

        robot_hidden = torch.cat((state_hidden, action_hidden), dim=1)
        self._log_non_finite_once("robot_hidden_concat", robot_hidden)
        self._register_gradient_probe("robot_hidden_concat", robot_hidden)

        if self.video_to_robot is not None:
            video_context = self._encode_video_context(backbone_output)
            robot_hidden = self.video_to_robot(robot_hidden, video_context)
            self._log_non_finite_once("robot_hidden_after_video_injection", robot_hidden)
            self._register_gradient_probe("robot_hidden_after_video_injection", robot_hidden)

        robot_hidden = self._apply_robot_mamba(robot_hidden)
        self._log_non_finite_once("robot_hidden_after_mamba", robot_hidden)
        self._register_gradient_probe("robot_hidden_after_mamba", robot_hidden)
        pred = self._safe_action_decoder(robot_hidden, embodiment_id)
        self._log_non_finite_once("action_decoder_pred", pred)
        self._register_gradient_probe("action_decoder_pred", pred)
        return pred[:, -action_trajectory.shape[1] :]

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)
        self._assert_contract(backbone_output, action_input, require_action=True)
        self._log_contract_once(
            mode="train",
            backbone_output=backbone_output,
            action_input=action_input,
            action=action_input["action"],
            action_mask=action_input["action_mask"],
        )

        embodiment_id = action_input["embodiment_id"]
        state_features = self._encode_state_features(action_input)

        actions = action_input["action"]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        t_discretized = t_discretized.clamp(min=0, max=self.num_timestep_buckets - 1)

        pred_actions = self._predict_velocity(
            action_trajectory=noisy_trajectory,
            t_discretized=t_discretized,
            state_features=state_features,
            embodiment_id=embodiment_id,
            backbone_output=backbone_output,
        )

        pred_actions_f = pred_actions.float()
        velocity_f = velocity.float()
        action_mask = action_input["action_mask"].float()
        self._register_gradient_probe("pred_actions_float", pred_actions_f)

        pred_non_finite = ~torch.isfinite(pred_actions_f)
        vel_non_finite = ~torch.isfinite(velocity_f)
        if (pred_non_finite.any() or vel_non_finite.any()) and not self._logged_non_finite_warning:
            logging.warning(
                "[System1:mamba] Non-finite values detected before loss. "
                "pred_non_finite=%d vel_non_finite=%d pred_abs_max=%.4f vel_abs_max=%.4f",
                int(pred_non_finite.sum().item()),
                int(vel_non_finite.sum().item()),
                float(torch.nan_to_num(pred_actions_f).abs().max().item()),
                float(torch.nan_to_num(velocity_f).abs().max().item()),
            )
            self._logged_non_finite_warning = True

        pred_actions_f = torch.nan_to_num(pred_actions_f, nan=0.0, posinf=1e4, neginf=-1e4)
        velocity_f = torch.nan_to_num(velocity_f, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_minus_velocity = pred_actions_f - velocity_f
        self._log_non_finite_once("pred_minus_velocity", pred_minus_velocity)
        self._register_gradient_probe("pred_minus_velocity", pred_minus_velocity)
        squared_error = torch.clamp(pred_minus_velocity**2, max=1e4)
        action_loss = squared_error * action_mask
        loss = action_loss.sum() / action_mask.sum().clamp_min(1.0)
        self._register_gradient_probe("squared_error", squared_error)
        self._register_gradient_probe("action_loss", action_loss)
        self._register_gradient_probe("loss", loss)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": backbone_output["backbone_features"],
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        self._assert_contract(backbone_output, action_input, require_action=False)
        self._log_contract_once(
            mode="infer",
            backbone_output=backbone_output,
            action_input=action_input,
        )

        state_features = self.state_encoder(action_input["state"], action_input["embodiment_id"])
        return BatchFeature(
            data={
                "backbone_features": backbone_output["backbone_features"],
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        batch_size = backbone_features.shape[0]
        device = backbone_features.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=backbone_features.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device,
                dtype=torch.long,
            )

            pred_velocity = self._predict_velocity(
                action_trajectory=actions,
                t_discretized=timesteps_tensor,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
            )
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": backbone_features,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input["embodiment_id"],
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
