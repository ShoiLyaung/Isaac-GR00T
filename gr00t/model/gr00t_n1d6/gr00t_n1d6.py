import logging
from typing import Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.mamba_system1 import MambaSystem1ActionHead
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


def _shape(x):
    if x is None:
        return "None"
    if torch.is_tensor(x):
        return str(tuple(x.shape))
    return str(type(x))


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

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

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.system1_debug_shapes = config.system1_debug_shapes
        self._logged_train_contract = False
        self._logged_infer_contract = False
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

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
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

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
        self, backbone_output: BatchFeature, action_input: BatchFeature, require_action: bool
    ):
        required_backbone_keys = ["backbone_features", "backbone_attention_mask"]
        required_action_keys = ["state", "embodiment_id"]
        if require_action:
            required_action_keys.extend(["action", "action_mask"])

        missing_backbone = [k for k in required_backbone_keys if k not in backbone_output]
        missing_action = [k for k in required_action_keys if k not in action_input]
        if missing_backbone:
            raise KeyError(f"DiT System1 missing backbone keys: {missing_backbone}")
        if missing_action:
            raise KeyError(f"DiT System1 missing action_input keys: {missing_action}")

        backbone_features = backbone_output["backbone_features"]
        backbone_attention_mask = backbone_output["backbone_attention_mask"]
        state = action_input["state"]
        embodiment_id = action_input["embodiment_id"]

        assert backbone_features.ndim == 3, (
            f"Expected backbone_features [B,L,C], got {_shape(backbone_features)}"
        )
        assert backbone_attention_mask.ndim == 2, (
            "Expected backbone_attention_mask [B,L], "
            f"got {_shape(backbone_attention_mask)}"
        )
        assert backbone_attention_mask.shape[:2] == backbone_features.shape[:2], (
            f"Mask/backbone mismatch: {_shape(backbone_attention_mask)} "
            f"vs {_shape(backbone_features)}"
        )
        assert state.ndim == 3, f"Expected state [B,S,D], got {_shape(state)}"
        assert embodiment_id.ndim == 1, f"Expected embodiment_id [B], got {_shape(embodiment_id)}"
        assert state.shape[0] == backbone_features.shape[0], (
            f"Batch mismatch state/backbone: {_shape(state)} vs {_shape(backbone_features)}"
        )
        assert embodiment_id.shape[0] == state.shape[0], (
            f"Batch mismatch embodiment/state: {_shape(embodiment_id)} vs {_shape(state)}"
        )

        if "image_mask" in backbone_output:
            image_mask = backbone_output["image_mask"]
            assert image_mask.ndim == 2, f"Expected image_mask [B,L], got {_shape(image_mask)}"
            assert image_mask.shape[:2] == backbone_features.shape[:2], (
                f"image_mask/backbone mismatch: {_shape(image_mask)} vs {_shape(backbone_features)}"
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
            "[System1:dit][%s] action_input keys=%s backbone_output keys=%s",
            mode,
            sorted(action_input.keys()),
            sorted(backbone_output.keys()),
        )
        logging.info(
            "[System1:dit][%s] shapes: state=%s embodiment_id=%s action=%s action_mask=%s "
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

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_horizon, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
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

        # Get vision and language embeddings.
        vl_embeds = backbone_output["backbone_features"]
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input["embodiment_id"]

        # Embed state.
        state_features = self.state_encoder(action_input["state"], embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input["action"]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output["backbone_attention_mask"]

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output["image_mask"]
            backbone_attention_mask = backbone_output["backbone_attention_mask"]
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input["action_mask"]
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_horizon, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)
        self._assert_contract(backbone_output, action_input, require_action=False)
        self._log_contract_once(
            mode="infer",
            backbone_output=backbone_output,
            action_input=action_input,
        )

        # Get vision and language embeddings.
        vl_embeds = backbone_output["backbone_features"]
        embodiment_id = action_input["embodiment_id"]

        # Embed state.
        state_features = self.state_encoder(action_input["state"], embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_horizon, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
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
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize System1 action head
        if config.system1_type == "dit":
            self.action_head = Gr00tN1d6ActionHead(config)
        elif config.system1_type == "mamba":
            self.action_head = MambaSystem1ActionHead(config)
        else:
            raise ValueError(
                f"Unsupported system1_type={config.system1_type}. "
                "Expected one of ['dit', 'mamba']."
            )
        logging.info("Initialized System1 type: %s", config.system1_type)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # In DataParallel replicas, parameters can be materialized as plain tensors,
        # so iter(self.parameters()) may be empty. Fall back to keeping tensor device/dtype.
        model_device, model_dtype = self._get_target_device_dtype()
        preferred_input_dtype = self._get_preferred_input_dtype()

        # Move to device and dtype
        def to_device_with_dtype(x):
            if not torch.is_tensor(x):
                return x
            target_device = model_device if model_device is not None else x.device
            if torch.is_floating_point(x):
                target_dtype = preferred_input_dtype or model_dtype or x.dtype
                return x.to(target_device, dtype=target_dtype)
            else:
                return x.to(target_device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def _get_target_device_dtype(self):
        for p in self.parameters():
            return p.device, p.dtype
        # DataParallel replicas may move parameters into `_former_parameters`.
        for module in self.modules():
            former = getattr(module, "_former_parameters", None)
            if former:
                for p in former.values():
                    if p is not None and torch.is_tensor(p):
                        return p.device, p.dtype
        for b in self.buffers():
            return b.device, b.dtype
        return None, None

    def _get_preferred_input_dtype(self):
        model_dtype = str(getattr(self.config, "model_dtype", "")).lower()
        if model_dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if model_dtype in ("fp16", "float16", "half"):
            return torch.float16
        if model_dtype in ("fp32", "float32"):
            return torch.float32
        return None

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype
        for b in self.buffers():
            if torch.is_floating_point(b):
                return b.dtype
        return torch.get_default_dtype()


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
