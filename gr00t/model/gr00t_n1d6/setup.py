import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.registry import register_model
import numpy as np
from termcolor import colored
import torch
from torch import nn
from transformers import AutoModel, AutoProcessor


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d6Pipeline(ModelPipeline):
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""

        # Build transformers loading kwargs from training config

        if self.config.training.start_from_checkpoint is not None:
            ignore_mismatched_sizes = self.config.model.system1_type == "mamba"
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                system1_type=self.config.model.system1_type,
                system1_debug_shapes=self.config.model.system1_debug_shapes,
                system1_debug_gradients=self.config.model.system1_debug_gradients,
                debug_robot_mamba_mode=self.config.model.debug_robot_mamba_mode,
                mamba_num_layers=self.config.model.mamba_num_layers,
                mamba_d_state=self.config.model.mamba_d_state,
                mamba_d_conv=self.config.model.mamba_d_conv,
                mamba_expand=self.config.model.mamba_expand,
                mamba_dropout=self.config.model.mamba_dropout,
                require_real_mamba_backend=self.config.model.require_real_mamba_backend,
                mamba_enable_video_injection=self.config.model.mamba_enable_video_injection,
                mamba_video_num_layers=self.config.model.mamba_video_num_layers,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            if ignore_mismatched_sizes:
                logging.info(
                    "System1 type is mamba; checkpoint loading uses ignore_mismatched_sizes=True "
                    "(equivalent to non-strict for shape-mismatched keys)."
                )
            logging.info(
                "Checkpoint load summary: missing=%d unexpected=%d mismatched=%d",
                len(loading_info.get("missing_keys", [])),
                len(loading_info.get("unexpected_keys", [])),
                len(loading_info.get("mismatched_keys", [])),
            )

            # Initialize mask_tokens if they are not present in the base checkpoint
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)

            if mask_token_missing and model.action_head.mask_token is not None:
                # Initialize mask_token
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            if self.config.model.system1_type == "mamba":
                self._reinitialize_mamba_action_head_parameters(model, loading_info)

        else:
            model = self.model_class(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        print("Model: ", model)

        return model

    def _reinitialize_mamba_action_head_parameters(self, model, loading_info: dict):
        named_modules = dict(model.named_modules())
        modules_to_reset = set()

        for key in loading_info.get("missing_keys", []):
            if key.startswith("action_head."):
                modules_to_reset.add(key.rsplit(".", 1)[0])

        for name, param in model.named_parameters():
            if name.startswith("action_head.") and not torch.isfinite(param).all():
                modules_to_reset.add(name.rsplit(".", 1)[0])

        reset_count = 0
        for module_name in sorted(modules_to_reset, key=lambda x: x.count(".")):
            module = named_modules.get(module_name)
            if module is None:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
                reset_count += 1
            elif hasattr(module, "_reset_parameters"):
                module._reset_parameters()
                reset_count += 1

        remaining_non_finite = []
        for name, param in model.named_parameters():
            if not name.startswith("action_head.") or torch.isfinite(param).all():
                continue
            remaining_non_finite.append(name)
            with torch.no_grad():
                if name.endswith(".bias"):
                    param.zero_()
                elif param.ndim == 1 and "norm" in name:
                    param.fill_(1.0)
                elif param.ndim == 1:
                    param.zero_()
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)

        final_non_finite = [
            name
            for name, param in model.named_parameters()
            if name.startswith("action_head.") and not torch.isfinite(param).all()
        ]
        logging.info(
            "Reinitialized Mamba action head parameters: reset_modules=%d remaining_non_finite_after_manual=%d",
            reset_count,
            len(final_non_finite),
        )
        if final_non_finite:
            logging.warning("Still non-finite after Mamba reinit: sample=%s", final_non_finite[:20])

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""

        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        print(
            colored(
                f"These are all the processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}",
                "yellow",
            )
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d6Config, Gr00tN1d6Pipeline)
