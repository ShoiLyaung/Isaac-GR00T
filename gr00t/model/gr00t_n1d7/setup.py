# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import run_on_rank0
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.registry import register_model


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


def _save_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def _save_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(value)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

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
        self.codec_evidence_config = self._resolve_codec_evidence_config()
        if self.codec_evidence_config is not None:
            self.model_config.codec_evidence = self.codec_evidence_config.to_model_dict()
        else:
            preseeded_codec = getattr(self.model_config, "codec_evidence", None)
            if preseeded_codec is not None:
                if type(preseeded_codec) is not dict:
                    raise TypeError("Preseeded model codec_evidence must be a dict or None")
                preseeded_enabled = preseeded_codec.get("enabled", False)
                if type(preseeded_enabled) is not bool:
                    raise TypeError("Preseeded model codec_evidence.enabled must be boolean")
                if preseeded_enabled:
                    raise ValueError(
                        "Enabled model codec_evidence requires an enabled all-LeRobot adapter config"
                    )
            self.model_config.codec_evidence = None
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _resolve_codec_evidence_config(self):
        """Resolve one run-wide optional Codec contract from leaf adapter configs.

        Disabled blocks intentionally avoid importing the Codec package. If any
        leaf enables Codec evidence, every leaf must enable the exact same
        fixture/model contract so the mixture cannot silently drop evidence.
        """

        codec_capable_adapter = "g1_all_lerobot_h40_v1"
        resolved = []
        for dataset_spec in self.config.data.datasets:
            if dataset_spec.adapter_config_path is None:
                if dataset_spec.dataset_adapter == codec_capable_adapter:
                    raise ValueError(f"{dataset_spec.dataset_adapter} requires adapter_config_path")
                resolved.append(None)
                continue
            with Path(dataset_spec.adapter_config_path).open("r", encoding="utf-8") as stream:
                adapter_config = json.load(stream, object_pairs_hook=_reject_duplicate_json_keys)
            codec_block = adapter_config.get("codec_evidence")
            if codec_block is None:
                resolved.append(None)
                continue
            if not isinstance(codec_block, dict):
                raise TypeError("codec_evidence must be an object")
            enabled = codec_block.get("enabled", False)
            if not isinstance(enabled, bool):
                raise TypeError("codec_evidence.enabled must be boolean")
            if not enabled:
                resolved.append(None)
                continue
            if dataset_spec.dataset_adapter != codec_capable_adapter:
                raise ValueError(
                    "Enabled codec_evidence is supported only for "
                    f"{codec_capable_adapter}, not {dataset_spec.dataset_adapter}"
                )

            from gr00t.data.codec_evidence.config import parse_codec_evidence_config

            parsed_codec = parse_codec_evidence_config(codec_block)
            if parsed_codec.schema_version != "codec-evidence-episode-bundle-v1":
                raise ValueError(
                    "This all-LeRobot Codec branch requires "
                    "codec-evidence-episode-bundle-v1 before model creation"
                )
            resolved.append(parsed_codec)

        if not resolved or all(item is None for item in resolved):
            return None
        first = resolved[0]
        if first is None or any(item != first for item in resolved[1:]):
            raise ValueError(
                "Every dataset in one run must use identical enabled codec_evidence config"
            )
        return first

    def _checkpoint_codec_load_mode(self, checkpoint: str | Path, codec_config) -> str:
        """Classify a checkpoint before model construction using read-only config identity."""

        checkpoint_config = AutoConfig.from_pretrained(
            checkpoint, **self.transformers_loading_kwargs
        )
        expected_model_type = getattr(self.model_config, "model_type", None)
        checkpoint_model_type = getattr(checkpoint_config, "model_type", None)
        if expected_model_type is not None and checkpoint_model_type != expected_model_type:
            raise ValueError(
                "Checkpoint model_type does not match the requested model: "
                f"{checkpoint_model_type!r} != {expected_model_type!r}"
            )

        saved_codec = getattr(checkpoint_config, "codec_evidence", None)
        if saved_codec is not None:
            if type(saved_codec) is not dict:
                raise TypeError("Checkpoint codec_evidence must be a dict or None")
            saved_enabled = saved_codec.get("enabled", False)
            if type(saved_enabled) is not bool:
                raise TypeError("Checkpoint codec_evidence.enabled must be boolean")
            if not saved_enabled:
                saved_codec = None
            else:
                from gr00t.data.codec_evidence.config import parse_codec_model_config

                saved_codec = parse_codec_model_config(saved_codec)

        requested_codec = codec_config.to_model_dict() if codec_config is not None else None
        if saved_codec is None:
            return "native_to_codec" if requested_codec is not None else "native"
        if requested_codec is None:
            raise ValueError(
                "Checkpoint enables codec_evidence but the current dataset configuration does not"
            )
        if saved_codec != requested_codec:
            raise ValueError(
                "Checkpoint codec_evidence config differs from the current dataset config: "
                f"saved={saved_codec!r}, requested={requested_codec!r}"
            )
        return "codec_reload"

    def _checkpoint_processor_codec_config(self, checkpoint: str | Path) -> dict | None:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute() or not checkpoint_path.is_dir():
            raise ValueError(
                "Offline processor identity validation requires an existing absolute "
                "checkpoint directory"
            )
        processor_config_path = checkpoint_path / "processor_config.json"
        if not processor_config_path.is_file():
            return None

        with processor_config_path.open("r", encoding="utf-8") as stream:
            processor_config = json.load(stream, object_pairs_hook=_reject_duplicate_json_keys)
        if not isinstance(processor_config, dict):
            raise TypeError("Checkpoint processor_config.json must contain an object")
        processor_kwargs = processor_config.get("processor_kwargs")
        if not isinstance(processor_kwargs, dict):
            raise TypeError("Checkpoint processor_config.json requires processor_kwargs")
        saved_codec = processor_kwargs.get("codec_evidence")
        if saved_codec is None:
            return None
        if type(saved_codec) is not dict:
            raise TypeError("Checkpoint processor codec_evidence must be a dict or absent")
        saved_enabled = saved_codec.get("enabled", False)
        if type(saved_enabled) is not bool:
            raise TypeError("Checkpoint processor codec_evidence.enabled must be boolean")
        if not saved_enabled:
            return None

        from gr00t.data.codec_evidence.config import parse_codec_model_config

        return parse_codec_model_config(saved_codec)

    def _validate_checkpoint_processor_codec_identity(
        self,
        checkpoint: str | Path,
        codec_config,
        load_mode: str,
    ) -> None:
        if load_mode == "native":
            return
        saved_processor_codec = self._checkpoint_processor_codec_config(checkpoint)
        requested_codec = codec_config.to_model_dict() if codec_config is not None else None
        if load_mode == "codec_reload":
            if saved_processor_codec is None:
                raise ValueError(
                    "Codec checkpoint requires enabled codec_evidence in processor_config.json"
                )
            if saved_processor_codec != requested_codec:
                raise ValueError(
                    "Checkpoint processor codec_evidence differs from the model/dataset config"
                )
            return
        if saved_processor_codec is not None:
            raise ValueError(
                "Native checkpoint model identity conflicts with enabled processor codec_evidence"
            )

    @staticmethod
    def _expected_codec_state_keys(model) -> set[str]:
        conditioning = getattr(model.action_head, "codec_conditioning", None)
        if conditioning is None:
            raise RuntimeError("Enabled codec load has no action_head.codec_conditioning module")
        prefix = "action_head.codec_conditioning."
        expected = {f"{prefix}{key}" for key in conditioning.state_dict()}
        if not expected:
            raise RuntimeError("Codec conditioning module has no checkpoint state")
        return expected

    def _validate_codec_missing_keys(self, model, loading_info: dict, load_mode: str) -> set[str]:
        """Return the only Codec missing keys allowed by the checkpoint identity."""

        prefix = "action_head.codec_conditioning."
        missing = set(loading_info.get("missing_keys", []))
        codec_missing = {key for key in missing if key.startswith(prefix)}
        if load_mode == "native_to_codec":
            expected = self._expected_codec_state_keys(model)
            if codec_missing != expected:
                raise RuntimeError(
                    "Native-to-Codec initialization requires the complete new module key set: "
                    f"expected={sorted(expected)}, observed={sorted(codec_missing)}"
                )
            return expected
        if codec_missing:
            raise RuntimeError(
                f"{load_mode} checkpoint is missing Codec keys: {sorted(codec_missing)}"
            )
        return set()

    @staticmethod
    def _initialize_native_to_codec_gate(
        model, load_mode: str, allowed_codec_missing: set[str]
    ) -> None:
        """Restore the exact-zero gate after Transformers initializes missing extension keys."""

        if load_mode != "native_to_codec":
            return
        gate_key = "action_head.codec_conditioning.conditioner.gate"
        if gate_key not in allowed_codec_missing:
            raise RuntimeError("Native-to-Codec initialization did not classify the gate as new")
        conditioning = getattr(model.action_head, "codec_conditioning", None)
        if conditioning is None:
            raise RuntimeError("Native-to-Codec initialization lacks the conditioning stack")
        gate = conditioning.conditioner.gate
        with torch.no_grad():
            gate.zero_()
        if gate.ndim != 0 or gate.item() != 0.0:
            raise RuntimeError("Native-to-Codec gate failed exact-zero initialization")

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        codec_model_kwargs = {}
        codec_config = getattr(self, "codec_evidence_config", None)
        if codec_config is not None:
            codec_model_kwargs["codec_evidence"] = codec_config.to_model_dict()
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            codec_load_mode = self._checkpoint_codec_load_mode(
                self.config.training.start_from_checkpoint, codec_config
            )
            self._validate_checkpoint_processor_codec_identity(
                self.config.training.start_from_checkpoint,
                codec_config,
                codec_load_mode,
            )
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                action_horizon=self.model_config.action_horizon,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                codec_trainability=self.config.model.codec_trainability,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                load_bf16=self.config.model.load_bf16,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **codec_model_kwargs,
                **self.transformers_loading_kwargs,
            )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            allowed_codec_missing = self._validate_codec_missing_keys(
                model, loading_info, codec_load_mode
            )
            self._initialize_native_to_codec_gate(
                model,
                codec_load_mode,
                allowed_codec_missing,
            )
            other_missing = [
                key
                for key in missing_keys
                if "mask_token" not in key and key not in allowed_codec_missing
            ]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(f"Model Config: {model.config}")
        run_on_rank0(
            _save_text,
            self.save_cfg_dir / "final_model_config.json",
            model.config.to_filtered_json(),
            label="gr00t_n1d7.final_model_config",
        )
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        codec_processor_kwargs = {}
        codec_config = getattr(self, "codec_evidence_config", None)
        if codec_config is not None:
            codec_processor_kwargs["codec_evidence"] = codec_config.to_model_dict()
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
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
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **codec_processor_kwargs,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
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
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **codec_processor_kwargs,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        run_on_rank0(
            _save_json,
            self.save_cfg_dir / "final_processor_config.json",
            {key: str(value) for key, value in vars(processor).items()},
            label="gr00t_n1d7.final_processor_config",
        )

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        run_on_rank0(
            _save_json,
            save_cfg_dir / "dataset_statistics.json",
            stats_dict,
            label="gr00t_n1d7.dataset_statistics",
        )
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
