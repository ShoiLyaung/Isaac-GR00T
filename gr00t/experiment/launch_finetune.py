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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import math
import os
from pathlib import Path
from typing import Any

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


DATASET_MANIFEST_SCHEMA_VERSION = "gr00t-finetune-dataset-manifest-v1"


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {path}")
    return value


def resolve_dataset_specs(ft_config: FinetuneConfig, embodiment_tag: str) -> list[dict[str, Any]]:
    """Resolve the native dataset path mode or one strict adapter manifest."""

    has_path = bool(ft_config.dataset_path)
    has_manifest = bool(ft_config.dataset_manifest_path)
    if has_path == has_manifest:
        raise ValueError("Exactly one of dataset_path and dataset_manifest_path is required")

    if has_path:
        dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]
        if not dataset_paths:
            raise ValueError("dataset_path did not contain a dataset root")
        if ft_config.dataset_adapter is not None and len(dataset_paths) != 1:
            raise ValueError("Adapter-backed dataset_path mode requires exactly one root")
        if ft_config.dataset_adapter is not None and ft_config.adapter_config_path is None:
            raise ValueError("Adapter-backed dataset_path mode requires adapter_config_path")
        return [
            {
                "dataset_paths": dataset_paths,
                "mix_ratio": 1.0,
                "embodiment_tag": embodiment_tag,
                "dataset_adapter": ft_config.dataset_adapter,
                "adapter_config_path": ft_config.adapter_config_path,
            }
        ]

    manifest_path = Path(ft_config.dataset_manifest_path).resolve()
    manifest = _load_json_object(manifest_path)
    if set(manifest) != {"schema_version", "datasets"}:
        raise ValueError("Dataset manifest keys must be exactly schema_version and datasets")
    if manifest["schema_version"] != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset manifest schema: {manifest['schema_version']!r}")
    entries = manifest["datasets"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Dataset manifest datasets must be a non-empty list")
    if ft_config.dataset_adapter is None:
        raise ValueError("dataset_manifest_path requires dataset_adapter")

    specs: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        required = {"dataset_path", "mix_ratio", "adapter_config_path"}
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(
                f"Dataset manifest entry {index} keys must be exactly "
                "dataset_path, mix_ratio, and adapter_config_path"
            )
        dataset_path = Path(entry["dataset_path"])
        if not dataset_path.is_absolute() or not dataset_path.is_dir():
            raise ValueError(f"Dataset manifest entry {index} must name an existing absolute root")
        dataset_path = dataset_path.resolve()
        if dataset_path in seen_paths:
            raise ValueError(f"Duplicate dataset root in manifest: {dataset_path}")
        seen_paths.add(dataset_path)

        mix_ratio = entry["mix_ratio"]
        if (
            isinstance(mix_ratio, bool)
            or not isinstance(mix_ratio, (int, float))
            or not math.isfinite(float(mix_ratio))
            or float(mix_ratio) <= 0
        ):
            raise ValueError(f"Dataset manifest entry {index} mix_ratio must be finite and > 0")
        adapter_config = Path(entry["adapter_config_path"])
        if not adapter_config.is_absolute() or not adapter_config.is_file():
            raise ValueError(
                f"Dataset manifest entry {index} must name an existing absolute adapter config"
            )
        specs.append(
            {
                "dataset_paths": [str(dataset_path)],
                "mix_ratio": float(mix_ratio),
                "embodiment_tag": embodiment_tag,
                "dataset_adapter": ft_config.dataset_adapter,
                "adapter_config_path": str(adapter_config.resolve()),
            }
        )
    return specs


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_specs = resolve_dataset_specs(ft_config, embodiment_tag)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": dataset_specs,
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
