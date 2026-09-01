"""Launch finetuning for GR00T-N1.7 on a single node.

Mirror of ``gr00t/experiment/launch_finetune.py`` but with N1.7-specific
configuration (Cosmos-Reason2-2B / Qwen3-VL-2B-Instruct backbone, Qwen3VL
processor, ``finetune-gr00t-n1d7`` wandb project).

Key N1.7 differences from N1.6:
  - ``eagle_collator = False`` (Qwen3VLProcessor instead of Eagle)
  - ``model_name = "Qwen/Qwen3-VL-2B-Instruct"`` (substitute for gated
    ``nvidia/Cosmos-Reason2-2B``; same Qwen3-VL architecture)
  - ``backbone_model_type = "qwen"`` (not "eagle")
  - ``use_alternate_vl_dit = True``
  - ``max_state_dim / max_action_dim = 132`` (not 29)
  - ``action_horizon = 40`` (not 16)
  - ``system1_type = "dit"`` only (N1.7 dropped Mamba System 1)
  - ``wandb_project = "finetune-gr00t-n1d7"`` (not n1d6)
"""

import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.experiment.experiment import run
from gr00t.model import _N1D7_AVAILABLE  # registers N1.7 vendored classes


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
        raise FileNotFoundError(f"Modality config path does not exist: {path}")


def build_n1d7_config(ft_config):
    """Build an N1.7-flavored Config from a FinetuneConfig."""
    if not _N1D7_AVAILABLE:
        raise RuntimeError(
            "N1.7 vendored modules are not available. "
            "Make sure gr00t/model/gr00t_n1d7/ is on the path "
            "and transformers >= 4.57 is installed."
        )

    embodiment_tag = ft_config.embodiment_tag.value

    config = get_default_config()
    # Force the model to be a Gr00tN1d7Config.
    config.model = Gr00tN1d7Config()

    # Use a SingleDatasetConfig instance (not a dict) so the validator
    # can read dataset_type / embodiment_tag attributes.
    from gr00t.configs.data.data_config import SingleDatasetConfig

    config.data.download_cache = False
    config.data.datasets = [
        SingleDatasetConfig(
            dataset_paths=[ft_config.dataset_path],
            mix_ratio=1.0,
            embodiment_tag=embodiment_tag,
        )
    ]
    config.load_config_path = None

    # --- N1.7-specific model settings ---
    config.model.model_name = ft_config.backbone_substitute
    config.model.backbone_model_type = "qwen"
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.system1_type = "dit"  # N1.7 dropped Mamba
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.load_bf16 = True
    config.model.reproject_vision = False
    config.model.eagle_collator = False  # N1.7 uses Qwen3VLProcessor
    config.model.gemma_collator = False
    config.model.collator_overwrite_image_inputs = False
    config.model.backbone_trainable_params_fp32 = ft_config.backbone_trainable_params_fp32
    config.model.use_relative_action = True
    # N1.7 defaults from config.json:
    #   action_horizon=40, max_state_dim=132, max_action_dim=132
    # These are already the defaults in Gr00tN1d7Config.

    # --- Training settings ---
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.bf16 = ft_config.bf16
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.finite_check_every_steps = ft_config.finite_check_every_steps
    config.training.stop_on_non_finite_params = ft_config.stop_on_non_finite_params
    config.training.detect_anomaly = ft_config.detect_anomaly
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d7"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    return config


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = build_n1d7_config(ft_config)
    run(config)