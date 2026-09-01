# Finetune config used for single node post-training.
from dataclasses import dataclass
from typing import Literal

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    backbone_substitute: str = "Qwen/Qwen3-VL-2B-Instruct"
    """Public Qwen3-VL substitute for the gated ``nvidia/Cosmos-Reason2-2B``
    (same Qwen3-VL architecture). Only used by the N1.7 finetune launcher
    (``launch_finetune_n1d7.py``); ignored by N1.6."""

    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    system1_type: Literal["dit", "mamba"] = "dit"
    """System1 backbone type. Keep `dit` for full backward compatibility; use `mamba` to enable MambaSystem1."""

    mamba_enable_video_injection: bool = False
    """Enable Phase-2 VideoMamba + unidirectional Video->Robot injection when system1_type='mamba'."""

    require_real_mamba_backend: bool = False
    """If True, fail before training when `mamba_ssm` cannot be imported and the code would fall back."""

    system1_debug_shapes: bool = True
    """Print System1 input/output keys and tensor shapes once during train/inference for debugging."""

    system1_debug_gradients: bool = False
    """Register backward hooks on key System1 tensors and log non-finite gradients once."""

    debug_robot_mamba_mode: Literal["full", "skip", "norm_only", "core_only"] = "full"
    """Isolation mode for the robot_mamba stack."""

    backbone_trainable_params_fp32: bool = True
    """Whether to cast trainable backbone params to fp32. Disable when running FlashAttention dtype-sensitive smoke tests."""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    bf16: bool = True
    """Enable bfloat16 mixed precision training."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    finite_check_every_steps: int = 0
    """If > 0, inspect model parameters every N optimizer steps and log the first non-finite tensors."""

    stop_on_non_finite_params: bool = False
    """If True, stop training as soon as non-finite parameters are detected by the finite monitor."""

    detect_anomaly: bool = False
    """If True, enable torch autograd anomaly detection during training."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""
