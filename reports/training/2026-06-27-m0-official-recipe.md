# Milestone 0 — 官方 GR00T N1.7 Training Recipe 参考卡

> **目的**：在干净 worktree `/home/sl/Isaac-GR00T-m0-clean`（commit `23ace64`）上，
> 把"N1.7 finetune 一个 new embodiment"的官方做法完整记录下来。
> **只读分析**。所有数值直接来自 m0-clean 文件，不来自记忆。
>
> **核心结论**：`launch_finetune_n1d7.py` 在 m0-clean 中**不存在**——
> N1.7 复用通用的 `launch_finetune.py`，并通过 `get_default_config()` 选用
> `Gr00tN1d7Config`（在 `base_config.py` 中注册为 default model）。
>
> **默认 hardware pattern**：
> - `examples/finetune.sh` 在 `NUM_GPUS=1` 时直接 `python`；`>=2` 时 `torchrun`。
> - `examples/SO100/README.md` 给出的 single-GPU finetune 是 `CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1`。
> - `examples/LIBERO/README.md` 给出的 multi-GPU finetune 是 `NUM_GPUS=8`。

---

## 1. CLI 参数完整表格

### 1.1 通用 finetune CLI（`gr00t/experiment/launch_finetune.py`，N1.7 用此文件）

Tyro 把 `FinetuneConfig` 暴露为 CLI，所有字段（除 required 外）都使用 default。

| CLI flag (tyro) | Type | Default | 说明 |
|---|---|---|---|
| `--base_model_path` | `str` | **required** | 预训练 base checkpoint 路径（HF hub 或本地）。对应 `FinetuneConfig.base_model_path` |
| `--dataset_path` | `str` | **required** | 数据集根目录。对应 `FinetuneConfig.dataset_path` |
| `--embodiment_tag` | `str` | **required** | Embodiment tag 名称或 value（大小写不敏感），由 `EmbodimentTag.resolve()` 解析。对应 `FinetuneConfig.embodiment_tag` |
| `--modality_config_path` | `str \| None` | `None` | 自定义 modality config `.py` 路径；缺省时使用 `gr00t/configs/data/embodiment_configs.py` 里预注册的 |
| `--tune_llm` | `bool` | `False` | 是否 finetune LLM backbone |
| `--tune_visual` | `bool` | `False` | 是否 finetune visual encoder |
| `--tune_projector` | `bool` | `True` | 是否 finetune multimodal projector |
| `--tune_diffusion_model` | `bool` | `True` | 是否 finetune 扩散 action head |
| `--state_dropout_prob` | `float` | `0.2` | state 输入的 dropout 概率 |
| `--random_rotation_angle` | `int \| None` | `None` | 图像随机旋转最大角度（度）；`None` = 用 base model 预训练时的设置 |
| `--color_jitter_params` | `dict[str, float] \| None` | `None` | color jitter 参数；`None` = 用 base model 预训练时的设置 |
| `--extra_augmentation_config` | `str \| None` | `None` | JSON 字符串：mask-based 等额外 image augmentation |
| `--global_batch_size` | `int` | `64` | 全局有效 batch size（所有 GPU + 累积） |
| `--dataloader_num_workers` | `int` | `2` | DataLoader worker 数 |
| `--learning_rate` | `float` | `1e-4` | 初始学习率 |
| `--gradient_accumulation_steps` | `int` | `1` | 累积步数 |
| `--output_dir` | `str` | `"./outputs"` | checkpoint/日志输出目录 |
| `--experiment_name` | `str \| None` | `None` | W&B run 名；缺省为 `output_dir` basename |
| `--wandb_project` | `str` | `"finetune-gr00t-n1d7"` | W&B 项目名 |
| `--save_steps` | `int` | `1000` | 保存 checkpoint 的 step 间隔 |
| `--save_total_limit` | `int` | `5` | 保留的最大 checkpoint 数 |
| `--num_gpus` | `int` | `1` | 可用 GPU 数 |
| `--use_wandb` | `bool` | `False` | 是否启用 W&B |
| `--max_steps` | `int` | `10000` | 总训练步数 |
| `--weight_decay` | `float` | `1e-5` | 优化器 L2 正则 |
| `--warmup_ratio` | `float` | `0.05` | 学习率 warmup 占总步数的比例 |
| `--shard_size` | `int` | `2**10` (=1024) | 数据预加载分片大小 |
| `--episode_sampling_rate` | `float` | `0.1` | episode 采样率 |
| `--num_shards_per_epoch` | `int` | `int(1e5)` (=100000) | 每 epoch 使用的 shard 数（VRAM 受限时减小） |
| `--save_only_model` | `bool` | `False` | 只保存模型权重，跳过 optimizer/scheduler/RNG 状态（无法 resume） |
| `--skip_weight_loading` | `bool` | `False` | 跳过从 `base_model_path` 加载权重（仅加载架构 + processor） |

### 1.2 Server CLI（`gr00t/eval/run_gr00t_server.py`，用于 N1.7 policy 推理）

| CLI flag | Type | Default | 说明 |
|---|---|---|---|
| `--model_path` | `str \| None` | `None` | 模型 checkpoint 目录 |
| `--embodiment_tag` | `str` | `"new_embodiment"` | 解析为 `EmbodimentTag` enum |
| `--device` | `str` | `"cuda"` | 推理设备 |
| `--dataset_path` | `str \| None` | `None` | ReplayPolicy 的 dataset 路径（与 `--model_path` 二选一） |
| `--modality_config_path` | `str \| None` | `None` | `.py`（执行 `register_modality_config`）或 `.json` |
| `--execution_horizon` | `int \| None` | `None` | ReplayPolicy 必填的正整数 |
| `--host` | `str` | `"0.0.0.0"` | server 监听地址 |
| `--port` | `int` | `5555` (`DEFAULT_MODEL_SERVER_PORT`) | server 监听端口 |
| `--strict` | `bool` | `True` | 是否严格校验输入/输出 |
| `--use_sim_policy_wrapper` | `bool` | `False` | 是否套用 sim policy wrapper |

### 1.3 Rollout/Client CLI（`gr00t/eval/rollout_policy.py`）

| CLI flag | Type | Default | 说明 |
|---|---|---|---|
| `--max_episode_steps` | `int` | `504` | 单 episode 最大步数 |
| `--n_episodes` | `int` | `50` | 总 episode 数 |
| `--model_path` | `str` | `""` | 离线模式 checkpoint 路径 |
| `--policy_client_host` | `str` | `""` | client 模式 server host |
| `--policy_client_port` | `int \| None` | `None` | client 模式 server port |
| `--env_name` | `str` | `"libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"` | gymnasium env 名 |
| `--n_envs` | `int` | `8` | 并行 env 数 |
| `--n_action_steps` | `int` | `8` | 每次执行的动作步数 |
| `--video_dir` | `str \| None` | `None` | 视频保存目录（`None` → `/tmp/sim_eval_videos_<env>_<uuid>`） |
| `--trt_engine_path` | `str` | `""` | TRT 推理引擎路径 |
| `--trt_mode` | `TrtMode` enum | `N17_FULL_PIPELINE` | `n17_full_pipeline` / `vit_llm_only` / `action_head` |

`--model_path` 与 `--policy_client_host/--policy_client_port` **互斥**（assert 在 `__main__` 中硬校验）。

### 1.4 Open-Loop Eval CLI（`gr00t/eval/open_loop_eval.py`，SO100 README 中用到）

| CLI flag (SO100 README 例子) | 取值 |
|---|---|
| `--dataset-path` | `examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich/` |
| `--embodiment-tag` | `NEW_EMBODIMENT` |
| `--model-path` | `/tmp/so100_finetune/checkpoint-10000` |
| `--traj-ids` | `0` |
| `--action-horizon` | `16` |
| `--steps` | `400` |

### 1.5 Environment Variables（launch_finetune.py 中显式处理的）

| Env var | Default | 说明 |
|---|---|---|
| `LOGURU_LEVEL` | `"INFO"` | `launch_finetune.py` 第 47 行：未设置时设为 `INFO` |
| `CUDA_VISIBLE_DEVICES` | `"0"` | `examples/finetune.sh` 在 `NUM_GPUS=1` 时设置 |
| `MASTER_PORT` | `"29500"` | `examples/finetune.sh` 中 `torchrun` 端口 |
| `WANDB_API_KEY` | — | LIBERO README 提示需设置（因为 `--use-wandb` 默认开） |

`examples/finetune.sh` 中可通过 env var 覆盖： `NUM_GPUS`、`MASTER_PORT`、`SAVE_STEPS` (default 1000)、`MAX_STEPS` (default 10000)、`USE_WANDB` (default 1)、`DATALOADER_NUM_WORKERS` (default 4)、`GLOBAL_BATCH_SIZE` (default 32)、`SHARD_SIZE` (default 1024)、`NUM_SHARDS_PER_EPOCH` (default 100000)、`EPISODE_SAMPLING_RATE` (default 0.1)。

---

## 2. dataclass 完整记录

### 2.1 `FinetuneConfig`（`gr00t/configs/finetune_config.py`）

**完整字段表**（已在上节 1.1 列出 — 共 30 个字段，全部为 required-only `base_model_path`/`dataset_path`/`embodiment_tag`，其余有 default）。

特别说明：
- `modality_config_path`（`.py`）由 `load_modality_config()` 在 `launch_finetune.py` 第 31–41 行用 `importlib.import_module(path.stem)` 加载；该模块必须**作为 side-effect 调用 `register_modality_config()`**（参 `examples/SO100/so100_config.py` 第 70 行）。
- `launch_finetune.py` 在第 76–93 行**硬覆写**部分 model / training 字段：
  - `config.model.tune_llm/visual/projector/diffusion_model` ← `ft_config.*`
  - `config.model.state_dropout_prob` ← `ft_config.state_dropout_prob`（**注意**：FT CLI default 是 0.2；模型 config `Gr00tN1d7Config` default 是 0.8，launch_finetune 用 0.2 覆盖）
  - `config.model.random_rotation_angle` ← `ft_config.random_rotation_angle`（`None` 保留）
  - `config.model.color_jitter_params` ← `ft_config.color_jitter_params`
  - `config.model.extra_augmentation_config` ← `json.loads(ft_config.extra_augmentation_config)` 或 `None`
  - 硬编码覆写：
    - `config.model.load_bf16 = False`
    - `config.model.reproject_vision = False`
    - `config.model.model_name = "nvidia/Cosmos-Reason2-2B"`
    - `config.model.backbone_trainable_params_fp32 = True`
    - `config.model.use_relative_action = True`
  - `config.training.optim = "adamw_torch"`（**注意**：覆盖了 `TrainingConfig` 中 `optim: str = "adamw_torch_fused"`；`examples/finetune.sh` 不传 `optim`，所以实际跑的 optim 是 `adamw_torch`）
  - `config.training.start_from_checkpoint = ft_config.base_model_path`
  - 其余 training 字段（`global_batch_size`, `learning_rate`, `max_steps` 等）从 FT CLI 取值

### 2.2 `TrainingConfig`（`gr00t/configs/training/training_config.py`）

完整字段（dataclass，默认值直接来自 m0-clean）：

| Field | Type | Default |
|---|---|---|
| `output_dir` | `str` | `"./outputs"` |
| `experiment_name` | `Optional[str]` | `None` |
| `max_steps` | `int` | `30000` |
| `global_batch_size` | `int` | `1024` |
| `batch_size` | `Optional[int]` | `None` |
| `gradient_accumulation_steps` | `int` | `1` |
| `learning_rate` | `float` | `1e-4` |
| `lr_scheduler_type` | `str` | `"cosine"` |
| `weight_decay` | `float` | `1e-5` |
| `warmup_ratio` | `float` | `0.05` |
| `warmup_steps` | `int` | `0` |
| `max_grad_norm` | `float` | `1.0` |
| `optim` | `str` | `"adamw_torch_fused"` |
| `start_from_checkpoint` | `Optional[str]` | `None` |
| `skip_weight_loading` | `bool` | `False` |
| `tf32` | `bool` | `True` |
| `fp16` | `bool` | `False` |
| `bf16` | `bool` | `True` |
| `eval_bf16` | `bool` | `True` |
| `logging_steps` | `int` | `10` |
| `save_steps` | `int` | `1000` |
| `save_total_limit` | `int` | `5` |
| `save_vl_model` | `bool` | `False` |
| `save_only_model` | `bool` | `False` |
| `upload_checkpoints` | `bool` | `False` |
| `upload_every` | `int` | `1000` |
| `upload_last_n_checkpoints` | `int` | `5` |
| `max_concurrent_uploads` | `int` | `2` |
| `eval_strategy` | `str` | `"no"` |
| `eval_steps` | `int` | `500` |
| `eval_set_split_ratio` | `float` | `0.1` |
| `eval_batch_size` | `int` | `2` |
| `save_best_eval_metric_name` | `str` | `""` |
| `save_best_eval_metric_greater_is_better` | `bool` | `True` |
| `deepspeed_stage` | `int` | `2` |
| `gradient_checkpointing` | `bool` | `False` |
| `transformers_trust_remote_code` | `bool` | `True` |
| `transformers_local_files_only` | `bool` | `False` |
| `transformers_cache_dir` | `Optional[str]` | `None` |
| `transformers_access_token` | `Optional[str]` | `None` |
| `use_ddp` | `bool` | `False` |
| `ddp_bucket_cap_mb` | `int` | `100` |
| `num_gpus` | `int` | `1` |
| `dataloader_num_workers` | `int` | `2` |
| `remove_unused_columns` | `bool` | `False` |
| `use_wandb` | `bool` | `False` |
| `wandb_project` | `str` | `"finetune-gr00t-n1d7"` |
| `enable_profiling` | `bool` | `False` |
| `max_retries` | `int` | `3` |
| `assert_loss_less_than` | `Optional[float]` | `None` |
| `add_rl_callback` | `bool` | `False` |
| `enable_open_loop_eval` | `bool` | `False` |
| `open_loop_eval_traj_ids` | `list[int]` | `[0]` |
| `open_loop_eval_steps_per_traj` | `int` | `100` |
| `open_loop_eval_plot_indices` | `Optional[list[int]]` | `None` |

### 2.3 `Gr00tN1d7Config`（`gr00t/configs/model/gr00t_n1d7.py`）

| Field | Type | Default |
|---|---|---|
| `model_type` | `str` | `"Gr00tN1d7"` |
| `model_dtype` | `str` | `"bfloat16"` |
| `model_name` | `str` | `"nvidia/Cosmos-Reason2-2B"` |
| `backbone_model_type` | `str` | `"qwen"` |
| `model_revision` | `Optional[str]` | `None` |
| `tune_top_llm_layers` | `int` | `0` |
| `backbone_embedding_dim` | `int` | `2048` |
| `tune_llm` | `bool` | `False` |
| `tune_visual` | `bool` | `False` |
| `select_layer` | `int` | `12` |
| `reproject_vision` | `bool` | `False` |
| `use_flash_attention` | `bool` | `True` |
| `load_bf16` | `bool` | `False` |
| `backbone_trainable_params_fp32` | `bool` | `True` |
| `image_crop_size` | `tuple[int,int]` | `(230, 230)` |
| `image_target_size` | `tuple[int,int]` | `(256, 256)` |
| `shortest_image_edge` | `Optional[int]` | `None` |
| `crop_fraction` | `Optional[float]` | `None` |
| `random_rotation_angle` | `Optional[int]` | `None` |
| `color_jitter_params` | `Optional[dict]` | `None` |
| `use_albumentations_transforms` | `bool` | `True` |
| `extra_augmentation_config` | `Optional[dict]` | `None` |
| `formalize_language` | `bool` | `True` |
| `apply_sincos_state_encoding` | `bool` | `False` |
| `use_percentiles` | `bool` | `True` |
| `use_relative_action` | `bool` | `False` |
| `max_state_dim` | `int` | `132` |
| `max_action_dim` | `int` | `132` |
| `action_horizon` | `int` | `40` |
| `hidden_size` | `int` | `1024` |
| `input_embedding_dim` | `int` | `1536` |
| `state_history_length` | `int` | `1` |
| `add_pos_embed` | `bool` | `True` |
| `attn_dropout` | `float` | `0.2` |
| `use_vlln` | `bool` | `True` |
| `max_seq_len` | `int` | `1024` |
| `use_alternate_vl_dit` | `bool` | `True` |
| `attend_text_every_n_blocks` | `int` | `2` |
| `diffusion_model_cfg` | `dict` | 见下 |
| `num_inference_timesteps` | `int` | `4` |
| `noise_beta_alpha` | `float` | `1.5` |
| `noise_beta_beta` | `float` | `1.0` |
| `noise_s` | `float` | `0.999` |
| `num_timestep_buckets` | `int` | `1000` |
| `tune_projector` | `bool` | `True` |
| `tune_diffusion_model` | `bool` | `True` |
| `tune_vlln` | `bool` | `True` |
| `state_dropout_prob` | `float` | `0.8`（**注意**：FinetuneConfig 覆盖为 0.2） |
| `exclude_state` | `bool` | `False` |
| `use_mean_std` | `bool` | `False` |
| `max_num_embodiments` | `int` | `32` |

`diffusion_model_cfg`（`default_factory` dict）：

```python
{
    "positional_embeddings": None,
    "num_layers": 16,
    "num_attention_heads": 32,
    "attention_head_dim": 48,
    "norm_type": "ada_norm",
    "dropout": 0.2,
    "final_dropout": True,
    "output_dim": 1024,
    "interleave_self_attention": True,
}
```

### 2.4 `Config`（`gr00t/configs/base_config.py`）— 顶层容器

```python
@dataclass
class Config:
    load_config_path: Optional[str] = None
    model: ModelUnionType = field(default_factory=lambda: Gr00tN1d7Config())
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

`get_default_config()` 返回 `Config()`，即 N1.7 是默认 model。

### 2.5 `DataConfig`（`gr00t/configs/data/data_config.py`）

完整字段：

| Field | Type | Default |
|---|---|---|
| `datasets` | `list[SingleDatasetConfig]` | `[]` |
| `modality_configs` | `dict[str, dict[str, ModalityConfig]]` | `MODALITY_CONFIGS`（来自 `embodiment_configs.py`） |
| `download_cache` | `bool` | `False` |
| `shard_size` | `int` | `2**10` |
| `episode_sampling_rate` | `float` | `0.1` |
| `num_shards_per_epoch` | `int` | `int(1e5)` |
| `override_pretraining_statistics` | `bool` | `True` |
| `mode` | `str` | `"single_turn"` |
| `random_chop` | `float` | `0.0` |
| `mock_dataset_mode` | `bool` | `False` |
| `shuffle` | `bool` | `True` |
| `seed` | `int` | `42` |
| `multiprocessing_context` | `str` | `"fork"` |
| `allow_padding` | `bool` | `False` |
| `subsample_ratio` | `float` | `1.0` |
| `image_crop_size` | `list[int]` | `[244, 244]` |
| `image_target_size` | `list[int]` | `[224, 224]` |
| `video_backend` | `str` | `"torchcodec"` |

`SingleDatasetConfig`：

| Field | Type | Default |
|---|---|---|
| `dataset_paths` | `list[Any]` | (required) |
| `embodiment_tag` | `Optional[str]` | `None` |
| `mix_ratio` | `float` | `1.0` |
| `dataset_type` | `str` | `"physical_embodiment"` |
| `val_dataset_path` | `Optional[str]` | `None` |

### 2.6 `ModalityConfig`（`gr00t/data/types.py`）

| Field | Type | Default |
|---|---|---|
| `delta_indices` | `list[int]` | (required) |
| `modality_keys` | `list[str]` | (required) |
| `sin_cos_embedding_keys` | `Optional[list[str]]` | `None` |
| `mean_std_embedding_keys` | `Optional[list[str]]` | `None` |
| `action_configs` | `Optional[list[ActionConfig]]` | `None` |

`__post_init__` 校验：`action_configs` 若非空，长度必须等于 `modality_keys`。

`ActionConfig`：

| Field | Type | Default |
|---|---|---|
| `rep` | `ActionRepresentation` enum | (required) — `RELATIVE`/`DELTA`/`ABSOLUTE` |
| `type` | `ActionType` enum | (required) — `EEF`/`NON_EEF` |
| `format` | `ActionFormat` enum | (required) — `DEFAULT`/`XYZ_ROT6D`/`XYZ_ROTVEC` |
| `state_key` | `Optional[str]` | `None` |

---

## 3. default 超参一览（直接来自 m0-clean）

注：以下"实际"列是 m0-clean 中**实际跑 finetune**时用到的值（`launch_finetune.py` 硬覆写 + `examples/finetune.sh` 显式传）。"model default" 列是 `Gr00tN1d7Config`/`TrainingConfig` 的 default。"FinetuneConfig default" 列是 `FinetuneConfig` 自己的 default。

| 超参 | model/training default | FinetuneConfig default | 实际跑 (N1.7) |
|---|---|---|---|
| `learning_rate` | 1e-4 | 1e-4 | **1e-4** |
| `global_batch_size` | 1024 | 64 | **64**（CLI 显式传） |
| `max_steps` | 30000 | 10000 | **10000**（SO100 单卡例子）；LIBERO 例子里用 `MAX_STEPS=20000` |
| `warmup_ratio` | 0.05 | 0.05 | **0.05** |
| `weight_decay` | 1e-5 | 1e-5 | **1e-5** |
| `max_grad_norm` | 1.0 | — | **1.0** |
| `optim` | `adamw_torch_fused` | — | **`adamw_torch`**（launch_finetune.py 第 96 行硬覆写） |
| `lr_scheduler_type` | `cosine` | — | **`cosine`**（不传则用 default） |
| `save_steps` | 1000 | 1000 | **1000** |
| `save_total_limit` | 5 | 5 | **5** |
| `gradient_checkpointing` | False | — | **False** |
| `state_dropout_prob` | 0.8（model） | 0.2 | **0.2**（launch_finetune.py 把 0.8 覆写为 0.2） |
| `bf16` | True | — | **True** |
| `fp16` | False | — | **False** |
| `tf32` | True | — | **True** |
| `tune_llm` | False | False | **False** |
| `tune_visual` | False | False | **False** |
| `tune_projector` | True | True | **True** |
| `tune_diffusion_model` | True | True | **True** |
| `tune_vlln` | True | — | **True**（model default；launch_finetune 不覆写） |
| `gradient_accumulation_steps` | 1 | 1 | **1** |
| `dataloader_num_workers` | 2 | 2 | LIBERO 例子里用 `4`（env var 覆盖）；SO100 默认 2 |
| `shard_size` | 2**10 | 2**10 | **1024** |
| `num_shards_per_epoch` | 100000 | 100000 | **100000** |
| `episode_sampling_rate` | 0.1 | 0.1 | **0.1** |
| `save_only_model` | False | False | **False** |
| `skip_weight_loading` | False | False | **False** |
| `use_wandb` | False | False | **False**（CLI）；LIBERO 例子里用 `1`（env var） |
| `wandb_project` | `finetune-gr00t-n1d7` | `finetune-gr00t-n1d7` | **finetune-gr00t-n1d7** |
| `deepspeed_stage` | 2 | — | **2**（仅在多卡时使用） |
| `use_alternate_vl_dit` | True | — | **True** |
| `action_horizon` | 40 | — | **40** |
| `attn_dropout` | 0.2 | — | **0.2** |
| `diffusion_model_cfg.dropout` | 0.2 | — | **0.2** |
| `num_inference_timesteps` | 4 | — | **4** |
| `state_dropout_prob`（来自 FT CLI） | — | 0.2 | **0.2**（LIBERO README 里显式 `--state-dropout-prob 0.2`） |

**官方 color_jitter 例子**（`examples/finetune.sh` 第 123 行）：

```bash
--color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

**官方 random_rotation_angle**：官方 examples/SO100、LIBERO 都没传 → `None`（保留 model default）。

---

## 4. 官方 launch bash 模板

### 4.1 单卡 finetune（来自 `examples/SO100/README.md`）

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path examples/SO100/finish_sandwich_lerobot/izuluaga/finish_sandwich \
  --modality-config-path examples/SO100/so100_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/so100_finetune
```

可覆盖 env vars：`USE_WANDB=0`、 `SAVE_STEPS=...`、`MAX_STEPS=...`、`GLOBAL_BATCH_SIZE=...`、`DATALOADER_NUM_WORKERS=4` 等。

### 4.2 多卡 finetune（来自 `examples/LIBERO/README.md`）

```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_10 \
    --state-dropout-prob 0.2
```

### 4.3 底层命令（`examples/finetune.sh` 内部展开的 tyro CLI）

`gr00t/experiment/launch_finetune.py`（tyro 把 `--` 转成 `_`，把 kebab-case 转成 snake_case）：

```bash
# 单卡（NUM_GPUS=1）
CUDA_VISIBLE_DEVICES=0 \
python gr00t/experiment/launch_finetune.py \
  --base_model_path nvidia/GR00T-N1.7-3B \
  --dataset_path /path/to/dataset \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 1 \
  --output_dir /tmp/run \
  --save_steps 1000 \
  --save_total_limit 5 \
  --max_steps 10000 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --learning_rate 1e-4 \
  --global_batch_size 32 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader_num_workers 4 \
  --shard_size 1024 \
  --num_shards_per_epoch 100000 \
  --episode_sampling_rate 0.1 \
  --modality_config_path /path/to/embodiment_config.py
# 末尾 `--` 之后是额外 tyro 参数
```

```bash
# 多卡（NUM_GPUS>=2）
torchrun --nproc_per_node=8 --master_port=29500 \
  gr00t/experiment/launch_finetune.py ... （同上）
```

`launch_finetune.py` 入口逻辑（精简）：
1. tyro 解析 `FinetuneConfig` → CLI。
2. `EmbodimentTag.resolve(ft_config.embodiment_tag)` → 标准 enum。
3. `load_modality_config(ft_config.modality_config_path)`（可选）：`importlib.import_module(path.stem)`，要求该模块调用 `register_modality_config()`。
4. `get_default_config().load_dict({"data": {"download_cache": False, "datasets": [{"dataset_paths": [...], "mix_ratio": 1.0, "embodiment_tag": "new_embodiment"}]}})`。
5. 把 FT CLI 字段覆写到 `config.model.*` 和 `config.training.*`（第 76–116 行）。
6. `run(config)` → 进入 `gr00t/experiment/experiment.py`。

**重要**：第 4 步中的 `"new_embodiment"` 字面量是 `ft_config.embodiment_tag.value`（例如 `NEW_EMBODIMENT` → `"new_embodiment"`、`LIBERO_PANDA` → `"libero_sim"`）。这是 `MODALITY_CONFIGS` registry 的 key。

### 4.4 完整 Stage 0/1/2 流程（参照 SO100 README）

```bash
# Stage 1: 数据准备（如果 LeRobot v3 → v2）
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id <hf_repo> \
  --root <local_lerobot_root>

# Stage 1b: 拷贝 modality.json 到 meta/
cp examples/SO100/modality.json <dataset_root>/meta/modality.json

# Stage 2: 启动 finetune
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <dataset_root> \
  --modality-config-path <my_embodiment_config.py> \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/ft_run
```

---

## 5. 官方 eval 启动模板

### 5.1 Open-Loop Eval（`gr00t/eval/open_loop_eval.py`）— SO100 README

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path <dataset_root>/ \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/ft_run/checkpoint-10000 \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 400
```

### 5.2 Closed-Loop Server + Client（SO100/LIBERO README）

**Server 端**（`gr00t/eval/run_gr00t_server.py`）：

```bash
# SO100 (real robot)
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /tmp/so100_finetune/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT

# LIBERO (sim) — 加 --use-sim-policy-wrapper
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /tmp/libero_10/checkpoint-20000 \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper
```

Server 默认监听 `0.0.0.0:5555`（`DEFAULT_MODEL_SERVER_PORT` in `run_gr00t_server.py`）。

**Client 端**（`gr00t/eval/rollout_policy.py`）：

```bash
# LIBERO
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
  --n-episodes 10 \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --max-episode-steps 720 \
  --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --n-action-steps 8 \
  --n-envs 5
```

```bash
# SO100 (real robot)
uv run python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower --robot.port=/dev/ttyACM2 \
  --robot.id=orange_follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
  --policy-host=localhost --policy-port=5555 --lang-instruction="..."
```

### 5.3 Rollout 独立模式（不连 server，inline 加载模型）

```bash
uv run python gr00t/eval/rollout_policy.py \
  --n-episodes 50 \
  --model-path /tmp/ft_run/checkpoint-10000 \
  --env-name libero_sim/<TASK> \
  --n-envs 8 \
  --n-action-steps 8 \
  --max-episode-steps 504
```

（注：`--model_path` 与 `--policy_client_host/--policy_client_port` 互斥）

### 5.4 SIM env setup（LIBERO）

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

---

## 6. 官方环境要求

### 6.1 Python / CUDA / Torch（来自 `pyproject.toml` + `docker/Dockerfile`）

| 项 | 版本 |
|---|---|
| Python | `==3.10.*` |
| CUDA | 12.8.0-devel-ubuntu22.04（docker base）；pytorch 索引 `cu128` |
| PyTorch | `==2.7.1`（从 `pytorch-cu128` 索引） |
| Torchvision | `==0.22.1` |
| transformers | `==4.57.3` |
| diffusers | `==0.35.1` |
| peft | `==0.17.1` |
| datasets | `==3.6.0` |
| tyro | `==0.9.17` |
| gymnasium | `==1.2.2` |
| numpy | `==1.26.4` |
| pandas | `==2.2.3` |
| scipy | `==1.15.3` |
| opencv-python-headless | `>=4.5,<4.13` |
| albumentations | `==1.4.18` |
| lmdb | `==1.7.5` |
| msgpack / msgpack-numpy | `==1.1.0` / `==0.4.8` |
| flash-attn | `==2.7.4.post1` |
| deepspeed | `==0.17.6`（仅 x86_64 / aarch64 + 非 linux） |
| wandb | `==0.23.0` |
| torchcodec | `==0.4.0`（x86_64）/`==0.10.0a0`（aarch64） |
| tensorrt | `cu12>=10.15.1.29`（x86_64）/`cu13>=10.15.1.29`（aarch64） |
| triton | `==3.3.1`（仅 aarch64 linux） |
| omegaconf | `==2.3.0` |
| onnx | `>=1.20.0` |
| einops | `==0.8.1` |
| jsonlines | `==4.0.0` |
| pyzmq | `==27.0.1` |
| click | `==8.1.8` |
| dm-tree | (无版本) |
| matplotlib | `==3.10.1` |
| gitpython | `==3.1.46` |
| cryptography | `==42.0.8` |
| termcolor | `==3.2.0` |
| huggingface-hub | `[cli]`（无版本） |

### 6.2 系统依赖（来自 `docker/Dockerfile` 第 30–42 行）

```
build-essential, git, git-lfs, curl, wget,
python3.10, python3.10-venv, python3.10-dev, python3-pip, python-is-python3,
software-properties-common, ffmpeg
```

EGL/Vulkan（MuJoCo/PyOpenGL headless rendering）：`libegl1` + 两个 ICD JSON（`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`、`/usr/share/vulkan/icd.d/nvidia_icd.json`）；env vars `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`__EGL_VENDOR_LIBRARY_FILENAMES=...`。

### 6.3 GPU 显存

- 没有在 `pyproject.toml` 或 README 中显式声明。
- LIBERO README 显示：`8 GPUs × GLOBAL_BATCH_SIZE=640` 跑 20K steps（= per-GPU batch size 80）。
- SO100 README 用单卡 finetune（`NUM_GPUS=1`），意味着单卡足以跑 GR00T-N1.7-3B 推理 + 小数据集 finetune（具体显存数无声明）。

### 6.4 数据集目录约定（LeRobot 格式）

参照 `examples/SO100/modality.json` 和 `examples/LIBERO/modality.json`，finetune 数据集根目录下必须含 `meta/modality.json`，结构：

```json
{
    "state": { "<key>": { "start": <int>, "end": <int> }, ... },
    "action": { "<key>": { "start": <int>, "end": <int> }, ... },
    "video": { "<view>": { "original_key": "<lerobot_feature>" }, ... },
    "annotation": { "<lang_key>": { "original_key": "<lerobot_feature>" }, ... }
}
```

LeRobot 自身结构（`info.json` / `stats.json` / `data/*/*.parquet` / `videos/...`）也必须在 root 下。`stats.json` 包含 `mean` / `std` / `min` / `max` / `q01` / `q99`（见 `gr00t/data/stats.py` 第 84–93 行）。

### 6.5 EmbodimentTag 命名（`gr00t/data/embodiment_tags.py`）

- Pretrain 标签（base model 自带）：`OXE_DROID_*`、`XDOF`、`XDOF_SUBTASK`、`REAL_G1`、`REAL_R1_PRO_SHARPA*`。
- Posttrain 标签（需要 finetuned ckpt）：`UNITREE_G1`、`SIMPLER_ENV_GOOGLE`、`SIMPLER_ENV_WIDOWX`、`LIBERO_PANDA`（对应 value `libero_sim`）。
- Finetune-only：`NEW_EMBODIMENT`（value `"new_embodiment"`）。
- `EmbodimentTag.resolve(s)` 大小写不敏感，先比 enum name，再比 value。

### 6.6 Embodiment id mapping（`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` 第 56–72 行）

```python
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}
```

`new_embodiment` 默认对应 projector index `10`。`max_num_embodiments=32` 允许扩展。

### 6.7 dataset_statistics 结构

`gr00t/model/gr00t_n1d7/setup.py` 第 222–226 行：训练过程中 `train_dataset.get_dataset_statistics()` 被调用并保存到 `<output_dir>/dataset_statistics.json`。

`gr00t/data/dataset/lerobot_episode_loader.py` 第 499–534 行的结构：

```python
{
    "state": {
        "<joint_key>": {
            "mean": [...], "std": [...], "min": [...], "max": [...],
            "q01": [...], "q99": [...]
        },
        ...
    },
    "action": {
        "<joint_key>": { "mean": [...], ... },
        ...
    },
    # 可选
    "relative_action": { ... }
}
```

该文件是推理时**自动从数据计算并保存的**（`gr00t/model/gr00t_n1d7/setup.py` 第 222–226 行），不需要用户手动提供。

---

## 7. 给 Supervisor 的判定 — 在 m0-clean 上 finetune 我们的 300 ep 数据集需要什么**新增**文件

> **核心结论**（基于官方 recipe，不改 default 超参、不改架构、不改训练算法）：
> 新增**最少两个文件 + 一个数据集目录约定**就足够。无需改 m0-clean 中任何现有文件。
> 我们的 finetune 仍然跑通用 `examples/finetune.sh` + 通用 `gr00t/experiment/launch_finetune.py`，
> 只提供自定义的 modality config `.py` 和数据集中的 `meta/modality.json`。

### 7.1 必须新增

#### (a) `examples/<our_embodiment>/<our_embodiment>_config.py`

参考 `examples/SO100/so100_config.py` 的最小结构：

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

OUR_CONFIG = {
    "video":   ModalityConfig(delta_indices=[0],   modality_keys=[<video_keys>]),
    "state":   ModalityConfig(delta_indices=[0],   modality_keys=[<state_keys>]),
    "action":  ModalityConfig(
        delta_indices=list(range(<H>)),
        modality_keys=[<action_keys>],
        action_configs=[ActionConfig(rep=..., type=..., format=...) for _ in <action_keys>],
    ),
    "language": ModalityConfig(delta_indices=[0],   modality_keys=[<lang_key>]),
}

register_modality_config(OUR_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

**关键约束**（来自 `gr00t/configs/data/embodiment_configs.py` 第 202–208 行）：
- `register_modality_config()` 只接受**单个** embodiment_tag；`assert embodiment_tag.value not in MODALITY_CONFIGS`。
- 第二次 register 相同 tag 会 AssertionError → 一个 `*_config.py` 对应一个 embodiment_tag。
- 我们的数据集若只有一个 embodiment 形态（如 dual-arm Unitree），一个 `.py` 文件即可。

#### (b) `<dataset_root>/meta/modality.json`

参考 `examples/SO100/modality.json`（`examples/SO100/README.md` 第 23–25 行：`cp examples/SO100/modality.json <dataset_root>/meta/modality.json`）。

结构约束（来自 `gr00t/data/dataset/lerobot_episode_loader.py` 第 514–529 行）：
- `state[*].start/end`、`action[*].start/end`：**整数切片**，对应 `info.json` 中 parquet feature 的向量切片。
- `video[*].original_key`：必须存在（缺省时 fallback 到 `observation.images.<key>`）；`annotation[*].original_key` 同样。

LeRobot 数据集本身要符合：`meta/info.json`（含 `features`）、`meta/stats.json`（含 mean/std/min/max/q01/q99 — 缺则可由 `gr00t/data/stats.py` 自动生成）、`data/*/*.parquet`、视频文件（`videos/...`）。

### 7.2 可选 / 视情况需要

- `<dataset_root>/meta/relative_stats.json`（如果 action_config 中 `rep=RELATIVE`）— 由 `python gr00t/data/stats.py --dataset-path ... --embodiment-tag NEW_EMBODIMENT --modality-config-path <cfg.py>` 自动生成。
- `--extra_augmentation_config` JSON 字符串（**默认 None**，官方 SO100/LIBERO 都没传）。
- `--random_rotation_angle`、`--color_jitter_params`（**默认 None**，官方 SO100 传了 color_jitter；`examples/finetune.sh` 默认会传 color_jitter）。

### 7.3 **不需要**新增或修改

- 不需要改 `gr00t/configs/data/embodiment_configs.py`：用 `--modality_config_path` 动态加载。
- 不需要改 `gr00t/configs/finetune_config.py`、`training_config.py`、`gr00t_n1d7.py`、或 `base_config.py`。
- 不需要改 `gr00t/experiment/launch_finetune.py`：N1.7 通用入口。
- 不需要改 `examples/finetune.sh`：通用 bash 模板。
- 不需要改超参：用 `FinetuneConfig` 的 default（除根据显存/数据集规模调小 `GLOBAL_BATCH_SIZE`、`NUM_SHARDS_PER_EPOCH` 等运行时 env var 外）。
- 不需要准备 `dataset_statistics.json`：训练时由 `gr00t/model/gr00t_n1d7/setup.py` 第 222–226 行自动计算并保存。

### 7.4 Milestone 0 跑通检查清单

1. 数据集已转 LeRobot v2 + 写入 `meta/modality.json`、`meta/stats.json`。
2. 写出 `examples/<our_embodiment>/<our_embodiment>_config.py`，调用 `register_modality_config(..., EmbodimentTag.NEW_EMBODIMENT)`。
3. 运行 `examples/finetune.sh`：

   ```bash
   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 USE_WANDB=0 \
     uv run bash examples/finetune.sh \
       --base-model-path nvidia/GR00T-N1.7-3B \
       --dataset-path <our_lerobot_root> \
       --modality-config-path examples/<our_embodiment>/<our_embodiment>_config.py \
       --embodiment-tag NEW_EMBODIMENT \
       --output-dir /tmp/m0_ft
   ```

4. (可选) Open-loop eval：

   ```bash
   uv run python gr00t/eval/open_loop_eval.py \
     --dataset-path <our_lerobot_root>/ \
     --embodiment-tag NEW_EMBODIMENT \
     --model-path /tmp/m0_ft/checkpoint-10000 \
     --traj-ids 0 \
     --action-horizon <H> \
     --steps 400
   ```

---

## 附录 A — 文件路径清单（m0-clean 全部为绝对路径）

- 启动器：`/home/sl/Isaac-GR00T-m0-clean/gr00t/experiment/launch_finetune.py`
- Training pipeline：`/home/sl/Isaac-GR00T-m0-clean/gr00t/experiment/experiment.py`
- Trainer：`/home/sl/Isaac-GR00T-m0-clean/gr00t/experiment/trainer.py`
- FinetuneConfig：`/home/sl/Isaac-GR00T-m0-clean/gr00t/configs/finetune_config.py`
- TrainingConfig：`/home/sl/Isaac-GR00T-m0-clean/gr00t/configs/training/training_config.py`
- Model config：`/home/sl/Isaac-GR00T-m0-clean/gr00t/configs/model/gr00t_n1d7.py`
- Base config：`/home/sl/Isaac-GR00T-m0-clean/gr00t/configs/base_config.py`
- Data config：`/home/sl/Isaac-GR00T-m0-clean/gr00t/configs/data/data_config.py`
- 预注册 modality configs：`/home/sl/Isaac-GR00T-m0-clean/gr00t/configs/data/embodiment_configs.py`
- Types / ActionConfig / ModalityConfig：`/home/sl/Isaac-GR00T-m0-clean/gr00t/data/types.py`
- EmbodimentTag enum：`/home/sl/Isaac-GR00T-m0-clean/gr00t/data/embodiment_tags.py`
- Stats 计算：`/home/sl/Isaac-GR00T-m0-clean/gr00t/data/stats.py`
- LeRobotEpisodeLoader：`/home/sl/Isaac-GR00T-m0-clean/gr00t/data/dataset/lerobot_episode_loader.py`
- N1.7 processor / embodiment_id_mapping：`/home/sl/Isaac-GR00T-m0-clean/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`
- N1.7 setup（自动计算 stats）：`/home/sl/Isaac-GR00T-m0-clean/gr00t/model/gr00t_n1d7/setup.py`
- 通用 finetune bash 模板：`/home/sl/Isaac-GR00T-m0-clean/examples/finetune.sh`
- SO100 范例（python config + modality.json + README）：`/home/sl/Isaac-GR00T-m0-clean/examples/SO100/`
- LIBERO 范例（modality.json + README）：`/home/sl/Isaac-GR00T-m0-clean/examples/LIBERO/`
- LeRobot v3→v2 转换脚本：`/home/sl/Isaac-GR00T-m0-clean/scripts/lerobot_conversion/convert_v3_to_v2.py`
- Eval server：`/home/sl/Isaac-GR00T-m0-clean/gr00t/eval/run_gr00t_server.py`
- Eval client/rollout：`/home/sl/Isaac-GR00T-m0-clean/gr00t/eval/rollout_policy.py`
- Open-loop eval：`/home/sl/Isaac-GR00T-m0-clean/gr00t/eval/open_loop_eval.py`
- LIBERO sim setup：`/home/sl/Isaac-GR00T-m0-clean/gr00t/eval/sim/LIBERO/setup_libero.sh`
- pyproject：`/home/sl/Isaac-GR00T-m0-clean/pyproject.toml`
- docker base：`/home/sl/Isaac-GR00T-m0-clean/docker/Dockerfile`

---

## 附录 B — 关键引用

- SO100 single-GPU finetune 例子：`/home/sl/Isaac-GR00T-m0-clean/examples/SO100/README.md` 第 32–38 行。
- LIBERO 8-GPU finetune 例子：`/home/sl/Isaac-GR00T-m0-clean/examples/LIBERO/README.md` 第 39–45 行。
- LIBERO 官方 4-suite 准确率（97.65 / 97.5 / 98.45 / 94.35%）使用 `--state-dropout-prob 0.2`（FinetuneConfig default），20K steps，batch_size 640，grad_accum 1。
- `launch_finetune.py` 把 `state_dropout_prob` 硬覆写为 0.2（第 80 行），与 FinetuneConfig default 一致。
- 官方 LIBERO 4-suite 全部用相同 `state_dropout_prob=0.2`、相同 `max_steps=20K`、相同 `global_batch_size=640`。

###TASK_COMPLETED###
