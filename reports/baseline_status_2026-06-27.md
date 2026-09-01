# Baseline Pipeline 中期状态报告

**日期：** 2026-06-27 02:30（本地）
**作者：** Supervisor
**目录：** `/home/sl/Isaac-GR00T`

---

## 0. 一句话总结

**数据 pipeline 100% 完成**，**finetune 因 GPU OOM 卡住**，正在等 GPU 释放。

## 1. 已完成的工作

### ✅ Stage 1: raw → LeRobot v3（55 分钟）
- 输入：`groot_datasets/raw/insertion_shaft/2026_06_02_main_300ep_NEWEST/`（16 GB / 300 episodes）
- 输出：`/home/sl/.cache/huggingface/lerobot/yuqi/insertion_shaft_main_300ep/`
- 内容：300 episodes / 87944 frames / 30 fps / 3 cameras（cam_left_high/wrist/right_wrist）/ 26-dim state/action / 含 torque
- 视频格式：LeRobot v3 chunked（`videos/<cam>/chunk-000/file-NNN.mp4`）

### ✅ Stage 2: LeRobot v3 → GR00T-flavored（~3.5 小时）
- 输出：`/home/sl/datasets/insertion_shaft_main_300ep_groot/`（548 MB）
- 内容：
  - 3 路相机各 300 个 per-episode mp4（共 900 个）
  - meta/info.json（LeRobot v3.0 schema）
  - meta/modality.json（4 流 26-dim：left_arm/right_arm 7+7、left_hand/right_hand 6+6，与训练 modality_config 完全对齐）
  - meta/stats.json（含 mean/std/min/max/q01-99，无 NaN/Inf）
  - meta/episodes.jsonl（300 行）
  - meta/tasks.jsonl（单任务 placeholder `"insertion_shaft_main_300ep"`）
- 平均 episode 长度：293 帧（9.8s @ 30fps）

### ✅ Wrapper 脚本
- `scripts/baselines/run_finetune_n1d7_main_300ep.sh`（finetune 启动器，含 preflight + 防御性检查）
- `scripts/baselines/run_eval_n1d7_main_300ep.sh`（eval orchestration：起 server → poll port → 跑 client → 写 summary → cleanup）
- 两个脚本均 `bash -n` 语法验证通过

### ✅ Schema gap 文档化
- 训练数据 26-dim vs env 期望 33-dim（缺 waist/base_height/navigate）
- N1.7 slim policy 在 `policy.py:_extract_trained_state_keys` 自动零填充（**这是 v1 finetune 已知的设计 gap**）

## 2. 修改 / 修复

| 文件 | 修改 | 原因 |
|---|---|---|
| `scripts/lerobot_conversion/convert_insertion_shaft_to_lerobot.sh` | 加 PYTHONPATH 指向 vendored lerobot 0.4.1 + unitree_lerobot | env_isaaclab 的 lerobot 0.1.0 太旧，缺 `lerobot.datasets` 模块 |
| `scripts/lerobot_conversion/convert_lerobot_to_groot.sh` | 同上 | 同上 |
| `gr00t/configs/training/training_config.py` | `gradient_checkpointing: bool = False` → `True` | 节省 GPU 显存（OOM 缓解）|
| `scripts/baselines/run_finetune_n1d7_main_300ep.sh` | `--use-wandb ${USE_WANDB:-false}` → `--use-wandb` / `--no-use-wandb` | tyro 不接受 `--use-wandb false` 双参数语法 |

## 3. ⚠️ 当前阻塞：Finetune OOM

**3 次尝试均失败**（同根因）：

| # | 配置 | GPU 状态 | 失败位置 |
|---|---|---|---|
| 1 | bs=4, grad ckpt off, GPU 1 | GPU 1 被 yuqi 占用 93%（45.7 GB/49 GB）| model.to(device) — 分配 16 MiB 失败 |
| 2 | bs=1, grad ckpt off, GPU 0 | GPU 0 被 yuqi 占用 79%（37.6 GB/47.5 GB），仅 8.9 GB free | model.to(device) — 分配 576 MiB 失败 |
| 3 | bs=1, grad ckpt on, GPU 0 | GPU 0 free 10.9 GB（yuqi 进程稍减） | **forward pass** (`dit.py:190 norm1`) — 分配 20 MiB 失败（说明模型加载成功，但 forward+backward 仍 OOM）|

**根因**：N1.7 3B 模型即便冻结 LLM+visual（剩 ~350M trainable），仍需：
- 模型加载（BF16）：~6 GB
- Optimizer states（Adam FP32，仅 trainable）：~3 GB
- Activations（即使 gradient checkpointing，diffusion model + state encoder 仍需）：~2-4 GB
- 合计：~11-13 GB

而 GPU 0 当前 free 10.9 GB，**临界 OOM**。

**GPU 占用方**（user `yuqi`，5-8h 已跑，估计还需数小时）：
| PID | GPU 0 占用 | 命令片段 |
|---|---|---|
| 1283574 | 17.40 GB | unifolm-vla eval server (port 5557) |
| 1349570 | 17.40 GB | unifolm-vla eval server (port 5555) |
| 1126612 | 1.97 GB | openpi eval client |
| 1367513 | 1.97 GB | unifolm eval client |

## 4. 已批准但未启动的配置

- **Finetune**: 5000 steps / global_batch_size=1 / grad_accum=16（effective bs=16）/ LR=1e-4 / bf16 / save every 500 steps / max 3 checkpoints kept
- **Output dir**: `outputs/n1d7_main_300ep_baseline_<YYYYMMDD_HHMMSS>/`
- **Eval**: 10 episodes × Isaac-Insertion-Shaft-G129-Inspire-Joint-Play / port 5755/5756（避开 yuqi 占用的 5555/5556）/ UNITREE_SIM_HAND_DIM=6

## 5. 恢复命令（GPU 释放后可直接执行）

```bash
cd /home/sl/Isaac-GR00T

# 1) Pre-flight
nvidia-smi --query-gpu=index,memory.free --format=csv  # GPU 0 应有 ≥ 12 GB free
ls /home/sl/datasets/insertion_shaft_main_300ep_groot/meta/info.json  # 应存在
ss -lnt 'sport = :5755 or sport = :5756'  # 端口空

# 2) Launch finetune
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH=/home/sl/datasets/insertion_shaft_main_300ep_groot \
OUTPUT_DIR=/home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_$(date +%Y%m%d_%H%M%S) \
LOG_DIR=/home/sl/Isaac-GR00T/outputs/logs/baseline_$(date +%Y%m%d)/finetune \
MAX_STEPS=5000 GLOBAL_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=16 \
SAVE_STEPS=500 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup bash scripts/baselines/run_finetune_n1d7_main_300ep.sh \
  > /home/sl/Isaac-GR00T/outputs/logs/baseline_$(date +%Y%m%d)/finetune.log 2>&1 &

# 3) Monitor
tail -f /home/sl/Isaac-GR00T/outputs/logs/baseline_$(date +%Y%m%d)/finetune.log

# 4) Eval (after finetune done)
MODEL_PATH=/home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_<DATE>/checkpoint-5000 \
N_EPISODES=10 MAX_EPISODE_STEPS=800 N_ACTION_STEPS=20 \
HOST=127.0.0.1 PORT=5755 \
nohup bash scripts/baselines/run_eval_n1d7_main_300ep.sh \
  > /home/sl/Isaac-GR00T/outputs/logs/baseline_$(date +%Y%m%d)/eval.log 2>&1 &
```

## 6. 文件证据索引

### 数据
- `/home/sl/.cache/huggingface/lerobot/yuqi/insertion_shaft_main_300ep/` — Stage 1 output
- `/home/sl/datasets/insertion_shaft_main_300ep_groot/` — Stage 2 output（**finetune 输入**）

### Wrapper 脚本
- `/home/sl/Isaac-GR00T/scripts/baselines/run_finetune_n1d7_main_300ep.sh`
- `/home/sl/Isaac-GR00T/scripts/baselines/run_eval_n1d7_main_300ep.sh`

### 修改的源文件
- `/home/sl/Isaac-GR00T/scripts/lerobot_conversion/convert_insertion_shaft_to_lerobot.sh`
- `/home/sl/Isaac-GR00T/scripts/lerobot_conversion/convert_lerobot_to_groot.sh`
- `/home/sl/Isaac-GR00T/gr00t/configs/training/training_config.py`（gradient_checkpointing default True）

### 训练/评估脚本（原项目）
- `/home/sl/Isaac-GR00T/scripts/train/finetune_insertion_shaft_n1d7.sh`
- `/home/sl/Isaac-GR00T/scripts/eval/run_insertion_shaft_server_n1d7.sh`
- `/home/sl/Isaac-GR00T/scripts/eval/run_insertion_shaft_client.sh`

### 之前的项目现状报告（背景）
- `/home/sl/Isaac-GR00T/reports/project_status_2026-06-26.md`

### 监督报告（本报告同目录）
- `/home/sl/Isaac-GR00T/reports/baseline_status_2026-06-27.md`

---

## 7. 后续 Agent 接手建议

1. 优先检查 GPU 0 free memory 是否 ≥ 12 GB（`nvidia-smi --query-gpu=memory.free --format=csv -i 0`）
2. 若 ≥ 12 GB，直接执行 §5 的 finetune 命令
3. 若 < 12 GB，继续等待（每 10 min 检查一次 yuqi 进程状态）
4. Finetune 完成后，按 §5 步骤 4 启动 eval
5. Eval 完成后，写最终 baseline 报告（路径 `evaluation/reports/n1d7_main_300ep_baseline_<DATE>.md`）