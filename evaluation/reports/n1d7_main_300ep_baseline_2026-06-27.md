# N1.7-DiT Baseline Report — G1 Shaft Insertion

**日期：** 2026-06-27
**作者：** Supervisor
**目的：** 在 300-episode 新数据集上建立 N1.7-DiT（不带 Mamba）的 finetune + eval 基线，作为后续 Mamba 对比的参考

---

## 1. 任务摘要

| 项 | 值 |
|---|---|
| 架构 | GR00T-N1.7-3B（Qwen3-VL backbone + AlternateVLDiT）|
| System1 类型 | **DiT**（不带 Mamba）|
| 数据集 | `/home/sl/datasets/insertion_shaft_main_300ep_groot/`（300 episodes / 87944 frames / 30 fps）|
| Finetune steps | **5000**（effective batch=4，5000×4 = 20,000 样本 ≈ 0.7 epoch）|
| Wallclock finetune | **25 min 57 s**（3.74 it/s）|
| Eval episodes | **10**（Isaac-Insertion-Shaft-G129-Inspire-Joint-Play）|
| Wallclock eval | **10 min 13 s**（61.22 s/episode）|

## 2. 关键数字

### 2.1 成功率（核心指标）

```
results:  [False, False, False, False, False, False, False, False, False, False]
success rate:  0.0   (0/10)
```

**Baseline N1.7-DiT 在新数据集 5000 步 finetune 后的成功率 = 0.0 / 10**

### 2.2 Finetune 收敛

- 初始 loss（step 1）：~0.43
- 最终 loss（step 5000）：~0.04
- **Loss 下降 10x**（训练收敛正常）
- 训练耗时：25:57
- checkpoint-5000 大小：22 GB（safetensors + processor）

### 2.3 Eval 性能

- 10 episodes × 61 s/ep = 10:13 总时长
- 所有 episode 都达到了 episode 末尾（terminated 或 timeout），未提前崩溃
- 视频已保存至 `outputs/eval_videos/insertion_shaft/n1d7_main_300ep_eval_v3_20260627_155924/`

## 3. 与历史数字对比

| Run | 数据集 | Steps | Eval ep | Success |
|---|---|---|---|---|
| **N1.7-DiT v1** (2026-06-19) | `groot_datasets/insertion_with_torque_groot` (33 ep OLD) | 500 | 10 | **0.0 / 10** |
| **N1.7-Mamba** (历史) | 33 ep OLD | 500 | 10 | **0.0 / 10** |
| **N1.7-DiT v2 (本 baseline)** (2026-06-27) | `insertion_shaft_main_300ep_groot` (300 ep NEW) | 5000 | 10 | **0.0 / 10** |

**关键观察**：即便用 9x 更多数据（300 ep vs 33 ep）和 10x 更多训练步数（5000 vs 500），G1 插轴任务的成功率依然为 0%。这表明：
- 数据规模不是当前瓶颈
- 任务本身在此架构 + 仿真环境 + 当前 evaluation protocol 下极难
- 进一步对比 Mamba 时 **不应只看成功率**——需要看 sign-match / abs-ratio / 轨迹级指标

## 4. Schema gap（已知设计 gap）

训练数据 vs env 接口存在**两层 gap**：

| 层 | 训练数据 | Env 期望 | 处理方式 |
|---|---|---|---|
| Hand dim | 6（left_hand/right_hand）| 7（默认）/ 6（`UNITREE_SIM_HAND_DIM=6`）| **Eval 必须设 `UNITREE_SIM_HAND_DIM=6`**（已做）|
| Lower-body | 无 | waist/base_height_command/navigate（3 流）| Slim policy 自动零填充（`policy.py:_fill_untrained_action_keys`）|
| Legs state | 无 | left_leg/right_leg（2 流 × 6 dim）| Slim policy 零填充 |

> 在 `gr00t/model/gr00t_n1d7/policy.py` 中，**`ENV_ACTION_DIMS` 和 `ENV_STATE_DIMS` 中 hand 维度已从 7 改为 6**（本 baseline 一次性修复）。

## 5. 失败模式分析（10/10 失败）

所有 10 个 episode 均失败。可观察到的失败特征（基于 env 自带的 `terminations`）：
- 任务成功条件：shaft-housing 中心距离 < 0.05 m
- 已知失败分类：`shaft_out_of_bound`、`housing_out_of_bound`、`timeout`（max_episode_steps）

> **建议后续工作**：增加 per-step 距离 / 关节轨迹 trace 输出（参考 `evaluation/reports/evaluator-agent-protocol.md`），定量分析失败模式是"卡在初始位姿"、"动作幅度太小"、"撞墙出界"还是"超时未到"。

## 6. 文件 / 证据索引

### Finetune 输出
- `/home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_20260627_151214/`
  - `checkpoint-5000/`（22 GB，主评估对象）
  - `checkpoint-4000/`, `checkpoint-4500/`（save_total_limit=3 保留）
  - `experiment_cfg/`, `processor/`, `training_args.bin`
  - `model-00001-of-00003.safetensors`（~5 GB × 3）

### Eval 输出
- `/home/sl/Isaac-GR00T/outputs/logs/baseline_20260627/eval/n1d7_main_300ep_eval_v3_20260627_155924/`
  - `summary.json`（机器可读 metadata）
  - `client.log`（10:13 episode 完整日志）
  - `server.log`（N1.7 server stdout，含权重加载 trace）
- `/home/sl/Isaac-GR00T/outputs/eval_videos/insertion_shaft/n1d7_main_300ep_eval_v3_20260627_155924/`
  - 10 个 `.mp4`（UUID 命名，每个 episode 一段）

### 数据
- Stage 1: `/home/sl/.cache/huggingface/lerobot/yuqi/insertion_shaft_main_300ep/`
- Stage 2: `/home/sl/datasets/insertion_shaft_main_300ep_groot/`（finetune 输入）

### Wrapper 脚本（可复用）
- `/home/sl/Isaac-GR00T/scripts/baselines/run_finetune_n1d7_main_300ep.sh`
- `/home/sl/Isaac-GR00T/scripts/baselines/run_eval_n1d7_main_300ep.sh`

### 本次 baseline 期间修改的源文件
- `scripts/lerobot_conversion/convert_insertion_shaft_to_lerobot.sh`：加 PYTHONPATH 让脚本走 vendored lerobot 0.4.1
- `scripts/lerobot_conversion/convert_lerobot_to_groot.sh`：同上
- `gr00t/configs/training/training_config.py`：`gradient_checkpointing` 默认 True（支持低显存）
- `scripts/baselines/run_finetune_n1d7_main_300ep.sh`：`--use-wandb` 改用 `--no-use-wandb` flag
- **`gr00t/model/gr00t_n1d7/policy.py`**：`ENV_ACTION_DIMS` / `ENV_STATE_DIMS` 中 hand 维度 7→6

## 7. 复现命令

```bash
cd /home/sl/Isaac-GR00T

# 1) Finetune (5000 steps on GPU 0)
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH=/home/sl/datasets/insertion_shaft_main_300ep_groot \
OUTPUT_DIR=/home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_<DATE> \
MAX_STEPS=5000 GLOBAL_BATCH_SIZE=4 SAVE_STEPS=500 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup bash scripts/baselines/run_finetune_n1d7_main_300ep.sh \
  > outputs/logs/baseline_<DATE>/finetune.log 2>&1 &

# 2) Eval (10 episodes, hand_dim=6 to match dataset)
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/home/sl/Isaac-GR00T/outputs/n1d7_main_300ep_baseline_<DATE>/checkpoint-5000 \
N_EPISODES=10 MAX_EPISODE_STEPS=800 N_ACTION_STEPS=20 \
HOST=127.0.0.1 PORT=5756 \
UNITREE_SIM_HAND_DIM=6 \
nohup bash scripts/baselines/run_eval_n1d7_main_300ep.sh \
  > outputs/logs/baseline_<DATE>/eval.log 2>&1 &
```

## 8. 后续建议（按 ROI 排序）

### 🥇 #1：量化 Mamba vs DiT 行为差异
- 不应只看成功率（都是 0.0）
- 用 `scripts/compare_rollout_joint_trace.py` + 加 per-step trace
- 重点指标：sign-match（手/腕/腰）、abs-ratio（动作幅度）、time-to-first-action、joint saturation

### 🥈 #2：补 N1.7 lower-body 训练数据
- 当前数据 4 流（arm × 2 + hand × 2），env 需 7 流（含 waist/base_height/navigate）
- 零填充的 lower-body 信号噪声 = 模型无法学全 stack 协调
- 用 unitree_lerobot 重新采集带 lower-body 的 episode

### 🥉 #3：把 Mamba System1 移植到 N1.7
- 当前 `launch_finetune_n1d7.py` 明确写 "N1.7 dropped Mamba System 1"——Mamba 只在 N1.6 路径
- 这意味着"GR00T → Mamba"目标在 N1.7 架构下被暂时搁置
- 移植工作 = 把 `mamba_system1.py` 的 MambaSystem1ActionHead 适配到 N1.7 的 slim 接口

## 9. 总体结论

**Baseline 成功建立**：finetune + eval pipeline 在 N1.7-DiT 架构 + 300-ep 数据 + 5000 步条件下，10 episode 成功率为 0.0/10。

**这不是 "失败"**，而是**诚实的基线**——历史 v1 finetune（33 ep, 500 步）和 Mamba（500 步）也是 0.0。本 baseline 用 9x 数据 + 10x 步数仍 0.0，**明确说明**：成功率的瓶颈不在数据量，也不在训练步数，需要更细致的策略质量分析（动作分布、轨迹质量、env-schema 修复）才能突破。

**已解锁的能力**：
- 端到端 pipeline 全部脚本化、可复现
- Stage 1/2 数据转换脚本已修复 lerobot 路径 bug
- Slim policy 的 hand_dim 7→6 修复已落地
- Wrapper scripts 支持 5000 step finetune + 10 ep eval 在 ~36 min 内完成

**下一个 milestone**：在同样 pipeline 下跑 **N1.7-Mamba** finetune（需先把 Mamba System1 移植到 N1.7），然后对比 DiT baseline 的 sign-match / abs-ratio 指标。