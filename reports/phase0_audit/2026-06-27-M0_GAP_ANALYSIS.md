# Milestone 0 — Phase 0-B 差距分析（Supervisor 综合）

**日期**：2026-06-27
**作者**：Supervisor
**Worktree**：`/home/sl/Isaac-GR00T-m0-clean` @ `23ace64 GR00T N1.7 Release`（干净官方）
**输入**：5 个 sub-agent Phase 0-A 报告（github / dataset / environment / training / evaluator）

---

## 0. 一句话判定

**Milestone 0 路径在 m0-clean 上是可行的**，需要**数据层 6 个新文件 + eval 层 1 个新 wrapper + 2 个官方文件共 10 行改动**。**不动模型架构、不动训练超参、不动 processor**。详见 §3 最小改动清单。

---

## 1. m0-clean 官方支持什么（5 个 agent 一致确认）

### 1.1 训练侧 — **直接 out-of-the-box 支持**

| 组件 | m0-clean 状态 | 证据 |
|---|---|---|
| `EmbodimentTag.UNITREE_G1` enum | ✅ **已存在** | `gr00t/data/embodiment_tags.py:99` |
| `MODALITY_CONFIGS["unitree_g1_full_body_with_waist_height_nav_cmd"]` | ✅ **已预注册** | `gr00t/configs/data/embodiment_configs.py:67-144` |
| `register_modality_config` API | ✅ | `gr00t/configs/data/embodiment_configs.py:202-208` |
| LeRobot v2.1 loader | ✅ | `gr00t/data/dataset/lerobot_episode_loader.py:344-447`（`{episode_chunk:03d}` 占位符）|
| `examples/finetune.sh` | ✅ | `/home/sl/Isaac-GR00T-m0-clean/examples/finetune.sh`（158 行）|
| `examples/SO100/so100_config.py`（新 embodiment 范本）| ✅ | `examples/SO100/so100_config.py:1-71` |
| `gr00t/data/stats.py`（自动 stats 生成）| ✅ | `gr00t/data/stats.py:46-49, 115-131`（glob `data/*/*.parquet`）|
| `nvidia/GR00T-N1.7-3B` 本地缓存 | ✅ | `/home/sl/checkpoints/GR00T-N1.7-3B/config.json` 存在 |

### 1.2 Eval 侧 — **部分缺失**

| 组件 | m0-clean 状态 | 修复 |
|---|---|---|
| `gr00tlocomanip_g1*` 在 `ENV_PREFIX_TO_EMBODIMENT_TAG` | ✅ 注册了 | 但 prefix 选 "unitree_g1_isaaclab" 更精确 |
| `rollout_policy.py:128-143` `get_gym_env()` | ❌ **硬 raise** | 加 elif branch（`+1 行`）|
| `LiberoEnv` / `SimplerEnv` 范本 | ✅ | 仿 `libero_env.py:186-210` 写 `register_unitree_g1_envs` |
| `MultiStepWrapper` / `VideoRecordingWrapper` | ✅ 直接复用 | 只要 `obs` 键是 `video.*` / `state.*` / `annotation.*` 前缀 |
| `run_gr00t_server.py` ZMQ server | ✅ | `--embodiment-tag UNITREE_G1` 大小写不敏感解析 |

### 1.3 数据集兼容性 — **fit 官方 schema**

| 项 | 我们的值 | 官方可接受 | 评估 |
|---|---|---|---|
| state dim | 26 | 0..132 (`max_state_dim`) | ✅ |
| action dim | 26 | 0..132 | ✅ |
| camera 数 | 3 | 1/2/3/4 都支持 | ✅（需写 modality_config）|
| camera 分辨率 | 640×480×3 | 任意（processor 自动 crop 256×256）| ✅ |
| fps | 30 | 默认 30 | ✅ |
| task 数 | 1 | ≥1 OK | ✅ |
| dataset format | raw JSON (Unitree XR-teleop) | m0-clean 要 LeRobot v2.1 | ⚠️ **需转换** |

---

## 2. 关键风险（必须先确认的 4 件事）

### R1 — **Cosmos-Reason2-2B gated**
- 官方 launch_finetune 写死 `config.model.model_name = "nvidia/Cosmos-Reason2-2B"`
- m0-clean 的 `TrainingConfig.transformers_access_token` 是 gated 模型 token 入口
- **`FinetuneConfig` 不暴露这个字段**——只能通过 tyro 嵌套 `--training.transformers-access-token $HF_TOKEN` 或写 yaml config override
- ✅ **但**：`/home/sl/checkpoints/GR00T-N1.7-3B/` 已在本地缓存。`--base-model-path` 用本地路径 + `transformers_local_files_only=True` 即可绕过。
- **方案**：finetune launcher 加 `--base-model-path /home/sl/checkpoints/GR00T-N1.7-3B/ --training.transformers-local-files-only True`

### R2 — **`gr00tlocomanip_g1*` 三个 prefix 是声明但无实现**
- 官方 `env_utils.py:24-26` 注册了 `gr00tlocomanip_g1` / `gr00tlocomanip_g1_sim` / `gr00tlocomanip_g1_new` → `EmbodimentTag.UNITREE_G1`
- **但** `gr00t/eval/sim/` 下**只有** LIBERO / SimplerEnv，没有 G1 的 env 实现
- 这是官方 "声明了一半" 的状态——env_name 到 tag 映射在，但 gym env 实现没合并
- **方案**：用 prefix `unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play`（更精确），加新 `ENV_PREFIX_TO_EMBODIMENT_TAG` 行

### R3 — **`CameraBaseCfg.get_camera_config()` 默认 H×W 没明文**
- `gr00t/eval/sim/wrapper/multistep_wrapper.py` 不假设具体 H×W，只 crop 到 `image_target_size=[256,256]`
- Unitree 官方 sim 的 `head_rgb` 是默认配置（`g1_front_camera`），具体 (H, W) 需在跑时确认
- ✅ **不阻塞**——wrapper 只关心 dtype（HWC, uint8），不关心具体像素

### R4 — **v3 vs v2 LeRobot 占位符差异**
- m0-clean loader 走 `data/chunk-{episode_chunk:03d}/...`（**v2.1** 占位符）
- 我们旧的 `groot_datasets/lerobot/100_episode_no_torque/` 走 `data/chunk-{chunk_index:03d}/...`（**v3** 占位符）
- ✅ **不阻塞**——M0 全新写 v2.1，不用转换 v3

---

## 3. 最小改动清单（精确到文件 + 行数）

### 3.1 数据层（**6 个新文件**，0 个源文件修改）

| # | 文件路径 | 角色 | 行数估计 |
|---|---|---|---|
| D1 | `scripts/lerobot_conversion/m0/01_enumerate_episodes.py` | 扫 300 ep，写 `episodes.jsonl` | ~50 |
| D2 | `scripts/lerobot_conversion/m0/02_pick_camera_mapping.py` | 决定 `color_0/1/2` → `cam_left_high/wrist/right_wrist` 映射 | ~30 |
| D3 | `scripts/lerobot_conversion/m0/03_encode_videos.py` | raw jpg → mp4 (libx264) | ~80 |
| D4 | `scripts/lerobot_conversion/m0/04_write_parquets.py` | raw data.json → v2.1 parquet | ~100 |
| D5 | `scripts/lerobot_conversion/m0/05_write_meta.py` | 写 `meta/{info,modality,tasks,episodes}.json` | ~80 |
| D6 | `scripts/lerobot_conversion/m0/run_all.sh` | 串联 D1-D5 | ~30 |
| D7 | `examples/insertion_shaft_m0/modality_config_insertion_shaft_m0.py` | new embodiment 范本（仿 `so100_config.py`）| ~70 |

**总新增 ~440 行**。**0 个 m0-clean 源文件修改**。

### 3.2 Eval 层（**2 个源文件改 10 行 + 1 个新文件**）

| # | 文件路径 | 改动 | 行数 |
|---|---|---|---|
| E1 | `gr00t/eval/sim/env_utils.py` | 在 `ENV_PREFIX_TO_EMBODIMENT_TAG` 加 1 行：<br>`"unitree_g1_isaaclab": EmbodimentTag.UNITREE_G1,` | +1 |
| E2 | `gr00t/eval/rollout_policy.py` | (a) 加 `get_unitree_g1_env_fn()` 闭包 ~8 行<br>(b) `get_gym_env()` 加 `elif` branch +1 行 | +9 |
| E3 | `gr00t/eval/sim/UnitreeG1/__init__.py` | 新建（空 docstring）| +3 |
| E4 | `gr00t/eval/sim/UnitreeG1/unitree_g1_env.py` | 新建 `UnitreeG1Env(gym.Env)` + `register_unitree_g1_envs()` | ~250 |

**E4 不可避免的原因**（详 environment agent 报告）：
- IsaacLab `ManagerBasedRLEnv` 的 obs 是嵌套 dict `{"policy": {"robot_joint_state": (1,58), ...}}`
- GR00T `MultiStepWrapper` 期望 flat 键 `state.left_arm`、`video.ego_view` 等
- **没有官方机制在不写 env wrapper 的情况下桥接**（这是 IO contract 差异，不是架构差异）
- action 同样需要 7-stream 拆分（arm RELATIVE + hand ABSOLUTE）
- success 信号需要 wrapper 主动注入 `info["success"]`

### 3.3 Launchers（**0 个源文件修改 + 2 个新 shell 脚本**）

| # | 文件路径 | 角色 | 行数估计 |
|---|---|---|---|
| L1 | `examples/insertion_shaft_m0/finetune.sh` | 仿 `examples/finetune.sh` + 传 `--base-model-path /home/sl/checkpoints/GR00T-N1.7-3B/` | ~50 |
| L2 | `examples/insertion_shaft_m0/eval.sh` | 仿 LIBERO eval：先起 server + 后跑 client | ~80 |

---

## 4. 总改动统计

| 类别 | 新文件 | 源文件修改 | 总行数 |
|---|---|---|---|
| 数据层 | 7 | 0 | ~440 |
| Eval 层 | 2 | 2（+10 行）| ~260 |
| Launchers | 2 | 0 | ~130 |
| **合计** | **11 个新文件** | **2 个源文件改 10 行** | **~830 行** |

**所有 10 行源文件改动都是"加 entry 到 dict / 加 elif 分支"**，**不修改任何模型、policy、processor、loader、stats、training、optim、scheduler、augmentation 代码**。

---

## 5. 不做的事（重申）

- ❌ 不改 `gr00t/data/dataset/lerobot_episode_loader.py`（loader 已是 v2.1 contract 正路）
- ❌ 不改 `gr00t/data/stats.py`（glob 通用）
- ❌ 不改 `gr00t/model/gr00t_n1d7/`（任何文件）
- ❌ 不改 `gr00t/data/state_action/`（任何文件）
- ❌ 不复用 `external_dependencies/unitree_lerobot/`（社区 submodule，依赖已分叉代码）
- ❌ 不复用旧分支 `gr00t/eval/sim/UnitreeIsaacLab/`（基于已分叉代码）
- ❌ 不复用旧分支 `examples/GR00T-WholeBodyControl/`（同上）
- ❌ 不写 `--modality-config-path`（直接用 `EmbodimentTag.UNITREE_G1`，配置已预注册）
- ❌ 不调 `max_steps=5000`（用 `examples/finetune.sh` 默认 10000）
- ❌ 不调 `global_batch_size`（用 default 64）
- ❌ 不调 `learning_rate`（用 default 1e-4）
- ❌ 不调 `state_dropout_prob`（launch_finetune 强制 0.2）
- ❌ 不调 `warmup_ratio`（0.05）
- ❌ 不加 augmentation（不加 `color_jitter_params`、不调 `random_rotation_angle`）

---

## 6. 复现命令（M0 完成后预期可跑）

### 6.1 数据准备（一次）
```bash
cd /home/sl/Isaac-GR00T-m0-clean
bash scripts/lerobot_conversion/m0/run_all.sh \
  --raw-root /home/sl/Isaac-GR00T/groot_datasets/raw/insertion_shaft/2026_06_02_main_300ep_NEWEST \
  --out-root /home/sl/groot_datasets/lerobot/insertion_shaft_300ep_m0

# 生成 stats（走官方脚本）
uv run --project . python gr00t/data/stats.py \
  --dataset-path /home/sl/groot_datasets/lerobot/insertion_shaft_300ep_m0 \
  --embodiment-tag UNITREE_G1
```

### 6.2 Finetune
```bash
cd /home/sl/Isaac-GR00T-m0-clean
NUM_GPUS=1 \
MAX_STEPS=10000 \
SAVE_STEPS=1000 \
GLOBAL_BATCH_SIZE=32 \
DATALOADER_NUM_WORKERS=4 \
bash examples/insertion_shaft_m0/finetune.sh
```

### 6.3 Eval
```bash
# Server
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /home/sl/Isaac-GR00T-m0-clean/outputs/insertion_shaft_m0/checkpoint-10000 \
  --embodiment-tag UNITREE_G1 \
  --host 0.0.0.0 --port 5555 &

# Client
uv run python gr00t/eval/rollout_policy.py \
  --n-episodes 10 \
  --policy-client-host 127.0.0.1 --policy-client-port 5555 \
  --max-episode-steps 720 \
  --env-name unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play \
  --n-action-steps 16 \
  --n-envs 1
```

---

## 7. 给 Supervisor 的请示

### Q1 — 是否批准按 §3 清单实施 Phase 0-C？

- 11 个新文件 + 2 个源文件 10 行改动
- 全部是有 evidence 支持的最小必要修改
- 不动模型、不动超参、不动数据 schema 之外的部分

### Q2 — 三个 camera 命名需要用户确认

raw `data.json` 里 `color_0` / `color_1` / `color_2` 是相机索引，**没有 metadata 告诉我们哪个是哪个**。我们的约定是：
- `color_0` → `cam_left_high`
- `color_1` → `cam_left_wrist`
- `color_2` → `cam_right_wrist`

**这个映射**需要在 D2 里 hardcode，**需要你确认**。如果实际不是这个对应关系，需要调整。

### Q3 — modality config 用 `UNITREE_G1` tag 还是 `NEW_EMBODIMENT`？

- **UNITREE_G1**（推荐）：配置已预注册（7 streams、含 lower-body/waist/base_height/navigate），但**我们数据集只有 4 streams**——预注册 config 与数据集不匹配
- **NEW_EMBODIMENT**：写一个用户 modality_config.py 描述 4 streams + 3 cam，**更准确匹配数据**

**推荐用 NEW_EMBODIMENT**——因为 UNITREE_G1 预注册 config 期望的训练数据包含 lower-body 等 stream，与我们 4-stream 数据集 mismatch。

### Q4 — `--use-sim-policy-wrapper` 是否需要？

- 旧分支用 `--use-sim-policy-wrapper` 是因为有自定义 `Gr00tSimPolicyWrapper`
- m0-clean 默认走 `Gr00tSimPolicyWrapper`（在 `run_gr00t_server.py:146-149`，传 `--use-sim-policy-wrapper` 时启用）
- **如果我们的 env 输出 obs 已经是 flat 键格式**（`video.ego_view` / `state.left_arm` 等），**不需要** `--use-sim-policy-wrapper`（因为 server 默认不启用 → 直接走 `Gr00tPolicy.get_action()` 处理 flat obs）

**确认：M0 env wrapper 输出 flat 键，server 不传 `--use-sim-policy-wrapper`。**

---

## 8. Phase 0-D 输出承诺

Phase 0-C + 单次 finetune + eval 完成后，Phase 0-D 报告回答用户 5 个问题：

1. 官方 N1.7 baseline 在 m0-clean 上是否可复现？
2. 如果不可复现，差距在哪？
3. 哪些修改是绝对必要的？（答：§3 的 11 个新文件 + 2 个 10 行）
4. 哪些修改是故意避免的？（答：§5）
5. 最终 baseline 数字是多少？（success rate / loss 收敛 / inference latency）

**Phase 0-D 不做** Mamba / Diffusion / hyperparam 优化实验。**Phase 0-D 完就是 Phase 1+ 的起点**。

---

## 9. 文件路径索引

**本报告**：`/home/sl/Isaac-GR00T-m0-clean/reports/phase0_audit/2026-06-27-M0_GAP_ANALYSIS.md`

**Phase 0-A 报告**：
- `reports/github/2026-06-27-m0-official-inventory.md`（待 sub-agent 写）
- `reports/dataset/2026-06-27-m0-compat.md`（待 sub-agent 写）
- `reports/environment/2026-06-27-m0-env-compat.md`（待 sub-agent 写）
- `reports/training/2026-06-27-m0-official-recipe.md`（已写）
- `reports/evaluator/2026-06-27-m0-eval-protocol.md`（待 sub-agent 写）

**Worktree**：`/home/sl/Isaac-GR00T-m0-clean` @ commit `23ace64 GR00T N1.7 Release`

**Model cache**：`/home/sl/checkpoints/GR00T-N1.7-3B/`

**Raw data**：`/home/sl/Isaac-GR00T/groot_datasets/raw/insertion_shaft/2026_06_02_main_300ep_NEWEST/`（300 ep / 16 GB）
