# 项目现状总览（GR00T → Mamba 双臂操作）

**Supervisor 汇总报告**　**日期：** 2026-06-26
**作者：** Supervisor（基于 Repo Scout / Environment Agent / Goal Agent 三份独立分析）
**工作目录：** `/home/sl/Isaac-GR00T`
**所有声明均附文件证据。**

---

## 1. 项目在做什么

这是一个 **NVIDIA Isaac-GR00T N1.6 的本地 fork**（upstream `main` HEAD = `ae17468`，
N1.6 release = `4e62473`，提交于 2026-01-30）。在它的基础上，本地用户 **Wang Jiaxin**
（导师 Yan Shijun，署名见 `weekly_update_mamba_system1_20260326.md:12`）并行推进两条自研路线：

| 路线 | 关键文件 | 状态 |
|---|---|---|
| **Mamba System1**（在 N1.6 内替换 DiT action head） | `gr00t/model/gr00t_n1d6/mamba_system1.py`（43 KB / 1057 行）| 工程链路跑通，**仿真成功率 0.0**（vs DiT 0.8）|
| **GR00T-N1.7**（全新架构，Qwen3-VL + AlternateVLDiT） | `gr00t/model/gr00t_n1d7/`（整目录未跟踪）| 已 vendored，**仿真成功率 0.0**（零样本与 500 步 finetune）|

目标任务是 **Unitree G1 29-DoF + Inspire 灵巧手** 在仿真环境下的双臂插轴（shaft insertion）。

---

## 2. 当前研究目标

### 显式目标（来自文档）

- `weekly_update_mamba_system1_20260326.md:18`：在 Isaac-GR00T 中实现 **可替换的 System1**，
  引入 Mamba backbone，**保持现有训练与评测链路兼容**。
- `gr00t/model/gr00t_n1d6/SYSTEM1_IO_CONTRACT.md:7`：DiT 与 Mamba 两套 System1 必须共享
  同一 forward / inference I/O 契约。

### 隐式目标（从代码与未跟踪文件推断）

- 在 G1 双臂插轴任务上，让 **Mamba System1** 与 **N1.7** 两条路线都能达到"可复现、
  稳定、可论文化"的状态（依据：`outputs/n1d7_insertion_shaft_finetune_v1/` 与
  `outputs/eval_runs/n1d7_finetuned_insertion_20260619.md` 的存在）。

### 与 Agent Team 半年目标的衔接

- 长期目标（已存于长期记忆）：`half-year-research-goal.md` —— 6 个月内将
  GR00T → Mamba 双臂策略推到稳定、可复现基线。
- 当前距离"基线"还差 **1 块核心数字**（DiT-on-this-task 的成功率）+ **2 块策略质量工作**
  （Mamba 分布匹配、N1.7 schema 补齐）。

---

## 3. 已采集数据

| 路径 | 内容 | 关键属性 |
|---|---|---|
| `groot_datasets/raw/insertion_shaft/` | 原始 Unitree XR-teleop JSON | 25 GB / 663 unique raw eps（per memory）|
| `groot_datasets/lerobot/100_episode_no_torque/` | LeRobot v3，无扭矩版 | 100 episode 简化版 |
| `groot_datasets/lerobot/insertion_shaft_with_torque/` | LeRobot v3，含扭矩版 | 33 episodes / 11567 frames / 30 fps / 26-dim state+action（4 流）|
| `demo_data/cube_to_bowl_5/`, `demo_data/gr1.PickNPlace/` | 上游自带示例 | 与本任务无关 |

**关键发现 —— Schema 缺口**（Goal Agent 已识别，Environment Agent 在 env 端确认）：

- 数据集 state/action 只有 **4 流**：`left_arm / right_arm / left_hand / right_hand`，合计 **26 dim**。
- IsaacLab env 端要求 **7 流**：`left_leg / right_leg / waist / left_arm / right_arm / left_hand / right_hand`，合计 **43 dim**。
- N1.7 slim policy 通过 `Gr00tN1d7G1ShaftPolicy._extract_trained_state_keys` 零填缺失流
  （`gr00t/model/gr00t_n1d7/policy.py:303-330`），但这是 **训练阶段从未见过** 的信号，
  模型只能靠零值"猜"。

---

## 4. 任务环境

### 仿真 vs 真机

- **仿真器**：Isaac Lab via `unitree_sim_isaaclab`（离线包，路径 `/home/sl/humanoid-lfd/unitree_sim_isaaclab/`）。
- **本地包装**：`gr00t/eval/sim/UnitreeIsaacLab/insertion_env.py`（17 KB，自身 fork 的薄包装）。
- **真机**：**未启用**。`scripts/teleop/` 只有离线 XR-teleop 采集 → JSON/h5 转换，无真机 rollout 脚本。

### 关键参数

| 项 | 值 | 证据 |
|---|---|---|
| Sim dt | 0.005 s（200 Hz）| `insertion_shaft_g1_29dof_inspire_env_cfg.py:241` |
| Decimation | 4 | 同上 |
| **Env step Hz** | **50 Hz** | 200 / 4 |
| Policy Hz | **2.5 Hz**（每 0.4 s 一次推理，每次推 20 env 步）| `multistep_wrapper.py:256-276` |
| Episode | 800 steps × 0.02 s = **16 s** | `run_insertion_shaft_client.sh` 默认 |
| 相机 | 1 路 head RGB（480×640，env 端 20 fps / dataset 30 fps）| `info.json:89-111`, `VideoConfig.fps=20` |
| Action dim | 33（N1.7）/ 29（N1.6 default）| `insertion_env.py:165-185`, `gr00t_n1d6.py:56` |
| Arms | 7+7 dim，**RELATIVE delta** | `modality_config_*.py` |
| Hands / waist / base / nav | 6+6+3+1+3 dim，**ABSOLUTE** | 同上 + `insertion_env.py:359-373` |
| Action horizon | N1.6=16，N1.7=40（截断到 20）| `gr00t_n1d6.py:57`, `policy.py:173` |
| Success | shaft-housing distance < **0.05 m** | `base_termination_insertion_shaft.py:38-51` |

### Server / Client 接口

- **协议**：ZMQ + `pyzmq==27.0.1`（`pyproject.toml`）。
- **Server 启动**：`scripts/eval/run_insertion_shaft_server.sh`（N1.6 默认端口 5555）；
  N1.7 用 `run_insertion_shaft_server_n1d7.sh`（端口 5556）。
- **Client 启动**：`scripts/eval/run_insertion_shaft_client.sh`，env 变量见
  `scripts/eval/README_unitree_isaaclab_insertion.md`。
- **Policy 接口**：`Gr00tPolicy`（N1.6）+ `Gr00tN1d7G1ShaftPolicy`（N1.7 slim）。
- **强约束**：`N_ENVS=1`（`rollout_policy.py:133-136` raise ValueError）。

---

## 5. 当前进度

### 时间线（依据 `outputs/` 命名 + `wandb/` + `git status`）

- **2026-02 ~ 03-16**：早期 Mamba 训练与 finetune，多个 `g1_mamba_fulltrain_*` 产物。
  wandb 最近一次是 `run-20260316_130540-9sujfyjp`。
- **2026-03-25 ~ 03-26**：一整夜数值稳定性调试（fp32、grad、backend、action_head reinit）；
  Mamba 完整训练到 `checkpoint-10000`；产出 `weekly_update_mamba_system1_20260326.md`
  周报。**Mamba 仿真成功率定格 0.0**。
- **2026-04-16**：短暂回到 DiT，跑 `g1_dit_lerobot_notau_*` smoke + fulltrain。
- **2026-06-19**：**N1.7 路线首次跑通**（`outputs/n1d7_insertion_shaft_finetune_v1/`，
  5 GB / ckpt-200/400/500），eval=0.0。
- **2026-06-26（今天）**：最后一次 commit 没有 push，所有改动都是本地修改 + 未跟踪。

### 关键数字

| 路径 | 数字 |
|---|---|
| Mamba 插轴成功率（2026-03-26）| **0.0 / 10** |
| DiT 插轴成功率（同任务同环境）| **缺失** —— 这是关键盲点 |
| N1.7 零样本成功率（2026-06-19）| **0.0 / 10** |
| N1.7 finetune 500 步成功率（2026-06-19）| **0.0 / 10** |
| right_shoulder_yaw sign match | 1/10（reinit 变体 2/10）|
| right_wrist_yaw sign match | 0/10（reinit 变体 4/10）|
| navigate_command 幅值 | DiT 的 6% |
| target_upper_body_pose 幅值 | DiT 的 16.6% |

### 当前焦点（推断）

主线已从 **Mamba 策略质量** 转向 **N1.7 迁移 + 数据补齐**。证据：
- N1.7 训练/评估产物 mtime 最新（6/19），Mamba 路径 3/26 后无新数字。
- 但 `scripts/probe_mamba_alignment_effect.py`、
  `scripts/probe_right_hand_prediction.py`、
  `scripts/probe_right_hand_sampling.py` 持续新增 → Mamba 处于 **复盘而非突破** 状态。

---

## 6. 已完成的关键工作

| 任务 | 证据 |
|---|---|
| DiT System1 baseline | 上游 N1.6，`examples/GR00T-WholeBodyControl/finetune_g1.sh` |
| Mamba System1 替代 | `gr00t/model/gr00t_n1d6/mamba_system1.py`（1057 行）|
| System1 I/O 契约 | `SYSTEM1_IO_CONTRACT.md`（DiT ↔ Mamba 共享 forward/get_action 协议）|
| Mamba 完整训练到 checkpoint-10000 | 周报§3；`outputs/g1_mamba_fulltrain_20260316_143146/` |
| Mamba smoke + fp32 + grad probe 一系列 | `g1_mamba_conditioned_temporal_fp32_probe_*`、`g1_mamba_internal_grad_probe_*` |
| 数值稳定性修复（action_head reinit）| `g1_mamba_fulltrain_reinit_20260325_234131/` |
| Server/client eval（DiT & Mamba）| `scripts/eval/run_{gr00t_wholebody,insertion_shaft}_{server,client}.sh` |
| Rollout joint/action 轨迹对比 | `scripts/compare_rollout_joint_trace.py` + 5 个 `joint_rollout_trace_compare_summary_*.md` |
| Unitree IsaacLab 插轴 env wrapper | `gr00t/eval/sim/UnitreeIsaacLab/insertion_env.py` |
| Unitree XR-teleop → LeRobot 转换 | `external_dependencies/unitree_lerobot/`（独立仓库）+ `scripts/lerobot_conversion/convert_*_to_lerobot.sh` |
| N1.7 架构 vendored | `gr00t/model/gr00t_n1d7/`（gr00t_n1d7.py / setup.py / policy.py / run_server.py / processing*.py）|
| N1.7 finetune 链路 | `gr00t/experiment/launch_finetune_n1d7.py` + `scripts/train/finetune_insertion_shaft_n1d7.sh` + `scripts/eval/run_insertion_shaft_server_n1d7.sh` |

**TODO/FIXME 数量**：剔除 vendor 噪声后仅 2 处
（`multistep_wrapper.py:264`、`video_utils.py:385`）—— 工程层面基本干净。

---

## 7. 当前风险与缺口

### 高优先级风险

1. **盲点：DiT-on-this-task 的成功率从未记录**。Mamba=0.0、N1.7=0.0，但 DiT 是真的赢在 0.8
   还是和 Mamba 一样烂，**未知**。没有这个数字，所有"vs DiT"的说法都不可证伪。
2. **Mamba backend fallback**：`mamba_ssm` / `selective_scan_cuda` 与本地 Torch ABI 不兼容，
   当前用 `_StableTemporalMixerCore` 替代（`mamba_system1.py:37-86`）。这意味着 **Mamba 论文
   里的 SSM 优势未在实验中获得**。`require_real_mamba_backend` 开关存在，但打开会直接
   `RuntimeError`。
3. **N1.7 schema 缺口**（硬约束）：训练 4 流 vs env 7 流。N1.7 policy 用零填，但模型从未见过
   `waist/legs/nav` 信号 → 学不会任何与 lower-body 相关的策略。这是 N1.7=0.0 的最可能根因。
4. **N1.7 没接 Mamba System1**：`launch_finetune_n1d7.py` docstring 明确写
   "system1_type = 'dit' only (N1.7 dropped Mamba System 1)"。两条路线目前 **互相独立**，无
   internal ablation。
5. **wandb 已 3 个月无新 run**（最新 `20260316_130540-9sujfyjp`）。N1.7 训练可能未接 wandb 或
   仅写本地（`wandb_config.json` 仅 80 字节）。

### 中优先级风险

6. **Action horizon 错配**：N1.6 输出 16 步，env 给 20 步 → `rollout_policy.py:340-364` 把
   16 步放到 target 前 16 位、剩 4 位重复末值 → 可能放大末步抖动。
7. **Embodiment tag drift**：`unitree_g1` key 在 N1.6 与 N1.7 下 modality_config 不同（5 流 vs 7 流
   vs 数据集 4 流）。若 N1.7 路径忘了走 `Gr00tN1d7G1ShaftPolicy` 直接走 `Gr00tPolicy`，会 KeyError
   或静默乱码。
8. **N_ENVS=1 强约束**（`rollout_policy.py:133-136`）→ 无法做 batched eval，吞吐约 1 episode / 分钟。
9. **eval 工具脆弱点**：`insertion_env.py:397-398` `_compute_success` 静默吞 exception，真错误
   可能被掩盖；`_joint_target` 在 step 期间 self-reset 时不重置（line 124/408）。

### 工程缺口

10. **缺文档骨架**：无 `CLAUDE.md` / `NOTES.md` / `TODO.md` / `DESIGN.md` / `ROADMAP.md` /
    `PROJECT.md` / `dataset_manifest.*`。所有项目意图分散在 `weekly_update_*.md` +
    `SYSTEM1_IO_CONTRACT.md` + 几份 README 里。
11. **缺独立目录**：`evaluation/`、`experiments/`、`reports/`、`research/` 均未建立，产物全部
    沉在 `outputs/` 下 100+ 个命名相近的目录里。
12. **缺 DiT-on-this-task 的 eval 数字**（同 1）。

---

## 8. 下一步最值得做的 3 件事

按 ROI 排序（**仅分析，不实施** —— 实施需 Supervisor 审批后交给 Coder Agent）：

### 🥇 #1：跑 DiT baseline，建立"3 选 1"基线

- **做什么**：用现成 DiT checkpoint（`examples/GR00T-WholeBodyControl/finetune_g1.sh`）跑 10
  episode 的 `Isaac-Insertion-Shaft-G129-Inspire-Joint-Play` 评估。
- **为什么**：**当前最大盲点**。Mamba=0.0、N1.7=0.0 但 DiT 数字未在同任务上确认。
  没有 DiT 基线，写不出"vs DiT"的任何有意义的对比，也写不出论文主表。
- **成本**：1~2 小时 eval 计算（10 episode × 16 s ≈ 3 分钟，加 server 启动和 trace 生成）。
- **不做的后果**：所有后续 Mamba/N1.7 数字都缺少参照系。

### 🥈 #2：生成系统化的 Mamba vs DiT sign-match / abs-ratio 对照表

- **做什么**：把 `scripts/compare_rollout_joint_trace.py` 升级为统一脚本，对每个 trained
  checkpoint 输出 per-joint sign_match（0~1）、abs_ratio、target_pose_magnitude 三张表。
- **为什么**：现有 `joint_rollout_trace_compare_summary_reinit_20260326.md` 已有 5 个变体，
  但格式不统一、变量命名不一致。**基础设施已具备**，只差"产品化"。这能直接定位 Mamba
  是"幅度问题"还是"语义问题"，决定下一阶段是改 head、改 state 表征、还是改数据集。
- **成本**：半天到一天（脚本重构 + 跑批 + 写汇总报告）。
- **依赖**：#1 的 DiT baseline（否则没有对照）。

### 🥉 #3：补 N1.7 lower-body 数据并重训

- **做什么**：用 `external_dependencies/unitree_lerobot/`（已具备 XR-teleop → LeRobot 全链路，
  `CONVERT_NOTES.md` 6/26 mtime）采集 waist/legs/nav 信号，再跑 `scripts/train/finetune_insertion_shaft_n1d7.sh`。
- **为什么**：**N1.7 = 0.0 的最可能根因**是 schema 缺口（4 流训练 vs 7 流 env）。这是数据层
  失败，不是模型失败；不补数据，再调架构也是徒劳。
- **成本**：1~2 天（采集 + 转换 + 训练 + eval）。
- **风险**：采集可能需要重新遥操一段时间；真机遥操是否能覆盖 G1 lower-body 取决于
  `unitree_lerobot` 当前采集脚本是否含 lower-body 字段（需 Supervisor 派 Dataset Agent 复核）。

### 备选（未入选 TOP-3，但值得记录）

- **#4（备选）**：评估自研 temporal mixer vs 真 `mamba_ssm` SSM 的差距（前提：能解决 ABI）。
- **#5（备选）**：把 Mamba System1 移植到 N1.7，形成 N1.7 内部 DiT vs Mamba ablation。

---

## 9. 证据索引

### 核心文档

- `README.md` — upstream NVIDIA Isaac-GR00T N1.6 总览
- `weekly_update_mamba_system1_20260326.md` — Wang Jiaxin 3/26 周报（Mamba 0.0 vs DiT 0.8 的来源）
- `gr00t/model/gr00t_n1d6/SYSTEM1_IO_CONTRACT.md` — DiT ↔ Mamba 共享 I/O 契约
- `scripts/eval/README_unitree_isaaclab_insertion.md` — Unitree IsaacLab 评估流程
- `scripts/lerobot_conversion/README.md` — 数据转换说明
- `examples/GR00T-WholeBodyControl/README.md` — G1 全身示例

### 关键模型 / 环境 / 策略

- `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` — N1.6 主体
- `gr00t/model/gr00t_n1d6/mamba_system1.py` — Mamba System1 替代实现
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — N1.7 主体
- `gr00t/model/gr00t_n1d7/policy.py` — N1.7 G1 shaft slim policy（含 schema 零填逻辑）
- `gr00t/model/gr00t_n1d7/setup.py` — N1.7 配置装配
- `gr00t/policy/gr00t_policy.py` — 上游 Gr00tPolicy（hand scale 三档常量 line 22-52）
- `gr00t/eval/sim/UnitreeIsaacLab/insertion_env.py` — 仿真 env 包装
- `gr00t/eval/rollout_policy.py` — server/client rollout loop（N_ENVS=1 强约束 line 133-136）

### 关键配置 / 入口

- `gr00t/configs/model/gr00t_n1d6.py`, `gr00t_n1d7.py`
- `gr00t/experiment/launch_finetune.py`, `launch_finetune_n1d7.py`
- `examples/GR00T-WholeBodyControl/finetune_g1.sh`, `finetune_g1_mamba.sh`
- `examples/GR00T-WholeBodyControl/modality_config_{100_with_tau,lerobot_v3_notau,g1_shaft_n1d7,insertion_shaft_n1d7}.py`
- `scripts/eval/run_insertion_shaft_{server,client,server_n1d7}.sh`
- `scripts/train/finetune_insertion_shaft_n1d7.sh`
- `scripts/lerobot_conversion/convert_*_to_lerobot.sh`

### 数据

- `groot_datasets/README.md`, `INDEX.md`
- `groot_datasets/raw/insertion_shaft/` — 25 GB / 663 raw eps
- `groot_datasets/lerobot/insertion_shaft_with_torque/meta/info.json` — LeRobot v3 schema
- `groot_datasets/lerobot/insertion_shaft_with_torque/meta/relative_stats.json` — 状态/动作归一化分位数
- `external_dependencies/unitree_lerobot/CONVERT_NOTES.md` — 转换笔记
- `external_dependencies/unitree_lerobot/convert_unitree_inspire_oncam_to_groot.py` 等 — 主转换器

### 调试与探针

- `scripts/compare_rollout_joint_trace.py` — rollout joint/action 轨迹对比
- `scripts/probe_mamba_alignment_effect.py`
- `scripts/probe_right_hand_prediction.py`
- `scripts/probe_right_hand_sampling.py`
- `outputs/joint_rollout_trace_compare_summary_*.md`（5 个变体：reinit / weakalign / tunedalign / tunedalign2 / righthand1）
- `outputs/eval_runs/n1d7_g1_shaft_20260619.md` — N1.7 零样本报告
- `outputs/eval_runs/n1d7_finetuned_insertion_20260619.md` — N1.7 500 步 finetune 报告

### 长期记忆（跨会话可被 recall）

- `unitree-lerobot-conversion-pipeline.md` — 数据 pipeline 索引
- `groot-dataset-cleanup.md` — 数据集最终布局（25 GB / 663 eps）
- `agent-team-architecture.md` — Agent 团队 v1 → v2 升级路径
- `half-year-research-goal.md` — 6 个月目标 + 6 个里程碑

---

## 附录：Agent 贡献

| Agent | 角色 | 关键产出 |
|---|---|---|
| **Repo Scout** | 仓库结构 | 顶层目录、入口/脚本/配置/模型代码、outputs 命名规律、未跟踪文件分组、缺失文档清单 |
| **Environment Agent** | 任务环境 | 仿真/真机、观测/动作空间（含 dim/Hz/order）、控制频率、env↔policy 接口、env vs 数据集 schema gap |
| **Goal Agent** | 目标与进度 | 显式 vs 隐式目标、假设 H1-H4、已完成工作清单、时间线、5 个下一步候选 |
| **Supervisor** | 汇总 | 本报告（按 9 节模板整合，补全数字与缺口排序）|

—— 报告完 ——