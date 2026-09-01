# Phase 1 Audit 综合报告 — Supervisor 判定

**日期：** 2026-06-27
**作者：** Supervisor（PI）
**目标：** 在不动代码的前提下，定位 n1d7_main_300ep_baseline_2026-06-27 跑出 **0.0 / 10 success** 的根因
**输入：** 6 个 sub-agent audit 报告（dataset / environment / training / evaluator / github / baseline）

---

## 0. 关键结论（先看这个）

**外部 hypothesis "官方 N1.7 + 我们数据集 ≈ 90% success" 在 NVIDIA 官方仓库记录里没有依据：**

- N1.7 **没有发布任何 G1 finetuned checkpoint**（HuggingFace 上只有 base + LIBERO/DROID/SimplerEnv）
- N1.6 唯一 G1 checkpoint (`GR00T-N1.6-G1-PnPAppleToPlate`) 在社区 issue #574 / #590 复现 0-1%
- Issue #674 直接命中我们的现象："policy only learns motions without successful grasping"，**0 maintainer 回复**
- 我们用的 G1/shaft-insertion pipeline **完全来自外部 submodule**（`NVlabs/GR00T-WholeBodyControl` + Unitree IsaacLab sim 包），**不是 NVIDIA Isaac-GR00T 官方 reference**

→ **0% 不一定是"我们做错了什么"，可能是"这条路在当前 GR00T 状态本来就没人跑通过"**。但这不能成为接受 0% 的理由——我们需要确定**是不是我们 pipeline 里有可修的 bug 把可能性埋了**。

---

## 1. 根因候选矩阵（按"破坏性 × 可修性"排序）

> 全部都是 agent team 的 evidence-based 结论，不是猜测。每条标证据强度。

### R0 — **CRITICAL**：embodiment_id mismatch (训练=10，推理=25)

| 维度 | 详情 |
|---|---|
| **机制** | 训练 finetune 用 `NEW_EMBODIMENT`，saved `processor/embodiment_id.json` 写入 `"new_embodiment": 10`。推理 slim policy hardcode `unitree_g1 → 25`（`policy.py:126`）。`CategorySpecificMLP` 按 id 选 projector 权重——训练学到的 projector[10] 在推理时**完全不被读取**，实际用的是 projector[25]（**未训练的随机初始化**）。|
| **证据强度** | **强**（environment + training + evaluator 三 agent 一致）|
| **可修性** | **极高** — 不动模型，只改 slim policy 让它读 saved `embodiment_id.json`|
| **影响** | 单一原因即可解释 100% failure |

### R1 — **CRITICAL**：state statistics 公式不一致（真实 vs synthetic identity）

| 维度 | 详情 |
|---|---|
| **机制** | 训练时 processor 用 `new_embodiment` 的真实 dataset mean/std/q01/q99。推理时 `Gr00tN1d7G1ShaftProcessor` 在 `processing.py:165-220` 找 `"unitree_g1"` key 不存在 → fallback 到 `real_g1_relative_eef_relative_joints` → 然后用 `mean=[0]*dim, std=[1]*dim, min=[-1]*dim, max=[1]*dim` **覆盖**为 synthetic identity。Model 收到 OOD 输入。|
| **证据强度** | **强**（evaluator + environment + training 三 agent 一致）|
| **可修性** | **极高** — slim processor 改读 saved `statistics.json` 里的 `new_embodiment` 段 |
| **影响** | 解释为什么 model 即使过 projector 后也无法输出有意义的动作 |

### R2 — **CRITICAL**：schema slot 错位 (4 流 vs 7 流)

| 维度 | 详情 |
|---|---|
| **机制** | 训练 schema `[LA, RA, LH, RH]` 把 left_arm 放在 state_features `[0:7]`。推理 schema `[LL, RL, W, LA, RA, LH, RH]` 把 left_arm 放在 `[15:22]`。`state_encoder` 收到的同一个关节位置在 train/inference 时位于不同 slot，投影结果完全不同。|
| **证据强度** | **强**（training agent 直接算出 slot 索引 + 与 R0/R1 联动解释 0%）|
| **可修性** | **中** — 必须保证 train 和 inference 的 modality_configs 是**同一份文件**，并且 embodied_id 走同一个 slot layout。Slim processor 需要从 saved checkpoint 的 `modality_configs` 读，而不是 hardcode `N1D7_G1_SHAFT_MODALITY_CONFIG` |
| **影响** | 即使 R0/R1 修了，slot 错位也会让 model 输出的 left_arm action 变成"语义上的 waist" |

### R3 — **HIGH**：lower-body / waist / base_height_command / navigate_command 完全没训练

| 维度 | 详情 |
|---|---|
| **机制** | 训练数据集 stage 2 只 4 流 26-dim。`modality_config_insertion_shaft_n1d7.py` 只声明 4 流。env 期望 7 流 33-dim。推理时 slim policy 用 0 填充 3 个未训练 action key。|
| **反证** | evaluator agent 反驳："env 端 `insertion_env.py` 不使用 base_height_command/navigate_command；waist 是 absolute set-point，0 = 合法 stand still"。所以 zero-fill **不会让 WBC 失稳**。但 env 期望 slim policy 报告 7 流 action，slim policy 报告 7 流（4 真实 + 3 zero），env 接受。|
| **证据强度** | **中**（dataset + environment + training 三 agent 列出，但 evaluator 反驳机制）|
| **可修性** | **低** — 必须重新采集数据（改 `unitree_lerobot/utils/constants.py:270-307` 的 `json_state_data_name`），1-2 天工作量。|
| **影响** | 模型确实没学过 lower-body，但 **不是 0% 的独立根因**（与 R0/R1/R2 协同放大） |

### R4 — **HIGH**：language key 不一致 + 训练时是垃圾字符串

| 维度 | 详情 |
|---|---|
| **机制** | 训练 modality config 用 `annotation.human.action.task_description`；推理 slim processor 用 `annotation.human.task_description`。即使 key 对齐，stage 2 conversion 把所有 task description 硬编码成 `"insertion_shaft_main_300ep"`（dataset path），不是 raw `text.goal`（如 `"pick up cube."`）。Model 学到 "忽略语言信号"。|
| **证据强度** | **强**（dataset + environment 两 agent 直接追踪字段路径）|
| **可修性** | **高** — 修复 stage 2 conversion 让 `text.goal` 正确写入；统一 modality config 的 language key |
| **影响** | 影响相对小（单任务时 language 是 constant）但**违背了 simple > complex 原则**：引入了一个 silent wrong input |

### R5 — **MEDIUM**：DiT 容量减半 + backbone 冻结

| 维度 | 详情 |
|---|---|
| **机制** | 我们 fork config `diffusion_model_cfg.num_layers=16` vs 上游 32 层。`tune_llm=False, tune_visual=False` vs 上游 True。`state_dropout_prob=0.8`（fork 默认）vs 上游 0.2。|
| **证据强度** | **中**（training + github agent 一致；loss 1.29→0.04 说明 model 至少能拟合训练分布，所以容量不是 0% 的主因）|
| **可修性** | **低** — 改 config 涉及重新训练；改 num_layers 需要重新对齐 weight shape |
| **影响** | 单独不解释 0%，但与 R0/R1/R2 叠加时放大问题 |

### R6 — **MEDIUM**：eval pipeline instrumentation bug

| 维度 | 详情 |
|---|---|
| **机制** | (a) server stdout buffering 导致 server.log 截断 10 分钟输出，**无法验证权重是否完整加载**。(b) Episode 1 在 step 12 提前终止，video 被 silent 删除。(c) `rollout_policy.py` 不记录 per-step `shaft_housing_distance` / termination 来源 / action 分布。|
| **证据强度** | **强**（evaluator 直接看 log mtime 差）|
| **可修性** | **极高** — 加 `PYTHONUNBUFFERED=1` 到 server 启动 + 在 `rollout_policy.py` 加 per-step jsonl |
| **影响** | 让**任何后续 audit 都没法验证模型状态**，是元 bug |

### R7 — **LOW**：eval 协议其他小问题

| 维度 | 详情 |
|---|---|
| **机制** | `UNITREE_SIM_HAND_DIM=6` 没 export（但 action 端 6-dim stable，state 端第 7 dim 是 padded zero 兼容训练）；eval env 没有 reference baseline 数字；NVlabs client 默认 `max_episode_steps=1440`，我们用 800（可能物理上时间不够）；`success_term.params={}` fragile。|
| **证据强度** | **中**（environment + github agent 列出）|
| **可修性** | **高** — 加 env var export、延长 episode |
| **影响** | 单独不解释 0% |

---

## 2. "必须修" vs "可以先不修" 判定

按用户的新方向（**simple > complex / official > customized / reproducible > optimized**）：

### 2.1 必须 Phase 2 修（最小必要修改）

| # | 修复 | 性质 | 改几个文件 |
|---|---|---|---|
| **F1** | **slim policy 读 saved `embodiment_id.json`**，把 `unitree_g1` 映射到 `new_embodiment` 的 projector id=10 | bug fix | 1 个 (`policy.py`) |
| **F2** | **slim processor 读 saved `statistics.json`** 的 `new_embodiment` 段，不覆盖为 synthetic identity | bug fix | 1 个 (`processing.py`) |
| **F3** | **slim processor 复用 saved modality_configs**（从 checkpoint-5000 加载），不 hardcode `N1D7_G1_SHAFT_MODALITY_CONFIG` | bug fix | 1 个 (`processing.py`) |
| **F4** | **统一 language key**：让训练 modality config 用 `annotation.human.task_description`，且 stage 2 conversion 写入真实 task text（不是 dataset path） | 数据一致性 | 1 个 modality config + 1 个 conversion 脚本 |
| **F5** | **eval server 加 `PYTHONUNBUFFERED=1`** + **rollout_policy 加 per-step jsonl trace** | instrumentation | 2 个脚本 + 1 个 rollout 文件 |
| **F6** | **eval client export `UNITREE_SIM_HAND_DIM=6`**（防御性，避免未来 hand dim 改动时 silent fail） | hardening | 1 个 eval 脚本 |

**合计 6 项**，全部是"把 inference path 的 hardcode 改为读 saved checkpoint metadata"，**不修改模型架构、不修改训练超参、不增加数据**。

### 2.2 不修（违背 simple > complex）

- ❌ 不改 `diffusion_model_cfg.num_layers`（动架构）
- ❌ 不改 `tune_llm/visual`（动训练算法）
- ❌ 不加 `color_jitter_params` / `random_rotation_angle`（增加 augmentation）
- ❌ 不改 `max_steps=5000`（不调超参）
- ❌ 不重采 lower-body 数据（1-2 天工作，违背"先验证再优化"）
- ❌ 不修 `gradient_checkpointing=True`（虽然偏离官方 False，但这是为低显存改的，是兼容性 fix 而非性能优化）

### 2.3 必须先确认（Phase 1 续集）

- Q1: **F1-F3 修完之后，0% 是不是变成非零？** 如果 0 → 非零（如 30-50%），说明 R0/R1/R2 是真根因。
- Q2: **F4 修了之后，训练时 model 看到的 language 是不是真的变成了 task description？** 还是说 dataset metadata 已被 model 当 constant ignore？
- Q3: **F5 加的 per-step trace 是不是能区分"timeout" vs "out_of_bound" vs "success"？** 现在没法区分。

---

## 3. 重新审计旧 baseline run 的 verdict

旧 run `n1d7_main_300ep_baseline_2026-06-27` 的 verdict：

- **之前**（在 baseline agent 自己标准下）：`baseline_certified`（可复现 10/10）
- **现在**（在新方向下）：**`baseline_revoked`**，角色调整为 **`zero_baseline_anchor`**（零起点对照锚点）

理由：

1. **0% performance 远低于任何 baseline 阈值**（建议 ≥50%）
2. **多个 high-probability bug 已暴露**（R0/R1/R2）—— 旧 run 是在 inference path 损坏的条件下跑的，**不代表模型本身的 baseline**
3. metadata 冻结有效，但**只能作为"0 起点"用于验证修复后的 delta**

下一步：

- 把旧 `experiments/baselines/2026-06-27-n1d7-main-300ep-dit/` 保留，**新增 `experiments/baselines/2026-06-27-revocation/`** 记录 verdict 历史
- metadata.json 加 `verdict_history` 字段，标记旧 verdict `superseded`

---

## 4. Phase 2 建议方案

### 4.1 范围

**只改 inference path**（slim policy / slim processor / eval scripts）。**不动训练侧任何代码、不动训练 config、不动数据 conversion**。

### 4.2 修改清单（最小化）

| # | 文件 | 修改 | 行数估计 |
|---|---|---|---|
| F1 | `gr00t/model/gr00t_n1d7/policy.py` | 让 `Gr00tN1d7G1ShaftPolicy.__init__` 读 `processor/embodiment_id.json`，把 `unitree_g1 → 10` 而不是 hardcode `25` | ~10 行 |
| F2 | `gr00t/model/gr00t_n1d7/processing.py` | `Gr00tN1d7G1ShaftProcessor.__init__` 直接用 `statistics["new_embodiment"]`，不再 fallback + 不再覆盖为 synthetic | ~15 行 |
| F3 | `gr00t/model/gr00t_n1d7/processing.py` | `N1D7_G1_SHAFT_MODALITY_CONFIG` 改为从 saved checkpoint 读 modality_configs（如果存在 new_embodiment） | ~10 行 |
| F4a | `examples/GR00T-WholeBodyControl/modality_config_insertion_shaft_n1d7.py` | language key 改为 `annotation.human.task_description` | 1 行 |
| F4b | `external_dependencies/unitree_lerobot/convert_unitree_inspire_multicam_to_groot_keep_effort.py` | `annotation.human.task_description` 写入 `raw_data.json["text"]["goal"]` 而不是 dataset path | ~5 行 |
| F5a | `scripts/baselines/run_eval_n1d7_main_300ep.sh` | server 启动加 `PYTHONUNBUFFERED=1` | 1 行 |
| F5b | `gr00t/eval/rollout_policy.py` | 每 step 写 jsonl trace：`{step, obs_state, action, info, shaft_housing_distance, terminations}` | ~20 行 |
| F6 | `scripts/baselines/run_eval_n1d7_main_300ep.sh` | client 启动前 `export UNITREE_SIM_HAND_DIM=6` | 1 行 |

**合计 ~63 行改动**，**全部是 silent-bug fix + instrumentation**，**无架构改动、无超参改动、无 augmentation**。

### 4.3 验证计划

1. **Phase 2-A：实现 F1-F3 + F5a + F6**（不动数据、不动训练）→ 用现有 checkpoint-5000 重跑 10-episode eval
   - 期望：success rate 0% → 任意非零值
   - 如果仍是 0%：说明 R0/R1/R2 不是主因，bug 仍在别处（继续审计）
2. **Phase 2-B：实现 F4a + F4b**（数据一致性）→ 重新跑 stage 2 conversion → 重新 finetune 5000 步 → eval
   - 期望：success rate 比 Phase 2-A 高
3. **Phase 2-C（如果前两阶段成功）：不动模型、不动超参，仅添加 per-step trace 后跑全套 eval** → 写 Phase 3 report

### 4.4 Phase 3 报告承诺

回答用户的 5 个问题：

1. 官方 N1.7 baseline 是否可复现？（N1.7 在我们的 setup 下不可能"完全复现"——因为官方没发 G1 ckpt。但我们能复现 N1.7 base 在我们的 task 上的参考数字。）
2. 如果不可复现，什么 evidence 解释 discrepancy？
3. 哪些修改是绝对必要的？（答：F1-F6）
4. 哪些修改是故意避免的？（答：架构、超参、augmentation、数据重采）
5. 最终 baseline 是？

---

## 5. 给用户的最终请示

**是否批准按 §4.2 清单实施 Phase 2-A**（F1-F3 + F5a + F6，~37 行 inference path bug fix + instrumentation）？

批准后我立即执行 Phase 2-A，跑一轮 eval 验证 R0/R1/R2 是不是真根因。**不批准则继续 Phase 1 续集（Q1-Q3 验证）**。

**注意**：Phase 2-A 完成后我**不会**自动进入 Phase 2-B（涉及重新训练 + 改 stage 2 conversion）；会再次请示。

---

## 6. 文件路径索引

本报告：`reports/phase1_audit/2026-06-27/SUPERVISOR_SYNTHESIS.md`

各 sub-agent 详细 audit 报告（待 sub-agent 写入）：
- `reports/dataset/2026-06-27-phase1-audit.md`（dataset agent）
- `reports/environment/2026-06-27-phase1-audit.md`（environment agent）
- `reports/training/2026-06-27-phase1-audit.md`（training agent）
- `reports/evaluator/2026-06-27-phase1-audit.md`（evaluator agent）
- `reports/github/2026-06-27-official-n1d7-recipe.md`（github agent）
- `experiments/baselines/2026-06-27-re-audit/re-audit-report.md`（baseline agent）
