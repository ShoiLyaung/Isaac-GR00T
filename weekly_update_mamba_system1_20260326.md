---
marp: true
paginate: true
theme: default
_class: lead
size: 16:9
---

# 周报汇报
### NVIDIA Isaac-GR00T 中可替换 System1 与 Mamba Backbone 实现进展
**时间:** 2026-03-20 → 2026-03-26
**汇报人:** Wang Jiaxin
**指导老师:** Yan Shijun

---

### 1. 本周工作概述
本周主要工作是在 Isaac-GR00T 代码库中实现“可替换的 System1”，并引入基于 Mamba 的新 System1 Backbone，同时保持现有训练与评测链路兼容。

当前阶段结论如下：
- 已支持通过配置/CLI 在 `DiT System1` 与 `Mamba System1` 之间切换。
- 经过 checkpoint 加载后重初始化修复，Mamba 训练过程已具备数值稳定性。
- 训练链路与 server/client evaluation 链路均已跑通。
- 当前主要瓶颈已从“工程链路是否可运行”转为“策略效果不足”，正式仿真评测结果仍为 `DiT = 0.8`、`Mamba = 0.0`。

---

<style scoped>
section {
  font-size: 24px;
  line-height: 1.55;
}
table {
  font-size: 18px;
}
h3 {
  color: #2B547E;
}
</style>

### 2. 本周进展
| 模块 | 关键更新 |
|------|--------------|
| **System1 重构** | ✅ 完成原始 System1 实现与实例化位置定位，并梳理 forward I/O 契约，包括 `batch` keys、输出 keys、关键 tensor shape 与 loss 依赖。 <br> ✅ 新增 `system1_type = dit | mamba` 配置项，默认保持 `dit`，兼容原流程。 |
| **Mamba 集成** | ✅ 新增 `MambaSystem1`，并保证 forward 输入输出协议与原始 System1 保持一致。 <br> ✅ 保持 dataset format、dataloader 与 System2 不变，改动集中在 System1 和加载逻辑。 |
| **训练链路** | ✅ `launch_finetune.py` 与 `finetune_g1_mamba.sh` 已可直接运行 Mamba 训练。 <br> ✅ 修复 checkpoint 加载后新增 action head 参数初始化异常，Mamba smoke test 与完整训练均可完成。 |
| **评测链路** | ✅ Mamba checkpoint 已可通过 server/client evaluation 完整执行。 <br> ✅ 新增 rollout 级别分析脚本，用于与 DiT 做 joint/action 轨迹对比。 |

---

### 3. 关键实验与阶段性结论
| 实验 | 结果 | 结论 |
|------|------|------|
| **完整训练** | Mamba 完整训练已成功结束，最终产物保存至 `checkpoint-10000` | 说明训练基础设施已打通，训练不会因数值问题中断 |
| **正式评测** | DiT: `0.8`，Mamba: `0.0` | 当前核心问题是策略表现，而非训练/部署链路失效 |
| **Safety 定位** | 新增 per-joint safety 日志，定位到早期版本中多个 arm joint 速度超限 | 早期失败的主要原因是 upper-body 动作不稳定 |
| **Inference 对齐实验** | 弱化 action alignment，避免将 Mamba 输出过度压缩 | alignment 过强确实是问题之一，但不是最终根因 |
| **数值稳定性调试** | 将第 1 步 NaN 追踪到 checkpoint 加载后新增 `action_head` 参数已为非有限值 | 通过对 missing/non-finite 的 `action_head` 参数做加载后重初始化，解决了训练初期 NaN |
| **Right-hand 单因素实验** | 将 `right_hand` scale 从 `0.1` 提高到 `1.0` 后，正式评测仍为 `0.0` | `right_hand` 不是当前主阻塞项 |

---

### 4. 当前问题诊断
- 当前 Mamba checkpoint 已具备“可训练、可部署、可评测”的工程状态，但仍无法完成目标任务。
- 当前主要差距已经收敛到 **轨迹语义与控制分布不匹配**，具体表现为：
  - `right_arm` 多个关键 joint 的行为仍与 DiT 偏离明显
  - `navigate_command` 的方向与幅值分布不合理
  - `target_upper_body_pose` 在 self-rollout 中显著弱于 DiT
- 具有代表性的 rollout 结果包括：
  - `right_shoulder_yaw_joint` 的 sign match 仅为 `1 / 10`
  - `right_wrist_yaw_joint` 的 sign match 为 `0 / 10`
  - `target_upper_body_pose` 仅约为 DiT 的 `16.6%`
  - 即使单独放开 `right_hand` scale，正式评测成功率仍然为 `0.0`

---

### 5. 遇到的主要问题
- **本地无法直接使用官方 Mamba backend**：当前环境中的 `mamba_ssm` / `selective_scan_cuda` 与现有 Torch/ABI 不兼容，因此当前实现暂时采用稳定的本地 temporal mixer，而非官方 CUDA kernel。
- **基础设施问题解决后，问题重心转向策略表现**：NaN、shape mismatch、server/client 启动失败等问题已基本排除，当前挑战主要在行为质量。
- **Rollout 分布漂移明显**：一旦进入 Mamba 自身滚出的 observation 区域，动作语义进一步偏移，尤其集中在 right arm 与导航命令。
- **局部修补无法解决全局问题**：无论是弱化 alignment，还是单独调高 `right_hand` scale，都无法显著提升成功率。

---

### 6. 已尝试的方法
| 方法 | 目的 | 结果 |
|------|------|------|
| 在 System1 中加入 shape/assert 日志 | 明确 forward I/O 契约 | 已确认 Mamba 替换后与原 System1 I/O 兼容 |
| 增加 per-joint safety 日志 | 定位 safety fallback 触发原因 | 已识别并消除早期重复触发的 arm joint 速度超限问题 |
| 增加梯度/参数 finite probe | 追踪 NaN 根因 | 定位到新 `action_head` 模块加载后参数异常 |
| checkpoint 加载后重初始化缺失/非有限参数 | 恢复训练稳定性 | 成功解决第 1 步 backward 即出现 non-finite 的问题 |
| 用稳定 temporal block 替代不稳定本地 mixer | 保证 fallback 可训练 | 已支持 Mamba 完整训练完成 |
| 弱化 inference alignment | 避免动作幅值被过度压制 | 改善了动作量级，但未解决任务成功率问题 |
| 将 `right_hand` scale 提升至 `1.0` | 验证 hand actuation 是否是主瓶颈 | 对正式评测无显著帮助 |

---

### 7. 下周计划
| 任务 | 预期产出 |
|------|------------------|
| 联合分析 `right_arm` 与 `navigate_command` | 进一步定位 rollout 级别与 DiT 的真实偏差来源 |
| 设计更有针对性的 inference alignment / joint-wise correction | 验证是否可以在不改训练架构的前提下稳定 rollout |
| 启动 Phase 2：Video -> Robot injection | 评估场景先验是否能改善 manipulation 语义 |
| 基于新架构重新训练与评测 | 验证成功率能否从当前 `0.0` 提升 |

---

*汇报人:* **Wang Jiaxin**
*日期:* 2026-03-26
