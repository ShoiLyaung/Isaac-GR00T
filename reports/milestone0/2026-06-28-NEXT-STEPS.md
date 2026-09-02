# Milestone 0 — Next Steps（等 HF gated 访问批准）

**日期**：2026-06-28
**状态**：⏸️ 等待用户（人类）申请 HF gated 访问 `nvidia/Cosmos-Reason2-2B`

---

## 1. 用户需做的（人类手动操作）

### Step 1: 申请 HF 访问

1. 浏览器打开 https://huggingface.co/nvidia/Cosmos-Reason2-2B
2. 登录你的 HF 账号
3. 点击 **"Agree and access repository"** 或 **"Request access"** 按钮
4. NVIDIA 的 gated access 流程可能要求填一个简短问卷（说明用途）
5. 等待 NVIDIA 批准（**通常 1-24 小时**，有时更快）

### Step 2: 验证访问

批准后，在 shell 里验证：
```bash
curl -sL https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json \
  -H "Authorization: Bearer $(cat ~/.cache/huggingface/token 2>/dev/null || echo '')" \
  | head -10
```

或者直接：
```bash
huggingface-cli whoami
huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir /tmp/cosmos-test 2>&1 | head -5
```

如果看到 "Downloading config.json" 之类，说明访问通过。

### Step 3: 启动 finetune

M0 launcher 已经准备好，只需：

```bash
cd /home/sl/Isaac-GR00T-m0-clean
source .venv/bin/activate

# 把 HF_HUB_OFFLINE 和 TRANSFORMERS_OFFLINE 注释掉（设回默认行为）
# 然后跑：
export HF_TOKEN="<your_hf_token>"  # 或 huggingface-cli login
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=/home/sl/Isaac-GR00T-m0-clean/outputs/m0_finetune_$(date +%Y%m%d_%H%M%S) \
nohup bash examples/insertion_shaft_m0/finetune.sh \
  > /home/sl/Isaac-GR00T-m0-clean/outputs/logs/m0_d/finetune.log 2>&1 &
```

**或者** 编辑 `finetune.sh` 把 line 103-104 的：
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```
改为注释：
```bash
# export HF_HUB_OFFLINE=1   # disabled after HF gated access approved
# export TRANSFORMERS_OFFLINE=1
```

### Step 4: 监控

```bash
# Tail log
tail -f /home/sl/Isaac-GR00T-m0-clean/outputs/logs/m0_d/finetune.log

# GPU usage
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv -l 5
```

### Step 5: 跑 eval

finetune 完成后（约 30-60 分钟）：

```bash
cd /home/sl/Isaac-GR00T-m0-clean
source .venv/bin/activate

# 找最新的 finetune output
ls -dt outputs/m0_finetune_*/checkpoint-10000 | head -1
# 设为 MODEL_PATH，例如：
MODEL_PATH=/home/sl/Isaac-GR00T-m0-clean/outputs/m0_finetune_20260628_xxxxxx/checkpoint-10000

MODEL_PATH=$MODEL_PATH nohup bash examples/insertion_shaft_m0/eval.sh \
  > /home/sl/Isaac-GR00T-m0-clean/outputs/logs/m0_d/eval.log 2>&1 &
```

---

## 2. 当前 m0-clean 完整 state

### 代码
- **worktree**: `/home/sl/Isaac-GR00T-m0-clean` @ commit `23ace64 GR00T N1.7 Release`
- **修改的源文件**: 2 个，共 +22 行
  - `gr00t/eval/sim/env_utils.py` (+2)
  - `gr00t/eval/rollout_policy.py` (+20)
- **新文件**: 15 个，共 ~3100 行
  - `scripts/lerobot_conversion/m0/`: 6 个 conversion 脚本 + 1 orchestrator + 1 smoke test
  - `examples/insertion_shaft_m0/`: modality_config + finetune.sh + eval.sh
  - `gr00t/eval/sim/UnitreeG1/`: __init__.py + unitree_g1_env.py + test_smoke.py
  - `reports/phase0_audit/2026-06-27-M0_GAP_ANALYSIS.md`（M0 spec 文件）

### 数据
- **转换输出**: `/home/sl/groot_datasets/lerobot/insertion_shaft_300ep_m0/`（300 ep / 1.3 GB / LeRobot v2.1）
- **stats 已生成**: `meta/stats.json` + `meta/relative_stats.json`
- **dataloader smoke test**: PASS

### 环境
- **Python venv**: `.venv/`（Python 3.10.12）
- **关键 dep**: torch 2.7.1+cu128, transformers 4.57.3, flash-attn 2.7.4.post1
- **GPU**: 0 free, 1 free

---

## 3. 复现命令（批准后可直接跑）

### 3.1 准备环境（一次性，已完成）
```bash
cd /home/sl/Isaac-GR00T-m0-clean
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3.2 数据转换（一次性，已完成）
```bash
cd /home/sl/Isaac-GR00T-m0-clean
source .venv/bin/activate
bash scripts/lerobot_conversion/m0/run_all.sh \
  --raw-root /home/sl/Isaac-GR00T/groot_datasets/raw/insertion_shaft/2026_06_02_main_300ep_NEWEST \
  --out-root /home/sl/groot_datasets/lerobot/insertion_shaft_300ep_m0
python gr00t/data/stats.py \
  --dataset-path /home/sl/groot_datasets/lerobot/insertion_shaft_300ep_m0 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/insertion_shaft_m0/modality_config_insertion_shaft_m0.py
```

### 3.3 Finetune（每次运行）
```bash
cd /home/sl/Isaac-GR00T-m0-clean
source .venv/bin/activate
# After HF approval, comment out HF_HUB_OFFLINE in finetune.sh, or:
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_TOKEN="<your_hf_token>"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=/home/sl/Isaac-GR00T-m0-clean/outputs/m0_finetune_$(date +%Y%m%d_%H%M%S) \
nohup bash examples/insertion_shaft_m0/finetune.sh \
  > /home/sl/Isaac-GR00T-m0-clean/outputs/logs/m0_d/finetune.log 2>&1 &
```

### 3.4 Eval（finetune 完成后）
```bash
MODEL_PATH=/home/sl/Isaac-GR00T-m0-clean/outputs/m0_finetune_<DATE>/checkpoint-10000 \
nohup bash examples/insertion_shaft_m0/eval.sh \
  > /home/sl/Isaac-GR00T-m0-clean/outputs/logs/m0_d/eval.log 2>&1 &
```

---

## 4. M0 spec 涉及的所有改动总览

| 类别 | 文件 | 改动 | 行数 |
|---|---|---|---|
| 源改 | `gr00t/eval/sim/env_utils.py` | +1 prefix mapping | +2 |
| 源改 | `gr00t/eval/rollout_policy.py` | +9 (env_fn + elif) | +20 |
| 新 | `gr00t/eval/sim/UnitreeG1/__init__.py` | 空 docstring | 3 |
| 新 | `gr00t/eval/sim/UnitreeG1/unitree_g1_env.py` | 薄 wrapper | 324 |
| 新 | `gr00t/eval/sim/UnitreeG1/test_smoke.py` | 5 smoke tests | 191 |
| 新 | `scripts/lerobot_conversion/m0/01_enumerate_episodes.py` | raw→episodes.jsonl | 171 |
| 新 | `scripts/lerobot_conversion/m0/02_pick_camera_mapping.py` | hardcode camera map | 72 |
| 新 | `scripts/lerobot_conversion/m0/03_encode_videos.py` | ffmpeg mp4 | 170 |
| 新 | `scripts/lerobot_conversion/m0/04_write_parquets.py` | data.json→parquet | 168 |
| 新 | `scripts/lerobot_conversion/m0/05_write_meta.py` | write meta/*.json | 212 |
| 新 | `scripts/lerobot_conversion/m0/99_smoke_test.py` | dataloader smoke | 257 |
| 新 | `scripts/lerobot_conversion/m0/run_all.sh` | orchestrator | 72 |
| 新 | `examples/insertion_shaft_m0/modality_config_insertion_shaft_m0.py` | new embodiment | 101 |
| 新 | `examples/insertion_shaft_m0/finetune.sh` | launcher | 124 |
| 新 | `examples/insertion_shaft_m0/eval.sh` | server+client | 142 |
| 新 | `reports/phase0_audit/2026-06-27-M0_GAP_ANALYSIS.md` | M0 spec doc | ~400 |

**总计**: 2 个源文件 +22 行 / 15 个新文件 ~3100 行 / 0 个模型/超参/augmentation 修改

---

## 5. M0 spec vs 实际差距

### 满足的部分
- ✅ 完全没改模型架构
- ✅ 完全没动训练超参（用 default）
- ✅ 完全没加 augmentation（除 finetune.sh 写死的官方 color_jitter，仿官方 example）
- ✅ 数据层从零写，不复用旧 worktree
- ✅ Eval env 从零写，目录重命名为 `UnitreeG1/` 避免与旧 `UnitreeIsaacLab/` 混淆
- ✅ Launchers 仿官方 example 模板

### 偏离的部分
- ⚠️ m0-clean **必须**有 HF gated 访问 `nvidia/Cosmos-Reason2-2B` 才能跑（**官方**的 M0 recipe 就是这样写的）
- ⚠️ 数据集采集是单手任务（right_arm 几乎不动），**这是 raw 数据本身的特性**，不是我们改的
- ⚠️ Eval env 的 `_map_action` 可能需要 GPU 真机校准（arms 可能被叠加两次 default offset）

---

## 6. 下一步（等 HF 批准）

1. 用户：申请 + 等待 HF 批准
2. 用户：批准后 `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE` + 设 `HF_TOKEN`
3. 跑 `bash examples/insertion_shaft_m0/finetune.sh`
4. 监控 ~30-60 分钟
5. 跑 `bash examples/insertion_shaft_m0/eval.sh`
6. 监控 ~10 分钟
7. 写 Phase 0-D 报告

---

**当前 waiting for**：用户（人类）申请 HF gated 访问。
