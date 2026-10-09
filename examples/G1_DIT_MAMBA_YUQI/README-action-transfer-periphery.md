# Native periphery adaptation from task-only50k

This private runner implements the frozen ICRA27 `PERIPHERY_ADAPTATION.md` contract. It does not change shared GR00T/model/Graph helpers or launch itself. The Supervisor freezes the PLAN and runs stages explicitly. Historical runs, cache, final/reserve cohorts and simulator state remain untouched.

Both conditions load the valid task-only50k core semantic `567b48c8ddc18909ef9bf228920778a503fe0d943b910deb7434bb470aaf61a1`, its AdamW step40000 and RNG, plus independent B0 periphery copies. `core_only_continuation` freezes those copies; `joint_periphery` trains their FP32 masters. The original native head forward, loss, time/noise sampler and four-step action sampler remain the only algorithms. Each native periphery module executes through differentiable BF16 parameter views, including CategorySpecificLinear's bias; directly forwarding an FP32 bias would change B0 output dtype. Sampling recomputes the student's own raw-cache features. Teacher evaluation runs its own frozen B0 head and decoder.

Training Graph captures only core and preserves the live hidden/VL `requires_grad` flags. Preflight compares actual B4 eager/Graph native draws, loss, core/periphery gradients and input gradients under the historical gradient tolerances. Both conditions reproduce the complete parent512 dev output before any proof update. Three proof updates are discarded, every reachable periphery module gets a gradient/change witness, and native BF16 deployment materialization must be exactly equal. A private temporary composite checkpoint is saved, CPU-verified and GPU-reloaded; only that newly created scratch directory is deleted after passing. Its hashes and restore evidence remain. Preflight restores parent core/periphery/optimizer/RNG before completion.

B0 teacher/VLM content is checked using the existing canonical-owner `BaseAudit` at initial load, graph capture, proof, every1000 updates, probes, saves, reload and release. Differences retain tensor/byte fragments and end the attempt. There is no per-forward CPU content hash, retry, scheduler or automatic resume.

The two suffixes use B4, offsets200000..279999, global steps50001..70000 and cosine1e-5→1e-6 without warmup. Core AdamW continues40000→60000; joint periphery begins0→20000. Core groups are loaded unchanged. New periphery category biases (`b`, including their category dimension), ordinary biases and norm vectors have zero decay; other matrices have1e-4 decay. Clip1 includes the full active student, and logs retain preclip norms by core/periphery module and clipping frequency. Eval mode/dropout-off remains fixed.

Only final complete checkpoints persist. Exact parameter counts are230,785,600 core/158 tensors and515,142,048 periphery/81 tensors. Payload projections, excluding small serialization headers, RNG, logs and probes:

| Condition | FP32 weights + AdamW payload bytes |
| --- | ---: |
| core_only_continuation | 4,829,995,392 |
| joint_periphery | 8,951,131,776 |
| Total | 13,781,127,168 (12.835GiB) |

The total16GiB cap leaves about3.165GiB for other artifacts. Preflight ≤3600s plus two train stages ≤9000s each fits6GPUh; each process also enforces allocated44GiB. Preflight records measured proof step time and fails if projected20000 updates alone exceed the train bound. This projection excludes evaluation/loading/audits and is not a measured full-run throughput guarantee. Resource or numerical failure ends the attempt; it never changes horizon, precision or step count.

## PLAN and CLI

Start from the continuation PLAN's exact base/prepared/cache/cohort/grids/evaluation/optimizer/gradient/loss fields. New/overridden fields:

```text
source_base = 7b8244060322f1583bde8fa5f59fccb8622e426c
source_worktree, artifact_root, allowed_source_files
records, immutable_source_relative_paths (bind this worktree's unchanged helpers)
conditions = [core_only_continuation, joint_periphery]
starting_step = 50000; training_steps = 70000; sample_offset = 200000; batch_size = 4
schedule_file, schedule_sha256  # exactly20000 suffix rows: step, batch, forward_seed, lr
parent = {index: {path, sha256[, bytes]}, terminal: {path, sha256[, bytes]}}
run_ids = {preflight: ID, train: {core_only_continuation: ID, joint_periphery: ID}, evaluate: ID}
dev_probe_steps = [55000,60000,65000,70000]
train_probe_steps = [60000,70000]; weight_checkpoint_steps = [70000]
preflight_internal_wall_seconds = 3600; train_internal_wall_seconds = 9000
max_allocated_bytes = 47244640256; max_artifact_bytes = 17179869184
```

All commands need `--command`, `--plan`, `--plan-sha256`, `--artifact-root`, `--run-id`, `--output`, `--wall-seconds`. Train also needs `--condition`, `--preflight-result`, `--preflight-sha256` from this exact source/PLAN. CPU evaluate needs `--core-only-result`, `--core-only-sha256`, `--joint-result`, `--joint-sha256`. Example command shapes:

```bash
python examples/G1_DIT_MAMBA_YUQI/action_transfer_periphery.py --command preflight [common explicit identity arguments]
python examples/G1_DIT_MAMBA_YUQI/action_transfer_periphery.py --command train --condition joint_periphery --preflight-result /absolute/RESULT.json --preflight-sha256 SHA [common explicit identity arguments]
python examples/G1_DIT_MAMBA_YUQI/action_transfer_periphery.py --command evaluate --core-only-result /absolute/control/RESULT.json --core-only-sha256 SHA --joint-result /absolute/joint/RESULT.json --joint-sha256 SHA [common explicit identity arguments]
```

Use the original venv, dependency overlay and `TRITON_F32_DEFAULT=ieee`. GPU commands expose exactly the PLAN's physical GPU. CPU evaluation verifies both complete20000-row matched receipts and final payload identities, reports all metrics and paired episode differences, and runs10000 episode bootstrap replicates with seed20261009. It applies only the fixed70k development gate: flow and combined improve≥2%, each in≥6/8 episodes. Passing does not establish the separate native≤1.02/full-policy10%/final simulation goals.

Focused CPU coverage: actual native small-dimension B4 forward exactness, category bias dtype semantics and nonstale views, core/periphery gradient flow, teacher isolation, control freezing, dual AdamW clocks, composite names/moments/weights and deterministic paired bootstrap. Production512 and CUDA Graph equivalence require the explicit GPU preflight.
