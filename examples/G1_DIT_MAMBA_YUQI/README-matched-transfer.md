# Matched full-data action transfer

This entry point implements the frozen `g1-mamba-action-transfer-matched-r1`
contract. It consumes the completed, explicitly SHA-bound 180-train/8-dev cache
(21,988 stride-4 observations). Final and reserve episodes are inaccessible.
Existing models, original head flow loss, sampling, and earlier runners are
unchanged.

`action_transfer_matched_runner.py` has three stages:

- `preflight`: shared fresh student/native dev-0 baseline with 512 outputs before
  graph capture or any update; actual B4 native objective parity; two batches and both task/KD
  objectives at updates 0/1/3 with full gradient comparisons; three real KD
  updates discarded. It releases the B4 graph before a separate B1 four-Euler
  deployment graph check. No checkpoint is used to initialize training.
- `train`: exactly 20,000 matched B4 steps from the original student initialization.
  `task_only` uses native loss throughout; `distill_then_task` uses data-line
  teacher velocity KD for 10,000 steps and native loss for 10,000. Both execute
  the same teacher replay each step. Intermediate development quality never
  causes early stopping. Both conditions reference the same SHA-bound dev-0
  baseline from preflight. Each 10,000-step phase has a fresh AdamW and its frozen
  warmup/cosine schedule.
- `evaluate`: CPU-only joint summary of final development outputs and complete
  schedule/actual-input pairing, including explicit resume lineage. It does not
  measure deployment cost or read final/reserve outputs.

Training concatenates cached raw VLM features and original action inputs, then
runs native head preprocessing on B4. It does not concatenate stored B1 post-VL
features as a substitute for the actual B4 condition. The graph owns the original
student condition encoder and action core; native sampling, loss, teacher replay,
clipping and optimizer remain outside the graph. The student remains FP32 master,
BF16 autocast, eval mode with gradients enabled. Configured Dropout is unchanged
and effectively disabled. Graph signatures reject silent B1-to-B4 broadcasting.

## Invocation

Use the frozen plan and the completed cache report SHA. All stages require
`--plan`, `--artifact-root`, `--run-id`, `--output`, `--cache-report-sha256`, and
`--wall-seconds`. The GPU process environment must match the plan. Example fresh
identities are `batch-preflight-r1`, `train-task-only-r1`, and
`train-distill-then-task-r1`. Training additionally requires `--condition`.
`--output` is exactly `<artifact-root>/<run-id>/output` and must not exist.

Explicit recovery uses a new run ID plus `--resume-index` and
`--resume-index-sha256`. The same source/plan/cache/schedule must match. The
checkpoint binds student weights, optimizer parameter order, Python/NumPy/Torch
RNG and completed global step. Graph setup occurs before RNG restoration.
At step 10,000 recovery starts phase 2 with fresh optimizer state; at step 15,000
it restores phase-2 optimizer moments. Existing attempt files are never replaced.
There is no automatic retry or scheduler.

CPU evaluation defaults to the two canonical training `RESULT.json` files. To
select explicitly resumed terminal attempts, supply `--task-only-terminal` and
`--task-only-terminal-sha256`, and/or `--distill-then-task-terminal` and
`--distill-then-task-terminal-sha256`. Checkpoint parent indexes and fixed log
prefix hashes reconstruct and verify global steps 1 through 20,000; no mutable
latest-attempt pointer is used.

## Evidence and timing

Each `dev-stepN` and `train-probe-stepN` contains details, normalization, aggregate
results and arrays for normalized student/teacher actions, physical actions and
targets, both velocity fields and active masks. Development contains 512 outputs
per checkpoint; train probes contain 32. Checkpoints are saved at steps
5,000/10,000/15,000/20,000, including safe optimizer/RNG payloads and immutable
indexes.

`training_steps.jsonl` preserves the frozen batch, forward seed, LR, global/phase
step, both loss values, selected objective, actual time/noise/condition/target/mask
receipts and gradient groups. `gradient_norm_before_clip` is before clipping;
group statistics are after clipping. `batch_preparation_seconds` reports mmap,
first-access hash checks and batch concatenation separately.
`clean_step_wall_seconds` includes B4 native head processing, graph student,
eager teacher, backward, clipping and AdamW with boundary synchronization.
`elapsed_step_seconds` additionally includes preparation and evidence statistics.
These are actual training costs, not inference-only deployment timings.
Peak allocated/reserved counters are cumulative stage high-water values.

Run focused tests with CUDA disabled. CPU tests cover cache and proxy integrity,
B4 masked KD/teacher detachment, signature rejection, phase objectives,
optimizer-state transitions and explicit checkpoint/log lineage. Real CUDA Graph
compatibility and memory constraints are established only by formal preflight.
