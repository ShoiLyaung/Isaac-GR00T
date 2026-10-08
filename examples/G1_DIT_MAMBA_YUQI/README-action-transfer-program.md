# Action-transfer program, Phase A/B

The first executable phase runs fresh episode125 data-line velocity KD for5000
updates and fixed-checkpoint native-sampler diagnostics. It does not implement
matched multi-episode objectives, trajectory KD training or full-VLA training.
Those require new frozen experiments; current VLM/periphery and B0 stay frozen.
No existing model, Graph helper, sampler, loss or prior result is modified.

Fresh command:

```
action_transfer_program_runner.py long-kd --run-id long-kd-r1 \
  --plan ROOT/phase-a-plan-r1/PLAN.json --artifact-root ROOT \
  --output ROOT/long-kd-r1/output --wall-seconds 3550
```

The219-material PLAN and new5000-row schedule are checked before model loading.
The first200 anchors/seeds match the old short experiment, while the cosine LR
explicitly uses the new5000-step length. This does not isolate step count as a
single causal variable. Fresh initialization, teacher, native active16 mask,
Graph action student and eval-with-grad/dropout-disabled mode remain unchanged.
The old create_graph/graph_step/evaluate/probes/integrity helpers are reused.
The original200-step quality predicates are reported at every probe but never
terminate this authorized long phase. Nonfinite, identity, resource, Graph/eager
numerical or sampler-observation failures stop the current attempt, preserving
its completed-step count and any valid immutable checkpoint indexes.

`initial-quality` contains the64-output fresh baseline, checked against the
anchored old step0 arrays (old RESULT hashes verified first). `probes/stepN`
for0/200/1000/2500/5000 contains quality evidence, four-case Graph/eager numerical
checks and trajectory observations. Step0 refers to initial-quality; later
quality outputs use the unchanged evaluator's actions, physical arrays, true
targets, velocities, masks and min/max metadata. `training_steps.jsonl` preserves
all original actual-input receipts and synchronized full cached-input KD step
measurements, including teacher replay/clip/AdamW; hashes and diagnostics remain
outside those intervals. No VLM/RGB training-throughput claim is made.

Trajectory observations temporarily delegate the existing sampler's action
encoder, core and decoder. All4 Euler updates still execute the original
get_action_with_features implementation. Each fixed anchor0/320 and seed42 runs
teacher and student driver paths separately. Captured fullH40 x_t/time/core
kwargs are replayed through both models on that identical state; no epsilon is
reconstructed and no new integration formula is introduced. The traced final
action must match normal unobserved sampling bitwise. FullH40 driver, teacher
and student velocities, x_t, mask and time are stored. Shared-state driver
velocity replay differences are descriptive evidence rather than a new
threshold; they must be considered when interpreting the diagnostic.

Weights are written exclusively at200/1000/2500/5000. At1000/2500/5000 a separate
optimizer+RNG payload and JSON index reference the weight file by SHA, avoiding
weight duplication. RNG is represented only by safe-load-compatible primitives
and tensors, including NumPy's state as an integer list. Source, PLAN, schedule,
parameter ordering, completed/next global step, base/cache and probe lineage are
recorded. No files are overwritten; partial files without a completed valid
index cannot silently become a resume point.

Resume requires an explicitly new run ID and both `--resume-index PATH` and
`--resume-index-sha256 SHA`. The parent must match the same PLAN/source/schedule,
contain optimizer/RNG state, and have remaining scheduled steps. Models and
optimizer load first, a new Graph is captured and its numeric checks run, then
checkpoint RNG is restored immediately before continuation. LR/warmup is not
restarted: the next frozen global schedule row supplies LR. No watcher,
automatic retry or job scheduler exists. A source-changing repair needs a
separately explicit plan/attempt rather than silently relaxing resume identity.

The fresh program's checkpoint footprint is approximately9.3GB plus diagnostics,
within the frozen16GiB total artifact bound; each attempt has3550 internal
seconds and44GiB allocated-GPU bounds on GPU0. A completed Phase A is same-episode
engineering evidence, not new-episode generalization or a closed-loop result.
Full-VLA training additionally requires a newly verified differentiable Graph
input surface and an independently frozen teacher; this runner does not claim
that capability.

Focused tests use tiny CPU modules/AdamW to verify immutable checkpointing,
resume step/LR/RNG continuity, identity failures and observer restoration. They
do not run the VLA, CUDA Graphs or GPU training.
