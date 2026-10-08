# Explicit matched continuation from terminal 20k to fixed 50k

`action_transfer_continuation.py` extends the two frozen matched conditions without
changing model architecture, batch size, native objective, teacher, processor,
precision or the existing update/evaluation mathematics. It does not select a
best development checkpoint. Each condition imports its fixed terminal-20,000
student, all 158 AdamW states and saved RNG.

```text
action_transfer_continuation.py train --condition task_only|distill_then_task \
  --plan PLAN --plan-sha256 SHA --artifact-root ROOT --output ROOT/RUN/output \
  --wall-seconds 14350
```

Fresh run IDs come from the plan. Explicit recovery within this new contract uses
`--run-id NEW_ATTEMPT --resume-index PATH --resume-index-sha256 SHA`; it never
rewrites an existing attempt or silently accepts a latest checkpoint.

## Restoration before any update

The new plan binds both historical terminal results, parent checkpoint indexes,
weight/optimizer files, original source/plan/schedule/log identities and the same
full train/dev cache. The ordered optimizer parameter names and phase-2 counters
are checked: 158 states at step 10,000 on the initial import. Instantiating the
AdamW object is followed by loading its saved state; no moments or phase schedule
are reset.

After the new B4 graph is captured, the runner reproduces the full 512-output B1
development evaluation of the imported parent. All eight stored arrays, metrics,
input receipts and normalization must match exactly. There is no tolerance
fallback. The restored B4 eager/graph task and KD losses and all 28 gradient groups
then use the prior frozen numerical tolerances, without optimizer updates. Only
after these checks is the saved parent RNG restored, immediately before the first
real continuation update.

Historical interface preflight and dev-0 remain immutable evidence. They do not
substitute for the new restoration gate. An explicit later resume repeats the
same checks against that checkpoint's corresponding development evaluation.

## Optimization and evidence

Both conditions use native task loss throughout global steps 20,001–50,000. Every
step still computes the same-state frozen teacher velocity for the matched KD
diagnostic. The old update body is reused; only its condition label identifies the
originating matched condition. Batch4, FP32 masters, BF16 autocast, disabled
effective Dropout, clipping and optimizer hyperparameters remain fixed.

LR is `1e-6 + 9e-6 * (1 + cos(pi*u))/2`, where
`u=(global_step-20001)/29999`. It starts at 1e-5 and ends at 1e-6, with no warmup or
optimizer reset. The first new update advances AdamW to 10,001; the final update
advances it to 40,000. The 50,000-row schedule preserves the original first 20,000
rows and continues the same shuffled sample stream from offset 80,000.

Full dev512 is recorded every 5,000 additional steps. The existing train32 probe
runs at global 30k/40k/50k. Immutable weights, optimizer/RNG and indexes are saved
at those same three steps. Current-attempt gradient witnesses start false and
are used for `changes_since_import`; cumulative parent witnesses are stored
separately so old gradients cannot hide a dead continuation branch.

`evaluate` performs a CPU-only joint summary. It verifies old-prefix and new-suffix
checkpoint/log chains, all 50k batches/LRs/actual input receipts and fixed final
50k outputs. It reports 30k new updates/120k new samples and 50k cumulative
updates/200k cumulative samples per condition. Explicit resumed terminal paths
can be supplied with `--task-only-terminal`/`--distill-then-task-terminal` and their
SHA arguments. Intermediate development quality never stops training or chooses
a checkpoint. Final quality gates are reported separately from technical validity.

No GPU jobs, simulator runs, new confirmation episodes or automatic retries are
launched by implementation tests. Focused CPU checks cover all 30k LR values,
AdamW moment/counter preservation, exact development reproduction, restoration
ordering and immutable cross-contract log lineage.
