# Matched mixed-distribution velocity KD

This experiment continues one explicitly frozen KD-then-task terminal-20k student
into two variants, `data_line_mixed` and `teacher_trajectory_mixed`. Both import the
same student weights, all 158 AdamW states and saved RNG, then perform 30,000
additional active-H16 teacher-velocity KD updates without resetting moments.
No ground-truth flow objective is used for backward.

```text
action_transfer_trajectory_kd.py train --condition VARIANT \
  --plan PLAN --plan-sha256 SHA --artifact-root ROOT --output ROOT/RUN/output \
  --wall-seconds 17950
```

Fresh run IDs come from the plan. Manual recovery requires a new `--run-id` plus
`--resume-index` and `--resume-index-sha256`. Existing attempts and source trees
are immutable. The runner does not retry or choose a checkpoint automatically.

## Exactly what changes

The frozen suffix schedule has identical B4 samples, forward seeds and low-LR
cosine values for both variants. Odd global steps use the original native BF16
continuous data-line KD in both variants. Even steps use one of four inference
buckets, 0/250/500/750, exactly 3,750 times each.

Both discrete variants first execute the original frozen teacher's four-Euler
sampler. Thin Trace delegates clone its actual pre-step FP32 states, core kwargs
and velocities. Its initial full epsilon is taken directly from bucket 0. No
alternative integrator or teacher noise/time override is used during capture.

- `data_line_mixed` converts the cached action to FP32 without changing its values
  and calls the original head directly. A narrowly scoped override supplies that
  captured FP32 epsilon and an FP32 time to the head's exactly one noise and time
  call. Original interpolation, encoding, integer bucket calculation and decoding
  remain unchanged.
- `teacher_trajectory_mixed` queries the student on the captured teacher kwargs
  and actual pre-step state at the selected bucket.

Each variant queries the frozen teacher again on its own selected state and uses
the existing B4 masked velocity KD reduction. The mask supervises only the original
16x13 active region. Tail supervision, architecture, H40, four-step sampler and
student-core precision are unchanged. Teacher targets are detached.

Original native continuous action/noise/time are BF16; for example BF16
`0.75 * 1000` rounds to 752. Both new discrete variants explicitly use inference
FP32 epsilon/time/x_t so bucket 750 remains 750. This is not a claim of bitwise
identity with the old BF16 discrete data-line path. Bucket-0 state sources match
exactly at the same weights; only 37.5% of suffix updates differ in state source
between the two mixed variants.

The discrete teacher rollout, selected teacher query and student core-call counts
match. The data-line branch additionally performs native head preprocessing and
action encoding, so peripheral FLOPs are not claimed equal. Actual step wall times
are recorded separately.

## Restore and execution gates

The imported parent must reproduce its complete 512-output development evidence
exactly. Before any update, common and bucket-0/750 branches must pass the frozen
eager/Graph forward-loss and all-28-group gradient tolerances. Same-weight bucket-0
inputs, teacher targets and student outputs must match across state-source arms.
Teacher Trace capture must reproduce the uninstrumented native sampler exactly.

These gates use zero optimizer updates. Parent RNG is restored after capture and
all gates. The first real update advances AdamW from 10,000 to 10,001; then the same
forward/gradient gate is repeated at the updated weights without another optimizer
step, and that real update's RNG is restored before continuing. Module overrides
always restore on exceptions and enforce shape, dtype, device and call count.

The student remains FP32 master with BF16 autocast, eval mode and gradients enabled;
VLA/vision/native decoder and teacher remain frozen. Backward runs outside AMP.
No effective dropout, new precision shortcut, tail loss or sampler change is added.

## Evidence and final comparison

Every step records the branch/bucket, common epsilon/time/q/Z/target/mask hashes,
selected x_t and teacher-target hashes, loss, gradients, actual optimizer counters
and wall time. Large teacher kwargs are temporary and are not written every step.
The unchanged B1 native evaluator produces dev512 every 5k and train32 every 10k.
Weights and optimizer/RNG checkpoints are saved at global 30k/40k/50k.

`evaluate` is CPU-only. It reconstructs both complete parent-prefix/new-suffix log
chains, verifies shared draws and teacher trajectories, and compares fixed final50k
outputs. New variant terminal overrides require paths and SHAs together. The two
external task-control paths are fixed in the plan; their completed terminal hashes
must be passed with `--task-only-control-sha256` and
`--distill-then-task-control-sha256` after those results exist.

The primary comparison retains flow/combined-MAE 2% and six-of-eight-episode gates,
plus the existing native-teacher retention gate. Teacher-action physical MAE is
reported alongside ground-truth error; improving one does not imply improving the
other. All three metrics include eight paired episode differences and 10,000
paired episode-bootstrap intervals for mean differences and relative gains using
seed 20261009. Relative gains use ratios of resampled episode-equal means, not
means of per-episode percentages. Nonpositive control denominators are explicitly
undefined, and no bootstrap draws are silently dropped. Sampling seeds are not
independent training repetitions, and this development analysis is not a new
confirmation claim. Intermediate quality never stops the frozen training budget.
