# Frozen memory training: cache / preflight / overfit handoff

This first implementation exposes only the frozen `cache`, `preflight`, and
`overfit` commands in `memory_training.py`. The parent research runner owns all
GPU and optimizer execution. Pilot `train` and joint `evaluate` commands remain
unimplemented until the overfit continuation gate and a separate source handoff;
this entry point cannot generate confirmation features or adapter outputs.

The runner verifies the frozen PLAN, both schedules, EVALUATION_CONTRACT,
NUMERICAL_CONTRACT, prototype result/initialization and original Yuqi17000
identities. Source imports must come from the clean training worktree; the old
Python 3.10 / torch 2.7.1 / Mamba 2.3.2.post1 overlay is retained. Confirmation
source files may be hashed as planned, but are never decoded by these commands.

`cache --cohort train_dev` saves only the 334 legal grid observations from train8
and dev4. Raw VLM features are cloned before `_encode_features` mutates its
container. Separate post-VL tensors feed the adapter. Original masks, prepared
normalized state/action, physical decode inputs, and tensor/file hashes are
preserved. Every native flow call receives fresh raw and action containers,
executes native VL postprocessing once, and can never overwrite the cache.

The original native head.forward executes unchanged. A scoped with_kwargs
pre-hook changes only DiT `encoder_hidden_states` to the actual post-Z plus the
image residual. Its returned `backbone_features` remains preinjection Z;
effective condition identity is recorded separately. Scoped wrappers capture
actual `sample_time`, discrete time and noisy action identities. RNG resets
Python/NumPy/CPU Torch/CUDA after memory preparation and before the native head.
The native returned FP32 `loss`, including `mask.sum()+1e-6`, is used directly
for backward, with the common `0.01 * mean(image_residual.float()**2)` penalty.
The frozen base stays eval, so native state dropout is disabled; this differs
from the historical full-vision fine-tune training mode.

Full-sequence training uses the complete grid prefix from observation zero with
no TBPTT or persistent graph. Deployment evaluation uses step recurrence once
per observation and reuses that residual across flow and the four action seeds.
current-reset still consumes the current frame. zero-history zeros past features
before input projection, preserving prefix length and biased null transitions.
All Mamba conditions start from the saved prototype initialization. The GRU
control uses two d256 LayerNorm→GRUCell→residual layers, copies common projection
and norm values, and initializes recurrence independently with seed42. Its FP32
hidden cache is 0.375 MiB versus the Mamba cache's 51.75 MiB; parameter counts and
actual state layouts are reported rather than called compute-matched.

The zero-optimizer preflight checks native/full-model versus raw-cache and zero
hook flow identity (loss, action_loss, mask, state, post-Z), exact sampled action
parity, immutable cache, and no base gradients. All four controls test the
longest train/dev prefix, official full/step numerical tolerances with an
independent seed43/std1e-3 readout shadow, and native flow gradients. Zero
initialization requires readout gradients; the nonzero shadow requires finite
nonzero reachable group witnesses. Current-reset's length-one A_log is allowed
to be unreachable. This phase measures actual allocated GPU peak before training.

`overfit` requires the passing cache and preflight. It starts fresh from the
prototype and executes exactly the 200 frozen episode125 steps, AdamW settings,
LRs, seeds and clip threshold. Actual noise/time receipts, prefixes, gradient
witnesses and losses are retained per step. The branch is evaluated at step0
and step200 on the frozen 16 anchors × four seeds with the step path. Continuing
requires ≥10% native-flow improvement, non-increasing combined MAE, arm/hand
regression ≤2%, parameter changes and gradient witnesses in reachable groups,
base/cache integrity, disabled-native parity and learned full/step tolerance.
Per-hand-dimension errors are reported without inventing a new passing gate.
Overfit weights are never a pilot initialization.

Existing output directories are refused. Exceptions preserve logs and a failed
terminal result; interrupted optimizer loops additionally save partial module
and optimizer state for diagnosis, with no authorized resume. There is no retry,
scheduler or threshold fallback. Each first-stage command has an internal
1750-second limit and the research runner's outer 1800-second timeout; total
new artifact limit is 8 GiB and allocated GPU limit is 44 GiB. Only new-module
weights and optimizer state are saved, never base weights.

CPU tests execute the actual native flow method bodies against small CPU
collaborators to verify loss/seed/hook behavior without copying the algorithm.
They also cover null-history semantics, GRU state/common initialization/BPTT,
cache isolation, shared frozen schedules, overfit gate aggregation and failure
retention. These tests do not establish CUDA SSD backward, memory budgets,
learned numerical equivalence or any learning/generalization outcome.
