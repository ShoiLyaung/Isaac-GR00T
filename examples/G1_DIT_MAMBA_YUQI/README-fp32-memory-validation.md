# Local FP32 visual-memory repair validation

`fp32_memory_validation.py --plan ... --output ... --wall-seconds 1750` executes
one zero-update validation of the fixed step200 memory and original Yuqi17000.
The parent owns the only formal GPU1 attempt. Old production, diagnosis and
training files/results are unchanged.

`FP32VisualMemory` is a thin serial executor holding the original module. Its
full method includes the original full_sequence and learned readout and preserves
the autograd graph; step delegates to the original inference method. Inputs,
master weights and explicit caches must be FP32. Autocast and both Torch TF32
flags are locally disabled and restored even on exceptions. No SSM, sampler or
base arithmetic is rewritten and no hidden global cache is added.

For episode125 anchors288/320 (19/21 observations), BF16 full/step controls and
candidate FP32 full/step are measured. Candidates must equal same-run direct r2
FP32 calls tensor-for-tensor. Each of the 40 anchor/mode/seed groups freezes one
condition/state/mask context, captures the actual initial sampling noise, and
runs three same-seed actions: 120 saved outputs. Forty native flow calls retain
actual time/noisy-action receipts. Same-context repeat variation is reported
separately; nonzero variation does not trigger a new tiny-tolerance gate.

CPU FP64 comparisons preserve the old latent/action limits. Candidate full/step
and BF16 full→candidate step are the primary pairs, checking all three paired
repetitions for each seed. Other pairs are descriptive. Historical anchor320
outputs are compatibility evidence only; anchor288 has no old evaluation row.
State snapshot hashes and observation counts check the 19/21-step lifecycles.

Each anchor runs one original native-loss backward pass through FP32 full+readout,
without an optimizer. Six groups / 24 parameters are recorded. Forward memory
precision flags and backward ambient flags are reported separately: this proves
gradient reachability after a local FP32 forward, not all-FP32/IEEE backward or
gradient equivalence. Gradients are cleared and weight contents checked.

Clean policy cost uses only newly loaded current observations125/288 and125/320;
metadata loading may cover the global dataset, and video decoding may require
GOP reference frames. The timer includes current H2D, native VLM/VL/q, one memory
step and Euler4. Prefill uses frozen cached post-VL history with H2D and is timed
separately as memory-only work, not full RGB cold-start cost. Restore, seed,
decode and checks are outside timing. Prefix and working GPU state copies are
both charged to memory. Native blocks move the new module to CPU and release
all explicit diagnostic/state tensors; only the base remains intentionally
resident. Instrumented call audits run separately and are removed before timing.

For288 the order is native→BF16→FP32;320 reverses it. Each path has five warmups
and fifty measured calls (330 records), with finite normalized/native decoded
shapes checked. Trained-memory actions are never required to equal native.
FP32/native p95 overhead must be≤10% and extra allocated peak≤512MiB on both
anchors. BF16 relative cost and recomputed-policy repeat variation are descriptive.

Identity, shape/finite, wrapper equivalence, paired noise/context and final
integrity failures stop immediately. Numerical, gradient-reachability and cost
adoption gates are combined only after all prescribed measurements complete.
A failed adoption retains a completed technical report and stops downstream.
Existing output directories cannot be reused; there are no retries or training
fallbacks. Limits remain1750s internal/1800s external,12GiB allocated and1GiB
artifacts. Passing supports only this limited execution candidate, not broader
training quality, generalization, closed-loop or robot performance.
