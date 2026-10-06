# Pretrained causal visual memory: zero-step engineering probe

`memory_prototype.py` implements the frozen research package
`g1-pretrained-causal-mamba-memory-prototype-r1`. The parent research runner owns
the sole formal GPU attempt. This entry point does not train, optimize, install
an alternative action head, change old runners, or establish memory quality.

The original Yuqi17000 checkpoint remains frozen/eval. The new
`CausalVisualMemory` receives post-VL image features in their original 192
camera/spatial positions. Each position is an independent batch row and actual
observation time is sequence: 2048→256 projection, two LayerNorm→official Mamba2
residual layers, then a zero-initialized 256→2048 readout. The 17 nonimage tokens,
209-token layout, current q13 state and native Euler4 sampler remain intact.
New weights are initialized on CPU with seed42 and retained as FP32 master
parameters under BF16 autocast. All pretrained parameters stay frozen/eval.

`MemoryState` is explicit. Each layer uses the official FP32 allocator, including
the transposed `(192,640,4)` conv cache (stride `(2560,1,640)`) and
`(192,8,64,64)` SSM cache. Two layers occupy exactly 13,565,952 elements / 51.75 MiB.
Clones own independent storage and preserve strides. Reset clears both caches
and the observation count. `step` is inference-only and decorated with no_grad;
`full_sequence` uses the official differentiable SSD forward. Neither maintains
a hidden cache, adds attention, or advances state during action denoising.

Four bounded suffix windows use real observation indices, never future targets:
150:[13,29,45,61,77,93], 147:[13,29,45,61], 152:[0,16,32],
112:[72,88,104,120,136,152]. These are not full episode histories or confirmed
capture timestamps. The runner verifies frozen checkpoint/video/processor
identities and all 209 input IDs, camera order and grids before model outputs.

The single probe checks 64 exact historical reference outputs/metrics, all 19
observations' native/disabled/zero-readout action parity and finite decode,
cache carry/reset/deep clone/read-only snapshots, official step/full-sequence
numerical tolerance, nonzero shadow connectivity, and a 300-update synthetic
constant-feature state test. Actual `Mamba2.step` methods are patched only for a
separate call-count check (two calls per observation); a removable DiT hook
checks four sampling evaluations. Neither is present during clean timing.

Gradient checks call the original decorated sampler's `__wrapped__` body with
frozen base parameters. Loss is squared active16 sampled normalized action,
not a teacher-MSE loss that vanishes at zero recovery. Zero initialization must
have a readout gradient. An independent shadow (seed43, readout weight std1e-3)
must propagate through input projection, both official full-sequence Mambas and
readout. Inference `step` is never evidence of full SSM backward. Shadow weights
are discarded; no optimizer steps occur, and original contents are hashed.

For each stream's final observation, native and enabled complete policy paths
have five warmups and fifty measurements, alternating block order by stream.
Prefix work (H2D/VLM/VL/memory) is timed separately. Each enabled measurement
restores the same prefix outside timing and includes the current VLM, memory
update/readout and all four native denoising steps. Native blocks keep all unused
new weights/cache on CPU. Enabled memory peaks include both prefix and restored
cache copies; warmup memory peaks are not excluded. Per-input cost gates are
p95 ≤ native p95 ×1.10, additional allocated peak ≤512 MiB and total ≤44 GiB.
These are capability-expansion overhead gates, not claims of acceleration.

The runner writes small JSON/JSONL evidence plus new memory initialization only;
it never copies/saves the base checkpoint. It refuses an existing output path.
Any failed gate writes a terminal failed RESULT and exits without retry,
threshold adjustment, smaller-model fallback or training. The internal budget
is 1750 seconds and 2 GiB artifacts; the existing research runner handles the
outer 1800-second hard timeout, including terminal evidence after forced kills.
CPU tests isolate official constructor/allocator source bodies to avoid importing
Triton without a CUDA driver, and use a toy recurrence for orientation/gradient
plumbing. This is source/interface checking, not an official allocator runtime or
CUDA success claim. Only the formal GPU run can establish real kernel numerics and SSD
backward success.
