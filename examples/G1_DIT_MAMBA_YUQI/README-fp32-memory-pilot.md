# FP32 causal visual memory: one matched pilot

`fp32_memory_pilot.py {preflight,train,evaluate,cache-confirmation}` implements
one frozen experiment. Every stage requires `--plan`, `--artifact-root`,
`--output` and `--wall-seconds`; train additionally takes `--condition` and
joint evaluate takes `--cohort dev|confirmation`. Runs expose only physical GPU1,
use the original Python3.10 environment and Mamba overlay, and require clean,
committed source. There is no resume/retry or parameter-selection CLI.

The existing immutable334-observation train/dev cache is reused. Zero-update
preflight proves native/raw-cache/zero-readout parity, validates all cached
frame hashes, and checks the longest47-observation prefix for all four fresh
conditions. It freezes three identical Mamba initializations and a GRU whose
common projection/norm parameters match; GRU recurrent initialization is seed42.
The only pre-existing module change accepts GRU's explicit hidden-cache layout
in the FP32 dtype guard. Neither recurrent math nor the native VLA is changed.

Each of current-reset Mamba, same-length zero-history Mamba, GRU history and
Mamba history then executes the original500-row schedule in that order. The
original native flow objective, AdamW grouping, LR/noise/time schedule and0.01
feature penalty are unchanged. Memory forward/readout runs locally in FP32
without autocast/TF32; frozen native forward uses BF16/TF32. Backward executes
outside autocast with ambient TF32 enabled, as in the passed overfit recipe.
Actual native noise/time/preinjection-condition receipts are compared across
conditions, while different effective prefixes and condition outputs are
explicitly recorded. No compute/parameter-matching claim is made for GRU.

Every final checkpoint is saved before terminal gates, checked at the longest
training prefix, and hashed before any dev output. All five evaluation
conditions (native plus four trained branches) use256 canonical step outputs:
four episodes, sixteen anchors, four seeds. Per-condition finite normalized
H40 actions and physical activeH16x13 metrics include all six hand dimensions.
Final full/step numerical checks cover each episode's last legal anchor with
five fixed seeds; initial zero/disabled parity remains exact. Learned numerical
thresholds remain relative latent RMS0.02/maxabs0.1 and active-action RMS0.002/
maxabs0.02, computed on saved tensors with CPU FP64 reduction.

Joint evaluation completes both quality and native-versus-carry currentRGB cost
measurements before aggregating false scientific gates. Costs include current
H2D/VLM/VL/q/memory/Euler4 with five warmups and fifty measurements on each
last anchor, alternating order. Seed, decode, prefill and state restore are
outside the host timer; both GPU state copies count. Native timing removes the
unused memory module and caches from GPU. The two-Mamba/four-DiT call audit is
separate and leaves no instrumentation in measured calls. Exact repeat actions,
base/cache/checkpoint integrity, resource limits and all original gates apply.

Confirmation access is fail-closed behind the complete dev result. Only then
may the four predeclared119-observation confirmation caches be generated,
followed by one joint evaluation. Its gates are identical. All220 episodes were
seen by base pretraining: this is adapter-held-out, pretrained-domain engineering
evidence, not base-unseen generalization, closed-loop success or proof that
Mamba is generally superior. Old failures remain unchanged. Every failed gate
stops downstream work; the code does not automatically retry or tune.

Each stage has a30-minute wall limit, each training condition a2-hour limit,
44GiB peak allocated memory and a total8GiB new-artifact budget. Only new module
and optimizer checkpoints are saved. CPU tests cover cache layout/precision,
real GRU full/step math, causal controls, gradient graphs, pairing, quality
predicates and fail-closed prerequisites; they do not execute official CUDA
kernels or optimizer steps. Run focused tests with the original venv/overlay
and `CUDA_VISIBLE_DEVICES=''`, then `pre-commit run --all-files` before commit.
