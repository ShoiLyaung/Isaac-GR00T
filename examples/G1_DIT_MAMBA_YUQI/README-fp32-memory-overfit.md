# Fresh FP32 memory single-episode overfit

`fp32_memory_overfit.py --plan PLAN.json --output OUTPUT --wall-seconds 1750`
implements one bounded joint command: zero-update preflight, step0 evaluation,
200 optimizer steps, step200 evaluation, terminal numerical and integrity checks.
The frozen PLAN hash and worktree path are checked before model loading. The
operator exposes only GPU1; source preparation and CPU tests never launch it.

Only the original prototype `memory-initialization.pt` is accepted, including
its content hash, 1,915,440 FP32 parameters and exactly-zero output projection.
There is no warm-start/resume argument. Only episode125's existing cache is
loaded, with complete `0:16:anchor` prefixes. The processor is used for existing
physical action decoding; no new RGB data, other episodes or caches are read.
The original 200-row schedule, AdamW settings, native flow loss and 0.01 residual
penalty are retained. Base weights remain frozen/eval; no GradScaler is used.

The unchanged `FP32VisualMemory` executor covers the full differentiable memory
and readout under FP32 with autocast/Torch TF32 disabled. Native base forward
uses BF16 autocast with TF32 enabled. **Backward runs outside autocast with
ambient TF32 enabled**, matching the old formal overfit boundary. Actual flags
are recorded. This does not establish an all-IEEE FP32 backward implementation.
The zero-readout preflight requires readout gradients; an independent seed43,
std1e-3 shadow tests all six groups without changing the training initialization.

Both evaluations save 64 normalized actions, physical metrics, six hand MAEs,
and native noise/time receipts using the original seed mapping. Terminal checks
save FP32 full/step latents, residuals and 20 actions across anchors288/320,
plus raw-cache disabled parity, state summaries and final base/cache identities.
CPU FP64 numerical limits and original quality limits are unchanged. A false
numerical/quality/change gate is aggregated only after terminal evidence is
saved. Identity, nonfinite, resource or runtime errors stop immediately; partial
step logs and non-resumable checkpoint evidence are retained when training fails.

The 1750-second internal limit, 12-GiB peak allocation and 1-GiB artifact budget
are enforced. No automatic retry, pilot, confirmation or deployment is exposed.
Prior fixed-weight repair cost is cited only as preparation evidence: the new
terminal checkpoint's latency is **not measured** here. A passed result only
supports trainability on one episode already seen by base pretraining.

Focused CPU tests use toy modules and simulated autocast to check fresh-only
loading, graph/precision boundaries, state isolation, RNG ordering and terminal
evidence retention. They do not execute official CUDA kernels or an optimizer.
Run with the original Python3.10 environment and dependency overlay:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/examples/test_fp32_memory_overfit.py -q -o addopts='' -p no:cacheprovider
pre-commit run --all-files
```
