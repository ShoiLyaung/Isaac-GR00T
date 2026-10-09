# CPU weights with an explicitly assigned CUDA driver (r3)

This isolated repair starts at `04f17ffa0674db172243eb5f7de7a8a0b69aca8e`. Both earlier failed attempts and their source remain intact. The r2 full CPU attempt passed B0 construction and semantic identity, then failed before any forward/update while importing the official Mamba2 implementation: Triton's import-time `Autotuner` requests `driver.active.get_benchmarker()`, which rejects an environment with no active driver.

The correction changes only the fixture execution boundary. It keeps the real `BudgetedActionMamba` → official `Mamba2` constructors and the existing strict core key/shape/dtype/hash checks. There is no substitute Mamba class, copied architecture or new model format.

The new fixture PLAN must include:

```json
{"physical_gpu": 1, "fixture_execution": "cpu_weights_cuda_driver_only"}
```

Run the existing fixture CLI with `CUDA_VISIBLE_DEVICES='1'`; exactly one CUDA device must be visible. GPU1 is explicitly assigned by the Supervisor solely for the official import/driver context. The entrypoint never calls GPU neural forward, observation encoding or an optimizer. It does not move B0 or student tensors onto CUDA.

B0 still loads on CPU using the native single-load `use_flash_attention=False` override and must report SDPA. The complete frozen BF16 B0 semantic hash remains `2bff6e15f09adc0ee05c24d2445d347113d802043e41227e4eb6b6ce150a83ad`. The fixture checks B0, official core and composite student parameters **and buffers** are all on CPU. `torch.cuda.memory_allocated()` must equal zero before/after construction and at budget boundaries; a nonzero value fails and is recorded, rather than waived. CUDA driver/context overhead outside PyTorch tensor accounting is not described as zero device memory.

Unlike r1/r2, `cuda_initialized` is recorded truthfully and may be true. `RESULT.json` also records `fixture_execution`, `physical_gpu`, `torch_cuda_memory_allocated_bytes` and B0 residency. The fixture index records composite parameter/buffer CPU residency while retaining `fixture_only=true`, zero added updates, original core-parent/B0 identities and full/core/periphery semantic hashes. GPU policy loading, its default attention backend, action sampling and all training/Mamba mathematics are unchanged.

Focused tests run in a process launched with hidden GPUs. Pure mocks exercise permitted initialized/uninitialized driver states, explicit schema/device rejection, nonzero PyTorch allocation rejection and non-CPU parameter/buffer rejection. Existing tiny real CPU SDPA reload and ordinary-loader scope tests remain; fake driver observations are unit-test fixtures, not claims of real GPU execution. The real driver-only complete fixture remains the Supervisor's separately frozen r3 attempt.
