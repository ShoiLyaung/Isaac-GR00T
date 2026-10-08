# Fixed-weight budgeted-action execution diagnosis

`budgeted_action_execution_probe.py --plan PLAN --artifact-root ROOT --output
ROOT/joint-diagnosis-gpu0-r1/output --wall-seconds 1750` implements one bounded joint
run. Only the four newly added diagnostic/test/documentation files may differ
from the frozen source base. Existing student architecture, official kernels,
native sampler, clean task-loss helper and all old results remain untouched.
The original B0 teacher and failed-cost student's initialization are strictly
loaded; there is no optimizer, training, fusion, CUDA Graph or retry path.

Three inference modes share the existing prepared three-camera/q13 input and
native H40/active16/Euler4 execution:

* native: original frozen BF16 B0 action core;
* student_original: original FP32 master values, BF16 autocast, eval/no-grad
  calls, with original requires_grad flags retained;
* student_bf16_storage: an independent derived copy whose floating parameters
  and buffers use BF16, except exactly24 A_log/D/dt_bias parameters preserving
  their original FP32 values. This is a precision intervention, not a claim of
  bitwise-identical checkpoint values. Original and derived semantic hashes and
  complete dtype inventories are saved and rechecked at the end.

Eight old seed42 native/original-student actions must reproduce exactly before
new evidence. Four fixed inputs and direct seeds42–46 then provide20 paired
original/derived outputs, native flow losses and physical target metrics for
all three modes. Full normalized actions are saved; active16 numerical RMS and
maxabs use the unchanged0.002/0.02 limits. A false precision gate is reported
and forbids adoption, while the remaining preregistered diagnosis still runs.

Clean whole-policy measurement uses four inputs, three modes and5warm+50 calls
per mode, rotating the order by input index. Only the measured action core is on
GPU. Seeds, checks and hashes remain outside synchronized host timers. Current
H2D, VLM, VL/q, student condition and original Euler4 sampling stay inside.
Every same-mode seed42 repeat must match bitwise. Allocated/reserved peaks and
all action tensors are saved; prior timing numbers are descriptive references,
never exact-reproduction gates.

A separate5warm+10 segmented pass per input/mode uses the original
prepare_input/backbone/_encode_features/sample_packet functions and explicit
student condition binding. H2D, VLM, VL/q, condition and Euler4 have sequential,
non-nested host/CUDA-event ranges. Native condition is defined as exactly zero,
without an empty synchronized range. Segmented actions must match their own
clean actions. Both p50 and p95 perturbations must be within5% before stage
mean/whole-mean shares can be interpreted for that mode/input; otherwise they
are explicitly qualified. Never add stage p95s or call event intervals pure
kernel-busy time.

The native/original-student forward+backward comparison retains the prior clean
cached-raw-VL task-loss helper. Both cores use FP32 masters/AMPBF16, eval mode,
gradients enabled and5warm+20 calls. It includes native VL/q/action/decoder and
student condition, but no VLM/RGB, teacher replay, optimizer, hooks, hashing or
reseed inside the timer. It is not a full training-step throughput benchmark.

Finally, input150/336 receives three policy traces and two F+B traces, exactly
one profiled invocation each. Temporary record_function scopes mark VLM,
VL/q, condition, core, AdaLN/FFN, student down/up and Mamba projections/conv/RMS,
and the official namespace's mamba_chunk_scan_combined call. Native attention
ranges are included. F+B has separate forward/backward ranges and no module
backward hooks. All patches and hooks are removed even on exceptions. Profiled
and subsequent uninstrumented outputs must equal the mode's reference; F+B's
comparand is the native loss scalar, with finite gradients rather than an
unclaimed gradient-equality test.

Traces use record_shapes=True, profile_memory=False, with_stack=False. Summaries
retain CPU self/inclusive time, device self/inclusive time where available,
CUDA device-event counts and names, cast/copy operations, and observed module
input/parameter dtypes. CUDA device events include copies/memsets, not just
arithmetic kernels. Nested ranges overlap: do not sum profiler intervals or
subtract them from clean latency. Profiler references are released before the
next mode is activated.

The result separately reports technical completeness, numerical preservation,
derived-versus-original10% speed improvement and the original native-relative
speed/memory gate. **No result opens the old training gate.** The student remains
untrained, the original F+B problem is not repaired by an inference cast, and
this run establishes no task-quality or generalization gain. Wall time is
bounded to1750seconds internally, new artifacts to2GiB and allocated GPU memory
to44GiB. Source, original initialization and base/cache identities are rechecked.

Focused CPU tests cover storage-cast ownership and sensitive parameters,
sequential event boundaries (including native zero-condition), request cache
restoration, profiler namespace/hook cleanup on success and exception, accurate
event labels, fixed order, timing aggregation and closed training decisions.
No CPU test invokes the production model or CUDA kernels.

The frozen GPU0 execution identity replaces a GPU1 launch blocked before Popen.
Old PLAN/resolved/terminal evidence remains untouched. Math and measurement
contracts are unchanged; timing comparisons use contemporaneous GPU0 modes,
not historical GPU1 latency. The eight old actions must still match bitwise.
