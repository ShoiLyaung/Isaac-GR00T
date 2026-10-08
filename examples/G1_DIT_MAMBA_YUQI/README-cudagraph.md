# Fixed-weight action-core CUDA Graph comparison

Run `budgeted_action_cudagraph_probe.py --plan PLAN --artifact-root ROOT --output
ROOT/joint-graph-r1/output --wall-seconds 1750` only after the frozen input/source
identity and CPU checks pass. Existing model, sampler, task objective, kernel
and prior evidence files are unchanged. This run has no optimizer, checkpoint
updates, new dependencies, fusion or compilation path.

Both frozen B0 and the original untrained230,785,600-parameter student are tested
under production_eager, bridge_eager and graph. Production preserves default
AMP cache behavior. A new tensor-only wrapper gives bridge and graph identical
BF16 autocast/cache_enabled=False action-core arithmetic. Native inference
weights remain BF16, student inference masters FP32; both F+B cores use FP32.
VLM, VL/q, action encoding/decoding, external random draws and native Euler4 stay
eager. No sampler is copied or torch.randn patched.

The inference surface takes dynamic hidden states, VL features, time and masks,
plus the student's observation-local modulation. Condition encoding remains
once per observation outside the graph. A zero buffer supplies the unused code
field; only the actual modulation participates in the original core's math.
One manually captured forward graph runs four times per native policy call;
all input copy/replay overhead is inside the complete policy timer. Its output
is borrowed until the next replay and consumed immediately by the eager decoder
on the same stream. Persisted action evidence always moves to separately owned
CPU storage. Static buffers are refreshed and checked under an A-B-A sequence
with all four original time buckets0/250/500/750.

F+B uses torch.cuda.make_graphed_callables on a new wrapper only. Its student
surface includes condition_adapter, preserving its parameter gradients. The
factory runs outside autocast; only wrapper.forward opens BF16/cacheFalse.
The original task head.forward, random/time generation, mask-before-square loss
and decoder remain outside capture. The dispatcher adapts the ignored native
intermediate-list ABI to `(final,[final])`, never altering the final tensor.
Production/bridge/graph outputs and gradients independently validate that ABI.
The factory's allow_unused_input=True preserves None gradients, which are
checked per parameter instead of being replaced by zero tensors.

There is one inference capture and one factory F+B capture per model. The
native path order is production/bridge/graph; student is graph/bridge/production.
Each path consumes the four frozen inputs in order. Numeric eager calls may
coexist with graph pools but are never timed. Before eager cost measurement,
all graphs, wrappers, static output/gradient references and pools are released.
Captured parameter storage is checked and never moved/cast until release.
Each F+B pass first clears all parameter gradients to None. Small gradient
sample fingerprints are recorded outside timing; exact repeat is descriptive,
not a new threshold beyond the frozen gradient gates.

Inference first preserves eight prior production actions bitwise and obtains
20 production/bridge references per model. Bridge/production exactness and
graph-versus-both active16 action RMS/maxabs are explicit. Each model/path/input
uses5warm+50 clean policy calls, with no diagnostic hooks/profiler. F+B uses
5warm+20 cached-raw task forward/backward calls on150/336. The native target,
mask, actual noise/time/state/condition receipts are compared outside timing.
Core eval disables dropout, gradients remain enabled for F+B, backward runs
outside autocast, and no optimizer or teacher replay is performed.

For each model's four inputs and seeds42/43, three F+B paths are compared in all
three pairs. One case temporarily retains three detached CPU gradient copies,
then deletes them before the next case. No full-gradient files are written.
CPU FP64 per-parameter squared norms, dot/error/maxabs and eight fixed sample
coordinates support independent auditing. None-pattern/shape/finite checks,
global and every group relativeL2<=.02/cosine>=.999, and the1e-12 zero-reference
rule prevent a disconnected small condition branch being hidden by global norms.
Native has32 block groups plus three time/output groups; student has28 groups.
Loss tolerance is atol1e-6+rtol1e-3. False finite numeric gates remain reportable
and close adoption; capture/runtime/nonfinite/resource failures stop immediately
with no fallback or automatic retry.

Output layout: `inference-{native,student}` contains20-per-path numerical action
references, A-B-A evidence,24 clean timing/action files across all paths/inputs,
capture setup and break-even invocation estimates. `fb-{native,student}` stores
three timing series, eight numeric cases with three gradient-statistic files
each, native loss and complete receipt identities. Top-level RESULT reports
per-model graph versus production/bridge speed gates and student_graph versus
native_graph policy/F+B/memory gates separately. Graph reserved pools are
reported separately from allocated memory. No outcome automatically opens
prior distillation training or changes its frozen acceptance conditions.

Hard bounds are1750seconds,2GiB new artifacts and44GiB allocated GPU memory on
GPU0. Historical plans and unlaunched identities remain untouched. CPU tests
cover tensor ABI, condition gradient, wrapper ownership, dynamic copy buffers,
request cleanup and group-level gradient statistics. Actual CUDA graph/capture
compatibility is established only by the separately authorized formal run.
