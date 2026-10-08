# Budgeted attention-free action student: first source milestone

`budgeted_action_runner.py preflight|overfit --plan PLAN --artifact-root ROOT
--output OUTPUT --wall-seconds 1750` implements the two initially authorized
stages. Preflight uses the single canonical `preflight-execution-r1/output`
root; the earlier resolved-only `preflight-r1` remains unlaunched and immutable.
Overfit keeps `overfit-r1/output`. The frozen plan, source/checkpoint/cache/kernel hashes and worktree are
checked before model outputs. Only physical GPU1 is permitted. The train and
evaluate CLI names explicitly raise `NotImplementedError` before creating an
output or loading a model: the later two1000-step runs and joint dev evaluator
are **not implemented in this commit**. A passing first milestone cannot be
reported as completion of those stages.

The student replaces the entire `action_head.model` subtree, preserving native
external VLM, four-layer VL postprocessor, q/action encoders, action decoder and
Euler4 implementation. Eight1536-wide blocks copy AdaLN/FFN sections from
teacher layers3/7/11/15/19/23/27/31 plus native time/output projections. Norms
without affine parameters retain configuration, not fictitious copied weights.
All copied teacher values are promoted from BF16 to FP32 and verified exactly;
all student parameters are trainable even though the source teacher is frozen.
New mixers and adapter use the isolated CPU seed42 stream, never previous
Mamba or overfit parameters.

Each mixer executes its1536→512 projection once, two weight-shared official
Mamba2 forwards on41 tokens (one reversed), and its512→1536 projection once.
Mamba state64, conv4, expand1, head64, one group and chunk16 are fixed; no
attention or persistent state exists inside the student. The once-per-request
condition adapter retains the old fixed209-token permutation and bottleneck
math. `ActionCondition` is an explicit object supplied through a temporary
`ConditionedActionRequest`. The shared student owns no condition cache; binding
is restored in `finally`, and all four native Euler calls share the one object.

Preflight verifies12-point raw/cache parity,64 historical native actions and
flow values,56 copied tensors,230,785,600 student parameters, independent
condition/cache lifetime and zero-step gradient flow through the frozen decoder.
Actual hooks audit64 official Mamba forwards, one condition encoder and41-token
mixers, plus every action-core Linear invocation. Hooks are removed before
clean cost timing. Static expected linear MAC is29,579,837,440 per policy;
MAC excludes scan/convolution/normalization/QK-AV arithmetic and is not FLOPs.

Four current-RGB inputs use alternating native/student5warm+50 measured host
wall calls, with only the measured core on GPU. Current transfer/VLM/VL/q/
condition/Euler4 are included; seed/decode/processor/checks are outside. The
additional cached-raw-VL task forward/backward probe uses both cores' FP32
masters with BF16 autocast, frozen outer modules and **eval mode (no dropout)**,
5warm+20 measurements. It includes native postprocessing/encoders/decoder and
student condition construction, but no teacher replay, VLM or optimizer. The timer calls the original native head.forward directly using fresh cached
inputs, without capture hooks, evidence hashes, teacher replay or internal RNG
reseed. This is not a full training-step throughput benchmark. Per-call CUDA peak resets
are tracked separately from the stage high-water recorded at budget boundaries.
All cost measurements finish before aggregating a reportable false cost gate;
identity, nonfinite or resource errors stop immediately.

Only a passing preflight permits fresh initialization overfit on episode125.
One original native head.forward produces the exact noisy-action/time draw;
scoped hooks retain its original core kwargs and differentiable decoder output.
After hooks are removed, frozen B0 replays those same detached core inputs and
the native decoder. KD applies the original active16×13 mask before squaring,
uses FP32 arithmetic and denominator mask.sum()+1e-6, with detached teacher.
Native task loss is the original result.loss, not a copied implementation.
Only student trains; the surrounding head stays eval to disable state dropout.
The200 frozen steps use original AdamW/LR/clip settings with no GradScaler or
resume. Only student weights are saved; no VLA/optimizer copy is stored.

Before/after64-anchor-seed outputs report velocity KD, real-target native flow
and physical errors, and same-noise student/teacher sampled-action differences.
The primary imitation gate uses **physical combined MAE after separately decoding
student and teacher with the same state**; normalized action MAE is descriptive.
A30% KD reduction and10% physical teacher-action difference reduction, finite
reachable gradients/parameter changes, condition sensitivity and unchanged
base/cache are required. Overfit weights are never pilot initialization.

The stage wall cap is30minutes, allocated-memory cap44GiB and whole experiment
artifact cap12GiB. Each stage has one canonical output directory and no automatic
retry. Confirmation is outside this experiment. Passing engineering gates does
not prove equal-capacity architecture superiority, unseen-data generalization
or closed-loop success.

CPU tests use a test-only shape-compatible Mamba double; there is no production
fallback. Tests cover real projection/FFN dimensions and migration ownership,
shared bidirectional projection counts, condition permutation/gradient, native
masked KD and exact teacher replay, decoder-to-student gradients, request cleanup
and unavailable-stage guards. Official CUDA kernels are exercised only by the
operator's separately resolved preflight after CPU tests, pre-commit and review.
