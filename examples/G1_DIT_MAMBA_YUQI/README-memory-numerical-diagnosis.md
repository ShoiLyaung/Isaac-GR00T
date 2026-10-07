# Fixed learned-memory numerical diagnosis

`memory_numerical_diagnosis.py --plan ... --output ... --wall-seconds 1750`
is the independent zero-training diagnosis of the frozen step200 checkpoint.
The parent owns the only formal GPU attempt. Existing model, prototype, training,
checkpoint and failure artifacts remain unchanged.

Only episode125 anchors 0,16,...,320 are consumed from the existing raw/post
cache. No FeatureCache over other episodes, tokenizer/processor, RGB decoding,
new dataset cache or optimizer is created. Original Yuqi17000 is strictly loaded
and held BF16/eval/frozen. New memory master weights and step caches stay FP32.
The four modes are production full/step under original BF16 autocast, and
production full/step with autocast disabled. Torch TF32 is disabled only inside
the FP32 memory context and restored afterward; the frozen local Triton source
and `TRITON_F32_DEFAULT=ieee` are verified separately.

Uninstrumented production outputs are saved first: all-time latent and final
residual. BF16 must reproduce the old maxabs 0.125 exactly and the two old RMS
statistics within 1e-8. That old threshold failure is the subject of this probe,
not a reason to suppress its FP32/action comparisons.

A separate scoped trace observes projection/norm/activation/output modules and
wraps original layer methods and SSM functions without replacing arithmetic or
return values. Direct step calls require method wrappers. Full Conv1d output
is `[slot,640,time+3]`; its right padding is removed and it is transposed before
the production SiLU call, whose input/output is `[slot,time,640]`. Captures own
CPU storage and normalize to `[observation_time,slot,flattened_feature]`. The
actual SSM x/B/C/dt and output tensors are saved, with native shapes/dtypes for
all tensor arguments. Step state gets per-observation finite/shape/dtype/stride
and numeric summaries; full exposes no explicit state and is marked unavailable.
No giant per-step SSM states are serialized. Every traced latent/residual must
be exactly equal to its untraced counterpart. Trace timing is not deployment
latency.

Each mode evaluates the same final raw/post/state/target and five seed choices:
nominal 42–45 via the old deterministic mapping, plus direct42. Native flow
receipts record actual time/noisy-action hashes. BF16 step must reproduce the
four historical nominal actions and flow losses exactly. Disabled parity is
native head.get_action on fresh cached raw features versus the unchanged cached
sampling helper, with no new VLM/RGB call. All normalized actions, conditions
and residuals are saved for independent CPU checks.

All six cross-mode latent/residual/condition/action comparisons are reported.
Positions refer to stored tensor layouts: traces/latent `[time,slot,feature]`,
actions `[batch,active_horizon,action_dim]`, conditions `[batch,token,feature]`.
Intermediate first nonzero differences and first old-threshold crossings are
separate localization observations, not new model success predicates.

The frozen strong-evidence predicate requires FP32 full/step latent RMS and
maxabs both at most 10% of BF16, plus FP32 passing the old latent limits. Otherwise
convergence is partial or unresolved. Neither result establishes a unique root
cause, mathematical reference, safe deployment of BF16-trained weights through
another precision path, or permission to reopen training. Final base/memory/cache
hashes and allocator release are recorded. Technical failures preserve a failed
terminal result without retries. Limits are 12 GiB allocated, 1 GiB new artifacts,
and the parent's outer 1800-second timeout.

The original GPU0 attempt was rejected by resource preflight before launch or
model output. A separate setup-only migration uses `CUDA_VISIBLE_DEVICES=1` and
`joint-diagnosis-gpu1-r1`; the GPU0 resolved/terminal records and source
`e9cdb5b869f56c537ae60aa1b5fd05131857d8e0` remain preserved. GPU selection and
actual device name are recorded in IDENTITY. Scientific inputs and gates are
unchanged.
