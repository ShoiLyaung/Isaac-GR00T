# Graph update compatibility and one fresh KD overfit

`budgeted_graph_distillation_runner.py compatibility|overfit --plan PLAN
--artifact-root ROOT --output ROOT/{compatibility-r1|overfit-r1}/output
--wall-seconds {850|1750}` implements only the frozen three-update compatibility
and200-step episode125 velocity-distillation stages. No existing model, graph
adapter, sampler, native loss or prior evidence file is modified. GPU0,
44GiB allocated memory and2GiB total new artifacts are hard bounds. There is no
retry, resume, pilot, dev or confirmation evaluator.

Both stages strictly reload the same original FP32 student initialization and
frozen BF16 B0 teacher. The surrounding VLA stays frozen/eval. Student parameters
are trainable but **the entire student remains eval**: its configured Dropout
p=.2 is preserved while effective random dropout is zero. This is explicitly
different from the old unexecuted student.train recipe. One existing
TensorCore(student=True, backward=True)/BackwardGraph captures student condition
and action core F+B per stage. AdamW, teacher and the original head.forward
noise/time/mask/target logic remain outside capture.

The clean KD call temporarily delegates the native core, action encoder,
decoder and sample_time. Delegates retain actual tensor references without
hooks, hashes or CPU copies. The original head produces one actual noisy
trajectory and time. Student decoder output is retained before all delegates
restore; only then the frozen teacher replays those exact detached core kwargs
and the original decoder. Its velocity is detached. The existing
masked_velocity_kd applies the same active16×13 mask before FP32 squaring and
original denominator. Native task loss is reported but never optimized here.
Required native/KD finite checks remain part of the original algorithms.

Each optimizer step clears gradients and asserts None. Cache access and seed
setup precede the synchronized timer; cached H2D, native postprocessing,
student Graph, teacher replay, KD backward, clipping and AdamW are timed. CPU
hashes, receipts and group summaries follow synchronization outside the timer.
The preclip total norm comes from clip_grad_norm_; per-group statistics are
explicitly postclip. Optimizer-return count increments immediately after step
returns and is preserved even if later synchronization/evidence/logging fails.
This wall interval is not RGB/VLM training throughput or an online policy rate.

Compatibility consumes exactly the first three old schedule rows, using one
core trajectory whose actual updates consume Graph gradients. At updates0/1/3,
anchors0/320 × direct seeds42/43 compare independent eager KD against Graph KD:
actual receipts, native and KD losses, every parameter None/shape/finite pattern,
all28 groups and global FP64 relativeL2/cosine gates. Only sufficient gradient
statistics and small samples are written. The original Euler4 sampler also
compares eager and Graph-forward actions. Reusing the F+B graph in no_grad
recomputes the condition four times; this is a **numerical-only path**, with no
claim about its inference speed. Current weights must match eager, retain the
same storage, change content, and produce at least one changed velocity and
sampled action by update3 relative to update0. These updated weights and optimizer
state are discarded and never checkpointed.

Overfit requires a passed matching compatibility result from the same source
commit and PLAN before model creation. It reloads the original initialization,
creates a fresh optimizer and one new Graph, and performs exactly200 frozen KD
updates. All callbacks and checks preserve parameter storage. At step0/200,
quality uses the original eager once-per-observation condition and64 fixed
anchor/seed pairs. Physical teacher-action MAE is the primary action-imitation
metric; normalized teacher-action MAE is descriptive. It also reports real-target
flow and combined/arm/hand errors. The final four compatibility cases recheck
Graph/eager numerical agreement at learned weights. Zero modulation and camera
position interventions verify sensitivity without changing model parameters.

`training_steps.jsonl` retains step, teacher_velocity_kd, native_loss, lr,
gradient_norm_before_clip, gradient_groups and elapsed_step_seconds, with
clean_step_wall_seconds, actual receipts, peaks and storage assertions.
`update-{0,1,3}`/`terminal` contain four-case action/velocity tensors and gradient
sufficient-statistic JSON. `step0`/`step200` contain details/RESULT plus:
student_actions and teacher_actions [64,40,13]; student_physical,
teacher_physical and physical_targets [64,16,13]; student_velocity,
teacher_velocity and action_masks [64,40,13]. NORMALIZATION.json records the
absolute/minmax13D vectors/order for independent CPU auditing. Only the final
student-step200.pt is saved; no optimizer or full VLA weights are written.

The original gates remain: at least30% fixed-evaluation KD reduction and10%
physical teacher-action difference reduction, all28 reachable group updates,
condition sensitivity, final Graph/eager numerical checks and unchanged
teacher/base/cache. False quality gates still retain final evidence and stop.
Passing this experiment proves at most same-episode transfer and Graph update
compatibility, not task success or generalization, and does not launch pilot.

Focused CPU tests use small modules to verify actual-input teacher replay,
independent eager KD agreement, gradients through the frozen decoder, delegate
restoration on failure, parameter-content versus storage identity, effective
dropout, fresh-stage prerequisites and post-step exception accounting. Actual
CUDA updates are performed only by the separately resolved authorized stages.
