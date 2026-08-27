# G1 all-LeRobot Codec branch contract

Status: `C2_FIXTURE_CANARY_AUTHORIZED_IMPLEMENTED_NOT_YET_EXECUTED`

This short-lived branch starts from
`feat/groot-n17-all-lerobot-20k-r1@dd86aa8deb7c5ee314ca0cf082ab9468b0b61119`.
It preserves the five-source training baseline and does not inherit authority
to generate a real codec fixture, use a GPU, train a Codec model, or run an
evaluator.

## Native contract that must not change

- embodiment: `new_embodiment -> 10`
- active state/action: 26 dimensions padded to 132 dimensions
- action horizon: H40; `action_mask.sum() == 1040` per sample
- common camera: `cam_left_high -> ego_view`
- normalized language: `insert the shaft into the housing`
- five dataset roots, 806 episodes and 265,906 H40 anchors
- natural H40-anchor sampling weights
- native flow target, masked MSE, action encoder, decoder and four-step Euler sampler
- codec evidence never enters or lengthens the VLM/backbone token sequence

Changing any item above creates a different experiment rather than this Codec
branch.

## Observed video facts

The five `info.json` files claim AV1, but 27 representative files spanning all
11 source/camera groups probe as MP4/H.264 High, 640x480, yuv420p, 30 fps, with
B-frames and reordered PTS/DTS. Their color tags are unknown. The existing
files are decoded-RGB assets only; they are not causal codec fixtures and must
not be relabeled as such.

All five sources therefore require this external, read-only derivation path:

```text
canonical LeRobot decode to RGB
  -> fixed causal canonical encoder
  -> independently decoded row binding
  -> hashed CodecEvidence fixture/warehouse
```

The source datasets and their MP4 files remain unchanged.

## Recommended first implementation stage: C0/C1 CPU candidate

The first stage ports only the baseline-neutral Codec evidence transport and
the original GR00T action-side conditioner. Mamba/MAM action-expert work is
explicitly deferred.

The default or `enabled=false` path must be behavior-identical to the current
five-source baseline: no fixture read, tensor allocation, module RNG, or schema
requirement. Enabled mode adds one evidence window per current observation;
H40 action semantics stay unchanged.

The action-side insertion point is after native action encoding plus position
embedding and before state/action concatenation. One T8 window produces 32
context tokens. The proposal and gated residual are `[B,40,1536]`. The direct
gate starts at exact zero, contains no sigmoid or stochastic operation, and
must give `torch.equal` parity. All-invalid online evidence is a reason-coded
exact-zero native fallback; missing evidence during enabled training fails
closed.

The first stage excludes bitstream residual. It retains the RPE proxy, which
means decoded luma minus motion-warped decoded reference and must not be called
the bitstream residual.

### Batch fields

The candidate retains the established T8 transport:

- frame indices, PTS/DTS, GOP/reference identity, PAD/I/P type, per-frame
  validity and reason codes: `[B,8]`
- 11 lossless MV field arrays with batch-only ragged padding
  `[B,8,M_batch]`, plus `mv_mask` and `mv_count`
- RPE `[B,8,2,2,4]` plus mask
- nearest legal past I-frame appearance `[B,3,256,256]`, with at most two
  GOP30 anchor slots
- camera-motion summary `[B,8,4]` plus mask
- sample validity and reason code `[B]`

For record-v2, PAD temporal/reference indices are `-1`. For episode-bundle,
ordinary numeric fields including `frame_index` are zero and excluded by an
exact-false `frame_mask`; only `frame_anchor_slot` is `-1`. In both schemas,
PAD evidence is numerically zero and all PAD validity is false. No truncation,
pickle, silent drop, B/future reference, or repeated-first-frame padding is
allowed.

### T8/H40 alignment

For H40 anchor row `n`, history is `max(0,n-7)..n`. Rows 0..6 use left PAD;
row 0 has mask `00000001`, ..., row 6 has `01111111`, and rows >=7 have
`11111111`. Across all five sources this affects 5,642 anchors and 22,568 PAD
slots. The history describes the observation only; it does not create eight
actions or alter the 40-step target.

## Canonical encoder candidate

The current candidate, pending exact binary hashes and a real-fixture Human
gate, is:

- decoded RGB input, 640x480, 30 CFR
- H.264/libx264, yuv420p, preset `medium`, CRF 18
- closed GOP/keyint/min-keyint 30, `bframes=0`, `ref=1`
- scene cut and open GOP disabled
- deterministic single-thread/lookahead settings
- time base 1/30000, PTS/DTS `row * 1000`
- IDR exactly at `row % 30 == 0`; every P references `row - 1`
- no audio or inherited metadata

Color tags remain `unknown` for the first capability comparison. Declaring
BT.601/BT.709 or limited/full range would be a separate scientific decision.
Every source is decoded before re-encoding, regardless of its metadata claim.

The online generator should live at the external evaluator/robot client, which
owns ordered RGB history and episode reset boundaries. It should send an
explicit evidence payload through the policy RPC. The policy server validates
and collates it but remains stateless and does not infer history. Legacy BGR
clients must explicitly convert to canonical RGB before evidence generation.

## Evidence references

- Codex thread `01a024c9-740f-7b22-af88-a3f4c8474f73`, titled
  `确认Unitree G1相机串流格式`
- `GR00T_N17_CODEC_EVIDENCE_INTERFACE_CONTRACT_R4.md`
- `GR00T_N17_CODEC_DYNAMICS_ACTION_CONDITIONER_CONTRACT_R3.md`
- `CROSS_BASELINE_CODEC_EVIDENCE_GR00T_INTERFACE_EXECUTION_CONTRACT_R1.md`
- `GR00T_N17_CODEC_CAUSAL_FIXTURE_CAPABILITY_GATE_DECISION_R1.md`
- `FINAL_IMMUTABLE_CLOSURE_R3.json`, SHA-256
  `bafd735be8e206f99015ba8a1802fead8210823010495e0d86d28934b7616296`

The thread was readable during branch creation. Two follow-up send attempts
failed with the Codex App error that dynamic thread messaging was unavailable;
no reply is claimed. This contract uses only the thread's completed record and
the local immutable/hash-identified evidence above.

## Existing donor boundary

There is no Codec Git commit to cherry-pick. The only old implementation is a
dirty worktree based on `25bd54f7...`, and it also contains later unreviewed R5
work. R4 identity is the 14-file hash closure recorded by
`FINAL_IMMUTABLE_CLOSURE_R3.json`; its fresh review rejected two P1 issues:
PAD RPE/camera values were not forced to zero, and out-of-T8 I-frame identity
was not bound. Any port must select files by recorded hash, fix those negative
cases, and manually merge with current `g1_all_lerobot_h40_v1`, rank-0 write
safety, manifest CLI, sampling evidence and reload smoke. Whole-tree copying is
forbidden.

## Experiment sequence

1. **C0 — completed:** branch-owned schema/transport/conditioner port,
   default-disabled.
2. **C1 — completed:** synthetic and temporary CPU conformance tests, including
   strict PAD/causality/identity negatives, all-invalid online fallback,
   enabled-training fail-closed behavior, no-read/no-import disabled identity,
   ragged `M=0`, five-source DatasetFactory closure and exact-zero parity.
3. **C2:** one real bounded causal capability fixture after Human supplies the
   output root, binary identities and storage/time budget.
4. **C3:** separately authorized five-source fixture coverage.
5. **C4:** separately authorized GPU full-model parity.
6. **C5:** separately authorized training and evaluator experiments.

No stage automatically authorizes the next one.

## Authorized C2 fixture canary

The Human authorized one CPU-only, offline materialization attempt at
`/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c2-r1`. It selects the
minimum complete episode index with length at least 40 from each frozen source;
the current read-only resolution is episode 0 with lengths
`374/439/355/284/351` (1,803 frames total). This is a five-episode technical
canary and is explicitly not full 806-episode training coverage.

The producer first resolves and hashes the committed implementation,
`/usr/bin/ffmpeg`, `/usr/bin/ffprobe`, the loaded system libx264, PyAV
`export_mvs`, TorchCodec, all selected payloads, and the existing five-source
loader preparation. Only a clean committed worktree may execute. It writes a
durable attempt receipt before derivation, uses no retry, preserves the exact
11 raw MV fields and accepted bitstream, and publishes `fixture_manifest.json`
only after all five bundles pass runtime round-trip checks.

Read-only preflight:

```bash
PYTHONPATH=. NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/sl/miniconda3/envs/gr00t_n1d7/bin/python \
examples/G1_ALL_LEROBOT_CODEC/build_c2_fixture_canary.py --preflight
```

The authorized execute command adds an outer 7,200-second timeout and uses the
same offline environment. Its output root is no-clobber, storage is capped at
5,000,000,000 bytes, and canonical Parquet/MP4 plus metadata hashes are checked
before and after generation. No checkpoint is opened or produced.

```bash
C2_OUTER_TIMEOUT_SECONDS=7200 \
/usr/bin/timeout --signal=TERM --kill-after=30s 7200s \
/usr/bin/env PYTHONPATH=. NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/sl/miniconda3/envs/gr00t_n1d7/bin/python \
examples/G1_ALL_LEROBOT_CODEC/build_c2_fixture_canary.py --execute
```

The combined C0/C1/C2 pre-execution focused suite currently reports 169 passed
with one existing Transformers cache deprecation warning.

## Recommended comparison matrix

- F0: native RGB reference
- F2: full Codec evidence
- F3: same Codec stack with zero/all-invalid evidence
- F4: same stack with causal-window temporal shuffle

F1 decoded-RGB T8 and modality ablations should be added before any broad
representation claim. F2-F7 must execute the same 9,571,649-parameter stack;
F8 single-token fusion is currently unmatched cost and excluded.

## Decisions still required

1. C2 fixture materialization uses `episode-bundle-v1`; `record-v2` remains a
   bounded transport/online conformance surface and is rejected for enabled
   five-source training.
2. Fixture coverage and absolute output root: small five-source canary first,
   then separately authorize all 806 episodes.
3. Initialization: continue all arms from the retained checkpoint-20000, or
   retrain native and Codec arms from the original N1.7 base. A code baseline
   does not decide model initialization.
4. Trainability: Codec-only is recommended for the first attribution study;
   every control must use the same native trainability and optimizer contract.
5. Exact `ffmpeg`/`ffprobe`/libx264/MV extractor identities and hashes, plus
   attempt, time and storage budgets.

Until those decisions are supplied, this branch may contain only C0/C1
CPU-safe code and tests. It must not materialize real fixtures or start GPU
work.

Before C2/C5, move the hash-only index/identity preflight ahead of 3B model
allocation. The current C0/C1 factory validates the index before dataset batch
construction but is invoked after model construction; that ordering is a
future resource-efficiency prerequisite, not evidence of a real fixture.
Also tighten the bundle JSON Schema descriptions for the raw-MV 11-field audit
and accepted-bitstream dtype/shape; runtime validation is already strict. Model
checkpoints intentionally retain only fixture-free Codec semantics, so every
C5 Run must separately record and compare the adapter, fixture index and global
codec-contract identities rather than claiming the checkpoint is a data closure.

## C0/C1 validation result

- record-v2 and episode-bundle-v1 synthetic/in-memory/temp-disk paths execute
- R4 PAD/causality/identity negative cases fail closed
- all five H40 leaves share one verified store and require exact UID/identity
  coverage from a production (not synthetic) fixture contract
- enabled loader views bind data/video symlink targets and exact
  info/modality/stats/episodes/tasks/manifest semantics to the canonical source
- invalid enabled-training evidence fails closed; reason-coded exact-zero
  fallback is restricted to the online path
- absent/disabled paths do not import Codec modules, read fixture/episode
  payloads, allocate Codec modules, or perturb RNG
- gate=0 and all-invalid paths return exact native action features
- model sizing is exactly 9,571,649 parameters and the R4 analytic FLOP envelope
- processor save/reload retains Codec config only for enabled checkpoints
- focused command:
  `NO_ALBUMENTATIONS_UPDATE=1 /home/sl/Isaac-GR00T-m0-clean/.venv/bin/python -m pytest -q -p no:cacheprovider tests/codec_evidence tests/g1_all_lerobot_h40/test_cpu_adapter.py`
- focused suite: 157 passed; Ruff check/format and `git diff --check` passed

These are CPU interface results, not real-fixture, GPU-parity, training,
latency, task-success, or scientific-performance results.
