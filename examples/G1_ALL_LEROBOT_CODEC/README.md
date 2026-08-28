# G1 all-LeRobot Codec branch contract

Status: `C3_R2_MEMORY_CANARY_IMPLEMENTED_NOT_EXECUTED`

This short-lived branch starts from
`feat/groot-n17-all-lerobot-20k-r1@dd86aa8deb7c5ee314ca0cf082ab9468b0b61119`.
It preserves the five-source training baseline. The initial branch did not
inherit execution authority; the later bounded C2 fixture canary was separately
authorized and completed. GPU use, training and evaluator execution remain
unauthorized.

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
3. **C2 — completed:** one bounded five-source causal fixture canary under the
   Human-supplied output, toolchain, time and storage contract.
4. **C3 — R2 memory canary authorized:** all 806 inputs remain frozen. R1
   stopped twice at its incorrect summed historical RSS limit; R2 uses only a
   supervisor-sampled live process-tree RSS limit and first runs three fixed
   CPU episodes. Full R2 materialization remains a separate later action.
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

The one authorized attempt completed with terminal status `ACCEPTED` using
implementation commit `cbe9296be28687974b7cb73e51ccbad8f71c1e17`:

- 5/5 sources, 5 episodes, 1,803 frames and 62 GOP30 anchors;
- PyAV/TorchCodec exact equality on all 1,803 decoded RGB rows;
- byte-identical double encodes for all five bundles;
- 2,508,407 untruncated raw MV records with the exact 11-field schema;
- runtime full bundle/index load and Draft 2020-12 Schema validation passed;
- input, implementation and toolchain closures remained unchanged;
- wall time 43.612 seconds, peak RSS 4,982,816,768 bytes;
- 139,202,588 regular-file bytes after terminal receipts, below the 5 GB cap;
- no GPU, network, checkpoint mutation, dataset mutation, training or evaluator.

Material evidence:

- terminal SHA-256:
  `b79f2e72e3812759f381e10b18955eecac37ba55f10ee5010646cb72589d2aca`
- summary SHA-256:
  `ee7fa65e7ed9aa9f3ee2ed0ad080002043f1dcaed8e446aec2ca5f28bf50a65b`
- fixture manifest SHA-256:
  `23ff5969dc150c8f809870e82679a296f1a81206140b49c4ede0a6e43d8d78c6`
- codec contract SHA-256:
  `6fe9b1364713c6f6234e15ae55c04c9ea3da9e403d0625362a4264d5ed7bc9ef`
- toolchain identity SHA-256:
  `e1a78efd5bafdb609aa6b5106fb99790475f167f3343ef980111081fac09df3c`

This is a technically valid, non-training canary. It is not evidence of model
quality and cannot be relabeled as full-dataset fixture coverage.

## C3 full-fixture preparation

The authoritative read-only freeze is
`examples/G1_ALL_LEROBOT_CODEC/c3_full_input_freeze.json` (file SHA-256
`f2e950071eb6187be7747658ae46598b8a3f741c37b710a6e2288c041ea7b9cd`,
content SHA-256
`db9e78c8d7c6ada793f20ab3c41e1fb9e84550d2886646ea8b050f6c3f1de483`).
Its independent episode core digest is
`8d05f252b6ea388cb27a3155db8359ce7ea759436dfd9e9294c865fc47ea594a`.

The freeze proves 806 unique episodes across all five sources: 297,340 frames,
265,906 H40 anchors and 10,291 Codec GOP30 anchors. All 806 Parquet files have
complete row/episode/task/validity semantics; all 806 `cam_left_high` MP4 files
have matching metadata/decode counts and the frozen H.264 source signature.
The maximum complete episode is 1,163 frames. All 806 timestamp columns use a
float32 30 Hz representation; four long episodes exceeded the old fixed
`1e-6` tolerance despite matching that grid exactly. Validation now accepts
only the exact ideal-float64 or exact stored-float32 grid.

The full-fixture budget is an empirical C2 projection:

- central object storage: 22,955,473,233 bytes;
- conservative storage: 28,694,341,542 bytes;
- proposed fail-closed maximum: 30,000,000,000 bytes;
- central wall time: 7,192.17 seconds; proposed maximum: 10,800 seconds;
- proposed peak RSS: 16,000,000,000 bytes;
- proposed scratch: 256,000,000 bytes; one CPU worker;
- automatic retry/resume: zero; resume is explicit and manual only.

`gr00t.data.codec_evidence.warehouse` wraps the current strict episode bundle
format in content-addressed, non-overwritable objects. It binds one derivation
digest to one content digest, fully verifies objects before reuse, never scans
for a “latest” attempt, and publishes the final exact-coverage manifest with
an atomic hard-link create that cannot replace an existing file. The input and
budget preparation packet remains `c3_full_fixture_plan.json`.

Five earlier complete scans remain locally under gitignored `preliminary`
filenames as superseded receipts; they are not part of the reproducible Git
contract. Only tracked `c3_full_input_freeze.json` is authoritative.
The C3-R1 execution at
`/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-full-r1` is retained as
historical failure evidence. Attempt 1 and its one manual resume both stopped
at the same RSS root cause. They left 65 fully verified immutable objects
(24,890 frames, 857 GOP30 anchors) but no `fixture_manifest.json`. Their
terminal hashes are respectively
`fb4b7d86b5df228e630d926b50aba895d9bb6428caa1c0367c6e19e7baff1b7c`
and `24100b1ddbfba07e25498b96aacfc0af3993d00d28fe2fd8e5cc459492ede311`.
R1 is not resumed or migrated.

C3-R2 corrects only resource measurement. The 16 GB limit is enforced from an
independent supervisor thread sampling the current process tree every 0.25
seconds. `RUSAGE_SELF` and `RUSAGE_CHILDREN` maxima remain separate diagnostic
values; their historical maxima are never summed into the limit decision.
Dataset identities, input freeze, codec contract, sequential CPU worker,
offline execution, 256 MB scratch policy and automatic retry count remain
unchanged.

The first R2 execution is the fixed-order three-episode memory canary at
`/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-memory-canary-r2`:

1. `no_torque_100` episode 64 — 1,028 frames;
2. `no_torque_100` episode 65 — 241 frames;
3. `aug1p2mm_220` episode 7 — 1,163 frames.

It is bounded to 7,200 seconds, 5 GB persistent output, 256 MB scratch, one CPU
worker and no retry. It does not publish a full-coverage fixture manifest. The
future full-R2 root is
`/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-full-r2`, with the
previous 10,800-second/30-GB limits, but is not executed by the canary command.

## Recommended comparison matrix

- F0: native RGB reference
- F2: full Codec evidence
- F3: same Codec stack with zero/all-invalid evidence
- F4: same stack with causal-window temporal shuffle

F1 decoded-RGB T8 and modality ablations should be added before any broad
representation claim. F2-F7 must execute the same 9,571,649-parameter stack;
F8 single-token fusion is currently unmatched cost and excluded.

## Decisions still required

1. Accept the R2 memory canary before authorizing full-R2 fixture coverage.
2. Initialization: continue all arms from the retained checkpoint-20000, or
   retrain native and Codec arms from the original N1.7 base. A code baseline
   does not decide model initialization.
3. Trainability: Codec-only is recommended for the first attribution study;
   every control must use the same native trainability and optimizer contract.

Until those decisions are supplied, the R2 canary must not be expanded into
full fixture generation, GPU work, training or an evaluator.

Before C2/C5, move the hash-only index/identity preflight ahead of 3B model
allocation. The current C0/C1 factory validates the index before dataset batch
construction but is invoked after model construction; that ordering is a
future resource-efficiency prerequisite, not evidence of a real fixture.
The C2 change tightened both runtime and JSON Schema requirements for the raw-MV
11-field audit and accepted bitstream. Model checkpoints intentionally retain
only fixture-free Codec semantics, so every
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
