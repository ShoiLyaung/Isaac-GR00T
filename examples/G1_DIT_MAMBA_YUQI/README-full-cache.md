# Complete specified train/dev native feature cache

`action_transfer_full_cache.py --plan PLAN --artifact-root ROOT --output
ROOT/cache-train-dev-r1/output --wall-seconds 7150` builds the frozen180train+
8dev cohort at stride4. Each grid is exactly `range(0, length-15, 4)` with no
extra last anchor:21151 train and837 dev observations,21988 total. This is not
all220 episodes or every source frame. Final/reserve episodes are never encoded.

The full945-record input/source/cohort manifest is verified before loading B0.
The original prepared processor remains eval, with no augmentation, seed42 and
batch1. The original native prepare_input/backbone/_encode_features operations
are reused. Raw features are saved before postprocessing mutates a fresh
container. Raw BF16, post-VL FP32, action-input tensors, physical states/targets
and input identities remain in the original cache schema. No precision lowering,
VLM update, policy sampling or optimizer execution is performed.

One CPU episode dictionary is retained at a time. The old reference cache keeps
only the current matching episode loaded; its values are released after the
new episode is written. Every old334-observation intersection must match its
stored content hash, tensor values/dtypes and complete input identity exactly.
The resulting CACHE_RESULT inventory and episode files are readable by the
existing PilotCache contract with the new grids/episode inventory. Both initial
and final model content hashes must match frozen B0.

Each episode uses an exclusive new file. `episodes.jsonl` records only completed
files with counts, SHA, bytes, elapsed episode time and GPU peak. Console updates
every200 encoded observations distinguish encoded RAM frames from committed
files. Failure preserves partial files and reports current episode/anchor/stage;
overlap errors identify the section/tensor or metadata key and new/old hashes.
There is no resume scheduler or automatic retry. A later explicitly authorized
attempt may reuse completed episodes only after their recorded SHAs validate.

Limits are7150 internal seconds,75GiB total new artifacts and44GiB allocated
memory on GPU1. Existing plans/caches and the running Phase A source tree remain
unchanged. The expected output is approximately52.75GiB. This is one shared data
preparation for subsequent matched conditions, not a model-quality experiment.

CPU tests verify raw-before-post isolation, clone protection, detailed exact
intersection failures, exclusion/stride rules, exclusive episode creation and
actual loading of a generated toy episode through PilotCache. They do not run
the native model or any GPU computation.
