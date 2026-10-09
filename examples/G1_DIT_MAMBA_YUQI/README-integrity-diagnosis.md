# Frozen-base integrity diagnosis

`integrity_diagnosis.py --plan PLAN --plan-sha256 SHA --artifact-root ROOT
--output ROOT/RUN/output --wall-seconds 1750` executes only the explicitly frozen
short diagnostic matrix. It never resumes from the failed50k candidate or writes
new scientific checkpoints.

Each matrix entry starts from a fresh B0 and the same trusted20k student/AdamW/RNG.
The prescribed trajectory/data-line × Graph/eager cases execute continuous, t0 and
t750 queries, first without optimizer updates, then three diagnostic updates.
The whole matrix is limited to twelve diagnostic updates. There is no retry or
extension if a mutation is found or the budget is exhausted.

After validating the original complete B0 hash, initial CPU tensor copies are kept
in RAM. Disk evidence contains names, parameter/persistent-buffer classifications,
shape/dtype/hash/storage/version metadata, registered-buffer inventories and stage
records. It does not contain a complete base checkpoint. Ordinary Trace scopes
are audited through original module owners; restored boundaries additionally check
full state-dict keys and the original action-core/encoder/decoder bindings.

Checks cover setup, Graph construction, each teacher query, student forward,
backward, clipping, optimizer update, fixed cached evaluation and Graph release.
Teacher/student observers are installed only after capture and delegate unchanged
mathematics. CUDA synchronization and CPU readback perturb execution timing and
may hide a rare asynchronous/allocator issue; non-reproduction does not validate
the failed historical run.

The first difference saves complete before/after tensor hashes, chunk-scanned byte
difference counts and bounded value/byte fragments, plus external storage-overlap
information. It immediately stops the remaining matrix. No giant whole-tensor
boolean difference mask or complete base dump is allocated on disk. Fresh optimizer
payload copies prevent AdamW CPU step counters from contaminating later matrix
entries. No final data, new RGB cache, physical robot or downstream training is used.

CPU tests cover temporary Trace scopes, restored-key checks, RNG preservation,
chunked evidence, first-mutation stopping and mutation evidence after Graph inputs
have been released. Only the supervisor launches the frozen GPU diagnostic.
