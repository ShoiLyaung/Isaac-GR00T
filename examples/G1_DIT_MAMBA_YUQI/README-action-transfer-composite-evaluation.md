# Explicit core and composite evaluation interfaces

This worktree starts from periphery source `ef6e58177d6b5f61dfff59c0feb40f12ed4bba2b`. It prepares deployment and final-evaluation interfaces only. No GPU, final/reserve outputs, simulator run, new training or candidate selection is performed by source preparation. The Supervisor freezes each dev-only validation PLAN and any later complete candidate set separately.

Private file provenance (copied individually, then adapted in this worktree):

| Source commit | Files |
| --- | --- |
| evaluation `9c75e93f51f0293ccca2ac1a2d993eb0cdc80f58` | `action_transfer_policy.py`, `serve_action_transfer.py`, `action_transfer_serving_probe.py`, `action_transfer_evaluation.py`; prior serving/evaluation CPU tests |
| final `98c945df56d25def1e2b8cf244e06abeedb85753` | `action_transfer_final_evaluation.py`, `action_transfer_final_support.py` |
| New | `action_transfer_checkpoint.py`, composite CPU tests, this README |

No model, processor, shared simulator, wire serializer, periphery training helper or mathematical evaluation helper is changed.

## Checkpoint records

`student_checkpoint` in serving/cost PLANs and every `candidates[]` entry in the final PLAN use the same explicit record:

```json
{
  "kind": "core_periphery",
  "name": "explicit-candidate-name",
  "path": "/absolute/student.pt",
  "bytes": 2983710592,
  "sha256": "actual-file-sha256",
  "content_sha256": "full-checkpoint-semantic-sha256",
  "core_content_sha256": "core-semantic-sha256",
  "periphery_content_sha256": "periphery-semantic-sha256",
  "expected_step": 70000,
  "source_commit": "actual-source-commit",
  "training_index": {"path": "/absolute/checkpoint.json", "sha256": "actual-index-sha256"},
  "fixture_only": false
}
```

The byte value above describes tensor payload only; freeze the actual serialized file size. `kind` must be `core_only` or `core_periphery`. Core-only records have identical full/core semantic hashes and `periphery_content_sha256: null`. Composite keys are exactly `core.*` plus `periphery.<module>.*`, without wrapper `master` prefixes. Every key/shape/dtype/finite value is checked against the real constructed model for deployment and the frozen structural template before final input access. Full/core/periphery hashes and the source step/file/index relationship are independently checked.

A dev fixture may be produced on CPU by `action_transfer_checkpoint.save_deployment_fixture(model, core_record, base_content_sha256, output, source_commit)`, after strictly loading B0 as frozen CPU BF16. The helper verifies B0 and the trained core parent, then creates an independent FP32 composite, `student.pt`, and `fixture-index.json`. That index explicitly has `fixture_only: true`, zero new updates, the original `core_parent` record/index, B0 identity and all three semantic hashes. It returns the exact deployment checkpoint record. It is not a trained joint candidate. Payload is2,983,710,592bytes (about2.779GiB), plus serialization metadata. Every final candidate path rejects a `fixture_only` index even if a caller hides that flag in its record. Fixture deployment also checks each fixture episode/anchor against the PLAN's explicit `train + dev` and `grids` before model/RGB access.

## Deployment and residency

The loader retains canonical FP32 source owners on CPU. `DeployedStudent` materializes one independent native BF16 periphery at setup, retaining it across requests/modes. Its action core remains FP32. The entire student head is temporarily bound to the original frozen VLM model **before** `encode_observation`; bindings restore in `finally`, including VLM failures. The original four-Euler sampler receives one request-local condition computed from the student's current VL features. Inference Graph uses the original `TensorCore(backward=False)` and explicit modulation, never the training backward surface.

Each policy stage checks Graph closure before moving modules. Native mode has the complete teacher head on GPU and all student core/FP32 masters/BF16 periphery on CPU. Student mode has shared VLM + active FP32 core + BF16 student periphery on GPU, while the complete teacher head and all FP32 periphery masters remain CPU. Copies are made before timed regions, and closed Graphs precede device moves. Policy close checks canonical and deployment-head content hashes. Cost timing retains the old prepared-input/H2D/VLM/VL/state/one-condition/four-Euler/neural-decoder/synchronize boundary; processor, physical decoding and CPU evidence copies remain outside.

The real old-client subprocess, legacy response-only encoding, safe request decoder, seed42 after load/warmup/socket bind, metadata/reset RNG behavior, continuous action-noise sequence and CPU output ownership retain the original implementation. Private Kit copies/bootstrap, simulator process groups/seeds and the native success predicate are unchanged.

## CLI and additional PLAN fields

Original CLI forms are retained:

```text
action_transfer_checkpoint.py fixture --plan PATH --plan-sha256 SHA --artifact-root ROOT --output OUT --wall-seconds N
action_transfer_evaluation.py cost|simulate --plan PATH --plan-sha256 SHA --artifact-root ROOT --output OUT --wall-seconds N
serve_action_transfer.py --plan PATH --plan-sha256 SHA --mode MODE --port PORT --ready-file PATH
action_transfer_serving_probe.py --plan PATH --plan-sha256 SHA --artifact-root ROOT --output OUT --wall-seconds N
action_transfer_final_evaluation.py --plan PATH --plan-sha256 SHA --artifact-root ROOT --output OUT --wall-seconds N
```

The CPU `fixture` entrypoint requires `CUDA_VISIBLE_DEVICES=''`, pins the known task-only50k semantic parent, verifies a clean source, material file hashes and full frozen B0, and asserts no CUDA initialization at completion. Its minimal PLAN uses `source_worktree`, `source_base`, `allowed_source_files`, `artifact_root`, `run_id`, `records`, `base_checkpoint`, `base_content_sha256`, `core_checkpoint` (the explicit trained core-only record), `internal_wall_seconds`, and `max_artifact_bytes`. It writes `student.pt`, `fixture-index.json` and `RESULT.json`; no optimizer/RNG training recovery or observation encoding occurs.

`cost|simulate` is positional. Old mode names remain `native_eager/native_graph/student_eager/student_graph`. Existing PLAN source/material/kernel/config/budget/loopback/client/simulator fields remain required. Update `source_base`, `source_worktree`, `allowed_source_files`, and immutable-source records to this worktree.

For the composite **dev serving probe**, include the historical train/dev cache fields: `cache_root`, `cache_report_sha256`, `cache_plan_sha256`, `train`, `dev`, `grids`. Each original-RGB reference must equal the independent existing raw-cache action. Eager/Graph requests report exactly one condition call. The unchanged legacy-client loopback checks the A–B–A input ordering and continuous RNG sequence. A private CPU diagnostic periphery perturbation proves that accidentally using the old head cannot pass these routing tests; no real candidate is perturbed.

For **final**, replace `student_structure_template` with `student_structure_templates`, mapping each candidate `kind` to a frozen `{path, sha256, content_sha256}` structural-template record. A composite dev fixture can supply a template but cannot be a candidate. Freeze2..6 explicit candidates and all comparisons before invoking the final entrypoint. Every candidate's full state and index are validated before model loading, final RGB decoding or cache creation. A candidate failure stops later candidates and preserves existing partial evidence.

Core-only offline quality uses the old native evaluator. Composite quality uses the periphery evaluator with its own frozen B0 teacher head/decoder and `BaseAudit`, never teacher replay through student encodings/decoder. Final metadata explicitly records `checkpoint_kind`, full `student_content_sha256`/`checkpoint_content_sha256`, `core_content_sha256` and `periphery_content_sha256`. Each row records the shared cached raw/state/target/mask frame hash. Cross-candidate checks require equal teacher complete receipts, arrays, metrics and cache identity; student receipts compare only continuous time, discrete time, noisy action and forward seed. Their complete learned-condition receipts remain separately recorded. Original metrics, physical decoder and seed→anchor→episode aggregation remain unchanged.

Main method comparisons remain matched50k controls and joint70k/core-only70k. Cross-budget comparisons are descriptive. Final selection, cost and100 paired simulator decisions remain the Supervisor's separate frozen execution work.

CPU verification uses `CUDA_VISIBLE_DEVICES=''` and checks no CUDA initialization. It covers key/index/hash rejection, fixture rejection before final input, sixth-candidate prevalidation failure, perturbed complete-head routing, once-materialized parity, A–B–A, four Euler/one condition, exception restoration, graph-before-move ownership, final full/core/periphery metadata and teacher pairing, plus inherited wire/metadata/CPU-array/simulator ownership checks. Real neural CUDA Graph, full RGB/raw-cache and old-client execution require the later dev-only PLAN.
