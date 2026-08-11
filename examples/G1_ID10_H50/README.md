# G1 ID10 / H50 CPU adapter

This directory implements the Human-selected `ID10_INSPIRE6_TASK_BASELINE`
candidate CPU data boundary. It does not claim a frozen semantic contract until
the required fresh independent review. Its temporal targets are synthetic causal ZOH from a canonical
30 Hz row grid to a nominal 50 Hz H50 grid. They are not true 50 Hz captures.

The implementation is intentionally split into three non-overwriting steps:

1. `build_loader_view.py` creates a metadata-only view and read-only symlinks to
   canonical data/videos.
2. `compute_relative_stats.py` writes H50 relative arm statistics to a separate
   derived output.
3. `write_temporal_provenance.py` writes exact per-anchor/tick synthetic-ZOH
   lineage to a derived parquet without modifying canonical data.
4. `cpu_adapter_probe.py` validates one episode without video decoding, model
   loading, CUDA, training, inference, evaluation, or simulation.
5. `cpu_processor_probe.py` decodes one real head frame and exercises the real
   checkpoint processor metadata, H50 state/action processing, mock VLM
   collation, and processor save/reload without loading model weights.

After the separate runtime fixed-state gate is closed, the bounded native
baseline uses two dedicated evaluation entries:

- `python -m gr00t.eval.g1_id10_h50_open_loop` uses
  `G1Id10H50EpisodeLoader` for one non-overwriting connectivity result. Its
  MSE/MAE are diagnostics, not task success or a paper-result claim.
- `run_insertion_evaluator.py --execute-simulator ...` is the only named
  insertion-shaft launcher. It requires a separately authorized running policy
  server, one simulator environment, real request timeouts, typed termination
  causes, and scoped client/environment cleanup.
- `run_policy_server.py --execute-model-server ...` is the paired loopback-only
  server. It requests Transformers loading info and fails on every missing,
  unexpected, mismatched key or loading error before serving any request.

The formula `n+floor(3*k/5)` belongs to dataset-target construction only. The
simulator consumes decoded H50 targets as `chunk_index=k`, one target per 50 Hz
tick. Applying 30-to-50 ZOH again in the evaluator is forbidden.

The template fixed leg/waist values are CPU fixtures only. `DatasetFactory`
requires `runtime_verified_fixed_state=true` before the adapter can enter the
training path. The separate `LINEAR-50Hz` control is not implemented here.

Future training uses the normal `gr00t/experiment/launch_finetune.py` with:

- `--embodiment-tag NEW_EMBODIMENT`
- `--modality-config-path examples/G1_ID10_H50/g1_id10_h50_config.py`
- `--dataset-adapter g1_id10_h50_synthetic_zoh_v1`
- `--adapter-config-path <resolved-config.json>`
- `--action-horizon 50`

Training remains a separately authorized stage.
