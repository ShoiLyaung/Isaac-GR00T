# G1 ID25 / H50 CPU adapter

This directory implements the frozen `ID25_INSPIRE6_TASK_BASELINE` CPU data
boundary. Its temporal targets are synthetic causal ZOH from a canonical
30 Hz row grid to a nominal 50 Hz H50 grid. They are not true 50 Hz captures.

The implementation is intentionally split into three non-overwriting steps:

1. `build_loader_view.py` creates a metadata-only view and read-only symlinks to
   canonical data/videos.
2. `compute_relative_stats.py` writes H50 relative arm statistics to a separate
   derived output.
3. `cpu_adapter_probe.py` validates one episode without video decoding, model
   loading, CUDA, training, inference, evaluation, or simulation.

The template fixed leg/waist values are CPU fixtures only. `DatasetFactory`
requires `runtime_verified_fixed_state=true` before the adapter can enter the
training path. The separate `LINEAR-50Hz` control is not implemented here.

Future training uses the normal `gr00t/experiment/launch_finetune.py` with:

- `--embodiment-tag UNITREE_G1`
- `--modality-config-path examples/G1_ID25_H50/g1_id25_h50_config.py`
- `--dataset-adapter g1_id25_h50_synthetic_zoh_v1`
- `--adapter-config-path <resolved-config.json>`
- `--action-horizon 50`

Training remains a separately authorized stage.
