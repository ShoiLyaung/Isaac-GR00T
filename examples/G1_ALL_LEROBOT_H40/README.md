# G1 all-LeRobot H40 joint training

This example mixes the five explicitly listed LeRobot roots in
`recipe.20k.json` into one `new_embodiment` model without modifying or copying
their Parquet/video payloads.

The shared contract is 26 active dimensions, H40 on the represented 30 Hz row
grid, and the only camera present in every source (`cam_left_high`, exposed to
the model as `ego_view`). The three wrist-camera streams remain provenance and
are not model inputs. Left-arm-dominant sources still supervise all 26 target
dimensions; a nearly static right arm is not treated as missing data.

The two source labels `100_notau` and `insertion_with_torque` are retained in
each view manifest and checked against every Parquet row. The recipe explicitly
normalizes all five inputs to the evaluator-facing language `insert the shaft
into the housing`; this is never inferred from directory names.

Dry-run only resolves paths and weights:

```bash
PYTHONPATH=. python examples/G1_ALL_LEROBOT_H40/prepare_joint_dataset.py \
  --recipe examples/G1_ALL_LEROBOT_H40/recipe.20k.json \
  --output-root /absolute/new/preparation/root \
  --dry-run
```

Materialization creates small metadata overlays and H40 statistics in a new
output root. `data/` and `videos/` are symlinks to the canonical read-only
sources. The generated `DATASET_MANIFEST.json` is passed to the normal
`gr00t/experiment/launch_finetune.py` with
`--dataset-manifest-path`, `--dataset-adapter g1_all_lerobot_h40_v1`, and the
existing `G1_ID10_H40_NATIVE30` modality config.

The recipe uses natural H40-anchor weights. All five sources have non-zero
probability, but a finite 20,000-step run does not guarantee a literal visit to
every anchor. Exact shaft/housing geometry and formal single/dual-hand labels
are not present in the metadata and remain provenance unknowns.
