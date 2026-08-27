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

`launch_joint_training.py` is the bounded external supervisor used for GPU
execution. Its `plan` action is CPU-only. Its `profile` mode runs two optimizer
steps; its `formal` mode freezes 20,000 steps, two physical GPUs, global batch
32, H40, visual/projector/diffusion tuning, an eight-hour wall, a 128 GiB
attempt limit, offline local inputs, and zero automatic retries. The supervisor
owns a single isolated `torchrun` child and two cooperative GPU leases; it does
not add service orchestration to the GR00T core.

`continue_40k` is an isolated full-state continuation profile. It requires an
absolute `--resume-checkpoint` at global step 20,000, keeps that parent
checkpoint read-only, writes a new output directory, and stops at absolute
global step 40,000. The final checkpoint closes at 40,000 while finite-monitor
and sampling evidence cover only the 20,000 optimizer steps performed by the
continuation Run (640,000 samples at global batch 32).

The original 20k cosine schedule has already reached zero. Restoring its
scheduler state into a newly constructed 40k schedule makes the first resumed
optimizer update use zero learning rate and then raises the scheduler to about
`5.41248e-5`; this is not the same trajectory as an uninterrupted 40k plan.
The iterable dataset is also reseeded from 42 to 20042 rather than skipping the
first 20,000 steps. Both differences are recorded in the attempt contract.
