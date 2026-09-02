# G1 all-LeRobot causal Codec on current main

This main-based branch keeps the native five-source H40 contract and adds only
the causal `codec-evidence-episode-bundle-v1` action-side conditioner. The
native/default path does not import or read Codec fixtures, allocate Codec
modules, or change RNG consumption. The direct scalar gate starts at exact
zero; Codec evidence never enters the VLM token sequence.

The retained causal fixture is external and read-only:

```text
/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r2
fixture_manifest.json sha256=5e4ec3590dd20fc767957bc1f7b74d8eaf60a766a45be6d3e4a1f8c55412b4b9
```

Create small adapter overlays after preparing the native H40 dataset views:

```bash
PYTHONPATH=. python examples/G1_ALL_LEROBOT_CODEC/prepare_causal_codec_training.py \
  --baseline-manifest /absolute/native-preparation/DATASET_MANIFEST.json \
  --output-root /absolute/new/codec-preparation \
  --fixture-root /home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r2 \
  --fixture-manifest-sha256 5e4ec3590dd20fc767957bc1f7b74d8eaf60a766a45be6d3e4a1f8c55412b4b9
```

Pass the generated manifest with
`--dataset-adapter g1_all_lerobot_h40_v1`, disable every native tune flag,
and set `--codec-trainability codec_only`. Historical C5 20k results remain
bound to source commit `0696eaa`; this branch does not relabel them as new
training or evaluator evidence.

`checkpoint_smoke.py` is a bounded compatibility probe for a retained causal
checkpoint. It strictly compares model/processor Codec identity and requests
one public-policy action with one in-process `EpisodeT8View`; it does not run a
simulator or establish task quality.
