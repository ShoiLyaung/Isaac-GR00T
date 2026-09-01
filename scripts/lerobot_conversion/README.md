# LeRobot data conversion

This directory holds LeRobot-format conversion scripts and wrappers for the
**insertion_shaft** task on Unitree G1 (Inspire hands).

## Two scripts, two purposes

| Script | Purpose |
|---|---|
| `convert_v3_to_v2.py` | **LeRobot → LeRobot**: convert a v3.0 dataset back to v2.1 (uses lerobot's own converter). Self-contained uv package — has its own `pyproject.toml`. |
| `convert_insertion_shaft_to_lerobot.sh` | **Unitree raw JSON → basic LeRobot v3**: Stage 1 of the G1 Inspire conversion pipeline. |
| `convert_lerobot_to_groot.sh` | **basic LeRobot v3 → GR00T-flavored LeRobot**: Stage 2 of the G1 Inspire conversion pipeline. |
| `convert_all_insertion_shaft.sh` | Run Stage 1 + Stage 2 for **all 3 batches** in `groot_datasets/raw/insertion_shaft/`. Requires `CONFIRM=yes` env var. |

The 3 `convert_*_to_*.sh` shell wrappers are thin convenience layers over the
vendored scripts in `external_dependencies/unitree_lerobot/`. See that repo's
`CONVERT_NOTES.md` for the canonical-script decisions.

## Setup

### `convert_v3_to_v2.py` (uv package)

```bash
cd scripts/lerobot_conversion
uv venv
source .venv/bin/activate
uv pip install -e . --verbose

python convert_v3_to_v2.py --repo-id BobShan/double_folding_towel_v3.0
```

> May need: `GIT_LFS_SKIP_SMUDGE=1 uv pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"`

### `convert_*.sh` wrappers (no setup — call vendored scripts directly)

Prerequisites (already on the system):
- `ffmpeg` + `ffprobe` on PATH (`/usr/bin/`)
- Python 3.10+ with `lerobot` package (`env_isaaclab` conda env works)

The wrappers internally call:
- `external_dependencies/unitree_lerobot/unitree_lerobot/utils/convert_unitree_json_to_lerobot_multicam_keep_effort.py`
- `external_dependencies/unitree_lerobot/convert_unitree_inspire_multicam_to_groot_keep_effort.py`
- `external_dependencies/unitree_lerobot/unitree_lerobot/utils/sort_and_rename_folders.py`

## Usage examples

### Convert a single batch (Stage 1 only — for pi0/pi05/groot)

```bash
scripts/lerobot_conversion/convert_insertion_shaft_to_lerobot.sh \
    /home/sl/Isaac-GR00T/groot_datasets/raw/insertion_shaft/2026_06_02_main_300ep_NEWEST \
    yuqi/insertion_shaft_threecam_keep_effort \
    --keep-cameras cam_left_high,cam_left_wrist,cam_right_wrist
```

Output: `~/.cache/huggingface/lerobot/yuqi/insertion_shaft_threecam_keep_effort/`

### Convert a basic LeRobot dataset to GR00T-flavored (Stage 2)

```bash
scripts/lerobot_conversion/convert_lerobot_to_groot.sh \
    yuqi/insertion_shaft_threecam_keep_effort \
    /home/sl/datasets/insertion_shaft_threecam_groot \
    --keep-videos cam_left_high,cam_left_wrist,cam_right_wrist
```

Output: `/home/sl/datasets/insertion_shaft_threecam_groot/` (with `meta/modality.json`, `annotation.*`, etc.)

### End-to-end for all batches

```bash
CONFIRM=yes scripts/lerobot_conversion/convert_all_insertion_shaft.sh
```

This will:
1. Rename episodes in each batch to be contiguous
2. Stage 1 each batch → `~/.cache/huggingface/lerobot/<repo_id>`
3. Stage 2 each dataset → `~/datasets/<batch>_groot/`

## Invoking from Claude Code

A skill is registered at `~/.claude/skills/convert-unitree-data/SKILL.md` (also
mirrored at `.claude/skills/convert-unitree-data/SKILL.md` in the project root).
Type `/convert-unitree-data` (or describe what you want) and Claude will surface
the right commands for your input batch / output repo_id / camera config.

## What this skill does NOT do

The skill does not run the conversion automatically. It only plans and shows
commands. To actually run them, copy-paste into a shell.

## See also

- `external_dependencies/unitree_lerobot/CONVERT_NOTES.md` — canonical-script selection
- `external_dependencies/unitree_lerobot/docs/unitree_json_to_training_lerobot.md` — full upstream reference
- `groot_datasets/INDEX.md` — input dataset schema
- `groot_datasets/README.md` — input dataset overview
- `scripts/train/` — training scripts that consume the converted data
- `scripts/eval/` — eval scripts that consume the converted data
