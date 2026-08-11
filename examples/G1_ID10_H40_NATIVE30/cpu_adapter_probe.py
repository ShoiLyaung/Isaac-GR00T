"""No-video, no-model H40 loader/statistics probe."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTION_GROUPS,
    G1Id10H40Native30EpisodeLoader,
    STATE_GROUPS,
)


parser = argparse.ArgumentParser()
parser.add_argument("--view-root", required=True)
parser.add_argument("--adapter-config", required=True)
parser.add_argument("--modality-config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--episode", type=int, default=0)
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("g1_id10_h40_native30_config", args.modality_config)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load modality config")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

loader = G1Id10H40Native30EpisodeLoader(
    Path(args.view_root),
    MODALITY_CONFIGS["new_embodiment"],
    Path(args.adapter_config),
    require_runtime_verified_fixed_state=False,
)
frame = loader._load_parquet_data(args.episode)
statistics = loader.get_dataset_statistics()
expected_columns = {
    *(f"state.{key}" for key in STATE_GROUPS),
    *(f"action.{key}" for key in ACTION_GROUPS),
    "language.annotation.human.task_description",
}
if set(frame.columns) != expected_columns:
    raise AssertionError(f"Unexpected columns: {sorted(set(frame.columns) ^ expected_columns)}")
relative_shapes = {}
for side in ("left_arm", "right_arm"):
    for statistic in ("mean", "std", "min", "max", "q01", "q99"):
        shape = list(np.asarray(statistics["relative_action"][side][statistic]).shape)
        if shape != [40, 7]:
            raise AssertionError(f"{side}.{statistic}: {shape}")
        relative_shapes[f"{side}.{statistic}"] = shape
result = {
    "status": "H40_NATIVE30_CPU_ADAPTER_PROBE_PASS_NO_MODEL",
    "episode": args.episode,
    "rows": len(frame),
    "columns": sorted(frame.columns),
    "relative_shapes": relative_shapes,
    "action_delta_indices": list(MODALITY_CONFIGS["new_embodiment"]["action"].delta_indices),
    "video_delta_indices": list(MODALITY_CONFIGS["new_embodiment"]["video"].delta_indices),
    "state_delta_indices": list(MODALITY_CONFIGS["new_embodiment"]["state"].delta_indices),
    "row_grid_rate_hz": 30,
    "true_30hz_hardware_clock_proven": False,
    "true_50hz_capture": False,
}
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
with output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
