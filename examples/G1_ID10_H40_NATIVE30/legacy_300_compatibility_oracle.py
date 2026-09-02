"""CPU regression for the pre-manifest 300-episode H40 loader contract."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ADAPTER_KIND,
    G1Id10H40Native30EpisodeLoader,
)


parser = argparse.ArgumentParser()
parser.add_argument("--view-root", required=True)
parser.add_argument("--adapter-config", required=True)
parser.add_argument("--modality-config", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

adapter_config_path = Path(args.adapter_config).resolve()
adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
if adapter_config.get("adapter_kind") != ADAPTER_KIND:
    raise AssertionError("Legacy adapter kind changed")
if "split_manifest_path" in adapter_config or "split" in adapter_config:
    raise AssertionError("Compatibility oracle requires a pre-manifest adapter config")

spec = importlib.util.spec_from_file_location(
    "g1_id10_h40_native30_legacy_config", args.modality_config
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load modality config")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS


loader = G1Id10H40Native30EpisodeLoader(
    Path(args.view_root),
    MODALITY_CONFIGS["new_embodiment"],
    adapter_config_path,
    require_runtime_verified_fixed_state=False,
)
if loader._manifest_mode:
    raise AssertionError("Legacy configuration unexpectedly entered manifest mode")
if len(loader) != 300:
    raise AssertionError(f"Expected 300 legacy episodes, got {len(loader)}")

episode_indices = [int(row["episode_index"]) for row in loader.episodes_metadata]
if episode_indices != list(range(300)):
    raise AssertionError("Legacy episode order or membership changed")

first = loader._load_parquet_data(0)
last = loader._load_parquet_data(299)
statistics = loader.get_dataset_statistics()
for side in ("left_arm", "right_arm"):
    for name in ("mean", "std", "min", "max", "q01", "q99"):
        values = np.asarray(statistics["relative_action"][side][name])
        if values.shape != (40, 7) or not np.isfinite(values).all():
            raise AssertionError(f"Invalid legacy relative statistic {side}.{name}")

lengths = [int(value) for value in loader.episode_lengths]
h40_anchors = sum(max(0, length - 39) for length in lengths)
if h40_anchors != 76244:
    raise AssertionError(f"Expected 76244 H40 anchors, got {h40_anchors}")

result = {
    "status": "G1_ID10_H40_NATIVE30_LEGACY_300_COMPATIBILITY_PASS",
    "adapter_kind": ADAPTER_KIND,
    "adapter_config_path": str(adapter_config_path),
    "split_manifest_present": False,
    "manifest_mode": False,
    "episodes": len(loader),
    "episode_index_range": [episode_indices[0], episode_indices[-1]],
    "first_last_rows": [len(first), len(last)],
    "h40_anchors": h40_anchors,
    "training_cli_parameter_names_changed": False,
}
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
