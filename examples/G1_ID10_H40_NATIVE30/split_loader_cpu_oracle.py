"""CPU-only proof that the production H40 loader consumes split assignments."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    G1Id10H40ManifestEpisodeCollection,
    G1Id10H40Native30SingleStepDataset,
)
from gr00t.data.embodiment_tags import EmbodimentTag


parser = argparse.ArgumentParser()
parser.add_argument("--collection-root", required=True)
parser.add_argument("--train-config", required=True)
parser.add_argument("--val-config", required=True)
parser.add_argument("--test-config", required=True)
parser.add_argument("--modality-config", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("g1_id10_h40_native30_config", args.modality_config)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load modality config")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS


modality = MODALITY_CONFIGS["new_embodiment"]
configs = {
    "train": Path(args.train_config),
    "val": Path(args.val_config),
    "test": Path(args.test_config),
}
expected = {"train": 539, "val": 67, "test": 67}
collections = {}
records = {}
statistics = {}
for split, config in configs.items():
    collection = G1Id10H40ManifestEpisodeCollection(
        args.collection_root,
        modality,
        config,
        require_runtime_verified_fixed_state=True,
    )
    if len(collection) != expected[split]:
        raise AssertionError(f"{split}: expected {expected[split]}, got {len(collection)}")
    source_pairs = [
        (row["dataset_name"], int(row["source_episode_index"]))
        for row in collection.episodes_metadata
    ]
    if len(source_pairs) != len(set(source_pairs)):
        raise AssertionError(f"{split}: duplicate source episode")
    first = collection._load_parquet_data(0)
    last = collection._load_parquet_data(len(collection) - 1)
    collections[split] = collection
    statistics[split] = collection.get_dataset_statistics()
    records[split] = {
        "episodes": len(collection),
        "cohort_counts": dict(sorted(Counter(name for name, _ in source_pairs).items())),
        "source_pairs": source_pairs,
        "first_rows": len(first),
        "last_rows": len(last),
    }

if statistics["train"] != statistics["val"] or statistics["train"] != statistics["test"]:
    raise AssertionError("val/test did not reuse the exact train statistics")
split_sets = {
    split: set(map(tuple, value["source_pairs"])) for split, value in records.items()
}
if any(
    split_sets[left] & split_sets[right]
    for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
):
    raise AssertionError("source episode appears in more than one split")

# Exercise the actual sharded training dataset constructor, not only manifest
# metadata parsing.  No shard is decoded in this CPU oracle.
dataset = G1Id10H40Native30SingleStepDataset(
    dataset_path=args.collection_root,
    embodiment_tag=EmbodimentTag("new_embodiment"),
    modality_configs=modality,
    shard_size=65536,
    episode_sampling_rate=1.0,
    seed=42,
    allow_padding=False,
    adapter_config_path=configs["train"],
)
sharded_episode_indices = {
    episode_index
    for shard in dataset.sharded_episodes
    for episode_index, _ in shard
}
if sharded_episode_indices != set(range(expected["train"])):
    raise AssertionError("Production sharding does not cover exactly the filtered train loader")

result = {
    "status": "GR00T_H40_MANIFEST_FILTER_AND_TRAIN_STATS_CPU_ORACLE_PASS",
    "adapter_kind": "g1_id10_h40_native30_v1",
    "training_cli_parameter_names_changed": False,
    "records": records,
    "split_intersections": {"train_val": 0, "train_test": 0, "val_test": 0},
    "train_shards": len(dataset),
    "train_sharded_episode_count": len(sharded_episode_indices),
    "val_test_reuse_train_statistics": True,
    "payload_modified": False,
}
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
