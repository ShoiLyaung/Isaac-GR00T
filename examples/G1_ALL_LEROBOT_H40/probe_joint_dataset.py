"""Read-only CPU closure probe for a resolved all-LeRobot dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.dataset.g1_all_lerobot_h40_adapter import (
    G1AllLerobotH40EpisodeLoader,
    load_adapter_config,
)
from gr00t.data.dataset.g1_id10_h40_native30_adapter import HORIZON
from gr00t.experiment.launch_finetune import load_modality_config, resolve_dataset_specs
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--modality-config", required=True, type=Path)
    parser.add_argument("--decode-one-frame", action="store_true")
    parser.add_argument("--build-mixture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_modality_config(str(args.modality_config.resolve()))
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    tag = "new_embodiment"
    ft_config = FinetuneConfig(
        base_model_path="/read-only-probe/no-model-load",
        embodiment_tag=tag,
        dataset_manifest_path=str(args.dataset_manifest.resolve()),
        dataset_adapter="g1_all_lerobot_h40_v1",
    )
    specs = resolve_dataset_specs(ft_config, tag)
    sources = []
    for spec in specs:
        view = Path(spec["dataset_paths"][0])
        adapter_path = Path(spec["adapter_config_path"])
        adapter = load_adapter_config(adapter_path)
        loader = G1AllLerobotH40EpisodeLoader(
            view,
            MODALITY_CONFIGS[tag],
            adapter_path,
            require_runtime_verified_fixed_state=True,
        )
        first = loader._load_parquet_data(int(loader.episodes_metadata[0]["episode_index"]))
        anchors = sum(max(0, int(length) - HORIZON + 1) for length in loader.episode_lengths)
        statistics = loader.get_dataset_statistics()
        result = {
            "source_id": adapter["source_id"],
            "view": str(view.resolve()),
            "canonical_root": adapter["canonical_root"],
            "episodes": len(loader),
            "h40_anchors": anchors,
            "mix_ratio": spec["mix_ratio"],
            "first_episode_rows": len(first),
            "first_episode_language": first["language.annotation.human.task_description"]
            .unique()
            .tolist(),
            "relative_stats_shapes": {
                side: {
                    name: list(np.asarray(values).shape)
                    for name, values in statistics["relative_action"][side].items()
                }
                for side in ("left_arm", "right_arm")
            },
            "payload_links": {
                name: {
                    "is_symlink": (view / name).is_symlink(),
                    "target": str((view / name).resolve()),
                }
                for name in ("data", "videos")
            },
        }
        if args.decode_one_frame:
            episode_index = int(loader.episodes_metadata[0]["episode_index"])
            video = loader._load_video_data(episode_index, np.asarray([0], dtype=np.int64))
            frames = video["ego_view"]
            result["one_frame_decode"] = {
                "count": len(frames),
                "shape": list(np.asarray(frames[0]).shape),
                "finite": bool(np.isfinite(np.asarray(frames[0])).all()),
            }
        sources.append(result)
    summary = {
        "schema_version": "g1-all-lerobot-h40-cpu-closure-v1",
        "status": "PASS",
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "source_count": len(sources),
        "total_episodes": sum(item["episodes"] for item in sources),
        "total_h40_anchors": sum(item["h40_anchors"] for item in sources),
        "mix_ratio_sum": sum(item["mix_ratio"] for item in sources),
        "sources": sources,
    }
    if args.build_mixture:
        config = get_default_config().load_dict(
            {"data": {"download_cache": False, "datasets": specs}}
        )
        config.data.num_shards_per_epoch = 10000

        class StatisticsSink:
            def __init__(self) -> None:
                self.statistics = None

            def set_statistics(self, statistics, *, override: bool) -> None:
                self.statistics = statistics

        processor = StatisticsSink()
        mixture, evaluation = DatasetFactory(config).build(processor)
        schedule_counts = np.bincount(
            [dataset_index for dataset_index, _ in mixture.shard_sampling_schedule],
            minlength=len(specs),
        )
        summary["mixture_closure"] = {
            "evaluation_is_none": evaluation is None,
            "leaf_count": len(mixture.datasets),
            "leaf_weights": mixture.weights,
            "schedule_shards": len(mixture.shard_sampling_schedule),
            "schedule_dataset_counts": schedule_counts.tolist(),
            "statistics_embodiments": sorted(processor.statistics),
        }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
