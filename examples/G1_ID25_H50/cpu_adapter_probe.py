"""No-video, no-model CPU probe for a materialized loader view."""

import argparse
import importlib.util
import json
from pathlib import Path

from gr00t.data.dataset.g1_id25_h50_adapter import (
    ACTION_GROUPS,
    G1Id25H50EpisodeLoader,
    STATE_GROUPS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-root", required=True)
    parser.add_argument("--adapter-config", required=True)
    parser.add_argument("--modality-config", required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("g1_id25_h50_config", args.modality_config)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load modality config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    tag = "unitree_g1_full_body_with_waist_height_nav_cmd"
    loader = G1Id25H50EpisodeLoader(
        dataset_path=Path(args.view_root),
        modality_configs=MODALITY_CONFIGS[tag],
        adapter_config_path=Path(args.adapter_config),
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
    print(
        json.dumps(
            {
                "status": "CPU_ADAPTER_PROBE_PASS_NO_MODEL",
                "episode": args.episode,
                "rows": len(frame),
                "columns": sorted(frame.columns),
                "relative_left_arm_shape": [
                    len(statistics["relative_action"]["left_arm"]["mean"]),
                    len(statistics["relative_action"]["left_arm"]["mean"][0]),
                ],
                "true_50hz_capture": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
