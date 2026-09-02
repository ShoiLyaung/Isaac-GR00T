"""Build five non-overwriting H40 loader views and a resolved training manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from gr00t.data.dataset.g1_all_lerobot_h40_adapter import (
    ADAPTER_KIND,
    BASELINE_ID,
    CONFIG_SCHEMA_VERSION,
    build_metadata_loader_view,
)
from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    ACTIVE_ACTION_DIM,
    EMBODIMENT_ID,
    EMBODIMENT_TAG,
    HORIZON,
    TEMPORAL_OPTION,
    compute_h40_relative_statistics,
)
from gr00t.experiment.launch_finetune import DATASET_MANIFEST_SCHEMA_VERSION


RECIPE_SCHEMA_VERSION = "g1-all-lerobot-20k-recipe-v1"
EXPECTED_RECIPE_KEYS = {
    "schema_version",
    "dataset_root",
    "training_language",
    "camera_policy",
    "sampling_policy",
    "horizon",
    "total_episodes",
    "total_frames",
    "total_h40_anchors",
    "datasets",
}
EXPECTED_DATASET_KEYS = {
    "source_id",
    "relative_path",
    "raw_task_labels",
    "expected_info_sha256",
    "expected_modality_sha256",
    "episodes",
    "frames",
    "h40_anchors",
    "mix_ratio",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def validate_recipe(path: Path) -> dict[str, Any]:
    recipe = _load_json(path)
    if not isinstance(recipe, dict) or set(recipe) != EXPECTED_RECIPE_KEYS:
        raise ValueError("Joint-dataset recipe schema mismatch")
    if recipe["schema_version"] != RECIPE_SCHEMA_VERSION:
        raise ValueError("Unsupported joint-dataset recipe schema")
    if recipe["camera_policy"] != "common_cam_left_high_only":
        raise ValueError("Only the common cam_left_high camera policy is supported")
    if recipe["sampling_policy"] != "h40_anchor_proportional":
        raise ValueError("Only explicit H40-anchor-proportional sampling is supported")
    if recipe["horizon"] != HORIZON:
        raise ValueError(f"Recipe horizon must be {HORIZON}")
    root = Path(recipe["dataset_root"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("dataset_root must be an existing absolute directory")
    if not isinstance(recipe["training_language"], str) or not recipe["training_language"].strip():
        raise ValueError("training_language must be non-empty")
    entries = recipe["datasets"]
    if not isinstance(entries, list) or len(entries) != 5:
        raise ValueError("This frozen recipe requires exactly five dataset entries")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    totals = {"episodes": 0, "frames": 0, "h40_anchors": 0}
    weight_sum = 0.0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != EXPECTED_DATASET_KEYS:
            raise ValueError(f"Dataset recipe entry {index} schema mismatch")
        if entry["source_id"] in seen_ids:
            raise ValueError(f"Duplicate source_id: {entry['source_id']}")
        seen_ids.add(entry["source_id"])
        source = (root / entry["relative_path"]).resolve()
        if root.resolve() not in source.parents or not source.is_dir():
            raise ValueError(f"Dataset source {index} must be an existing child of dataset_root")
        if source in seen_paths:
            raise ValueError(f"Duplicate dataset source: {source}")
        seen_paths.add(source)
        weight = entry["mix_ratio"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise ValueError(f"Dataset source {index} mix_ratio must be finite and > 0")
        weight_sum += float(weight)
        for key in totals:
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Dataset source {index} {key} must be a positive integer")
            totals[key] += value
        expected_weight = entry["h40_anchors"] / recipe["total_h40_anchors"]
        if not math.isclose(float(weight), expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Dataset source {index} mix_ratio is not H40-anchor proportional")
    for key, total in totals.items():
        if recipe[f"total_{key}"] != total:
            raise ValueError(f"Recipe total_{key} mismatch")
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Dataset mix ratios must sum to one")
    return recipe


def _adapter_config(
    *, recipe: dict[str, Any], entry: dict[str, Any], source: Path, stats_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "adapter_kind": ADAPTER_KIND,
        "baseline_id": BASELINE_ID,
        "source_id": entry["source_id"],
        "canonical_root": str(source),
        "canonical_info_sha256": entry["expected_info_sha256"],
        "canonical_modality_sha256": entry["expected_modality_sha256"],
        "raw_task_labels": entry["raw_task_labels"],
        "training_language": recipe["training_language"],
        "embodiment_tag": EMBODIMENT_TAG,
        "embodiment_id": EMBODIMENT_ID,
        "temporal_option": TEMPORAL_OPTION,
        "source_index_formula": "n+k",
        "row_grid_rate_hz": 30,
        "true_30hz_hardware_clock_proven": False,
        "true_50hz_capture": False,
        "horizon": HORIZON,
        "active_action_dims": ACTIVE_ACTION_DIM,
        "hand_scope": "PROJECT_INSPIRE6_EXTENSION_NO_DIRECT_OFFICIAL_MATCH",
        "runtime_verified_fixed_state": True,
        "fixed_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
        },
        "fixed_action": {
            "waist": [0.0] * 3,
            "base_height_command": [0.0],
            "navigate_command": [0.0] * 3,
        },
        "derived_relative_stats_path": str(stats_path.resolve()),
    }


def plan(recipe: dict[str, Any], output_root: Path) -> dict[str, Any]:
    dataset_root = Path(recipe["dataset_root"]).resolve()
    return {
        "schema_version": "g1-all-lerobot-h40-preparation-plan-v1",
        "output_root": str(output_root.resolve()),
        "output_must_not_exist": True,
        "canonical_payload_policy": "read-only; no metadata or payload writes",
        "training_language": recipe["training_language"],
        "camera_policy": recipe["camera_policy"],
        "sampling_policy": recipe["sampling_policy"],
        "total_episodes": recipe["total_episodes"],
        "total_frames": recipe["total_frames"],
        "total_h40_anchors": recipe["total_h40_anchors"],
        "sources": [
            {
                "source_id": entry["source_id"],
                "canonical_root": str((dataset_root / entry["relative_path"]).resolve()),
                "loader_view": str(
                    (output_root / "sources" / entry["source_id"] / "LOADER_VIEW").resolve()
                ),
                "relative_stats": str(
                    (
                        output_root / "sources" / entry["source_id"] / "H40_RELATIVE_STATS.json"
                    ).resolve()
                ),
                "adapter_config": str(
                    (output_root / "sources" / entry["source_id"] / "ADAPTER_CONFIG.json").resolve()
                ),
                "raw_task_labels": entry["raw_task_labels"],
                "h40_anchors": entry["h40_anchors"],
                "mix_ratio": entry["mix_ratio"],
            }
            for entry in recipe["datasets"]
        ],
    }


def materialize(recipe: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    resolved_plan = plan(recipe, output_root)
    dataset_root = Path(recipe["dataset_root"]).resolve()
    manifest_entries = []
    source_results = []
    for entry in recipe["datasets"]:
        source = (dataset_root / entry["relative_path"]).resolve()
        source_root = output_root / "sources" / entry["source_id"]
        view = source_root / "LOADER_VIEW"
        stats_path = source_root / "H40_RELATIVE_STATS.json"
        config_path = source_root / "ADAPTER_CONFIG.json"
        view_manifest = build_metadata_loader_view(
            source,
            view,
            source_id=entry["source_id"],
            expected_info_sha256=entry["expected_info_sha256"],
            expected_modality_sha256=entry["expected_modality_sha256"],
            raw_task_labels=entry["raw_task_labels"],
            training_language=recipe["training_language"],
        )
        stats = compute_h40_relative_statistics(source, stats_path)
        if stats["episodes"] != entry["episodes"] or stats["anchors"] != entry["h40_anchors"]:
            raise ValueError(f"H40 closure mismatch for {entry['source_id']}")
        _write_json_new(
            config_path,
            _adapter_config(recipe=recipe, entry=entry, source=source, stats_path=stats_path),
        )
        manifest_entries.append(
            {
                "dataset_path": str(view.resolve()),
                "mix_ratio": entry["mix_ratio"],
                "adapter_config_path": str(config_path.resolve()),
            }
        )
        source_results.append(
            {
                "source_id": entry["source_id"],
                "view_manifest": view_manifest,
                "episodes": stats["episodes"],
                "h40_anchors": stats["anchors"],
                "mix_ratio": entry["mix_ratio"],
            }
        )
    dataset_manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "datasets": manifest_entries,
    }
    _write_json_new(output_root / "DATASET_MANIFEST.json", dataset_manifest)
    summary = {
        "schema_version": "g1-all-lerobot-h40-preparation-summary-v1",
        "status": "PASS",
        "plan": resolved_plan,
        "dataset_manifest": str((output_root / "DATASET_MANIFEST.json").resolve()),
        "sources": source_results,
    }
    _write_json_new(output_root / "PREPARATION_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recipe = validate_recipe(args.recipe.resolve())
    result = (
        plan(recipe, args.output_root) if args.dry_run else materialize(recipe, args.output_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
