#!/usr/bin/env python3
"""Build the one authorized CPU-only full-episode canonical canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time
from typing import Any, Mapping, Sequence

from gr00t.data.codec_evidence.full_episode import (
    CAMERA_ID,
    CanonicalEpisodeInput,
    DatasetFactoryCanonicalDecoder,
    build_canonical_object,
    install_forbidden_upstream_jpeg_audit,
    toolchain_identity,
)
from gr00t.data.codec_evidence.full_episode_verifier import verify_full_episode_snapshot
from gr00t.data.codec_evidence.implementation_manifest import (
    implementation_file_identities,
    verify_implementation_manifest,
)
from gr00t.data.codec_evidence.warehouse import (
    CANARY_COVERAGE,
    SOURCE_KIND,
    AttemptJournal,
    WarehouseExclusiveLease,
    canonical_json_sha256,
    ensure_warehouse_layout,
    sha256_path,
    write_snapshot_manifest,
)
from gr00t.data.dataset.dataset_generation_guard import DatasetGenerationGuard


RUN_ID = "GR00T-N17-CODEC-EVIDENCE-FULL-EPISODE-CANARY-R1"
ATTEMPT_ID = f"{RUN_ID}-ATTEMPT1"
SNAPSHOT_ID = ATTEMPT_ID
EXPECTED_WAREHOUSE = Path("/home/sl/groot_codec_fixtures/GR00T-N17-CODEC-EVIDENCE-WAREHOUSE-R1")
EXPECTED_SCRATCH = Path(
    "/home/sl/groot_codec_fixtures/GR00T-N17-CODEC-EVIDENCE-FULL-EPISODE-CANARY-R1-SCRATCH-ATTEMPT1"
)
MAX_WALL_SECONDS = 10_800
MAX_PERSISTENT_BYTES = 4 * 1024**3
MAX_SCRATCH_BYTES = 4 * 1024**3
MAX_RSS_BYTES = 24 * 1024**3
SEED = 42

AUTHORIZED_EPISODES = (
    {
        "episode_uid": "0b43a36f59c1848386d6aa754b109d51ba3bd963a4f20fcffe15b003b4a08ed3",
        "cohort_id": "aug_1p2mm_220_20260625_26",
        "dataset_name": "aug1.2mm_insert the shaft into the housing",
        "source_episode_index": 50,
        "num_frames": 1050,
    },
    {
        "episode_uid": "4db1f851a1607f73bc131a3c06cc3c1b33b15c7dc0b04ac41da3d8b5ecbe8443",
        "cohort_id": "aug_5p4mm_153_20260624",
        "dataset_name": "aug5.4mm_insert the shaft into the housing",
        "source_episode_index": 9,
        "num_frames": 635,
    },
    {
        "episode_uid": "f0351612e05387214b68855fccca5d4b1ef16fbc1393a97d16d315562da32441",
        "cohort_id": "main_300_20260602",
        "dataset_name": "insert the shaft into the housing",
        "source_episode_index": 122,
        "num_frames": 524,
    },
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input is missing, non-regular, or a symlink: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _check_budget(started: float, warehouse: Path, scratch: Path) -> dict[str, Any]:
    observed = {
        "elapsed_seconds": time.monotonic() - started,
        "warehouse_bytes": _directory_bytes(warehouse),
        "scratch_bytes": _directory_bytes(scratch),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    limits = {
        "elapsed_seconds": MAX_WALL_SECONDS,
        "warehouse_bytes": MAX_PERSISTENT_BYTES,
        "scratch_bytes": MAX_SCRATCH_BYTES,
        "peak_rss_bytes": MAX_RSS_BYTES,
    }
    exceeded = [name for name, limit in limits.items() if observed[name] > limit]
    if exceeded:
        raise RuntimeError(f"C1 resource budget exceeded: {exceeded}, observed={observed}")
    return observed


def _binding_from_guard(guard: DatasetGenerationGuard, train_count: int) -> dict[str, Any]:
    identity = guard.identity
    return {
        "dataset_root": str(identity.dataset_root),
        "catalog_database_path": str(identity.catalog_path),
        "catalog_database_sha256": identity.catalog_sha256,
        "split_manifest_path": str(identity.split_manifest_path),
        "split_manifest_sha256": identity.split_manifest_sha256,
        "payload_manifest_path": str(identity.payload_manifest_path),
        "payload_manifest_sha256": identity.payload_manifest_sha256,
        "train_statistics_path": str(identity.train_statistics_path),
        "train_statistics_sha256": identity.train_statistics_sha256,
        "train_episode_uid_digest": identity.train_episode_uid_digest,
        "resolved_train_episode_count": train_count,
    }


def _toolchain_paths(toolchain: Mapping[str, Any]) -> dict[str, Path]:
    """Flatten every path+SHA toolchain identity into one unique file closure."""

    result: dict[str, Path] = {}
    role_by_path: dict[Path, str] = {}

    def visit(value: Any, components: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            path = value.get("path")
            digest = value.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                resolved = Path(path).resolve(strict=True)
                if sha256_path(resolved) != digest:
                    raise ValueError(f"Toolchain path/SHA identity is already stale: {resolved}")
                role = ".".join(("toolchain", *components))
                prior = role_by_path.get(resolved)
                if prior is None:
                    result[role] = resolved
                    role_by_path[resolved] = role
            for key in sorted(value):
                visit(value[key], (*components, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*components, str(index)))

    visit(toolchain, ())
    if not result:
        raise ValueError("Toolchain identity exposed no exact path/SHA artifacts")
    return result


def expected_implementation_files(
    args: argparse.Namespace, toolchain: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return the exact C1 source/test/tool file set, including loaded decoder libs."""

    repository = Path(__file__).resolve(strict=True).parents[2]
    paths = _toolchain_paths(toolchain)
    existing = {path.resolve(strict=True) for path in paths.values()}
    additional = {
        "producer_cli": Path(__file__),
        "outer_supervisor": repository
        / "examples/G1_ID10_H40_NATIVE30/supervise_codec_full_episode_canary.py",
        "outer.python": Path(sys.executable).resolve(strict=True),
        "outer.bwrap": Path("/usr/bin/bwrap"),
        "outer.timeout": Path("/usr/bin/timeout"),
        "outer.time": Path("/usr/bin/time"),
        "codec_package_init": repository / "gr00t/data/codec_evidence/__init__.py",
        "adapter_config_input": Path(args.adapter_config_path),
        "modality_config_input": Path(args.modality_config_path),
        "launch_finetune_argument": Path(args.launch_finetune),
        "ffmpeg_argument": Path(args.ffmpeg),
        "ffprobe_argument": Path(args.ffprobe),
        "oracle.legacy_300_compatibility": repository
        / "examples/G1_ID10_H40_NATIVE30/legacy_300_compatibility_oracle.py",
        "oracle.split_loader_cpu": repository
        / "examples/G1_ID10_H40_NATIVE30/split_loader_cpu_oracle.py",
        "test.dataset_generation_guard": repository
        / "tests/g1_id10_h40_native30/test_dataset_generation_guard.py",
        "test.cpu_adapter": repository / "tests/g1_id10_h40_native30/test_cpu_adapter.py",
        "test.strict_load_contract": repository
        / "tests/g1_id10_h40_native30/test_strict_load_contract.py",
        "test.codec_evidence.conftest": repository / "tests/codec_evidence/conftest.py",
        "test.pyproject": repository / "pyproject.toml",
    }
    optional_g1_conftest = repository / "tests/g1_id10_h40_native30/conftest.py"
    if optional_g1_conftest.is_file():
        additional["test.g1_id10_h40_native30.conftest"] = optional_g1_conftest
    for test_path in sorted((repository / "tests/codec_evidence").glob("test_*.py")):
        additional[f"test.codec_evidence.{test_path.stem}"] = test_path
    for role, path in additional.items():
        resolved = path.resolve(strict=True)
        if resolved not in existing:
            paths[role] = resolved
            existing.add(resolved)
    return implementation_file_identities(paths)


def _read_only_output_preflight(warehouse: Path, scratch: Path) -> None:
    """Reject an already consumed attempt without creating any output path."""

    for root, label in ((warehouse.parent, "warehouse parent"), (scratch.parent, "scratch parent")):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"C1 {label} must be an existing real directory: {root}")
    if warehouse.exists() or warehouse.is_symlink():
        if warehouse.is_symlink() or not warehouse.is_dir():
            raise ValueError("Warehouse root exists but is not a real directory")
        for candidate, label in (
            (warehouse / "attempts" / ATTEMPT_ID, "attempt journal"),
            (warehouse / "canaries" / SNAPSHOT_ID, "canary snapshot"),
        ):
            if candidate.exists() or candidate.is_symlink():
                raise FileExistsError(f"Authorized C1 {label} already exists: {candidate}")
    if scratch.exists() or scratch.is_symlink():
        raise FileExistsError(f"Scratch attempt path already exists: {scratch}")


def resolve_authorized_episodes(
    split_manifest_path: str | Path,
) -> tuple[Path, tuple[CanonicalEpisodeInput, ...], tuple[str, ...]]:
    requested_split = Path(split_manifest_path)
    if requested_split.is_symlink():
        raise ValueError("Frozen split manifest must not be a symlink")
    split_path = requested_split.resolve(strict=True)
    split = _load_json(split_path)
    dataset_root = Path(split["dataset_root"]).resolve(strict=True)
    requested_payload = Path(split["payload_manifest_path"])
    if requested_payload.is_symlink():
        raise ValueError("Frozen payload manifest must not be a symlink")
    payload_path = requested_payload.resolve(strict=True)
    if sha256_path(payload_path) != split["payload_manifest_sha256"]:
        raise ValueError("Split manifest payload SHA-256 mismatch")
    payload = _load_json(payload_path)
    payload_by_uid = {row["episode_uid"]: row for row in payload["episodes"]}
    assignment_by_uid = {row["episode_uid"]: row for row in split["assignments"]}
    resolved: list[CanonicalEpisodeInput] = []
    for expected in AUTHORIZED_EPISODES:
        uid = expected["episode_uid"]
        assignment = assignment_by_uid.get(uid)
        source = payload_by_uid.get(uid)
        if assignment is None or source is None:
            raise ValueError(f"Authorized episode is absent from frozen manifests: {uid}")
        for field, source_field in (
            ("cohort_id", "cohort_id"),
            ("dataset_name", "dataset_name"),
            ("source_episode_index", "episode_index"),
            ("num_frames", "rows"),
        ):
            if (
                expected[field] != source[source_field]
                or expected[field] != assignment[source_field]
            ):
                raise ValueError(f"Authorized episode identity drift: {uid}:{field}")
        if assignment.get("split") != "train" or assignment.get("eligible") is not True:
            raise ValueError("Authorized C1 episode is no longer eligible train data")
        parquet = dataset_root / source["parquet_relative_path"]
        video = dataset_root / source["cam_left_high_relative_path"]
        for path, sha_field, byte_field in (
            (parquet, "parquet_sha256", "parquet_bytes"),
            (video, "cam_left_high_sha256", "cam_left_high_bytes"),
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Canonical input is missing or a symlink: {path}")
            if sha256_path(path) != source[sha_field] or path.stat().st_size != source[byte_field]:
                raise ValueError(f"Canonical input identity mismatch: {path}")
        resolved.append(
            CanonicalEpisodeInput(
                episode_uid=uid,
                cohort_id=expected["cohort_id"],
                dataset_name=expected["dataset_name"],
                source_episode_index=expected["source_episode_index"],
                num_frames=expected["num_frames"],
                parquet_path=parquet,
                parquet_sha256=source["parquet_sha256"],
                parquet_bytes=source["parquet_bytes"],
                canonical_video_path=video,
                canonical_video_sha256=source["cam_left_high_sha256"],
                canonical_video_bytes=source["cam_left_high_bytes"],
            )
        )
    train_uids = tuple(
        sorted(
            row["episode_uid"]
            for row in split["assignments"]
            if row.get("eligible") is True and row.get("split") == "train"
        )
    )
    if len(train_uids) != 539 or not set(item.episode_uid for item in resolved).issubset(
        train_uids
    ):
        raise ValueError("Frozen train UID closure differs from the authorized C1 plan")
    return dataset_root, tuple(resolved), train_uids


def _launch_cli(args: argparse.Namespace, unused_output: Path) -> list[str]:
    return [
        "--base-model-path",
        str(args.base_model_path),
        "--dataset-path",
        str(args.dataset_path),
        "--modality-config-path",
        str(args.modality_config_path),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--dataset-adapter",
        "g1_id10_h40_native30_v1",
        "--adapter-config-path",
        str(args.adapter_config_path),
        "--action-horizon",
        "40",
        "--output-dir",
        str(unused_output),
        "--num-gpus",
        "1",
        "--global-batch-size",
        "1",
        "--dataloader-num-workers",
        "0",
        "--no-use-wandb",
    ]


def capture_real_launch_config(args: argparse.Namespace, scratch: Path) -> Any:
    import gr00t.experiment.experiment as experiment

    requested_launch = Path(args.launch_finetune)
    if requested_launch.is_symlink():
        raise ValueError("launch_finetune.py must not be a symlink")
    launch = requested_launch.resolve(strict=True)
    captured: list[Any] = []
    original_argv = sys.argv[:]
    original_run = experiment.run

    def capture(config: Any) -> None:
        captured.append(config)

    try:
        experiment.run = capture
        sys.argv = [str(launch), *_launch_cli(args, scratch / "UNUSED_TRAIN_OUTPUT")]
        runpy.run_path(str(launch), run_name="__main__")
    finally:
        experiment.run = original_run
        sys.argv = original_argv
    if len(captured) != 1:
        raise RuntimeError("Real launch_finetune config capture did not occur exactly once")
    config = captured[0]
    config.data.seed = SEED
    config.validate()
    return config


def validate_resolved_h40_modality(config: Any) -> dict[str, Any]:
    """Validate the exact state1/video1/language1/action[0..39]/132-D surface."""

    modality = config.data.modality_configs.get("new_embodiment")
    if not isinstance(modality, dict) or set(modality) != {"video", "state", "action", "language"}:
        raise ValueError("Resolved NEW_EMBODIMENT modality surface drifted")
    deltas = {name: list(modality[name].delta_indices) for name in modality}
    if (
        deltas["action"] != list(range(40))
        or deltas["video"] != [0]
        or deltas["state"] != [0]
        or deltas["language"] != [0]
        or config.model.max_state_dim != 132
        or config.model.max_action_dim != 132
    ):
        raise ValueError("Resolved modality is not the exact H40/132-D canonical surface")
    return {
        "delta_indices": deltas,
        "max_state_dim": config.model.max_state_dim,
        "max_action_dim": config.model.max_action_dim,
    }


def validate_adapter_launch_cross_binding(
    args: argparse.Namespace,
    guard: DatasetGenerationGuard,
    config: Any,
    dataset_root: Path,
) -> dict[str, Any]:
    """Prove the guard, adapter JSON, CLI, and resolved DatasetFactory agree."""

    adapter_path = Path(args.adapter_config_path)
    adapter = _load_json(adapter_path)
    expected_adapter_fields = {
        "adapter_kind": "g1_id10_h40_native30_v1",
        "split": "train",
        "embodiment_id": 10,
        "embodiment_tag": "new_embodiment",
        "horizon": 40,
        "active_action_dims": 26,
    }
    for name, expected in expected_adapter_fields.items():
        if adapter.get(name) != expected:
            raise ValueError(f"Frozen C1 adapter semantic mismatch: {name}")
    codec_config = adapter.get("codec_evidence")
    if codec_config is not None and (
        not isinstance(codec_config, dict) or codec_config.get("enabled") is not False
    ):
        raise ValueError("C1 producer requires legacy/explicitly-disabled codec loading")
    for name, expected in (
        ("split_manifest_path", guard.identity.split_manifest_path),
        ("derived_relative_stats_path", guard.identity.train_statistics_path),
    ):
        value = adapter.get(name)
        declared = Path(value) if isinstance(value, str) else None
        if declared is None or declared.is_symlink() or declared.resolve(strict=True) != expected:
            raise ValueError(f"Adapter {name} differs from the held generation guard")

    if len(config.data.datasets) != 1:
        raise ValueError("Resolved launch config must contain exactly one dataset spec")
    spec = config.data.datasets[0]
    if Path(spec.adapter_config_path).resolve(strict=True) != adapter_path.resolve(strict=True):
        raise ValueError("Resolved DatasetFactory adapter path differs from the CLI")
    declared_dataset_paths = [Path(path) for path in spec.dataset_paths]
    expected_dataset_input = Path(args.dataset_path)
    if expected_dataset_input.is_symlink() or any(
        path.is_symlink() for path in declared_dataset_paths
    ):
        raise ValueError("Resolved DatasetFactory dataset path must not be a symlink")
    resolved_dataset_paths = [path.resolve(strict=True) for path in declared_dataset_paths]
    expected_dataset_path = expected_dataset_input.resolve(strict=True)
    if resolved_dataset_paths != [expected_dataset_path] or expected_dataset_path != (
        dataset_root / "lerobot"
    ):
        raise ValueError("Resolved DatasetFactory dataset path differs from the frozen root")
    if (
        spec.dataset_adapter != "g1_id10_h40_native30_v1"
        or spec.embodiment_tag != "new_embodiment"
        or config.model.action_horizon != 40
        or config.data.video_backend != "torchcodec"
    ):
        raise ValueError("Resolved DatasetFactory adapter/H40/TorchCodec semantics drifted")

    modality_path = Path(args.modality_config_path)
    if modality_path.is_symlink() or not modality_path.is_file() or modality_path.suffix != ".py":
        raise ValueError("C1 modality config must be one real Python source file")
    loaded_module = sys.modules.get(modality_path.stem)
    loaded_path = getattr(loaded_module, "__file__", None)
    if loaded_path is None or Path(loaded_path).resolve(strict=True) != modality_path.resolve(
        strict=True
    ):
        raise ValueError("Resolved launch did not import the exact CLI modality config")
    modality_evidence = validate_resolved_h40_modality(config)
    return {
        **expected_adapter_fields,
        "adapter_config_path": str(adapter_path.resolve(strict=True)),
        "split_manifest_path": str(guard.identity.split_manifest_path),
        "train_statistics_path": str(guard.identity.train_statistics_path),
        "dataset_paths": [str(path) for path in resolved_dataset_paths],
        "modality_config_path": str(modality_path.resolve(strict=True)),
        "video_backend": config.data.video_backend,
        "modality": modality_evidence,
    }


def run_canary(
    args: argparse.Namespace, *, enforce_authorized_paths: bool = True
) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "TORCH_COMPILE": "0",
        }
    )
    requested_warehouse = Path(args.warehouse_root)
    requested_scratch = Path(args.scratch_root)
    if requested_warehouse.is_symlink() or requested_scratch.is_symlink():
        raise ValueError("C1 warehouse/scratch roots must not be symlinks")
    warehouse = requested_warehouse.resolve(strict=False)
    scratch = requested_scratch.resolve(strict=False)
    if enforce_authorized_paths and (
        warehouse != EXPECTED_WAREHOUSE or scratch != EXPECTED_SCRATCH
    ):
        raise ValueError("C1 output paths differ from the one authorized attempt")
    _read_only_output_preflight(warehouse, scratch)

    decoder: DatasetFactoryCanonicalDecoder | None = None
    audit = None
    with DatasetGenerationGuard.acquire_for_manifest(
        args.split_manifest, args.train_statistics, blocking=False
    ) as preflight_guard:
        preflight_guard.verify_frozen_identity()
        dataset_root, episodes, train_uids = resolve_authorized_episodes(args.split_manifest)
        audit = install_forbidden_upstream_jpeg_audit(dataset_root)
        if Path(args.dataset_path).resolve(strict=True) != dataset_root / "lerobot":
            raise ValueError("Resolved launch dataset path differs from frozen lerobot root")
        config = capture_real_launch_config(args, scratch)
        adapter_launch_evidence = validate_adapter_launch_cross_binding(
            args, preflight_guard, config, dataset_root
        )
        decoder = DatasetFactoryCanonicalDecoder(config)
        decoder.bind_generation_guard(preflight_guard)
        try:
            if decoder.train_episode_uids != train_uids:
                raise ValueError("DatasetFactory train UID set differs from frozen split")
            binding_preflight = _binding_from_guard(preflight_guard, len(train_uids))
            # Import all real decoder libraries first, then compute this mapping exactly
            # once.  The same immutable mapping is used by all three derivations.
            tools = toolchain_identity(ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            tools_sha256 = canonical_json_sha256(tools)
            implementation_files = expected_implementation_files(args, tools)
            implementation_preflight = verify_implementation_manifest(
                args.implementation_manifest,
                expected_run_id=RUN_ID,
                expected_files=implementation_files,
            )
            preflight_guard.verify_frozen_identity()
        finally:
            decoder.unbind_generation_guard(preflight_guard)
            decoder.release_generation_lease()
    if decoder is None or audit is None:  # pragma: no cover - construction is fail-closed
        raise RuntimeError("C1 read-only preflight did not construct the real DatasetFactory")

    root = ensure_warehouse_layout(warehouse)
    plan = {
        "run_id": RUN_ID,
        "attempt": 1,
        "source_kind": SOURCE_KIND,
        "camera_id": CAMERA_ID,
        "episode_uids": [item.episode_uid for item in episodes],
        "max_workers": 1,
        "deterministic_encodes_per_episode": 2,
        "limits": {
            "wall_seconds": MAX_WALL_SECONDS,
            "persistent_bytes": MAX_PERSISTENT_BYTES,
            "scratch_bytes": MAX_SCRATCH_BYTES,
            "rss_bytes": MAX_RSS_BYTES,
        },
        "implementation_manifest_sha256": implementation_preflight["sha256"],
        "implementation_file_count": implementation_preflight["file_count"],
        "toolchain_identity_sha256": tools_sha256,
        "adapter_launch_cross_binding": adapter_launch_evidence,
        "scratch_policy": "PRESERVE_ALL_EPISODE_INTERMEDIATES_ON_PASS_OR_FAILURE",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with WarehouseExclusiveLease(root):
        journal = AttemptJournal(root, ATTEMPT_ID, plan)
        try:
            # ATTEMPT_STARTED is now durable.  Only now may the one authorized
            # scratch directory be created.
            scratch.mkdir(mode=0o755, parents=False, exist_ok=False)
            published = []
            episode_results = []
            for episode in episodes:
                with DatasetGenerationGuard.acquire_for_manifest(
                    args.split_manifest, args.train_statistics, blocking=False
                ) as guard:
                    guard.verify_frozen_identity()
                    decoder.bind_generation_guard(guard)
                    try:
                        obj, evidence = build_canonical_object(
                            root,
                            episode,
                            scratch_root=scratch,
                            ffmpeg=args.ffmpeg,
                            ffprobe=args.ffprobe,
                            tool_identity=tools,
                            dataset_factory_decoder=decoder,
                        )
                    finally:
                        decoder.unbind_generation_guard(guard)
                    if (
                        evidence["toolchain_identity"] != tools
                        or canonical_json_sha256(evidence["toolchain_identity"]) != tools_sha256
                    ):
                        raise RuntimeError("One episode used a different derivation toolchain")
                    guard.verify_frozen_identity()
                published.append(obj)
                episode_results.append(evidence)
                _check_budget(started, root, scratch)

            # One continuous shared generation lease covers the final snapshot
            # commit and every base-payload read made by the production verifier.
            with DatasetGenerationGuard.acquire_for_manifest(
                args.split_manifest, args.train_statistics, blocking=False
            ) as guard:
                guard.verify_frozen_identity()
                decoder.bind_generation_guard(guard)
                try:
                    implementation_before_commit = verify_implementation_manifest(
                        args.implementation_manifest,
                        expected_run_id=RUN_ID,
                        expected_files=implementation_files,
                    )
                    binding = _binding_from_guard(guard, len(train_uids))
                    if binding != binding_preflight:
                        raise ValueError("Dataset binding drifted between preflight and commit")
                    manifest_path = write_snapshot_manifest(
                        root,
                        snapshot_id=SNAPSHOT_ID,
                        coverage_kind=CANARY_COVERAGE,
                        dataset_binding=binding,
                        objects=published,
                        producer_identity={
                            "run_id": RUN_ID,
                            "attempt": 1,
                            "implementation_manifest_path": implementation_preflight["path"],
                            "implementation_manifest_sha256": implementation_preflight["sha256"],
                            "implementation_file_count": implementation_preflight["file_count"],
                            "producer_path": str(Path(__file__).resolve(strict=True)),
                            "producer_sha256": sha256_path(Path(__file__)),
                            "toolchain_identity_sha256": tools_sha256,
                            "toolchain": tools,
                        },
                    )
                    manifest_sha = sha256_path(manifest_path)
                    verification = verify_full_episode_snapshot(
                        snapshot_root=manifest_path.parent,
                        snapshot_manifest_sha256=manifest_sha,
                        expected_dataset_binding=binding,
                        expected_train_episode_uids=train_uids,
                        expected_canary_episode_uids=[item.episode_uid for item in episodes],
                        base_dataset=decoder.single_dataset,
                    )
                    implementation_final = verify_implementation_manifest(
                        args.implementation_manifest,
                        expected_run_id=RUN_ID,
                        expected_files=implementation_files,
                    )
                    if not (
                        implementation_preflight
                        == implementation_before_commit
                        == implementation_final
                    ):
                        raise RuntimeError("C1 implementation identity changed during the attempt")
                    guard.verify_frozen_identity()
                finally:
                    decoder.unbind_generation_guard(guard)
            resources = _check_budget(started, root, scratch)
            if audit.forbidden_open_count != 0:
                raise RuntimeError("Canonical-only producer attempted an upstream JPEG source open")
            result = {
                "schema_version": "codec-evidence-full-episode-canary-result-v1",
                "status": "READY_FOR_SUPERVISOR_FINALIZATION",
                "run_id": RUN_ID,
                "attempt": 1,
                "snapshot_manifest_path": str(manifest_path),
                "snapshot_manifest_sha256": manifest_sha,
                "episodes": episode_results,
                "dataset_factory": decoder.runtime_evidence,
                "adapter_launch_cross_binding": adapter_launch_evidence,
                "implementation_identity": implementation_final,
                "toolchain_identity_sha256": tools_sha256,
                "verification": verification,
                "resources": resources,
                "upstream_jpeg_source_open_count": audit.forbidden_open_count,
                "gpu_used": False,
                "network_used": False,
                "training_steps": 0,
            }
            journal.ready(result)
            return result
        except BaseException as error:
            journal.finish(
                "FAILED",
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "output_preserved": True,
                    "cleanup_performed": False,
                    "scratch_root": str(scratch),
                    "scratch_exists": scratch.is_dir(),
                    "scratch_bytes": _directory_bytes(scratch),
                },
            )
            raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-statistics", type=Path, required=True)
    parser.add_argument("--adapter-config-path", type=Path, required=True)
    parser.add_argument("--modality-config-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--launch-finetune", type=Path, required=True)
    parser.add_argument("--implementation-manifest", type=Path, required=True)
    parser.add_argument("--warehouse-root", type=Path, default=EXPECTED_WAREHOUSE)
    parser.add_argument("--scratch-root", type=Path, default=EXPECTED_SCRATCH)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_canary(_parse_args(argv))
    print(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
