#!/usr/bin/env python3
"""Run source acceptance or the single train-only learned-resampler/Mamba phase."""

from __future__ import annotations

# The frozen Mamba overlay is inserted by the imported technical runner.
# ruff: noqa: E402
import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Final

import numpy as np
import torch

import gr00t
from gr00t.experimental import (
    run_mikasa_history_hard_learned_spatial_resampler_mamba as technical_runner,
)
from gr00t.experimental.mikasa_history_hard_learned_spatial_resampler_mamba import (
    PAIR_CODES,
    ROWS,
    ROWS_PER_GROUP,
)
from gr00t.experimental.mikasa_history_hard_learned_spatial_resampler_mamba_train import (
    ARTIFACT_LIMIT,
    CONDITION_ORDER,
    EVAL_SHUFFLE_SEEDS,
    EXPERIMENT_ID,
    FOLD_ORDER,
    REMAINING_SECONDS,
    TECHNICAL_CACHE_SHA256,
    TRAIN_SEED,
    TRAIN_STEPS,
    combine_condition_evidence,
    effective_samples_sha256,
    evaluate_fold,
    evaluate_ordered_substitutions,
    fold_spec,
    load_checkpoint,
    model_initialization_hash,
    new_model,
    optimizer_schedule_sha256,
    train_fold,
    train_scientific_gates,
)
from gr00t.experimental.mikasa_history_hard_order_runtime import (
    CORPUS_ROOT,
    TARGET_FIELDS,
    TARGET_MANIFEST_SHA256,
    HistoryHardSources,
    RuntimeContractError,
    canonical_json_bytes,
    freeze_directory,
    sha256_bytes,
    sha256_file,
    validate_frozen_evidence,
    write_once,
)


REPO: Final = Path(__file__).resolve().parents[2]
BASE_COMMIT: Final = "91cf0401add04f92e6dcb7aaa3deabd8c98cd108"
EXPECTED_BRANCH: Final = "codex/g1-mikasa-history-hard-learned-spatial-resampler-mamba-train-r1"
SOURCE_PATHS: Final = (
    "gr00t/experimental/mikasa_history_hard_learned_spatial_resampler_mamba_train.py",
    "gr00t/experimental/run_mikasa_history_hard_learned_spatial_resampler_mamba_train.py",
    "tests/test_mikasa_history_hard_learned_spatial_resampler_mamba_train.py",
)
DEPENDENCY_PATHS: Final = (
    "gr00t/experimental/mikasa_history_hard_learned_spatial_resampler_mamba.py",
    "gr00t/experimental/run_mikasa_history_hard_learned_spatial_resampler_mamba.py",
    "gr00t/experimental/mikasa_history_hard_order_runtime.py",
)
PHASES: Final = ("source_acceptance", "train")
TECHNICAL_EXPERIMENT_ID: Final = "g1-mikasa-history-hard-learned-spatial-resampler-mamba-r1"
TECHNICAL_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/g1-mikasa-history-hard-learned-spatial-resampler-mamba-r1"
)
TECHNICAL_RESULTS: Final = {
    "source": (
        TECHNICAL_ROOT / "source-acceptance-cpu-r1/RESULT.json",
        18_873,
        "9b344aed0332a3347b173f1d6cb773523ce57b0ec804fbc23901afbf86e66b9b",
    ),
    "layout": (
        TECHNICAL_ROOT / "layout-probe-gpu0-r1/RESULT.json",
        17_883,
        "d731af753ab43133be6da92a1435249cedffc682291deb4789d75ca5465a2fef",
    ),
    "cache": (
        TECHNICAL_ROOT / "target-blind-full-grid-cache-gpu0-r1/RESULT.json",
        28_716,
        "e2bf628f41ab13df865d076682560f41bfcace8428b5851ea5b2abcaa6b9e0d4",
    ),
    "preflight": (
        TECHNICAL_ROOT / "training-preflight-gpu0-r1/RESULT.json",
        30_699,
        "282594c9b05c952e84d69dbfcc8b8189745eee35b54b3ca26ccfb699b3f7a09b",
    ),
    "overfit": (
        TECHNICAL_ROOT / "group0-overfit-gpu0-r1/RESULT.json",
        113_512,
        "9abee72570fb60e68e5b9916be45c468a294bada818fda73e4433132dd8bd55b",
    ),
}
CACHE_ROOT: Final = TECHNICAL_ROOT / "target-blind-full-grid-cache-gpu0-r1"
FORMAL_OUTPUT_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/"
    "g1-mikasa-history-hard-learned-spatial-resampler-mamba-train-r1/"
    "train-only-gpu0-r1"
)
CACHE_FILES: Final = {
    "cue": (
        CACHE_ROOT / "CUE.npy",
        6_543_114_368,
        "00f7bbbb1aa4ff54f3612ebeed5c2455b278a23fc3f64d7159cd663b93206223",
    ),
    "current": (
        CACHE_ROOT / "CURRENT.npy",
        50_331_776,
        "eb3f49f2f2d425f1c1c9677511690ee97626d6316e0453b28bc39e15af79551d",
    ),
}
PREDECESSOR_PREFLIGHT_WALL: Final = 36.612442183017265
PREDECESSOR_OVERFIT_WALL: Final = 42.25428877695231
CUMULATIVE_LIMIT_SECONDS: Final = 21_600.0
TRAIN_GATE_KEYS: Final = (
    "condition_fold_order_exact",
    "twenty_four_final_checkpoints_exact",
    "all_steps_exact",
    "candidate_heldout_isolation_exact",
    "all_train_targets_opened_once",
    "pre_target_plan_frozen_exact",
    "shared_initialization_exact",
    "optimizer_schedule_exact",
    "effective_samples_matched_across_conditions",
    "curves_logits_gradients_parameters_finite",
    "all_checkpoint_reload_parity_exact",
    "ordered_all_modules_gradient_and_change",
    "heldout_denominators_complete",
    "ordered_substitution_denominator_180",
    "target_manifest_opened_once",
    "dev_fresh_action_outputs_zero",
    "wall_within_remaining_six_hour_budget",
    "artifact_within_512mib",
    "zero_retry",
)


class TrainRunnerContractError(RuntimeError):
    """The frozen train-only source, predecessor, label, or result contract changed."""


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise TrainRunnerContractError(exc.stderr.decode(errors="replace").strip()) from exc


def _bind_repo_root(repo: Path) -> Path:
    resolved = repo.resolve(strict=True)
    expected = REPO.resolve(strict=True)
    if resolved != expected:
        raise TrainRunnerContractError(f"train repo-root differs from imported runner: {resolved}")
    import_roots = {
        Path(__file__).resolve(),
        Path(gr00t.__file__).resolve(),
        Path(sys.modules[train_fold.__module__].__file__).resolve(),
        Path(sys.modules[technical_runner.__name__].__file__).resolve(),
    }
    if any(expected not in path.parents for path in import_roots):
        raise TrainRunnerContractError("train-only imports escaped the active worktree")
    return resolved


def source_identity(repo: Path) -> dict[str, Any]:
    repo = _bind_repo_root(repo)
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    _git(repo, "merge-base", "--is-ancestor", BASE_COMMIT, head)
    observed = {}
    raw = _git(repo, "diff", "--name-status", f"{BASE_COMMIT}..{head}").decode()
    for line in raw.splitlines():
        status_code, path = line.split("\t", maxsplit=1)
        observed[path] = status_code
    expected = {path: "A" for path in SOURCE_PATHS}
    if observed != expected:
        raise TrainRunnerContractError(f"train-only exact source diff changed: {observed}")
    bound = (*SOURCE_PATHS, *DEPENDENCY_PATHS)
    dirty = (
        _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--", *bound)
        .decode()
        .splitlines()
    )
    if dirty:
        raise TrainRunnerContractError(f"train-only source is dirty: {dirty}")
    files = {}
    for relative in bound:
        path = repo / relative
        raw_file = path.read_bytes()
        if path.is_symlink() or raw_file != _git(repo, "show", f"HEAD:{relative}"):
            raise TrainRunnerContractError(f"train-only source differs from HEAD: {relative}")
        files[relative] = {"bytes": len(raw_file), "sha256": sha256_bytes(raw_file)}
    branch = _git(repo, "branch", "--show-current").decode().strip()
    if branch != EXPECTED_BRANCH:
        raise TrainRunnerContractError(f"train-only branch changed: {branch}")
    return {
        "base_commit": BASE_COMMIT,
        "head_commit": head,
        "branch": branch,
        "exact_diff": observed,
        "files": {path: files[path] for path in SOURCE_PATHS},
        "dependency_files": {path: files[path] for path in DEPENDENCY_PATHS},
        "owned_source_clean": True,
    }


def _regular_identity(path: Path, size: int, digest: str) -> dict[str, Any]:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_size != size
        or sha256_file(path) != digest
    ):
        raise TrainRunnerContractError(f"frozen predecessor identity changed: {path}")
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest, "mode": "0444"}


def _exact_passed_gates(value: Mapping[str, Any], expected: Sequence[str]) -> bool:
    gates = value.get("gates")
    return bool(
        isinstance(gates, Mapping)
        and len(gates) == len(expected)
        and set(gates) == set(expected)
        and all(type(gate) is bool and gate for gate in gates.values())
    )


def validate_technical_predecessors() -> dict[str, Any]:
    identities = {
        name: _regular_identity(path, size, digest)
        for name, (path, size, digest) in TECHNICAL_RESULTS.items()
    }
    values = {
        name: json.loads(path.read_bytes())
        for name, (path, _size, _digest) in TECHNICAL_RESULTS.items()
    }
    source = values["source"]
    if (
        source.get("schema") != f"{TECHNICAL_EXPERIMENT_ID}/source-acceptance-v1"
        or source.get("status") != "passed"
        or source.get("mode") != "source_acceptance"
        or source.get("source_identity", {}).get("head_commit") != BASE_COMMIT
        or source.get("target_manifest_opened") is not False
        or source.get("target_npz_opened") != []
        or source.get("fresh_data_opened") is not False
    ):
        raise TrainRunnerContractError("technical source acceptance changed")
    layout = values["layout"]
    if (
        layout.get("schema") != f"{TECHNICAL_EXPERIMENT_ID}/layout-v1"
        or layout.get("status") != "passed"
        or layout.get("mode") != "layout_probe"
        or not _exact_passed_gates(layout, technical_runner.LAYOUT_GATE_KEYS)
        or layout.get("boundary_indices") != [519, 1039]
        or layout.get("target_npz_opened") != []
        or layout.get("fresh_data_opened") is not False
    ):
        raise TrainRunnerContractError("technical layout changed")
    cache = values["cache"]
    if (
        cache.get("schema") != f"{TECHNICAL_EXPERIMENT_ID}/cache-v1"
        or cache.get("status") != "passed"
        or cache.get("mode") != "target_blind_cache"
        or not _exact_passed_gates(cache, technical_runner.CACHE_GATE_KEYS)
        or cache.get("target_npz_opened") != []
        or cache.get("fresh_data_opened") is not False
    ):
        raise TrainRunnerContractError("technical full-grid cache changed")
    cache_files = {
        name: _regular_identity(path, size, digest)
        for name, (path, size, digest) in CACHE_FILES.items()
    }
    if any(
        cache.get("features", {}).get(name, {}).get("sha256") != CACHE_FILES[name][2]
        or Path(str(cache.get("features", {}).get(name, {}).get("path"))).resolve()
        != CACHE_FILES[name][0].resolve()
        for name in CACHE_FILES
    ):
        raise TrainRunnerContractError("technical cache file receipt changed")
    preflight = values["preflight"]
    if (
        preflight.get("schema") != f"{TECHNICAL_EXPERIMENT_ID}/preflight-v1"
        or preflight.get("status") != "passed"
        or preflight.get("mode") != "training_preflight"
        or not _exact_passed_gates(preflight, technical_runner.PREFLIGHT_GATE_KEYS)
        or preflight.get("optimizer_steps") != 20
        or preflight.get("checkpoint_written") is not False
        or [record.get("row_id") for record in preflight.get("target_npz_opened", ())]
        != list(range(30))
        or preflight.get("fresh_data_opened") is not False
    ):
        raise TrainRunnerContractError("technical preflight changed")
    overfit = values["overfit"]
    if (
        overfit.get("schema") != f"{TECHNICAL_EXPERIMENT_ID}/overfit-v1"
        or overfit.get("status") != "passed"
        or overfit.get("mode") != "group0_overfit"
        or not all(type(gate) is bool and gate for gate in overfit.get("gates", {}).values())
        or set(overfit.get("gates", {}))
        != {
            "all_outputs_and_metrics_finite",
            "all_six_modules_changed",
            "all_six_modules_have_nonzero_gradient",
            "all_six_pairs_exact",
            "checkpoint_not_written",
            "donor_substitution_at_least_80pct",
            "exact_500_optimizer_steps",
            "final50_ce_reduced_at_least_90pct",
            "learned_queries_changed",
            "wall_within_30_minutes",
        }
        or overfit.get("optimizer_steps") != 500
        or overfit.get("checkpoint_written") is not False
        or overfit.get("substitution", {}).get("total") != 30
        or [record.get("row_id") for record in overfit.get("target_npz_opened", ())]
        != list(range(6))
        or overfit.get("fresh_data_opened") is not False
    ):
        raise TrainRunnerContractError("technical overfit changed")
    remaining = (
        CUMULATIVE_LIMIT_SECONDS - float(preflight["wall_seconds"]) - float(overfit["wall_seconds"])
    )
    if abs(remaining - REMAINING_SECONDS) > 1e-9:
        raise TrainRunnerContractError("remaining six-hour budget changed")
    return {
        "results": identities,
        "cache_files": cache_files,
        "technical_source_head": BASE_COMMIT,
        "preflight_wall_seconds": preflight["wall_seconds"],
        "overfit_wall_seconds": overfit["wall_seconds"],
        "remaining_seconds": remaining,
    }


def protocol() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "phases": list(PHASES),
        "technical_result_sha256": {name: record[2] for name, record in TECHNICAL_RESULTS.items()},
        "cache_file_sha256": {name: record[2] for name, record in CACHE_FILES.items()},
        "target_manifest_sha256": TARGET_MANIFEST_SHA256,
        "condition_order": list(CONDITION_ORDER),
        "fold_order": list(FOLD_ORDER),
        "folds": {
            str(group): {
                "candidate_groups": list(fold_spec(group).candidate_groups),
                "candidate_rows": list(fold_spec(group).candidate_rows),
                "heldout_rows": list(fold_spec(group).heldout_rows),
                "effective_samples_sha256": effective_samples_sha256(fold_spec(group)),
            }
            for group in FOLD_ORDER
        },
        "train_steps": TRAIN_STEPS,
        "seed": TRAIN_SEED,
        "microbatch_rows": ROWS_PER_GROUP,
        "gradient_accumulation": 5,
        "optimizer": {
            "name": "AdamW",
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 1e-4,
            "warmup_steps": 100,
            "final_lr": 3e-5,
            "gradient_clip": 1.0,
            "schedule_sha256": optimizer_schedule_sha256(),
        },
        "checkpoint_schema": f"{EXPERIMENT_ID}/checkpoint-v1",
        "checkpoint_selection": "24_final_step_1000_only",
        "shuffle": {
            "training_seed": TRAIN_SEED,
            "same_step_permutation_shared_across_five_microbatches": True,
            "evaluation_seeds": list(EVAL_SHUFFLE_SEEDS),
            "evaluation_step": 0,
            "aggregation": "row_argmax_then_seed_within_geometry_then_geometry_equal",
        },
        "ordered_substitution_total": 180,
        "remaining_seconds": REMAINING_SECONDS,
        "artifact_limit_bytes": ARTIFACT_LIMIT,
        "formal_output_root": str(FORMAL_OUTPUT_ROOT),
        "dev_phase": None,
        "fresh_phase": None,
        "action_outputs": 0,
        "retry_count": 0,
    }


def _run_focused_tests(repo: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        "tests/test_mikasa_history_hard_learned_spatial_resampler_mamba_train.py",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repo),
        }
    )
    completed = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise TrainRunnerContractError(
            f"train-only focused tests failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2_000:],
        "cuda_visible_devices": "",
    }


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run_source_acceptance(output_root: Path, repo: Path = REPO) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    identity = source_identity(repo)
    technical = validate_technical_predecessors()
    evidence = validate_frozen_evidence(include_target=False)
    tests = _run_focused_tests(repo)
    result = {
        "schema": f"{EXPERIMENT_ID}/source-acceptance-v1",
        "status": "passed",
        "mode": "source_acceptance",
        "source_identity": identity,
        "technical_predecessors": technical,
        "frozen_evidence": evidence,
        "focused_tests": tests,
        "protocol": protocol(),
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "dev_rows_opened": [],
        "fresh_data_opened": False,
        "gpu_contexts": 0,
        "model_calls": 0,
        "action_outputs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return result


def _require_source_receipt(
    path: Path, expected_sha256: str, repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise PermissionError("source receipt must be one regular mode-0444 file")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if len(expected_sha256) != 64 or observed != expected_sha256:
        raise PermissionError("source receipt identity changed")
    value = json.loads(raw)
    if (
        value.get("schema") != f"{EXPERIMENT_ID}/source-acceptance-v1"
        or value.get("status") != "passed"
        or value.get("mode") != "source_acceptance"
        or value.get("protocol") != protocol()
        or value.get("target_manifest_opened") is not False
        or value.get("target_npz_opened") != []
        or value.get("dev_rows_opened") != []
        or value.get("fresh_data_opened") is not False
        or value.get("retry_count") != 0
    ):
        raise PermissionError("source acceptance boundary changed")
    current_source = source_identity(repo)
    if value.get("source_identity") != current_source:
        raise PermissionError("runtime source identity changed")
    current_technical = validate_technical_predecessors()
    if value.get("technical_predecessors") != current_technical:
        raise PermissionError("runtime technical predecessor identity changed")
    return value, {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": observed,
        "mode": "0444",
    }


def _load_train_labels() -> tuple[np.ndarray, list[dict[str, Any]]]:
    sources = HistoryHardSources.discover()
    labels = np.full((ROWS, 2), -1, dtype=np.int64)
    opened = []
    for row_id in range(36):
        row = sources.rows[row_id]
        row.target.verify(root=CORPUS_ROOT)
        with np.load(row.target.path, allow_pickle=False) as archive:
            if set(archive.files) != set(TARGET_FIELDS):
                raise RuntimeContractError(f"row {row_id} target field allowlist changed")
            target_row = int(np.asarray(archive["row_id"]).item())
            group = int(np.asarray(archive["group_id"]).item())
            pair_code = str(np.asarray(archive["pair_code"]).item())
            ordered = np.asarray(archive["ordered_path_ids"])
        expected_pair = PAIR_CODES[row_id % ROWS_PER_GROUP]
        if (
            target_row != row_id
            or group != row_id // ROWS_PER_GROUP
            or pair_code != expected_pair
            or ordered.shape != (2,)
            or ordered.dtype != np.int64
            or ordered.tolist() != [int(expected_pair[0]), int(expected_pair[1])]
        ):
            raise RuntimeContractError(f"row {row_id} train target changed")
        labels[row_id] = ordered
        opened.append(
            {
                "row_id": row_id,
                "path": str(row.target.path.resolve()),
                "bytes": row.target.bytes,
                "sha256": row.target.sha256,
            }
        )
    if np.any(labels[:36] < 0) or np.any(labels[36:] != -1):
        raise RuntimeContractError("train/dev target boundary changed")
    return labels, opened


def _curve_finite(record: Mapping[str, Any]) -> bool:
    curve = record.get("curve")
    fields = (
        "cross_entropy",
        "pair_accuracy",
        "slot0_accuracy",
        "slot1_accuracy",
        "lr",
        "grad_norm",
    )
    return bool(
        isinstance(curve, list)
        and len(curve) == TRAIN_STEPS
        and all(
            point.get("step") == index + 1
            and all(np.isfinite(float(point[field])) for field in fields)
            for index, point in enumerate(curve)
        )
    )


def _checkpoint_paths(output_root: Path) -> dict[str, dict[int, Path]]:
    return {
        condition: {
            heldout: output_root
            / "checkpoints"
            / condition
            / f"heldout-{heldout}"
            / "checkpoint-final.pt"
            for heldout in FOLD_ORDER
        }
        for condition in CONDITION_ORDER
    }


def _checkpoint_receipts_exact(output_root: Path, records: Sequence[Mapping[str, Any]]) -> bool:
    expected = {
        (condition, heldout): path.resolve()
        for condition, folds in _checkpoint_paths(output_root).items()
        for heldout, path in folds.items()
    }
    seen = set()
    for record in records:
        key = (record.get("condition"), record.get("heldout_group"))
        receipt = record.get("checkpoint")
        if key not in expected or key in seen or not isinstance(receipt, Mapping):
            return False
        path = Path(str(receipt.get("path")))
        if (
            path.resolve() != expected[key]
            or path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or path.stat().st_size != receipt.get("bytes")
            or sha256_file(path) != receipt.get("sha256")
        ):
            return False
        seen.add(key)
    actual = {path.resolve() for path in (output_root / "checkpoints").rglob("*.pt")}
    return seen == set(expected) and actual == set(expected.values())


def run_train(
    output_root: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
    phase_started: float | None = None,
) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    started = time.monotonic() if phase_started is None else float(phase_started)
    deadline = started + REMAINING_SECONDS
    if time.monotonic() >= deadline:
        raise TimeoutError("remaining six-hour train-only budget exhausted before model setup")
    train_source_sha256 = str(provenance.get("source_acceptance", {}).get("sha256", ""))
    if len(train_source_sha256) != 64:
        raise TrainRunnerContractError("train source receipt hash is missing")
    # Freeze the shared initialization and every schedule/evaluator identity in
    # a write-once plan before the target manifest or any target NPZ is opened.
    reference = new_model(CONDITION_ORDER[0], device)
    expected_initial_hash = model_initialization_hash(reference)
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    output_root.mkdir(parents=True)
    frozen_plan_value = {
        "schema": f"{EXPERIMENT_ID}/frozen-train-plan-v1",
        "source_acceptance_sha256": train_source_sha256,
        "technical_source_result_sha256": TECHNICAL_RESULTS["source"][2],
        "technical_cache_result_sha256": TECHNICAL_CACHE_SHA256,
        "cache_file_sha256": {name: record[2] for name, record in CACHE_FILES.items()},
        "target_manifest_sha256_identity_only": TARGET_MANIFEST_SHA256,
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "condition_order": list(CONDITION_ORDER),
        "fold_order": list(FOLD_ORDER),
        "folds": protocol()["folds"],
        "initial_state_sha256": expected_initial_hash,
        "optimizer_schedule_sha256": optimizer_schedule_sha256(),
        "checkpoint_schema": f"{EXPERIMENT_ID}/checkpoint-v1",
        "checkpoint_selection": "24_final_step_1000_only",
        "shuffle": protocol()["shuffle"],
        "evaluator": "evaluate_fold_then_combine_condition_evidence",
        "aggregate": "row_argmax_then_seed_within_geometry_then_geometry_equal",
        "ordered_substitution_total": 180,
        "remaining_seconds": REMAINING_SECONDS,
        "artifact_limit_bytes": ARTIFACT_LIMIT,
        "dev_phase": None,
        "fresh_phase": None,
        "retry_count": 0,
    }
    frozen_plan = write_once(
        output_root / "FROZEN_PLAN.json", canonical_json_bytes(frozen_plan_value)
    )
    target_evidence = validate_frozen_evidence(include_target=True)
    labels, opened = _load_train_labels()
    paths = _checkpoint_paths(output_root)
    records = []
    for condition in CONDITION_ORDER:
        for heldout in FOLD_ORDER:
            if time.monotonic() >= deadline:
                raise TimeoutError("remaining six-hour train-only budget exhausted")
            record = train_fold(
                arrays,
                labels,
                condition=condition,
                heldout_group=heldout,
                device=device,
                deadline=deadline,
                checkpoint_path=paths[condition][heldout],
                source_sha256=train_source_sha256,
                cache_sha256=TECHNICAL_CACHE_SHA256,
                expected_initial_hash=expected_initial_hash,
            )
            records.append(record)
            if _directory_bytes(output_root) > ARTIFACT_LIMIT:
                raise RuntimeError("train-only artifact exceeded 512 MiB")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if not _checkpoint_receipts_exact(output_root, records):
        raise TrainRunnerContractError("24 final checkpoint receipts changed")
    freeze_directory(output_root / "checkpoints")

    evaluations: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITION_ORDER}
    ordered_models = {}
    records_by_key = {
        (record["condition"], int(record["heldout_group"])): record for record in records
    }
    for condition in CONDITION_ORDER:
        for heldout in FOLD_ORDER:
            if time.monotonic() >= deadline:
                raise TimeoutError("remaining six-hour train-only budget exhausted")
            record = records_by_key[(condition, heldout)]
            model = load_checkpoint(
                paths[condition][heldout],
                condition=condition,
                heldout_group=heldout,
                source_sha256=train_source_sha256,
                cache_sha256=TECHNICAL_CACHE_SHA256,
                device=device,
                expected_receipt=record["checkpoint"],
            )
            evaluations[condition].append(
                evaluate_fold(
                    model,
                    arrays,
                    labels,
                    condition=condition,
                    heldout_group=heldout,
                    device=device,
                    deadline=deadline,
                )
            )
            if condition == "mamba_ordered":
                ordered_models[heldout] = model
            else:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    evidence = {
        condition: combine_condition_evidence(condition, evaluations[condition])
        for condition in CONDITION_ORDER
    }
    substitution = evaluate_ordered_substitutions(
        ordered_models, arrays, labels, device=device, deadline=deadline
    )
    del ordered_models
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if time.monotonic() >= deadline:
        raise TimeoutError("remaining six-hour budget exhausted before scientific aggregation")
    wall = time.monotonic() - started

    expected_order = [
        (condition, heldout) for condition in CONDITION_ORDER for heldout in FOLD_ORDER
    ]
    observed_order = [(record.get("condition"), record.get("heldout_group")) for record in records]
    initial_hashes = {record.get("initial_state_sha256") for record in records}
    schedule_hash = optimizer_schedule_sha256()
    effective_matched = all(
        len(
            {
                record.get("effective_samples_sha256")
                for record in records
                if record.get("heldout_group") == heldout
            }
        )
        == 1
        and next(
            record.get("effective_samples_sha256")
            for record in records
            if record.get("heldout_group") == heldout
        )
        == effective_samples_sha256(fold_spec(heldout))
        for heldout in FOLD_ORDER
    )
    ordered_records = [record for record in records if record.get("condition") == "mamba_ordered"]
    expected_checkpoint_bytes = _directory_bytes(output_root)
    technical_gates = {
        "condition_fold_order_exact": observed_order == expected_order,
        "twenty_four_final_checkpoints_exact": len(records) == 24
        and _checkpoint_receipts_exact(output_root, records),
        "all_steps_exact": all(record.get("steps") == TRAIN_STEPS for record in records),
        "candidate_heldout_isolation_exact": all(
            record.get("candidate_groups")
            == list(fold_spec(record["heldout_group"]).candidate_groups)
            and record.get("candidate_rows")
            == list(fold_spec(record["heldout_group"]).candidate_rows)
            and record.get("heldout_rows") == list(fold_spec(record["heldout_group"]).heldout_rows)
            for record in records
        ),
        "all_train_targets_opened_once": [record["row_id"] for record in opened] == list(range(36)),
        "pre_target_plan_frozen_exact": Path(str(frozen_plan.get("path"))).resolve()
        == (output_root / "FROZEN_PLAN.json").resolve()
        and frozen_plan.get("mode") == "0444"
        and stat.S_IMODE((output_root / "FROZEN_PLAN.json").stat().st_mode) == 0o444
        and (output_root / "FROZEN_PLAN.json").stat().st_size == frozen_plan.get("bytes")
        and sha256_file(output_root / "FROZEN_PLAN.json") == frozen_plan.get("sha256")
        and frozen_plan_value["target_manifest_opened"] is False
        and frozen_plan_value["target_npz_opened"] == [],
        "shared_initialization_exact": initial_hashes == {expected_initial_hash},
        "optimizer_schedule_exact": all(
            record.get("optimizer_schedule_sha256") == schedule_hash for record in records
        ),
        "effective_samples_matched_across_conditions": effective_matched,
        "curves_logits_gradients_parameters_finite": all(
            _curve_finite(record) and record.get("gradients_finite") is True for record in records
        )
        and all(
            result.get("finite") is True for values in evaluations.values() for result in values
        ),
        "all_checkpoint_reload_parity_exact": all(
            record.get("reload_parity") is True for record in records
        ),
        "ordered_all_modules_gradient_and_change": all(
            all(record.get("gradient_seen", {}).values())
            and all(record.get("module_changed", {}).values())
            and record.get("module_changed", {}).get("learned_queries") is True
            for record in ordered_records
        ),
        "heldout_denominators_complete": all(
            set(section.get("by_group", {})) == {str(group)}
            and section["by_group"][str(group)].get("rows") == ROWS_PER_GROUP
            for condition in CONDITION_ORDER
            for group, section in enumerate(
                [result["section"] for result in evaluations[condition]]
            )
        ),
        "ordered_substitution_denominator_180": substitution.get("total") == 180,
        "target_manifest_opened_once": target_evidence.get("target_manifest_opened") is True
        and target_evidence.get("target_manifest", {}).get("sha256") == TARGET_MANIFEST_SHA256,
        "dev_fresh_action_outputs_zero": all(record["row_id"] < 36 for record in opened)
        and target_evidence.get("fresh_data_opened") is False,
        "wall_within_remaining_six_hour_budget": wall <= REMAINING_SECONDS,
        "artifact_within_512mib": expected_checkpoint_bytes <= ARTIFACT_LIMIT,
        "zero_retry": output_root.resolve() == FORMAL_OUTPUT_ROOT.resolve(),
    }
    if tuple(technical_gates) != TRAIN_GATE_KEYS:
        raise TrainRunnerContractError("train technical gate set/order changed")
    scientific = train_scientific_gates(evidence, substitution)
    result = {
        "schema": f"{EXPERIMENT_ID}/train-v1",
        "status": "passed"
        if all(technical_gates.values()) and all(scientific.values())
        else "gate_failed",
        "mode": "train_only",
        "condition_order": list(CONDITION_ORDER),
        "fold_order": list(FOLD_ORDER),
        "seed": TRAIN_SEED,
        "steps_per_fold": TRAIN_STEPS,
        "frozen_plan": frozen_plan,
        "records": records,
        "evidence": evidence,
        "ordered_substitution": substitution,
        "technical_gates": technical_gates,
        "scientific_gates": scientific,
        "shared_initialization_sha256": expected_initial_hash,
        "optimizer_schedule_sha256": schedule_hash,
        "target_evidence": target_evidence,
        "target_manifest_opened": True,
        "target_npz_opened": opened,
        "dev_rows_opened": [],
        "fresh_data_opened": False,
        "dev_model_outputs": 0,
        "fresh_model_outputs": 0,
        "action_outputs": 0,
        "optimizer_steps": len(records) * TRAIN_STEPS,
        "checkpoint_count": len(records),
        "wall_seconds": wall,
        "wall_limit_seconds": REMAINING_SECONDS,
        "artifact_limit_bytes": ARTIFACT_LIMIT,
        "provenance": dict(provenance),
        "retry_count": 0,
    }
    payload = canonical_json_bytes(result)
    if time.monotonic() >= deadline:
        raise TimeoutError("remaining six-hour budget exhausted before result commit")
    if _directory_bytes(output_root) + len(payload) > ARTIFACT_LIMIT:
        raise RuntimeError("train-only artifact exceeded 512 MiB")
    write_once(output_root / "RESULT.json", payload)
    freeze_directory(output_root)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"missing required {label}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.phase == "source_acceptance":
        result = run_source_acceptance(args.output_root, args.repo_root)
    else:
        phase_started = time.monotonic()
        if args.output_root.resolve() != FORMAL_OUTPUT_ROOT.resolve():
            raise PermissionError("formal train output-root changed")
        source, source_receipt = _require_source_receipt(
            _required(args.source_result, "--source-result"),
            _required(args.source_sha256, "--source-sha256"),
            args.repo_root,
        )
        del source
        if args.cache_root.resolve() != CACHE_ROOT.resolve():
            raise PermissionError("train cache-root changed")
        arrays, _sample_ids, cache, cache_receipt = technical_runner._load_cache(
            args.cache_root, TECHNICAL_CACHE_SHA256
        )
        if cache_receipt.get("sha256") != TECHNICAL_CACHE_SHA256:
            raise PermissionError("technical cache receipt changed")
        device, runtime = technical_runner._formal_runtime(args.device)
        result = run_train(
            args.output_root,
            arrays=arrays,
            device=device,
            runtime=runtime,
            provenance={
                "source_acceptance": source_receipt,
                "technical_cache": cache_receipt,
            },
            phase_started=phase_started,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
