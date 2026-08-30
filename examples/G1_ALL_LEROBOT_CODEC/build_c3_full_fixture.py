#!/usr/bin/env python3
"""Materialize one frozen C3-R2 Codec selection, sequentially on CPU.

``--preflight`` is read-only. ``--execute-initial`` consumes the one fixed
initial attempt. A later run may use ``--manual-resume`` only with an explicit
new ``--attempt-id`` and exact ``--resume-from`` attempt. There is no automatic
retry, latest-attempt lookup, network use, GPU path, training, or evaluator.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
    validate_codec_contract,
)
from gr00t.data.codec_evidence.contract import MV_FIELD_DTYPES
from gr00t.data.codec_evidence.episode_bundle import (
    SOURCE_KIND_NATIVE_CANONICAL,
    EpisodeBundleKey,
    EpisodeBundleStore,
)
from gr00t.data.codec_evidence.identity import EPISODE_IDENTITY_SCHEMA_VERSION
from gr00t.data.codec_evidence.production import (
    CAMERA_ID,
    CRF,
    GOP,
    X264_CAUSAL_PARAMETERS,
    CanonicalEpisodeInput,
    assert_torchcodec_pyav_exact,
    build_episode_arrays,
    decode_canonical_pyav,
    encode_h264_twice,
    extract_mvs_and_decoded_rgb,
    probe_packets,
    probe_x264_encoder_options,
    raster_sha256,
    sha256_path,
    toolchain_identity,
    trace_h264_headers,
    validate_canonical_parquet,
    verify_canonical_input_identity,
)
from gr00t.data.codec_evidence.warehouse import (
    ATTEMPT_SCHEMA_VERSION,
    FINAL_MANIFEST_FILENAME,
    AttemptJournal,
    PublishedObject,
    canonical_json_bytes,
    canonical_json_sha256,
    commit_full_fixture_manifest,
    derivation_sha256,
    publish_episode_object,
)
import numpy as np


SCHEMA_VERSION = "g1-all-lerobot-c3-full-fixture-execution-r2-v1"
DERIVATION_SCHEMA_VERSION = "g1-all-lerobot-c3-episode-derivation-r2-v1"
FIXTURE_ID = "g1-all-lerobot-c3-full-r2"
INITIAL_ATTEMPT_ID = "G1-ALL-LEROBOT-C3-FULL-R2-ATTEMPT1"
EXPECTED_OUTPUT_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-full-r2")
PRODUCER_CLI_RELATIVE = Path("examples/G1_ALL_LEROBOT_CODEC/build_c3_full_fixture.py")
PREPARATION_PLAN_RELATIVE = Path("examples/G1_ALL_LEROBOT_CODEC/c3_full_fixture_plan.json")
INPUT_FREEZE_RELATIVE = Path("examples/G1_ALL_LEROBOT_CODEC/c3_full_input_freeze.json")
PREPARATION_PLAN_FILE_SHA256 = "1e9e34e3134429792e4b8bd2573a3b68e2a30a6349accc6b60aedc52c86749cb"
INPUT_FREEZE_FILE_SHA256 = "f2e950071eb6187be7747658ae46598b8a3f741c37b710a6e2288c041ea7b9cd"
EXPECTED_BASELINE = "dd86aa8deb7c5ee314ca0cf082ab9468b0b61119"
PREPARATION_SCHEMA_VERSION = "g1-all-lerobot-c3-full-fixture-plan-v1"
PREPARATION_STATUS = "C3_PREPARATION"
CONTRACT_IDENTITY_VERSION = "c3-r2"
CAUSAL_PROOF_REQUIRED = False
CAUSAL_PROOF_PATH = Path(
    "/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-contract-r1/"
    "CAUSAL_PREFIX_EQUIVALENCE.json"
)
CAUSAL_PROOF_FILE_SHA256 = "44a0e88cd1577dee942e84fc9420c111b86754957e966d39d155d87446c903c9"
FROZEN_EPISODES = 806
FROZEN_FRAMES = 297_340
FROZEN_CODEC_ANCHORS = 10_291
EXPECTED_EPISODES = FROZEN_EPISODES
EXPECTED_FRAMES = FROZEN_FRAMES
EXPECTED_CODEC_ANCHORS = FROZEN_CODEC_ANCHORS
EPISODE_SELECTION: tuple[tuple[str, int], ...] | None = None
PUBLISH_FULL_FIXTURE_MANIFEST = True
COVERAGE_SCOPE = "EXACT_FROZEN_806_COMPLETE_EPISODES_FULL_TECHNICAL_FIXTURE_R2"
PROFILE_NAME = "full-r2"
MAX_WALL_SECONDS = 10_800
MAX_NEW_BYTES = 30_000_000_000
MAX_PEAK_RSS_BYTES = 16_000_000_000
MAX_SCRATCH_BYTES = 256_000_000
MAX_WORKERS = 1
AUTOMATIC_RETRIES = 0
FINAL_COMMIT_RESERVE_BYTES = 16_000_000
FINAL_COMMIT_WALL_RESERVE_SECONDS = 900.0
PUBLISH_OVERHEAD_RESERVE_BYTES = 1_000_000
SUPERVISOR_POLL_SECONDS = 0.25
SUPERVISOR_TERM_GRACE_SECONDS = 10.0

MARKERS = (
    Path("/home/sl/groot_datasets/catalog/SYNC_IN_PROGRESS.json"),
    Path("/home/sl/groot_datasets/catalog/INDEX_IN_PROGRESS.json"),
    Path("/home/sl/groot_datasets/catalog/STALE_AFTER_SYNC.json"),
)
TERMINAL_NAMES = ("ATTEMPT_TERMINAL.json",)


def configure_memory_canary_r2() -> None:
    """Select the fixed three-episode R2 live-RSS canary profile."""

    global SCHEMA_VERSION, FIXTURE_ID, INITIAL_ATTEMPT_ID, EXPECTED_OUTPUT_ROOT
    global PRODUCER_CLI_RELATIVE, EXPECTED_EPISODES, EXPECTED_FRAMES
    global EXPECTED_CODEC_ANCHORS, EPISODE_SELECTION, PUBLISH_FULL_FIXTURE_MANIFEST
    global COVERAGE_SCOPE, PROFILE_NAME, MAX_WALL_SECONDS, MAX_NEW_BYTES
    SCHEMA_VERSION = "g1-all-lerobot-c3-memory-canary-r2-v1"
    FIXTURE_ID = "g1-all-lerobot-c3-memory-canary-r2"
    INITIAL_ATTEMPT_ID = "G1-ALL-LEROBOT-C3-MEMORY-CANARY-R2-ATTEMPT1"
    EXPECTED_OUTPUT_ROOT = Path(
        "/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-c3-memory-canary-r2"
    )
    PRODUCER_CLI_RELATIVE = Path("examples/G1_ALL_LEROBOT_CODEC/build_c3_r2_memory_canary.py")
    EPISODE_SELECTION = (
        ("no_torque_100", 64),
        ("no_torque_100", 65),
        ("aug1p2mm_220", 7),
    )
    EXPECTED_EPISODES = 3
    EXPECTED_FRAMES = 2_432
    EXPECTED_CODEC_ANCHORS = 83
    PUBLISH_FULL_FIXTURE_MANIFEST = False
    COVERAGE_SCOPE = "FIXED_THREE_EPISODE_LIVE_RSS_MEMORY_CANARY_NOT_FULL_COVERAGE"
    PROFILE_NAME = "memory-canary-r2"
    MAX_WALL_SECONDS = 7_200
    MAX_NEW_BYTES = 5_000_000_000


def configure_causal_full_r1() -> None:
    """Select the new no-overwrite 806-episode causal fixture profile."""

    global SCHEMA_VERSION, DERIVATION_SCHEMA_VERSION, FIXTURE_ID, INITIAL_ATTEMPT_ID
    global EXPECTED_OUTPUT_ROOT, PREPARATION_PLAN_RELATIVE, PREPARATION_PLAN_FILE_SHA256
    global PREPARATION_SCHEMA_VERSION, PREPARATION_STATUS, CONTRACT_IDENTITY_VERSION
    global CAUSAL_PROOF_REQUIRED, COVERAGE_SCOPE, PROFILE_NAME
    SCHEMA_VERSION = "g1-all-lerobot-causal-full-fixture-execution-r1-v1"
    DERIVATION_SCHEMA_VERSION = "g1-all-lerobot-causal-episode-derivation-r1-v1"
    FIXTURE_ID = "g1-all-lerobot-causal-full-r1"
    INITIAL_ATTEMPT_ID = "G1-ALL-LEROBOT-CAUSAL-FULL-R1-ATTEMPT1"
    EXPECTED_OUTPUT_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r1")
    PREPARATION_PLAN_RELATIVE = Path("examples/G1_ALL_LEROBOT_CODEC/causal_full_fixture_plan.json")
    PREPARATION_PLAN_FILE_SHA256 = (
        "f982d30ecc114dd88ee8ee19e5e907456408e116d9bded63f40120c33955948a"
    )
    PREPARATION_SCHEMA_VERSION = "g1-all-lerobot-causal-full-fixture-plan-v1"
    PREPARATION_STATUS = "CAUSAL_PREFIX_EQUIVALENCE_PROVEN"
    CONTRACT_IDENTITY_VERSION = "causal-r1"
    CAUSAL_PROOF_REQUIRED = True
    COVERAGE_SCOPE = "EXACT_FROZEN_806_COMPLETE_EPISODES_CAUSAL_PREFIX_EQUIVALENT_FIXTURE_R1"
    PROFILE_NAME = "causal-full-r1"


def configure_causal_full_r2() -> None:
    """Select the race-safe no-overwrite 806-episode causal fixture profile."""

    configure_causal_full_r1()
    global SCHEMA_VERSION, DERIVATION_SCHEMA_VERSION, FIXTURE_ID, INITIAL_ATTEMPT_ID
    global EXPECTED_OUTPUT_ROOT, PREPARATION_PLAN_RELATIVE, PREPARATION_PLAN_FILE_SHA256
    global PREPARATION_SCHEMA_VERSION, PREPARATION_STATUS, CONTRACT_IDENTITY_VERSION
    global COVERAGE_SCOPE, PROFILE_NAME
    SCHEMA_VERSION = "g1-all-lerobot-causal-full-fixture-execution-r2-v1"
    DERIVATION_SCHEMA_VERSION = "g1-all-lerobot-causal-episode-derivation-r2-v1"
    FIXTURE_ID = "g1-all-lerobot-causal-full-r2"
    INITIAL_ATTEMPT_ID = "G1-ALL-LEROBOT-CAUSAL-FULL-R2-ATTEMPT1"
    EXPECTED_OUTPUT_ROOT = Path("/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r2")
    PREPARATION_PLAN_RELATIVE = Path(
        "examples/G1_ALL_LEROBOT_CODEC/causal_full_fixture_plan_r2.json"
    )
    PREPARATION_PLAN_FILE_SHA256 = (
        "83e8ada3fd1cc206b5a1e7b72cf410b3904ee531db13285cffd25a73b0a094a2"
    )
    PREPARATION_SCHEMA_VERSION = "g1-all-lerobot-causal-full-fixture-plan-r2-v1"
    PREPARATION_STATUS = "CAUSAL_PREFIX_EQUIVALENCE_PROVEN_RACE_SAFE_SUPERVISION"
    CONTRACT_IDENTITY_VERSION = "causal-r2"
    COVERAGE_SCOPE = "EXACT_FROZEN_806_COMPLETE_EPISODES_CAUSAL_PREFIX_EQUIVALENT_FIXTURE_R2"
    PROFILE_NAME = "causal-full-r2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"JSON input must be a regular non-symlink file: {requested}")
    with requested.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {requested}")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def _append_event(path: Path, event: str, **fields: Any) -> None:
    value = {"event": event, "utc_time": _utc_now(), **fields}
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _file_identity(path: str | Path) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"Identity input must be a regular non-symlink file: {requested}")
    resolved = requested.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_path(resolved),
    }


def _verify_file_closure(closure: Mapping[str, Mapping[str, Any]]) -> None:
    for role, expected in closure.items():
        if _file_identity(expected["path"]) != dict(expected):
            raise RuntimeError(f"Frozen implementation file drifted: {role}")


def _verify_toolchain_identity(identity: Mapping[str, Any]) -> None:
    expected_digest = identity.get("identity_sha256")
    observed_digest = canonical_json_sha256(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    if expected_digest != observed_digest:
        raise RuntimeError("Frozen toolchain identity digest is internally inconsistent")
    seen: set[Path] = set()

    def visit(value: Any, role: str) -> None:
        if isinstance(value, Mapping):
            path_value = value.get("path")
            digest = value.get("sha256")
            if isinstance(path_value, str) and isinstance(digest, str):
                path = Path(path_value)
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"Frozen toolchain path disappeared: {role}:{path}")
                resolved = path.resolve(strict=True)
                if resolved not in seen:
                    seen.add(resolved)
                    if sha256_path(resolved) != digest:
                        raise RuntimeError(f"Frozen toolchain file drifted: {role}:{resolved}")
            for key, item in value.items():
                visit(item, f"{role}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{role}.{index}")

    visit(identity, "toolchain")
    if not seen:
        raise RuntimeError("Frozen toolchain contains no exact file closure")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def _git_identity(repository: Path, preparation: Mapping[str, Any]) -> dict[str, Any]:
    dirty = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"C3 execution requires a clean implementation worktree:\n{dirty}")
    branch = _git(repository, "branch", "--show-current")
    if branch != preparation["branch"]:
        raise RuntimeError(f"C3 branch differs from the tracked preparation plan: {branch}")
    baseline = preparation["baseline_commit"]
    if (
        baseline != EXPECTED_BASELINE
        or _git(repository, "merge-base", "HEAD", baseline) != baseline
    ):
        raise RuntimeError("C3 implementation no longer descends from the frozen baseline")
    tracked = _git(
        repository,
        "ls-files",
        "--error-unmatch",
        PREPARATION_PLAN_RELATIVE.as_posix(),
        INPUT_FREEZE_RELATIVE.as_posix(),
    ).splitlines()
    if set(tracked) != {
        PREPARATION_PLAN_RELATIVE.as_posix(),
        INPUT_FREEZE_RELATIVE.as_posix(),
    }:
        raise RuntimeError("C3 preparation plan and input freeze must both be tracked")
    return {
        "repository": str(repository.resolve(strict=True)),
        "branch": branch,
        "head": _git(repository, "rev-parse", "HEAD"),
        "baseline": baseline,
        "dirty_paths": [],
    }


def _validate_causal_prefix_proof(repository: Path) -> dict[str, Any] | None:
    if not CAUSAL_PROOF_REQUIRED:
        return None
    identity = _file_identity(CAUSAL_PROOF_PATH)
    if identity["sha256"] != CAUSAL_PROOF_FILE_SHA256:
        raise RuntimeError("Frozen causal prefix-equivalence proof identity drifted")
    proof = _load_json(CAUSAL_PROOF_PATH)
    if not (
        proof.get("schema_version") == "g1-all-lerobot-causal-prefix-equivalence-proof-v1"
        and proof.get("status") == "PASS"
        and proof.get("technical_valid") is True
        and proof.get("tests", {})
        .get("legacy_negative_control", {})
        .get("known_noncausal_detected")
        is True
        and proof.get("rc_lookahead_zero_proof", {}).get("empirical_prefix_equivalence") is True
    ):
        raise RuntimeError("Causal prefix-equivalence proof is not an accepted PASS")
    detached = dict(proof)
    embedded = detached.pop("proof_payload_sha256", None)
    if embedded != canonical_json_sha256(detached):
        raise RuntimeError("Causal prefix-equivalence proof digest is internally inconsistent")
    proof_head = proof.get("git", {}).get("head")
    if (
        not isinstance(proof_head, str)
        or _git(repository, "merge-base", "HEAD", proof_head) != proof_head
    ):
        raise RuntimeError("Causal proof Git identity is not an ancestor of this execution")
    protected = (
        "gr00t/data/codec_evidence/codec_contract.py",
        "gr00t/data/codec_evidence/production.py",
        "gr00t/data/codec_evidence/schemas/codec-evidence-global-contract-v2.schema.json",
        "examples/G1_ALL_LEROBOT_CODEC/prove_causal_prefix_equivalence.py",
        "tests/codec_evidence/test_production_fixture.py",
        "tests/codec_evidence/test_episode_bundle.py",
        "tests/codec_evidence/test_dataset_protocol.py",
    )
    changed = _git(repository, "diff", "--name-only", f"{proof_head}..HEAD", "--", *protected)
    if changed:
        raise RuntimeError(f"Causal proof-protected implementation changed after proof: {changed}")
    if proof.get("x264_requested_parameters") != list(X264_CAUSAL_PARAMETERS):
        raise RuntimeError("Causal proof x264 request differs from the production implementation")
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
    )
    if proof.get("contract") != contract or proof.get("contract_sha256") != codec_contract_sha256(
        contract
    ):
        raise RuntimeError("Causal proof semantic contract differs from the current contract")
    return identity


def _check_dataset_markers() -> None:
    present = [str(path) for path in MARKERS if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"Dataset generation marker blocks C3 read: {present}")


def _force_cpu_offline_environment() -> None:
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        }
    )


def _contained_payload(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Frozen dataset payload path escapes its source root: {relative_path}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"Frozen dataset payload may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"Frozen dataset payload is not a contained regular file: {relative_path}")
    return resolved


def _validate_preparation_and_freeze(
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preparation_path = (repository / PREPARATION_PLAN_RELATIVE).resolve(strict=True)
    freeze_path = (repository / INPUT_FREEZE_RELATIVE).resolve(strict=True)
    preparation_identity = _file_identity(preparation_path)
    freeze_identity = _file_identity(freeze_path)
    if preparation_identity["sha256"] != PREPARATION_PLAN_FILE_SHA256:
        raise RuntimeError("Tracked C3 preparation plan file identity drifted")
    if freeze_identity["sha256"] != INPUT_FREEZE_FILE_SHA256:
        raise RuntimeError("Tracked C3 input freeze file identity drifted")
    preparation = _load_json(preparation_path)
    frozen = _load_json(freeze_path)
    if preparation.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ValueError("Unsupported C3 preparation plan schema")
    if preparation.get("status") != PREPARATION_STATUS:
        raise ValueError("C3 preparation plan is not frozen preparation state")
    if CAUSAL_PROOF_REQUIRED and preparation.get("causal_prefix_equivalence_proof") != {
        "path": str(CAUSAL_PROOF_PATH),
        "sha256": CAUSAL_PROOF_FILE_SHA256,
    }:
        raise ValueError("Causal preparation plan does not bind the accepted prefix proof")
    if preparation.get("input_freeze") != {
        "path": INPUT_FREEZE_RELATIVE.as_posix(),
        "file_sha256": INPUT_FREEZE_FILE_SHA256,
        "content_sha256": frozen.get("content_sha256"),
        "episode_freeze_core_sha256": frozen.get("episode_freeze_core_sha256"),
        "budget_sha256": canonical_json_sha256(frozen.get("budget")),
    }:
        raise ValueError("C3 preparation plan does not exactly bind the tracked input freeze")
    detached = dict(frozen)
    content_digest = detached.pop("content_sha256", None)
    if content_digest != canonical_json_sha256(detached):
        raise ValueError("C3 input freeze content digest is internally inconsistent")
    budget = frozen.get("budget")
    expected_budget = {
        "recommended_cumulative_wall_limit_seconds": 10_800,
        "recommended_persistent_limit_bytes": 30_000_000_000,
        "recommended_peak_rss_limit_bytes": 16_000_000_000,
        "recommended_scratch_limit_bytes": 256_000_000,
        "max_workers": 1,
        "automatic_retries": 0,
        "resume": "EXPLICIT_MANUAL_ONLY_SAME_PLAN",
    }
    if not isinstance(budget, dict) or any(
        budget.get(key) != value for key, value in expected_budget.items()
    ):
        raise ValueError("Tracked C3 budget differs from the authorized exact limits")
    if (
        frozen.get("episode_count") != FROZEN_EPISODES
        or frozen.get("total_frames") != FROZEN_FRAMES
        or frozen.get("total_codec_anchors") != FROZEN_CODEC_ANCHORS
        or frozen.get("camera_id") != CAMERA_ID
        or frozen.get("source_kind") != SOURCE_KIND_NATIVE_CANONICAL
    ):
        raise ValueError("Tracked C3 coverage differs from the exact 806-episode scope")
    episodes = frozen.get("episodes")
    sources = frozen.get("sources")
    if not isinstance(episodes, list) or not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("C3 freeze source/episode closure is malformed")
    expected_order = sorted(
        episodes,
        key=lambda item: (item.get("recipe_order"), item.get("episode_index")),
    )
    if episodes != expected_order:
        raise ValueError("C3 episodes are not in frozen recipe/index order")
    uids = [item.get("episode_uid") for item in episodes]
    if len(set(uids)) != FROZEN_EPISODES or canonical_json_sha256(sorted(uids)) != frozen.get(
        "episode_uid_digest"
    ):
        raise ValueError("C3 frozen UID closure is inconsistent")
    core = {
        "schema_version": "g1-all-lerobot-c3-episode-input-freeze-core-v1",
        "camera_id": CAMERA_ID,
        "episode_identity_schema_version": EPISODE_IDENTITY_SCHEMA_VERSION,
        "episodes": [
            {
                "source_id": item["source_id"],
                "episode_index": item["episode_index"],
                "episode_uid": item["episode_uid"],
                "episode_identity": item["episode_identity"],
            }
            for item in episodes
        ],
    }
    if canonical_json_sha256(core) != frozen.get("episode_freeze_core_sha256"):
        raise ValueError("C3 episode freeze core digest is inconsistent")
    return (
        preparation,
        frozen,
        {
            "preparation_plan": preparation_identity,
            "input_freeze": freeze_identity,
        },
    )


def _select_execution_episodes(frozen: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(frozen)
    if EPISODE_SELECTION is None:
        episodes = list(frozen["episodes"])
    else:
        by_key = {(item["source_id"], item["episode_index"]): item for item in frozen["episodes"]}
        if len(by_key) != FROZEN_EPISODES:
            raise RuntimeError("Frozen episode keys are not unique")
        try:
            episodes = [by_key[key] for key in EPISODE_SELECTION]
        except KeyError as error:
            raise RuntimeError(
                f"R2 execution selection is absent from the freeze: {error}"
            ) from error
    if (
        len(episodes) != EXPECTED_EPISODES
        or sum(item["num_frames"] for item in episodes) != EXPECTED_FRAMES
        or sum(item["gop30_anchors"] for item in episodes) != EXPECTED_CODEC_ANCHORS
    ):
        raise RuntimeError("R2 execution selection differs from its exact count closure")
    selected["episodes"] = episodes
    selected["episode_count"] = len(episodes)
    selected["total_frames"] = sum(item["num_frames"] for item in episodes)
    selected["total_codec_anchors"] = sum(item["gop30_anchors"] for item in episodes)
    selected["episode_uid_digest"] = canonical_json_sha256(
        sorted(item["episode_uid"] for item in episodes)
    )
    return selected


def _implementation_closure(repository: Path) -> dict[str, dict[str, Any]]:
    relative_paths = {
        "producer_cli": PRODUCER_CLI_RELATIVE.as_posix(),
        "execution_engine": "examples/G1_ALL_LEROBOT_CODEC/build_c3_full_fixture.py",
        "preparation_plan": PREPARATION_PLAN_RELATIVE.as_posix(),
        "input_freeze": INPUT_FREEZE_RELATIVE.as_posix(),
        "production_primitives": "gr00t/data/codec_evidence/production.py",
        "warehouse": "gr00t/data/codec_evidence/warehouse.py",
        "episode_bundle_writer": "gr00t/data/codec_evidence/episode_bundle.py",
        "codec_contract": "gr00t/data/codec_evidence/codec_contract.py",
        "global_codec_contract_schema": (
            "gr00t/data/codec_evidence/schemas/codec-evidence-global-contract-v2.schema.json"
        ),
        "causal_prefix_proof_cli": (
            "examples/G1_ALL_LEROBOT_CODEC/prove_causal_prefix_equivalence.py"
        ),
        "episode_identity": "gr00t/data/codec_evidence/identity.py",
        "adapter_identity": "gr00t/data/dataset/g1_all_lerobot_h40_adapter.py",
        "mv_normalization_reference": "gr00t/model/gr00t_n1d7/codec_dynamics.py",
        "bundle_schema": (
            "gr00t/data/codec_evidence/schemas/codec-evidence-episode-bundle-v1.schema.json"
        ),
        "full_manifest_schema": (
            "gr00t/data/codec_evidence/schemas/codec-evidence-full-fixture-manifest-v1.schema.json"
        ),
        "test_c3_cli": "tests/codec_evidence/test_c3_full_fixture_cli.py",
        "test_production_fixture": "tests/codec_evidence/test_production_fixture.py",
        "test_warehouse": "tests/codec_evidence/test_warehouse.py",
        "project_configuration": "pyproject.toml",
    }
    return {
        role: _file_identity(repository / relative) for role, relative in relative_paths.items()
    }


def _compact_identity(name: str, value: Any) -> dict[str, str]:
    return {
        "name": name,
        "version": CONTRACT_IDENTITY_VERSION,
        "sha256": canonical_json_sha256(value),
    }


def _codec_contract(toolchain: Mapping[str, Any]) -> dict[str, Any]:
    encoder = {
        "ffmpeg": toolchain["ffmpeg"],
        "libx264": toolchain["libx264"],
        "recipe": {
            "codec": "h264",
            "container": "mp4",
            "pixel_format": "yuv420p",
            "preset": "medium",
            "crf": CRF,
            "gop": GOP,
            "keyint": GOP,
            "min_keyint": 16,
            "bframes": 0,
            "refs": 1,
            "threads": 1,
            "lookahead_threads": 1,
            "sliced_threads": 0,
            "sync_lookahead": 0,
            "rc_lookahead": 0,
            "mbtree": 0,
            "time_base": "1/30000",
            "pts_step": 1000,
        },
    }
    decoder = {
        "pyav": toolchain["pyav"],
        "torchcodec": toolchain["torchcodec"],
        "parity": "EVERY_FROZEN_EPISODE_ROW_EXACT_UINT8_RGB_EQUAL",
    }
    extractor = {
        "motion_vector_extractor": toolchain["motion_vector_extractor"],
        "mv_fields": list(MV_FIELD_DTYPES),
        "mv_term": "codec encoder decision; not optical flow",
        "rpe_term": "decoded-luma MV-warp proxy; not bitstream residual",
        "aggregation_reference": toolchain["implementation"]["mv_normalization_reference"],
    }
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
    )
    contract.update(
        {
            "tool_identity": _compact_identity("g1-all-lerobot-c3-toolchain", toolchain),
            "encoder_identity": _compact_identity("g1-all-lerobot-c3-encoder", encoder),
            "decoder_identity": _compact_identity("g1-all-lerobot-c3-decoder", decoder),
            "extractor_identity": _compact_identity("g1-all-lerobot-c3-mv-extractor", extractor),
        }
    )
    return validate_codec_contract(contract)


def _output_filesystem_identity() -> dict[str, Any]:
    parent = EXPECTED_OUTPUT_ROOT.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"C3 output parent must be a real directory: {parent}")
    usage = shutil.disk_usage(parent)
    if usage.free < MAX_NEW_BYTES:
        raise RuntimeError(f"C3 filesystem has less than the authorized 30 GB budget: {usage.free}")
    return {
        "output_parent": str(parent.resolve(strict=True)),
        "required_free_bytes": MAX_NEW_BYTES,
    }


def _verify_all_frozen_inputs(frozen: Mapping[str, Any]) -> None:
    """Re-hash the entire read-only metadata and 806-episode payload closure."""

    _check_dataset_markers()
    sources = {item["source_id"]: item for item in frozen["sources"]}
    for source in frozen["sources"]:
        _verify_source_metadata(source)
    total_frames = 0
    for row in frozen["episodes"]:
        episode = _episode_input(row, sources[row["source_id"]])
        verify_canonical_input_identity(episode, stage="FULL_FROZEN_INPUT_CLOSURE")
        total_frames += episode.num_frames
    if total_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"R2 input re-hash did not cover exactly {EXPECTED_FRAMES} selected frames"
        )
    _check_dataset_markers()


def build_execution_plan(repository: Path) -> dict[str, Any]:
    """Freeze clean Git, exact tools, implementation, contract, inputs, and budget."""

    _force_cpu_offline_environment()
    _check_dataset_markers()
    preparation, full_frozen, artifacts = _validate_preparation_and_freeze(repository)
    frozen = _select_execution_episodes(full_frozen)
    git_identity = _git_identity(repository, preparation)
    implementation = _implementation_closure(repository)
    tools = toolchain_identity(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe")
    tools = {
        **{key: value for key, value in tools.items() if key != "identity_sha256"},
        "runtime_python": _file_identity(Path(sys.executable).resolve(strict=True)),
        "outer_timeout": _file_identity("/usr/bin/timeout"),
        "implementation": implementation,
        "git": git_identity,
    }
    tools["identity_sha256"] = canonical_json_sha256(tools)
    contract = _codec_contract(tools)
    causal_proof = _validate_causal_prefix_proof(repository)
    _verify_all_frozen_inputs(frozen)
    expected_lengths = sorted(
        (
            {"episode_uid": item["episode_uid"], "num_frames": item["num_frames"]}
            for item in frozen["episodes"]
        ),
        key=lambda item: item["episode_uid"],
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "output_root": str(EXPECTED_OUTPUT_ROOT),
        "preparation_plan": artifacts["preparation_plan"],
        "input_freeze": artifacts["input_freeze"],
        "input_snapshot_sha256": frozen["content_sha256"],
        "episode_freeze_core_sha256": frozen["episode_freeze_core_sha256"],
        "budget_sha256": preparation["input_freeze"]["budget_sha256"],
        "expected_episode_count": EXPECTED_EPISODES,
        "expected_total_frames": EXPECTED_FRAMES,
        "expected_total_codec_anchors": EXPECTED_CODEC_ANCHORS,
        "expected_episode_uid_digest": frozen["episode_uid_digest"],
        "expected_episode_lengths_sha256": canonical_json_sha256(expected_lengths),
        "camera_id": CAMERA_ID,
        "source_kind": SOURCE_KIND_NATIVE_CANONICAL,
        "execution": {
            "cpu_only": True,
            "network": "offline_no_download",
            "gpu": False,
            "outer_timeout_seconds": MAX_WALL_SECONDS,
            "maximum_cumulative_wall_seconds": MAX_WALL_SECONDS,
            "maximum_new_bytes": MAX_NEW_BYTES,
            "maximum_live_process_tree_rss_bytes": MAX_PEAK_RSS_BYTES,
            "rss_enforcement": "SUPERVISOR_LIVE_PROCESS_TREE_SAMPLED_EVERY_0P25_SECONDS",
            "rusage_diagnostics": ["RUSAGE_SELF_MAXRSS", "RUSAGE_CHILDREN_MAXRSS"],
            "maximum_scratch_bytes_including_encode_and_publish_stage": MAX_SCRATCH_BYTES,
            "max_workers": MAX_WORKERS,
            "automatic_retries": AUTOMATIC_RETRIES,
            "source_inputs_read_only": True,
            "output_filesystem": _output_filesystem_identity(),
        },
        "toolchain": tools,
        "toolchain_identity_sha256": tools["identity_sha256"],
        "implementation_identity_sha256": canonical_json_sha256(implementation),
        "codec_contract": contract,
        "codec_contract_sha256": codec_contract_sha256(contract),
        "causal_prefix_equivalence_proof": causal_proof,
        "scope": COVERAGE_SCOPE,
        "episode_selection": (
            None
            if EPISODE_SELECTION is None
            else [
                {"source_id": source_id, "episode_index": episode_index}
                for source_id, episode_index in EPISODE_SELECTION
            ]
        ),
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return plan


# Compatibility with the smaller C2 command's public naming.
build_plan = build_execution_plan


def _verify_execution_identity(plan: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    detached_plan = dict(plan)
    embedded_digest = detached_plan.pop("plan_sha256", None)
    if embedded_digest != canonical_json_sha256(detached_plan):
        raise RuntimeError("C3 execution plan digest is internally inconsistent")
    preparation, full_frozen, artifacts = _validate_preparation_and_freeze(repository)
    frozen = _select_execution_episodes(full_frozen)
    if (
        artifacts["preparation_plan"] != plan["preparation_plan"]
        or artifacts["input_freeze"] != plan["input_freeze"]
    ):
        raise RuntimeError("Tracked C3 plan/freeze file identities drifted")
    if (
        frozen["content_sha256"] != plan["input_snapshot_sha256"]
        or frozen["episode_freeze_core_sha256"] != plan["episode_freeze_core_sha256"]
        or canonical_json_sha256(frozen["budget"]) != plan["budget_sha256"]
    ):
        raise RuntimeError("Tracked C3 freeze content drifted")
    if _git_identity(repository, preparation) != plan["toolchain"]["git"]:
        raise RuntimeError("Frozen clean Git identity drifted")
    _verify_file_closure(plan["toolchain"]["implementation"])
    _verify_toolchain_identity(plan["toolchain"])
    if _codec_contract(plan["toolchain"]) != plan["codec_contract"]:
        raise RuntimeError("Frozen codec contract drifted from the toolchain")
    if _validate_causal_prefix_proof(repository) != plan.get("causal_prefix_equivalence_proof"):
        raise RuntimeError("Frozen causal prefix-equivalence proof drifted")
    return frozen


def _verify_source_metadata(source: Mapping[str, Any]) -> None:
    root = Path(source["canonical_root"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Frozen source root is missing or a symlink: {root}")
    for name, relative in (
        ("info", "meta/info.json"),
        ("modality", "meta/modality.json"),
        ("episodes", "meta/episodes.jsonl"),
    ):
        expected = source["metadata"][name]
        observed = _file_identity(root / relative)
        if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Frozen source metadata drifted: {source['source_id']}:{name}")


def _episode_input(episode: Mapping[str, Any], source: Mapping[str, Any]) -> CanonicalEpisodeInput:
    root = Path(source["canonical_root"]).resolve(strict=True)
    parquet = _contained_payload(root, episode["parquet"]["relative_path"])
    video = _contained_payload(root, episode["video"]["relative_path"])
    return CanonicalEpisodeInput(
        episode_uid=episode["episode_uid"],
        cohort_id=episode["source_id"],
        dataset_name=root.name,
        source_episode_index=episode["episode_index"],
        num_frames=episode["num_frames"],
        parquet_path=parquet,
        parquet_sha256=episode["parquet"]["sha256"],
        parquet_bytes=episode["parquet"]["bytes"],
        canonical_video_path=video,
        canonical_video_sha256=episode["video"]["sha256"],
        canonical_video_bytes=episode["video"]["bytes"],
    )


def episode_derivation(plan: Mapping[str, Any], episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "episode_uid": episode["episode_uid"],
        "source_id": episode["source_id"],
        "source_episode_index": episode["episode_index"],
        "num_frames": episode["num_frames"],
        "episode_identity": episode["episode_identity"],
        "canonical_parquet": {
            "bytes": episode["parquet"]["bytes"],
            "sha256": episode["parquet"]["sha256"],
        },
        "canonical_video": {
            "bytes": episode["video"]["bytes"],
            "sha256": episode["video"]["sha256"],
        },
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "episode_freeze_core_sha256": plan["episode_freeze_core_sha256"],
        "codec_contract_sha256": plan["codec_contract_sha256"],
        "implementation_identity_sha256": plan["implementation_identity_sha256"],
        "toolchain_identity_sha256": plan["toolchain_identity_sha256"],
        "git_head": plan["toolchain"]["git"]["head"],
    }


def _tree_bytes(path: Path) -> int:
    for attempt in range(3):
        try:
            try:
                root_stat = path.lstat()
            except FileNotFoundError:
                return 0
            if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
                raise ValueError(f"Budget namespace must be a real directory: {path}")
            total = 0
            for item in path.rglob("*"):
                item_stat = item.lstat()
                if stat.S_ISLNK(item_stat.st_mode):
                    raise ValueError(f"Symlink is forbidden in the C3 warehouse: {item}")
                if stat.S_ISREG(item_stat.st_mode):
                    total += item_stat.st_size
                elif not stat.S_ISDIR(item_stat.st_mode):
                    raise ValueError(f"Special file is forbidden in the C3 warehouse: {item}")
            return total
        except FileNotFoundError:
            if attempt == 2:
                raise RuntimeError(f"Budget namespace changed during three scans: {path}")
    raise AssertionError("unreachable")


def _make_tree_user_writable(path: Path) -> None:
    """Make only one exact failed-attempt temporary tree removable.

    Bundle staging intentionally becomes read-only before publication.  If a
    worker is interrupted in that interval, manual resume must be able to
    remove the uncommitted staging inode without weakening committed objects.
    """

    if not path.exists():
        return
    root_stat = path.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"Resume temporary is not a real directory: {path}")
    entries = list(path.rglob("*"))
    for item in reversed(entries):
        item_stat = item.lstat()
        if stat.S_ISLNK(item_stat.st_mode):
            raise RuntimeError(f"Resume temporary contains a symlink: {item}")
        if not (stat.S_ISREG(item_stat.st_mode) or stat.S_ISDIR(item_stat.st_mode)):
            raise RuntimeError(f"Resume temporary contains a special file: {item}")
        item.chmod(
            item_stat.st_mode
            | stat.S_IWUSR
            | (stat.S_IXUSR if stat.S_ISDIR(item_stat.st_mode) else 0)
        )
    path.chmod(root_stat.st_mode | stat.S_IWUSR | stat.S_IXUSR)


def _persistent_bytes(root: Path) -> int:
    total = 0
    for name in ("objects", "derivations", "attempts"):
        total += _tree_bytes(root / name)
    manifest = root / FINAL_MANIFEST_FILENAME
    if manifest.exists():
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError("Final manifest path is not a regular file")
        total += manifest.stat().st_size
    return total


def _scratch_bytes(root: Path) -> int:
    return _tree_bytes(root / ".scratch") + _tree_bytes(root / ".staging")


def _rusage_diagnostics() -> dict[str, int]:
    self_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    children_maxrss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024
    return {
        "rusage_self_maxrss_bytes": self_maxrss,
        "rusage_children_maxrss_bytes": children_maxrss,
        "rusage_historical_sum_diagnostic_bytes": self_maxrss + children_maxrss,
    }


class BudgetMonitor:
    def __init__(
        self,
        *,
        root: Path,
        started: float,
        prior_cumulative_wall_seconds: float,
        persistent_bytes: int,
    ) -> None:
        self.root = root
        self.started = started
        self.prior_cumulative_wall_seconds = prior_cumulative_wall_seconds
        self.persistent_bytes = persistent_bytes

    @classmethod
    def start(
        cls,
        root: Path,
        *,
        prior_cumulative_wall_seconds: float,
        attempt_started: float | None = None,
    ) -> "BudgetMonitor":
        monitor = cls(
            root=root,
            started=time.monotonic() if attempt_started is None else attempt_started,
            prior_cumulative_wall_seconds=prior_cumulative_wall_seconds,
            persistent_bytes=_persistent_bytes(root),
        )
        monitor.check(refresh_persistent=True)
        return monitor

    def observed(self, *, refresh_persistent: bool = False) -> dict[str, Any]:
        if refresh_persistent:
            self.persistent_bytes = _persistent_bytes(self.root)
        attempt_wall = time.monotonic() - self.started
        scratch = _scratch_bytes(self.root)
        return {
            "attempt_wall_seconds": attempt_wall,
            "prior_cumulative_wall_seconds": self.prior_cumulative_wall_seconds,
            "cumulative_wall_seconds": self.prior_cumulative_wall_seconds + attempt_wall,
            "persistent_bytes": self.persistent_bytes,
            "scratch_bytes": scratch,
            "new_bytes_including_scratch": self.persistent_bytes + scratch,
            **_rusage_diagnostics(),
        }

    def check(
        self,
        *,
        refresh_persistent: bool = False,
        reserve_bytes: int = FINAL_COMMIT_RESERVE_BYTES,
    ) -> dict[str, Any]:
        observed = self.observed(refresh_persistent=refresh_persistent)
        if observed["cumulative_wall_seconds"] > MAX_WALL_SECONDS:
            raise RuntimeError(f"C3 cumulative wall budget exceeded: {observed}")
        if observed["scratch_bytes"] > MAX_SCRATCH_BYTES:
            raise RuntimeError(f"C3 scratch budget exceeded: {observed}")
        if observed["new_bytes_including_scratch"] + reserve_bytes > MAX_NEW_BYTES:
            raise RuntimeError(f"C3 storage budget/reserve exceeded: {observed}")
        return observed

    def prepublish_guard(self, stage_bytes: int) -> None:
        if type(stage_bytes) is not int or stage_bytes < 0:
            raise ValueError("stage_bytes must be a non-negative integer")
        # The warehouse calls this after staging but before derivation/object mutation.
        observed = self.observed(refresh_persistent=True)
        encode_scratch = _tree_bytes(self.root / ".scratch")
        staged_now = _tree_bytes(self.root / ".staging")
        effective_scratch = max(observed["scratch_bytes"], encode_scratch + stage_bytes, staged_now)
        if effective_scratch > MAX_SCRATCH_BYTES:
            raise RuntimeError(
                f"C3 pre-publish scratch quota exceeded: {effective_scratch}>{MAX_SCRATCH_BYTES}"
            )
        proposed = (
            self.persistent_bytes
            + effective_scratch
            + PUBLISH_OVERHEAD_RESERVE_BYTES
            + FINAL_COMMIT_RESERVE_BYTES
        )
        if proposed > MAX_NEW_BYTES:
            raise RuntimeError(f"C3 pre-publish storage quota exceeded: {proposed}>{MAX_NEW_BYTES}")
        if observed["cumulative_wall_seconds"] > MAX_WALL_SECONDS:
            raise RuntimeError("C3 cumulative wall budget exceeded before CAS publish")


def _predicted_episode_bytes(source: Mapping[str, Any], episode: Mapping[str, Any]) -> int:
    rate = float(source["c2_rate_basis"]["bytes_per_frame"])
    return int(rate * int(episode["num_frames"]) * 1.25) + PUBLISH_OVERHEAD_RESERVE_BYTES


def _remove_success_scratch(path: Path, root: Path) -> None:
    resolved = path.resolve(strict=True)
    scratch_root = (root / ".scratch").resolve(strict=True)
    if scratch_root not in resolved.parents:
        raise RuntimeError("Refusing to clean scratch outside the C3 warehouse")
    shutil.rmtree(resolved)


def _verified_encode_identity(encoded: Any) -> dict[str, Any]:
    identity = _file_identity(encoded.bitstream_path)
    if (
        identity["sha256"] != encoded.bitstream_sha256
        or encoded.attempt_sha256 != (encoded.bitstream_sha256,) * 2
    ):
        raise RuntimeError("Accepted encode drifted after deterministic double encoding")
    return identity


def _content_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {"bytes": identity["bytes"], "sha256": identity["sha256"]}


def _estimated_publish_stage_bytes(
    arrays: Mapping[str, np.ndarray],
    raw_mv_offsets: np.ndarray,
    raw_mv_fields: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    accepted_bitstream: Path,
) -> int:
    """Conservative pre-write bound for the publish staging tree."""

    array_bytes = sum(int(np.asarray(value).nbytes) for value in arrays.values())
    raw_bytes = int(np.asarray(raw_mv_offsets).nbytes) + sum(
        int(np.asarray(value).nbytes) for value in raw_mv_fields.values()
    )
    metadata_bytes = len(canonical_json_bytes(dict(metadata)))
    bitstream_bytes = accepted_bitstream.stat().st_size
    return array_bytes + raw_bytes + metadata_bytes * 2 + bitstream_bytes + 2_000_000


def _materialize_episode(
    root: Path,
    journal: AttemptJournal,
    plan: Mapping[str, Any],
    source: Mapping[str, Any],
    episode_row: Mapping[str, Any],
    derivation: Mapping[str, Any],
    budget: BudgetMonitor,
) -> tuple[PublishedObject, dict[str, Any]]:
    episode = _episode_input(episode_row, source)
    before = verify_canonical_input_identity(episode, stage="BEFORE_EPISODE_READ")
    timestamps = validate_canonical_parquet(episode)
    canonical_rgb, canonical_decoder = decode_canonical_pyav(episode, timestamps)
    parity = assert_torchcodec_pyav_exact(episode.canonical_video_path, canonical_rgb)
    scratch = root / ".scratch" / journal.attempt_id / episode.episode_uid
    scratch.mkdir(mode=0o755, parents=True, exist_ok=False)
    budget.check()
    ffmpeg_path = plan["toolchain"]["ffmpeg"]["path"]
    ffprobe_path = plan["toolchain"]["ffprobe"]["path"]
    encoded = encode_h264_twice(canonical_rgb, scratch, ffmpeg=ffmpeg_path)
    encoder_options = probe_x264_encoder_options(encoded.bitstream_path)
    budget.check()
    packets = probe_packets(
        encoded.bitstream_path,
        num_frames=episode.num_frames,
        ffprobe=ffprobe_path,
    )
    headers = trace_h264_headers(
        encoded.bitstream_path,
        num_frames=episode.num_frames,
        ffmpeg=ffmpeg_path,
    )
    extracted = extract_mvs_and_decoded_rgb(encoded.bitstream_path, num_frames=episode.num_frames)
    arrays = build_episode_arrays(extracted, packets, headers)
    encoded_identity = _verified_encode_identity(encoded)
    contract = plan["codec_contract"]
    metadata = {
        "split": "full",
        "cohort_id": episode.cohort_id,
        "dataset_name": episode.dataset_name,
        "source_episode_index": episode.source_episode_index,
        "width": int(canonical_rgb.shape[2]),
        "height": int(canonical_rgb.shape[1]),
        "time_base_num": 1,
        "time_base_den": 30_000,
        "gop_size": GOP,
        "source_frame_sha256": [raster_sha256(frame) for frame in canonical_rgb],
        "canonical_frame_sha256": [raster_sha256(frame) for frame in canonical_rgb],
        "fixture_frame_sha256": [raster_sha256(frame) for frame in extracted.rgb],
        "tool_identity": contract["tool_identity"],
        "encoder_identity": contract["encoder_identity"],
        "decoder_identity": contract["decoder_identity"],
        "extractor_identity": contract["extractor_identity"],
        "episode_identity": episode_row["episode_identity"],
        "codec_contract": contract,
        "codec_contract_sha256": plan["codec_contract_sha256"],
    }
    stage_bound = _estimated_publish_stage_bytes(
        arrays,
        extracted.mv_offsets,
        extracted.mv_fields,
        metadata,
        encoded.bitstream_path,
    )
    encode_scratch = _tree_bytes(root / ".scratch")
    if encode_scratch + stage_bound > MAX_SCRATCH_BYTES:
        raise RuntimeError("C3 estimated encode+publish staging closure exceeds the scratch budget")
    published = publish_episode_object(
        root,
        derivation_payload=derivation,
        episode_uid=episode.episode_uid,
        camera_id=CAMERA_ID,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
        metadata=metadata,
        arrays=arrays,
        raw_mv_offsets=extracted.mv_offsets,
        raw_mv_fields=extracted.mv_fields,
        accepted_bitstream=encoded.bitstream_path,
        prepublish_guard=budget.prepublish_guard,
    )
    bundle_store = EpisodeBundleStore(
        published.object_path / "payload/bundle",
        manifest_sha256_by_key={published.key: published.bundle_manifest_sha256},
    )
    loaded = bundle_store.get(published.key)
    expected_anchors = np.arange(0, episode.num_frames, GOP, dtype=np.int64)
    if not np.array_equal(loaded.anchor_frame_index, expected_anchors):
        raise RuntimeError("Published C3 anchor_frame_index differs from exact GOP30 coverage")
    raw_audit = bundle_store.audit_raw_motion_vectors(published.key)
    retained = bundle_store.audit_accepted_bitstream(published.key)
    if _content_identity(_file_identity(retained)) != _content_identity(encoded_identity):
        raise RuntimeError("Published CAS bitstream differs from the double-encoded source")
    after = verify_canonical_input_identity(episode, stage="AFTER_EPISODE_PUBLISH")
    if (
        before["canonical_video"] != after["canonical_video"]
        or before["canonical_parquet"] != after["canonical_parquet"]
    ):
        raise RuntimeError("Canonical input identity changed during episode materialization")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": journal.attempt_id,
        "episode_uid": episode.episode_uid,
        "source_id": episode.cohort_id,
        "source_episode_index": episode.source_episode_index,
        "num_frames": episode.num_frames,
        "num_anchors": loaded.num_anchors,
        "derivation_sha256": derivation_sha256(derivation),
        "object_content_sha256": published.object_content_sha256,
        "input_identity_checks": [before, after],
        "input_identity_unchanged": True,
        "canonical_decoder": canonical_decoder,
        "torchcodec_pyav_exact": parity,
        "deterministic_encode_sha256": list(encoded.attempt_sha256),
        "deterministic_encode_equal": len(set(encoded.attempt_sha256)) == 1,
        "x264_encoder_options": encoder_options,
        "accepted_bitstream": _file_identity(retained),
        "packet_stream": dict(packets.stream),
        "header_proof_sha256": headers.normalized_trace_sha256,
        "header_proof_claims": dict(headers.proof_claims),
        "raw_mv_count": int(raw_audit.offsets[-1]),
        "raw_mv_fields": list(raw_audit.fields),
        "reused": published.reused,
    }
    evidence_path = journal.path / "episode_evidence" / f"{episode.episode_uid}.json"
    _write_json_new(evidence_path, evidence)
    journal.record_object(published)
    _remove_success_scratch(scratch, root)
    return published, evidence


def _reuse_episode(
    journal: AttemptJournal,
    episode: CanonicalEpisodeInput,
    derivation: Mapping[str, Any],
    published: PublishedObject,
) -> dict[str, Any]:
    input_check = verify_canonical_input_identity(episode, stage="MANUAL_RESUME_REUSE_CHECK")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": journal.attempt_id,
        "episode_uid": episode.episode_uid,
        "source_id": episode.cohort_id,
        "source_episode_index": episode.source_episode_index,
        "num_frames": episode.num_frames,
        "derivation_sha256": derivation_sha256(derivation),
        "object_content_sha256": published.object_content_sha256,
        "input_identity_checks": [input_check],
        "input_identity_unchanged": True,
        "reused": True,
    }
    _write_json_new(journal.path / "episode_evidence" / f"{episode.episode_uid}.json", evidence)
    journal.record_object(published)
    return evidence


def _attempt_terminal_path(journal: AttemptJournal) -> Path:
    return journal.path / "ATTEMPT_TERMINAL.json"


def _finish_attempt_once(
    journal: AttemptJournal,
    *,
    status: str,
    result: Mapping[str, Any],
    resume_from_attempt_id: str | None,
) -> Path:
    if status not in {"READY", "FAILED", "STOPPED"}:
        raise ValueError("C3 terminal status must be READY, FAILED, or STOPPED")
    collisions = [name for name in TERMINAL_NAMES if (journal.path / name).exists()]
    if collisions:
        raise FileExistsError(f"C3 attempt already has terminal state: {collisions}")
    terminal_result = {
        **dict(result),
        "resume_from_attempt_id": resume_from_attempt_id,
        "terminal_utc_time": _utc_now(),
    }
    return journal.finish(status, terminal_result)


def _write_worker_result(journal: AttemptJournal, result: Mapping[str, Any]) -> Path:
    path = journal.path / "WORKER_RESULT.json"
    _write_json_new(
        path,
        {
            "schema_version": "g1-all-lerobot-c3-worker-result-r2-v1",
            "attempt_id": journal.attempt_id,
            "plan_sha256": journal.plan_sha256,
            "result": dict(result),
        },
    )
    return path


def _load_prior_chain(
    root: Path,
    resume_from_attempt_id: str,
    *,
    expected_plan: Mapping[str, Any] | None = None,
) -> tuple[list[str], float]:
    current: str | None = resume_from_attempt_id
    seen: set[str] = set()
    newest_to_oldest: list[str] = []
    prior_cumulative: float | None = None
    normalized_plan = (
        json.loads(canonical_json_bytes(dict(expected_plan))) if expected_plan is not None else None
    )
    expected_plan_sha256 = (
        canonical_json_sha256(normalized_plan) if normalized_plan is not None else None
    )
    while current is not None:
        if current in seen or not current or "/" in current or "\\" in current:
            raise ValueError("Manual resume attempt chain is cyclic or malformed")
        seen.add(current)
        newest_to_oldest.append(current)
        attempt = root / "attempts" / current
        started = _load_json(attempt / "ATTEMPT_STARTED.json")
        terminal = _load_json(attempt / "ATTEMPT_TERMINAL.json")
        if (
            set(started)
            != {
                "schema_version",
                "attempt_id",
                "mode",
                "resume_from_attempt_id",
                "plan_sha256",
                "plan",
            }
            or started.get("schema_version") != ATTEMPT_SCHEMA_VERSION
            or started.get("attempt_id") != current
            or terminal.get("attempt_id") != current
        ):
            raise ValueError("Manual resume attempt ID differs from its explicit directory")
        if terminal.get("plan_sha256") != started.get("plan_sha256") or canonical_json_sha256(
            started.get("plan")
        ) != started.get("plan_sha256"):
            raise ValueError("Manual resume terminal and start plan digests differ")
        if (
            set(terminal)
            != {
                "schema_version",
                "attempt_id",
                "plan_sha256",
                "status",
                "result",
            }
            or terminal.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        ):
            raise ValueError("Manual resume terminal schema/fields differ from the warehouse")
        if terminal.get("status") not in {"FAILED", "STOPPED"}:
            raise ValueError("Every resumed C3 predecessor must be explicitly failed/stopped")
        if normalized_plan is not None and (
            started.get("plan") != normalized_plan
            or started.get("plan_sha256") != expected_plan_sha256
        ):
            raise ValueError("Manual resume chain differs from the exact current plan")
        predecessor = started.get("resume_from_attempt_id")
        expected_mode = "INITIAL" if predecessor is None else "MANUAL_RESUME"
        if (
            started.get("mode") != expected_mode
            or terminal.get("result", {}).get("resume_from_attempt_id") != predecessor
        ):
            raise ValueError("Manual resume lineage fields are inconsistent")
        if current == resume_from_attempt_id:
            value = terminal.get("result", {}).get("resources", {}).get("cumulative_wall_seconds")
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError("Prior terminal lacks exact cumulative wall accounting")
            prior_cumulative = float(value)
        current = predecessor
        if current is not None and not isinstance(current, str):
            raise ValueError("Manual resume chain predecessor must be a string or null")
    assert prior_cumulative is not None
    return list(reversed(newest_to_oldest)), prior_cumulative


def _load_existing_journal(root: Path, attempt_id: str, plan: Mapping[str, Any]) -> AttemptJournal:
    started = _load_json(root / "attempts" / attempt_id / "ATTEMPT_STARTED.json")
    normalized_plan = json.loads(canonical_json_bytes(dict(plan)))
    if (
        started.get("attempt_id") != attempt_id
        or started.get("plan") != normalized_plan
        or started.get("plan_sha256") != canonical_json_sha256(normalized_plan)
    ):
        raise RuntimeError("Worker attempt start record differs from the exact execution plan")
    return AttemptJournal(root, root / "attempts" / attempt_id, started)


def _require_exact_warehouse_top_level(root: Path, *, allow_scratch: bool = False) -> None:
    expected = {"objects", "derivations", "attempts", ".staging"}
    if allow_scratch:
        expected.add(".scratch")
    observed = {path.name for path in root.iterdir()}
    if allow_scratch and ".scratch" not in observed:
        expected.remove(".scratch")
    if observed != expected:
        raise RuntimeError(f"C3 warehouse has foreign/missing top-level entries: {observed}")


def _clean_explicit_resume_temporaries(root: Path, chain: Sequence[str]) -> None:
    """Remove only uncommitted scratch after every attempt is durably terminal."""

    for attempt in (root / "attempts").iterdir():
        if attempt.is_symlink() or not attempt.is_dir():
            raise RuntimeError("C3 attempts namespace contains a non-directory")
        if not (attempt / "ATTEMPT_TERMINAL.json").is_file():
            raise RuntimeError(f"Another C3 attempt is not terminal: {attempt.name}")
    scratch_root = root / ".scratch"
    if scratch_root.exists():
        if scratch_root.is_symlink() or not scratch_root.is_dir():
            raise RuntimeError("C3 scratch namespace is not a real directory")
        allowed = set(chain)
        foreign = {path.name for path in scratch_root.iterdir()} - allowed
        if foreign:
            raise RuntimeError(f"C3 scratch has non-lineage owners: {sorted(foreign)}")
        for attempt_id in chain:
            path = scratch_root / attempt_id
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise RuntimeError("C3 lineage scratch is not a real directory")
                _make_tree_user_writable(path)
                shutil.rmtree(path)
    staging = root / ".staging"
    if staging.is_symlink() or not staging.is_dir():
        raise RuntimeError("C3 staging namespace is not a real directory")
    for path in list(staging.iterdir()):
        if path.is_symlink():
            raise RuntimeError("C3 staging contains a symlink")
        if path.is_dir():
            _make_tree_user_writable(path)
            shutil.rmtree(path)
        elif path.is_file():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
            path.unlink()
        else:
            raise RuntimeError("C3 staging contains a special file")


def _create_attempt(
    plan: Mapping[str, Any],
    *,
    attempt_id: str,
    resume_from_attempt_id: str | None,
) -> tuple[AttemptJournal, float, list[str]]:
    if EXPECTED_OUTPUT_ROOT.is_symlink():
        raise ValueError("C3 output root must not be a symlink")
    if resume_from_attempt_id is None:
        if attempt_id != INITIAL_ATTEMPT_ID:
            raise ValueError(f"Initial C3 attempt is fixed to {INITIAL_ATTEMPT_ID}")
        parent = EXPECTED_OUTPUT_ROOT.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("C3 output parent must be an existing real directory")
        # This is the no-clobber reservation. AttemptJournal may only populate
        # the exact empty inode we created here; it never adopts a raced owner.
        EXPECTED_OUTPUT_ROOT.mkdir(mode=0o755, exist_ok=False)
        journal = AttemptJournal.start(EXPECTED_OUTPUT_ROOT, attempt_id=attempt_id, plan=plan)
        _require_exact_warehouse_top_level(EXPECTED_OUTPUT_ROOT)
        return journal, 0.0, []
    if not EXPECTED_OUTPUT_ROOT.is_dir():
        raise FileNotFoundError("Manual resume requires the exact existing C3 warehouse")
    if (EXPECTED_OUTPUT_ROOT / FINAL_MANIFEST_FILENAME).exists():
        raise FileExistsError("A finalized C3 fixture cannot be resumed")
    chain, prior_wall = _load_prior_chain(
        EXPECTED_OUTPUT_ROOT,
        resume_from_attempt_id,
        expected_plan=plan,
    )
    if prior_wall >= MAX_WALL_SECONDS:
        raise RuntimeError("C3 cumulative wall budget is already exhausted")
    _require_exact_warehouse_top_level(EXPECTED_OUTPUT_ROOT, allow_scratch=True)
    attempt_names = {path.name for path in (EXPECTED_OUTPUT_ROOT / "attempts").iterdir()}
    if attempt_names != set(chain):
        raise RuntimeError(
            f"Manual resume requires one unbranched explicit attempt chain: {attempt_names} != {set(chain)}"
        )
    _clean_explicit_resume_temporaries(EXPECTED_OUTPUT_ROOT, chain)
    journal = AttemptJournal.manual_resume(
        EXPECTED_OUTPUT_ROOT,
        attempt_id=attempt_id,
        resume_from_attempt_id=resume_from_attempt_id,
        plan=plan,
    )
    return journal, prior_wall, chain


def _producer_identity(
    plan: Mapping[str, Any],
    journal: AttemptJournal,
    objects: Sequence[PublishedObject],
    frozen: Mapping[str, Any],
    resume_chain: Sequence[str],
) -> dict[str, Any]:
    expected_lengths = sorted(
        (
            {"episode_uid": item["episode_uid"], "num_frames": item["num_frames"]}
            for item in frozen["episodes"]
        ),
        key=lambda item: item["episode_uid"],
    )
    derivations = sorted(
        (
            {"episode_uid": item.key.episode_uid, "derivation_sha256": item.derivation_sha256}
            for item in objects
        ),
        key=lambda item: item["episode_uid"],
    )
    if canonical_json_sha256(expected_lengths) != plan["expected_episode_lengths_sha256"]:
        raise RuntimeError("Final expected episode-length closure drifted")
    return {
        "schema_version": "g1-all-lerobot-c3-producer-identity-v1",
        "attempt_id": journal.attempt_id,
        "resume_chain": list(resume_chain),
        "attempt_plan_sha256": journal.plan_sha256,
        "execution_plan_sha256": plan["plan_sha256"],
        "git_head": plan["toolchain"]["git"]["head"],
        "toolchain_identity_sha256": plan["toolchain_identity_sha256"],
        "implementation_identity_sha256": plan["implementation_identity_sha256"],
        "input_freeze_file_sha256": plan["input_freeze"]["sha256"],
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "episode_freeze_core_sha256": plan["episode_freeze_core_sha256"],
        "expected_episode_lengths_sha256": plan["expected_episode_lengths_sha256"],
        "expected_derivations_sha256": canonical_json_sha256(derivations),
        "automatic_retries": AUTOMATIC_RETRIES,
    }


def _verify_final_bundle_facts(
    objects: Sequence[PublishedObject], frozen: Mapping[str, Any]
) -> tuple[int, int]:
    """Verify exact per-object GOP30 anchors and aggregate frozen coverage."""

    expected = {item["episode_uid"]: item for item in frozen["episodes"]}
    total_frames = 0
    total_anchors = 0
    for published in objects:
        row = expected[published.key.episode_uid]
        store = EpisodeBundleStore(
            published.object_path / "payload/bundle",
            manifest_sha256_by_key={published.key: published.bundle_manifest_sha256},
        )
        bundle = store.get(published.key)
        expected_anchor_index = np.arange(0, row["num_frames"], GOP, dtype=np.int64)
        if bundle.num_frames != row["num_frames"] or not np.array_equal(
            bundle.anchor_frame_index, expected_anchor_index
        ):
            raise RuntimeError(
                f"Final C3 object length/anchor closure drifted: {published.key.episode_uid}"
            )
        total_frames += bundle.num_frames
        total_anchors += bundle.num_anchors
    if total_frames != EXPECTED_FRAMES or total_anchors != EXPECTED_CODEC_ANCHORS:
        raise RuntimeError(
            f"Final C3 aggregate coverage differs from "
            f"{EXPECTED_FRAMES} frames/{EXPECTED_CODEC_ANCHORS} anchors: "
            f"{total_frames}/{total_anchors}"
        )
    return total_frames, total_anchors


def _run_attempt(
    plan: Mapping[str, Any],
    repository: Path,
    journal: AttemptJournal,
    *,
    resume_from_attempt_id: str | None,
    prior_cumulative_wall_seconds: float,
    resume_chain: Sequence[str],
    attempt_started: float | None = None,
) -> dict[str, Any]:
    root = journal.root
    event_path = journal.path / "events.jsonl"
    budget: BudgetMonitor | None = None
    effective_attempt_started = time.monotonic() if attempt_started is None else attempt_started
    try:
        frozen = _verify_execution_identity(plan, repository)
        _verify_all_frozen_inputs(frozen)
        _check_dataset_markers()
        _write_json_new(journal.path / "TOOLCHAIN_IDENTITY.json", plan["toolchain"])
        _write_json_new(journal.path / "CODEC_CONTRACT.json", plan["codec_contract"])
        budget = BudgetMonitor.start(
            root,
            prior_cumulative_wall_seconds=prior_cumulative_wall_seconds,
            attempt_started=effective_attempt_started,
        )
        _append_event(
            event_path,
            "ATTEMPT_STARTED",
            attempt_id=journal.attempt_id,
            resume_from_attempt_id=resume_from_attempt_id,
            resume_chain=list(resume_chain),
        )
        source_by_id = {item["source_id"]: item for item in frozen["sources"]}
        objects: list[PublishedObject] = []
        attestations: list[dict[str, Any]] = []
        active_source: str | None = None
        for ordinal, episode_row in enumerate(frozen["episodes"]):
            source_id = episode_row["source_id"]
            source = source_by_id[source_id]
            if source_id != active_source:
                if active_source is not None:
                    budget.check(refresh_persistent=True)
                _verify_execution_identity(plan, repository)
                _verify_source_metadata(source)
                active_source = source_id
            _check_dataset_markers()
            budget.check()
            derivation = episode_derivation(plan, episode_row)
            key = EpisodeBundleKey(
                episode_row["episode_uid"], CAMERA_ID, SOURCE_KIND_NATIVE_CANONICAL
            )
            canonical = _episode_input(episode_row, source)
            verify_canonical_input_identity(canonical, stage="BEFORE_DERIVATION_LOOKUP")
            reusable = journal.reusable_object(derivation, expected_key=key)
            if reusable is not None:
                if reusable.num_frames != episode_row["num_frames"]:
                    raise RuntimeError("Reusable C3 object frame count differs from the freeze")
                evidence = _reuse_episode(journal, canonical, derivation, reusable)
                published = reusable
            else:
                projected = _predicted_episode_bytes(source, episode_row)
                observed = budget.check()
                if (
                    observed["new_bytes_including_scratch"] + projected + FINAL_COMMIT_RESERVE_BYTES
                    > MAX_NEW_BYTES
                ):
                    raise RuntimeError("C3 projected next episode would exceed the storage budget")
                published, evidence = _materialize_episode(
                    root,
                    journal,
                    plan,
                    source,
                    episode_row,
                    derivation,
                    budget,
                )
            if published.key != key or published.num_frames != episode_row["num_frames"]:
                raise RuntimeError("Published C3 object differs from the frozen episode key/length")
            objects.append(published)
            attestations.append(
                {
                    "episode_uid": key.episode_uid,
                    "num_frames": published.num_frames,
                    "derivation_sha256": published.derivation_sha256,
                    "object_content_sha256": published.object_content_sha256,
                    "reused": published.reused,
                    "input_identity_unchanged": evidence["input_identity_unchanged"],
                }
            )
            _append_event(
                event_path,
                "EPISODE_ACCEPTED",
                ordinal=ordinal,
                episode_uid=key.episode_uid,
                source_id=source_id,
                object_content_sha256=published.object_content_sha256,
                reused=published.reused,
                resources=budget.observed(),
            )
            budget.check(refresh_persistent=True)
        _verify_execution_identity(plan, repository)
        _check_dataset_markers()
        observed_uids = [item.key.episode_uid for item in objects]
        expected_uids = [item["episode_uid"] for item in frozen["episodes"]]
        expected_frames = {item["episode_uid"]: item["num_frames"] for item in frozen["episodes"]}
        if (
            len(objects) != EXPECTED_EPISODES
            or set(observed_uids) != set(expected_uids)
            or len(set(observed_uids)) != EXPECTED_EPISODES
            or any(item.num_frames != expected_frames[item.key.episode_uid] for item in objects)
        ):
            raise RuntimeError("C3 final object closure differs from the exact execution selection")
        for item, episode_row in zip(objects, frozen["episodes"]):
            if item.derivation_sha256 != derivation_sha256(episode_derivation(plan, episode_row)):
                raise RuntimeError("C3 final derivation closure differs from the frozen plan")
        _verify_all_frozen_inputs(frozen)
        total_frames, total_anchors = _verify_final_bundle_facts(objects, frozen)
        producer = _producer_identity(plan, journal, objects, frozen, resume_chain)
        before_commit = budget.check(refresh_persistent=True)
        common_summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "READY",
            "fixture_id": FIXTURE_ID,
            "attempt_id": journal.attempt_id,
            "resume_from_attempt_id": resume_from_attempt_id,
            "resume_chain": list(resume_chain),
            "coverage_scope": COVERAGE_SCOPE,
            "episode_selection": plan["episode_selection"],
            "episode_count": len(objects),
            "total_frames": total_frames,
            "total_codec_anchors": total_anchors,
            "reused_episode_count": sum(item.reused for item in objects),
            "materialized_episode_count": sum(not item.reused for item in objects),
            "codec_contract_sha256": plan["codec_contract_sha256"],
            "toolchain_identity_sha256": plan["toolchain_identity_sha256"],
            "input_snapshot_sha256": plan["input_snapshot_sha256"],
            "expected_episode_lengths_sha256": plan["expected_episode_lengths_sha256"],
            "expected_derivations_sha256": producer["expected_derivations_sha256"],
            "episode_attestations_sha256": canonical_json_sha256(attestations),
            "resources": before_commit,
            "rss_enforcement": "SUPERVISOR_LIVE_PROCESS_TREE_SAMPLED_EVERY_0P25_SECONDS",
            "supervisor_resources_path": "SUPERVISOR_RESOURCES.json",
            "cpu_only": True,
            "network": "offline_no_download",
            "gpu_used": False,
            "training_run": False,
            "evaluator_run": False,
            "automatic_retries": AUTOMATIC_RETRIES,
        }
        if not PUBLISH_FULL_FIXTURE_MANIFEST:
            summary = {
                **common_summary,
                "fixture_manifest_sha256": None,
                "full_fixture": False,
            }
            _write_json_new(journal.path / "CANARY_SUMMARY.json", summary)
            _write_worker_result(journal, summary)
            return summary
        if (
            before_commit["cumulative_wall_seconds"] + FINAL_COMMIT_WALL_RESERVE_SECONDS
            > MAX_WALL_SECONDS
        ):
            raise RuntimeError("C3 final manifest commit wall-time reserve is unavailable")
        manifest_path = commit_full_fixture_manifest(
            root,
            fixture_id=FIXTURE_ID,
            expected_episode_count=EXPECTED_EPISODES,
            expected_episode_uids=expected_uids,
            objects=objects,
            input_snapshot_sha256=plan["input_snapshot_sha256"],
            budget_sha256=plan["budget_sha256"],
            codec_contract=plan["codec_contract"],
            producer_identity=producer,
        )
        # Returning from commit means its exact full-selection validation and
        # atomic no-replace hard-link are authoritative. Keep the post-commit
        # path minimal; the supervisor can recover READY from this manifest.
        manifest_sha256 = sha256_path(manifest_path)
        budget.persistent_bytes += manifest_path.stat().st_size
        resources = {
            **before_commit,
            "persistent_bytes": budget.persistent_bytes,
            "new_bytes_including_scratch": budget.persistent_bytes + before_commit["scratch_bytes"],
        }
        summary = {
            **common_summary,
            "fixture_manifest_sha256": manifest_sha256,
            "resources": resources,
            "full_fixture": True,
        }
        _write_worker_result(journal, summary)
        return summary
    except BaseException as error:
        if (root / FINAL_MANIFEST_FILENAME).is_file():
            # A loader-visible manifest is the authoritative success commit.
            # Never publish FAILED after it; the supervisor validates and
            # recovers the exclusive READY terminal if this worker dies here.
            raise
        if budget is None:
            attempt_wall = time.monotonic() - effective_attempt_started
            resources = {
                "attempt_wall_seconds": attempt_wall,
                "prior_cumulative_wall_seconds": prior_cumulative_wall_seconds,
                "cumulative_wall_seconds": prior_cumulative_wall_seconds + attempt_wall,
                "persistent_bytes": _persistent_bytes(root),
                "scratch_bytes": _scratch_bytes(root),
                "new_bytes_including_scratch": _persistent_bytes(root) + _scratch_bytes(root),
                **_rusage_diagnostics(),
            }
        else:
            resources = budget.observed(refresh_persistent=True)
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "resources": resources,
            "automatic_retries": AUTOMATIC_RETRIES,
            "scratch_preserved": True,
        }
        if not _attempt_terminal_path(journal).exists():
            _append_event(event_path, "ATTEMPT_FAILED", failure=failure)
            _finish_attempt_once(
                journal,
                status="FAILED",
                result=failure,
                resume_from_attempt_id=resume_from_attempt_id,
            )
        raise


def execute_attempt(
    plan: Mapping[str, Any],
    repository: Path,
    *,
    attempt_id: str = INITIAL_ATTEMPT_ID,
    resume_from_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Run inline; the CLI uses the separate supervisor wrapper below."""

    attempt_started = time.monotonic()
    journal, prior_wall, chain = _create_attempt(
        plan,
        attempt_id=attempt_id,
        resume_from_attempt_id=resume_from_attempt_id,
    )
    return _run_attempt(
        plan,
        repository,
        journal,
        resume_from_attempt_id=resume_from_attempt_id,
        prior_cumulative_wall_seconds=prior_wall,
        resume_chain=chain,
        attempt_started=attempt_started,
    )


def _verify_outer_timeout_supervisor() -> None:
    if os.environ.get("C3_OUTER_TIMEOUT_SECONDS") != str(MAX_WALL_SECONDS):
        raise RuntimeError(
            f"C3 execute requires the frozen outer {MAX_WALL_SECONDS}-second timeout"
        )
    parent_executable = Path(f"/proc/{os.getppid()}/exe").resolve(strict=True)
    if parent_executable != Path("/usr/bin/timeout").resolve(strict=True):
        raise RuntimeError(f"C3 execute parent must be /usr/bin/timeout, got {parent_executable}")
    command = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    expected_prefix = [
        b"/usr/bin/timeout",
        b"--signal=TERM",
        b"--kill-after=30s",
        f"{MAX_WALL_SECONDS}s".encode("ascii"),
    ]
    if command[:4] != expected_prefix:
        raise RuntimeError(
            f"C3 outer timeout argv must be --signal=TERM --kill-after=30s {MAX_WALL_SECONDS}s"
        )


def _process_tree_rss_bytes(root_pid: int) -> int:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text(encoding="utf-8")
            command_end = raw.rfind(")")
            if command_end < 0:
                raise ValueError("malformed /proc stat")
            pid = int(raw[: raw.index("(")].strip())
            fields_after_command = raw[command_end + 2 :].split()
            parents[pid] = int(fields_after_command[1])
            rss[pid] = int(fields_after_command[21]) * page_bytes
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in descendants)


class LiveProcessTreeRssMonitor:
    """Sample one live process tree on an independent 0.25-second cadence."""

    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self._stop = threading.Event()
        self._triggered = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._last = 0
        self._samples = 0
        self._error: str | None = None

    def _sample(self) -> None:
        try:
            live = _process_tree_rss_bytes(self.root_pid)
        except BaseException as error:
            with self._lock:
                self._error = f"{type(error).__name__}:{error}"
            self._triggered.set()
            return
        with self._lock:
            self._last = live
            self._peak = max(self._peak, live)
            self._samples += 1
            if live > MAX_PEAK_RSS_BYTES:
                self._triggered.set()

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            self._sample()
            deadline += SUPERVISOR_POLL_SECONDS
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Live RSS monitor may only be started once")
        self._thread = threading.Thread(target=self._run, name="c3-live-rss", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("Live RSS monitor did not stop")

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "peak_live_rss_bytes": self._peak,
                "last_live_rss_bytes": self._last,
                "sample_count": self._samples,
                "error": self._error,
            }


def _terminate_worker(worker: subprocess.Popen[bytes]) -> None:
    if worker.poll() is not None:
        return
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + SUPERVISOR_TERM_GRACE_SECONDS
    while worker.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if worker.poll() is None:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait(timeout=5)


def _arm_parent_death_signal() -> None:
    """Stop an isolated worker if its supervisor is killed unexpectedly."""

    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != parent_pid:
        raise RuntimeError("C3 supervisor exited before the worker armed its death signal")


def _write_supervisor_resources(
    journal: AttemptJournal,
    *,
    peak_live_rss_bytes: int,
    last_live_rss_bytes: int,
    sample_count: int,
    supervisor_started: float,
    prior_wall: float,
) -> Path:
    path = journal.path / "SUPERVISOR_RESOURCES.json"
    if path.exists() or path.is_symlink():
        existing = _load_json(path)
        if existing.get("attempt_id") != journal.attempt_id:
            raise RuntimeError("Existing supervisor resource receipt belongs to another attempt")
        return path
    attempt_wall = time.monotonic() - supervisor_started
    _write_json_new(
        path,
        {
            "schema_version": "g1-all-lerobot-c3-supervisor-resources-r2-v1",
            "attempt_id": journal.attempt_id,
            "sample_interval_seconds": SUPERVISOR_POLL_SECONDS,
            "sample_count": sample_count,
            "last_live_process_tree_rss_bytes": last_live_rss_bytes,
            "peak_live_process_tree_rss_bytes": peak_live_rss_bytes,
            "hard_limit_live_process_tree_rss_bytes": MAX_PEAK_RSS_BYTES,
            "live_rss_within_limit": peak_live_rss_bytes <= MAX_PEAK_RSS_BYTES,
            "attempt_wall_seconds": attempt_wall,
            "prior_cumulative_wall_seconds": prior_wall,
            "cumulative_wall_seconds": prior_wall + attempt_wall,
            "diagnostic_only": _rusage_diagnostics(),
        },
    )
    return path


def _finalize_worker_result(
    journal: AttemptJournal,
    *,
    resume_from_attempt_id: str | None,
) -> dict[str, Any]:
    payload = _load_json(journal.path / "WORKER_RESULT.json")
    if (
        set(payload) != {"schema_version", "attempt_id", "plan_sha256", "result"}
        or payload.get("schema_version") != "g1-all-lerobot-c3-worker-result-r2-v1"
        or payload.get("attempt_id") != journal.attempt_id
        or payload.get("plan_sha256") != journal.plan_sha256
        or not isinstance(payload.get("result"), dict)
        or payload["result"].get("status") != "READY"
    ):
        raise RuntimeError("C3 worker result differs from its exact attempt")
    supervisor_path = journal.path / "SUPERVISOR_RESOURCES.json"
    supervisor = _load_json(supervisor_path)
    if (
        supervisor.get("attempt_id") != journal.attempt_id
        or supervisor.get("live_rss_within_limit") is not True
        or supervisor.get("peak_live_process_tree_rss_bytes", MAX_PEAK_RSS_BYTES + 1)
        > MAX_PEAK_RSS_BYTES
    ):
        raise RuntimeError("C3 supervisor live process-tree RSS result exceeds its limit")
    result = dict(payload["result"])
    result["resources"] = {
        **dict(result["resources"]),
        "supervisor_live_process_tree_peak_rss_bytes": supervisor[
            "peak_live_process_tree_rss_bytes"
        ],
        "supervisor_rss_sample_count": supervisor["sample_count"],
        "supervisor_resources": _file_identity(supervisor_path),
    }
    _write_json_new(journal.path / "FINAL_SUMMARY.json", result)
    _finish_attempt_once(
        journal,
        status="READY",
        result=result,
        resume_from_attempt_id=resume_from_attempt_id,
    )
    return result


def _supervised_terminal(
    journal: AttemptJournal,
    *,
    status: str,
    reason: str,
    resume_from_attempt_id: str | None,
    prior_wall: float,
    supervisor_started: float,
    supervisor_peak_live_rss_bytes: int = 0,
) -> None:
    if _attempt_terminal_path(journal).exists():
        return
    attempt_wall = time.monotonic() - supervisor_started
    resources = {
        "attempt_wall_seconds": attempt_wall,
        "prior_cumulative_wall_seconds": prior_wall,
        "cumulative_wall_seconds": prior_wall + attempt_wall,
        "persistent_bytes": _persistent_bytes(journal.root),
        "scratch_bytes": _scratch_bytes(journal.root),
        "new_bytes_including_scratch": _persistent_bytes(journal.root)
        + _scratch_bytes(journal.root),
        "supervisor_live_process_tree_peak_rss_bytes": supervisor_peak_live_rss_bytes,
        **_rusage_diagnostics(),
    }
    _finish_attempt_once(
        journal,
        status=status,
        result={
            "reason": reason,
            "resources": resources,
            "automatic_retries": AUTOMATIC_RETRIES,
            "scratch_preserved": True,
            "supervisor_terminal": True,
        },
        resume_from_attempt_id=resume_from_attempt_id,
    )


def _recover_ready_from_manifest(
    journal: AttemptJournal,
    plan: Mapping[str, Any],
    *,
    resume_from_attempt_id: str | None,
    prior_wall: float,
    supervisor_started: float,
    supervisor_peak_live_rss_bytes: int,
) -> dict[str, Any]:
    """Validate an authoritative final commit and recover its READY terminal."""

    manifest_path = journal.root / FINAL_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("Committed C3 manifest is missing, non-regular, or a symlink")
    manifest_sha256 = sha256_path(manifest_path)
    manifest = _load_json(manifest_path)
    expected_manifest_fields = {
        "schema_version",
        "fixture_id",
        "complete",
        "coverage_kind",
        "camera_id",
        "source_kind",
        "expected_episode_count",
        "covered_episode_count",
        "covered_episode_uids",
        "covered_episode_uid_digest",
        "input_snapshot_sha256",
        "budget_sha256",
        "codec_contract",
        "codec_contract_sha256",
        "objects",
        "producer_identity",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != "codec-evidence-full-fixture-manifest-v1"
        or manifest.get("fixture_id") != FIXTURE_ID
        or manifest.get("complete") is not True
        or manifest.get("coverage_kind") != "FULL_EPISODE_SET"
    ):
        raise RuntimeError("Committed C3 manifest top-level contract is invalid")
    producer = manifest["producer_identity"]
    expected_producer_fields = {
        "attempt_id": journal.attempt_id,
        "attempt_plan_sha256": journal.plan_sha256,
        "execution_plan_sha256": plan["plan_sha256"],
        "git_head": plan["toolchain"]["git"]["head"],
        "toolchain_identity_sha256": plan["toolchain_identity_sha256"],
        "implementation_identity_sha256": plan["implementation_identity_sha256"],
        "input_freeze_file_sha256": plan["input_freeze"]["sha256"],
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "episode_freeze_core_sha256": plan["episode_freeze_core_sha256"],
        "expected_episode_lengths_sha256": plan["expected_episode_lengths_sha256"],
        "automatic_retries": AUTOMATIC_RETRIES,
    }
    if any(producer.get(key) != value for key, value in expected_producer_fields.items()):
        raise RuntimeError("Committed C3 manifest producer identity differs from the attempt")
    if (
        manifest["expected_episode_count"] != EXPECTED_EPISODES
        or manifest["covered_episode_count"] != EXPECTED_EPISODES
        or sum(item["num_frames"] for item in manifest["objects"]) != EXPECTED_FRAMES
        or manifest["input_snapshot_sha256"] != plan["input_snapshot_sha256"]
        or manifest["budget_sha256"] != plan["budget_sha256"]
        or manifest["codec_contract_sha256"] != plan["codec_contract_sha256"]
        or manifest["codec_contract"] != plan["codec_contract"]
    ):
        raise RuntimeError("Committed C3 manifest coverage/contract differs from the plan")
    entries = manifest["objects"]
    uids = [item["episode_uid"] for item in entries]
    lengths = sorted(
        (
            {"episode_uid": item["episode_uid"], "num_frames": item["num_frames"]}
            for item in entries
        ),
        key=lambda item: item["episode_uid"],
    )
    derivations = sorted(
        (
            {"episode_uid": item["episode_uid"], "derivation_sha256": item["derivation_sha256"]}
            for item in entries
        ),
        key=lambda item: item["episode_uid"],
    )
    if (
        uids != sorted(uids)
        or len(set(uids)) != EXPECTED_EPISODES
        or manifest["covered_episode_uids"] != uids
        or manifest["covered_episode_uid_digest"] != plan["expected_episode_uid_digest"]
        or canonical_json_sha256(sorted(uids)) != plan["expected_episode_uid_digest"]
        or canonical_json_sha256(lengths) != plan["expected_episode_lengths_sha256"]
        or canonical_json_sha256(derivations) != producer["expected_derivations_sha256"]
    ):
        raise RuntimeError("Committed C3 manifest UID/length/derivation closure is invalid")
    for entry in entries:
        object_digest = entry["object_content_sha256"]
        commit = _load_json(journal.root / "objects" / object_digest / "OBJECT_COMMITTED.json")
        binding = _load_json(journal.root / "derivations" / f"{entry['derivation_sha256']}.json")
        if (
            commit.get("schema_version") != "codec-evidence-warehouse-object-commit-v1"
            or commit.get("object_content_sha256") != object_digest
            or binding.get("derivation_sha256") != entry["derivation_sha256"]
            or binding.get("object_content_sha256") != object_digest
            or binding.get("episode_uid") != entry["episode_uid"]
        ):
            raise RuntimeError("Committed C3 object marker/binding closure is invalid")
    attempt_wall = time.monotonic() - supervisor_started
    resources = {
        "attempt_wall_seconds": attempt_wall,
        "prior_cumulative_wall_seconds": prior_wall,
        "cumulative_wall_seconds": prior_wall + attempt_wall,
        "persistent_bytes": _persistent_bytes(journal.root),
        "scratch_bytes": _scratch_bytes(journal.root),
        "new_bytes_including_scratch": _persistent_bytes(journal.root)
        + _scratch_bytes(journal.root),
        "supervisor_live_process_tree_peak_rss_bytes": supervisor_peak_live_rss_bytes,
        **_rusage_diagnostics(),
    }
    if (
        resources["cumulative_wall_seconds"] > MAX_WALL_SECONDS
        or resources["new_bytes_including_scratch"] > MAX_NEW_BYTES
        or resources["scratch_bytes"] > MAX_SCRATCH_BYTES
        or resources["supervisor_live_process_tree_peak_rss_bytes"] > MAX_PEAK_RSS_BYTES
    ):
        raise RuntimeError("Committed C3 manifest recovery exceeds the frozen resource budget")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY",
        "fixture_id": FIXTURE_ID,
        "attempt_id": journal.attempt_id,
        "resume_from_attempt_id": resume_from_attempt_id,
        "episode_count": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "total_codec_anchors": EXPECTED_CODEC_ANCHORS,
        "fixture_manifest_sha256": manifest_sha256,
        "codec_contract_sha256": plan["codec_contract_sha256"],
        "toolchain_identity_sha256": plan["toolchain_identity_sha256"],
        "input_snapshot_sha256": plan["input_snapshot_sha256"],
        "expected_episode_lengths_sha256": plan["expected_episode_lengths_sha256"],
        "expected_derivations_sha256": producer["expected_derivations_sha256"],
        "resources": resources,
        "recovered_by_supervisor": True,
        "automatic_retries": AUTOMATIC_RETRIES,
    }
    _finish_attempt_once(
        journal,
        status="READY",
        result=result,
        resume_from_attempt_id=resume_from_attempt_id,
    )
    return result


def _supervise_attempt(
    plan: Mapping[str, Any],
    repository: Path,
    *,
    attempt_id: str,
    resume_from_attempt_id: str | None,
    attempt_started: float,
) -> int:
    del repository
    if resume_from_attempt_id is not None:
        _chain, preflight_prior_wall = _load_prior_chain(
            EXPECTED_OUTPUT_ROOT,
            resume_from_attempt_id,
            expected_plan=plan,
        )
        if preflight_prior_wall + (time.monotonic() - attempt_started) >= MAX_WALL_SECONDS:
            raise RuntimeError("C3 cumulative wall budget expired during resume preflight")
    journal, prior_wall, _chain = _create_attempt(
        plan,
        attempt_id=attempt_id,
        resume_from_attempt_id=resume_from_attempt_id,
    )
    supervisor_started = attempt_started
    command = [
        str(Path(sys.executable).resolve(strict=True)),
        str(Path(__file__).resolve(strict=True)),
        "--_worker-attempt",
        "--profile",
        PROFILE_NAME,
        "--attempt-id",
        attempt_id,
        "--output-root",
        str(EXPECTED_OUTPUT_ROOT),
    ]
    if resume_from_attempt_id is not None:
        command.extend(("--resume-from", resume_from_attempt_id))
    timeout_requested = False

    def handle_outer_timeout(_signum: int, _frame: Any) -> None:
        nonlocal timeout_requested
        timeout_requested = True

    previous = signal.signal(signal.SIGTERM, handle_outer_timeout)
    worker: subprocess.Popen[bytes] | None = None
    stop_reason: str | None = None
    rss_monitor: LiveProcessTreeRssMonitor | None = None

    def rss_snapshot() -> dict[str, Any]:
        if rss_monitor is not None:
            return rss_monitor.snapshot()
        live = _process_tree_rss_bytes(os.getpid())
        return {
            "peak_live_rss_bytes": live,
            "last_live_rss_bytes": live,
            "sample_count": 1,
            "error": None,
        }

    def write_supervisor_receipt() -> Path:
        snapshot = rss_snapshot()
        return _write_supervisor_resources(
            journal,
            peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
            last_live_rss_bytes=snapshot["last_live_rss_bytes"],
            sample_count=snapshot["sample_count"],
            supervisor_started=supervisor_started,
            prior_wall=prior_wall,
        )

    try:
        worker_environment = os.environ.copy()
        worker_environment["C3_ATTEMPT_MONOTONIC_STARTED"] = repr(attempt_started)
        try:
            worker = subprocess.Popen(
                command,
                env=worker_environment,
                start_new_session=True,
            )
        except BaseException as error:
            write_supervisor_receipt()
            snapshot = rss_snapshot()
            _supervised_terminal(
                journal,
                status="FAILED",
                reason=f"WORKER_SPAWN_FAILED:{type(error).__name__}:{error}",
                resume_from_attempt_id=resume_from_attempt_id,
                prior_wall=prior_wall,
                supervisor_started=supervisor_started,
                supervisor_peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
            )
            return 1
        rss_monitor = LiveProcessTreeRssMonitor(os.getpid())
        rss_monitor.start()
        last_persistent_scan = 0.0
        while worker.poll() is None:
            if timeout_requested:
                stop_reason = f"OUTER_TIMEOUT_{MAX_WALL_SECONDS}_SECONDS"
                break
            now = time.monotonic()
            if prior_wall + (now - supervisor_started) > MAX_WALL_SECONDS:
                stop_reason = "CUMULATIVE_WALL_EXCEEDED"
                break
            snapshot = rss_snapshot()
            if rss_monitor.triggered:
                if snapshot["error"] is not None:
                    stop_reason = f"LIVE_RSS_MONITOR_FAILED:{snapshot['error']}"
                else:
                    stop_reason = (
                        f"LIVE_PROCESS_TREE_RSS_EXCEEDED:{snapshot['peak_live_rss_bytes']}"
                    )
                break
            scratch = _scratch_bytes(journal.root)
            if scratch > MAX_SCRATCH_BYTES:
                stop_reason = f"LIVE_SCRATCH_EXCEEDED:{scratch}"
                break
            if now - last_persistent_scan >= 30.0:
                persistent = _persistent_bytes(journal.root)
                if persistent + scratch + FINAL_COMMIT_RESERVE_BYTES > MAX_NEW_BYTES:
                    stop_reason = f"LIVE_WAREHOUSE_STORAGE_EXCEEDED:{persistent + scratch}"
                    break
                last_persistent_scan = now
            time.sleep(SUPERVISOR_POLL_SECONDS)
        if stop_reason is not None:
            _terminate_worker(worker)
            rss_monitor.stop()
            snapshot = rss_snapshot()
            write_supervisor_receipt()
            if _attempt_terminal_path(journal).exists():
                terminal = _load_json(_attempt_terminal_path(journal))
                if terminal.get("status") == "READY":
                    print(json.dumps(terminal["result"], indent=2, sort_keys=True))
                    return 0
            if (
                not _attempt_terminal_path(journal).exists()
                and (journal.root / FINAL_MANIFEST_FILENAME).is_file()
            ):
                _recover_ready_from_manifest(
                    journal,
                    plan,
                    resume_from_attempt_id=resume_from_attempt_id,
                    prior_wall=prior_wall,
                    supervisor_started=supervisor_started,
                    supervisor_peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
                )
                return 0
            _supervised_terminal(
                journal,
                status="STOPPED",
                reason=stop_reason,
                resume_from_attempt_id=resume_from_attempt_id,
                prior_wall=prior_wall,
                supervisor_started=supervisor_started,
                supervisor_peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
            )
            return 124 if timeout_requested else 1
        returncode = worker.wait()
        rss_monitor.stop()
        snapshot = rss_snapshot()
        write_supervisor_receipt()
        if snapshot["error"] is not None or snapshot["peak_live_rss_bytes"] > MAX_PEAK_RSS_BYTES:
            if not _attempt_terminal_path(journal).exists():
                _supervised_terminal(
                    journal,
                    status="STOPPED",
                    reason=(
                        f"LIVE_RSS_MONITOR_FAILED:{snapshot['error']}"
                        if snapshot["error"] is not None
                        else f"LIVE_PROCESS_TREE_RSS_EXCEEDED:{snapshot['peak_live_rss_bytes']}"
                    ),
                    resume_from_attempt_id=resume_from_attempt_id,
                    prior_wall=prior_wall,
                    supervisor_started=supervisor_started,
                    supervisor_peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
                )
            return 1
        if not _attempt_terminal_path(journal).exists():
            if returncode == 0 and (journal.path / "WORKER_RESULT.json").is_file():
                _finalize_worker_result(
                    journal,
                    resume_from_attempt_id=resume_from_attempt_id,
                )
            elif (journal.root / FINAL_MANIFEST_FILENAME).is_file():
                _recover_ready_from_manifest(
                    journal,
                    plan,
                    resume_from_attempt_id=resume_from_attempt_id,
                    prior_wall=prior_wall,
                    supervisor_started=supervisor_started,
                    supervisor_peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
                )
                returncode = 0
            else:
                _supervised_terminal(
                    journal,
                    status="FAILED",
                    reason=f"WORKER_EXITED_WITHOUT_TERMINAL:{returncode}",
                    resume_from_attempt_id=resume_from_attempt_id,
                    prior_wall=prior_wall,
                    supervisor_started=supervisor_started,
                    supervisor_peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
                )
        terminal = _load_json(_attempt_terminal_path(journal))
        if terminal.get("status") == "READY":
            print(json.dumps(terminal["result"], indent=2, sort_keys=True))
            return 0
        return returncode or 1
    except BaseException as error:
        if worker is not None and worker.poll() is None:
            _terminate_worker(worker)
        if rss_monitor is not None and rss_monitor.snapshot()["sample_count"] > 0:
            rss_monitor.stop()
        if (
            not _attempt_terminal_path(journal).exists()
            and not (journal.root / FINAL_MANIFEST_FILENAME).is_file()
        ):
            snapshot = rss_snapshot()
            write_supervisor_receipt()
            _supervised_terminal(
                journal,
                status="FAILED",
                reason=f"SUPERVISOR_EXCEPTION:{type(error).__name__}:{error}",
                resume_from_attempt_id=resume_from_attempt_id,
                prior_wall=prior_wall,
                supervisor_started=supervisor_started,
                supervisor_peak_live_rss_bytes=snapshot["peak_live_rss_bytes"],
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if worker is not None and worker.poll() is None:
            _terminate_worker(worker)
        if (
            rss_monitor is not None
            and rss_monitor._thread is not None
            and rss_monitor._thread.is_alive()
        ):
            rss_monitor.stop()


def _worker_attempt(repository: Path, *, attempt_id: str, resume_from: str | None) -> int:
    _arm_parent_death_signal()
    started_path = EXPECTED_OUTPUT_ROOT / "attempts" / attempt_id / "ATTEMPT_STARTED.json"
    started = _load_json(started_path)
    plan = started.get("plan")
    if not isinstance(plan, dict) or canonical_json_sha256(plan) != started.get("plan_sha256"):
        raise RuntimeError("Worker attempt contains an invalid exact plan")
    journal = _load_existing_journal(EXPECTED_OUTPUT_ROOT, attempt_id, plan)
    if started.get("resume_from_attempt_id") != resume_from:
        raise RuntimeError("Worker resume predecessor differs from attempt start")
    if resume_from is None:
        prior_wall, chain = 0.0, []
    else:
        chain, prior_wall = _load_prior_chain(
            EXPECTED_OUTPUT_ROOT,
            resume_from,
            expected_plan=plan,
        )
    attempt_started_text = os.environ.get("C3_ATTEMPT_MONOTONIC_STARTED")
    if attempt_started_text is None:
        raise RuntimeError("C3 worker lacks the supervisor monotonic start")
    attempt_started = float(attempt_started_text)
    if not np.isfinite(attempt_started) or attempt_started > time.monotonic():
        raise RuntimeError("C3 worker supervisor monotonic start is invalid")
    _run_attempt(
        plan,
        repository,
        journal,
        resume_from_attempt_id=resume_from,
        prior_cumulative_wall_seconds=prior_wall,
        resume_chain=chain,
        attempt_started=attempt_started,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute-initial", action="store_true")
    mode.add_argument("--manual-resume", action="store_true")
    mode.add_argument("--_worker-attempt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--profile",
        choices=("full-r2", "memory-canary-r2", "causal-full-r1", "causal-full-r2"),
        default=PROFILE_NAME,
    )
    parser.add_argument("--attempt-id")
    parser.add_argument("--resume-from")
    parser.add_argument("--output-root", type=Path, default=EXPECTED_OUTPUT_ROOT)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError(f"C3 output root is frozen to {EXPECTED_OUTPUT_ROOT}")
    if args.preflight and (args.attempt_id is not None or args.resume_from is not None):
        raise ValueError("C3 preflight does not accept attempt arguments")
    if args.execute_initial:
        if args.resume_from is not None:
            raise ValueError("Initial C3 execution cannot resume another attempt")
        if args.attempt_id not in {None, INITIAL_ATTEMPT_ID}:
            raise ValueError(f"Initial C3 attempt is fixed to {INITIAL_ATTEMPT_ID}")
    if args.manual_resume and (args.attempt_id is None or args.resume_from is None):
        raise ValueError("Manual resume requires explicit --attempt-id and --resume-from")
    if args._worker_attempt and args.attempt_id is None:
        raise ValueError("Internal C3 worker requires --attempt-id")


def _exact_prior_plan(resume_from_attempt_id: str) -> dict[str, Any]:
    """Load the immutable prior plan before the new manual attempt is created."""

    started = _load_json(
        EXPECTED_OUTPUT_ROOT / "attempts" / resume_from_attempt_id / "ATTEMPT_STARTED.json"
    )
    plan = started.get("plan")
    if (
        not isinstance(plan, dict)
        or canonical_json_sha256(plan) != started.get("plan_sha256")
        or started.get("attempt_id") != resume_from_attempt_id
    ):
        raise RuntimeError("Explicit manual-resume predecessor contains an invalid exact plan")
    return plan


def main() -> int:
    args = _parser().parse_args()
    if args.profile == "memory-canary-r2" and PROFILE_NAME != "memory-canary-r2":
        configure_memory_canary_r2()
    elif args.profile == "causal-full-r1" and PROFILE_NAME != "causal-full-r1":
        configure_causal_full_r1()
    elif args.profile == "causal-full-r2" and PROFILE_NAME != "causal-full-r2":
        configure_causal_full_r2()
    elif args.profile != PROFILE_NAME:
        raise ValueError(
            f"Configured R2 profile differs from CLI: {PROFILE_NAME} != {args.profile}"
        )
    _validate_args(args)
    _force_cpu_offline_environment()
    repository = Path(__file__).resolve(strict=True).parents[2]
    if args._worker_attempt:
        return _worker_attempt(repository, attempt_id=args.attempt_id, resume_from=args.resume_from)
    if not args.preflight:
        _verify_outer_timeout_supervisor()
    attempt_started = time.monotonic()
    plan = (
        _exact_prior_plan(args.resume_from)
        if args.manual_resume
        else build_execution_plan(repository)
    )
    if args.preflight:
        availability = {
            "initial_attempt_id": INITIAL_ATTEMPT_ID,
            "candidate_root_exists": EXPECTED_OUTPUT_ROOT.exists()
            or EXPECTED_OUTPUT_ROOT.is_symlink(),
            "initial_execution_available": not EXPECTED_OUTPUT_ROOT.exists()
            and not EXPECTED_OUTPUT_ROOT.is_symlink(),
        }
        print(json.dumps({"plan": plan, "availability": availability}, indent=2, sort_keys=True))
        return 0
    if args.execute_initial:
        return _supervise_attempt(
            plan,
            repository,
            attempt_id=INITIAL_ATTEMPT_ID,
            resume_from_attempt_id=None,
            attempt_started=attempt_started,
        )
    return _supervise_attempt(
        plan,
        repository,
        attempt_id=args.attempt_id,
        resume_from_attempt_id=args.resume_from,
        attempt_started=attempt_started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
