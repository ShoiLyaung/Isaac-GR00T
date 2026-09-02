#!/usr/bin/env python3
"""Bound and supervise the one authorized full-episode CPU canary process tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import time
from typing import Any, Sequence


SCHEMA_VERSION = "codec-evidence-full-episode-supervisor-v1"
DEFAULT_MAX_WALL_SECONDS = 10_800.0
DEFAULT_MAX_PROCESS_RSS_BYTES = 24 * 1024**3
DEFAULT_MAX_WAREHOUSE_BYTES = 4 * 1024**3
DEFAULT_MAX_SCRATCH_BYTES = 4 * 1024**3
LIMIT_EXIT_CODE = 125
WALL_EXIT_CODE = 124
ATTEMPT_SCHEMA_VERSION = "codec-evidence-warehouse-attempt-v1"
IMPLEMENTATION_SCHEMA_VERSION = "codec-evidence-c1-implementation-manifest-v1"
SNAPSHOT_SCHEMA_VERSION = "codec-evidence-fixture-snapshot-v1"
RUN_ID = "GR00T-N17-CODEC-EVIDENCE-FULL-EPISODE-CANARY-R1"
ATTEMPT_ID = f"{RUN_ID}-ATTEMPT1"
WORKTREE = Path(
    "/home/sl/agent-team-workspaces/icra27-gr00t-worktrees/"
    "g1-id10-h40-native30-cpu-r1"
)
PYTHON = Path("/home/sl/miniconda3/envs/gr00t_n1d7/bin/python")
BWRAP = Path("/usr/bin/bwrap")
FIXTURE_PARENT = Path("/home/sl/groot_codec_fixtures")
PROTECTED_R5_ENTRIES = tuple(
    FIXTURE_PARENT / name
    for name in (
        "R5-CPU-PILOT-20260820T014534+0800-ATTEMPT1",
        "R5-CPU-PILOT-20260820T014534+0800-ATTEMPT1-VERIFICATION.json",
        "R5-CPU-PILOT-20260820T014534+0800-ATTEMPT1-FINAL-REVIEW-R2.json",
        "R5-CPU-PILOT-20260820T014534+0800-ATTEMPT1-RESULT.json",
        "R5-CPU-PILOT-20260820T014534+0800-ATTEMPT1-VERIFICATION-R2.json",
    )
)
WAREHOUSE = FIXTURE_PARENT / "GR00T-N17-CODEC-EVIDENCE-WAREHOUSE-R1"
SCRATCH = (
    FIXTURE_PARENT
    / "GR00T-N17-CODEC-EVIDENCE-FULL-EPISODE-CANARY-R1-SCRATCH-ATTEMPT1"
)
GENERATION_LOCK = Path("/home/sl/groot_datasets/catalog/DATASET_GENERATION.lock")
SPLIT_MANIFEST = Path(
    "/home/sl/agent-team-workspaces/icra27-control/runs/"
    "DATASET-DUAL-VLA-CONTRACT-P0P2-CPU-R4/20260814T134737+0800/"
    "CONTRACT/split_manifest.json"
)
TRAIN_STATISTICS = SPLIT_MANIFEST.parents[1] / "GR00T_TRAIN_STATISTICS.json"
PAYLOAD_MANIFEST = SPLIT_MANIFEST.with_name("payload_manifest.json")
CATALOG = Path("/home/sl/groot_datasets/catalog/current.sqlite3")
ADAPTER_CONFIG = SPLIT_MANIFEST.parents[1] / "GR00T_ADAPTER_CONFIG_TRAIN.json"
MODALITY_CONFIG = WORKTREE / "examples/G1_ID10_H40_NATIVE30/g1_id10_h40_native30_config.py"
RUNNER = WORKTREE / "examples/G1_ID10_H40_NATIVE30/build_codec_full_episode_canary.py"
LAUNCH_FINETUNE = WORKTREE / "gr00t/experiment/launch_finetune.py"
BASE_MODEL = Path("/home/sl/checkpoints/GR00T-N1.7-3B")
_SHA_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FrozenIdentity:
    path: Path
    sha256: str


@dataclass(frozen=True)
class TerminalContext:
    attempt_id: str
    implementation_manifest: FrozenIdentity
    generation_lock: Path
    catalog: FrozenIdentity
    split_manifest: FrozenIdentity
    payload_manifest: FrozenIdentity
    train_statistics: FrozenIdentity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required JSON is missing, non-regular, or a symlink: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA-256 for {label}")
    return value


def _verify_frozen_file(identity: FrozenIdentity, label: str) -> dict[str, Any]:
    requested = identity.path
    if requested.is_symlink():
        raise ValueError(f"Frozen {label} must not be a symlink")
    path = requested.resolve(strict=True)
    if not path.is_file() or _sha256_path(path) != _validate_sha(identity.sha256, label):
        raise ValueError(f"Frozen {label} identity drifted")
    return {"path": str(path), "sha256": identity.sha256, "bytes": path.stat().st_size}


def _verify_implementation_manifest(identity: FrozenIdentity) -> dict[str, Any]:
    manifest_identity = _verify_frozen_file(identity, "implementation manifest")
    manifest = _load_json(identity.path)
    if (
        set(manifest) != {"schema_version", "run_id", "complete", "files"}
        or manifest["schema_version"] != IMPLEMENTATION_SCHEMA_VERSION
        or manifest["run_id"] != RUN_ID
        or manifest["complete"] is not True
        or not isinstance(manifest["files"], dict)
        or not manifest["files"]
    ):
        raise ValueError("Implementation manifest schema/run/completeness mismatch")
    verified_files: dict[str, Any] = {}
    observed_paths: set[Path] = set()
    for role, descriptor in sorted(manifest["files"].items()):
        if (
            not isinstance(role, str)
            or not isinstance(descriptor, dict)
            or set(descriptor) != {"path", "sha256", "bytes"}
        ):
            raise ValueError("Implementation file descriptor schema mismatch")
        requested = Path(descriptor["path"])
        if not requested.is_absolute() or requested.is_symlink():
            raise ValueError(f"Implementation path is unsafe: {role}")
        path = requested.resolve(strict=True)
        if path in observed_paths or not path.is_file():
            raise ValueError(f"Implementation path is duplicate or non-regular: {role}")
        observed_paths.add(path)
        digest = _validate_sha(descriptor["sha256"], role)
        if (
            type(descriptor["bytes"]) is not int
            or descriptor["bytes"] != path.stat().st_size
            or _sha256_path(path) != digest
        ):
            raise ValueError(f"Implementation artifact drifted: {role}")
        verified_files[role] = dict(descriptor)
    return {**manifest_identity, "file_count": len(verified_files), "files": verified_files}


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _directory_bytes(path: Path) -> int:
    """Count regular-file bytes without following any symlink."""

    if path.is_symlink():
        raise ValueError(f"Supervised output root must not be a symlink: {path}")
    if not path.exists():
        return 0
    if not path.is_dir():
        raise ValueError(f"Supervised output root is not a directory: {path}")
    total = 0
    for current, directory_names, file_names in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            if (current_path / name).is_symlink():
                raise ValueError("Symlinks are forbidden in supervised output trees")
        for name in file_names:
            child = current_path / name
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("Symlinks are forbidden in supervised output trees")
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Only regular files are allowed in supervised output trees")
            total += metadata.st_size
    return total


def production_command(implementation_manifest: str | Path) -> list[str]:
    """Return the sole accepted, network-isolated C1 child command."""

    implementation = Path(implementation_manifest)
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": "/tmp/hf",
        "HF_HUB_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "",
        "HOME": "/tmp/home",
        "LANG": "C.UTF-8",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "42",
        "PYTHONPATH": str(WORKTREE),
        "ROCR_VISIBLE_DEVICES": "",
        "TMPDIR": "/tmp",
        "TORCH_COMPILE": "0",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/tmp/xdg",
    }
    command = [
        str(BWRAP),
        "--die-with-parent",
        "--unshare-net",
        "--new-session",
        "--clearenv",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/home",
        "--dir",
        "/tmp/hf",
        "--dir",
        "/tmp/xdg",
        "--bind",
        str(FIXTURE_PARENT),
        str(FIXTURE_PARENT),
    ]
    for protected in PROTECTED_R5_ENTRIES:
        command.extend(("--ro-bind", str(protected), str(protected)))
    command.extend(
        (
            "--bind",
            str(GENERATION_LOCK),
            str(GENERATION_LOCK),
            "--chdir",
            str(WORKTREE),
        )
    )
    for name, value in sorted(environment.items()):
        command.extend(("--setenv", name, value))
    command.extend(
        (
            "--",
            str(PYTHON),
            str(RUNNER),
            "--split-manifest",
            str(SPLIT_MANIFEST),
            "--train-statistics",
            str(TRAIN_STATISTICS),
            "--adapter-config-path",
            str(ADAPTER_CONFIG),
            "--modality-config-path",
            str(MODALITY_CONFIG),
            "--dataset-path",
            "/home/sl/groot_datasets/lerobot",
            "--base-model-path",
            str(BASE_MODEL),
            "--launch-finetune",
            str(LAUNCH_FINETUNE),
            "--implementation-manifest",
            str(implementation),
            "--warehouse-root",
            str(WAREHOUSE),
            "--scratch-root",
            str(SCRATCH),
            "--ffmpeg",
            "/usr/bin/ffmpeg",
            "--ffprobe",
            "/usr/bin/ffprobe",
        )
    )
    return command


def validate_production_command(
    command: Sequence[str], implementation_manifest: str | Path
) -> list[str]:
    normalized = [str(item) for item in command]
    expected = production_command(implementation_manifest)
    if normalized != expected:
        raise ValueError(
            "Supervisor CLI accepts only the exact bwrap-isolated authorized C1 command"
        )
    return normalized


def _protected_entry_identities(entries: Sequence[Path]) -> dict[str, Any]:
    """Hash the complete pre-existing fixture surface without following links."""

    result: dict[str, Any] = {}
    for entry in entries:
        if entry.is_symlink() or not entry.exists():
            raise ValueError(f"Protected fixture entry is missing or a symlink: {entry}")
        descriptors: list[dict[str, Any]] = []
        if entry.is_file():
            descriptors.append(
                {
                    "relative_path": entry.name,
                    "bytes": entry.stat().st_size,
                    "sha256": _sha256_path(entry),
                }
            )
        elif entry.is_dir():
            for current, directory_names, file_names in os.walk(entry, followlinks=False):
                current_path = Path(current)
                for name in directory_names:
                    if (current_path / name).is_symlink():
                        raise ValueError("Protected fixture directory contains a symlink")
                for name in file_names:
                    path = current_path / name
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("Protected fixture contains a non-regular file")
                    descriptors.append(
                        {
                            "relative_path": path.relative_to(entry).as_posix(),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256_path(path),
                        }
                    )
            descriptors.sort(key=lambda item: item["relative_path"])
        else:
            raise ValueError("Protected fixture entry must be a regular file or directory")
        encoded = json.dumps(
            descriptors,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        result[str(entry)] = {
            "kind": "directory" if entry.is_dir() else "file",
            "regular_file_count": len(descriptors),
            "regular_file_bytes": sum(item["bytes"] for item in descriptors),
            "closure_sha256": sha256(encoded).hexdigest(),
            "files": descriptors,
        }
    return result


class _FinalizationLimit(RuntimeError):
    def __init__(self, reason: str, detail: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _sample_terminal_budget(
    warehouse: Path,
    scratch: Path,
    *,
    started_monotonic: float,
    max_wall_seconds: float,
    max_warehouse_bytes: int,
    max_scratch_bytes: int,
    wall_margin_seconds: float = 0.0,
) -> dict[str, Any]:
    sample = {
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "warehouse_bytes": _directory_bytes(warehouse),
        "scratch_bytes": _directory_bytes(scratch),
    }
    if sample["elapsed_seconds"] + wall_margin_seconds > max_wall_seconds:
        raise _FinalizationLimit(
            "WALL_LIMIT_EXCEEDED_AFTER_READY",
            {"observed": sample["elapsed_seconds"], "limit": max_wall_seconds},
        )
    if sample["warehouse_bytes"] > max_warehouse_bytes:
        raise _FinalizationLimit(
            "WAREHOUSE_LIMIT_EXCEEDED_AFTER_READY",
            {"observed": sample["warehouse_bytes"], "limit": max_warehouse_bytes},
        )
    if sample["scratch_bytes"] > max_scratch_bytes:
        raise _FinalizationLimit(
            "SCRATCH_LIMIT_EXCEEDED_AFTER_READY",
            {"observed": sample["scratch_bytes"], "limit": max_scratch_bytes},
        )
    return sample


def _validate_ready_attempt(warehouse: Path, attempt_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_id) is None:
        raise ValueError("Unsafe attempt_id")
    attempt = warehouse / "attempts" / attempt_id
    if attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("Worker exited zero without an attempt directory")
    terminal_paths = [attempt / f"ATTEMPT_{status}.json" for status in ("PASS", "FAILED", "STOPPED")]
    if any(path.exists() or path.is_symlink() for path in terminal_paths):
        raise ValueError("Worker must leave READY with no terminal attempt record")
    started = _load_json(attempt / "ATTEMPT_STARTED.json")
    ready_path = attempt / "ATTEMPT_READY.json"
    ready = _load_json(ready_path)
    if (
        set(started) != {"schema_version", "attempt_id", "status", "plan_sha256", "plan"}
        or started["schema_version"] != ATTEMPT_SCHEMA_VERSION
        or started["attempt_id"] != attempt_id
        or started["status"] != "STARTED"
        or set(ready) != {"schema_version", "attempt_id", "status", "result"}
        or ready["schema_version"] != ATTEMPT_SCHEMA_VERSION
        or ready["attempt_id"] != attempt_id
        or ready["status"] != "READY"
        or not isinstance(ready["result"], dict)
        or ready["result"].get("status") != "READY_FOR_SUPERVISOR_FINALIZATION"
    ):
        raise ValueError("ATTEMPT_STARTED/READY schema or status mismatch")
    snapshot_path = Path(str(ready["result"].get("snapshot_manifest_path", "")))
    expected_snapshot = warehouse / "canaries" / attempt_id / "fixture_snapshot_manifest.json"
    if snapshot_path.is_symlink() or snapshot_path.resolve(strict=True) != expected_snapshot.resolve(
        strict=True
    ):
        raise ValueError("READY points outside the exact canary snapshot path")
    declared_snapshot_sha = _validate_sha(
        ready["result"].get("snapshot_manifest_sha256"), "READY snapshot"
    )
    if _sha256_path(snapshot_path) != declared_snapshot_sha:
        raise ValueError("READY snapshot SHA-256 drifted")
    snapshot = _load_json(snapshot_path)
    if (
        snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or snapshot.get("snapshot_id") != attempt_id
        or snapshot.get("complete") is not True
        or snapshot.get("coverage_kind") != "CANARY_SUBSET_NONTRAINING"
        or snapshot.get("training_eligible") is not False
        or snapshot.get("source_kind") != "native_canonical_decoded_derived"
        or snapshot.get("camera_id") != "cam_left_high"
    ):
        raise ValueError("READY snapshot is not the exact nontraining canonical canary")
    return {
        "attempt_path": str(attempt),
        "ready_path": str(ready_path),
        "ready_sha256": _sha256_path(ready_path),
        "snapshot_manifest_path": str(snapshot_path),
        "snapshot_manifest_sha256": declared_snapshot_sha,
    }


def _verify_terminal_identities(context: TerminalContext) -> dict[str, Any]:
    requested_lock = context.generation_lock
    if requested_lock.is_symlink() or not requested_lock.is_file():
        raise ValueError("Dataset generation lock is missing or a symlink")
    lock_stat = requested_lock.stat(follow_symlinks=False)
    with requested_lock.open("rb") as generation_stream:
        opened_stat = os.fstat(generation_stream.fileno())
        if (lock_stat.st_dev, lock_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            raise ValueError("Dataset generation lock path was replaced")
        fcntl.flock(generation_stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            catalog_root = context.catalog.path.parent
            for marker in (
                catalog_root / "SYNC_IN_PROGRESS.json",
                catalog_root / "INDEX_IN_PROGRESS.json",
                catalog_root / "STALE_AFTER_SYNC.json",
            ):
                if marker.exists() or marker.is_symlink():
                    raise ValueError(f"Dataset generation marker is present: {marker.name}")
            frozen = {
                "catalog": _verify_frozen_file(context.catalog, "catalog"),
                "split_manifest": _verify_frozen_file(context.split_manifest, "split manifest"),
                "payload_manifest": _verify_frozen_file(
                    context.payload_manifest, "payload manifest"
                ),
                "train_statistics": _verify_frozen_file(
                    context.train_statistics, "train statistics"
                ),
            }
            implementation = _verify_implementation_manifest(
                context.implementation_manifest
            )
            current_lock = requested_lock.stat(follow_symlinks=False)
            if (current_lock.st_dev, current_lock.st_ino) != (
                opened_stat.st_dev,
                opened_stat.st_ino,
            ):
                raise ValueError("Dataset generation lock changed during terminal verification")
        finally:
            fcntl.flock(generation_stream.fileno(), fcntl.LOCK_UN)
    return {"frozen_dataset": frozen, "implementation": implementation}


def _finalize_ready_attempt(
    warehouse: Path,
    scratch: Path,
    context: TerminalContext,
    *,
    started_monotonic: float,
    max_wall_seconds: float,
    max_warehouse_bytes: int,
    max_scratch_bytes: int,
) -> dict[str, Any]:
    """Promote READY to PASS only after a counted candidate and frozen recheck."""

    producer_lock = warehouse / "WAREHOUSE_PRODUCER.lock"
    if producer_lock.is_symlink() or not producer_lock.is_file():
        raise ValueError("Warehouse coordinator lock is missing or a symlink")
    with producer_lock.open("a+b") as producer_stream:
        fcntl.flock(producer_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with context.generation_lock.open("rb") as terminal_generation_lease:
                fcntl.flock(
                    terminal_generation_lease.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB
                )
                ready = _validate_ready_attempt(warehouse, context.attempt_id)
                identities = _verify_terminal_identities(context)
                attempt = warehouse / "attempts" / context.attempt_id
                pending = attempt / "ATTEMPT_PASS_PENDING.json"
                terminal = attempt / "ATTEMPT_PASS.json"
                if (
                    pending.exists()
                    or pending.is_symlink()
                    or terminal.exists()
                    or terminal.is_symlink()
                ):
                    raise FileExistsError("Supervisor terminal candidate/PASS already exists")
                pass_payload = {
                    "schema_version": ATTEMPT_SCHEMA_VERSION,
                    "attempt_id": context.attempt_id,
                    "status": "PASS",
                    "result": {
                        "authority": "OUTER_RESOURCE_AND_IDENTITY_SUPERVISOR",
                        "ready": ready,
                        "terminal_identities": identities,
                    },
                }
                _write_json_exclusive(pending, pass_payload)
                before_promotion = _sample_terminal_budget(
                    warehouse,
                    scratch,
                    started_monotonic=started_monotonic,
                    max_wall_seconds=max_wall_seconds,
                    max_warehouse_bytes=max_warehouse_bytes,
                    max_scratch_bytes=max_scratch_bytes,
                    wall_margin_seconds=1.0,
                )
                os.rename(pending, terminal)
                directory_fd = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                after_promotion = _sample_terminal_budget(
                    warehouse,
                    scratch,
                    started_monotonic=started_monotonic,
                    max_wall_seconds=max_wall_seconds,
                    max_warehouse_bytes=max_warehouse_bytes,
                    max_scratch_bytes=max_scratch_bytes,
                )
                if (
                    before_promotion["warehouse_bytes"]
                    != after_promotion["warehouse_bytes"]
                    or before_promotion["scratch_bytes"]
                    != after_promotion["scratch_bytes"]
                ):
                    raise RuntimeError("Output bytes changed across terminal promotion")
                fcntl.flock(terminal_generation_lease.fileno(), fcntl.LOCK_UN)
            return {
                "attempt_pass_path": str(terminal),
                "attempt_pass_sha256": _sha256_path(terminal),
                "ready": ready,
                "terminal_identities": identities,
                "budget_before_promotion": before_promotion,
                "budget_after_promotion": after_promotion,
            }
        finally:
            fcntl.flock(producer_stream.fileno(), fcntl.LOCK_UN)


def _process_table() -> dict[int, tuple[int, int, int, str]]:
    """Read Linux procfs as pid -> (ppid, process-group, RSS bytes, state)."""

    page_size = os.sysconf("SC_PAGE_SIZE")
    table: dict[int, tuple[int, int, int, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            closing_parenthesis = raw.rfind(")")
            fields = raw[closing_parenthesis + 2 :].split()
            if closing_parenthesis < 0 or len(fields) <= 21:
                continue
            pid = int(entry.name)
            state = fields[0]
            ppid = int(fields[1])
            process_group = int(fields[2])
            rss_bytes = max(0, int(fields[21])) * page_size
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        table[pid] = (ppid, process_group, rss_bytes, state)
    return table


def _active_tree(
    root_pid: int,
    table: dict[int, tuple[int, int, int, str]],
    known_descendants: set[int],
) -> tuple[set[int], set[int]]:
    """Return the live recursive descendants plus every member of the root PGID."""

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _group, _rss, state) in table.items():
            if state != "Z" and ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    known_descendants.update(descendants)
    active = {
        pid
        for pid, (_ppid, process_group, _rss, state) in table.items()
        if state != "Z"
        and (pid in known_descendants or process_group == root_pid or pid == root_pid)
    }
    return active, known_descendants


def _signal_process_tree(process_group: int, pids: set[int], signum: int) -> None:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        pass
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _terminate_process_tree(
    process: subprocess.Popen[Any],
    active_pids: set[int],
    known_descendants: set[int],
    grace_seconds: float,
) -> dict[str, Any]:
    """Terminate the whole process group, then hard-kill any surviving descendant."""

    process_group = process.pid
    _signal_process_tree(process_group, active_pids, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    survivors = set(active_pids)
    while survivors and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        table = _process_table()
        survivors, known_descendants = _active_tree(process.pid, table, known_descendants)
    hard_kill_sent = bool(survivors)
    if survivors:
        _signal_process_tree(process_group, survivors, signal.SIGKILL)
    try:
        return_code = process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        _signal_process_tree(process_group, survivors | {process.pid}, signal.SIGKILL)
        return_code = process.wait(timeout=2.0)
        hard_kill_sent = True
    return {
        "sigterm_sent": True,
        "sigkill_sent": hard_kill_sent,
        "last_observed_process_ids": sorted(active_pids),
        "child_returncode": return_code,
    }


def _propagated_child_exit_code(return_code: int) -> int:
    if return_code >= 0:
        return min(return_code, 255)
    return min(128 + abs(return_code), 255)


def _child_rlimit_setter(max_address_space_bytes: int, max_file_bytes: int):
    def apply_limits() -> None:
        resource.setrlimit(
            resource.RLIMIT_AS, (max_address_space_bytes, max_address_space_bytes)
        )
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))

    return apply_limits


def _write_reserved_evidence(file_descriptor: int, path: Path, payload: dict[str, Any]) -> None:
    with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def supervise(
    command: Sequence[str],
    *,
    evidence_path: str | Path,
    warehouse_root: str | Path,
    scratch_root: str | Path,
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS,
    max_process_rss_bytes: int = DEFAULT_MAX_PROCESS_RSS_BYTES,
    max_warehouse_bytes: int = DEFAULT_MAX_WAREHOUSE_BYTES,
    max_scratch_bytes: int = DEFAULT_MAX_SCRATCH_BYTES,
    poll_interval_seconds: float = 0.25,
    termination_grace_seconds: float = 2.0,
    terminal_context: TerminalContext | None = None,
    protected_entries: Sequence[Path] = (),
) -> tuple[int, dict[str, Any]]:
    """Run one command under continuous process-tree and output-byte limits."""

    normalized_command = [str(item) for item in command]
    if not normalized_command:
        raise ValueError("A supervised command is required")
    limits: dict[str, int | float] = {
        "max_wall_seconds": max_wall_seconds,
        "max_process_rss_bytes": max_process_rss_bytes,
        "max_tree_rss_bytes": max_process_rss_bytes,
        "max_warehouse_bytes": max_warehouse_bytes,
        "max_scratch_bytes": max_scratch_bytes,
    }
    if (
        max_wall_seconds <= 0
        or max_process_rss_bytes <= 0
        or max_warehouse_bytes <= 0
        or max_scratch_bytes <= 0
        or poll_interval_seconds <= 0
        or termination_grace_seconds < 0
    ):
        raise ValueError("Supervisor limits and poll interval must be positive")
    evidence = Path(evidence_path)
    if not evidence.is_absolute() or evidence.is_symlink() or not evidence.parent.is_dir():
        raise ValueError("Evidence path needs an existing real absolute parent")
    # Reserving the exact JSON inode before Popen makes no-clobber a prelaunch fact.
    evidence_fd = os.open(evidence, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    warehouse = Path(warehouse_root)
    scratch = Path(scratch_root)
    started_monotonic = time.monotonic()
    started_utc = _utc_now()
    process: subprocess.Popen[Any] | None = None
    active_pids: set[int] = set()
    known_descendants: set[int] = set()
    child_returncode: int | None = None
    termination_reason = "SUPERVISOR_INTERNAL_ERROR"
    termination_detail: dict[str, Any] = {}
    kill_evidence = {
        "sigterm_sent": False,
        "sigkill_sent": False,
        "last_observed_process_ids": [],
        "child_returncode": None,
    }
    peaks = {
        "process_count": 0,
        "any_process_rss_bytes": 0,
        "tree_rss_bytes": 0,
        "warehouse_bytes": 0,
        "scratch_bytes": 0,
    }
    poll_count = 0
    pending_signal: list[int | None] = [None]
    previous_handlers: dict[int, Any] = {}
    terminal_authority: dict[str, Any] | None = None
    protected_prelaunch: dict[str, Any] = {}
    protected_postrun: dict[str, Any] = {}
    child_rlimits = {
        "RLIMIT_AS_bytes": max_process_rss_bytes,
        "RLIMIT_FSIZE_bytes": max(max_warehouse_bytes, max_scratch_bytes),
    }

    def receive_signal(signum: int, _frame: Any) -> None:
        pending_signal[0] = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, receive_signal)
        protected_prelaunch = _protected_entry_identities(protected_entries)
        process = subprocess.Popen(
            normalized_command,
            start_new_session=True,
            preexec_fn=_child_rlimit_setter(
                max_process_rss_bytes,
                max(max_warehouse_bytes, max_scratch_bytes),
            ),
        )
        known_descendants.add(process.pid)
        while True:
            poll_count += 1
            elapsed = time.monotonic() - started_monotonic
            table = _process_table()
            active_pids, known_descendants = _active_tree(process.pid, table, known_descendants)
            rss_values = [table[pid][2] for pid in active_pids if pid in table]
            any_rss = max(rss_values, default=0)
            tree_rss = sum(rss_values)
            warehouse_bytes = _directory_bytes(warehouse)
            scratch_bytes = _directory_bytes(scratch)
            peaks["process_count"] = max(peaks["process_count"], len(active_pids))
            peaks["any_process_rss_bytes"] = max(peaks["any_process_rss_bytes"], any_rss)
            peaks["tree_rss_bytes"] = max(peaks["tree_rss_bytes"], tree_rss)
            peaks["warehouse_bytes"] = max(peaks["warehouse_bytes"], warehouse_bytes)
            peaks["scratch_bytes"] = max(peaks["scratch_bytes"], scratch_bytes)

            if pending_signal[0] is not None:
                termination_reason = "SUPERVISOR_SIGNAL"
                termination_detail = {"signal": pending_signal[0]}
            elif elapsed > max_wall_seconds:
                termination_reason = "WALL_LIMIT_EXCEEDED"
                termination_detail = {"observed": elapsed, "limit": max_wall_seconds}
            elif any_rss > max_process_rss_bytes:
                termination_reason = "PROCESS_RSS_LIMIT_EXCEEDED"
                termination_detail = {
                    "observed": any_rss,
                    "limit": max_process_rss_bytes,
                }
            elif tree_rss > max_process_rss_bytes:
                termination_reason = "TREE_RSS_LIMIT_EXCEEDED"
                termination_detail = {
                    "observed": tree_rss,
                    "limit": max_process_rss_bytes,
                }
            elif warehouse_bytes > max_warehouse_bytes:
                termination_reason = "WAREHOUSE_LIMIT_EXCEEDED"
                termination_detail = {
                    "observed": warehouse_bytes,
                    "limit": max_warehouse_bytes,
                }
            elif scratch_bytes > max_scratch_bytes:
                termination_reason = "SCRATCH_LIMIT_EXCEEDED"
                termination_detail = {
                    "observed": scratch_bytes,
                    "limit": max_scratch_bytes,
                }
            else:
                root_returncode = process.poll()
                if root_returncode is not None and not active_pids:
                    child_returncode = root_returncode
                    termination_reason = "CHILD_EXITED"
                    break
                time.sleep(poll_interval_seconds)
                continue

            kill_evidence = _terminate_process_tree(
                process,
                active_pids,
                known_descendants,
                termination_grace_seconds,
            )
            child_returncode = int(kill_evidence["child_returncode"])
            break
    except BaseException as error:
        termination_reason = "SUPERVISOR_INTERNAL_ERROR"
        termination_detail = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        if process is not None and process.poll() is None:
            table = _process_table()
            active_pids, known_descendants = _active_tree(process.pid, table, known_descendants)
            kill_evidence = _terminate_process_tree(
                process,
                active_pids,
                known_descendants,
                termination_grace_seconds,
            )
            child_returncode = int(kill_evidence["child_returncode"])
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if (
        termination_reason == "CHILD_EXITED"
        and child_returncode == 0
        and process is not None
    ):
        try:
            protected_postrun = _protected_entry_identities(protected_entries)
            if protected_postrun != protected_prelaunch:
                raise ValueError("A protected pre-existing fixture changed during C1")
            if terminal_context is not None:
                terminal_authority = _finalize_ready_attempt(
                    warehouse,
                    scratch,
                    terminal_context,
                    started_monotonic=started_monotonic,
                    max_wall_seconds=max_wall_seconds,
                    max_warehouse_bytes=max_warehouse_bytes,
                    max_scratch_bytes=max_scratch_bytes,
                )
                peaks["warehouse_bytes"] = max(
                    peaks["warehouse_bytes"],
                    terminal_authority["budget_after_promotion"]["warehouse_bytes"],
                )
                peaks["scratch_bytes"] = max(
                    peaks["scratch_bytes"],
                    terminal_authority["budget_after_promotion"]["scratch_bytes"],
                )
        except _FinalizationLimit as error:
            termination_reason = error.reason
            termination_detail = error.detail
        except BaseException as error:
            termination_reason = "TERMINAL_FINALIZATION_FAILED"
            termination_detail = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

    elapsed = time.monotonic() - started_monotonic
    if termination_reason == "CHILD_EXITED" and child_returncode is not None:
        exit_code = _propagated_child_exit_code(child_returncode)
    elif termination_reason == "WALL_LIMIT_EXCEEDED":
        exit_code = WALL_EXIT_CODE
    elif termination_reason == "SUPERVISOR_SIGNAL":
        exit_code = 128 + int(termination_detail["signal"])
    else:
        exit_code = LIMIT_EXIT_CODE
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "status": "PASS" if termination_reason == "CHILD_EXITED" and exit_code == 0 else "FAIL",
        "termination_reason": termination_reason,
        "termination_detail": termination_detail,
        "command": normalized_command,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "limits": limits,
        "observed_peaks": peaks,
        "poll_count": poll_count,
        "child_pid": process.pid if process is not None else None,
        "process_group_id": process.pid if process is not None else None,
        "child_returncode": child_returncode,
        "propagated_exit_code": exit_code,
        "kill_evidence": kill_evidence,
        "child_rlimits": child_rlimits,
        "protected_fixture_prelaunch": protected_prelaunch,
        "protected_fixture_postrun": protected_postrun,
        "terminal_authority": terminal_authority,
    }
    _write_reserved_evidence(evidence_fd, evidence, payload)
    return exit_code, payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--warehouse-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS)
    parser.add_argument("--max-process-rss-bytes", type=int, default=DEFAULT_MAX_PROCESS_RSS_BYTES)
    parser.add_argument("--max-warehouse-bytes", type=int, default=DEFAULT_MAX_WAREHOUSE_BYTES)
    parser.add_argument("--max-scratch-bytes", type=int, default=DEFAULT_MAX_SCRATCH_BYTES)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.25)
    parser.add_argument("--termination-grace-seconds", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command must follow --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    exit_code, payload = supervise(
        args.command,
        evidence_path=args.evidence,
        warehouse_root=args.warehouse_root,
        scratch_root=args.scratch_root,
        max_wall_seconds=args.max_wall_seconds,
        max_process_rss_bytes=args.max_process_rss_bytes,
        max_warehouse_bytes=args.max_warehouse_bytes,
        max_scratch_bytes=args.max_scratch_bytes,
        poll_interval_seconds=args.poll_interval_seconds,
        termination_grace_seconds=args.termination_grace_seconds,
    )
    print(json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
