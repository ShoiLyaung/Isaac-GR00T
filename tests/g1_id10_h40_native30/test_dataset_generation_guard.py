from __future__ import annotations

import fcntl
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import pickle

from gr00t.data.dataset.dataset_generation_guard import (
    BLOCKING_MARKERS,
    DatasetGenerationGuard,
    DatasetGenerationGuardError,
)
import pytest


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "dataset"
    catalog_dir = root / "catalog"
    catalog_dir.mkdir(parents=True)
    (catalog_dir / "DATASET_GENERATION.lock").touch(mode=0o600)
    catalog = catalog_dir / "current.sqlite3"
    catalog.write_bytes(b"frozen-catalog")
    catalog_sha = _hash(catalog)
    payload = tmp_path / "payload_manifest.json"
    _write_json(payload, {"payload": "frozen"})
    uid = "a" * 64
    train_digest = "b" * 64
    split = tmp_path / "split_manifest.json"
    _write_json(
        split,
        {
            "dataset_root": str(root),
            "catalog": {
                "database_path": str(catalog),
                "database_sha256": catalog_sha,
            },
            "payload_manifest_path": str(payload),
            "payload_manifest_sha256": _hash(payload),
            "train_episode_uid_digest": train_digest,
            "assignments": [{"episode_uid": uid, "eligible": True, "split": "train"}],
        },
    )
    stats = tmp_path / "train_statistics.json"
    _write_json(
        stats,
        {
            "catalog_database_sha256": catalog_sha,
            "train_episode_uid_digest": train_digest,
        },
    )
    return root, split, stats


def _ensure_worker(guard: DatasetGenerationGuard, queue) -> None:
    try:
        guard.ensure_current_process()
        queue.put(("ok", os.getpid(), guard.owner_pid))
    except BaseException as exc:  # pragma: no cover - returned to parent
        queue.put(("error", type(exc).__name__, str(exc)))
    finally:
        guard.close()


def test_shared_guard_pins_all_four_identities_and_blocks_exclusive(tmp_path):
    root, split, stats = _contract(tmp_path)
    guard = DatasetGenerationGuard.acquire_for_manifest(split, stats)
    assert guard.is_held_by_current_process
    assert set(guard.identity.sha256_by_name()) == {
        "catalog",
        "split_manifest",
        "payload_manifest",
        "train_statistics",
    }
    descriptor = os.open(root / "catalog/DATASET_GENERATION.lock", os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)
        guard.close()


@pytest.mark.parametrize("marker", BLOCKING_MARKERS)
def test_every_generation_marker_fails_closed(tmp_path, marker):
    root, split, stats = _contract(tmp_path)
    _write_json(root / "catalog" / marker, {"state": "blocked"})
    with pytest.raises(DatasetGenerationGuardError, match="blocked by marker"):
        DatasetGenerationGuard.acquire_for_manifest(split, stats)


def test_dangling_marker_symlink_fails_closed(tmp_path):
    root, split, stats = _contract(tmp_path)
    (root / "catalog/SYNC_IN_PROGRESS.json").symlink_to(tmp_path / "missing")
    with pytest.raises(DatasetGenerationGuardError, match="not a regular file"):
        DatasetGenerationGuard.acquire_for_manifest(split, stats)


def test_direct_identity_drift_is_detected_even_while_lease_is_held(tmp_path):
    _, split, stats = _contract(tmp_path)
    guard = DatasetGenerationGuard.acquire_for_manifest(split, stats)
    stats.write_text(stats.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(DatasetGenerationGuardError, match="train statistics SHA-256 changed"):
        guard.verify_frozen_identity()
    guard.close()


def test_lock_path_replacement_is_detected_while_held(tmp_path):
    root, split, stats = _contract(tmp_path)
    guard = DatasetGenerationGuard.acquire_for_manifest(split, stats)
    lock = root / "catalog/DATASET_GENERATION.lock"
    moved = root / "catalog/DATASET_GENERATION.lock.replaced"
    lock.rename(moved)
    lock.touch(mode=0o600)
    with pytest.raises(DatasetGenerationGuardError, match="identity changed"):
        guard.ensure_current_process()
    guard.close()


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_worker_reacquires_process_owned_shared_lease(tmp_path, start_method):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable")
    _, split, stats = _contract(tmp_path)
    guard = DatasetGenerationGuard.acquire_for_manifest(split, stats)
    parent_pid = guard.owner_pid
    context = multiprocessing.get_context(start_method)
    queue = context.Queue()
    process = context.Process(target=_ensure_worker, args=(guard, queue))
    process.start()
    result = queue.get(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
    assert result[0] == "ok", result
    assert result[1] == result[2]
    assert result[1] != parent_pid
    assert guard.owner_pid == parent_pid
    guard.close()


def test_pickle_never_serializes_a_live_file_descriptor(tmp_path):
    _, split, stats = _contract(tmp_path)
    guard = DatasetGenerationGuard.acquire_for_manifest(split, stats)
    restored = pickle.loads(pickle.dumps(guard))
    assert restored.owner_pid is None
    assert not restored.is_held_by_current_process
    restored.ensure_current_process()
    assert restored.owner_pid == os.getpid()
    restored.close()
    guard.close()


def test_reader_never_creates_a_missing_generation_lock(tmp_path):
    root, split, stats = _contract(tmp_path)
    lock = root / "catalog/DATASET_GENERATION.lock"
    lock.unlink()
    with pytest.raises(DatasetGenerationGuardError, match="lock is unavailable"):
        DatasetGenerationGuard.acquire_for_manifest(split, stats)
    assert not lock.exists()
