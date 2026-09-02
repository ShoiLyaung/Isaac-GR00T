"""Fail-closed shared lease for one immutable dataset generation.

The control-plane writer and this reader deliberately share only a filesystem
protocol.  Writers take an exclusive ``flock`` on ``DATASET_GENERATION.lock``;
manifest-backed codec consumers keep a shared lock for the lifetime of the
parent dataset and reacquire an independent shared lock in every worker
process.  In-progress or stale markers are always blocking.

The lock is advisory.  Direct writes that bypass the control-plane protocol
remain unsupported, so every process also verifies the frozen input hashes
when it acquires its lease.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


LOCK_NAME = "DATASET_GENERATION.lock"
BLOCKING_MARKERS = (
    "SYNC_IN_PROGRESS.json",
    "INDEX_IN_PROGRESS.json",
    "STALE_AFTER_SYNC.json",
)


class DatasetGenerationGuardError(RuntimeError):
    """The requested dataset generation is not safe to read."""


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DatasetGenerationGuardError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise DatasetGenerationGuardError(f"Frozen identity is missing or not regular: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetGenerationGuardError(f"Cannot read frozen JSON identity: {path}") from exc
    if not isinstance(value, dict):
        raise DatasetGenerationGuardError(f"Frozen JSON identity must be an object: {path}")
    return value


@dataclass(frozen=True)
class FrozenDatasetGenerationIdentity:
    """Exact files pinned while a reader owns a shared generation lease."""

    dataset_root: Path
    catalog_path: Path
    catalog_sha256: str
    split_manifest_path: Path
    split_manifest_sha256: str
    payload_manifest_path: Path
    payload_manifest_sha256: str
    train_statistics_path: Path
    train_statistics_sha256: str
    train_episode_uid_digest: str

    def sha256_by_name(self) -> dict[str, str]:
        return {
            "catalog": self.catalog_sha256,
            "split_manifest": self.split_manifest_sha256,
            "payload_manifest": self.payload_manifest_sha256,
            "train_statistics": self.train_statistics_sha256,
        }


def _resolve_prelease_contract(
    split_manifest_path: str | Path,
) -> tuple[Path, Path, str, Path, str, str]:
    """Resolve only enough information to locate the generation lock.

    Every value is read and validated again after the lock is acquired.  This
    first read therefore cannot authorize dataset access.
    """

    split_path = Path(split_manifest_path).resolve()
    manifest = _load_json(split_path)
    dataset_root_value = manifest.get("dataset_root")
    if not isinstance(dataset_root_value, str) or not dataset_root_value:
        raise DatasetGenerationGuardError("Split manifest has no dataset_root")
    dataset_root = Path(dataset_root_value).resolve()
    catalog_path = Path(str(manifest.get("catalog", {}).get("database_path", ""))).resolve()
    catalog_sha = manifest.get("catalog", {}).get("database_sha256")
    payload_sha = manifest.get("payload_manifest_sha256")
    train_uid_digest = manifest.get("train_episode_uid_digest")
    for value, name in (
        (catalog_sha, "catalog SHA-256"),
        (payload_sha, "payload-manifest SHA-256"),
        (train_uid_digest, "train UID digest"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise DatasetGenerationGuardError(f"Split manifest has no valid {name}")
    sibling = split_path.with_name("payload_manifest.json")
    declared = Path(str(manifest.get("payload_manifest_path", ""))).resolve()
    payload_path = (
        sibling
        if sibling.is_file() and not sibling.is_symlink() and _sha256_path(sibling) == payload_sha
        else declared
    )
    return dataset_root, catalog_path, catalog_sha, payload_path, payload_sha, train_uid_digest


class DatasetGenerationGuard:
    """Process-aware lifetime shared lease plus frozen identity verifier."""

    def __init__(self, identity: FrozenDatasetGenerationIdentity, *, blocking: bool = False):
        self.identity = identity
        self.blocking = bool(blocking)
        self._descriptor: int | None = None
        self._owner_pid: int | None = None
        self.ensure_current_process()

    @classmethod
    def acquire_for_manifest(
        cls,
        split_manifest_path: str | Path,
        train_statistics_path: str | Path,
        *,
        blocking: bool = False,
    ) -> "DatasetGenerationGuard":
        split_path = Path(split_manifest_path).resolve()
        stats_path = Path(train_statistics_path).resolve()
        (
            dataset_root,
            catalog_path,
            catalog_sha,
            payload_path,
            payload_sha,
            train_uid_digest,
        ) = _resolve_prelease_contract(split_path)

        # Acquire before trusting any of the values read above.  Readers never
        # create the lock sidecar; its absence means the control protocol has
        # not committed a readable generation.
        provisional = object.__new__(cls)
        provisional.identity = FrozenDatasetGenerationIdentity(
            dataset_root=dataset_root,
            catalog_path=catalog_path,
            catalog_sha256=catalog_sha,
            split_manifest_path=split_path,
            split_manifest_sha256="0" * 64,
            payload_manifest_path=payload_path,
            payload_manifest_sha256=payload_sha,
            train_statistics_path=stats_path,
            train_statistics_sha256="0" * 64,
            train_episode_uid_digest=train_uid_digest,
        )
        provisional.blocking = bool(blocking)
        provisional._descriptor = None
        provisional._owner_pid = None
        provisional._acquire_descriptor(verify_identity=False)
        try:
            # Re-resolve under the lease and reject a generation switch between
            # the pre-read and the lock acquisition.
            resolved = _resolve_prelease_contract(split_path)
            expected = (
                dataset_root,
                catalog_path,
                catalog_sha,
                payload_path,
                payload_sha,
                train_uid_digest,
            )
            if resolved != expected:
                raise DatasetGenerationGuardError(
                    "Dataset generation changed while acquiring the shared lease"
                )
            expected_catalog_path = dataset_root / "catalog/current.sqlite3"
            if catalog_path != expected_catalog_path:
                raise DatasetGenerationGuardError(
                    "Split manifest does not bind the current dataset catalog"
                )
            for path, expected_sha, name in (
                (catalog_path, catalog_sha, "catalog"),
                (payload_path, payload_sha, "payload manifest"),
            ):
                if path.is_symlink() or not path.is_file() or _sha256_path(path) != expected_sha:
                    raise DatasetGenerationGuardError(f"Frozen {name} identity mismatch")
            stats = _load_json(stats_path)
            if stats.get("train_episode_uid_digest") != train_uid_digest:
                raise DatasetGenerationGuardError(
                    "Train statistics use a different train episode UID set"
                )
            if stats.get("catalog_database_sha256") != catalog_sha:
                raise DatasetGenerationGuardError("Train statistics use a different catalog")
            identity = FrozenDatasetGenerationIdentity(
                dataset_root=dataset_root,
                catalog_path=catalog_path,
                catalog_sha256=catalog_sha,
                split_manifest_path=split_path,
                split_manifest_sha256=_sha256_path(split_path),
                payload_manifest_path=payload_path,
                payload_manifest_sha256=payload_sha,
                train_statistics_path=stats_path,
                train_statistics_sha256=_sha256_path(stats_path),
                train_episode_uid_digest=train_uid_digest,
            )
            provisional.identity = identity
            provisional.verify_frozen_identity()
            return provisional
        except BaseException:
            provisional.close()
            raise

    @property
    def catalog_dir(self) -> Path:
        return self.identity.dataset_root / "catalog"

    @property
    def lock_path(self) -> Path:
        return self.catalog_dir / LOCK_NAME

    @property
    def owner_pid(self) -> int | None:
        return self._owner_pid

    @property
    def is_held_by_current_process(self) -> bool:
        return self._descriptor is not None and self._owner_pid == os.getpid()

    def _check_markers(self) -> None:
        present: list[str] = []
        for name in BLOCKING_MARKERS:
            path = self.catalog_dir / name
            try:
                status = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
                raise DatasetGenerationGuardError(
                    f"Dataset generation marker is not a regular file: {path}"
                )
            present.append(name)
        if present:
            raise DatasetGenerationGuardError(
                f"Dataset generation is blocked by marker(s): {', '.join(present)}"
            )

    def _open_lock(self) -> int:
        for path, label in (
            (self.identity.dataset_root, "dataset root"),
            (self.catalog_dir, "dataset catalog"),
        ):
            try:
                status = path.lstat()
            except FileNotFoundError as exc:
                raise DatasetGenerationGuardError(f"{label} is missing: {path}") from exc
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise DatasetGenerationGuardError(f"{label} is not a real directory: {path}")
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.lock_path, flags)
        except OSError as exc:
            raise DatasetGenerationGuardError(
                f"Dataset generation lock is unavailable: {self.lock_path}"
            ) from exc
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            os.close(descriptor)
            raise DatasetGenerationGuardError("Dataset generation lock must be a regular file")
        return descriptor

    def _assert_lock_identity(self) -> None:
        if self._descriptor is None:
            raise DatasetGenerationGuardError("Dataset generation lease is not held")
        descriptor_status = os.fstat(self._descriptor)
        try:
            path_status = self.lock_path.lstat()
        except FileNotFoundError as exc:
            raise DatasetGenerationGuardError(
                "Dataset generation lock disappeared while the lease was held"
            ) from exc
        if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
            raise DatasetGenerationGuardError(
                "Dataset generation lock path is no longer a regular file"
            )
        if (descriptor_status.st_dev, descriptor_status.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            raise DatasetGenerationGuardError(
                "Dataset generation lock identity changed while the lease was held"
            )

    def _acquire_descriptor(self, *, verify_identity: bool) -> None:
        descriptor = self._open_lock()
        operation = fcntl.LOCK_SH | (0 if self.blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise DatasetGenerationGuardError(
                "Dataset generation has an active sync/index writer"
            ) from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self._owner_pid = os.getpid()
        try:
            self._assert_lock_identity()
            self._check_markers()
            if verify_identity:
                self.verify_frozen_identity()
        except BaseException:
            self.close()
            raise

    def ensure_current_process(self) -> None:
        """Own an independent lease in the current parent or worker process."""

        current_pid = os.getpid()
        if self._descriptor is not None and self._owner_pid == current_pid:
            self._assert_lock_identity()
            self._check_markers()
            return
        if self._descriptor is not None:
            # A fork inherited the descriptor/open-file description.  Drop the
            # child copy, then acquire a fresh shared lock owned by this worker.
            try:
                os.close(self._descriptor)
            finally:
                self._descriptor = None
                self._owner_pid = None
        self._acquire_descriptor(verify_identity=True)

    def verify_frozen_identity(self) -> None:
        self._assert_lock_identity()
        self._check_markers()
        identities: tuple[tuple[Path, str, str], ...] = (
            (self.identity.catalog_path, self.identity.catalog_sha256, "catalog"),
            (
                self.identity.split_manifest_path,
                self.identity.split_manifest_sha256,
                "split manifest",
            ),
            (
                self.identity.payload_manifest_path,
                self.identity.payload_manifest_sha256,
                "payload manifest",
            ),
            (
                self.identity.train_statistics_path,
                self.identity.train_statistics_sha256,
                "train statistics",
            ),
        )
        for path, expected, name in identities:
            if path.is_symlink() or not path.is_file():
                raise DatasetGenerationGuardError(f"Frozen {name} is missing or not regular")
            if _sha256_path(path) != expected:
                raise DatasetGenerationGuardError(f"Frozen {name} SHA-256 changed")

    def identity_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.identity.dataset_root),
            "catalog_path": str(self.identity.catalog_path),
            "catalog_sha256": self.identity.catalog_sha256,
            "split_manifest_path": str(self.identity.split_manifest_path),
            "split_manifest_sha256": self.identity.split_manifest_sha256,
            "payload_manifest_path": str(self.identity.payload_manifest_path),
            "payload_manifest_sha256": self.identity.payload_manifest_sha256,
            "train_statistics_path": str(self.identity.train_statistics_path),
            "train_statistics_sha256": self.identity.train_statistics_sha256,
            "train_episode_uid_digest": self.identity.train_episode_uid_digest,
        }

    def close(self) -> None:
        descriptor, owner_pid = self._descriptor, self._owner_pid
        self._descriptor = None
        self._owner_pid = None
        if descriptor is None:
            return
        try:
            # Never unlock an inherited open-file description from a child;
            # closing the child duplicate is sufficient and preserves parent
            # ownership until the parent closes its own duplicate.
            if owner_pid == os.getpid():
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __getstate__(self) -> dict[str, Any]:
        return {"identity": self.identity, "blocking": self.blocking}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.identity = state["identity"]
        self.blocking = bool(state["blocking"])
        self._descriptor = None
        self._owner_pid = None

    def __enter__(self) -> "DatasetGenerationGuard":
        self.ensure_current_process()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


__all__ = [
    "BLOCKING_MARKERS",
    "DatasetGenerationGuard",
    "DatasetGenerationGuardError",
    "FrozenDatasetGenerationIdentity",
    "LOCK_NAME",
]
