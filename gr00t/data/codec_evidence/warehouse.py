"""Read-only validation and loading for immutable causal fixture warehouses."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .codec_contract import PRODUCTION_FIXTURE_MODE, codec_contract_sha256
from .contract import SHA256_RE
from .episode_bundle import (
    SOURCE_KINDS,
    EpisodeBundle,
    EpisodeBundleKey,
    EpisodeBundleStore,
    RawMotionVectorAudit,
)


WAREHOUSE_OBJECT_SCHEMA_VERSION = "codec-evidence-warehouse-object-v1"
OBJECT_COMMIT_SCHEMA_VERSION = "codec-evidence-warehouse-object-commit-v1"
DERIVATION_BINDING_SCHEMA_VERSION = "codec-evidence-derivation-binding-v1"
FULL_FIXTURE_MANIFEST_SCHEMA_VERSION = "codec-evidence-full-fixture-manifest-v1"
OBJECT_MANIFEST_FILENAME = "object_manifest.json"
OBJECT_COMMIT_FILENAME = "OBJECT_COMMITTED.json"
FINAL_MANIFEST_FILENAME = "fixture_manifest.json"
FULL_COVERAGE_KIND = "FULL_EPISODE_SET"

_OBJECT_MANIFEST_FIELDS = {
    "schema_version",
    "object_content_sha256",
    "episode_uid",
    "camera_id",
    "source_kind",
    "num_frames",
    "bundle_manifest",
    "artifacts",
}
_ARTIFACT_FIELDS = {"relative_path", "bytes", "sha256"}
_BUNDLE_DESCRIPTOR_FIELDS = {"relative_path", "bytes", "sha256"}
_COMMIT_FIELDS = {
    "schema_version",
    "object_content_sha256",
    "object_manifest_sha256",
}
_DERIVATION_BINDING_FIELDS = {
    "schema_version",
    "derivation_sha256",
    "object_content_sha256",
    "episode_uid",
    "camera_id",
    "source_kind",
}
_MANIFEST_OBJECT_FIELDS = {
    "episode_uid",
    "camera_id",
    "source_kind",
    "num_frames",
    "derivation_sha256",
    "object_content_sha256",
    "bundle_manifest_sha256",
    "bundle_manifest_bytes",
}
_FULL_MANIFEST_FIELDS = {
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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one byte encoding used for warehouse identities."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_plain_component(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be one plain path component")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact is missing, non-regular, or a symlink: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _safe_relative_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe warehouse artifact path: {relative_path!r}")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Symlinks are forbidden in warehouse objects: {relative_path}")
    resolved = (root / relative).resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"Warehouse artifact escapes its object: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"Warehouse artifact is not a regular file: {relative_path}")
    return resolved


def episode_uid_digest(episode_uids: Sequence[str]) -> str:
    values = sorted(episode_uids)
    if len(values) != len(set(values)):
        raise ValueError("episode_uids must be unique")
    for value in values:
        _require_sha(value, "episode_uid")
    return canonical_json_sha256(values)


def _content_digest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "episode_uid": manifest["episode_uid"],
        "camera_id": manifest["camera_id"],
        "source_kind": manifest["source_kind"],
        "num_frames": manifest["num_frames"],
        "bundle_manifest": manifest["bundle_manifest"],
        "artifacts": manifest["artifacts"],
    }


def _object_paths(root: Path, digest: str) -> tuple[Path, Path, Path]:
    object_root = root / "objects" / digest
    return (
        object_root,
        object_root / "payload" / OBJECT_MANIFEST_FILENAME,
        object_root / OBJECT_COMMIT_FILENAME,
    )


def _validate_object_payload(
    root: Path,
    digest: str,
    *,
    require_commit: bool,
    full_bundle: bool,
) -> tuple[dict[str, Any], EpisodeBundleStore]:
    object_root, manifest_path, commit_path = _object_paths(root, digest)
    if object_root.is_symlink() or not object_root.is_dir():
        raise ValueError("Warehouse object is missing or a symlink")
    payload_root = object_root / "payload"
    if payload_root.is_symlink() or not payload_root.is_dir():
        raise ValueError("Warehouse object payload is missing or a symlink")
    manifest = _load_json(manifest_path)
    if (
        set(manifest) != _OBJECT_MANIFEST_FIELDS
        or manifest.get("schema_version") != WAREHOUSE_OBJECT_SCHEMA_VERSION
    ):
        raise ValueError("Warehouse object manifest keys/schema mismatch")
    if manifest.get("object_content_sha256") != digest:
        raise ValueError("Object directory and content digest differ")
    key = EpisodeBundleKey(
        manifest.get("episode_uid"),
        manifest.get("camera_id"),
        manifest.get("source_kind"),
    )
    if type(manifest.get("num_frames")) is not int or manifest["num_frames"] <= 0:
        raise ValueError("Warehouse object num_frames must be positive")
    descriptor = manifest.get("bundle_manifest")
    if not isinstance(descriptor, dict) or set(descriptor) != _BUNDLE_DESCRIPTOR_FIELDS:
        raise ValueError("Warehouse bundle manifest descriptor is malformed")
    expected_relative = f"bundle/{key.episode_uid}/{key.camera_id}/{key.source_kind}/bundle.json"
    if descriptor["relative_path"] != expected_relative:
        raise ValueError("Warehouse bundle manifest path is not canonical")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Warehouse object artifacts must be a non-empty list")
    observed_paths: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise ValueError("Warehouse object artifact descriptor is malformed")
        relative = artifact["relative_path"]
        if not isinstance(relative, str):
            raise ValueError("Warehouse object artifact path must be a string")
        path = _safe_relative_file(payload_root, relative)
        if type(artifact["bytes"]) is not int or artifact["bytes"] != path.stat().st_size:
            raise ValueError(f"Warehouse artifact byte count mismatch: {relative}")
        if sha256_path(path) != _require_sha(artifact["sha256"], "artifact.sha256"):
            raise ValueError(f"Warehouse artifact SHA-256 mismatch: {relative}")
        observed_paths.append(relative)
    if observed_paths != sorted(observed_paths) or len(observed_paths) != len(set(observed_paths)):
        raise ValueError("Warehouse artifact descriptors must be sorted and unique")
    if descriptor not in artifacts:
        raise ValueError("Bundle manifest descriptor is absent from object artifacts")
    if canonical_json_sha256(_content_digest_payload(manifest)) != digest:
        raise ValueError("Warehouse object content digest mismatch")
    expected_payload_files = set(observed_paths) | {OBJECT_MANIFEST_FILENAME}
    if require_commit:
        commit = _load_json(commit_path)
        if (
            set(commit) != _COMMIT_FIELDS
            or commit.get("schema_version") != OBJECT_COMMIT_SCHEMA_VERSION
        ):
            raise ValueError("Warehouse object commit marker keys/schema mismatch")
        expected_commit = {
            "schema_version": OBJECT_COMMIT_SCHEMA_VERSION,
            "object_content_sha256": digest,
            "object_manifest_sha256": sha256_path(manifest_path),
        }
        if commit != expected_commit:
            raise ValueError("Warehouse object commit marker does not bind its manifest")
    elif commit_path.exists() or commit_path.is_symlink():
        raise ValueError("Uncommitted object unexpectedly contains a commit marker")
    observed_payload_files: set[str] = set()
    for current, directory_names, file_names in os.walk(payload_root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if (current_path / directory_name).is_symlink():
                raise ValueError("Symlinks are forbidden inside warehouse objects")
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Only regular files are allowed inside warehouse objects")
            observed_payload_files.add(path.relative_to(payload_root).as_posix())
    if observed_payload_files != expected_payload_files:
        raise ValueError("Warehouse object contains unclosed or missing files")
    root_entries = {path.name for path in object_root.iterdir()}
    expected_root_entries = {"payload"}
    if require_commit:
        expected_root_entries.add(OBJECT_COMMIT_FILENAME)
    if root_entries != expected_root_entries:
        raise ValueError("Warehouse object root contains unclosed or missing entries")

    bundle_root = payload_root / "bundle"
    store = EpisodeBundleStore(
        bundle_root,
        manifest_sha256_by_key={key: descriptor["sha256"]},
    )
    identity = store.identity(key)
    if full_bundle:
        bundle = store.get(key)
        if bundle.num_frames != manifest["num_frames"]:
            raise ValueError("Warehouse object frame count differs from its bundle")
        bundle_manifest = _load_json(_safe_relative_file(payload_root, expected_relative))
        if bundle_manifest.get("raw_motion_vector_audit") is not None:
            store.audit_raw_motion_vectors(key)
        if bundle_manifest.get("accepted_bitstream") is not None:
            store.audit_accepted_bitstream(key)
        if identity["codec_contract"]["fixture_mode"] == PRODUCTION_FIXTURE_MODE and (
            bundle_manifest.get("raw_motion_vector_audit") is None
            or bundle_manifest.get("accepted_bitstream") is None
        ):
            raise ValueError("Production warehouse object lacks required audit artifacts")
    return manifest, store


def verify_object(
    warehouse_root: str | Path,
    object_content_sha256: str,
    *,
    full_bundle: bool = True,
) -> dict[str, Any]:
    requested = Path(warehouse_root)
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    root = requested.resolve(strict=True)
    digest = _require_sha(object_content_sha256, "object_content_sha256")
    manifest, _store = _validate_object_payload(
        root,
        digest,
        require_commit=True,
        full_bundle=full_bundle,
    )
    return manifest


def _binding_path(root: Path, digest: str) -> Path:
    return root / "derivations" / f"{digest}.json"


def _validate_binding(root: Path, derivation_digest: str) -> dict[str, Any]:
    binding = _load_json(_binding_path(root, derivation_digest))
    if (
        set(binding) != _DERIVATION_BINDING_FIELDS
        or binding.get("schema_version") != DERIVATION_BINDING_SCHEMA_VERSION
        or binding.get("derivation_sha256") != derivation_digest
    ):
        raise ValueError("Derivation binding keys/schema/digest mismatch")
    _require_sha(binding.get("object_content_sha256"), "object_content_sha256")
    EpisodeBundleKey(
        binding.get("episode_uid"),
        binding.get("camera_id"),
        binding.get("source_kind"),
    )
    return binding


def _validate_manifest_mapping(
    root: Path,
    manifest: Any,
    *,
    verify_objects: bool,
) -> dict[str, Any]:
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _FULL_MANIFEST_FIELDS
        or manifest.get("schema_version") != FULL_FIXTURE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Full fixture manifest fields/schema mismatch")
    _require_plain_component(manifest.get("fixture_id"), "fixture_id")
    if manifest.get("complete") is not True or manifest.get("coverage_kind") != FULL_COVERAGE_KIND:
        raise ValueError("Full fixture manifest is incomplete or not full episode coverage")
    camera_id = _require_plain_component(manifest.get("camera_id"), "camera_id")
    source_kind = manifest.get("source_kind")
    if source_kind not in SOURCE_KINDS:
        raise ValueError("Full fixture manifest source_kind is unsupported")
    expected_count = manifest.get("expected_episode_count")
    covered_count = manifest.get("covered_episode_count")
    if type(expected_count) is not int or expected_count <= 0 or covered_count != expected_count:
        raise ValueError("Full fixture expected/covered episode counts differ")
    uids = manifest.get("covered_episode_uids")
    if (
        not isinstance(uids, list)
        or uids != sorted(uids)
        or len(uids) != expected_count
        or len(uids) != len(set(uids))
        or manifest.get("covered_episode_uid_digest") != episode_uid_digest(uids)
    ):
        raise ValueError("Full fixture UID closure mismatch")
    _require_sha(manifest.get("input_snapshot_sha256"), "input_snapshot_sha256")
    _require_sha(manifest.get("budget_sha256"), "budget_sha256")
    contract = manifest.get("codec_contract")
    if not isinstance(contract, dict):
        raise ValueError("Full fixture codec_contract must be an object")
    if codec_contract_sha256(contract) != _require_sha(
        manifest.get("codec_contract_sha256"), "codec_contract_sha256"
    ):
        raise ValueError("Full fixture codec contract digest mismatch")
    if contract.get("fixture_mode") != PRODUCTION_FIXTURE_MODE:
        raise ValueError("A complete full fixture requires production fixture mode")
    if contract.get("camera_id") != camera_id or contract.get("source_kind") != source_kind:
        raise ValueError("Full fixture codec contract differs from camera/source identity")
    if not isinstance(manifest.get("producer_identity"), dict) or not manifest["producer_identity"]:
        raise ValueError("Full fixture producer_identity must be a non-empty object")
    entries = manifest.get("objects")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError("Full fixture object count differs from expected coverage")
    if entries != sorted(
        entries,
        key=lambda entry: (
            entry.get("episode_uid"),
            entry.get("camera_id"),
            entry.get("source_kind"),
        ),
    ):
        raise ValueError("Full fixture objects must be in canonical key order")
    observed_uids: list[str] = []
    seen_keys: set[EpisodeBundleKey] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _MANIFEST_OBJECT_FIELDS:
            raise ValueError("Full fixture object entry is malformed")
        key = EpisodeBundleKey(entry["episode_uid"], entry["camera_id"], entry["source_kind"])
        if key in seen_keys or key.camera_id != camera_id or key.source_kind != source_kind:
            raise ValueError("Full fixture object keys are duplicate or mixed")
        seen_keys.add(key)
        observed_uids.append(key.episode_uid)
        if type(entry["num_frames"]) is not int or entry["num_frames"] <= 0:
            raise ValueError("Full fixture object num_frames must be positive")
        derivation_digest = _require_sha(entry["derivation_sha256"], "derivation_sha256")
        object_digest = _require_sha(entry["object_content_sha256"], "object_content_sha256")
        binding = _validate_binding(root, derivation_digest)
        expected_binding = {
            "schema_version": DERIVATION_BINDING_SCHEMA_VERSION,
            "derivation_sha256": derivation_digest,
            "object_content_sha256": object_digest,
            "episode_uid": key.episode_uid,
            "camera_id": key.camera_id,
            "source_kind": key.source_kind,
        }
        if binding != expected_binding:
            raise ValueError("Full fixture derivation binding differs from its entry")
        object_manifest = verify_object(
            root,
            object_digest,
            full_bundle=verify_objects,
        )
        descriptor = object_manifest["bundle_manifest"]
        for name, expected in (
            ("episode_uid", key.episode_uid),
            ("camera_id", key.camera_id),
            ("source_kind", key.source_kind),
            ("num_frames", entry["num_frames"]),
        ):
            if object_manifest[name] != expected:
                raise ValueError(f"Full fixture object {name} differs from its entry")
        if (
            descriptor["sha256"] != entry["bundle_manifest_sha256"]
            or descriptor["bytes"] != entry["bundle_manifest_bytes"]
        ):
            raise ValueError("Full fixture bundle descriptor differs from its object")
        identity = EpisodeBundleStore(
            root / "objects" / object_digest / "payload" / "bundle",
            manifest_sha256_by_key={key: descriptor["sha256"]},
        ).identity(key)
        if (
            identity["codec_contract_sha256"] != manifest["codec_contract_sha256"]
            or identity["codec_contract"] != contract
        ):
            raise ValueError("Full fixture object codec contract drifted")
    if observed_uids != uids:
        raise ValueError("Full fixture object UIDs differ from covered UID closure")
    return manifest


def validate_full_fixture_manifest(
    warehouse_root: str | Path,
    *,
    fixture_manifest_sha256: str,
    verify_objects: bool = True,
) -> dict[str, Any]:
    requested = Path(warehouse_root)
    if requested.is_symlink():
        raise ValueError("warehouse root must not be a symlink")
    root = requested.resolve(strict=True)
    manifest_path = root / FINAL_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Final fixture manifest is missing or a symlink")
    if sha256_path(manifest_path) != _require_sha(
        fixture_manifest_sha256, "fixture_manifest_sha256"
    ):
        raise ValueError("Final fixture manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path)
    return _validate_manifest_mapping(root, manifest, verify_objects=verify_objects)


class WarehouseEpisodeBundleStore:
    """Read current EpisodeBundle objects through one pinned full manifest."""

    def __init__(
        self,
        warehouse_root: str | Path,
        *,
        fixture_manifest_sha256: str,
    ) -> None:
        requested = Path(warehouse_root)
        if requested.is_symlink():
            raise ValueError("warehouse root must not be a symlink")
        self.root = requested.resolve(strict=True)
        self.manifest = validate_full_fixture_manifest(
            self.root,
            fixture_manifest_sha256=fixture_manifest_sha256,
            verify_objects=False,
        )
        self.coverage_kind = self.manifest["coverage_kind"]
        self.fixture_manifest_schema_version = self.manifest["schema_version"]
        self._entries = {
            EpisodeBundleKey(entry["episode_uid"], entry["camera_id"], entry["source_kind"]): entry
            for entry in self.manifest["objects"]
        }
        self._stores: dict[EpisodeBundleKey, EpisodeBundleStore] = {}

    @property
    def keys(self) -> frozenset[EpisodeBundleKey]:
        return frozenset(self._entries)

    def _store(self, key: EpisodeBundleKey) -> EpisodeBundleStore:
        if key not in self._entries:
            raise KeyError(f"Full fixture has no object for {key.canonical}")
        cached = self._stores.get(key)
        if cached is not None:
            return cached
        entry = self._entries[key]
        manifest = verify_object(
            self.root,
            entry["object_content_sha256"],
            full_bundle=True,
        )
        descriptor = manifest["bundle_manifest"]
        store = EpisodeBundleStore(
            self.root / "objects" / entry["object_content_sha256"] / "payload" / "bundle",
            manifest_sha256_by_key={key: descriptor["sha256"]},
        )
        self._stores[key] = store
        return store

    def get(self, key: EpisodeBundleKey) -> EpisodeBundle:
        return self._store(key).get(key)

    def get_view(self, key: EpisodeBundleKey, observation_row: int):
        return self._store(key).get_view(key, observation_row)

    def identity(self, key: EpisodeBundleKey) -> Mapping[str, Any]:
        return self._store(key).identity(key)

    def audit_raw_motion_vectors(
        self,
        key: EpisodeBundleKey,
        *,
        frame_index: int | None = None,
    ) -> RawMotionVectorAudit:
        return self._store(key).audit_raw_motion_vectors(key, frame_index=frame_index)

    def audit_accepted_bitstream(self, key: EpisodeBundleKey) -> Path:
        return self._store(key).audit_accepted_bitstream(key)


__all__ = [
    "FINAL_MANIFEST_FILENAME",
    "FULL_COVERAGE_KIND",
    "FULL_FIXTURE_MANIFEST_SCHEMA_VERSION",
    "WarehouseEpisodeBundleStore",
    "validate_full_fixture_manifest",
    "verify_object",
]
