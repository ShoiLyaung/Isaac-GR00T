from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from gr00t.data.codec_evidence.codec_contract import (
    PRODUCTION_FIXTURE_MODE,
    codec_contract_sha256,
    make_codec_contract,
)
from gr00t.data.codec_evidence.episode_bundle import SOURCE_KIND_NATIVE_CANONICAL, EpisodeBundleKey
import pytest


def _load_cli():
    path = Path(__file__).parents[2] / "examples/G1_ALL_LEROBOT_CODEC/build_c2_fixture_canary.py"
    spec = importlib.util.spec_from_file_location("c2_fixture_canary_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(module):
    identity = {"name": "focused", "version": "1", "sha256": "f" * 64}
    contract = make_codec_contract(
        fixture_mode=PRODUCTION_FIXTURE_MODE,
        source_kind=SOURCE_KIND_NATIVE_CANONICAL,
        identity=identity,
    )
    sources = [
        {
            "source_id": f"source-{index}",
            "episode_uid": f"{index + 1:064x}",
            "num_frames": 40,
        }
        for index in range(5)
    ]
    return {
        "schema_version": module.SCHEMA_VERSION,
        "attempt_id": module.ATTEMPT_ID,
        "plan_sha256": "a" * 64,
        "selection_policy": "focused",
        "scope": "focused",
        "sources": sources,
        "input_file_closure": {},
        "toolchain": {"implementation": {}, "identity_sha256": "b" * 64},
        "codec_contract": contract,
        "codec_contract_sha256": codec_contract_sha256(contract),
    }


def _make_writable(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)
    root.chmod(0o755)


def test_execute_commits_exact_five_index_last_and_is_non_overwritable(tmp_path, monkeypatch):
    module = _load_cli()
    output = tmp_path / "fixture"
    monkeypatch.setattr(module, "EXPECTED_OUTPUT_ROOT", output)
    monkeypatch.setattr(module, "_check_dataset_markers", lambda: None)
    monkeypatch.setattr(module, "_verify_toolchain_identity", lambda _value: None)
    write_order: list[str] = []
    original_write_json = module._write_json_new

    def recording_write(path, value, *, mode=0o444):
        write_order.append(Path(path).name)
        return original_write_json(path, value, mode=mode)

    monkeypatch.setattr(module, "_write_json_new", recording_write)
    keys: list[EpisodeBundleKey] = []

    def fake_write_bundle(output_root, source, plan, event_path):
        del plan, event_path
        key = EpisodeBundleKey(
            source["episode_uid"], module.CAMERA_ID, SOURCE_KIND_NATIVE_CANONICAL
        )
        keys.append(key)
        bitstream = output_root / "fake" / f"{source['source_id']}.bin"
        bitstream.parent.mkdir(parents=True, exist_ok=True)
        bitstream.write_bytes(b"accepted")
        result = {
            "anchors": 2,
            "raw_mv_count": 1,
            "accepted_bitstream": module._file_identity(bitstream),
        }
        module._write_json_new(
            output_root / "episode_results" / f"{source['source_id']}.json", result
        )
        return (
            {
                "episode_uid": key.episode_uid,
                "camera_id": key.camera_id,
                "source_kind": key.source_kind,
                "relative_path": f"fake/{source['source_id']}.json",
                "sha256": "c" * 64,
                "bytes": 1,
            },
            key,
        )

    monkeypatch.setattr(module, "_write_bundle", fake_write_bundle)

    class FakeStore:
        def __init__(self):
            self.keys = frozenset(keys)

        def get(self, key):
            assert key in self.keys

        def audit_raw_motion_vectors(self, key):
            assert key in self.keys

        def audit_accepted_bitstream(self, key):
            assert key in self.keys

    monkeypatch.setattr(
        module,
        "episode_bundle_store_from_index_manifest",
        lambda *args, **kwargs: FakeStore(),
    )
    summary = module.execute(_plan(module), Path(__file__).parents[2])
    assert summary["status"] == "TECHNICALLY_VERIFIED_READY_FOR_TERMINAL_COMMIT"
    assert summary["source_count"] == 5
    assert (output / "fixture_manifest.json").is_file()
    assert (output / "ATTEMPT_TERMINAL.json").is_file()
    assert write_order[-1] == "ATTEMPT_TERMINAL.json"
    with pytest.raises(FileExistsError):
        module._reserve_output_root()
    _make_writable(output)


def test_execute_failure_consumes_attempt_without_retry(tmp_path, monkeypatch):
    module = _load_cli()
    output = tmp_path / "failed"
    monkeypatch.setattr(module, "EXPECTED_OUTPUT_ROOT", output)
    monkeypatch.setattr(module, "_check_dataset_markers", lambda: None)
    monkeypatch.setattr(module, "_verify_toolchain_identity", lambda _value: None)

    def fail(*args, **kwargs):
        raise RuntimeError("focused fixture failure")

    monkeypatch.setattr(module, "_write_bundle", fail)
    with pytest.raises(RuntimeError, match="focused fixture failure"):
        module.execute(_plan(module), Path(__file__).parents[2])
    failure = module._load_json(output / "ATTEMPT_FAILED.json")
    assert failure["status"] == "FAILED_ATTEMPT_CONSUMED_NO_RETRY"
    assert failure["error_type"] == "RuntimeError"
    assert not (output / "fixture_manifest.json").exists()
    _make_writable(output)


def test_toolchain_file_closure_is_rechecked_after_freeze(tmp_path):
    module = _load_cli()
    tool = tmp_path / "tool.bin"
    tool.write_bytes(b"frozen")
    identity = {"tool": module._file_identity(tool)}
    identity["identity_sha256"] = module._canonical_json_sha256(identity)
    module._verify_toolchain_identity(identity)
    tool.write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="toolchain file drifted"):
        module._verify_toolchain_identity(identity)


def test_execute_requires_frozen_outer_timeout_supervisor(monkeypatch):
    module = _load_cli()
    monkeypatch.delenv("C2_OUTER_TIMEOUT_SECONDS", raising=False)
    with pytest.raises(RuntimeError, match="requires the frozen outer"):
        module._verify_outer_timeout_supervisor()


def test_double_hashed_encode_source_is_rechecked_before_bundle_copy(tmp_path):
    module = _load_cli()
    bitstream = tmp_path / "accepted.mp4"
    bitstream.write_bytes(b"frozen-bitstream")
    digest = module.sha256_path(bitstream)
    encoded = SimpleNamespace(
        bitstream_path=bitstream,
        bitstream_sha256=digest,
        attempt_sha256=(digest, digest),
    )
    source_identity = module._verified_encode_source_identity(encoded)
    retained = tmp_path / "copied-under-a-different-name.bin"
    retained.write_bytes(bitstream.read_bytes())
    retained_identity = module._file_identity(retained)
    assert retained_identity != source_identity
    assert module._content_identity(retained_identity) == module._content_identity(source_identity)
    bitstream.write_bytes(b"drifted-after-double-hash")
    with pytest.raises(RuntimeError, match="drifted after the deterministic double hash"):
        module._verified_encode_source_identity(encoded)
