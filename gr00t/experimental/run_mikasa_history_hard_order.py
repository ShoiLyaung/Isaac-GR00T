#!/usr/bin/env python3
"""Run one write-once phase of the MIKASA history-hard native-order experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Final, Mapping


REPO: Final = Path(__file__).resolve().parents[2]
OVERLAY_ROOT: Final = Path(
    "/home/sl/research/icra27-artifacts/"
    "g1-arena-codec-streaming-backbone-r1/dependencies/python-overlay"
)
if str(OVERLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERLAY_ROOT))
if str(REPO) not in sys.path:
    sys.path.insert(1, str(REPO))

import torch  # noqa: E402

from gr00t.experimental.mikasa_history_hard_order import CONDITIONS  # noqa: E402
from gr00t.experimental.mikasa_history_hard_order_experiment import (  # noqa: E402
    DEV_SHUFFLE_SEEDS,
    TRAIN_SEEDS,
    run_dev_evaluation,
    validate_matched_checkpoints,
)
from gr00t.experimental.mikasa_history_hard_order_runtime import (  # noqa: E402
    FRESH_PURE_ROOT,
    TARGET_MANIFEST_SHA256,
    canonical_json_bytes,
    freeze_directory,
    load_order_data,
    sha256_bytes,
    sha256_file,
    validate_frozen_evidence,
    write_once,
)


EXPERIMENT_ID: Final = "g1-mikasa-history-hard-native-order-dev-evaluator-r2"
TRAINING_EXPERIMENT_ID: Final = "g1-mikasa-history-hard-native-order-backbones-r1"
TRAINING_SOURCE_COMMIT: Final = "6f9bf220132fc896d3ec0ce3decd13d316bd0692"
BASE_COMMIT: Final = TRAINING_SOURCE_COMMIT
TRAINING_SOURCE_RESULT_SHA256: Final = (
    "28bd9dd5525f7b5f627b2beef24f262ffc529490c0d75bae163900110157057d"
)
TRAINING_CACHE_RESULT_SHA256: Final = (
    "4ad8bca4d6c1f9f71880cc73260e247380de0f28616639abfea73343a1c87c0f"
)
TRAINING_MATCHED_RESULT_SHA256: Final = (
    "5dd8e56d05fa81cb42c603b2c88397286f32bbb7f27e048b1e39d7668aba5eaa"
)
MAMBA_VERSION: Final = "2.3.2.post1"
SOURCE_PATHS: Final = (
    "gr00t/experimental/mikasa_history_hard_order.py",
    "gr00t/experimental/mikasa_history_hard_order_runtime.py",
    "gr00t/experimental/mikasa_history_hard_order_experiment.py",
    "gr00t/experimental/run_mikasa_history_hard_order.py",
    "tests/test_mikasa_history_hard_order.py",
    "tests/test_mikasa_history_hard_order_runtime.py",
    "tests/test_mikasa_history_hard_order_experiment.py",
)
EVALUATOR_REPAIR_PATHS: Final = (
    "gr00t/experimental/mikasa_history_hard_order_experiment.py",
    "gr00t/experimental/run_mikasa_history_hard_order.py",
    "tests/test_mikasa_history_hard_order_experiment.py",
)
OVERLAY_MATERIALS: Final = {
    "mamba2.py": (
        OVERLAY_ROOT / "mamba_ssm/modules/mamba2.py",
        17_479,
        "605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707",
    ),
    "selective_scan_cuda.so": (
        OVERLAY_ROOT / "selective_scan_cuda.cpython-312-x86_64-linux-gnu.so",
        419_076_688,
        "e9489a6a560600d87f540636a46bf7f0ac86b69b19b5231eb838c5de06166de6",
    ),
    "RECORD": (
        OVERLAY_ROOT / "mamba_ssm-2.3.2.post1.dist-info/RECORD",
        5_974,
        "c6aedc73d29161fdb121caae8ec793617bfd05ed48992706f7981a0df19b4d4d",
    ),
}
PHASES: Final = ("source_acceptance", "dev")


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode(errors="replace").strip()) from exc


def verify_overlay(*, import_package: bool) -> dict[str, Any]:
    records = {}
    for name, (path, size, digest) in OVERLAY_MATERIALS.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"frozen Mamba overlay material changed: {name}")
        observed = sha256_file(path)
        if observed != digest:
            raise RuntimeError(f"frozen Mamba overlay hash changed: {name}")
        records[name] = {"path": str(path), "bytes": size, "sha256": digest}
    package_file = None
    if import_package:
        package = importlib.import_module("mamba_ssm")
        package_file = Path(package.__file__).resolve()
        if not package_file.is_relative_to(OVERLAY_ROOT.resolve()):
            raise RuntimeError(f"mamba_ssm did not resolve from overlay: {package_file}")
        if importlib.metadata.version("mamba-ssm") != MAMBA_VERSION:
            raise RuntimeError("mamba_ssm version changed")
        if importlib.util.find_spec("causal_conv1d") is not None or "causal_conv1d" in sys.modules:
            raise RuntimeError("causal_conv1d must remain absent for the qualified overlay")
    return {
        "root": str(OVERLAY_ROOT.resolve()),
        "version": MAMBA_VERSION,
        "package_file": str(package_file) if package_file is not None else None,
        "causal_conv1d_absent": importlib.util.find_spec("causal_conv1d") is None,
        "materials": records,
    }


def source_identity(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    _git(repo, "merge-base", "--is-ancestor", TRAINING_SOURCE_COMMIT, head)
    raw = _git(repo, "diff", "--name-status", f"{TRAINING_SOURCE_COMMIT}..{head}").decode()
    observed = {}
    for line in raw.splitlines():
        status_code, path = line.split("\t", maxsplit=1)
        observed[path] = status_code
    expected = {path: "M" for path in EVALUATOR_REPAIR_PATHS}
    if observed != expected:
        raise RuntimeError(f"native-order evaluator exact source diff changed: {observed}")
    dirty = (
        _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_PATHS)
        .decode()
        .splitlines()
    )
    if dirty:
        raise RuntimeError(f"native-order evaluator source is dirty: {dirty}")
    files = {}
    for relative in SOURCE_PATHS:
        path = repo / relative
        value = path.read_bytes()
        if path.is_symlink() or value != _git(repo, "show", f"HEAD:{relative}"):
            raise RuntimeError(f"native-order source differs from HEAD: {relative}")
        files[relative] = {"bytes": len(value), "sha256": sha256_bytes(value)}
    return {
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "head_commit": head,
        "branch": _git(repo, "branch", "--show-current").decode().strip(),
        "exact_diff": observed,
        "files": files,
        "owned_source_clean": True,
    }


def training_source_identity() -> dict[str, Any]:
    return {
        "experiment_id": TRAINING_EXPERIMENT_ID,
        "head_commit": TRAINING_SOURCE_COMMIT,
        "source_acceptance_result_sha256": TRAINING_SOURCE_RESULT_SHA256,
        "target_blind_cache_result_sha256": TRAINING_CACHE_RESULT_SHA256,
        "matched_training_result_sha256": TRAINING_MATCHED_RESULT_SHA256,
        "checkpoint_schema": f"{TRAINING_EXPERIMENT_ID}/checkpoint-v1",
        "checkpoint_selection": "all_18_final_step_1500_only",
    }


def protocol() -> dict[str, Any]:
    return {
        "contract": "evaluator-r2",
        "conditions": list(CONDITIONS),
        "training_seeds": list(TRAIN_SEEDS),
        "dev_shuffle_seeds": list(DEV_SHUFFLE_SEEDS),
        "dev_groups": [6, 7],
        "cue_substitution": {
            "rolled_channels": [0, 1, 2],
            "recipient_current_channels": [3, 4, 5],
            "recomputed_delta_channels": [6, 7, 8],
            "label_permutation": "roll_minus_one_within_sixpack",
        },
        "matched_noninferiority_control": "per_metric_max_of_gru_full_and_transformer_full",
        "training_source": training_source_identity(),
        "dev_wall_seconds": 15 * 60,
        "dev_artifact_limit_bytes": 64 * 1024 * 1024,
        "retry_count": 0,
        "target_manifest_sha256_frozen_but_unopened": TARGET_MANIFEST_SHA256,
    }


def run_source_acceptance(output_root: Path, repo: Path = REPO) -> dict[str, Any]:
    if os.path.lexists(output_root):
        raise FileExistsError(output_root)
    evaluator_source = source_identity(repo)
    overlay = verify_overlay(import_package=True)
    result = {
        "schema": f"{EXPERIMENT_ID}/source-acceptance-v1",
        "status": "passed",
        "mode": "source_acceptance",
        "source_identity": evaluator_source,
        "evaluator_source_identity": evaluator_source,
        "training_source_identity": training_source_identity(),
        "frozen_evidence": validate_frozen_evidence(include_target=False),
        "mamba_overlay": overlay,
        "protocol": protocol(),
        "fresh_pure_root_opened_only_for_result": str(FRESH_PURE_ROOT),
        "fresh_data_paths_opened": False,
        "gpu_contexts": 0,
        "simulator_runs": 0,
        "model_runs": 0,
        "optimizer_steps": 0,
        "retry_count": 0,
    }
    output_root.mkdir(parents=True)
    write_once(output_root / "RESULT.json", canonical_json_bytes(result))
    freeze_directory(output_root)
    return result


def bind_runtime_source(source: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Recompute the evaluator identity before any data or checkpoint is opened."""

    if (
        source.get("schema") != f"{EXPERIMENT_ID}/source-acceptance-v1"
        or source.get("protocol") != protocol()
        or source.get("training_source_identity") != training_source_identity()
    ):
        raise PermissionError("evaluator-r2 source acceptance contract changed")
    accepted = source.get("evaluator_source_identity")
    if accepted != source.get("source_identity"):
        raise PermissionError("evaluator-r2 source identities disagree")
    current = source_identity(repo)
    if current != accepted:
        raise PermissionError("runtime evaluator source identity changed")
    return current


def validate_training_lineage(
    cache: Mapping[str, Any], matched: Mapping[str, Any] | None = None
) -> None:
    source_receipt = cache.get("source_acceptance")
    if (
        not isinstance(source_receipt, Mapping)
        or source_receipt.get("sha256") != TRAINING_SOURCE_RESULT_SHA256
    ):
        raise PermissionError("target-blind cache training-source lineage changed")
    if matched is None:
        return
    provenance = matched.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PermissionError("matched-training provenance missing")
    matched_source = provenance.get("source_acceptance")
    matched_cache = provenance.get("cache")
    if (
        not isinstance(matched_source, Mapping)
        or matched_source.get("sha256") != TRAINING_SOURCE_RESULT_SHA256
        or not isinstance(matched_cache, Mapping)
        or matched_cache.get("sha256") != TRAINING_CACHE_RESULT_SHA256
    ):
        raise PermissionError("matched-training frozen lineage changed")


def _read_receipt(
    path: Path, expected_sha256: str, mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise PermissionError(f"{mode} receipt must be regular mode 0444")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if len(expected_sha256) != 64 or observed != expected_sha256:
        raise PermissionError(f"{mode} receipt identity changed")
    value = json.loads(raw)
    if value.get("status") != "passed" or value.get("mode") != mode:
        raise PermissionError(f"{mode} did not pass")
    return value, {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": observed,
        "mode": "0444",
    }


def _cuda0(name: str) -> torch.device:
    device = torch.device(name)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "0"):
        raise RuntimeError("formal GPU phases require CUDA_VISIBLE_DEVICES unset or exactly 0")
    if device.type != "cuda" or device.index not in (None, 0) or not torch.cuda.is_available():
        raise RuntimeError("formal GPU phases require physical cuda:0")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _cache_identity(
    cache_root: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_receipt(cache_root / "RESULT.json", expected_sha256, "target_blind_cache")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-sha256")
    parser.add_argument("--matched-root", type=Path)
    parser.add_argument("--matched-sha256")
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
        source, source_identity_record = _read_receipt(
            _required(args.source_result, "--source-result"),
            _required(args.source_sha256, "--source-sha256"),
            "source_acceptance",
        )
        evaluator_source = bind_runtime_source(source, args.repo_root)
        if source.get("mamba_overlay") != verify_overlay(import_package=True):
            raise PermissionError("source acceptance protocol/overlay changed")
        cache_sha256 = _required(args.cache_sha256, "--cache-sha256")
        if cache_sha256 != TRAINING_CACHE_RESULT_SHA256:
            raise PermissionError("unfrozen target-blind cache result")
        cache_root = _required(args.cache_root, "--cache-root")
        cache, cache_identity_record = _cache_identity(cache_root, cache_sha256)
        validate_training_lineage(cache)
        data, data_receipt = load_order_data(cache_root)

        matched_sha256 = _required(args.matched_sha256, "--matched-sha256")
        if matched_sha256 != TRAINING_MATCHED_RESULT_SHA256:
            raise PermissionError("unfrozen matched-training result")
        matched_root = _required(args.matched_root, "--matched-root")
        matched, matched_identity = _read_receipt(
            matched_root / "RESULT.json",
            matched_sha256,
            "matched_training",
        )
        if not all(matched.get("gates", {}).values()):
            raise PermissionError("matched training gates did not all pass")
        validate_training_lineage(cache, matched)
        validate_matched_checkpoints(matched, matched_root)
        provenance = {
            "evaluator_source_acceptance": source_identity_record,
            "evaluator_source_identity": evaluator_source,
            "training_source_identity": training_source_identity(),
            "cache": cache_identity_record,
            "data": data_receipt,
            "matched_training": matched_identity,
            "mamba_overlay": verify_overlay(import_package=True),
        }
        device = _cuda0(args.device)
        result = run_dev_evaluation(
            data,
            matched_root,
            args.output_root,
            device=device,
            cache_result_sha256=cache_identity_record["sha256"],
            provenance=provenance,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
