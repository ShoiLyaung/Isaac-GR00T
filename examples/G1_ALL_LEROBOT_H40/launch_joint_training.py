#!/usr/bin/env python3
"""Bounded offline supervisor for the joint five-source GR00T training run.

This file is deliberately outside the GR00T training core.  It owns one
``bwrap``/``torchrun`` process, two cooperative local GPU leases, and bounded
resource evidence for either a two-step engineering profile or the frozen
20,000-step training attempt.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from typing import Any, TextIO


SOURCE_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = Path("/home/sl/checkpoints/GR00T-N1.7-3B")
MODALITY_CONFIG = SOURCE_ROOT / "examples/G1_ID10_H40_NATIVE30/g1_id10_h40_native30_config.py"
PYTHON_ENV = Path("/home/sl/miniconda3/envs/gr00t_n1d7")
HF_HOME = Path("/home/sl/.cache/huggingface")
COSMOS_SNAPSHOT = (
    HF_HOME
    / "hub/models--nvidia--Cosmos-Reason2-2B/snapshots"
    / "9ce19a195e423419c349abfc86fd07178b230561/model.safetensors"
)
BASE_HASHES = {
    "model-00001-of-00002.safetensors": (
        "8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d"
    ),
    "model-00002-of-00002.safetensors": (
        "c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661"
    ),
    "model.safetensors.index.json": (
        "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
    ),
}
GPU_QUERY = (
    "index,uuid,memory.total,memory.used,memory.free,utilization.gpu,"
    "ecc.errors.uncorrected.volatile.total"
)


@dataclass(frozen=True)
class Mode:
    max_steps: int
    save_steps: int
    save_total_limit: int
    save_only_model: bool
    timeout_seconds: int
    storage_limit_bytes: int


MODES = {
    "profile": Mode(
        max_steps=2,
        save_steps=2,
        save_total_limit=1,
        save_only_model=True,
        timeout_seconds=3300,
        storage_limit_bytes=64 * 1024**3,
    ),
    "formal": Mode(
        max_steps=20000,
        save_steps=1000,
        save_total_limit=2,
        save_only_model=False,
        timeout_seconds=28200,
        storage_limit_bytes=128 * 1024**3,
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=SOURCE_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def descendant_pids(root_pid: int) -> set[int]:
    """Return the current procfs descendants of one owned process."""
    owned = {root_pid}
    queue = [root_pid]
    while queue:
        parent = queue.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for raw in children:
            child = int(raw)
            if child not in owned:
                owned.add(child)
                queue.append(child)
    return owned


def process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def choose_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def gpu_snapshot() -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 7:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        gpus.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "total_mib": int(fields[2]),
                "used_mib": int(fields[3]),
                "free_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
                "uncorrected_ecc": fields[6],
            }
        )
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    compute_apps = []
    for line in apps.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi compute row: {line}")
        compute_apps.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "used_mib": int(fields[2]),
                "process_name": fields[3],
            }
        )
    return {
        "observed_at": now(),
        "gpus": gpus,
        "compute_apps": compute_apps,
    }


def gate_gpus(snapshot: dict[str, Any], minimum_free_mib: int) -> None:
    by_index = {gpu["index"]: gpu for gpu in snapshot["gpus"]}
    if any(index not in by_index for index in (0, 1)):
        raise RuntimeError("physical GPU0 and GPU1 are both required")
    if snapshot["compute_apps"]:
        raise RuntimeError(
            "training requires conflict-free GPUs; active compute processes: "
            + json.dumps(snapshot["compute_apps"], sort_keys=True)
        )
    for index in (0, 1):
        gpu = by_index[index]
        if gpu["free_mib"] < minimum_free_mib:
            raise RuntimeError(
                f"GPU{index} free={gpu['free_mib']} MiB is below {minimum_free_mib} MiB"
            )
        ecc = gpu["uncorrected_ecc"]
        if ecc not in {"0", "[N/A]", "N/A"}:
            raise RuntimeError(f"GPU{index} has uncorrected ECC={ecc}")


def _lease_filename(gpu: dict[str, Any]) -> str:
    safe_uuid = re.sub(r"[^A-Za-z0-9_.-]+", "_", gpu["uuid"]).strip("._")
    return f"gpu-{gpu['index']}-{safe_uuid or 'unknown'}.lease"


class GpuLease:
    """A lock compatible with research-control's cooperative lease files."""

    def __init__(self, gpu: dict[str, Any], root: Path, run_id: str) -> None:
        self.gpu = gpu
        self.path = root / _lease_filename(gpu)
        self.run_id = run_id
        self.handle: TextIO | None = None

    def __enter__(self) -> "GpuLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(f"GPU lease is active: {self.path}") from exc
            raise
        metadata = {
            "owner": os.environ.get("USER", str(os.geteuid())),
            "run_id": self.run_id,
            "pid": os.getpid(),
            "time": now(),
            "gpu": self.gpu["uuid"],
        }
        handle.seek(0)
        handle.truncate()
        json.dump(metadata, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self.handle = handle
        return self

    def __exit__(self, *_args: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def validate_source(expected_commit: str) -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(f"source commit drift: expected={expected_commit}, actual={head}")
    dirty = git_output("status", "--porcelain")
    if dirty:
        raise RuntimeError(f"source worktree must be clean: {dirty.splitlines()}")
    if not MODALITY_CONFIG.is_file():
        raise RuntimeError(f"modality config missing: {MODALITY_CONFIG}")
    return {"source_root": str(SOURCE_ROOT), "source_commit": head}


def validate_materials(preparation_root: Path) -> dict[str, Any]:
    manifest_path = preparation_root / "DATASET_MANIFEST.json"
    summary_path = preparation_root / "PREPARATION_SUMMARY.json"
    closure_path = preparation_root / "CPU_MIXTURE_CLOSURE.json"
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    closure = load_json(closure_path)
    if manifest.get("schema_version") != "gr00t-finetune-dataset-manifest-v1":
        raise RuntimeError("unsupported dataset manifest schema")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 5:
        raise RuntimeError("joint training requires exactly five manifest datasets")
    if summary.get("status") != "PASS":
        raise RuntimeError("preparation summary is not PASS")
    if closure.get("status") != "PASS":
        raise RuntimeError("CPU mixture closure is not PASS")
    weights: list[float] = []
    source_ids: list[str] = []
    for index, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            raise RuntimeError(f"dataset manifest entry {index} is not an object")
        dataset_path = Path(entry["dataset_path"]).resolve(strict=True)
        adapter_path = Path(entry["adapter_config_path"]).resolve(strict=True)
        adapter = load_json(adapter_path)
        try:
            dataset_path.relative_to(preparation_root)
            adapter_path.relative_to(preparation_root)
        except ValueError as exc:
            raise RuntimeError("prepared dataset references must stay in preparation root") from exc
        for name in ("data", "videos"):
            if not (dataset_path / name).is_symlink():
                raise RuntimeError(f"loader-view {name} must be a read-only payload symlink")
        weight = float(entry["mix_ratio"])
        if not weight > 0:
            raise RuntimeError("every dataset mix_ratio must be positive")
        weights.append(weight)
        source_id = adapter.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError(f"adapter source_id is missing: {adapter_path}")
        source_ids.append(source_id)
    if abs(sum(weights) - 1.0) > 1e-9:
        raise RuntimeError(f"dataset weights must sum to one: {sum(weights)}")
    for name, expected in BASE_HASHES.items():
        path = BASE_MODEL / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"base checkpoint hash mismatch: {path}")
    if not COSMOS_SNAPSHOT.is_file():
        raise RuntimeError(f"offline Cosmos snapshot missing: {COSMOS_SNAPSHOT}")
    return {
        "preparation_root": str(preparation_root),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "preparation_summary_sha256": sha256_file(summary_path),
        "cpu_mixture_closure_sha256": sha256_file(closure_path),
        "base_model": str(BASE_MODEL),
        "base_hashes": BASE_HASHES,
        "mix_ratios": weights,
        "source_ids": source_ids,
    }


def build_train_command(
    *, mode: Mode, dataset_manifest: Path, train_output: Path, master_port: int
) -> list[str]:
    command = [
        "uv",
        "run",
        "torchrun",
        "--nproc_per_node=2",
        f"--master_port={master_port}",
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        str(BASE_MODEL),
        "--dataset-manifest-path",
        str(dataset_manifest),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(MODALITY_CONFIG),
        "--num-gpus",
        "2",
        "--output-dir",
        str(train_output),
        "--max-steps",
        str(mode.max_steps),
        "--global-batch-size",
        "32",
        "--gradient-accumulation-steps",
        "1",
        "--dataloader-num-workers",
        "0",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "1e-05",
        "--warmup-ratio",
        "0.05",
        "--save-steps",
        str(mode.save_steps),
        "--save-total-limit",
        str(mode.save_total_limit),
        "--action-horizon",
        "40",
        "--no-tune-llm",
        "--tune-projector",
        "--tune-diffusion-model",
        "--no-use-wandb",
        "--dataset-adapter",
        "g1_all_lerobot_h40_v1",
        "--tune-visual",
        "--save-only-model" if mode.save_only_model else "--no-save-only-model",
    ]
    return command


def isolation_command(command: list[str], run_dir: Path) -> list[str]:
    return [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(run_dir),
        str(run_dir),
        "--unshare-net",
        "--chdir",
        str(SOURCE_ROOT),
        "--",
        *command,
    ]


def child_environment(run_dir: Path, source_commit: str) -> dict[str, str]:
    inherited = {
        name: os.environ[name]
        for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "LANG", "LC_ALL", "TZ")
        if name in os.environ
    }
    inherited.update(
        {
            "HOME": str(run_dir / "home"),
            "XDG_CACHE_HOME": str(run_dir / "cache"),
            "MPLCONFIGDIR": str(run_dir / "matplotlib"),
            "UV_CACHE_DIR": str(run_dir / "uv-cache"),
            "UV_PROJECT_ENVIRONMENT": str(PYTHON_ENV),
            "UV_OFFLINE": "1",
            "UV_NO_SYNC": "1",
            "HF_HOME": str(HF_HOME),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PYTHONPATH": str(SOURCE_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "GROOT_COMMIT_HASH": source_commit,
            "GROOT_CLEANROOM_EVIDENCE": "1",
            "GROOT_CLEANROOM_REQUIRE_NCCL_TRACE": "1",
            # Historical 10k tune-visual used 263,099 trace entries. 20k can
            # exceed 524,288, so retain a full non-wrapping sequence at 1 Mi.
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
            "TORCH_NCCL_TRACE_CPP_STACK": "0",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_DEBUG": "INFO",
            "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
        }
    )
    return inherited


def _terminate_group(process: subprocess.Popen[bytes]) -> str:
    process.poll()
    if not process_group_exists(process.pid):
        return "already_exited"
    for sig, label, grace in (
        (signal.SIGTERM, "term", 15.0),
        (signal.SIGKILL, "kill", 5.0),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return label
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()
            if not process_group_exists(process.pid):
                return label
            time.sleep(0.2)
    return "kill_wait_expired"


def _posttrain_evidence(
    train_output: Path, expected_steps: int, source_ids: list[str]
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "rank_files": [],
        "global_steps": [],
        "training_monitors_complete": [],
        "nccl_complete": [],
        "nccl_no_wrap": [],
        "nccl_sequence_sha256": [],
        "rank_sample_totals": [],
        "source_sample_totals": {source_id: 0 for source_id in source_ids},
        "complete": False,
    }
    for path in sorted((train_output / "distributed_evidence").glob("rank_*_posttrain.json")):
        value = load_json(path)
        evidence["rank_files"].append(str(path))
        evidence["global_steps"].append(value.get("trainer_global_step"))
        monitor = value.get("training_monitor", {})
        evidence["training_monitors_complete"].append(
            monitor.get("complete") is True
            and monitor.get("raw_loss_count") == expected_steps
            and monitor.get("finite_loss_count") == expected_steps
            and monitor.get("gradient_norm_count") == expected_steps
            and monitor.get("finite_gradient_norm_count") == expected_steps
            and monitor.get("skipped_optimizer_steps") == 0
            and monitor.get("overflow_events") == 0
        )
        nccl = value.get("nccl_flight_recorder", {})
        evidence["nccl_complete"].append(
            nccl.get("required") is True and nccl.get("complete") is True
        )
        evidence["nccl_no_wrap"].append(nccl.get("no_circular_buffer_wrap") is True)
        evidence["nccl_sequence_sha256"].append(nccl.get("operation_sequence_sha256"))
        sampling = value.get("dataset_sampling", {})
        sampling_evidence = sampling.get("evidence", {})
        evidence["rank_sample_totals"].append(sampling_evidence.get("total_samples_yielded"))
        datasets = sampling_evidence.get("datasets", [])
        if sampling.get("rank_local_counts_exact") is True and len(datasets) == len(source_ids):
            for source_id, dataset in zip(source_ids, datasets):
                evidence["source_sample_totals"][source_id] += int(dataset["samples_yielded"])
    expected_samples = expected_steps * 32
    evidence["expected_total_samples"] = expected_samples
    evidence["actual_total_samples"] = sum(
        value for value in evidence["rank_sample_totals"] if isinstance(value, int)
    )
    evidence["complete"] = (
        len(evidence["rank_files"]) == 2
        and evidence["global_steps"] == [expected_steps, expected_steps]
        and evidence["training_monitors_complete"] == [True, True]
        and evidence["nccl_complete"] == [True, True]
        and evidence["nccl_no_wrap"] == [True, True]
        and len(set(evidence["nccl_sequence_sha256"])) == 1
        and None not in evidence["nccl_sequence_sha256"]
        and evidence["actual_total_samples"] == expected_samples
    )
    return evidence


def run(args: argparse.Namespace) -> int:
    mode = MODES[args.mode]
    preparation_root = args.preparation_root.resolve(strict=True)
    source = validate_source(args.expected_source_commit)
    materials = validate_materials(preparation_root)
    train_output = args.run_dir / "train"
    port = choose_port()
    train_command = build_train_command(
        mode=mode,
        dataset_manifest=Path(materials["dataset_manifest"]),
        train_output=train_output,
        master_port=port,
    )
    executed_command = isolation_command(train_command, args.run_dir)
    plan = {
        "schema_version": "g1-all-lerobot-h40-training-plan-v1",
        "mode": args.mode,
        "run_dir": str(args.run_dir),
        "source": source,
        "materials": materials,
        "training": {
            "max_steps": mode.max_steps,
            "global_batch_size": 32,
            "seed": 42,
            "action_horizon": 40,
            "tune_visual": True,
            "timeout_seconds": mode.timeout_seconds,
            "storage_limit_bytes": mode.storage_limit_bytes,
            "automatic_retries": 0,
        },
        "official_command": train_command,
        "executed_command": executed_command,
        "environment": child_environment(args.run_dir, source["source_commit"]),
        "network": "bwrap network namespace unshared; all model/data inputs local",
        "gpu_gate": {"physical_indices": [0, 1], "minimum_free_mib": args.minimum_free_mib},
    }
    if args.action == "plan":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.run_dir.exists():
        raise FileExistsError(f"attempt directory already exists: {args.run_dir}")
    if not args.run_dir.parent.is_dir():
        raise FileNotFoundError(f"attempt parent does not exist: {args.run_dir.parent}")
    if (
        os.statvfs(args.run_dir.parent).f_bavail * os.statvfs(args.run_dir.parent).f_frsize
        < mode.storage_limit_bytes
    ):
        raise RuntimeError("insufficient free disk for bounded attempt")

    gate_snapshot = gpu_snapshot()
    gate_gpus(gate_snapshot, args.minimum_free_mib)
    by_index = {gpu["index"]: gpu for gpu in gate_snapshot["gpus"]}
    run_id = args.run_dir.name
    with ExitStack() as leases:
        for index in (0, 1):
            leases.enter_context(GpuLease(by_index[index], args.lease_root, run_id))
        # Close the probe/lease race against other cooperating callers.
        gate_snapshot = gpu_snapshot()
        gate_gpus(gate_snapshot, args.minimum_free_mib)

        args.run_dir.mkdir()
        for name in ("home", "cache", "matplotlib", "uv-cache"):
            (args.run_dir / name).mkdir()
        plan["gpu_at_gate"] = gate_snapshot
        plan["created_at"] = now()
        atomic_json(args.run_dir / "ATTEMPT.json", plan)

        start = time.monotonic()
        stop_reason: str | None = None
        cleanup = "not_needed"
        exit_code: int | None = None
        peak_used = {"0": 0, "1": 0}
        sample_count = 0
        owned_pids: set[int] = set()
        last_storage_check = 0.0
        process: subprocess.Popen[bytes] | None = None
        environment = plan["environment"]
        with (
            (args.run_dir / "training.stdout.log").open("xb") as stdout,
            (args.run_dir / "training.stderr.log").open("xb") as stderr,
            (args.run_dir / "RESOURCE_SAMPLES.jsonl").open("x", encoding="utf-8") as samples,
        ):
            try:
                process = subprocess.Popen(
                    executed_command,
                    cwd=SOURCE_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                while process.poll() is None:
                    owned_pids.update(descendant_pids(process.pid))
                    snapshot = gpu_snapshot()
                    unknown_apps = [
                        app for app in snapshot["compute_apps"] if app["pid"] not in owned_pids
                    ]
                    if unknown_apps:
                        # Close a short fork/procfs observation race before failing closed.
                        owned_pids.update(descendant_pids(process.pid))
                        unknown_apps = [
                            app for app in snapshot["compute_apps"] if app["pid"] not in owned_pids
                        ]
                    if unknown_apps:
                        stop_reason = "unrelated GPU compute process appeared: " + json.dumps(
                            unknown_apps, sort_keys=True
                        )
                    for gpu in snapshot["gpus"]:
                        key = str(gpu["index"])
                        if key in peak_used:
                            peak_used[key] = max(peak_used[key], gpu["used_mib"])
                            if gpu["used_mib"] >= 47000:
                                stop_reason = f"GPU{key} used-memory stop line reached"
                    samples.write(json.dumps(snapshot, sort_keys=True) + "\n")
                    samples.flush()
                    sample_count += 1
                    elapsed = time.monotonic() - start
                    if elapsed >= mode.timeout_seconds:
                        stop_reason = "hard wall exceeded"
                    if elapsed - last_storage_check >= 30:
                        last_storage_check = elapsed
                        if directory_size(args.run_dir) > mode.storage_limit_bytes:
                            stop_reason = "attempt storage limit exceeded"
                    if stop_reason:
                        cleanup = _terminate_group(process)
                        break
                    time.sleep(10)
                exit_code = process.wait()
                if process_group_exists(process.pid):
                    cleanup = _terminate_group(process)
            except BaseException as exc:
                stop_reason = stop_reason or f"{type(exc).__name__}: {exc}"
                if process is not None:
                    cleanup = _terminate_group(process)
                    exit_code = process.poll()

        wall_seconds = time.monotonic() - start
        posttrain = _posttrain_evidence(train_output, mode.max_steps, materials["source_ids"])
        optimizer_state_complete = True
        if args.mode == "formal":
            optimizer_root = (
                train_output / f"checkpoint-{mode.max_steps}/global_step{mode.max_steps}"
            )
            optimizer_state_complete = (
                len(list(optimizer_root.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt")))
                == 2
            )
        technical_valid = (
            exit_code == 0
            and stop_reason is None
            and posttrain["complete"]
            and (train_output / "model.safetensors.index.json").is_file()
            and optimizer_state_complete
            and (
                args.mode != "formal"
                or all(value > 0 for value in posttrain["source_sample_totals"].values())
            )
        )
        summary = {
            "schema_version": "g1-all-lerobot-h40-training-summary-v1",
            "mode": args.mode,
            "run_id": run_id,
            "finished_at": now(),
            "exit_code": exit_code,
            "stop_reason": stop_reason,
            "cleanup": cleanup,
            "wall_seconds": wall_seconds,
            "peak_total_used_mib": peak_used,
            "resource_sample_count": sample_count,
            "attempt_bytes": directory_size(args.run_dir),
            "storage_limit_bytes": mode.storage_limit_bytes,
            "posttrain_evidence": posttrain,
            "optimizer_state_complete": optimizer_state_complete,
            "technical_valid": technical_valid,
            "scientific_interpretation": (
                "training execution evidence only; no evaluator/native success is inferred"
            ),
            "source_commit": source["source_commit"],
            "dataset_manifest_sha256": materials["dataset_manifest_sha256"],
            "model_output": str(train_output),
        }
        atomic_json(args.run_dir / "SUPERVISOR_SUMMARY.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if technical_valid else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--mode", required=True, choices=tuple(MODES))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--lease-root", type=Path, default=Path("/tmp/research-control-gpu-leases"))
    parser.add_argument("--minimum-free-mib", type=int, default=40000)
    args = parser.parse_args(argv)
    if not args.run_dir.is_absolute() or not args.preparation_root.is_absolute():
        parser.error("run-dir and preparation-root must be absolute")
    if args.minimum_free_mib < 0:
        parser.error("minimum-free-mib must be non-negative")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
