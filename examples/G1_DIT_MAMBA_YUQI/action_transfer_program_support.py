"""Immutable checkpoints and explicit resume state for bounded action-transfer attempts."""

import json
from pathlib import Path
import random

from contract import sha, write_new
import memory_prototype as proto
import numpy as np
import torch


PLAN_SHA = "6037b109e2e65c914f3c41771947bbce18376e1a156f17174dbbdd584cd73f32"


def read(path):
    return json.loads(Path(path).read_text())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def rng_state():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy[0],
            "keys": numpy[1].tolist(),
            "position": int(numpy[2]),
            "has_gauss": int(numpy[3]),
            "cached_gaussian": float(numpy[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
    }


def restore_rng(values):
    random.setstate(values["python"])
    state = values["numpy"]
    np.random.set_state(
        (
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )
    torch.set_rng_state(values["torch_cpu"])
    if len(values["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume visible GPU RNG inventory changed")
    torch.cuda.set_rng_state_all(values["torch_cuda"])


def optimizer_names(core, optimizer):
    names = {id(parameter): name for name, parameter in core.named_parameters()}
    return [
        [names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups
    ]


def save_tensor_new(path, values):
    with Path(path).open("xb") as stream:
        torch.save(values, stream)
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha(path),
        "bytes": Path(path).stat().st_size,
    }


def validate_run_id(args, plan):
    if not args.run_id or any(
        ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in args.run_id
    ):
        raise ValueError("Expected a simple immutable run_id")
    if args.output.resolve() != (args.artifact_root / args.run_id / "output").resolve():
        raise ValueError("Output/run identity mismatch")
    if (args.resume_index is None) != (args.resume_index_sha256 is None):
        raise ValueError("Resume requires explicit parent index path and SHA together")
    if args.resume_index is None and args.run_id != plan["initial_run_id"]:
        raise ValueError("Fresh attempt requires the frozen initial run_id")
    if args.resume_index is not None and args.run_id == plan["initial_run_id"]:
        raise ValueError("Resume must use a new immutable attempt identity")


def load_resume(args, plan, source_commit):
    if args.resume_index is None:
        return None, None
    if sha(args.resume_index) != args.resume_index_sha256:
        raise ValueError("Parent checkpoint-index SHA mismatch")
    index = read(args.resume_index)
    if (
        index["plan_sha256"] != PLAN_SHA
        or index["source_commit"] != source_commit
        or index["schedule_sha256"] != plan["schedule_sha256"]["overfit"]
    ):
        raise ValueError(
            "Resume source/PLAN/schedule changed; requires a separately explicit new experiment"
        )
    if (
        index["run_id"] == args.run_id
        or index["completed_step"] >= plan["training_steps"]
        or index["next_step"] != index["completed_step"] + 1
    ):
        raise ValueError("Invalid or already-completed resume boundary")
    if "optimizer_rng" not in index:
        raise ValueError("Weight-only checkpoint is not a resumable optimizer boundary")
    for name in ("weights", "optimizer_rng"):
        value = index[name]
        if (
            Path(value["path"]).stat().st_size != value["bytes"]
            or sha(value["path"]) != value["sha256"]
        ):
            raise ValueError("Parent resume payload identity changed")
    state = torch.load(index["optimizer_rng"]["path"], map_location="cpu", weights_only=True)
    if (
        state["completed_step"] != index["completed_step"]
        or state["plan_sha256"] != PLAN_SHA
        or state["source_commit"] != source_commit
    ):
        raise ValueError("Resume payload metadata mismatch")
    return index, state


def restore_training(core, optimizer, index, state):
    weights = torch.load(index["weights"]["path"], map_location="cpu", weights_only=True)
    if proto.content_hash(weights) != index["student_content_sha256"]:
        raise ValueError("Resume student semantic hash mismatch")
    core.load_state_dict(weights, strict=True)
    if optimizer_names(core, optimizer) != state["optimizer_parameter_names"]:
        raise ValueError("Resume optimizer parameter ordering changed")
    optimizer.load_state_dict(state["optimizer"])
    return state["rng"]


def checkpoint(context, optimizer, step, probe_records, seen_gradients):
    root = context.args.output / "checkpoints"
    root.mkdir(exist_ok=True)
    weights = save_tensor_new(
        root / f"student-step{step}.pt",
        {name: value.detach().cpu() for name, value in context.student.state_dict().items()},
    )
    index = {
        "run_id": context.args.run_id,
        "plan_sha256": PLAN_SHA,
        "source_commit": context.source_commit,
        "schedule_sha256": context.plan["schedule_sha256"]["overfit"],
        "completed_step": step,
        "next_step": step + 1,
        "weights": weights,
        "student_content_sha256": proto.content_hash(context.student.state_dict()),
        "base_content_sha256": context.base_before,
        "cache_report_sha256": context.plan["cache_report_sha256"],
        "probe_records": probe_records,
        "gradient_groups_seen": seen_gradients,
    }
    if step in context.plan["resume_checkpoint_steps"]:
        payload = {
            "completed_step": step,
            "plan_sha256": PLAN_SHA,
            "source_commit": context.source_commit,
            "optimizer": cpu_tree(optimizer.state_dict()),
            "optimizer_parameter_names": optimizer_names(context.student, optimizer),
            "rng": rng_state(),
        }
        index["optimizer_rng"] = save_tensor_new(root / f"optimizer-rng-step{step}.pt", payload)
        del payload
    path = root / f"checkpoint-step{step}.json"
    write_new(path, index)
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "completed_step": step,
        "resumable": "optimizer_rng" in index,
    }
