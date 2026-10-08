"""Bounded mmap cache, batch-native KD and strict graph surfaces for matched training."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from budgeted_action_cudagraph import TENSOR_KEYS, CoreDispatcher
from budgeted_graph_distillation import CoreDelegate, DecoderDelegate, EncoderDelegate
from contract import sha
from memory_training_support import ACTION_KEYS, RAW_KEYS, frame_hash, fresh
import torch


class MMapCache:
    def __init__(self, root, plan, report_sha256, budget=lambda: None):
        import json

        self.root = Path(root)
        self.plan = plan
        self.loaded = {}
        self.verified_frames = set()
        report_path = self.root / "CACHE_RESULT.json"
        if not report_sha256 or sha(report_path) != report_sha256:
            raise ValueError("Explicit completed cache-report SHA required")
        self.report = json.loads(report_path.read_text())
        expected = plan["train"] + plan["dev"]
        if (
            self.report["status"] != "passed"
            or self.report["plan_sha256"] != plan["cache_plan_sha256"]
            or self.report["cohort"] != "train_dev"
            or self.report["episodes"] != expected
        ):
            raise ValueError("Full cache cohort/status mismatch")
        self.paths = {}
        for row in self.report["files"]:
            budget()
            path = self.root / row["file"]
            if (
                path.parent != self.root
                or path.stat().st_size != row["bytes"]
                or sha(path) != row["sha256"]
            ):
                raise ValueError(f"Frozen episode cache identity mismatch: {path}")
            self.paths[row["episode"]] = path
        if set(self.paths) != set(expected):
            raise ValueError("Cache episode inventory mismatch")
        self.report_sha256 = report_sha256

    def frame(self, episode, anchor):
        if episode not in self.paths or anchor not in self.plan["grids"][str(episode)]:
            raise ValueError("Sample is outside frozen train/dev cache")
        if episode not in self.loaded:
            frames = torch.load(
                self.paths[episode], map_location="cpu", weights_only=True, mmap=True
            )
            if list(frames) != self.plan["grids"][str(episode)]:
                raise ValueError("Mapped episode grid mismatch")
            self.loaded[episode] = frames
        value = self.loaded[episode][anchor]
        if (episode, anchor) not in self.verified_frames:
            if (
                value["episode"] != episode
                or value["anchor"] != anchor
                or frame_hash(value) != value["content_sha256"]
            ):
                raise ValueError(f"Mapped cache content mismatch: {episode}/{anchor}")
            self.verified_frames.add((episode, anchor))
        return value

    def assert_readonly(self):
        for episode, anchor in self.verified_frames:
            value = self.loaded[episode][anchor]
            if frame_hash(value) != value["content_sha256"]:
                raise ValueError("Read-only mapped frame was modified")
        if sha(self.root / "CACHE_RESULT.json") != self.report_sha256:
            raise ValueError("Cache report changed during run")


def assemble_batch(cache, pairs, batch_size=4):
    normalized = [(int(pair["episode"]), int(pair["anchor"])) for pair in pairs]
    if len(normalized) != batch_size or len(set(normalized)) != batch_size:
        raise ValueError("Expected fixed distinct B4 samples")
    frames = [cache.frame(ep, anchor) for ep, anchor in normalized]
    # Do not concatenate stored B1 post-Z: native head.forward computes actual
    # B4 postprocessing, which may round differently from four separate B1 calls.
    result = {
        "raw": {key: torch.cat([frame["raw"][key] for frame in frames], dim=0) for key in RAW_KEYS},
        "action_input": {
            key: torch.cat([frame["action_input"][key] for frame in frames], dim=0)
            for key in ACTION_KEYS
        },
        "samples": normalized,
        "frame_content_sha256": [frame["content_sha256"] for frame in frames],
    }
    if result["raw"]["backbone_features"].shape[0] != batch_size or result["action_input"][
        "action"
    ].shape != (batch_size, 40, 13):
        raise ValueError("Native batch tensor dimensions changed")
    return result


def masked_batch_kd(student, teacher, mask):
    if (
        student.ndim != 3
        or tuple(student.shape[1:]) != (40, 13)
        or student.shape != teacher.shape
        or mask.shape != student.shape
        or student.shape[0] < 1
    ):
        raise ValueError("Expected matching Bx40x13 velocity/mask tensors")
    active = mask > 0
    a = torch.where(active, student.float(), torch.zeros_like(student.float()))
    b = torch.where(active, teacher.detach().float(), torch.zeros_like(teacher.float()))
    denominator = mask.float().sum()
    if (
        not bool(denominator > 0)
        or not bool(torch.isfinite(a).all())
        or not bool(torch.isfinite(b).all())
    ):
        raise ValueError("Invalid active batched KD inputs")
    return (a - b).square().sum() / (denominator + 1e-6)


class StrictBatchDispatcher(CoreDispatcher):
    def __init__(self, surface, backend, sample_args):
        super().__init__(surface, backend)
        self.signature = tuple(
            (tuple(value.shape), value.dtype, value.device, value.requires_grad)
            for value in sample_args
        )
        if any(shape[0] != 4 for shape, _, _, _ in self.signature):
            raise ValueError("Training graph must capture exactlyB4")

    def forward(self, *args, **kwargs):
        values = tuple(kwargs[key] for key in TENSOR_KEYS)
        actual = tuple(
            (tuple(value.shape), value.dtype, value.device, value.requires_grad) for value in values
        )
        if actual != self.signature:
            raise ValueError("Graph input signature changed; broadcasting B1 into B4 is forbidden")
        return super().forward(*args, **kwargs)


def batch_forward(model, teacher, route, frame):
    """Only batch dimension generalizes; original head loss and true draws remain."""
    head = model.action_head
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Frozen eval teacher required")
    core = CoreDelegate(route)
    decoder = DecoderDelegate(head.action_decoder)
    encoder = EncoderDelegate(head.action_encoder)
    original_decoder = head.action_decoder
    original_time = head.sample_time
    times = []

    def time_delegate(*args, **kwargs):
        value = original_time(*args, **kwargs)
        times.append(value)
        return value

    with ExitStack() as scope:
        scope.enter_context(patch.object(head, "model", core))
        scope.enter_context(patch.object(head, "action_decoder", decoder))
        scope.enter_context(patch.object(head, "action_encoder", encoder))
        scope.enter_context(patch.object(head, "sample_time", time_delegate))
        result = head(fresh(frame["raw"], model.device), fresh(frame["action_input"], model.device))
        if (core.calls, decoder.calls, encoder.calls, len(times)) != (1, 1, 1, 1):
            raise ValueError("Batched native call cadence changed")
        prediction = decoder.output[:, -40:]
        kwargs = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in core.kwargs.items()
        }
        evidence = {
            "continuous_time": times[0].detach(),
            "discrete_time": encoder.inputs[1].detach(),
            "noisy_action": encoder.inputs[0].detach(),
            "post_condition": kwargs["encoder_hidden_states"],
            "state_features": result["state_features"].detach(),
            "target": frame["action_input"]["action"],
            "mask": result["action_mask"].detach(),
            "core_inputs": kwargs,
        }
    with torch.no_grad():
        target = teacher(**kwargs)
        if isinstance(target, tuple):
            target = target[0]
        target = original_decoder(target, frame["action_input"]["embodiment_id"].to(model.device))[
            :, -40:
        ].detach()
    kd = masked_batch_kd(prediction, target, result["action_mask"])
    return {
        "native": result,
        "kd": kd,
        "student_velocity": prediction,
        "teacher_velocity": target,
        "evidence": evidence,
    }


def phase_for_step(step):
    if not 1 <= step <= 20000:
        raise ValueError("Expected global step1..20000")
    return 1 if step <= 10000 else 2


def objective_for(condition, step):
    if condition not in ("task_only", "distill_then_task"):
        raise ValueError("Unknown matched condition")
    return "kd" if condition == "distill_then_task" and phase_for_step(step) == 1 else "task"


PLAN_SHA = "799213a729a3f69d69d66626f6312e1a592ffd4a8b21cce784623b047d503e49"


def save_checkpoint(context, optimizer, step, evaluations, seen):
    from action_transfer_program_support import (
        cpu_tree,
        optimizer_names,
        rng_state,
        save_tensor_new,
    )
    from contract import write_new
    import memory_prototype as proto

    root = context.args.output / "checkpoints"
    root.mkdir(exist_ok=True)
    weights = save_tensor_new(
        root / f"student-step{step}.pt",
        {k: v.detach().cpu() for k, v in context.student.state_dict().items()},
    )
    phase = phase_for_step(step)
    payload = {
        "optimizer": cpu_tree(optimizer.state_dict()),
        "optimizer_parameter_names": optimizer_names(context.student, optimizer),
        "rng": rng_state(),
        "completed_step": step,
        "phase": phase,
        "phase_optimizer_steps": step - (phase - 1) * 10000,
        "plan_sha256": PLAN_SHA,
        "source_commit": context.source_commit,
        "condition": context.args.condition,
    }
    optimizer_rng = save_tensor_new(root / f"optimizer-rng-step{step}.pt", payload)
    index = {
        "run_id": context.args.run_id,
        "condition": context.args.condition,
        "source_commit": context.source_commit,
        "plan_sha256": PLAN_SHA,
        "cache_report_sha256": context.args.cache_report_sha256,
        "schedule_sha256": context.plan["schedule_sha256"],
        "completed_step": step,
        "next_step": step + 1,
        "phase": phase,
        "next_phase": 2 if step >= 10000 else 1,
        "next_phase_reinitialize_optimizer": step == 10000,
        "weights": weights,
        "optimizer_rng": optimizer_rng,
        "student_content_sha256": proto.content_hash(context.student.state_dict()),
        "evaluations": evaluations,
        "gradient_groups_seen": seen,
        "parent_index": context.parent_index,
        "starting_step": context.starting_step,
        "training_log": log_prefix(context.args.output / "training_steps.jsonl"),
    }
    path = root / f"checkpoint-step{step}.json"
    write_new(path, index)
    return {"path": str(path.resolve()), "sha256": sha(path), "step": step}


def read_resume(args, plan, source_commit):
    from action_transfer_program_support import read

    if args.resume_index is None:
        return None, None
    if sha(args.resume_index) != args.resume_index_sha256:
        raise ValueError("Explicit parent resume SHA mismatch")
    index = read(args.resume_index)
    expected = {
        "condition": args.condition,
        "source_commit": source_commit,
        "plan_sha256": PLAN_SHA,
        "cache_report_sha256": args.cache_report_sha256,
        "schedule_sha256": plan["schedule_sha256"],
    }
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError("Resume scientific/source/cache identity changed")
    if (
        index["run_id"] == args.run_id
        or index["completed_step"] >= 20000
        or index["next_step"] != index["completed_step"] + 1
        or index["completed_step"] not in plan["weight_checkpoint_steps"]
    ):
        raise ValueError("Invalid matched resume boundary")
    if index["phase"] != phase_for_step(index["completed_step"]) or index[
        "next_phase_reinitialize_optimizer"
    ] != (index["completed_step"] == 10000):
        raise ValueError("Resume phase-reset semantics changed")
    for key in ("weights", "optimizer_rng"):
        value = index[key]
        if (
            Path(value["path"]).stat().st_size != value["bytes"]
            or sha(value["path"]) != value["sha256"]
        ):
            raise ValueError("Resume payload changed")
    payload = torch.load(index["optimizer_rng"]["path"], map_location="cpu", weights_only=True)
    if (
        payload["completed_step"] != index["completed_step"]
        or payload["phase"] != index["phase"]
        or payload["condition"] != args.condition
        or payload["plan_sha256"] != PLAN_SHA
        or payload["source_commit"] != source_commit
        or payload["phase_optimizer_steps"]
        != index["completed_step"] - (index["phase"] - 1) * 10000
    ):
        raise ValueError("Resume payload/header mismatch")
    return index, payload


def restore_checkpoint(core, optimizer, index, payload):
    from action_transfer_program_support import optimizer_names
    import memory_prototype as proto

    weights = torch.load(index["weights"]["path"], map_location="cpu", weights_only=True)
    if proto.content_hash(weights) != index["student_content_sha256"]:
        raise ValueError("Resume weight content mismatch")
    core.load_state_dict(weights, strict=True)
    if optimizer_names(core, optimizer) != payload["optimizer_parameter_names"]:
        raise ValueError("Resume parameter order differs")
    if not index["next_phase_reinitialize_optimizer"]:
        optimizer.load_state_dict(payload["optimizer"])
    elif optimizer.state:
        raise ValueError("Phase2 boundary requires fresh optimizer state")
    return payload["rng"]


def log_prefix(path):
    """Pin complete written rows at this checkpoint, even if the log later grows."""
    path = Path(path)
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("Checkpoint cannot bind a partial log row")
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(data.splitlines()),
    }


def checkpoint_log_rows(record, expected, seen=None):
    """Read explicit immutable parent indexes, never guess a latest attempt."""
    seen = set() if seen is None else seen
    path = Path(record["path"])
    if record["sha256"] in seen or sha(path) != record["sha256"]:
        raise ValueError("Checkpoint lineage cycle or SHA mismatch")
    seen.add(record["sha256"])
    index = json.loads(path.read_text())
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint lineage identity changed")
    parent = index["parent_index"]
    rows = [] if parent is None else checkpoint_log_rows(parent, expected, seen)
    if len(rows) != index["starting_step"]:
        raise ValueError("Resume lineage starting step mismatch")
    meta = index["training_log"]
    with Path(meta["path"]).open("rb") as stream:
        data = stream.read(meta["bytes"])
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise ValueError("Checkpoint log prefix changed")
    suffix = [json.loads(line) for line in data.splitlines()]
    if len(suffix) != meta["rows"]:
        raise ValueError("Checkpoint log prefix row count changed")
    rows.extend(suffix)
    if [row["step"] for row in rows] != list(range(1, index["completed_step"] + 1)):
        raise ValueError("Checkpoint lineage lost or repeated global optimizer steps")
    return rows
