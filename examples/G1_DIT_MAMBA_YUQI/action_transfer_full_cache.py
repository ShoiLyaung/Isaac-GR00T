"""Frozen native raw/post cache for the specified180train+8dev stride4 cohort."""

import argparse
import gc
import json
from pathlib import Path
import time

from budgeted_graph_distillation_runner import environment
from contract import sha, tensor_identity, write_new
from fp32_pilot_support import PilotCache
import memory_prototype as proto
from memory_training_support import ACTION_KEYS, RAW_KEYS, frame_hash, fresh
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
PLAN_SHA = "f8f022de2bde6ad52f80525dfcaeb72b4f22c188d19462742676e2a9b2b2478a"


def read(path):
    return json.loads(Path(path).read_text())


def verify_cohort(plan, cohort):
    if plan["train"] != cohort["train"] or plan["dev"] != cohort["dev"]:
        raise ValueError("Frozen train/dev membership or ordering changed")
    allowed = plan["train"] + plan["dev"]
    excluded = set(cohort["final"] + cohort["reserve"])
    if len(allowed) != 188 or len(set(allowed)) != 188 or set(allowed) & excluded:
        raise ValueError("Expected188 distinct train/dev episodes with no final/reserve")
    counts = {}
    for label in ("train", "dev"):
        total = 0
        for episode in plan[label]:
            expected = list(range(0, plan["lengths"][str(episode)] - 15, 4))
            if plan["grids"][str(episode)] != expected:
                raise ValueError("Stride4 grid changed or an unscheduled final anchor was added")
            total += len(expected)
        counts[label] = total
    if (
        counts != {"train": 21151, "dev": 837}
        or sum(counts.values()) != plan["expected_observations"]
    ):
        raise ValueError("Frozen21988-observation inventory changed")
    return allowed


def verify_plan(args):
    if sha(args.plan) != PLAN_SHA:
        raise ValueError("Frozen full-cache PLAN changed")
    plan = read(args.plan)
    if (
        REPO != Path(plan["source_worktree"])
        or args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve()
    ):
        raise ValueError("Cache source/artifact identity mismatch")
    if plan["training_authorized"] or plan["optimizer_steps"] != 0:
        raise ValueError("Cache stage forbids training")
    for row in plan["records"]:
        source = Path(row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError(f"Frozen cache input changed: {source}")
    for relative in plan["immutable_source_relative_paths"]:
        expected = {
            row["sha256"] for row in plan["records"] if row["path"].endswith("/" + relative)
        }
        if len(expected) != 1 or sha(REPO / relative) not in expected:
            raise ValueError(f"Native cache implementation changed: {relative}")
    if sha(plan["program_cohort_file"]) != plan["program_cohort_sha256"]:
        raise ValueError("Program cohort identity changed")
    verify_cohort(plan, read(plan["program_cohort_file"]))
    return plan


def encode_frame(model, source, episode, anchor):
    """Original cache math: preserve raw BEFORE native postprocess mutates a clone."""
    batch, raw = source.batch([(episode, anchor)])
    observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
    proto.validate_input_layout(observation, source.modality["video"].modality_keys)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_input, action_input = model.prepare_input(dict(batch))
        backbone = model.backbone(backbone_input)
        raw_cpu = {k: backbone[k].detach().cpu().clone() for k in RAW_KEYS}
        clean_raw = fresh({k: backbone[k] for k in RAW_KEYS}, model.device)
        clean_action = fresh({k: action_input[k] for k in ACTION_KEYS}, model.device)
        features = model.action_head._encode_features(clean_raw, clean_action)
        proto.validate_packet({"features": features, "backbone": clean_raw}, observation)
    frame = {
        "episode": episode,
        "anchor": anchor,
        "raw": raw_cpu,
        "post_vl_features": features.backbone_features.detach().cpu().clone(),
        "action_input": {k: action_input[k].detach().cpu().clone() for k in ACTION_KEYS},
        "physical_states": {k: torch.as_tensor(v).clone() for k, v in raw[0].states.items()},
        "physical_actions": {k: torch.as_tensor(v).clone() for k, v in raw[0].actions.items()},
        "input_identity": {k: tensor_identity(v) for k, v in batch.items()},
    }
    if (
        frame["raw"]["backbone_features"].dtype != torch.bfloat16
        or frame["post_vl_features"].dtype != torch.float32
    ):
        raise ValueError("Native raw BF16/post FP32 cache precision changed")
    for tensor in [
        *frame["raw"].values(),
        frame["post_vl_features"],
        *frame["action_input"].values(),
        *frame["physical_states"].values(),
        *frame["physical_actions"].values(),
    ]:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite native cache tensor")
    frame["content_sha256"] = frame_hash(frame)
    return frame


def compare_frame(actual, reference):
    location = f"episode={actual.get('episode')} anchor={actual.get('anchor')}"
    hashes = f"new_frame_sha256={actual.get('content_sha256')} old_frame_sha256={reference.get('content_sha256')}"

    def fail(key, new, old):
        raise ValueError(
            f"Old/new cache mismatch {location} section={key} new={new} old={old} {hashes}"
        )

    for key in ("episode", "anchor"):
        if actual[key] != reference[key]:
            fail("metadata." + key, actual[key], reference[key])
    if actual["input_identity"].keys() != reference["input_identity"].keys():
        fail(
            "input_identity.keys", list(actual["input_identity"]), list(reference["input_identity"])
        )
    for key, value in actual["input_identity"].items():
        if value != reference["input_identity"][key]:
            fail("input_identity." + key, value, reference["input_identity"][key])
    for group in ("raw", "action_input", "physical_states", "physical_actions"):
        if actual[group].keys() != reference[group].keys():
            fail(group + ".keys", list(actual[group]), list(reference[group]))
        for name, value in actual[group].items():
            other = reference[group][name]
            if value.dtype != other.dtype or not torch.equal(value, other):
                fail(group + "." + name, tensor_identity(value), tensor_identity(other))
    value, other = actual["post_vl_features"], reference["post_vl_features"]
    if value.dtype != other.dtype or not torch.equal(value, other):
        fail("post_vl_features", tensor_identity(value), tensor_identity(other))
    actual_hash, reference_hash = frame_hash(actual), frame_hash(reference)
    if actual_hash != actual["content_sha256"]:
        fail("content_sha256.actual", actual_hash, actual["content_sha256"])
    if reference_hash != reference["content_sha256"] or actual_hash != reference_hash:
        fail("content_sha256.reference", actual_hash, reference_hash)
    return {
        "episode": actual["episode"],
        "anchor": actual["anchor"],
        "content_sha256": actual_hash,
        "all_tensor_and_input_identities_exact": True,
    }


def save_episode(path, values):
    with path.open("xb") as stream:
        torch.save(values, stream)
    return {
        "file": path.name,
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "observations": len(values),
    }


def run(args, deadline):
    from run_train_eval import NativeAnchors, strict_reload

    args.current_episode = None
    args.current_anchor = None
    args.failure_stage = "initialization"
    plan = verify_plan(args)
    source_identity = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    set_seed(42, cuda=True)
    peak = 0
    committed_bytes = 0
    encoded = 0

    def budget():
        nonlocal peak
        peak = max(peak, torch.cuda.max_memory_allocated())
        if time.monotonic() > deadline:
            raise TimeoutError("Full train/dev cache wall budget exceeded")
        if committed_bytes > plan["max_artifact_bytes"] or peak > plan["max_allocated_bytes"]:
            raise RuntimeError("Cache artifact/GPU resource bound exceeded")

    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    model = model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    proto.assert_frozen(model)
    before = proto.content_hash(model.state_dict())
    if before != plan["base_content_sha256"]:
        raise ValueError("Frozen native B0 identity mismatch")
    source = NativeAnchors(Path(plan["prepared"]))
    source.processor.eval()
    old_plan = read(plan["old_cache_plan"])
    if sha(plan["old_cache_plan"]) != plan["old_cache_plan_sha256"]:
        raise ValueError("Old334 cache contract changed")
    old = PilotCache(plan["cache_root"], old_plan, expected_report_sha=plan["cache_report_sha256"])
    if old.report["integrity"]["base_before"] != before:
        raise ValueError("Old reference cache/base identity mismatch")
    allowed = plan["train"] + plan["dev"]
    files = []
    overlap = []
    write_new(
        args.output / "IDENTITY.json",
        {
            "plan_sha256": PLAN_SHA,
            "source": source_identity,
            "strict_loading": loading,
            "base_before": before,
            "cohort_sha256": plan["program_cohort_sha256"],
            "seed": 42,
            "mode": "native processor/model eval, no augmentation, no policy sampling",
            "optimizer_steps": 0,
        },
    )
    with (args.output / "episodes.jsonl").open("x") as log:
        for episode in allowed:
            args.current_episode = episode
            args.current_anchor = None
            args.failure_stage = "episode_reference"
            budget()
            tick = time.monotonic()
            values = {}
            if source.metadata[episode]["length"] != plan["lengths"][str(episode)]:
                raise ValueError("Original episode metadata length changed")
            reference = old.episode(episode) if episode in old.paths else {}
            for anchor in plan["grids"][str(episode)]:
                args.current_anchor = anchor
                args.failure_stage = "encode_frame"
                budget()
                frame = encode_frame(model, source, episode, anchor)
                if anchor in reference:
                    args.failure_stage = "old_cache_overlap"
                    overlap.append(compare_frame(frame, reference[anchor]))
                values[anchor] = frame
                encoded += 1
                if encoded % 200 == 0:
                    print(
                        json.dumps(
                            {
                                "observations_encoded": encoded,
                                "episodes_committed": len(files),
                                "current_episode": episode,
                                "current_episode_observations": len(values),
                                "committed_bytes": committed_bytes,
                            }
                        ),
                        flush=True,
                    )
            args.failure_stage = "save_episode"
            path = args.output / f"episode-{episode:06d}.pt"
            item = {
                "episode": episode,
                **save_episode(path, values),
                "elapsed_episode_seconds": time.monotonic() - tick,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            files.append(item)
            committed_bytes += item["bytes"]
            log.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            print(
                json.dumps(
                    {
                        "episode_committed": episode,
                        "observations": len(values),
                        "total_encoded": encoded,
                    }
                ),
                flush=True,
            )
            del values, reference, frame
            old.loaded.clear()
            gc.collect()
            budget()
    if (
        encoded != plan["expected_observations"]
        or len(files) != 188
        or len(overlap) != plan["expected_overlap_observations"]
    ):
        raise ValueError("Final cohort/observation/334-overlap inventory mismatch")
    args.failure_stage = "final_integrity"
    proto.assert_frozen(model)
    after = proto.content_hash(model.state_dict())
    if after != before:
        raise ValueError("Native base changed while building cache")
    old.assert_unchanged()
    budget()
    total = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
    if total > plan["max_artifact_bytes"]:
        raise RuntimeError("Total cache artifact bound exceeded")
    write_new(
        args.output / "OLD_CACHE_INTERSECTION.json",
        {"count": len(overlap), "exact": True, "frames": overlap},
    )
    return {
        "status": "passed",
        "plan_sha256": PLAN_SHA,
        "cohort": "train_dev",
        "episodes": allowed,
        "files": files,
        "observations": encoded,
        "grids": plan["grids"],
        "stride": 4,
        "raw_post_separate": True,
        "raw_dtype": "torch.bfloat16",
        "post_dtype": "torch.float32",
        "old_overlap_exact": len(overlap),
        "optimizer_steps": 0,
        "integrity": {
            "base_before": before,
            "base_after": after,
            "base_frozen_eval_no_grad": True,
            "peak_allocated_bytes": peak,
            "artifact_bytes": total,
        },
        "scope": "Complete specified180train+8dev cohort at stride4;21988 observations, not all220 episodes or all source frames; final/reserve not encoded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=7150)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 7150:
        parser.error("Frozen cache wall bound exceeded")
    if args.output.resolve() != (args.artifact_root / "cache-train-dev-r1/output").resolve():
        parser.error("One canonical cache output required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = run(args, started + args.wall_seconds)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        result.update(
            elapsed_seconds=time.monotonic() - started,
            gpu_after_release={
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            },
        )
        write_new(args.output / "CACHE_RESULT.json", result)
        write_new(args.output / "RESULT.json", result)
    except Exception as exc:
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "plan_sha256": PLAN_SHA,
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
                "partials_preserved": True,
                "current_episode": getattr(args, "current_episode", None),
                "current_anchor": getattr(args, "current_anchor", None),
                "failure_stage": getattr(args, "failure_stage", "setup"),
                "complete_episode_evidence": str(args.output / "episodes.jsonl"),
            },
        )
        raise


if __name__ == "__main__":
    main()
