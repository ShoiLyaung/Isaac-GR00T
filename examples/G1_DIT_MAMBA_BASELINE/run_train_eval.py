"""Matched head-only training/evaluation over the frozen native VLM cache."""

import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup
from transformers.feature_extraction_utils import BatchFeature

from G1_DIT_MAMBA_BASELINE.run_baseline import (  # noqa: E402
    GROUPS,
    REPO,
    VAL_IDS,
    copy_features,
    initialize_head,
    load_records,
    pack_records,
    parse_args,
    read,
    seed,
    sha,
    tensor_sha,
    write,
)


def save_head(head, directory, args, audit):
    directory.mkdir(parents=True, exist_ok=False)
    head.config.save_pretrained(directory)
    state = {key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()}
    save_file(state, str(directory / "head.safetensors"))
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        shutil.copy2(args.cache_root / "processor" / name, directory / name)
    write(
        directory / "HEAD_CHECKPOINT.json",
        {
            "format": "strict-full-action-head-plus-frozen-official-VLM-reference-v1",
            "base_checkpoint": str(args.base_checkpoint),
            "base_config_sha256": sha(args.base_checkpoint / "config.json"),
            "head_sha256": sha(directory / "head.safetensors"),
            "condition": args.condition,
            "audit": audit,
        },
    )


def load_compact_head(directory):
    metadata = read(directory / "HEAD_CHECKPOINT.json")
    if sha(directory / "head.safetensors") != metadata["head_sha256"]:
        raise ValueError("Compact head checksum changed")
    cfg = Gr00tN1d7Config.from_pretrained(directory, local_files_only=True)
    head = Gr00tN1d7ActionHead(cfg)
    head.load_state_dict(load_file(directory / "head.safetensors"), strict=True)
    return head.float(), metadata


def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records, manifest = load_records(args.cache_root, "train", args.base_checkpoint)
    schedule_file = args.cache_root / "train_schedule.npy"
    if sha(schedule_file) != manifest["schedule_sha256"]:
        raise ValueError("Training schedule changed")
    schedule = np.load(schedule_file)
    if args.steps != manifest["steps"] or schedule.shape != (args.steps, 8):
        raise ValueError("Optimizer steps and global batch must match frozen cache manifest")
    head, audit = initialize_head(args.base_checkpoint, args.condition, args.seed)
    head = head.cuda().train()
    if any(parameter.dtype != torch.float32 for parameter in head.parameters()):
        raise ValueError("Expected FP32 master parameters")
    # Match HF Trainer AdamW: biases and normalization weights have no decay.
    no_decay = set()
    for module_name, module in head.named_modules():
        if isinstance(module, nn.LayerNorm):
            no_decay.update(
                f"{module_name}.{name}" for name, _ in module.named_parameters(recurse=False)
            )
    groups = [
        {
            "params": [
                p
                for name, p in head.named_parameters()
                if p.requires_grad and not (name.endswith("bias") or name in no_decay)
            ],
            "weight_decay": 1e-5,
        },
        {
            "params": [
                p
                for name, p in head.named_parameters()
                if p.requires_grad and (name.endswith("bias") or name in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(args.steps * 0.05), num_training_steps=args.steps
    )
    write(
        args.output / "RESOLVED.json",
        {
            **audit,
            "cache_manifest_sha256": sha(args.cache_root / "CACHE_RESULT.json"),
            "schedule_sha256": manifest["schedule_sha256"],
            "steps": args.steps,
            "wall_seconds": args.wall_seconds,
            "global_batch_size": 8,
            "seed": args.seed,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "warmup_fraction": 0.05,
            "scheduler": "cosine",
            "clip_grad_norm": 1.0,
            "precision": "FP32 master with BF16 autocast",
            "state_dropout": 0.0,
            "raw_VLM_cached": True,
            "trainable_VL_projection_cached": False,
            "runtime_verified_fixed_state": False,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
        },
    )
    steps = []
    grad_seen = set()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (args.output / "train_steps.jsonl").open("x") as log:
        for index, positions in enumerate(schedule):
            if time.monotonic() - started > args.wall_seconds:
                raise TimeoutError("Training wall budget exceeded; no retry or continuation")
            raw, action = pack_records([records[int(i)] for i in positions])
            seed(args.seed + index)
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = head(copy_features(raw), action)
                loss = result["loss"]
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at step {index}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            if index == 0:
                by_module = {}
                for name, parameter in head.named_parameters():
                    if parameter.grad is not None:
                        group = name.split(".")[0]
                        value = parameter.grad.detach().square().sum()
                        by_module[group] = by_module.get(group, 0) + value
                grad_seen.update(name for name, value in by_module.items() if value.item() > 0)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            row = {
                "step": index + 1,
                "seed": args.seed + index,
                "loss": float(loss.detach()),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "service_ms": (time.perf_counter() - tick) * 1000,
                "frame_indices": positions.tolist(),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            steps.append(row)
            if (index + 1) % 25 == 0:
                print(json.dumps(row), flush=True)
    if time.monotonic() - started > args.wall_seconds:
        raise TimeoutError("Training wall budget exceeded before completion")
    required = {
        "model",
        "state_encoder",
        "action_encoder",
        "action_decoder",
        "vlln",
        "vl_self_attention",
    }
    if not required.issubset(grad_seen):
        raise ValueError(f"Missing finite nonzero gradients: {required - grad_seen}")
    head = head.eval()
    raw, action = pack_records(records[:1])
    observation = BatchFeature(
        data={key: value for key, value in action.items() if key not in ("action", "action_mask")}
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        seed(args.seed)
        before = head.get_action(copy_features(raw), observation)["action_pred"].cpu()
    save_head(head, args.output / "checkpoint", args, audit)
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated()
    del optimizer, scheduler, head, result, loss
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_compact_head(args.output / "checkpoint")
    reloaded = reloaded.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        seed(args.seed)
        after = reloaded.get_action(copy_features(raw), observation)["action_pred"].cpu()
    parity = float((before - after).abs().max())
    if parity != 0.0:
        raise ValueError(f"Strict reload action changed: {parity}")
    report = {
        "status": "completed",
        "technical_validity": True,
        "condition": args.condition,
        "steps": len(steps),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": peak,
        "first20_loss_median": float(np.median([row["loss"] for row in steps[:20]])),
        "last20_loss_median": float(np.median([row["loss"] for row in steps[-20:]])),
        "step_p50_ms": float(np.median([row["service_ms"] for row in steps])),
        "step_p95_ms": float(np.percentile([row["service_ms"] for row in steps], 95)),
        "gradient_modules": sorted(grad_seen),
        "strict_reload_action_max_abs": parity,
        "head_checkpoint": str(args.output / "checkpoint"),
    }
    write(args.output / "TRAIN_RESULT.json", report)
    print(json.dumps(report), flush=True)


def physical_metrics(prediction, record, processor):
    states = {key: value.numpy() for key, value in record["states"].items()}
    decoded = processor.decode_action(prediction, EmbodimentTag.NEW_EMBODIMENT, state=states)
    physical = np.concatenate([decoded[key] for key in GROUPS], axis=-1)
    if physical.shape != (40, 26) or not np.isfinite(physical).all():
        raise ValueError("Nonfinite or non-H40 active physical prediction")
    error = np.abs(physical - record["target"].numpy())
    return {
        "left_arm_mae_rad": float(error[:, :7].mean()),
        "right_arm_mae_rad": float(error[:, 7:14].mean()),
        "left_hand_mae_u": float(error[:, 14:20].mean()),
        "right_hand_mae_u": float(error[:, 20:26].mean()),
        "arm_mae_rad": float(error[:, :14].mean()),
        "hand_mae_u": float(error[:, 14:].mean()),
        "combined_mae_mixed_units": float(error.mean()),
        "hand_out_of_range_fraction": float(
            ((physical[:, 14:] < 0) | (physical[:, 14:] > 1)).mean()
        ),
    }


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records, manifest = load_records(args.cache_root, "validation", args.base_checkpoint)
    if args.head_checkpoint:
        head, metadata = load_compact_head(args.head_checkpoint)
        condition = metadata["condition"]
    else:
        head, _ = initialize_head(args.base_checkpoint, args.condition, args.seed)
        condition = args.condition + "_untrained"
    head = head.cuda().eval()
    processor = Gr00tN1d7Processor.from_pretrained(
        args.cache_root / "processor",
        local_files_only=True,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
    )
    processor.eval()
    started = time.monotonic()
    rows = []
    policy_times = []
    predictions = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for noise_seed in manifest["noise_seeds"]:
            for index, record in enumerate(records):
                if time.monotonic() - started > args.wall_seconds:
                    raise TimeoutError("Evaluation wall budget exceeded")
                raw, action = pack_records([record])
                paired_seed = int(noise_seed) + index * 10000
                seed(paired_seed)
                output = head(copy_features(raw), BatchFeature(data=dict(action)))
                loss = output["loss"]
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite evaluation flow loss")
                observation = BatchFeature(
                    data={
                        key: value
                        for key, value in action.items()
                        if key not in ("action", "action_mask")
                    }
                )
                seed(paired_seed)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                prediction = head.get_action(copy_features(raw), observation)["action_pred"]
                torch.cuda.synchronize()
                service_ms = (time.perf_counter() - tick) * 1000
                policy_times.append(service_ms)
                if prediction.shape != (1, 40, 132) or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid H40 evaluation prediction")
                row = {
                    "episode": record["episode"],
                    "anchor": record["anchor"],
                    "seed": noise_seed,
                    "paired_rng_seed": paired_seed,
                    "flow_loss": float(loss),
                    "action_sha256": tensor_sha(prediction),
                    "policy_service_ms": service_ms,
                    **physical_metrics(prediction[0].float().cpu().numpy(), record, processor),
                }
                rows.append(row)
                predictions.append(prediction[0].float().cpu().numpy())
                if (index + 1) % 32 == 0:
                    print(
                        json.dumps(
                            {"condition": condition, "seed": noise_seed, "anchors_done": index + 1}
                        ),
                        flush=True,
                    )
    metric_keys = [
        key
        for key in rows[0]
        if key
        in (
            "flow_loss",
            "combined_mae_mixed_units",
            "arm_mae_rad",
            "hand_mae_u",
            "hand_out_of_range_fraction",
        )
        or key.startswith(("left_", "right_"))
    ]
    per_episode = {}
    for episode in VAL_IDS:
        values = [row for row in rows if row["episode"] == episode]
        per_episode[str(episode)] = {
            key: float(np.mean([row[key] for row in values])) for key in metric_keys
        }
    aggregate = {
        key: float(np.mean([value[key] for value in per_episode.values()])) for key in metric_keys
    }
    report = {
        "status": "completed",
        "technical_validity": True,
        "condition": condition,
        "metrics": aggregate,
        "per_episode": per_episode,
        "anchors": len(records),
        "paired_outputs": len(rows),
        "head_policy_p50_ms": float(np.median(policy_times[5:])),
        "head_policy_p95_ms": float(np.percentile(policy_times[5:], 95)),
        "elapsed_seconds": time.monotonic() - started,
        "claim_scope": "small-cohort offline engineering baseline; combined MAE mixes radians and hand command u",
        "head_checkpoint": str(args.head_checkpoint),
        "cache_manifest_sha256": sha(args.cache_root / "CACHE_RESULT.json"),
    }
    write(args.output / "eval_details.json", rows)
    np.save(args.output / "normalized_predictions.npy", np.stack(predictions))
    write(args.output / "EVALUATION_RESULT.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(4)
    seed(args.seed)
    try:
        {"train": train, "evaluate": evaluate}[args.command](args)
    except Exception as exc:
        if (
            args.output is not None
            and args.output.exists()
            and not (args.output / "FAILURE.json").exists()
        ):
            write(
                args.output / "FAILURE.json",
                {"status": "failed", "exception": type(exc).__name__, "message": str(exc)},
            )
        raise
