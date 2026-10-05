"""Bounded matched G1 native-DiT / attention-free conditional-Mamba experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.g1_all_lerobot_h40_adapter import G1AllLerobotH40EpisodeLoader
from gr00t.data.dataset.g1_id10_h40_native30_adapter import apply_active_action_mask
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.experiment.launch_finetune import load_modality_config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.modules.mamba_action_mixer import convert_native_model
import numpy as np
from safetensors import safe_open
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


REPO = Path(__file__).resolve().parents[2]
GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")
CONDITIONS = ("dit_pretrained", "dit_reinit_backbone", "conditional_mamba2")
TRAIN_IDS = [158, 203, 71, 104, 42, 128, 192, 173]
VAL_IDS = [150, 147, 152, 112]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            result.update(part)
    return result.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def seed(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    return value


def copy_features(raw):
    # Native process_backbone_output mutates its mapping, never the source tensor.
    return BatchFeature(data=dict(raw))


def strict_info(info):
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Non-strict native checkpoint load: {info}")
    return info


def load_full_native(checkpoint):
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise RuntimeError("Refusing skipped model weights")
    model, info = Gr00tN1d7.from_pretrained(
        checkpoint,
        output_loading_info=True,
        local_files_only=True,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
        torch_dtype=torch.float32,
        load_bf16=False,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
    )
    strict_info(info)
    if model.config.action_backbone != "dit" or model.config.action_horizon != 40:
        raise ValueError("Expected native H40 checkpoint")
    return model.eval().cuda(), info


def make_processor(stage, checkpoint):
    load_modality_config(str(REPO / "examples/G1_ID10_H40_NATIVE30/g1_id10_h40_native30_config.py"))
    modality = MODALITY_CONFIGS["new_embodiment"]
    loader = G1AllLerobotH40EpisodeLoader(
        stage / "train",
        modality,
        stage / "ADAPTER_CONFIG.json",
        require_runtime_verified_fixed_state=False,
    )
    if loader.adapter_config["runtime_verified_fixed_state"]:
        raise ValueError(
            "Expected disclosed missing-leg/waist zero placeholders, not a runtime claim"
        )
    processor = Gr00tN1d7Processor.from_pretrained(
        checkpoint,
        modality_configs={"new_embodiment": modality},
        use_relative_action=True,
        use_percentiles=True,
        max_action_horizon=40,
        image_crop_size=None,
        image_target_size=None,
        shortest_image_edge=256,
        crop_fraction=1.0,
        color_jitter_params={"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
        random_rotation_angle=0,
        state_dropout_prob=0.0,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
        local_files_only=True,
    )
    processor.set_statistics({"new_embodiment": loader.get_dataset_statistics()}, override=True)
    processor.eval()
    return processor, modality


def pack_records(records, device="cuda"):
    lengths = {record["raw"]["backbone_features"].shape[1] for record in records}
    if len(lengths) != 1:
        raise ValueError(
            "Frozen cache has varying token lengths; refusing unregistered padding change"
        )
    raw = {
        key: torch.cat([record["raw"][key] for record in records]).to(device)
        for key in records[0]["raw"]
    }
    action = {
        key: torch.cat([record["action_input"][key] for record in records]).to(device)
        for key in records[0]["action_input"]
    }
    return BatchFeature(data=raw), BatchFeature(data=action)


def cache(args):
    started = time.monotonic()
    root = args.cache_root
    root.mkdir(parents=True, exist_ok=False)
    plan = read(args.stage / "OFFLINE_EVAL_PLAN.json")
    if [row["episode_index"] for row in plan["train"]] != TRAIN_IDS:
        raise ValueError("Frozen train cohort changed")
    if [row["episode_index"] for row in plan["validation"]] != VAL_IDS:
        raise ValueError("Frozen validation cohort changed")
    processor, modality = make_processor(args.stage, args.base_checkpoint)
    processor.save_pretrained(root / "processor")
    model, loading = load_full_native(args.base_checkpoint)
    manifest = {
        "base_checkpoint": str(args.base_checkpoint),
        "stage": str(args.stage),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "base_config_sha256": sha(args.base_checkpoint / "config.json"),
        "plan_sha256": sha(args.stage / "OFFLINE_EVAL_PLAN.json"),
        "adapter_sha256": sha(args.stage / "ADAPTER_CONFIG.json"),
        "cache_boundary": "raw frozen VLM; before trainable vlln and VL self-attention",
        "runtime_verified_fixed_state": False,
        "fixed_state_interpretation": "missing leg/waist zero placeholders",
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": 8,
        "noise_seeds": plan["noise_seeds"],
        "files": [],
        "strict_loading": loading,
    }
    train_keys = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for split in ("train", "validation"):
            loader = G1AllLerobotH40EpisodeLoader(
                args.stage / split,
                modality,
                args.stage / "ADAPTER_CONFIG.json",
                require_runtime_verified_fixed_state=False,
            )
            positions = {row["episode_index"]: i for i, row in enumerate(loader.episodes_metadata)}
            for selection in plan[split]:
                ep_id = selection["episode_index"]
                episode = loader[positions[ep_id]]
                anchors = (
                    list(range(len(episode) - 39))
                    if split == "train"
                    else selection["anchor_indices"]
                )
                records = []
                for anchor in anchors:
                    if time.monotonic() - started > args.wall_seconds:
                        raise TimeoutError("Cache wall budget exceeded")
                    step = extract_step_data(
                        episode, anchor, modality, EmbodimentTag.NEW_EMBODIMENT, allow_padding=False
                    )
                    item = apply_active_action_mask(
                        processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
                    )
                    batch = processor.collator([item])["inputs"]
                    backbone_input, action_input = model.prepare_input(dict(batch))
                    raw = model.backbone(backbone_input)
                    if not torch.isfinite(raw.backbone_features).all():
                        raise ValueError("Nonfinite frozen VLM features")
                    target = np.concatenate([step.actions[key] for key in GROUPS], axis=-1)
                    normalized_target = action_input.action[0].float().cpu().numpy()
                    decoded = processor.decode_action(
                        normalized_target, EmbodimentTag.NEW_EMBODIMENT, state=step.states
                    )
                    roundtrip = max(
                        float(np.abs(decoded[key] - step.actions[key]).max()) for key in GROUPS
                    )
                    if roundtrip > 5e-5:
                        raise ValueError(f"Target roundtrip error {roundtrip}")
                    record = {
                        "episode": ep_id,
                        "anchor": anchor,
                        "raw": tree_to(dict(raw), "cpu"),
                        "action_input": tree_to(dict(action_input), "cpu"),
                        "states": {
                            key: torch.from_numpy(value.copy())
                            for key, value in step.states.items()
                        },
                        "target": torch.from_numpy(target.copy()),
                        "target_roundtrip_max_abs": roundtrip,
                    }
                    if "native_smoke" not in manifest:
                        observation = {
                            key: value
                            for key, value in batch.items()
                            if key not in ("action", "action_mask")
                        }
                        seed(args.seed)
                        direct = model.get_action(dict(observation))["action_pred"]
                        cached_obs = BatchFeature(
                            data={
                                key: value
                                for key, value in action_input.items()
                                if key not in ("action", "action_mask")
                            }
                        )
                        seed(args.seed)
                        cached = model.action_head.get_action(copy_features(raw), cached_obs)[
                            "action_pred"
                        ]
                        parity = float((direct.float() - cached.float()).abs().max())
                        seed(args.seed)
                        direct_loss = model(dict(batch))["loss"]
                        seed(args.seed)
                        cached_loss = model.action_head(
                            copy_features(raw), BatchFeature(data=dict(action_input))
                        )["loss"]
                        loss_parity = float((direct_loss - cached_loss).abs())
                        if parity > 1e-5 or loss_parity > 1e-5 or not torch.isfinite(direct).all():
                            raise ValueError(
                                f"Native raw-cache parity failed: {parity}, {loss_parity}"
                            )
                        manifest["native_smoke"] = {
                            "action_shape": list(direct.shape),
                            "action_parity_max_abs": parity,
                            "loss_parity_abs": loss_parity,
                            "finite": True,
                        }
                        write(root / "NATIVE_SMOKE_RESULT.json", manifest["native_smoke"])
                    records.append(record)
                    if split == "train":
                        train_keys.append([ep_id, anchor])
                path = root / f"{split}_{ep_id:06d}.pt"
                torch.save(records, path)
                manifest["files"].append(
                    {
                        "split": split,
                        "episode": ep_id,
                        "path": str(path),
                        "samples": len(records),
                        "sha256": sha(path),
                        "bytes": path.stat().st_size,
                    }
                )
                print(json.dumps({"cached": path.name, "samples": len(records)}), flush=True)
    rng = np.random.default_rng(args.seed)
    schedule = rng.integers(0, len(train_keys), size=(args.steps, 8), dtype=np.int64)
    np.save(root / "train_schedule.npy", schedule)
    write(root / "train_keys.json", train_keys)
    manifest.update(
        schedule_sha256=sha(root / "train_schedule.npy"),
        train_keys_sha256=sha(root / "train_keys.json"),
        train_samples=len(train_keys),
        elapsed_seconds=time.monotonic() - started,
        status="completed",
        technical_validity=True,
    )
    write(root / "CACHE_RESULT.json", manifest)


def load_head(checkpoint):
    cfg = Gr00tN1d7Config.from_pretrained(checkpoint, local_files_only=True)
    cfg.tune_projector = cfg.tune_diffusion_model = cfg.tune_vlln = True
    cfg.tune_llm = cfg.tune_visual = False
    cfg.state_dropout_prob = 0.0
    if cfg.action_backbone != "dit" or cfg.action_horizon != 40 or cfg.num_inference_timesteps != 4:
        raise ValueError("Native action contract mismatch")
    head = Gr00tN1d7ActionHead(cfg)
    index = read(checkpoint / "model.safetensors.index.json")["weight_map"]
    weights = {}
    for shard in sorted({value for key, value in index.items() if key.startswith("action_head.")}):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith("action_head."):
                    weights[key.removeprefix("action_head.")] = stream.get_tensor(key)
    head.load_state_dict(weights, strict=True)
    return head.float()


def initialize_head(checkpoint, condition, init_seed):
    seed(init_seed)
    head = load_head(checkpoint)
    audit = {"condition": condition, "native_head_strict_load": True, "init_seed": init_seed}
    if condition == "conditional_mamba2":
        holder = nn.Module()
        holder.action_head = head
        holder.config = head.config
        seed(init_seed)
        audit.update(convert_native_model(holder))
    elif condition == "dit_reinit_backbone":
        original = head.model
        seed(init_seed)
        # Same constructor as official AlternateVLDiT; reset the entire declared model.
        head.model = type(original)(
            **head.config.diffusion_model_cfg,
            cross_attention_dim=head.config.backbone_embedding_dim,
            attend_text_every_n_blocks=head.config.attend_text_every_n_blocks,
        )
        audit["replacement_prefix"] = "action_head.model."
    elif condition != "dit_pretrained":
        raise ValueError(condition)
    audit.update(
        total_head_parameters=sum(p.numel() for p in head.parameters()),
        backbone_parameters=sum(p.numel() for p in head.model.parameters()),
    )
    return head, audit


def load_records(cache_root, split):
    manifest = read(cache_root / "CACHE_RESULT.json")
    if not manifest["technical_validity"]:
        raise ValueError("Invalid cache")
    records = []
    for row in manifest["files"]:
        if row["split"] == split:
            if sha(row["path"]) != row["sha256"]:
                raise ValueError("Cache shard hash changed")
            records.extend(torch.load(row["path"], map_location="cpu", weights_only=True))
    return records, manifest


def probe(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records, manifest = load_records(args.cache_root, "validation")
    head, audit = initialize_head(args.base_checkpoint, args.condition, args.seed)
    head = head.cuda().eval()
    raw, action = pack_records(records[:1])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        seed(args.seed)
        loss = head(copy_features(raw), BatchFeature(data=dict(action)))["loss"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite probe loss")
        observation = BatchFeature(
            data={
                key: value for key, value in action.items() if key not in ("action", "action_mask")
            }
        )
        samples = []
        for index in range(15):
            seed(args.seed)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            output = head.get_action(copy_features(raw), observation)["action_pred"]
            torch.cuda.synchronize()
            if output.shape != (1, 40, 132) or not torch.isfinite(output).all():
                raise ValueError("Bad probe actions")
            if index >= 5:
                samples.append((time.perf_counter() - tick) * 1000)
    report = {
        **audit,
        "status": "completed",
        "technical_validity": True,
        "native_flow_loss": float(loss),
        "latency_boundary": "head only; raw VLM outputs precomputed; all native head projections and four Euler calls",
        "head_policy_p50_ms": float(np.median(samples)),
        "head_policy_p95_ms": float(np.percentile(samples, 95)),
        "head_policy_samples_ms": samples,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.monotonic() - started,
        "cache_config_sha256": manifest["base_config_sha256"],
    }
    write(args.output / "PROBE_RESULT.json", report)
    print(json.dumps(report), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("cache", "probe", "train", "evaluate"))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--head-checkpoint", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(4)
    seed(args.seed)
    try:
        {"cache": cache, "probe": probe}[args.command](args)
    except Exception as exc:
        root = args.cache_root if args.command == "cache" else args.output
        if root is not None and root.exists() and not (root / "FAILURE.json").exists():
            write(
                root / "FAILURE.json",
                {"status": "failed", "exception": type(exc).__name__, "message": str(exc)},
            )
        raise
