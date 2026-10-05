"""CPU input parity and bounded full-visual ZeRO-2 preflight for the Yuqi recipe.

Run this same file with PYTHONPATH pointing to Yuqi for reference preparation,
then with PYTHONPATH pointing to this worktree for the candidate implementation.
No model weights are loaded by the prepare command.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import time

from contract import (
    LLM_PREFIXES,
    VISUAL_PREFIXES,
    assert_compact13_contract,
    preserve_yuqi_language,
    processor_compatibility_config,
    seed_microbatch,
    sha,
    tensor_identity,
    write_new,
)
import numpy as np
from runner_core import assert_mask, gradient_checks, rank_slots
import torch


HERE = Path(__file__).resolve().parent


def prepare(args):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    args.output.mkdir(parents=True, exist_ok=False)
    processor_dir = args.output / "processor"
    processor_dir.mkdir()
    source = json.loads((args.checkpoint / "processor_config.json").read_text())
    if args.reference:
        adapted, adaptation = source, {"reference": True, "processor_config_unchanged": True}
    else:
        adapted, adaptation = processor_compatibility_config(source, legacy_letterbox=True)
    write_new(processor_dir / "processor_config.json", adapted)
    for name in ("statistics.json", "embodiment_id.json", "config.json"):
        shutil.copy2(args.checkpoint / name, processor_dir / name)
    processor = Gr00tN1d7Processor.from_pretrained(
        processor_dir,
        local_files_only=True,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
    )
    view = args.output / "metadata_view"
    meta = view / "meta"
    meta.mkdir(parents=True)
    for path in (args.dataset / "meta").iterdir():
        if path.is_file():
            shutil.copy2(path, meta / path.name)
    info = json.loads((meta / "info.json").read_text())
    if info["total_episodes"] != 220:
        raise ValueError("Expected the original 220 narrow episodes")
    for key in ("data_path", "video_path"):
        info[key] = info[key].replace("{chunk_index", "{episode_chunk")
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    for name in ("data", "videos"):
        (view / name).symlink_to((args.dataset / name).resolve(), target_is_directory=True)
    modality = processor.get_modality_configs()["new_embodiment"]
    loader_kwargs = {}
    if "video_backend" in inspect.signature(LeRobotEpisodeLoader).parameters:
        loader_kwargs["video_backend"] = "torchcodec"
    loader = LeRobotEpisodeLoader(view, modality, **loader_kwargs)
    preserve_yuqi_language(loader)
    index = next(
        i for i, row in enumerate(loader.episodes_metadata) if row["episode_index"] == args.episode
    )
    episode = loader[index]
    if len(episode) < 47:
        raise ValueError("Probe episode needs at least 47 frames for B32/H16")
    results = {}
    for mode in ("eval", "train"):
        processor.train() if mode == "train" else processor.eval()
        items = []
        for anchor in range(32):
            seed_microbatch(anchor, 0, initialize_cuda=False)
            step = extract_step_data(
                episode, anchor, modality, EmbodimentTag.NEW_EMBODIMENT, allow_padding=False
            )
            item = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            assert_compact13_contract(processor, item)
            items.append(item)
        batch = processor.collator(items)["inputs"]
        filename = args.output / f"{mode}_batch.pt"
        torch.save(batch, filename)
        identities = {key: tensor_identity(value) for key, value in batch.items()}
        results[mode] = {"path": str(filename), "sha256": sha(filename), "tensors": identities}
    import gr00t

    report = {
        "status": "prepared",
        "reference": args.reference,
        "gr00t_source": gr00t.__file__,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "episode": args.episode,
        "anchors": list(range(32)),
        "global_batch": 32,
        "compatibility": adaptation,
        "checkpoint_config_sha256": sha(args.checkpoint / "config.json"),
        "checkpoint_processor_sha256": sha(args.checkpoint / "processor_config.json"),
        "batches": results,
    }
    write_new(args.output / "PREPARED.json", report)
    print(json.dumps(report, indent=2), flush=True)


def compare(args):
    a = json.loads((args.prepared / "PREPARED.json").read_text())
    b = json.loads((args.reference_prepared / "PREPARED.json").read_text())
    differences = []
    for mode in ("eval", "train"):
        for key in sorted(set(a["batches"][mode]["tensors"]) | set(b["batches"][mode]["tensors"])):
            if a["batches"][mode]["tensors"].get(key) != b["batches"][mode]["tensors"].get(key):
                differences.append([mode, key])
    result = {
        "status": "passed" if not differences else "failed",
        "tensor_differences": differences,
        "candidate_manifest_sha256": sha(args.prepared / "PREPARED.json"),
        "reference_manifest_sha256": sha(args.reference_prepared / "PREPARED.json"),
    }
    write_new(args.output, result)
    print(json.dumps(result), flush=True)
    if differences:
        raise RuntimeError(f"Original/candidate processor input mismatch: {differences}")


def full_model(args):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    model, info = Gr00tN1d7.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
        tune_visual=True,
        tune_llm=False,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        max_state_dim=13,
        max_action_dim=13,
        action_horizon=40,
        state_dropout_prob=0.2,
        load_bf16=False,
        backbone_trainable_params_fp32=True,
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Strict compact13 load failed: {info}")
    cfg = model.config
    if (cfg.max_state_dim, cfg.max_action_dim, cfg.action_horizon, cfg.num_inference_timesteps) != (
        13,
        13,
        40,
        4,
    ):
        raise ValueError("Yuqi model contract drift")
    if cfg.diffusion_model_cfg["num_layers"] != 32:
        raise ValueError("Use checkpoint-effective 32 layers, not input YAML 16")
    audit = {"strict_native_loading": info, "condition": args.condition}
    seed_microbatch(0, 0, initialize_cuda=False)
    if args.condition == "conditional_mamba2":
        from gr00t.model.modules.mamba_action_mixer import convert_native_model

        audit.update(convert_native_model(model))
    elif args.condition == "dit_reinit_backbone":
        model.action_head.model = type(model.action_head.model)(
            **cfg.diffusion_model_cfg,
            cross_attention_dim=cfg.backbone_embedding_dim,
            attend_text_every_n_blocks=cfg.attend_text_every_n_blocks,
        )
    return model, audit


def initialization_evidence(model):
    """Read actual post-load layouts/hooks and RoPE values without changing them."""
    qwen = model.backbone.model
    visual, language = qwen.visual, qwen.language_model
    module_names = {id(module): name for name, module in model.named_modules()}
    convolutions = []
    for module in visual.modules():
        if not isinstance(module, torch.nn.Conv3d):
            continue
        hooks = [
            getattr(hook, "__qualname__", type(hook).__qualname__)
            for hook in module._forward_pre_hooks.values()
        ]
        convolutions.append(
            {
                "module": module_names[id(module)],
                "weight_shape": list(module.weight.shape),
                "weight_dtype": str(module.weight.dtype),
                "weight_device": str(module.weight.device),
                "weight_stride": list(module.weight.stride()),
                "weight_contiguous_ncdhw": module.weight.is_contiguous(),
                "weight_contiguous_channels_last_3d": module.weight.is_contiguous(
                    memory_format=torch.channels_last_3d
                ),
                "forward_pre_hook_count": len(hooks),
                "forward_pre_hooks": hooks,
            }
        )
    rope = {}
    for group, rotary in (("vision", visual.rotary_pos_emb), ("text", language.rotary_emb)):
        values = {}
        for name in ("inv_freq", "original_inv_freq"):
            value = getattr(rotary, name, None)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected tensor at {group} RoPE {name}")
            is_buffer = name in rotary._buffers
            values[name] = {
                **tensor_identity(value),
                "device": str(value.device),
                "stride": list(value.stride()),
                "registered_buffer": is_buffer,
                "persistent": name not in rotary._non_persistent_buffers_set if is_buffer else None,
            }
        rope[group] = {"module": module_names[id(rotary)], "values": values}
    return {"torch_version": str(torch.__version__), "vision_conv3d": convolutions, "rope": rope}


def cuda_probe(args):
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 2:
        raise ValueError("Yuqi preflight requires two GPUs/global32")
    slots = rank_slots(rank, args.microbatch, args.accumulation)
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    deepspeed.init_distributed(dist_backend="nccl")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    prepared = json.loads((args.prepared / "PREPARED.json").read_text())
    for mode in ("train", "eval"):
        if sha(prepared["batches"][mode]["path"]) != prepared["batches"][mode]["sha256"]:
            raise ValueError("Prepared batch changed")
    batch = torch.load(prepared["batches"]["train"]["path"], map_location="cpu", weights_only=True)
    # Pixel tensors concatenate image patches, unlike batch-leading state/text.
    # Re-collate per-rank input by slicing image_grid_thw and patch spans together.
    batches = [slice_batch(batch, part[0], part[-1] + 1) for part in slots]
    for batch in batches:
        assert_mask(batch["action_mask"])
    eval_batch = slice_batch(
        torch.load(prepared["batches"]["eval"]["path"], map_location="cpu", weights_only=True),
        slots[0][0],
        slots[0][-1] + 1,
    )
    model, audit = full_model(args)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not any(name.startswith(VISUAL_PREFIXES) for name, _ in parameters):
        raise ValueError("Visual training is required; raw VLM caching is forbidden")
    if any(name.startswith(LLM_PREFIXES) for name, _ in parameters):
        raise ValueError("Language weights must remain frozen")
    norm_names = {
        f"{name}.{key}"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
        for key, _ in module.named_parameters(recurse=False)
    }
    groups = [
        {
            "params": [
                p for name, p in parameters if not (name.endswith("bias") or name in norm_names)
            ],
            "weight_decay": 1e-5,
        },
        {
            "params": [p for name, p in parameters if name.endswith("bias") or name in norm_names],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    ds_config = json.loads(Path(args.zero_config).read_text())
    ds_config.update(
        train_batch_size=32,
        train_micro_batch_size_per_gpu=args.microbatch,
        gradient_accumulation_steps=args.accumulation,
        gradient_clipping=1.0,
    )
    ds_config["bf16"]["enabled"] = True
    ds_config["fp16"]["enabled"] = False
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config
    )
    engine.eval()
    initialized = initialization_evidence(engine.module)
    seed_microbatch(0, rank)
    with torch.no_grad():
        observation = {
            key: value for key, value in eval_batch.items() if key not in ("action", "action_mask")
        }
        initial_action = engine.module.get_action(
            {key: value.cuda(local_rank) for key, value in observation.items()}
        )["action_pred"]
    if (
        initial_action.shape != (args.microbatch, 40, 13)
        or not torch.isfinite(initial_action).all()
    ):
        raise ValueError("Nonfinite or non-native preflight action")
    np.save(args.output / f"initial_action_rank{rank}.npy", initial_action.float().cpu().numpy())
    initialization_path = args.output / f"INITIALIZATION_RANK{rank}.json"
    write_new(
        initialization_path,
        {
            "post_deepspeed_initialization": initialized,
            "post_initial_inference": initialization_evidence(engine.module),
        },
    )
    engine.train()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    gradient_report = {}
    started = time.monotonic()
    first_step = {
        "rank": rank,
        "condition": args.condition,
        "microbatches": [],
        "selected_gradients": {},
        "gradient_stage": "after first optimizer-step accumulation backward, before clipping/update",
        "time_sampler": {
            "alpha": float(engine.module.action_head.beta_dist.concentration1),
            "beta": float(engine.module.action_head.beta_dist.concentration0),
            "dtype": str(engine.module.action_head.beta_dist.concentration1.dtype),
            "device": str(engine.module.action_head.beta_dist.concentration1.device),
        },
    }
    captured_encoder_inputs = {}

    def capture_native_noisy_input(module, inputs):
        # This observes existing tensors without drawing RNG or changing model outputs.
        captured_encoder_inputs["noisy_trajectory"] = inputs[0].detach().cpu().clone()
        captured_encoder_inputs["time_buckets"] = inputs[1].detach().cpu().clone()

    capture_handle = engine.module.action_head.action_encoder.register_forward_pre_hook(
        capture_native_noisy_input
    )
    first_step_report = None
    for step in range(args.steps):
        if time.monotonic() - started > args.wall_seconds:
            raise TimeoutError("Preflight wall budget exhausted")
        tick = time.perf_counter()
        for micro, batch in enumerate(batches):
            seed = seed_microbatch(step * args.accumulation + micro, rank)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = engine(
                    inputs={key: value.cuda(local_rank) for key, value in batch.items()}
                )
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native masked loss")
            if step == 0:
                first_step["microbatches"].append(
                    {
                        "micro": micro,
                        "seed": seed,
                        "global_slots": slots[micro],
                        "loss": loss.detach().cpu().clone(),
                        "action_loss": result["action_loss"].detach().cpu().clone(),
                        "action_mask": result["action_mask"].detach().cpu().clone(),
                        **captured_encoder_inputs,
                    }
                )
            engine.backward(loss)
            if step == 0 and micro == args.accumulation - 1:
                gradient_report = gradient_checks(parameters)
                parameter_map = dict(parameters)
                for group, check in gradient_report.items():
                    name = check["parameter"]
                    gradient = safe_get_full_grad(parameter_map[name])
                    if gradient is None:
                        raise ValueError(f"Missing parity gradient: {name}")
                    first_step["selected_gradients"][group] = {
                        "parameter": name,
                        "tensor": gradient.detach().cpu().clone(),
                    }
                capture_handle.remove()
                evidence_file = args.output / f"first_step_training_rank{rank}.pt"
                torch.save(first_step, evidence_file)
                first_step_report = {
                    "path": str(evidence_file.resolve()),
                    "sha256": sha(evidence_file),
                    "time_sampler": first_step["time_sampler"],
                    "gradient_stage": first_step["gradient_stage"],
                    "microbatches": [
                        {
                            "micro": row["micro"],
                            "seed": row["seed"],
                            "loss": float(row["loss"]),
                            "loss_dtype": str(row["loss"].dtype),
                            "tensors": {
                                key: tensor_identity(row[key])
                                for key in (
                                    "action_loss",
                                    "action_mask",
                                    "noisy_trajectory",
                                    "time_buckets",
                                )
                            },
                        }
                        for row in first_step["microbatches"]
                    ],
                    "selected_gradients": {
                        group: {
                            "parameter": row["parameter"],
                            "tensor": tensor_identity(row["tensor"]),
                        }
                        for group, row in first_step["selected_gradients"].items()
                    },
                }
                write_new(args.output / f"FIRST_STEP_TRAINING_RANK{rank}.json", first_step_report)
            engine.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
    report = {
        **audit,
        "rank": rank,
        "status": "completed",
        "global_batch": 32,
        "micro_batch": args.microbatch,
        "gradient_accumulation_steps": args.accumulation,
        "rank_slots": slots,
        "active_mask_sum_per_microbatch": [int(batch["action_mask"].sum()) for batch in batches],
        "steps": args.steps,
        "loss": float(loss),
        "durations_seconds": durations,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gradient_checks": gradient_report,
        "first_step_training_evidence": first_step_report,
        "initial_action_identity": tensor_identity(initial_action),
        "initialization_evidence": {
            "path": str(initialization_path.resolve()),
            "sha256": sha(initialization_path),
        },
        "prepared_manifest_sha256": sha(args.prepared / "PREPARED.json"),
        "training_precision": "native Yuqi DeepSpeed BF16/TF32; no raw VLM cache",
    }
    write_new(args.output / f"PROBE_RANK{rank}.json", report)
    dist.barrier()
    dist.destroy_process_group()


def slice_batch(batch, start, stop):
    size = batch["state"].shape[0]
    if batch["image_grid_thw"].shape[0] != size * 3:
        raise ValueError("Expected three image grids per sample")
    grids = batch["image_grid_thw"]
    patch_counts = grids.prod(-1)
    patch_start = int(patch_counts[: start * 3].sum())
    patch_stop = int(patch_counts[: stop * 3].sum())
    return {
        key: (
            value[patch_start:patch_stop]
            if key == "pixel_values"
            else value[start * 3 : stop * 3]
            if key == "image_grid_thw"
            else value[start:stop]
        )
        for key, value in batch.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "compare-inputs", "cuda"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(json.loads((HERE / "yuqi_recipe.json").read_text())["base_checkpoint"]),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(json.loads((HERE / "yuqi_recipe.json").read_text())["dataset"]),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--reference-prepared", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--condition",
        choices=("dit_pretrained", "dit_reinit_backbone", "conditional_mamba2"),
        default="dit_pretrained",
    )
    parser.add_argument("--microbatch", type=int, choices=(8, 16), default=16)
    parser.add_argument("--accumulation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--wall-seconds", type=int, default=900)
    parser.add_argument(
        "--zero-config",
        type=Path,
        default=Path("/home/yuqi/Isaac-GR00T-N1.7/gr00t/configs/deepspeed/zero2_config.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"prepare": prepare, "compare-inputs": compare, "cuda": cuda_probe}[args.command](args)


if __name__ == "__main__":
    main()
