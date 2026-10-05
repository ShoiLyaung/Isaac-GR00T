"""Frozen-anchor Yuqi compact13 training with trainable visual backbone.

Preparation is CPU-only. Training requires torchrun --nproc-per-node=2 and the
separate passing input/forward parity reports. No cached VLM features are used.
Only the final checkpoint and resumable DeepSpeed state are saved.
"""

import argparse
from collections import OrderedDict
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from unittest.mock import patch

from contract import (
    LLM_PREFIXES,
    VISUAL_PREFIXES,
    assert_compact13_contract,
    preserve_yuqi_language,
    sha,
    tensor_identity,
    write_new,
)
import numpy as np
from preflight import full_model
from runner_core import (
    GLOBAL_BATCH,
    NOISE_SEEDS,
    PROBE_STEPS,
    STEPS,
    aggregate_metrics,
    deterministic_seed,
    gradient_checks,
    make_schedule,
    masked_flow,
    preserve_eval_state,
    rank_slots,
    set_seed,
)
import torch
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECIPE = json.loads((HERE / "yuqi_recipe.json").read_text())
METRICS = (
    "flow_loss",
    "arm_mae_rad",
    "hand_mae_u",
    "combined_mae_mixed_units",
    "normalized_out_of_range_before",
    "normalized_out_of_range_after",
    "hand_u_out_of_range_before",
    "hand_u_out_of_range_after",
)


def read(path):
    return json.loads(Path(path).read_text())


def processor_from(path):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    return Gr00tN1d7Processor.from_pretrained(
        path,
        local_files_only=True,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
    )


class NativeAnchors:
    """Native loader/extractor with bounded raw data and decoder caches.

    Only current RGB is decoded. A local H16 dataframe is then passed through the
    native extract_step_data; actions, states and language use native parquet
    parsing. At most 24 decoders, 220 small numeric dataframes, and 256MiB of raw
    RGB samples are retained per process. No encoded visual features are cached.
    """

    def __init__(self, prepared):
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        self.prepared = Path(prepared)
        self.processor = processor_from(self.prepared / "processor")
        self.modality = self.processor.get_modality_configs()["new_embodiment"]
        for key in ("video", "state", "language"):
            if self.modality[key].delta_indices != [0]:
                raise ValueError(f"Expected current-only Yuqi {key}")
        kwargs = {}
        if "video_backend" in inspect.signature(LeRobotEpisodeLoader).parameters:
            kwargs["video_backend"] = "torchcodec"
        self.loader = LeRobotEpisodeLoader(self.prepared / "metadata_view", self.modality, **kwargs)
        preserve_yuqi_language(self.loader)
        self.metadata = {row["episode_index"]: row for row in self.loader.episodes_metadata}
        self.tables = OrderedDict()
        self.decoders = OrderedDict()
        self.raw_cache = OrderedDict()
        self.raw_bytes = 0

    def table(self, episode):
        if episode not in self.tables:
            meta = self.metadata[episode]
            frame = self.loader._load_parquet_data(episode)
            if len(frame) != meta["length"]:
                raise ValueError(f"Episode {episode}: actual/metadata length mismatch")
            lang = self.modality["language"].modality_keys[0]
            if lang in ("task", "sub_task"):
                if lang == "task" and len(meta["tasks"]) != 1:
                    raise ValueError("This frozen recipe requires one task string per episode")
                frame[f"language.{lang}"] = self.loader.create_language_from_meta(
                    meta, len(frame), lang
                )
            self.tables[episode] = frame
            if len(self.tables) > 220:
                self.tables.popitem(last=False)
        self.tables.move_to_end(episode)
        return self.tables[episode]

    def frames(self, video_path, indices, decoder_kwargs=None, **kwargs):
        from gr00t.utils.video_utils import _build_decoder

        if video_path not in self.decoders:
            self.decoders[video_path] = _build_decoder(
                video_path, {"num_ffmpeg_threads": 1, **(decoder_kwargs or {})}
            )
            if len(self.decoders) > 24:
                self.decoders.popitem(last=False)
        self.decoders.move_to_end(video_path)
        return self.decoders[video_path].get_frames_at(indices=indices).data.numpy()

    def raw(self, episode, anchor):
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag

        key = (int(episode), int(anchor))
        if key in self.raw_cache:
            self.raw_cache.move_to_end(key)
            return self.raw_cache[key][0]
        episode, anchor = key
        table = self.table(episode)
        if not 0 <= anchor <= len(table) - 16:
            raise ValueError(f"Illegal H16 anchor: {key}")
        frame = table.iloc[anchor : anchor + 16].copy().reset_index(drop=True)
        # Reuse native filename/key resolution; only decoder construction is cached.
        with patch("gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices", self.frames):
            videos = self.loader._load_video_data(episode, np.asarray([anchor]))
        for camera, values in videos.items():
            frame[f"video.{camera}"] = [values[0]] + [None] * 15
        raw = extract_step_data(
            frame, 0, self.modality, EmbodimentTag.NEW_EMBODIMENT, allow_padding=False
        )
        nbytes = sum(image.nbytes for images in raw.images.values() for image in images)
        self.raw_cache[key] = (raw, nbytes)
        self.raw_bytes += nbytes
        while self.raw_bytes > 256 * 2**20:
            _, (_, removed_bytes) = self.raw_cache.popitem(last=False)
            self.raw_bytes -= removed_bytes
        return raw

    def item(self, episode, anchor, augmentation_seed=None):
        from gr00t.data.types import MessageType

        raw = self.raw(episode, anchor)
        if augmentation_seed is not None:
            set_seed(augmentation_seed)
            transform = self.processor.train_image_transform
            if hasattr(transform, "set_random_seed"):
                transform.set_random_seed(augmentation_seed)
        item = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": raw}])
        assert_compact13_contract(self.processor, item)
        return item, raw

    def batch(self, pairs):
        records = [self.item(int(ep), int(anchor)) for ep, anchor in pairs]
        return self.processor.collator([item for item, _ in records])["inputs"], [
            raw for _, raw in records
        ]


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = read(args.prepared / "PREPARED.json")
    if prepared.get("reference"):
        raise ValueError("Schedule must use the candidate parity-prepared native input view")
    source = NativeAnchors(args.prepared)
    lengths = {}
    parquet_identities = []
    for episode in sorted(source.metadata):
        frame = source.table(episode)
        lengths[episode] = len(frame)
        for key in ("action.left_arm", "action.left_hand", "state.left_arm", "state.left_hand"):
            value = np.stack(frame[key])
            expected_dim = 7 if key.endswith("left_arm") else 6
            if value.shape != (len(frame), expected_dim) or not np.isfinite(value).all():
                raise ValueError(f"Invalid physical values in episode{episode}:{key}")
        path = source.loader.dataset_path / source.loader.data_path_pattern.format(
            episode_chunk=episode // source.loader.chunk_size, episode_index=episode
        )
        parquet_identities.append(
            {
                "episode": episode,
                "path": str(path.resolve()),
                "sha256": sha(path),
                "length": len(frame),
            }
        )
    if len(lengths) != 220 or sum(lengths.values()) != 105027:
        raise ValueError("Expected original narrow train220 / 105027 frames")
    schedule, anchors = make_schedule(lengths)
    np.save(args.output / "train_schedule.npy", schedule)
    write_new(args.output / "anchors.json", anchors)
    # Verify the optimized raw access against the already frozen CPU eval batch.
    source.processor.eval()
    items = [source.item(prepared["episode"], anchor)[0] for anchor in prepared["anchors"]]
    batch = source.processor.collator(items)["inputs"]
    expected = torch.load(args.prepared / "eval_batch.pt", weights_only=True, map_location="cpu")
    differences = [
        key
        for key in sorted(set(batch) | set(expected))
        if key not in batch or key not in expected or not torch.equal(batch[key], expected[key])
    ]
    if differences:
        raise ValueError(f"Native partial-loader input parity failed: {differences}")
    video_inventory = []
    for path in sorted((source.loader.dataset_path / "videos").rglob("*.mp4")):
        st = path.stat()
        video_inventory.append(
            {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        )
    write_new(args.output / "video_inventory.json", video_inventory)
    write_new(
        args.output / "SCHEDULE.json",
        {
            "status": "prepared",
            "prepared": str(args.prepared.resolve()),
            "prepared_manifest_sha256": sha(args.prepared / "PREPARED.json"),
            "schedule_sha256": sha(args.output / "train_schedule.npy"),
            "anchors_sha256": sha(args.output / "anchors.json"),
            "video_inventory_sha256": sha(args.output / "video_inventory.json"),
            "video_identity_policy": "absolute path, byte size, mtime_ns; no transcode",
            "parquet": parquet_identities,
            "steps": STEPS,
            "global_batch": GLOBAL_BATCH,
            "sampling": "seed42 complete shuffled epochs over all legal H16 anchors",
            "shape": list(schedule.shape),
            "train_episodes": len(lengths),
            "total_frames": sum(lengths.values()),
            "legal_anchors": sum(n - 15 for n in lengths.values()),
            "native_partial_loader_input_parity": "exact CPU eval tensor equality, B32 three cameras",
            "noise_seeds": list(NOISE_SEEDS),
            "diagnostic_scope": "seen-data diagnostic; no unseen validation",
        },
    )
    print(
        json.dumps({"status": "prepared", "schedule": str(args.output / "SCHEDULE.json")}),
        flush=True,
    )


def load_plan(path):
    manifest = read(path / "SCHEDULE.json")
    for name, key in (
        ("train_schedule.npy", "schedule_sha256"),
        ("anchors.json", "anchors_sha256"),
        ("video_inventory.json", "video_inventory_sha256"),
    ):
        if sha(path / name) != manifest[key]:
            raise ValueError(f"Frozen plan changed: {name}")
    prepared = Path(manifest["prepared"])
    if sha(prepared / "PREPARED.json") != manifest["prepared_manifest_sha256"]:
        raise ValueError("Prepared input manifest changed")
    for row in manifest["parquet"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"Parquet changed: {row['path']}")
    for row in read(path / "video_inventory.json"):
        st = Path(row["path"]).stat()
        if st.st_size != row["size"] or st.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"Video changed: {row['path']}")
    schedule = np.load(path / "train_schedule.npy")
    if schedule.shape != (STEPS, GLOBAL_BATCH, 2):
        raise ValueError("Expected frozen 5000x32 schedule")
    return manifest, prepared, schedule, read(path / "anchors.json")


class ScheduledMicrobatches(Dataset):
    """Workers transform whole microbatches; no worker-side GPU or VLM execution."""

    def __init__(self, prepared, schedule, rank, microbatch, accumulation):
        self.prepared, self.schedule = prepared, schedule
        self.rank, self.microbatch, self.accumulation = rank, microbatch, accumulation
        self.slots = rank_slots(rank, microbatch, accumulation)
        self.source = None

    def __len__(self):
        return STEPS * self.accumulation

    def __getitem__(self, index):
        if self.source is None:
            self.source = NativeAnchors(self.prepared)
            self.source.processor.train()
        step, micro = divmod(index, self.accumulation)
        slots = self.slots[micro]
        pairs = self.schedule[step, slots]
        items = [
            self.source.item(int(ep), int(anchor), deterministic_seed("augmentation", step, slot))[
                0
            ]
            for slot, (ep, anchor) in zip(slots, pairs)
        ]
        batch = self.source.processor.collator(items)["inputs"]
        return {
            "inputs": batch,
            "step": step,
            "micro": micro,
            "slots": slots,
            "pairs": pairs,
            "noise_seed": deterministic_seed("forward", step, micro, self.rank),
        }


def physical_metrics(prediction, raw, processor):
    from gr00t.data.embodiment_tags import EmbodimentTag

    if prediction.shape != (40, 13) or not np.isfinite(prediction).all():
        raise ValueError("Expected finite normalized H40x13 prediction")
    decoded = processor.decode_action(prediction, EmbodimentTag.NEW_EMBODIMENT, state=raw.states)
    physical = np.concatenate([decoded[key] for key in ("left_arm", "left_hand")], axis=-1)
    target = np.concatenate([raw.actions[key] for key in ("left_arm", "left_hand")], axis=-1)
    if target.shape != (16, 13) or physical.shape != target.shape:
        raise ValueError("Physical target/decoded action must be active H16x13")
    if processor.get_modality_configs()["new_embodiment"]["action"].mean_std_embedding_keys:
        raise ValueError("Yuqi physical metrics require native minmax normalization")
    params = processor.state_action_processor.norm_params["new_embodiment"]["action"]
    # Contract is absolute/minmax; compute the same inverse affine before native clipping.
    lo = np.concatenate([np.asarray(params[key]["min"]) for key in ("left_arm", "left_hand")])
    hi = np.concatenate([np.asarray(params[key]["max"]) for key in ("left_arm", "left_hand")])
    unclipped = (prediction[:16] + 1) * 0.5 * (hi - lo) + lo
    error = np.abs(physical - target)
    return {
        "arm_mae_rad": float(error[:, :7].mean()),
        "hand_mae_u": float(error[:, 7:].mean()),
        "combined_mae_mixed_units": float(error.mean()),
        "normalized_out_of_range_before": float((np.abs(prediction[:16]) > 1).mean()),
        "normalized_out_of_range_after": float(
            (np.abs(np.clip(prediction[:16], -1, 1)) > 1).mean()
        ),
        "hand_u_out_of_range_before": float(
            ((unclipped[:, 7:] < 0) | (unclipped[:, 7:] > 1)).mean()
        ),
        "hand_u_out_of_range_after": float(((physical[:, 7:] < 0) | (physical[:, 7:] > 1)).mean()),
    }


def evaluate_pairs(model, source, pairs, output, *, noise_seeds=NOISE_SEEDS, wall_seconds=5400):
    output.mkdir(parents=True, exist_ok=False)
    rows, predictions = [], []
    started = time.monotonic()
    with preserve_eval_state(model, source.processor), torch.no_grad():
        for index, (episode, anchor) in enumerate(pairs):
            batch, raw = source.batch([(episode, anchor)])
            observation = {
                key: value for key, value in batch.items() if key not in ("action", "action_mask")
            }
            for noise_seed in noise_seeds:
                if time.monotonic() - started > wall_seconds:
                    raise TimeoutError("Seen-data diagnostic wall budget exceeded")
                paired_seed = deterministic_seed(
                    "diagnostic", int(episode), int(anchor), int(noise_seed)
                )
                set_seed(paired_seed, cuda=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(inputs=dict(batch))
                    flow = masked_flow(result["action_loss"], result["action_mask"])[0]
                del result
                set_seed(paired_seed, cuda=True)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model.get_action(dict(observation))["action_pred"]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                prediction = predicted[0].float().cpu().numpy()
                rows.append(
                    {
                        "episode": int(episode),
                        "anchor": int(anchor),
                        "seed": int(noise_seed),
                        "paired_rng_seed": paired_seed,
                        "flow_loss": float(flow),
                        "active_mask_sum": 208,
                        "policy_service_ms": elapsed * 1000,
                        "action_identity": tensor_identity(predicted),
                        **physical_metrics(prediction, raw[0], source.processor),
                    }
                )
                predictions.append(prediction)
            if (index + 1) % 8 == 0:
                print(
                    json.dumps({"diagnostic": str(output), "anchors_done": index + 1}), flush=True
                )
    per_episode, metrics = aggregate_metrics(rows, METRICS)
    write_new(output / "details.json", rows)
    np.save(output / "normalized_predictions.npy", np.stack(predictions))
    report = {
        "status": "completed",
        "scope": "seen-data diagnostic; seed-anchor-episode aggregation",
        "anchors": len(pairs),
        "paired_outputs": len(rows),
        "per_episode": per_episode,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "prediction_sha256": sha(output / "normalized_predictions.npy"),
    }
    write_new(output / "EVALUATION_RESULT.json", report)
    return report


def strict_reload(path):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    model, info = Gr00tN1d7.from_pretrained(
        path,
        local_files_only=True,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Strict checkpoint reload failed: {info}")
    return model, info


def optimizer_for(model):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not any(name.startswith(VISUAL_PREFIXES) for name, _ in parameters):
        raise ValueError("Vision must remain trainable")
    if any(name.startswith(LLM_PREFIXES) for name, _ in parameters):
        raise ValueError("LLM must remain frozen")
    norms = {
        f"{name}.{key}"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
        for key, _ in module.named_parameters(recurse=False)
    }
    groups = [
        {
            "params": [
                p for name, p in parameters if (name.endswith("bias") or name in norms) == no_decay
            ],
            "weight_decay": 0.0 if no_decay else 1e-5,
        }
        for no_decay in (False, True)
    ]
    return torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.999), eps=1e-8), parameters


def source_identity():
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
        "files": {
            str(path.relative_to(REPO)): sha(path)
            for path in (
                HERE / "run_train_eval.py",
                HERE / "runner_core.py",
                HERE / "preflight.py",
                HERE / "contract.py",
                REPO / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
                REPO / "gr00t/model/modules/conditional_mamba.py",
            )
        },
    }


def validate_gate(path):
    if path is None or read(path).get("status") != "passed":
        raise ValueError(f"A passing parity report is required before training: {path}")
    return {"path": str(path.resolve()), "sha256": sha(path)}


def external_weight_identity(model):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in model.state_dict().items():
        if name.startswith("action_head.model."):
            continue
        digest.update(name.encode())
        data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(data))
        count += 1
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": count,
        "excluded_prefix": "action_head.model.",
    }


def train(args):
    import deepspeed
    import torch.distributed as dist
    from transformers import get_cosine_schedule_with_warmup

    started = time.monotonic()
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Training forbids skipped checkpoint weights")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise ValueError("Use torchrun --nproc-per-node=2")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    slots = rank_slots(rank, args.microbatch, args.accumulation)
    gates = {
        "input_parity": validate_gate(args.input_parity),
        "forward_parity": validate_gate(args.forward_parity),
    }
    plan, prepared, schedule, anchors = load_plan(args.plan)
    if Path(read(prepared / "PREPARED.json")["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Training base checkpoint differs from the frozen native input plan")
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    deepspeed.init_distributed(dist_backend="nccl")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    set_seed(42, cuda=True)
    model, audit = full_model(args)
    external_identity = external_weight_identity(model)
    optimizer, parameters = optimizer_for(model)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=250, num_training_steps=STEPS
    )
    config = read(args.zero_config)
    config.update(
        train_batch_size=GLOBAL_BATCH,
        train_micro_batch_size_per_gpu=args.microbatch,
        gradient_accumulation_steps=args.accumulation,
        gradient_clipping=1.0,
        steps_per_print=1000000,
        wall_clock_breakdown=False,
    )
    config["bf16"]["enabled"], config["fp16"]["enabled"] = True, False
    if config["zero_optimization"]["stage"] != 2:
        raise ValueError("Require ZeRO stage2")
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, config=config
    )
    dataset = ScheduledMicrobatches(prepared, schedule, rank, args.microbatch, args.accumulation)
    loader_options = {"batch_size": None, "num_workers": args.workers, "pin_memory": True}
    if args.workers:
        loader_options.update(
            multiprocessing_context="spawn", prefetch_factor=1, persistent_workers=True
        )
    loader = DataLoader(dataset, **loader_options)
    source = NativeAnchors(prepared)
    resolved = {
        **audit,
        **gates,
        "condition": args.condition,
        "source": source_identity(),
        "external_initial_weights": external_identity,
        "plan_manifest_sha256": sha(args.plan / "SCHEDULE.json"),
        "schedule_sha256": plan["schedule_sha256"],
        "checkpoint": str(args.checkpoint.resolve()),
        "base_config_sha256": sha(args.checkpoint / "config.json"),
        "global_batch": GLOBAL_BATCH,
        "microbatch": args.microbatch,
        "accumulation": args.accumulation,
        "rank_slots": slots,
        "steps": STEPS,
        "warmup_steps": 250,
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-5,
        "scheduler": "cosine to zero",
        "bf16": True,
        "tf32": True,
        "visual_trainable": True,
        "llm_trainable": False,
        "raw_vlm_cache": False,
        "wall_seconds": args.wall_seconds,
        "deepspeed": config,
        "augmentation_seed": "sha256(42,augmentation,zero_based_step,global_slot)",
        "forward_seed": "sha256(42,forward,zero_based_step,micro,rank)",
    }
    write_new(args.output / f"RESOLVED_RANK{rank}.json", resolved)
    torch.cuda.reset_peak_memory_stats()
    probe_reports = {}

    def probe(step, probe_model):
        # Both ZeRO2 ranks have complete model weights; each evaluates 16 anchors.
        label = "train_probe" if rank == 0 else "seen_probe"
        probe_reports[step] = evaluate_pairs(
            probe_model,
            source,
            anchors[label],
            args.output / "probes" / f"step{step:04d}_{label}",
            wall_seconds=max(0, args.wall_seconds - (time.monotonic() - started)),
        )
        dist.barrier()

    engine.train()
    probe(0, engine.module)
    gradient_report, step_times, losses = {}, [], []
    sample_digest = hashlib.sha256()
    iterator = iter(loader)
    with (args.output / f"train_rank{rank}.jsonl").open("x") as stream:
        for step in range(STEPS):
            if time.monotonic() - started > args.wall_seconds:
                raise TimeoutError("Training wall budget exhausted; no automatic retry")
            tick = time.perf_counter()
            micro_losses, noise_seeds = [], []
            lr = optimizer.param_groups[0]["lr"]
            for micro in range(args.accumulation):
                record = next(iterator)
                if int(record["step"]) != step or int(record["micro"]) != micro:
                    raise ValueError("Dataloader advanced out of frozen schedule order")
                pairs = np.asarray(record["pairs"], dtype=np.int32)
                if not np.array_equal(pairs, schedule[step, slots[micro]]):
                    raise ValueError("Rank samples differ from frozen global order")
                sample_digest.update(pairs.tobytes())
                seed = int(record["noise_seed"])
                set_seed(seed, cuda=True)
                batch = {
                    key: value.cuda(local_rank, non_blocking=True)
                    for key, value in record["inputs"].items()
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = engine(inputs=batch)
                    loss = result["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step{step}/micro{micro}")
                engine.backward(loss)
                if step == 0 and micro == args.accumulation - 1:
                    gradient_report = gradient_checks(parameters)
                    write_new(args.output / f"GRADIENTS_RANK{rank}.json", gradient_report)
                micro_losses.append(float(loss.detach()))
                noise_seeds.append(seed)
                engine.step()
                del result, loss, batch, record
            torch.cuda.synchronize()
            duration = time.perf_counter() - tick
            step_times.append(duration)
            losses.append(float(np.mean(micro_losses)))
            row = {
                "step": step + 1,
                "rank": rank,
                "loss": losses[-1],
                "lr": lr,
                "seconds": duration,
                "noise_seeds": noise_seeds,
                "slots": slots,
                "pairs": [schedule[step, value].tolist() for value in slots],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if (step + 1) % 25 == 0:
                print(
                    json.dumps(
                        {key: row[key] for key in ("step", "rank", "loss", "lr", "seconds")}
                    ),
                    flush=True,
                )
            if step + 1 in PROBE_STEPS:
                probe(step + 1, engine.module)
    del iterator, loader, dataset
    if time.monotonic() - started > args.wall_seconds:
        raise TimeoutError("Training wall budget exhausted before checkpoint save")
    # Fixed final input/seed; verify same-model strict reload on both ranks.
    with preserve_eval_state(engine.module, source.processor), torch.no_grad():
        reload_batch, _ = source.batch([anchors["seen_probe"][0]])
        observation = {
            key: value
            for key, value in reload_batch.items()
            if key not in ("action", "action_mask")
        }
        set_seed(42, cuda=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            before = engine.module.get_action(dict(observation))["action_pred"].float().cpu()
    checkpoint = args.output / "checkpoint-5000"
    if rank == 0:
        engine.module.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
        for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
            shutil.copy2(prepared / "processor" / name, checkpoint / name)
    dist.barrier()
    engine.save_checkpoint(
        str(args.output / "deepspeed"),
        tag="step5000",
        client_state={
            "completed_steps": STEPS,
            "schedule_sha256": plan["schedule_sha256"],
            "source": resolved["source"],
            "condition": args.condition,
            "rng_files": [f"rng_rank{i}.pt" for i in range(2)],
        },
        exclude_frozen_parameters=True,
    )
    torch.save(
        {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        args.output / "deepspeed" / f"rng_rank{rank}.pt",
    )
    peak_allocated, peak_reserved = (
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )
    del engine, model, parameters, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, loading_info = strict_reload(checkpoint)
    reloaded = reloaded.to(device=f"cuda:{local_rank}", dtype=torch.bfloat16).eval()
    set_seed(42, cuda=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = reloaded.get_action(dict(observation))["action_pred"].float().cpu()
    delta = float((before - after).abs().max())
    if delta != 0:
        raise ValueError(f"Same-model strict reload prediction mismatch: {delta}")
    report = {
        "status": "completed",
        "condition": args.condition,
        "steps": STEPS,
        "rank": rank,
        "checkpoint": str(checkpoint.resolve()),
        "deepspeed_state": str((args.output / "deepspeed").resolve()),
        "resumption": "load full HF model first, then DeepSpeed step5000 with load_module_strict=False plus per-rank RNG; no automatic resume",
        "strict_loading_info": loading_info,
        "strict_reload_action_max_abs": delta,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "gradient_checks": gradient_report,
        "sample_order_sha256": sample_digest.hexdigest(),
        "first20_loss_median": float(np.median(losses[:20])),
        "last20_loss_median": float(np.median(losses[-20:])),
        "step_p50_seconds": float(np.median(step_times)),
        "step_p95_seconds": float(np.percentile(step_times, 95)),
        "schedule_sha256": plan["schedule_sha256"],
        "external_initial_weights": external_identity,
    }
    write_new(args.output / f"TRAIN_RESULT_RANK{rank}.json", report)
    dist.barrier()
    if rank == 0:
        write_new(
            args.output / "TRAIN_RESULT.json",
            {
                **report,
                "rank_reports": [str(args.output / f"TRAIN_RESULT_RANK{i}.json") for i in range(2)],
            },
        )
    dist.destroy_process_group()
    print(json.dumps(report), flush=True)


def evaluate(args):
    manifest, prepared, _, anchors = load_plan(args.plan)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, info = strict_reload(args.checkpoint)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    source = NativeAnchors(prepared)
    report = evaluate_pairs(
        model, source, anchors["diagnostic"], args.output, wall_seconds=args.wall_seconds
    )
    write_new(
        args.output / "IDENTITY.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "loading": info,
            "config_sha256": sha(args.checkpoint / "config.json"),
            "source": source_identity(),
            "schedule_sha256": manifest["schedule_sha256"],
            "rows_expected": 512,
        },
    )
    if report["paired_outputs"] != 512:
        raise ValueError("Expected seen512 diagnostic outputs")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(RECIPE["base_checkpoint"]))
    parser.add_argument("--condition", choices=RECIPE["conditions"], default="dit_pretrained")
    parser.add_argument("--microbatch", type=int, choices=(8, 16), default=16)
    parser.add_argument("--accumulation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--workers", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--input-parity", type=Path)
    parser.add_argument("--forward-parity", type=Path)
    parser.add_argument("--wall-seconds", type=int, default=21600)
    parser.add_argument(
        "--zero-config",
        type=Path,
        default=Path("/home/yuqi/Isaac-GR00T-N1.7/gr00t/configs/deepspeed/zero2_config.json"),
    )
    args = parser.parse_args()
    if args.wall_seconds > 21600:
        parser.error("Training/evaluation cannot expand the authorized wall budget")
    if args.command == "prepare" and args.prepared is None:
        parser.error("prepare requires --prepared")
    if args.command != "prepare" and args.plan is None:
        parser.error("train/evaluate requires --plan")
    torch.set_num_threads(4)
    try:
        {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command](args)
    except Exception as exc:
        rank = os.environ.get("RANK", "0")
        failure = args.output / f"FAILURE_RANK{rank}.json"
        if args.output.is_dir() and not failure.exists():
            write_new(
                failure, {"status": "failed", "exception": type(exc).__name__, "message": str(exc)}
            )
        raise


if __name__ == "__main__":
    main()
