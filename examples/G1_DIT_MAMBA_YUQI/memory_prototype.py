"""One bounded, zero-optimizer-step probe of pretrained causal visual memory.

This standalone wrapper never installs a replacement action head. Zero readout
protects the native function; shadow readout tests connectivity, not quality.
The parent research runner owns the only formal GPU attempt and outer timeout.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from contract import CAMERAS, sha, tensor_identity, write_new
from gr00t.model.modules.causal_visual_memory import (
    CACHE_ELEMENTS,
    MAMBA_CONFIG,
    CausalVisualMemory,
    inject_image_residual,
)
import numpy as np
from runner_core import set_seed
import torch


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path(
    "/home/sl/research/icra27-worktrees/Isaac-GR00T/g1-pretrained-causal-mamba-memory-prototype-r1"
)
EXPECTED_VENV = Path("/home/yuqi/Isaac-GR00T-N1.7/.venv")
ARTIFACTS = Path("/home/sl/research/icra27-artifacts")
PRE_RUN = ARTIFACTS / "g1-pretrained-causal-mamba-memory-prototype-r1/PRE_RUN_IDENTITIES.json"
PRE_RUN_SHA = "0570e37475c640a375a687e38a6615fdfd32e1816608bbf8c469398b8b84b132"
REFERENCES = ARTIFACTS / "g1-dit-self-mamba-yuqi-r1/REFERENCE_IDENTITIES.json"
LAYOUT = ARTIFACTS / "g1-mamba-independent-condition-cost-r1/INPUT_LAYOUT_IDENTITY.json"
LAYOUT_SHA = "3d69fd87d534fe2760f26b677de26c03fb2255ca470bde71d2887c1efc539233"
STREAMS = {
    150: [13, 29, 45, 61, 77, 93],
    147: [13, 29, 45, 61],
    152: [0, 16, 32],
    112: [72, 88, 104, 120, 136, 152],
}
IMAGE_POSITIONS = list(range(4, 68)) + list(range(70, 134)) + list(range(136, 200))
NONIMAGE_POSITIONS = [0, 1, 2, 3, 68, 69, 134, 135, 200, 201, 202, 203, 204, 205, 206, 207, 208]
NONIMAGE_IDS = [
    151644,
    872,
    198,
    151652,
    151653,
    151652,
    151653,
    151652,
    151653,
    4208,
    279,
    30368,
    1119,
    279,
    11721,
    151645,
    198,
]
MAX_BYTES, MAX_ALLOCATED = 2 * 2**30, 44 * 2**30
WARMUP, MEASURED = 5, 50


def read(path):
    return json.loads(Path(path).read_text())


def check_budget(deadline, output, *, cuda=True):
    if time.monotonic() > deadline:
        raise TimeoutError("Memory prototype exceeded its 1750-second internal wall budget")
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size > MAX_BYTES:
        raise RuntimeError("Memory prototype exceeded 2 GiB artifact budget")
    if cuda and torch.cuda.max_memory_allocated() > MAX_ALLOCATED:
        raise RuntimeError("Memory prototype exceeded 44 GiB allocated GPU memory")
    return size


def content_hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def state_hash(state):
    return content_hash(
        {f"{i}.{j}": t for i, pair in enumerate(state.layers) for j, t in enumerate(pair)}
    )


def assert_frozen(model):
    if any(m.training for m in model.modules()):
        raise ValueError("Every pretrained module must remain in eval mode")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Pretrained parameters must be frozen and have no gradients")


def validate_input_layout(observation, cameras):
    if tuple(cameras) != CAMERAS:
        raise ValueError("Current RGB camera identity/order changed")
    expected = torch.full((1, 209), 151655, dtype=torch.long)
    expected[0, NONIMAGE_POSITIONS] = torch.tensor(NONIMAGE_IDS)
    if not torch.equal(observation["input_ids"].cpu(), expected):
        raise ValueError("Frozen full 209-token image/language layout changed")
    if not torch.equal(observation["attention_mask"].cpu(), torch.ones_like(expected)):
        raise ValueError("Frozen attention mask changed")
    if not torch.equal(observation["image_grid_thw"].cpu(), torch.tensor([[1, 16, 16]] * 3)):
        raise ValueError("Expected three ordered 64-slot image grids")
    if tuple(observation["state"].shape) != (1, 1, 13):
        raise ValueError("Expected current-only q13 state")
    if "action" in observation or "action_mask" in observation:
        raise ValueError("Memory inference must not receive targets or RTC actions")


def encode_observation(model, observation):
    backbone_inputs, action_input = model.prepare_input(dict(observation))
    backbone_output = model.backbone(backbone_inputs)
    features = model.action_head._encode_features(backbone_output, action_input)
    return {"features": features, "backbone": backbone_output, "action": action_input}


def validate_packet(packet, observation):
    backbone, features = packet["backbone"], packet["features"]
    if tuple(features.backbone_features.shape) != (1, 209, 2048):
        raise ValueError("Expected native post-VL [1,209,2048] features")
    if not torch.equal(backbone.image_mask.cpu(), observation["input_ids"] == 151655):
        raise ValueError("Backbone image mask does not match frozen positions")
    if not torch.equal(backbone.backbone_attention_mask.cpu(), observation["attention_mask"] == 1):
        raise ValueError("Backbone attention mask changed after native VL postprocessing")
    if (
        backbone.image_mask.dtype != torch.bool
        or backbone.backbone_attention_mask.dtype != torch.bool
    ):
        raise ValueError("Expected native boolean masks")
    if not bool(torch.isfinite(features.backbone_features).all()):
        raise ValueError("Nonfinite native post-VL features")


def packet_to(packet, device):
    from transformers.feature_extraction_utils import BatchFeature

    return {
        name: BatchFeature(data={k: v.detach().to(device) for k, v in values.items()})
        for name, values in packet.items()
    }


def slots(packet):
    value = packet["features"].backbone_features
    return value[0].index_select(0, torch.tensor(IMAGE_POSITIONS, device=value.device))


def sample_packet(model, packet, residual=None, *, gradient_body=False):
    head = model.action_head
    features, backbone, action = packet["features"], packet["backbone"], packet["action"]
    if "action" in action or "action_mask" in action:
        raise ValueError("Sampling forbids teacher/RTC action inputs")
    value = features.backbone_features
    if residual is not None:
        value = inject_image_residual(
            value, torch.tensor(IMAGE_POSITIONS, device=value.device), residual
        )
    kwargs = dict(
        backbone_features=value,
        state_features=features.state_features,
        embodiment_id=action.embodiment_id,
        backbone_output=backbone,
        action_input=action,
        options=None,
    )
    if gradient_body:
        # enable_grad cannot override the public method's @no_grad decorator.
        result = type(head).get_action_with_features.__wrapped__(head, **kwargs)
    else:
        result = head.get_action_with_features(**kwargs)
    return result["action_pred"]


@torch.no_grad()
def policy_with_memory(model, memory, state, observation, *, enabled=True, current_reset=False):
    if not enabled:
        return model.get_action(dict(observation), options=None)["action_pred"]
    if current_reset:
        state.reset()
    packet = encode_observation(model, observation)
    _, residual = memory.step(slots(packet), state)
    return sample_packet(model, packet, residual)


def action_record(predicted, raw, processor, expected=None):
    from gr00t.data.embodiment_tags import EmbodimentTag

    if tuple(predicted.shape) != (1, 40, 13) or not bool(torch.isfinite(predicted).all()):
        raise ValueError("Expected finite native internal (1,40,13) action")
    value = predicted.detach().cpu()
    if expected is not None and not torch.equal(value, expected):
        raise ValueError("Native/disabled/zero-readout exact action parity failed")
    decoded = processor.decode_action(
        value[0].float().numpy(), EmbodimentTag.NEW_EMBODIMENT, state=raw.states
    )
    physical = np.concatenate([decoded[k] for k in ("left_arm", "left_hand")], axis=-1)
    if physical.shape != (16, 13) or not np.isfinite(physical).all():
        raise ValueError("Expected finite native (16,13) decoded action")
    return value, {
        "action_identity": tensor_identity(value),
        "decoded_shape": [16, 13],
        "decoded_sha256": hashlib.sha256(physical.tobytes()).hexdigest(),
        "finite": True,
    }


def differences(actual, reference):
    delta = actual.float() - reference.float()
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("Nonfinite numerical comparison")
    rms = float(delta.square().mean().sqrt())
    scale = float(reference.float().square().mean().sqrt())
    return {"rms": rms, "relative_rms": rms / max(scale, 1e-12), "maxabs": float(delta.abs().max())}


def check_tolerance(report, *, action=False):
    if action:
        passed = report["rms"] <= 0.002 and report["maxabs"] <= 0.02
    else:
        passed = report["relative_rms"] <= 0.02 and report["maxabs"] <= 0.1
    if not passed:
        raise ValueError(
            f"Frozen {'action' if action else 'latent'} numerical tolerance failed: {report}"
        )


@torch.no_grad()
def step_sequence(memory, sequence, state=None):
    state = memory.new_state() if state is None else state
    latent = [memory.step(current, state)[0] for current in sequence]
    return torch.stack(latent), state


def verify_environment(historical_root):
    import gr00t
    from gr00t.model.gr00t_n1d7 import gr00t_n1d7
    from gr00t.model.modules import causal_visual_memory
    import mamba_ssm.modules.mamba2 as official
    import run_train_eval
    import self_mamba_preflight

    modules = (gr00t, gr00t_n1d7, causal_visual_memory, run_train_eval, self_mamba_preflight)
    if REPO != EXPECTED_REPO or any(
        REPO not in Path(m.__file__).resolve().parents for m in modules
    ):
        raise ValueError("A model/helper import escaped this prototype worktree")
    if sys.version_info[:2] != (3, 10) or Path(sys.prefix) != EXPECTED_VENV:
        raise ValueError("Expected original Python 3.10 environment")
    if (
        torch.__version__.split("+")[0] != "2.7.1"
        or EXPECTED_VENV not in Path(torch.__file__).resolve().parents
    ):
        raise ValueError("Expected original torch 2.7.1 environment")
    overlay = (historical_root / "dependencies/python-overlay").resolve()
    if (
        overlay not in Path(official.__file__).resolve().parents
        or official.causal_conv1d_fn is not None
    ):
        raise ValueError("Expected original Mamba overlay without causal-conv1d")
    if importlib.metadata.version("mamba-ssm") != "2.3.2.post1":
        raise ValueError("Expected frozen mamba-ssm 2.3.2.post1")
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Checkpoint weights may not be skipped")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty:
        raise ValueError(f"Prototype requires clean committed source: {dirty}")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "dirty": dirty,
        "imports": {m.__name__: m.__file__ for m in modules},
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "mamba_file": official.__file__,
        "mamba_version": importlib.metadata.version("mamba-ssm"),
        "runner_sha256": sha(__file__),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }


def verify_frozen_inputs(args):
    if sha(PRE_RUN) != PRE_RUN_SHA or sha(LAYOUT) != LAYOUT_SHA:
        raise ValueError("Frozen pre-run or full input layout identity changed")
    frozen, layout, references = read(PRE_RUN), read(LAYOUT), read(REFERENCES)
    reference = references["checkpoints"]["frozen_reference"]
    if frozen["status"] != "passed" or references["status"] != "passed":
        raise ValueError("Missing passing input/checkpoint identity")
    if (
        args.checkpoint.resolve() != Path(reference["checkpoint"])
        or frozen["base_checkpoint"] != reference["checkpoint"]
    ):
        raise ValueError(
            "Expected original Yuqi17000, not native5000 or an action-head Mamba checkpoint"
        )
    if {int(k): v for k, v in frozen["streams"].items()} != STREAMS:
        raise ValueError("Frozen observation streams changed")
    # PRE_RUN already binds checkpoint, audited videos/parquets and processor files.
    # Recheck each exact file before the first model output; hash duplicate paths once.
    verified = {}
    for row in frozen["source_files"] + frozen["records"]:
        path = Path(row["path"])
        if str(path) not in verified:
            actual = {"bytes": path.stat().st_size, "sha256": sha(path)}
            if actual != {key: row[key] for key in actual}:
                raise ValueError(f"Frozen input changed: {path}")
            verified[str(path)] = actual
    for name, expected in reference["files"].items():
        if verified.get(str(args.checkpoint / name)) != expected:
            raise ValueError(f"Checkpoint identity not bound by pre-run record: {name}")
    if layout["image_positions"] != IMAGE_POSITIONS or layout["cameras"] != list(CAMERAS):
        raise ValueError("Prior image-slot identity does not match this prototype")
    return {
        "pre_run_path": str(PRE_RUN),
        "pre_run_sha256": PRE_RUN_SHA,
        "reference_identity_sha256": sha(REFERENCES),
        "checkpoint": reference,
        "layout_identity_path": str(LAYOUT),
        "layout_identity_sha256": LAYOUT_SHA,
        "verified_file_count": len(verified),
    }


def freeze_observations(source, output):
    streams, identities = [], []
    for episode, anchors in STREAMS.items():
        records = []
        for anchor in anchors:
            batch, raw = source.batch([(episode, anchor)])
            observation = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}
            validate_input_layout(observation, source.modality["video"].modality_keys)
            identity = {k: tensor_identity(v) for k, v in observation.items()}
            records.append(
                {
                    "episode": episode,
                    "anchor": anchor,
                    "observation": observation,
                    "raw": raw[0],
                    "identity": identity,
                }
            )
            identities.append({"episode": episode, "anchor": anchor, "tensors": identity})
        streams.append(records)
    write_new(
        output / "INPUTS.json",
        {
            "status": "passed",
            "observations": identities,
            "image_positions": IMAGE_POSITIONS,
            "cameras": CAMERAS,
            "scope": "Bounded suffix windows; stride16 observation frames, no future/target memory input",
        },
    )
    return streams


def zero_recovery(model, memory, source, streams, output, deadline):
    reports, packets = [], []
    for records in streams:
        state, encoded_stream = memory.new_state(), []
        for row in records:
            check_budget(deadline, output)
            obs = row["observation"]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                set_seed(42, cuda=True)
                native = model.get_action(dict(obs))["action_pred"]
                expected, native_record = action_record(native, row["raw"], source.processor)
                versions = [v._version for pair in state.layers for v in pair]
                before_steps = state.observations
                set_seed(42, cuda=True)
                disabled = policy_with_memory(model, memory, state, obs, enabled=False)
                action_record(disabled, row["raw"], source.processor, expected)
                if state.observations != before_steps or versions != [
                    v._version for pair in state.layers for v in pair
                ]:
                    raise ValueError("Disable touched temporal state")
                set_seed(42, cuda=True)
                enabled = policy_with_memory(model, memory, state, obs)
                action_record(enabled, row["raw"], source.processor, expected)
                if state.observations != before_steps + 1:
                    raise ValueError("Memory must update exactly once per observation")
                packet = encode_observation(model, obs)
                validate_packet(packet, obs)
                encoded_stream.append(packet_to(packet, "cpu"))
                reports.append(
                    {
                        "episode": row["episode"],
                        "anchor": row["anchor"],
                        **native_record,
                        "native_disabled_enabled_exact": True,
                        "memory_observations": state.observations,
                        "post_vl_identity": tensor_identity(packet["features"].backbone_features),
                    }
                )
            if row["identity"] != {k: tensor_identity(v) for k, v in obs.items()}:
                raise ValueError("Input contents changed during recovery check")
        packets.append(encoded_stream)
    write_new(
        output / "ZERO_RECOVERY.json",
        {"status": "passed", "rows": reports, "observations": len(reports)},
    )
    return packets


def make_shadow(memory):
    shadow = deepcopy(memory)
    generator = torch.Generator(device="cpu").manual_seed(43)
    weight = (
        torch.randn(shadow.output_projection.weight.shape, generator=generator, dtype=torch.float32)
        * 1e-3
    )
    with torch.no_grad():
        shadow.output_projection.weight.copy_(weight)
        shadow.output_projection.bias.zero_()
    return shadow


def check_memory_streams(model, memory, shadow, packets, streams, source, output, deadline):
    reports = []
    for index, encoded in enumerate(packets):
        check_budget(deadline, output)
        sequence = torch.stack([slots(p) for p in encoded]).cuda()
        packet = packet_to(encoded[-1], "cuda")
        row = streams[index][-1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            step_latent, state = step_sequence(memory, sequence)
            whole = memory.full_sequence(sequence)
            numerical = differences(step_latent, whole)
            check_tolerance(numerical)
            split = len(sequence) // 2
            left, carried = step_sequence(memory, sequence[:split])
            right, carried = step_sequence(memory, sequence[split:], carried)
            if not torch.equal(torch.cat([left, right]), step_latent) or state_hash(
                carried
            ) != state_hash(state):
                raise ValueError("Chunk boundary changed carried-cache results")
            main_hash = state_hash(state)
            cloned = state.clone()
            for original_pair, clone_pair in zip(state.layers, cloned.layers):
                if any(
                    a.data_ptr() == b.data_ptr() or a.stride() != b.stride()
                    for a, b in zip(original_pair, clone_pair)
                ):
                    raise ValueError(
                        "Shadow state clone aliases storage or changes official stride"
                    )
            memory.step(sequence[0], cloned)
            if state_hash(state) != main_hash:
                raise ValueError("Shadow cache update changed main cache")
            # Reset a previously used cache, then compare the next episode's first observation.
            next_sequence = packets[(index + 1) % len(packets)]
            next_current = slots(next_sequence[0]).cuda()
            state.reset()
            reset_latent, _ = memory.step(next_current, state)
            fresh = memory.new_state()
            fresh_latent, _ = memory.step(next_current, fresh)
            if not torch.equal(reset_latent, fresh_latent) or state_hash(state) != state_hash(
                fresh
            ):
                raise ValueError("Cross-episode reset differs from fresh state")
            before_read = state_hash(carried)
            set_seed(42, cuda=True)
            first = sample_packet(model, packet, memory.output_projection(step_latent[-1]))
            set_seed(42, cuda=True)
            again = sample_packet(model, packet, memory.output_projection(step_latent[-1]))
            if not torch.equal(first, again) or before_read != state_hash(carried):
                raise ValueError("Snapshot/action reads changed state or action")
            set_seed(42, cuda=True)
            step_action = sample_packet(model, packet, shadow.output_projection(step_latent[-1]))
            set_seed(42, cuda=True)
            forward_action = sample_packet(model, packet, shadow.output_projection(whole[-1]))
            _, step_action_record = action_record(step_action, row["raw"], source.processor)
            _, forward_action_record = action_record(forward_action, row["raw"], source.processor)
            action_delta = differences(step_action[:, :16], forward_action[:, :16])
            check_tolerance(action_delta, action=True)
            changed_current = sequence.clone()
            changed_current[-1].zero_()
            changed_history = sequence.clone()
            changed_history[:-1].zero_()
            current_delta = differences(shadow.full_sequence(changed_current)[-1], whole[-1])
            history_delta = differences(shadow.full_sequence(changed_history)[-1], whole[-1])
            connection = differences(forward_action[:, :16], first[:, :16])
            if min(current_delta["rms"], history_delta["rms"], connection["rms"]) <= 0:
                raise ValueError("Nonzero shadow failed current/history/action connectivity")
            row = streams[index][-1]
            state.reset()
            set_seed(42, cuda=True)
            reset_action = policy_with_memory(
                model, memory, state, row["observation"], current_reset=True
            )
            if state.observations != 1 or not torch.equal(reset_action, first):
                raise ValueError(
                    "Current-reset must execute one memory step and retain zero-init recovery"
                )
        reports.append(
            {
                "episode": streams[index][0]["episode"],
                "observations": len(encoded),
                "step_full_latent": numerical,
                "shadow_step_full_active_action": action_delta,
                "shadow_step_action": step_action_record,
                "shadow_forward_action": forward_action_record,
                "chunk_carry_exact": True,
                "reset_fresh_exact": True,
                "clone_independent": True,
                "snapshot_read_immutable": True,
                "current_reset_updates_once": True,
                "shadow_current_intervention": current_delta,
                "shadow_history_intervention": history_delta,
                "shadow_residual_action_connection": connection,
            }
        )
    write_new(
        output / "MEMORY_SEMANTICS.json",
        {
            "status": "passed",
            "streams": reports,
            "scope": "Engineering connectivity and fixed numerical tolerances; no random-module quality claims",
        },
    )
    return reports


def gradient_group(module):
    rows = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"Nonfinite gradient: {name}")
        rows.append(
            {
                "name": name,
                "nonzero": bool(torch.any(gradient != 0)),
                "norm": float(gradient.float().norm()),
            }
        )
    return {
        "parameters_with_gradient": len(rows),
        "finite_nonzero_parameters": sum(r["nonzero"] for r in rows),
        "parameters": rows,
    }


def gradient_dry_run(model, memory, shadow, encoded, output, deadline):
    check_budget(deadline, output)
    packet = packet_to(encoded[-1], "cuda")
    sequence = torch.stack([slots(p) for p in encoded]).cuda()
    residual = torch.zeros_like(slots(packet), requires_grad=True)
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        set_seed(42, cuda=True)
        action = sample_packet(model, packet, residual, gradient_body=True)
        loss = action[:, :16].float().square().mean()
        loss.backward()
    if (
        residual.grad is None
        or not bool(torch.isfinite(residual.grad).all())
        or not bool(torch.any(residual.grad != 0))
    ):
        raise ValueError("Residual leaf has no finite nonzero gradient through frozen native DiT")
    leaf_report = {
        "loss": float(loss.detach()),
        "gradient_norm": float(residual.grad.float().norm()),
        "finite_nonzero": True,
    }
    del residual, action, loss
    reports = {}
    for label, module in [("zero_initialization", memory), ("nonzero_shadow", shadow)]:
        module.zero_grad(set_to_none=True)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = module.full_sequence(sequence)
            readout = module.output_projection(latent[-1])
            set_seed(42, cuda=True)
            action = sample_packet(model, packet, readout, gradient_body=True)
            loss = action[:, :16].float().square().mean()
            loss.backward()
        groups = {
            "input_projection": gradient_group(module.input_projection),
            "mamba0": gradient_group(module.layers[0].mamba),
            "mamba1": gradient_group(module.layers[1].mamba),
            "output_projection": gradient_group(module.output_projection),
        }
        required = ["output_projection"] if label == "zero_initialization" else list(groups)
        if any(groups[name]["finite_nonzero_parameters"] == 0 for name in required):
            raise ValueError(f"Full-sequence gradient gate failed: {label}")
        reports[label] = {
            "loss": float(loss.detach()),
            "groups": groups,
            "path": "official full_sequence SSD backward; never inference step",
        }
        module.zero_grad(set_to_none=True)
        assert_frozen(model)
        check_budget(deadline, output)
        del latent, readout, action, loss
    report = {
        "status": "passed",
        "optimizer_steps": 0,
        "residual_leaf": leaf_report,
        "module_gradients": reports,
        "native_sampling_body": "type(head).get_action_with_features.__wrapped__",
        "loss": "mean(square(active16 normalized sampled actions))",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_new(output / "GRADIENT_DRY_RUN.json", report)
    return report


def count_real_calls(model, memory, row, source, output):
    counts = {"mamba_steps": [0, 0], "dit_forward": 0}
    order = []
    state = memory.new_state()

    def counted(index, original):
        def call(*args, **kwargs):
            counts["mamba_steps"][index] += 1
            order.append(f"mamba{index}")
            return original(*args, **kwargs)

        return call

    def count_dit(*args):
        counts["dit_forward"] += 1
        order.append("dit")

    with ExitStack() as stack:
        for index, layer in enumerate(memory.layers):
            stack.enter_context(patch.object(layer.mamba, "step", counted(index, layer.mamba.step)))
        handle = model.action_head.model.register_forward_pre_hook(count_dit)
        stack.callback(handle.remove)
        set_seed(42, cuda=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            instrumented = policy_with_memory(model, memory, state, row["observation"])
    if (
        counts != {"mamba_steps": [1, 1], "dit_forward": 4}
        or state.observations != 1
        or order != ["mamba0", "mamba1", "dit", "dit", "dit", "dit"]
    ):
        raise ValueError(f"Expected two real Mamba steps before four DiT evaluations: {counts}")
    set_seed(42, cuda=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = model.get_action(dict(row["observation"]))["action_pred"]
    action_record(instrumented, row["raw"], source.processor, clean.cpu())
    write_new(
        output / "CALL_COUNTS.json",
        {
            "status": "passed",
            "counts": counts,
            "order": order,
            "hooked_clean_action_exact": True,
            "instrumentation_removed_before_cost": True,
        },
    )


@torch.no_grad()
def synthetic_state_check(memory, current, output, deadline):
    state = memory.new_state()
    rows = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range(1, 301):
            check_budget(deadline, output)
            latent, residual = memory.step(current, state)
            if latent.requires_grad or residual.requires_grad:
                raise ValueError("Inference step retained an autograd graph")
            del latent, residual
            if step in (1, 150, 300):
                torch.cuda.synchronize()
                description = state.describe()
                if description["elements"] != CACHE_ELEMENTS or state.observations != step:
                    raise ValueError("Synthetic recurrent state grew with history length")
                if any(not bool(torch.isfinite(t).all()) for pair in state.layers for t in pair):
                    raise ValueError("Nonfinite synthetic memory state")
                rows.append(
                    {
                        "step": step,
                        "state": description,
                        "allocated_bytes": torch.cuda.memory_allocated(),
                    }
                )
    growth = rows[-1]["allocated_bytes"] - rows[-2]["allocated_bytes"]
    if growth > 2**20:
        raise ValueError(f"Second-half synthetic allocated growth exceeds 1 MiB: {growth}")
    state.reset()
    write_new(
        output / "SYNTHETIC_STATE.json",
        {
            "status": "passed",
            "steps": 300,
            "checkpoints": rows,
            "second_half_allocated_growth_bytes": growth,
            "state_cleared": True,
            "scope": "Repeated fixed feature; not a robot trajectory or policy-cost sample",
        },
    )


def timing_stats(rows):
    values = np.asarray([r["wall_ms"] for r in rows if not r["warmup"]], dtype=np.float64)
    if len(rows) != 55 or len(values) != 50 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Expected five warmups and fifty positive finite timings")
    return {
        "count": 50,
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
    }


def cost_gate(native, enabled):
    delta = enabled["peak_allocated_bytes"] - native["peak_allocated_bytes"]
    return {
        "latency_passed": enabled["p95_ms"] <= native["p95_ms"] * 1.10,
        "extra_peak_allocated_passed": delta <= 512 * 2**20,
        "absolute_allocated_passed": max(
            native["peak_allocated_bytes"], enabled["peak_allocated_bytes"]
        )
        <= MAX_ALLOCATED,
        "enabled_p95_relative_difference": enabled["p95_ms"] / native["p95_ms"] - 1,
        "extra_peak_allocated_bytes": delta,
    }


def cost_input(model, memory, source, records, index, output, deadline):
    order = ["native", "enabled"] if index % 2 == 0 else ["enabled", "native"]
    last, all_rows, prefill_report = records[-1], [], None
    expected = None
    for mode in order:
        memory.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        prefix = state = None
        if mode == "enabled":
            memory.to("cuda")
            prefix = memory.new_state()
            set_seed(42, cuda=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                for history in records[:-1]:
                    packet = encode_observation(model, history["observation"])
                    memory.step(slots(packet), prefix)
                torch.cuda.synchronize()
                prefill_ms = (time.perf_counter() - tick) * 1000
            del packet
            prefix_hash = state_hash(prefix)
            prefill_report = {
                "observations": len(records) - 1,
                "wall_ms": prefill_ms,
                "scope": "Full H2D/VLM/VL/memory prefix work; excludes action sampling",
                "state": prefix.describe(),
                "state_sha256": prefix_hash,
            }
            write_new(output / f"input-{index}-prefill.json", prefill_report)
        elif any(p.is_cuda for p in memory.parameters()):
            raise ValueError("Native baseline must not retain unused memory weights on GPU")
        with (output / f"input-{index}-{mode}.jsonl").open("x") as stream:
            for iteration in range(WARMUP + MEASURED):
                check_budget(deadline, output)
                if prefix is not None:
                    state = prefix.clone()
                set_seed(42, cuda=True)
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    if mode == "native":
                        predicted = model.get_action(dict(last["observation"]))["action_pred"]
                    else:
                        predicted = policy_with_memory(model, memory, state, last["observation"])
                    torch.cuda.synchronize()
                    wall_ms = (time.perf_counter() - tick) * 1000
                allocated, reserved = (
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                value, action = action_record(predicted, last["raw"], source.processor, expected)
                if expected is None:
                    expected = value.clone()
                if state is not None and state.observations != len(records):
                    raise ValueError(
                        "Timed policy must add one current observation to restored prefix"
                    )
                row = {
                    "path": mode,
                    "iteration": iteration,
                    "warmup": iteration < WARMUP,
                    "rng_seed": 42,
                    "episode": last["episode"],
                    "anchor": last["anchor"],
                    "wall_ms": wall_ms,
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "gpu_state_copies": 2 if prefix is not None else 0,
                    "exact_native_parity": True,
                    **action,
                }
                all_rows.append(row)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                check_budget(deadline, output)
                del predicted, value
                state = None
        if prefix is not None and state_hash(prefix) != prefix_hash:
            raise ValueError("Repeated-call restoration mutated the shared prefix snapshot")
        prefix = state = None
    memory.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    summaries = {
        mode: timing_stats([r for r in all_rows if r["path"] == mode])
        for mode in ("native", "enabled")
    }
    gate = cost_gate(summaries["native"], summaries["enabled"])
    report = {
        "episode": last["episode"],
        "anchor": last["anchor"],
        "order": order,
        "prefill": prefill_report,
        "timings": summaries,
        "gate": gate,
        "boundary": "Current H2D/VLM/VL/memory/Euler4 included; seed/prefix restore/decode/checks excluded",
    }
    write_new(output / f"input-{index}-COST.json", report)
    if not all(
        gate[k]
        for k in ("latency_passed", "extra_peak_allocated_passed", "absolute_allocated_passed")
    ):
        raise ValueError(f"Frozen memory cost gate failed on input {index}: {gate}")
    return report


def run(args, deadline):
    from run_train_eval import NativeAnchors, evaluate_pairs, load_plan, strict_reload
    from self_mamba_preflight import exact_historical_parity

    environment = verify_environment(args.historical_root)
    frozen = verify_frozen_inputs(args)
    check_budget(deadline, args.output, cuda=False)
    plan, prepared, _, anchors = load_plan(args.plan)
    source = NativeAnchors(prepared)
    source.processor.eval()
    streams = freeze_observations(source, args.output)
    write_new(
        args.output / "IDENTITY.json",
        {
            "environment": environment,
            "frozen_inputs": frozen,
            "schedule_sha256": plan["schedule_sha256"],
            "prepared_manifest_sha256": plan["prepared_manifest_sha256"],
            "precision": "base BF16/autocast TF32; new master weights and official caches FP32",
            "streams": STREAMS,
            "optimizer_steps": 0,
        },
    )
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose only the authorized GPU0")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, loading = strict_reload(args.checkpoint)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    assert_frozen(model)
    if (
        model.backbone.model.config.image_token_id != 151655
        or model.action_head.num_inference_timesteps != 4
    ):
        raise ValueError("Native image-token/Euler4 contract changed")
    base_before = content_hash(model.state_dict())
    memory = CausalVisualMemory().eval()
    memory_before = content_hash(memory.state_dict())
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in memory.parameters()):
        raise ValueError("New memory must initialize on CPU with FP32 master weights")
    write_new(
        args.output / "INITIAL_WEIGHTS.json",
        {"base_sha256": base_before, "memory_sha256": memory_before},
    )
    torch.save(memory.state_dict(), args.output / "memory-initialization.pt")
    write_new(
        args.output / "MEMORY_CONFIG.json",
        {
            "mamba": MAMBA_CONFIG,
            "layers": 2,
            "width": 256,
            "slots": 192,
            "input_dim": 2048,
            "output_projection_zero": True,
            "initialization_seed": 42,
            "shadow_seed": 43,
            "dt_limit": [0, "infinity"],
            "parameters": sum(p.numel() for p in memory.parameters()),
            "master_dtype": "float32",
            "initialization_sha256": sha(args.output / "memory-initialization.pt"),
            "initial_content_sha256": memory_before,
        },
    )
    check_budget(deadline, args.output)
    evaluate_pairs(
        model,
        source,
        anchors["seen_probe"],
        args.output / "historical-parity",
        noise_seeds=(42, 43, 44, 45),
        wall_seconds=max(0, deadline - time.monotonic()),
    )
    parity = exact_historical_parity(
        args.output / "historical-parity",
        args.historical_root / "evaluate-frozen_reference-r1/evaluate",
    )
    parity["strict_loading"] = loading
    write_new(args.output / "REFERENCE_PARITY.json", parity)
    memory.to("cuda")
    cache = memory.new_state()
    write_new(args.output / "CACHE_LAYOUT.json", cache.describe())
    del cache
    packets = zero_recovery(model, memory, source, streams, args.output, deadline)
    shadow = make_shadow(memory)
    semantics = check_memory_streams(
        model, memory, shadow, packets, streams, source, args.output, deadline
    )
    gradients = gradient_dry_run(model, memory, shadow, packets[0], args.output, deadline)
    del shadow
    memory.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    # Original initialization is restored (it was never overwritten or optimized).
    if content_hash(memory.state_dict()) != memory_before:
        raise ValueError("Gradient/shadow checks changed initial memory parameters")
    restored = []
    for records in streams:
        state = memory.new_state()
        row = records[-1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            set_seed(42, cuda=True)
            native = model.get_action(dict(row["observation"]))["action_pred"]
            set_seed(42, cuda=True)
            enabled = policy_with_memory(model, memory, state, row["observation"])
            _, record = action_record(enabled, row["raw"], source.processor, native.cpu())
        restored.append(
            {"episode": row["episode"], "anchor": row["anchor"], **record, "native_exact": True}
        )
        del state, native, enabled
    write_new(args.output / "RESTORED_PARITY.json", {"status": "passed", "rows": restored})
    count_real_calls(model, memory, streams[0][-1], source, args.output)
    synthetic_state_check(memory, slots(packets[0][0]).cuda(), args.output, deadline)
    assert_frozen(model)
    check_budget(deadline, args.output)
    base_after_technical = content_hash(model.state_dict())
    memory_after_technical = content_hash(memory.state_dict())
    if base_after_technical != base_before or memory_after_technical != memory_before:
        raise ValueError("Technical probes changed frozen base or initial memory contents")
    write_new(
        args.output / "TECHNICAL_WEIGHT_INTEGRITY.json",
        {
            "status": "passed",
            "base_before": base_before,
            "base_after": base_after_technical,
            "memory_before": memory_before,
            "memory_after": memory_after_technical,
        },
    )
    write_new(
        args.output / "TECHNICAL_RESULT.json",
        {
            "status": "passed",
            "historical_outputs": 64,
            "zero_recovery_observations": 19,
            "memory_streams": semantics,
            "gradient_path": gradients["module_gradients"]["nonzero_shadow"]["path"],
            "optimizer_steps": 0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    costs = []
    for index, records in enumerate(streams):
        costs.append(cost_input(model, memory, source, records, index, args.output, deadline))
        print(json.dumps({"cost_input_completed": index}), flush=True)
    assert_frozen(model)
    base_after = content_hash(model.state_dict())
    memory_after = content_hash(memory.state_dict())
    if base_after != base_before or memory_after != memory_before:
        raise ValueError("Fixed pretrained or initial memory parameter content changed")
    if any(p.dtype != torch.float32 for p in memory.parameters()):
        raise ValueError("New memory master weights must remain FP32")
    size = check_budget(deadline, args.output)
    return {
        "status": "passed",
        "technical_validity": "passed",
        "cost_gate": "passed",
        "optimizer_steps": 0,
        "scientific_outcome": "not_established_no_training",
        "claim_scope": "Engineering prototype only; no memory quality, long-horizon or training success claim",
        "reference_parity": parity,
        "costs": costs,
        "base_sha256_before": base_before,
        "base_sha256_after": base_after,
        "memory_sha256_before": memory_before,
        "memory_sha256_after": memory_after,
        "artifact_bytes_before_result": size,
        "measured_policy_calls": 400,
        "warmup_policy_calls": 40,
        "gpu": torch.cuda.get_device_name(0),
    }


def execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        report = run(args, started + args.wall_seconds)
        report["elapsed_seconds"] = time.monotonic() - started
        write_new(args.output / "RESULT.json", report)
    except Exception as exc:
        technical = args.output / "TECHNICAL_RESULT.json"
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "exception": type(exc).__name__,
                "message": str(exc),
                "technical_validity": "passed"
                if technical.exists() and read(technical)["status"] == "passed"
                else "not_established",
                "scientific_outcome": "not_established_no_training",
                "optimizer_steps": 0,
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "checkpoint", "historical-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=1750)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1750:
        parser.error("Wall budget must be in (0,1750] seconds")
    execute(args)


if __name__ == "__main__":
    main()
