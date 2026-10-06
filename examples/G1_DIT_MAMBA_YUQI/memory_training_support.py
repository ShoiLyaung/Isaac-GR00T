"""Thin cache, residual-injection and pairing contracts for the frozen memory plan."""

from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

from contract import tensor_identity
from gr00t.model.modules.causal_visual_memory import inject_image_residual
from memory_prototype import IMAGE_POSITIONS, content_hash
from runner_core import set_seed
import torch
from transformers.feature_extraction_utils import BatchFeature


CONDITIONS = ("current_reset_mamba", "zero_history_mamba", "gru_history", "mamba_history")
RAW_KEYS = ("backbone_features", "backbone_attention_mask", "image_mask")
ACTION_KEYS = ("state", "action", "action_mask", "embodiment_id")


def fresh(values, device):
    return BatchFeature(data={k: v.to(device).clone() for k, v in values.items()})


def selected_prefix(sequence, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown memory condition: {condition}")
    if condition == "current_reset_mamba":
        return sequence[-1:]
    if condition == "zero_history_mamba":
        sequence = sequence.clone()
        sequence[:-1].zero_()  # Before input projection; biased null transitions remain.
    return sequence


def prepare_residual(module, sequence, condition, *, deployment=False):
    current = selected_prefix(sequence, condition)
    if deployment:
        with torch.no_grad():
            state = module.new_state()
            latent = [module.step(value, state)[0] for value in current]
            latent = torch.stack(latent)
    else:
        latent = module.full_sequence(current)
    return latent, module.output_projection(latent[-1]), len(current)


def native_flow(model, frame, residual, seed, *, full_batch=None):
    """Run the original FP32 flow objective, replacing only actual DiT conditioning.

    Capture actual native draws; no noise/time or flow-loss algorithm is copied.
    The residual remains attached to its full-sequence graph.
    """
    head = model.action_head
    captured = {}
    calls = {"time": 0, "noisy": 0, "condition": 0}
    sample_time = head.sample_time

    def capture_time(*args, **kwargs):
        value = sample_time(*args, **kwargs)
        calls["time"] += 1
        captured["continuous_time"] = value.detach().clone()
        return value

    def capture_noisy(_module, args, kwargs):
        calls["noisy"] += 1
        captured["noisy_action"] = args[0].detach().clone()
        captured["discrete_time"] = args[1].detach().clone()

    def condition_hook(_module, args, kwargs):
        calls["condition"] += 1
        actual = kwargs["encoder_hidden_states"]
        effective = (
            actual
            if residual is None
            else inject_image_residual(
                actual, torch.tensor(IMAGE_POSITIONS, device=actual.device), residual
            )
        )
        captured["preinjection_post"] = actual.detach().clone()
        captured["effective_condition"] = effective.detach().clone()
        return args, {**kwargs, "encoder_hidden_states": effective}

    with ExitStack() as context:
        context.enter_context(patch.object(head, "sample_time", capture_time))
        context.callback(
            head.action_encoder.register_forward_pre_hook(capture_noisy, with_kwargs=True).remove
        )
        context.callback(
            head.model.register_forward_pre_hook(condition_hook, with_kwargs=True).remove
        )
        set_seed(seed, cuda=True)  # After memory preparation; both native RNG sources.
        if full_batch is None:
            result = head(
                fresh(frame["raw"], model.device), fresh(frame["action_input"], model.device)
            )
        else:
            result = model(inputs=dict(full_batch))
    if calls != {"time": 1, "noisy": 1, "condition": 1}:
        raise ValueError(f"Native flow capture call count changed: {calls}")
    if result["loss"].dtype != torch.float32 or not bool(torch.isfinite(result["loss"])):
        raise ValueError("Expected native finite FP32 result[loss]")
    receipt = {key: tensor_identity(value) for key, value in captured.items()}
    receipt["continuous_time_values"] = captured["continuous_time"].float().cpu().tolist()
    receipt["discrete_time_values"] = captured["discrete_time"].cpu().tolist()
    receipt["forward_seed"] = seed
    return result, receipt


def sampling_packet(model, frame):
    raw = fresh(frame["raw"], model.device)
    observation = fresh(
        {k: v for k, v in frame["action_input"].items() if k not in ("action", "action_mask")},
        model.device,
    )
    features = model.action_head._encode_features(raw, observation)
    return {"features": features, "backbone": raw, "action": observation}


def frame_hash(frame):
    tensors = {"raw." + k: v for k, v in frame["raw"].items()}
    tensors["post_vl_features"] = frame["post_vl_features"]
    tensors.update({"action." + k: v for k, v in frame["action_input"].items()})
    tensors.update({"physical_state." + k: v for k, v in frame["physical_states"].items()})
    tensors.update({"physical_action." + k: v for k, v in frame["physical_actions"].items()})
    return content_hash(tensors)


def authorize_cache_cohort(plan, cohort, dev_result=None):
    if cohort == "train_dev":
        return plan["train"] + plan["dev"]
    if cohort == "confirmation":
        if (
            dev_result is None
            or dev_result.get("status") != "passed"
            or dev_result.get("continuation_gate") is not True
        ):
            raise ValueError("Confirmation feature access requires a passing frozen dev gate")
        return plan["confirmation"]
    raise ValueError(f"Unsupported cohort: {cohort}")


class FeatureCache:
    """Read-only episode tensors bound to the selected cohort and file identities."""

    def __init__(self, root, plan, plan_hash):
        from contract import sha

        self.root = Path(root)
        self.report = json.loads((self.root / "CACHE_RESULT.json").read_text())
        if self.report["status"] != "passed" or self.report["plan_sha256"] != plan_hash:
            raise ValueError("Missing matching frozen cache result")
        expected = plan["train"] + plan["dev"]
        if self.report["cohort"] != "train_dev" or self.report["episodes"] != expected:
            raise ValueError("First-stage commands only consume train/dev cache")
        self.paths = {}
        for row in self.report["files"]:
            path = self.root / row["file"]
            if path.parent != self.root or sha(path) != row["sha256"]:
                raise ValueError("Cache file identity changed")
            self.paths[row["episode"]] = path
        if set(self.paths) != set(expected):
            raise ValueError("Cache episode inventory changed")
        self.loaded = {}
        self.plan = plan

    def episode(self, episode):
        if episode not in self.paths:
            raise ValueError("Episode is outside the authorized train/dev cache")
        if episode not in self.loaded:
            values = torch.load(self.paths[episode], map_location="cpu", weights_only=True)
            if list(values) != self.plan["grids"][str(episode)]:
                raise ValueError("Cached grid order differs from frozen plan")
            for frame in values.values():
                if frame_hash(frame) != frame["content_sha256"]:
                    raise ValueError("Cached tensor content identity failed")
            self.loaded[episode] = values
        return self.loaded[episode]

    def frame(self, episode, anchor):
        return self.episode(episode)[anchor]

    def prefix(self, episode, anchor, device):
        grid = self.plan["grids"][str(episode)]
        if anchor not in grid:
            raise ValueError("Anchor is outside the legal observation grid")
        indices = grid[: grid.index(anchor) + 1]
        values = [self.frame(episode, a)["post_vl_features"][0, IMAGE_POSITIONS] for a in indices]
        return torch.stack(values).to(device), indices

    def assert_unchanged(self):
        for values in self.loaded.values():
            if any(frame_hash(frame) != frame["content_sha256"] for frame in values.values()):
                raise ValueError("In-memory frozen cache was mutated")


def overfit_gate(before, after):
    old, new = before["metrics"], after["metrics"]
    return {
        "flow_improved_10pct": new["flow_loss"] <= old["flow_loss"] * 0.90,
        "combined_not_worse": new["combined_mae_mixed_units"] <= old["combined_mae_mixed_units"],
        "arm_within_2pct": new["arm_mae_rad"] <= old["arm_mae_rad"] * 1.02,
        "hand_within_2pct": new["hand_mae_u"] <= old["hand_mae_u"] * 1.02,
    }
