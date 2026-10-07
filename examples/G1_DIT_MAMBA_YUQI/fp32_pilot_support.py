"""Concrete pairing, cache and numerical contracts for one FP32 memory pilot."""

import json
from pathlib import Path

from contract import sha, write_new
from fp32_memory_overfit import sample_cached
from gr00t.model.modules.causal_visual_gru import CausalVisualGRU
from gr00t.model.modules.causal_visual_memory import CausalVisualMemory
from gr00t.model.modules.fp32_visual_memory import FP32VisualMemory, explicit_cache_values
from memory_numerical_diagnosis import old_gate
from memory_numerical_trace import error_summary
import memory_prototype as proto
from memory_training import raw_record
from memory_training_support import CONDITIONS, frame_hash, selected_prefix
from runner_core import deterministic_seed
import torch


INITIAL_SHA = "08469967ccb48df87bd2094a9aa7a6a9fb6a144bac8b5d7a2752eeaaf90b9c55"
GRU_INITIAL_SHA = "3e7a86c21afadbe0b6b76ba209aa050cde3fd3576fff3b7b4bce073005661a07"
SCHEDULE_SHA = "30eb1b694947f5f56f070f540dd6b85461841b8d836820471c62613f4db96cc2"
METRICS = ("flow_loss", "combined_mae_mixed_units", "arm_mae_rad", "hand_mae_u")
ALL_METRICS = (*METRICS, *[f"hand_mae_u_{i}" for i in range(6)])
NOISE_KEYS = (
    "continuous_time",
    "discrete_time",
    "noisy_action",
    "continuous_time_values",
    "discrete_time_values",
    "forward_seed",
    "preinjection_post",
)


def read(path):
    return json.loads(Path(path).read_text())


def make_memory(initial, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unexpected condition")
    if condition == "gru_history":
        module = CausalVisualGRU(initial)
    else:
        module = CausalVisualMemory()
        module.load_state_dict(initial, strict=True)
    expected = GRU_INITIAL_SHA if condition == "gru_history" else INITIAL_SHA
    if proto.content_hash(module.state_dict()) != expected:
        raise ValueError("Fresh initialization differs from the frozen prototype/preflight")
    return module


def state_values(state):
    return {str(i): value for i, value in enumerate(explicit_cache_values(state))}


def memory_outputs(module, sequence, condition, deployment=False):
    sequence = selected_prefix(sequence, condition)
    executor = FP32VisualMemory(module)
    if not deployment:
        latent, residual = executor.full(sequence)
        return latent, residual, len(sequence), None
    state = module.new_state()
    with torch.no_grad():
        outputs = []
        for current in sequence:
            latent, residual = executor.step(current, state)
            outputs.append(latent)
        latent = torch.stack(outputs)
    if state.observations != len(sequence):
        raise ValueError("Deployment state cadence changed")
    if any(
        not bool(torch.isfinite(v).all()) for v in (latent, residual, *state_values(state).values())
    ):
        raise ValueError("Nonfinite deployment memory values")
    return latent, residual, len(sequence), state


class PilotCache:
    """Immutable existing train/dev cache or gated newly created confirmation cache."""

    def __init__(self, root, plan, *, confirmation=False, expected_report_sha=None):
        self.root, self.plan = Path(root), plan
        report_path = self.root / "CACHE_RESULT.json"
        if expected_report_sha is not None and sha(report_path) != expected_report_sha:
            raise ValueError("Frozen cache report changed")
        self.report = read(report_path)
        expected = plan["confirmation"] if confirmation else plan["train"] + plan["dev"]
        cohort = "confirmation" if confirmation else "train_dev"
        if (
            self.report["status"] != "passed"
            or self.report["episodes"] != expected
            or self.report["cohort"] != cohort
        ):
            raise ValueError("Cache cohort/result mismatch")
        self.paths, self.loaded = {}, {}
        for row in self.report["files"]:
            path = self.root / row["file"]
            if path.parent != self.root or sha(path) != row["sha256"]:
                raise ValueError("Cache file identity mismatch")
            self.paths[row["episode"]] = (path, row["sha256"])
        if set(self.paths) != set(expected):
            raise ValueError("Cache episode inventory mismatch")

    def episode(self, episode):
        if episode not in self.paths:
            raise ValueError("Unauthorized cache episode")
        if episode not in self.loaded:
            values = torch.load(self.paths[episode][0], map_location="cpu", weights_only=True)
            if list(values) != self.plan["grids"][str(episode)]:
                raise ValueError("Frozen cache grid mismatch")
            for anchor, frame in values.items():
                if (
                    frame["episode"] != episode
                    or frame["anchor"] != anchor
                    or frame_hash(frame) != frame["content_sha256"]
                ):
                    raise ValueError("Frozen cache content mismatch")
                if frame["post_vl_features"].dtype != torch.float32:
                    raise ValueError("Expected original FP32 post-VL features")
            self.loaded[episode] = values
        return self.loaded[episode]

    def frame(self, episode, anchor):
        return self.episode(episode)[anchor]

    def prefix(self, episode, anchor, device="cuda"):
        grid = self.plan["grids"][str(episode)]
        if anchor not in grid:
            raise ValueError("Illegal causal anchor")
        indices = grid[: grid.index(anchor) + 1]
        return torch.stack(
            [self.frame(episode, a)["post_vl_features"][0, proto.IMAGE_POSITIONS] for a in indices]
        ).to(device), indices

    def assert_unchanged(self):
        for path, expected in self.paths.values():
            if sha(path) != expected:
                raise ValueError("Read-only cache file changed")
        for values in self.loaded.values():
            for frame in values.values():
                if frame_hash(frame) != frame["content_sha256"]:
                    raise ValueError("Read-only in-memory cache changed")


def numerical_probe(context, cache, module, condition, episode, anchor, label):
    sequence, indices = cache.prefix(episode, anchor)
    with torch.no_grad():
        full, full_residual, length, _ = memory_outputs(module, sequence, condition)
        step, step_residual, _, state = memory_outputs(module, sequence, condition, True)
    latent = error_summary(step, full)
    frame = cache.frame(episode, anchor)
    actions, tensors = (
        [],
        {
            "full_latent": full.cpu(),
            "step_latent": step.cpu(),
            "full_residual": full_residual.cpu(),
            "step_residual": step_residual.cpu(),
            "actions": {},
        },
    )
    seeds = [
        (f"nominal_{s}", deterministic_seed("diagnostic", episode, anchor, s))
        for s in context.plan["noise_seeds"]
    ]
    seeds.append(("direct_42", 42))
    for name, seed in seeds:
        context.budget()
        actual = sample_cached(context.model, frame, step_residual, seed)
        expected = sample_cached(context.model, frame, full_residual, seed)
        proto.action_record(actual, raw_record(frame), context.source.processor)
        proto.action_record(expected, raw_record(frame), context.source.processor)
        difference = error_summary(actual[:, :16], expected[:, :16])
        actions.append(
            {
                "seed_label": name,
                "actual_seed": seed,
                **difference,
                "old_action_gate": old_gate(difference, context.plan["action_limits"]),
            }
        )
        tensors["actions"][name] = {"step": actual, "full": expected}
    path = context.args.output / f"numerical-{condition}-{episode}-{anchor}-{label}.pt"
    torch.save(tensors, path)
    report = {
        "condition": condition,
        "episode": episode,
        "anchor": anchor,
        "prefix": indices,
        "effective_length": length,
        "label": label,
        "latent": latent,
        "old_latent_gate": old_gate(latent, context.plan["latent_limits"]),
        "active_actions": actions,
        "state": state.describe(),
        "state_sha256": proto.content_hash(state_values(state)),
        "tensors": {"path": str(path), "sha256": sha(path)},
    }
    report["passed"] = report["old_latent_gate"] and all(r["old_action_gate"] for r in actions)
    write_new(path.with_suffix(".json"), report)
    return report


def parameter_witness(module, before, seen):
    recurrence = "cell" if isinstance(module, CausalVisualGRU) else "mamba"
    prefixes = {
        "input_projection": "input_projection.",
        "output_projection": "output_projection.",
        **{f"norm{i}": f"layers.{i}.norm." for i in range(2)},
        **{f"recurrence{i}": f"layers.{i}.{recurrence}." for i in range(2)},
    }
    parameters = {
        name: {
            "changed": not torch.equal(value.detach().cpu(), before[name]),
            "finite": bool(torch.isfinite(value).all()),
            "maxabs_change": float((value.detach().cpu() - before[name]).abs().max()),
        }
        for name, value in module.state_dict().items()
    }
    groups = {
        group: {
            "finite_nonzero_gradient_seen": seen.get(group, False),
            "changed_parameters": [
                name for name, v in parameters.items() if v["changed"] and name.startswith(prefix)
            ],
        }
        for group, prefix in prefixes.items()
    }
    return {
        "passed": all(v["finite"] for v in parameters.values())
        and all(
            v["finite_nonzero_gradient_seen"] and v["changed_parameters"] for v in groups.values()
        ),
        "groups": groups,
        "parameters": parameters,
    }


def compare_receipts(condition_rows, schedule):
    if list(condition_rows) != list(CONDITIONS) or len(schedule) != 500:
        raise ValueError("Expected four conditions and exactly500 scheduled steps")
    reference = condition_rows[CONDITIONS[0]]
    for condition, rows in condition_rows.items():
        if len(rows) != 500:
            raise ValueError("Missing optimizer-step receipt")
        for index, (row, planned, ref) in enumerate(zip(rows, schedule, reference)):
            if (
                any(row.get(key) != value for key, value in planned.items())
                or row["optimizer_steps_completed"] != index + 1
            ):
                raise ValueError("Anchor/seed/LR/scheduled prefix pairing changed")
            expected_length = 1 if condition == "current_reset_mamba" else len(planned["prefix"])
            if row["condition"] != condition or row["effective_length"] != expected_length:
                raise ValueError("Actual condition/prefix cadence mismatch")
            if any(row["noise_receipt"][key] != ref["noise_receipt"][key] for key in NOISE_KEYS):
                raise ValueError("Actual native noise/time/preinjection condition mismatch")
    return {
        "passed": True,
        "conditions": list(CONDITIONS),
        "steps_per_condition": 500,
        "receipt_keys_compared": list(NOISE_KEYS),
        "effective_condition_equal_required": False,
    }


def quality_gates(evaluations, episodes):
    candidate = evaluations["mamba_history"]
    comparisons = {}
    for control in CONDITIONS[:-1]:
        other = evaluations[control]
        comparisons[control] = {}
        for key in METRICS[:2]:
            baseline = other["metrics"][key]
            improvement = (
                (baseline - candidate["metrics"][key]) / baseline if baseline > 0 else None
            )
            directional = sum(
                candidate["per_episode"][str(ep)][key] < other["per_episode"][str(ep)][key]
                for ep in episodes
            )
            comparisons[control][key] = {
                "relative_improvement": improvement,
                "episodes_better": directional,
                "passed": improvement is not None and improvement >= 0.02 and directional >= 3,
            }
    preservation = {
        key: {
            "native": evaluations["native"]["metrics"][key],
            "candidate": candidate["metrics"][key],
            "passed": candidate["metrics"][key] <= evaluations["native"]["metrics"][key] * 1.02,
        }
        for key in METRICS
    }
    return {
        "passed": all(v["passed"] for row in comparisons.values() for v in row.values())
        and all(v["passed"] for v in preservation.values()),
        "matched_comparisons": comparisons,
        "native_error_retention": preservation,
    }
