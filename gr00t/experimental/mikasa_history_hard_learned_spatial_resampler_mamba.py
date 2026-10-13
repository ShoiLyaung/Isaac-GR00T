"""Learned fixed-query spatial resampler and causal Mamba for history-hard cues."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT_ID: Final = "g1-mikasa-history-hard-learned-spatial-resampler-mamba-r1"
CAMERAS: Final = 2
PATCHES: Final = 64
GRID_SIDE: Final = 8
INPUT_DIM: Final = 2048
WIDTH: Final = 128
QUERIES: Final = 4
FRAMES: Final = 130
SLOT_FRAMES: Final = 65
SEQUENCE_LENGTH: Final = FRAMES * CAMERAS * QUERIES
BOUNDARY_INDICES: Final = (SLOT_FRAMES * CAMERAS * QUERIES - 1, SEQUENCE_LENGTH - 1)
CLASSES: Final = 3
CONDITIONS: Final = ("mamba_current", "mamba_zero", "mamba_shuffle", "mamba_ordered")
TRAIN_SEED: Final = 230923
DEV_SHUFFLE_SEEDS: Final = (230941, 230942, 230943, 230944)
SHUFFLE_DOMAIN: Final = "history-shuffle-v1"
PARAMETER_LIMIT: Final = 2_500_000
ROWS: Final = 48
ROWS_PER_GROUP: Final = 6
TRAIN_GROUPS: Final = (0, 1, 2, 3, 4, 5)
DEV_GROUPS: Final = (6, 7)
TRAIN_ROWS: Final = tuple(range(36))
DEV_ROWS: Final = tuple(range(36, 48))
PAIR_CODES: Final = ("01", "02", "10", "12", "20", "21")


class LearnedSpatialContractError(RuntimeError):
    """A frozen model, intervention, aggregation, or evidence contract changed."""


def slot_shuffle_permutation(seed: int, step: int, *, frames: int = FRAMES) -> torch.Tensor:
    """Two deterministic slot-local permutations shared by the effective batch."""

    if step < 0 or frames < 2 or frames % 2:
        raise ValueError("shuffle requires nonnegative step and an even frame count")
    half = frames // 2
    values = []
    for slot in range(2):
        payload = f"{SHUFFLE_DOMAIN}|{int(seed)}|{int(step)}|{slot}".encode()
        generator_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        generator = torch.Generator(device="cpu").manual_seed(generator_seed)
        values.append(torch.randperm(half, generator=generator) + slot * half)
    return torch.cat(values)


def row_major_coordinates(*, patches: int = PATCHES, grid_side: int = GRID_SIDE) -> torch.Tensor:
    if patches != grid_side * grid_side or grid_side < 2:
        raise ValueError("patches must form the frozen square row-major grid")
    return torch.tensor(
        [
            (index // grid_side / (grid_side - 1), index % grid_side / (grid_side - 1))
            for index in range(patches)
        ],
        dtype=torch.float32,
    )


def _normalize_patches(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise LearnedSpatialContractError("patch tokens must have finite nonzero FP32 norm")
    return value / norms


def apply_condition(
    cue: torch.Tensor,
    current: torch.Tensor,
    condition: str,
    *,
    shuffle_seed: int | None = None,
    shuffle_step: int | None = None,
    input_dim: int = INPUT_DIM,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
]:
    """Return raw cue, current, delta, coordinates and reliability at the frozen boundary."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown learned-spatial condition: {condition}")
    if cue.ndim != 5 or current.ndim != 4:
        raise ValueError("cue/current must be [B,T,C,P,D] and [B,C,P,D]")
    batch, frames, cameras, patches, width = cue.shape
    if current.shape != (batch, cameras, patches, width):
        raise ValueError("cue/current shape mismatch")
    if cameras != CAMERAS or patches != PATCHES or width != input_dim:
        raise ValueError("formal learned-spatial input shape changed")
    permutation = None
    ordered = cue
    if condition == "mamba_shuffle":
        if shuffle_seed is None or shuffle_step is None:
            raise ValueError("mamba_shuffle requires explicit seed and optimizer/eval step")
        permutation = slot_shuffle_permutation(shuffle_seed, shuffle_step, frames=frames)
        ordered = cue.index_select(1, permutation.to(cue.device))
    cue_norm = _normalize_patches(ordered)
    current_norm = _normalize_patches(current)
    current_full = current_norm[:, None].expand_as(cue_norm)
    coords = row_major_coordinates().to(cue_norm.device)
    coords = coords.view(1, 1, 1, patches, 2).expand(batch, frames, cameras, -1, -1)
    if condition == "mamba_current":
        cue_norm = current_full
    delta = cue_norm - current_full
    reliability = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    if condition == "mamba_zero":
        cue_norm = torch.zeros_like(cue_norm)
        current_full = torch.zeros_like(current_full)
        delta = torch.zeros_like(delta)
        coords = torch.zeros_like(coords)
        reliability = torch.zeros_like(reliability)
    return cue_norm, current_full, delta, coords, reliability, permutation


class CausalMambaLayer(nn.Module):
    def __init__(
        self,
        index: int,
        *,
        width: int = WIDTH,
        mixer_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        if mixer_factory is None:
            from mamba_ssm import Mamba2

            self.mixer = Mamba2(
                d_model=width,
                d_state=16,
                d_conv=4,
                expand=2,
                headdim=64,
                ngroups=1,
                chunk_size=16,
                use_mem_eff_path=False,
                layer_idx=index,
                device="cpu",
                dtype=torch.float32,
            )
        else:
            self.mixer = mixer_factory(index, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mixer(self.norm(value))

    def allocate(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen=1, dtype=torch.float32, device=device
        )

    def step(self, value: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        update, conv, ssm = self.mixer.step(self.norm(value), *cache)
        if conv is not cache[0] or ssm is not cache[1]:
            raise LearnedSpatialContractError("Mamba step must update supplied row state in place")
        return value + update


@dataclass
class TemporalState:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    batch_size: int
    tokens: int = 0

    def clone(self) -> "TemporalState":
        return TemporalState(
            [tuple(tensor.clone() for tensor in pair) for pair in self.layers],
            self.batch_size,
            self.tokens,
        )

    @torch.no_grad()
    def reset(self) -> None:
        for pair in self.layers:
            for tensor in pair:
                tensor.zero_()
        self.tokens = 0


@dataclass(frozen=True)
class LearnedSpatialOutput:
    logits: torch.Tensor
    sequence: torch.Tensor
    permutation: torch.Tensor | None

    @property
    def predictions(self) -> torch.Tensor:
        # torch.argmax resolves an exact tie to the lower class ID.
        return self.logits.argmax(dim=-1)


class LearnedSpatialResamplerMamba(nn.Module):
    """Fixed four-query spatial resampler followed by two causal Mamba layers."""

    def __init__(
        self,
        condition: str,
        *,
        seed: int = TRAIN_SEED,
        input_dim: int = INPUT_DIM,
        width: int = WIDTH,
        frames: int = FRAMES,
        cameras: int = CAMERAS,
        patches: int = PATCHES,
        queries: int = QUERIES,
        mixer_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown learned-spatial condition: {condition}")
        if frames % 2 or cameras != CAMERAS or patches != PATCHES or queries != QUERIES:
            raise ValueError("formal cadence requires even frames, two cameras, 64 patches and K4")
        self.condition = condition
        self.seed = int(seed)
        self.input_dim = input_dim
        self.width = width
        self.frames = frames
        self.cameras = cameras
        self.patches = patches
        self.queries = queries
        self.slot_frames = frames // 2
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.raw_projection = nn.Linear(input_dim, width)
            self.residual_projection = nn.Linear(input_dim + 3, width)
            nn.init.zeros_(self.residual_projection.weight)
            nn.init.zeros_(self.residual_projection.bias)
            self.resampler_queries = nn.Parameter(torch.empty(queries, width))
            nn.init.normal_(self.resampler_queries, std=0.02)
            self.query_norm = nn.LayerNorm(width)
            self.patch_norm = nn.LayerNorm(width)
            self.cross_attention = nn.MultiheadAttention(
                width, num_heads=4, dropout=0.0, batch_first=True
            )
            self.ff_norm = nn.LayerNorm(width)
            self.ff = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, width))
            self.camera_embedding = nn.Embedding(cameras, width)
            self.query_embedding = nn.Embedding(queries, width)
            self.slot_embedding = nn.Embedding(2, width)
            self.temporal = nn.ModuleList(
                [
                    CausalMambaLayer(index, width=width, mixer_factory=mixer_factory)
                    for index in range(2)
                ]
            )
            self.classifier = nn.Linear(width, CLASSES)

    def encode_resampled_sequence(
        self,
        cue: torch.Tensor,
        current: torch.Tensor,
        *,
        shuffle_seed: int | None = None,
        shuffle_step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cue.shape[-1] != self.input_dim or cue.shape[1:4] != (
            self.frames,
            self.cameras,
            self.patches,
        ):
            raise ValueError("model cue shape differs from configured cadence")
        cue_norm, _current, delta, coords, reliability, permutation = apply_condition(
            cue,
            current,
            self.condition,
            shuffle_seed=shuffle_seed,
            shuffle_step=shuffle_step,
            input_dim=self.input_dim,
        )
        raw = self.raw_projection(cue_norm)
        residual_input = torch.cat((delta, coords, reliability), dim=-1)
        patch = raw + self.residual_projection(residual_input)
        batch = cue.shape[0]
        flat = patch.reshape(batch * self.frames * self.cameras, self.patches, self.width)
        query = self.resampler_queries.unsqueeze(0).expand(flat.shape[0], -1, -1)
        attention, _ = self.cross_attention(
            self.query_norm(query),
            self.patch_norm(flat),
            self.patch_norm(flat),
            need_weights=False,
        )
        value = query + attention
        value = value + self.ff(self.ff_norm(value))
        value = value.reshape(batch, self.frames, self.cameras, self.queries, self.width)
        camera_ids = torch.arange(self.cameras, device=value.device)
        query_ids = torch.arange(self.queries, device=value.device)
        slot_ids = torch.cat(
            (
                torch.zeros(self.slot_frames, dtype=torch.long, device=value.device),
                torch.ones(self.slot_frames, dtype=torch.long, device=value.device),
            )
        )
        value = (
            value
            + self.camera_embedding(camera_ids)[None, None, :, None]
            + self.query_embedding(query_ids)[None, None, None, :]
            + self.slot_embedding(slot_ids)[None, :, None, None]
        )
        sequence = value.reshape(batch, self.frames * self.cameras * self.queries, self.width)
        return sequence, permutation

    def temporal_full(self, sequence: torch.Tensor) -> torch.Tensor:
        value = sequence
        for layer in self.temporal:
            value = layer(value)
        return value

    def new_state(self, batch_size: int, device: torch.device | None = None) -> TemporalState:
        actual = device or next(self.parameters()).device
        return TemporalState(
            [layer.allocate(batch_size, actual) for layer in self.temporal],
            batch_size,
        )

    def temporal_step(self, token: torch.Tensor, state: TemporalState) -> torch.Tensor:
        if token.shape != (state.batch_size, 1, self.width):
            raise ValueError("streaming token must be [batch,1,width]")
        value = token
        for layer, cache in zip(self.temporal, state.layers, strict=True):
            value = layer.step(value, cache)
        state.tokens += 1
        return value

    def temporal_chunk(self, sequence: torch.Tensor, state: TemporalState) -> torch.Tensor:
        return torch.cat(
            [self.temporal_step(token, state) for token in sequence.split(1, dim=1)], dim=1
        )

    def forward(
        self,
        cue: torch.Tensor,
        current: torch.Tensor,
        *,
        shuffle_seed: int | None = None,
        shuffle_step: int | None = None,
    ) -> LearnedSpatialOutput:
        sequence, permutation = self.encode_resampled_sequence(
            cue,
            current,
            shuffle_seed=shuffle_seed,
            shuffle_step=shuffle_step,
        )
        hidden = self.temporal_full(sequence)
        boundary = (
            self.slot_frames * self.cameras * self.queries - 1,
            self.frames * self.cameras * self.queries - 1,
        )
        logits = self.classifier(hidden[:, boundary])
        if logits.shape != (cue.shape[0], 2, CLASSES) or not bool(torch.isfinite(logits).all()):
            raise LearnedSpatialContractError("shared classifier output is invalid")
        return LearnedSpatialOutput(logits=logits, sequence=hidden, permutation=permutation)


def order_cross_entropy(output: LearnedSpatialOutput, labels: torch.Tensor) -> torch.Tensor:
    if labels.shape != (output.logits.shape[0], 2) or labels.dtype != torch.int64:
        raise ValueError("ordered labels must be int64 [batch,2]")
    return F.cross_entropy(output.logits.reshape(-1, CLASSES), labels.reshape(-1))


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def module_has_finite_nonzero_gradient(module: nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    return (
        bool(gradients)
        and all(
            gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        and any(
            bool(torch.count_nonzero(gradient)) for gradient in gradients if gradient is not None
        )
    )


def score_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    row_ids: Sequence[int],
) -> dict[str, Any]:
    rows = tuple(int(row) for row in row_ids)
    if predictions.shape != (len(rows), 2) or labels.shape != (ROWS, 2):
        raise LearnedSpatialContractError("prediction/label shape changed")
    by_group = {}
    records = []
    for offset, row in enumerate(rows):
        prediction = predictions[offset].astype(np.int64)
        target = labels[row]
        records.append(
            {
                "row_id": row,
                "group_id": int(groups[row]),
                "prediction": prediction.tolist(),
                "target": target.tolist(),
                "slot0_accuracy": float(prediction[0] == target[0]),
                "slot1_accuracy": float(prediction[1] == target[1]),
                "pair_accuracy": float(np.array_equal(prediction, target)),
            }
        )
    for group in sorted({int(groups[row]) for row in rows}):
        selected = [record for record in records if record["group_id"] == group]
        by_group[str(group)] = {
            key: float(np.mean([record[key] for record in selected]))
            for key in ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
        }
        by_group[str(group)]["rows"] = len(selected)
    return {
        "records": records,
        "by_group": by_group,
        "group_equal": {
            key: float(np.mean([value[key] for value in by_group.values()]))
            for key in ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
        },
    }


def mean_shuffle_geometry_metrics(sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(sections) != len(DEV_SHUFFLE_SEEDS):
        raise LearnedSpatialContractError("shuffle evaluation requires four frozen seeds")
    groups = tuple(sections[0]["by_group"])
    metrics = ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    if any(tuple(section["by_group"]) != groups for section in sections):
        raise LearnedSpatialContractError("shuffle seed geometry order changed")
    by_group = {
        group: {
            "rows": sections[0]["by_group"][group]["rows"],
            **{
                metric: float(np.mean([section["by_group"][group][metric] for section in sections]))
                for metric in metrics
            },
        }
        for group in groups
    }
    return {
        "seed_order": list(DEV_SHUFFLE_SEEDS),
        "seed_sections": list(sections),
        "by_group": by_group,
        "group_equal": {
            metric: float(np.mean([by_group[group][metric] for group in groups]))
            for metric in metrics
        },
    }


def scientific_gates(
    evidence: Mapping[str, Any],
    *,
    expected_groups: int,
    substitution_correct: int,
    substitution_total: int,
) -> dict[str, bool]:
    ordered = evidence["mamba_ordered"]
    controls = ("mamba_current", "mamba_zero", "mamba_shuffle")
    gates: dict[str, bool] = {
        "ordered_pair_at_least_83_33": ordered["group_equal"]["pair_accuracy"] >= 5 / 6,
        "ordered_slot0_at_least_91_67": ordered["group_equal"]["slot0_accuracy"] >= 11 / 12,
        "ordered_slot1_at_least_91_67": ordered["group_equal"]["slot1_accuracy"] >= 11 / 12,
        "substitution_denominator_exact": substitution_total
        == (180 if expected_groups == 6 else 60),
        "substitution_accuracy_at_least_80": substitution_total > 0
        and substitution_correct / substitution_total >= 0.80,
    }
    required_direction = 5 if expected_groups == 6 else expected_groups
    for control in controls:
        other = evidence[control]
        gates[f"ordered_vs_{control}_pair_plus_50"] = (
            ordered["group_equal"]["pair_accuracy"] - other["group_equal"]["pair_accuracy"] >= 0.50
        )
        for slot in ("slot0_accuracy", "slot1_accuracy"):
            gates[f"ordered_vs_{control}_{slot}_plus_25"] = (
                ordered["group_equal"][slot] - other["group_equal"][slot] >= 0.25
            )
        directions = sum(
            ordered["by_group"][group]["pair_accuracy"] > other["by_group"][group]["pair_accuracy"]
            for group in ordered["by_group"]
        )
        gates[f"ordered_vs_{control}_pair_direction"] = directions >= required_direction
    gates["group_count_exact"] = len(ordered["by_group"]) == expected_groups
    gates["all_metrics_finite"] = all(
        np.isfinite(float(section["group_equal"][metric]))
        for section in evidence.values()
        for metric in ("pair_accuracy", "slot0_accuracy", "slot1_accuracy")
    )
    return {key: bool(value) for key, value in gates.items()}
