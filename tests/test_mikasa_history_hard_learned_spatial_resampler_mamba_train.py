from __future__ import annotations

import copy
from io import BytesIO
from pathlib import Path
import time
from types import SimpleNamespace

from gr00t.experimental import (
    mikasa_history_hard_learned_spatial_resampler_mamba as learned,
    mikasa_history_hard_learned_spatial_resampler_mamba_train as train,
    run_mikasa_history_hard_learned_spatial_resampler_mamba_train as runner,
)
import numpy as np
import pytest
import torch
from torch import nn


class _FakeCausalMixer(nn.Module):
    """Small deterministic recurrent mixer with the qualified full-forward API."""

    def __init__(self, _index: int, width: int):
        super().__init__()
        self.width = width
        self.gain = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(value.shape[0], self.width, dtype=value.dtype, device=value.device)
        outputs = []
        for token in value.unbind(dim=1):
            state = state * 0.5 + token
            outputs.append((state * self.gain).unsqueeze(1))
        return torch.cat(outputs, dim=1)


def _fake_mixer(index: int, width: int) -> nn.Module:
    return _FakeCausalMixer(index, width)


def _learned_model(condition: str, seed: int = train.TRAIN_SEED) -> nn.Module:
    return learned.LearnedSpatialResamplerMamba(
        condition,
        seed=seed,
        input_dim=4,
        width=8,
        frames=4,
        mixer_factory=_fake_mixer,
    )


def _tiny_arrays(rows: int = 36) -> dict[str, np.ndarray]:
    values = np.arange(rows * 4 * 2 * 64 * 4, dtype=np.float32)
    cue = values.reshape(rows, 4, 2, 64, 4) / float(values.size) + 0.1
    current = np.arange(rows * 2 * 64 * 4, dtype=np.float32).reshape(rows, 2, 64, 4)
    current = current / float(current.size) + 0.1
    return {"cue": cue, "current": current}


def _train_labels() -> np.ndarray:
    pairs = np.asarray([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
    labels = np.full((48, 2), -1, dtype=np.int64)
    labels[:36] = np.tile(pairs, (6, 1))
    return labels


def test_condition_and_fold_order_are_exact_and_isolated() -> None:
    assert train.CONDITION_ORDER == (
        "mamba_current",
        "mamba_zero",
        "mamba_shuffle",
        "mamba_ordered",
    )
    assert train.FOLD_ORDER == (0, 1, 2, 3, 4, 5)

    all_rows = set(range(36))
    for held_out in train.FOLD_ORDER:
        spec = train.fold_spec(held_out)
        expected_groups = tuple(group for group in train.FOLD_ORDER if group != held_out)
        expected_candidates = tuple(
            row for group in expected_groups for row in range(group * 6, group * 6 + 6)
        )
        expected_held_out = tuple(range(held_out * 6, held_out * 6 + 6))
        assert spec.heldout_group == held_out
        assert spec.candidate_groups == expected_groups
        assert spec.candidate_rows == expected_candidates
        assert spec.heldout_rows == expected_held_out
        assert set(spec.candidate_rows).isdisjoint(spec.heldout_rows)
        assert set(spec.candidate_rows) | set(spec.heldout_rows) == all_rows
        assert len(spec.candidate_rows) == 30
        assert len(spec.heldout_rows) == 6

    with pytest.raises(train.TrainContractError, match="held-out"):
        train.fold_spec(6)


def test_all_conditions_and_folds_have_identical_seeded_initialization() -> None:
    hashes = {
        (condition, held_out): train.state_dict_sha256(
            _learned_model(condition, seed=train.TRAIN_SEED)
        )
        for condition in train.CONDITION_ORDER
        for held_out in train.FOLD_ORDER
    }
    assert len(set(hashes.values())) == 1

    changed_seed = train.state_dict_sha256(
        _learned_model("mamba_ordered", seed=train.TRAIN_SEED + 1)
    )
    assert changed_seed != next(iter(hashes.values()))


def test_frozen_learning_rate_schedule_and_hash() -> None:
    assert train.TRAIN_STEPS == 1000
    assert train.lr_multiplier(0, train.TRAIN_STEPS) == pytest.approx(0.01)
    assert train.lr_multiplier(98, train.TRAIN_STEPS) == pytest.approx(0.99)
    assert train.lr_multiplier(99, train.TRAIN_STEPS) == pytest.approx(1.0)
    assert train.lr_multiplier(100, train.TRAIN_STEPS) < 1.0
    assert train.lr_multiplier(999, train.TRAIN_STEPS) == pytest.approx(0.1)
    assert train.optimizer_schedule_sha256(train.TRAIN_STEPS) == train.optimizer_schedule_sha256(
        train.TRAIN_STEPS
    )
    assert len(train.optimizer_schedule_sha256(train.TRAIN_STEPS)) == 64
    with pytest.raises(ValueError, match="learning-rate"):
        train.lr_multiplier(-1, train.TRAIN_STEPS)
    with pytest.raises(ValueError, match="learning-rate"):
        train.lr_multiplier(0, 0)


def test_effective_samples_are_condition_matched_and_fold_specific() -> None:
    for held_out in train.FOLD_ORDER:
        spec = train.fold_spec(held_out)
        hashes = {
            train.effective_samples_sha256(spec, total_steps=train.TRAIN_STEPS)
            for _condition in train.CONDITION_ORDER
        }
        assert len(hashes) == 1
    assert train.effective_samples_sha256(
        train.fold_spec(0), total_steps=train.TRAIN_STEPS
    ) != train.effective_samples_sha256(train.fold_spec(1), total_steps=train.TRAIN_STEPS)


class _TinyTrainModel(nn.Module):
    def __init__(self, condition: str, seed: int):
        super().__init__()
        self.condition = condition
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.class_bias = nn.Parameter(torch.randn(3, dtype=torch.float32))
        self.calls: list[tuple[int | None, int | None]] = []

    def forward(
        self,
        cue: torch.Tensor,
        current: torch.Tensor,
        *,
        shuffle_seed: int | None = None,
        shuffle_step: int | None = None,
    ) -> learned.LearnedSpatialOutput:
        del current
        self.calls.append((shuffle_seed, shuffle_step))
        logits = self.class_bias.view(1, 1, 3).expand(cue.shape[0], 2, 3)
        permutation = (
            learned.slot_shuffle_permutation(shuffle_seed, shuffle_step)
            if self.condition == "mamba_shuffle"
            and shuffle_seed is not None
            and shuffle_step is not None
            else None
        )
        return learned.LearnedSpatialOutput(
            logits=logits,
            sequence=torch.zeros(cue.shape[0], 1, 1),
            permutation=permutation,
        )


def _tiny_model_factory(condition: str, seed: int) -> nn.Module:
    return _TinyTrainModel(condition, seed)


def test_optimizer_uses_five_complete_six_row_microbatches_loss_over_five_and_one_shuffle() -> None:
    model = _TinyTrainModel("mamba_shuffle", train.TRAIN_SEED)
    reference = _TinyTrainModel("mamba_shuffle", train.TRAIN_SEED)
    spec = train.fold_spec(2)
    visited: list[tuple[int, int]] = []
    target = torch.zeros(6, 2, dtype=torch.int64)

    def group_loader(
        _arrays: dict[str, np.ndarray],
        _labels: np.ndarray,
        group: int,
        _device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visited.append((group, len(target)))
        return torch.full((6, 1), float(group)), torch.zeros(6, 1), target

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    record, gradients, finite = train.optimizer_step(
        model,
        optimizer,
        {},
        _train_labels(),
        spec,
        step=17,
        device=torch.device("cpu"),
        group_loader=group_loader,
        module_parameter_fn=lambda value: {"tiny": [value.class_bias]},
    )
    reference_output = reference(torch.ones(6, 1), torch.zeros(6, 1))
    learned.order_cross_entropy(reference_output, target).backward()

    assert visited == [(group, 6) for group in spec.candidate_groups]
    assert len(model.calls) == 5
    assert model.calls == [(train.TRAIN_SEED, 17)] * 5
    permutations = [learned.slot_shuffle_permutation(*call) for call in model.calls]
    assert all(torch.equal(permutations[0], value) for value in permutations[1:])
    assert sorted(permutations[0][:65].tolist()) == list(range(65))
    assert sorted(permutations[0][65:].tolist()) == list(range(65, 130))
    assert torch.allclose(model.class_bias.grad, reference.class_bias.grad)
    assert gradients == {"tiny": True}
    assert finite
    assert record["step"] == 18
    assert record["lr"] == pytest.approx(
        train.LEARNING_RATE * train.lr_multiplier(17, train.TRAIN_STEPS)
    )


class _SeededEvaluationModel(nn.Module):
    def __init__(self, condition: str, labels: np.ndarray):
        super().__init__()
        self.condition = condition
        self.labels = labels
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[int | None, int | None]] = []

    def forward(
        self,
        cue: torch.Tensor,
        current: torch.Tensor,
        *,
        shuffle_seed: int | None = None,
        shuffle_step: int | None = None,
    ) -> learned.LearnedSpatialOutput:
        del current
        self.calls.append((shuffle_seed, shuffle_step))
        rows = cue[:, 0].to(torch.int64).cpu().numpy()
        target = torch.from_numpy(self.labels[rows]).to(cue.device)
        prediction = target.clone()
        if self.condition == "mamba_shuffle":
            seed_index = train.EVAL_SHUFFLE_SEEDS.index(int(shuffle_seed))
            if seed_index in (1, 3):
                prediction[:, 1] = (prediction[:, 1] + 1) % 3
            if seed_index in (2, 3):
                prediction[:, 0] = (prediction[:, 0] + 1) % 3
        logits = torch.full((len(rows), 2, 3), -5.0, device=cue.device)
        logits.scatter_(2, prediction.unsqueeze(-1), 5.0)
        return learned.LearnedSpatialOutput(
            logits=logits + self.anchor * 0.0,
            sequence=torch.zeros(len(rows), 1, 1),
            permutation=None,
        )


def _row_group_loader(
    _arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    group: int,
    _device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(group * 6, group * 6 + 6)
    return rows[:, None].float(), torch.zeros(6, 1), torch.from_numpy(labels[rows.numpy()])


def test_shuffle_evaluation_uses_four_fixed_seeds_step_zero_then_seed_geometry_mean() -> None:
    labels = _train_labels()
    model = _SeededEvaluationModel("mamba_shuffle", labels)
    result = train.evaluate_fold(
        model,
        {},
        labels,
        condition="mamba_shuffle",
        heldout_group=3,
        device=torch.device("cpu"),
        group_loader=_row_group_loader,
    )

    assert (
        result["evaluation_seeds"]
        == list(train.EVAL_SHUFFLE_SEEDS)
        == [
            230941,
            230942,
            230943,
            230944,
        ]
    )
    assert model.calls == [(seed, 0) for seed in train.EVAL_SHUFFLE_SEEDS]
    assert len(result["section"]["records"]) == 4 * 6
    assert result["section"]["by_group"]["3"] == {
        "rows": 6,
        "pair_accuracy": pytest.approx(0.25),
        "slot0_accuracy": pytest.approx(0.5),
        "slot1_accuracy": pytest.approx(0.5),
    }
    assert result["section"]["group_equal"] == {
        "pair_accuracy": pytest.approx(0.25),
        "slot0_accuracy": pytest.approx(0.5),
        "slot1_accuracy": pytest.approx(0.5),
    }


def test_evaluator_opens_only_the_six_heldout_rows() -> None:
    labels = _train_labels()
    visited: list[int] = []

    def loader(
        arrays: dict[str, np.ndarray],
        target: np.ndarray,
        group: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visited.append(group)
        return _row_group_loader(arrays, target, group, device)

    result = train.evaluate_fold(
        _SeededEvaluationModel("mamba_ordered", labels),
        {},
        labels,
        condition="mamba_ordered",
        heldout_group=4,
        device=torch.device("cpu"),
        group_loader=loader,
    )
    assert visited == [4]
    assert [record["row_id"] for record in result["section"]["records"]] == list(range(24, 30))
    assert tuple(result["section"]["by_group"]) == ("4",)


def test_checkpoint_binds_full_fold_identity_reloads_with_logit_parity_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    condition = "mamba_ordered"
    spec = train.fold_spec(1)
    model = train.new_model(condition, torch.device("cpu"), model_factory=_tiny_model_factory)
    initial_hash = train.model_initialization_hash(model)
    with torch.no_grad():
        model.class_bias.add_(torch.tensor([0.5, -0.25, 0.125]))
    final_hash = train.state_dict_sha256(model)
    raw = train.checkpoint_bytes(
        model,
        condition=condition,
        spec=spec,
        source_sha256="a" * 64,
        cache_sha256="b" * 64,
        initial_hash=initial_hash,
        final_hash=final_hash,
    )
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    assert payload["schema"] == train.CHECKPOINT_SCHEMA
    assert payload["condition"] == condition
    assert payload["heldout_group"] == 1
    assert payload["candidate_groups"] == list(spec.candidate_groups)
    assert payload["heldout_rows"] == list(spec.heldout_rows)
    assert payload["candidate_rows"] == list(spec.candidate_rows)
    assert payload["steps"] == 1000
    assert payload["train_source_result_sha256"] == "a" * 64
    assert payload["technical_source_result_sha256"] == train.TECHNICAL_SOURCE_SHA256
    assert payload["technical_cache_result_sha256"] == "b" * 64
    assert payload["initial_state_sha256"] == initial_hash
    assert payload["final_state_sha256"] == final_hash
    assert "optimizer_state_dict" not in payload

    path = tmp_path / "final.pt"
    path.write_bytes(raw)
    loaded = train.load_checkpoint(
        path,
        condition=condition,
        heldout_group=1,
        source_sha256="a" * 64,
        cache_sha256="b" * 64,
        device=torch.device("cpu"),
        model_factory=_tiny_model_factory,
    )
    cue = torch.ones(3, 1)
    current = torch.zeros(3, 1)
    assert torch.equal(model(cue, current).logits, loaded(cue, current).logits)

    tamper_values = {
        "condition": "mamba_zero",
        "heldout_group": 2,
        "candidate_groups": list(reversed(spec.candidate_groups)),
        "heldout_rows": list(spec.heldout_rows[:-1]),
        "candidate_rows": list(spec.candidate_rows[:-1]),
        "steps": 999,
        "train_source_result_sha256": "c" * 64,
        "technical_source_result_sha256": "d" * 64,
        "technical_cache_result_sha256": "e" * 64,
        "initial_state_sha256": "f" * 64,
        "final_state_sha256": "0" * 64,
    }
    for index, (key, value) in enumerate(tamper_values.items()):
        changed = copy.deepcopy(payload)
        changed[key] = value
        tampered = tmp_path / f"tampered-{index}.pt"
        torch.save(changed, tampered)
        with pytest.raises(train.TrainContractError):
            train.load_checkpoint(
                tampered,
                condition=condition,
                heldout_group=1,
                source_sha256="a" * 64,
                cache_sha256="b" * 64,
                device=torch.device("cpu"),
                model_factory=_tiny_model_factory,
            )


def test_train_fold_writes_only_one_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train, "TRAIN_STEPS", 2)
    model = train.new_model("mamba_current", torch.device("cpu"), model_factory=_tiny_model_factory)
    expected_initial = train.model_initialization_hash(model)
    target = torch.zeros(6, 2, dtype=torch.int64)

    def loader(
        _arrays: dict[str, np.ndarray],
        _labels: np.ndarray,
        group: int,
        _device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.full((6, 1), float(group)), torch.zeros(6, 1), target

    checkpoint = tmp_path / "checkpoint-final.pt"
    record = train.train_fold(
        {},
        _train_labels(),
        condition="mamba_current",
        heldout_group=0,
        device=torch.device("cpu"),
        deadline=time.monotonic() + 30,
        checkpoint_path=checkpoint,
        expected_initial_hash=expected_initial,
        model_factory=_tiny_model_factory,
        group_loader=loader,
        module_parameter_fn=lambda value: {"tiny": [value.class_bias]},
    )
    assert record["steps"] == 2
    assert record["reload_parity"]
    assert [path.name for path in tmp_path.rglob("*.pt")] == ["checkpoint-final.pt"]
    assert checkpoint.stat().st_mode & 0o777 == 0o444


class _DonorReadingModel(nn.Module):
    condition = "mamba_ordered"

    def __init__(self, labels: np.ndarray):
        super().__init__()
        self.labels = labels
        self.anchor = nn.Parameter(torch.zeros(()))
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, cue: torch.Tensor, current: torch.Tensor) -> learned.LearnedSpatialOutput:
        self.inputs.append((cue.clone(), current.clone()))
        donor_rows = cue[:, 0, 0, 0, 0].to(torch.int64).cpu().numpy()
        prediction = torch.from_numpy(self.labels[donor_rows]).to(cue.device)
        logits = torch.full((len(donor_rows), 2, 3), -5.0, device=cue.device)
        logits.scatter_(2, prediction.unsqueeze(-1), 5.0)
        return learned.LearnedSpatialOutput(
            logits=logits + self.anchor * 0.0,
            sequence=torch.zeros(len(donor_rows), 1, 1),
            permutation=None,
        )


def _row_encoded_arrays() -> dict[str, np.ndarray]:
    cue = np.empty((36, 2, 2, 64, 1), dtype=np.float32)
    current = np.empty((36, 2, 64, 1), dtype=np.float32)
    for row in range(36):
        cue[row].fill(row)
        current[row].fill(1000 + row)
    return {"cue": cue, "current": current}


def test_ordered_substitution_has_exact_180_recipient_current_donor_cue_probes() -> None:
    arrays = _row_encoded_arrays()
    labels = _train_labels()
    cue, current, target, donors = train.substitution_batch(
        arrays,
        labels,
        group=3,
        recipient_row=19,
        device=torch.device("cpu"),
    )
    assert donors == (18, 20, 21, 22, 23)
    assert cue[:, 0, 0, 0, 0].tolist() == list(donors)
    assert current[:, 0, 0, 0].tolist() == [1019.0] * 5
    assert torch.equal(target, torch.from_numpy(labels[list(donors)]))

    models = {group: _DonorReadingModel(labels) for group in train.FOLD_ORDER}
    result = train.evaluate_ordered_substitutions(
        models,
        arrays,
        labels,
        device=torch.device("cpu"),
    )
    assert result["correct"] == result["total"] == 180
    assert result["accuracy"] == 1.0
    assert len(result["records"]) == 180
    assert (
        len({(record["recipient_row"], record["donor_row"]) for record in result["records"]}) == 180
    )
    for group, model in models.items():
        assert len(model.inputs) == 6
        for offset, (observed_cue, observed_current) in enumerate(model.inputs):
            recipient = group * 6 + offset
            assert torch.all(observed_current == float(1000 + recipient))
            donor_values = observed_cue[:, 0, 0, 0, 0].to(torch.int64).tolist()
            assert donor_values == [
                row for row in range(group * 6, group * 6 + 6) if row != recipient
            ]


def _evidence_section(
    condition: str, *, pair: float, slot0: float, slot1: float
) -> dict[str, object]:
    records = []
    by_group = {}
    seeds: tuple[int | None, ...] = (
        train.EVAL_SHUFFLE_SEEDS if condition == "mamba_shuffle" else (None,)
    )
    for group in train.FOLD_ORDER:
        by_group[str(group)] = {
            "rows": 6,
            "pair_accuracy": pair,
            "slot0_accuracy": slot0,
            "slot1_accuracy": slot1,
        }
        for seed in seeds:
            for row in range(group * 6, group * 6 + 6):
                record = {
                    "row_id": row,
                    "group_id": group,
                    "pair_accuracy": pair,
                    "slot0_accuracy": slot0,
                    "slot1_accuracy": slot1,
                }
                if seed is not None:
                    record["evaluation_seed"] = seed
                records.append(record)
    return {
        "condition": condition,
        "records": records,
        "by_group": by_group,
        "group_equal": {
            "pair_accuracy": pair,
            "slot0_accuracy": slot0,
            "slot1_accuracy": slot1,
        },
    }


def test_scientific_gates_require_finite_recomputable_metrics_and_exact_denominator() -> None:
    evidence = {
        "mamba_current": _evidence_section("mamba_current", pair=0.0, slot0=0.5, slot1=0.5),
        "mamba_zero": _evidence_section("mamba_zero", pair=0.0, slot0=0.5, slot1=0.5),
        "mamba_shuffle": _evidence_section("mamba_shuffle", pair=0.0, slot0=0.5, slot1=0.5),
        "mamba_ordered": _evidence_section("mamba_ordered", pair=1.0, slot0=1.0, slot1=1.0),
    }
    assert all(train.recompute_evidence(condition, evidence[condition]) for condition in evidence)
    gates = train.train_scientific_gates(evidence, {"correct": 144, "total": 180})
    assert gates and all(gates.values())

    wrong_denominator = train.train_scientific_gates(evidence, {"correct": 144, "total": 179})
    assert not wrong_denominator["substitution_denominator_exact"]
    corrupted = copy.deepcopy(evidence)
    corrupted["mamba_ordered"]["group_equal"]["pair_accuracy"] = float("nan")
    nonfinite = train.train_scientific_gates(corrupted, {"correct": 144, "total": 180})
    assert not nonfinite["all_metrics_finite"]
    assert not nonfinite["all_conditions_independently_recomputable"]


def test_cli_exposes_only_source_acceptance_and_train() -> None:
    assert runner.PHASES == ("source_acceptance", "train")
    protocol = runner.protocol()
    assert protocol["phases"] == ["source_acceptance", "train"]
    assert protocol["checkpoint_selection"] == "24_final_step_1000_only"
    assert protocol["dev_phase"] is None
    assert protocol["fresh_phase"] is None
    assert protocol["action_outputs"] == 0
    for forbidden in ("dev", "final", "fresh"):
        with pytest.raises(SystemExit):
            runner._parse_args(
                ["--phase", forbidden, "--output-root", f"/tmp/forbidden-{forbidden}"]
            )


def test_predecessor_gate_allowlists_reject_empty_extra_and_nonboolean_values() -> None:
    expected = ("alpha", "beta")
    assert runner._exact_passed_gates({"gates": {"alpha": True, "beta": True}}, expected)
    for gates in (
        {},
        {"alpha": True},
        {"alpha": True, "beta": True, "extra": True},
        {"alpha": 1, "beta": True},
        {"alpha": True, "beta": False},
    ):
        assert not runner._exact_passed_gates({"gates": gates}, expected)


def test_source_receipt_is_exact_and_boundary_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_identity = {"head_commit": "synthetic", "files": {}}
    technical = {"results": {"synthetic": {"sha256": "a" * 64}}}
    monkeypatch.setattr(runner, "source_identity", lambda _repo: source_identity)
    monkeypatch.setattr(runner, "validate_technical_predecessors", lambda: technical)
    base = {
        "schema": f"{train.EXPERIMENT_ID}/source-acceptance-v1",
        "status": "passed",
        "mode": "source_acceptance",
        "source_identity": source_identity,
        "technical_predecessors": technical,
        "protocol": runner.protocol(),
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "dev_rows_opened": [],
        "fresh_data_opened": False,
        "retry_count": 0,
    }

    def write(name: str, value: dict[str, object]) -> tuple[Path, str]:
        raw = runner.canonical_json_bytes(value)
        path = tmp_path / f"{name}.json"
        path.write_bytes(raw)
        path.chmod(0o444)
        return path, runner.sha256_bytes(raw)

    valid_path, valid_hash = write("valid", base)
    value, receipt = runner._require_source_receipt(valid_path, valid_hash, tmp_path)
    assert value == base
    assert receipt["sha256"] == valid_hash

    mutations = (
        ("targets", "target_npz_opened", [{"row_id": 0}]),
        ("dev", "dev_rows_opened", [36]),
        ("fresh", "fresh_data_opened", True),
        ("retry", "retry_count", 1),
        ("protocol", "protocol", {**runner.protocol(), "unexpected": True}),
    )
    for name, key, replacement in mutations:
        changed = copy.deepcopy(base)
        changed[key] = replacement
        path, digest = write(name, changed)
        with pytest.raises(PermissionError):
            runner._require_source_receipt(path, digest, tmp_path)


class _FakeArchive:
    def __init__(self, row_id: int):
        self.row_id = row_id
        self.files = list(runner.TARGET_FIELDS)

    def __enter__(self) -> "_FakeArchive":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def __getitem__(self, key: str) -> np.ndarray:
        pair = runner.PAIR_CODES[self.row_id % 6]
        values: dict[str, np.ndarray] = {
            "row_id": np.asarray(self.row_id),
            "group_id": np.asarray(self.row_id // 6),
            "pair_code": np.asarray(pair),
            "ordered_path_ids": np.asarray([int(pair[0]), int(pair[1])], dtype=np.int64),
        }
        return values[key]


def test_train_label_loader_opens_rows_zero_through_35_once_and_no_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified: list[int] = []
    opened_paths: list[Path] = []

    class Target:
        def __init__(self, row_id: int):
            self.row_id = row_id
            self.path = tmp_path / f"row-{row_id}.npz"
            self.bytes = row_id + 1
            self.sha256 = f"{row_id:064x}"

        def verify(self, *, root: Path) -> None:
            assert root == runner.CORPUS_ROOT
            verified.append(self.row_id)

    sources = SimpleNamespace(rows=[SimpleNamespace(target=Target(row_id)) for row_id in range(48)])
    monkeypatch.setattr(
        runner.HistoryHardSources,
        "discover",
        classmethod(lambda _cls: sources),
    )

    def fake_load(path: Path, *, allow_pickle: bool) -> _FakeArchive:
        assert allow_pickle is False
        path = Path(path)
        opened_paths.append(path)
        row_id = int(path.stem.removeprefix("row-"))
        return _FakeArchive(row_id)

    monkeypatch.setattr(runner.np, "load", fake_load)
    labels, opened = runner._load_train_labels()

    assert verified == list(range(36))
    assert [int(path.stem.removeprefix("row-")) for path in opened_paths] == list(range(36))
    assert [record["row_id"] for record in opened] == list(range(36))
    assert labels.shape == (48, 2)
    assert np.array_equal(labels, _train_labels())
    assert np.all(labels[36:] == -1)
