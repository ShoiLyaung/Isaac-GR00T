from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from gr00t.experimental import (
    mikasa_history_hard_learned_spatial_resampler_mamba as learned,
    run_mikasa_history_hard_learned_spatial_resampler_mamba as runner,
)
import numpy as np
import pytest
import torch
from torch import nn


class _FakeCausalMixer(nn.Module):
    """Small recurrent mixer with matching full/step semantics for CPU tests."""

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

    def allocate_inference_cache(
        self,
        batch_size: int,
        _max_seqlen: int | None = None,
        *,
        max_seqlen: int | None = None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del _max_seqlen, max_seqlen
        return (
            torch.zeros(batch_size, self.width, dtype=dtype, device=device),
            torch.zeros(batch_size, self.width, dtype=dtype, device=device),
        )

    def step(
        self,
        value: torch.Tensor,
        convolution_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        convolution_state.mul_(0.5).add_(value[:, 0])
        ssm_state.copy_(convolution_state)
        return (
            convolution_state[:, None] * self.gain,
            convolution_state,
            ssm_state,
        )


def _fake_mixer(index: int, width: int) -> nn.Module:
    return _FakeCausalMixer(index, width)


def _inputs(
    *, frames: int = 4, input_dim: int = 4, batch: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    cue = torch.rand(
        batch,
        frames,
        learned.CAMERAS,
        learned.PATCHES,
        input_dim,
        generator=generator,
    )
    current = torch.rand(
        batch,
        learned.CAMERAS,
        learned.PATCHES,
        input_dim,
        generator=generator,
    )
    return cue + 0.1, current + 0.1


def _model(
    condition: str = "mamba_ordered",
    *,
    frames: int = 4,
    input_dim: int = 4,
    width: int = 8,
) -> learned.LearnedSpatialResamplerMamba:
    return learned.LearnedSpatialResamplerMamba(
        condition,
        seed=learned.TRAIN_SEED,
        input_dim=input_dim,
        width=width,
        frames=frames,
        mixer_factory=_fake_mixer,
    )


def _score_section(
    pair: float,
    slot0: float,
    slot1: float,
    *,
    groups: tuple[str, ...],
    rows: int = 6,
) -> dict:
    by_group = {
        group: {
            "rows": rows,
            "pair_accuracy": pair,
            "slot0_accuracy": slot0,
            "slot1_accuracy": slot1,
        }
        for group in groups
    }
    return {
        "records": [],
        "by_group": by_group,
        "group_equal": {
            "pair_accuracy": pair,
            "slot0_accuracy": slot0,
            "slot1_accuracy": slot1,
        },
    }


def test_slot_shuffle_is_deterministic_slot_local_and_domain_separated() -> None:
    first = learned.slot_shuffle_permutation(230923, 7)
    repeated = learned.slot_shuffle_permutation(230923, 7)
    next_step = learned.slot_shuffle_permutation(230923, 8)
    next_seed = learned.slot_shuffle_permutation(230924, 7)

    assert torch.equal(first, repeated)
    assert sorted(first[:65].tolist()) == list(range(65))
    assert sorted(first[65:].tolist()) == list(range(65, 130))
    assert not torch.equal(first[:65], first[65:] - 65)
    assert not torch.equal(first, next_step)
    assert not torch.equal(first, next_seed)
    with pytest.raises(ValueError, match="even frame count"):
        learned.slot_shuffle_permutation(1, 0, frames=3)


def test_condition_interventions_have_exact_frozen_boundaries() -> None:
    cue, current = _inputs(frames=4)
    ordered = learned.apply_condition(cue, current, "mamba_ordered", input_dim=4)
    current_only = learned.apply_condition(cue, current, "mamba_current", input_dim=4)
    zero = learned.apply_condition(cue, current, "mamba_zero", input_dim=4)
    shuffled = learned.apply_condition(
        cue,
        current,
        "mamba_shuffle",
        shuffle_seed=230923,
        shuffle_step=11,
        input_dim=4,
    )

    cue_norm, current_full, delta, coordinates, reliability, permutation = ordered
    assert permutation is None
    assert torch.allclose(
        torch.linalg.vector_norm(cue_norm, dim=-1), torch.ones_like(reliability[..., 0])
    )
    assert torch.allclose(delta, cue_norm - current_full)
    assert torch.allclose(reliability, torch.linalg.vector_norm(delta, dim=-1, keepdim=True))
    assert torch.equal(coordinates[0, 0, 0], learned.row_major_coordinates())

    current_cue, current_full_2, current_delta, current_coords, current_rel, _ = current_only
    assert torch.equal(current_cue, current_full_2)
    assert torch.count_nonzero(current_delta) == 0
    assert torch.count_nonzero(current_rel) == 0
    assert torch.equal(current_coords, coordinates)

    for value in zero[:5]:
        assert torch.count_nonzero(value) == 0

    shuffled_cue, shuffled_current, shuffled_delta, shuffled_coords, shuffled_rel, perm = shuffled
    assert perm is not None
    assert torch.equal(shuffled_cue, cue_norm.index_select(1, perm))
    assert torch.equal(shuffled_current, current_full)
    assert torch.equal(shuffled_coords, coordinates)
    assert torch.equal(shuffled_delta, shuffled_cue - shuffled_current)
    assert torch.equal(
        shuffled_rel,
        torch.linalg.vector_norm(shuffled_delta, dim=-1, keepdim=True),
    )


def test_row_major_coordinates_are_exact() -> None:
    coordinates = learned.row_major_coordinates()
    assert coordinates.shape == (64, 2)
    assert coordinates.dtype == torch.float32
    assert coordinates[0].tolist() == [0.0, 0.0]
    assert coordinates[7].tolist() == [0.0, 1.0]
    assert coordinates[8].tolist() == pytest.approx([1 / 7, 0.0])
    assert coordinates[63].tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="square row-major grid"):
        learned.row_major_coordinates(patches=63)


def test_residual_projection_is_exactly_zero_initialized() -> None:
    model = _model()
    assert torch.count_nonzero(model.residual_projection.weight) == 0
    assert torch.count_nonzero(model.residual_projection.bias) == 0


def test_all_conditions_keep_shape_and_module_call_counts() -> None:
    cue, current = _inputs(frames=4)
    expected_shape = (1, 2, learned.CLASSES)
    for condition in learned.CONDITIONS:
        model = _model(condition)
        calls: dict[str, int] = {}
        hooks = []
        modules = {
            "raw": model.raw_projection,
            "residual": model.residual_projection,
            "attention": model.cross_attention,
            "ffn": model.ff,
            "mamba0": model.temporal[0].mixer,
            "mamba1": model.temporal[1].mixer,
            "classifier": model.classifier,
        }
        for name, module in modules.items():
            hooks.append(
                module.register_forward_hook(
                    lambda _module, _inputs, _output, name=name: calls.__setitem__(
                        name, calls.get(name, 0) + 1
                    )
                )
            )
        kwargs = {"shuffle_seed": 230923, "shuffle_step": 3} if condition == "mamba_shuffle" else {}
        output = model(cue, current, **kwargs)
        for hook in hooks:
            hook.remove()
        assert output.logits.shape == expected_shape
        assert output.sequence.shape == (1, 4 * 2 * 4, 8)
        assert calls == {name: 1 for name in modules}


def test_zero_control_retains_only_downstream_camera_query_slot_structure() -> None:
    model = _model("mamba_zero")
    cue, current = _inputs(frames=4)
    with torch.no_grad():
        model.raw_projection.weight.zero_()
        model.raw_projection.bias.zero_()
        model.residual_projection.weight.zero_()
        model.residual_projection.bias.zero_()
        model.resampler_queries.zero_()
        for parameter in model.cross_attention.parameters():
            parameter.zero_()
        for parameter in model.ff.parameters():
            parameter.zero_()
        model.camera_embedding.weight.zero_()
        model.query_embedding.weight.zero_()
        model.slot_embedding.weight.zero_()
        model.camera_embedding.weight[1, 0] = 10
        model.query_embedding.weight[:, 1] = torch.arange(4)
        model.slot_embedding.weight[1, 2] = 100
    sequence, _ = model.encode_resampled_sequence(cue, current)
    view = sequence.reshape(1, 4, 2, 4, 8)
    assert view[0, 0, 0, 0, :3].tolist() == [0.0, 0.0, 0.0]
    assert view[0, 0, 1, 0, :3].tolist() == [10.0, 0.0, 0.0]
    assert view[0, 0, 0, 3, :3].tolist() == [0.0, 3.0, 0.0]
    assert view[0, 2, 0, 0, :3].tolist() == [0.0, 0.0, 100.0]


def test_sequence_order_and_classifier_boundaries_are_exact() -> None:
    model = _model(frames=130)
    cue, current = _inputs(frames=130)
    with torch.no_grad():
        model.raw_projection.weight.zero_()
        model.raw_projection.bias.zero_()
        model.resampler_queries.zero_()
        for parameter in model.cross_attention.parameters():
            parameter.zero_()
        for parameter in model.ff.parameters():
            parameter.zero_()
        model.camera_embedding.weight.zero_()
        model.query_embedding.weight.zero_()
        model.slot_embedding.weight.zero_()
        model.camera_embedding.weight[1, 0] = 10
        model.query_embedding.weight[:, 1] = torch.arange(4)
        model.slot_embedding.weight[1, 2] = 100
    sequence, _ = model.encode_resampled_sequence(cue, current)
    view = sequence.reshape(1, 130, 2, 4, 8)
    locations = ((0, 0, 0), (0, 1, 3), (64, 1, 3), (65, 0, 0), (129, 1, 3))
    for frame, camera, query in locations:
        index = (frame * 2 + camera) * 4 + query
        assert torch.equal(sequence[0, index], view[0, frame, camera, query])
    assert learned.BOUNDARY_INDICES == (519, 1039)
    assert sequence.shape[1] == learned.SEQUENCE_LENGTH == 1040
    assert sequence[0, 519, :3].tolist() == [10.0, 3.0, 0.0]
    assert sequence[0, 1039, :3].tolist() == [10.0, 3.0, 100.0]


def test_causal_full_forward_blocks_future_slot_from_slot0() -> None:
    model = _model(frames=4)
    model.eval()
    cue, current = _inputs(frames=4)
    changed = cue.clone()
    changed[:, 2:] = torch.flip(changed[:, 2:], dims=(1,)) * 7
    with torch.no_grad():
        original = model(cue, current)
        future_changed = model(changed, current)
    assert torch.equal(original.logits[:, 0], future_changed.logits[:, 0])
    assert not torch.equal(original.logits[:, 1], future_changed.logits[:, 1])


def test_full_and_token_step_are_equivalent_and_row_reset_is_exact() -> None:
    model = _model()
    model.eval()
    sequence = torch.randn(2, 17, 8, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        full = model.temporal_full(sequence)
        state = model.new_state(2, torch.device("cpu"))
        stepped = model.temporal_chunk(sequence, state)
        assert state.tokens == 17
        state.reset()
        after_reset = model.temporal_chunk(sequence, state)
        fresh = model.temporal_chunk(sequence, model.new_state(2, torch.device("cpu")))
    assert torch.allclose(full, stepped, atol=1e-6, rtol=1e-6)
    assert torch.equal(after_reset, fresh)
    assert state.tokens == 17


def test_formal_model_parameter_count_is_below_frozen_limit() -> None:
    model = learned.LearnedSpatialResamplerMamba("mamba_ordered", mixer_factory=_fake_mixer)
    count = learned.trainable_parameter_count(model)
    assert 0 < count < learned.PARAMETER_LIMIT == 2_500_000


def test_cross_entropy_and_lower_class_argmax_tie_contract() -> None:
    output = learned.LearnedSpatialOutput(
        logits=torch.zeros(1, 2, 3),
        sequence=torch.zeros(1, 2, 4),
        permutation=None,
    )
    assert output.predictions.tolist() == [[0, 0]]
    loss = learned.order_cross_entropy(output, torch.tensor([[0, 2]], dtype=torch.int64))
    assert loss.item() == pytest.approx(np.log(3.0))
    with pytest.raises(ValueError, match="int64"):
        learned.order_cross_entropy(output, torch.tensor([[0, 2]], dtype=torch.int32))


def test_shuffle_aggregation_averages_seeds_inside_each_geometry_then_geometries() -> None:
    sections = []
    for index, _seed in enumerate(learned.DEV_SHUFFLE_SEEDS):
        sections.append(
            {
                "by_group": {
                    "6": {
                        "rows": 1,
                        "pair_accuracy": index / 3,
                        "slot0_accuracy": 1.0,
                        "slot1_accuracy": 0.0,
                    },
                    "7": {
                        "rows": 99,
                        "pair_accuracy": 1.0 - index / 3,
                        "slot0_accuracy": 0.0,
                        "slot1_accuracy": 1.0,
                    },
                }
            }
        )
    result = learned.mean_shuffle_geometry_metrics(sections)
    assert result["seed_order"] == list(learned.DEV_SHUFFLE_SEEDS)
    assert result["by_group"]["6"]["pair_accuracy"] == pytest.approx(0.5)
    assert result["by_group"]["7"]["pair_accuracy"] == pytest.approx(0.5)
    assert result["group_equal"] == {
        "pair_accuracy": pytest.approx(0.5),
        "slot0_accuracy": pytest.approx(0.5),
        "slot1_accuracy": pytest.approx(0.5),
    }


@pytest.mark.parametrize(
    ("expected_groups", "substitution_correct", "substitution_total"),
    ((6, 144, 180), (2, 48, 60)),
)
def test_scientific_gate_uses_exact_substitution_denominator(
    expected_groups: int, substitution_correct: int, substitution_total: int
) -> None:
    groups = tuple(str(index) for index in range(expected_groups))
    evidence = {
        "mamba_ordered": _score_section(1.0, 1.0, 1.0, groups=groups),
        "mamba_current": _score_section(0.0, 0.5, 0.5, groups=groups),
        "mamba_zero": _score_section(0.0, 0.5, 0.5, groups=groups),
        "mamba_shuffle": _score_section(0.0, 0.5, 0.5, groups=groups),
    }
    passing = learned.scientific_gates(
        evidence,
        expected_groups=expected_groups,
        substitution_correct=substitution_correct,
        substitution_total=substitution_total,
    )
    assert all(passing.values())
    wrong_denominator = learned.scientific_gates(
        evidence,
        expected_groups=expected_groups,
        substitution_correct=substitution_correct,
        substitution_total=substitution_total - 1,
    )
    assert not wrong_denominator["substitution_denominator_exact"]


def test_public_boundary_requires_full_grid_fields_and_ignores_topk_fields() -> None:
    receipt = {
        "images": 4,
        "merged_tokens_per_image": 64,
        "embedding_dim": runner.EMBED_DIM,
        "grid_side": 8,
        "owner_type": "transformers.Qwen3VLForConditionalGeneration",
        "method_qualname": "Qwen3VLForConditionalGeneration.get_image_features",
    }
    assert runner._public_boundary_exact(receipt, 4)
    assert runner._public_boundary_exact({**receipt, "top_k": -1, "event_width": -1}, 4)
    assert not runner._public_boundary_exact({**receipt, "grid_side": 7}, 4)


def test_cache_loader_checks_tiny_full_grid_shape_and_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, frames, patches, width = 2, 4, 64, 3
    cue = np.ones((rows, frames, 2, patches, width), dtype=np.float32)
    current = np.ones((rows, 2, patches, width), dtype=np.float32)
    cue_path = tmp_path / "CUE.npy"
    current_path = tmp_path / "CURRENT.npy"
    np.save(cue_path, cue, allow_pickle=False)
    np.save(current_path, current, allow_pickle=False)
    cue_path.chmod(0o444)
    current_path.chmod(0o444)

    monkeypatch.setattr(runner, "ROWS", rows)
    monkeypatch.setattr(runner, "FRAMES", frames)
    monkeypatch.setattr(runner, "INPUT_DIM", width)
    sources = SimpleNamespace(
        rows=[
            SimpleNamespace(
                sample_id=f"sample-{row}",
                model=SimpleNamespace(sha256=f"model-{row}"),
            )
            for row in range(rows)
        ]
    )
    monkeypatch.setattr(
        runner.HistoryHardSources,
        "discover",
        classmethod(lambda _cls: sources),
    )
    features = {}
    for name, path in (("cue", cue_path), ("current", current_path)):
        features[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": runner.sha256_file(path),
        }
    result = {
        "schema": f"{runner.EXPERIMENT_ID}/cache-v1",
        "target_manifest_opened": False,
        "target_npz_opened": [],
        "fresh_data_opened": False,
        "gates": {key: True for key in runner.CACHE_GATE_KEYS},
        "rows": rows,
        "camera_order": list(runner.CAMERA_ORDER),
        "frame_batch_size": runner.FRAME_BATCH_SIZE,
        "sample_ids": [row.sample_id for row in sources.rows],
        "source_model_sha256": [row.model.sha256 for row in sources.rows],
        "features": features,
    }
    monkeypatch.setattr(
        runner,
        "_read_receipt",
        lambda *_args, **_kwargs: (result, {"sha256": "receipt"}),
    )
    arrays, sample_ids, loaded, identity = runner._load_cache(tmp_path, "a" * 64)
    assert arrays["cue"].shape == cue.shape
    assert arrays["current"].shape == current.shape
    assert arrays["cue"].dtype == arrays["current"].dtype == np.float32
    assert sample_ids == ("sample-0", "sample-1")
    assert loaded is result and identity == {"sha256": "receipt"}

    current_path.chmod(0o644)
    current_path.write_bytes(current_path.read_bytes() + b"tamper")
    with pytest.raises(PermissionError, match="current identity changed"):
        runner._load_cache(tmp_path, "a" * 64)


class _FakeTargetArchive:
    def __init__(self, row: int, accesses: list[tuple[int, str]]):
        self.files = list(runner.TARGET_FIELDS)
        self.row = row
        self.accesses = accesses

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getitem__(self, key: str):
        self.accesses.append((self.row, key))
        pair = runner.PAIR_CODES[self.row % runner.ROWS_PER_GROUP]
        readable = {
            "row_id": np.asarray(self.row),
            "group_id": np.asarray(self.row // runner.ROWS_PER_GROUP),
            "pair_code": np.asarray(pair),
            "ordered_path_ids": np.asarray([int(pair[0]), int(pair[1])], dtype=np.int64),
        }
        if key not in readable:
            raise AssertionError(f"non-allowlisted target field was read: {key}")
        return readable[key]


def test_overfit_label_reader_uses_exact_group0_and_field_allowlists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accesses: list[tuple[int, str]] = []
    verified: list[int] = []
    rows = []
    for row in range(runner.ROWS):
        target = SimpleNamespace(
            path=tmp_path / f"row-{row}.npz",
            bytes=100 + row,
            sha256=f"{row:064x}",
            verify=lambda *, root, row=row: verified.append(row),
        )
        rows.append(SimpleNamespace(target=target))
    monkeypatch.setattr(
        runner.HistoryHardSources,
        "discover",
        classmethod(lambda _cls: SimpleNamespace(rows=rows)),
    )

    def fake_load(path: Path, *, allow_pickle: bool):
        assert allow_pickle is False
        row = int(Path(path).stem.split("-")[-1])
        return _FakeTargetArchive(row, accesses)

    monkeypatch.setattr(runner.np, "load", fake_load)
    labels, opened = runner._load_labels(tuple(range(6)), "overfit")
    assert verified == list(range(6))
    assert [record["row_id"] for record in opened] == list(range(6))
    assert np.all(labels[:6] >= 0) and np.all(labels[6:] == -1)
    assert {row for row, _key in accesses} == set(range(6))
    assert {key for _row, key in accesses} == {
        "row_id",
        "group_id",
        "pair_code",
        "ordered_path_ids",
    }


def test_label_phases_reject_dev_rows_for_preflight_and_overfit() -> None:
    assert runner._validate_label_rows(tuple(range(6)), "overfit") == tuple(range(6))
    for phase in ("preflight", "overfit"):
        with pytest.raises(PermissionError, match="exact frozen target rows"):
            runner._validate_label_rows(runner.DEV_ROWS, phase)


def test_preflight_and_overfit_code_cannot_open_dev_or_target_manifest() -> None:
    assert "TARGET_MANIFEST" not in runner.__dict__
    for function_name in ("run_preflight", "run_overfit"):
        function = getattr(runner, function_name)
        source = inspect.getsource(function)
        assert "DEV_ROWS" not in source
        assert "TARGET_MANIFEST" not in source


def test_cli_has_no_legacy_train_audit_and_unimplemented_phases_fail_closed(
    tmp_path: Path,
) -> None:
    assert "train_audit" not in runner.PHASES
    with pytest.raises(SystemExit):
        runner._parse_args(["--phase", "train_audit", "--output-root", str(tmp_path / "legacy")])
    for phase in ("train", "dev"):
        if phase not in runner.PHASES:
            with pytest.raises(SystemExit):
                runner._parse_args(["--phase", phase, "--output-root", str(tmp_path / phase)])


def test_source_contract_is_dedicated_and_protocol_freezes_mamba_shape() -> None:
    assert runner.BASE_COMMIT == "2f792adfd8b5f44ff4a0a3d2c8401a77cdb49d46"
    assert (
        runner.EXPECTED_BRANCH == "codex/g1-mikasa-history-hard-learned-spatial-resampler-mamba-r1"
    )
    assert runner.SOURCE_PATHS == (
        "gr00t/experimental/mikasa_history_hard_learned_spatial_resampler_mamba.py",
        "gr00t/experimental/run_mikasa_history_hard_learned_spatial_resampler_mamba.py",
        "tests/test_mikasa_history_hard_learned_spatial_resampler_mamba.py",
    )
    protocol = runner.protocol()
    assert protocol["full_grid_shape"] == [130, 2, 64, 2048]
    assert protocol["sequence_order"] == "time-major,camera-major,query-major"
    assert protocol["sequence_length"] == 1040
    assert protocol["boundary_indices"] == [519, 1039]
    assert protocol["parameter_limit"] == 2_500_000
    assert protocol["mamba"] == {
        "layers": 2,
        "d_model": 128,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
        "chunk_size": 16,
        "use_mem_eff_path": False,
        "causal": True,
    }
    assert protocol["action_outputs"] == 0


def test_repo_root_must_match_imported_runner(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="differs from imported runner"):
        runner._bind_repo_root(tmp_path)


def test_phase_receipt_rejects_missing_empty_or_nonboolean_gates(tmp_path: Path) -> None:
    def write_receipt(name: str, gates: dict) -> tuple[Path, str]:
        path = tmp_path / f"{name}.json"
        payload = runner.canonical_json_bytes(
            {
                "schema": f"{runner.EXPERIMENT_ID}/synthetic-v1",
                "status": "passed",
                "mode": "synthetic",
                "gates": gates,
                "provenance": {"source": {"sha256": "a" * 64}},
                "target_manifest_opened": False,
                "target_npz_opened": [],
                "fresh_data_opened": False,
            }
        )
        path.write_bytes(payload)
        path.chmod(0o444)
        return path, runner.sha256_bytes(payload)

    for name, gates in (
        ("missing", {}),
        ("nonboolean", {"required": 1}),
        ("extra", {"required": True, "unexpected": True}),
    ):
        path, digest = write_receipt(name, gates)
        with pytest.raises(PermissionError, match="exact gates"):
            runner._require_phase_receipt(
                path,
                digest,
                mode="synthetic",
                schema=f"{runner.EXPERIMENT_ID}/synthetic-v1",
                expected_gate_keys=("required",),
                provenance_key="source",
                provenance_value={"sha256": "a" * 64},
            )

    valid_path, valid_digest = write_receipt("valid-sorted", {"alpha": True, "zeta": True})
    value, _receipt = runner._require_phase_receipt(
        valid_path,
        valid_digest,
        mode="synthetic",
        schema=f"{runner.EXPERIMENT_ID}/synthetic-v1",
        expected_gate_keys=("zeta", "alpha"),
        provenance_key="source",
        provenance_value={"sha256": "a" * 64},
    )
    assert value["gates"] == {"alpha": True, "zeta": True}


def test_five_accumulation_microbatches_share_one_slot_local_shuffle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeShuffleModel(nn.Module):
        condition = "mamba_shuffle"

        def __init__(self):
            super().__init__()
            self.class_bias = nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
            self.permutations: list[torch.Tensor] = []

        def forward(
            self,
            cue: torch.Tensor,
            current: torch.Tensor,
            *,
            shuffle_seed: int | None,
            shuffle_step: int | None,
        ) -> learned.LearnedSpatialOutput:
            del current
            assert shuffle_seed is not None and shuffle_step is not None
            self.permutations.append(
                learned.slot_shuffle_permutation(shuffle_seed, shuffle_step).clone()
            )
            logits = self.class_bias.view(1, 1, 3).expand(cue.shape[0], 2, 3)
            return learned.LearnedSpatialOutput(
                logits=logits,
                sequence=torch.zeros(cue.shape[0], 1, 1),
                permutation=self.permutations[-1],
            )

    model = FakeShuffleModel()
    target = torch.zeros(6, 2, dtype=torch.int64)
    monkeypatch.setattr(
        runner,
        "_group_tensors",
        lambda _arrays, _labels, _group, _device: (
            torch.ones(6, 1),
            torch.ones(6, 1),
            target,
        ),
    )
    monkeypatch.setattr(runner, "_module_parameters", lambda _model: {"fake": [model.class_bias]})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    record, gradients = runner._optimizer_step(
        model,
        optimizer,
        {},
        np.zeros((runner.ROWS, 2), dtype=np.int64),
        groups=(0, 1, 2, 3, 4),
        step=7,
        total_steps=runner.PREFLIGHT_STEPS,
        device=torch.device("cpu"),
    )
    assert len(model.permutations) == runner.ACCUMULATION == 5
    assert all(torch.equal(model.permutations[0], value) for value in model.permutations[1:])
    assert torch.all(model.permutations[0][:65] < 65)
    assert torch.all(model.permutations[0][65:] >= 65)
    assert gradients == {"fake": True}
    assert record["step"] == 8
