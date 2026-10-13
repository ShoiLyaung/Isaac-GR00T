from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import types
from types import SimpleNamespace

from gr00t.experimental import (
    mikasa_history_hard_pretrained_vlm_cue_audit as audit,
    run_mikasa_history_hard_pretrained_vlm_cue_audit as runner,
)
import numpy as np
import pytest
import torch


def _small_descriptor_contract(monkeypatch: pytest.MonkeyPatch, dim: int = 4) -> None:
    monkeypatch.setattr(audit, "EMBED_DIM", dim)
    monkeypatch.setattr(audit, "TOKENS_PER_IMAGE", 2)


def test_public_vision_boundary_is_merged_and_accepts_no_language_state_or_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_descriptor_contract(monkeypatch)
    calls = []

    class Owner:
        def get_image_features(self, *, pixel_values, image_grid_thw):
            calls.append((pixel_values.clone(), image_grid_thw.clone()))
            return (
                [
                    torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]),
                    torch.tensor([[2.0, 4.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0]]),
                ],
                [torch.ones(4, 4)],
            )

    backbone = SimpleNamespace(model=Owner())
    grid = torch.tensor([[1, 16, 16], [1, 16, 16]])
    value, receipt = audit.encode_public_merged_descriptors(
        backbone,
        torch.ones(2, 3),
        grid,
        frames=1,
        hash_tokens=True,
    )
    assert value.shape == (1, 2, 4)
    assert torch.equal(value[0, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(value[0, 1], torch.tensor([3.0, 5.0, 7.0, 9.0]))
    assert receipt["merged_tokens_per_image"] == 2
    assert len(calls) == 1
    assert set(inspect.signature(audit.encode_public_merged_descriptors).parameters) == {
        "backbone",
        "pixel_values",
        "image_grid_thw",
        "frames",
        "hash_tokens",
    }
    assert len(receipt["merged_token_sha256"]) == 2


def test_public_boundary_rejects_grid_shape_and_nonfinite_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_descriptor_contract(monkeypatch)
    owner = SimpleNamespace(
        get_image_features=lambda **_kwargs: ([torch.full((2, 4), float("nan"))] * 2, [])
    )
    backbone = SimpleNamespace(model=owner)
    with pytest.raises(audit.CueAuditContractError, match="image_grid"):
        audit.encode_public_merged_descriptors(
            backbone,
            torch.ones(2, 3),
            torch.tensor([[1, 8, 8], [1, 8, 8]]),
            frames=1,
        )
    with pytest.raises(audit.CueAuditContractError, match="finite"):
        audit.encode_public_merged_descriptors(
            backbone,
            torch.ones(2, 3),
            torch.tensor([[1, 16, 16], [1, 16, 16]]),
            frames=1,
        )


def test_native_packaging_is_time_major_camera_minor_and_contains_no_state_or_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    def fake_apply(_transform, images):
        tensors = [torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1) for image in images]
        return tensors, None

    module = types.ModuleType("gr00t.model.gr00t_n1d7.image_augmentations")
    module.apply_with_replay = fake_apply
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Processor:
        eval_image_transform = object()

        def _apply_vlm_processing(self, images, prompt):
            seen["images"] = images.clone()
            seen["prompt"] = prompt
            return {"vlm_content": {"text": prompt, "images": list(images)}}

        def collator(self, features):
            assert set(features[0]) == {"vlm_content"}
            return {
                "inputs": {
                    "pixel_values": torch.ones(4, 3),
                    "image_grid_thw": torch.tensor([[1, 16, 16]] * 4),
                }
            }

    rgb = np.zeros((2, 2, 128, 128, 3), dtype=np.uint8)
    rgb[0, 0].fill(1)
    rgb[0, 1].fill(2)
    rgb[1, 0].fill(3)
    rgb[1, 1].fill(4)
    pixels, grid = runner._package_rgb(Processor(), rgb, runner.PROMPT)
    assert pixels.shape == (4, 3)
    assert grid.tolist() == [[1, 16, 16]] * 4
    assert seen["prompt"] == runner.PROMPT
    assert [int(frame[0, 0, 0]) for frame in seen["images"]] == [1, 2, 3, 4]


def test_condition_controls_preserve_shape_and_shuffle_is_sample_slot_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_descriptor_contract(monkeypatch, dim=2)
    rng = np.random.default_rng(7)
    cue = rng.normal(size=(3, 130, 2, 2)).astype(np.float32)
    current = rng.normal(size=(3, 2, 2)).astype(np.float32)
    sample_ids = tuple(f"{index:064x}" for index in range(3))
    primary = audit.condition_slots(cue, current, sample_ids, audit.PRIMARY_CONDITION)
    current_repeat = audit.condition_slots(cue, current, sample_ids, "current_repeat")
    endpoint = audit.condition_slots(cue, current, sample_ids, "endpoint_repeat")
    first = audit.condition_slots(
        cue, current, sample_ids, "sample_shuffle", shuffle_seed=audit.SHUFFLE_SEEDS[0]
    )
    repeated = audit.condition_slots(
        cue, current, sample_ids, "sample_shuffle", shuffle_seed=audit.SHUFFLE_SEEDS[0]
    )
    second = audit.condition_slots(
        cue, current, sample_ids, "sample_shuffle", shuffle_seed=audit.SHUFFLE_SEEDS[1]
    )
    assert primary.shape == current_repeat.shape == endpoint.shape == (3, 2, 65, 2, 2)
    assert np.count_nonzero(current_repeat) == 0
    assert np.all(endpoint[:, :, 0] == endpoint[:, :, -1])
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)


def test_standardizer_is_train_only_and_requires_complete_geometry_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_descriptor_contract(monkeypatch, dim=2)
    slots = np.zeros((48, 2, 65, 2, 2), dtype=np.float32)
    slots[:36, ..., 0] = np.arange(36, dtype=np.float32)[:, None, None, None]
    candidate_rows = tuple(range(6, 36))
    standardizer = audit.fit_standardizer(slots, candidate_rows)
    standardizer.validate()
    assert standardizer.mean[0, 0] == pytest.approx(np.mean(np.arange(6, 36)))
    with pytest.raises(audit.CueAuditContractError, match="train-only"):
        audit.fit_standardizer(slots, tuple(range(1, 36)))
    with pytest.raises(audit.CueAuditContractError, match="train-only"):
        audit.fit_standardizer(slots, tuple(range(6, 42)))


def test_train_loo_fits_only_the_five_candidate_geometries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_descriptor_contract(monkeypatch, dim=2)
    rng = np.random.default_rng(11)
    cue = rng.normal(size=(48, 130, 2, 2)).astype(np.float32)
    current = rng.normal(size=(48, 2, 2)).astype(np.float32)
    sample_ids = tuple(f"{index:064x}" for index in range(48))
    labels = np.tile(np.asarray([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]), (8, 1))
    groups = np.arange(48, dtype=np.int64) // 6
    observed = []
    original = audit.fit_standardizer

    def record(slots, rows):
        observed.append(tuple(rows))
        return original(slots, rows)

    monkeypatch.setattr(runner, "fit_standardizer", record)
    evidence = runner._train_loo_evidence(cue, current, sample_ids, labels, groups)
    assert set(evidence) == set(audit.ALL_CONDITIONS)
    assert len(observed) == 48
    for rows in observed:
        assert len(rows) == 30
        assert all(row in audit.TRAIN_ROWS for row in rows)
        assert len({row // 6 for row in rows}) == 5


def test_train_shuffle_reuses_ordered_fold_standardizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_descriptor_contract(monkeypatch, dim=2)
    rng = np.random.default_rng(19)
    cue = rng.normal(size=(48, 130, 2, 2)).astype(np.float32)
    current = rng.normal(size=(48, 2, 2)).astype(np.float32)
    sample_ids = tuple(f"{index:064x}" for index in range(48))
    labels = np.tile(np.asarray([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]), (8, 1))
    groups = np.arange(48, dtype=np.int64) // 6
    primary = audit.condition_slots(cue, current, sample_ids, audit.PRIMARY_CONDITION)
    seen = []
    original = audit.fit_standardizer

    def record(slots, rows):
        seen.append(np.array_equal(slots, primary))
        return original(slots, rows)

    monkeypatch.setattr(runner, "fit_standardizer", record)
    runner._train_loo_evidence(cue, current, sample_ids, labels, groups)
    # Four gated/descriptive conditions x six folds, then four shuffle seeds x
    # six folds. Every shuffle fit must use the primary ordered tensor.
    assert len(seen) == 48
    assert all(seen[-24:])


def test_neighbor_mapping_is_label_blind_and_ties_use_sample_id_then_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_descriptor_contract(monkeypatch, dim=1)
    slots = np.zeros((3, 2, 65, 2, 1), dtype=np.float32)
    groups = np.asarray([0, 1, 2], dtype=np.int64)
    sample_ids = ("f" * 64, "b" * 64, "a" * 64)
    mapping = audit.build_neighbor_mapping(
        slots,
        sample_ids,
        groups,
        query_rows=(0,),
        candidate_rows=(1, 2),
    )
    assert [record["candidate_sample_id"] for record in mapping] == ["a" * 64, "a" * 64]
    assert [record["candidate_slot"] for record in mapping] == [0, 0]
    assert "labels" not in inspect.signature(audit.build_neighbor_mapping).parameters


def _score_section(pair: float, slot0: float, slot1: float, groups=("0", "1")) -> dict:
    values = {
        group: {
            "rows": 6,
            "pair_accuracy": pair,
            "slot0_accuracy": slot0,
            "slot1_accuracy": slot1,
        }
        for group in groups
    }
    return {
        "records": [],
        "by_group": values,
        "group_equal": {
            "pair_accuracy": pair,
            "slot0_accuracy": slot0,
            "slot1_accuracy": slot1,
        },
    }


def test_audit_gates_apply_absolute_control_margin_and_group_direction() -> None:
    evidence = {
        audit.PRIMARY_CONDITION: _score_section(0.90, 0.96, 0.96),
        "current_repeat": _score_section(0.20, 0.60, 0.60),
        "endpoint_repeat": _score_section(0.20, 0.60, 0.60),
        "sample_shuffle": _score_section(0.20, 0.60, 0.60),
        audit.DESCRIPTIVE_CONDITION: _score_section(0.70, 0.85, 0.85),
    }
    gates = audit.audit_gates(evidence, expected_groups=2)
    assert all(gates.values())
    evidence["endpoint_repeat"] = _score_section(0.45, 0.75, 0.75)
    assert not audit.audit_gates(evidence, expected_groups=2)[
        "primary_vs_endpoint_repeat_pair_plus_50"
    ]


class _FakeArchive:
    def __init__(self, row: int, accesses: list[tuple[int, str]]):
        self.files = list(runner.TARGET_FIELDS)
        self.row = row
        self.accesses = accesses

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getitem__(self, key):
        self.accesses.append((self.row, key))
        pair = runner.PAIR_CODES[self.row % runner.ROWS_PER_GROUP]
        values = {
            "row_id": np.asarray(self.row),
            "group_id": np.asarray(self.row // runner.ROWS_PER_GROUP),
            "pair_code": np.asarray(pair),
            "ordered_path_ids": np.asarray([int(pair[0]), int(pair[1])], dtype=np.int64),
        }
        if key not in values:
            raise AssertionError(f"oracle/unused target field was read: {key}")
        return values[key]


def test_train_target_reader_opens_only_rows_0_35_and_never_target_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accesses: list[tuple[int, str]] = []
    verified = []
    rows = []
    for row in range(48):
        target = SimpleNamespace(
            path=tmp_path / f"row-{row}.npz",
            bytes=10 + row,
            sha256=f"{row:064x}",
            verify=lambda *, root, row=row: verified.append((row, root)),
        )
        rows.append(SimpleNamespace(target=target))
    monkeypatch.setattr(
        runner.HistoryHardSources,
        "discover",
        classmethod(lambda _cls: SimpleNamespace(rows=rows)),
    )

    def fake_load(path, *, allow_pickle):
        assert allow_pickle is False
        row = int(Path(path).stem.split("-")[-1])
        return _FakeArchive(row, accesses)

    monkeypatch.setattr(runner.np, "load", fake_load)
    labels, opened = runner._load_labels(runner.TRAIN_ROWS, "train_audit")
    assert [row for row, _root in verified] == list(range(36))
    assert [record["row_id"] for record in opened] == list(range(36))
    assert np.all(labels[:36] >= 0) and np.all(labels[36:] == -1)
    assert {row for row, _key in accesses} == set(range(36))
    assert {key for _row, key in accesses} == {
        "row_id",
        "group_id",
        "pair_code",
        "ordered_path_ids",
    }
    assert "TARGET_MANIFEST" not in runner.__dict__
    with pytest.raises(PermissionError, match="exact frozen target rows"):
        runner._load_labels(tuple(range(37)), "train_audit")


def test_dev_gate_failure_prevents_any_dev_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = []
    monkeypatch.setattr(
        runner,
        "_load_labels",
        lambda *_args, **_kwargs: opened.append(True),
    )
    with pytest.raises(PermissionError, match="train-only"):
        runner.run_dev(
            tmp_path / "dev",
            cue=np.empty(0),
            current=np.empty(0),
            sample_ids=(),
            train={"gates": {"failed": False}},
            train_root=tmp_path,
            provenance={},
        )
    assert opened == []
    assert not (tmp_path / "dev").exists()


def test_source_contract_adds_only_dedicated_files_and_no_core_model_diff() -> None:
    assert runner.BASE_COMMIT == "7aa3ec5a31d6c3276abda0e660c378ce939aa70b"
    assert runner.PHASES == ("source_acceptance", "layout", "cache", "train_audit", "dev")
    assert runner.SOURCE_PATHS == (
        "gr00t/experimental/mikasa_history_hard_pretrained_vlm_cue_audit.py",
        "gr00t/experimental/run_mikasa_history_hard_pretrained_vlm_cue_audit.py",
        "tests/test_mikasa_history_hard_pretrained_vlm_cue_audit.py",
    )
    assert runner.protocol()["public_boundary"] == "model.backbone.model.get_image_features"
    assert runner.protocol()["action_outputs"] == 0


def test_train_phase_validates_cache_before_label_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    source = {
        "schema": f"{runner.EXPERIMENT_ID}/source-acceptance-v1",
        "status": "passed",
        "mode": "source_acceptance",
        "source_identity": {},
        "protocol": runner.protocol(),
        "target_manifest_opened": False,
        "fresh_data_opened": False,
    }
    layout = {
        "schema": f"{runner.EXPERIMENT_ID}/layout-v1",
        "status": "passed",
        "mode": "layout_probe",
        "gates": {"passed": True},
        "provenance": {"source_acceptance": {"mode": "source_acceptance"}},
    }

    def receipt(_path, _sha, mode):
        return (source if mode == "source_acceptance" else layout), {"mode": mode}

    monkeypatch.setattr(runner, "_read_receipt", receipt)
    monkeypatch.setattr(runner, "bind_runtime_source", lambda *_args: {})

    def cache_first(*_args, **_kwargs):
        calls.append("cache")
        raise PermissionError("cache invalid")

    monkeypatch.setattr(runner, "_load_cache", cache_first)
    monkeypatch.setattr(
        runner,
        "run_train_audit",
        lambda *_args, **_kwargs: calls.append("labels"),
    )
    with pytest.raises(PermissionError, match="cache invalid"):
        runner.main(
            [
                "--phase",
                "train_audit",
                "--output-root",
                str(tmp_path / "out"),
                "--source-result",
                str(tmp_path / "source"),
                "--source-sha256",
                "a" * 64,
                "--layout-result",
                str(tmp_path / "layout"),
                "--layout-sha256",
                "b" * 64,
                "--cache-root",
                str(tmp_path / "cache"),
                "--cache-sha256",
                "c" * 64,
            ]
        )
    assert calls == ["cache"]


def test_existing_source_output_fails_before_git_or_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    calls = []
    monkeypatch.setattr(runner, "source_identity", lambda *_args: calls.append("git"))
    with pytest.raises(FileExistsError):
        runner.run_source_acceptance(output)
    assert calls == []


def test_regular_identity_rehashes_same_size_material(tmp_path: Path) -> None:
    material = tmp_path / "weights.safetensors"
    material.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    identity = runner._regular_identity(material, 3, expected)
    assert identity["sha256"] == expected
    material.write_bytes(b"abd")
    with pytest.raises(RuntimeError, match="hash changed"):
        runner._regular_identity(material, 3, expected)
