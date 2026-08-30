from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.experiment.distributed_evidence import (
    EvidenceTrainerMixin,
    FiniteTrainingMonitor,
    _dataset_contract_summary,
    record_posttrain,
    record_pretrain,
)
import pytest
import torch


class _TrainabilityToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.zeros(()))
        self.weight = torch.nn.Parameter(torch.ones((2,)))

    @staticmethod
    def trainability_evidence_contract():
        return {
            "mode": "toy",
            "parameter_names": ["gate", "weight"],
            "gate_parameter_name": "gate",
            "expected_numel": 3,
            "required_gradient_steps": 2,
            "gradient_capture_steps": 3,
        }


class _BackwardTrainer:
    def training_step(self, model, inputs):
        loss = inputs["loss_fn"](model)
        loss.backward()
        return loss.detach()


class _EvidenceBackwardTrainer(EvidenceTrainerMixin, _BackwardTrainer):
    pass


def _trainer_with_monitor(model, monitor):
    trainer = _EvidenceBackwardTrainer()
    trainer.model = model
    trainer.args = SimpleNamespace(gradient_accumulation_steps=1)
    monitor.bind_trainer(trainer)
    return trainer


def test_trainability_evidence_requires_gate_then_upstream_gradient():
    model = _TrainabilityToy()
    monitor = FiniteTrainingMonitor()
    trainer = _trainer_with_monitor(model, monitor)

    trainer.training_step(
        model,
        {"loss_fn": lambda value: value.gate + 0.0 * value.weight.sum()},
    )
    model.zero_grad(set_to_none=True)  # Simulate ZeRO clearing .grad before the callback.
    monitor.on_pre_optimizer_step(None, None, None)
    with torch.no_grad():
        model.gate.add_(0.1)

    trainer.training_step(model, {"loss_fn": lambda value: value.gate + value.weight.sum()})
    model.zero_grad(set_to_none=True)
    monitor.on_pre_optimizer_step(None, None, None)
    with torch.no_grad():
        model.gate.add_(0.1)
        model.weight.add_(0.1)

    evidence = monitor.summary(expected_steps=2)["trainability"]
    assert evidence["complete"] is True
    assert evidence["initial_gate"] == 0.0
    assert evidence["final_gate"] == pytest.approx(0.2)
    assert evidence["first_gradient_steps"][0]["non_gate_nonzero_count"] == 0
    assert evidence["first_gradient_steps"][1]["non_gate_nonzero_count"] == 1
    assert monitor._trainability_hook_handles == []


def test_trainability_autograd_hook_requires_no_gradient_accumulation():
    model = _TrainabilityToy()
    monitor = FiniteTrainingMonitor()
    with pytest.raises(ValueError, match="accumulation=1"):
        monitor.bind_trainer(
            SimpleNamespace(model=model, args=SimpleNamespace(gradient_accumulation_steps=2))
        )


@pytest.mark.parametrize("failure", ["duplicate", "nonfinite"])
def test_trainability_autograd_hook_rejects_duplicate_or_nonfinite(failure):
    model = _TrainabilityToy()
    monitor = FiniteTrainingMonitor()
    monitor.bind_trainer(
        SimpleNamespace(model=model, args=SimpleNamespace(gradient_accumulation_steps=1))
    )

    monitor.begin_training_step()
    if failure == "duplicate":
        monitor._capture_trainability_gradient("gate", torch.ones_like(model.gate))
    coefficient = float("nan") if failure == "nonfinite" else 1.0
    (coefficient * model.gate + 0.0 * model.weight.sum()).backward()
    monitor.end_training_step()
    model.zero_grad(set_to_none=True)
    monitor.on_pre_optimizer_step(None, None, None)
    with torch.no_grad():
        model.gate.add_(0.1)

    monitor.begin_training_step()
    (model.gate + model.weight.sum()).backward()
    monitor.end_training_step()
    model.zero_grad(set_to_none=True)
    monitor.on_pre_optimizer_step(None, None, None)
    with torch.no_grad():
        model.gate.add_(0.1)
        model.weight.add_(0.1)

    evidence = monitor.summary(expected_steps=2)["trainability"]
    assert evidence["complete"] is False
    gate = evidence["first_gradient_steps"][0]["gate"]
    if failure == "duplicate":
        assert gate["call_count"] == 2
    else:
        assert gate["finite"] is False


@pytest.mark.parametrize(
    "view_manifest_name",
    [
        "g1_id10_h40_native30_view_manifest.json",
        "g1_all_lerobot_h40_view_manifest.json",
    ],
)
def test_dataset_contract_discovers_loader_view_manifest(tmp_path, view_manifest_name):
    dataset_root = tmp_path / "loader-view"
    meta = dataset_root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        '{"total_episodes": 2, "total_frames": 10, "fps": 30, '
        '"splits": {"train": "0:2"}, "features": {'
        '"observation.state": {"shape": [26]}, "action": {"shape": [26]}}}',
        encoding="utf-8",
    )
    view_manifest = meta / view_manifest_name
    view_manifest.write_text(
        '{"schema_version": "loader-view-v1", "source": "/read/only/source"}',
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        dataset_paths=[str(dataset_root)],
        dataset_adapter="generic-adapter",
        adapter_config_path=None,
        embodiment_tag="new_embodiment",
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            datasets=[spec],
            modality_configs={"new_embodiment": {"video": [0]}},
        )
    )
    leaf = SimpleNamespace(action_horizon=40)
    train_dataset = SimpleNamespace(
        datasets=[leaf],
        get_dataset_statistics=lambda: {"new_embodiment": {"action": {}}},
    )

    summary = _dataset_contract_summary(config, train_dataset)

    entry = summary["datasets"][0]
    assert entry["loader_view_manifest_path"] == str(view_manifest)
    assert entry["loader_view_manifest"]["source"] == "/read/only/source"
    assert len(entry["loader_view_manifest_sha256"]) == 64


def test_posttrain_rank_evidence_contains_actual_dataset_sampling():
    sampling_evidence = {
        "rank": 0,
        "world_size": 1,
        "iterator_worker_id": 0,
        "iterator_num_workers": 1,
        "rank_local_counts_available": True,
        "total_samples_yielded": 32,
        "datasets": [
            {
                "dataset_index": 0,
                "dataset_path": "/loader/source_a",
                "sampling_weight": 0.25,
                "normalized_sampling_weight": 0.25,
                "samples_yielded": 8,
            },
            {
                "dataset_index": 1,
                "dataset_path": "/loader/source_b",
                "sampling_weight": 0.75,
                "normalized_sampling_weight": 0.75,
                "samples_yielded": 24,
            },
        ],
    }
    train_dataset = MagicMock()
    train_dataset.get_sampling_evidence.return_value = sampling_evidence
    trainer = SimpleNamespace(
        deepspeed=None,
        model_wrapped=None,
        state=SimpleNamespace(global_step=2),
        train_dataset=train_dataset,
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            num_gpus=1,
            use_ddp=False,
            max_steps=2,
            dataloader_num_workers=0,
        )
    )
    monitor = MagicMock()
    monitor.summary.return_value = {"complete": True}
    gathered = {}

    def capture_summary(output_dir, phase, local_summary, equal_fields):
        gathered.update(local_summary)

    with (
        patch("gr00t.experiment.distributed_evidence.evidence_enabled", return_value=True),
        patch(
            "gr00t.experiment.distributed_evidence._gather_verify_write",
            side_effect=capture_summary,
        ),
        patch(
            "gr00t.experiment.distributed_evidence._nccl_flight_recorder_summary",
            return_value={"required": False, "complete": None},
        ),
    ):
        record_posttrain(
            output_dir=MagicMock(),
            config=config,
            model=torch.nn.Linear(2, 1),
            trainer=trainer,
            monitor=monitor,
        )

    assert gathered["dataset_sampling"] == {
        "available": True,
        "dataloader_num_workers": 0,
        "rank_local_counts_exact": True,
        "evidence": sampling_evidence,
    }


def test_posttrain_uses_pretrain_cached_zero_stage_when_config_file_disappears(tmp_path):
    deepspeed_config = tmp_path / "zero2_config.json"
    deepspeed_config.write_text('{"zero_optimization": {"stage": 2}}', encoding="utf-8")
    config = SimpleNamespace(
        training=SimpleNamespace(
            num_gpus=2,
            use_ddp=False,
            max_steps=2,
            dataloader_num_workers=0,
        ),
        get_deepspeed_config=MagicMock(
            side_effect=lambda: __import__("json").loads(
                deepspeed_config.read_text(encoding="utf-8")
            )
        ),
    )
    trainer = SimpleNamespace(
        deepspeed=SimpleNamespace(zero_optimization_stage=lambda: 2),
        model_wrapped=None,
        state=SimpleNamespace(global_step=2),
        train_dataset=SimpleNamespace(),
    )
    monitor = MagicMock()
    monitor.summary.return_value = {"complete": True}
    summaries = []

    with (
        patch("gr00t.experiment.distributed_evidence.evidence_enabled", return_value=True),
        patch(
            "gr00t.experiment.distributed_evidence._dataset_contract_summary",
            return_value={},
        ),
        patch(
            "gr00t.experiment.distributed_evidence._gather_verify_write",
            side_effect=lambda output_dir, phase, local_summary, equal_fields: summaries.append(
                local_summary
            ),
        ),
        patch(
            "gr00t.experiment.distributed_evidence._nccl_flight_recorder_summary",
            return_value={"required": False, "complete": None},
        ),
    ):
        model = torch.nn.Linear(2, 1)
        record_pretrain(tmp_path, config, model, SimpleNamespace())
        deepspeed_config.unlink()
        record_posttrain(tmp_path, config, model, trainer, monitor)

    assert config.get_deepspeed_config.call_count == 1
    assert [summary["declared_deepspeed_zero_stage"] for summary in summaries] == [2, 2]
