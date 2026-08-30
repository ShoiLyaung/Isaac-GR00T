from __future__ import annotations

import json
from types import SimpleNamespace

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.codec_evidence.episode_bundle import make_synthetic_episode_bundle
from gr00t.data.codec_evidence.processor import CodecEvidenceProcessorCore
from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.model.gr00t_n1d7.codec_dynamics import Gr00tCodecConditioningStack
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import (
    _CODEC_ONLY_PARAMETER_NAMES,
    _CODEC_ONLY_TRAINABLE_NUMEL,
    Gr00tN1d7,
    Gr00tN1d7ActionHead,
    _parse_optional_codec_model_config as parse_action_codec_config,
)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    Gr00tN1d7DataCollator,
    Gr00tN1d7Processor,
    _parse_optional_codec_model_config as parse_processor_codec_config,
)
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
import numpy as np
import pytest
import torch
from torch import nn
from transformers import Trainer
from transformers.feature_extraction_utils import BatchFeature


def _bare_head(*, enabled: bool, conditioning=None) -> Gr00tN1d7ActionHead:
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    nn.Module.__init__(head)
    head.codec_evidence_enabled = enabled
    head.codec_conditioning = conditioning
    return head


class _NoRead:
    def __contains__(self, key):
        raise AssertionError(f"disabled path read key {key!r}")

    def __getitem__(self, key):
        raise AssertionError(f"disabled path read key {key!r}")


def test_disabled_action_path_does_not_read_allocate_or_consume_rng():
    head = _bare_head(enabled=False)
    action = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    rng_before = torch.random.get_rng_state().clone()

    context = head._encode_codec_context(_NoRead())
    output = head._condition_action_features(action, context)

    assert context is None
    assert output is action
    assert torch.equal(output, action)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


class _CodecStub:
    def encode(self, evidence):
        assert evidence == {"sentinel": 7}
        return torch.zeros((1, 32, 8)), torch.ones((1, 32), dtype=torch.bool)

    def condition(self, action, context, mask):
        return action


def test_enabled_model_has_one_action_input_entry_and_fails_closed_if_missing():
    head = _bare_head(enabled=True, conditioning=_CodecStub())

    context, mask = head._encode_codec_context(
        BatchFeature(data={"codec_evidence": {"sentinel": 7}})
    )
    assert context.shape == (1, 32, 8)
    assert mask.all()

    with pytest.raises(KeyError, match=r"action_input\['codec_evidence'\]"):
        head._encode_codec_context(BatchFeature(data={"codec_evidence_sample": object()}))


class _BatchCodecCollator:
    def __call__(self, values):
        return {"count": torch.tensor(len(values))}


def _bare_collator(*, enabled: bool) -> Gr00tN1d7DataCollator:
    collator = Gr00tN1d7DataCollator.__new__(Gr00tN1d7DataCollator)
    collator.model_name = "unused"
    collator.model_type = "qwen"
    collator.processor = SimpleNamespace()
    if enabled:
        collator.codec_evidence_enabled = True
        collator.codec_evidence_collator = _BatchCodecCollator()
    return collator


def test_training_collator_emits_only_codec_evidence_action_key():
    collator = _bare_collator(enabled=True)
    output = collator(
        [
            {"state": np.zeros((1, 2)), "codec_evidence_sample": object()},
            {"state": np.zeros((1, 2)), "codec_evidence_sample": object()},
        ]
    )["inputs"]

    assert "codec_evidence" in output
    assert "codec_evidence_sample" not in output
    assert output["codec_evidence"]["count"].item() == 2


def test_training_collator_rejects_missing_member_raw_key_and_disabled_mode():
    enabled = _bare_collator(enabled=True)
    with pytest.raises(ValueError, match="missing"):
        enabled(
            [
                {"state": np.zeros((1, 2)), "codec_evidence_sample": object()},
                {"state": np.zeros((1, 2))},
            ]
        )
    with pytest.raises(ValueError, match="Only codec_evidence_sample"):
        enabled([{"codec_evidence": {}}])

    disabled = _bare_collator(enabled=False)
    with pytest.raises(ValueError, match="disabled"):
        disabled([{"codec_evidence_sample": object()}])


def test_online_processor_uses_same_collator_and_fails_closed(synthetic_codec_record):
    sample = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)
    processor._collator = SimpleNamespace(codec_evidence_collator=_BatchCodecCollator())
    processor.codec_evidence = {
        "schema_version": "codec-evidence-record-v2",
        "camera_id": "cam_left_high",
    }

    batch = processor._collate_codec_evidence_observation(
        {"codec_evidence_sample": [sample, sample]}, 2
    )
    assert batch["count"].item() == 2

    with pytest.raises(KeyError, match="codec_evidence_sample"):
        processor._collate_codec_evidence_observation({}, 1)
    with pytest.raises(ValueError, match="batch size"):
        processor._collate_codec_evidence_observation({"codec_evidence_sample": [sample]}, 2)


def test_online_processor_rejects_wrong_schema_camera_and_source(synthetic_codec_record):
    legacy = CodecEvidenceProcessorCore().process(synthetic_codec_record)
    episode = make_synthetic_episode_bundle().view(7)
    processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)
    processor._collator = SimpleNamespace(codec_evidence_collator=_BatchCodecCollator())

    processor.codec_evidence = {
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": episode.key.source_kind,
    }
    with pytest.raises(TypeError, match="requires EpisodeT8View"):
        processor._collate_codec_evidence_observation({"codec_evidence_sample": legacy}, 1)

    processor.codec_evidence = {
        "schema_version": "codec-evidence-record-v2",
        "camera_id": "wrong_camera",
    }
    with pytest.raises(ValueError, match="camera_id"):
        processor._collate_codec_evidence_observation({"codec_evidence_sample": legacy}, 1)

    processor.codec_evidence = {
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": episode.key.camera_id,
        "source_kind": "RAW_JPEG_DERIVED",
    }
    with pytest.raises(ValueError, match="source_kind"):
        processor._collate_codec_evidence_observation({"codec_evidence_sample": episode}, 1)


def _serializable_processor(*, codec_evidence=None) -> Gr00tN1d7Processor:
    processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)
    processor.modality_configs = {}
    processor.image_crop_size = None
    processor.image_target_size = None
    processor.use_albumentations = False
    processor.random_rotation_angle = None
    processor.color_jitter_params = None
    processor.shortest_image_edge = 256
    processor.crop_fraction = 0.95
    processor.letter_box_transform = False
    processor.model_name = "local-model"
    processor.model_type = "qwen"
    processor.formalize_language = True
    processor.max_state_dim = 132
    processor.max_action_dim = 132
    processor.max_action_horizon = 40
    processor.use_percentiles = True
    processor.use_mean_std = False
    processor.clip_outliers = True
    processor.apply_sincos_state_encoding = False
    processor.use_relative_action = False
    processor.exclude_state = False
    processor.state_dropout_prob = 0.0
    processor.state_action_processor = SimpleNamespace(statistics={})
    processor.embodiment_id_mapping = {"new_embodiment": 10}
    if codec_evidence is not None and codec_evidence.get("enabled", False):
        processor.codec_evidence_enabled = True
        processor.codec_evidence = dict(codec_evidence)
    return processor


def test_processor_checkpoint_serializes_only_enabled_codec_config(tmp_path, monkeypatch):
    enabled_config = {
        "enabled": True,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "d_codec": 64,
        "conditioner_heads": 8,
        "gate_init": 0.0,
    }
    enabled_root = tmp_path / "enabled"
    _serializable_processor(codec_evidence=enabled_config).save_pretrained(enabled_root)
    saved = json.loads((enabled_root / "processor_config.json").read_text())
    assert saved["processor_kwargs"]["codec_evidence"] == enabled_config

    native_root = tmp_path / "native"
    _serializable_processor().save_pretrained(native_root)
    native = json.loads((native_root / "processor_config.json").read_text())
    assert "codec_evidence" not in native["processor_kwargs"]

    captured = {}

    def capture_init(self, **kwargs):
        del self
        captured.update(kwargs)

    monkeypatch.setattr(Gr00tN1d7Processor, "__init__", capture_init)
    Gr00tN1d7Processor.from_pretrained(enabled_root)
    assert captured["codec_evidence"] == enabled_config

    override = {**enabled_config, "d_codec": 32}
    captured.clear()
    Gr00tN1d7Processor.from_pretrained(enabled_root, codec_evidence=override)
    assert captured["codec_evidence"] == override


class _InputRecorder(nn.Module):
    def __init__(self, *, codec_enabled: bool | None = None):
        super().__init__()
        if codec_enabled is not None:
            self.codec_evidence_enabled = codec_enabled
        self.seen = None

    def prepare_input(self, inputs):
        self.seen = inputs
        return inputs


def test_model_prepare_input_routes_nested_codec_only_to_action_head():
    model = Gr00tN1d7.__new__(Gr00tN1d7)
    nn.Module.__init__(model)
    model.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
    model.backbone = _InputRecorder()
    model.action_head = _InputRecorder(codec_enabled=True)
    evidence = {"nested": {"tensor": torch.ones((1, 8), dtype=torch.float32)}}
    inputs = {
        "state": torch.zeros((1, 1, 2)),
        "codec_evidence": evidence,
    }

    backbone_inputs, action_inputs = model.prepare_input(inputs)

    assert "codec_evidence" not in model.backbone.seen
    assert "codec_evidence" not in backbone_inputs
    assert model.action_head.seen["codec_evidence"] is evidence
    assert torch.equal(
        action_inputs["codec_evidence"]["nested"]["tensor"],
        evidence["nested"]["tensor"],
    )


def _bare_pipeline(adapter_paths, *, dataset_adapter="g1_all_lerobot_h40_v1") -> Gr00tN1d7Pipeline:
    pipeline = Gr00tN1d7Pipeline.__new__(Gr00tN1d7Pipeline)
    pipeline.config = SimpleNamespace(
        data=SimpleNamespace(
            datasets=[
                SimpleNamespace(
                    dataset_adapter=dataset_adapter,
                    adapter_config_path=str(path),
                )
                for path in adapter_paths
            ]
        )
    )
    return pipeline


def test_setup_requires_one_identical_five_leaf_codec_contract(tmp_path):
    codec_block = {
        "enabled": True,
        "fixture_root": "/tmp/codec-fixture-not-read-by-config-parser",
        "fixture_manifest_sha256": "a" * 64,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "d_codec": 64,
        "conditioner_heads": 8,
    }
    paths = []
    for index in range(5):
        path = tmp_path / f"adapter-{index}.json"
        path.write_text(json.dumps({"codec_evidence": codec_block}))
        paths.append(path)

    resolved = _bare_pipeline(paths)._resolve_codec_evidence_config()

    assert resolved is not None
    assert resolved.to_model_dict()["enabled"] is True
    assert resolved.to_model_dict()["d_codec"] == 64

    paths[-1].write_text(json.dumps({"codec_evidence": {"enabled": False}}))
    with pytest.raises(ValueError, match="Every dataset"):
        _bare_pipeline(paths)._resolve_codec_evidence_config()


def test_setup_disabled_blocks_do_not_parse_fixture_or_consume_rng(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"disabled-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "codec_evidence": {
                        "enabled": False,
                        "fixture_root": "/path/is/intentionally/not/read",
                    }
                }
            )
        )
        paths.append(path)
    rng_before = torch.random.get_rng_state().clone()

    resolved = _bare_pipeline(paths)._resolve_codec_evidence_config()

    assert resolved is None
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_setup_rejects_enabled_codec_on_adapter_factory_does_not_wrap(tmp_path):
    adapter_path = tmp_path / "unsupported-adapter.json"
    adapter_path.write_text(
        json.dumps(
            {
                "codec_evidence": {
                    "enabled": True,
                    "fixture_root": "/tmp/not-read-before-adapter-rejection",
                    "fixture_manifest_sha256": "a" * 64,
                    "schema_version": "codec-evidence-episode-bundle-v1",
                    "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
                }
            }
        )
    )

    with pytest.raises(ValueError, match="supported only for g1_all_lerobot_h40_v1"):
        _bare_pipeline(
            [adapter_path], dataset_adapter="g1_id10_h40_native30_v1"
        )._resolve_codec_evidence_config()

    adapter_path.write_text(json.dumps({"codec_evidence": {"enabled": False}}))
    assert (
        _bare_pipeline(
            [adapter_path], dataset_adapter="g1_id10_h40_native30_v1"
        )._resolve_codec_evidence_config()
        is None
    )


def test_setup_rejects_record_v2_before_model_creation(tmp_path):
    adapter_path = tmp_path / "record-v2-adapter.json"
    adapter_path.write_text(
        json.dumps(
            {
                "codec_evidence": {
                    "enabled": True,
                    "fixture_root": "/tmp/record-v2-not-opened",
                    "fixture_manifest_sha256": "a" * 64,
                    "record_schema_sha256": "b" * 64,
                    "schema_version": "codec-evidence-record-v2",
                }
            }
        )
    )
    pipeline = _bare_pipeline([adapter_path])
    create_model_called = False

    def reject_model_creation():
        nonlocal create_model_called
        create_model_called = True
        raise AssertionError("model creation must not be reached")

    pipeline._create_model = reject_model_creation
    with pytest.raises(ValueError, match="episode-bundle-v1 before model creation"):
        pipeline.setup()
    assert create_model_called is False


def test_setup_rejects_duplicate_adapter_keys_before_model_creation(tmp_path):
    adapter_path = tmp_path / "duplicate-adapter.json"
    adapter_path.write_text(
        '{"codec_evidence":{"enabled":false},"codec_evidence":{"enabled":true}}'
    )
    pipeline = _bare_pipeline([adapter_path])
    create_model_called = False

    def reject_model_creation():
        nonlocal create_model_called
        create_model_called = True
        raise AssertionError("model creation must not be reached")

    pipeline._create_model = reject_model_creation
    with pytest.raises(ValueError, match="Duplicate JSON key: codec_evidence"):
        pipeline.setup()
    assert create_model_called is False


def test_setup_rejects_stale_preseeded_codec_when_adapter_is_native(tmp_path):
    adapter_path = tmp_path / "native-adapter.json"
    adapter_path.write_text(json.dumps({"codec_evidence": {"enabled": False}}))
    pipeline = _bare_pipeline([adapter_path])
    pipeline.config.model = SimpleNamespace(codec_evidence=_model_codec_config())
    create_model_called = False

    def reject_model_creation():
        nonlocal create_model_called
        create_model_called = True
        raise AssertionError("model creation must not be reached")

    pipeline._create_model = reject_model_creation
    with pytest.raises(ValueError, match="requires an enabled all-LeRobot adapter"):
        pipeline.setup()
    assert create_model_called is False


def test_setup_clears_disabled_preseed_before_native_components(tmp_path):
    adapter_path = tmp_path / "native-disabled-adapter.json"
    adapter_path.write_text(json.dumps({"codec_evidence": {"enabled": False}}))
    pipeline = _bare_pipeline([adapter_path])
    pipeline.config.model = SimpleNamespace(
        codec_evidence={"enabled": False, "stale": "must disappear"}
    )
    pipeline.save_cfg_dir = tmp_path
    pipeline._create_model = lambda: object()
    pipeline._create_dataset = lambda save_cfg_dir: (save_cfg_dir, None)
    pipeline._create_collator = lambda: object()

    pipeline.setup()

    assert pipeline.config.model.codec_evidence is None
    assert pipeline.train_dataset == tmp_path


def _checkpoint_identity_pipeline(requested_codec: dict | None) -> tuple:
    pipeline = Gr00tN1d7Pipeline.__new__(Gr00tN1d7Pipeline)
    pipeline.config = SimpleNamespace(model=SimpleNamespace(model_type="Gr00tN1d7"))
    pipeline.transformers_loading_kwargs = {"local_files_only": True}
    codec_config = (
        SimpleNamespace(to_model_dict=lambda: dict(requested_codec))
        if requested_codec is not None
        else None
    )
    return pipeline, codec_config


def _codec_model_stub():
    conditioning = SimpleNamespace(
        state_dict=lambda: {
            "core.weight": object(),
            "projector.bias": object(),
        }
    )
    return SimpleNamespace(action_head=SimpleNamespace(codec_conditioning=conditioning))


def _model_codec_config() -> dict:
    return {
        "enabled": True,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": "NATIVE_CANONICAL_DECODED_DERIVED",
        "d_codec": 64,
        "conditioner_heads": 8,
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: {**value, "schema_version": "bogus-schema"},
        lambda value: {**value, "camera_id": "wrong_camera"},
        lambda value: {**value, "source_kind": None},
        lambda value: {**value, "fixture_root": "/forbidden/model/field"},
        lambda value: {**value, "d_codec": True},
        lambda value: {**value, "conditioner_heads": False},
        lambda value: {**value, "gate_init": True},
        lambda value: {**value, "gate_init": "0.0"},
    ],
)
@pytest.mark.parametrize("parser", [parse_action_codec_config, parse_processor_codec_config])
def test_direct_model_and_processor_configs_reject_bogus_semantics(parser, mutator):
    with pytest.raises((TypeError, ValueError)):
        parser(mutator(_model_codec_config()))


@pytest.mark.parametrize("parser", [parse_action_codec_config, parse_processor_codec_config])
def test_direct_disabled_config_bypasses_model_parser_and_rng(parser, monkeypatch):
    def reject_model_parser(value):
        del value
        raise AssertionError("disabled config must not call the strict model parser")

    monkeypatch.setattr(
        "gr00t.data.codec_evidence.config.parse_codec_model_config",
        reject_model_parser,
    )
    rng_before = torch.random.get_rng_state().clone()
    assert parser({"enabled": False, "gate_init": "ignored", "unknown": object()}) is None
    assert parser({}) is None
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    with pytest.raises(TypeError, match="enabled must be boolean"):
        parser({"enabled": 0})


def test_full_model_rejects_malformed_codec_before_backbone_creation(monkeypatch):
    backbone_factory_called = False

    def reject_backbone_factory(config):
        nonlocal backbone_factory_called
        del config
        backbone_factory_called = True
        raise AssertionError("backbone factory must not be reached")

    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls",
        reject_backbone_factory,
    )
    malformed = {**_model_codec_config(), "gate_init": "0.0"}
    with pytest.raises(TypeError, match="gate_init"):
        Gr00tN1d7(Gr00tN1d7Config(codec_evidence=malformed))
    assert backbone_factory_called is False


@pytest.mark.parametrize(
    ("raw_codec", "enabled"),
    [
        (_model_codec_config(), True),
        ({"enabled": False, "garbage": object()}, False),
        (None, False),
    ],
)
def test_full_model_writes_canonical_codec_config_before_backbone(raw_codec, enabled, monkeypatch):
    config = Gr00tN1d7Config(codec_evidence=raw_codec)

    def stop_after_canonicalization(model_config):
        assert model_config is config
        raise RuntimeError("STOP_AFTER_CANONICALIZATION")

    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls",
        stop_after_canonicalization,
    )
    with pytest.raises(RuntimeError, match="STOP_AFTER_CANONICALIZATION"):
        Gr00tN1d7(config)

    if enabled:
        assert config.codec_evidence == _model_codec_config()
        assert config.codec_evidence is not raw_codec
        assert type(config.codec_evidence["gate_init"]) is float
    else:
        assert config.codec_evidence is None


@pytest.mark.parametrize(
    ("codec_config", "overrides", "message"),
    [
        (_model_codec_config(), {"codec_trainability": "bogus"}, "Unsupported"),
        (None, {"codec_trainability": "codec_only"}, "requires enabled Codec"),
        (
            _model_codec_config(),
            {"codec_trainability": "codec_only", "tune_visual": True},
            "every native tune flag disabled",
        ),
        (
            {**_model_codec_config(), "d_codec": 32},
            {"codec_trainability": "codec_only"},
            "sizing differs",
        ),
    ],
)
def test_codec_only_contract_fails_before_backbone_allocation(
    codec_config, overrides, message, monkeypatch
):
    values = {
        "codec_evidence": codec_config,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": False,
        "tune_diffusion_model": False,
        "tune_vlln": False,
        "tune_top_llm_layers": 0,
        **overrides,
    }
    config = Gr00tN1d7Config(**values)
    backbone_factory_called = False

    def reject_backbone_factory(_config):
        nonlocal backbone_factory_called
        backbone_factory_called = True
        raise AssertionError("backbone allocation must not be reached")

    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls",
        reject_backbone_factory,
    )
    with pytest.raises(ValueError, match=message):
        Gr00tN1d7(config)
    assert backbone_factory_called is False


class _CodecOnlyActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
        self.codec_conditioning = Gr00tCodecConditioningStack(
            action_width=1536,
            d_codec=64,
            conditioner_heads=8,
            gate_init=0.0,
            modalities=("mv", "rpe", "i_frame", "metadata"),
        )

    def set_frozen_modules_to_eval_mode(self):
        self.native.eval()


def _bare_codec_only_model() -> Gr00tN1d7:
    model = Gr00tN1d7.__new__(Gr00tN1d7)
    nn.Module.__init__(model)
    model.config = Gr00tN1d7Config(
        codec_evidence=_model_codec_config(),
        codec_trainability="codec_only",
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        tune_top_llm_layers=0,
    )
    model.codec_trainability = "codec_only"
    model.backbone = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
    model.action_head = _CodecOnlyActionHead()
    model._apply_codec_trainability(_model_codec_config())
    return model


def test_codec_only_closure_and_training_modes_are_exact():
    model = _bare_codec_only_model()
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert tuple(name for name, _parameter in trainable) == _CODEC_ONLY_PARAMETER_NAMES
    assert len(trainable) == 27
    assert sum(parameter.numel() for _name, parameter in trainable) == _CODEC_ONLY_TRAINABLE_NUMEL

    model.train()
    assert model.training is True
    assert model.backbone.training is False
    assert model.action_head.training is True
    assert model.action_head.native.training is False
    assert model.action_head.codec_conditioning.training is True


def test_codec_only_optimizer_closure_is_identity_exact():
    model = _bare_codec_only_model()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = SimpleNamespace(
        param_groups=[{"params": trainable[:14]}, {"params": trainable[14:]}]
    )

    summary = model.validate_optimizer_trainability(optimizer)

    assert summary["parameter_count"] == 27
    assert summary["trainable_numel"] == _CODEC_ONLY_TRAINABLE_NUMEL
    assert summary["optimizer_group_parameter_counts"] == [14, 13]

    with pytest.raises(RuntimeError, match="duplicate"):
        model.validate_optimizer_trainability(
            SimpleNamespace(param_groups=[{"params": trainable + [trainable[0]]}])
        )
    with pytest.raises(RuntimeError, match="differs"):
        model.validate_optimizer_trainability(
            SimpleNamespace(param_groups=[{"params": trainable[:-1]}])
        )


def test_trainer_validates_realized_optimizer_closure(monkeypatch):
    optimizer = SimpleNamespace(param_groups=[])
    expected_summary = {"mode": "codec_only", "parameter_count": 27}
    calls = []
    model = SimpleNamespace(
        validate_optimizer_trainability=lambda value: calls.append(value) or expected_summary
    )
    trainer = Gr00tTrainer.__new__(Gr00tTrainer)
    trainer.model = model
    monkeypatch.setattr(Trainer, "create_optimizer", lambda _trainer: optimizer)

    assert trainer.create_optimizer() is optimizer
    assert calls == [optimizer]
    assert trainer.optimizer_trainability_summary == expected_summary


def test_enabled_checkpoint_partial_codec_missing_fails(monkeypatch):
    requested = _model_codec_config()
    pipeline, codec_config = _checkpoint_identity_pipeline(requested)
    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.setup.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="Gr00tN1d7", codec_evidence=dict(requested)
        ),
    )
    mode = pipeline._checkpoint_codec_load_mode("checkpoint", codec_config)
    assert mode == "codec_reload"

    with pytest.raises(RuntimeError, match="codec_reload checkpoint is missing Codec keys"):
        pipeline._validate_codec_missing_keys(
            _codec_model_stub(),
            {"missing_keys": ["action_head.codec_conditioning.core.weight"]},
            mode,
        )


def test_saved_enabled_checkpoint_config_drift_or_disabled_current_fails(monkeypatch):
    requested = _model_codec_config()
    saved = {**requested, "d_codec": 32}
    pipeline, codec_config = _checkpoint_identity_pipeline(requested)
    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.setup.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="Gr00tN1d7", codec_evidence=saved),
    )
    with pytest.raises(ValueError, match="differs from the current dataset config"):
        pipeline._checkpoint_codec_load_mode("checkpoint", codec_config)

    saved = {**requested, "gate_init": False}
    with pytest.raises(TypeError, match="gate_init"):
        pipeline._checkpoint_codec_load_mode("checkpoint", codec_config)

    saved = dict(requested)
    pipeline, codec_config = _checkpoint_identity_pipeline(None)
    with pytest.raises(ValueError, match="current dataset configuration does not"):
        pipeline._checkpoint_codec_load_mode("checkpoint", codec_config)


def test_native_checkpoint_allows_only_complete_first_codec_extension(monkeypatch):
    requested = _model_codec_config()
    pipeline, codec_config = _checkpoint_identity_pipeline(requested)
    calls = []

    def native_config(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return SimpleNamespace(model_type="Gr00tN1d7", codec_evidence=None)

    monkeypatch.setattr("gr00t.model.gr00t_n1d7.setup.AutoConfig.from_pretrained", native_config)
    mode = pipeline._checkpoint_codec_load_mode("native-checkpoint", codec_config)
    assert mode == "native_to_codec"
    assert calls == [("native-checkpoint", {"local_files_only": True})]

    complete = {
        "action_head.codec_conditioning.core.weight",
        "action_head.codec_conditioning.projector.bias",
    }
    assert (
        pipeline._validate_codec_missing_keys(
            _codec_model_stub(), {"missing_keys": sorted(complete)}, mode
        )
        == complete
    )
    with pytest.raises(RuntimeError, match="complete new module key set"):
        pipeline._validate_codec_missing_keys(
            _codec_model_stub(),
            {"missing_keys": ["action_head.codec_conditioning.core.weight"]},
            mode,
        )


def test_codec_reload_rejects_missing_disabled_or_drifted_processor_before_model_load(
    tmp_path, monkeypatch
):
    requested = _model_codec_config()
    auto_model_calls = []

    def reject_auto_model(*args, **kwargs):
        auto_model_calls.append((args, kwargs))
        raise AssertionError("AutoModel must not be reached")

    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.setup.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="Gr00tN1d7", codec_evidence=dict(requested)
        ),
    )
    monkeypatch.setattr("gr00t.model.gr00t_n1d7.setup.AutoModel.from_pretrained", reject_auto_model)

    cases = {
        "missing": None,
        "disabled": {"enabled": False, "ignored": "native"},
        "drift": {**requested, "d_codec": 32},
    }
    for name, processor_codec in cases.items():
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        if processor_codec is not None:
            (checkpoint / "processor_config.json").write_text(
                json.dumps({"processor_kwargs": {"codec_evidence": processor_codec}})
            )
        pipeline, codec_config = _checkpoint_identity_pipeline(requested)
        pipeline.codec_evidence_config = codec_config
        pipeline.config.training = SimpleNamespace(
            start_from_checkpoint=str(checkpoint), skip_weight_loading=False
        )

        with pytest.raises(ValueError, match="processor"):
            pipeline._create_model()
        assert auto_model_calls == []


def test_native_to_codec_accepts_absent_processor_identity(tmp_path):
    requested = _model_codec_config()
    pipeline, codec_config = _checkpoint_identity_pipeline(requested)
    checkpoint = tmp_path / "native"
    checkpoint.mkdir()

    pipeline._validate_checkpoint_processor_codec_identity(
        checkpoint, codec_config, "native_to_codec"
    )


def test_native_checkpoint_relative_or_hf_identity_does_not_read_processor(monkeypatch):
    pipeline, codec_config = _checkpoint_identity_pipeline(None)

    def reject_processor_read(checkpoint):
        del checkpoint
        raise AssertionError("native path must not read processor identity")

    monkeypatch.setattr(pipeline, "_checkpoint_processor_codec_config", reject_processor_read)
    rng_before = torch.random.get_rng_state().clone()

    pipeline._validate_checkpoint_processor_codec_identity(
        "relative-or-hf-id", codec_config, "native"
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_native_to_codec_model_load_receives_strict_trainability(tmp_path, monkeypatch):
    pipeline = Gr00tN1d7Pipeline.__new__(Gr00tN1d7Pipeline)
    model_config = SimpleNamespace(
        action_horizon=40,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        codec_trainability="codec_only",
        state_dropout_prob=0.2,
        backbone_trainable_params_fp32=True,
        load_bf16=False,
    )
    pipeline.config = SimpleNamespace(
        model=model_config,
        training=SimpleNamespace(
            start_from_checkpoint="native-checkpoint",
            skip_weight_loading=False,
        ),
    )
    pipeline.transformers_loading_kwargs = {"local_files_only": True}
    pipeline.codec_evidence_config = SimpleNamespace(to_model_dict=lambda: _model_codec_config())
    pipeline.save_cfg_dir = tmp_path
    pipeline._checkpoint_codec_load_mode = lambda *_args: "native_to_codec"
    pipeline._validate_checkpoint_processor_codec_identity = lambda *_args: None
    pipeline._validate_codec_missing_keys = lambda *_args: set()
    pipeline._initialize_native_to_codec_gate = lambda *_args: None

    parameter = nn.Parameter(torch.zeros(()))
    loaded_model = SimpleNamespace(
        action_head=SimpleNamespace(mask_token=None),
        config=SimpleNamespace(to_filtered_json=lambda: "{}"),
        parameters=lambda: [parameter],
    )
    captured = {}

    def fake_from_pretrained(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return loaded_model, {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
        }

    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.setup.AutoModel.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.setup.run_on_rank0",
        lambda function, *args, **kwargs: function(*args),
    )

    assert pipeline._create_model() is loaded_model
    assert captured["checkpoint"] == "native-checkpoint"
    assert captured["codec_trainability"] == "codec_only"
    assert captured["codec_evidence"] == _model_codec_config()
