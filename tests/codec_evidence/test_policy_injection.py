from __future__ import annotations

from types import SimpleNamespace

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, _validate_model_processor_codec_identity
import numpy as np
import pytest
import torch


class _Processor:
    def __init__(self, *, codec_enabled: bool):
        self.codec_evidence_enabled = codec_enabled

    def __call__(self, messages):
        del messages
        return {"native": np.zeros((1,), dtype=np.float32)}

    def decode_action(self, action, embodiment_tag, states):
        del action, embodiment_tag, states
        return {"left_arm": np.zeros((1, 40, 7), dtype=np.float32)}


class _Model:
    def __init__(self):
        self.options = None

    def get_action(self, *, inputs, options):
        del inputs
        self.options = options
        return {"action_pred": torch.zeros((1, 40, 132), dtype=torch.float32)}


def _policy(*, codec_enabled: bool, evidence: object):
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.processor = _Processor(codec_enabled=codec_enabled)
    policy.model = _Model()
    policy.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    policy.language_key = "task"
    policy.modality_configs = {
        "state": SimpleNamespace(modality_keys=["left_arm"]),
    }

    def collate(rows):
        if codec_enabled:
            assert rows[0]["codec_evidence_sample"] is evidence
        return {"inputs": {"sentinel": torch.ones(())}}

    policy.collate_fn = collate
    return policy


def _observation():
    return {
        "video": {"ego_view": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)},
        "state": {"left_arm": np.zeros((1, 1, 7), dtype=np.float32)},
        "language": {"task": [["insert the shaft into the housing"]]},
    }


def test_policy_injects_codec_sample_in_process_and_strips_it_from_model_options():
    evidence = object()
    policy = _policy(codec_enabled=True, evidence=evidence)

    action, info = policy._get_action(
        _observation(),
        options={"codec_evidence_sample": evidence, "rtc_overlap_steps": 2},
    )

    assert action["left_arm"].shape == (1, 40, 7)
    assert info == {}
    assert policy.model.options == {"rtc_overlap_steps": 2}


def test_policy_codec_option_fails_closed_for_missing_native_or_wrong_batch():
    evidence = object()
    native = _policy(codec_enabled=False, evidence=evidence)
    with pytest.raises(ValueError, match="native processor"):
        native._get_action(_observation(), options={"codec_evidence_sample": evidence})

    enabled = _policy(codec_enabled=True, evidence=evidence)
    with pytest.raises(KeyError, match="codec_evidence_sample"):
        enabled._get_action(_observation())
    with pytest.raises(ValueError, match="batch size"):
        enabled._get_action(
            _observation(),
            options={"codec_evidence_sample": [evidence, evidence]},
        )


def _codec_identity(*, source_kind="NATIVE_CANONICAL_DECODED_DERIVED"):
    return {
        "enabled": True,
        "modalities": ["mv", "rpe", "i_frame", "metadata"],
        "gate_init": 0.0,
        "schema_version": "codec-evidence-episode-bundle-v1",
        "camera_id": "cam_left_high",
        "source_kind": source_kind,
        "d_codec": 64,
        "conditioner_heads": 8,
    }


def test_strict_policy_compares_model_and_processor_codec_identity(tmp_path):
    native_model = SimpleNamespace(config=SimpleNamespace(codec_evidence=None))
    native_processor = SimpleNamespace(codec_evidence_enabled=False)
    _validate_model_processor_codec_identity(native_model, native_processor, tmp_path)

    codec = _codec_identity()
    codec_model = SimpleNamespace(config=SimpleNamespace(codec_evidence=codec))
    codec_processor = SimpleNamespace(codec_evidence_enabled=True, codec_evidence=codec)
    _validate_model_processor_codec_identity(codec_model, codec_processor, tmp_path)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_model_processor_codec_identity(codec_model, native_processor, tmp_path)
    drifted = SimpleNamespace(
        codec_evidence_enabled=True,
        codec_evidence=_codec_identity(source_kind="RAW_JPEG_DERIVED"),
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_model_processor_codec_identity(codec_model, drifted, tmp_path)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_model_processor_codec_identity(native_model, codec_processor, tmp_path)
