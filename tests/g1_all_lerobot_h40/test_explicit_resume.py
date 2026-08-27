from __future__ import annotations

from pathlib import Path

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.training.training_config import TrainingConfig
from gr00t.experiment.experiment import resolve_resume_from_checkpoint, resolve_resume_global_step
import pytest


def test_resume_config_is_exposed_at_finetune_and_training_layers(tmp_path):
    checkpoint = tmp_path / "checkpoint-20000"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 20000}', encoding="utf-8")
    finetune = FinetuneConfig(
        base_model_path="/base",
        embodiment_tag="NEW_EMBODIMENT",
        resume_from_checkpoint=str(checkpoint),
    )
    training = TrainingConfig(resume_from_checkpoint=finetune.resume_from_checkpoint)

    assert training.resume_from_checkpoint == str(checkpoint)
    assert resolve_resume_from_checkpoint(training.resume_from_checkpoint) == str(
        checkpoint.resolve()
    )
    assert resolve_resume_global_step(str(checkpoint), 40000) == 20000


def test_resume_defaults_to_no_resume_for_a_new_output(tmp_path):
    assert resolve_resume_from_checkpoint(None, tmp_path / "new-output") is False


def test_existing_output_checkpoint_requires_explicit_resume(tmp_path):
    output = tmp_path / "output"
    (output / "checkpoint-2").mkdir(parents=True)
    with pytest.raises(ValueError, match="require explicit"):
        resolve_resume_from_checkpoint(None, output)


@pytest.mark.parametrize("value", ["checkpoint-20000", "/missing/checkpoint-20000"])
def test_explicit_resume_requires_existing_absolute_directory(value):
    with pytest.raises(ValueError, match="existing absolute directory"):
        resolve_resume_from_checkpoint(value)


def test_resume_does_not_alias_new_output_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint-20000"
    output = tmp_path / "new-output"
    checkpoint.mkdir()

    resolved = Path(resolve_resume_from_checkpoint(str(checkpoint)))
    assert resolved != output
