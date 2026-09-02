from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
import numpy as np
import torch


def _numpy_state_equal(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_absent_and_disabled_setup_do_not_import_codec_or_consume_rng(tmp_path):
    assert "codec_evidence" not in Gr00tN1d7Config().to_filtered_dict()
    for value in ({}, {"codec_evidence": {"enabled": False}}):
        adapter_path = tmp_path / f"adapter-{len(list(tmp_path.iterdir()))}.json"
        adapter_path.write_text(json.dumps(value), encoding="utf-8")
        pipeline = Gr00tN1d7Pipeline.__new__(Gr00tN1d7Pipeline)
        pipeline.config = SimpleNamespace(
            data=SimpleNamespace(
                datasets=[
                    SimpleNamespace(
                        dataset_adapter="g1_id10_h40_native30_v1",
                        adapter_config_path=str(adapter_path),
                    )
                ]
            )
        )
        python_state = random.getstate()
        numpy_state = deepcopy(np.random.get_state())
        torch_state = torch.random.get_rng_state().clone()
        before_modules = {
            name
            for name in __import__("sys").modules
            if name.startswith("gr00t.data.codec_evidence")
        }
        assert pipeline._resolve_codec_evidence_config() is None
        after_modules = {
            name
            for name in __import__("sys").modules
            if name.startswith("gr00t.data.codec_evidence")
        }
        assert after_modules == before_modules
        assert random.getstate() == python_state
        assert _numpy_state_equal(np.random.get_state(), numpy_state)
        assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_disabled_collator_has_native_serialized_surface_and_rng(monkeypatch):
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side=None))
    monkeypatch.setattr(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        lambda *args, **kwargs: processor,
    )
    torch_state = torch.random.get_rng_state().clone()
    collator = Gr00tN1d7DataCollator("local", transformers_loading_kwargs={})
    assert "codec_evidence_enabled" not in vars(collator)
    assert "codec_evidence_collator" not in vars(collator)
    assert str(collator) == "Gr00tN1d7DataCollator(model_name=local, model_type=qwen)"
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_absent_config_isolated_process_never_loads_codec_modules(tmp_path):
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text("{}\n", encoding="utf-8")
    script = f"""
from types import SimpleNamespace
import random
import sys
import numpy as np
import torch
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
assert not any(name.startswith('gr00t.data.codec_evidence') for name in sys.modules)
p = Gr00tN1d7Pipeline.__new__(Gr00tN1d7Pipeline)
p.config = SimpleNamespace(data=SimpleNamespace(datasets=[SimpleNamespace(
    dataset_adapter='g1_id10_h40_native30_v1',
    adapter_config_path={str(adapter_path)!r},
)]))
py_state = random.getstate()
np_state = np.random.get_state()
torch_state = torch.random.get_rng_state().clone()
assert p._resolve_codec_evidence_config() is None
assert not any(name.startswith('gr00t.data.codec_evidence') for name in sys.modules)
assert random.getstate() == py_state
after_np = np.random.get_state()
assert after_np[0] == np_state[0] and np.array_equal(after_np[1], np_state[1]) and after_np[2:] == np_state[2:]
assert torch.equal(torch.random.get_rng_state(), torch_state)
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
