from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from gr00t.experimental.arena_functional_mamba_transition import FunctionalMamba2State
import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples/G1_ARENA_N17/online_state_scaling.py"
EXAMPLE_DIR = str(EXAMPLE.parent)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)
SPEC = importlib.util.spec_from_file_location("online_state_scaling_runner", EXAMPLE)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class _CPUFunctionalMambaStub(torch.nn.Module):
    """Exercise runner state semantics without invoking GPU-only Mamba kernels."""

    def new_functional_state(self, batch_size: int) -> FunctionalMamba2State:
        return FunctionalMamba2State(
            tuple(
                (
                    torch.zeros(batch_size, 640, 4),
                    torch.zeros(batch_size, 8, 64, 64),
                )
                for _ in range(2)
            ),
            batch_size,
        )

    def functional_step(
        self,
        descriptor: torch.Tensor,
        state: FunctionalMamba2State,
    ) -> tuple[torch.Tensor, FunctionalMamba2State]:
        return descriptor, FunctionalMamba2State(
            tuple((conv.clone(), ssm.clone()) for conv, ssm in state.layers),
            state.batch_size,
            state.observations + 1,
        )


def test_frozen_five_process_orders_lengths_and_incomplete_status() -> None:
    assert runner.CONDITIONS == (
        "functional_mamba",
        "persistent_gru",
        "transformer_functional_cat",
        "transformer_preallocated",
        "current_descriptor",
    )
    assert runner.LENGTHS == (128, 256, 512, 925)
    assert runner.COST_RUN_IDS == {
        "A": "cost-gpu0-r1a",
        "B": "cost-gpu0-r1b",
        "C": "cost-gpu0-r1c",
        "D": "cost-gpu0-r1d",
        "E": "cost-gpu0-r1e",
    }
    for offset, label in enumerate(runner.COST_RUN_IDS):
        assert runner.CONDITION_ORDERS[label] == (
            runner.CONDITIONS[offset:] + runner.CONDITIONS[:offset]
        )
        length_offset = offset % len(runner.LENGTHS)
        assert runner.LENGTH_ORDERS[label] == (
            runner.LENGTHS[length_offset:] + runner.LENGTHS[:length_offset]
        )
    assert runner.WARMUP_SAMPLES == 20
    assert runner.MEASURED_SAMPLES == 100
    assert runner.IMPLEMENTATION_STATUS == {
        "preallocated_control": True,
        "independent_correctness": True,
        "cost_cli_contract": True,
        "cost_measurement": False,
        "aggregate_decision": False,
        "formal_gpu_run_allowed": False,
    }


def test_model_identity_records_per_tensor_shared_transformer_hashes() -> None:
    identity = runner.model_identity()
    functional = identity["conditions"]["transformer_functional_cat"]
    preallocated = identity["conditions"]["transformer_preallocated"]
    assert functional["parameters"] == preallocated["parameters"] == 857_216
    assert functional["state_dict_sha256"] == preallocated["state_dict_sha256"]
    assert functional["state_dict_tensor_sha256"] == preallocated["state_dict_tensor_sha256"]
    assert identity["transformer_weight_hash_shared"]
    assert identity["transformer_per_tensor_hash_shared"]
    assert (
        len(functional["state_dict_tensor_sha256"]) == len(functional["state_dict_sha256"][:0]) + 24
    )


def test_closed_routes_and_opaque_full_endpoint_are_fail_closed() -> None:
    assert runner.assert_closed_routes_absent() == {
        "closed_helper_calls": [],
        "closed_runner_imported": False,
        "opaque_full_transformer_calls": 0,
    }
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "CausalTransformerBackbone" not in source
    module = runner.build_condition("transformer_preallocated").eval()
    with pytest.raises(RuntimeError, match="full-sequence.*forbidden"):
        module(torch.randn(1, 2, 256))


def test_closed_route_audit_ignores_preimported_runner_from_shared_test_process() -> None:
    imported = importlib.import_module("functional_streaming_controls")
    assert (
        Path(imported.__file__).resolve()
        == (REPO / "examples/G1_ARENA_N17/functional_streaming_controls.py").resolve()
    )
    assert "functional_streaming_controls" in sys.modules
    assert runner.assert_closed_routes_absent()["closed_runner_imported"] is False


def test_closed_route_audit_rejects_import_in_the_new_runner_ast() -> None:
    with pytest.raises(RuntimeError, match="runner import"):
        runner._assert_closed_route_source(
            "import functional_streaming_controls as closed_runner\n",
            filename="synthetic_new_runner.py",
        )
    with pytest.raises(RuntimeError, match="helper call"):
        runner._assert_closed_route_source(
            "def bad():\n    evaluate_cost_gates()\n",
            filename="synthetic_new_runner.py",
        )


def test_generic_dispatch_keeps_preallocated_mutable_semantics_separate() -> None:
    module = runner.build_condition("transformer_preallocated").eval()
    state = runner.new_state("transformer_preallocated", module, 1)
    ids = state.storage_identities()
    with torch.inference_mode():
        output, returned = runner.online_step(
            "transformer_preallocated", module, torch.randn(1, 256), state
        )
    assert output.shape == (1, 256)
    assert returned is state
    assert state.live_index == 1
    assert state.storage_identities() == ids
    assert runner._state_logical_bytes(state) == 4_096
    assert runner._state_capacity_bytes(state) == 3_788_800

    functional = runner.build_condition("transformer_functional_cat").eval()
    functional_state = runner.new_state("transformer_functional_cat", functional, 1)
    with torch.inference_mode():
        _, functional_returned = runner.online_step(
            "transformer_functional_cat",
            functional,
            torch.randn(1, 256),
            functional_state,
        )
    assert functional_returned is not functional_state
    assert functional_returned.observations == 1


@pytest.mark.parametrize("condition", runner.CONDITIONS)
def test_recurrence_invariants_enforce_reset_return_and_replay_contract(
    condition: str,
) -> None:
    module = (
        _CPUFunctionalMambaStub()
        if condition == "functional_mamba"
        else runner.build_condition(condition).eval()
    )
    descriptors = torch.randn(
        1,
        128,
        256,
        generator=torch.Generator().manual_seed(230923),
    )
    invariants, replay_output, replay_state = runner.recurrence_invariants(
        condition,
        module,
        descriptors,
    )
    assert replay_output.shape == (1, 128, 256)
    assert replay_state.observations == 128
    assert invariants["reset_returned_observations"] == 0
    assert invariants["reset_alias_contract"]
    assert invariants["reset_returned_is_input"] is (condition == "transformer_preallocated")
    assert invariants["episode_reset_output_exact"]
    assert invariants["episode_reset_terminal_state_exact"]
    assert invariants["episode_reset_exact"]


def test_plan_and_dry_run_are_explicitly_nonformal_and_cost_fail_closed() -> None:
    payload = runner.dry_run_payload()
    assert payload["status"] == "dry_run_only"
    assert payload["formal_execution_permitted"] is False
    assert payload["result_schema"] == {
        "status": "engineering_incomplete",
        "correctness_not_run": True,
        "cost_not_implemented": True,
        "aggregate_not_implemented": True,
        "training_allowed": False,
        "vla_model_instantiated": False,
        "optimizer_steps": 0,
    }
    assert len(payload["cost_process_interface"]) == 5
    assert payload["plan"]["closed_evidence_excluded"] == {
        "functional_streaming_controls": True,
        "transformer_parity_diagnosis": True,
    }
    assert payload["plan"]["implementation_status"]["cost_measurement"] is False


def test_correctness_gates_fail_closed_before_gradient_and_full_lengths() -> None:
    execution = {
        "native_vla_modules": {"before": [], "after": []},
        "optimizer": {"constructor_calls": [], "steps_observed": 0},
    }
    gates = runner.evaluate_correctness({}, execution)
    assert gates["native_vla_modules_absent"]
    assert gates["optimizer_steps_zero"]
    assert not gates[
        "all_lengths_model_specific_oracles_invariants_state_and_shared_transformer_pass"
    ]
    assert not gates["t128_directional_jvp_gradcheck"]
    assert not all(gates.values())


@pytest.mark.parametrize("condition", ["persistent_gru", "transformer_functional_cat"])
def test_cpu_directional_gradient_probe_is_finite_nonzero(condition: str) -> None:
    module = runner.build_condition(condition).eval()
    descriptors = torch.randn(1, 4, 256, generator=torch.Generator().manual_seed(29))
    result = runner.directional_gradient_probe(condition, module, descriptors)
    assert result["length"] == 4
    assert result["gradient_finite"]
    assert result["gradient_nonzero"]
    assert result["passed"], result


def test_cli_dry_run_is_json_and_does_not_materialize_artifact() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = runner.FROZEN_RUNTIME["pythonpath"]
    before = runner.ARTIFACT_ROOT.exists()
    completed = subprocess.run(
        [runner.FROZEN_RUNTIME["python_executable"], str(EXAMPLE), "--dry-run"],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run_only"
    assert payload["result_schema"]["cost_not_implemented"]
    assert runner.ARTIFACT_ROOT.exists() is before
