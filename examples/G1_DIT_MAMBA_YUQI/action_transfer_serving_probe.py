"""Real legacy-client neural loopback, with independent prepared-data action references."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from action_transfer_checkpoint import DeployedStudent
from action_transfer_matched_support import MMapCache
from action_transfer_policy import (
    LANGUAGE_KEY,
    ActionTransferPolicy,
    load_models,
    observation_from_raw,
)
from action_transfer_program_support import restore_rng, rng_state, save_tensor_new
from budgeted_action_support import cached_action
from contract import GROUPS, sha, tensor_identity, write_new
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.server_client import MsgSerializer, PolicyClient, PolicyServer
from memory_numerical_trace import error_summary
import memory_prototype as proto
import msgpack
import numpy as np
from run_train_eval import NativeAnchors
from runner_core import set_seed
from serve_action_transfer import policy_identity, rng_identity, runtime, verify_plan
import torch
from yuqi_wire import legacy_to_bytes


# Executed only by the frozen old simulator Python with the old GR00T source.
# Plain numeric/string NPZ fixtures do not cross a pickle boundary.
OLD_CLIENT = r"""
import json, sys, numpy as np
sys.path.append(sys.argv[4])
import filelock
from gr00t.policy.server_client import PolicyClient
import gr00t.policy.server_client as wire
port, fixture_file, output = int(sys.argv[1]), sys.argv[2], sys.argv[3]
fixture = np.load(fixture_file, allow_pickle=False)
c = PolicyClient(host="127.0.0.1", port=port, timeout_ms=30000)
try:
    assert c.call_endpoint("ping", requires_input=False)["status"] == "ok"
    config = c.get_modality_config()
    assert config["video"].modality_keys == ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]
    assert config["state"].modality_keys == config["action"].modality_keys == ["left_arm", "left_hand"]
    assert config["action"].delta_indices == list(range(16))
    arrays = {}
    c.reset(options={"seed": 2026})
    for i, index in enumerate(fixture["order"].tolist()):
        obs = {"video": {k:fixture[f"{index}.video.{k}"].copy() for k in config["video"].modality_keys},
               "state": {k:fixture[f"{index}.state.{k}"].copy() for k in config["state"].modality_keys},
               "language": {config["language"].modality_keys[0]:[[str(fixture[f"{index}.text"].item())]]}}
        action, info = c.get_action(obs)
        for k, d in [("left_arm",7),("left_hand",6)]:
            assert action[k].shape == (1,16,d) and action[k].dtype == np.float32
            assert np.isfinite(action[k]).all()
            arrays[f"{i}.{k}"] = action[k].copy()
        c.get_modality_config()
        c.reset(options={"seed": i})
    np.savez(output, **arrays)
    print(json.dumps({"requests":3,"old_client_source":wire.__file__,"python":sys.executable,
                      "dependency_path":sys.argv[4],"filelock_version":filelock.__version__,"filelock_origin":filelock.__file__,
                      "action_shapes":{"left_arm":[1,16,7],"left_hand":[1,16,6]},"legacy_handshake":True}), flush=True)
finally:
    c.call_endpoint("kill", requires_input=False)
    c.socket.close(linger=0)
    c.context.term()
"""


def validate_client_identity(result, plan):
    if (
        Path(result["old_client_source"]).resolve()
        != Path(plan["client_pythonpath"]) / "gr00t/policy/server_client.py"
    ):
        raise ValueError("Subprocess did not import the frozen real old client")
    dependency = Path(plan["client_dependency_path"]).resolve()
    origin = Path(result["filelock_origin"]).resolve()
    if (
        Path(result["dependency_path"]).resolve() != dependency
        or origin != dependency / "filelock/__init__.py"
        or result["filelock_version"] != plan["client_filelock_version"]
    ):
        raise ValueError("Legacy client dependency origin/version changed")
    expected = {row["sha256"] for row in plan["records"] if Path(row["path"]).resolve() == origin}
    if len(expected) != 1 or sha(origin) not in expected:
        raise ValueError("Legacy client filelock is not bound to frozen material")


def save_fixtures(path, observations, order):
    arrays = {"order": np.asarray(order, dtype=np.int64)}
    for index, observation in enumerate(observations):
        for modality in ("video", "state"):
            for key, value in observation[modality].items():
                arrays[f"{index}.{modality}.{key}"] = value
        arrays[f"{index}.text"] = np.asarray(observation["language"][LANGUAGE_KEY][0][0])
    np.savez(path, **arrays)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def rejection_checks():
    refused = {}
    for name, call in (
        ("object_encode", lambda: legacy_to_bytes(np.array([object()], dtype=object))),
        (
            "pickle_decode",
            lambda: MsgSerializer.from_bytes(
                msgpack.packb({b"nd": True, b"kind": b"O", b"data": b"never unpickle"})
            ),
        ),
    ):
        try:
            call()
        except (TypeError, ValueError):
            refused[name] = True
        else:
            raise ValueError(f"Unsafe wire payload accepted: {name}")
    return refused


def counted_prediction(policy, inputs, *, graph):
    if not policy.mode.startswith("student"):
        return policy.gpu_action(inputs, graph=graph), 0
    core = policy.student.core if isinstance(policy.student, DeployedStudent) else policy.student
    original = core.encode_condition
    count = 0

    def encode(features):
        nonlocal count
        count += 1
        return original(features)

    with patch.object(core, "encode_condition", encode):
        result = policy.gpu_action(inputs, graph=graph)
    if count != 1:
        raise ValueError("Student condition must be computed exactly once per current RGB request")
    return result, count


def independent_parity(policy, source, plan, output, budget, cache=None):
    rows, arrays = [], {}
    for index, case in enumerate(plan["fixtures"]):
        budget()
        reference_input, raw = source.batch([(case["episode"], case["anchor"])])
        reference_input = {
            k: v for k, v in reference_input.items() if k not in ("action", "action_mask")
        }
        observation = observation_from_raw(raw[0])
        actual_input, states = policy.prepare_observation(observation)
        if set(actual_input) != set(reference_input) or any(
            not torch.equal(actual_input[k], reference_input[k].to(dtype=actual_input[k].dtype))
            for k in actual_input
        ):
            raise ValueError("Policy preparation differs from independent NativeAnchors.batch")
        set_seed(42, cuda=True)
        reference, eager_condition_calls = counted_prediction(policy, reference_input, graph=False)
        reference = reference.detach().float().cpu().numpy().copy()
        set_seed(42, cuda=True)
        actual_tensor, graph_condition_calls = counted_prediction(
            policy, actual_input, graph=policy.mode.endswith("graph")
        )
        actual = actual_tensor.detach().float().cpu().numpy().copy()
        cache_exact = None
        if cache is not None:
            frame = cache.frame(case["episode"], case["anchor"])
            deployment = policy.student if isinstance(policy.student, DeployedStudent) else None
            student_mode = policy.mode.startswith("student")
            core = (
                (deployment.core if deployment is not None else policy.student)
                if student_mode
                else None
            )
            with (
                deployment.bind(policy.model)
                if deployment is not None and student_mode
                else nullcontext()
            ):
                cached = cached_action(policy.model, frame, 42, student=core).float().numpy()
            cache_exact = np.array_equal(reference, cached)
            if not cache_exact:
                raise ValueError("Full current RGB and existing raw-cache student actions differ")
        stats = error_summary(torch.from_numpy(actual[:, :16]), torch.from_numpy(reference[:, :16]))
        exact = np.array_equal(actual, reference)
        passed = (
            exact
            if policy.mode.endswith("eager")
            else all(stats[k] <= bound for k, bound in plan["action_limits"].items())
        )
        # Independent decoder object and original raw state avoid validating a
        # shared bad observation conversion against itself.
        expected_physical = source.processor.decode_action(
            reference[0], EmbodimentTag.NEW_EMBODIMENT, state=raw[0].states
        )
        actual_physical = policy.processor.decode_action(actual, policy.embodiment_tag, states)
        physical_delta = {}
        for key in GROUPS:
            arrays[f"{index}.reference.{key}"] = np.asarray(expected_physical[key], np.float32)[
                None
            ]
            arrays[f"{index}.actual.{key}"] = np.asarray(actual_physical[key], np.float32)
            physical_delta[key] = float(
                np.max(np.abs(arrays[f"{index}.actual.{key}"] - arrays[f"{index}.reference.{key}"]))
            )
        if exact and any(value != 0 for value in physical_delta.values()):
            raise ValueError("Independent absolute physical decoder parity failed")
        arrays[f"{index}.reference.normalized"] = reference
        arrays[f"{index}.actual.normalized"] = actual
        rows.append(
            {
                **case,
                "seed": 42,
                "normalized_error": stats,
                "normalized_exact": exact,
                "raw_cache_exact": cache_exact,
                "condition_calls": {
                    "eager": eager_condition_calls,
                    "deployed": graph_condition_calls,
                },
                "physical_maxabs": physical_delta,
                "passed": passed,
                "prepared_input": {k: tensor_identity(v) for k, v in actual_input.items()},
            }
        )
    np.savez(output / "REFERENCE_ACTIONS.npz", **arrays)
    if not all(row["passed"] for row in rows):
        raise ValueError("Independent current-RGB serving parity failed")
    return {"passed": True, "cases": rows, "arrays_sha256": sha(output / "REFERENCE_ACTIONS.npz")}


def audit_endpoint(server, name, audits, *, action=False):
    original = server._endpoints[name].handler

    def wrapped(*args, **kwargs):
        before = rng_identity()
        result = original(*args, **kwargs)
        after = rng_identity()
        if not action and before != after:
            raise ValueError(f"Metadata/reset unexpectedly consumed policy RNG: {name}")
        audits.append({"endpoint": name, "rng_before": before, "rng_after": after})
        return result

    server.register_endpoint(name, wrapped, requires_input=server._endpoints[name].requires_input)


def loopback(policy, observations, fixture_path, plan, output, identity, deadline):
    direct, direct_copies, direct_rng, audits = [], [], [], []
    with PolicyServer(policy, host="127.0.0.1", port=plan["port"]) as server:
        for name in ("get_modality_config", "reset", "ping"):
            audit_endpoint(server, name, audits)
        audit_endpoint(server, "get_action", audits, action=True)
        # Socket binding and graph warmup precede the one serving seed operation.
        set_seed(42, cuda=True)
        initial = rng_state()
        initial_record = save_tensor_new(output / "INITIAL_RNG.pt", initial)
        write_new(
            output / "READY.json",
            {
                "status": "ready",
                **identity,
                "initial_rng": initial_record,
                "initial_rng_identity": rng_identity(initial),
                "requests_before_seed": policy.requests_completed,
            },
        )
        for index in plan["request_order"]:
            before = rng_identity()
            action, _ = policy.get_action(observations[index])
            direct.append(action)
            direct_copies.append({k: v.copy() for k, v in action.items()})
            direct_rng.append({"rng_before": before, "rng_after": rng_identity()})
        restore_rng(initial)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        env = dict(
            os.environ,
            PYTHONPATH=plan["client_pythonpath"],
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            CONDA_PREFIX=str(Path(plan["client_python"]).parent.parent),
        )
        try:
            completed = subprocess.run(
                [
                    plan["client_python"],
                    "-c",
                    OLD_CLIENT,
                    str(plan["port"]),
                    str(fixture_path),
                    str(output / "WIRE_ACTIONS.npz"),
                    plan["client_dependency_path"],
                ],
                env=env,
                cwd=plan["client_pythonpath"],
                text=True,
                capture_output=True,
                timeout=max(1, min(90, deadline - time.monotonic())),
            )
            (output / "CLIENT.stdout.txt").write_text(completed.stdout)
            (output / "CLIENT.stderr.txt").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError("Real old-client loopback failed: " + completed.stderr[-2000:])
        finally:
            thread.join(timeout=0.5)
            if thread.is_alive() and server.running:
                with PolicyClient(host="127.0.0.1", port=plan["port"], timeout_ms=3000) as client:
                    client.call_endpoint("kill", requires_input=False)
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("Owned loopback server thread did not terminate")
        client_result = json.loads(completed.stdout.splitlines()[-1])
        validate_client_identity(client_result, plan)
        wire = np.load(output / "WIRE_ACTIONS.npz", allow_pickle=False)
        request_audits = [row for row in audits if row["endpoint"] == "get_action"]
        if len(request_audits) != 3 or len(direct_rng) != 3 or policy.requests_completed != 6:
            raise ValueError("Legacy handshake consumed an extra get_action request")
        rows = []
        for i, expected in enumerate(direct):
            actual_rng = {key: request_audits[i][key] for key in ("rng_before", "rng_after")}
            if actual_rng != direct_rng[i]:
                raise ValueError("Direct/wire continuous action RNG sequence differs")
            if actual_rng["rng_before"]["torch_sha256"] == actual_rng["rng_after"]["torch_sha256"]:
                raise ValueError("Request did not advance the native noise RNG")
            for key in GROUPS:
                if not np.array_equal(expected[key], direct_copies[i][key]) or not np.array_equal(
                    expected[key], wire[f"{i}.{key}"]
                ):
                    raise ValueError("Wire output or preserved CPU action copy differs")
            rows.append(
                {
                    "request": i,
                    "fixture_index": plan["request_order"][i],
                    "physical_actions_exact": True,
                    "continuous_rng_exact": True,
                    **actual_rng,
                }
            )
        np.savez(
            output / "DIRECT_ACTIONS.npz",
            **{f"{i}.{k}": v for i, a in enumerate(direct) for k, v in a.items()},
        )
        result = {
            "status": "passed",
            "requests": rows,
            "endpoint_audits": audits,
            "client": client_result,
            "direct_actions_sha256": sha(output / "DIRECT_ACTIONS.npz"),
            "wire_actions_sha256": sha(output / "WIRE_ACTIONS.npz"),
            "independent_cpu_copies": True,
            "graph_replays": 0 if policy.route is None else policy.route.calls,
            "socket_closed_on_context_exit": True,
        }
    return result


def offline_adapter_proof(model, deployment, source, cache, plan, output, budget):
    """Run both real offline adapters on existing dev pairs before any final use."""
    from action_transfer_checkpoint import semantic_identity
    from action_transfer_final_support import (
        assert_development_teacher_equal,
        development_pairs,
        evaluate_development_candidate,
        verify_development_evaluation,
    )
    from integrity_diagnosis_support import BaseAudit

    pairs = development_pairs(plan)
    if (
        not isinstance(deployment, DeployedStudent)
        or deployment.record["kind"] != "core_periphery"
        or deployment.active_graphs
    ):
        raise ValueError("Development offline proof needs a composite owner with no live graph")
    output.mkdir(exist_ok=False)
    audit = BaseAudit(model, output, plan["base_content_sha256"], budget=budget)
    owner = deployment.owner
    ctx = SimpleNamespace(
        model=model,
        teacher=model.action_head.model,
        source=source,
        cache=cache,
        plan=plan,
        args=SimpleNamespace(cache_report_sha256=plan["cache_report_sha256"]),
        audit=audit,
        budget=budget,
    )
    records = {}
    private = None

    def evaluate(name, student, kind, expected_hashes):
        ctx.student = student
        ctx.candidate = {"kind": kind, "development_adapter_only": True}
        audit.check("before-development-adapter:" + name, stable=True)
        record = evaluate_development_candidate(ctx, pairs, output / name)
        value, rows = verify_development_evaluation(record, pairs)
        fields = {
            "student_content_sha256": "content_sha256",
            "checkpoint_content_sha256": "content_sha256",
            "core_content_sha256": "core_content_sha256",
            "periphery_content_sha256": "periphery_content_sha256",
        }
        if value["checkpoint_kind"] != kind or any(
            value[key] != expected_hashes[source_key] for key, source_key in fields.items()
        ):
            raise ValueError("Development evaluator full/core/periphery metadata binding differs")
        audit.check("after-development-adapter:" + name, stable=True)
        records[name] = record
        return value, rows

    try:
        core = deployment.core.cuda().eval().requires_grad_(False)
        core_hash = deployment.record["core_content_sha256"]
        core_result, _ = evaluate(
            "core-only",
            core,
            "core_only",
            {
                "content_sha256": core_hash,
                "core_content_sha256": core_hash,
                "periphery_content_sha256": None,
            },
        )
        owner.cuda().eval().requires_grad_(False)
        composite_result, composite_rows = evaluate(
            "composite", owner, "core_periphery", deployment.record
        )
        assert_development_teacher_equal(records["core-only"], records["composite"], pairs)
        fixture_exact = None
        if deployment.record.get("fixture_only", False):
            fixture_exact = all(
                core_result["array_sha256"][name] == composite_result["array_sha256"][name]
                for name in ("student_actions", "student_velocity", "student_physical")
            )
            if not fixture_exact:
                raise ValueError(
                    "B0-periphery fixture differs from the same core-only offline adapter"
                )
        # The source object/file is never perturbed. Copy on CPU to avoid keeping
        # two full FP32 peripheries resident on the GPU during this diagnostic.
        owner.cpu()
        ctx.student = None
        private = deepcopy(owner).cuda().eval().requires_grad_(False)
        with torch.no_grad():
            bias = private.periphery["vlln"].bias
            alternating = (torch.arange(bias.numel(), device=bias.device) % 2).to(
                bias.dtype
            ) * 2 - 1
            bias.add_(alternating.reshape_as(bias) / 64)
            private.periphery["action_decoder"].layer2.b.add_(1 / 64)
        private_hashes = semantic_identity(private.state_dict(), "core_periphery")
        _, changed_rows = evaluate(
            "diagnostic-private-periphery", private, "core_periphery", private_hashes
        )
        assert_development_teacher_equal(
            records["composite"], records["diagnostic-private-periphery"], pairs
        )
        original_actions = np.load(
            Path(records["composite"]["path"]).parent / "student_actions.npy"
        )
        changed_actions = np.load(
            Path(records["diagnostic-private-periphery"]["path"]).parent / "student_actions.npy"
        )
        witness = [
            {
                **pair,
                "student_condition_changed": a["noise_receipt"]["effective_condition"]
                != b["noise_receipt"]["effective_condition"],
                "student_action_changed": not np.array_equal(
                    original_actions[i], changed_actions[i]
                ),
            }
            for i, (pair, a, b) in enumerate(zip(pairs, composite_rows, changed_rows))
        ]
        if not all(
            row["student_condition_changed"] and row["student_action_changed"] for row in witness
        ):
            raise ValueError(
                "Private periphery perturbation did not witness changed student condition/action"
            )
        cache.assert_readonly()
        deployment.assert_unchanged()
        audit.check("development-proof-complete", stable=True)
        result = {
            "status": "passed",
            "development_only": True,
            "cohort": "dev",
            "pairs": pairs,
            "source_checkpoint": deployment.record,
            "optimizer_steps": 0,
            "base_content_sha256": plan["base_content_sha256"],
            "teacher_arrays_receipts_targets_masks_equal": True,
            "shared_noise_time_equal": True,
            "three_semantic_hash_bindings_verified": True,
            "b0_fixture_core_composite_exact": fixture_exact,
            "diagnostic_private_perturbation": {
                "saved_weights": False,
                "vlln.bias_alternating_delta": 1 / 64,
                "action_decoder.layer2.b_delta": 1 / 64,
                "witness": witness,
                "semantic_identity": private_hashes,
                "source_checkpoint_unchanged": True,
            },
            "evaluations": {
                name: {"path": record["path"], "sha256": record["sha256"]}
                for name, record in records.items()
            },
        }
    finally:
        ctx.student = None
        private = None
        owner.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        audit.check("development-private-copy-released", stable=True)
    write_new(output / "RESULT.json", result)
    return {"path": str((output / "RESULT.json").resolve()), "sha256": sha(output / "RESULT.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    plan = verify_plan(args.plan, args.plan_sha256)
    if (
        args.artifact_root.resolve() != Path(plan["artifact_root"]).resolve()
        or args.output.resolve() != args.artifact_root / plan["run_id"] / "output"
        or not 0 < args.wall_seconds <= plan["internal_wall_seconds"]
    ):
        parser.error("Frozen probe output/time identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.wall_seconds
    policy = None

    def budget():
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if (
            time.monotonic() > deadline
            or size > plan["max_artifact_bytes"]
            or torch.cuda.max_memory_allocated() > plan["max_allocated_bytes"]
        ):
            raise RuntimeError("Frozen serving probe resource bound exceeded")

    try:
        source_identity = runtime(plan)
        rejection = rejection_checks()
        model, student, processor, loading = load_models(plan)
        source = NativeAnchors(Path(plan["prepared"]))
        source.processor.eval()
        observations = [
            observation_from_raw(source.raw(c["episode"], c["anchor"])) for c in plan["fixtures"]
        ]
        fixture = save_fixtures(args.output / "FIXTURES.npz", observations, plan["request_order"])
        cache = (
            MMapCache(plan["cache_root"], plan, plan["cache_report_sha256"], budget)
            if "cache_root" in plan
            else None
        )
        if plan["student_checkpoint"]["kind"] == "core_periphery" and cache is None:
            raise ValueError(
                "Composite deployment probe requires frozen train/dev raw cache parity"
            )
        offline = None
        if plan["student_checkpoint"]["kind"] == "core_periphery":
            offline = offline_adapter_proof(
                model, student, source, cache, plan, args.output / "offline-adapter-proof", budget
            )
        rows = []
        for mode in plan["modes"]:
            budget()
            mode_output = args.output / mode
            mode_output.mkdir()
            policy = ActionTransferPolicy(
                model, processor, mode, student=student, task_text=plan["task_text"]
            )
            try:
                setup = policy.initialize_graph(observations[0]) if mode.endswith("graph") else None
                parity = independent_parity(policy, source, plan, mode_output, budget, cache)
                identity = policy_identity(policy, plan, source_identity, args.plan_sha256)
                result = loopback(
                    policy,
                    observations,
                    Path(fixture["path"]),
                    plan,
                    mode_output,
                    identity,
                    deadline,
                )
                result.update(identity=identity, graph_setup=setup, independent_reference=parity)
                write_new(mode_output / "RESULT.json", result)
                rows.append(
                    {
                        "mode": mode,
                        "path": str(mode_output / "RESULT.json"),
                        "sha256": sha(mode_output / "RESULT.json"),
                    }
                )
            finally:
                policy.close()
                policy = None
                gc.collect()
                torch.cuda.empty_cache()
            print(json.dumps({"mode": mode, "status": "passed", "wire_requests": 3}), flush=True)
        budget()
        if cache is not None:
            cache.assert_readonly()
        if (
            proto.content_hash(model.state_dict()) != plan["base_content_sha256"]
            or proto.content_hash(student.state_dict())
            != plan["student_checkpoint"]["content_sha256"]
        ):
            raise ValueError("Serving changed fixed model weights")
        result = {
            "status": "passed",
            "technical_validity": "passed",
            "plan_sha256": args.plan_sha256,
            "source": source_identity,
            "strict_loading": loading,
            "fixtures": fixture,
            "modes": rows,
            "safe_decoder": rejection,
            "offline_adapter_proof": offline,
            "optimizer_steps": 0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Frozen explicit core/composite checkpoint service compatibility only; no final model quality or simulation claim",
        }
        write_new(args.output / "RESULT.json", result)
    except Exception as exc:
        if policy is not None:
            policy.close()
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "plan_sha256": args.plan_sha256,
                "exception": type(exc).__name__,
                "message": str(exc),
                "optimizer_steps": 0,
                "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
