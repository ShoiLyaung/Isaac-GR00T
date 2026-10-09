"""Explicit candidate cost or one isolated native Yuqi simulator run; no scheduling."""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from contract import sha, write_new


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def verify_plan(args):
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Explicit evaluation PLAN SHA mismatch")
    plan = read(args.plan)
    if (
        Path(plan["source_worktree"]).resolve() != REPO
        or Path(plan["artifact_root"]).resolve() != args.artifact_root.resolve()
        or args.output.resolve() != args.artifact_root.resolve() / plan["run_id"] / "output"
        or plan["command"] != args.command
        or not 0 < args.wall_seconds <= plan["internal_wall_seconds"]
    ):
        raise ValueError("Frozen evaluation source/output/command/time identity changed")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen evaluation material changed: {path}")
    identities = {Path(r["path"]).resolve(): r["sha256"] for r in plan["records"]}
    for relative in plan["immutable_source_relative_paths"]:
        path = (REPO / relative).resolve()
        if identities.get(path) != sha(path):
            raise ValueError(f"Immutable current source changed: {relative}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=REPO, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(plan["allowed_source_files"]):
        raise ValueError("Expected clean committed evaluation-only source changes")
    return plan


def artifact_budget(args, plan, deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Evaluation wall budget exceeded")
    size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
    if size > plan["max_artifact_bytes"]:
        raise RuntimeError("Evaluation artifact budget exceeded")
    return size


def validate_kit(simulator, artifact_root):
    manifest_path = Path(simulator["kit_manifest_path"])
    if sha(manifest_path) != simulator["kit_manifest_sha256"]:
        raise ValueError("Private Kit initialization manifest changed")
    kit = Path(simulator["kit_root"])
    if kit.is_symlink() or not kit.resolve().is_relative_to(artifact_root.resolve()):
        raise ValueError("Kit root must be a private directory in this artifact root")
    rows = read(manifest_path)
    paths = []
    for row in rows:
        relative = Path(row["relative_path"])
        path = kit / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] not in ("data", "cache")
            or any(parent.is_symlink() for parent in (path, *path.parents) if parent != kit.parent)
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.stat().st_size != row["bytes"]
            or sha(path) != row["sha256"]
        ):
            raise ValueError(f"Private Kit copy differs or aliases shared storage: {path}")
        paths.append(path.relative_to(kit).as_posix())
    actual = sorted(p.relative_to(kit).as_posix() for p in kit.rglob("*") if p.is_file())
    if actual != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Kit initial inventory must contain exactly frozen data/cache files")
    return {
        "root": str(kit.resolve()),
        "manifest_sha256": sha(manifest_path),
        "files": len(paths),
        "bytes": sum(r["bytes"] for r in rows),
        "private_regular_files": True,
    }


def child_environment(spec, physical_gpu):
    env = dict(os.environ)
    env.pop("INSERTION_EVAL_ROOM_WALLS_USD", None)
    env.update({k: str(v) for k, v in spec["environment"].items()})
    env.pop("INSERTION_EVAL_ROOM_WALLS_USD", None)
    if env.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("Child physical GPU differs from frozen plan")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def terminate_owned(proc):
    """Only a Popen child started in its own session can be signalled here."""
    if proc is None or proc.poll() is not None:
        return {"terminated": False}
    if os.getpgid(proc.pid) != proc.pid:
        raise RuntimeError("Refusing to signal a process group not owned by this child")
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return {"terminated": True, "signal": "SIGTERM", "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        return {"terminated": True, "signal": "SIGKILL", "returncode": proc.returncode}


def wait_child(proc, deadline, check):
    while True:
        check()
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned child exceeded this bounded evaluation deadline")
        try:
            return proc.wait(timeout=min(5, max(0.01, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass


def validate_ready(path, server):
    value = read(path)
    if (
        value.get("status") != "ready"
        or value["plan_sha256"] != server["plan_sha256"]
        or value["mode"] != server["mode"]
        or value["source"]["commit"] != server["source_commit"]
        or value["host"] != "127.0.0.1"
        or value["port"] != server["port"]
    ):
        raise ValueError("Owned server READY mode/source/PLAN identity mismatch")
    expected = "budgeted_mamba" if server["mode"].startswith("student") else "dit"
    if value["effective_action_backbone"] != expected:
        raise ValueError("READY generator identity does not match selected mode")
    return {"path": str(path.resolve()), "sha256": sha(path), "identity": value}


def metadata_handshake(server):
    from gr00t.policy.server_client import PolicyClient

    with PolicyClient(host="127.0.0.1", port=server["port"], timeout_ms=3000) as client:
        config = client.get_modality_config()
        if (
            config["action"].delta_indices != list(range(16))
            or config["action"].modality_keys != ["left_arm", "left_hand"]
            or config["video"].modality_keys
            != ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]
        ):
            raise ValueError("Owned server metadata handshake contract changed")
    return {"metadata_only": True, "action_requests": 0, "returned_action_horizon": 16}


def summarize_simulation(output, simulator, returncode):
    """Retain planned denominator even when the native evaluator aborts or times out."""
    output = Path(output)
    path = output / "evaluation_results.jsonl"
    rows, issues = [], []
    if path.exists():
        for index, line in enumerate(path.read_text().splitlines()):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                issues.append(f"invalid episode JSON row {index}")
    planned = int(simulator["episodes"])
    seeds = simulator["seeds"]
    if len(seeds) != planned:
        raise ValueError("Frozen planned seed count differs")
    valid = []
    for index, row in enumerate(rows):
        if (
            index >= planned
            or row.get("episode_index") != index
            or row.get("seed") != seeds[index]
            or type(row.get("success")) is not bool
            or type(row.get("stable_success")) is not bool
            or not isinstance(row.get("failure_reason"), str)
        ):
            issues.append(f"episode identity/status mismatch at row {index}")
        else:
            valid.append(row)
    summary_path = output / "summary.json"
    summary = read(summary_path) if summary_path.exists() else None
    if summary is None:
        issues.append("summary missing")
    else:
        if (
            summary.get("episodes") != len(rows)
            or summary.get("successes") != sum(r["success"] for r in valid)
            or summary.get("stable_successes") != sum(r["stable_success"] for r in valid)
        ):
            issues.append("summary and episode counts disagree")
        for key, value in simulator["expected_config"].items():
            if summary.get("config", {}).get(key) != value:
                issues.append(f"simulator config mismatch: {key}")
        if summary.get("aborted") is not False:
            issues.append("evaluator aborted or lacks terminal status")
    completed = len(valid)
    stable = sum(r["stable_success"] for r in valid)
    success = sum(r["success"] for r in valid)
    if completed != planned:
        issues.append("planned episode cohort incomplete")
    if returncode != 0:
        issues.append(f"evaluator returncode {returncode}")
    return {
        "technical_validity": "passed" if not issues else "not_established",
        "complete": not issues,
        "planned_episodes": planned,
        "completed_episodes": completed,
        "not_completed": max(0, planned - completed),
        "successes": success,
        "stable_successes": stable,
        "stable_success_rate_completed": stable / completed if completed else None,
        "stable_success_fraction_of_planned": stable / planned,
        "outcome_counts": dict(
            Counter("success" if r["stable_success"] else r["failure_reason"] for r in valid)
        ),
        "issues": issues,
        "summary_sha256": sha(summary_path) if summary_path.exists() else None,
        "episodes_sha256": sha(path) if path.exists() else None,
        "scope": "Original Yuqi root-pose/stable predicate, with incomplete denominator preserved; no real-robot claim",
    }


def simulate(args, plan, deadline):
    server_spec, sim_spec = plan["server"], plan["simulator"]
    if sha(server_spec["plan_path"]) != server_spec["plan_sha256"]:
        raise ValueError("Explicit serving subplan changed")
    ready_path = Path(server_spec["ready_path"])
    sim_output = Path(sim_spec["output_dir"])
    if (
        ready_path.exists()
        or sim_output.exists()
        or not ready_path.resolve().is_relative_to(args.output.resolve())
        or not sim_output.resolve().is_relative_to(args.output.resolve())
    ):
        raise ValueError("Fresh private READY/evaluation output paths required")
    kit = validate_kit(sim_spec, args.artifact_root)
    temporary = Path(sim_spec["environment"]["TMPDIR"])
    if (
        temporary.is_symlink()
        or not temporary.is_dir()
        or not temporary.resolve().is_relative_to(args.artifact_root.resolve())
        or any(temporary.iterdir())
    ):
        raise ValueError("Simulator TMPDIR must be a fresh private artifact directory")
    argv = sim_spec["argv"]
    if (
        "--kit_args" not in argv
        or argv[argv.index("--kit_args") + 1] != "--portable-root " + sim_spec["kit_root"]
        or "--output-dir" not in argv
        or Path(argv[argv.index("--output-dir") + 1]).resolve() != sim_output.resolve()
    ):
        raise ValueError("Simulator argv does not use the frozen private Kit/output")
    for label, spec in (("server", server_spec), ("simulator", sim_spec)):
        if not spec["argv"] or not Path(spec["argv"][0]).is_absolute():
            raise ValueError(f"Frozen absolute {label} argv required")
    server = evaluator = None
    ready = None
    returncode = None
    error = None
    cleanup = {}
    logs = []
    try:
        for label in ("server", "simulator"):
            logs.append((args.output / f"{label}.stdout.txt").open("x"))
        server = subprocess.Popen(
            server_spec["argv"],
            cwd=server_spec["cwd"],
            env=child_environment(server_spec, plan["physical_gpu"]),
            stdout=logs[0],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_new(
            args.output / "SERVER_PROCESS.json",
            {
                "pid": server.pid,
                "process_group": server.pid,
                "argv": server_spec["argv"],
                "owned": True,
            },
        )
        ready_deadline = min(deadline, time.monotonic() + server_spec["ready_timeout_seconds"])
        while not ready_path.exists():
            artifact_budget(args, plan, deadline)
            if server.poll() is not None:
                raise RuntimeError(f"Owned server exited before READY: {server.returncode}")
            if time.monotonic() > ready_deadline:
                raise TimeoutError("Owned server READY deadline exceeded")
            time.sleep(0.2)
        if server.poll() is not None:
            raise RuntimeError("Owned server stopped before READY validation")
        ready = validate_ready(ready_path, server_spec)
        handshake = metadata_handshake(server_spec)
        write_new(args.output / "SERVER_IDENTITY.json", {"ready": ready, "handshake": handshake})
        evaluator = subprocess.Popen(
            sim_spec["argv"],
            cwd=sim_spec["cwd"],
            env=child_environment(sim_spec, plan["physical_gpu"]),
            stdout=logs[1],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_new(
            args.output / "EVALUATOR_PROCESS.json",
            {
                "pid": evaluator.pid,
                "process_group": evaluator.pid,
                "argv": sim_spec["argv"],
                "owned": True,
            },
        )
        returncode = wait_child(evaluator, deadline, lambda: artifact_budget(args, plan, deadline))
    except Exception as exc:
        error = {"exception": type(exc).__name__, "message": str(exc)}
    finally:
        cleanup["evaluator"] = terminate_owned(evaluator)
        if evaluator is not None:
            returncode = evaluator.returncode
        if server is not None and server.poll() is None and ready is not None:
            try:
                from gr00t.policy.server_client import PolicyClient

                with PolicyClient(
                    host="127.0.0.1", port=server_spec["port"], timeout_ms=2000
                ) as client:
                    client.call_endpoint("kill", requires_input=False)
                cleanup["owned_kill_endpoint"] = True
                server.wait(timeout=5)
            except Exception as exc:
                cleanup["kill_endpoint_error"] = str(exc)
        cleanup["server"] = terminate_owned(server)
        for stream in logs:
            stream.close()
    results = summarize_simulation(sim_output, sim_spec, returncode)
    return {
        "status": "completed" if results["complete"] and error is None else "failed",
        "technical_validity": results["technical_validity"],
        "simulation": results,
        "server_ready": ready,
        "kit_initial": kit,
        "cleanup": cleanup,
        "error": error,
        "optimizer_steps": 0,
        "model_identity_source": "owned SHA-bound server READY, not evaluator --model-path label",
    }


def gpu_action(policy, inputs, *, graph):
    """Return GPU tensor; processing/decoder/CPU evidence stay outside cost timing."""
    return policy.gpu_action(inputs, graph=graph)


def cost(args, plan, deadline):
    from action_transfer_policy import ActionTransferPolicy, load_models, observation_from_raw
    from budgeted_action_cudagraph_probe import summarize
    from memory_numerical_trace import error_summary
    import memory_prototype as proto
    from run_train_eval import NativeAnchors
    from runner_core import set_seed
    from serve_action_transfer import runtime
    import torch

    source_identity = runtime(plan)
    model, student, processor, loading = load_models(plan)
    teacher = model.action_head
    initial_peak = torch.cuda.max_memory_allocated()
    source = NativeAnchors(Path(plan["prepared"]))
    source.processor.eval()
    observations = [
        observation_from_raw(source.raw(c["episode"], c["anchor"])) for c in plan["fixtures"]
    ]
    if plan["modes"] != ["native_graph", "student_graph"]:
        raise ValueError("Matched core-graph cost requires native then student")
    timings, parity, setups = {}, {}, {}
    stage_peak = initial_peak

    def budget():
        nonlocal stage_peak
        stage_peak = max(stage_peak, torch.cuda.max_memory_allocated())
        artifact_budget(args, plan, deadline)
        if stage_peak > plan["max_allocated_bytes"]:
            raise RuntimeError("GPU allocated bound exceeded")

    try:
        for mode in plan["modes"]:
            budget()
            policy = ActionTransferPolicy(
                model, processor, mode, student=student, task_text=plan["task_text"]
            )
            active = teacher if mode == "native_graph" else student.head
            inactive = student.head if mode == "native_graph" else teacher
            if any(p.device.type != "cpu" for p in inactive.parameters()):
                raise ValueError("Inactive complete action head was not offloaded")
            gc.collect()
            torch.cuda.empty_cache()
            output = args.output / mode
            output.mkdir()
            torch.cuda.reset_peak_memory_stats()
            try:
                setup = policy.initialize_graph(observations[0])
                setup.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    inactive_head_device="cpu",
                    fp32_periphery_master_device="cpu",
                    active_core_device="cuda",
                    active_parameter_dtypes=sorted({str(p.dtype) for p in active.parameters()}),
                )
                budget()
                setups[mode] = setup
                prepared = [policy.prepare_observation(obs)[0] for obs in observations]
                arrays, checks = {}, []
                for ordinal, index in enumerate(plan["parity_order"]):
                    for seed in plan["parity_seeds"]:
                        budget()
                        set_seed(seed, cuda=True)
                        reference = (
                            gpu_action(policy, prepared[index], graph=False).detach().float().cpu()
                        )
                        set_seed(seed, cuda=True)
                        actual = (
                            gpu_action(policy, prepared[index], graph=True).detach().float().cpu()
                        )
                        if (
                            actual.shape != (1, 40, 13)
                            or not torch.isfinite(actual).all()
                            or not torch.isfinite(reference).all()
                        ):
                            raise ValueError("Nonfinite or wrong-shape cost parity output")
                        stats = error_summary(actual[:, :16], reference[:, :16])
                        passed = all(stats[k] <= v for k, v in plan["action_limits"].items())
                        checks.append(
                            {
                                "ordinal": ordinal,
                                "fixture_index": index,
                                "seed": seed,
                                "error": stats,
                                "passed": passed,
                            }
                        )
                        arrays[f"{ordinal}.{seed}.eager"] = reference
                        arrays[f"{ordinal}.{seed}.graph"] = actual
                torch.save(arrays, output / "PARITY_ACTIONS.pt")
                parity[mode] = {
                    "cases": checks,
                    "passed": all(r["passed"] for r in checks),
                    "actions_sha256": sha(output / "PARITY_ACTIONS.pt"),
                }
                timings[mode] = []
                for index, inputs in enumerate(prepared):
                    actions, rows = [], []
                    for repeat in range(plan["timing"]["warmup"] + plan["timing"]["measured"]):
                        budget()
                        set_seed(42, cuda=True)
                        policy.lease.assert_storage()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        tick = time.perf_counter()
                        value = gpu_action(policy, inputs, graph=True)
                        torch.cuda.synchronize()
                        wall = (time.perf_counter() - tick) * 1000
                        allocated, reserved = (
                            torch.cuda.max_memory_allocated(),
                            torch.cuda.max_memory_reserved(),
                        )
                        stage_peak = max(stage_peak, allocated)
                        action = value.detach().float().cpu()
                        del value
                        if action.shape != (1, 40, 13) or not torch.isfinite(action).all():
                            raise ValueError("Invalid measured full policy output")
                        if actions and not torch.equal(action, actions[0]):
                            raise ValueError("Fixed-seed graph timing output changed")
                        actions.append(action)
                        rows.append(
                            {
                                "fixture_index": index,
                                "repeat": repeat,
                                "warmup": repeat < plan["timing"]["warmup"],
                                "wall_ms": wall,
                                "peak_allocated_bytes": allocated,
                                "peak_reserved_bytes": reserved,
                            }
                        )
                    torch.save(torch.stack(actions), output / f"fixture-{index}-actions.pt")
                    write_new(output / f"fixture-{index}-timings.json", rows)
                    timings[mode].append(
                        {
                            "fixture": plan["fixtures"][index],
                            **summarize(rows),
                            "timings_sha256": sha(output / f"fixture-{index}-timings.json"),
                            "actions_sha256": sha(output / f"fixture-{index}-actions.pt"),
                        }
                    )
            finally:
                policy.close()
                del policy
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    finally:
        # The active graph is closed before either core is moved by any caller.
        teacher.zero_grad(set_to_none=True)
        student.zero_grad(set_to_none=True)
    gates = []
    for index, (native, candidate) in enumerate(
        zip(timings["native_graph"], timings["student_graph"])
    ):
        ratio = candidate["p95_ms"] / native["p95_ms"]
        memory_ratio = candidate["peak_allocated_bytes"] / native["peak_allocated_bytes"]
        gates.append(
            {
                "fixture_index": index,
                "p95_ratio": ratio,
                "allocated_ratio": memory_ratio,
                "latency_passed": ratio <= plan["decision_limits"]["p95_ratio"],
                "memory_passed": memory_ratio <= plan["decision_limits"]["allocated_ratio"],
            }
        )
    unchanged = (
        proto.content_hash(model.state_dict()) == plan["base_content_sha256"]
        and proto.content_hash(student.state_dict()) == plan["student_checkpoint"]["content_sha256"]
        and sha(plan["student_checkpoint"]["path"]) == plan["student_checkpoint"]["sha256"]
    )
    if not unchanged:
        raise ValueError("Fixed cost model identity changed")
    budget()
    return {
        "status": "completed",
        "technical_validity": "passed",
        "source": source_identity,
        "strict_loading": loading,
        "timings": timings,
        "parity": parity,
        "graph_setup": setups,
        "initial_load_peak_allocated_bytes": initial_peak,
        "stage_peak_allocated_bytes": stage_peak,
        "fixed_weights_unchanged": True,
        "gates": gates,
        "numerical_gate": all(r["passed"] for r in parity.values()),
        "cost_gate": all(r["latency_passed"] and r["memory_passed"] for r in gates),
        "optimizer_steps": 0,
        "scope": "B1 prepared-CPU to H2D/VLM/VL/q/one-condition/four-Euler/sync; processor/decode/CPU evidence excluded; inactive complete head and FP32 periphery masters offloaded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cost", "simulate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    plan = verify_plan(args)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = (
            simulate(args, plan, started + args.wall_seconds)
            if args.command == "simulate"
            else cost(args, plan, started + args.wall_seconds)
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "technical_validity": "not_established",
            "exception": type(exc).__name__,
            "message": str(exc),
            "optimizer_steps": 0,
        }
    result.update(
        plan_sha256=args.plan_sha256,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        elapsed_seconds=time.monotonic() - started,
        automatic_retry=False,
    )
    write_new(args.output / "RESULT.json", result)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
