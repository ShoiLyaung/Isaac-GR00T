"""Serve an explicitly frozen action-transfer identity on loopback only."""

import argparse
import hashlib
import json
from pathlib import Path

from action_transfer_policy import MODES, ActionTransferPolicy, load_models, observation_from_raw
from action_transfer_program_support import rng_state, save_tensor_new
from budgeted_graph_distillation_runner import environment
from contract import sha, write_new
from gr00t.policy.server_client import MsgSerializer, PolicyServer
import memory_prototype as proto
from run_train_eval import NativeAnchors
from runner_core import set_seed
import torch
from yuqi_wire import install_legacy_yuqi_response_encoding


REPO = Path(__file__).resolve().parents[2]


def verify_plan(path, expected_sha256):
    if not expected_sha256 or sha(path) != expected_sha256:
        raise ValueError("Explicit frozen serving PLAN SHA required")
    plan = json.loads(Path(path).read_text())
    if Path(plan["source_worktree"]).resolve() != REPO or plan["host"] != "127.0.0.1":
        raise ValueError("Serving source/loopback identity changed")
    for row in plan["records"]:
        source = Path(row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError(f"Frozen serving material changed: {source}")
    for relative in plan["immutable_source_relative_paths"]:
        expected = {
            r["sha256"]
            for r in plan["records"]
            if Path(r["path"]).resolve() == (REPO / relative).resolve()
        }
        if len(expected) != 1 or sha(REPO / relative) not in expected:
            raise ValueError(f"Existing implementation changed: {relative}")
    checkpoint = plan["student_checkpoint"]
    if sha(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Student file changed")
    return plan


def runtime(plan):
    identity = environment(plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    safe_decoder = MsgSerializer.from_bytes
    install_legacy_yuqi_response_encoding()
    if MsgSerializer.from_bytes is not safe_decoder:
        raise ValueError("Legacy response setup changed the safe request decoder")
    return identity


def rng_identity(value=None):
    value = rng_state() if value is None else value
    return {
        "python_numpy_sha256": hashlib.sha256(
            json.dumps(
                {"python": value["python"], "numpy": value["numpy"]}, sort_keys=True
            ).encode()
        ).hexdigest(),
        "torch_sha256": proto.content_hash(
            {
                "cpu": value["torch_cpu"],
                **{f"cuda{i}": v for i, v in enumerate(value["torch_cuda"])},
            }
        ),
    }


def policy_identity(policy, plan, source, plan_sha):
    return {
        "mode": policy.mode,
        "plan_sha256": plan_sha,
        "source": source,
        "base_checkpoint": plan["base_checkpoint"],
        "base_content_sha256": plan["base_content_sha256"],
        "student_checkpoint": plan["student_checkpoint"],
        "prepared": plan["prepared"],
        "processor_config_sha256": sha(Path(plan["prepared"]) / "processor/processor_config.json"),
        "base_contract": policy.base_contract,
        "contract": policy.contract,
        "condition": policy.mode,
        "effective_action_backbone": policy.contract["action_backbone"],
        "effective_precision": policy.contract["precision"],
        "student_precision": "float32 masters; BF16 AMP",
        "base_precision": "BF16",
        "host": "127.0.0.1",
        "port": plan["port"],
        "task_text": plan["task_text"],
        "rng": "seed42 once after loading/warmup/bind; no reseeding on request/reset/metadata",
        "wire": "original safe no-pickle request decoder; legacy response encoding only",
        "condition_lifetime": "once per current observation, request-local, no history",
        "student_scope": plan["student_scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    plan = verify_plan(args.plan, args.plan_sha256)
    if args.port != plan["port"] or args.mode not in plan["modes"]:
        parser.error("Port/mode differs from explicit serving PLAN")
    if args.ready_file.exists():
        raise FileExistsError(args.ready_file)
    source_identity = runtime(plan)
    model, student, processor, loading = load_models(plan)
    policy = ActionTransferPolicy(
        model, processor, args.mode, student=student, task_text=plan["task_text"]
    )
    try:
        setup = None
        if args.mode.endswith("graph"):
            source = NativeAnchors(Path(plan["prepared"]))
            source.processor.eval()
            case = plan["fixtures"][0]
            setup = policy.initialize_graph(
                observation_from_raw(source.raw(case["episode"], case["anchor"]))
            )
        with PolicyServer(policy, host="127.0.0.1", port=args.port) as server:
            set_seed(42, cuda=True)
            initial = rng_state()
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            rng_record = save_tensor_new(args.ready_file.with_name("INITIAL_RNG.pt"), initial)
            ready = {
                "status": "ready",
                **policy_identity(policy, plan, source_identity, args.plan_sha256),
                "strict_loading": loading,
                "graph_setup": setup,
                "initial_rng": rng_record,
                "initial_rng_identity": rng_identity(initial),
                "requests_before_ready": policy.requests_completed,
            }
            write_new(args.ready_file, ready)
            print("ACTUAL_SERVER_IDENTITY " + json.dumps(ready), flush=True)
            server.run()
    finally:
        policy.close()


if __name__ == "__main__":
    main()
