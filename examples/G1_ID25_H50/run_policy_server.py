"""Authorized-only strict fresh-reload server for the G1 ID25/H50 baseline."""

from __future__ import annotations

import argparse

from gr00t.data.dataset.g1_id25_h50_adapter import EMBODIMENT_TAG
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.g1_id25_h50_open_loop import validate_loaded_policy_contract
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-model-server", action="store_true")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    if not args.execute_model_server:
        raise RuntimeError("Model load/server execution requires --execute-model-server")
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The baseline server is restricted to loopback")
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid TCP port")

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        model_path=args.model_path,
        device=args.device,
        strict_checkpoint_loading=True,
    )
    validate_loaded_policy_contract(policy)
    PolicyServer.start_server(policy, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
