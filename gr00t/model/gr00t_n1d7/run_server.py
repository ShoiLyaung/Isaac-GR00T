"""Standalone GR00T-N1.7 inference server entry point.

Mirrors ``gr00t/eval/run_gr00t_server.py`` but uses the slim N1.7 policy
adapter (which bypasses the upstream EEF modality config and the gated
Cosmos-Reason2-2B backbone).

Run inside the ``gr00t_n1d7`` conda env (transformers >= 4.57) via
``scripts/eval/run_insertion_shaft_server_n1d7.sh``.
"""

from __future__ import annotations

from dataclasses import dataclass

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer
import tyro

from gr00t.model.gr00t_n1d7.policy import (
    DEFAULT_BACKBONE_SUBSTITUTE,
    Gr00tN1d7G1ShaftPolicy,
)


@dataclass
class N1d7ServerConfig:
    """Configuration for the GR00T-N1.7 inference server."""

    model_path: str
    """Path to the N1.7-3B checkpoint directory (e.g.
    /home/sl/checkpoints/GR00T-N1.7-3B)."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.UNITREE_G1
    """Embodiment tag (must match the slim processor's modality config key)."""

    device: str = "cuda:0"
    """Device to run the model on (e.g. "cuda:0", "cpu")."""

    backbone_substitute: str = DEFAULT_BACKBONE_SUBSTITUTE
    """Public Qwen3-VL checkpoint used as processor/image-preprocessor source.
    Substitute for the gated Cosmos-Reason2-2B; same architecture."""

    host: str = "127.0.0.1"
    """Host address for the server."""

    port: int = 5556
    """Port number (default 5556 to avoid collision with the N1.6 server on 5555)."""

    n_action_steps: int = 20
    """Per-call action horizon (the env's MultiStepWrapper n_action_steps)."""

    use_sim_policy_wrapper: bool = True
    """Whether to wrap with Gr00tSimPolicyWrapper (the env produces Gr00t-sim
    flat format; required for the existing client)."""

    strict: bool = True
    """Whether to enforce strict input/output validation."""


def main(config: N1d7ServerConfig):
    print("[Gr00tN1d7Server] Starting GR00T-N1.7 inference server...")
    print(f"  Model path:           {config.model_path}")
    print(f"  Embodiment tag:       {config.embodiment_tag}")
    print(f"  Device:               {config.device}")
    print(f"  Backbone substitute:  {config.backbone_substitute}")
    print(f"  Host:                 {config.host}")
    print(f"  Port:                 {config.port}")
    print(f"  n_action_steps:       {config.n_action_steps}")
    print(f"  use_sim_policy_wrapper: {config.use_sim_policy_wrapper}")

    policy = Gr00tN1d7G1ShaftPolicy(
        model_path=config.model_path,
        embodiment_tag=config.embodiment_tag,
        device=config.device,
        backbone_substitute=config.backbone_substitute,
        n_action_steps=config.n_action_steps,
        strict=config.strict,
    )

    if config.use_sim_policy_wrapper:
        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[Gr00tN1d7Server] Shutting down server...")


if __name__ == "__main__":
    config = tyro.cli(N1d7ServerConfig)
    main(config)