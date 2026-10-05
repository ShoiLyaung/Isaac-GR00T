"""Loopback-only policy server for a strict compact head + frozen native VLM."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import PolicyServer

from G1_DIT_MAMBA_BASELINE.insertion_bridge import PROTOCOL_ID  # noqa: E402
from G1_DIT_MAMBA_BASELINE.run_baseline import (  # noqa: E402
    load_full_native,
    read,
    seed,
    validate_checkpoint_identity,
    write,
)
from G1_DIT_MAMBA_BASELINE.run_train_eval import load_compact_head  # noqa: E402


STATE_KEYS = {"left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"}


class CompactHeadPolicy(Gr00tPolicy):
    def __init__(self, checkpoint, cache_root, policy_seed=42):
        BasePolicy.__init__(self, strict=True)
        checkpoint = Path(checkpoint)
        metadata = read(checkpoint / "HEAD_CHECKPOINT.json")
        base = Path(metadata["base_checkpoint"])
        validate_checkpoint_identity(base, read(Path(cache_root) / "CACHE_RESULT.json"))
        self.model, self.native_loading_info = load_full_native(base)
        head, self.head_metadata = load_compact_head(checkpoint)
        self.model.action_head = head.cuda().eval()
        self.model.config = head.config
        self.model.eval()
        self.processor = Gr00tN1d7Processor.from_pretrained(
            checkpoint,
            local_files_only=True,
            transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
        )
        self.processor.eval()
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator
        self.language_key = self.modality_configs["language"].modality_keys[0]
        if set(self.modality_configs["state"].modality_keys) != STATE_KEYS:
            raise ValueError("q-only policy cannot accept extra state, velocity or torque channels")
        if (
            self.model.config.action_horizon != 40
            or self.model.action_head.num_inference_timesteps != 4
        ):
            raise ValueError("Server must preserve H40 and four Euler steps")
        self.policy_seed = policy_seed
        self.calls = 0
        seed(self.policy_seed)
        self.reset()

    def check_observation(self, observation):
        if set(observation["state"]) != STATE_KEYS:
            raise ValueError("Expected exactly the seven native q-only state groups")
        super().check_observation(observation)

    def _get_action(self, observation, options=None):
        if options:
            raise ValueError("This baseline has no RTC or inference-option variants")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        unbatched = self._unbatch_observation(observation)
        states = []
        items = []
        for obs in unbatched:
            step = self._to_vla_step_data(obs)
            states.append(step.states)
            items.append(
                self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            )
        inputs = self.collate_fn(items)
        # Match training/evaluation: FP32 master/state, BF16 compute; no implicit
        # BF16 state rounding introduced by the generic policy wrapper.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.model.get_action(**inputs)["action_pred"].float()
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite policy action")
        batched_state = {key: np.stack([state[key] for state in states]) for key in STATE_KEYS}
        decoded = self.processor.decode_action(
            prediction.cpu().numpy(), self.embodiment_tag, batched_state
        )
        result = {key: np.asarray(value, dtype=np.float32) for key, value in decoded.items()}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite physical action")
        torch.cuda.synchronize()
        service_ms = (time.perf_counter() - tick) * 1000
        self.calls += 1
        info = {
            "full_policy_service_ms": service_ms,
            "policy_call": self.calls,
            "protocol_id": PROTOCOL_ID,
            "condition": self.head_metadata["condition"],
        }
        print(json.dumps({"event": "policy_call", **info}), flush=True)
        return result, info

    def reset(self, options=None):
        # Native baseline resets no RNG. Keep the single startup-seeded stream
        # across episodes; the entire action SSM is already stateless per call.
        del options
        self.calls = 0
        return {
            "startup_policy_seed": self.policy_seed,
            "rng_reseeded": False,
            "persistent_ssm_state": False,
            "protocol_id": PROTOCOL_ID,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5567)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy = CompactHeadPolicy(args.checkpoint, args.cache_root, args.policy_seed)
    with PolicyServer(policy, host="127.0.0.1", port=args.port) as server:
        write(
            args.ready_file,
            {
                "checkpoint": str(args.checkpoint),
                "strict_reload": "PASS",
                "condition": policy.head_metadata["condition"],
                "port": args.port,
                "policy_seed": args.policy_seed,
                "protocol_id": PROTOCOL_ID,
                "runtime_verified_fixed_state": False,
            },
        )
        server.run()


if __name__ == "__main__":
    main()
