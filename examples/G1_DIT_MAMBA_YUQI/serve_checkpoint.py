"""Serve a final Yuqi5000 compact13 checkpoint through the native policy/server.

The policy RNG is seeded once, after all loading and socket binding. Native reset
is left unchanged and does not reseed. No q26, OpenPI, simulation, or action bridge
is imported; observations and decoded actions follow Gr00tPolicy directly.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

from contract import CAMERAS, GROUPS, sha, write_new
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CONDITIONS = (
    "dit_pretrained",
    "dit_reinit_backbone",
    "conditional_mamba2",
    "dit_reinit_self_attention",
    "dit_self_mamba2",
)


def assert_source():
    import gr00t

    actual = Path(gr00t.__file__).resolve()
    expected = REPO / "gr00t" / "__init__.py"
    if actual != expected.resolve():
        raise RuntimeError(f"Wrong GR00T source: {actual}; expected {expected}")
    unrelated = [name for name in sys.modules if name == "openpi" or name.startswith("openpi.")]
    if unrelated:
        raise RuntimeError(f"OpenPI must not be imported in the native Yuqi server: {unrelated}")
    return str(actual)


def training_identity(checkpoint, result_path):
    result = json.loads(result_path.read_text())
    condition = result.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("Expected a supported final Yuqi5000 condition")
    if result.get("status") != "completed" or result.get("steps") != 5000:
        raise ValueError("Serve only the completed additional-5000-step checkpoint")
    if Path(result.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError("Training result points to a different checkpoint")
    if result.get("strict_reload_action_max_abs") != 0.0:
        raise ValueError("Final same-model strict reload must have passed exactly")
    return condition, {"path": str(result_path.resolve()), "sha256": sha(result_path)}


def validate_policy(policy, condition):
    cfg, head, processor = policy.model.config, policy.model.action_head, policy.processor
    if condition not in CONDITIONS:
        raise ValueError("Unknown training condition")
    expected_backbone = (
        condition if condition in ("conditional_mamba2", "dit_self_mamba2") else "dit"
    )
    if cfg.action_backbone != expected_backbone:
        raise ValueError("Checkpoint action backbone does not match its training condition")
    if (cfg.max_state_dim, cfg.max_action_dim, cfg.action_horizon, cfg.state_history_length) != (
        13,
        13,
        40,
        1,
    ):
        raise ValueError("Model must remain compact13, stateT1, modelH40")
    if (
        head.action_dim,
        head.action_horizon,
        head.num_inference_timesteps,
        cfg.num_inference_timesteps,
        cfg.num_timestep_buckets,
        head.num_timestep_buckets,
    ) != (13, 40, 4, 4, 1000, 1000):
        raise ValueError("Native H40/four-step Euler/1000-bucket contract changed")
    if (processor.max_state_dim, processor.max_action_dim, processor.max_action_horizon) != (
        13,
        13,
        40,
    ):
        raise ValueError("Processor dimensions differ from the compact model contract")
    if processor.embodiment_id_mapping.get("new_embodiment") != 10:
        raise ValueError("Yuqi compact13 must use native embodiment ID10")
    modalities = policy.get_modality_config()
    if tuple(modalities["video"].modality_keys) != CAMERAS:
        raise ValueError("Expected exactly the original three Yuqi cameras in native order")
    for modality in ("state", "action"):
        if tuple(modalities[modality].modality_keys) != GROUPS:
            raise ValueError("Only absolute left_arm q7 and left_hand6 may be served")
    for modality in ("video", "state", "language"):
        if modalities[modality].delta_indices != [0]:
            raise ValueError(f"Expected only the current {modality} observation")
    if modalities["action"].delta_indices != list(range(16)):
        raise ValueError("Native processor must return exactly the effective 16 action steps")
    if any(action.rep.name != "ABSOLUTE" for action in modalities["action"].action_configs):
        raise ValueError("Yuqi arm/hand actions must remain absolute")
    dimensions = {}
    params = processor.state_action_processor.norm_params["new_embodiment"]
    for modality in ("state", "action"):
        dimensions[modality] = {key: int(params[modality][key]["dim"]) for key in GROUPS}
        if list(dimensions[modality].values()) != [7, 6]:
            raise ValueError("Native statistics do not describe left q7 plus hand6")
    if not processor.letter_box_transform or not processor.state_action_processor.clip_outliers:
        raise ValueError(
            "Preserve the frozen effective Yuqi letterbox/native clipping configuration"
        )
    if policy.model.dtype != torch.bfloat16 or any(
        parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        for parameter in policy.model.parameters()
    ):
        raise ValueError("Native policy must load its complete model in BF16")
    if any(module.training for module in policy.model.modules()) or processor.training:
        raise ValueError("The complete policy must remain in eval mode")
    if not policy.strict:
        raise ValueError("Native observation/action validation must stay enabled")
    return {
        "condition": condition,
        "action_backbone": expected_backbone,
        "cameras": list(CAMERAS),
        "embodiment_id": 10,
        "state_dimensions": dimensions["state"],
        "action_dimensions": dimensions["action"],
        "state_dim": 13,
        "action_dim": 13,
        "model_action_horizon": 40,
        "effective_denoising_steps": 4,
        "num_timestep_buckets": 1000,
        "returned_action_horizon": 16,
        "action_delta_indices": list(range(16)),
        "action_representation": "absolute",
        "precision": "BF16",
        "native_decode_clipping": True,
    }


def seed_after_loading():
    # This function is invoked exactly once before the native run loop, never on reset.
    random.seed(42)
    np.random.seed(42)
    torch.random.default_generator.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def serve(args):
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("Strict native serving forbids skipping checkpoint weights")
    checkpoint = args.checkpoint.resolve(strict=True)
    result_path = args.training_result or checkpoint.parent / "TRAIN_RESULT.json"
    condition, result_identity = training_identity(checkpoint, result_path)
    if args.ready_file.exists():
        raise FileExistsError(f"Ready file already exists: {args.ready_file}")
    rng_path = args.ready_file.with_name("INITIAL_TORCH_RNG.pt")
    if rng_path.exists():
        raise FileExistsError(f"Initial RNG evidence already exists: {rng_path}")
    assert_source()
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from yuqi_wire import install_legacy_yuqi_response_encoding

    install_legacy_yuqi_response_encoding()

    policy = Gr00tPolicy(
        "new_embodiment",
        str(checkpoint),
        device=args.device,
        strict=True,
        strict_checkpoint_loading=True,
    )
    contract = validate_policy(policy, condition)
    source = assert_source()
    effective_config = policy.model.config.to_dict()
    full_config_sha = hashlib.sha256(
        json.dumps(effective_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ready = {
        "status": "ready",
        "checkpoint": str(checkpoint),
        **contract,
        "full_config_sha256": full_config_sha,
        "effective_model_config": effective_config,
        "checkpoint_config_sha256": sha(checkpoint / "config.json"),
        "processor_config_sha256": sha(checkpoint / "processor_config.json"),
        "statistics_sha256": sha(checkpoint / "statistics.json"),
        "embodiment_id_sha256": sha(checkpoint / "embodiment_id.json"),
        "training_result": result_identity,
        "gr00t_source": source,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "server_source_sha256": sha(__file__),
        "policy_source_sha256": sha(REPO / "gr00t/policy/gr00t_policy.py"),
        "loader": "native new-worktree Gr00tPolicy/AutoModel; strict_checkpoint_loading=True",
        "interface_strict": True,
        "host": "127.0.0.1",
        "port": args.port,
        "policy_rng_seed": 42,
        "rng_seed_timing": "once after loading and binding, immediately before serving",
        "episode_rng_reset": False,
        "initial_rng_path": str(rng_path.resolve()),
        "openpi_imports": [],
        "bridge": None,
        "wire_protocol": "Yuqi legacy ModalityConfig/NPY response envelopes; native safe decoder retained",
        "wire_encoder_sha256": sha(Path(__file__).with_name("yuqi_wire.py")),
        "rtc": False,
        "protocol": "Yuqi three-camera absolute left13, native processor H16 decode from model H40",
    }
    with PolicyServer(policy, host="127.0.0.1", port=args.port) as server:
        rng = seed_after_loading()
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        with rng_path.open("xb") as stream:
            torch.save(rng, stream)
        ready["initial_rng_sha256"] = sha(rng_path)
        write_new(args.ready_file, ready)
        print(
            "ACTUAL_SERVER_IDENTITY "
            + json.dumps(
                {key: value for key, value in ready.items() if key != "effective_model_config"}
            ),
            flush=True,
        )
        try:
            server.run()
        except KeyboardInterrupt:
            print("Native Yuqi server stopped.", flush=True)


def selftest():
    def modality(keys, delta, action=False):
        return SimpleNamespace(
            modality_keys=keys,
            delta_indices=delta,
            action_configs=[SimpleNamespace(rep=SimpleNamespace(name="ABSOLUTE")) for _ in keys]
            if action
            else None,
        )

    modalities = {
        "video": modality(list(CAMERAS), [0]),
        "state": modality(list(GROUPS), [0]),
        "language": modality(["annotation.human.action.task_description"], [0]),
        "action": modality(list(GROUPS), list(range(16)), action=True),
    }
    cfg = SimpleNamespace(
        action_backbone="dit",
        max_state_dim=13,
        max_action_dim=13,
        action_horizon=40,
        state_history_length=1,
        num_inference_timesteps=4,
        num_timestep_buckets=1000,
    )
    head = SimpleNamespace(
        action_dim=13, action_horizon=40, num_inference_timesteps=4, num_timestep_buckets=1000
    )
    params = {mod: {"left_arm": {"dim": 7}, "left_hand": {"dim": 6}} for mod in ("state", "action")}
    processor = SimpleNamespace(
        max_state_dim=13,
        max_action_dim=13,
        max_action_horizon=40,
        letter_box_transform=True,
        training=False,
        embodiment_id_mapping={"new_embodiment": 10},
        state_action_processor=SimpleNamespace(
            norm_params={"new_embodiment": params}, clip_outliers=True
        ),
    )
    model = SimpleNamespace(
        config=cfg,
        action_head=head,
        dtype=torch.bfloat16,
        parameters=lambda: iter([torch.zeros(1, dtype=torch.bfloat16)]),
        modules=lambda: iter([SimpleNamespace(training=False)]),
    )
    policy = SimpleNamespace(
        model=model, processor=processor, strict=True, get_modality_config=lambda: modalities
    )
    assert validate_policy(policy, "dit_pretrained")["returned_action_horizon"] == 16
    assert validate_policy(policy, "dit_reinit_backbone")["action_backbone"] == "dit"
    assert validate_policy(policy, "dit_reinit_self_attention")["action_backbone"] == "dit"
    cfg.action_backbone = "dit_self_mamba2"
    assert validate_policy(policy, "dit_self_mamba2")["action_backbone"] == "dit_self_mamba2"
    for wrong_condition in ("dit_pretrained", "dit_reinit_self_attention", "conditional_mamba2"):
        try:
            validate_policy(policy, wrong_condition)
        except ValueError as exc:
            assert "backbone" in str(exc)
        else:
            raise AssertionError("Wrong condition/backbone pairing was accepted")
    cfg.action_backbone = "conditional_mamba2"
    assert validate_policy(policy, "conditional_mamba2")["state_dim"] == 13
    modalities["action"].delta_indices = list(range(40))
    try:
        validate_policy(policy, "conditional_mamba2")
    except ValueError as exc:
        assert "effective 16" in str(exc)
    else:
        raise AssertionError("A mistaken H40 policy return horizon was accepted")
    print(
        json.dumps(
            {
                "status": "passed",
                "cpu_native_server_contract": True,
                "gpu_initialized": torch.cuda.is_initialized(),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--model-path", type=Path)
    parser.add_argument("--training-result", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
        return
    if args.checkpoint is None or args.ready_file is None:
        parser.error("--checkpoint and --ready-file are required")
    if not 1 <= args.port <= 65535 or torch.device(args.device).type != "cuda":
        parser.error("Use a valid TCP port and a CUDA device for native BF16 serving")
    torch.set_num_threads(4)
    serve(args)


if __name__ == "__main__":
    main()
