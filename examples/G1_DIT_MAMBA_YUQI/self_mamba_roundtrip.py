"""Official CUDA Mamba2 tiny HF strict-save/reload smoke; no training or VLM weights."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import time
from unittest.mock import patch

from contract import sha, tensor_identity, write_new
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.modules.self_attention_mamba import (
    modify_native_self_attention,
    tensor_content_hash,
)
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


OLD_OVERLAY = Path(
    "/home/sl/research/icra27-artifacts/g1-dit-mamba-yuqi-5000step-r1/dependencies/python-overlay"
)


class TinyBackbone(nn.Module):
    """The expensive VLM alone is stubbed; all action-head modules are real."""

    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def tiny_config():
    return Gr00tN1d7Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=13,
        max_action_dim=13,
        action_horizon=40,
        state_history_length=1,
        num_inference_timesteps=4,
        max_num_embodiments=16,
        max_seq_len=64,
        use_alternate_vl_dit=True,
        use_vlln=True,
        state_dropout_prob=0.0,
        action_mamba2_cfg=dict(
            d_state=16, d_conv=4, expand=2, headdim=16, chunk_size=16, use_mem_eff_path=False
        ),
        diffusion_model_cfg=dict(
            num_layers=32,
            num_attention_heads=2,
            attention_head_dim=16,
            norm_type="ada_norm",
            dropout=0.0,
            final_dropout=False,
            output_dim=32,
            positional_embeddings=None,
            interleave_self_attention=True,
        ),
    )


def clone_feature(values):
    # process_backbone_output replaces its feature entry, so every call gets a new container.
    return BatchFeature(data={key: value.clone() for key, value in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 600:
        parser.error("Wall budget must be <=600 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.wall_seconds

    def check_deadline():
        if time.monotonic() > deadline:
            raise TimeoutError("Official Mamba HF roundtrip exceeded wall budget")

    try:
        import gr00t
        import mamba_ssm.modules.mamba2 as official

        repo = Path(__file__).resolve().parents[2]
        if Path(gr00t.__file__).resolve() != repo / "gr00t/__init__.py":
            raise ValueError("GR00T source is outside this worktree")
        if OLD_OVERLAY not in Path(official.__file__).resolve().parents:
            raise ValueError("Use the original frozen Mamba overlay")
        if official.causal_conv1d_fn is not None:
            raise ValueError("Original convolution fallback contract changed")
        if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
            raise ValueError("Roundtrip must load actual checkpoint tensors")
        if not torch.cuda.is_available():
            raise RuntimeError("Official Mamba roundtrip requires CUDA")
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        checkpoint = args.output / "tiny-checkpoint"
        with (
            patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls", return_value=TinyBackbone),
            patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"),
        ):
            model = Gr00tN1d7(tiny_config())
            audit = modify_native_self_attention(model)
            model = model.to(device="cuda", dtype=torch.bfloat16).eval()
            model.save_pretrained(checkpoint)
            check_deadline()
            restored, loading = Gr00tN1d7.from_pretrained(
                checkpoint,
                local_files_only=True,
                output_loading_info=True,
                ignore_mismatched_sizes=False,
            )
            if any(
                loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise ValueError(f"Strict tiny checkpoint load failed: {loading}")
            restored = restored.to(device="cuda", dtype=torch.bfloat16).eval()
        before, after = model.state_dict(), restored.state_dict()
        if before.keys() != after.keys() or any(
            before[name].dtype != after[name].dtype or not torch.equal(before[name], after[name])
            for name in before
        ):
            raise ValueError("Saved/reloaded state_dict tensors are not byte equal")
        weights_sha = tensor_content_hash(before)
        if weights_sha != tensor_content_hash(after):
            raise ValueError("Saved/reloaded state_dict hashes differ")
        for candidate in (model, restored):
            modules = [m for m in candidate.modules() if isinstance(m, official.Mamba2)]
            if len(modules) != 16 or any(m.use_mem_eff_path for m in modules):
                raise ValueError("Expected sixteen official unfused Mamba2 modules")
        torch.manual_seed(73)
        torch.cuda.manual_seed_all(73)
        features = {
            "backbone_features": torch.randn(1, 6, 32, device="cuda", dtype=torch.bfloat16),
            "backbone_attention_mask": torch.tensor(
                [[True, True, True, True, False, False]], device="cuda"
            ),
            "image_mask": torch.tensor([[True, True, False, False, True, False]], device="cuda"),
        }
        inputs = {
            "state": torch.randn(1, 1, 13, device="cuda", dtype=torch.bfloat16),
            "embodiment_id": torch.tensor([10], device="cuda"),
            "action": torch.randn(1, 40, 13, device="cuda", dtype=torch.bfloat16),
            "action_mask": torch.cat((torch.ones(1, 16, 13), torch.zeros(1, 24, 13)), dim=1).to(
                "cuda"
            ),
        }
        actions = {
            key: value for key, value in inputs.items() if key not in ("action", "action_mask")
        }

        def run(candidate):
            check_deadline()
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = candidate.action_head(clone_feature(features), clone_feature(inputs))
            if not bool(torch.isfinite(output["action_loss"]).all()):
                raise ValueError("Official action-head forward is nonfinite")
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = candidate.action_head.get_action(
                    clone_feature(features), clone_feature(actions)
                )["action_pred"]
            torch.cuda.synchronize()
            if predicted.shape != (1, 40, 13) or not bool(torch.isfinite(predicted).all()):
                raise ValueError("Official sampled action must be finite (1,40,13)")
            return predicted, output["action_loss"]

        for candidate in (model, restored):
            for _ in range(3):
                run(candidate)
        original_action, original_loss = run(model)
        restored_action, restored_loss = run(restored)
        if not torch.equal(original_action, restored_action) or not torch.equal(
            original_loss, restored_loss
        ):
            raise ValueError("Fixed-seed official Mamba action/loss roundtrip differs")
        check_deadline()
        write_new(
            args.output / "RESULT.json",
            {
                "status": "passed",
                "scope": "tiny official CUDA SSM HF constructor/save/load smoke; synthetic VLM features; no task-quality evidence",
                "optimizer_steps": 0,
                "layers": 32,
                "official_mamba_modules": 16,
                "model_horizon": 40,
                "active_dimensions": 13,
                "denoising_steps": 4,
                "precision": "BF16",
                "strict_loading": loading,
                "weights_byte_equal": True,
                "state_dict_sha256": weights_sha,
                "action_max_abs": 0.0,
                "forward_action_loss_max_abs": 0.0,
                "action": tensor_identity(original_action),
                "forward_action_loss": tensor_identity(original_loss),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_config_sha256": sha(checkpoint / "config.json"),
                "files": {f.name: sha(f) for f in checkpoint.iterdir() if f.is_file()},
                "gr00t_file": gr00t.__file__,
                "mamba_file": official.__file__,
                "versions": {
                    key: importlib.metadata.version(key) for key in ("torch", "mamba-ssm", "triton")
                },
                "input_identity": {
                    key: tensor_identity(value) for key, value in {**features, **inputs}.items()
                },
                "audit": audit,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    except Exception as exc:
        write_new(
            args.output / "FAILURE.json",
            {
                "status": "failed",
                "exception": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
