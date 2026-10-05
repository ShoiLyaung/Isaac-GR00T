"""Contract tests use an explicit CPU stand-in; GPU tests use official Mamba2."""

from unittest.mock import patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.modules import mamba_action_mixer as mixer
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class StandIn(nn.Module):
    """Causal CPU test double, never selected by production code."""

    def __init__(self, d_model, device=None, dtype=None, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, hidden):
        return self.proj(hidden).cumsum(dim=1)


class TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def config(**overrides):
    options = dict(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=2,
        max_seq_len=16,
        use_alternate_vl_dit=True,
        use_vlln=True,
        state_dropout_prob=0.0,
        action_mamba2_cfg=dict(
            d_state=16, d_conv=4, expand=2, headdim=16, chunk_size=64, use_mem_eff_path=False
        ),
        diffusion_model_cfg=dict(
            num_layers=4,
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
    options.update(overrides)
    return Gr00tN1d7Config(**options)


def features(cfg, device="cpu"):
    return BatchFeature(
        data=dict(
            backbone_features=torch.randn(1, 6, cfg.backbone_embedding_dim, device=device),
            backbone_attention_mask=torch.tensor(
                [[True, True, True, True, False, False]], device=device
            ),
            image_mask=torch.tensor([[True, True, False, False, True, False]], device=device),
        )
    )


def inputs(cfg, device="cpu"):
    return BatchFeature(
        data=dict(
            state=torch.randn(1, 1, cfg.max_state_dim, device=device),
            action=torch.randn(1, cfg.action_horizon, cfg.max_action_dim, device=device),
            action_mask=torch.ones(1, cfg.action_horizon, cfg.max_action_dim, device=device),
            embodiment_id=torch.zeros(1, dtype=torch.long, device=device),
        )
    )


@pytest.fixture
def cpu_double(monkeypatch):
    monkeypatch.setattr(mixer, "_mamba2_class", lambda: StandIn)


def test_native_default_is_bitwise_unchanged():
    torch.manual_seed(31)
    default = Gr00tN1d7ActionHead(config())
    torch.manual_seed(31)
    explicit = Gr00tN1d7ActionHead(config(action_self_mixer="dit"))
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert all(
        torch.equal(value, explicit.state_dict()[name])
        for name, value in default.state_dict().items()
    )


def test_bidirectional_is_equivariant_and_repeatable(cpu_double):
    module = mixer.BidirectionalMamba2SelfMixer(4, {})
    sequence = torch.randn(2, 5, 4)
    first = module(sequence)
    assert torch.equal(first, module(sequence))
    assert torch.equal(first.flip(1), module(sequence.flip(1)))
    changed = sequence.clone()
    changed[:, -1] += 10
    assert not torch.equal(first[:, 0], module(changed)[:, 0])
    with pytest.raises(ValueError, match="only the native unmasked"):
        module(sequence, encoder_hidden_states=sequence)


def test_only_self_mixers_change_and_gradients_flow(cpu_double):
    cfg = config()
    head = Gr00tN1d7ActionHead(cfg)
    holder = nn.Module()
    holder.config = cfg
    holder.action_head = head
    before = {name: tensor.clone() for name, tensor in holder.state_dict().items()}
    report = mixer.convert_native_model(holder)
    assert report["replacement_indices"] == [1, 3]
    assert report["unchanged_tensor_identity_and_version"]
    assert cfg.action_self_mixer == "mamba2"
    for name, value in holder.state_dict().items():
        if not name.startswith(tuple(report["replacement_prefixes"])):
            assert torch.equal(value, before[name])
    loss = head(features(cfg), inputs(cfg))["loss"]
    loss.backward()
    gradients = [p.grad for name, p in holder.named_parameters() if ".mamba2." in name]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_masked_vlm_tokens_remain_excluded(cpu_double):
    cfg = config(action_self_mixer="mamba2")
    head = Gr00tN1d7ActionHead(cfg).eval()
    original = features(cfg)
    altered = BatchFeature(data={key: value.clone() for key, value in original.items()})
    altered.backbone_features[:, 4:] += 1000 * torch.randn_like(altered.backbone_features[:, 4:])
    action = inputs(cfg)
    action.pop("action")
    action.pop("action_mask")
    torch.manual_seed(1)
    first = head.get_action(original, action)["action_pred"]
    torch.manual_seed(1)
    second = head.get_action(altered, action)["action_pred"]
    assert torch.equal(first, second)


def test_candidate_save_and_strict_reload(tmp_path, cpu_double, monkeypatch):
    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    with (
        patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls", return_value=TinyBackbone),
        patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"),
    ):
        native = Gr00tN1d7(config())
        native.save_pretrained(tmp_path / "native")
        candidate, report = mixer.load_native_then_convert(
            tmp_path / "native", local_files_only=True
        )
        assert report["unchanged_tensor_identity_and_version"]
        candidate.save_pretrained(tmp_path / "candidate")
        loaded, info = mixer.load_candidate_strict(tmp_path / "candidate", local_files_only=True)
        assert not any(
            info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
        )
        assert loaded.config.action_mamba2_cfg == candidate.config.action_mamba2_cfg
        assert loaded.state_dict().keys() == candidate.state_dict().keys()
        for name, value in loaded.state_dict().items():
            assert torch.equal(value, candidate.state_dict()[name]), name
        with pytest.raises(ValueError, match="does not declare"):
            mixer.load_candidate_strict(tmp_path / "native", local_files_only=True)
        with pytest.raises(ValueError, match="only a native"):
            mixer.load_native_then_convert(tmp_path / "candidate", local_files_only=True)


def test_missing_backend_fails_closed(monkeypatch):
    def unavailable():
        raise RuntimeError("official backend unavailable")

    monkeypatch.setattr(mixer, "_mamba2_class", unavailable)
    with pytest.raises(RuntimeError, match="official backend unavailable"):
        Gr00tN1d7ActionHead(config(action_self_mixer="mamba2"))


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA and official Mamba2 overlay"
)
def test_official_mamba2_cuda_forward_backward():
    module = mixer.BidirectionalMamba2SelfMixer(
        64,
        dict(
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            chunk_size=64,
            use_mem_eff_path=False,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    data = torch.randn(1, 41, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = module(data)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(module(data), module(data))
