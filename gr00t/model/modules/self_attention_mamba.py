"""Replace only native odd self-attention modules, retaining AlternateVLDiT."""

import hashlib

import torch
from torch import nn

from gr00t.model.modules.dit import AlternateVLDiT
from gr00t.model.modules.mamba_action_mixer import BidirectionalMamba2SelfMixer


def self_attention_indices(backbone):
    if not isinstance(backbone, AlternateVLDiT):
        raise ValueError("Self-Mamba requires AlternateVLDiT")
    if not backbone.config.interleave_self_attention or len(backbone.transformer_blocks) % 2:
        raise ValueError("Expected alternating cross/self blocks with even layer count")
    indices = list(range(1, len(backbone.transformer_blocks), 2))
    for index, block in enumerate(backbone.transformer_blocks):
        if (block.cross_attention_dim is None) != (index in indices):
            raise ValueError("Native cross/self block order changed")
    return indices


def install_self_mamba(backbone, config):
    """Constructor path used by strict HF reload; the native forward is unchanged."""
    indices = self_attention_indices(backbone)
    for index in indices:
        old = backbone.transformer_blocks[index].attn1
        parameter = next(old.parameters())
        candidate = BidirectionalMamba2SelfMixer(
            backbone.inner_dim, config, device=parameter.device, dtype=parameter.dtype
        )
        candidate.train(old.training)
        candidate.requires_grad_(parameter.requires_grad)
        backbone.transformer_blocks[index].attn1 = candidate
    return indices


def _tensors(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def tensor_content_hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def modify_native_self_attention(model, *, reset_only=False, mamba2_config=None):
    """Audit identity, version, and bytes of every tensor outside changed modules."""
    if model.config.action_backbone != "dit":
        raise ValueError("Conversion accepts only a native DiT model")
    backbone = model.action_head.model
    indices = self_attention_indices(backbone)
    prefixes = tuple(f"action_head.model.transformer_blocks.{i}.attn1." for i in indices)
    before = _tensors(model)
    retained = {name: value for name, value in before.items() if not name.startswith(prefixes)}
    versions = {name: value._version for name, value in retained.items()}
    before_hash = tensor_content_hash(retained)
    changed_before = {name: value for name, value in before.items() if name.startswith(prefixes)}
    changed_before_hash = tensor_content_hash(changed_before)
    config = dict(model.config.action_mamba2_cfg if mamba2_config is None else mamba2_config)
    reset_names = []
    if reset_only:
        for index in indices:
            attention = backbone.transformer_blocks[index].attn1
            covered = set()
            for name, module in attention.named_modules():
                if isinstance(module, nn.Linear):
                    covered.update(id(value) for value in module.parameters(recurse=False))
                    module.reset_parameters()
                    reset_names.append(f"{prefixes[indices.index(index)]}{name}")
            if covered != {id(value) for value in attention.parameters()}:
                raise ValueError("Self-attention reset must cover exactly its Linear parameters")
    else:
        install_self_mamba(backbone, config)
    after = _tensors(model)
    retained_after = {name: value for name, value in after.items() if not name.startswith(prefixes)}
    if retained.keys() != retained_after.keys() or any(
        retained_after[name] is not value or value._version != versions[name]
        for name, value in retained.items()
    ):
        raise RuntimeError("Conversion changed tensor identity/version outside self-attention")
    after_hash = tensor_content_hash(retained_after)
    if before_hash != after_hash:
        raise RuntimeError("Conversion changed tensor content outside self-attention")
    changed_after = {name: value for name, value in after.items() if name.startswith(prefixes)}
    changed_after_hash = tensor_content_hash(changed_after)
    if changed_before_hash == changed_after_hash:
        raise RuntimeError("Declared self-attention modification changed no tensor content")
    if not reset_only:
        model.config.action_backbone = "dit_self_mamba2"
        model.config.action_mamba2_cfg = config
        model.action_head.config.action_backbone = "dit_self_mamba2"
        model.action_head.config.action_mamba2_cfg = config
    return {
        "action_backbone": model.config.action_backbone,
        "condition": "dit_reinit_self_attention" if reset_only else "dit_self_mamba2",
        "replacement_prefixes": list(prefixes),
        "self_block_indices": indices,
        "cross_attention_blocks_retained": len(indices),
        "retained_tensor_count": len(retained),
        "retained_identity_and_version_equal": True,
        "retained_content_sha256_before": before_hash,
        "retained_content_sha256_after": after_hash,
        "changed_content_sha256_before": changed_before_hash,
        "changed_content_sha256_after": changed_after_hash,
        "removed_parameter_count": sum(v.numel() for v in changed_before.values()),
        "replacement_parameter_count": sum(v.numel() for v in changed_after.values()),
        "retained_backbone_parameter_count": sum(
            v.numel()
            for k, v in model.named_parameters()
            if k.startswith("action_head.model.") and not k.startswith(prefixes)
        ),
        "reset_linear_modules": reset_names,
        "mamba2_config": None if reset_only else config,
        "native_forward_unchanged": True,
        "time_and_output_projection_pretrained_transfer": True,
        "persistent_ssm_state": False,
    }
