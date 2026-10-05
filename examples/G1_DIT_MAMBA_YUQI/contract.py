"""Explicit compatibility for Yuqi compact13 checkpoints, never a silent migration."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


CAMERAS = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
GROUPS = ("left_arm", "left_hand")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            digest.update(part)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def processor_compatibility_config(raw, *, legacy_letterbox):
    result = deepcopy(raw)
    kwargs = result["processor_kwargs"]
    removed = 0
    for modalities in kwargs["modality_configs"].values():
        for config in modalities.values():
            if "active_action_keys" in config:
                if config["active_action_keys"] is not None:
                    raise ValueError("Cannot discard a nonempty Yuqi active-action mask")
                del config["active_action_keys"]
                removed += 1
    original_flag = kwargs.get("letter_box_transform", False)
    if legacy_letterbox:
        # Yuqi's old builder always pads, regardless of its saved False flag.
        # The newer builder is conditional, so True preserves actual semantics.
        kwargs["letter_box_transform"] = True
    return result, {
        "removed_null_active_action_keys": removed,
        "original_letter_box_transform": original_flag,
        "effective_letter_box_transform": kwargs.get("letter_box_transform"),
        "reason": "Preserve Yuqi legacy unconditional LetterBoxPad semantics",
    }


def assert_compact13_contract(processor, item):
    modality = processor.get_modality_configs()["new_embodiment"]
    if tuple(modality["video"].modality_keys) != CAMERAS:
        raise ValueError("Expected the three Yuqi RGB cameras")
    if (
        tuple(modality["state"].modality_keys) != GROUPS
        or tuple(modality["action"].modality_keys) != GROUPS
    ):
        raise ValueError("Expected left q7 plus hand6 only")
    if modality["action"].delta_indices != list(range(16)):
        raise ValueError("Expected H16 supervised actions")
    if any(config.rep.name != "ABSOLUTE" for config in modality["action"].action_configs):
        raise ValueError("Yuqi arm and hand actions must both be absolute")
    if tuple(item["state"].shape) != (1, 13) or tuple(item["action"].shape) != (40, 13):
        raise ValueError("Compact13/modelH40 input shape changed")
    mask = torch.as_tensor(item["action_mask"])
    if not torch.all(mask[:16] == 1) or not torch.all(mask[16:] == 0) or int(mask.sum()) != 208:
        raise ValueError("Supervision must be exactly the first 16x13 entries")
    if int(item["embodiment_id"]) != 10:
        raise ValueError("Expected ID10")


def seed_microbatch(step, rank, *, base_seed=42, initialize_cuda=True):
    value = base_seed + step * 1000 + rank
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if initialize_cuda:
        torch.cuda.manual_seed_all(value)
    return value


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
