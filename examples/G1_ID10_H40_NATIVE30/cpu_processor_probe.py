"""Real-sample H40 processor/collator/save-reload probe without model weights."""

import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    EMBODIMENT_TAG,
    G1Id10H40Native30EpisodeLoader,
    apply_active_action_mask,
)
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor


def _load_modality(path: Path):
    spec = importlib.util.spec_from_file_location("g1_id10_h40_native30_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load modality config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    return MODALITY_CONFIGS[EMBODIMENT_TAG]


def _mock_vlm_processor() -> MagicMock:
    mock = MagicMock()
    mock.tokenizer.padding_side = "left"
    mock.apply_chat_template.return_value = "mock exact task"
    mock.return_value = {
        "input_ids": torch.ones((1, 4), dtype=torch.int64),
        "attention_mask": torch.ones((1, 4), dtype=torch.int64),
        "pixel_values": torch.zeros((1, 3, 16, 16), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.int64),
    }
    return mock


def _normalize(value):
    if isinstance(value, dict):
        return {str(getattr(key, "value", key)): _normalize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(child) for child in value]
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


parser = argparse.ArgumentParser()
parser.add_argument("--view-root", required=True)
parser.add_argument("--adapter-config", required=True)
parser.add_argument("--modality-config", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--processor-save", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--step", type=int, default=0)
args = parser.parse_args()

save_root = Path(args.processor_save)
output_path = Path(args.output)
if save_root.exists() or output_path.exists():
    raise FileExistsError(save_root if save_root.exists() else output_path)
modality = _load_modality(Path(args.modality_config))
embodiment = EmbodimentTag(EMBODIMENT_TAG)
loader = G1Id10H40Native30EpisodeLoader(
    args.view_root,
    modality,
    args.adapter_config,
    require_runtime_verified_fixed_state=False,
)
episode = loader[args.episode]
step = extract_step_data(episode, args.step, modality, embodiment, allow_padding=False)
statistics = {EMBODIMENT_TAG: loader.get_dataset_statistics()}
mock_vlm = _mock_vlm_processor()
loading = {"local_files_only": True, "trust_remote_code": True}

with patch(
    "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
    return_value=mock_vlm,
):
    processor = Gr00tN1d7Processor.from_pretrained(
        args.checkpoint,
        modality_configs={EMBODIMENT_TAG: modality},
        max_action_horizon=40,
        use_relative_action=True,
        transformers_loading_kwargs=loading,
    )
    processor.set_statistics(statistics, override=True)
    processor.eval()
    item = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    item = apply_active_action_mask(item)
    batch = processor.collator([item])["inputs"]
    decoded = processor.decode_action(item["action"].numpy(), embodiment, state=step.states)
    processor.save_pretrained(save_root)
    reloaded = Gr00tN1d7Processor.from_pretrained(
        save_root, transformers_loading_kwargs=loading
    )

expected = {"state": (1, 132), "action": (40, 132), "action_mask": (40, 132)}
for key, shape in expected.items():
    if tuple(item[key].shape) != shape:
        raise AssertionError(f"{key}: {tuple(item[key].shape)} != {shape}")
if tuple(batch["action"].shape) != (1, 40, 132):
    raise AssertionError("Collated action is not [1,40,132]")
if int(item["action_mask"].sum().item()) != 40 * 26:
    raise AssertionError("Active mask is not exactly 40x26")
if int(item["embodiment_id"]) != 10:
    raise AssertionError("Embodiment projector id is not 10")
if processor.max_action_horizon != 40 or reloaded.max_action_horizon != 40:
    raise AssertionError("H40 was not preserved across processor save/reload")
if reloaded.embodiment_id_mapping[EMBODIMENT_TAG] != 10:
    raise AssertionError("ID10 mapping was not preserved")
before_stats = _normalize(processor.state_action_processor.statistics)[EMBODIMENT_TAG]
after_stats = _normalize(reloaded.state_action_processor.statistics)[EMBODIMENT_TAG]
if before_stats != after_stats:
    raise AssertionError("ID10 H40 statistics changed across save/reload")
if reloaded.modality_configs[EMBODIMENT_TAG]["action"].delta_indices != list(range(40)):
    raise AssertionError("Native H40 row offsets changed across save/reload")
if reloaded.modality_configs[EMBODIMENT_TAG]["video"].delta_indices != [0]:
    raise AssertionError("Video input is not a single current frame")
if reloaded.modality_configs[EMBODIMENT_TAG]["state"].delta_indices != [0]:
    raise AssertionError("State input is not a single current row")

roundtrip = {
    key: float(abs(decoded[key] - step.actions[key]).max())
    for key in ("left_arm", "right_arm", "left_hand", "right_hand")
}
if max(roundtrip["left_arm"], roundtrip["right_arm"]) > 1e-5:
    raise AssertionError(f"Arm roundtrip exceeds 1e-5: {roundtrip}")
if max(roundtrip["left_hand"], roundtrip["right_hand"]) > 5e-5:
    raise AssertionError(f"Hand roundtrip exceeds 5e-5: {roundtrip}")

result = {
    "status": "H40_NATIVE30_CPU_ONE_BATCH_PROCESSOR_SAVE_RELOAD_PASS_NO_MODEL",
    "baseline_id": "ID10_INSPIRE6_TASK_BASELINE_H40_NATIVE30",
    "embodiment_tag": EMBODIMENT_TAG,
    "embodiment_id": int(item["embodiment_id"]),
    "reloaded_embodiment_id_mapping": int(reloaded.embodiment_id_mapping[EMBODIMENT_TAG]),
    "state_shape": list(item["state"].shape),
    "action_shape": list(item["action"].shape),
    "active_mask_values": int(item["action_mask"].sum().item()),
    "expected_active_mask_values": 1040,
    "max_action_horizon": processor.max_action_horizon,
    "reloaded_max_action_horizon": reloaded.max_action_horizon,
    "real_camera_shape": list(episode["video.ego_view"].iloc[args.step].shape),
    "active_roundtrip_max_abs": roundtrip,
    "statistics_roundtrip_equal": True,
    "action_delta_indices": list(range(40)),
    "video_delta_indices": [0],
    "state_delta_indices": [0],
    "row_grid_rate_hz": 30,
    "true_30hz_hardware_clock_proven": False,
    "strict_model_weight_reload_scope": "MODEL_WEIGHTS_NOT_LOADED",
}
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
