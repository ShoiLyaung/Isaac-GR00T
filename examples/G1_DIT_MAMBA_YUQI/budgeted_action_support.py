"""Native-objective and request-lifetime helpers for a budgeted action student."""

from contextlib import ExitStack, contextmanager

from contract import tensor_identity
from gr00t.model.modules.budgeted_action_mamba import SELECTED_LAYERS, ConditionedActionRequest
import memory_prototype as proto
from memory_training_support import fresh, native_flow, sampling_packet
from runner_core import set_seed
import torch


@contextmanager
def bind_request(model, student, condition):
    original = model.action_head.model
    request = ConditionedActionRequest(student, condition)
    try:
        model.action_head.model = request
        yield request
    finally:
        model.action_head.model = original


def clean_task_loss(model, frame, *, student=None):
    """Original task loss only: no capture, hooks, evidence hashes or RNG reset.

    The caller seeds before its timer. Input transfers/native postprocessing and
    optional condition construction remain inside this call and its graph.
    """
    raw = fresh(frame["raw"], model.device)
    action = fresh(frame["action_input"], model.device)
    if student is None:
        return model.action_head(raw, action)["loss"]
    condition = student.encode_condition(frame["post_vl_features"].to(model.device))
    with bind_request(model, student, condition):
        return model.action_head(raw, action)["loss"]


def capture_velocity(model, frame, seed, *, student=None, teacher=None, replay_teacher=False):
    """One original head.forward draw; optional teacher replays actual core inputs.

    The native objective is never reimplemented. Student decoder output remains
    attached to autograd; the teacher receives detached copies of identical core
    kwargs and its decoder output is detached. No second noise/time sampling.
    """
    captured = {}
    calls = {"core": 0, "decoder": 0}

    def core_input(_module, args, kwargs):
        if args:
            raise ValueError("Expected native keyword-only core invocation")
        calls["core"] += 1
        captured["core_kwargs"] = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }

    def decoder_output(_module, args, output):
        calls["decoder"] += 1
        captured["prediction"] = output[:, -40:]

    condition = None
    with ExitStack() as scope:
        if student is not None:
            condition = student.encode_condition(frame["post_vl_features"].to(model.device))
            request = scope.enter_context(bind_request(model, student, condition))
        else:
            request = model.action_head.model
        scope.callback(request.register_forward_pre_hook(core_input, with_kwargs=True).remove)
        scope.callback(
            model.action_head.action_decoder.register_forward_hook(decoder_output).remove
        )
        result, receipt = native_flow(model, frame, None, seed)
    if calls != {"core": 1, "decoder": 1}:
        raise ValueError(f"Native core/decoder call count changed: {calls}")
    predicted = captured["prediction"]
    if predicted.shape != (1, 40, 13) or not bool(torch.isfinite(predicted).all()):
        raise ValueError("Expected finite normalized native velocity [1,40,13]")
    core_kwargs = captured["core_kwargs"]
    input_identity = {
        k: tensor_identity(v) if isinstance(v, torch.Tensor) else v for k, v in core_kwargs.items()
    }
    teacher_velocity = None
    kd = None
    if replay_teacher:
        if (
            teacher is None
            or any(p.requires_grad for p in teacher.parameters())
            or teacher.training
        ):
            raise ValueError("KD requires frozen eval B0 teacher")
        with torch.no_grad():
            teacher_output = teacher(**core_kwargs)
            if isinstance(teacher_output, tuple):
                teacher_output = teacher_output[0]
            teacher_velocity = model.action_head.action_decoder(
                teacher_output, frame["action_input"]["embodiment_id"].to(model.device)
            )[:, -40:].detach()
        kd = masked_velocity_kd(predicted, teacher_velocity, result["action_mask"])
    return {
        "native": result,
        "velocity": predicted,
        "teacher_velocity": teacher_velocity,
        "kd": kd,
        "receipt": receipt,
        "core_input_identity": input_identity,
        "condition": condition,
        "core_kwargs": core_kwargs,
    }


def masked_velocity_kd(student, teacher, mask):
    if student.shape != teacher.shape or tuple(student.shape) != (1, 40, 13):
        raise ValueError("KD velocities must share native [1,40,13] shape")
    mask = mask.float()
    if mask.shape != student.shape or not bool((mask.sum() > 0)):
        raise ValueError("Invalid native action mask")
    active = mask > 0
    a = torch.where(active, student.float(), torch.zeros_like(student.float()))
    b = torch.where(active, teacher.detach().float(), torch.zeros_like(teacher.float()))
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("Nonfinite active velocity for distillation")
    return (a - b).square().sum() / (mask.sum() + 1e-6)


def cached_action(model, frame, seed, *, student=None, modulation_zero=False):
    from gr00t.model.modules.budgeted_action_mamba import ActionCondition

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        packet = sampling_packet(model, frame)
        set_seed(seed, cuda=True)
        if student is None:
            predicted = proto.sample_packet(model, packet)
        else:
            condition = student.encode_condition(packet["features"].backbone_features)
            if modulation_zero:
                condition = ActionCondition(condition.code, torch.zeros_like(condition.modulation))
            with bind_request(model, student, condition):
                predicted = proto.sample_packet(model, packet)
    if predicted.shape != (1, 40, 13) or not bool(torch.isfinite(predicted).all()):
        raise ValueError("Invalid sampled native-shape action")
    return predicted.detach().cpu()


def current_policy(model, observation, student=None):
    if student is None:
        return model.get_action(dict(observation))["action_pred"]
    packet = proto.encode_observation(model, observation)
    condition = student.encode_condition(packet["features"].backbone_features)
    with bind_request(model, student, condition):
        return proto.sample_packet(model, packet)


def transfer_audit(student, teacher):
    mappings = [
        (f"blocks.{j}.{part}.", f"transformer_blocks.{i}.{part}.")
        for j, i in enumerate(SELECTED_LAYERS)
        for part in ("norm1", "norm3", "ff")
    ]
    mappings += [
        (name + ".", name + ".")
        for name in ("timestep_encoder", "norm_out", "proj_out_1", "proj_out_2")
    ]
    source = teacher.state_dict()
    target = student.state_dict()
    copied = {}
    for dst, src in mappings:
        for name, value in source.items():
            if name.startswith(src):
                new = dst + name[len(src) :]
                expected = value.detach().cpu().float()
                if new not in target or not torch.equal(target[new].detach().cpu(), expected):
                    raise ValueError(f"Teacher-to-student migration failed: {new}")
                copied[new] = expected
    if len(copied) != 56 or sum(v.numel() for v in copied.values()) != 197_880_832:
        raise ValueError("Selected teacher migration inventory changed")
    return {
        "copied_tensors": len(copied),
        "copied_parameters": sum(v.numel() for v in copied.values()),
        "fp32_value_hash": proto.content_hash(copied),
        "native_layer_indices": list(SELECTED_LAYERS),
        "copy_exact_after_teacher_bf16_to_fp32": True,
        "non_affine_norms": "norm1.norm/norm3/norm_out have no parameters; configuration retained",
    }


def group_modules(student):
    result = {
        "condition_adapter": student.condition_adapter,
        "timestep_encoder": student.timestep_encoder,
        "output_projection": student.proj_out_2,
        "output_modulation": student.proj_out_1,
    }
    for index, block in enumerate(student.blocks):
        result[f"block{index}.mixer"] = block.mixer
        result[f"block{index}.adaln"] = block.norm1
        result[f"block{index}.ffn"] = block.ff
    return result


def gradient_groups(student):
    return {name: proto.gradient_group(module) for name, module in group_modules(student).items()}


def parameter_changes(student, initial, seen):
    changed = {
        name: not torch.equal(value.detach().cpu(), initial[name])
        for name, value in student.state_dict().items()
    }
    prefix = {
        "condition_adapter": "condition_adapter.",
        "timestep_encoder": "timestep_encoder.",
        "output_projection": "proj_out_2.",
        "output_modulation": "proj_out_1.",
    }
    for i in range(8):
        prefix.update(
            {
                f"block{i}.mixer": f"blocks.{i}.mixer.",
                f"block{i}.adaln": f"blocks.{i}.norm1.",
                f"block{i}.ffn": f"blocks.{i}.ff.",
            }
        )
    groups = {
        name: {
            "gradient_seen": seen.get(name, False),
            "changed_tensors": [k for k, v in changed.items() if v and k.startswith(p)],
        }
        for name, p in prefix.items()
    }
    return {
        "passed": all(r["gradient_seen"] and r["changed_tensors"] for r in groups.values()),
        "groups": groups,
        "parameters": changed,
    }
