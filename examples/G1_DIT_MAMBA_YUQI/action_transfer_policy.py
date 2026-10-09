"""B1 absolute-action serving with the original input, decoder and four-Euler sampler."""

from contextlib import nullcontext
from pathlib import Path
import threading

from action_transfer_checkpoint import DeployedStudent, load_student, verify_fixture_scope
from budgeted_action_cudagraph import (
    CoreDispatcher,
    ForwardGraph,
    TensorCore,
    replace_core,
    tensor_args,
)
from budgeted_action_support import current_policy
from contract import CAMERAS, GROUPS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from gr00t.policy.policy import BasePolicy
import memory_prototype as proto
import numpy as np
from run_train_eval import processor_from, strict_reload
from serve_checkpoint import validate_policy
import torch


MODES = ("native_eager", "native_graph", "student_eager", "student_graph")
LANGUAGE_KEY = "annotation.human.action.task_description"


class ActionTransferPolicy(Gr00tPolicy):
    """No persistent condition/history; graph storage is owned by this instance."""

    def __init__(self, model, processor, mode, *, student=None, task_text):
        BasePolicy.__init__(self, strict=True)
        if mode not in MODES or (mode.startswith("student") and student is None):
            raise ValueError("Unsupported serving mode or missing student")
        self.model = model
        self.processor = processor
        self.processor.eval()
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.modality_configs = {
            k: v
            for k, v in processor.get_modality_configs()["new_embodiment"].items()
            if k != "rl_info"
        }
        self.collate_fn = processor.collator
        self.language_key = self.modality_configs["language"].modality_keys[0]
        if self.language_key != LANGUAGE_KEY:
            raise ValueError("Frozen Yuqi language key changed")
        self.mode, self.student, self.task_text = mode, student, task_text
        if isinstance(student, DeployedStudent):
            student.place(model.action_head, mode)
        self.lease = self.route = None
        self.request_lock = threading.Lock()
        self.requests_completed = 0
        self.closed = False
        self.base_contract = validate_policy(self, "dit_pretrained")
        self.contract = {
            **self.base_contract,
            "condition": mode,
            "action_backbone": "budgeted_mamba" if mode.startswith("student") else "dit",
            "precision": "FP32 student parameters; BF16 autocast; BF16 frozen VLM/periphery"
            if mode.startswith("student")
            else "BF16 parameters and autocast",
            "validation_scope": "Native input/decoder contract inherited from frozen B0; action generator identified by current mode",
        }
        if isinstance(student, DeployedStudent):
            self.contract["checkpoint_kind"] = student.record["kind"]
            if student.record["kind"] == "core_periphery" and mode.startswith("student"):
                self.contract["precision"] = (
                    "FP32 core; once-materialized BF16 student periphery; frozen BF16 VLM"
                )
            self.contract["student_semantics"] = {
                key: student.record[key]
                for key in ("content_sha256", "core_content_sha256", "periphery_content_sha256")
            }
        proto.assert_frozen(model)
        if student is not None and (
            student.training or any(p.dtype != torch.float32 for p in student.parameters())
        ):
            raise ValueError("Student must remain eval with FP32 masters")

    def check_observation(self, observation):
        super().check_observation(observation)
        for key in CAMERAS:
            if observation["video"][key].shape[:2] != (1, 1):
                raise ValueError("Only B1 current-frame three-camera requests are supported")
        for key, width in zip(GROUPS, (7, 6)):
            value = observation["state"][key]
            if value.shape != (1, 1, width) or not np.isfinite(value).all():
                raise ValueError("Expected finite current q7/hand6")
        if observation["language"][LANGUAGE_KEY] != [[self.task_text]]:
            raise ValueError("Frozen task text/token layout changed")

    def check_action(self, action):
        super().check_action(action)
        for key, width in zip(GROUPS, (7, 6)):
            if action[key].shape != (1, 16, width) or not np.isfinite(action[key]).all():
                raise ValueError("Expected finite B1 absolute H16 arm7/hand6")
        if np.any((action["left_hand"] < 0) | (action["left_hand"] > 1)):
            raise ValueError("Native clipped hand-u output must remain in [0,1]")

    def prepare_observation(self, observation):
        self.check_observation(observation)
        raw = self._to_vla_step_data(self._unbatch_observation(observation)[0])
        item = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": raw}])
        inputs = _rec_to_dtype(self.collate_fn([item]), dtype=torch.bfloat16)["inputs"]
        return inputs, {k: np.stack([v]) for k, v in raw.states.items()}

    def predict_normalized(self, inputs):
        if self.closed:
            raise RuntimeError("Policy graph lifetime already closed")
        if self.mode.endswith("graph") and self.lease is None:
            raise RuntimeError("Graph must be initialized before serving")
        output = self.gpu_action(inputs, graph=self.mode.endswith("graph"))
        if output.shape != (1, 40, 13) or not bool(torch.isfinite(output).all()):
            raise ValueError("Invalid H40 normalized action")
        # Own the result before any later graph replay overwrites its static buffer.
        return output.detach().float().cpu().numpy().copy()

    def gpu_action(self, inputs, *, graph):
        """Whole RGB/VLM/head path; the caller owns timing and CPU evidence copies."""
        if self.closed:
            raise RuntimeError("Policy graph lifetime already closed")
        native_route = self.route if graph and self.mode == "native_graph" else None
        if graph and self.mode.endswith("graph") and self.lease is None:
            raise RuntimeError("Graph must be initialized before inference")
        deployment = self.student if isinstance(self.student, DeployedStudent) else None
        core = deployment.core if deployment is not None else self.student
        route = self.route if graph and self.route is not None else core
        student = route if self.mode.startswith("student") else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Bind BEFORE encode_observation: learned VL/state/encoder/decoder
            # all belong to this independent student head, while VLM is shared.
            with (
                deployment.bind(self.model)
                if deployment is not None and student is not None
                else nullcontext()
            ):
                with (
                    replace_core(self.model, native_route)
                    if native_route is not None
                    else nullcontext()
                ):
                    return current_policy(self.model, inputs, student)

    def _get_action(self, observation, options=None):
        if not self.request_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent use of one action graph is forbidden")
        try:
            inputs, states = self.prepare_observation(observation)
            normalized = self.predict_normalized(inputs)
            decoded = self.processor.decode_action(normalized, self.embodiment_tag, states)
            action = {key: np.array(decoded[key], dtype=np.float32, copy=True) for key in GROUPS}
            self.requests_completed += 1
            return action, {}
        finally:
            self.request_lock.release()

    def initialize_graph(self, observation):
        if not self.mode.endswith("graph") or self.lease is not None:
            raise ValueError("Initialize each graph mode exactly once")
        inputs, _ = self.prepare_observation(observation)
        core = (
            (self.student.core if isinstance(self.student, DeployedStudent) else self.student)
            if self.mode.startswith("student")
            else self.model.action_head.model
        )
        samples = []

        def capture(_module, _args, kwargs):
            samples.append(
                tensor_args(kwargs, student=self.mode.startswith("student"), backward=False)
            )

        handle = core.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            self.gpu_action(inputs, graph=False)
        finally:
            handle.remove()
        if len(samples) != 4:
            raise ValueError("Native sampler no longer makes four core calls")
        surface = TensorCore(core, student=self.mode.startswith("student"), backward=False)
        self.lease = ForwardGraph(surface, samples[0])
        self.route = CoreDispatcher(surface, self.lease)
        if isinstance(self.student, DeployedStudent):
            self.student.active_graphs += 1
        return dict(self.lease.setup)

    def close(self):
        if self.lease is not None:
            self.lease.close()
            if isinstance(self.student, DeployedStudent):
                self.student.active_graphs -= 1
            self.route.backend = self.route.surface = None
            self.lease = self.route = None
        if isinstance(self.student, DeployedStudent):
            self.student.assert_unchanged()
        self.closed = True


def load_models(plan):
    verify_fixture_scope(plan)
    model, loading = strict_reload(Path(plan["base_checkpoint"]))
    model.to(device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)
    if proto.content_hash(model.state_dict()) != plan["base_content_sha256"]:
        raise ValueError("B0 semantic identity changed")
    owner = load_student(model.action_head, plan["student_checkpoint"], allow_fixture=True)
    student = DeployedStudent(owner, model.action_head, plan["student_checkpoint"])
    model.cuda()
    processor = processor_from(Path(plan["prepared"]) / "processor")
    processor.eval()
    return model, student, processor, loading


def observation_from_raw(raw):
    return {
        "video": {key: np.asarray(raw.images[key], dtype=np.uint8)[None].copy() for key in CAMERAS},
        "state": {
            key: np.asarray(raw.states[key], dtype=np.float32)[None].copy() for key in GROUPS
        },
        "language": {LANGUAGE_KEY: [[raw.text]]},
    }
