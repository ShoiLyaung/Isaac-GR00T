"""Explicit core/composite checkpoint identity and private inference-head ownership."""

from contextlib import contextmanager
from copy import copy, deepcopy
import json
from pathlib import Path

from action_transfer_periphery_support import NativeStudent
from contract import sha
from gr00t.model.modules.budgeted_action_mamba import BudgetedActionMamba
import memory_prototype as proto
import torch


KINDS = ("core_only", "core_periphery")


def read(path):
    return json.loads(Path(path).read_text())


def signature(state):
    result = {}
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Expected finite FP32 checkpoint tensor: {name}")
        result[name] = (tuple(value.shape), value.dtype)
    return result


def semantic_identity(state, kind):
    if kind not in KINDS:
        raise ValueError("Explicit core_only/core_periphery checkpoint kind required")
    full = proto.content_hash(state)
    if kind == "core_only":
        if any(k.startswith(("core.", "periphery.")) for k in state):
            raise ValueError("Composite keys cannot masquerade as core_only")
        return {
            "content_sha256": full,
            "core_content_sha256": full,
            "periphery_content_sha256": None,
        }
    core = {k.removeprefix("core."): v for k, v in state.items() if k.startswith("core.")}
    periphery = {
        k.removeprefix("periphery."): v for k, v in state.items() if k.startswith("periphery.")
    }
    if not core or not periphery or len(core) + len(periphery) != len(state):
        raise ValueError("Expected complete explicit core.* and periphery.* composite keys")
    return {
        "content_sha256": full,
        "core_content_sha256": proto.content_hash(core),
        "periphery_content_sha256": proto.content_hash(periphery),
    }


def load_structure_template(record):
    if sha(record["path"]) != record["sha256"]:
        raise ValueError("Frozen student structure-template file changed")
    state = torch.load(record["path"], map_location="cpu", weights_only=True, mmap=True)
    if proto.content_hash(state) != record["content_sha256"]:
        raise ValueError("Frozen student structure-template semantic identity changed")
    return signature(state)


def verified_state(record, expected_signature, *, allow_fixture=False):
    kind = record["kind"]
    if kind not in KINDS:
        raise ValueError("Explicit checkpoint kind required")
    path = Path(record["path"])
    if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise ValueError("Frozen candidate weight file changed")
    reference = record["training_index"]
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Frozen candidate training index changed")
    index = read(reference["path"])
    if index.get("fixture_only", False) and not allow_fixture:
        raise ValueError("A deployment fixture is not a final trained candidate")
    if (
        index["completed_step"] != record["expected_step"]
        or index["source_commit"] != record["source_commit"]
        or index["student_content_sha256"] != record["content_sha256"]
        or Path(index["weights"]["path"]).resolve() != path.resolve()
        or index["weights"]["sha256"] != record["sha256"]
        or index["weights"]["bytes"] != record["bytes"]
    ):
        raise ValueError("Candidate expected-step/source/weight/index identity mismatch")
    if bool(record.get("fixture_only", False)) != bool(index.get("fixture_only", False)):
        raise ValueError("Fixture identity must be explicit in record and index")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(state) != set(expected_signature):
        raise ValueError("Candidate tensor key inventory differs from its explicit structure")
    if signature(state) != expected_signature:
        raise ValueError("Candidate tensor shape/dtype differs from its explicit structure")
    identities = semantic_identity(state, kind)
    if any(record.get(k) != v for k, v in identities.items()):
        raise ValueError("Candidate full/core/periphery semantic identity differs")
    if kind == "core_periphery" and any(
        index.get(k) != identities[k] for k in ("core_content_sha256", "periphery_content_sha256")
    ):
        raise ValueError("Composite index core/periphery semantic identity differs")
    return state


def verify_candidate(record, expected_signature, *, allow_fixture=False):
    verified_state(record, expected_signature, allow_fixture=allow_fixture)
    return {
        key: record[key]
        for key in (
            "kind",
            "expected_step",
            "source_commit",
            "training_index",
            "content_sha256",
            "core_content_sha256",
            "periphery_content_sha256",
        )
    } | {
        "name": record.get("name"),
        "checkpoint_sha256": record["sha256"],
        "all_tensor_keys_shapes_dtypes_verified": True,
    }


def verify_fixture_scope(plan):
    if not plan["student_checkpoint"].get("fixture_only", False):
        return
    allowed = set(plan["train"] + plan["dev"])
    for pair in plan["fixtures"]:
        if (
            pair["episode"] not in allowed
            or pair["anchor"] not in plan["grids"][str(pair["episode"])]
        ):
            raise ValueError("Deployment fixture may only use explicitly frozen train/dev inputs")


def load_student(teacher_head, record, *, device="cpu", allow_fixture=False):
    core = BudgetedActionMamba(teacher_head.model)
    if record["kind"] == "core_only":
        owner = core
    elif record["kind"] == "core_periphery":
        owner = NativeStudent(teacher_head, core, "core_only_continuation")
    else:
        raise ValueError("Explicit checkpoint kind required")
    state = verified_state(record, signature(owner.state_dict()), allow_fixture=allow_fixture)
    owner.load_state_dict(state, strict=True)
    owner.to(device=device).eval().requires_grad_(False)
    return owner


def inference_head(owner, teacher_head):
    """Materialize BF16 once, while retaining canonical FP32 masters on CPU."""
    source = owner.head if isinstance(owner, NativeStudent) else teacher_head
    head = copy(source)
    head._modules = dict(source._modules)
    head._parameters = dict(source._parameters)
    head._buffers = {
        name: None if value is None else value.detach().clone()
        for name, value in source._buffers.items()
    }
    core = owner.core if isinstance(owner, NativeStudent) else owner
    for name, module in source.named_children():
        if name == "model":
            setattr(head, name, core)
        else:
            master = (
                owner.periphery[name]
                if isinstance(owner, NativeStudent) and name in owner.periphery
                else module
            )
            setattr(
                head,
                name,
                deepcopy(master)
                .to(device="cpu", dtype=torch.bfloat16)
                .eval()
                .requires_grad_(False),
            )
    return head.eval()


class DeployedStudent:
    """One source owner and one deployment head, with explicit per-phase residency."""

    def __init__(self, owner, teacher_head, record):
        self.owner = owner
        self.core = owner.core if isinstance(owner, NativeStudent) else owner
        self.head = inference_head(owner, teacher_head)
        self.record = record
        self.active_graphs = 0
        self.deployment_content_sha256 = proto.content_hash(self.head.state_dict())

    def state_dict(self):
        return self.owner.state_dict()

    def parameters(self):
        return self.owner.parameters()

    @property
    def training(self):
        return self.owner.training

    def zero_grad(self, **kwargs):
        self.owner.zero_grad(**kwargs)

    def place(self, teacher_head, mode, device="cuda"):
        if self.active_graphs:
            raise ValueError("Close the action graph before moving either head")
        self.owner.cpu()
        self.head.cpu()
        teacher_head.cpu()
        active = teacher_head if mode.startswith("native") else self.head
        active.to(device=device)
        if not mode.startswith("native") and isinstance(self.owner, NativeStudent):
            if any(p.device.type != "cpu" for p in self.owner.periphery.parameters()):
                raise ValueError("FP32 periphery masters must be CPU-only during deployment")
        inactive = [self.owner, self.head] if mode.startswith("native") else [teacher_head]
        if any(p.device.type != "cpu" for module in inactive for p in module.parameters()):
            raise ValueError("Inactive complete action head remained on accelerator")
        return active

    def assert_unchanged(self):
        if (
            proto.content_hash(self.owner.state_dict()) != self.record["content_sha256"]
            or proto.content_hash(self.head.state_dict()) != self.deployment_content_sha256
        ):
            raise ValueError("Deployment changed canonical masters or materialized head")

    @contextmanager
    def bind(self, model):
        original = model.action_head
        try:
            model.action_head = self.head
            yield
        finally:
            model.action_head = original


def save_deployment_fixture(
    model, core_record, base_content_sha256, output, source_commit, budget=lambda: None
):
    """Supervisor-only CPU construction; this index explicitly is not training evidence."""
    from action_transfer_program_support import cpu_tree, save_tensor_new
    from contract import write_new

    if core_record["kind"] != "core_only" or core_record.get("fixture_only", False):
        raise ValueError("Fixture requires an explicitly trained core-only parent")
    assert_fixture_cpu_residency(model, "B0")
    proto.assert_frozen(model)
    if proto.content_hash(model.state_dict()) != base_content_sha256:
        raise ValueError("Fixture source model differs from frozen B0")
    teacher_head = model.action_head
    if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("Build deployment fixtures from frozen CPU BF16 B0")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    budget()
    core = load_student(teacher_head, core_record, device="cpu")
    assert_fixture_cpu_residency(core, "official Mamba core")
    student = (
        NativeStudent(teacher_head, core, "core_only_continuation").eval().requires_grad_(False)
    )
    residency = assert_fixture_cpu_residency(student, "composite student")
    budget()
    weights = save_tensor_new(output / "student.pt", cpu_tree(student.state_dict()))
    budget()
    hashes = semantic_identity(student.state_dict(), "core_periphery")
    if hashes["core_content_sha256"] != core_record["core_content_sha256"]:
        raise ValueError("Fixture construction changed the trained parent core")
    index = {
        "fixture_only": True,
        "optimizer_steps": 0,
        "added_steps": 0,
        "completed_step": core_record["expected_step"],
        "source_commit": source_commit,
        "student_content_sha256": hashes["content_sha256"],
        "core_content_sha256": hashes["core_content_sha256"],
        "periphery_content_sha256": hashes["periphery_content_sha256"],
        "weights": weights,
        "core_parent": core_record,
        "base_content_sha256": base_content_sha256,
        "residency": residency,
        "scope": "Existing trained core plus unchanged B0 periphery; dev deployment fixture only",
    }
    path = output / "fixture-index.json"
    write_new(path, index)
    return {
        **weights,
        **hashes,
        "kind": "core_periphery",
        "fixture_only": True,
        "expected_step": core_record["expected_step"],
        "source_commit": source_commit,
        "training_index": {"path": str(path.resolve()), "sha256": sha(path)},
    }


CPU_FIXTURE_BASE_SHA256 = "2bff6e15f09adc0ee05c24d2445d347113d802043e41227e4eb6b6ce150a83ad"


FIXTURE_EXECUTION = "cpu_weights_cuda_driver_only"


def fixture_driver_contract(physical_gpu, execution):
    import os

    if (
        physical_gpu != 1
        or execution != FIXTURE_EXECUTION
        or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu)
    ):
        raise ValueError(
            "Fixture requires physical_gpu=1, cpu_weights_cuda_driver_only and CUDA_VISIBLE_DEVICES='1'"
        )
    if torch.cuda.device_count() != 1:
        raise ValueError("Driver-only fixture must expose exactly one CUDA device")
    return assert_fixture_zero_cuda_allocations()


def fixture_cuda_state():
    return {
        "cuda_initialized": torch.cuda.is_initialized(),
        "torch_cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
    }


def assert_fixture_zero_cuda_allocations():
    state = fixture_cuda_state()
    if state["torch_cuda_memory_allocated_bytes"] != 0:
        raise ValueError(
            f"Driver-only fixture has PyTorch CUDA allocations: {state['torch_cuda_memory_allocated_bytes']}"
        )
    return state


def assert_fixture_cpu_residency(module, owner):
    values = list(module.named_parameters()) + list(module.named_buffers())
    outside = [name for name, value in values if value.device.type != "cpu"]
    if outside:
        raise ValueError(f"Fixture {owner} parameters/buffers escaped CPU: {outside}")
    state = assert_fixture_zero_cuda_allocations()
    return {
        "owner": owner,
        "parameter_and_buffer_tensors": len(values),
        "all_parameters_and_buffers_cpu": True,
        **state,
    }


def strict_cpu_fixture_reload(path):
    """Use the native per-load CPU config override; no factory/global patch escapes."""
    import os

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    fixture_driver_contract(1, FIXTURE_EXECUTION)
    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS") == "1":
        raise ValueError("CPU fixture cannot skip pretrained weights")
    # HF consumes this native config field on the fresh config for THIS load.
    # Qwen3Backbone then omits its forced flash_attention_2 kwarg, allowing the
    # pinned Qwen3-VL implementation to select its supported SDPA default.
    model, info = Gr00tN1d7.from_pretrained(
        path,
        local_files_only=True,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
        use_flash_attention=False,
        transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Strict CPU checkpoint reload failed: {info}")
    backend = model.backbone.model.config._attn_implementation
    if backend != "sdpa" or model.config.use_flash_attention is not False:
        raise ValueError("CPU fixture did not construct the native SDPA backbone")
    assert_fixture_cpu_residency(model, "loaded B0")
    return model, {
        **info,
        "cpu_fixture_attention_backend": backend,
        "cpu_fixture_override_scope": "native use_flash_attention=False on this load only",
    }


def fixture_main():
    import argparse
    import os
    import subprocess
    import time

    from action_transfer_periphery_support import PARENT_SEMANTIC
    from contract import write_new

    parser = argparse.ArgumentParser(
        description="Create CPU task50k+B0 weights with one explicitly assigned CUDA driver context"
    )
    parser.add_argument("command", choices=("fixture",))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Driver-only fixture requires CUDA_VISIBLE_DEVICES='1'")
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Explicit CPU fixture PLAN SHA mismatch")
    plan = read(args.plan)
    fixture_driver_contract(plan["physical_gpu"], plan["fixture_execution"])
    repo = Path(__file__).resolve().parents[2]
    if (
        Path(plan["source_worktree"]).resolve() != repo
        or Path(plan["artifact_root"]).resolve() != args.artifact_root.resolve()
        or args.output.resolve() != args.artifact_root.resolve() / plan["run_id"] / "output"
        or not 0 < args.wall_seconds <= plan["internal_wall_seconds"]
        or plan["core_checkpoint"]["kind"] != "core_only"
        or plan["core_checkpoint"]["expected_step"] != 50000
        or plan["core_checkpoint"]["content_sha256"] != PARENT_SEMANTIC
        or plan["base_content_sha256"] != CPU_FIXTURE_BASE_SHA256
    ):
        raise ValueError("CPU fixture source/parent/output/budget identity differs")
    for row in plan["records"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen CPU fixture material changed: {path}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", plan["source_base"], "HEAD"], cwd=repo, text=True
        ).splitlines()
    )
    if dirty or not changed.issubset(plan["allowed_source_files"]):
        raise ValueError("CPU fixture requires clean committed private source")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()

    def budget():
        size = sum(p.stat().st_size for p in args.artifact_root.rglob("*") if p.is_file())
        if time.monotonic() - started > args.wall_seconds or size > plan["max_artifact_bytes"]:
            raise RuntimeError("CPU fixture wall/storage budget exceeded")
        assert_fixture_zero_cuda_allocations()
        return size

    torch.set_num_threads(4)
    try:
        model, loading = strict_cpu_fixture_reload(Path(plan["base_checkpoint"]))
        model.to(device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)
        record = save_deployment_fixture(
            model,
            plan["core_checkpoint"],
            plan["base_content_sha256"],
            args.output,
            source_commit,
            budget,
        )
        budget()
        residency = assert_fixture_cpu_residency(model, "final B0")
        cuda_state = assert_fixture_zero_cuda_allocations()
        write_new(
            args.output / "RESULT.json",
            {
                "status": "passed",
                "fixture_only": True,
                "checkpoint": record,
                "source_commit": source_commit,
                "plan_sha256": args.plan_sha256,
                "strict_loading": loading,
                "optimizer_steps": 0,
                "encoded_observations": 0,
                "fixture_execution": plan["fixture_execution"],
                "physical_gpu": plan["physical_gpu"],
                "b0_residency": residency,
                **cuda_state,
                "elapsed_seconds": time.monotonic() - started,
                "artifact_bytes": budget(),
            },
        )
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_new(
            args.output / "RESULT.json",
            {
                "status": "failed",
                "fixture_only": True,
                "plan_sha256": args.plan_sha256,
                "source_commit": source_commit,
                "optimizer_steps": 0,
                "encoded_observations": 0,
                "fixture_execution": plan["fixture_execution"],
                "physical_gpu": plan["physical_gpu"],
                **fixture_cuda_state(),
                "exception": type(exc).__name__,
                "message": str(exc),
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    fixture_main()
