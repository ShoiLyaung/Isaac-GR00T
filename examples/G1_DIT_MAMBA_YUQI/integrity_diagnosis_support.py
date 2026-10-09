"""Read-only canonical-owner tensor checks for one frozen-base failure diagnosis."""

import hashlib
import json
from pathlib import Path

from contract import write_new
import memory_prototype as proto
import numpy as np
import torch


class BaseMutation(RuntimeError):
    pass


def cpu_copy(value):
    return value.detach().to(device="cpu", copy=True).contiguous()


def digest(value, chunk_bytes):
    raw = value.reshape(-1).view(torch.uint8).numpy()
    h = hashlib.sha256()
    for start in range(0, raw.size, chunk_bytes):
        h.update(memoryview(raw[start : start + chunk_bytes]))
    return h.hexdigest()


def storage(value):
    try:
        version = value._version
    except RuntimeError:
        version = None
    return {
        "device": str(value.device),
        "data_ptr": value.data_ptr(),
        "storage_ptr": value.untyped_storage().data_ptr(),
        "storage_bytes": value.untyped_storage().nbytes(),
        "offset": value.storage_offset(),
        "version": version,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "requires_grad": value.requires_grad,
    }


def difference(before, after, chunk_bytes, samples):
    result = {"comparable": before.shape == after.shape and before.dtype == after.dtype}
    if not result["comparable"]:
        return {
            **result,
            "before_values": list(map(str, before.reshape(-1)[:samples].tolist())),
            "after_values": list(map(str, after.reshape(-1)[:samples].tolist())),
        }
    a = before.reshape(-1).view(torch.uint8).numpy()
    b = after.reshape(-1).view(torch.uint8).numpy()
    count = 0
    offsets = []
    for start in range(0, a.size, chunk_bytes):
        local = np.flatnonzero(a[start : start + chunk_bytes] != b[start : start + chunk_bytes])
        count += int(local.size)
        if len(offsets) < samples:
            offsets.extend((local[: samples - len(offsets)] + start).tolist())
    elements = list(dict.fromkeys(index // before.element_size() for index in offsets))[:samples]
    return {
        **result,
        "changed_bytes": count,
        "byte_offsets_sample": offsets,
        "before_bytes_sample": [int(a[i]) for i in offsets],
        "after_bytes_sample": [int(b[i]) for i in offsets],
        "element_indices_sample": elements,
        "before_values_sample": [str(before.reshape(-1)[i].item()) for i in elements],
        "after_values_sample": [str(after.reshape(-1)[i].item()) for i in elements],
    }


class BaseAudit:
    """Stable original owners bypass only the known temporary head Trace wrappers."""

    def __init__(
        self,
        model,
        output,
        expected_hash,
        chunk_bytes=1 << 20,
        samples=64,
        synchronize=None,
        budget=lambda: None,
    ):
        self.model = model
        self.output = Path(output)
        self.chunk_bytes = chunk_bytes
        self.samples = samples
        self.budget = budget
        self.synchronize = synchronize or (
            lambda: torch.cuda.synchronize() if torch.cuda.is_initialized() else None
        )
        self.refs = {}
        self.before = {}
        self.initial = {}
        self.checks = 0
        self.failed = False
        self.phase = "initialization"
        self.extra = lambda: {}
        self.bindings = {
            name: getattr(model.action_head, name)
            for name in ("model", "action_encoder", "action_decoder")
        }
        self.synchronize()
        parameters = dict(model.named_parameters(remove_duplicate=False))
        buffers = dict(model.named_buffers(remove_duplicate=False))
        for name, value in model.state_dict(keep_vars=True).items():
            prefix, _, leaf = name.rpartition(".")
            owner = model.get_submodule(prefix) if prefix else model
            kind = (
                "parameter"
                if name in parameters
                else "persistent_buffer"
                if name in buffers
                else "unknown"
            )
            if kind == "unknown":
                raise ValueError(f"Unclassified base state tensor: {name}")
            self.refs[name] = (owner, leaf, kind)
            copy = cpu_copy(value)
            self.before[name] = copy
            self.initial[name] = {
                "kind": kind,
                "sha256": digest(copy, chunk_bytes),
                **storage(value),
            }
        self.full_hash = proto.content_hash(self.before)
        if self.full_hash != expected_hash:
            raise ValueError("Initial frozen base differs from original B0")
        write_new(
            self.output / "INITIAL_BASE.json",
            {
                "base_hash": self.full_hash,
                "tensors": self.initial,
                "registered_buffers": {name: storage(v) for name, v in buffers.items()},
                "cpu_clone_bytes": sum(v.numel() * v.element_size() for v in self.before.values()),
                "full_base_saved_to_disk": False,
            },
        )

    def live(self, name):
        owner, leaf, kind = self.refs[name]
        return (owner._parameters if kind == "parameter" else owner._buffers).get(leaf)

    def binding_state(self):
        return {
            name: {
                "restored": getattr(self.model.action_head, name) is module,
                "actual_type": type(getattr(self.model.action_head, name)).__name__,
                "expected_type": type(module).__name__,
            }
            for name, module in self.bindings.items()
        }

    def aliases(self):
        extras = {
            name: storage(value)
            for name, value in self.extra().items()
            if isinstance(value, torch.Tensor)
        }
        hits = []
        for name in self.refs:
            value = self.live(name)
            if value is None:
                continue
            a = storage(value)
            lo = a["storage_ptr"]
            hi = lo + a["storage_bytes"]
            for other, b in extras.items():
                if a["device"] == b["device"] and max(lo, b["storage_ptr"]) < min(
                    hi, b["storage_ptr"] + b["storage_bytes"]
                ):
                    hits.append(
                        {"base": name, "other": other, "base_storage": a, "other_storage": b}
                    )
        return {"external_tensor_inventory": extras, "base_external_storage_overlaps": hits}

    def check(self, stage, *, stable=False):
        self.budget()
        self.synchronize()
        self.checks += 1
        bindings = self.binding_state()
        live_state = self.model.state_dict(keep_vars=True) if stable else None
        keys = list(live_state) if stable else None
        structural = {}
        if stable:
            if set(keys) != set(self.refs):
                structural["keys"] = {
                    "added": sorted(set(keys) - set(self.refs)),
                    "removed": sorted(set(self.refs) - set(keys)),
                }
            if not all(v["restored"] for v in bindings.values()):
                structural["bindings"] = bindings
        changed = []
        fragments = {}
        for name in sorted(self.refs):
            value = live_state.get(name) if stable else self.live(name)
            old = self.initial[name]
            if value is None:
                changed.append(
                    {"name": name, "kind": old["kind"], "removed_from_original_owner": True}
                )
                continue
            now = cpu_copy(value)
            current = {
                "kind": old["kind"],
                "sha256": digest(now, self.chunk_bytes),
                **storage(value),
            }
            if (current["sha256"], current["shape"], current["dtype"]) != (
                old["sha256"],
                old["shape"],
                old["dtype"],
            ):
                delta = difference(self.before[name], now, self.chunk_bytes, self.samples)
                changed.append({"name": name, "before": old, "after": current, "difference": delta})
                indices = delta.get(
                    "element_indices_sample", list(range(min(self.samples, now.numel())))
                )
                fragments[name] = {
                    "before": self.before[name].reshape(-1)[: self.samples].clone()
                    if not delta["comparable"]
                    else self.before[name].reshape(-1)[indices].clone(),
                    "after": now.reshape(-1)[: self.samples].clone()
                    if not delta["comparable"]
                    else now.reshape(-1)[indices].clone(),
                }
            del now
        event = {
            "check": self.checks,
            "phase": self.phase,
            "stage": stage,
            "stable_boundary": stable,
            "tensor_count": len(self.refs),
            "bindings": bindings,
            "changed_tensor_count": len(changed),
            "structural_change": structural,
        }
        with (self.output / "AUDIT.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if changed or structural:
            self.failed = True
            report = {
                **event,
                "initial_base_hash": self.full_hash,
                "changed_tensors": changed,
                "alias_information": self.aliases(),
                "first_mutation_stops_matrix": True,
                "full_base_saved_to_disk": False,
            }
            write_new(self.output / "FIRST_MUTATION.json", report)
            with (self.output / "DIFFERENCE_FRAGMENTS.pt").open("xb") as stream:
                torch.save(fragments, stream)
            raise BaseMutation(f"First base difference at {self.phase}/{stage}")
        return event
