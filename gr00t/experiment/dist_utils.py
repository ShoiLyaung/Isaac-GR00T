# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Make training work w/ and w/o distributed training.
from contextlib import contextmanager

import torch


_COLLECTIVE_TRACE: list[dict[str, object]] = []


def is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if is_dist_avail_and_initialized():
        return torch.distributed.get_rank()
    return 0


def barrier():
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


def _record_collective(label: str, operations: list[str]) -> None:
    _COLLECTIVE_TRACE.append(
        {
            "sequence": len(_COLLECTIVE_TRACE),
            "label": label,
            "operations": list(operations),
        }
    )


def record_collective(label: str, operation: str) -> None:
    """Record a clean-room evidence collective without changing its execution."""

    _record_collective(label, [operation])


def get_collective_trace() -> list[dict[str, object]]:
    """Return an immutable copy of the local collective trace."""

    return [dict(item) for item in _COLLECTIVE_TRACE]


def _collective_device() -> torch.device:
    """Select a status-collective device compatible with the active backend."""

    if torch.distributed.get_backend() == "nccl":
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@contextmanager
def run_or_wait_on_rank0(label: str | None = None):
    """Run a body on rank 0 and synchronize its failure with every peer.

    This is the narrow helper from official hardening commit ``4c0662fc``.
    It prevents a rank-0 shared-write exception from stranding peers at a later
    NCCL collective. Non-distributed execution degenerates to a normal block.
    """

    if not is_dist_avail_and_initialized():
        yield True
        return

    rank = torch.distributed.get_rank()
    is_rank0 = rank == 0
    device = _collective_device()
    status = torch.zeros(1, dtype=torch.int, device=device)
    rank0_err: BaseException | None = None
    error_summary: list[str | None] = [None]
    operations = ["all_reduce_status"]
    try:
        yield is_rank0
    except BaseException as exc:
        if is_rank0:
            status += 1
            rank0_err = exc
            error_summary[0] = f"{type(exc).__name__}: {exc}"
        else:
            raise
    finally:
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
        if status.item() != 0:
            torch.distributed.broadcast_object_list(error_summary, src=0)
            operations.append("broadcast_object_list_error")
        torch.distributed.barrier()
        operations.append("barrier")
        _record_collective(label or "run_or_wait_on_rank0", operations)

    if status.item() == 0:
        return
    if rank0_err is not None:
        raise rank0_err
    label_prefix = f"{label}: " if label else ""
    raise RuntimeError(
        f"{label_prefix}rank-0 failed inside run_or_wait_on_rank0() with "
        f"{error_summary[0] or '<no error info broadcast>'}; "
        "see rank-0's traceback for the full stack."
    ) from None


def run_on_rank0(fn, *args, label: str | None = None, **kwargs):
    """Call ``fn`` on rank 0 only and synchronize rank-0 failures."""

    result = None
    with run_or_wait_on_rank0(label=label or getattr(fn, "__qualname__", None)) as is_rank0:
        if is_rank0:
            result = fn(*args, **kwargs)
    return result
