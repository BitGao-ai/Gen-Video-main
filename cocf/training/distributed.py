"""Optional data-parallel support for Stage C; a set of no-ops when single-process.

Stage C's step runs the whole accelerated engine plus several plugin losses, so it
cannot be wrapped in ``DistributedDataParallel``. This module provides the two pieces
DDP would have supplied directly: gradient averaging (:func:`average_gradients`) and
replica agreement (:func:`broadcast_parameters`). Every helper degrades to a no-op when
the process was not launched under ``torchrun``; :func:`context` reads
``torch.distributed``'s own state, so a single-process run is unaffected.

The caller must give every rank the same seed and a different data shard
(``DistributedSampler``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional, Sequence

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cocf.common.logging import get_logger

Tensor = torch.Tensor
_log = get_logger(__name__)

# Ceiling on how long a rank waits inside a collective before the backend gives up.
_COLLECTIVE_TIMEOUT = timedelta(minutes=15)

__all__ = [
    "DistContext",
    "context",
    "init_distributed",
    "shutdown",
    "resolve_device",
    "average_gradients",
    "broadcast_parameters",
    "all_agree",
    "all_reduce_mean",
    "all_reduce_min",
    "assert_same",
]


@dataclass(frozen=True)
class DistContext:
    """This process's place in the job. ``enabled=False`` for a plain single run."""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        """True on the one rank that should log, save checkpoints and print progress."""
        return self.rank == 0


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def context() -> DistContext:
    """The current context, derived from ``torch.distributed``'s own state.

    Stateless on purpose, so no module-level flag can go stale between runs. When the
    process group is not initialised this is the single-process context and every
    helper short-circuits.
    """
    if dist.is_available() and dist.is_initialized():
        return DistContext(
            enabled=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=_env_int("LOCAL_RANK", 0),
        )
    return DistContext()


def _device_index(device: str, default: int) -> int:
    """CUDA index named by ``device`` ("cuda:2" → 2), else ``default``."""
    try:
        idx = torch.device(str(device)).index
    except (RuntimeError, TypeError, ValueError):
        return default
    return default if idx is None else int(idx)


def init_distributed(device: str = "cuda") -> DistContext:
    """Join the process group when launched under ``torchrun``; otherwise do nothing.

    Detection is ``WORLD_SIZE > 1`` in the environment. The backend follows the device:
    NCCL for CUDA, Gloo otherwise. ``torch.cuda.set_device`` is called here, before any
    weight is built.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return DistContext()
    if not dist.is_available():  # pragma: no cover - torch built without distributed
        _log.warning(
            "WORLD_SIZE=%d but this torch build has no distributed support; "
            "running as a single process (ranks will NOT share gradients).", world_size,
        )
        return DistContext()
    if dist.is_initialized():  # already joined (nested call)
        return context()

    local_rank = _env_int("LOCAL_RANK", 0)
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        # Follow an explicit index when the caller gave one (e.g. "--device cuda:2").
        torch.cuda.set_device(_device_index(device, local_rank))
    dist.init_process_group(backend="nccl" if use_cuda else "gloo",
                            timeout=_COLLECTIVE_TIMEOUT)
    ctx = context()
    _log.info(
        "distributed: rank %d/%d (local_rank %d) on %s backend, collective timeout %s",
        ctx.rank, ctx.world_size, ctx.local_rank, "nccl" if use_cuda else "gloo",
        _COLLECTIVE_TIMEOUT,
    )
    return ctx


def shutdown() -> None:
    """Leave the process group if this process joined one. Safe to call always."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(device: str, ctx: Optional[DistContext] = None) -> str:
    """Pin a bare ``"cuda"`` to this rank's card; leave anything explicit alone."""
    ctx = ctx or context()
    if not ctx.enabled or device != "cuda" or not torch.cuda.is_available():
        return device
    return f"cuda:{ctx.local_rank}"


def average_gradients(params: Sequence[torch.nn.Parameter],
                      ctx: Optional[DistContext] = None) -> None:
    """All-reduce ``params``' gradients to their mean across ranks (DDP semantics).

    Called after ``backward()`` and before gradient clipping. A parameter with a
    ``None`` gradient is given an explicit zero so every rank agrees on which tensors
    take part in the collective. Gradients are flattened into one buffer for a single
    NCCL launch.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return
    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.mul_(1.0 / ctx.world_size)
    for g, reduced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(reduced)


def broadcast_parameters(tensors: Iterable[Tensor],
                         ctx: Optional[DistContext] = None, *, src: int = 0) -> int:
    """Copy rank ``src``'s weights over every other rank's, in place. Returns the count.

    Ranks normally start identical; this covers layers re-initialised on a shape
    mismatch, injected LoRA, or plugins added after the load, which would otherwise
    silently diverge across replicas.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return 0
    n = 0
    for t in tensors:
        if isinstance(t, Tensor) and t.numel():
            dist.broadcast(t.data, src=src)
            n += 1
    return n


def assert_same(value: int, what: str, ctx: Optional[DistContext] = None) -> None:
    """Fail loudly if ``value`` differs across ranks.

    Checks the two quantities that could differ (trainable-parameter count, batch
    count) up front, so a mismatch names itself instead of deadlocking in a later
    collective. This is itself a collective and must be called by every rank.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return
    t = torch.tensor([float(value)], dtype=torch.float64,
                     device=torch.device(f"cuda:{ctx.local_rank}")
                     if torch.cuda.is_available() else torch.device("cpu"))
    lo, hi = t.clone(), t.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if int(lo.item()) != int(hi.item()):
        raise RuntimeError(
            f"distributed: ranks disagree on {what} (this rank {value}, "
            f"min {int(lo.item())}, max {int(hi.item())}). Every rank must run the "
            f"same configuration — same checkpoint, same --use_lora / --wan-variant, "
            f"same data source. Continuing would deadlock in the first mismatched "
            f"collective rather than fail."
        )


def _reduced(value: float, op, ctx: DistContext,
             device: Optional[torch.device]) -> torch.Tensor:
    """Shared body of the scalar reductions: one f64 element, reduced in place."""
    if device is None:
        device = torch.device(f"cuda:{ctx.local_rank}") if torch.cuda.is_available() \
            else torch.device("cpu")
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return t


def all_reduce_min(value: float, ctx: Optional[DistContext] = None,
                   device: Optional[torch.device] = None) -> float:
    """Smallest ``value`` across ranks — for settings that MUST agree.

    Used for Stage C's per-rank VRAM-clamped batch size so the whole job adopts the
    size the smallest card can hold and no rank runs out of batches early.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return float(value)
    return float(_reduced(value, dist.ReduceOp.MIN, ctx, device).item())


def all_reduce_mean(value: float, ctx: Optional[DistContext] = None,
                    device: Optional[torch.device] = None) -> float:
    """Mean of a scalar across ranks — for logged losses and best-checkpoint decisions."""
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return float(value)
    return float(_reduced(value, dist.ReduceOp.SUM, ctx, device).item() / ctx.world_size)


def all_agree(ok: bool, ctx: Optional[DistContext] = None,
              device: Optional[torch.device] = None) -> bool:
    """``True`` only when every rank passed ``ok=True``.

    Turns a locally-failing batch (OOM, geometry mismatch, unreadable baseline) into a
    shared decision so all ranks skip it together and keep the collective sequence
    identical. This is itself a collective and must be called by every rank.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return bool(ok)
    return _reduced(1.0 if ok else 0.0, dist.ReduceOp.MIN, ctx, device).item() > 0.0
