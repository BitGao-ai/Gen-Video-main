"""Optional data-parallel support for Stage C (§4.2), as a set of no-ops when single-process.

Stage C's forward is not a single ``nn.Module.forward``: one step runs the whole
accelerated engine (20 denoise steps, tube segmentation, a windowed differentiable
decode) and then several losses over a handful of small plugins. ``DistributedDataParallel``
cannot wrap that — it hooks one module's autograd graph and expects every rank to
traverse it identically — so this module implements the two pieces DDP would have
provided, directly:

    * **gradient averaging** — :func:`average_gradients`, an all-reduce over the
      ~7M trainable parameters after ``backward()``. That is the whole communication
      cost: the 27B frozen backbone never has a gradient, so a Stage-C step moves
      ~28 MB over the interconnect regardless of how big the model is.
    * **replica agreement** — :func:`broadcast_parameters` before the first step, so a
      layer that was *re-initialised* rather than restored (a shape-mismatched
      ``vis_proj``, a freshly injected LoRA) is identical on every rank rather than
      silently diverging into eight different models.

Everything here degrades to a no-op when the process was not launched under
``torchrun`` — :func:`context` reads ``torch.distributed``'s own state rather than a
module global, so a single-process run behaves exactly as it did before this file
existed, with no flag to remember and no state to reset between runs.

Launch (8 GPUs, one process each)::

    torchrun --standalone --nproc_per_node=8 scripts/train/train_stage_c.py ...

The caller is responsible for two things this module cannot do for it: giving every
rank the *same* seed (so the plugins start identical) and a *different* data shard
(``DistributedSampler``). Both are wired up in ``scripts/train/train_stage_c.py`` and
:meth:`cocf.training.stage_c_finetune.FinettuneStage.run`.
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
# The default is 30 minutes, which is how a job whose rank 3 died on an OOM keeps
# eight cards at 100% utilisation for half an hour with nothing in the log. A Stage-C
# step is minutes long, so this is comfortably above any legitimate skew.
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

    Stateless on purpose: there is no module-level flag that could go stale between a
    pipeline's stages or be left set by a previous run in the same process (a test
    suite, a notebook). If the process group is not initialised, this is the
    single-process context and every helper below short-circuits.
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

    Detection is ``WORLD_SIZE > 1`` in the environment, which is what ``torchrun``
    sets and what nothing else does — so a normal ``python scripts/...`` invocation
    takes the single-process path with no flag and no risk of a hung collective.

    The backend follows the device: NCCL for CUDA (the only sane choice for a
    gradient all-reduce between GPUs), Gloo otherwise so a CPU smoke test can still
    exercise this path. ``torch.cuda.set_device`` is called *here*, before any weight
    is built, because everything downstream resolves "cuda" against the current device.
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
    if dist.is_initialized():  # already joined (nested call) — reuse it
        return context()

    local_rank = _env_int("LOCAL_RANK", 0)
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        # Follow an *explicit* index when the caller gave one ("--device cuda:2", the
        # paired-GPU layout where rank r computes on 2r and parks on 2r+1). Setting the
        # current device to LOCAL_RANK there would leave every implicit allocation —
        # NCCL's buffers included — on a card this rank does not compute on.
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
    """Pin a bare ``"cuda"`` to *this rank's* card; leave anything explicit alone.

    ``torchrun`` gives each process a ``LOCAL_RANK`` and expects it to use that device.
    A device string that already names an index (``cuda:2``) or another backend
    (``cpu``) is the caller being deliberate and is returned untouched.
    """
    ctx = ctx or context()
    if not ctx.enabled or device != "cuda" or not torch.cuda.is_available():
        return device
    return f"cuda:{ctx.local_rank}"


def average_gradients(params: Sequence[torch.nn.Parameter],
                      ctx: Optional[DistContext] = None) -> None:
    """All-reduce ``params``' gradients to their mean across ranks (DDP semantics).

    Called after ``backward()`` and **before** gradient clipping, so every rank clips
    the same averaged gradient and therefore takes an identical optimiser step.

    A parameter whose gradient is ``None`` on this rank (its branch of the §4.2 loss
    did not fire this batch — a repair that never triggered, a tube set with no
    alignment term) is given an explicit zero first. Skipping it instead would make
    ranks disagree about *which* tensors take part in the collective, and a mismatched
    all-reduce order does not error — it hangs.

    The gradients are flattened into one buffer for the collective. The payload is the
    same ~28 MB either way, but Stage B's step is milliseconds long, so paying one
    NCCL launch instead of one per tensor is the difference between communication
    being free and being the bottleneck.
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

    Ranks start from the same seed and the same checkpoint, so in the normal case this
    changes nothing. It exists for the case where they do *not*: a checkpoint tensor
    dropped on a shape mismatch and re-initialised, a LoRA adapter injected after the
    load, a plugin added to the accelerator later. Data parallelism silently computes
    nonsense if the replicas differ, and the cost of ruling that out is one 28 MB
    broadcast per run.
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
    """Fail loudly, now, if ``value`` differs across ranks.

    Everything in this module assumes the ranks agree on *how many* collectives a step
    performs and *in what order*: :func:`average_gradients` walks one parameter list,
    and the epoch loop performs one all-reduce per batch. If two ranks disagree on the
    length of that list or the number of batches, nothing raises — the shorter rank
    simply stops calling, and the others block in a collective until the job is killed
    hours later with no diagnostic at all.

    So the two quantities that could differ (trainable-parameter count, batch count)
    are checked once, up front, where the failure can name itself. The check is itself
    a collective and must therefore be called by **every** rank, unconditionally.
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

    Stage C's batch size is the case that matters: it is clamped per rank against that
    card's free VRAM, and a rank whose card has slightly less free memory (a display
    attached, a stray process) would clamp lower, take fewer batches per epoch, and
    reach the end of its shard while the others are still calling all-reduce. That
    does not fail — it hangs. Taking the minimum makes the whole job use the batch
    size the smallest card can hold.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return float(value)
    return float(_reduced(value, dist.ReduceOp.MIN, ctx, device).item())


def all_reduce_mean(value: float, ctx: Optional[DistContext] = None,
                    device: Optional[torch.device] = None) -> float:
    """Mean of a scalar across ranks — for logged losses and best-checkpoint decisions.

    Both uses need every rank to agree: a per-rank epoch mean would make rank 3 keep a
    "best" checkpoint that rank 0 discarded, on a shard the other ranks never saw.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return float(value)
    return float(_reduced(value, dist.ReduceOp.SUM, ctx, device).item() / ctx.world_size)


def all_agree(ok: bool, ctx: Optional[DistContext] = None,
              device: Optional[torch.device] = None) -> bool:
    """``True`` only when **every** rank passed ``ok=True``.

    The counterpart to :func:`assert_same` for failures that are local by nature: a
    per-clip OOM, a reference whose geometry does not match, an unreadable baseline.
    Such a rank cannot simply skip its batch — the others would block forever in the
    next gradient all-reduce — and it cannot raise either, for the same reason. Turning
    the local outcome into a shared one lets all ranks skip the batch *together*,
    keeping the collective sequence identical.

    This is itself a collective, so it must be called unconditionally by every rank.
    """
    ctx = ctx or context()
    if not ctx.enabled or ctx.world_size == 1:
        return bool(ok)
    return _reduced(1.0 if ok else 0.0, dist.ReduceOp.MIN, ctx, device).item() > 0.0
