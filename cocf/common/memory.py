"""Memory-saving utilities — the toolbox behind user requirement #1.

These helpers are deliberately backbone-agnostic and side-effect-light so any
subsystem (trainer, teacher generator, inference engine) can opt into them
without coupling. They cover the standard levers for fitting large video-DiT
training/inference on limited VRAM:

    * mixed-precision autocast (bf16/fp16) context
    * gradient checkpointing toggling on arbitrary ``nn.Module`` trees
    * CPU<->GPU module offload (keep the frozen backbone off the GPU until used)
    * ``no_grad`` teacher-forward helper (Stage-A runs the backbone label-only)
    * a peak-memory profiler context for reporting
    * parameter freezing + trainable-parameter accounting

Design notes
------------
The single most effective memory saving in this framework is *architectural*, not
a flag: the backbone is **frozen** and the bulk of training operates on cached
latents and cached counterfactual labels (Stages A→B), so the VAE, text encoder
and (mostly) the DiT need not hold activations/optimizer state. These helpers
support that design rather than replace it.
"""

from __future__ import annotations

import contextlib
import functools
import gc
from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch
import torch.nn as nn

_DTYPES: Dict[str, Optional[torch.dtype]] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "none": None,
    "": None,
}


def resolve_dtype(name: str) -> Optional[torch.dtype]:
    key = (name or "none").lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype '{name}'. choices: {sorted(_DTYPES)}")
    return _DTYPES[key]


@contextlib.contextmanager
def autocast(device: str, dtype_name: str) -> Iterator[None]:
    """Mixed-precision context. A no-op when ``dtype_name`` resolves to None/fp32."""
    dtype = resolve_dtype(dtype_name)
    if dtype is None or dtype == torch.float32:
        yield
        return
    device_type = "cuda" if str(device).startswith("cuda") else (
        "mps" if str(device).startswith("mps") else "cpu"
    )
    # CPU autocast only supports bf16; guard so tests on CPU don't error.
    if device_type == "cpu" and dtype == torch.float16:
        dtype = torch.bfloat16
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def set_gradient_checkpointing(module: nn.Module, enabled: bool = True) -> int:
    """Enable gradient checkpointing on every submodule that supports it.

    Recognises both the HuggingFace convention (``gradient_checkpointing_enable``)
    and a plain ``gradient_checkpointing`` boolean attribute. Returns the number
    of modules toggled so callers can log/verify it took effect.
    """
    count = 0
    if hasattr(module, "gradient_checkpointing_enable") and enabled:
        module.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        count += 1
    for sub in module.modules():
        if hasattr(sub, "gradient_checkpointing") and isinstance(
            getattr(sub, "gradient_checkpointing"), bool
        ):
            setattr(sub, "gradient_checkpointing", enabled)
            count += 1
    return count


def checkpointed(fn):
    """Wrap a forward callable so its activations are recomputed in backward.

    Useful for the small trainable plugins (predictor, tube encoder, repair net)
    when they are stacked deeply. Non-reentrant variant avoids the RNG/duplicate
    warnings of the legacy implementation.
    """

    @functools.wraps(fn)
    def _inner(*args, **kwargs):
        if not torch.is_grad_enabled():
            return fn(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)

    return _inner


def freeze(module: nn.Module) -> nn.Module:
    """Freeze a module in place (no grad + eval). Returns it for chaining."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in module.parameters() if p.requires_grad)


def resolve_device(requested: str) -> str:
    """Return ``requested`` if its backend is actually available, else ``"cpu"``.

    Lets configs default to ``"cuda"`` for production while the same code runs on
    a CPU-only box (CI, this repo's tests) without manual overrides.
    """
    r = str(requested)
    if r.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if r.startswith("mps") and not getattr(torch.backends, "mps", None):
        return "cpu"
    if r.startswith("mps") and not torch.backends.mps.is_available():
        return "cpu"
    return r


@contextlib.contextmanager
def on_device(module: nn.Module, device: str, offload_to: str = "cpu") -> Iterator[nn.Module]:
    """Temporarily move ``module`` to ``device``, returning it to ``offload_to`` after.

    Lets a frozen backbone live on CPU/disk and only occupy GPU memory for the
    duration of a forward pass (``MemoryConfig.offload_backbone_to_cpu``). The
    target falls back to CPU when its backend is unavailable, so the helper is a
    safe no-op on CPU-only machines.
    """
    device = resolve_device(device)
    origin = next((p.device for p in module.parameters()), torch.device(offload_to))
    if str(origin) == device:  # already there → nothing to move/offload
        yield module
        return
    module.to(device)
    try:
        yield module
    finally:
        module.to(origin if str(origin) != device else offload_to)
        free_memory()


def free_memory() -> None:
    """Drop cached allocator blocks. Cheap no-op when CUDA is unavailable."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def teacher_forward() -> Iterator[None]:
    """``no_grad`` + ``inference_mode`` context for label-only backbone passes.

    Stage-A teacher generation and the CMSC ``Y_full`` reference pass never need
    gradients; this guarantees no activation graph is retained.
    """
    with torch.inference_mode():
        yield


@contextlib.contextmanager
def peak_memory(tag: str = "", enabled: bool = True) -> Iterator[Dict[str, float]]:
    """Profile peak CUDA memory (GiB) over a block; result filled on exit."""
    stats: Dict[str, float] = {"peak_gib": 0.0, "alloc_gib": 0.0}
    if not (enabled and torch.cuda.is_available()):
        yield stats
        return
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    try:
        yield stats
    finally:
        stats["peak_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        stats["alloc_gib"] = (torch.cuda.memory_allocated() - start) / 1024 ** 3
