"""Memory-saving helpers for training and inference."""

from __future__ import annotations

import contextlib
import functools
import gc
from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

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
    """Resolve dtype name to torch dtype."""
    key = (name or "none").lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype '{name}'. choices: {sorted(_DTYPES)}")
    return _DTYPES[key]


@contextlib.contextmanager
def autocast(device: str, dtype_name: str) -> Iterator[None]:
    """Mixed-precision autocast context."""
    dtype = resolve_dtype(dtype_name)
    if dtype is None or dtype == torch.float32:
        yield
        return
    device_type = "cuda" if str(device).startswith("cuda") else (
        "mps" if str(device).startswith("mps") else "cpu"
    )
    if device_type == "cpu" and dtype == torch.float16:
        dtype = torch.bfloat16
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def set_gradient_checkpointing(module: nn.Module, enabled: bool = True) -> int:
    """Enable gradient checkpointing on submodules."""
    count = 0
    for on_name, off_name in (
        ("enable_gradient_checkpointing", "disable_gradient_checkpointing"),
        ("gradient_checkpointing_enable", "gradient_checkpointing_disable"),
    ):
        fn = getattr(module, on_name if enabled else off_name, None)
        if callable(fn):
            fn()
            count += 1
            break
    for sub in module.modules():
        if hasattr(sub, "gradient_checkpointing") and isinstance(
            getattr(sub, "gradient_checkpointing"), bool
        ):
            if enabled and getattr(sub, "_gradient_checkpointing_func", None) is None:
                sub._gradient_checkpointing_func = functools.partial(
                    torch.utils.checkpoint.checkpoint, use_reentrant=False
                )
            setattr(sub, "gradient_checkpointing", enabled)
            count += 1
    return count


def checkpointed(fn):
    """Wrap callable with gradient checkpointing."""

    @functools.wraps(fn)
    def _inner(*args, **kwargs):
        if not torch.is_grad_enabled():
            return fn(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)

    return _inner


def freeze(module: nn.Module) -> nn.Module:
    """Freeze module in place."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    """Return trainable and total parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    """Yield trainable parameters."""
    return (p for p in module.parameters() if p.requires_grad)


def resolve_device(requested: str) -> str:
    """Resolve requested device or fall back to CPU."""
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
    """Temporarily move module to device."""
    device = resolve_device(device)
    origin = next((p.device for p in module.parameters()), torch.device(offload_to))
    if str(origin) == device:
        yield module
        return
    module.to(device)
    try:
        yield module
    finally:
        module.to(origin if str(origin) != device else offload_to)
        free_memory()


def free_memory() -> None:
    """Free cached memory blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def normal_mode() -> Iterator[None]:
    """Suspend ambient inference mode."""
    with torch.inference_mode(False):
        yield


@contextlib.contextmanager
def teacher_forward() -> Iterator[None]:
    """Run label-only backbone passes."""
    with torch.inference_mode():
        yield


@contextlib.contextmanager
def peak_memory(tag: str = "", enabled: bool = True) -> Iterator[Dict[str, float]]:
    """Profile peak CUDA memory over block."""
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
