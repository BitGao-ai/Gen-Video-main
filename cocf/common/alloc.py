"""Configure CUDA caching allocator before torch import."""

from __future__ import annotations

import os

_KEY = "expandable_segments"
_VALUE = "True"


def merge_alloc_conf(existing: str, key: str = _KEY, value: str = _VALUE) -> str:
    """Merge key:value into existing alloc conf string."""
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if any(p.split(":", 1)[0].strip() == key for p in parts):
        return ",".join(parts)
    return ",".join(parts + [f"{key}:{value}"])


def configure_cuda_allocator() -> str:
    """Apply alloc conf to environment and return it."""
    conf = merge_alloc_conf(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
    return conf
