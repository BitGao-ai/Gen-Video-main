"""CUDA caching-allocator configuration, importable *before* ``torch``.

Deliberately imports nothing but :mod:`os`. ``PYTORCH_CUDA_ALLOC_CONF`` is read once,
at the first CUDA use, so it has to be set before ``import torch`` initialises
anything — which means the entry scripts cannot reach it through any module that
transitively imports torch. Hence a leaf module of its own rather than a helper in
:mod:`cocf.common.vram`.
"""

from __future__ import annotations

import os

#: Allocator key this module guarantees. Stage A allocates large, *variably sized*
#: transients (VAE decode tiles, RAFT correlation volumes, expert swaps) against a
#: small residual after the frozen backbone's weights, which fragments the default
#: fixed-segment allocator badly: the resulting OOM typically reports several GiB
#: "reserved but unallocated". Expandable segments let those blocks be reused across
#: sizes.
_KEY = "expandable_segments"
_VALUE = "True"


def merge_alloc_conf(existing: str, key: str = _KEY, value: str = _VALUE) -> str:
    """Return ``existing`` with ``key:value`` added unless it is already present.

    Merged rather than overwritten: a launcher that exports the variable for an
    unrelated key (``max_split_size_mb``, ``garbage_collection_threshold``) would
    otherwise silently drop expandable segments and reintroduce exactly the
    fragmentation this guards against. An explicit ``expandable_segments`` in the
    environment still wins.
    """
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if any(p.split(":", 1)[0].strip() == key for p in parts):
        return ",".join(parts)
    return ",".join(parts + [f"{key}:{value}"])


def configure_cuda_allocator() -> str:
    """Apply :func:`merge_alloc_conf` to the environment. Returns the resolved value.

    Call at the top of an entry script, **before** ``import torch``. Returning the
    string lets the caller log the setting that is actually in force, which is the
    only way to tell a run that got expandable segments from one that did not.
    """
    conf = merge_alloc_conf(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
    return conf
