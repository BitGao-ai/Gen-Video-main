"""Lightweight, rank-aware logging used throughout the framework."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

_CONFIGURED = False
_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S"
)


def _is_main_process() -> bool:
    # Works whether or not torch.distributed is initialised.
    return int(os.environ.get("RANK", "0")) == 0


def _ensure_stream_handler() -> logging.Logger:
    """Attach the stdout handler to the ``cocf`` root logger exactly once."""
    global _CONFIGURED
    root = logging.getLogger("cocf")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_FORMATTER)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return root


def get_logger(name: str = "cocf", level: int = logging.INFO) -> logging.Logger:
    """Return a process-aware logger; only rank 0 emits at ``level``, others WARNING."""
    _ensure_stream_handler()
    logger = logging.getLogger(name)
    logger.setLevel(level if _is_main_process() else logging.WARNING)
    return logger


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure the framework's root ``cocf`` logger and return it.

    The entry scripts call this once at start-up. It is **idempotent** — the stdout
    handler is attached only on the first call (shared with :func:`get_logger`), so
    re-invoking it never duplicates log lines. When ``log_file`` is given a
    :class:`logging.FileHandler` is added (at most one, keyed by resolved path), so a
    run can tee its log to disk without losing the console stream. Only rank 0 emits at
    ``level`` (other ranks stay at WARNING), matching :func:`get_logger`.
    """
    root = _ensure_stream_handler()
    root.setLevel(level if _is_main_process() else logging.WARNING)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path.resolve())
        already = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == target
            for h in root.handlers
        )
        if not already:
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(_FORMATTER)
            root.addHandler(fh)
    return root
