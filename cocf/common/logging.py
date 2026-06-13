"""Lightweight, rank-aware logging used throughout the framework."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False


def _is_main_process() -> bool:
    # Works whether or not torch.distributed is initialised.
    return int(os.environ.get("RANK", "0")) == 0


def get_logger(name: str = "cocf", level: int = logging.INFO) -> logging.Logger:
    """Return a process-aware logger; only rank 0 emits at ``level``, others WARNING."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("cocf")
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    logger.setLevel(level if _is_main_process() else logging.WARNING)
    return logger
