"""Core — the top-level accelerator that wires every subsystem together (§2, §7).

This package holds :class:`~cocf.core.accelerator.Accelerator`, the single object
shared by the inference engine, the Stage-A teacher generator and the trainer. It
instantiates the frozen backbone and the four learnable plugins (L-COCF, RAEC,
CMSC alignment) with mutually-consistent, backbone-derived dimensions.
"""

from __future__ import annotations

from cocf.core.accelerator import Accelerator

__all__ = ["Accelerator"]
