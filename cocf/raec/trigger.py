"""Risk trigger & force-FULL bookkeeping (§5.3.2).

Maps each tube's error certificate onto one of three actions, with two thresholds::

    E_cert ≤ τ_low              KEEP      trust the skip, do nothing
    τ_low < E_cert ≤ τ_high     REPAIR    boundary fusion + cache refresh (local fix)
    E_cert > τ_high             ROLLBACK  restore last safe anchor + force FULL for q steps

After a rollback a tube is pinned to FULL for ``force_full_steps`` (``q``) steps so
the trajectory re-stabilises before the tube is allowed to skip again — the
discrete analogue of the late-step contraction that bounds error propagation
(§5.4). This module owns only the *policy*; the actual latent edits live in
:mod:`cocf.raec.repair` and the anchor state in :mod:`cocf.raec.anchor_store`.
"""

from __future__ import annotations

from typing import Dict

from cocf.common.config import TriggerConfig
from cocf.common.types import ErrorCertificate, TriggerLevel

__all__ = ["RiskTrigger"]


class RiskTrigger:
    """Thresholds certificates into :class:`TriggerLevel` and tracks force-FULL pins."""

    def __init__(self, config: TriggerConfig) -> None:
        self.cfg = config
        # tube_id -> remaining steps it must stay FULL after a rollback
        self._force_full: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # classification
    # ------------------------------------------------------------------ #

    def classify_value(self, value: float) -> TriggerLevel:
        if value > self.cfg.tau_high:
            return TriggerLevel.ROLLBACK
        if value > self.cfg.tau_low:
            return TriggerLevel.REPAIR
        return TriggerLevel.KEEP

    def classify(self, cert: ErrorCertificate) -> TriggerLevel:
        return self.classify_value(cert.value)

    def classify_all(
        self, certs: Dict[int, ErrorCertificate]
    ) -> Dict[int, TriggerLevel]:
        return {tid: self.classify(c) for tid, c in certs.items()}

    # ------------------------------------------------------------------ #
    # force-FULL bookkeeping (the "q steps" after a rollback)
    # ------------------------------------------------------------------ #

    def register_rollback(self, tube_id: int) -> None:
        """Pin a tube to FULL for the next ``force_full_steps`` steps."""
        self._force_full[tube_id] = self.cfg.force_full_steps

    def is_forced_full(self, tube_id: int) -> bool:
        return self._force_full.get(tube_id, 0) > 0

    def forced_full_tubes(self) -> set:
        return {tid for tid, n in self._force_full.items() if n > 0}

    def step(self) -> None:
        """Advance one denoising step: decrement every active force-FULL counter."""
        expired = []
        for tid in self._force_full:
            self._force_full[tid] -= 1
            if self._force_full[tid] <= 0:
                expired.append(tid)
        for tid in expired:
            del self._force_full[tid]

    def reset(self) -> None:
        self._force_full.clear()
