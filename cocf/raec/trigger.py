"""Risk trigger and force-FULL bookkeeping."""

from __future__ import annotations

from typing import Dict

from cocf.common.config import TriggerConfig
from cocf.common.types import ErrorCertificate, TriggerLevel

__all__ = ["RiskTrigger"]


class RiskTrigger:
    """Thresholds certificates into trigger levels."""

    def __init__(self, config: TriggerConfig) -> None:
        self.cfg = config
        self._force_full: Dict[int, int] = {}
        self._unmeasured: Dict[int, int] = {}

    def note_measured(self, tube_ids) -> None:
        """Clear unmeasured counters for measured tubes."""
        for tid in tube_ids:
            self._unmeasured.pop(tid, None)

    def note_unmeasured(self, tube_ids) -> None:
        """Count steps without measured residuals per tube."""
        for tid in tube_ids:
            self._unmeasured[tid] = self._unmeasured.get(tid, 0) + 1

    def coverage_exhausted(self, max_unmeasured: int) -> bool:
        """Check if any tube exceeded max unmeasured steps."""
        if max_unmeasured <= 0:
            return False
        return any(n >= max_unmeasured for n in self._unmeasured.values())

    def unmeasured_steps(self, tube_id: int) -> int:
        """Return unmeasured count for a tube."""
        return self._unmeasured.get(tube_id, 0)

    def retain(self, live_ids) -> None:
        """Drop bookkeeping for retired tubes."""
        live = set(live_ids)
        self._unmeasured = {tid: n for tid, n in self._unmeasured.items() if tid in live}
        self._force_full = {tid: n for tid, n in self._force_full.items() if tid in live}

    def classify_value(self, value: float) -> TriggerLevel:
        """Classify risk value into trigger level."""
        if value > self.cfg.tau_high:
            return TriggerLevel.ROLLBACK
        if value > self.cfg.tau_low:
            return TriggerLevel.REPAIR
        return TriggerLevel.KEEP

    def classify(self, cert: ErrorCertificate) -> TriggerLevel:
        """Classify certificate into trigger level."""
        return self.classify_value(cert.value)

    def classify_all(
        self, certs: Dict[int, ErrorCertificate]
    ) -> Dict[int, TriggerLevel]:
        """Classify all certificates into trigger levels."""
        return {tid: self.classify(c) for tid, c in certs.items()}

    def register_rollback(self, tube_id: int) -> None:
        """Pin tube to FULL for configured steps."""
        self._force_full[tube_id] = self.cfg.force_full_steps

    def register_repair(self, tube_id: int, steps: int = 1) -> None:
        """Pin repaired tube to FULL for refresh steps."""
        self._force_full[tube_id] = max(self._force_full.get(tube_id, 0), int(steps))

    def is_forced_full(self, tube_id: int) -> bool:
        """Check if tube is pinned to FULL."""
        return self._force_full.get(tube_id, 0) > 0

    def forced_full_tubes(self) -> set:
        """Return tubes currently pinned to FULL."""
        return {tid for tid, n in self._force_full.items() if n > 0}

    def step(self) -> None:
        """Advance counters by one denoising step."""
        expired = []
        for tid in self._force_full:
            self._force_full[tid] -= 1
            if self._force_full[tid] <= 0:
                expired.append(tid)
        for tid in expired:
            del self._force_full[tid]

    def reset(self) -> None:
        """Clear all trigger bookkeeping."""
        self._force_full.clear()
        self._unmeasured.clear()
