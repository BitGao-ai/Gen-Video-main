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
        # tube_id -> consecutive steps whose skip residual δ went unmeasured (§P4-A2)
        self._unmeasured: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # certificate-coverage bookkeeping (what gates the whole-step skip)
    # ------------------------------------------------------------------ #

    def note_measured(self, tube_ids) -> None:
        """Clear the unmeasured counter for tubes whose δ the transition just measured."""
        for tid in tube_ids:
            self._unmeasured.pop(tid, None)

    def note_unmeasured(self, tube_ids) -> None:
        """Charge a step to tubes whose skip produced no residual to certify against.

        A whole-step skip computes nothing, so there is no reference to measure a
        skipped tube's δ against and the certificate's λ_res term sees 0 — it cannot
        price the error the skip introduced. That is exactly why the promotion is
        opt-in. Counting how long each tube has gone uncertified turns "we might be
        flying blind" into a bounded, checkable quantity (see
        :meth:`coverage_exhausted`).
        """
        for tid in tube_ids:
            self._unmeasured[tid] = self._unmeasured.get(tid, 0) + 1

    def coverage_exhausted(self, max_unmeasured: int) -> bool:
        """Whether any tube has gone ``max_unmeasured`` steps without a measured δ.

        The transition executor consults this before promoting a step to a whole-step
        skip: once a tube hits the bound, the step must run so its residual can be
        measured and its certificate re-grounded. ``max_unmeasured <= 0`` disables the
        invariant (unbounded skipping, the pre-§P4 behaviour).
        """
        if max_unmeasured <= 0:
            return False
        return any(n >= max_unmeasured for n in self._unmeasured.values())

    def unmeasured_steps(self, tube_id: int) -> int:
        return self._unmeasured.get(tube_id, 0)

    def retain(self, live_ids) -> None:
        """Drop bookkeeping for tubes a re-segmentation retired.

        Without this a dead tube id that had reached ``max_unmeasured`` would make
        :meth:`coverage_exhausted` true forever — silently vetoing every later
        whole-step-skip promotion for the rest of the generation. Called alongside
        ``AnchorStore.retain`` by the engine.
        """
        live = set(live_ids)
        self._unmeasured = {tid: n for tid, n in self._unmeasured.items() if tid in live}
        self._force_full = {tid: n for tid, n in self._force_full.items() if tid in live}

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

    def register_repair(self, tube_id: int, steps: int = 1) -> None:
        """Pin a *repaired* (medium-risk) tube to FULL for one refresh step.

        Shorter than a rollback's window on purpose: the tube was fused toward the
        freshly computed latent rather than revoked, so it needs one recompute to
        settle, not ``q``. Never shortens an active rollback pin.
        """
        self._force_full[tube_id] = max(self._force_full.get(tube_id, 0), int(steps))

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
        self._unmeasured.clear()
