"""Hierarchical discrete compute-field mapping (§3.3.3).

L-COCF's third simplification: skip the continuous-field iterative solve and map
causal strength *directly* to one of four discrete compute tiers — the four
actions reused from the original SS-DCA action set. This is a pure forward
lookup (no gradient-field optimisation, no iteration), which is exactly what
makes the inference path cheap and parallel-friendly.

    s > θ₁              HIGH → FULL        (causally critical: full fidelity)
    θ₂ < s ≤ θ₁         MID  → LOWFREQ     (moderate: low-frequency compute)
    s ≤ θ₂             LOW  → INTERP       (background: interpolate)
                       (ANCHOR is left to the allocator/RAEC under budget+risk)

The map produces a *prior* action; the budget allocator (§2.2) may downgrade a LOW
tube to ANCHOR when the budget is tight and its risk certificate is safe. The one
hard rule applied here is the STA stability override (§4.3.1): an unstable tube
(identity confidence < threshold) is forced to FULL regardless of strength.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from cocf.common.config import StrengthConfig, TubeConfig
from cocf.common.types import (
    Action,
    DEFAULT_LEVEL_TO_ACTION,
    SemanticTube,
    StrengthLevel,
    TubeState,
)

Tensor = torch.Tensor


class ComputeFieldMapping:
    """Maps causal strength → tier → prior action, with the STA stability override."""

    def __init__(
        self,
        strength_cfg: StrengthConfig,
        tube_cfg: TubeConfig,
        level_to_action: Optional[Dict[StrengthLevel, Action]] = None,
    ) -> None:
        self.s_cfg = strength_cfg
        self.t_cfg = tube_cfg
        self.level_to_action = level_to_action or DEFAULT_LEVEL_TO_ACTION

    def tier(self, strength: float) -> StrengthLevel:
        if strength > self.s_cfg.theta1:
            return StrengthLevel.HIGH
        if strength > self.s_cfg.theta2:
            return StrengthLevel.MID
        return StrengthLevel.LOW

    def prior_action(self, strength: float, state: Optional[TubeState] = None) -> Action:
        """Cold-start / prior action for a tube given its strength and state."""
        if state is not None and state.identity_confidence < self.t_cfg.identity_unstable_threshold:
            return Action.FULL  # unstable tube → full compute (§4.3.1)
        return self.level_to_action[self.tier(strength)]

    def prior_actions(
        self, strengths: Dict[int, float], states: Optional[Dict[int, TubeState]] = None
    ) -> Dict[int, Action]:
        states = states or {}
        return {
            tid: self.prior_action(s, states.get(tid)) for tid, s in strengths.items()
        }
