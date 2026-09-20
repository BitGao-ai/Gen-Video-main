"""Hierarchical discrete compute-field mapping."""

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
    """Maps strength to tier and prior action."""

    def __init__(
        self,
        strength_cfg: StrengthConfig,
        tube_cfg: TubeConfig,
        level_to_action: Optional[Dict[StrengthLevel, Action]] = None,
    ) -> None:
        """Store configs and level map."""
        self.s_cfg = strength_cfg
        self.t_cfg = tube_cfg
        self.level_to_action = level_to_action or DEFAULT_LEVEL_TO_ACTION

    def tier(self, strength: float) -> StrengthLevel:
        """Strength value to discrete tier."""
        if strength > self.s_cfg.theta1:
            return StrengthLevel.HIGH
        if strength > self.s_cfg.theta2:
            return StrengthLevel.MID
        return StrengthLevel.LOW

    def prior_action(self, strength: float, state: Optional[TubeState] = None) -> Action:
        """Prior action for strength and state."""
        if state is not None and state.identity_confidence < self.t_cfg.identity_unstable_threshold:
            return Action.FULL
        return self.level_to_action[self.tier(strength)]

    def prior_actions(
        self, strengths: Dict[int, float], states: Optional[Dict[int, TubeState]] = None
    ) -> Dict[int, Action]:
        """Prior actions for all tubes."""
        states = states or {}
        return {
            tid: self.prior_action(s, states.get(tid)) for tid, s in strengths.items()
        }
