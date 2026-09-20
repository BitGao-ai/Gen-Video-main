"""Dynamic per-step compute budget schedule."""

from __future__ import annotations

import re
from typing import Optional

from cocf.common.config import BudgetConfig
from cocf.common.types import CausalSubgraph


class PromptComplexity:
    """Scores prompt complexity in [0, 1]."""

    _SPATIAL = ("left", "right", "above", "below", "behind", "front", "next",
                "on", "under", "between", "near", "上", "下", "左", "右", "旁")

    def score(self, prompt: str, subgraph: Optional[CausalSubgraph] = None) -> float:
        """Score prompt complexity from text and subgraph."""
        tokens = re.findall(r"[\w']+", prompt.lower())
        n_len = min(len(tokens) / 30.0, 1.0)
        n_ent = 0.0
        n_act = 0.0
        n_crit = 0.0
        if subgraph is not None:
            n_ent = min(len(subgraph.entity_importance) / 5.0, 1.0)
            n_act = min(
                sum(1 for t in subgraph.triplets if t.action not in ("exists", "")) / 3.0, 1.0
            )
            n_crit = 1.0 if subgraph.critical_entities else 0.0
        n_spatial = 1.0 if any(s in tokens for s in self._SPATIAL) else 0.0
        score = (
            0.20 * n_len + 0.25 * n_ent + 0.20 * n_act
            + 0.15 * n_spatial + 0.20 * n_crit
        )
        return float(min(max(score, 0.0), 1.0))


class BudgetScheduler:
    """Produces per-step compute budget from demand signals."""

    def __init__(self, config: BudgetConfig) -> None:
        """Create scheduler from budget config."""
        self.cfg = config
        self.complexity = PromptComplexity()

    def time_weight(self, step_frac: float) -> float:
        """Compute U-shaped time weight for step fraction."""
        sf = float(min(max(step_frac, 0.0), 1.0))
        early = max(0.0, (sf - 0.60) / 0.40)
        late = max(0.0, (0.40 - sf) / 0.40)
        c = self.cfg
        raw = c.q_mid_floor + c.q_early_boost * early + c.q_late_boost * late
        q_max = c.q_mid_floor + max(c.q_early_boost, c.q_late_boost)
        return float(min(raw / q_max, 1.0)) if q_max > 0 else 0.0

    def budget(
        self,
        step_frac: float,
        *,
        complexity: float = 0.0,
        mean_uncertainty: float = 0.0,
        interaction_density: float = 0.0,
    ) -> float:
        """Compute allowed compute fraction for this step."""
        c = self.cfg
        demand = (
            c.eta_scene * complexity
            + c.eta_uncertainty * mean_uncertainty
            + c.eta_interaction * interaction_density
        )
        weight = min(max(self.time_weight(step_frac) + demand, 0.0), 1.0)
        return c.b_min + (c.b_max - c.b_min) * weight

    def score_complexity(self, prompt: str, subgraph: Optional[CausalSubgraph] = None) -> float:
        """Score prompt complexity."""
        return self.complexity.score(prompt, subgraph)
