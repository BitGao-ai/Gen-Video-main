"""Dynamic per-step compute-budget schedule ``B_t`` (§7.3).

The budget is the fraction of full compute the allocator may spend on a denoising
step. It is *not* constant: it tracks where compute actually buys quality, via a
U-shaped time profile plus three demand signals::

    demand = q(t)                                   U-shaped time weight (§7.3)
           + η_scene      · complexity(prompt)       harder prompts → more compute
           + η_uncertainty· mean σ                   unsure damage → more compute
           + η_interaction· interaction density      multi-subject scenes → more
    B_t    = b_min + (b_max − b_min) · clip(demand, 0, 1)

``q(t)`` is high in the early *structure* phase and the late *detail* phase and low
in the easy middle (the empirical sweet spot for where skipping is safe, §7.3),
expressed over ``step_frac = t/T ∈ [0,1]`` (1 = pure noise, 0 = clean). All weights
live in :class:`~cocf.common.config.BudgetConfig`; the mapping to ``[b_min, b_max]``
keeps the budget interpretable as "between 30% and 100% of full compute".
"""

from __future__ import annotations

import re
from typing import Optional

from cocf.common.config import BudgetConfig
from cocf.common.types import CausalSubgraph


class PromptComplexity:
    """Scores prompt complexity ∈ [0, 1] from its causal sub-graph (§7.3).

    Combines prompt length, entity count, named-action count, spatial-relation
    cues and the presence of quality-critical content (text/face/hands). Pure and
    deterministic; the sub-graph is already parsed by L-COCF so this is free.
    """

    _SPATIAL = ("left", "right", "above", "below", "behind", "front", "next",
                "on", "under", "between", "near", "上", "下", "左", "右", "旁")

    def score(self, prompt: str, subgraph: Optional[CausalSubgraph] = None) -> float:
        tokens = re.findall(r"[\w']+", prompt.lower())
        n_len = min(len(tokens) / 30.0, 1.0)                # length, saturating at 30
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
        # weighted blend, normalised to [0, 1]
        score = (
            0.20 * n_len + 0.25 * n_ent + 0.20 * n_act
            + 0.15 * n_spatial + 0.20 * n_crit
        )
        return float(min(max(score, 0.0), 1.0))


class BudgetScheduler:
    """Produces ``B_t`` for a denoising step from the time profile + demand signals."""

    def __init__(self, config: BudgetConfig) -> None:
        self.cfg = config
        self.complexity = PromptComplexity()

    # ------------------------------------------------------------------ #
    # U-shaped time weight q(t)
    # ------------------------------------------------------------------ #

    def time_weight(self, step_frac: float) -> float:
        """U-shaped ``q(t) ∈ [floor, 1]`` over ``step_frac=t/T`` (high at both ends).

        Normalised by the largest value the shape can attain, because ``early`` and
        ``late`` are mutually exclusive ramps: the raw sum could only ever reach
        ``q_mid_floor + max(boost)`` — 0.50 on the defaults — so ``B_t`` topped out at
        ``b_min + 0.7·0.5 = 0.65`` and ``b_max = 1.0`` was a dead config value the
        system could never spend, leaving it permanently in "downgrade" mode (§P1-9).
        Dividing by that maximum keeps the configured early/late *ratio* intact while
        making the ends of the trajectory actually reach the ceiling.
        """
        sf = float(min(max(step_frac, 0.0), 1.0))
        early = max(0.0, (sf - 0.60) / 0.40)   # ramps up as sf → 1 (structure phase)
        late = max(0.0, (0.40 - sf) / 0.40)    # ramps up as sf → 0 (detail phase)
        c = self.cfg
        raw = c.q_mid_floor + c.q_early_boost * early + c.q_late_boost * late
        q_max = c.q_mid_floor + max(c.q_early_boost, c.q_late_boost)
        return float(min(raw / q_max, 1.0)) if q_max > 0 else 0.0

    # ------------------------------------------------------------------ #
    # budget
    # ------------------------------------------------------------------ #

    def budget(
        self,
        step_frac: float,
        *,
        complexity: float = 0.0,
        mean_uncertainty: float = 0.0,
        interaction_density: float = 0.0,
    ) -> float:
        """Fraction of full compute allowed this step, in ``[b_min, b_max]`` (§7.3).

        ``B_t = b_min + (b_max − b_min) · clip(q(t) + demand, 0, 1)``.

        Everything inside the clip is a *weight* in [0, 1]: the U-shaped time profile
        plus the three demand signals (scene complexity, mean damage uncertainty,
        tube-interaction density). Adding the demand terms in absolute compute-fraction
        units instead — and clamping ``B_t`` afterwards, as this used to — made the
        band boundaries do the work of the formula and did not match the documented
        form, so a caller reading either one was misled about the other (§P1-9).
        """
        c = self.cfg
        demand = (
            c.eta_scene * complexity
            + c.eta_uncertainty * mean_uncertainty
            + c.eta_interaction * interaction_density
        )
        weight = min(max(self.time_weight(step_frac) + demand, 0.0), 1.0)
        return c.b_min + (c.b_max - c.b_min) * weight

    def score_complexity(self, prompt: str, subgraph: Optional[CausalSubgraph] = None) -> float:
        return self.complexity.score(prompt, subgraph)
