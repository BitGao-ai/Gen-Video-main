"""Budget-constrained per-tube action allocation."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch

from cocf.common.config import AllocatorConfig
from cocf.common.logging import get_logger
from cocf.common.types import (
    Action,
    AllocationDecision,
    DamagePrediction,
    SemanticTube,
    TubeState,
)

Tensor = torch.Tensor
_log = get_logger(__name__)
_NUM_ACTIONS = len(Action)


class ActionAllocator:
    """Greedy multiple-choice knapsack over per-tube actions."""

    def __init__(self, config: AllocatorConfig, lowfreq_stride: Optional[int] = None,
                 identity_unstable_threshold: float = 0.5) -> None:
        """Create allocator from config and stride."""
        self.cfg = config
        self.identity_unstable_threshold = float(identity_unstable_threshold)
        self.action_cost = list(config.action_cost)
        if lowfreq_stride:
            self.action_cost[int(Action.LOWFREQ)] = 1.0 / float(max(1, lowfreq_stride) ** 2)
        self._warn_if_ladder_collapses()

    def _warn_if_ladder_collapses(self) -> None:
        """Warn when adjacent actions share the same cost."""
        ladder = sorted(Action, key=self._rank_key_for)
        for lo, hi in zip(ladder[:-1], ladder[1:]):
            if abs(self.action_cost[int(hi)] - self.action_cost[int(lo)]) <= 1e-12:
                _log.warning(
                    "allocator: %s and %s are both priced at %.4f, so the greedy cannot "
                    "move between them — a tube seeded at %s can never be upgraded past "
                    "it. Check AllocatorConfig.action_cost (engine.lowfreq_stride == 1 "
                    "is the one benign case).",
                    lo.name, hi.name, self.action_cost[int(lo)], lo.name,
                )

    def _rank_key_for(self, action: Action):
        """Return ladder sort key for action."""
        return (self.action_cost[int(action)], -int(action))

    @staticmethod
    def distinct_token_count(tubes: List[SemanticTube]) -> int:
        """Count distinct tokens covered by tubes."""
        if not tubes:
            return 0
        seen = torch.cat([t.all_token_indices() for t in tubes]) if tubes else torch.empty(0)
        return int(torch.unique(seen).numel()) if seen.numel() else 0

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #

    def allocate(
        self,
        tubes: List[SemanticTube],
        predictions: Dict[int, DamagePrediction],
        *,
        budget: float = 1.0,
        states: Optional[Dict[int, TubeState]] = None,
        prior_actions: Optional[Dict[int, Action]] = None,
        forced_full: Optional[Set[int]] = None,
        action_risk: Optional[Dict[int, Tensor]] = None,
        step: int = 0,
    ) -> AllocationDecision:
        """Allocate actions to tubes under budget."""
        states = states or {}
        prior_actions = prior_actions or {}
        forced_full = forced_full or set()

        total_size = max(1, self.distinct_token_count(tubes))
        budget_tokens = float(budget) * total_size

        admissible: Dict[int, List[Action]] = {}
        cost: Dict[int, Dict[Action, float]] = {}
        dmg: Dict[int, Dict[Action, float]] = {}
        for tube in tubes:
            tid = tube.tube_id
            adm = self._admissible(tube, states.get(tid), tid in forced_full,
                                   action_risk.get(tid) if action_risk else None)
            admissible[tid] = adm
            cost[tid] = {a: self.action_cost[int(a)] * tube.size for a in adm}
            mu = predictions[tid].mu.detach().tolist() if tid in predictions else None
            dmg[tid] = {
                a: (float(mu[int(a)]) if mu is not None else _prior_damage(a, prior_actions.get(tid)))
                for a in adm
            }

        chosen = self._seeded_knapsack(
            admissible, cost, dmg, budget_tokens, prior_actions
        )

        used = sum(cost[t.tube_id][chosen[t.tube_id]] for t in tubes)
        return AllocationDecision(
            step=step,
            actions=chosen,
            predicted_cost=used / total_size,
            budget=float(budget),
            chosen_damage={t.tube_id: dmg[t.tube_id][chosen[t.tube_id]] for t in tubes},
        )

    # ------------------------------------------------------------------ #
    # seeded multiple-choice knapsack
    # ------------------------------------------------------------------ #

    def _seeded_knapsack(
        self,
        admissible: Dict[int, List[Action]],
        cost: Dict[int, Dict[Action, float]],
        dmg: Dict[int, Dict[Action, float]],
        budget_tokens: float,
        prior_actions: Dict[int, Action],
    ) -> Dict[int, Action]:
        """Solve per-tube assignment seeded at prior."""
        ranked = {
            tid: sorted(acts, key=self._rank_key_for) for tid, acts in admissible.items()
        }
        pos = {}
        chosen: Dict[int, Action] = {}
        for tid, order in ranked.items():
            seed = prior_actions.get(tid)
            if seed not in order:
                seed = order[-1]
            pos[tid] = order.index(seed)
            chosen[tid] = seed
        used = sum(cost[tid][chosen[tid]] for tid in ranked)

        if used > budget_tokens + 1e-9:
            used = self._downgrade_to_fit(ranked, pos, chosen, cost, dmg, budget_tokens, used)
        else:
            used = self._upgrade_into_budget(ranked, pos, chosen, cost, dmg, budget_tokens, used)
        return chosen

    @staticmethod
    def _downgrade_to_fit(ranked, pos, chosen, cost, dmg, budget_tokens, used) -> float:
        """Downgrade cheapest damage-per-saving first."""
        while used > budget_tokens + 1e-9:
            best = None
            for tid, order in ranked.items():
                i = pos[tid]
                if i == 0:
                    continue
                cur, prv = order[i], order[i - 1]
                saved = cost[tid][cur] - cost[tid][prv]
                if saved <= 0:
                    continue
                increase = max(0.0, dmg[tid][prv] - dmg[tid][cur])
                ratio = increase / saved
                if best is None or ratio < best[0]:
                    best = (ratio, tid, prv, saved)
            if best is None:
                break
            _, tid, prv, saved = best
            chosen[tid] = prv
            pos[tid] -= 1
            used -= saved
        return used

    @staticmethod
    def _upgrade_into_budget(ranked, pos, chosen, cost, dmg, budget_tokens, used) -> float:
        """Upgrade best damage-reduction-per-cost first."""
        while True:
            best = None
            for tid, order in ranked.items():
                i = pos[tid]
                if i + 1 >= len(order):
                    continue
                cur, nxt = order[i], order[i + 1]
                extra = cost[tid][nxt] - cost[tid][cur]
                if extra <= 0 or used + extra > budget_tokens + 1e-9:
                    continue
                benefit = max(0.0, dmg[tid][cur] - dmg[tid][nxt])
                ratio = benefit / extra
                if best is None or ratio > best[0]:
                    best = (ratio, tid, nxt, extra)
            if best is None or best[0] <= 0.0:
                break
            _, tid, nxt, extra = best
            chosen[tid] = nxt
            pos[tid] += 1
            used += extra
        return used

    def _admissible(
        self,
        tube: SemanticTube,
        state: Optional[TubeState],
        forced_full: bool,
        risk: Optional[Tensor],
    ) -> List[Action]:
        """Return admissible actions for a tube."""
        if forced_full or (state is not None
                           and state.is_unstable(self.identity_unstable_threshold)):
            return [Action.FULL]
        risks = risk.detach().tolist() if risk is not None else None
        acts = []
        for a in Action:
            if a == Action.FULL:
                acts.append(a)
                continue
            if risks is not None and float(risks[int(a)]) > self.cfg.risk_threshold:
                continue
            acts.append(a)
        return acts

def _prior_damage(action: Action, prior: Optional[Action]) -> float:
    """Return fallback damage when no prediction exists."""
    if prior is not None and action == prior:
        return 0.0
    return 0.1 * int(action)
