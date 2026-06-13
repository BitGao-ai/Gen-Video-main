"""Budget-constrained per-tube action allocation (§2.2).

Solves the core optimisation of the framework — minimise the predicted *final-video*
damage of skipping, subject to the step's compute budget and the RAEC risk
constraint::

    min_{a_k}  Σ_k  μ_k(a_k)                    predicted damage (L-COCF)
    s.t.       Σ_k  C(a_k)·|g_k|  ≤  B_t·Σ_k|g_k|     budget (§2.2)
               E_cert_k(a_k)      ≤  τ_r               risk   (§5.3.2)
               a_k = FULL                              if tube unstable / force-FULL

This is a *multiple-choice knapsack* (each tube picks exactly one action with a
(cost, damage) pair). It is solved by a deterministic greedy: start every tube at
its cheapest admissible action, then repeatedly apply the single upgrade with the
best damage-reduction-per-extra-cost that still fits the budget. That is near-optimal
for MCKP and needs no solver (so it runs anywhere, user requirement #3); an exact LP/
MILP path is used instead when SciPy is present and ``greedy_fallback`` is off.

The allocator also exposes differentiable per-tube action *probabilities* (softmax
over −μ) for the tube smoothing loss (§4.3.2) and Stage-C training — the hard
decision is for inference, the soft distribution is for learning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch

from cocf.common.config import AllocatorConfig
from cocf.common.types import (
    Action,
    AllocationDecision,
    DamagePrediction,
    SemanticTube,
    TubeState,
)

Tensor = torch.Tensor
_NUM_ACTIONS = len(Action)


class ActionAllocator:
    """Greedy (or LP) multiple-choice knapsack over per-tube actions (§2.2)."""

    def __init__(self, config: AllocatorConfig) -> None:
        self.cfg = config
        self.action_cost = list(config.action_cost)  # indexed by Action value

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
        states = states or {}
        prior_actions = prior_actions or {}
        forced_full = forced_full or set()

        total_size = max(1, sum(t.size for t in tubes))
        budget_tokens = float(budget) * total_size

        # admissible actions + (cost, damage) tables per tube
        admissible: Dict[int, List[Action]] = {}
        cost: Dict[int, Dict[Action, float]] = {}
        dmg: Dict[int, Dict[Action, float]] = {}
        for tube in tubes:
            tid = tube.tube_id
            adm = self._admissible(tube, states.get(tid), tid in forced_full,
                                   action_risk.get(tid) if action_risk else None)
            admissible[tid] = adm
            cost[tid] = {a: self.action_cost[int(a)] * tube.size for a in adm}
            # detach: μ here drives the non-differentiable control flow (knapsack);
            # the differentiable training path uses ``action_probabilities`` instead.
            mu = predictions[tid].mu.detach() if tid in predictions else None
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
        """Solve the per-tube action assignment, *seeded at the strength prior*.

        Unlike a cheapest-first knapsack, every tube starts at its §3.3.3 prior
        action (the cold-start fallback of §1.3). From that seed the solver moves in
        exactly one direction:

        * **over budget** → *downgrade* (toward cheaper actions) the tube whose
          extra damage-per-token-saved is smallest, until the plan fits. A LOW tube
          may thus fall to ANCHOR only under genuine budget pressure (§3.3.3).
        * **under budget** → *upgrade* (toward FULL) the tube whose damage-reduction
          -per-extra-token is largest, while a beneficial upgrade still fits.

        With an untrained (flat-μ) predictor no upgrade has positive benefit, so the
        plan stays at the priors — i.e. the system degrades gracefully to the §1.3
        threshold policy until the predictor has learned. Forced-FULL / unstable
        tubes have a singleton admissible set and are never moved.
        """
        ranked = {  # cheapest → most expensive
            tid: sorted(acts, key=lambda a: cost[tid][a]) for tid, acts in admissible.items()
        }
        pos = {}
        chosen: Dict[int, Action] = {}
        for tid, order in ranked.items():
            seed = prior_actions.get(tid)
            if seed not in order:  # prior not admissible (e.g. forced FULL) → safest
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
        """Shed cost by the smallest-damage-increase-per-token-saved downgrade first."""
        while used > budget_tokens + 1e-9:
            best = None  # (ratio, tid, prev_action, saved)
            for tid, order in ranked.items():
                i = pos[tid]
                if i == 0:
                    continue  # already cheapest admissible
                cur, prv = order[i], order[i - 1]
                saved = cost[tid][cur] - cost[tid][prv]
                if saved <= 0:
                    continue
                increase = max(0.0, dmg[tid][prv] - dmg[tid][cur])
                ratio = increase / saved  # smaller = cheaper to give up
                if best is None or ratio < best[0]:
                    best = (ratio, tid, prv, saved)
            if best is None:
                break  # nothing left to downgrade (all forced/at floor)
            _, tid, prv, saved = best
            chosen[tid] = prv
            pos[tid] -= 1
            used -= saved
        return used

    @staticmethod
    def _upgrade_into_budget(ranked, pos, chosen, cost, dmg, budget_tokens, used) -> float:
        """Spend spare budget on the largest-damage-reduction-per-extra-token upgrade."""
        while True:
            best = None  # (ratio, tid, next_action, extra_cost)
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

    # ------------------------------------------------------------------ #
    # admissibility
    # ------------------------------------------------------------------ #

    def _admissible(
        self,
        tube: SemanticTube,
        state: Optional[TubeState],
        forced_full: bool,
        risk: Optional[Tensor],
    ) -> List[Action]:
        """Actions a tube may take. FULL is always admissible (the safe fallback)."""
        if forced_full or (state is not None and state.is_unstable):
            return [Action.FULL]
        acts = []
        for a in Action:
            if a == Action.FULL:
                acts.append(a)
                continue
            if risk is not None and float(risk[int(a)]) > self.cfg.risk_threshold:
                continue  # this skip is too risky (§5.3.2) — forbid it
            acts.append(a)
        return acts

    # ------------------------------------------------------------------ #
    # differentiable action probabilities (training / smoothing loss)
    # ------------------------------------------------------------------ #

    @staticmethod
    def action_probabilities(
        predictions: Dict[int, DamagePrediction], sharpness: float = 4.0
    ) -> Dict[int, Tensor]:
        """``{tube_id: softmax(−sharpness·μ) [num_actions]}`` — low damage ⇒ high prob.

        Differentiable w.r.t. the predictor's μ, so the smoothing loss (§4.3.2) and
        Stage-C objective shape the allocator's preferences during training.
        """
        return {
            tid: torch.softmax(-sharpness * p.mu, dim=-1) for tid, p in predictions.items()
        }


def _prior_damage(action: Action, prior: Optional[Action]) -> float:
    """Fallback damage when no prediction exists: 0 if it matches the cold-start
    prior action, else a mild penalty ordered by how aggressive the skip is.

    Lets the allocator degrade gracefully to the §1.3 threshold prior before the
    predictor has converged, without special-casing the call site.
    """
    if prior is not None and action == prior:
        return 0.0
    return 0.1 * int(action)
