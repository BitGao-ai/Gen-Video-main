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

The differentiable counterpart used for training (softmax over −μ, feeding the tube
smoothing and budget losses) lives with the losses that consume it, in
:mod:`cocf.training.stage_b_losses`.
"""

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
    """Greedy (or LP) multiple-choice knapsack over per-tube actions (§2.2)."""

    def __init__(self, config: AllocatorConfig, lowfreq_stride: Optional[int] = None) -> None:
        self.cfg = config
        self.action_cost = list(config.action_cost)  # indexed by Action value
        # LOWFREQ's true cost is set by the executor's spatial stride (a stride-s
        # lattice computes 1/s² of the tube's tokens), so derive it rather than trust a
        # constant that silently disagrees with the transition the engine performs.
        if lowfreq_stride:
            self.action_cost[int(Action.LOWFREQ)] = 1.0 / float(max(1, lowfreq_stride) ** 2)
        self._warn_if_ladder_collapses()

    def _warn_if_ladder_collapses(self) -> None:
        """Say so when two adjacent rungs of the action ladder share a cost (§P4-1).

        The greedy walks the ladder one rung at a time and only moves where the cost
        delta is non-zero (a zero-delta move buys no budget, so there is nothing to
        trade). Two adjacent actions priced identically therefore make the *cheaper*
        one a one-way door: reachable by downgrade, never escapable by upgrade.

        This is a configuration smell, not an error — ``lowfreq_stride == 1`` legitimately
        makes LOWFREQ cost the same as FULL because it then computes every token, and
        the two really are the same operation. So we log rather than raise, naming the
        pair so a mis-edited ``action_cost`` is diagnosable from the first line of the run.
        """
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
        """Ladder sort key: cheapest first, ties broken by *most destructive* first.

        ``Action`` is already ordered FULL(0) … ANCHOR(3) by descending expense, so
        ``-int(action)`` puts the more destructive member of a cost tie lower on the
        ladder. Without it ``sorted`` falls back to input order, which put INTERP below
        ANCHOR and made budget pressure skip the *less* destructive action (§P4-1).
        """
        return (self.action_cost[int(action)], -int(action))

    @staticmethod
    def distinct_token_count(tubes: List[SemanticTube]) -> int:
        """``|⋃ g_k|`` — tokens covered by at least one tube.

        Tubes overlap by design (the state vector models it as ``interaction``), so
        ``Σ|g_k|`` counts shared tokens once per tube and inflates the budget
        denominator: the allocator then believes it may spend compute it does not
        have. Counting the union keeps ``B_t · |⋃ g_k|`` an actual token budget.
        """
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
        states = states or {}
        prior_actions = prior_actions or {}
        forced_full = forced_full or set()

        # Budget denominator = distinct covered tokens, so overlapping tubes do not
        # inflate the allowance (§P1-6). Per-tube costs still use each tube's own size:
        # a shared token genuinely costs both tubes' actions, and charging it twice is
        # the conservative direction.
        total_size = max(1, self.distinct_token_count(tubes))
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
            # the differentiable training path uses ``stage_b_losses.action_probs``.
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

        The ladder each tube walks is ``[ANCHOR, INTERP, LOWFREQ, FULL]`` — sorted by
        cost, ties broken most-destructive-first (:meth:`_rank_key_for`). Unlike a
        cheapest-first knapsack, every tube starts at its §3.3.3 prior action (the
        cold-start fallback of §1.3). From that seed the solver moves in exactly one
        direction:

        * **over budget** → *downgrade* (toward cheaper actions) the tube whose
          extra damage-per-token-saved is smallest, until the plan fits. A LOW tube
          may thus fall to ANCHOR only under genuine budget pressure (§3.3.3), and it
          passes through INTERP on the way rather than jumping straight to the freeze.
        * **under budget** → *upgrade* (toward FULL) the tube whose damage-reduction
          -per-extra-token is largest, while a beneficial upgrade still fits.

        Both moves require a non-zero cost delta — a zero-delta step buys no budget, so
        there is nothing to trade. That is why the cost vector must be strictly ordered
        along the ladder; :meth:`_warn_if_ladder_collapses` flags a config that isn't.

        With an untrained (flat-μ) predictor no upgrade has positive benefit, so the
        plan stays at the priors — i.e. the system degrades gracefully to the §1.3
        threshold policy until the predictor has learned. Forced-FULL / unstable
        tubes have a singleton admissible set and are never moved.
        """
        ranked = {  # cheapest → most expensive; ties broken most-destructive-first
            tid: sorted(acts, key=self._rank_key_for) for tid, acts in admissible.items()
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

def _prior_damage(action: Action, prior: Optional[Action]) -> float:
    """Fallback damage when no prediction exists: 0 if it matches the cold-start
    prior action, else a mild penalty ordered by how aggressive the skip is.

    Lets the allocator degrade gracefully to the §1.3 threshold prior before the
    predictor has converged, without special-casing the call site.
    """
    if prior is not None and action == prior:
        return 0.0
    return 0.1 * int(action)
