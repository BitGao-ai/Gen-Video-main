"""Local single-hop counterfactual verification & repair (§3.3.4).

L-COCF's fourth simplification: replace full-domain *multi-hop* counterfactual
reasoning (infeasible, §3.1) with a *single-hop* intervention fired **only at
temporal causal mutation points** (``s_T > θ_sT``) and capped per step. This is
justified by the single-hop sufficiency theorem (§3.2.3): because the video
causal coupling coefficient is < 1, high-order causal effects decay exponentially,
so one hop suffices to certify causal consistency. Net effect: the cost of the
causal check drops ~two orders of magnitude vs native COCF (§3.4).

The procedure for a triggered, *skipped* tube:

    1. intervene  — actually compute the FULL transition on the tube's tokens
                    (do(¬skip)); the engine supplies this via a callback so this
                    module stays backbone-agnostic.
    2. residual   — Δ = ‖z_full − z_skip‖ over the tube.
    3. repair     — if Δ > η the skip omitted a causal effect ("causal omission");
                    a lightweight residual-repair sub-net corrects z_skip toward
                    z_full locally (the trainable plugin fine-tuned in Stage C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from cocf.common.config import CounterfactualConfig
from cocf.common.types import SemanticTube
from cocf.lcocf.strength import StrengthFeatures

Tensor = torch.Tensor


@dataclass
class VerificationResult:
    """Outcome of a single-hop counterfactual check on one tube."""

    tube_id: int
    residual: float
    repaired: bool
    z_corrected: Optional[Tensor] = None  # [n_tok, d] corrected tube latent (if repaired)


class ResidualRepairNet(nn.Module):
    """Lightweight per-token residual corrector (one of L-COCF's trainable plugins).

    Predicts a local correction ``Δz`` for a skipped tube from its (cheap) skipped
    latent and a pooled tube context, added back to repair causal omissions. Small
    by design (§3.3.5: ~1–10 M params) and independent of the frozen backbone.
    """

    def __init__(self, token_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # start as identity (zero correction)

    def forward(self, z_skip_tube: Tensor) -> Tensor:
        """``z_skip_tube`` is ``[n_tok, d]``; returns the corrected latent ``[n_tok, d]``."""
        ctx = z_skip_tube.mean(0, keepdim=True).expand_as(z_skip_tube)
        delta = self.net(torch.cat([z_skip_tube, ctx], dim=-1))
        return z_skip_tube + delta


class CounterfactualVerifier:
    """Triggers single-hop checks and applies repairs at mutation points."""

    def __init__(self, config: CounterfactualConfig, repair_net: Optional[ResidualRepairNet] = None) -> None:
        self.cfg = config
        self.repair_net = repair_net

    def triggered_tubes(
        self,
        strength_feats: Dict[int, StrengthFeatures],
        skipped: Dict[int, bool],
    ) -> List[int]:
        """Tube ids to check: temporal-mutation (s_T>θ) ∧ skipped, capped per step.

        Ranked by ``s_T`` so the scarce check budget goes to the most volatile
        tubes first.
        """
        candidates = [
            (tid, f.s_T)
            for tid, f in strength_feats.items()
            if f.s_T > self.cfg.theta_sT and skipped.get(tid, False)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in candidates[: self.cfg.max_checks_per_step]]

    def verify_and_repair(
        self,
        tube: SemanticTube,
        z_skip_tube: Tensor,
        z_full_tube: Tensor,
    ) -> VerificationResult:
        """Compare skip vs full on a tube; repair locally if the residual exceeds η."""
        residual = float((z_full_tube - z_skip_tube).pow(2).mean().sqrt().item())
        if residual <= self.cfg.eta:
            return VerificationResult(tube.tube_id, residual, repaired=False)
        # causal omission detected → local residual correction
        if self.repair_net is not None:
            corrected = self.repair_net(z_skip_tube)
        else:  # no trained net yet → fall back to the verified full latent
            corrected = z_full_tube
        return VerificationResult(tube.tube_id, residual, repaired=True, z_corrected=corrected)
