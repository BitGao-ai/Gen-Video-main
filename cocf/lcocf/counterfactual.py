"""Local single-hop counterfactual verification and repair."""

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
    """Single-hop check outcome for one tube."""

    tube_id: int
    residual: float
    repaired: bool
    z_corrected: Optional[Tensor] = None


class ResidualRepairNet(nn.Module):
    """Per-token residual corrector for skipped tubes."""

    def __init__(self, token_dim: int, hidden: int = 128) -> None:
        """Build correction MLP."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_skip_tube: Tensor) -> Tensor:
        """Correct skipped tube latent."""
        ctx = z_skip_tube.mean(0, keepdim=True).expand_as(z_skip_tube)
        delta = self.net(torch.cat([z_skip_tube, ctx], dim=-1))
        return z_skip_tube + delta


class CounterfactualVerifier:
    """Triggers single-hop checks and repairs."""

    def __init__(self, config: CounterfactualConfig, repair_net: Optional[ResidualRepairNet] = None) -> None:
        """Store config and repair net."""
        self.cfg = config
        self.repair_net = repair_net

    def triggered_tubes(
        self,
        strength_feats: Dict[int, StrengthFeatures],
        skipped: Dict[int, bool],
    ) -> List[int]:
        """Tube ids needing verification."""
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
        """Compare skip vs full and repair if needed."""
        residual = float((z_full_tube - z_skip_tube).pow(2).mean().sqrt().item())
        if residual <= self.cfg.eta:
            return VerificationResult(tube.tube_id, residual, repaired=False)
        if self.repair_net is not None:
            corrected = self.repair_net(z_skip_tube)
        else:
            corrected = z_full_tube
        return VerificationResult(tube.tube_id, residual, repaired=True, z_corrected=corrected)
