"""Lightweight causal-strength field s = a*s_E + b*s_A + g*s_T."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.common.config import StrengthConfig
from cocf.common.types import CausalSubgraph, SemanticTube, StrengthLevel, TubeState

Tensor = torch.Tensor


@dataclass
class StrengthFeatures:
    """Three causal signals for one tube."""

    s_E: float
    s_A: float
    s_T: float

    def as_tensor(self, device=None, dtype=torch.float32) -> Tensor:
        """Features as tensor."""
        return torch.tensor([self.s_E, self.s_A, self.s_T], device=device, dtype=dtype)


class CausalStrengthFeatureBuilder:
    """Derives strength features from tube and sub-graph."""

    def build(
        self,
        tube: SemanticTube,
        state: TubeState,
        subgraph: CausalSubgraph,
        entity_importance: Optional[float] = None,
        action_alignment: Optional[float] = None,
    ) -> StrengthFeatures:
        """Build strength features for one tube."""
        # s_E: importance of the entity this tube depicts. Use the resolved
        # tube-entity match if given, else the sub-graph mean (boosted when the tube is
        # flagged critical).
        if entity_importance is None:
            base = (
                sum(subgraph.entity_importance.values()) / len(subgraph.entity_importance)
                if subgraph.entity_importance else 0.5
            )
            # No tube-entity match: a bounded additive boost keeps critical prompts
            # stronger without pinning every tube to the top tier.
            entity_importance = min(1.0, base + 0.25) if subgraph.critical_entities else base
        s_E = float(min(max(entity_importance, 0.0), 1.0))

        # s_A: motion magnitude gated by text-action alignment.
        align = action_alignment if action_alignment is not None else (
            1.0 if any(t.action != "exists" for t in subgraph.triplets) else 0.5
        )
        s_A = float(min(max(state.motion_phase * align, 0.0), 1.0))

        # s_T: inter-frame semantic change (occlusion plus identity drift).
        s_T = float(min(max(0.5 * state.occlusion + 0.5 * (1.0 - state.identity_confidence), 0.0), 1.0))
        return StrengthFeatures(s_E=s_E, s_A=s_A, s_T=s_T)


class CausalStrengthField(nn.Module):
    """The 3-parameter learnable combiner producing the causal strength ``s``.

    ``s = softplus(a)*s_E + softplus(b)*s_A + softplus(c)*s_T`` (softplus keeps the
    weights non-negative so larger signals never reduce allocated compute), then
    optionally squashed to ``[0, 1]`` before tier thresholding.
    """

    def __init__(self, config: StrengthConfig) -> None:
        super().__init__()
        self.cfg = config
        self.alpha = nn.Parameter(torch.tensor(float(config.alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(config.beta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(config.gamma_init)))

    def weights(self) -> Tensor:
        return F.softplus(torch.stack([self.alpha, self.beta, self.gamma]))

    def forward(self, features: Tensor) -> Tensor:
        """``features`` is ``[..., 3]`` (s_E, s_A, s_T); returns strength ``[...]``."""
        w = self.weights().to(features.dtype)
        s = (features * w).sum(-1)
        if self.cfg.normalize_strength:
            s = s / w.sum().clamp_min(1e-6)  # convex combination, already in [0, 1]
        return s

    def level(self, strength: Tensor) -> Tensor:
        """Discretise strength to :class:`StrengthLevel` codes."""
        level = torch.full_like(strength, float(StrengthLevel.LOW), dtype=torch.float32)
        level = torch.where(strength > self.cfg.theta2, torch.full_like(level, float(StrengthLevel.MID)), level)
        level = torch.where(strength > self.cfg.theta1, torch.full_like(level, float(StrengthLevel.HIGH)), level)
        return level.long()
