"""Lightweight causal-strength field ``s = α·s_E + β·s_A + γ·s_T`` (§3.3.2).

L-COCF's second simplification: a *first-order linear* approximation of the causal
effect, justified by the causal-effect hierarchy axiom (§3.2.2 — quality is driven
by entities, actions and temporal transitions; everything else is ~constant). It
has only **three** learnable scalars, so it trains on a handful of samples and
adds negligible cost.

The three causal signals, all derived from cheap, already-available quantities:

    s_E  entity strength      VLM entity importance for the entity the tube depicts
    s_A  action strength      tube motion magnitude × text-action alignment
    s_T  temporal strength    inter-frame semantic change (occlusion + id drift)

This module separates *feature construction* (``CausalStrengthFeatureBuilder`` —
pure, reused verbatim by the data pipeline so training/inference features match)
from the *learnable combination* (``CausalStrengthField`` — the three weights).
That separation is what lets §3.3.5's plugin training touch only three params.
"""

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
    """The three causal signals for one tube (inputs to the strength field)."""

    s_E: float  # entity importance ∈ [0, 1]
    s_A: float  # action strength  ∈ [0, 1]
    s_T: float  # temporal-transition strength ∈ [0, 1]

    def as_tensor(self, device=None, dtype=torch.float32) -> Tensor:
        return torch.tensor([self.s_E, self.s_A, self.s_T], device=device, dtype=dtype)


class CausalStrengthFeatureBuilder:
    """Derives :class:`StrengthFeatures` from a tube, its state and the sub-graph.

    Stateless and dependency-free → identical features at train and inference time
    (a common source of train/serve skew, avoided here by construction).
    """

    def build(
        self,
        tube: SemanticTube,
        state: TubeState,
        subgraph: CausalSubgraph,
        entity_importance: Optional[float] = None,
        action_alignment: Optional[float] = None,
    ) -> StrengthFeatures:
        # s_E: importance of the entity this tube depicts. If the caller resolved a
        # tube→entity match, use it; else fall back to the sub-graph's mean (or a
        # boost when the tube is flagged critical, text/face/hands).
        if entity_importance is None:
            base = (
                sum(subgraph.entity_importance.values()) / len(subgraph.entity_importance)
                if subgraph.entity_importance else 0.5
            )
            entity_importance = 1.0 if subgraph.critical_entities else base
        s_E = float(min(max(entity_importance, 0.0), 1.0))

        # s_A: motion magnitude (already normalised in the state) gated by how much
        # the tube participates in a *named action* (text-action alignment).
        align = action_alignment if action_alignment is not None else (
            1.0 if any(t.action != "exists" for t in subgraph.triplets) else 0.5
        )
        s_A = float(min(max(state.motion_phase * align, 0.0), 1.0))

        # s_T: inter-frame semantic change — occlusion plus identity drift.
        s_T = float(min(max(0.5 * state.occlusion + 0.5 * (1.0 - state.identity_confidence), 0.0), 1.0))
        return StrengthFeatures(s_E=s_E, s_A=s_A, s_T=s_T)


class CausalStrengthField(nn.Module):
    """The 3-parameter learnable combiner producing the causal strength ``s``.

    ``s = softplus(α)·s_E + softplus(β)·s_A + softplus(γ)·s_T`` (softplus keeps the
    weights non-negative so larger signals never *reduce* allocated compute), then
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
            s = s / w.sum().clamp_min(1e-6)  # convex combination → already in [0,1]
        return s

    def level(self, strength: Tensor) -> Tensor:
        """Discretise strength to :class:`StrengthLevel` codes (§3.3.3 thresholds)."""
        level = torch.full_like(strength, float(StrengthLevel.LOW), dtype=torch.float32)
        level = torch.where(strength > self.cfg.theta2, torch.full_like(level, float(StrengthLevel.MID)), level)
        level = torch.where(strength > self.cfg.theta1, torch.full_like(level, float(StrengthLevel.HIGH)), level)
        return level.long()
