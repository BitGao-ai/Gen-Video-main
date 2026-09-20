"""L-COCF facade wiring sub-modules into one interface."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from cocf.common.config import LCOCFConfig, TubeConfig
from cocf.common.types import (
    Action,
    CausalSubgraph,
    DamagePrediction,
    SemanticTube,
    TubeState,
)
from cocf.lcocf.counterfactual import CounterfactualVerifier, ResidualRepairNet
from cocf.lcocf.mapping import ComputeFieldMapping
from cocf.lcocf.predictor import (
    DamagePredictor,
    build_predictor_input,
)
from cocf.lcocf.strength import (
    CausalStrengthField,
    CausalStrengthFeatureBuilder,
    StrengthFeatures,
)
from cocf.lcocf.triplets import CausalParser, build_parser

Tensor = torch.Tensor


class LCOCFModule(nn.Module):
    """Lightweight counterfactual causal compute field."""

    def __init__(
        self,
        lcocf_cfg: LCOCFConfig,
        tube_cfg: TubeConfig,
        *,
        token_dim: int,
        parser: Optional[CausalParser] = None,
    ) -> None:
        """Store configs and build sub-modules."""
        super().__init__()
        self.cfg = lcocf_cfg
        self.tube_cfg = tube_cfg

        self.feature_builder = CausalStrengthFeatureBuilder()
        self.mapping = ComputeFieldMapping(lcocf_cfg.strength, tube_cfg)
        self.parser = parser or build_parser(lcocf_cfg)

        self.strength_field = CausalStrengthField(lcocf_cfg.strength)
        self.predictor = DamagePredictor(lcocf_cfg.predictor)
        self.repair_net = ResidualRepairNet(
            token_dim, hidden=lcocf_cfg.counterfactual.repair_net_dim
        )
        self.verifier = CounterfactualVerifier(
            lcocf_cfg.counterfactual, repair_net=self.repair_net
        )

    def parse(self, prompt: str) -> CausalSubgraph:
        """Parse prompt into causal sub-graph."""
        return self.parser.parse(prompt)

    # ------------------------------------------------------------------ #
    # causal strength field
    # ------------------------------------------------------------------ #

    def strength_features(
        self,
        tubes: List[SemanticTube],
        states: Dict[int, TubeState],
        subgraph: CausalSubgraph,
        tube_entity: Optional[Dict[int, float]] = None,
        tube_action_align: Optional[Dict[int, float]] = None,
    ) -> Dict[int, StrengthFeatures]:
        """Per-tube strength features."""
        tube_entity = tube_entity or {}
        tube_action_align = tube_action_align or {}
        feats: Dict[int, StrengthFeatures] = {}
        for tube in tubes:
            feats[tube.tube_id] = self.feature_builder.build(
                tube,
                states[tube.tube_id],
                subgraph,
                entity_importance=tube_entity.get(tube.tube_id),
                action_alignment=tube_action_align.get(tube.tube_id),
            )
        return feats

    def strengths(self, feats: Dict[int, StrengthFeatures]) -> Dict[int, float]:
        """Scalar strength per tube."""
        if not feats:
            return {}
        ids = list(feats)
        dev = self.strength_field.alpha.device
        stacked = torch.stack([feats[i].as_tensor(device=dev) for i in ids])
        s = self.strength_field(stacked).detach()
        return {i: float(s[j]) for j, i in enumerate(ids)}

    def prior_actions(
        self, strengths: Dict[int, float], states: Optional[Dict[int, TubeState]] = None
    ) -> Dict[int, Action]:
        """Prior actions for all tubes."""
        return self.mapping.prior_actions(strengths, states)

    def predict(
        self,
        tubes: List[SemanticTube],
        states: Dict[int, TubeState],
        feats: Dict[int, StrengthFeatures],
        strengths: Dict[int, float],
        budget: float,
        step_frac: float,
        *,
        device=None,
        dtype=torch.float32,
    ) -> Dict[int, DamagePrediction]:
        """Predict per-action damage for every tube."""
        if not tubes:
            return {}
        if device is None:
            device = self.predictor.mu_head.weight.device
        ids = [t.tube_id for t in tubes]
        rows = [
            build_predictor_input(
                states[i], feats[i], strengths[i], budget, step_frac,
                self.cfg.predictor.context_dim, device=device, dtype=dtype,
            )
            for i in ids
        ]
        batch = torch.stack(rows)
        out = self.predictor(batch)
        return {
            i: DamagePrediction(mu=out.mu[j], sigma=out.sigma[j])
            for j, i in enumerate(ids)
        }

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Trainable parameter groups by sub-module."""
        return {
            "strength": [self.strength_field.alpha, self.strength_field.beta, self.strength_field.gamma],
            "predictor": list(self.predictor.parameters()),
            "repair": list(self.repair_net.parameters()),
        }
