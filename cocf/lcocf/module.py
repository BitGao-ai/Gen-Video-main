"""L-COCF facade — wires the five sub-modules into one consumable interface (§3.3).

The engine and the trainer should not have to know the internal ordering of the
L-COCF pipeline (parse → features → strength → tier/prior → predict → verify).
This module assembles those pieces behind a single :class:`LCOCFModule`, which is
itself an ``nn.Module`` so its *only* trainable tensors — the three strength
weights (α, β, γ), the damage-predictor head and the residual-repair net — are
tracked and checkpointed as a unit (§3.3.5: ~千万级 params + 3 hyper-params).

The split between the pure feature builders (imported verbatim by the data
pipeline) and the learnable combiners is preserved here: this facade only
*orchestrates*, it adds no new maths, so training/inference parity is intact.

Typical per-step use inside the inference loop (§7.2 steps 2–4)::

    sub      = lcocf.parse(prompt)                    # once per prompt
    feats    = lcocf.strength_features(tubes, states, sub)
    strength = lcocf.strengths(feats)
    priors   = lcocf.prior_actions(strength, states)
    preds    = lcocf.predict(tubes, states, feats, strength, budget, step_frac)
"""

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
    """The lightweight counterfactual causal compute field as one component (§3).

    Parameters
    ----------
    lcocf_cfg
        L-COCF configuration slice (strength weights, predictor head, CF check).
    tube_cfg
        Needed only for the STA stability override threshold used by the mapping
        (an unstable tube is forced to FULL, §4.3.1) — kept explicit so the
        coupling is visible rather than hidden in a global config.
    parser
        Frozen prompt→sub-graph parser; defaults to the config-selected one
        (rule-based fallback or a real VLM). Injected for testability.
    token_dim
        DiT token width ``d`` — sizes the residual-repair net. Supplied by the
        backbone adapter at build time so this module stays backbone-agnostic.
    """

    def __init__(
        self,
        lcocf_cfg: LCOCFConfig,
        tube_cfg: TubeConfig,
        *,
        token_dim: int,
        parser: Optional[CausalParser] = None,
    ) -> None:
        super().__init__()
        self.cfg = lcocf_cfg
        self.tube_cfg = tube_cfg

        # -- pure, dependency-free feature builders (shared with the data pipeline)
        self.feature_builder = CausalStrengthFeatureBuilder()
        self.mapping = ComputeFieldMapping(lcocf_cfg.strength, tube_cfg)
        self.parser = parser or build_parser(lcocf_cfg)

        # -- the (only) trainable tensors of L-COCF -------------------------- #
        self.strength_field = CausalStrengthField(lcocf_cfg.strength)
        self.predictor = DamagePredictor(lcocf_cfg.predictor)
        self.repair_net = ResidualRepairNet(
            token_dim, hidden=lcocf_cfg.counterfactual.repair_net_dim
        )
        self.verifier = CounterfactualVerifier(
            lcocf_cfg.counterfactual, repair_net=self.repair_net
        )

    # ------------------------------------------------------------------ #
    # §3.3.1 — causal sub-graph (once per prompt)
    # ------------------------------------------------------------------ #

    def parse(self, prompt: str) -> CausalSubgraph:
        return self.parser.parse(prompt)

    # ------------------------------------------------------------------ #
    # §3.3.2 — causal strength field
    # ------------------------------------------------------------------ #

    def strength_features(
        self,
        tubes: List[SemanticTube],
        states: Dict[int, TubeState],
        subgraph: CausalSubgraph,
        tube_entity: Optional[Dict[int, float]] = None,
        tube_action_align: Optional[Dict[int, float]] = None,
    ) -> Dict[int, StrengthFeatures]:
        """Per-tube ``(s_E, s_A, s_T)`` features (§3.3.2).

        ``tube_entity`` / ``tube_action_align`` optionally override the entity
        importance and text-action alignment for a tube when a tube→triplet match
        has been resolved (else the builder falls back to sub-graph statistics).
        """
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
        """Combine each tube's ``(s_E, s_A, s_T)`` into a scalar strength ``s``."""
        if not feats:
            return {}
        ids = list(feats)
        stacked = torch.stack([feats[i].as_tensor() for i in ids])  # [K, 3]
        # detach: the float strengths drive non-differentiable tier/action control
        # flow; the differentiable training path calls ``strength_field`` directly.
        s = self.strength_field(stacked).detach()  # [K]
        return {i: float(s[j]) for j, i in enumerate(ids)}

    # ------------------------------------------------------------------ #
    # §3.3.3 — hierarchical discrete mapping → prior action
    # ------------------------------------------------------------------ #

    def prior_actions(
        self, strengths: Dict[int, float], states: Optional[Dict[int, TubeState]] = None
    ) -> Dict[int, Action]:
        return self.mapping.prior_actions(strengths, states)

    # ------------------------------------------------------------------ #
    # §3.3 — damage prediction (μ, σ) per action
    # ------------------------------------------------------------------ #

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
        """Predict per-action damage ``(μ, σ)`` for every tube in one batched pass.

        Returns ``{tube_id: DamagePrediction}`` where each prediction holds
        ``mu``/``sigma`` of shape ``[num_actions]``. Batching all tubes into a
        single forward keeps the head call O(1) per step regardless of tube count.
        """
        if not tubes:
            return {}
        ids = [t.tube_id for t in tubes]
        rows = [
            build_predictor_input(
                states[i], feats[i], strengths[i], budget, step_frac,
                self.cfg.predictor.context_dim, device=device, dtype=dtype,
            )
            for i in ids
        ]
        batch = torch.stack(rows)  # [K, in_dim]
        out = self.predictor(batch)  # mu/sigma: [K, num_actions]
        return {
            i: DamagePrediction(mu=out.mu[j], sigma=out.sigma[j])
            for j, i in enumerate(ids)
        }

    # ------------------------------------------------------------------ #
    # convenience: trainable-parameter accounting (§3.4 — 千万级 + 3 hyper-params)
    # ------------------------------------------------------------------ #

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Named groups so the trainer can apply different LRs / log counts."""
        return {
            "strength": [self.strength_field.alpha, self.strength_field.beta, self.strength_field.gamma],
            "predictor": list(self.predictor.parameters()),
            "repair": list(self.repair_net.parameters()),
        }
