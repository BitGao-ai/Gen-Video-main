"""Counterfactual damage predictor: the learnable core of L-COCF.

For each candidate action ``a`` on a tube at step ``t`` it predicts the final-video
counterfactual damage as a Gaussian ``(mu, sigma)``: ``mu`` is the expected marginal
damage of the cheap action versus FULL, ``sigma`` its epistemic uncertainty. ``mu``
feeds the budget-constrained allocator and the error certificate; ``sigma`` drives the
uncertainty term of the budget schedule and the certificate's risk margin.

The exact input assembly lives here (``build_predictor_input``) and is imported by the
data pipeline, so training and inference see identical features.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from cocf.common.config import PredictorConfig
from cocf.common.types import Action, DamagePrediction, TubeState
from cocf.lcocf.strength import StrengthFeatures

Tensor = torch.Tensor


def sinusoidal_embedding(values: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal embedding of a ``[...]`` scalar tensor to ``[..., dim]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / max(half, 1)
    )
    args = values.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:  # pad odd dims
        emb = torch.cat([emb, torch.zeros(*emb.shape[:-1], 1, device=values.device)], dim=-1)
    return emb


# scalar features concatenated before the step embedding:
#   state(7) + [s_E, s_A, s_T, strength, budget] (5)
_SCALAR_DIM = 7 + 5


def build_predictor_input(
    state: TubeState,
    strength_feats: StrengthFeatures,
    strength: float,
    budget: float,
    step_frac: float,
    step_embed_dim: int,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Assemble the predictor input vector for one tube at one step.

    ``step_frac`` in [0, 1] is ``t / T`` (1 = pure noise, 0 = clean), embedded
    sinusoidally so the predictor is conditioned on the denoising phase.
    """
    scalars = torch.tensor(
        [
            state.identity_confidence, state.occlusion, state.interaction,
            state.boundary_uncertainty, state.motion_phase, state.causal_value,
            state.anchor_age,
            strength_feats.s_E, strength_feats.s_A, strength_feats.s_T,
            float(strength), float(budget),
        ],
        device=device, dtype=dtype,
    )
    step_emb = sinusoidal_embedding(
        torch.tensor(float(step_frac), device=device), step_embed_dim
    ).to(dtype)
    return torch.cat([scalars, step_emb], dim=-1)


def predictor_input_dim(cfg: PredictorConfig) -> int:
    return _SCALAR_DIM + cfg.context_dim


def build_predictor_input_batch(
    states: Tensor,          # [B, 7] tube-state vectors
    strength_feats: Tensor,  # [B, 3] (s_E, s_A, s_T)
    strength: Tensor,        # [B] combined causal strength
    budget: Tensor,          # [B] per-step budget
    step_frac: Tensor,       # [B] t / T in [0, 1]
    step_embed_dim: int,
) -> Tensor:
    """Batched, differentiable predictor input ``[B, in_dim]``.

    Concatenates in the same order as :func:`build_predictor_input`
    (``[state(7), s_E, s_A, s_T, strength, budget]`` then the step embedding) so Stage-B
    training and inference see identical features. Stays in the autograd graph.
    """
    scalars = torch.cat(
        [
            states,                       # [B, 7]
            strength_feats,               # [B, 3]
            strength.reshape(-1, 1),      # [B, 1]
            budget.reshape(-1, 1),        # [B, 1]
        ],
        dim=-1,
    )  # [B, 12]
    step_emb = sinusoidal_embedding(step_frac.reshape(-1), step_embed_dim).to(scalars.dtype)
    return torch.cat([scalars, step_emb], dim=-1)  # [B, 12 + step_embed_dim]


class DamagePredictor(nn.Module):
    """MLP head mapping a tube's features to per-action ``(mu, sigma)``."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.cfg = config
        in_dim = predictor_input_dim(config)
        dims = [in_dim] + [config.hidden_dim] * config.num_layers
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(config.hidden_dim, config.num_actions)
        # log-variance head for a calibrated, heteroscedastic sigma
        self.var_head = nn.Linear(config.hidden_dim, config.num_actions)
        # Bias-init both heads so an untrained predictor yields small mu and sigma;
        # the cold-start error certificate (mu + k*sigma) must sit below the rollback
        # threshold. See PredictorConfig.mu_init / log_var_init.
        nn.init.constant_(self.mu_head.bias, _inv_softplus(config.mu_init))
        nn.init.constant_(self.var_head.bias, float(config.log_var_init))
        # Small-weight init so the biases dominate at step 0.
        nn.init.normal_(self.mu_head.weight, std=1e-3)
        nn.init.normal_(self.var_head.weight, std=1e-3)

    def forward(self, features: Tensor) -> DamagePrediction:
        """``features`` is ``[B, in_dim]`` (B = number of tubes); returns batched mu, sigma."""
        h = self.trunk(features)
        mu = torch.nn.functional.softplus(self.mu_head(h))  # damage >= 0
        if self.cfg.pin_full_zero:
            # FULL is the reference damage is measured against, so its column is pinned
            # to zero to keep the allocator's benefit/cost arithmetic consistent.
            keep = torch.ones_like(mu)
            keep[..., int(Action.FULL)] = 0.0
            mu = mu * keep
        if self.cfg.predict_log_variance:
            sigma = torch.exp(0.5 * self.var_head(h).clamp(-10.0, 10.0))
        else:
            sigma = torch.nn.functional.softplus(self.var_head(h)) + 1e-4
        return DamagePrediction(mu=mu, sigma=sigma)

    @torch.no_grad()
    def predict_one(self, features: Tensor) -> DamagePrediction:
        """Single-tube convenience: accepts ``[in_dim]`` or ``[1, in_dim]``."""
        was_training = self.training
        self.eval()
        f = features.unsqueeze(0) if features.dim() == 1 else features
        out = self.forward(f)
        if was_training:
            self.train()
        return DamagePrediction(mu=out.mu[0], sigma=out.sigma[0])


def _inv_softplus(y: float) -> float:
    """``x`` such that ``softplus(x) == y``, so ``mu_head.bias`` init lands on ``mu_0``."""
    y = max(float(y), 1e-6)
    # log(exp(y) - 1), via expm1 for stability at small y.
    return float(math.log(math.expm1(y))) if y < 20.0 else y
