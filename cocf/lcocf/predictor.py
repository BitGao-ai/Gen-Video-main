"""Counterfactual damage predictor ``H_φ`` (§3.3, the learnable core of L-COCF).

Predicts, for each candidate action ``a`` on a tube ``g_k`` at step ``t``, the
*final-video* counterfactual damage ``y_{k,t,a}`` as a Gaussian ``(μ, σ)``:

    μ   expected marginal damage of taking the cheap action (vs FULL)
    σ   epistemic uncertainty of that estimate

This is the single trainable network of L-COCF (a few-MLP head, ~1–10 M params,
§3.3.5). It consumes the 7-dim tube state plus a compact causal/temporal context;
``μ`` feeds the budget-constrained allocator (§2.2) and the error certificate
(§5.3.1), while ``σ`` drives the uncertainty term of the budget schedule (§7.3)
and the certificate's risk margin.

The *exact* input assembly lives here (``build_predictor_input``) and is imported
verbatim by the data pipeline, so the features the predictor is trained on are
byte-for-byte the features it sees at inference — no train/serve skew.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from cocf.common.config import PredictorConfig
from cocf.common.memory import checkpointed
from cocf.common.types import DamagePrediction, TubeState
from cocf.lcocf.strength import StrengthFeatures

Tensor = torch.Tensor


def sinusoidal_embedding(values: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal embedding of a ``[...]`` scalar tensor → ``[..., dim]``."""
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

    ``step_frac`` ∈ [0,1] is ``t / T`` (1=pure noise, 0=clean), embedded
    sinusoidally so the predictor can be conditioned on the denoising phase
    (early structure vs late detail, §7.3).
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


class DamagePredictor(nn.Module):
    """MLP head mapping a tube's features → per-action ``(μ, σ)`` (§3.3)."""

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
        # log-variance head for a calibrated, heteroscedastic σ
        self.var_head = nn.Linear(config.hidden_dim, config.num_actions)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.var_head.bias)

    def forward(self, features: Tensor) -> DamagePrediction:
        """``features`` is ``[B, in_dim]`` (B = number of tubes); returns batched μ, σ."""
        h = checkpointed(self.trunk)(features) if self.training else self.trunk(features)
        mu = torch.nn.functional.softplus(self.mu_head(h))  # damage ≥ 0
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
