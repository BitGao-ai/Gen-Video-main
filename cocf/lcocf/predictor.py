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
from cocf.common.types import Action, DamagePrediction, TubeState
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


def build_predictor_input_batch(
    states: Tensor,          # [B, 7]   tube-state vectors
    strength_feats: Tensor,  # [B, 3]   (s_E, s_A, s_T)
    strength: Tensor,        # [B]      combined causal strength s
    budget: Tensor,          # [B]      per-step budget B_t
    step_frac: Tensor,       # [B]      t / T ∈ [0,1]
    step_embed_dim: int,
) -> Tensor:
    """Batched, differentiable predictor input ``[B, in_dim]``.

    Concatenates in the **exact same order** as the per-sample
    :func:`build_predictor_input` — ``[state(7), s_E, s_A, s_T, strength, budget]``
    then the sinusoidal step embedding — so the features the predictor is trained on
    in Stage B are byte-for-byte the features it sees at inference (no train/serve
    skew). Stays in the autograd graph (``strength`` flows from the strength field).
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
        # Bias-init both heads so an *untrained* predictor yields a small μ and a
        # small σ rather than μ≈0.70/σ≈1.0 (the softplus/exp values at bias 0). The
        # error certificate is μ + κ·σ with κ=1.96, so zero biases put E_cert at ~2.7
        # against a τ_high of 0.80 and every tube is rolled back on step 2 — the
        # accelerator would disable itself before the predictor ever learns anything
        # (§5.3.1 cold start). See PredictorConfig.mu_init / log_var_init.
        nn.init.constant_(self.mu_head.bias, _inv_softplus(config.mu_init))
        nn.init.constant_(self.var_head.bias, float(config.log_var_init))
        # Small-weight init on the heads so the biases (not the random projections)
        # dominate at step 0 and the cold-start certificate is actually calibrated.
        nn.init.normal_(self.mu_head.weight, std=1e-3)
        nn.init.normal_(self.var_head.weight, std=1e-3)

    def forward(self, features: Tensor) -> DamagePrediction:
        """``features`` is ``[B, in_dim]`` (B = number of tubes); returns batched μ, σ."""
        # No gradient checkpointing: the trunk is 3 layers of width 128, so its
        # activations are a few hundred KB while recomputing them costs a second
        # forward on every backward — the trade is inverted at this size (§P2-10).
        h = self.trunk(features)
        mu = torch.nn.functional.softplus(self.mu_head(h))  # damage ≥ 0
        if self.cfg.pin_full_zero:
            # FULL is the *reference* the whole framework measures damage against:
            # Stage A labels it as exactly zero (§1.5) and the allocator's benefit /
            # cost arithmetic assumes it. Left free, softplus can emit μ[FULL] > μ[skip],
            # and then upgrading toward FULL has negative benefit while downgrading is
            # clamped to zero cost by ``max(0, ·)`` — a downgrade looks *free* and the
            # allocation degenerates (§P1-12). Anchoring the column removes the
            # degree of freedom instead of hoping training removes it.
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
    """``x`` such that ``softplus(x) == y`` — so ``mu_head.bias`` init lands on ``μ₀``."""
    y = max(float(y), 1e-6)
    # log(exp(y) − 1), via expm1 for stability at small y (the regime we init in).
    return float(math.log(math.expm1(y))) if y < 20.0 else y
