"""Error certificate ``E_cert(k, t)`` and its training loss (§5.3.1).

Every anchoring (skip) decision is issued a *risk certificate* quantifying the
potential of that skip to propagate error to the final video. It is an
upper-confidence bound on the predicted damage plus the local risk factors the
L-COCF predictor does not directly model::

    E_cert(k,t) = μ_{k,t}                  predicted damage mean   (L-COCF)
                + κ · σ_{k,t}              epistemic margin        (L-COCF)
                + λ_res · δ_{k,t}          residual to last anchor (skip residual)
                + λ_bnd · b_{k,t}          boundary uncertainty    (tube state)
                + λ_age · age_{k,t}        steps since last anchor (staleness)
                + λ_cmsc · c_{k,t}         local CMSC violation    (§6/§5.3.1)

The coefficients are *learnable* (kept positive via softplus, initialised from
config) so the certificate can be calibrated by :meth:`ErrorCertificateModule.loss`
against realised teacher damage — making "risk" a trained quantity rather than a
hand-tuned heuristic. The value path is fully tensorised so a whole step's tubes
are certified in one shot, and differentiably (the predictor's μ/σ receive
gradient through the certificate loss).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.common.config import CertificateConfig
from cocf.common.types import Action, DamagePrediction, ErrorCertificate

Tensor = torch.Tensor

# Staleness normaliser: anchor age in steps is squashed by age/(age+AGE_TAU) so the
# term saturates and a very old anchor cannot dominate the certificate.
AGE_TAU = 8.0


class ErrorCertificateModule(nn.Module):
    """Computes (and learns) the per-tube error certificate ``E_cert`` (§5.3.1)."""

    def __init__(self, cfg: CertificateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Store the *inverse-softplus* of the init so that softplus(param) == init.
        self._kappa = nn.Parameter(_inv_softplus(cfg.kappa))
        self._res = nn.Parameter(_inv_softplus(cfg.lambda_res))
        self._bnd = nn.Parameter(_inv_softplus(cfg.lambda_bnd))
        self._age = nn.Parameter(_inv_softplus(cfg.lambda_age))
        self._cmsc = nn.Parameter(_inv_softplus(cfg.lambda_cmsc))

    # -- coefficients (always positive) --------------------------------- #

    @property
    def coeffs(self) -> Dict[str, float]:
        with torch.no_grad():
            return {
                "kappa": float(F.softplus(self._kappa)),
                "lambda_res": float(F.softplus(self._res)),
                "lambda_bnd": float(F.softplus(self._bnd)),
                "lambda_age": float(F.softplus(self._age)),
                "lambda_cmsc": float(F.softplus(self._cmsc)),
            }

    # -- tensorised value (training path) ------------------------------- #

    def value(
        self,
        mu: Tensor,
        sigma: Tensor,
        residual: Tensor,
        boundary: Tensor,
        anchor_age: Tensor,
        local_cmsc: Tensor,
    ) -> Tensor:
        """Batched certificate value ``[K]`` from per-tube tensors (all ``[K]``).

        Differentiable w.r.t. ``mu``/``sigma`` (the predictor) and the coefficients.
        """
        age = anchor_age / (anchor_age + AGE_TAU)
        return (
            mu
            + F.softplus(self._kappa) * sigma
            + F.softplus(self._res) * residual
            + F.softplus(self._bnd) * boundary
            + F.softplus(self._age) * age
            + F.softplus(self._cmsc) * local_cmsc
        )

    # -- single-tube dataclass (inference path) ------------------------- #

    @torch.no_grad()
    def compute(
        self,
        tube_id: int,
        step: int,
        action: Action,
        prediction: DamagePrediction,
        *,
        residual: float = 0.0,
        boundary: float = 0.0,
        anchor_age: float = 0.0,
        local_cmsc: float = 0.0,
    ) -> ErrorCertificate:
        """Certificate for one tube's chosen action, as a logged dataclass."""
        mu, sigma = prediction.of(action)
        c = self.coeffs
        age_term = anchor_age / (anchor_age + AGE_TAU)
        components = {
            "mu": float(mu),
            "kappa_sigma": c["kappa"] * float(sigma),
            "residual": c["lambda_res"] * residual,
            "boundary": c["lambda_bnd"] * boundary,
            "age": c["lambda_age"] * age_term,
            "cmsc": c["lambda_cmsc"] * local_cmsc,
        }
        return ErrorCertificate(
            value=float(sum(components.values())),
            tube_id=tube_id,
            step=step,
            action=action,
            components=components,
        )

    # -- certificate calibration loss (§5.3.1) -------------------------- #

    def loss(self, e_cert: Tensor, damage_true: Tensor) -> Tensor:
        """Certificate calibration loss, exactly as §5.3.1::

            L_cert = mean( max(0, y − E_cert)² )                 upper-bound hinge
                   + α_cert · mean( max(0, E_cert − τ_safe) )    safe-threshold penalty

        The first (squared) term penalises **under-certification** only — a certificate
        below the realised damage ``y`` is a false "safe", the costly error (§5.2) — so
        it drives ``E_cert`` to be an *upper bound* on ``y``. The second term keeps a
        calibrated certificate from growing needlessly far past the safe threshold
        ``τ_safe`` (so risk stays tight and skips are not over-suppressed). Both
        ``e_cert`` and ``damage_true`` are ``[K]``.
        """
        upper = F.relu(damage_true - e_cert).pow(2).mean()       # max(0, y − E_cert)²
        safe = F.relu(e_cert - self.cfg.tau_safe).mean()         # max(0, E_cert − τ_safe)
        return upper + self.cfg.alpha_cert * safe


def _inv_softplus(y: float) -> Tensor:
    """Return ``x`` such that ``softplus(x) == y`` (so params init to config values)."""
    import math

    y = max(float(y), 1e-4)
    # softplus⁻¹(y) = log(exp(y) − 1); use log1p/expm1 for numerical stability.
    return torch.tensor(math.log(math.expm1(y)) if y < 20 else y, dtype=torch.float32)
