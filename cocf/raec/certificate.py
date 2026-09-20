"""Error certificate value and calibration loss."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.common.config import CertificateConfig
from cocf.common.types import Action, DamagePrediction, ErrorCertificate

Tensor = torch.Tensor
AGE_TAU = 8.0


class ErrorCertificateModule(nn.Module):
    """Computes and learns per-tube error certificates."""

    def __init__(self, cfg: CertificateConfig) -> None:
        """Create certificate module from config."""
        super().__init__()
        self.cfg = cfg
        self._kappa = nn.Parameter(_inv_softplus(cfg.kappa))
        self._res = nn.Parameter(_inv_softplus(cfg.lambda_res))
        self._bnd = nn.Parameter(_inv_softplus(cfg.lambda_bnd))
        self._age = nn.Parameter(_inv_softplus(cfg.lambda_age))
        self._cmsc = nn.Parameter(_inv_softplus(cfg.lambda_cmsc))

    @property
    def coeffs(self) -> Dict[str, float]:
        """Return positive coefficient values."""
        with torch.no_grad():
            return {
                "kappa": float(F.softplus(self._kappa)),
                "lambda_res": float(F.softplus(self._res)),
                "lambda_bnd": float(F.softplus(self._bnd)),
                "lambda_age": float(F.softplus(self._age)),
                "lambda_cmsc": float(F.softplus(self._cmsc)),
            }

    def value(
        self,
        mu: Tensor,
        sigma: Tensor,
        residual: Tensor,
        boundary: Tensor,
        anchor_age: Tensor,
        local_cmsc: Tensor,
    ) -> Tensor:
        """Compute batched certificate values."""
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
        """Compute certificate for one tube action."""
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

    def loss(self, e_cert: Tensor, damage_true: Tensor) -> Tensor:
        """Compute certificate calibration loss."""
        upper = F.relu(damage_true - e_cert).pow(2).mean()
        safe = F.relu(e_cert - self.cfg.tau_safe).mean()
        return upper + self.cfg.alpha_cert * safe


def _inv_softplus(y: float) -> Tensor:
    """Return x with softplus(x) equal to y."""
    import math

    y = max(float(y), 1e-4)
    return torch.tensor(math.log(math.expm1(y)) if y < 20 else y, dtype=torch.float32)
