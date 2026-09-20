"""RAEC facade bundling certificate, trigger, and repair."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from cocf.common.config import CertificateConfig, MemoryConfig, TriggerConfig
from cocf.common.types import Action, DamagePrediction, ErrorCertificate, TubeState
from cocf.raec.anchor_store import AnchorStore
from cocf.raec.certificate import ErrorCertificateModule
from cocf.raec.repair import BoundaryRepair, RepairResult
from cocf.raec.trigger import RiskTrigger

Tensor = torch.Tensor


class RAECModule(nn.Module):
    """Revocable anchoring and error certificates."""

    def __init__(self, cert_cfg: CertificateConfig, trigger_cfg: TriggerConfig) -> None:
        super().__init__()
        self.certificate = ErrorCertificateModule(cert_cfg)
        self.trigger = RiskTrigger(trigger_cfg)
        self.repair = BoundaryRepair(trigger_cfg)

    def new_anchor_store(self, memory: Optional[MemoryConfig] = None) -> AnchorStore:
        """Create fresh anchor store for one generation."""
        offload = bool(memory.offload_backbone_to_cpu) if memory else False
        return AnchorStore(offload_to_cpu=offload)

    def reset(self) -> None:
        """Clear per-run trigger bookkeeping."""
        self.trigger.reset()

    def certify(
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
        return self.certificate.compute(
            tube_id, step, action, prediction,
            residual=residual, boundary=boundary,
            anchor_age=anchor_age, local_cmsc=local_cmsc,
        )

    @torch.no_grad()
    def action_risk(
        self,
        prediction: DamagePrediction,
        *,
        boundary: float = 0.0,
        anchor_age: float = 0.0,
        local_cmsc: float = 0.0,
    ) -> Tensor:
        """Compute prior risk for every candidate action."""
        zeros = torch.zeros_like(prediction.mu)
        return self.certificate.value(
            prediction.mu,
            prediction.sigma,
            zeros,
            zeros + float(boundary),
            zeros + float(anchor_age),
            zeros + float(local_cmsc),
        )
