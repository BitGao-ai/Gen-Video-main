"""RAEC facade — certificate + trigger + repair as one component (§5).

Bundles the four RAEC pieces so the engine and trainer see a single object:

    certificate  ErrorCertificateModule  (learnable — the only RAEC params)
    trigger      RiskTrigger             (per-run policy state: force-FULL pins)
    repair       BoundaryRepair          (stateless latent operators)
    anchor store created per generation by :meth:`new_anchor_store` (trajectory state)

The certificate carries the learnable coefficients and is the part that is saved /
fine-tuned; the trigger and repair are deterministic policy. The anchor store is
*trajectory* state (one per generated video), so it is created fresh per run rather
than held here.
"""

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


class RAECModule(nn.Module):
    """Revocable anchoring & error certificates, wired together (§5)."""

    def __init__(self, cert_cfg: CertificateConfig, trigger_cfg: TriggerConfig) -> None:
        super().__init__()
        self.certificate = ErrorCertificateModule(cert_cfg)
        self.trigger = RiskTrigger(trigger_cfg)
        self.repair = BoundaryRepair(trigger_cfg)

    # ------------------------------------------------------------------ #
    # per-run state
    # ------------------------------------------------------------------ #

    def new_anchor_store(self, memory: Optional[MemoryConfig] = None) -> AnchorStore:
        offload = bool(memory.offload_backbone_to_cpu) if memory else False
        return AnchorStore(offload_to_cpu=offload)

    def reset(self) -> None:
        """Clear per-run trigger bookkeeping (call at the start of each generation)."""
        self.trigger.reset()

    # ------------------------------------------------------------------ #
    # certification (inference path)
    # ------------------------------------------------------------------ #

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
        return self.certificate.compute(
            tube_id, step, action, prediction,
            residual=residual, boundary=boundary,
            anchor_age=anchor_age, local_cmsc=local_cmsc,
        )
