"""RAEC — Revocable Anchoring & Error Certificates (§5).

Replaces irreversible caching with a risk-controlled, *revocable* mechanism that
bounds error propagation (the §5.4 upper bound):

    certificate   §5.3.1  E_cert(k,t) risk score + its calibration loss
    trigger       §5.3.2  KEEP / REPAIR / ROLLBACK policy + force-FULL pins
    repair        §5.3.2  rollback, boundary fusion, cache-refresh operators
    anchor_store  §5.3.2  per-tube last-verified-safe latent library
    module        §5      the RAECModule facade
"""

from __future__ import annotations

from cocf.raec.anchor_store import AnchorStore
from cocf.raec.certificate import AGE_TAU, ErrorCertificateModule
from cocf.raec.module import RAECModule
from cocf.raec.repair import BoundaryRepair, RepairResult
from cocf.raec.trigger import RiskTrigger

__all__ = [
    "RAECModule",
    "ErrorCertificateModule",
    "AGE_TAU",
    "RiskTrigger",
    "BoundaryRepair",
    "RepairResult",
    "AnchorStore",
]
