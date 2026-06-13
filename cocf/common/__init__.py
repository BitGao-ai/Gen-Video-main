"""Infrastructure layer: shared types, config, registry, memory helpers, logging."""

from __future__ import annotations

from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.registry import BACKBONES, Registry, register_backbone
from cocf.common.types import (
    Action,
    AllocationDecision,
    CausalSubgraph,
    CausalTriplet,
    DamagePrediction,
    ErrorCertificate,
    Region,
    SemanticTube,
    StrengthLevel,
    TokenGrid,
    TriggerLevel,
    TubeState,
    TUBE_STATE_DIM,
    TUBE_STATE_FIELDS,
)

__all__ = [
    "Config",
    "get_logger",
    "BACKBONES",
    "Registry",
    "register_backbone",
    "Action",
    "AllocationDecision",
    "CausalSubgraph",
    "CausalTriplet",
    "DamagePrediction",
    "ErrorCertificate",
    "Region",
    "SemanticTube",
    "StrengthLevel",
    "TokenGrid",
    "TriggerLevel",
    "TubeState",
    "TUBE_STATE_DIM",
    "TUBE_STATE_FIELDS",
]
