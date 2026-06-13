"""Per-generation engine state & result containers (§7.2).

The :class:`InferenceEngine` is stateless across calls; everything that evolves
along a denoising trajectory lives in an :class:`EngineState` (owned by the engine
for the duration of one ``generate``). Keeping the mutable state in one explicit,
inspectable object — rather than scattered attributes — is what lets the loop stay
readable and the whole thing be unit-tested deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from cocf.backbones.base import BackboneCache, TextConditioning
from cocf.common.types import CausalSubgraph, SemanticTube, TokenGrid
from cocf.raec.anchor_store import AnchorStore

Tensor = torch.Tensor


@dataclass
class StepTrace:
    """Diagnostics for one denoising step (§9.4 efficiency/quality logging)."""

    step: int
    active_ratio: float
    budget: float
    predicted_cost: float
    num_tubes: int
    actions: Dict[int, str] = field(default_factory=dict)
    rollbacks: int = 0
    repairs: int = 0
    cf_checks: int = 0
    cf_repairs: int = 0


@dataclass
class EngineState:
    """Mutable trajectory state for one accelerated generation."""

    z: Tensor                                   # current latent [B, N, d] (tokens)
    grid: TokenGrid
    cond: TextConditioning
    subgraph: CausalSubgraph
    anchor_store: AnchorStore
    cache: Optional[BackboneCache] = None
    tubes: List[SemanticTube] = field(default_factory=list)
    # per-tube running quantities used by the next step's certificate / smoothing
    prev_probs: Dict[int, Tensor] = field(default_factory=dict)
    prev_residual: Dict[int, float] = field(default_factory=dict)
    anchor_age: Dict[int, int] = field(default_factory=dict)
    # per-tube countdown of remaining forced-FULL steps after a rollback/repair
    # (§5.3.2): while > 0 the allocator pins the tube to FULL so it is recomputed
    # forward instead of being allowed to skip again.
    force_full_countdown: Dict[int, int] = field(default_factory=dict)
    tube_visual_embed: Dict[int, Tensor] = field(default_factory=dict)
    prev_z: Optional[Tensor] = None
    # accumulated metrics
    traces: List[StepTrace] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        return self.cond.prompts[0] if self.cond.prompts else ""


@dataclass
class GenerationResult:
    """Output of :meth:`InferenceEngine.generate`."""

    video: Tensor                # [B, 3, F, H, W] decoded video
    z0: Tensor                   # [B, N, d] final clean latent (token form)
    traces: List[StepTrace] = field(default_factory=list)

    # -- convenience efficiency summaries (§9.4) ------------------------ #

    @property
    def mean_active_ratio(self) -> float:
        return sum(t.active_ratio for t in self.traces) / max(1, len(self.traces))

    @property
    def num_rollbacks(self) -> int:
        return sum(t.rollbacks for t in self.traces)

    @property
    def num_repairs(self) -> int:
        return sum(t.repairs for t in self.traces)

    def summary(self) -> Dict[str, float]:
        return {
            "steps": len(self.traces),
            "mean_active_ratio": round(self.mean_active_ratio, 4),
            "rollbacks": self.num_rollbacks,
            "repairs": self.num_repairs,
            "cf_repairs": sum(t.cf_repairs for t in self.traces),
        }
