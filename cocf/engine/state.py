"""Per-generation engine state & result containers.

The :class:`InferenceEngine` is stateless across calls; everything that evolves along
a denoising trajectory lives in an :class:`EngineState`, owned by the engine for the
duration of one ``generate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from cocf.backbones.base import BackboneCache, TextConditioning
from cocf.common.types import CausalSubgraph, SemanticTube, TokenGrid
from cocf.raec.anchor_store import AnchorStore

Tensor = torch.Tensor


@dataclass
class StepTrace:
    """Diagnostics for one denoising step."""

    step: int
    # Fraction of tokens the allocation marked "recompute" — a plan statistic, not a cost.
    mask_ratio: float
    budget: float
    predicted_cost: float
    num_tubes: int
    # Fraction of a dense denoiser forward actually executed this step (0.0 = skipped,
    # 1.0 = full dense forward); the only number that may be quoted as a saving.
    compute_ratio: float = 1.0
    actions: Dict[int, str] = field(default_factory=dict)
    rollbacks: int = 0
    repairs: int = 0
    cf_checks: int = 0
    cf_repairs: int = 0
    predicted_damage: float = 0.0

    @property
    def active_ratio(self) -> float:
        """Deprecated alias of :attr:`mask_ratio` (see :class:`TransitionResult`)."""
        return self.mask_ratio


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
    # tube_id -> pooled CLIP visual embed [d_v], computed at tube-build time and reused
    # every step, so the certificate's λ_cmsc term needs no per-step perception forward.
    tube_embeds: Dict[int, Tensor] = field(default_factory=dict)
    # Mean damage uncertainty (mean σ over tubes) of the previous step, fed into the next
    # step's dynamic budget. Carried on the state because the budget is sized before this
    # step's σ is known.
    prev_mean_uncertainty: float = 0.0
    grad_window: int = 0
    retained_computed: int = 0
    graph_cuts: int = 0
    latent_flows: Dict[int, Tensor] = field(default_factory=dict)
    # accumulated metrics
    traces: List[StepTrace] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        return self.cond.prompts[0] if self.cond.prompts else ""


@dataclass
class GenerationResult:
    """Output of :meth:`InferenceEngine.generate`."""

    video: Tensor                # [B, 3, F, H, W] decoded video, in [0, 1]
    z0: Tensor                   # [B, N, d] final clean latent (token form)
    traces: List[StepTrace] = field(default_factory=list)
    # The tube set the trajectory ended on and the latent geometry it was built against;
    # Stage C's per-tube conservation loss pools each tube's visual embed from the render.
    tubes: List[SemanticTube] = field(default_factory=list)
    grid: Optional[TokenGrid] = None
    # Pixel-frame range ``[start, stop)`` that :attr:`video` covers; ``None`` means the
    # whole clip. Stage C slices ``Y_full`` by this before comparing.
    frame_span: Optional[Tuple[int, int]] = None
    # Peak CUDA memory (GiB) over the whole trajectory, measured by the engine; 0.0 off CUDA.
    peak_gib: float = 0.0

    # -- convenience efficiency summaries ------------------------ #

    @property
    def mean_compute_ratio(self) -> float:
        """Mean fraction of a dense denoiser forward actually executed per step.

        The efficiency metric: ``1.0`` means no compute was saved.
        """
        return sum(t.compute_ratio for t in self.traces) / max(1, len(self.traces))

    @property
    def mean_mask_ratio(self) -> float:
        """Mean allocated-token occupancy — a *plan* statistic, not a saving."""
        return sum(t.mask_ratio for t in self.traces) / max(1, len(self.traces))

    @property
    def mean_active_ratio(self) -> float:
        """Deprecated alias of :attr:`mean_mask_ratio`."""
        return self.mean_mask_ratio

    @property
    def num_rollbacks(self) -> int:
        return sum(t.rollbacks for t in self.traces)

    @property
    def num_repairs(self) -> int:
        return sum(t.repairs for t in self.traces)

    def summary(self) -> Dict[str, float]:
        """Efficiency/quality summary.

        ``mean_compute_ratio`` is the cost figure; ``mean_mask_ratio`` is the plan
        statistic kept alongside it.
        """
        return {
            "steps": len(self.traces),
            "mean_compute_ratio": round(self.mean_compute_ratio, 4),
            "mean_mask_ratio": round(self.mean_mask_ratio, 4),
            "rollbacks": self.num_rollbacks,
            "repairs": self.num_repairs,
            "cf_repairs": sum(t.cf_repairs for t in self.traces),
            "peak_gib": round(self.peak_gib, 3),
        }
