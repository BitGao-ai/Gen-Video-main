"""Per-generation engine state & result containers (§7.2).

The :class:`InferenceEngine` is stateless across calls; everything that evolves
along a denoising trajectory lives in an :class:`EngineState` (owned by the engine
for the duration of one ``generate``). Keeping the mutable state in one explicit,
inspectable object — rather than scattered attributes — is what lets the loop stay
readable and the whole thing be unit-tested deterministically.
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
    """Diagnostics for one denoising step (§9.4 efficiency/quality logging)."""

    step: int
    # Fraction of tokens the allocation marked "recompute" — a *plan* statistic, not
    # a cost. Reported separately from, and never as a substitute for, compute_ratio.
    mask_ratio: float
    budget: float
    predicted_cost: float
    num_tubes: int
    # Fraction of a dense denoiser forward actually executed this step, as reported by
    # the backbone adapter (0.0 = the transformer was skipped entirely, 1.0 = full
    # dense forward). This is the only number that may be quoted as a saving.
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
    # tube_id -> pooled CLIP visual embed [d_v], computed from the preview decode at
    # tube-build time and reused every step. This is what makes the certificate's
    # §5.3.1 λ_cmsc term computable inside the accelerated loop: scoring it per step
    # would mean a perception forward per tube per step, while the embed only changes
    # when the tubes are rebuilt (§P4-4).
    tube_embeds: Dict[int, Tensor] = field(default_factory=dict)
    # mean damage uncertainty (mean σ over tubes) of the *previous* step, fed into the
    # next step's dynamic budget (§7.3 平均损害不确定度 term). Carried on the state
    # because the budget is sized before this step's σ is known (it conditions the
    # predictor), so the causal, no-future-info signal is the previous step's σ.
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
    # The tube set the trajectory ended on, and the latent geometry it was built
    # against. Stage C's §6.3.2 conservation loss is *per tube* — it needs these to
    # pool each tube's visual embed out of the render — and they are already held by
    # the engine, so handing them back costs nothing (§P4-4/§P4-A3).
    tubes: List[SemanticTube] = field(default_factory=list)
    grid: Optional[TokenGrid] = None
    # Pixel-frame range ``[start, stop)`` of the source clip that :attr:`video` covers.
    # ``None`` means the whole clip (every inference render, and any training render
    # that did not window its differentiable decode). Stage C slices ``Y_full`` by this
    # before comparing, so a windowed render is never scored against the wrong frames.
    frame_span: Optional[Tuple[int, int]] = None
    # Peak CUDA memory (GiB) over the whole trajectory, measured by the engine.
    # 0.0 off CUDA. §9.4 lists 峰值显存 among the efficiency metrics to report, and an
    # accelerator that buys latency with memory should have that visible rather than
    # inferred.
    peak_gib: float = 0.0

    # -- convenience efficiency summaries (§9.4) ------------------------ #

    @property
    def mean_compute_ratio(self) -> float:
        """Mean fraction of a dense denoiser forward actually executed per step.

        **This is the efficiency metric.** ``1.0`` means no compute was saved; a run
        on a backbone without a token-sparse attention kernel will report ~1.0 minus
        the fraction of steps that were skipped outright, no matter how aggressive
        the tube allocation looks.
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

        ``mean_compute_ratio`` is the honest cost figure; ``mean_mask_ratio`` is kept
        alongside it (clearly named) because the gap between the two is exactly the
        saving a sparse-attention kernel would unlock and is worth watching.
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
