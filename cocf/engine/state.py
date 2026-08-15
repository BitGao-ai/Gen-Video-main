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
    # mean damage uncertainty (mean σ over tubes) of the *previous* step, fed into the
    # next step's dynamic budget (§7.3 平均损害不确定度 term). Carried on the state
    # because the budget is sized before this step's σ is known (it conditions the
    # predictor), so the causal, no-future-info signal is the previous step's σ.
    prev_mean_uncertainty: float = 0.0
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
        }
