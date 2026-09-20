"""Action-aware transition executor: turns allocation decisions into latent updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from cocf.backbones.base import BackboneAdapter, BackboneCache, TextConditioning
from cocf.common.logging import get_logger
from cocf.common.types import Action, AllocationDecision, SemanticTube, TokenGrid

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class TransitionResult:
    """Output of one accelerated denoising step."""

    z_next: Tensor  # [B, N, d] latent at t_next
    cache: BackboneCache  # refreshed eps cache
    active_mask: Tensor  # [N] bool tokens planned to compute
    tube_residual: Dict[int, float]  # per-tube residual against the full-compute reference
    mask_ratio: float  # |active| / N (plan occupancy, not an efficiency metric)
    compute_ratio: float  # fraction of a dense forward actually spent
    z_full: Optional[Tensor] = None  # compute-everywhere candidate latent
    downgraded: Dict[int, Action] = field(default_factory=dict)  # tubes whose executed action changed

    @property
    def active_ratio(self) -> float:
        """Deprecated alias of :attr:`mask_ratio`."""
        return self.mask_ratio


class TransitionExecutor:
    """Executes a per-tube allocation decision into a latent update (stateless)."""

    def __init__(
        self,
        adapter: BackboneAdapter,
        lowfreq_stride: int = 2,
        dense_step_skip_below: float = 0.0,
        background_refresh_every: int = 0,
        max_unmeasured_steps: int = 0,
    ) -> None:
        self.adapter = adapter
        self.background_refresh_every = max(0, int(background_refresh_every))  # recompute background every N steps (0 = never)
        self.lowfreq_stride = max(1, int(lowfreq_stride))  # LOWFREQ spatial stride
        self.dense_step_skip_below = max(0.0, float(dense_step_skip_below))  # promote near-empty masks to a whole-step skip
        self.max_unmeasured_steps = max(0, int(max_unmeasured_steps))  # bound on steps without a measured residual (0 = none)
        self.risk_trigger = None  # optional RAEC coverage counters, set by the engine

    def build_active_mask(
        self,
        decision: AllocationDecision,
        tubes: List[SemanticTube],
        grid: TokenGrid,
        *,
        device,
    ) -> Tensor:
        """Bool [N] mask of tokens to freshly compute this step."""
        mask = torch.zeros(grid.num_tokens, dtype=torch.bool, device=device)
        for tube in tubes:
            action = decision.action_for(tube.tube_id, default=Action.FULL)
            if action == Action.FULL:
                mask[tube.all_token_indices().to(device)] = True
            elif action == Action.LOWFREQ:
                mask[self._strided_indices(tube, grid).to(device)] = True
        if self._refresh_background_at(decision.step):
            covered = torch.zeros_like(mask)
            for tube in tubes:
                covered[tube.all_token_indices().to(device)] = True
            mask |= ~covered
        return mask

    def _refresh_background_at(self, step: int) -> bool:
        """Whether this step recomputes the un-tubed background."""
        n = self.background_refresh_every
        return n > 0 and step > 0 and step % n == 0

    def _maybe_promote_to_step_skip(
        self,
        active_mask: Tensor,
        cache: Optional[BackboneCache],
        decision: AllocationDecision,
        tubes: List[SemanticTube],
    ) -> Tensor:
        """Clear a near-empty mask into a whole-step skip on dense-only adapters."""
        if self.adapter.supports_token_sparsity or self.dense_step_skip_below <= 0.0:
            return active_mask
        if cache is None or cache.model_output is None:
            return active_mask
        if any(
            decision.action_for(t.tube_id, default=Action.FULL) == Action.FULL
            for t in tubes
        ):
            return active_mask
        if self.risk_trigger is not None and self.risk_trigger.coverage_exhausted(
            self.max_unmeasured_steps
        ):
            _log.debug(
                "transition: a tube has gone %d steps without a measured skip residual "
                "— running the forward so RAEC's certificate is re-grounded",
                self.max_unmeasured_steps,
            )
            return active_mask
        occupancy = float(active_mask.float().mean().item())
        if 0.0 < occupancy <= self.dense_step_skip_below:
            _log.debug(
                "transition: mask occupancy %.4f ≤ %.4f on a dense-only adapter with no "
                "FULL tube — promoting to a whole-step skip (a partial dense forward "
                "saves nothing)", occupancy, self.dense_step_skip_below,
            )
            return torch.zeros_like(active_mask)
        return active_mask

    def _strided_indices(self, tube: SemanticTube, grid: TokenGrid) -> Tensor:
        """Spatially strided subset of a tube's tokens for LOWFREQ compute."""
        stride = self.lowfreq_stride
        kept: List[Tensor] = []
        for frame, idx in tube.tokens_by_frame.items():
            if stride == 1:
                kept.append(idx)
                continue
            local = idx - frame * grid.tokens_per_frame
            hi = torch.div(local, grid.w, rounding_mode="floor")
            wi = local - hi * grid.w
            keep = (hi % stride == 0) & (wi % stride == 0)
            kept.append(idx[keep])
        if not kept:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(kept)

    def prepare_masks(self, decision, tubes, grid, *, device, cache=None):
        """Return (planned, executed) active masks after step-skip promotion."""
        planned = self.build_active_mask(decision, tubes, grid, device=device)
        active = self._maybe_promote_to_step_skip(planned, cache, decision, tubes)
        return planned, active

    def step(
        self,
        z_t: Tensor,
        t: Tensor,
        t_next: Tensor,
        cond: TextConditioning,
        decision: AllocationDecision,
        tubes: List[SemanticTube],
        grid: TokenGrid,
        *,
        cache: Optional[BackboneCache] = None,
        anchor_latent: Optional[Tensor] = None,
        want_attention: bool = False,
        measure_residual: bool = False,
        prepared_masks: Optional[Tuple[Tensor, Tensor]] = None,
        fill_lowfreq: bool = True,
    ) -> TransitionResult:
        """Advance ``z_t`` to ``z_{t_next}`` honouring the per-tube allocation."""
        device = z_t.device
        planned_mask, active_mask = prepared_masks if prepared_masks is not None else self.prepare_masks(
            decision, tubes, grid, device=device, cache=cache
        )
        downgraded: Dict[int, Action] = {}
        if active_mask is not planned_mask:
            downgraded = {
                t.tube_id: Action.ANCHOR
                for t in tubes
                if decision.action_for(t.tube_id, default=Action.FULL) == Action.LOWFREQ
            }

        out = self.adapter.denoise(
            z_t, t, cond, grid=grid, active_mask=active_mask,
            cache=cache, want_attention=want_attention,
        )
        eps = out.model_output  # [B, N, d_out] full-resolution eps (grad-attached)

        z_full = self.adapter.scheduler_step(eps, t, t_next, z_t)

        computed = float(getattr(out, "compute_fraction", 1.0)) > 0.0

        skips = sorted(
            (t for t in tubes
             if decision.action_for(t.tube_id, default=Action.FULL).is_skip),
            key=lambda t: -int(decision.action_for(t.tube_id, default=Action.FULL)),
        )
        lowfreq = [
            t for t in tubes
            if decision.action_for(t.tube_id, default=Action.FULL) == Action.LOWFREQ
        ]
        do_fill = fill_lowfreq and computed and self.lowfreq_stride > 1
        needs_write = bool(skips) or bool(lowfreq and do_fill)
        z_next = z_full.clone() if needs_write else z_full
        tube_residual: Dict[int, float] = {}
        if do_fill:
            for tube in lowfreq:
                reconstructed = self.coarsen_lowfreq(
                    z_full, tube, grid, protected=active_mask
                )
                idx = tube.all_token_indices().to(device)
                keep = ~active_mask.to(device).index_select(0, idx)
                z_next.index_copy_(1, idx[keep], reconstructed.index_select(1, idx[keep]))

        active_now = active_mask.to(device)
        rode_tokens = 0
        for tube in skips:
            action = decision.action_for(tube.tube_id, default=Action.FULL)
            idx = tube.all_token_indices().to(device)
            if action == Action.ANCHOR and anchor_latent is not None:
                z_skip = anchor_latent.index_select(1, idx)
            elif action == Action.INTERP:
                rows = self.interp_rows(z_full, tube, grid)
                if rows is None:
                    rode_tokens += int(idx.numel())
                    continue
                z_skip = rows
            else:
                rode_tokens += int(idx.numel())
                continue
            if measure_residual:
                ref = z_full.index_select(1, idx)
                tube_residual[tube.tube_id] = float(
                    (ref - z_skip).pow(2).mean().sqrt().item()
                )
            keep = ~active_now.index_select(0, idx)
            if bool(keep.all()):
                z_next.index_copy_(1, idx, z_skip.to(z_next.dtype))
            elif bool(keep.any()):
                sel = keep.nonzero(as_tuple=True)[0]
                z_next.index_copy_(
                    1, idx.index_select(0, sel),
                    z_skip.index_select(1, sel).to(z_next.dtype),
                )

        if rode_tokens:
            _log.info(
                "transition: %d token(s) rode the spliced-ε step (freeze disabled)",
                rode_tokens,
            )

        active_ratio = float(active_mask.float().mean().item())
        if self.risk_trigger is not None:
            certified, blind = [], []
            for tube in tubes:
                tid = tube.tube_id
                executed = downgraded.get(tid, decision.action_for(tid, default=Action.FULL))
                if executed == Action.FULL or (executed == Action.LOWFREQ and computed):
                    certified.append(tid)
                elif tid in tube_residual:
                    certified.append(tid)
                else:
                    blind.append(tid)
            self.risk_trigger.note_measured(certified)
            self.risk_trigger.note_unmeasured(blind)
        return TransitionResult(
            z_next=z_next,
            cache=out.cache,
            active_mask=active_mask,
            tube_residual=tube_residual,
            mask_ratio=active_ratio,
            compute_ratio=float(getattr(out, "compute_fraction", 1.0)),
            z_full=z_full,
            downgraded=downgraded,
        )

    def coarsen_lowfreq(
        self, z: Tensor, tube: SemanticTube, grid: TokenGrid,
        *, protected: Optional[Tensor] = None,
    ) -> Tensor:
        """Coarsen the latent: LOWFREQ holes take their lattice anchor's value."""
        if self.lowfreq_stride <= 1:
            return z
        out = z.clone()
        self._fill_lowfreq(out, z, tube, grid, protected=protected)
        return out

    def interp_temporal(
        self, z: Tensor, tube: SemanticTube, grid: TokenGrid,
        freeze_to: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a copy of ``z`` with ``tube``'s tokens temporally interpolated."""
        rows = self.interp_rows(z, tube, grid)
        idx = tube.all_token_indices().to(z.device)
        if rows is None:
            if freeze_to is None or idx.numel() == 0:
                return z
            rows = freeze_to.index_select(1, idx)
        out = z.clone()
        out.index_copy_(1, idx, rows.to(out.dtype))
        return out

    def interp_rows(
        self, z: Tensor, tube: SemanticTube, grid: TokenGrid
    ) -> Optional[Tensor]:
        """INTERP latent operation: temporally blend each frame from its neighbours.

        Returns ``None`` when the tube spans fewer than two frames.
        """
        frames = tube.frames
        if len(frames) < 2:
            return None
        device = z.device
        per_frame = grid.tokens_per_frame
        rows: List[Tensor] = []
        for i, f in enumerate(frames):
            idx = tube.tokens_by_frame[f].to(device)
            local = idx - f * per_frame
            lo = frames[i - 1] if i > 0 else None
            hi = frames[i + 1] if i + 1 < len(frames) else None
            if lo is None:
                rows.append(z.index_select(1, hi * per_frame + local))
            elif hi is None:
                rows.append(z.index_select(1, lo * per_frame + local))
            else:
                w = (f - lo) / float(hi - lo)
                a = z.index_select(1, lo * per_frame + local)
                b = z.index_select(1, hi * per_frame + local)
                rows.append(a * (1.0 - w) + b * w)
        return torch.cat(rows, dim=1)

    def _fill_lowfreq(
        self, z_next: Tensor, z_full: Tensor, tube: SemanticTube, grid: TokenGrid,
        *, protected: Optional[Tensor] = None,
    ) -> None:
        """Fill the uncomputed LOWFREQ positions of ``z_next`` from ``z_full``."""
        stride = self.lowfreq_stride
        device = z_next.device
        for frame, idx in tube.tokens_by_frame.items():
            local = idx.to(device) - frame * grid.tokens_per_frame
            hi = torch.div(local, grid.w, rounding_mode="floor")
            wi = local - hi * grid.w
            kept = (hi % stride == 0) & (wi % stride == 0)
            if kept.all():
                continue
            hi_a = (torch.div(hi, stride, rounding_mode="floor") * stride).clamp_max(grid.h - 1)
            wi_a = (torch.div(wi, stride, rounding_mode="floor") * stride).clamp_max(grid.w - 1)
            src_flat = frame * grid.tokens_per_frame + hi_a * grid.w + wi_a
            holes = ~kept
            holes &= torch.isin(src_flat, idx.to(device))
            if protected is not None:
                holes &= ~protected.to(device=device, dtype=torch.bool).index_select(0, idx.to(device))
            z_next.index_copy_(
                1, idx[holes].to(device),
                z_full.index_select(1, src_flat[holes].to(device)),
            )
