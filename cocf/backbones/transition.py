"""Action-aware transition executor — where the FLOPs/VRAM saving is realised.

The accelerator decides *what* to do per tube (the :class:`AllocationDecision`);
this module turns that decision into an actual latent update ``z_t -> z_{t-1}``
while computing the denoiser ε_θ on as few tokens as possible.

It is written entirely against the backbone-agnostic :class:`BackboneAdapter`
contract, so the four compute actions are defined *once* here and work for every
backbone (user requirement #2). The mapping from action → which tokens are
"active" (freshly computed) is the single lever that produces the speed-up
(user requirement #1):

    FULL     all of the tube's tokens are active            cost ∝ 1.00·|g_k|
    LOWFREQ  a strided spatial subset is active, the rest    cost ∝ 0.45·|g_k|
             are upsampled from the computed neighbours
    INTERP   no token is active; the latent is advanced by   cost ∝ 0.15·|g_k|
             a fresh scheduler step reusing the *cached* ε_θ
    ANCHOR   no token is active; the latent is frozen to     cost ∝ 0.00
             the last verified-safe anchor verbatim

Only FULL/LOWFREQ tokens enter ``adapter.denoise(active_mask=…)``; ANCHOR/INTERP
tokens never touch the (expensive) attention, which is what saves both compute
and the activation memory that dominates video-DiT VRAM.

Design note — axis conventions
------------------------------
``ANCHOR`` reuses across the *denoising-step* axis (skip this step, keep the last
latent), matching cache methods like DeepCache/TeaCache. ``INTERP`` reuses the
*cached ε_θ* but still takes a fresh, cheap scheduler step so the tube keeps
moving with the global trajectory instead of stalling. ``LOWFREQ`` keeps the
low-frequency band fresh (strided compute + upsample) and inherits high-frequency
detail from cache. These choices are local to this file; the rest of the
framework only sees "an action was executed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from cocf.backbones.base import BackboneAdapter, BackboneCache, TextConditioning
from cocf.common.types import Action, AllocationDecision, SemanticTube, TokenGrid

Tensor = torch.Tensor


@dataclass
class TransitionResult:
    """Output of one accelerated step (§7.2 step 6)."""

    z_next: Tensor  # [B, N, d] latent at t_next
    cache: BackboneCache  # refreshed ε_θ cache (full-resolution, for reuse)
    active_mask: Tensor  # [N] bool — tokens freshly computed this step
    # per-tube residual δ_k = ‖z_full − z_action‖ on the tube, for RAEC (§5.3.1).
    # Only populated for tubes that skipped (INTERP/ANCHOR) when ``measure_residual``.
    tube_residual: Dict[int, float]
    active_ratio: float  # |active| / N — the efficiency metric (§9.4)
    # the "compute-everywhere" candidate latent z_full (cached-ε scheduler step).
    # Returned so RAEC boundary fusion (§5.3.2) and the single-hop counterfactual
    # check (§3.3.4) have the full-compute reference without recomputing it.
    z_full: Optional[Tensor] = None


class TransitionExecutor:
    """Executes a per-tube :class:`AllocationDecision` into a latent update.

    Stateless w.r.t. the diffusion trajectory: all evolving state (the anchor
    latent, the ε cache) is passed in and returned, so the engine owns memory and
    this class stays trivially testable and reusable.
    """

    def __init__(self, adapter: BackboneAdapter, lowfreq_stride: int = 2) -> None:
        self.adapter = adapter
        # spatial stride for LOWFREQ active subsampling (2 ⇒ ~1/4 tokens computed)
        self.lowfreq_stride = max(1, int(lowfreq_stride))

    # ------------------------------------------------------------------ #
    # Active-mask construction (the FLOPs lever)
    # ------------------------------------------------------------------ #

    def build_active_mask(
        self,
        decision: AllocationDecision,
        tubes: List[SemanticTube],
        grid: TokenGrid,
        *,
        device,
    ) -> Tensor:
        """Bool ``[N]`` mask of tokens to freshly compute this step.

        Tokens not covered by *any* tube (background) default to inactive — they
        are the cheapest to reuse and the axioms (§3.2.2) say their causal effect
        is ~constant. FULL marks every tube token active; LOWFREQ marks a strided
        spatial subset; INTERP/ANCHOR mark none.
        """
        mask = torch.zeros(grid.num_tokens, dtype=torch.bool, device=device)
        for tube in tubes:
            action = decision.action_for(tube.tube_id, default=Action.FULL)
            if action == Action.FULL:
                mask[tube.all_token_indices().to(device)] = True
            elif action == Action.LOWFREQ:
                mask[self._strided_indices(tube, grid).to(device)] = True
            # INTERP / ANCHOR contribute no active tokens
        return mask

    def _strided_indices(self, tube: SemanticTube, grid: TokenGrid) -> Tensor:
        """Spatially strided subset of a tube's tokens for LOWFREQ compute.

        We keep every ``stride``-th column/row *within each frame* so the kept
        tokens form a coarse grid that bilinear upsampling can reconstruct from.
        """
        stride = self.lowfreq_stride
        kept: List[Tensor] = []
        for frame, idx in tube.tokens_by_frame.items():
            if stride == 1:
                kept.append(idx)
                continue
            # decode (hi, wi) of each flat index, keep those on the coarse lattice
            local = idx - frame * grid.tokens_per_frame
            hi = torch.div(local, grid.w, rounding_mode="floor")
            wi = local - hi * grid.w
            keep = (hi % stride == 0) & (wi % stride == 0)
            kept.append(idx[keep])
        if not kept:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(kept)

    # ------------------------------------------------------------------ #
    # The step
    # ------------------------------------------------------------------ #

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
    ) -> TransitionResult:
        """Advance ``z_t`` to ``z_{t_next}`` honouring the per-tube allocation.

        Parameters
        ----------
        anchor_latent
            ``[B, N, d]`` last verified-safe latent per token (for ANCHOR reuse and
            the RAEC residual). If ``None``, ANCHOR falls back to a cheap ε-reuse
            step (still correct, just not frozen).
        measure_residual
            When True, also compute ‖z_full − z_action‖ per skipped tube against a
            *cache-reused full step* reference, feeding the error certificate.
        """
        device = z_t.device
        active_mask = self.build_active_mask(decision, tubes, grid, device=device)

        # 1) Denoise only the active tokens; the adapter splices inactive ε from cache.
        out = self.adapter.denoise(
            z_t, t, cond, grid=grid, active_mask=active_mask,
            cache=cache, want_attention=want_attention,
        )
        eps = out.cache.model_output  # [B, N, d_out] full-resolution ε (active fresh)

        # 2) A single scheduler step gives the "compute everywhere" candidate.
        z_full = self.adapter.scheduler_step(eps, t, t_next, z_t)

        # 3) Per-action latent reconstruction for the skipped tubes.
        z_next = z_full.clone()
        tube_residual: Dict[int, float] = {}
        for tube in tubes:
            action = decision.action_for(tube.tube_id, default=Action.FULL)
            if action in (Action.FULL, Action.LOWFREQ):
                if action == Action.LOWFREQ and self.lowfreq_stride > 1:
                    self._fill_lowfreq(z_next, z_full, tube, grid)
                continue  # already in z_full
            idx = tube.all_token_indices().to(device)
            if action == Action.ANCHOR and anchor_latent is not None:
                z_skip = anchor_latent.index_select(1, idx)
            else:  # INTERP, or ANCHOR without an anchor latent → reuse cached ε step
                z_skip = z_full.index_select(1, idx)  # already a cheap ε-reuse step
            if measure_residual:
                ref = z_full.index_select(1, idx)
                tube_residual[tube.tube_id] = float(
                    (ref - z_skip).pow(2).mean().sqrt().item()
                )
            z_next.index_copy_(1, idx, z_skip.to(z_next.dtype))

        active_ratio = float(active_mask.float().mean().item())
        return TransitionResult(
            z_next=z_next,
            cache=out.cache,
            active_mask=active_mask,
            tube_residual=tube_residual,
            active_ratio=active_ratio,
            z_full=z_full,
        )

    def _fill_lowfreq(
        self, z_next: Tensor, z_full: Tensor, tube: SemanticTube, grid: TokenGrid
    ) -> None:
        """Reconstruct LOWFREQ tokens that were *not* computed by upsampling.

        The strided lattice was computed in ``z_full``; the holes are filled by
        nearest-neighbour copy from the kept lattice within the same frame (a cheap,
        artefact-free stand-in for true bilinear interpolation on the token grid).
        """
        stride = self.lowfreq_stride
        device = z_next.device
        for frame, idx in tube.tokens_by_frame.items():
            local = idx.to(device) - frame * grid.tokens_per_frame
            hi = torch.div(local, grid.w, rounding_mode="floor")
            wi = local - hi * grid.w
            kept = (hi % stride == 0) & (wi % stride == 0)
            if kept.all():
                continue
            # snap each hole to its kept lattice anchor (floor to multiple of stride)
            hi_a = (torch.div(hi, stride, rounding_mode="floor") * stride).clamp_max(grid.h - 1)
            wi_a = (torch.div(wi, stride, rounding_mode="floor") * stride).clamp_max(grid.w - 1)
            src_flat = frame * grid.tokens_per_frame + hi_a * grid.w + wi_a
            holes = ~kept
            z_next.index_copy_(
                1, idx[holes].to(device),
                z_full.index_select(1, src_flat[holes].to(device)),
            )
