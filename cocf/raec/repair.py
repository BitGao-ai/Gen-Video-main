"""Local repair operators — rollback, boundary fusion, cache refresh (§5.3.2).

When the risk trigger fires, the fix must be *local* and *seam-free*: rolling a
tube back to its safe anchor leaves a discontinuity against the neighbouring
tokens that advanced normally, and a hard seam is itself an artefact. The repair
operators here resolve that:

    rollback        restore a tube's tokens from the anchor store (the revocation)
    boundary_fuse   blend the tube's *boundary band* between the rolled-back latent
                    (interior, frozen-safe) and the freshly computed latent
                    (exterior-consistent) using a soft mask of bandwidth σ_bnd, so
                    the seam vanishes (the boundary operator B(·) of §5.3.2)
    refreshed_tokens the indices whose ε cache is now stale and must be recomputed

Everything is pure latent/token-grid arithmetic against the backbone-agnostic
:class:`~cocf.common.types.TokenGrid`, so RAEC works for every backbone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from cocf.common.config import TriggerConfig
from cocf.common.types import SemanticTube, TokenGrid
from cocf.raec.anchor_store import AnchorStore

Tensor = torch.Tensor


@dataclass
class RepairResult:
    """Outcome of repairing one tube."""

    z: Tensor                 # [B, N, d] latent after repair
    refreshed: Tensor         # [n] flat token indices whose ε cache is now stale
    rolled_back: bool         # whether a full rollback to anchor happened


class BoundaryRepair:
    """Rollback + boundary fusion + cache-refresh accounting."""

    def __init__(self, config: TriggerConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------ #
    # ROLLBACK (+ boundary fusion against the advanced latent)
    # ------------------------------------------------------------------ #

    def rollback(
        self,
        z_full: Tensor,
        tube: SemanticTube,
        grid: TokenGrid,
        anchor_store: AnchorStore,
    ) -> RepairResult:
        """Restore ``tube`` to its safe anchor and fuse the boundary against ``z_full``.

        ``z_full`` is the *advanced* (compute-everywhere) latent for this step,
        used as the exterior-consistent reference at the tube boundary. If the tube
        has no anchor yet, returns ``z_full`` unchanged but still flags the tube's
        tokens as refreshed (the engine will force FULL on it).
        """
        idx = tube.all_token_indices().to(z_full.device)
        if not anchor_store.has(tube.tube_id):
            return RepairResult(z=z_full, refreshed=idx, rolled_back=False)
        z = anchor_store.rollback(z_full, tube)            # interior = safe anchor
        z = self._fuse_boundary(z, z_full, tube, grid)     # smooth the seam
        return RepairResult(z=z, refreshed=idx, rolled_back=True)

    # ------------------------------------------------------------------ #
    # REPAIR (no rollback): just refresh the cache + light boundary fuse
    # ------------------------------------------------------------------ #

    def repair(
        self,
        z_current: Tensor,
        z_full: Tensor,
        tube: SemanticTube,
        grid: TokenGrid,
    ) -> RepairResult:
        """Moderate-risk fix: pull the tube toward the freshly computed ``z_full``
        at the boundary and mark it for cache refresh, without a full rollback."""
        idx = tube.all_token_indices().to(z_current.device)
        z = self._fuse_boundary(z_current, z_full, tube, grid, toward_interior=False)
        return RepairResult(z=z, refreshed=idx, rolled_back=False)

    # ------------------------------------------------------------------ #
    # boundary soft-mask fusion
    # ------------------------------------------------------------------ #

    def _fuse_boundary(
        self,
        z_interior: Tensor,
        z_exterior: Tensor,
        tube: SemanticTube,
        grid: TokenGrid,
        toward_interior: bool = True,
    ) -> Tensor:
        """Blend ``z_interior`` (e.g. anchor) and ``z_exterior`` (e.g. z_full) over a
        tube's tokens using a per-token weight derived from depth-into-the-tube.

        ``w = 1 − exp(−depth / σ_bnd)`` → ~0 at the edge (favour the exterior,
        neighbour-consistent latent) and →1 deep inside (favour the interior,
        safe latent). With ``toward_interior=False`` the roles invert (used by the
        lighter REPAIR path, which keeps the current latent inside and only
        reconciles the rim with z_full).
        """
        sigma = max(self.cfg.sigma_bnd, 1e-3)
        out = z_interior.clone()
        for frame, idx in tube.tokens_by_frame.items():
            mask = tube.masks_by_frame.get(frame)
            if mask is None or idx.numel() == 0:
                continue
            depth = self._erosion_depth(mask, max_depth=int(math.ceil(3 * sigma)))
            local = idx.to(mask.device) - frame * grid.tokens_per_frame
            hi = torch.div(local, grid.w, rounding_mode="floor")
            wi = local - hi * grid.w
            d_tok = depth[hi.clamp(0, grid.h - 1), wi.clamp(0, grid.w - 1)].float()
            w = 1.0 - torch.exp(-d_tok / sigma)            # [n_tok] ∈ [0,1)
            if not toward_interior:
                w = 1.0 - w
            w = w.view(1, -1, 1).to(out.device, out.dtype)
            gidx = idx.to(out.device)
            blended = w * z_interior.index_select(1, gidx) + (1 - w) * z_exterior.index_select(1, gidx)
            out.index_copy_(1, gidx, blended.to(out.dtype))
        return out

    @staticmethod
    def _erosion_depth(mask: Tensor, max_depth: int) -> Tensor:
        """Per-pixel depth into a boolean mask ``[H, W]`` via iterated 4-neighbour erosion.

        ``depth = 0`` outside; ``1`` on the boundary; increasing toward the interior
        (capped at ``max_depth``). Vectorised: each iteration keeps only pixels whose
        4 neighbours are all still set.
        """
        cur = mask.bool()
        depth = cur.to(torch.int32)
        for _ in range(max(1, max_depth)):
            up = torch.zeros_like(cur); up[:-1] = cur[1:]
            dn = torch.zeros_like(cur); dn[1:] = cur[:-1]
            lf = torch.zeros_like(cur); lf[:, :-1] = cur[:, 1:]
            rt = torch.zeros_like(cur); rt[:, 1:] = cur[:, :-1]
            cur = cur & up & dn & lf & rt   # survives erosion
            depth = depth + cur.to(torch.int32)
            if not cur.any():
                break
        return depth
