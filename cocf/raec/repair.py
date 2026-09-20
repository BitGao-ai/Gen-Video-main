"""Local repair operators for rollback and boundary fusion."""

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

    z: Tensor
    refreshed: Tensor
    rolled_back: bool


class BoundaryRepair:
    """Rollback plus boundary fusion operators."""

    def __init__(self, config: TriggerConfig) -> None:
        """Create repair with trigger config."""
        self.cfg = config

    def rollback(
        self,
        z_current: Tensor,
        z_full: Tensor,
        tube: SemanticTube,
        grid: TokenGrid,
        anchor_store: AnchorStore,
    ) -> RepairResult:
        """Restore tube to anchor and fuse boundary."""
        idx = tube.all_token_indices().to(z_current.device)
        if not anchor_store.has(tube.tube_id):
            return RepairResult(z=z_current, refreshed=idx, rolled_back=False)
        z = anchor_store.rollback(z_current, tube)
        z = self._fuse_boundary(z, z_full, tube, grid)
        return RepairResult(z=z, refreshed=idx, rolled_back=True)

    def repair(
        self,
        z_current: Tensor,
        z_full: Tensor,
        tube: SemanticTube,
        grid: TokenGrid,
    ) -> RepairResult:
        """Fix moderate-risk tube without full rollback."""
        idx = tube.all_token_indices().to(z_current.device)
        z = self._fuse_boundary(z_current, z_full, tube, grid)
        return RepairResult(z=z, refreshed=idx, rolled_back=False)

    def _fuse_boundary(
        self,
        z_interior: Tensor,
        z_exterior: Tensor,
        tube: SemanticTube,
        grid: TokenGrid,
    ) -> Tensor:
        """Blend interior and exterior latents over tube boundary."""
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
            w = 1.0 - torch.exp(-d_tok / sigma)
            w = w.view(1, -1, 1).to(out.device, out.dtype)
            gidx = idx.to(out.device)
            blended = w * z_interior.index_select(1, gidx) + (1 - w) * z_exterior.index_select(1, gidx)
            out.index_copy_(1, gidx, blended.to(out.dtype))
        return out

    @staticmethod
    def _erosion_depth(mask: Tensor, max_depth: int) -> Tensor:
        """Compute per-pixel depth into boolean mask."""
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        cur = mask.bool()
        depth = cur.to(torch.int32)
        for _ in range(max_depth - 1):
            up = torch.zeros_like(cur); up[:-1] = cur[1:]
            dn = torch.zeros_like(cur); dn[1:] = cur[:-1]
            lf = torch.zeros_like(cur); lf[:, :-1] = cur[:, 1:]
            rt = torch.zeros_like(cur); rt[:, 1:] = cur[:, :-1]
            cur = cur & up & dn & lf & rt
            depth = depth + cur.to(torch.int32)
            if not cur.any():
                break
        return depth
