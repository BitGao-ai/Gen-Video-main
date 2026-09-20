"""Cross-frame region affinity for tube linking."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from cocf.common.config import AffinityConfig
from cocf.common.types import Region

Tensor = torch.Tensor


class AffinityComputer:
    """Computes region affinity matrix between two frames."""

    def __init__(self, config: AffinityConfig) -> None:
        """Store affinity config."""
        self.cfg = config

    def matrix(
        self,
        regions_a: List[Region],
        regions_b: List[Region],
        latent_flow: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute affinity matrix in [0, 1]."""
        ra, rb = len(regions_a), len(regions_b)
        if ra == 0 or rb == 0:
            return torch.zeros(ra, rb)

        c = self.cfg
        device = regions_a[0].mask.device
        id_a = self._stack_feats(regions_a, "identity_feat")
        id_b = self._stack_feats(regions_b, "identity_feat")
        txt_a = self._stack_feats(regions_a, "text_feat")
        txt_b = self._stack_feats(regions_b, "text_feat")
        cen_a = torch.tensor([r.center for r in regions_a], device=device)
        cen_b = torch.tensor([r.center for r in regions_b], device=device)

        id_sim = self._cosine_matrix(id_a, id_b, device)
        txt_sim = self._cosine_matrix(txt_a, txt_b, device)

        warped = self._warp_centroids(cen_a, latent_flow)
        dist = torch.cdist(warped, cen_b)
        flow_sim = torch.exp(-dist / max(c.flow_scale, 1e-6))
        pos_dist = torch.cdist(cen_a, cen_b)
        pos_sim = torch.exp(-(pos_dist ** 2) / (2 * c.sigma_p ** 2))

        iou = self._warped_iou(regions_a, regions_b, latent_flow)

        aff = (
            c.w_id * id_sim
            + c.w_flow * flow_sim
            + c.w_iou * iou
            + c.w_txt * txt_sim
            + c.w_pos * pos_sim
        )
        return aff.clamp(0.0, 1.0)

    @staticmethod
    def _stack_feats(regions: List[Region], attr: str) -> Optional[Tensor]:
        """Stack region features or None."""
        feats = [getattr(r, attr) for r in regions]
        if any(f is None for f in feats):
            return None
        return torch.stack(feats)

    @staticmethod
    def _cosine_matrix(a: Optional[Tensor], b: Optional[Tensor], device=None) -> Tensor:
        """Cosine similarity matrix mapped to [0, 1]."""
        if a is None or b is None:
            return torch.full((1, 1), 0.5, device=device)
        a = F.normalize(a.float(), dim=-1)
        b = F.normalize(b.float(), dim=-1)
        return ((a @ b.T) + 1.0) * 0.5

    def _warp_centroids(self, centroids: Tensor, latent_flow: Optional[Tensor]) -> Tensor:
        """Warp centroids by latent flow."""
        if latent_flow is None:
            return centroids
        h, w = latent_flow.shape[-2:]
        out = centroids.clone()
        for i, (cy, cx) in enumerate(centroids):
            yi = int(min(max(cy, 0), h - 1))
            xi = int(min(max(cx, 0), w - 1))
            dy, dx = latent_flow[0, yi, xi], latent_flow[1, yi, xi]
            out[i, 0] = cy + dy
            out[i, 1] = cx + dx
        return out

    def _warped_iou(
        self, regions_a: List[Region], regions_b: List[Region], latent_flow: Optional[Tensor]
    ) -> Tensor:
        """Warped mask IoU matrix."""
        iou = torch.zeros(len(regions_a), len(regions_b), device=regions_a[0].mask.device)
        warped_masks = [self._warp_mask(r.mask, latent_flow) for r in regions_a]
        for i, wma in enumerate(warped_masks):
            for j, rb in enumerate(regions_b):
                inter = (wma & rb.mask).sum().float()
                union = (wma | rb.mask).sum().float().clamp_min(1.0)
                iou[i, j] = inter / union
        return iou

    @staticmethod
    def _warp_mask(mask: Tensor, latent_flow: Optional[Tensor]) -> Tensor:
        """Forward-warp mask by integer-rounded flow."""
        if latent_flow is None:
            return mask
        h, w = mask.shape
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            return mask
        dy = latent_flow[0, ys, xs].round().long()
        dx = latent_flow[1, ys, xs].round().long()
        ny = (ys + dy).clamp(0, h - 1)
        nx = (xs + dx).clamp(0, w - 1)
        out = torch.zeros_like(mask)
        out[ny, nx] = True
        return out
