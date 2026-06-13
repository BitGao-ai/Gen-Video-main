"""Cross-frame region affinity ``Aff(i, j)`` for tube linking (§4.3.1).

Given regions on frame ``a`` and frame ``b`` (typically consecutive), produces an
affinity matrix that the matcher (:mod:`cocf.tubes.matching`) feeds to the
Hungarian algorithm. The score fuses five complementary cues, weighted by
:class:`~cocf.common.config.AffinityConfig`:

    Aff = w_id·cos(id_i, id_j)              identity (DINOv2)  — robust to motion
        + w_flow·exp(−‖warp(c_i)−c_j‖/s)    flow agreement     — motion-consistent
        + w_iou·IoU(warp(M_i), M_j)         warped overlap     — shape/extent
        + w_txt·cos(txt_i, txt_j)           text alignment     — semantic identity
        + w_pos·exp(−‖c_i−c_j‖²/2σ²)        proximity          — spatial prior

Everything operates at **latent resolution** so it is cheap and aligns with the
token grid; the builder is responsible for down-sampling the pixel-space RAFT
flow to latent coordinates before calling here. The module is pure tensor maths
with no model dependency, so it is fully deterministic and testable.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from cocf.common.config import AffinityConfig
from cocf.common.types import Region

Tensor = torch.Tensor


class AffinityComputer:
    """Computes the region-to-region affinity matrix between two frames."""

    def __init__(self, config: AffinityConfig) -> None:
        self.cfg = config

    def matrix(
        self,
        regions_a: List[Region],
        regions_b: List[Region],
        latent_flow: Optional[Tensor] = None,
    ) -> Tensor:
        """Affinity matrix ``[R_a, R_b]`` in ``[0, 1]``.

        ``latent_flow`` is ``[2, H_l, W_l]`` (dy, dx) mapping frame ``a`` → ``b`` at
        latent resolution; if ``None`` the flow term degrades gracefully to the
        positional prior.
        """
        ra, rb = len(regions_a), len(regions_b)
        if ra == 0 or rb == 0:
            return torch.zeros(ra, rb)

        c = self.cfg
        id_a = self._stack_feats(regions_a, "identity_feat")
        id_b = self._stack_feats(regions_b, "identity_feat")
        txt_a = self._stack_feats(regions_a, "text_feat")
        txt_b = self._stack_feats(regions_b, "text_feat")
        cen_a = torch.tensor([r.center for r in regions_a])  # [R_a, 2]
        cen_b = torch.tensor([r.center for r in regions_b])  # [R_b, 2]

        id_sim = self._cosine_matrix(id_a, id_b)  # [R_a, R_b]
        txt_sim = self._cosine_matrix(txt_a, txt_b)

        # flow-warped centroids of A, then distance to each centroid of B
        warped = self._warp_centroids(cen_a, latent_flow)  # [R_a, 2]
        dist = torch.cdist(warped, cen_b)  # [R_a, R_b]
        flow_sim = torch.exp(-dist / max(c.flow_scale, 1e-6))
        pos_dist = torch.cdist(cen_a, cen_b)
        pos_sim = torch.exp(-(pos_dist ** 2) / (2 * c.sigma_p ** 2))

        iou = self._warped_iou(regions_a, regions_b, latent_flow)  # [R_a, R_b]

        aff = (
            c.w_id * id_sim
            + c.w_flow * flow_sim
            + c.w_iou * iou
            + c.w_txt * txt_sim
            + c.w_pos * pos_sim
        )
        return aff.clamp(0.0, 1.0)

    # -- pieces ---------------------------------------------------------- #

    @staticmethod
    def _stack_feats(regions: List[Region], attr: str) -> Optional[Tensor]:
        feats = [getattr(r, attr) for r in regions]
        if any(f is None for f in feats):
            return None
        return torch.stack(feats)

    @staticmethod
    def _cosine_matrix(a: Optional[Tensor], b: Optional[Tensor]) -> Tensor:
        if a is None or b is None:
            # neutral 0.5 when a feature is unavailable (keeps the term unbiased)
            return torch.full((1, 1), 0.5)
        a = F.normalize(a.float(), dim=-1)
        b = F.normalize(b.float(), dim=-1)
        return ((a @ b.T) + 1.0) * 0.5  # map cos∈[-1,1] → [0,1]

    def _warp_centroids(self, centroids: Tensor, latent_flow: Optional[Tensor]) -> Tensor:
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
        iou = torch.zeros(len(regions_a), len(regions_b))
        warped_masks = [self._warp_mask(r.mask, latent_flow) for r in regions_a]
        for i, wma in enumerate(warped_masks):
            for j, rb in enumerate(regions_b):
                inter = (wma & rb.mask).sum().float()
                union = (wma | rb.mask).sum().float().clamp_min(1.0)
                iou[i, j] = inter / union
        return iou

    @staticmethod
    def _warp_mask(mask: Tensor, latent_flow: Optional[Tensor]) -> Tensor:
        """Forward-warp a latent mask by integer-rounded flow (cheap, robust)."""
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
