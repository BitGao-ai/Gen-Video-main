"""Frame-level region extraction — the first stage of tube construction (§4.3.1).

Pipeline per frame:

    SAM masks  →  filter tiny (<0.1% pixels) & low-semantic (CLIP<0.2) regions
               →  down-sample masks to *latent* resolution (H_l × W_l)
               →  attach DINOv2 identity features + CLIP text-alignment features
               →  :class:`~cocf.common.types.Region`

All heavy perception models (SAM, CLIP, DINOv2, RAFT) sit behind the
:class:`PerceptionProvider` protocol so this module — and every STA module — is
**decoupled** from any particular checkpoint and is unit-testable with the
:class:`MockPerception` stand-in. Swapping SAM-2 for FastSAM, or CLIP for SigLIP,
is a provider change with zero edits to the algorithm.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from cocf.common.config import TubeConfig
from cocf.common.types import Region, TokenGrid

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Perception provider (dependency-injected, backbone/checkpoint agnostic)
# --------------------------------------------------------------------------- #


class PerceptionProvider(abc.ABC):
    """Abstract perception backend used by the whole STA subsystem.

    Implementations wrap concrete models; the algorithm depends only on this
    contract. All spatial outputs are at *pixel* resolution unless noted; the
    region extractor down-samples them to latent resolution.
    """

    @abc.abstractmethod
    def segment(self, frame: Tensor) -> Tensor:
        """RGB frame ``[3, Hp, Wp]`` → instance masks ``[R, Hp, Wp]`` (bool/float)."""

    @abc.abstractmethod
    def identity_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """Pooled DINOv2 identity embedding ``[d_id]`` of the masked region crop."""

    @abc.abstractmethod
    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        """CLIP image-text match in ``[0, 1]`` for region vs prompt (semantic filter)."""

    @abc.abstractmethod
    def clip_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """CLIP visual embedding ``[d_clip]`` of the region (for text-tube alignment)."""

    @abc.abstractmethod
    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """RAFT flow ``[2, Hp, Wp]`` mapping ``frame_a`` pixels to ``frame_b``."""


# --------------------------------------------------------------------------- #
# Region extraction
# --------------------------------------------------------------------------- #


class RegionExtractor:
    """Turns RGB frames into filtered, latent-resolution :class:`Region` objects."""

    def __init__(self, config: TubeConfig, perception: PerceptionProvider) -> None:
        self.cfg = config
        self.perception = perception

    def extract_frame(
        self, frame_idx: int, frame_rgb: Tensor, grid: TokenGrid, prompt: str = ""
    ) -> List[Region]:
        """Extract regions for one RGB frame ``[3, Hp, Wp]`` at latent ``grid`` res."""
        masks = self.perception.segment(frame_rgb)  # [R, Hp, Wp]
        if masks.numel() == 0:
            return []
        total_px = float(masks.shape[-1] * masks.shape[-2])
        regions: List[Region] = []
        rid = 0
        for r in range(masks.shape[0]):
            mask_px = masks[r]
            area_ratio = float(mask_px.float().mean().item())
            if area_ratio < self.cfg.min_region_ratio:  # drop tiny regions
                continue
            score = self.perception.clip_score(frame_rgb, mask_px, prompt) if prompt else 1.0
            if score < self.cfg.min_clip_score:  # drop low-semantic regions
                continue
            lat_mask = self._to_latent_mask(mask_px, grid)  # [H_l, W_l] bool
            tok = self._mask_to_tokens(lat_mask, frame_idx, grid)
            if tok.numel() == 0:
                continue
            regions.append(
                Region(
                    frame=frame_idx,
                    region_id=rid,
                    mask=lat_mask,
                    token_indices=tok,
                    identity_feat=self.perception.identity_feature(frame_rgb, mask_px),
                    text_feat=self.perception.clip_feature(frame_rgb, mask_px),
                    center=self._centroid(lat_mask),
                    clip_score=score,
                )
            )
            rid += 1
        return regions

    # -- helpers --------------------------------------------------------- #

    @staticmethod
    def _to_latent_mask(mask_px: Tensor, grid: TokenGrid) -> Tensor:
        """Down-sample a pixel mask to latent resolution by area-average + threshold."""
        m = mask_px.float()[None, None]  # [1,1,Hp,Wp]
        down = F.interpolate(m, size=(grid.h, grid.w), mode="area")[0, 0]
        return down > 0.5

    @staticmethod
    def _mask_to_tokens(lat_mask: Tensor, frame_idx: int, grid: TokenGrid) -> Tensor:
        """Flat token indices (into the full ``[N]`` axis) for a latent-frame mask."""
        flat = lat_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        return flat + frame_idx * grid.tokens_per_frame

    @staticmethod
    def _centroid(lat_mask: Tensor) -> tuple:
        ys, xs = torch.nonzero(lat_mask, as_tuple=True)
        if ys.numel() == 0:
            return (0.0, 0.0)
        return (float(ys.float().mean()), float(xs.float().mean()))
