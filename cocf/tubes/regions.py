"""Frame-level region extraction for tube construction."""

from __future__ import annotations

import abc
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from cocf.common.config import TubeConfig
from cocf.common.types import Region, TokenGrid

Tensor = torch.Tensor


class PerceptionProvider(abc.ABC):
    """Abstract perception backend for STA."""

    @abc.abstractmethod
    def segment(self, frame: Tensor) -> Tensor:
        """Segment frame into instance masks."""

    @abc.abstractmethod
    def identity_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """Identity embedding of masked region."""

    @abc.abstractmethod
    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        """CLIP match score for region vs prompt."""

    @abc.abstractmethod
    def clip_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """CLIP visual embedding of region."""

    def text_feature(self, prompt: str) -> Optional[Tensor]:
        """CLIP text embedding of prompt or None."""
        return None

    @abc.abstractmethod
    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """Optical flow from frame_a to frame_b."""

    def region_features(
        self, frame: Tensor, masks: Sequence[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Identity and CLIP features for frame regions."""
        return (
            [self.identity_feature(frame, m) for m in masks],
            [self.clip_feature(frame, m) for m in masks],
        )


class RegionExtractor:
    """Extracts filtered latent-resolution regions."""

    def __init__(self, config: TubeConfig, perception: PerceptionProvider) -> None:
        """Store config and perception backend."""
        self.cfg = config
        self.perception = perception

    def extract_frame(
        self, frame_idx: int, frame_rgb: Tensor, grid: TokenGrid, prompt: str = ""
    ) -> List[Region]:
        """Extract regions for one frame."""
        masks = self.perception.segment(frame_rgb)
        if masks.numel() == 0:
            return []
        areas = masks.flatten(1).float().mean(dim=1).tolist()
        kept: List[tuple] = []
        for r, area_ratio in enumerate(areas):
            if area_ratio < self.cfg.min_region_ratio:
                continue
            lat_mask = self._to_latent_mask(masks[r], grid)
            tok = self._mask_to_tokens(lat_mask, frame_idx, grid)
            if tok.numel() == 0:
                continue
            kept.append((masks[r], lat_mask, tok))

        ident, textf = self.perception.region_features(
            frame_rgb, [k[0] for k in kept]
        )
        scores = self._clip_scores(frame_rgb, [k[0] for k in kept], textf, prompt)
        regions: List[Region] = []
        for i, (_, lat_mask, tok) in enumerate(kept):
            if scores[i] < self.cfg.min_clip_score:
                continue
            regions.append(Region(
                frame=frame_idx,
                region_id=len(regions),
                mask=lat_mask,
                token_indices=tok,
                identity_feat=ident[i],
                text_feat=textf[i],
                center=self._centroid(lat_mask),
                clip_score=scores[i],
            ))
        return regions

    def _clip_scores(
        self, frame_rgb: Tensor, masks: Sequence[Tensor],
        text_feats: Sequence[Tensor], prompt: str,
    ) -> List[float]:
        """CLIP match scores for frame regions."""
        if not prompt or not masks:
            return [1.0] * len(masks)
        txt = self.perception.text_feature(prompt)
        if txt is None:
            return [
                self.perception.clip_score(frame_rgb, m, prompt) for m in masks
            ]
        v = F.normalize(torch.stack([f.float() for f in text_feats]), dim=-1)
        t = F.normalize(txt.float().to(v.device), dim=-1)
        return ((v @ t).clamp(-1.0, 1.0) * 0.5 + 0.5).tolist()

    @staticmethod
    def _to_latent_mask(mask_px: Tensor, grid: TokenGrid) -> Tensor:
        """Down-sample pixel mask to latent resolution."""
        m = mask_px.float()[None, None]
        down = F.interpolate(m, size=(grid.h, grid.w), mode="area")[0, 0]
        return down > 0.5

    @staticmethod
    def _mask_to_tokens(lat_mask: Tensor, frame_idx: int, grid: TokenGrid) -> Tensor:
        """Flat token indices for latent mask."""
        flat = lat_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        return flat + frame_idx * grid.tokens_per_frame

    @staticmethod
    def _centroid(lat_mask: Tensor) -> tuple:
        """Mask centroid coordinates."""
        ys, xs = torch.nonzero(lat_mask, as_tuple=True)
        if ys.numel() == 0:
            return (0.0, 0.0)
        return (float(ys.float().mean()), float(xs.float().mean()))
