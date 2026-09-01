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
from typing import List, Optional, Sequence, Tuple

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

    def text_feature(self, prompt: str) -> Optional[Tensor]:
        """CLIP text embedding ``[d_clip]`` of the prompt, or ``None`` if unsupported.

        Exposed so the semantic filter can score a whole frame's regions with one dot
        product against the features :meth:`region_features` already computed, instead
        of one image encode per region on top of them. A provider without a text tower
        returns ``None`` and the filter falls back to :meth:`clip_score`.
        """
        return None

    @abc.abstractmethod
    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """RAFT flow ``[2, Hp, Wp]`` mapping ``frame_a`` pixels to ``frame_b``.

        **Channel order is ``(dy, dx)``** — vertical displacement first — which is what
        every consumer in the framework assumes: :meth:`TubeBuilder._downsample_flow`
        rescales channel 0 by ``H`` and channel 1 by ``W``, and the mask/centroid warps
        in :mod:`cocf.tubes.state` and :mod:`cocf.tubes.affinity` add channel 0 to the
        row index. Note this is the *opposite* of torchvision's RAFT, which returns
        ``(dx, dy)``; a provider wrapping it must transpose the channels (see
        :func:`cocf.tubes.model_perception.raft_to_framework_flow`) or every occlusion,
        motion-phase and affinity term is computed from a displacement pointing the
        wrong way — invisibly so on square inputs with isotropic motion (§P1-11).
        """

    def region_features(
        self, frame: Tensor, masks: Sequence[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """``(identity, clip)`` features for several regions of **one** frame.

        Concrete default: call the per-region methods, so an existing provider needs
        no change. A provider backed by real checkpoints should override it with a
        single batched forward — the per-region contract costs one processor
        round-trip and one batch-of-1 forward per mask, i.e. 24 masks × 13 frames =
        312 serial DINOv2 forwards plus 312 CLIP forwards for one clip (§P2-8).
        """
        return (
            [self.identity_feature(frame, m) for m in masks],
            [self.clip_feature(frame, m) for m in masks],
        )


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
        # Areas for every mask in one reduction: the per-mask ``.item()`` this replaces
        # was a device sync per region, ~300 per clip before any model ran.
        areas = masks.flatten(1).float().mean(dim=1).tolist()
        kept: List[tuple] = []
        for r, area_ratio in enumerate(areas):
            if area_ratio < self.cfg.min_region_ratio:  # drop tiny regions
                continue
            lat_mask = self._to_latent_mask(masks[r], grid)  # [H_l, W_l] bool
            tok = self._mask_to_tokens(lat_mask, frame_idx, grid)
            if tok.numel() == 0:
                continue
            kept.append((masks[r], lat_mask, tok))

        # Features for all surviving regions in one call, so a real backend can run a
        # single batched forward per frame instead of one per region (§P2-8).
        ident, textf = self.perception.region_features(
            frame_rgb, [k[0] for k in kept]
        )
        scores = self._clip_scores(frame_rgb, [k[0] for k in kept], textf, prompt)
        # The semantic filter (§4.3.1) is applied here rather than before the feature
        # pass: scoring needs the very CLIP embedding that pass produces, and computing
        # it twice — once to filter, once to keep — was the frame's dominant non-SAM
        # cost. ``region_id`` numbers the survivors, which is what the affinity
        # matrices index.
        regions: List[Region] = []
        for i, (_, lat_mask, tok) in enumerate(kept):
            if scores[i] < self.cfg.min_clip_score:  # drop low-semantic regions
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
        """Region-vs-prompt CLIP match in ``[0,1]`` for a whole frame.

        Uses one text encode plus a dot product against the already-computed region
        embeddings when the provider exposes :meth:`PerceptionProvider.text_feature`;
        otherwise falls back to the per-region :meth:`clip_score`.
        """
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
