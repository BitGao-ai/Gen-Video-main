"""Semantic-tube builder — orchestrates the STA pipeline (§4.3.1).

This is the single entry point the engine uses for tube construction/maintenance.
It wires the four STA sub-modules together while keeping each one ignorant of the
others:

    RegionExtractor → (RAFT flow, down-sampled to latent) → AffinityComputer
                    → TubeMatcher → TubeStateEncoder

It exposes two cost tiers (matching the inference loop's needs, §7.2 step 1):

    build()   full segmentation + matching from RGB frames — run once (or rarely)
    update()  cheap per-step state refresh on existing tubes (no re-segmentation)

so that the expensive SAM pass is amortised while the per-step ``s_{k,t}`` update
stays in the inner denoising loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from cocf.common.config import TubeConfig
from cocf.common.types import Region, SemanticTube, TokenGrid, TubeState
from cocf.tubes.affinity import AffinityComputer
from cocf.tubes.matching import TubeMatcher
from cocf.tubes.regions import PerceptionProvider, RegionExtractor
from cocf.tubes.state import TubeStateEncoder

Tensor = torch.Tensor


class TubeBuilder:
    """Builds and maintains the semantic-tube set ``G_t`` across denoising."""

    def __init__(self, config: TubeConfig, perception: PerceptionProvider) -> None:
        self.cfg = config
        self.perception = perception
        self.regions = RegionExtractor(config, perception)
        self.affinity = AffinityComputer(config.affinity)
        self.matcher = TubeMatcher(config)
        self.state = TubeStateEncoder(config)

    # ------------------------------------------------------------------ #
    # Full build (run once, e.g. on a preview decode or the GT video)
    # ------------------------------------------------------------------ #

    def build(
        self, frames_rgb: Tensor, grid: TokenGrid, prompt: str = ""
    ) -> List[SemanticTube]:
        """Construct tubes from RGB frames ``[F, 3, Hp, Wp]``.

        ``F`` should equal ``grid.t`` (one RGB frame per latent-temporal slot); the
        caller decodes the latent (or uses the GT video in training).
        """
        f = frames_rgb.shape[0]
        regions_by_frame: Dict[int, List[Region]] = {}
        for fi in range(f):
            regions_by_frame[fi] = self.regions.extract_frame(
                fi, frames_rgb[fi], grid, prompt
            )
        latent_flows = self._latent_flows(frames_rgb, grid)
        affinity_by_pair = {}
        for a, b in zip(range(f - 1), range(1, f)):
            affinity_by_pair[(a, b)] = self.affinity.matrix(
                regions_by_frame[a], regions_by_frame[b], latent_flows.get(a)
            )
        tubes = self.matcher.build_tubes(regions_by_frame, affinity_by_pair, grid)
        self.update(tubes, latent_flow_by_frame=latent_flows)
        return tubes

    # ------------------------------------------------------------------ #
    # Cheap per-step state update
    # ------------------------------------------------------------------ #

    def update(
        self,
        tubes: List[SemanticTube],
        latent_flow_by_frame: Optional[Dict[int, Tensor]] = None,
        causal_values: Optional[Dict[int, float]] = None,
    ) -> Dict[int, TubeState]:
        """Refresh ``s_{k,t}`` for all tubes (cheap; called every step)."""
        return self.state.encode_all(tubes, latent_flow_by_frame, causal_values)

    # ------------------------------------------------------------------ #
    # Flow helpers
    # ------------------------------------------------------------------ #

    def _latent_flows(self, frames_rgb: Tensor, grid: TokenGrid) -> Dict[int, Tensor]:
        """Compute pixel RAFT flow per consecutive pair, down-sampled to latent res."""
        flows: Dict[int, Tensor] = {}
        f = frames_rgb.shape[0]
        for a in range(f - 1):
            pix = self.perception.optical_flow(frames_rgb[a], frames_rgb[a + 1])  # [2,Hp,Wp]
            flows[a] = self._downsample_flow(pix, grid)
        return flows

    @staticmethod
    def _downsample_flow(pixel_flow: Tensor, grid: TokenGrid) -> Tensor:
        """``[2, Hp, Wp]`` pixel flow → ``[2, H_l, W_l]`` latent flow (vectors rescaled)."""
        _, hp, wp = pixel_flow.shape
        down = F.interpolate(
            pixel_flow[None], size=(grid.h, grid.w), mode="bilinear", align_corners=False
        )[0]
        down[0] *= grid.h / hp  # dy scaled to latent rows
        down[1] *= grid.w / wp  # dx scaled to latent cols
        return down
