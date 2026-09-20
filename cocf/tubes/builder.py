"""Semantic-tube builder orchestrating the STA pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    """Builds and maintains semantic tubes."""

    def __init__(self, config: TubeConfig, perception: PerceptionProvider) -> None:
        """Store config and wire STA sub-modules."""
        self.cfg = config
        self.perception = perception
        self.regions = RegionExtractor(config, perception)
        self.affinity = AffinityComputer(config.affinity)
        self.matcher = TubeMatcher(config)
        self.state = TubeStateEncoder(config)

    def build(
        self, frames_rgb: Tensor, grid: TokenGrid, prompt: str = ""
    ) -> List[SemanticTube]:
        """Build tubes from RGB frames."""
        return self.build_with_states(frames_rgb, grid, prompt)[0]

    def build_with_states(
        self, frames_rgb: Tensor, grid: TokenGrid, prompt: str = ""
    ) -> Tuple[List[SemanticTube], Dict[int, TubeState], Dict[int, Tensor]]:
        """Build tubes and return states and latent flows."""
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
        tubes = self.matcher.build_tubes(
            regions_by_frame, affinity_by_pair, grid,
            affinity_fn=lambda fa, fb: (
                self.affinity.matrix(
                    regions_by_frame[fa], regions_by_frame[fb], latent_flows.get(fa)
                )
                if fa in regions_by_frame and fb in regions_by_frame else None
            ),
        )
        states = self.update(tubes, latent_flow_by_frame=latent_flows)
        return tubes, states, latent_flows

    def update(
        self,
        tubes: List[SemanticTube],
        latent_flow_by_frame: Optional[Dict[int, Tensor]] = None,
        causal_values: Optional[Dict[int, float]] = None,
    ) -> Dict[int, TubeState]:
        """Refresh tube states for one step."""
        return self.state.encode_all(tubes, latent_flow_by_frame, causal_values)

    def _latent_flows(self, frames_rgb: Tensor, grid: TokenGrid) -> Dict[int, Tensor]:
        """Compute latent-resolution flow per frame pair."""
        flows: Dict[int, Tensor] = {}
        f = frames_rgb.shape[0]
        for a in range(f - 1):
            pix = self.perception.optical_flow(frames_rgb[a], frames_rgb[a + 1])
            flows[a] = self._downsample_flow(pix, grid)
        return flows

    @staticmethod
    def _downsample_flow(pixel_flow: Tensor, grid: TokenGrid) -> Tensor:
        """Down-sample pixel flow to latent resolution."""
        _, hp, wp = pixel_flow.shape
        down = F.interpolate(
            pixel_flow[None], size=(grid.h, grid.w), mode="bilinear", align_corners=False
        )[0]
        down[0] *= grid.h / hp
        down[1] *= grid.w / wp
        return down
