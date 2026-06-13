"""STA — Semantic-Tube Anchoring subsystem (§4).

Upgrades the unit of compute allocation from isolated tokens to cross-frame
*semantic tubes*, which (provably, §4.4) lowers decision variance and removes the
flicker/tearing/identity-drift failure modes of token-level skipping.

Public surface:

    TubeBuilder        orchestrates region→affinity→matching→state (entry point)
    SemanticTube       the tube object (re-exported from common.types)
    PerceptionProvider the injected SAM/CLIP/DINOv2/RAFT backend contract
    TubeSmoothingLoss  the §4.3.2 action-consistency regulariser
"""

from __future__ import annotations

from cocf.common.types import SemanticTube, TubeState
from cocf.tubes.affinity import AffinityComputer
from cocf.tubes.builder import TubeBuilder
from cocf.tubes.matching import TubeMatcher, solve_assignment
from cocf.tubes.regions import PerceptionProvider, RegionExtractor
from cocf.tubes.smoothing import TubeSmoothingLoss
from cocf.tubes.state import TubeStateEncoder

__all__ = [
    "TubeBuilder",
    "SemanticTube",
    "TubeState",
    "PerceptionProvider",
    "RegionExtractor",
    "AffinityComputer",
    "TubeMatcher",
    "solve_assignment",
    "TubeStateEncoder",
    "TubeSmoothingLoss",
]
