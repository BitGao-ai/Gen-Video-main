"""STA semantic-tube anchoring subsystem."""

from __future__ import annotations

from cocf.common.types import SemanticTube, TubeState
from cocf.tubes.affinity import AffinityComputer
from cocf.tubes.builder import TubeBuilder
from cocf.tubes.matching import TubeMatcher, solve_assignment
from cocf.tubes.mock_perception import MockPerception
from cocf.tubes.model_perception import ModelPerception
from cocf.tubes.regions import PerceptionProvider, RegionExtractor
from cocf.tubes.smoothing import TubeSmoothingLoss
from cocf.tubes.state import TubeStateEncoder

__all__ = [
    "TubeBuilder",
    "SemanticTube",
    "TubeState",
    "PerceptionProvider",
    "MockPerception",
    "ModelPerception",
    "RegionExtractor",
    "AffinityComputer",
    "TubeMatcher",
    "solve_assignment",
    "TubeStateEncoder",
    "TubeSmoothingLoss",
]
