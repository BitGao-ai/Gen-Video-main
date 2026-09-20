"""L-COCF lightweight counterfactual causal compute field."""

from __future__ import annotations

from cocf.lcocf.counterfactual import (
    CounterfactualVerifier,
    ResidualRepairNet,
    VerificationResult,
)
from cocf.lcocf.damage import (
    DAMAGE_DIMENSIONS,
    DEFAULT_DAMAGE_WEIGHTS,
    NUM_DAMAGE_DIMS,
    MetricExtractor,
    MultiDimDamageComputer,
    VideoFeatures,
)
from cocf.lcocf.mapping import ComputeFieldMapping
from cocf.lcocf.module import LCOCFModule
from cocf.lcocf.predictor import (
    DamagePredictor,
    build_predictor_input,
    predictor_input_dim,
    sinusoidal_embedding,
)
from cocf.lcocf.strength import (
    CausalStrengthField,
    CausalStrengthFeatureBuilder,
    StrengthFeatures,
)
from cocf.lcocf.triplets import (
    CausalParser,
    RuleBasedCausalParser,
    VLMCausalParser,
    build_parser,
    build_subgraph,
)

__all__ = [
    # facade
    "LCOCFModule",
    # sub-graph
    "CausalParser",
    "RuleBasedCausalParser",
    "VLMCausalParser",
    "build_parser",
    "build_subgraph",
    # strength
    "StrengthFeatures",
    "CausalStrengthFeatureBuilder",
    "CausalStrengthField",
    # mapping
    "ComputeFieldMapping",
    # predictor
    "DamagePredictor",
    "build_predictor_input",
    "predictor_input_dim",
    "sinusoidal_embedding",
    # counterfactual
    "CounterfactualVerifier",
    "ResidualRepairNet",
    "VerificationResult",
    # damage label
    "MultiDimDamageComputer",
    "MetricExtractor",
    "VideoFeatures",
    "DAMAGE_DIMENSIONS",
    "DEFAULT_DAMAGE_WEIGHTS",
    "NUM_DAMAGE_DIMS",
]
