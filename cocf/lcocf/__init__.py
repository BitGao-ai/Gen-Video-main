"""L-COCF — Lightweight Counterfactual Causal Compute Field (§3).

Turns the theoretically-complete-but-infeasible native COCF into an engineering-
realisable component through four scientifically-grounded simplifications, each in
its own module:

    triplets        §3.3.1  local causal sub-graph from a frozen VLM (no NP-hard
                            global graph learning)
    strength        §3.3.2  first-order linear causal-strength field s=αs_E+βs_A+γs_T
                            (3 learnable scalars)
    mapping         §3.3.3  hierarchical discrete strength→tier→action map
                            (no continuous-field iterative solve)
    counterfactual  §3.3.4  local single-hop CF check at temporal mutation points
                            (no full-domain multi-hop reasoning)
    predictor       §3.3    the damage predictor H_φ regressing final-video damage
    damage          §7.1.1  the multi-dimensional damage *label* definition
    module          §3      the LCOCFModule facade wiring all of the above

The public surface is re-exported here so callers import from ``cocf.lcocf``.
"""

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
    # §3.3.1 sub-graph
    "CausalParser",
    "RuleBasedCausalParser",
    "VLMCausalParser",
    "build_parser",
    "build_subgraph",
    # §3.3.2 strength
    "StrengthFeatures",
    "CausalStrengthFeatureBuilder",
    "CausalStrengthField",
    # §3.3.3 mapping
    "ComputeFieldMapping",
    # §3.3 predictor
    "DamagePredictor",
    "build_predictor_input",
    "predictor_input_dim",
    "sinusoidal_embedding",
    # §3.3.4 counterfactual
    "CounterfactualVerifier",
    "ResidualRepairNet",
    "VerificationResult",
    # §7.1.1 damage label
    "MultiDimDamageComputer",
    "MetricExtractor",
    "VideoFeatures",
    "DAMAGE_DIMENSIONS",
    "DEFAULT_DAMAGE_WEIGHTS",
    "NUM_DAMAGE_DIMS",
]
