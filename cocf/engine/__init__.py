"""Engine — the end-to-end accelerated inference loop.

Holds the object that drives a generation: it threads a frozen backbone through the
four innovations on every denoising step, turning per-tube causal-damage predictions
into a budget- and risk-constrained compute plan and executing it. Trajectory state
lives in an explicit :class:`~cocf.engine.state.EngineState`; the engine itself is
stateless across calls, so it is reusable and deterministic under a fixed seed.

    state    EngineState / GenerationResult / StepTrace — the trajectory containers
    engine   InferenceEngine — the loop wiring STA, L-COCF, budget/allocator, action
             execution, RAEC certify/repair and the single-hop CF check
"""

from __future__ import annotations

from cocf.engine.inference import InferenceEngine
from cocf.engine.state import EngineState, GenerationResult, StepTrace

__all__ = ["InferenceEngine", "EngineState", "GenerationResult", "StepTrace"]
