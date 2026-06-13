"""Engine — the end-to-end accelerated inference loop (§7.2).

This package holds the one object that drives a *generation*: it threads a frozen
backbone through the four innovations on every denoising step, turning per-tube
causal-damage predictions into a budget- and risk-constrained compute plan and
executing it. Everything that evolves along the trajectory lives in an explicit
:class:`~cocf.engine.state.EngineState`; the engine itself is stateless across
calls, so it is trivially reusable and deterministic under a fixed seed.

    state    EngineState / GenerationResult / StepTrace — the trajectory containers
    engine   InferenceEngine — the §7.2 loop wiring STA → L-COCF → budget/allocator
             → action execution → RAEC certify/repair → single-hop CF check
"""

from __future__ import annotations

from cocf.engine.inference import InferenceEngine
from cocf.engine.state import EngineState, GenerationResult, StepTrace

__all__ = ["InferenceEngine", "EngineState", "GenerationResult", "StepTrace"]
