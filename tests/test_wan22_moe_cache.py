"""Wan2.2 MoE cache-ownership: a cached velocity belongs to the expert that
produced it and must never be spliced into the other expert's noise regime."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cocf.backbones.base import BackboneCache, TextConditioning
from cocf.backbones.wan22 import Wan22Backbone
from cocf.common.config import BackboneConfig
from cocf.common.types import TokenGrid

# model_sigma with flow_shift=5: t=0.95 → timestep ≈ 990 ≥ 875 (high-noise
# expert), t=0.5 → ≈ 833 < 875 (low-noise expert).
T_HI = torch.tensor([0.95])
T_LO = torch.tensor([0.5])
GRID = TokenGrid(t=1, h=2, w=2)  # 4 tokens; token dim = 16·1·2·2 = 64


class _Expert(nn.Module):
    """Stand-in WanTransformer3DModel: emits a constant velocity identifying itself."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden_states=None, **kwargs):
        return SimpleNamespace(sample=torch.full_like(hidden_states, self.value))


def _backbone() -> Wan22Backbone:
    bb = Wan22Backbone(BackboneConfig(device="cpu", dtype="float32",
                                      extra={"flow_shift": 5.0}))
    bb.transformer = _Expert(1.0)    # high-noise expert
    bb.transformer_2 = _Expert(2.0)  # low-noise expert
    bb._loaded = True  # bypass weight loading; forwards hit the stand-ins
    return bb


def _inputs():
    tokens = torch.zeros(1, GRID.num_tokens, 64)
    cond = TextConditioning(embeds=torch.zeros(1, 4, 8))
    return tokens, cond


def test_cross_expert_cache_is_dropped():
    bb = _backbone()
    tokens, cond = _inputs()
    bb.denoise(tokens, T_HI, cond, grid=GRID)  # cache now belongs to the hi expert
    foreign = BackboneCache(model_output=torch.full((1, GRID.num_tokens, 64), 7.0), step=1)
    active = torch.tensor([True, False, False, False])
    out = bb.denoise(tokens, T_LO, cond, grid=GRID, active_mask=active, cache=foreign)
    # The boundary was crossed → no splice: every token carries the lo expert's
    # fresh value, none the cached 7.0.
    assert torch.all(out.model_output == 2.0)
    assert out.compute_fraction == 1.0


def test_cross_expert_whole_step_skip_is_forced_to_compute():
    bb = _backbone()
    tokens, cond = _inputs()
    bb.denoise(tokens, T_HI, cond, grid=GRID)
    foreign = BackboneCache(model_output=torch.full((1, GRID.num_tokens, 64), 7.0), step=1)
    out = bb.denoise(tokens, T_LO, cond, grid=GRID,
                     active_mask=torch.zeros(GRID.num_tokens, dtype=torch.bool),
                     cache=foreign)
    assert torch.all(out.model_output == 2.0)
    assert out.compute_fraction == 1.0


def test_same_expert_cache_is_still_spliced():
    bb = _backbone()
    tokens, cond = _inputs()
    bb.denoise(tokens, T_HI, cond, grid=GRID)
    own = BackboneCache(model_output=torch.full((1, GRID.num_tokens, 64), 7.0), step=1)
    active = torch.tensor([True, False, False, False])
    out = bb.denoise(tokens, T_HI, cond, grid=GRID, active_mask=active, cache=own)
    # Same expert: TeaCache-style reuse is untouched — inactive tokens keep 7.0.
    assert torch.all(out.model_output[0, 0] == 1.0)
    assert torch.all(out.model_output[0, 1:] == 7.0)


def test_single_expert_variant_never_invalidates():
    bb = _backbone()
    bb.transformer_2 = None
    tokens, cond = _inputs()
    bb.denoise(tokens, T_HI, cond, grid=GRID)
    own = BackboneCache(model_output=torch.full((1, GRID.num_tokens, 64), 7.0), step=1)
    out = bb.denoise(tokens, T_LO, cond, grid=GRID,
                     active_mask=torch.tensor([True, False, False, False]), cache=own)
    assert torch.all(out.model_output[0, 0] == 1.0)
    assert torch.all(out.model_output[0, 1:] == 7.0)
