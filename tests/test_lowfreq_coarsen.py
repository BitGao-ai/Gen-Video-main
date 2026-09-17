import torch

from cocf.backbones.transition import TransitionExecutor
from cocf.common.types import SemanticTube, TokenGrid


def test_holes_take_anchor_value_and_respect_protected():
    executor = TransitionExecutor(None, lowfreq_stride=2)
    grid = TokenGrid(t=1, h=2, w=2)
    tube = SemanticTube(0, tokens_by_frame={0: torch.arange(4)})
    z = torch.tensor([[[11.], [22.], [33.], [44.]]], requires_grad=True)
    protected = torch.tensor([True, True, False, False])
    result = executor.coarsen_lowfreq(z, tube, grid, protected=protected)
    torch.testing.assert_close(result, torch.tensor([[[11.], [22.], [11.], [11.]]]))
    result.sum().backward()
    torch.testing.assert_close(z.grad, torch.tensor([[[3.], [1.], [0.], [0.]]]))


def test_holes_do_not_freeze_their_own_noise_offset():
    # The multi-step failure this guards: coarsening the step *increment* keeps
    # each hole's step-1 noise offset against its anchor frozen for the whole
    # trajectory (hole_T = hole_1 + anchor_T − anchor_1), which decodes as
    # saturated noise blocks. Coarsening the *value* lets the hole's noise die
    # with the anchor's.
    executor = TransitionExecutor(None, lowfreq_stride=2)
    grid = TokenGrid(t=1, h=2, w=2)
    tube = SemanticTube(0, tokens_by_frame={0: torch.arange(4)})
    z = torch.randn(1, 4, 8)
    out = executor.coarsen_lowfreq(z, tube, grid)
    torch.testing.assert_close(out[0, 1:], z[0, :1].expand(3, 8))
    torch.testing.assert_close(out[0, :1], z[0, :1])


def test_irregular_tube_does_not_read_foreign_lattice():
    executor = TransitionExecutor(None)
    grid = TokenGrid(t=1, h=2, w=2)
    tube = SemanticTube(0, tokens_by_frame={0: torch.tensor([1, 3])})
    z = torch.tensor([[[999.], [2.], [3.], [4.]]])
    assert torch.equal(executor.coarsen_lowfreq(z, tube, grid), z)
