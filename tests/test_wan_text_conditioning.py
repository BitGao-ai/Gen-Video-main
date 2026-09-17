import pytest
import torch

from cocf.backbones.base import TextConditioning
from cocf.backbones.wan21 import Wan21Backbone
from cocf.backbones.wan22 import Wan22Backbone
from cocf.common.config import BackboneConfig


@pytest.mark.parametrize("cls", [Wan21Backbone, Wan22Backbone])
def test_full_zero_padded_sequence(cls):
    bb = cls(BackboneConfig(device="cpu", dtype="float32"))
    embeds = torch.randn(2, 512, 4, requires_grad=True)
    mask = torch.zeros(2, 512, dtype=torch.long)
    mask[0, :21] = 1
    mask[1, :7] = 1
    before = embeds.detach().clone()
    kwargs = bb._text_kwargs(TextConditioning(embeds=embeds, mask=mask), None)
    assert set(kwargs) == {"encoder_hidden_states"}
    result = kwargs["encoder_hidden_states"]
    assert result.shape == (2, 512, 4)
    assert torch.equal(result[mask.bool()], embeds[mask.bool()])
    assert torch.count_nonzero(result[~mask.bool()]) == 0
    assert torch.equal(embeds, before)
    result.sum().backward()
    assert torch.count_nonzero(embeds.grad[~mask.bool()]) == 0
    assert torch.all(embeds.grad[mask.bool()] == 1)


def test_precomputed_unmasked_condition_is_preserved():
    bb = Wan22Backbone(BackboneConfig(device="cpu", dtype="float32"))
    embeds = torch.randn(1, 512, 4)
    assert torch.equal(bb._text_kwargs(TextConditioning(embeds=embeds), None)
                       ["encoder_hidden_states"], embeds)


def test_invalid_mask_rejected():
    bb = Wan22Backbone(BackboneConfig(device="cpu", dtype="float32"))
    with pytest.raises(ValueError, match="mask"):
        bb._text_kwargs(TextConditioning(embeds=torch.ones(1, 512, 4),
                                        mask=torch.ones(1, 21)), None)
