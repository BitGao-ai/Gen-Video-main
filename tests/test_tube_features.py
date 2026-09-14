import unittest

import torch

from cocf.common.types import SemanticTube, TokenGrid
from cocf.lcocf.data import tube_clip_embed, tube_pixel_mask
from cocf.tubes.model_perception import ModelPerception


class TubeFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.grid = TokenGrid(t=13, h=2, w=2)
        self.tube = SemanticTube(
            tube_id=0,
            tokens_by_frame={i: torch.arange(4) + i * 4 for i in range(13)},
            masks_by_frame={i: torch.ones(2, 2, dtype=torch.bool) for i in range(13)},
        )
        self.video = torch.arange(49.).view(49, 1, 1, 1).expand(49, 3, 2, 2).clone()
        feature = lambda frame, mask: frame.mean().reshape(1)
        self.perception = ModelPerception(
            lambda frame: None, feature, feature, lambda text: None,
            lambda a, b: None, d_id=1, d_clip=1, clip_grad_fn=feature,
        )

    def test_temporal_expansion_and_middle_frame(self):
        mask = tube_pixel_mask(self.video, self.tube, self.grid)
        self.assertTrue(mask.all())
        embed = tube_clip_embed(self.video, self.tube, self.grid, self.perception)
        self.assertEqual(embed.item(), 24)

    def test_window_matches_full_mask_slice(self):
        self.tube.masks_by_frame[6] = torch.zeros(2, 2, dtype=torch.bool)
        full = tube_pixel_mask(self.video, self.tube, self.grid)
        window = tube_pixel_mask(self.video[20:29], self.tube, self.grid,
                                 frame_span=(20, 29), full_frame_count=49)
        torch.testing.assert_close(window, full[20:29])
        with self.assertRaises(ValueError):
            tube_pixel_mask(self.video[20:29], self.tube, self.grid, frame_span=(20, 29))

    def test_training_features_reach_pixels_reference_is_detached(self):
        self.video.requires_grad_()
        grad = tube_clip_embed(self.video, self.tube, self.grid, self.perception,
                               differentiable=True)
        ref = tube_clip_embed(self.video, self.tube, self.grid, self.perception)
        torch.testing.assert_close(grad, ref)
        self.assertFalse(ref.requires_grad)
        grad.sum().backward()
        self.assertGreater(self.video.grad[24].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
