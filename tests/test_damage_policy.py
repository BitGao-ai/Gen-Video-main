import unittest
from types import SimpleNamespace

import torch

from cocf.lcocf.damage import DEFAULT_DAMAGE_WEIGHTS, DAMAGE_DIMENSIONS
from cocf.training.stage_b_losses import damage_scalar_batch, tube_temporal_smoothness


class DamagePolicyTests(unittest.TestCase):
    def test_ocr_excluded_and_weights_normalized(self):
        self.assertEqual(DEFAULT_DAMAGE_WEIGHTS['ocr_accuracy'], 0)
        self.assertAlmostEqual(sum(DEFAULT_DAMAGE_WEIGHTS.values()), 1)
        labels = torch.ones(2, len(DAMAGE_DIMENSIONS))
        labels[0, DAMAGE_DIMENSIONS.index('ocr_accuracy')] = 0
        torch.testing.assert_close(damage_scalar_batch(labels), torch.ones(2))

    def test_temporal_pairs_require_distinct_steps(self):
        acc = SimpleNamespace(tube_smoothing=lambda a, b: (a[0] - b[0]).square().mean())
        probs = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        batch = dict(video_id=['v'] * 3, tube_id=torch.zeros(3), timestep=torch.ones(3))
        value, pairs = tube_temporal_smoothness(acc, probs, batch, return_pairs=True)
        self.assertEqual(pairs, 0)
        batch['timestep'][2] = 2
        value, pairs = tube_temporal_smoothness(acc, probs, batch, return_pairs=True)
        self.assertEqual(pairs, 1)
        self.assertAlmostEqual(value.item(), .25)
        value.backward()
        self.assertIsNotNone(probs.grad)
