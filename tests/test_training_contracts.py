import unittest

import torch

from cocf.common.config import Config
from cocf.common.types import TokenGrid
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine
from cocf.training.checkpoint import build_checkpoint, load_checkpoint
from cocf.training.stage_b_losses import _local_cmsc_violation


class TrainingContractsTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.backbone.name = "mock"
        self.config.backbone.device = "cpu"
        self.acc = Accelerator.from_config(self.config)

    def test_risk_matches_inference_with_padding(self):
        text = torch.randn(1, 4, self.acc.text_dim)
        visual = torch.randn(1, self.acc.visual_dim)
        for mask in (torch.tensor([[1, 0, 1, 0]]), torch.zeros(1, 4)):
            batch = {"text_embed": text, "tube_visual_embed_cf": visual, "text_mask": mask}
            trained = _local_cmsc_violation(self.acc, batch, torch.zeros(1))
            served = self.acc.cmsc_loss.local_conservation(text[0], {0: visual[0]}, mask[0])
            self.assertAlmostEqual(trained.item(), served[0])

    def test_checkpoint_roundtrip_and_reject_partial(self):
        checkpoint = build_checkpoint(self.acc)
        self.assertEqual(load_checkpoint(self.acc, checkpoint), 0)
        checkpoint["accelerator"] = dict(checkpoint["accelerator"])
        checkpoint["accelerator"].pop(next(iter(checkpoint["accelerator"])))
        with self.assertRaises(RuntimeError):
            load_checkpoint(self.acc, checkpoint)

    def test_checkpoint_rejects_wrong_backbone(self):
        checkpoint = build_checkpoint(self.acc)
        checkpoint["model_metadata"]["backbone"] = "wan22"
        with self.assertRaises(ValueError):
            load_checkpoint(self.acc, checkpoint)

    def test_checkpoint_rejects_old_scoring(self):
        checkpoint = build_checkpoint(self.acc)
        checkpoint.pop('damage_weights')
        with self.assertRaisesRegex(ValueError, 'scoring policy'):
            load_checkpoint(self.acc, checkpoint)

    def test_batch_rejected_before_render(self):
        engine = InferenceEngine(self.acc, self.config.engine, self.config.trigger)
        grid = TokenGrid(t=1, h=2, w=2)
        with self.assertRaisesRegex(ValueError, "single video"):
            engine.generate(["a", "b"], torch.zeros(2, 4, self.acc.token_dim), grid,
                            self.acc.backbone.encode_text(["a", "b"]), self.acc.backbone)


if __name__ == "__main__":
    unittest.main()
