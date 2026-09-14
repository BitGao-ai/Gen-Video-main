import unittest

import torch

from cocf.common.config import TriggerConfig
from cocf.common.types import SemanticTube, TokenGrid
from cocf.raec.repair import BoundaryRepair


class BoundaryRepairTest(unittest.TestCase):
    def test_repair_favors_reference_at_rim_and_current_inside(self):
        repair = BoundaryRepair(TriggerConfig(sigma_bnd=4.0))
        grid = TokenGrid(t=2, h=31, w=31)
        idx = torch.arange(grid.tokens_per_frame) + grid.tokens_per_frame
        tube = SemanticTube(
            tube_id=0,
            tokens_by_frame={1: idx},
            masks_by_frame={1: torch.ones(31, 31, dtype=torch.bool)},
        )
        current = torch.zeros(1, grid.num_tokens, 1, requires_grad=True)
        reference = torch.ones_like(current, requires_grad=True)
        result = repair.repair(current, reference, tube, grid)
        edge, center = int(idx[0]), int(idx[480])
        self.assertGreater(result.z[0, edge, 0].item(), 0.7)
        self.assertLess(result.z[0, center, 0].item(), 0.1)
        torch.testing.assert_close(result.z[:, :grid.tokens_per_frame],
                                   current[:, :grid.tokens_per_frame])
        torch.testing.assert_close(current.detach(), torch.zeros_like(current))
        torch.testing.assert_close(result.refreshed, idx)
        self.assertFalse(result.rolled_back)
        result.z.sum().backward()
        for value in (current, reference):
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(current.grad[0, center, 0], current.grad[0, edge, 0])
        self.assertGreater(reference.grad[0, edge, 0], reference.grad[0, center, 0])

    def test_erosion_depth_respects_cap_and_mask_boundary(self):
        mask = torch.zeros(33, 33, dtype=torch.bool)
        mask[1:-1, 1:-1] = True
        for cap in (1, 2, 12):
            with self.subTest(cap=cap):
                depth = BoundaryRepair._erosion_depth(mask, cap)
                self.assertEqual(depth[0, 0].item(), 0)
                self.assertEqual(depth[1, 1].item(), 1)
                self.assertEqual(depth[16, 16].item(), cap)
                self.assertEqual(depth.max().item(), cap)
        self.assertFalse(BoundaryRepair._erosion_depth(torch.zeros_like(mask), 12).any())

    def test_erosion_depth_rejects_nonpositive_cap(self):
        for cap in (0, -1):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                BoundaryRepair._erosion_depth(torch.ones(3, 3), cap)


if __name__ == "__main__":
    unittest.main()
