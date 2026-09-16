import unittest
import torch
from scripts.train.diagnose_stage_b_fit import regression_loss


class FitLossTests(unittest.TestCase):
    def test_predictor_can_fit_synthetic_nonfull_targets(self):
        from cocf.common.config import PredictorConfig
        from cocf.lcocf.predictor import DamagePredictor, predictor_input_dim
        torch.manual_seed(5)
        cfg = PredictorConfig()
        model = DamagePredictor(cfg).eval()
        x = torch.randn(12, predictor_input_dim(cfg))
        actions = torch.tensor([1, 2, 3] * 4)
        y = torch.linspace(.01, .05, 12)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        losses = []
        for _ in range(60):
            pred = model(x)
            loss = regression_loss(y, pred.mu.gather(1, actions[:, None]).squeeze(1),
                                   pred.sigma.gather(1, actions[:, None]).squeeze(1), actions, 'mse')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * .5)

    def test_full_is_excluded(self):
        for objective in ('mse', 'huber', 'nll'):
            mu = torch.tensor([.8, .2], requires_grad=True)
            sigma = torch.tensor([.1, .1], requires_grad=True)
            loss = regression_loss(torch.tensor([0., .3]), mu, sigma,
                                   torch.tensor([0, 1]), objective)
            loss.backward()
            self.assertEqual(mu.grad[0], 0)
            self.assertNotEqual(mu.grad[1], 0)
            if sigma.grad is not None:
                self.assertEqual(sigma.grad[0], 0)

    def test_full_only_skips_update(self):
        self.assertIsNone(regression_loss(torch.zeros(2), torch.ones(2), torch.ones(2),
                                         torch.zeros(2), 'mse'))
