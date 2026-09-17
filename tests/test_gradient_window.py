import types
import unittest

import torch

from cocf.common.config import Config
from cocf.common.types import TokenGrid, SemanticTube, AllocationDecision, Action
from cocf.backbones.base import sigma_from_step
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine
from cocf.engine.state import StepTrace, EngineState


class GradientWindowTest(unittest.TestCase):
    def test_no_cache_control_discards_cache_before_execution(self):
        from cocf.backbones.base import BackboneCache
        config = Config()
        config.backbone.device = "cpu"
        config.engine.diagnostic_no_cache = True
        acc = Accelerator.from_config(config)
        engine = InferenceEngine(acc, config.engine, config.trigger)
        grid = TokenGrid(t=1, h=2, w=2)
        state = EngineState(
            z=torch.ones(1, 4, acc.token_dim, requires_grad=True), grid=grid,
            cond=acc.backbone.encode_text(["test"]), subgraph=None,
            anchor_store=acc.raec.new_anchor_store(),
            tubes=[SemanticTube(0, tokens_by_frame={0: torch.arange(4)})],
            cache=BackboneCache(model_output=torch.full((1, 4, acc.token_dim), 999.0)),
            grad_window=1, retained_computed=1,
        )
        from unittest.mock import patch
        with patch.object(acc.transition, "step", wraps=acc.transition.step) as spy:
            engine._execute_transition(state, 2, AllocationDecision(0, {0: Action.ANCHOR}, 0, 0), acc.backbone)
        self.assertIsNone(spy.call_args.kwargs["cache"])
        self.assertEqual(state.graph_cuts, 1)

    def run_trajectory(self, window, computed_steps):
        torch.manual_seed(7)
        config = Config()
        config.backbone.name = "mock"
        config.backbone.device = "cpu"
        config.engine.num_inference_steps = 10
        config.engine.grad_window_steps = window
        acc = Accelerator.from_config(config)
        engine = InferenceEngine(acc, config.engine, config.trigger)
        weight = torch.nn.Parameter(torch.tensor(1.0))
        captured = {}

        def step(self, state, step_idx, t, backbone, record_sink=None):
            computed = step_idx < computed_steps
            if computed:
                # Exercise the real warmup truncation path with a trainable input.
                state.z, state.cache, _ = self._warmup_step(state, t, backbone)
                state.z = state.z + weight
            captured["state"] = state
            return StepTrace(step_idx, 1.0, 1.0, 0.0, 0,
                             compute_ratio=float(computed))

        engine._step = types.MethodType(step, engine)
        grid = TokenGrid(t=1, h=2, w=2)
        result = engine.generate(
            ["test"], torch.zeros(1, grid.num_tokens, acc.token_dim), grid,
            acc.backbone.encode_text(["test"]), acc.backbone, decode_grad=True,
        )
        result.video.square().mean().backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(weight.grad))
        self.assertGreater(weight.grad.abs().item(), 0)
        return result, captured["state"]

    def test_trailing_skips_keep_final_segment(self):
        result, state = self.run_trajectory(1, 3)
        reference, _ = self.run_trajectory(0, 3)
        self.assertEqual(state.graph_cuts, 2)
        self.assertEqual(state.retained_computed, 1)
        torch.testing.assert_close(result.video, reference.video)

    def test_continuous_compute_bounds_segment(self):
        _, state = self.run_trajectory(2, 10)
        self.assertEqual(state.graph_cuts, 4)
        self.assertEqual(state.retained_computed, 2)

    def test_executor_anchor_skip_preserves_graph_without_saved_anchor(self):
        config = Config()
        config.backbone.name = "mock"
        config.backbone.device = "cpu"
        acc = Accelerator.from_config(config)
        engine = InferenceEngine(acc, config.engine, config.trigger)
        grid = TokenGrid(t=1, h=2, w=2)
        weight = torch.nn.Parameter(torch.tensor(1.0))
        tube = SemanticTube(0, tokens_by_frame={0: torch.arange(4)})
        state = EngineState(z=torch.ones(1, 4, acc.token_dim) * weight,
                            grid=grid, cond=acc.backbone.encode_text(["test"]),
                            subgraph=None, anchor_store=acc.raec.new_anchor_store(),
                            tubes=[tube], grad_window=1)
        with acc.backbone.grad_mode(True):
            full = AllocationDecision(0, {0: Action.FULL}, 1.0, 1.0)
            out = engine._execute_transition(state, 3, full, acc.backbone)
            state.z, state.cache = out.z_next, out.cache
            state.retained_computed = 1
            # ANCHOR without a saved anchor no longer freezes (a freeze pins the
            # tokens at this step's noise level for the rest of the trajectory and
            # decodes as noise blocks): the tube rides the cached-velocity Euler
            # step, so z keeps advancing while compute stays 0.
            expected = state.z.detach().clone()
            T = engine.engine_cfg.num_inference_steps
            for step in (1, 2):
                eps_cached = state.cache.model_output
                t_now = torch.full((1,), sigma_from_step(3 - step, T))
                t_next = torch.full((1,), sigma_from_step(2 - step, T))
                expected = acc.backbone.scheduler_step(eps_cached, t_now, t_next, expected)
                skip = AllocationDecision(step, {0: Action.ANCHOR}, 0.0, 0.0)
                out = engine._execute_transition(state, 3 - step, skip, acc.backbone)
                state.z, state.cache = out.z_next, out.cache
                self.assertEqual(out.compute_ratio, 0.0)
            self.assertEqual(state.graph_cuts, 0)
            torch.testing.assert_close(state.z, expected)
            state.z.square().mean().backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(weight.grad))
        self.assertGreater(weight.grad.abs().item(), 0)


if __name__ == "__main__":
    unittest.main()
