import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cocf.common.config import Config
from cocf.core.accelerator import Accelerator
from cocf.training.stage_b_joint import JointTrainingStage, StageBConfig
from cocf.training.stage_b_losses import (
    compute_joint_loss,
    damage_scalar_batch,
    gaussian_nll,
    per_sample_budget,
    predictor_regression_loss,
)


def _mock_accelerator():
    config = Config()
    config.backbone.name = "mock"
    config.backbone.device = "cpu"
    torch.manual_seed(7)
    return Accelerator.from_config(config)


def _batch(n=8, actions=None):
    g = torch.Generator().manual_seed(3)
    return {
        "tube_features": torch.rand(n, 7, generator=g),
        "strength_features": torch.rand(n, 3, generator=g),
        "action": actions if actions is not None else torch.randint(0, 4, (n,), generator=g),
        "step_frac": torch.rand(n, generator=g),
        "damage_label": torch.rand(n, 8, generator=g) * 0.05,
        "tube_id": torch.arange(n) % 2,
        "timestep": torch.arange(n) % 3,
    }


def _trainer(**kw):
    trainer = object.__new__(JointTrainingStage)
    trainer.config = StageBConfig(processed_root=Path("."), **kw)
    return trainer


class RegressionLossTests(unittest.TestCase):
    def test_excludes_full_and_scales(self):
        target = torch.tensor([0.0, 0.02, 0.03])
        mu = torch.tensor([0.5, 0.01, 0.04])
        actions = torch.tensor([0, 1, 2])
        loss = predictor_regression_loss(target, mu, actions, objective="mse", scale=100.0)
        expected = torch.nn.functional.mse_loss(mu[1:] * 100.0, target[1:] * 100.0)
        torch.testing.assert_close(loss, expected)

    def test_full_only_batch_is_zero(self):
        loss = predictor_regression_loss(
            torch.zeros(2), torch.ones(2), torch.zeros(2, dtype=torch.long))
        self.assertEqual(float(loss), 0.0)

    def test_unknown_objective_rejected(self):
        with self.assertRaises(ValueError):
            predictor_regression_loss(torch.zeros(1), torch.zeros(1), objective="nll")


class PhasedLossTests(unittest.TestCase):
    def test_mean_phase_keeps_variance_out_of_loss(self):
        acc = _mock_accelerator()
        loss, comps = compute_joint_loss(acc, _batch(), phase="mean")
        loss.backward()
        var = acc.lcocf.predictor.var_head
        for p in var.parameters():
            self.assertTrue(p.grad is None or bool((p.grad == 0).all()),
                            "variance head must receive no gradient in the mean phase")
        self.assertIsNotNone(acc.lcocf.predictor.mu_head.weight.grad)
        self.assertTrue(bool((acc.lcocf.predictor.mu_head.weight.grad != 0).any()))

    def test_mean_phase_isolation_drops_aux_terms(self):
        acc = _mock_accelerator()
        cfg = acc.config.training
        _, comps = compute_joint_loss(acc, _batch(), phase="mean", isolate_aux=True)
        expected = (comps["cocf"] + cfg.lambda_cert * comps["cert"]
                    + cfg.lambda_cmsc * comps["cmsc"])
        self.assertAlmostEqual(comps["total"], expected, places=5)

    def test_var_phase_excludes_full(self):
        acc = _mock_accelerator()
        batch = _batch()
        loss, comps = compute_joint_loss(acc, batch, phase="var")
        self.assertEqual(set(comps), {"cocf", "total"})
        from cocf.lcocf.predictor import build_predictor_input_batch
        strength = acc.lcocf.strength_field(batch["strength_features"])
        inp = build_predictor_input_batch(
            batch["tube_features"], batch["strength_features"], strength,
            per_sample_budget(acc, batch), batch["step_frac"],
            acc.config.lcocf.predictor.context_dim,
        )
        pred = acc.lcocf.predictor(inp)
        idx = batch["action"].clamp(0, pred.mu.shape[-1] - 1).unsqueeze(-1)
        keep = batch["action"] != 0
        ref = gaussian_nll(
            damage_scalar_batch(batch["damage_label"])[keep],
            pred.mu.gather(-1, idx).squeeze(-1)[keep],
            pred.sigma.gather(-1, idx).squeeze(-1)[keep],
        )
        torch.testing.assert_close(loss, ref)

    def test_var_phase_all_full_batch_is_zero_but_differentiable(self):
        acc = _mock_accelerator()
        batch = _batch(actions=torch.zeros(6, dtype=torch.long))
        loss, _ = compute_joint_loss(acc, batch, phase="var")
        self.assertEqual(float(loss), 0.0)
        loss.backward()  # must not raise

    def test_joint_isolation_excludes_full(self):
        """Phased-mode joint fine-tune keeps FULL out of the NLL too."""
        acc = _mock_accelerator()
        batch = _batch()
        loss, comps = compute_joint_loss(acc, batch, phase="joint", isolate_aux=True)
        from cocf.lcocf.predictor import build_predictor_input_batch
        strength = acc.lcocf.strength_field(batch["strength_features"])
        inp = build_predictor_input_batch(
            batch["tube_features"], batch["strength_features"], strength,
            per_sample_budget(acc, batch), batch["step_frac"],
            acc.config.lcocf.predictor.context_dim,
        )
        pred = acc.lcocf.predictor(inp)
        idx = batch["action"].clamp(0, pred.mu.shape[-1] - 1).unsqueeze(-1)
        keep = batch["action"] != 0
        ref = gaussian_nll(
            damage_scalar_batch(batch["damage_label"])[keep],
            pred.mu.gather(-1, idx).squeeze(-1)[keep],
            pred.sigma.gather(-1, idx).squeeze(-1)[keep],
        )
        self.assertAlmostEqual(comps["cocf"], float(ref), places=5)

    def test_joint_classic_keeps_full_in_nll(self):
        """Classic single-phase training (isolate_aux=False) is unchanged."""
        acc = _mock_accelerator()
        batch = _batch()
        _, comps = compute_joint_loss(acc, batch, phase="joint", isolate_aux=False)
        from cocf.lcocf.predictor import build_predictor_input_batch
        strength = acc.lcocf.strength_field(batch["strength_features"])
        inp = build_predictor_input_batch(
            batch["tube_features"], batch["strength_features"], strength,
            per_sample_budget(acc, batch), batch["step_frac"],
            acc.config.lcocf.predictor.context_dim,
        )
        pred = acc.lcocf.predictor(inp)
        idx = batch["action"].clamp(0, pred.mu.shape[-1] - 1).unsqueeze(-1)
        ref = gaussian_nll(
            damage_scalar_batch(batch["damage_label"]),
            pred.mu.gather(-1, idx).squeeze(-1),
            pred.sigma.gather(-1, idx).squeeze(-1),
        )
        self.assertAlmostEqual(comps["cocf"], float(ref), places=5)


class PhaseScheduleTests(unittest.TestCase):
    def test_phase_boundaries_in_steps(self):
        trainer = _trainer(predictor_mean_steps=4, predictor_var_steps=2,
                           predictor_joint_lr_scale=0.1)
        self.assertEqual([trainer._phase_for_step(s) for s in (0, 3, 4, 5, 6, 99)],
                         ["mean", "mean", "var", "var", "joint", "joint"])

    def test_no_joint_phase_stops_after_var(self):
        trainer = _trainer(predictor_mean_steps=4, predictor_var_steps=2)
        self.assertIsNone(trainer._phase_for_step(6))

    def test_phased_config_requires_mean_steps(self):
        with self.assertRaises(ValueError):
            StageBConfig(processed_root=Path("."), predictor_mean_steps=0,
                         predictor_var_steps=10)

    def test_rebase_preserves_remaining_phase_lengths(self):
        trainer = _trainer(predictor_mean_steps=10, predictor_var_steps=5,
                           predictor_joint_lr_scale=0.1)
        trainer._phase_bounds = trainer._build_phase_schedule()
        trainer._rebase_schedule("mean", 3)  # mean early-stopped at step 3
        self.assertEqual(trainer._phase_bounds, [["mean", 3], ["var", 8], ["joint", None]])
        self.assertEqual(trainer._phase_for_step(3), "var")
        self.assertEqual(trainer._phase_for_step(8), "joint")

    def test_successor_phase(self):
        trainer = _trainer(predictor_mean_steps=4, predictor_var_steps=2,
                           predictor_joint_lr_scale=0.1)
        trainer._phase_bounds = trainer._build_phase_schedule()
        self.assertEqual(trainer._successor_phase("mean"), "var")
        self.assertEqual(trainer._successor_phase("var"), "joint")
        self.assertIsNone(trainer._successor_phase("joint"))


class PhaseTransitionTests(unittest.TestCase):
    def _full_trainer(self, **kw):
        acc = _mock_accelerator()
        trainer = _trainer(predictor_mean_steps=4, predictor_var_steps=2,
                           predictor_joint_lr_scale=0.1, **kw)
        trainer.accelerator = acc
        trainer.train_cfg = acc.config.training
        trainer.trainable_params = acc.trainable_parameters()
        trainer.active_params = list(trainer.trainable_params)
        opt = trainer.train_cfg.optim
        trainer.optimizer = torch.optim.AdamW(trainer.trainable_params, lr=opt.lr)
        return trainer

    def test_var_head_only_freezing(self):
        trainer = self._full_trainer()
        trainer._set_var_head_only(True)
        var_ids = {id(p) for p in trainer.accelerator.lcocf.predictor.var_head.parameters()}
        for p in trainer.trainable_params:
            self.assertEqual(p.requires_grad, id(p) in var_ids)
        trainer._set_var_head_only(False)
        self.assertTrue(all(p.requires_grad for p in trainer.trainable_params))

    def test_enter_phase_rebuilds_optimizer(self):
        trainer = self._full_trainer()
        base_lr = trainer.train_cfg.optim.lr
        trainer._enter_phase("var")
        var_ids = {id(p) for p in trainer.accelerator.lcocf.predictor.var_head.parameters()}
        opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        self.assertEqual(opt_ids, var_ids)
        self.assertEqual({id(p) for p in trainer.active_params}, var_ids)
        trainer._enter_phase("joint")
        opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        self.assertEqual(opt_ids, {id(p) for p in trainer.trainable_params})
        for g in trainer.optimizer.param_groups:
            self.assertAlmostEqual(g["lr"], base_lr * 0.1)

    def test_frozen_params_do_not_move_in_var_phase(self):
        """Issue: stale AdamW momentum + weight decay moved frozen params on zero grads."""
        trainer = self._full_trainer()
        acc = trainer.accelerator
        # Prime the optimizer with real gradients on every parameter (mean phase).
        loss, _ = compute_joint_loss(acc, _batch(), phase="mean")
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.optimizer.step()
        trainer._enter_phase("var")
        frozen = [p for p in trainer.trainable_params if not p.requires_grad]
        snapshot = [p.detach().clone() for p in frozen]
        loss, _ = compute_joint_loss(acc, _batch(), phase="var")
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.active_params, 1.0)
        trainer.optimizer.step()
        for p, ref in zip(frozen, snapshot):
            torch.testing.assert_close(p.detach(), ref)
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any())
                            for p in acc.lcocf.predictor.var_head.parameters()))

    def test_save_and_restore_best_roundtrip(self):
        trainer = self._full_trainer()
        acc = trainer.accelerator
        trainer.dctx = SimpleNamespace(is_main=False)   # skip the file write
        trainer._phased = True
        trainer._best_metric = 0.5
        original = {k: v.detach().clone() for k, v in acc.state_dict().items()}
        trainer._save_best("mean")
        with torch.no_grad():
            for p in acc.lcocf.predictor.mu_head.parameters():
                p.add_(1.0)
        trainer._restore_best()
        for k, v in acc.state_dict().items():
            torch.testing.assert_close(v, original[k])

    def test_monitor_for_scores_each_phase_on_its_own_objective(self):
        metrics = {"mae": 1.0, "nll_nonfull": 2.0, "mae_nonfull": 3.0}
        self.assertEqual(JointTrainingStage._monitor_for("var", metrics), 2.0)
        self.assertEqual(JointTrainingStage._monitor_for("mean", metrics), 3.0)
        self.assertEqual(JointTrainingStage._monitor_for("joint", metrics), 1.0)

    def test_enter_phase_sets_module_mode(self):
        """Variance phase runs in eval mode; every other phase in train mode."""
        trainer = self._full_trainer()
        trainer._enter_phase("var")
        self.assertFalse(trainer.accelerator.training)
        trainer._enter_phase("joint")
        self.assertTrue(trainer.accelerator.training)
        trainer._enter_phase("mean")
        self.assertTrue(trainer.accelerator.training)


class PhaseStateTests(unittest.TestCase):
    def _trainer_with_schedule(self, **kw):
        trainer = _trainer(predictor_mean_steps=4, predictor_var_steps=2,
                           predictor_joint_lr_scale=0.1, **kw)
        trainer._phase_bounds = trainer._build_phase_schedule()
        return trainer

    def test_calibration_complete_follows_explicit_flag(self):
        """calibration_complete mirrors _var_calibrated — not the phase name."""
        trainer = self._trainer_with_schedule()
        for phase, updates in (("mean", 2), ("var", 5), ("joint", 7)):
            self.assertFalse(trainer._phase_state_dict(phase, updates)["calibration_complete"])
        trainer._var_calibrated = True
        for phase, updates in (("mean", 2), ("var", 5), ("joint", 7)):
            self.assertTrue(trainer._phase_state_dict(phase, updates)["calibration_complete"])

    def test_mean_only_schedule_is_never_calibration_complete(self):
        """Issue: a mean-only experiment that finished its schedule was stamped
        calibration_complete=True despite never training the variance head."""
        trainer = _trainer(predictor_mean_steps=4)   # no var, no joint
        trainer._phase_bounds = trainer._build_phase_schedule()
        state = trainer._phase_state_dict("mean", 4)  # schedule fully spent
        self.assertFalse(state["calibration_complete"])

    def test_var_budget_spent(self):
        """Issue: stopping exactly on the variance phase's last step (a joint
        phase configured but never entered) must still count as calibrated."""
        trainer = self._trainer_with_schedule()   # mean=4, var ends at 6, joint open
        self.assertFalse(trainer._var_budget_spent(5))
        self.assertTrue(trainer._var_budget_spent(6))
        self.assertTrue(trainer._var_budget_spent(7))
        solo = _trainer(predictor_mean_steps=4)
        solo._phase_bounds = solo._build_phase_schedule()
        self.assertFalse(solo._var_budget_spent(99))   # no var phase at all

    def test_restamp_marks_best_var_file_complete(self):
        """Issue: stage_b_best_var.pt kept calibration_complete=False after the
        variance phase completed, so the default loader rejected it."""
        import tempfile
        acc = _mock_accelerator()
        trainer = self._trainer_with_schedule()
        trainer.accelerator = acc
        trainer.dctx = SimpleNamespace(is_main=True)
        trainer._phased = True
        trainer._best_metric = 0.5
        trainer._updates = 6
        with tempfile.TemporaryDirectory() as tmp:
            trainer.config.checkpoint_dir = Path(tmp)
            trainer._save_best("var")
            path = Path(tmp) / "stage_b_best_var.pt"
            before = torch.load(path, weights_only=False)["phase_state"]
            self.assertFalse(before["calibration_complete"])
            trainer._var_calibrated = True
            trainer._restamp_best("var", 6)
            after = torch.load(path, weights_only=False)["phase_state"]
            self.assertTrue(after["calibration_complete"])
            self.assertEqual(after["phase"], "var")
            self.assertEqual(after["updates"], 6)

    def test_restamp_skips_when_nothing_saved(self):
        trainer = self._trainer_with_schedule()
        trainer._phased = True
        trainer._best_state = None
        trainer.dctx = SimpleNamespace(is_main=True)
        trainer._restamp_best("var", 6)   # must not raise

    def test_save_best_records_phase_state(self):
        import tempfile
        acc = _mock_accelerator()
        trainer = self._trainer_with_schedule()
        trainer.accelerator = acc
        trainer.dctx = SimpleNamespace(is_main=True)
        trainer._phased = True
        trainer._best_metric = 0.5
        trainer._updates = 3
        with tempfile.TemporaryDirectory() as tmp:
            trainer.config.checkpoint_dir = Path(tmp)
            trainer._save_best("mean")
            ckpt = torch.load(Path(tmp) / "stage_b_best_mean.pt", weights_only=False)
        state = ckpt["phase_state"]
        self.assertTrue(state["phased"])
        self.assertEqual(state["phase"], "mean")
        self.assertEqual(state["updates"], 3)
        self.assertFalse(state["calibration_complete"])

    def test_load_checkpoint_gates_incomplete_phased_run(self):
        from cocf.training.checkpoint import build_checkpoint, load_checkpoint
        acc = _mock_accelerator()
        ckpt = build_checkpoint(acc)
        ckpt["phase_state"] = {"phased": True, "phase": "mean", "updates": 3,
                               "schedule": [("mean", 4), ("var", 2)],
                               "calibration_complete": False}
        with self.assertRaises(ValueError):
            load_checkpoint(acc, ckpt)
        # Diagnostics opt in explicitly.
        self.assertEqual(load_checkpoint(acc, ckpt, allow_incomplete=True), 0)
        # A completed run loads without the flag.
        ckpt["phase_state"]["calibration_complete"] = True
        self.assertEqual(load_checkpoint(acc, ckpt), 0)
        # Classic checkpoints carry no phase_state and are unaffected.
        del ckpt["phase_state"]
        self.assertEqual(load_checkpoint(acc, ckpt), 0)


if __name__ == "__main__":
    unittest.main()
