"""Stage B: joint module training on the counterfactual store.

Trains the four learnable plugins together on the Stage-A samples with a
backbone-frozen objective ``L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc +
λ_cost·L_budget`` (see :func:`cocf.training.stage_b_losses.compute_joint_loss`).
Samples come from the counterfactual LMDB ``train`` split via a stratified,
action-balanced sampler; the ``val`` split is scored each epoch and the best model
by MAE is checkpointed with early stopping. Phased mode scores each phase on the
metric it optimises (see ``_monitor_for``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.memory import autocast, resolve_dtype
from cocf.core.accelerator import Accelerator
from cocf.data import (
    CounterfactualLMDBDataset,
    ProcessedLayout,
    StratifiedBatchSampler,
    collate_cocf_samples,
    timestep_stratum,
)
from cocf.lcocf.predictor import build_predictor_input_batch
from cocf.training.checkpoint import build_checkpoint
from cocf.training.distributed import (
    all_agree,
    all_reduce_mean,
    assert_same,
    average_gradients,
    broadcast_parameters,
    context as dist_context,
)
from cocf.training.stage_b_losses import (
    action_probs,
    compute_joint_loss,
    damage_scalar_batch,
    gaussian_nll,
    per_sample_budget,
    tube_temporal_smoothness,
    batch_float,
    _local_cmsc_violation,
)

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class StageBConfig:
    """Stage-B run settings. Loss weights / optimiser live in ``config.training``."""

    processed_root: Path                      # processed store root
    config: Config = field(default_factory=Config)
    batch_size: int = 32
    num_epochs: int = 10
    num_workers: int = 0                      # 0 keeps LMDB handles single-process safe
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    mixed_precision: bool = False
    checkpoint_dir: Path = Path("./checkpoints/stage_b")
    log_every: int = 20

    # Optional phased predictor training (mean → var → joint). Stage lengths are
    # optimiser steps; ``num_epochs`` is then only a budget cap.
    predictor_mean_steps: int = 0
    predictor_var_steps: int = 0
    predictor_joint_lr_scale: float = 0.0
    predictor_mean_objective: str = "mse"       # "mse" | "huber"
    predictor_target_scale: float = 100.0
    # Detach the certificate's (mu, sigma) and drop STA/budget terms in the mean phase.
    predictor_aux_isolation: bool = True

    def __post_init__(self) -> None:
        self.processed_root = Path(self.processed_root)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.predictor_mean_steps < 0 or self.predictor_var_steps < 0:
            raise ValueError("phase step counts must be nonnegative")
        if self.predictor_mean_steps == 0 and (
            self.predictor_var_steps or self.predictor_joint_lr_scale
        ):
            raise ValueError("phased training requires predictor_mean_steps > 0")


class JointTrainingStage:
    """Stage B: joint training of L-COCF / STA / RAEC / CMSC."""

    def __init__(self, accelerator: Accelerator, config: StageBConfig) -> None:
        self.accelerator = accelerator
        self.config = config
        self.device = config.device
        self.train_cfg = config.config.training
        self.layout = ProcessedLayout(config.processed_root)
        # Data-parallel context; a single-process run gets a disabled no-op context.
        self.dctx = dist_context()

        # Freeze the backbone; only the plugins remain trainable.
        self.accelerator.freeze_backbone()
        self.accelerator.to(self.device)

        self.trainable_params: List[torch.nn.Parameter] = self.accelerator.trainable_parameters()
        _log.info("Stage B: %s trainable plugin parameters",
                  f"{sum(p.numel() for p in self.trainable_params):,}")

        opt = self.train_cfg.optim
        self.optimizer = optim.AdamW(
            self.trainable_params, lr=opt.lr, betas=opt.betas, weight_decay=opt.weight_decay
        )
        # The parameter set the current phase optimises; the optimizer is rebuilt
        # from it at every phase switch.
        self.active_params: List[torch.nn.Parameter] = list(self.trainable_params)
        # AMP: autocast wraps the forward; the scaler is only meaningful for fp16 on CUDA.
        self._amp_dtype = (
            self.config.config.memory.amp_dtype if config.mixed_precision else "none"
        )
        use_scaler = (
            config.mixed_precision
            and str(self.device).startswith("cuda")
            and resolve_dtype(self._amp_dtype) is torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        # Completion record of the last run() (phased mode only).
        self.phase_state: Optional[Dict[str, Any]] = None
        # Set only when the variance phase completes; reset by run().
        self._var_calibrated = False

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def _build_loader(self, split: str, *, stratified: bool) -> Optional[DataLoader]:
        """DataLoader over one split's LMDB samples (stratified for train; plain for val)."""
        ids = self.layout.read_split(split)
        if not ids:
            return None
        # Prompt embeddings are stored once per clip and joined on read.
        dataset = CounterfactualLMDBDataset(
            self.layout.lmdb_dir, ids, text_embed_dir=self.layout.text_embed_dir
        )
        if len(dataset) == 0:
            return None

        index = {r["sample_id"]: r for r in self.layout.read_sample_index()}
        T = self.config.config.teacher.num_inference_steps
        actions, scenes, strata = [], [], []
        for k in dataset.keys:
            row = index.get(k, {})
            actions.append(int(row.get("action", 0) or 0))
            scenes.append(str(row.get("scene_type", "")))
            strata.append(timestep_stratum(int(row.get("timestep", 0) or 0), T))

        if stratified:
            sampler = StratifiedBatchSampler(
                actions, scenes, strata,
                batch_size=self.config.batch_size, seed=self.config.config.seed,
                rank=self.dctx.rank, world_size=self.dctx.world_size,
            )
            return DataLoader(
                dataset, batch_sampler=sampler, num_workers=self.config.num_workers,
                collate_fn=collate_cocf_samples,
            )
        # Validation runs on the whole split (not sharded) and is reduced across ranks.
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, collate_fn=collate_cocf_samples,
        )

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def run(self) -> Accelerator:
        """Execute Stage B and return the trained accelerator."""
        _log.info("=== Stage B: Joint Module Training (§4.1) ===")
        from cocf.lcocf.damage import DEFAULT_DAMAGE_WEIGHTS
        _log.info("Damage scoring weights: %s", DEFAULT_DAMAGE_WEIGHTS)
        if self.dctx.is_main:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_loader = self._build_loader("train", stratified=True)
        if train_loader is None:
            raise RuntimeError(
                f"Stage B found no training samples under {self.layout.lmdb_dir} / "
                f"{self.layout.train_list}. Run Stage A first."
            )
        val_loader = self._build_loader("val", stratified=False)
        _log.info("Stage B: %d train batches%s", len(train_loader),
                  f", {len(val_loader)} val batches" if val_loader else " (no val split)")

        if self.dctx.enabled:
            # Guard the first collective: ranks must agree on parameter and batch counts.
            assert_same(len(self.trainable_params), "trainable-parameter count", self.dctx)
            assert_same(len(train_loader), "batches per epoch", self.dctx)
            n = broadcast_parameters(
                list(self.trainable_params) + list(self.accelerator.buffers()), self.dctx
            )
            _log.info("Stage B: rank %d/%d, %d batches per epoch; %d tensor(s) synced "
                      "from rank 0", self.dctx.rank, self.dctx.world_size,
                      len(train_loader), n)

        sampler = train_loader.batch_sampler
        opt = self.train_cfg.optim
        # ``global_step`` counts successful optimiser updates only.
        global_step = 0
        self._updates = 0
        self._best_metric = float("inf")
        self._epochs_no_improve = 0
        self._best_state: Optional[Dict[str, Tensor]] = None
        # Set only when the variance phase verifiably completes; drives ``calibration_complete``.
        self._var_calibrated = False
        self._phased = phased = self.config.predictor_mean_steps > 0
        phase = "mean" if phased else "joint"
        phases_done = False
        if phased:
            self._phase_bounds = self._build_phase_schedule()
            _log.info(
                "Phased predictor training: mean=%d steps (%s, scale=%g) → var=%d steps"
                "%s; num_epochs=%d is a budget cap only; auxiliary isolation %s",
                self.config.predictor_mean_steps, self.config.predictor_mean_objective,
                self.config.predictor_target_scale, self.config.predictor_var_steps,
                f" → joint at lr×{self.config.predictor_joint_lr_scale:g}"
                if self.config.predictor_joint_lr_scale > 0 else " (no joint phase)",
                self.config.num_epochs,
                "ON" if self.config.predictor_aux_isolation else "OFF",
            )

        for epoch in range(self.config.num_epochs):
            # The variance phase freezes the mean path; keep it in eval mode.
            if phased and phase == "var":
                self.accelerator.eval()
            else:
                self.accelerator.train()
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            running: Dict[str, float] = {}
            n_batches = 0
            n_text_missing = 0

            for batch_idx, batch in enumerate(train_loader):
                if phased:
                    next_phase = self._phase_for_step(global_step)
                    if next_phase is None or next_phase != phase:
                        # Mid-epoch phase boundary: finalize and restore before advancing.
                        self._finalize_phase(phase, val_loader)
                        self._restore_best()
                        if phase == "var":
                            self._var_calibrated = True
                            self._restamp_best("var", global_step)
                        if next_phase is None:
                            phases_done = True
                            break
                        self._rebase_schedule(phase, global_step)
                        phase = next_phase
                        self._enter_phase(phase)   # also sets train/eval mode
                        self._best_metric = float("inf")
                        self._epochs_no_improve = 0
                        self._best_state = None

                if global_step + 1 <= opt.warmup_steps and not (phased and phase == "joint"):
                    for g in self.optimizer.param_groups:
                        g["lr"] = opt.lr * (global_step + 1) / max(1, opt.warmup_steps)
                    if global_step + 1 == opt.warmup_steps:
                        _log.info("Stage B warmup complete: step=%d lr=%.8g",
                                  global_step + 1, opt.lr)

                with autocast(str(self.device), self._amp_dtype):
                    total, comps = compute_joint_loss(
                        self.accelerator, batch, training_cfg=self.train_cfg,
                        phase=phase,
                        mean_objective=self.config.predictor_mean_objective,
                        target_scale=self.config.predictor_target_scale,
                        isolate_aux=phased and self.config.predictor_aux_isolation,
                    )
                n_text_missing += int(batch.get("text_missing", 0) or 0)
                self.optimizer.zero_grad(set_to_none=True)
                max_norm = self.config.config.memory.max_grad_norm
                if self.scaler is not None:
                    self.scaler.scale(total).backward()
                    self.scaler.unscale_(self.optimizer)
                    finite = all_agree(
                        all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.active_params),
                        self.dctx, device=torch.device(self.device))
                    if not finite:
                        self.scaler.update(new_scale=self.scaler.get_scale() * self.scaler.get_backoff_factor())
                        self.optimizer.zero_grad(set_to_none=True)
                        _log.warning("Stage B step %d: nonfinite gradients; all ranks skip update", global_step)
                        continue
                    # Average across ranks before clipping so all ranks step identically.
                    average_gradients(self.active_params, self.dctx)
                    finite = all_agree(
                        all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.active_params),
                        self.dctx, device=torch.device(self.device))
                    if not finite:
                        self.scaler.update(new_scale=self.scaler.get_scale() * self.scaler.get_backoff_factor())
                        self.optimizer.zero_grad(set_to_none=True)
                        _log.warning("Stage B step %d: gradient reduction overflow; all ranks skip update", global_step)
                        continue
                    torch.nn.utils.clip_grad_norm_(self.active_params, max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total.backward()
                    average_gradients(self.active_params, self.dctx)
                    torch.nn.utils.clip_grad_norm_(self.active_params, max_norm)
                    self.optimizer.step()
                global_step += 1
                self._updates = global_step

                for k, v in comps.items():
                    running[k] = running.get(k, 0.0) + v
                n_batches += 1
                if batch_idx % self.config.log_every == 0:
                    _log.info("  epoch %d/%d batch %d/%d  loss=%.4f  %s",
                              epoch + 1, self.config.num_epochs, batch_idx, len(train_loader),
                              comps["total"], self._fmt(comps))

            avg = {k: self._reduce(v / max(1, n_batches)) for k, v in running.items()}
            _log.info("epoch %d done  avg_total=%.4f", epoch + 1, avg.get("total", 0.0))
            if phases_done:
                _log.info("Phased predictor training complete after %d successful "
                          "updates; final phase best restored", global_step)
                break
            if n_text_missing:
                _log.warning(
                    "epoch %d: %d sample(s) had no prompt embedding, so their CMSC "
                    "terms contributed nothing. Check text_embeds/ against the train "
                    "split.", epoch + 1, n_text_missing,
                )

            # --- validation & early stopping --------------------- #
            if val_loader is not None:
                if self.train_cfg.val_every_epochs <= 0 or (epoch + 1) % self.train_cfg.val_every_epochs != 0:
                    continue
                metrics = self._validate(val_loader)
                _log.info("  val  %s", self._fmt(metrics))
                monitor = self._monitor_for(phase, metrics)
            else:
                monitor = avg.get("total", float("inf"))

            if monitor < self._best_metric - 1e-5:
                self._best_metric = monitor
                self._epochs_no_improve = 0
                self._save_best(phase)
            else:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.train_cfg.early_stop_patience:
                    if not phased:
                        _log.info("early stop after %d evaluations without improvement",
                                  self._epochs_no_improve)
                        break
                    # Phased mode: an exhausted phase finishes early — restore its best and advance.
                    _log.info("phase %s early-stopped after %d evaluations; restoring "
                              "its best and advancing", phase, self._epochs_no_improve)
                    self._restore_best()
                    if phase == "var":
                        self._var_calibrated = True
                        self._restamp_best("var", global_step)
                    successor = self._successor_phase(phase)
                    if successor is None:
                        phases_done = True
                        _log.info("all phases complete (final phase early-stopped; its "
                                  "best restored)")
                        break
                    self._rebase_schedule(phase, global_step)
                    phase = successor
                    self._enter_phase(phase)
                    self._best_metric = float("inf")
                    self._epochs_no_improve = 0
                    self._best_state = None

        if phased and not phases_done:
            # The last batch may exhaust the schedule exactly; detect completion here.
            if self._phase_for_step(global_step) is None:
                phases_done = True
                _log.info("all phases complete: schedule exhausted at %d successful "
                          "updates", global_step)
            # Close out the in-progress phase on its final weights.
            self._finalize_phase(phase, val_loader)
            self._restore_best()
            # The loop may end exactly on the variance phase's last step; σ is then calibrated.
            if phase == "var" and self._var_budget_spent(global_step):
                self._var_calibrated = True
                self._restamp_best("var", global_step)
        if phased and not phases_done:
            if self._var_calibrated:
                _log.info("Stage B stopped at the epoch budget after variance "
                          "calibration (phase %s); optional later phases did not "
                          "run — the checkpoint is calibration-complete", phase)
            else:
                _log.warning("Stage B stopped at the epoch budget mid-phase (%s); "
                             "variance calibration did NOT finish — the checkpoint "
                             "is gated as phase-incomplete", phase)
        self.phase_state = (
            self._phase_state_dict(phase, global_step) if phased else None
        )
        _log.info("Stage B training complete (best=%.4f)", self._best_metric)
        return self.accelerator

    # ------------------------------------------------------------------ #
    # phased predictor training (mean → variance → optional joint)
    # ------------------------------------------------------------------ #

    def _build_phase_schedule(self) -> List[list]:
        """Phase boundaries as cumulative successful-update counts; ``None`` = open-ended."""
        c = self.config
        bounds: List[list] = [["mean", c.predictor_mean_steps]]
        if c.predictor_var_steps:
            bounds.append(["var", bounds[-1][1] + c.predictor_var_steps])
        if c.predictor_joint_lr_scale > 0:
            bounds.append(["joint", None])
        return bounds

    def _phase_for_step(self, steps_completed: int) -> Optional[str]:
        """Phase for ``steps_completed`` updates; ``None`` when all phases are spent."""
        bounds = getattr(self, "_phase_bounds", None) or self._build_phase_schedule()
        for name, end in bounds:
            if end is None or steps_completed < end:
                return name
        return None

    def _rebase_schedule(self, phase: str, steps_completed: int) -> None:
        """End ``phase`` at ``steps_completed`` and re-anchor the later phases."""
        bounds = getattr(self, "_phase_bounds", None)
        if not bounds:
            return
        for i, (name, end) in enumerate(bounds):
            if name != phase:
                continue
            old_prev = end
            bounds[i][1] = steps_completed
            for j in range(i + 1, len(bounds)):
                _, e = bounds[j]
                if e is None:
                    continue
                length = e - old_prev
                bounds[j][1] = bounds[j - 1][1] + length
                old_prev = e
            return

    def _successor_phase(self, phase: str) -> Optional[str]:
        names = [n for n, _ in getattr(self, "_phase_bounds", None) or self._build_phase_schedule()]
        if phase not in names:
            return None
        i = names.index(phase)
        return names[i + 1] if i + 1 < len(names) else None

    @staticmethod
    def _monitor_for(phase: str, metrics: Dict[str, float]) -> float:
        # Score each phase on the metric it optimises.
        if phase == "var":
            return metrics["nll_nonfull"]
        if phase == "mean":
            return metrics["mae_nonfull"]
        return metrics["mae"]

    def _finalize_phase(self, phase: str, val_loader: Optional[DataLoader]) -> None:
        """Close a phase out mid-epoch: validate and checkpoint its best now."""
        if val_loader is None:
            return
        metrics = self._validate(val_loader)
        _log.info("  val (phase %s final)  %s", phase, self._fmt(metrics))
        monitor = self._monitor_for(phase, metrics)
        if monitor < self._best_metric - 1e-5:
            self._best_metric = monitor
            self._epochs_no_improve = 0
            self._save_best(phase)

    def _phase_state_dict(self, phase: str, updates: int) -> Dict[str, Any]:
        """Completion record written into every phased-mode checkpoint.

        ``calibration_complete`` mirrors ``self._var_calibrated`` and gates load_checkpoint.
        """
        bounds = getattr(self, "_phase_bounds", None) or self._build_phase_schedule()
        return {
            "phased": True,
            "phase": phase,
            "updates": int(updates),
            "schedule": [(n, e) for n, e in bounds],
            "calibration_complete": bool(getattr(self, "_var_calibrated", False)),
        }

    def _var_budget_spent(self, updates: int) -> bool:
        """True when ``updates`` has reached the variance phase's end boundary."""
        for name, end in getattr(self, "_phase_bounds", None) or []:
            if name == "var":
                return end is not None and updates >= end
        return False

    def _restamp_best(self, phase: str, updates: int) -> None:
        """Re-write a completed phase's best file with its final phase_state."""
        if not self._phased or self._best_state is None or not self.dctx.is_main:
            return
        ckpt = build_checkpoint(self.accelerator)
        ckpt["phase_state"] = self._phase_state_dict(phase, updates)
        torch.save(ckpt, self.config.checkpoint_dir / f"stage_b_best_{phase}.pt")

    def _save_best(self, phase: str) -> None:
        # Each rank keeps its own in-memory copy so a phase switch needs no file round-trip.
        self._best_state = {k: v.detach().clone() for k, v in self.accelerator.state_dict().items()}
        if self.dctx.is_main:
            name = f"stage_b_best_{phase}.pt" if self._phased else "stage_b_best.pt"
            ckpt = build_checkpoint(self.accelerator)
            if self._phased:
                # Stamp the phase-best file with the run's progress record.
                ckpt["phase_state"] = self._phase_state_dict(
                    phase, getattr(self, "_updates", 0))
            torch.save(ckpt, self.config.checkpoint_dir / name)
            _log.info("  new best (%.4f) → %s", self._best_metric, name)

    def _restore_best(self) -> None:
        if self._best_state is None:
            return
        self.accelerator.load_state_dict(self._best_state)
        _log.info("restored best weights of the completed phase (%.4f)", self._best_metric)

    def _enter_phase(self, phase: str) -> None:
        """Switch the trainable set, module mode and optimizer for the new phase."""
        opt = self.train_cfg.optim
        self._set_var_head_only(phase == "var")
        self.active_params = [p for p in self.trainable_params if p.requires_grad]
        lr = opt.lr if phase != "joint" else opt.lr * self.config.predictor_joint_lr_scale
        self.optimizer = optim.AdamW(self.active_params, lr=lr, betas=opt.betas,
                                     weight_decay=opt.weight_decay)
        if phase == "var":
            self.accelerator.eval()
        else:
            self.accelerator.train()
        _log.info("phase %s: %d trainable parameter(s), lr=%.8g",
                  phase, sum(p.numel() for p in self.active_params), lr)

    def _set_var_head_only(self, var_only: bool) -> None:
        """Freeze every plugin parameter except ``predictor.var_head`` (or restore)."""
        var_params = {id(p) for p in self.accelerator.lcocf.predictor.var_head.parameters()}
        for p in self.trainable_params:
            p.requires_grad_(not var_only or id(p) in var_params)

    def _reduce(self, value: float) -> float:
        """Mean of a per-rank scalar across the job (identity when single-process)."""
        return all_reduce_mean(value, self.dctx, device=torch.device(self.device))

    # ------------------------------------------------------------------ #
    # validation metrics (MAE / cert-violation / budget-hit / smoothness)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.accelerator.eval()
        acc = self.accelerator
        ctx_dim = acc.config.lcocf.predictor.context_dim

        n = 0
        abs_err = cert_viol = budget_hit = smooth = 0.0
        zero_err = sigma_sum = 0.0
        pair_count = 0
        action_errors = [0.0] * 4
        action_counts = [0] * 4
        action_cost = torch.tensor(acc.allocator.action_cost, device=self.device)
        # Effective σ floor of the variance head; ``sigma_floor_frac`` reports how much
        # of the split sits on it.
        pcfg = acc.config.lcocf.predictor
        sigma_floor = math.exp(-5.0) * 1.05 if pcfg.predict_log_variance else 1.05e-4
        nll_sum = floor_hits = 0.0
        n_nf = 0
        nll_nf = cover_nf = floor_nf = 0.0
        for batch in loader:
            tube_features = batch["tube_features"].to(self.device).float()
            strength_features = batch["strength_features"].to(self.device).float()
            actions = batch["action"].to(self.device).long()
            step_frac = batch["step_frac"].to(self.device).float()
            damage_true = damage_scalar_batch(batch["damage_label"].to(self.device).float())

            strength = acc.lcocf.strength_field(strength_features)
            budget = per_sample_budget(acc, batch, device=self.device)
            inp = build_predictor_input_batch(
                states=tube_features, strength_feats=strength_features, strength=strength,
                budget=budget, step_frac=step_frac, step_embed_dim=ctx_dim,
            )
            pred = acc.lcocf.predictor(inp)
            idx = actions.clamp(0, pred.mu.shape[-1] - 1).unsqueeze(-1)
            mu_a = pred.mu.gather(-1, idx).squeeze(-1)
            sigma_a = pred.sigma.gather(-1, idx).squeeze(-1)
            e_cert = acc.raec.certificate.value(
                mu_a, sigma_a, residual=batch_float(batch, "skip_residual", mu_a),
                boundary=tube_features[:, 3], anchor_age=tube_features[:, 6],
                local_cmsc=_local_cmsc_violation(acc, batch, mu_a),
            )
            probs = action_probs(pred.mu)
            expected_cost = (probs * action_cost).sum(-1)

            bs = mu_a.shape[0]
            abs_err += float((mu_a - damage_true).abs().sum())
            zero_err += float(damage_true.abs().sum())
            sigma_sum += float(sigma_a.sum())
            nll_sum += float(gaussian_nll(damage_true, mu_a, sigma_a)) * bs
            floor_hits += float((sigma_a <= sigma_floor).sum())
            nonfull = actions != 0
            nf = int(nonfull.sum())
            if nf:
                # Non-FULL-only calibration view.
                nll_nf += float(gaussian_nll(
                    damage_true[nonfull], mu_a[nonfull], sigma_a[nonfull])) * nf
                cover_nf += float(((mu_a[nonfull] - damage_true[nonfull]).abs()
                                   <= 2 * sigma_a[nonfull]).sum())
                floor_nf += float((sigma_a[nonfull] <= sigma_floor).sum())
                n_nf += nf
            for action in range(4):
                selected = actions == action
                action_counts[action] += int(selected.sum())
                action_errors[action] += float((mu_a[selected] - damage_true[selected]).abs().sum())
            cert_viol += float((e_cert < damage_true).sum())   # cert failed to upper-bound
            budget_hit += float((expected_cost <= budget + 1e-6).sum())
            temporal, pairs = tube_temporal_smoothness(acc, probs, batch, return_pairs=True)
            smooth += float(temporal) * pairs
            pair_count += pairs
            n += bs

        n = max(1, n)
        # Reduce sums and counts separately so uneven rank populations remain weighted.
        global_n = self._reduce(n)
        global_nf = self._reduce(max(1, n_nf))
        global_pairs = self._reduce(pair_count)
        world_size = getattr(getattr(self, "dctx", None), "world_size", 1)
        metrics = {
            "mae": self._reduce(abs_err) / global_n,
            "cert_violation": self._reduce(cert_viol) / global_n,
            "budget_hit": self._reduce(budget_hit) / global_n,
            "zero_baseline_mae": self._reduce(zero_err) / global_n,
            "sigma_mean": self._reduce(sigma_sum) / global_n,
            "nll": self._reduce(nll_sum) / global_n,
            "sigma_floor_frac": self._reduce(floor_hits) / global_n,
            "nll_nonfull": self._reduce(nll_nf) / global_nf,
            "coverage_nonfull": self._reduce(cover_nf) / global_nf,
            "sigma_floor_frac_nonfull": self._reduce(floor_nf) / global_nf,
            "smoothness": self._reduce(smooth) / global_pairs if global_pairs else float("nan"),
            "temporal_pairs": global_pairs * world_size,
        }
        counts = [self._reduce(v) for v in action_counts]
        errors = [self._reduce(v) for v in action_errors]
        for i, name in enumerate(("full", "lowfreq", "interp", "anchor")):
            metrics[f"mae_{name}"] = errors[i] / counts[i] if counts[i] else float("nan")
            metrics[f"n_{name}"] = counts[i] * world_size
        nonfull_count = sum(counts[1:])
        metrics["mae_nonfull"] = sum(errors[1:]) / nonfull_count if nonfull_count else float("nan")
        return metrics

    @staticmethod
    def _fmt(d: Dict[str, float]) -> str:
        return " ".join(f"{k}={v:.4f}" for k, v in d.items() if k != "total")
