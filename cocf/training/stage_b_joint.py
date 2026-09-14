"""Stage B: joint module training on the counterfactual store (§4.1).

Trains the four learnable plugins together on the Stage-A samples, minimising::

    L_total = L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc + λ_cost·L_budget

(assembled by :func:`cocf.training.stage_b_losses.compute_joint_loss`). Training is
**backbone-frozen**, so every gradient lands on the tiny plugin parameter set
(strength weights + damage predictor + residual-repair net + certificate coeffs +
CMSC alignment head), keeping VRAM at the plugin footprint (user requirement #1).

Data path (§4.1 读取方式): the §3 level-5 counterfactual LMDB, restricted to the
``train`` split, read through a :class:`~cocf.data.cocf_batch.StratifiedBatchSampler`
that enforces the **action-balanced 1:1:1:1** ratio and mixes scenes / denoising
phases within a batch — planned entirely from the lightweight ``sample_index.csv``
so no payload is read to assemble a batch.

Validation (§4.1 验证环节) runs on the ``val`` split each epoch: degradation-prediction
MAE, certificate-violation rate, budget-hit rate and tube action smoothness; the best
model by MAE is checkpointed and training early-stops after
``training.early_stop_patience`` epochs without improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

    processed_root: Path                      # LCOCF_OpenVid1M_Processed root (§3)
    config: Config = field(default_factory=Config)
    batch_size: int = 32
    num_epochs: int = 10
    num_workers: int = 0                      # 0 keeps LMDB handles single-process safe
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    mixed_precision: bool = False
    checkpoint_dir: Path = Path("./checkpoints/stage_b")
    log_every: int = 20

    def __post_init__(self) -> None:
        self.processed_root = Path(self.processed_root)
        self.checkpoint_dir = Path(self.checkpoint_dir)


class JointTrainingStage:
    """Stage B: joint training of L-COCF / STA / RAEC / CMSC (§4.1)."""

    def __init__(self, accelerator: Accelerator, config: StageBConfig) -> None:
        self.accelerator = accelerator
        self.config = config
        self.device = config.device
        self.train_cfg = config.config.training
        self.layout = ProcessedLayout(config.processed_root)
        # Data-parallel context, derived from torch.distributed's own state: a plain
        # single-process run gets a disabled context and every collective below is a
        # no-op (cocf.training.distributed).
        self.dctx = dist_context()

        # Freeze the backbone (the adapter's weights live on `.module`); only the
        # plugins remain trainable. Done here so the freezing is provable without
        # running the loop.
        self.accelerator.freeze_backbone()
        self.accelerator.to(self.device)

        self.trainable_params: List[torch.nn.Parameter] = self.accelerator.trainable_parameters()
        _log.info("Stage B: %s trainable plugin parameters",
                  f"{sum(p.numel() for p in self.trainable_params):,}")

        opt = self.train_cfg.optim
        self.optimizer = optim.AdamW(
            self.trainable_params, lr=opt.lr, betas=opt.betas, weight_decay=opt.weight_decay
        )
        # AMP needs *both* halves: an autocast region for the forward (where the
        # memory/throughput saving actually comes from) and a loss scaler for the
        # backward. Only the scaler existed, so --mixed-precision bought scaling
        # overhead and nothing else (§P2-9). The forward is wrapped in ``_autocast``
        # below; the scaler is only meaningful for fp16 on CUDA.
        self._amp_dtype = (
            self.config.config.memory.amp_dtype if config.mixed_precision else "none"
        )
        use_scaler = (
            config.mixed_precision
            and str(self.device).startswith("cuda")
            and resolve_dtype(self._amp_dtype) is torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #

    def _build_loader(self, split: str, *, stratified: bool) -> Optional[DataLoader]:
        """DataLoader over one split's LMDB samples (stratified for train; plain for val)."""
        ids = self.layout.read_split(split)
        if not ids:
            return None
        # Prompt embeddings are stored once per clip and joined on read (§P2-3).
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
        # Validation is scored on the *whole* split by every rank and then reduced, so
        # it is not sharded: the metric each rank reports must describe the same data.
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
            # Before the first collective: average_gradients walks this exact list and
            # the epoch loop performs one all-reduce per batch, so a rank that disagrees
            # on either length would not error — it would hang.
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
        global_step = 0
        best_metric = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.config.num_epochs):
            self.accelerator.train()
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            running: Dict[str, float] = {}
            n_batches = 0
            n_text_missing = 0

            for batch_idx, batch in enumerate(train_loader):
                global_step += 1
                if global_step <= opt.warmup_steps:  # include the target-LR boundary
                    for g in self.optimizer.param_groups:
                        g["lr"] = opt.lr * global_step / max(1, opt.warmup_steps)
                    if global_step == opt.warmup_steps:
                        _log.info("Stage B warmup complete: step=%d lr=%.8g", global_step, opt.lr)

                with autocast(str(self.device), self._amp_dtype):
                    total, comps = compute_joint_loss(
                        self.accelerator, batch, training_cfg=self.train_cfg
                    )
                n_text_missing += int(batch.get("text_missing", 0) or 0)
                self.optimizer.zero_grad(set_to_none=True)
                max_norm = self.config.config.memory.max_grad_norm
                if self.scaler is not None:
                    self.scaler.scale(total).backward()
                    self.scaler.unscale_(self.optimizer)
                    finite = all_agree(
                        all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.trainable_params),
                        self.dctx, device=torch.device(self.device))
                    if not finite:
                        self.scaler.update(new_scale=self.scaler.get_scale() * self.scaler.get_backoff_factor())
                        self.optimizer.zero_grad(set_to_none=True)
                        _log.warning("Stage B step %d: nonfinite gradients; all ranks skip update", global_step)
                        continue
                    # Average across ranks *before* clipping, so every rank clips the
                    # same gradient and therefore takes an identical step.
                    average_gradients(self.trainable_params, self.dctx)
                    finite = all_agree(
                        all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.trainable_params),
                        self.dctx, device=torch.device(self.device))
                    if not finite:
                        self.scaler.update(new_scale=self.scaler.get_scale() * self.scaler.get_backoff_factor())
                        self.optimizer.zero_grad(set_to_none=True)
                        _log.warning("Stage B step %d: gradient reduction overflow; all ranks skip update", global_step)
                        continue
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total.backward()
                    average_gradients(self.trainable_params, self.dctx)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm)
                    self.optimizer.step()

                for k, v in comps.items():
                    running[k] = running.get(k, 0.0) + v
                n_batches += 1
                if batch_idx % self.config.log_every == 0:
                    _log.info("  epoch %d/%d batch %d/%d  loss=%.4f  %s",
                              epoch + 1, self.config.num_epochs, batch_idx, len(train_loader),
                              comps["total"], self._fmt(comps))

            avg = {k: self._reduce(v / max(1, n_batches)) for k, v in running.items()}
            _log.info("epoch %d done  avg_total=%.4f", epoch + 1, avg.get("total", 0.0))
            if n_text_missing:
                _log.warning(
                    "epoch %d: %d sample(s) had no prompt embedding, so their CMSC "
                    "terms contributed nothing. Check text_embeds/ against the train "
                    "split.", epoch + 1, n_text_missing,
                )

            # --- validation & early stopping (§4.1) --------------------- #
            # One monitored quantity for the whole run: mixing the training total with
            # the validation MAE (they differ by an order of magnitude) made every
            # non-validation epoch score as "no improvement" and tripped the patience
            # counter on a run that was still converging.
            if val_loader is not None:
                if self.train_cfg.val_every_epochs <= 0 or (epoch + 1) % self.train_cfg.val_every_epochs != 0:
                    continue
                metrics = self._validate(val_loader)
                _log.info("  val  %s", self._fmt(metrics))
                monitor = metrics["mae"]
            else:
                monitor = avg.get("total", float("inf"))

            if monitor < best_metric - 1e-5:
                best_metric = monitor
                epochs_no_improve = 0
                if self.dctx.is_main:
                    torch.save(build_checkpoint(self.accelerator),
                               self.config.checkpoint_dir / "stage_b_best.pt")
                _log.info("  new best (%.4f) → stage_b_best.pt", best_metric)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.train_cfg.early_stop_patience:
                    _log.info("early stop after %d evaluations without improvement",
                              epochs_no_improve)
                    break

        _log.info("Stage B training complete (best=%.4f)", best_metric)
        return self.accelerator

    def _reduce(self, value: float) -> float:
        """Mean of a per-rank scalar across the job (identity when single-process)."""
        return all_reduce_mean(value, self.dctx, device=torch.device(self.device))

    # ------------------------------------------------------------------ #
    # validation metrics (§4.1: MAE / cert-violation / budget-hit / smoothness)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.accelerator.eval()
        acc = self.accelerator
        ctx_dim = acc.config.lcocf.predictor.context_dim

        n = 0
        abs_err = cert_viol = budget_hit = smooth = 0.0
        action_cost = torch.tensor(acc.allocator.action_cost, device=self.device)
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
            cert_viol += float((e_cert < damage_true).sum())   # cert failed to upper-bound
            budget_hit += float((expected_cost <= budget + 1e-6).sum())
            smooth += float(tube_temporal_smoothness(acc, probs, batch)) * bs
            n += bs

        n = max(1, n)
        return {
            "mae": self._reduce(abs_err / n),
            "cert_violation": self._reduce(cert_viol / n),
            "budget_hit": self._reduce(budget_hit / n),
            "smoothness": self._reduce(smooth / n),
        }

    @staticmethod
    def _fmt(d: Dict[str, float]) -> str:
        return " ".join(f"{k}={v:.4f}" for k, v in d.items() if k != "total")
