"""Stage B: Joint module training (§7.1.2).

Trains all four learnable components together on the Stage A counterfactual data:
    - L-COCF predictor (damage regression + uncertainty)
    - STA tube encoder + action smoothing
    - RAEC error certificate predictor
    - CMSC text-tube alignment & conservation loss

The combined loss (§7.1.2) is:
    L_total = L_cocf + 0.2·L_tube + 0.1·L_cert + 0.3·L_cmsc + 0.1·L_budget

Training is **backbone-frozen**, so all FLOPs are on the tiny plugin parameters:
the 3 strength scalars, the damage predictor MLP, residual-repair net, and alignment
head (total ~千万 params).

This stage is the bulk of training; it should run for ~10 epochs over Stage A data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.data import LatentCacheDataset
from cocf.core.accelerator import Accelerator

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class StageBConfig:
    """Hyperparameters for Stage B joint training."""

    # Data
    cache_dir: Path  # Path to Stage A cache
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 10

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    gradient_clip: float = 1.0

    # Loss weights (from §7.1.2)
    loss_weights: Dict[str, float] = None

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    mixed_precision: bool = False

    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints/stage_b")
    save_interval: int = 500  # Save every N batches

    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                "cocf": 1.0,
                "tube": 0.2,
                "cert": 0.1,
                "cmsc": 0.3,
                "budget": 0.1,
            }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class JointTrainingStage:
    """Stage B: Joint training of L-COCF, STA, RAEC, CMSC (§7.1.2).

    Training loop that:
        1. Loads Stage A counterfactual data
        2. For each batch:
           a. Forward pass through predictor + feature extractors
           b. Compute L_cocf (damage regression) + L_tube + L_cert + L_cmsc + L_budget
           c. Backward pass, gradient clip, optimizer step
        3. Log efficiency metrics & save checkpoints
        4. Return trained accelerator

    All backbone parameters are frozen (requires_grad=False).
    """

    def __init__(
        self,
        accelerator: Accelerator,
        config: StageBConfig,
    ) -> None:
        self.accelerator = accelerator
        self.config = config
        self.device = config.device

        # Freeze backbone. The backbone is an *adapter*, not an nn.Module — its
        # trainable weights live on `.module`. Use the accelerator's helper, which
        # freezes `self.backbone.module` (a no-op-safe call).
        self.accelerator.freeze_backbone()

        # Trainable parameters: L-COCF (strength + predictor), STA, RAEC, CMSC
        self.trainable_params = [
            p for p in accelerator.parameters() if p.requires_grad
        ]
        _log.info(f"Stage B: {sum(p.numel() for p in self.trainable_params):,} trainable parameters")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Optional mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

    def run(self) -> Accelerator:
        """Execute Stage B training and return the updated accelerator."""
        _log.info("=== Stage B: Joint Module Training ===")

        # Load dataset. LatentCacheDataset reads every cached .pt under cache_dir;
        # it has no train/val split concept, so we must not pass a bogus `paths`
        # (a str would be iterated character-by-character into fake file paths).
        dataset = LatentCacheDataset(
            cache_dir=str(self.config.cache_dir),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

        _log.info(f"Stage B: {len(dataset)} training samples, {len(dataloader)} batches")

        global_step = 0
        best_loss = float("inf")

        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                global_step += 1

                # Learning rate warmup
                if global_step < self.config.warmup_steps:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = (
                            self.config.learning_rate
                            * global_step
                            / self.config.warmup_steps
                        )

                # Forward pass
                loss_dict = self._forward_batch(batch)
                total_loss = sum(loss_dict.values())

                # Backward pass
                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, self.config.gradient_clip
                    )
                    self.optimizer.step()

                epoch_loss += float(total_loss)
                num_batches += 1

                # Logging & checkpointing
                if batch_idx % 50 == 0:
                    avg_loss = epoch_loss / num_batches
                    _log.info(
                        f"  Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"batch {batch_idx}/{len(dataloader)}, "
                        f"loss: {total_loss:.4f} (avg: {avg_loss:.4f})"
                    )
                    _log.debug(f"    Loss components: {loss_dict}")

                if global_step % self.config.save_interval == 0:
                    ckpt_path = (
                        self.config.checkpoint_dir
                        / f"stage_b_step_{global_step:06d}.pt"
                    )
                    torch.save(self.accelerator.state_dict(), ckpt_path)
                    _log.info(f"    Saved checkpoint to {ckpt_path}")

            avg_epoch_loss = epoch_loss / num_batches
            _log.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_ckpt = (
                    self.config.checkpoint_dir / "stage_b_best.pt"
                )
                torch.save(self.accelerator.state_dict(), best_ckpt)
                _log.info(f"  New best loss! Saved to {best_ckpt}")

        _log.info("Stage B training complete")
        return self.accelerator

    def _forward_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute all loss components for a batch.

        Returns:
            Dictionary with keys: cocf, tube, cert, cmsc, budget (each weighted).
        """
        # Real impl would:
        #   1. Forward batch through accelerator
        #   2. Compute damage predictions from L-COCF
        #   3. Regress against ground-truth damage labels (from Stage A)
        #   4. Compute tube smoothing loss (STA)
        #   5. Compute certificate training loss (RAEC)
        #   6. Compute semantic conservation loss (CMSC)
        #   7. Compute budget constraint penalty
        #
        # Until that exists, this MUST NOT return constant tensors: they carry no
        # grad_fn, so `total_loss.backward()` raises, and even if it didn't the
        # optimizer would "train" on a meaningless zero loss. Fail loudly instead
        # of silently no-op'ing the entire Stage-B training run.
        raise NotImplementedError(
            "Stage-B loss computation is not implemented yet. Wire the batch "
            "through the accelerator and compute the cocf/tube/cert/cmsc/budget "
            "losses against the Stage-A labels before training; constant tensors "
            "have no grad_fn and cannot be backpropagated."
        )
