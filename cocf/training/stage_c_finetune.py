"""Stage C: Lightweight fine-tuning (§7.1.3).

Final training stage that fine-tunes the engine ↔ backbone interactions. Unlike
Stage B (which only used isolated counterfactual labels), Stage C runs the full
inference loop and tunes to minimize end-to-end video quality.

Key differences from Stage B:
    - Backbone is mostly frozen, but optionally includes LoRA adapters (rank=8)
    - Trains residual-repair nets more heavily
    - Includes pixel-level L1 loss as an auxiliary objective
    - Runs for only 3 epochs over full video data (not counterfactual pairs)

This stage is optional but recommended for production quality. It converges quickly
(2-3 epochs) because the Stage B initialization is already good.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cocf.common.config import Config, DataConfig
from cocf.common.logging import get_logger
from cocf.core.accelerator import Accelerator
from cocf.data import VideoTextDataset, collate_video_samples
from cocf.engine import InferenceEngine

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class StageCConfig:
    """Hyperparameters for Stage C fine-tuning."""

    # Data
    manifest_path: Path  # Full video dataset manifest
    batch_size: int = 4  # Smaller batches due to full-pipeline overhead
    num_workers: int = 2
    num_epochs: int = 3

    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # LoRA configuration
    use_lora: bool = False  # Optional LoRA on backbone
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_layers: int = 3  # LoRA on last N DiT blocks

    # Loss weights
    lambda_pixel: float = 0.10  # Auxiliary L1 loss on pixels
    lambda_quality: float = 0.90  # Primary quality loss (CMSC, etc.)

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32

    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints/stage_c")
    save_interval: int = 100

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class FinettuneStage:
    """Stage C: End-to-end lightweight fine-tuning (§7.1.3).

    Runs the full accelerated inference engine on real video data and tunes to
    minimize end-to-end quality loss. Optional LoRA adapters on the backbone.

    Key insight: Stage B training converges quickly, so this stage mainly refines
    the residual-repair nets and boundary fusion, with minimal backbone tuning.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        engine: InferenceEngine,
        config: StageCConfig,
    ) -> None:
        self.accelerator = accelerator
        self.engine = engine
        self.config = config
        self.device = config.device

        # Freeze the backbone before anything else (matches Stage B). Stage C tunes
        # the plugins — and optionally LoRA — never the full backbone. But autograd
        # still allocates gradients and retains activations for any parameter left
        # with requires_grad=True, so an unfrozen backbone silently inflates VRAM by
        # a large multiple even though the optimizer never sees those params. Freeze
        # first, then add LoRA so the freshly-inserted LoRA params stay trainable.
        self.accelerator.freeze_backbone()

        # Optionally add LoRA adapters to backbone
        if config.use_lora:
            _log.info(f"Stage C: Adding LoRA adapters (rank={config.lora_rank})")
            self._add_lora_adapters()

        # Identify trainable parameters
        self.trainable_params = self._get_trainable_params()
        _log.info(f"Stage C: {sum(p.numel() for p in self.trainable_params):,} trainable parameters")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def run(self) -> Accelerator:
        """Execute Stage C fine-tuning."""
        _log.info("=== Stage C: Lightweight Fine-tuning ===")

        # Load dataset. VideoTextDataset takes a DataConfig (its ``meta_file`` is the
        # manifest path); ``manifest_path`` / ``num_workers`` were never valid
        # constructor kwargs, so the old call raised TypeError on entry.
        #
        # NOTE (ambiguous, intentionally not auto-resolved): the frame-sampling /
        # bucketing params fall back to DataConfig defaults; thread the full
        # ``Config.data`` into StageCConfig if Stage C must match the main run.
        data_cfg = DataConfig(
            meta_file=str(self.config.manifest_path),
            num_workers=self.config.num_workers,
        )
        dataset = VideoTextDataset(data_cfg)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            # VideoSample is a dataclass; PyTorch's default collate can't stack it.
            # collate_video_samples stacks shape-homogeneous (bucketed) clips into a
            # batch dict, and pin_memory enables a faster async host→device copy on
            # CUDA (a no-op on CPU).
            collate_fn=collate_video_samples,
            pin_memory=(self.device.type == "cuda"),
        )

        _log.info(f"Stage C: {len(dataset)} videos, {len(dataloader)} batches")

        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(dataloader):
                # Run accelerated generation for the batch
                # (in practice, this would be batched; here shown per-video for clarity)
                loss = self._finetune_batch(batch)

                # Backward pass. set_to_none frees the grad tensors between steps
                # instead of zeroing them in place — lower memory held across the
                # step boundary and marginally faster.
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params, self.config.gradient_clip
                )
                self.optimizer.step()

                epoch_loss += float(loss)

                if batch_idx % 10 == 0:
                    # ``batch_idx + 1`` batches have been summed into epoch_loss;
                    # the old ``max(1, batch_idx)`` divisor skewed the running mean.
                    avg_loss = epoch_loss / (batch_idx + 1)
                    _log.info(
                        f"  Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"batch {batch_idx}/{len(dataloader)}, "
                        f"loss: {loss:.4f} (avg: {avg_loss:.4f})"
                    )

                if (batch_idx + 1) % self.config.save_interval == 0:
                    ckpt_path = (
                        self.config.checkpoint_dir
                        / f"stage_c_epoch_{epoch+1}_batch_{batch_idx+1}.pt"
                    )
                    torch.save(self.accelerator.state_dict(), ckpt_path)

            avg_epoch_loss = epoch_loss / len(dataloader)
            _log.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_ckpt = self.config.checkpoint_dir / "stage_c_best.pt"
                torch.save(self.accelerator.state_dict(), best_ckpt)

        _log.info("Stage C fine-tuning complete")
        return self.accelerator

    def _finetune_batch(self, batch) -> Tensor:
        """Forward pass and loss computation for a batch.

        Runs the accelerated engine on batch videos and computes:
            - Quality loss (CMSC, stability, temporal smoothness)
            - Auxiliary pixel loss (optional)
        """
        # Stub: would call self.engine.generate() and compute losses
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to the backbone's final layers."""
        # This is a stub; real impl would use the peft library or similar
        # to wrap the backbone's DiT blocks with LoRA
        pass

    def _get_trainable_params(self):
        """Return all parameters that should be trained in Stage C."""
        trainable = []

        # Always trainable: residual-repair nets, CMSC modules
        for module_name in ["raec", "cmsc"]:
            if hasattr(self.accelerator, module_name):
                module = getattr(self.accelerator, module_name)
                trainable.extend([p for p in module.parameters() if p.requires_grad])

        # Optionally trainable: backbone LoRA
        if self.config.use_lora:
            if hasattr(self.accelerator.backbone, "lora_params"):
                trainable.extend(self.accelerator.backbone.lora_params())

        return trainable
