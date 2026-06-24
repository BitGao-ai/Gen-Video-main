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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from cocf.common.config import Config, DataConfig
from cocf.common.logging import get_logger
from cocf.core.accelerator import Accelerator
from cocf.data import ProcessedLayout, VideoTextDataset, collate_video_samples
from cocf.engine import InferenceEngine
from cocf.lcocf.predictor import build_predictor_input_batch
from cocf.training.lora import inject_lora
from cocf.training.stage_b_losses import action_probs, budget_penalty
from cocf.training.teacher_forward import TeacherForwardConfig, TeacherForwardRunner

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class StageCConfig:
    """Hyperparameters for Stage C fine-tuning.

    The §4.2 data source is the processed store's ``raw_filtered/`` (preferred,
    ``processed_root``) or a plain video/caption ``manifest_path`` fallback — both
    optional so the pipeline (which threads ``processed_root``) and the standalone
    script (which may pass ``--manifest``) construct this identically. ``config`` is
    the full :class:`~cocf.common.config.Config` so the stage can size budgets / read
    data knobs consistently with the rest of the run.
    """

    # Data — at least one of these resolves the §4.2 source (see :meth:`run`).
    manifest_path: Optional[Path] = None  # fallback video/caption manifest
    processed_root: Optional[Path] = None  # preferred: §3 store (raw_filtered/)
    config: Config = field(default_factory=Config)  # full run config
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

        # Optionally add LoRA adapters to backbone (§4.2 "最后若干层 DiT 的 LoRA 适配器")
        self._lora_params: list = []
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
        # Eval mode: the plugins carry no dropout/BN, so this only skips the predictor's
        # gradient-checkpointing wrapper — keeping the autograd graph simple and
        # deterministic while parameters still receive gradient.
        self.accelerator.eval()

        # Resolve the §4.2 data source: an explicit ``manifest_path`` wins, else the
        # processed store's ``raw_filtered/captions.jsonl`` (written by Stage A).
        # VideoTextDataset reads either as a {path/caption} manifest (its ``meta_file``).
        #
        # NOTE (ambiguous, intentionally not auto-resolved): the frame-sampling /
        # bucketing params fall back to DataConfig defaults; thread the full
        # ``Config.data`` into StageCConfig if Stage C must match the main run.
        meta_file = ""
        if self.config.manifest_path is not None:
            meta_file = str(self.config.manifest_path)
        elif self.config.processed_root is not None:
            captions = ProcessedLayout(self.config.processed_root).raw_filtered_dir / "captions.jsonl"
            meta_file = str(captions)
        else:
            _log.warning("Stage C: no --manifest or --processed-root given; nothing to train on.")
            return self.accelerator
        data_cfg = DataConfig(
            meta_file=meta_file,
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
        if len(dataloader) == 0:
            # No clips resolved (empty/missing manifest). Bail out cleanly instead of
            # dividing by a zero batch count in the epoch-average below.
            _log.warning("Stage C: data source '%s' yielded no batches; skipping fine-tune.", meta_file)
            return self.accelerator

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
        """One §4.2 end-to-end step on a batch of captions.

        Runs the accelerated engine, scores the render against the full-compute
        baseline ``Y_full`` (same initial noise), and adds the schedule regularisers
        recomputed on the engine's real per-step features::

            L = λ_pixel·L1(Y_accel, Y_full)          (§4.2 主损失 — pixel)
              + λ_quality·L_sem(Y_accel, Y_full)      (§4.2 主损失 — 多维语义守恒, §6.3.2)
              + L_reg(schedule)                       (§4.2 正则 — 复用阶段B管平滑/预算)

        The pixel/semantic terms flow gradient through the differentiable decode into
        the residual-repair net (and any LoRA adapters); the schedule term flows into
        the L-COCF strength field + damage predictor — exactly the §4.2 gradient scope.
        """
        bb = self.accelerator.backbone
        captions = list(batch["captions"])
        if not captions:
            return torch.zeros((), device=self.device, requires_grad=True)

        cond, grid, z_init = self._build_inputs(captions)

        # Full-compute baseline on the *same* noise (the §4.2 主损失 reference; no grad).
        y_full = self._full_baseline(z_init, cond, grid)

        # Accelerated render with the differentiable decode + per-step feature capture.
        steps: list = []
        result = self.engine.generate(
            captions, z_init, grid, cond, bb,
            record_sink=lambda **kw: steps.append(kw),
            decode_grad=True,
        )
        y_accel = result.video

        l_pixel = (y_accel - y_full).abs().mean()
        l_quality = self._semantic_loss(y_accel, y_full, captions[0])
        l_reg = self._schedule_reg(steps)
        return (
            self.config.lambda_pixel * l_pixel
            + self.config.lambda_quality * l_quality
            + l_reg
        )

    # ------------------------------------------------------------------ #
    # §4.2 forward building blocks
    # ------------------------------------------------------------------ #

    def _build_inputs(self, captions):
        """Encode captions and sample the shared initial noise / token grid (§7.2)."""
        bb = self.accelerator.backbone
        d = self.config.config.data
        grid = bb.token_grid(d.num_frames, d.height, d.width)
        cond = bb.encode_text(captions).to(self.device)
        z_init = bb.initial_latent(grid, batch=len(captions), device=self.device)
        return cond, grid, z_init

    def _full_baseline(self, z_init, cond, grid) -> Tensor:
        """Decode the un-accelerated baseline ``Y_full`` for the §4.2 main loss, on the
        *same* initial noise as the accelerated run (label-only, no grad)."""
        tf_cfg = TeacherForwardConfig.from_config(self.config.config)
        tf_cfg.num_inference_steps = self.engine.engine_cfg.num_inference_steps
        runner = TeacherForwardRunner(self.accelerator, tf_cfg, device=self.device)
        bb = self.accelerator.backbone
        with torch.no_grad():
            z0, _ = runner.full_denoise(z_init, cond, grid)
            return bb.decode_latent(bb.to_grid(z0, grid))

    def _semantic_loss(self, y_accel: Tensor, y_full: Tensor, prompt: str) -> Tensor:
        """Multi-dimensional cross-modal conservation (§6.3.2) between the accelerated
        render and the full baseline via the injected metric extractor (DINO/CLIP/RAFT).

        The baseline features are detached, so gradient flows only through ``y_accel``
        — the "conserve the reference relations" semantics — training the repair net /
        LoRA toward identity / appearance / motion fidelity of the full render.
        """
        me = self.accelerator.metric_extractor
        if me is None:
            return y_accel.new_zeros(())
        # Accelerated branch keeps the autograd graph + the render's device
        # (``differentiable=True``); the baseline is a detached, no-grad reference.
        dev = y_accel.device
        fa = me.extract(self._to_fchw(y_accel), prompt, differentiable=True)
        ff = me.extract(self._to_fchw(y_full), prompt)
        c = self.accelerator.config.cmsc

        def cos_dev(a: Tensor, b: Tensor) -> Tensor:
            # ``b`` (baseline) may sit on CPU; align it to the accelerated device so the
            # term composes with the on-device pixel/schedule losses.
            return (1.0 - F.cosine_similarity(
                a.float().mean(0).to(dev),
                b.float().mean(0).to(dev), dim=0)).clamp_min(0.0)

        l_id = cos_dev(fa.dino_per_frame, ff.dino_per_frame)      # identity conservation
        l_app = cos_dev(fa.clip_per_frame, ff.clip_per_frame)     # appearance / text-tube
        n = min(fa.flow_mag_per_pair.numel(), ff.flow_mag_per_pair.numel())
        l_mot = (
            (fa.flow_mag_per_pair[:n].to(dev) - ff.flow_mag_per_pair[:n].to(dev))
            .abs().mean()
            if n else y_accel.new_zeros(())
        )                                                          # motion conservation
        return c.lambda_id * l_id + c.lambda_align * l_app + c.lambda_motion * l_mot

    def _schedule_reg(self, steps) -> Tensor:
        """§4.2 regulariser: recompute the Stage-B tube-smoothing + budget penalties on
        the engine's *actual* per-step scheduling features (captured via ``record_sink``),
        differentiable through the strength field and damage predictor — so the §4.2
        end-to-end fine-tune shapes the scheduler on the trajectory it really produced.
        """
        acc = self.accelerator
        total = torch.zeros((), device=self.device)
        ctx = acc.config.lcocf.predictor.context_dim
        cost = torch.tensor(acc.config.allocator.action_cost, device=self.device)
        prev_probs = None
        for st in steps:
            sf, ts = st["strength_feats"], st["tube_states"]
            ids = list(sf)
            if not ids:
                prev_probs = None
                continue
            S = torch.stack([sf[i].as_tensor(self.device) for i in ids])    # [K, 3]
            X = torch.stack([ts[i].as_tensor(self.device) for i in ids])    # [K, 7]
            strength = acc.lcocf.strength_field(S)                          # [K] (grad)
            bud = torch.full((len(ids),), float(st["budget"]), device=self.device)
            frac = torch.full((len(ids),), float(st["step_frac"]), device=self.device)
            mu = acc.lcocf.predictor(
                build_predictor_input_batch(X, S, strength, bud, frac, ctx)
            ).mu                                                            # [K, A] (grad)
            probs = action_probs(mu)
            total = total + acc.config.training.lambda_cost * budget_penalty(probs, cost, bud)
            cur = {i: probs[j] for j, i in enumerate(ids)}
            if prev_probs is not None:
                total = total + acc.config.training.lambda_sta * acc.tube_smoothing(cur, prev_probs)
            prev_probs = cur
        return total

    @staticmethod
    def _to_fchw(video: Tensor) -> Tensor:
        """``[B,3,F,H,W]`` (or ``[3,F,H,W]``) → ``[F,3,H,W]`` in [0,1] for the extractor."""
        v = video[0] if video.dim() == 5 else video
        return v.permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # LoRA + trainable-parameter scope (§4.2)
    # ------------------------------------------------------------------ #

    def _add_lora_adapters(self) -> None:
        """Inject LoRA into the backbone's last-``lora_layers`` DiT blocks (§4.2),
        storing the new trainable params. A logged no-op when the backbone exposes no
        ``dit_blocks()`` (the plugin fine-tune still runs)."""
        self._lora_params, self._lora_modules = inject_lora(
            self.accelerator.backbone,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            last_n_blocks=self.config.lora_layers,
        )

    def _get_trainable_params(self):
        """The §4.2 gradient scope: the L-COCF predictor + strength field + residual
        repair net, the RAEC certificate, the CMSC alignment head, and (optionally) the
        last-N-DiT-block LoRA adapters. The backbone bulk stays frozen.
        """
        params = [p for group in self.accelerator.parameter_groups().values() for p in group]
        params += self._lora_params
        seen, unique = set(), []
        for p in params:  # de-dup (a param may appear in two groups) and keep trainable
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique
