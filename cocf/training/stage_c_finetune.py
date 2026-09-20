"""Stage C: lightweight end-to-end fine-tuning.

Runs the full accelerated inference loop on real video data and tunes the plugins
(and optional LoRA adapters) to minimise end-to-end quality loss against the
full-compute render. The backbone stays frozen; it converges in a few epochs from
the Stage-B initialisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from cocf.backbones.base import TextConditioning
from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.memory import free_memory, set_gradient_checkpointing
from cocf.common.types import Action, TubeState
from cocf.core.accelerator import Accelerator
from cocf.data import (
    HardSamplePrioritySampler,
    ProcessedLayout,
    RawFilteredDataset,
    collate_raw_filtered,
)
from cocf.engine import InferenceEngine
from cocf.lcocf.damage import (
    DAMAGE_DIMENSIONS,
    DEFAULT_DAMAGE_WEIGHTS,
    MultiDimDamageComputer,
    VideoFeatures,
)
from cocf.training.checkpoint import build_checkpoint
from cocf.training.distributed import (
    all_agree,
    all_reduce_mean,
    assert_same,
    average_gradients,
    broadcast_parameters,
    context as dist_context,
)
from cocf.training.lora import inject_lora
from cocf.training.stage_c_losses import (
    StepRecord,
    build_cmsc_observation,
    cmsc_quality_loss,
    collate_step_records,
    stage_c_regularizers,
)
from cocf.training.teacher_forward import TeacherForwardConfig, TeacherForwardRunner

Tensor = torch.Tensor
_log = get_logger(__name__)

# Activation headroom one Stage-C clip needs on top of the frozen weights, split into
# the terms the autograd graph is actually made of.
_GIB_CHECKPOINT_STASH = 5.0   # one block input per block, per retained BPTT segment
_GIB_BLOCK_RECOMPUTE = 3.0    # the in-block peak while a checkpointed block reruns
_GIB_DECODE_PER_SLOT = 0.375  # retained VAE decode graph, per latent slot on the graph
_GIB_DECODE_FULL = 12.0       # decode_grad_frames=0 ⇒ the whole clip is on the graph


# One render is one smoothing group; every record comes from the same trajectory.
_RENDER_GROUP = "__render__"

_DAMAGE_COMPUTER = MultiDimDamageComputer()


def _step_records(
    *, step_idx: int, t: int, budget: float, step_frac: float,
    tube_states: Dict[int, TubeState], strength_feats, actions,
    tube_residual: Optional[Dict[int, float]] = None,
    local_cmsc: Optional[Dict[int, float]] = None,
) -> List[StepRecord]:
    """Fan one ``record_sink`` callback out into per-(tube, step) :class:`StepRecord`s."""
    tube_residual = tube_residual or {}
    local_cmsc = local_cmsc or {}
    out: List[StepRecord] = []
    for tid, state in tube_states.items():
        feats = strength_feats.get(tid)
        if feats is None:
            continue
        out.append(
            StepRecord(
                tube_features=state.as_tensor(),
                strength_features=feats.as_tensor(),
                action=int(actions.get(tid, Action.FULL)),
                step_frac=float(step_frac),
                budget=float(budget),
                tube_id=int(tid),
                timestep=int(t),
                video_id=_RENDER_GROUP,
                interaction_density=float(state.interaction),
                skip_residual=float(tube_residual.get(tid, 0.0)),
                local_cmsc=float(local_cmsc.get(tid, 0.0)),
            )
        )
    return out


def _scalar_damage(full: VideoFeatures, accel: VideoFeatures) -> float:
    """Realised end-to-end degradation of the accelerated render, as one scalar.

    Uses the same weighted reduction over the damage axes as Stage A/B so the stages
    agree on what "damage" means. The reference is co-located onto the accelerated
    branch's device first.
    """
    accel = accel.detached()
    ref = full.detached().to(accel.dino_per_frame.device)
    per_axis = _DAMAGE_COMPUTER.compute(ref, accel)
    return min(
        1.0,
        sum(per_axis.get(a, 0.0) * DEFAULT_DAMAGE_WEIGHTS.get(a, 0.0)
            for a in DAMAGE_DIMENSIONS),
    )


@dataclass
class StageCConfig:
    """Hyperparameters for Stage C fine-tuning.

    The data source is the processed store's ``raw_filtered/`` (``processed_root``) or a
    plain video/caption ``manifest_path`` fallback; ``config`` is the full run config.
    """

    # Data — at least one of these resolves the source (see :meth:`run`).
    manifest_path: Optional[Path] = None  # fallback video/caption manifest
    processed_root: Optional[Path] = None  # preferred: processed store (raw_filtered/)
    config: Config = field(default_factory=Config)  # full run config
    batch_size: int = 1  # Engine owns one semantic-tube trajectory per call.
    num_workers: int = 2
    num_epochs: int = 3
    # Denoising steps for the accelerated render. ``None`` (default) matches the schedule
    # Stage A rendered Y_full with, keeping the persisted baseline a valid target; any
    # other value discards the cached baseline and re-denoises every batch.
    num_inference_steps: Optional[int] = None

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



class FinettuneStage:
    """Stage C: end-to-end lightweight fine-tuning.

    Runs the full accelerated inference engine on real video data and tunes the plugins
    (and optional LoRA adapters) to minimise end-to-end quality loss.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        engine: InferenceEngine,
        config: StageCConfig,
    ) -> None:
        if config.batch_size != 1:
            raise ValueError("Stage C requires batch_size=1: semantic-tube state is per video")
        self.accelerator = accelerator
        self.engine = engine
        self.config = config
        self.device = config.device
        self._warned_baseline_mismatch = False
        self._warned_no_grad_batch = False

        # Pin the accelerated render to Stage A's schedule unless the caller overrode it.
        self._resolve_inference_steps()

        # Freeze the backbone first, then add LoRA so the new LoRA params stay trainable.
        self.accelerator.freeze_backbone()

        # Place the learnable plugins on the run device to match the backbone/z_init.
        self.accelerator.to(self.device)

        # Optionally add LoRA adapters to the last few DiT blocks.
        self._lora_params: list = []
        if config.use_lora:
            _log.info(f"Stage C: Adding LoRA adapters (rank={config.lora_rank})")
            self._add_lora_adapters()
        # Unconditional: the repair net puts ``z`` on the graph even without LoRA.
        self._enable_backbone_checkpointing()

        free_gib = self._free_vram_gib()
        need = self._headroom_gib()
        if free_gib is not None and free_gib < need:
            _log.warning(
                "Stage C: %.1f GiB free, estimated single-video headroom %.1f GiB; "
                "reduce decode window or resolution before training.",
                free_gib, need,
            )

        # Identify trainable parameters
        self.trainable_params = self._get_trainable_params()
        _log.info(f"Stage C: {sum(p.numel() for p in self.trainable_params):,} trainable parameters")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _resolve_inference_steps(self) -> None:
        """Settle the accelerated render's step count (see ``StageCConfig.num_inference_steps``)."""
        teacher_steps = int(self.config.config.teacher.num_inference_steps)
        want = self.config.num_inference_steps
        if want is None:
            want = teacher_steps
        want = int(want)
        self.config.config.engine.num_inference_steps = want
        self.engine.engine_cfg.num_inference_steps = want
        if want != teacher_steps:
            _log.warning(
                "Stage C: rendering at %d steps but Stage A generated Y_full at %d. The "
                "persisted baseline is not a valid target for a different schedule, so "
                "every batch will re-denoise one — roughly doubling step time and peak "
                "memory. Drop the explicit step override to reuse it.",
                want, teacher_steps,
            )
        else:
            _log.info("Stage C: rendering at %d steps (matches Stage A's baseline)", want)

    def _free_vram_gib(self) -> Optional[float]:
        """GiB left on the compute device after the frozen weights, or ``None`` off-CUDA.

        Read after the backbone is placed, so it reflects the real residual budget the
        activations have to fit into rather than the card's nameplate capacity.
        """
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return None
        # Read the current device, not card 0 (a bare "cuda" has index None).
        idx = torch.device(self.device).index
        idx = torch.cuda.current_device() if idx is None else idx
        total = torch.cuda.get_device_properties(idx).total_memory
        return (total - torch.cuda.memory_allocated(idx)) / 1024 ** 3

    def _stash_gib(self) -> float:
        """Retained per-block checkpoint inputs, one set per BPTT segment held."""
        return _GIB_CHECKPOINT_STASH * max(1, self.config.config.engine.grad_window_steps)

    def _decode_gib(self) -> float:
        """Retained VAE decode graph — linear in the slots decoded on the graph."""
        k = int(self.config.config.engine.decode_grad_frames)
        return _GIB_DECODE_PER_SLOT * k if k > 0 else _GIB_DECODE_FULL

    def _headroom_gib(self) -> float:
        """Activation headroom one clip needs, for this run's graph bounds."""
        return self._stash_gib() + _GIB_BLOCK_RECOMPUTE + self._decode_gib()

    def _enable_backbone_checkpointing(self) -> None:
        """Turn on activation checkpointing in every DiT expert.

        Unconditional (gated only by ``MemoryConfig.gradient_checkpointing``): the
        residual-repair net puts ``z`` on the graph, so autograd retains block inputs
        whether or not the DiT is trainable. Applied to every denoiser via
        :meth:`BackboneAdapter.lora_roots`, not just the primary transformer.
        """
        if not self.config.config.memory.gradient_checkpointing:
            return
        bb = self.accelerator.backbone
        roots = list(bb.lora_roots()) if hasattr(bb, "lora_roots") else []
        if not roots:
            module = bb.module
            roots = [("module", module)] if module is not None else []
        total = 0
        params = 0
        for name, module in roots:
            n = set_gradient_checkpointing(module, True)
            total += n
            params += sum(p.numel() for p in module.parameters())
            _log.info("Stage C: activation checkpointing on '%s' → %d module(s)", name, n)
        if total or params < 10 ** 8:
            # A mock/toy adapter needs no checkpointing; only warn for a real DiT.
            return
        _log.warning(
            "Stage C: a %.1fB-parameter backbone accepted no activation checkpointing. "
            "Peak activation memory is then bounded only by engine.grad_window_steps "
            "(=%d) and this will almost certainly OOM. Check the installed diffusers "
            "version exposes enable_gradient_checkpointing().",
            params / 1e9, self.config.config.engine.grad_window_steps,
        )

    def run(self) -> Accelerator:
        """Execute Stage C fine-tuning."""
        _log.info("=== Stage C: Lightweight Fine-tuning ===")
        # Eval mode: the plugins carry no dropout/BN; parameters still receive gradient.
        self.accelerator.eval()

        # Resolve the data source. ``RawFilteredDataset`` reads captions as metadata only
        # and carries each clip's ``video_id``, used to fetch Stage A's cached ``Y_full``.
        if self.config.manifest_path is None and self.config.processed_root is None:
            _log.warning("Stage C: no --manifest or --processed-root given; nothing to train on.")
            return self.accelerator
        source = self.config.manifest_path or self.config.processed_root
        dataset = RawFilteredDataset(
            processed_root=self.config.processed_root,
            manifest_path=self.config.manifest_path,
        )
        # Hard-sample priority: multi / occlusion / text / face clips are up-weighted. The
        # sampler draws with replacement so every rank keeps the same step count.
        dctx = dist_context()
        sampler = HardSamplePrioritySampler(
            dataset.scene_types, seed=self.config.config.seed,
            rank=dctx.rank, world_size=dctx.world_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler=sampler,
            drop_last=True,
            # Identity collate: the engine runs per clip because Y_full is per video.
            collate_fn=collate_raw_filtered,
        )

        if dctx.enabled:
            # Before the first collective: sync parameters from rank 0 and assert that all
            # ranks agree on the parameter and batch counts (a mismatch would hang).
            assert_same(len(self.trainable_params), "trainable-parameter count", dctx)
            assert_same(len(dataloader), "batches per epoch", dctx)
            n = broadcast_parameters(
                list(self.trainable_params) + [b for b in self.accelerator.buffers()],
                dctx,
            )
            _log.info(
                "Stage C: rank %d/%d, %d clips in this shard (%d batches); "
                "%d tensor(s) synced from rank 0",
                dctx.rank, dctx.world_size, len(sampler), len(dataloader), n,
            )
        else:
            _log.info("Stage C: %d clips, %d batches (hard scenes up-weighted)",
                      len(dataset), len(dataloader))
        if len(dataloader) == 0:
            # No clips resolved (empty/missing manifest); bail out cleanly.
            _log.warning("Stage C: data source '%s' yielded no batches; skipping fine-tune.", source)
            return self.accelerator

        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            trained_batches = 0
            # Reseed the sampler so each epoch draws different indices.
            sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                # A per-clip failure is local to one rank; fail softly then agree via
                # ``all_agree`` so every rank runs the same collective sequence.
                try:
                    loss = self._finetune_batch(batch)
                    ok = True
                except RuntimeError as exc:   # OOM included: it subclasses RuntimeError
                    _log.warning("Stage C: batch %d failed on rank %d (%s: %s); "
                                 "skipping it on every rank",
                                 batch_idx, dctx.rank, type(exc).__name__, exc)
                    loss, ok = torch.zeros((), device=self.device), False
                    free_memory()
                self.optimizer.zero_grad(set_to_none=True)
                if not all_agree(ok, dctx, device=torch.device(self.device)):
                    continue

                # A batch can produce a loss with no autograd graph (no repair fired, no
                # tube, LoRA off); skip backward but keep the collective sequence identical
                # across ranks.
                if loss.requires_grad:
                    loss.backward()
                # Average across ranks before clipping so all ranks step identically.
                average_gradients(self.trainable_params, dctx)
                trained = all_reduce_mean(
                    float(loss.requires_grad), dctx, device=torch.device(self.device)
                ) > 0.0
                if trained:
                    trained_batches += 1
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, self.config.gradient_clip
                    )
                    self.optimizer.step()
                elif not self._warned_no_grad_batch:
                    self._warned_no_grad_batch = True  # once per run, not per batch
                    _log.warning(
                        "Stage C: batch %d produced a loss with no autograd graph on "
                        "any rank — no repair fired, no tube was segmented and LoRA is "
                        "off, so nothing is trainable from it. Skipping the optimiser "
                        "step (this message is not repeated). If it is the common case, "
                        "the run is not learning: raise engine.grad_window_steps or "
                        "lower the skip pressure (budget.b_min).", batch_idx,
                    )

                epoch_loss += float(loss)

                if batch_idx % 10 == 0 and dctx.is_main:
                    # Divide by the number of batches summed so far.
                    avg_loss = epoch_loss / (batch_idx + 1)
                    _log.info(
                        f"  Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"batch {batch_idx}/{len(dataloader)}, "
                        f"loss: {loss:.4f} (avg: {avg_loss:.4f})"
                    )

                if (batch_idx + 1) % self.config.save_interval == 0 and dctx.is_main:
                    # Created on first write, not in __post_init__.
                    self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = (
                        self.config.checkpoint_dir
                        / f"stage_c_epoch_{epoch+1}_batch_{batch_idx+1}.pt"
                    )
                    torch.save(self.checkpoint(), ckpt_path)

            # Reduced across ranks so every rank agrees on the epoch mean.
            avg_epoch_loss = all_reduce_mean(
                epoch_loss / len(dataloader), dctx, device=torch.device(self.device)
            )
            # Report how many batches actually trained, not just the loss.
            if dctx.is_main:
                _log.info(
                    "Epoch %d complete. Average loss: %.4f (optimiser stepped on "
                    "%d/%d batches)",
                    epoch + 1, avg_epoch_loss, trained_batches, len(dataloader),
                )
            if trained_batches == 0:
                _log.error(
                    "Stage C: epoch %d trained on 0 of %d batches — no batch carried "
                    "an autograd graph, so the checkpoint this epoch writes is the one "
                    "it started from. With --use_lora off, the only path from the main "
                    "loss to the backbone is the residual-repair net, which has to fire "
                    "*inside* the BPTT window: raise engine.grad_window_steps "
                    "(--grad-window-steps), lower the skip pressure (budget.b_min), or "
                    "turn LoRA on if the card has room.", epoch + 1, len(dataloader),
                )

            # Every rank computes the same all-reduced mean; only rank 0 writes.
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                if dctx.is_main:
                    self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    best_ckpt = self.config.checkpoint_dir / "stage_c_best.pt"
                    torch.save(self.checkpoint(), best_ckpt)

        if dctx.is_main:
            _log.info("Stage C fine-tuning complete")
        return self.accelerator

    def _finetune_batch(self, batch) -> Tensor:
        """One end-to-end step on a batch of captions.

        Runs the accelerated engine, scores the render against the full-compute baseline
        ``Y_full`` (same initial noise), and adds the schedule regularisers recomputed on
        the engine's per-step features::

            L = λ_pixel·L_pixel + λ_quality·L_CMSC + λ_sta·L_tube + λ_cert·L_cert
              + λ_cost·L_budget

        Every term is assembled by :mod:`cocf.training.stage_c_losses`. The pixel/CMSC
        terms flow gradient into the residual-repair net (and LoRA); the regularisers
        into the strength field, damage predictor and certificate coefficients.
        """
        bb = self.accelerator.backbone
        captions = [item.caption for item in batch]
        if not captions:
            return torch.zeros((), device=self.device, requires_grad=True)

        # Reuse Stage A's persisted baseline when it exists (z_T and Y_full together),
        # else recompute the full-compute reference.
        video_ids = [item.video_id for item in batch]
        grid = bb.token_grid(*self._frame_shape())
        z_init = self._cached_baseline(video_ids, grid)
        y_full = None  # cached path: read after the render, at the decode window only
        cond = self._cached_cond(video_ids, captions)
        if z_init is not None:
            if cond is None:
                cond = bb.encode_text(captions).to(self.device)
        else:
            cond, grid, z_init = self._build_inputs(captions, cond=cond)
            y_full = self._full_baseline(z_init, cond, grid)

        # Accelerated render with the differentiable decode + per-step feature capture.
        records: list = []
        result = self.engine.generate(
            captions, z_init, grid, cond, bb,
            record_sink=lambda **kw: records.extend(_step_records(**kw)),
            decode_grad=True,
        )
        y_accel = result.video
        # A windowed decode covers only part of the clip, so cut the reference to the same
        # pixel frames before comparing.
        if y_full is None:
            y_full = self._windowed_reference(video_ids, result.frame_span)
            if y_full is None:
                # The bucket disappeared between the pre-render check and now.
                y_full = self._align_reference(
                    self._full_baseline(z_init, cond, grid), result.frame_span
                )
        else:
            y_full = self._align_reference(y_full, result.frame_span)
        self._check_alignment(y_accel, y_full, result.frame_span)

        # --- main loss: pixel + the conservation loss --- #
        l_pixel = F.l1_loss(y_accel.float(), y_full.float().to(y_accel.device))
        l_quality, cmsc_terms = self._conservation_loss(
            y_accel, y_full, captions, result, text_embeds=cond.embeds
        )
        l_reg, _ = self._regularizers(records, cmsc_terms.get("measured_damage", 0.0))
        _log.info(
            "Stage C loss: pixel=%.6f (grad=%s) quality=%.6f (grad=%s) "
            "regularizer=%.6f (grad=%s) video_grad=%s records=%d",
            float(l_pixel.detach()), l_pixel.requires_grad,
            float(l_quality.detach()), l_quality.requires_grad,
            float(l_reg.detach()), l_reg.requires_grad, y_accel.requires_grad, len(records),
        )
        return (
            self.config.lambda_pixel * l_pixel
            + self.config.lambda_quality * l_quality
            + l_reg
        )

    @staticmethod
    def _align_reference(y_full: Tensor, frame_span) -> Tensor:
        """Slice ``Y_full`` ``[B,3,F,H,W]`` to the pixel frames the render covers.

        ``frame_span is None`` leaves it untouched.
        """
        if frame_span is None:
            return y_full
        start, stop = frame_span
        return y_full[:, :, start:stop]

    @staticmethod
    def _check_alignment(y_accel: Tensor, y_full: Tensor, frame_span) -> None:
        """Fail loudly when the render and its reference do not describe the same frames.

        Both main-loss terms compare these element-wise, so a geometry mismatch would
        otherwise surface as an opaque broadcast error (or train on nonsense silently).
        """
        if y_accel.shape == y_full.shape:
            return
        raise RuntimeError(
            f"Stage C: the accelerated render {tuple(y_accel.shape)} and the "
            f"full-compute reference {tuple(y_full.shape)} do not cover the same frames "
            f"(differentiable-decode window frame_span={frame_span}). The §4.2 pixel and "
            f"§6.3.2 conservation losses compare them element-wise, so this cannot be "
            f"scored. Check that the backbone's pixel_span() reports the frame count "
            f"decode_latent() really emits for that window, and that Stage A's Y_full "
            f"was generated at this run's data.num_frames/height/width."
        )

    # ------------------------------------------------------------------ #
    # forward building blocks
    # ------------------------------------------------------------------ #

    def _conservation_loss(self, y_accel, y_full, captions, result, *, text_embeds=None):
        """``L_CMSC`` averaged over every clip in the batch.

        Returns ``(loss, {per-term components, measured_damage})``, where
        ``measured_damage`` is the realised degradation the certificate regulariser
        calibrates against.
        """
        me = self.accelerator.metric_extractor
        if me is None or not result.tubes:
            return y_accel.new_zeros(()), {}
        grid = result.grid or self.accelerator.backbone.token_grid(*self._frame_shape())
        # Reuse the conditioning the caller already encoded instead of re-encoding.
        text = text_embeds if text_embeds is not None else \
            self.accelerator.backbone.encode_text(captions).embeds

        total = y_accel.new_zeros(())
        comps: Dict[str, float] = {}
        damages: list = []
        n = min(y_accel.shape[0], len(captions))
        _log.info("CMSC geometry: decoded_frames=%d full_frames=%d span=%s latent_slots=%d tubes=%d",
                  y_accel.shape[2], self._frame_shape()[0], result.frame_span, grid.t,
                  len(result.tubes))
        for b in range(n):
            # The accelerated branch keeps the autograd graph; the baseline is detached.
            accel_obs = build_cmsc_observation(
                me, self.accelerator.perception, self._to_fchw(y_accel, b),
                captions[b], result.tubes, grid, text[b], differentiable=True,
                frame_span=result.frame_span, full_frame_count=self._frame_shape()[0],
            )
            full_obs = build_cmsc_observation(
                me, self.accelerator.perception, self._to_fchw(y_full, b),
                captions[b], result.tubes, grid, text[b],
                frame_span=result.frame_span, full_frame_count=self._frame_shape()[0],
            )
            loss_b, comps_b = cmsc_quality_loss(
                self.accelerator.cmsc_loss, full_obs, accel_obs
            )
            total = total + loss_b
            for k, v in comps_b.items():
                comps[k] = comps.get(k, 0.0) + v / n
            damages.append(_scalar_damage(full_obs.video, accel_obs.video))
        comps["measured_damage"] = float(sum(damages) / max(1, len(damages)))
        return total / n, comps

    def _regularizers(self, records, measured_damage: float):
        """Reuse the Stage-B tube-smoothing / certificate / budget terms on the engine's
        per-step scheduling features (captured via ``record_sink``).
        """
        batch = collate_step_records(records)
        if not batch:
            return torch.zeros((), device=self.device), {}
        target = torch.tensor(float(measured_damage), device=self.device)
        return stage_c_regularizers(
            self.accelerator, batch, target,
            training_cfg=self.accelerator.config.training,
        )

    def _frame_shape(self):
        d = self.config.config.data
        return d.num_frames, d.height, d.width

    def _cached_baseline(self, video_ids, grid):
        """Stage A's cached ``z_T`` for the batch, or ``None`` to recompute.

        Requires every clip to have both halves of the baseline bucket; only ``z_T`` is
        returned here and ``Y_full`` is fetched afterwards by :meth:`_windowed_reference`.
        Returns ``None`` for a store built without a baseline or at a different geometry.
        """
        if self.config.processed_root is None:
            return None
        # The baseline is only valid for the schedule it was rendered with; require the
        # step counts to agree.
        teacher_steps = self.config.config.teacher.num_inference_steps
        if self.engine.engine_cfg.num_inference_steps != teacher_steps:
            return None
        layout = ProcessedLayout(self.config.processed_root)
        want = (grid.num_tokens, self.accelerator.backbone.hidden_dim)
        zs = []
        for vid in video_ids:
            z = layout.load_z_init(vid, device=self.device)
            if z is None or not layout.has_y_full(vid):
                return None
            if tuple(z.shape[-2:]) != want:
                if not self._warned_baseline_mismatch:
                    self._warned_baseline_mismatch = True  # once per run, not per batch
                    _log.warning(
                        "Stage C: %s was generated at token geometry %s but this run uses "
                        "%s — recomputing the baseline instead of reusing it (this "
                        "message is not repeated).",
                        vid, tuple(z.shape[-2:]), want,
                    )
                return None
            zs.append(z.reshape(1, *z.shape[-2:]))
        return torch.cat(zs, dim=0)

    def _windowed_reference(self, video_ids, frame_span) -> Optional[Tensor]:
        """Stage A's ``Y_full`` for the batch, read at ``frame_span`` only.

        Returns ``[B, 3, f, H, W]`` already cut to the frames the decode covers (no
        :meth:`_align_reference` needed), or ``None`` if any clip's baseline is missing.
        """
        if self.config.processed_root is None:
            return None
        layout = ProcessedLayout(self.config.processed_root)
        frames = None if frame_span is None else (int(frame_span[0]), int(frame_span[1]))
        ys = []
        for vid in video_ids:
            y = layout.load_y_full(vid, device=self.device, frames=frames)
            if y is None:
                return None
            # Stage A stores Y_full as [F, 3, H, W] in [0, 1]; the loss compares
            # against the engine's [B, 3, F, H, W] render, so restore that layout.
            ys.append(y.permute(1, 0, 2, 3).contiguous())
        return torch.stack(ys)

    def _build_inputs(self, captions, *, cond=None):
        """Encode captions and sample the shared initial noise / token grid."""
        bb = self.accelerator.backbone
        d = self.config.config.data
        grid = bb.token_grid(d.num_frames, d.height, d.width)
        if cond is None:
            cond = bb.encode_text(captions).to(self.device)
        z_init = bb.initial_latent(grid, batch=len(captions), device=self.device)
        return cond, grid, z_init

    def _cached_cond(self, video_ids, captions) -> Optional[TextConditioning]:
        """Stage A's persisted prompt embeddings for the batch, or ``None``.

        Loads ``text_embeds/<video_id>.pt`` from disk so the text encoder never wakes up.
        ``None`` when any clip is missing one, so the caller falls back to encoding.
        """
        if self.config.processed_root is None:
            return None
        layout = ProcessedLayout(self.config.processed_root)
        embeds = []
        for vid in video_ids:
            path = layout.text_embed_path(vid)
            if not path.exists():
                return None
            embeds.append(torch.load(path, map_location="cpu", weights_only=False).float())
        if not embeds or any(e.dim() != 2 for e in embeds):
            return None
        length = max(e.shape[0] for e in embeds)
        padded = torch.zeros(len(embeds), length, embeds[0].shape[-1])
        mask = torch.zeros(len(embeds), length)
        for i, e in enumerate(embeds):
            padded[i, : e.shape[0]] = e
            mask[i, : e.shape[0]] = 1.0
        return TextConditioning(
            embeds=padded.to(self.device), mask=mask.to(self.device),
            prompts=tuple(captions),
        )

    def _full_baseline(self, z_init, cond, grid) -> Tensor:
        """Decode the un-accelerated baseline ``Y_full`` on the same initial noise as the
        accelerated run (label-only, no grad).

        Run under ``grad_mode`` + ``no_grad`` rather than ``inference_mode`` so the
        reference can be multiplied against the accelerated render in the loss.
        """
        tf_cfg = TeacherForwardConfig.from_config(self.config.config)
        tf_cfg.num_inference_steps = self.engine.engine_cfg.num_inference_steps
        runner = TeacherForwardRunner(self.accelerator, tf_cfg, device=self.device)
        bb = self.accelerator.backbone
        with bb.grad_mode(True), torch.no_grad():
            z0, _ = runner.full_denoise(z_init, cond, grid)
            return bb.decode_to_unit(bb.to_grid(z0, grid))

    @staticmethod
    def _to_fchw(video: Tensor, index: int = 0) -> Tensor:
        """``[B,3,F,H,W]`` (or ``[3,F,H,W]``) → clip ``index`` as ``[F,3,H,W]``.

        Slices before permuting so only the read clip is materialised.
        """
        v = video[index] if video.dim() == 5 else video
        return v.permute(1, 0, 2, 3).contiguous()
    # ------------------------------------------------------------------ #
    # LoRA + trainable-parameter scope
    # ------------------------------------------------------------------ #

    def _add_lora_adapters(self) -> None:
        """Inject LoRA into the backbone's last-``lora_layers`` DiT blocks, storing the new
        trainable params. A logged no-op when the backbone exposes no ``dit_blocks()``.
        """
        self._lora_params, self._lora_modules = inject_lora(
            self.accelerator.backbone,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            last_n_blocks=self.config.lora_layers,
        )

    # ------------------------------------------------------------------ #
    # Checkpointing (the adapters live inside the frozen backbone)
    # ------------------------------------------------------------------ #

    def checkpoint(self) -> Dict[str, object]:
        """Full Stage-C checkpoint: plugin weights + LoRA adapters + their geometry.

        The rank / target-block count are stored so the adapters can be re-injected with
        the same geometry before loading (see :func:`cocf.training.lora.attach_lora`).
        """
        return build_checkpoint(
            self.accelerator,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            last_n_blocks=self.config.lora_layers,
        )

    def _get_trainable_params(self):
        """The gradient scope: the L-COCF predictor + strength field + repair net, the RAEC
        certificate, the CMSC alignment head, and any LoRA adapters. The backbone stays frozen.
        """
        params = [p for group in self.accelerator.parameter_groups().values() for p in group]
        params += self._lora_params
        seen, unique = set(), []
        for p in params:  # de-dup (a param may appear in two groups) and keep trainable
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique
