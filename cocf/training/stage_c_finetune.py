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
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.memory import set_gradient_checkpointing
from cocf.common.types import Action, TubeState
from cocf.core.accelerator import Accelerator
from cocf.data import ProcessedLayout, RawFilteredDataset, collate_raw_filtered
from cocf.engine import InferenceEngine
from cocf.lcocf.damage import (
    DAMAGE_DIMENSIONS,
    DEFAULT_DAMAGE_WEIGHTS,
    MultiDimDamageComputer,
    VideoFeatures,
)
from cocf.training.checkpoint import build_checkpoint
from cocf.training.distributed import (
    all_reduce_mean,
    all_reduce_min,
    assert_same,
    average_gradients,
    broadcast_parameters,
    context as dist_context,
)
from cocf.training.lora import inject_lora, lora_state_dict
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

# Activation headroom one Stage-C clip needs on top of the frozen weights, at the
# §4.2 reference geometry (49×384×640 ⇒ 12480 tokens) with a 14B-class expert,
# grad_window_steps=1 and a windowed differentiable decode. Measured as the sum of
# the per-block checkpoint stash (~5 GiB), the in-block recompute peak (~3 GiB) and
# the retained VAE decode graph (~8-14 GiB, see ``--decode-grad-frames``). Used only
# to clamp an over-large batch before it OOMs minutes into the epoch.
_STAGE_C_GIB_PER_CLIP = 20.0


# One render is one smoothing group: :func:`tube_temporal_smoothness` keys on
# ``(video_id, tube_id)`` to find the same tube at adjacent steps, and every record
# here comes from the same accelerated trajectory.
_RENDER_GROUP = "__render__"

_DAMAGE_COMPUTER = MultiDimDamageComputer()


def _step_records(
    *, step_idx: int, t: int, budget: float, step_frac: float,
    tube_states: Dict[int, TubeState], strength_feats, actions,
) -> List[StepRecord]:
    """Fan one ``record_sink`` callback out into per-(tube, step) :class:`StepRecord`s.

    The engine reports a step at a time while the Stage-C regularisers consume the
    same flat, per-sample shape Stage B uses. This adapter is the glue that was
    missing — which is the direct reason ``stage_c_losses`` had no caller (§P4-A3).
    """
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
            )
        )
    return out


def _scalar_damage(full: VideoFeatures, accel: VideoFeatures) -> float:
    """Realised end-to-end degradation of the accelerated render, as one scalar.

    The §4.2 certificate regulariser calibrates ``E_cert`` to upper-bound the damage
    that actually occurred, so it needs the same weighted reduction of the same 8 axes
    Stage A labels and Stage B trains on (:func:`damage_scalar_batch`) — computed here
    from the two observations' features rather than re-derived, so the two stages
    agree on what "damage" means.

    The reference is co-located onto the accelerated branch's device first: every axis
    is a binary reduction and :class:`MultiDimDamageComputer` carries no device logic,
    so a reference that was extracted with the CPU offload on would fault here.
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
    # Denoising steps for the accelerated render. ``None`` — the default — means "match
    # the schedule Stage A rendered Y_full with", which is what makes the persisted
    # baseline a valid target. Set it only to deliberately render on a different
    # schedule, accepting that the cached baseline is then discarded and re-denoised
    # every batch (two full trajectories + two decodes per step, §P2-4).
    #
    # This lives here, rather than in each caller, because the two entry points had
    # drifted: the CLI aligned the counts while ``TrainingPipeline`` did not, so the
    # pipeline path silently paid double for every Stage-C step (§P4-3).
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
        self._warned_baseline_mismatch = False
        self._warned_no_grad_batch = False

        # Pin the accelerated render to the schedule Stage A's Y_full was generated
        # with, unless the caller explicitly asked for another one. Doing it here — not
        # in each entry point — is what keeps the CLI and TrainingPipeline in step
        # (§P4-3). ``engine.engine_cfg`` is normally the *same object* as
        # ``config.config.engine``; both are written so a caller that passed a separate
        # EngineConfig still gets a consistent pair.
        self._resolve_inference_steps()

        # Freeze the backbone before anything else (matches Stage B). Stage C tunes
        # the plugins — and optionally LoRA — never the full backbone. But autograd
        # still allocates gradients and retains activations for any parameter left
        # with requires_grad=True, so an unfrozen backbone silently inflates VRAM by
        # a large multiple even though the optimizer never sees those params. Freeze
        # first, then add LoRA so the freshly-inserted LoRA params stay trainable.
        self.accelerator.freeze_backbone()

        # Place the learnable plugins on the run device (mirrors JointTrainingStage,
        # stage_b_joint.py). The engine follows ``z.device`` throughout but never
        # *moves* the accelerator, and the frozen backbone is a plain attribute placed
        # via ``config.backbone.device`` — so without this the plugins (strength field,
        # damage predictor, repair net, CMSC alignment, certificate) stay on CPU while
        # ``z_init``/backbone are on GPU, and the first plugin call inside
        # ``engine.generate()`` (and the schedule regularisers) hit a CPU×CUDA mismatch.
        self.accelerator.to(self.device)

        # Optionally add LoRA adapters to backbone (§4.2 "最后若干层 DiT 的 LoRA 适配器")
        self._lora_params: list = []
        if config.use_lora:
            _log.info(f"Stage C: Adding LoRA adapters (rank={config.lora_rank})")
            self._add_lora_adapters()
        # Unconditional: the repair net puts ``z`` on the graph, so the DiT retains
        # block inputs with or without LoRA (see the method docstring).
        self._enable_backbone_checkpointing()

        # The whole batch goes through a single ``engine.generate`` call
        # (:meth:`_finetune_batch`), so activations scale with batch_size and there is
        # no gradient-accumulation path to trade against it. On a 40 GB card with a
        # 14B expert resident, anything above 1 is an OOM rather than a slowdown.
        #
        # The budget is *per clip*, not a fixed floor: with a 14B expert the retained
        # graph is dominated by terms linear in the batch — the per-DiT-block stash
        # (B·N·d, ~0.5 GiB/block-boundary at B=4, 12480 tokens, d=5120) and the
        # differentiable VAE decode (every tile of every clip retained for backward).
        # A card with 25 GiB free therefore clears an "is it ≥ 24 GiB" floor and still
        # OOMs on batch 4 several minutes in; requiring the headroom to scale with the
        # batch turns that into an immediate, explained clamp.
        if config.batch_size > 1:
            free_gib = self._free_vram_gib()
            if free_gib is not None and free_gib < _STAGE_C_GIB_PER_CLIP * config.batch_size:
                _log.warning(
                    "Stage C: batch_size=%d needs ~%.0f GiB of activation headroom but "
                    "only %.1f GiB is free after the frozen backbone; clamping to %d. "
                    "One engine.generate call renders the whole batch, so the retained "
                    "graph scales with it. To keep the effective batch, free residency "
                    "instead (--offload-idle-expert frees ~27 GiB of the 54 GiB two "
                    "resident Wan2.2 experts take) or shrink the graph "
                    "(--decode-grad-frames, --grad-window-steps).",
                    config.batch_size, _STAGE_C_GIB_PER_CLIP * config.batch_size,
                    free_gib, max(1, int(free_gib // _STAGE_C_GIB_PER_CLIP)),
                )
                config.batch_size = max(1, int(free_gib // _STAGE_C_GIB_PER_CLIP))
        else:
            # batch_size == 1 already: there is nothing left to clamp, but the run can
            # still be short on headroom — and then it OOMs minutes in with no prior
            # hint, because the clamp above never looked. Say so at startup instead.
            free_gib = self._free_vram_gib()
            if free_gib is not None and free_gib < _STAGE_C_GIB_PER_CLIP:
                _log.warning(
                    "Stage C: only %.1f GiB free after the frozen backbone; one clip at "
                    "%dx%dx%d needs ~%.0f GiB (DiT checkpoint stash ~5 + in-block "
                    "recompute ~3 + differentiable VAE decode ~8-14). Expect an OOM in "
                    "the first batch. Lower --decode-grad-frames (1 is the floor), then "
                    "--metric-frame-chunk, then the render geometry — or free residency "
                    "with --offload-idle-expert / a single-expert --wan-variant.",
                    free_gib, *self._frame_shape(), _STAGE_C_GIB_PER_CLIP,
                )

        # Under data parallelism every rank must agree on the batch size: it decides
        # how many batches an epoch has, and a rank that clamped lower would finish its
        # shard early and leave the others blocked in an all-reduce forever. The
        # smallest card decides. No-op in a single process.
        agreed = int(all_reduce_min(config.batch_size))
        if agreed != config.batch_size:
            _log.warning("Stage C: batch_size %d -> %d to match the smallest rank.",
                         config.batch_size, agreed)
            config.batch_size = agreed

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
        # ``index or 0`` would read card 0 for a bare "cuda" — wrong on every rank but
        # rank 0 under torchrun, and this figure decides the batch clamp.
        idx = torch.device(self.device).index
        idx = torch.cuda.current_device() if idx is None else idx
        total = torch.cuda.get_device_properties(idx).total_memory
        return (total - torch.cuda.memory_allocated(idx)) / 1024 ** 3

    def _enable_backbone_checkpointing(self) -> None:
        """Turn on activation checkpointing in every DiT expert.

        This used to be gated on ``use_lora``, on the reasoning that a fully frozen
        backbone "retains nothing". That is wrong for this stage: the residual-repair
        net edits ``z`` mid-trajectory, so from that point on ``z`` requires grad and
        autograd retains each block's inputs to backprop *through* the DiT — whether or
        not the DiT itself has trainable parameters. Without checkpointing a
        Wan2.2-A14B step at 12,480 tokens keeps roughly 2 GiB per block across 40
        blocks, which OOMs a 40 GB card long before the LoRA question arises. So it is
        now unconditional, and ``MemoryConfig.gradient_checkpointing`` is the only
        switch.

        Applied to **every** denoiser, not ``backbone.module``: that property returns
        the primary transformer only, so on an A14B MoE the low-noise expert — the one
        that runs at every σ below the boundary, i.e. all of the steps inside the BPTT
        window — was left un-checkpointed. :meth:`BackboneAdapter.lora_roots` is the
        existing enumeration of both experts.
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
            # A mock/toy adapter exposes no checkpointing hook and needs none — its
            # activations are megabytes. Only a real DiT makes this a failure worth
            # shouting about, so gate the alarm on there being real weights to protect.
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
        # Eval mode: the plugins carry no dropout/BN, so this only skips the predictor's
        # gradient-checkpointing wrapper — keeping the autograd graph simple and
        # deterministic while parameters still receive gradient.
        self.accelerator.eval()

        # Resolve the §4.2 data source. ``RawFilteredDataset`` reads the processed
        # store's ``raw_filtered/captions.jsonl`` (or a fallback manifest) as *metadata
        # only* and carries each clip's ``video_id`` — two things that matter here:
        # this stage never touches ``batch["video"]`` (it renders from the caption), so
        # decoding clips was pure waste, and the ``video_id`` is what lets it fetch the
        # ``Y_full`` Stage A already computed instead of re-denoising it (§P2-4).
        if self.config.manifest_path is None and self.config.processed_root is None:
            _log.warning("Stage C: no --manifest or --processed-root given; nothing to train on.")
            return self.accelerator
        source = self.config.manifest_path or self.config.processed_root
        dataset = RawFilteredDataset(
            processed_root=self.config.processed_root,
            manifest_path=self.config.manifest_path,
        )
        # Data parallelism (§4.2 on a multi-GPU box): each rank owns a disjoint shard
        # and the gradients are averaged after every backward. ``drop_last=True`` is
        # what keeps that safe — it gives every rank the *same* number of batches, and
        # a rank that ran out early would leave the others waiting in an all-reduce
        # that never completes. Single-process runs take the original path untouched.
        dctx = dist_context()
        sampler = None
        if dctx.enabled:
            sampler = DistributedSampler(
                dataset, num_replicas=dctx.world_size, rank=dctx.rank,
                shuffle=True, drop_last=True,
            )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler=sampler,
            shuffle=sampler is None,
            # Identity collate: the engine runs per clip because Y_full is per video.
            collate_fn=collate_raw_filtered,
        )

        if dctx.enabled:
            # Ranks share a seed and a checkpoint, so this is normally a no-op — but a
            # tensor the checkpoint could not restore (a shape-mismatched vis_proj) or
            # a freshly injected LoRA would otherwise start different on every rank,
            # and data parallelism over diverged replicas trains nothing coherent.
            # Before the first collective of the run: the gradient all-reduce walks
            # this exact list, and a rank with a different length would not error, it
            # would hang. Same for the batch count, which drives one collective a step.
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
            _log.info(f"Stage C: {len(dataset)} clips, {len(dataloader)} batches")
        if len(dataloader) == 0:
            # No clips resolved (empty/missing manifest). Bail out cleanly instead of
            # dividing by a zero batch count in the epoch-average below.
            _log.warning("Stage C: data source '%s' yielded no batches; skipping fine-tune.", source)
            return self.accelerator

        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            trained_batches = 0
            if sampler is not None:
                # Without this every epoch reshuffles to the *same* permutation, so a
                # rank sees one fixed shard for the whole run.
                sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                # Run accelerated generation for the batch
                # (in practice, this would be batched; here shown per-video for clarity)
                loss = self._finetune_batch(batch)

                # Backward pass. set_to_none frees the grad tensors between steps
                # instead of zeroing them in place — lower memory held across the
                # step boundary and marginally faster.
                self.optimizer.zero_grad(set_to_none=True)
                # A batch can legitimately produce a loss with no autograd graph: the
                # render skipped every repair (nothing puts z on the graph), the clip
                # segmented no tube (no regularisers) and LoRA is off. ``backward()``
                # on that raises "does not require grad", which used to kill the run at
                # a random batch hours in. Skipping it is the correct no-op — there is
                # nothing to learn from this batch — but the *decision* must be shared:
                # a rank that skipped its collectives while the others all-reduce does
                # not fail, it hangs.
                if loss.requires_grad:
                    loss.backward()
                # Average across ranks *before* clipping, so every rank clips the same
                # gradient and therefore takes an identical optimiser step. Called
                # unconditionally so the collective count matches on every rank; a
                # no-op when this is a single process.
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
                    # ``batch_idx + 1`` batches have been summed into epoch_loss;
                    # the old ``max(1, batch_idx)`` divisor skewed the running mean.
                    avg_loss = epoch_loss / (batch_idx + 1)
                    _log.info(
                        f"  Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"batch {batch_idx}/{len(dataloader)}, "
                        f"loss: {loss:.4f} (avg: {avg_loss:.4f})"
                    )

                if (batch_idx + 1) % self.config.save_interval == 0 and dctx.is_main:
                    # Created on first write, not in __post_init__: building a config
                    # object should not touch the filesystem.
                    self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = (
                        self.config.checkpoint_dir
                        / f"stage_c_epoch_{epoch+1}_batch_{batch_idx+1}.pt"
                    )
                    torch.save(self.checkpoint(), ckpt_path)

            # Reduced across ranks: each rank only saw its own shard, and a per-rank
            # mean would have rank 3 keeping a "best" checkpoint rank 0 rejected.
            avg_epoch_loss = all_reduce_mean(
                epoch_loss / len(dataloader), dctx, device=torch.device(self.device)
            )
            # Report what actually trained, not just the loss: skipping a graph-less
            # batch is a quiet no-op, and a run where *every* batch is skipped would
            # otherwise look identical in the log to one that is learning — the exact
            # failure the crash this replaced used to make impossible to miss.
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

            # The comparison runs on every rank (all-reduced, hence identical), but
            # only rank 0 writes — eight processes writing one path is a corrupt file.
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
        """One §4.2 end-to-end step on a batch of captions.

        Runs the accelerated engine, scores the render against the full-compute
        baseline ``Y_full`` (same initial noise), and adds the schedule regularisers
        recomputed on the engine's real per-step features::

            L = λ_pixel·L_pixel(Y_accel, Y_full)      (§4.2 主损失 — 像素)
              + λ_quality·L_CMSC(Y_accel, Y_full)     (§4.2 主损失 — 多维语义守恒, §6.3.2)
              + λ_sta·L_tube + λ_cert·L_cert + λ_cost·L_budget   (§4.2 正则)

        Every term is assembled by :mod:`cocf.training.stage_c_losses`, which is also
        where the objective is documented. This stage previously carried a hand-written
        3-term stand-in for the §6.3.2 loss (identity / appearance / motion) while the
        full six-term implementation sat unused, so ``λ_spatial``/``λ_ocr``/``λ_bnd``
        never trained (§P4-A3). The pixel/CMSC terms flow gradient through the
        differentiable decode into the residual-repair net (and any LoRA adapters); the
        regularisers flow into the L-COCF strength field, damage predictor and the RAEC
        certificate coefficients — exactly the §4.2 gradient scope.
        """
        bb = self.accelerator.backbone
        captions = [item.caption for item in batch]
        if not captions:
            return torch.zeros((), device=self.device, requires_grad=True)

        # Reuse Stage A's persisted baseline when it exists: it is the *same*
        # computation, already paid for, and re-running it made every Stage-C step two
        # full trajectories plus two decodes (§P2-4). It is only a valid reference for a
        # run starting from the same noise, so z_T is loaded with it or neither is used.
        video_ids = [item.video_id for item in batch]
        grid = bb.token_grid(*self._frame_shape())
        z_init = self._cached_baseline(video_ids, grid)
        y_full = None  # cached path: read after the render, at the decode window only
        if z_init is not None:
            cond = bb.encode_text(captions).to(self.device)
        else:
            cond, grid, z_init = self._build_inputs(captions)
            y_full = self._full_baseline(z_init, cond, grid)

        # Accelerated render with the differentiable decode + per-step feature capture.
        records: list = []
        result = self.engine.generate(
            captions, z_init, grid, cond, bb,
            record_sink=lambda **kw: records.extend(_step_records(**kw)),
            decode_grad=True,
        )
        y_accel = result.video
        # A windowed differentiable decode (``engine.decode_grad_frames``) covers only
        # part of the clip, so the reference must be cut to the *same* pixel frames —
        # comparing a window against the whole baseline would score the accelerator on
        # frames it did not render (§P4-B1). On the cached path the cut happens at the
        # *read*, so the discarded frames never reach the device at all; on the
        # recomputed path the whole clip is already in hand and is sliced here.
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

        # --- §4.2 主损失: pixel + the full §6.3.2 conservation loss ------------- #
        # ``F.l1_loss`` rather than ``(a - b).abs().mean()``: the latter materialises
        # two more full-size video tensors (a 49×480×832 batch is ~0.9 GB each) for a
        # value the fused kernel produces with none (§P4-B2).
        l_pixel = F.l1_loss(y_accel.float(), y_full.float().to(y_accel.device))
        l_quality, cmsc_terms = self._conservation_loss(
            y_accel, y_full, captions, result, text_embeds=cond.embeds
        )
        l_reg, _ = self._regularizers(records, cmsc_terms.get("measured_damage", 0.0))
        return (
            self.config.lambda_pixel * l_pixel
            + self.config.lambda_quality * l_quality
            + l_reg
        )

    @staticmethod
    def _align_reference(y_full: Tensor, frame_span) -> Tensor:
        """Slice ``Y_full`` ``[B,3,F,H,W]`` to the pixel frames the render covers.

        ``frame_span is None`` (inference, or a full decode) leaves it untouched. The
        span comes from the backbone's own :meth:`~cocf.backbones.base.BackboneAdapter.pixel_span`,
        so the two sides are cut by the same arithmetic that produced the render.
        """
        if frame_span is None:
            return y_full
        start, stop = frame_span
        return y_full[:, :, start:stop]

    @staticmethod
    def _check_alignment(y_accel: Tensor, y_full: Tensor, frame_span) -> None:
        """Fail loudly when the render and its reference do not describe the same frames.

        Both §4.2 main-loss terms compare these element-wise, so a geometry mismatch is a
        wrong training objective — and one that reports itself badly if left to the loss:
        a differing frame count surfaces as ``broadcast_tensors`` deep inside
        ``F.l1_loss`` with no mention of the window that caused it, while a mismatch on
        an axis that happens to be 1 would broadcast *silently* and train on nonsense.
        Checked once per batch against tensors already in hand, so it costs nothing.

        The cause is always the same shape of thing: whatever produced ``frame_span``
        disagrees with what ``decode_latent`` actually emitted for the window (see
        :meth:`~cocf.backbones.base.BackboneAdapter.pixel_span`'s contract), or Stage A
        persisted ``Y_full`` at a geometry this run does not render at.
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
    # §4.2 forward building blocks
    # ------------------------------------------------------------------ #

    def _conservation_loss(self, y_accel, y_full, captions, result, *, text_embeds=None):
        """§6.3.2 ``L_CMSC`` averaged over **every** clip in the batch.

        The batch is rendered in one engine call and the engine segments its tubes from
        the first element's preview decode, so all clips share one tube set — but each
        has its own caption and its own render, and scoring only ``[0]`` (as this stage
        used to) threw away ``(B-1)/B`` of the signal while paying the full memory bill
        for it (§P4-B2). Returns ``(loss, {per-term components, measured_damage})``,
        where ``measured_damage`` is the realised end-to-end degradation the §4.2
        certificate regulariser calibrates against.
        """
        me = self.accelerator.metric_extractor
        if me is None or not result.tubes:
            return y_accel.new_zeros(()), {}
        grid = result.grid or self.accelerator.backbone.token_grid(*self._frame_shape())
        # Reuse the conditioning the caller already encoded. Re-encoding here was the
        # same prompts through the same frozen umT5 for the same result — and under the
        # default residency policy (``text_encoder=cpu`` + ``te_exclusive``) each call
        # parks the resident ~27 GiB expert on the CPU, moves the ~11 GiB text encoder
        # onto the card and reverses it afterwards, so the duplicate cost a second
        # ~76 GiB round trip over PCIe on *every* training step.
        text = text_embeds if text_embeds is not None else \
            self.accelerator.backbone.encode_text(captions).embeds

        total = y_accel.new_zeros(())
        comps: Dict[str, float] = {}
        damages: list = []
        n = min(y_accel.shape[0], len(captions))
        for b in range(n):
            # Accelerated branch keeps the autograd graph (``differentiable=True``);
            # the baseline is a detached, no-grad reference — the "conserve the
            # reference relations" semantics of §6.
            accel_obs = build_cmsc_observation(
                me, self.accelerator.perception, self._to_fchw(y_accel, b),
                captions[b], result.tubes, grid, text[b], differentiable=True,
            )
            full_obs = build_cmsc_observation(
                me, self.accelerator.perception, self._to_fchw(y_full, b),
                captions[b], result.tubes, grid, text[b],
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
        """§4.2 正则: reuse the Stage-B tube-smoothing / certificate / budget terms on
        the engine's *actual* per-step scheduling features (captured via ``record_sink``).

        Differentiable through the strength field and damage predictor, so the §4.2
        end-to-end fine-tune shapes the scheduler on the trajectory it really produced.
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

        Requires *every* clip in the batch to have **both** halves of the level-3
        bucket, since the batch is rendered in one engine call — but only the ``z_T``
        half is returned here. ``Y_full`` is fetched afterwards by
        :meth:`_windowed_reference`, once the render has settled which frames the
        differentiable decode actually covers; presence is validated now
        (:meth:`ProcessedLayout.has_y_full`) so the decision to reuse the baseline is
        still made *before* the render commits to ``z_T``.

        Returns ``None`` — meaning "recompute" — for a store built with
        ``--no-baseline`` (the documented way to skip the ~1 TB bucket), and for a
        store whose geometry differs from this run's: Stage A may have generated at
        another resolution or frame count, and a ``z_T`` of the wrong token count
        would otherwise be reshaped into nonsense.
        """
        if self.config.processed_root is None:
            return None
        # Y_full is the full-compute render of a *specific* schedule. Reusing a
        # 20-step baseline as the target for a 30-step accelerated run would charge
        # the schedule difference to the accelerator, so the step counts must agree.
        # ``_resolve_inference_steps`` aligns them unless the caller overrode the
        # count, and already said so once — this is the per-batch guard, kept silent
        # so a deliberate override does not emit one line per batch for the whole run.
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

        Returns ``[B, 3, f, H, W]`` — already cut to the frames the differentiable
        decode covers, so it needs no :meth:`_align_reference` afterwards — or ``None``
        if any clip's baseline vanished between the pre-render check and here (a store
        edited mid-run; the caller then recomputes).

        Reading the window rather than the clip is what keeps the reference off the
        card during the render: at ``decode_grad_frames=2`` the loss scores 5 of 49
        frames, and the other 44 were being transferred and held for the whole
        trajectory for nothing.
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
        *same* initial noise as the accelerated run (label-only, no grad).

        Run under ``grad_mode`` + ``no_grad`` rather than ``inference_mode``: the
        reference is a *constant*, but it still gets multiplied against the accelerated
        render in the semantic loss, and an inference tensor cannot be saved for
        backward — it would raise the moment the §6.3.2 terms are evaluated. ``no_grad``
        keeps the pass graph-free all the same, so nothing is paid for the correctness.
        """
        tf_cfg = TeacherForwardConfig.from_config(self.config.config)
        tf_cfg.num_inference_steps = self.engine.engine_cfg.num_inference_steps
        runner = TeacherForwardRunner(self.accelerator, tf_cfg, device=self.device)
        bb = self.accelerator.backbone
        with bb.grad_mode(True), torch.no_grad():
            z0, _ = runner.full_denoise(z_init, cond, grid)
            return bb.decode_latent(bb.to_grid(z0, grid))

    @staticmethod
    def _to_fchw(video: Tensor, index: int = 0) -> Tensor:
        """``[B,3,F,H,W]`` (or ``[3,F,H,W]``) → clip ``index`` as ``[F,3,H,W]`` in [0,1].

        Slices *before* permuting: ``permute(...).contiguous()`` on the whole batch
        materialised two more full-size, gradient-carrying copies of every clip when
        only one of them was about to be read (§P4-B2).
        """
        v = video[index] if video.dim() == 5 else video
        return v.permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)
    # ------------------------------------------------------------------ #
    # LoRA + trainable-parameter scope (§4.2)
    # ------------------------------------------------------------------ #

    def _add_lora_adapters(self) -> None:
        """Inject LoRA into the backbone's last-``lora_layers`` DiT blocks (§4.2),
        storing the new trainable params. A logged no-op when the backbone exposes no
        ``dit_blocks()`` (the plugin fine-tune still runs).

        ``inject_lora`` materialises the backbone first: a real adapter builds its
        transformer lazily on the first forward, and this runs at *construction* —
        long before any forward — so without that the injection saw no blocks and
        ``--use_lora`` was a silent no-op.
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

    def lora_state_dict(self) -> Dict[str, Tensor]:
        """Trained LoRA weights, or ``{}`` when LoRA is off / was not injected.

        ``Accelerator.state_dict()`` cannot carry these — the backbone is a plain
        attribute by design — so Stage C's checkpoint must save them explicitly or the
        fine-tune is discarded at the moment it finishes.
        """
        if not self._lora_params:
            return {}
        return lora_state_dict(self.accelerator.backbone)

    def checkpoint(self) -> Dict[str, object]:
        """Full Stage-C checkpoint: plugin weights + LoRA adapters + their geometry.

        The rank / target-block count are stored alongside the tensors because the
        adapters must be re-injected with the *same* geometry before they can be
        loaded back (see :func:`cocf.training.lora.attach_lora`).
        """
        return build_checkpoint(
            self.accelerator,
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
