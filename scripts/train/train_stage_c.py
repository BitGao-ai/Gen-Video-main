#!/usr/bin/env python
"""Entry script for Stage C: end-to-end lightweight fine-tuning (§4.2).

Embeds the trained plugins into the full accelerated inference pipeline and fine-tunes
the differentiable plugins (L-COCF predictor + strength weights + residual-repair net,
plus an optional LoRA on the last DiT blocks) against the full-compute baseline
``Y_full`` that Stage A persisted — the backbone stays frozen.

Reads the §4.2 source: ``raw_filtered/captions.jsonl`` from the processed store
(``--processed-root``); falls back to a plain video/caption manifest (``--manifest``).

Usage (mock smoke run — no weights, CPU):

    python scripts/train/train_stage_c.py \
        --processed-root ./LCOCF_OpenVid1M_Processed \
        --checkpoint_load ./checkpoints/stage_b_final.pt \
        --backbone mock --device cpu --num_epochs 1

Usage (real run on a 40 GB card, Wan2.2-A14B, LoRA off):

    python scripts/train/train_stage_c.py \
        --processed-root ./LCOCF_OpenVid1M_Processed \
        --checkpoint_load ./checkpoints/stage_b_final.pt \
        --backbone wan22 --wan-variant a14b-t2v \
        --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
        --num-frames 49 --height 384 --width 640 \
        --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 4 \
        --real-models --metric-frame-chunk 2 --batch_size 1

**Geometry must match Stage A.** The quality loss compares this stage's render against
the ``Y_full`` Stage A wrote, and the cached-baseline path is only taken when the token
geometry and the step count agree. Both are read back from
``metadata/stage_a_env.json`` and applied automatically; explicit ``--num-frames/
--height/--width`` override them and will be warned about if they disagree.

VRAM on a 40 GB card with A14B: one expert is resident at ~26.1 GiB, leaving ~13 GiB.
Activation checkpointing is on for **both** experts, so the retained graph is one block
input per block per BPTT segment — ~5 GiB at 12,480 tokens with ``--grad-window-steps
1`` — plus ~3 GiB of in-block recompute and ~1.5 GiB for the windowed differentiable
decode at ``--decode-grad-frames 4``. ``FinettuneStage`` computes that budget from the
flags actually in force and warns when it does not fit.
``--use_lora`` is **not** feasible here: LoRA is injected into both experts, but only
one can be resident, and moving a module across devices while its activations are on
the autograd graph is not sound. If you OOM: ``--decode-grad-frames 2``, then
``--vae-tile 96``, then ``--metric-frame-chunk 1``.

Data-parallel (one process per GPU)::

    torchrun --standalone --nproc_per_node=8 scripts/train/train_stage_c.py ...
"""

import argparse
import logging
from pathlib import Path

# Must run before ``import torch``: the CUDA caching allocator reads
# PYTORCH_CUDA_ALLOC_CONF once, at first CUDA use. See cocf.common.alloc.
from cocf.common.alloc import configure_cuda_allocator

PYTORCH_CUDA_ALLOC_CONF = configure_cuda_allocator()

import torch

from cocf.common.config import Config
from cocf.common.logging import get_logger, setup_logging
from cocf.common.vram import (
    add_backbone_args,
    add_geometry_args,
    add_perception_args,
    apply_geometry,
    apply_wan_variant,
    build_perception_and_metrics,
    is_real_gpu_backbone,
    log_vram_policy,
    resolve_vram_policy,
)
from cocf.core.accelerator import Accelerator
from cocf.data.metrics import DEFAULT_FRAME_CHUNK
from cocf.data.processed_layout import ProcessedLayout
from cocf.engine import InferenceEngine
from cocf.training.checkpoint import load_checkpoint
from cocf.training.distributed import init_distributed
from cocf.training.distributed import resolve_device as dist_device
from cocf.training.distributed import shutdown as dist_shutdown
from cocf.training.stage_c_finetune import FinettuneStage, StageCConfig


def _check_visual_dim(accelerator, processed_root, log) -> None:
    """Fail fast when the perception provider disagrees with the store's tube embeds.

    The CMSC alignment head's ``vis_proj`` is sized from the *perception provider's*
    ``d_clip`` (``_probe_visual_dim``), while Stage B sized the very same layer from
    the ``tube_visual_embed_full`` width recorded in the store. Run Stage C without
    ``--real-models`` and the mock provider reports 64 against a store written by a
    real CLIP, so the Stage-B checkpoint will not load — and if the widths had merely
    been *compatible* rather than equal, it would have loaded and trained a projection
    on embeddings from a different encoder, which is worse than a crash.

    Checked before any checkpoint load so the message names the cause instead of a
    bare ``size mismatch for cmsc_alignment.vis_proj.weight``.
    """
    if processed_root is None:
        return
    from cocf.data import CounterfactualLMDBDataset
    layout = ProcessedLayout(processed_root)
    stored = None
    try:
        ds = CounterfactualLMDBDataset(layout.lmdb_dir)
        if len(ds) > 0:
            vf = ds[0].get("tube_visual_embed_full")
            if vf is not None:
                import numpy as np
                arr = np.asarray(vf)
                if arr.ndim >= 1 and arr.shape[-1] > 0:
                    stored = int(arr.shape[-1])
    except Exception as e:
        log.warning("Could not read tube_visual_embed_full to verify visual_dim: %s", e)
        return
    if stored is None or stored == accelerator.visual_dim:
        return
    raise SystemExit(
        f"visual_dim mismatch: this run's perception provider reports d_clip="
        f"{accelerator.visual_dim}, but the store's tube_visual_embed_full is {stored}-d "
        f"(and Stage B built cmsc_alignment.vis_proj at {stored}).\n"
        f"  * d_clip=64 means the mock provider is in use — pass --real-models.\n"
        f"  * Otherwise pass the same --clip-model Stage A used: d_clip is CLIP's "
        f"projection_dim (ViT-B/32 => 512, ViT-L/14 => 768).\n"
        f"Stage C's tube embeds must come from the same encoder as the store's, or the "
        f"alignment head is trained on two different spaces."
    )


def _apply_stage_a_env(config: Config, processed_root, args, log) -> None:
    """Adopt the store's recorded geometry / teacher schedule, and flag disagreements.

    Stage C is only a meaningful fine-tune if its render is comparable to the reference
    Stage A produced. Two things have to line up, and both fail *quietly* rather than
    loudly when they do not:

    * **Token geometry.** ``FinettuneStage._cached_baseline`` compares the cached
      ``z_init``'s token count against this run's grid and, on a mismatch, silently
      falls back to recomputing the baseline — two extra full trajectories and two extra
      decodes per batch, i.e. a several-fold slowdown that appears in the log only as a
      single warning.
    * **Step count.** ``Y_full`` is the full-compute render *of a specific schedule*;
      scoring a 20-step baseline against a 30-step accelerated run charges the schedule
      difference to the accelerator.

    Explicit CLI geometry still wins — a deliberate override is a legitimate experiment —
    but it is warned about, because getting it by accident is the common case.
    """
    if processed_root is None:
        return
    env = ProcessedLayout(processed_root).read_stage_a_env()
    if not env:
        log.warning(
            "No metadata/stage_a_env.json in this store. Cannot verify that this run's "
            "geometry matches the Y_full it will train against; if they differ, Stage C "
            "will silently recompute the baseline every batch."
        )
        return
    if "teacher_steps" in env:
        config.teacher.num_inference_steps = int(env["teacher_steps"])
    for key in ("num_frames", "height", "width"):
        if key not in env:
            continue
        cli = getattr(args, key, None)
        if cli is not None and int(cli) != int(env[key]):
            log.warning(
                "--%s %s overrides the store's %s=%s. Stage C will not be able to reuse "
                "the cached Y_full and will recompute it for every batch.",
                key.replace("_", "-"), cli, key, env[key],
            )
            continue
        setattr(config.data, key, int(env[key]))
    if env.get("use_real_video"):
        log.warning(
            "This store was generated with --use-real-video, so no z_init was persisted "
            "and the cached-baseline path is unavailable: every batch will re-denoise "
            "two full trajectories. Regenerate Stage A without --use-real-video for a "
            "store that feeds Stage C."
        )
    # The schedule shift decides both the trajectory and (on a Wan2.2 MoE) which expert
    # runs at each noise level, so a Stage C that renders on a different one is not
    # comparable to the Y_full it is scored against — and nothing downstream would
    # notice, because the shapes still match.
    stored_shift = (env.get("backbone_extra") or {}).get("flow_shift")
    run_shift = (config.backbone.extra or {}).get("flow_shift")
    if stored_shift is not None and run_shift is not None \
            and abs(float(stored_shift) - float(run_shift)) > 1e-6:
        log.warning(
            "flow_shift %.3f in this run vs %.3f in the store: the accelerated render "
            "and Y_full follow different noise schedules, so the quality loss charges "
            "the schedule difference to the accelerator. Pass --flow-shift %s.",
            float(run_shift), float(stored_shift), stored_shift,
        )
    log.info(
        "Stage A env: %dx%dx%d, %d teacher steps, token_dim=%s, flow_shift=%s",
        config.data.num_frames, config.data.height, config.data.width,
        config.teacher.num_inference_steps, env.get("token_dim", "?"),
        stored_shift if stored_shift is not None else "1.0",
    )


def main():
    parser = argparse.ArgumentParser(description="Stage C: end-to-end lightweight fine-tuning (§4.2)")
    parser.add_argument("--processed-root", type=Path,
                        help="Processed store root (§3); reads raw_filtered/ + full_baseline/")
    parser.add_argument("--manifest", type=Path,
                        help="Fallback video/caption manifest when no processed store is given")
    parser.add_argument("--checkpoint_load", type=Path, help="Load accelerator checkpoint (e.g. Stage B)")
    parser.add_argument("--checkpoint_save", type=Path, default=Path("./checkpoints/stage_c_final.pt"))
    # One engine.generate call renders the whole batch, so activations scale with it and
    # there is no accumulation path to trade against. 1 is the only value a 40 GB card
    # with a 14B expert resident can take; FinettuneStage clamps it if the card is small.
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None, help="Override config.training.optim.lr")
    parser.add_argument("--use_lora", action="store_true",
                        help="Inject LoRA into the last DiT blocks of every expert. Needs "
                             "all targeted experts co-resident, so it is not usable with "
                             "Wan2.2-A14B below ~64 GB.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Denoising steps for the accelerated run. Defaults to the "
                             "schedule Stage A generated Y_full with, so the persisted "
                             "baseline is a valid target; override to re-compute it.")
    # Backbone / residency / geometry / perception — shared with Stage A so the two
    # stages resolve every one of them identically (cocf/common/vram.py).
    add_backbone_args(
        parser,
        default_backbone="mock",
        default_wan_variant="a14b-t2v",
        # Unlike Stage A's label-only decode, this one runs *on the autograd graph*, so
        # every tile's intermediates are retained for backward and a large tile costs
        # more than it saves. Bounded further by --decode-grad-frames.
        default_vae_tile=128,
    )
    add_geometry_args(parser)
    add_perception_args(parser, default_frame_chunk=DEFAULT_FRAME_CHUNK)
    g = parser.add_argument_group("Stage-C graph bounds (§4.2)")
    g.add_argument("--grad-window-steps", type=int, default=None,
                   help="Truncated-BPTT segment length in *computed* denoising steps "
                        "(config.engine.grad_window_steps). Peak activation memory is "
                        "linear in this. Use 1 on a 40 GB card; 0 means full BPTT, which "
                        "no real backbone survives.")
    g.add_argument("--decode-grad-frames", type=int, default=None,
                   help="Latent temporal slots decoded on the autograd graph "
                        "(config.engine.decode_grad_frames). 0 = the whole clip.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # Join the process group first: it must happen before any weight is built (it
    # pins this rank's CUDA device) and before logging is configured (so the extra
    # ranks can be quietened). A run not launched under torchrun gets a disabled
    # context here and behaves exactly as it always did.
    dctx = init_distributed(args.device)
    args.device = dist_device(args.device, dctx)

    # Only rank 0 narrates. Eight interleaved copies of every line make a log that
    # cannot be read, and the per-rank detail that *is* worth having (shard size,
    # OOM, warnings) comes through at WARNING.
    setup_logging(level=logging.INFO if dctx.is_main else logging.WARNING)
    # setup_logging attaches the stdout handler to the "cocf" logger and sets
    # propagate=False, so a bare getLogger("__main__") would emit nothing at
    # INFO — this script's own progress lines included.
    log = get_logger("cocf.stage_c")
    # The *same* seed on every rank, deliberately: replicas must start from identical
    # weights (a re-initialised layer is seeded here), and the data is split by the
    # DistributedSampler rather than by diverging RNG streams.
    torch.manual_seed(args.seed)

    if not args.processed_root and not args.manifest:
        parser.error("provide --processed-root (preferred, §4.2) or --manifest (fallback)")

    config = Config()
    config.seed = args.seed
    if args.lr is not None:
        # Feed *both* sinks: FinettuneStage builds its optimiser from
        # StageCConfig.learning_rate, so setting only config.training.optim.lr made
        # --lr silently inert for this stage.
        config.training.optim.lr = args.lr

    config.backbone.name = args.backbone
    config.backbone.model_path = args.model_path
    # Keep the frozen backbone on the device the engine renders on, so its weights and
    # the latents/conditioning built on --device never split across cpu/cuda.
    config.backbone.device = args.device
    config.backbone.dtype = args.backbone_dtype
    real_gpu_backbone = is_real_gpu_backbone(args)
    resolve_vram_policy(config, args, real_gpu_backbone)
    apply_wan_variant(config, args)
    # Store first, CLI second: the store records what the target was rendered at, and an
    # explicit flag is a deliberate (warned-about) override of that.
    _apply_stage_a_env(config, args.processed_root, args, log)
    apply_geometry(config, args)
    if args.grad_window_steps is not None:
        config.engine.grad_window_steps = args.grad_window_steps
    if args.decode_grad_frames is not None:
        config.engine.decode_grad_frames = args.decode_grad_frames
    if real_gpu_backbone:
        log_vram_policy(config, log, PYTORCH_CUDA_ALLOC_CONF)
        log.info(
            "Stage C graph bounds: grad_window_steps=%d, decode_grad_frames=%d, "
            "gradient_checkpointing=%s",
            config.engine.grad_window_steps, config.engine.decode_grad_frames,
            config.memory.gradient_checkpointing,
        )
        if args.use_lora and config.backbone.offload_idle_expert:
            log.warning(
                "--use_lora with --offload-idle-expert: LoRA is injected into every "
                "expert, but only one is resident, and the offloaded expert's adapters "
                "would be moved across devices while their activations are on the "
                "autograd graph. Expect a device mismatch or silently dead gradients."
            )

    perception, metric_extractor = build_perception_and_metrics(args, log)

    log.info("Building accelerator and engine with backbone '%s' (perception=%s, metrics=%s)",
             args.backbone, "real" if perception else "mock",
             "real" if metric_extractor else "mock")
    accelerator = Accelerator.from_config(
        config, perception=perception, metric_extractor=metric_extractor,
    )
    engine = InferenceEngine(accelerator, config.engine, config.trigger)

    # Before the checkpoint load, so a provider/store disagreement is reported as such.
    _check_visual_dim(accelerator, args.processed_root, log)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info("Loading checkpoint from %s", args.checkpoint_load)
        ckpt = torch.load(args.checkpoint_load, map_location=args.device,
                          weights_only=False)
        # Either layout: a bare Stage-B state_dict, or a Stage-C checkpoint whose
        # LoRA adapters are re-attached here so a resumed run continues training them
        # (FinettuneStage's injection below finds and re-arms the restored wrappers).
        load_checkpoint(accelerator, ckpt, training_config=config.training)

    stage_c_config = StageCConfig(
        processed_root=args.processed_root,
        manifest_path=args.manifest,
        config=config,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        # None ⇒ FinettuneStage pins the render to Stage A's teacher schedule, so the
        # persisted Y_full is a valid target. The stage owns this so this script and
        # TrainingPipeline cannot drift apart (§P4-3).
        num_inference_steps=args.steps,
        use_lora=args.use_lora,
        device=torch.device(args.device),
    )
    if args.lr is not None:
        stage_c_config.learning_rate = args.lr

    log.info("Starting Stage C: end-to-end fine-tuning")
    stage_c = FinettuneStage(accelerator=accelerator, engine=engine, config=stage_c_config)
    accelerator = stage_c.run()

    # One writer: the ranks hold identical weights (gradients are averaged every
    # step), so rank 0's copy *is* the model — and eight processes writing one path
    # is a corrupt file, not a redundant one.
    if dctx.is_main:
        args.checkpoint_save.parent.mkdir(parents=True, exist_ok=True)
        # Save the plugins *and* the LoRA adapters. The adapters live inside the frozen
        # backbone, which is deliberately outside Accelerator.state_dict() — saving only
        # that discarded the entire --use_lora fine-tune at the last line of the run.
        ckpt = stage_c.checkpoint()
        torch.save(ckpt, args.checkpoint_save)
        log.info(
            "Saved final checkpoint to %s (%d LoRA tensors)",
            args.checkpoint_save, len(ckpt.get("lora", {})),
        )
    dist_shutdown()


if __name__ == "__main__":
    main()
