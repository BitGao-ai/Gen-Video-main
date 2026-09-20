#!/usr/bin/env python
"""Fine-tune plugins end to end in Stage C."""

import argparse
import logging
from pathlib import Path

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
    """Fail fast on perception and store visual dim mismatch."""
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
    """Adopt store geometry and schedule, warn on overrides."""
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
    """Run Stage-C fine-tuning."""
    parser = argparse.ArgumentParser(description="Stage C: end-to-end lightweight fine-tuning (§4.2)")
    parser.add_argument("--processed-root", type=Path,
                        help="Processed store root (§3); reads raw_filtered/ + full_baseline/")
    parser.add_argument("--manifest", type=Path,
                        help="Fallback video/caption manifest when no processed store is given")
    parser.add_argument("--checkpoint_load", type=Path, help="Load accelerator checkpoint (e.g. Stage B)")
    parser.add_argument("--checkpoint_save", type=Path, default=Path("./checkpoints/stage_c_final.pt"))
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
    add_backbone_args(
        parser,
        default_backbone="mock",
        default_wan_variant="a14b-t2v",
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
    if args.batch_size != 1:
        parser.error("--batch_size must be 1: semantic-tube state is per video")
    if args.checkpoint_load and not args.checkpoint_load.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint_load}")

    dctx = init_distributed(args.device)
    args.device = dist_device(args.device, dctx)

    setup_logging(level=logging.INFO if dctx.is_main else logging.WARNING)
    log = get_logger("cocf.stage_c")
    torch.manual_seed(args.seed)

    if not args.processed_root and not args.manifest:
        parser.error("provide --processed-root (preferred, §4.2) or --manifest (fallback)")

    config = Config()
    config.seed = args.seed
    if args.lr is not None:
        config.training.optim.lr = args.lr

    config.backbone.name = args.backbone
    config.backbone.model_path = args.model_path
    config.backbone.device = args.device
    config.backbone.dtype = args.backbone_dtype
    real_gpu_backbone = is_real_gpu_backbone(args)
    resolve_vram_policy(config, args, real_gpu_backbone)
    apply_wan_variant(config, args)
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

    _check_visual_dim(accelerator, args.processed_root, log)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info("Loading checkpoint from %s", args.checkpoint_load)
        ckpt = torch.load(args.checkpoint_load, map_location=args.device,
                          weights_only=False)
        load_checkpoint(accelerator, ckpt, training_config=config.training)

    stage_c_config = StageCConfig(
        processed_root=args.processed_root,
        manifest_path=args.manifest,
        config=config,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_inference_steps=args.steps,
        use_lora=args.use_lora,
        device=torch.device(args.device),
    )
    if args.lr is not None:
        stage_c_config.learning_rate = args.lr

    log.info("Starting Stage C: end-to-end fine-tuning")
    stage_c = FinettuneStage(accelerator=accelerator, engine=engine, config=stage_c_config)
    accelerator = stage_c.run()

    if dctx.is_main:
        args.checkpoint_save.parent.mkdir(parents=True, exist_ok=True)
        ckpt = stage_c.checkpoint()
        torch.save(ckpt, args.checkpoint_save)
        log.info(
            "Saved final checkpoint to %s (%d LoRA tensors)",
            args.checkpoint_save, len(ckpt.get("lora", {})),
        )
    dist_shutdown()


if __name__ == "__main__":
    main()
