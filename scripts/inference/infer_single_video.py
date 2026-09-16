#!/usr/bin/env python
"""Entry script for accelerated video inference (§7.2).

Runs the full COCF-SS-DCA loop end-to-end: encode the prompt, sample ``z_T`` in the
backbone's own latent layout, denoise with per-tube compute allocation, then decode
and write the video.

Usage:
    python scripts/inference/infer_single_video.py \
        --prompt "a cat jumping" \
        --backbone mock \
        --output ./output.mp4 \
        --quality balanced

    # with trained plugins (and Stage-C LoRA, if the checkpoint carries any)
    python scripts/inference/infer_single_video.py \
        --prompt "a cat jumping" \
        --backbone wan22 --model-path /weights/Wan2.2-T2V-A14B \
        --checkpoint ./checkpoints/stage_c_final.pt \
        --output ./output.mp4
"""

import argparse
import logging
from pathlib import Path

import torch

from cocf.common.config import Config
from cocf.common.logging import get_logger, setup_logging
from cocf.common.vram import (
    add_backbone_args,
    add_geometry_args,
    add_perception_args,
    build_perception_and_metrics,
    apply_geometry,
    apply_wan_variant,
    is_real_gpu_backbone,
    resolve_vram_policy,
)
from cocf.core.accelerator import Accelerator
from cocf.data.video_writer import save_video
from cocf.engine import InferenceEngine
from cocf.training.checkpoint import load_checkpoint

# §7.3 compute-budget floor per quality preset. The budget scheduler's B_t is clamped
# to [b_min, b_max], so this is the knob that decides *how much compute the run is
# allowed to skip* — which is what "quality" means here. (Step count is orthogonal
# and stays on --steps.)
QUALITY_B_MIN = {"fast": 0.30, "balanced": 0.50, "quality": 0.80}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Accelerated video inference (§7.2)")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt")
    p.add_argument("--checkpoint", type=Path,
                   help="Trained accelerator checkpoint (Stage B or C). Optional: "
                        "without it the plugins run at their cold-start init.")
    p.add_argument("--output", type=Path, default=Path("./output.mp4"))
    p.add_argument("--quality", choices=sorted(QUALITY_B_MIN), default="balanced",
                   help=f"Compute-budget floor b_min: {QUALITY_B_MIN}")
    p.add_argument("--steps", type=int, help="Override num inference steps")
    p.add_argument("--fps", type=int, default=16, help="Frame rate of the written file")
    # Backbone selection, §9.1 residency and render geometry come from the shared
    # helpers, so a render reproduces what Stage A/C were configured with rather than
    # silently falling back to BackboneConfig's defaults (cocf/common/vram.py).
    add_backbone_args(p, default_backbone="mock", default_wan_variant="a14b-t2v",
                      default_vae_tile=128)
    add_geometry_args(p)
    add_perception_args(p, default_frame_chunk=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    if args.checkpoint and not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")

    setup_logging(level=logging.INFO)
    # Under the framework's ``cocf`` logger namespace: setup_logging attaches the
    # stdout handler there, so a bare ``getLogger(__name__)`` ("__main__") would be
    # silently dropped and the run would print nothing but warnings.
    log = get_logger("cocf.infer")

    # -- config ----------------------------------------------------------- #
    # Seed the global RNG too, not just the z_init generator: without a checkpoint the
    # plugins are randomly initialised (the documented quick start), so leaving the
    # global RNG unseeded makes two runs with the same --seed produce different videos.
    torch.manual_seed(args.seed)
    config = Config()
    config.seed = args.seed
    config.backbone.name = args.backbone
    config.backbone.device = args.device
    config.backbone.dtype = args.backbone_dtype
    if args.model_path:
        config.backbone.model_path = args.model_path
    # Same resolution order as Stage A/C: residency policy, then the variant's
    # geometry/MoE keys, then the render geometry (validated).
    real_gpu_backbone = is_real_gpu_backbone(args)
    resolve_vram_policy(config, args, real_gpu_backbone)
    apply_wan_variant(config, args)
    frames, height, width = apply_geometry(config, args)

    # --quality sets the budget floor (§7.3), matching the documented semantics.
    config.budget.b_min = QUALITY_B_MIN[args.quality]
    config.budget.b_max = max(config.budget.b_max, config.budget.b_min)
    if args.steps:
        config.engine.num_inference_steps = args.steps

    # -- accelerator & engine --------------------------------------------- #
    if args.backbone != "mock":
        args.real_perception = True
        args.require_flow = True
        log.info("Real backbone selected: enabling real semantic-tube perception")
    perception, metric_extractor = build_perception_and_metrics(args, log)
    accelerator = Accelerator.from_config(config, perception=perception,
                                           metric_extractor=metric_extractor)
    backbone = accelerator.backbone
    # The adapter resolves an unavailable backend down to CPU; follow *its* choice so
    # the plugins, z_init and the frozen weights all land on one device.
    device = torch.device(backbone.device)
    if device != torch.device(args.device):
        log.warning("requested device %s is unavailable; running on %s", args.device, device)

    if args.checkpoint and args.checkpoint.exists():
        log.info("Loading checkpoint from %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=str(device), weights_only=False)
        # Two-part build_checkpoint payload only: bare state_dicts and pre-policy
        # checkpoints carry no damage_weights and are rejected by load_checkpoint.
        # Any LoRA the checkpoint carries is re-injected into the frozen backbone.
        n = load_checkpoint(accelerator, ckpt, training_config=config.training)
        if n:
            log.info("Re-attached %d Stage-C LoRA adapter(s)", n)
    else:
        log.info("No --checkpoint given: running with cold-start (untrained) plugins.")

    accelerator.to(device)
    accelerator.eval()
    engine = InferenceEngine(accelerator, config.engine, config.trigger)
    engine.to(device)

    # -- inputs (owned by the adapter: layout, text encoding, noise) -------- #
    # Materialise the weights *before* reading any geometry: a real adapter resolves
    # its true latent channel count / VAE compression during the lazy load, so a
    # token_grid computed beforehand can disagree with the z_init built afterwards.
    backbone.ensure_loaded()
    grid = backbone.token_grid(frames, height, width)
    log.info(
        "Generating %d frames @ %dx%d → token grid %s (%d tokens), %d steps, b_min=%.2f",
        frames, height, width, grid, grid.num_tokens,
        config.engine.num_inference_steps, config.budget.b_min,
    )
    cond = backbone.encode_text([args.prompt]).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z_init = backbone.initial_latent(grid, batch=1, generator=generator, device=device)

    # -- run ---------------------------------------------------------------- #
    log.info("Generating video: '%s'", args.prompt)
    with torch.no_grad():
        result = engine.generate(
            prompts=[args.prompt],
            z_init=z_init,
            grid=grid,
            cond=cond,
            backbone=backbone,
        )

    log.info("Generation complete. Efficiency summary:")
    for key, val in result.summary().items():
        log.info("  %s: %s", key, val)
    log.info(
        "  (mean_compute_ratio is the measured cost; mean_mask_ratio is the "
        "allocation plan and is not a FLOPs saving)"
    )

    # -- write --------------------------------------------------------------- #
    written, backend_name = save_video(result.video, args.output, fps=args.fps)
    log.info("Saved video to %s (via %s)", written, backend_name)


if __name__ == "__main__":
    main()
