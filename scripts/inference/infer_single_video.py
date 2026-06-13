#!/usr/bin/env python
"""Entry script for accelerated video inference (§7.2).

Usage:
    python scripts/inference/infer_single_video.py \
        --prompt "a cat jumping" \
        --checkpoint ./checkpoints/stage_c_final.pt \
        --output ./output.mp4 \
        --quality balanced
"""

import argparse
import logging
from pathlib import Path

import torch

from cocf.common.config import Config
from cocf.common.logging import setup_logging
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Accelerated video inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=Path, default=Path("./output.mp4"))
    parser.add_argument("--quality", choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--steps", type=int, help="Override num inference steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)

    torch.manual_seed(args.seed)

    # Build config from defaults, then apply the quality preset
    config = Config()
    if args.quality == "fast":
        config.engine.num_inference_steps = 20
    elif args.quality == "quality":
        config.engine.num_inference_steps = 50

    # Override steps if specified
    if args.steps:
        config.engine.num_inference_steps = args.steps

    # Build accelerator & engine
    log.info(f"Loading checkpoint from {args.checkpoint}")
    accelerator = Accelerator.from_config(config)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    accelerator.load_state_dict(state_dict)

    engine = InferenceEngine(accelerator, config.engine, config.trigger)
    engine.to(torch.device(args.device))

    # Prepare input
    log.info(f"Generating video: '{args.prompt}'")
    z_init = torch.randn(1, 1024, 64, device=torch.device(args.device))  # Stub shape

    # TODO: Prepare proper inputs: z_init, grid, cond, etc.

    # Run inference
    with torch.no_grad():
        result = engine.generate(
            prompts=[args.prompt],
            z_init=z_init,
            grid=None,  # TODO: build TokenGrid
            cond=None,  # TODO: encode prompts
            backbone=accelerator.backbone,
        )

    # Log efficiency
    log.info("Generation complete!")
    summary = result.summary()
    for key, val in summary.items():
        log.info(f"  {key}: {val}")

    # Save video (stub)
    log.info(f"Saving video to {args.output}")
    # TODO: Encode video.cpu().numpy() and save


if __name__ == "__main__":
    main()
