#!/usr/bin/env python
"""Entry script for Stage A: Generate counterfactual training data (§7.1.1).

Usage:
    python scripts/data/generate_counterfactual_data.py \
        --backbone hunyuan1.5 \
        --output ./stage_a_data
"""

import argparse
import logging
from pathlib import Path

import torch

from cocf.common.config import Config
from cocf.common.logging import setup_logging
from cocf.core.accelerator import Accelerator
from cocf.training.stage_a_data_gen import DataGenerationStage, StageAConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate counterfactual training data for L-COCF"
    )
    parser.add_argument("--backbone", type=str, default="hunyuan1.5", help="Backbone model")
    parser.add_argument("--output", type=Path, default=Path("./stage_a_data"))
    parser.add_argument("--manifest", type=Path, help="Video manifest CSV path")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Set seed
    torch.manual_seed(args.seed)

    # Build accelerator (module hyper-parameters come from Config() defaults)
    config = Config()

    log.info(f"Building accelerator with backbone {args.backbone}")
    config.backbone.name = args.backbone
    accelerator = Accelerator.from_config(config)

    # Create Stage A config
    stage_a_config = StageAConfig(
        manifest_path=args.manifest or Path("./data/prompts.csv"),
        output_dir=args.output,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_prompts=args.max_prompts,
        device=torch.device(args.device),
    )

    # Run Stage A
    log.info("Starting Stage A: Counterfactual data generation")
    stage_a = DataGenerationStage(
        config=stage_a_config,
        backbone=accelerator.backbone,
        metric_extractor=accelerator.metric_extractor,  # Injected (mock by default)
        accelerator=accelerator,
    )
    output_dir = stage_a.run()

    log.info(f"Stage A complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
