#!/usr/bin/env python
"""Entry script for Stage B: Joint module training (§7.1.2).

Usage:
    python scripts/train/train_stage_b.py \
        --cache_dir ./stage_a_data/cache \
        --checkpoint_load ./checkpoints/after_stage_a.pt
"""

import argparse
import logging
from pathlib import Path

import torch

from cocf.common.config import Config
from cocf.common.logging import setup_logging
from cocf.core.accelerator import Accelerator
from cocf.training.stage_b_joint import JointTrainingStage, StageBConfig


def main():
    parser = argparse.ArgumentParser(description="Stage B: Joint module training")
    parser.add_argument("--cache_dir", type=Path, help="Stage A cache directory")
    parser.add_argument("--checkpoint_load", type=Path, help="Load checkpoint")
    parser.add_argument("--checkpoint_save", type=Path, default=Path("./checkpoints/stage_b_final.pt"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)

    torch.manual_seed(args.seed)

    # Build accelerator (module hyper-parameters come from Config() defaults)
    config = Config()

    log.info("Building accelerator")
    accelerator = Accelerator.from_config(config)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info(f"Loading checkpoint from {args.checkpoint_load}")
        state_dict = torch.load(args.checkpoint_load, map_location=args.device)
        accelerator.load_state_dict(state_dict)

    # Create Stage B config
    stage_b_config = StageBConfig(
        cache_dir=args.cache_dir or Path("./stage_a_data/cache"),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=torch.device(args.device),
    )

    # Run Stage B
    log.info("Starting Stage B: Joint training")
    stage_b = JointTrainingStage(accelerator=accelerator, config=stage_b_config)
    accelerator = stage_b.run()

    # Save checkpoint
    args.checkpoint_save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(accelerator.state_dict(), args.checkpoint_save)
    log.info(f"Saved checkpoint to {args.checkpoint_save}")


if __name__ == "__main__":
    main()
