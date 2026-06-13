#!/usr/bin/env python
"""Entry script for Stage C: Lightweight fine-tuning (§7.1.3).

Usage:
    python scripts/train/train_stage_c.py \
        --manifest ./data/prompts.csv \
        --checkpoint_load ./checkpoints/after_stage_b.pt
"""

import argparse
import logging
from pathlib import Path

import torch

from cocf.common.config import Config
from cocf.common.logging import setup_logging
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine
from cocf.training.stage_c_finetune import FinettuneStage, StageCConfig


def main():
    parser = argparse.ArgumentParser(description="Stage C: Lightweight fine-tuning")
    parser.add_argument("--checkpoint_load", type=Path, help="Load checkpoint")
    parser.add_argument("--checkpoint_save", type=Path, default=Path("./checkpoints/stage_c_final.pt"))
    parser.add_argument("--manifest", type=Path, help="Video manifest")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)

    torch.manual_seed(args.seed)

    # Build accelerator & engine (module hyper-parameters come from Config() defaults)
    config = Config()

    log.info("Building accelerator and engine")
    accelerator = Accelerator.from_config(config)
    engine = InferenceEngine(accelerator, config.engine, config.trigger)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info(f"Loading checkpoint from {args.checkpoint_load}")
        state_dict = torch.load(args.checkpoint_load, map_location=args.device)
        accelerator.load_state_dict(state_dict)

    # Create Stage C config
    stage_c_config = StageCConfig(
        manifest_path=args.manifest or Path("./data/prompts.csv"),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        device=torch.device(args.device),
    )

    # Run Stage C
    log.info("Starting Stage C: Fine-tuning")
    stage_c = FinettuneStage(accelerator=accelerator, engine=engine, config=stage_c_config)
    accelerator = stage_c.run()

    # Save checkpoint
    args.checkpoint_save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(accelerator.state_dict(), args.checkpoint_save)
    log.info(f"Saved final checkpoint to {args.checkpoint_save}")


if __name__ == "__main__":
    main()
