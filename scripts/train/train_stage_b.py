#!/usr/bin/env python
"""Entry script for Stage B: joint module training (§4.1).

Trains the four learnable plugins (L-COCF predictor + strength weights, STA
smoothing, RAEC certificate, CMSC alignment) together on the Stage-A counterfactual
LMDB store, minimising::

    L_total = L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc + λ_cost·L_budget

The backbone stays frozen, so every gradient lands on the tiny plugin set.

Usage:
    python scripts/train/train_stage_b.py \
        --processed-root ./LCOCF_OpenVid1M_Processed \
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
    parser = argparse.ArgumentParser(description="Stage B: joint module training (§4.1)")
    parser.add_argument("--processed-root", type=Path, required=True,
                        help="Root of the six-level processed store (§3), written by Stage A")
    parser.add_argument("--checkpoint_load", type=Path, help="Resume from accelerator checkpoint")
    parser.add_argument("--checkpoint_save", type=Path, default=Path("./checkpoints/stage_b_final.pt"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 keeps the LMDB handle single-process safe")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config.training.optim.lr")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)
    torch.manual_seed(args.seed)

    config = Config()
    config.seed = args.seed
    if args.lr is not None:
        config.training.optim.lr = args.lr

    log.info("Building accelerator")
    accelerator = Accelerator.from_config(config)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info("Loading checkpoint from %s", args.checkpoint_load)
        state_dict = torch.load(args.checkpoint_load, map_location=args.device)
        accelerator.load_state_dict(state_dict)

    stage_b_config = StageBConfig(
        processed_root=args.processed_root,
        config=config,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        device=torch.device(args.device),
        mixed_precision=args.mixed_precision,
    )

    log.info("Starting Stage B: joint training")
    stage_b = JointTrainingStage(accelerator=accelerator, config=stage_b_config)
    accelerator = stage_b.run()

    args.checkpoint_save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(accelerator.state_dict(), args.checkpoint_save)
    log.info("Saved checkpoint to %s", args.checkpoint_save)


if __name__ == "__main__":
    main()
