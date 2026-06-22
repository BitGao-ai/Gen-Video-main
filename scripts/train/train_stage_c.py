#!/usr/bin/env python
"""Entry script for Stage C: end-to-end lightweight fine-tuning (§4.2).

Embeds the trained plugins into the full accelerated inference pipeline and fine-tunes
the differentiable plugins (L-COCF predictor + strength weights + residual-repair net,
plus an optional LoRA on the last DiT blocks) against the full-compute baseline
``Y_full`` that Stage A persisted — the backbone stays frozen.

Reads the §4.2 source: ``raw_filtered/captions.jsonl`` from the processed store
(``--processed-root``); falls back to a plain video/caption manifest (``--manifest``).

Usage:
    python scripts/train/train_stage_c.py \
        --processed-root ./LCOCF_OpenVid1M_Processed \
        --checkpoint_load ./checkpoints/stage_b_final.pt \
        --use_lora
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
    parser = argparse.ArgumentParser(description="Stage C: end-to-end lightweight fine-tuning (§4.2)")
    parser.add_argument("--processed-root", type=Path,
                        help="Processed store root (§3); reads raw_filtered/ + full_baseline/")
    parser.add_argument("--manifest", type=Path,
                        help="Fallback video/caption manifest when no processed store is given")
    parser.add_argument("--checkpoint_load", type=Path, help="Load accelerator checkpoint (e.g. Stage B)")
    parser.add_argument("--checkpoint_save", type=Path, default=Path("./checkpoints/stage_c_final.pt"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None, help="Override config.training.optim.lr")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)
    torch.manual_seed(args.seed)

    if not args.processed_root and not args.manifest:
        parser.error("provide --processed-root (preferred, §4.2) or --manifest (fallback)")

    config = Config()
    config.seed = args.seed
    if args.lr is not None:
        config.training.optim.lr = args.lr

    log.info("Building accelerator and engine")
    accelerator = Accelerator.from_config(config)
    engine = InferenceEngine(accelerator, config.engine, config.trigger)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info("Loading checkpoint from %s", args.checkpoint_load)
        state_dict = torch.load(args.checkpoint_load, map_location=args.device)
        accelerator.load_state_dict(state_dict)

    stage_c_config = StageCConfig(
        processed_root=args.processed_root,
        manifest_path=args.manifest,
        config=config,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        use_lora=args.use_lora,
        device=torch.device(args.device),
    )

    log.info("Starting Stage C: end-to-end fine-tuning")
    stage_c = FinettuneStage(accelerator=accelerator, engine=engine, config=stage_c_config)
    accelerator = stage_c.run()

    args.checkpoint_save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(accelerator.state_dict(), args.checkpoint_save)
    log.info("Saved final checkpoint to %s", args.checkpoint_save)


if __name__ == "__main__":
    main()
