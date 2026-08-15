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
from cocf.common.logging import get_logger, setup_logging
from cocf.core.accelerator import Accelerator
from cocf.training.checkpoint import load_checkpoint
from cocf.data import CounterfactualLMDBDataset, ProcessedLayout
from cocf.training.stage_b_joint import JointTrainingStage, StageBConfig

# Repo root (…/pro_011). Anchors the default processed-store path so the script runs
# with no flags — mirroring scripts/data/generate_counterfactual_data.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "LCOCF_OpenVid1M_Processed"


def _infer_dims_from_store(layout: ProcessedLayout, log: logging.Logger):
    """Infer (text_dim, visual_dim) from the processed store's sample payloads.

    Stage B must build the CMSC alignment projection with the same dims that Stage A
    used when writing the counterfactual samples. Probing the backbone can silently
    fall back to a wrong default (e.g. 4096 when the mock store holds 16-dim embeds),
    so reading the ground-truth from the data itself is the robust path.
    Returns (text_dim | None, visual_dim | None) — None when the field is absent.
    """
    text_dim = None
    visual_dim = None
    try:
        ds = CounterfactualLMDBDataset(layout.lmdb_dir)
        if len(ds) > 0:
            sample = ds[0]
            te = sample.get("text_embed")
            if te is not None:
                import numpy as np
                arr = np.asarray(te)
                if arr.ndim >= 2:
                    text_dim = int(arr.shape[-1])
            vf = sample.get("tube_visual_embed_full")
            if vf is not None:
                import numpy as np
                arr = np.asarray(vf)
                if arr.ndim >= 1 and arr.shape[-1] > 0:
                    visual_dim = int(arr.shape[-1])
    except Exception as e:
        log.warning("Could not infer dims from store (will probe backbone): %s", e)
    if text_dim is not None:
        log.info("Inferred text_dim=%s from processed store", text_dim)
    if visual_dim is not None:
        log.info("Inferred visual_dim=%s from processed store", visual_dim)
    return text_dim, visual_dim


def _preflight(layout: ProcessedLayout, log: logging.Logger) -> None:
    """Fail fast with an actionable message when the store can't feed Stage B.

    Stage A writes the sample index, the ``splits/`` lists and the committed sample
    store only in its final §1.6 step, so a run interrupted earlier leaves the heavy
    per-video buckets on disk but none of the three things Stage B actually reads.
    Detect that here and say exactly what is missing — instead of the late, generic
    "no training samples" error (or a silent no-op that trains on nothing).
    """
    missing = []
    if not layout.read_split("train"):
        missing.append(f"a non-empty splits/train_list.txt (at {layout.splits_dir})")
    if not layout.sample_index.exists():
        missing.append(f"metadata/sample_index.csv (at {layout.sample_index})")
    if len(CounterfactualLMDBDataset(layout.lmdb_dir)) == 0:
        missing.append(
            f"a committed sample store (at {layout.lmdb_dir}; its LMDB/shard manifest "
            "is empty — Stage A was likely interrupted before §1.6)"
        )
    if missing:
        raise SystemExit(
            f"Processed store '{layout.root}' is not ready for Stage B. Missing:\n  - "
            + "\n  - ".join(missing)
            + "\nRe-run scripts/data/generate_counterfactual_data.py to completion "
            "(its §1.6 step writes the index / splits / norm_stats), or repair the store."
        )


def main():
    parser = argparse.ArgumentParser(description="Stage B: joint module training (§4.1)")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT,
                        help="Root of the six-level processed store (§3), written by Stage A. "
                             f"Defaults to {DEFAULT_PROCESSED_ROOT}.")
    parser.add_argument("--checkpoint_load", type=Path, help="Resume from accelerator checkpoint")
    parser.add_argument("--checkpoint_save", type=Path, default=Path("./checkpoints/stage_b_final.pt"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 keeps the LMDB handle single-process safe")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config.training.optim.lr")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device; auto-detects cuda when available, else cpu")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    # setup_logging attaches the stdout handler to the "cocf" logger and sets
    # propagate=False, so a bare getLogger("__main__") would emit nothing at
    # INFO — this script's own progress lines included.
    log = get_logger("cocf.stage_b")
    torch.manual_seed(args.seed)

    # Fail fast (with a precise message) if Stage A never finished writing the store.
    layout = ProcessedLayout(args.processed_root)
    _preflight(layout, log)

    # The action-balanced sampler drops the last partial batch (drop_last=True), so a
    # batch larger than the whole train split yields zero batches and trains nothing.
    n_train = len(layout.read_split("train"))
    batch_size = args.batch_size
    if batch_size > n_train:
        log.warning("batch_size %d > %d train samples; clamping to %d.",
                    batch_size, n_train, n_train)
        batch_size = max(1, n_train)

    config = Config()
    config.seed = args.seed
    if args.lr is not None:
        config.training.optim.lr = args.lr

    # Infer text/visual embedding dims from the data store so the CMSC alignment
    # projection matches what Stage A actually wrote — avoids the silent fallback
    # in Accelerator._probe_text_dim that can yield a wrong default (e.g. 4096 vs 16).
    text_dim, visual_dim = _infer_dims_from_store(layout, log)

    log.info("Building accelerator")
    accelerator = Accelerator.from_config(config, text_dim=text_dim, visual_dim=visual_dim)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info("Loading checkpoint from %s", args.checkpoint_load)
        ckpt = torch.load(args.checkpoint_load, map_location=args.device,
                          weights_only=False)
        # Either layout: a bare state_dict, or Stage C's {"accelerator", "lora"}.
        # Stage B trains the plugins only, so any LoRA in the checkpoint is loaded
        # into the backbone but not touched by this stage's optimiser.
        load_checkpoint(accelerator, ckpt, training_config=config.training)

    stage_b_config = StageBConfig(
        processed_root=args.processed_root,
        config=config,
        batch_size=batch_size,
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
