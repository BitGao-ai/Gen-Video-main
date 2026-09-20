#!/usr/bin/env python
"""Train Stage-B plugins jointly on counterfactual store."""

import argparse
import logging
from pathlib import Path

import torch

from cocf.common.config import Config
from cocf.common.logging import get_logger, setup_logging
from cocf.core.accelerator import Accelerator
from cocf.training.checkpoint import load_checkpoint
from cocf.data import CounterfactualLMDBDataset, ProcessedLayout
from cocf.training.distributed import init_distributed
from cocf.training.distributed import resolve_device as dist_device
from cocf.training.distributed import shutdown as dist_shutdown
from cocf.training.stage_b_joint import JointTrainingStage, StageBConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "LCOCF_OpenVid1M_Processed"


def _infer_dims_from_store(layout: ProcessedLayout, log: logging.Logger):
    """Infer text and visual dims from store samples."""
    text_dim = None
    visual_dim = None
    try:
        ds = CounterfactualLMDBDataset(layout.lmdb_dir, text_embed_dir=layout.text_embed_dir)
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
    if text_dim is None:
        try:
            import torch
            for p in sorted(layout.text_embed_dir.glob("*.pt"))[:1]:
                emb = torch.load(p, map_location="cpu")
                text_dim = int(getattr(emb, "shape", [0])[-1]) or None
        except Exception as e:
            log.warning("Could not read text_embeds/: %s", e)
    if text_dim is not None:
        log.info("Inferred text_dim=%s from processed store", text_dim)
    if visual_dim is not None:
        log.info("Inferred visual_dim=%s from processed store", visual_dim)
    return text_dim, visual_dim


def _preflight(layout: ProcessedLayout, log: logging.Logger) -> None:
    """Check store readiness for Stage B."""
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


def _apply_stage_a_geometry(config, layout, log) -> None:
    """Rebuild Stage-A geometry into config from store env."""
    env = layout.read_stage_a_env()
    if not env:
        log.warning(
            "No metadata/stage_a_env.json in this store (written by Stage A). Falling "
            "back to the default backbone geometry — if Stage A ran on a real backbone, "
            "the repair net will be sized wrong and this checkpoint will not load into "
            "Stage C. Re-run Stage A's --finalize-only pass to write it."
        )
        return
    config.backbone.name = "mock"
    config.backbone.extra = {
        "hidden_dim": int(env["token_dim"]),
        "latent_channels": int(env.get("latent_channels", 4)),
    }
    for key in ("num_frames", "height", "width"):
        if key in env:
            setattr(config.data, key, int(env[key]))
    log.info(
        "Stage A env: backbone=%s variant-geometry token_dim=%d, %dx%dx%d, %d steps",
        env.get("backbone", "?"), int(env["token_dim"]),
        int(env.get("num_frames", config.data.num_frames)),
        int(env.get("height", config.data.height)),
        int(env.get("width", config.data.width)),
        int(env.get("teacher_steps", 0)),
    )


def main():
    """Run Stage-B joint training."""
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
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Override config.training.early_stop_patience (default 3). "
                             "Phased mode applies the patience per phase; a slow-converging "
                             "mean phase needs a larger value to avoid stopping at its "
                             "first plateau.")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device; auto-detects cuda when available, else cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--predictor_mean_steps", type=int, default=0,
                        help="Phase 1 length in optimiser steps: scaled MSE/Huber on the "
                             "predictor mean, variance excluded from the loss. "
                             "0 (default) keeps the classic single-phase joint NLL.")
    parser.add_argument("--predictor_var_steps", type=int, default=0,
                        help="Phase 2 length in optimiser steps: only predictor.var_head "
                             "trains, on the plain Gaussian NLL.")
    parser.add_argument("--predictor_joint_lr_scale", type=float, default=0.0,
                        help="If > 0, a phase 3 follows: full joint loss at lr×this scale. "
                             "num_epochs is a budget cap once phases are configured.")
    parser.add_argument("--predictor_mean_objective", choices=["mse", "huber"], default="mse",
                        help="Phase 1 regression form (targets inflated by the scale below).")
    parser.add_argument("--predictor_target_scale", type=float, default=100.0,
                        help="Phase 1 target inflation factor for numerical comfort.")
    parser.add_argument("--predictor_aux_isolation", type=int, choices=[0, 1], default=1,
                        help="1 (default): certificate gets detached mu/sigma in every "
                             "phase and the mean phase drops STA/budget — a pure "
                             "regression control. 0 restores auxiliary gradients.")
    args = parser.parse_args()
    if args.checkpoint_load and not args.checkpoint_load.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint_load}")

    dctx = init_distributed(args.device)
    args.device = dist_device(args.device, dctx)

    setup_logging(level=logging.INFO if dctx.is_main else logging.WARNING)
    log = get_logger("cocf.stage_b")
    torch.manual_seed(args.seed)

    layout = ProcessedLayout(args.processed_root)
    _preflight(layout, log)

    n_train = len(layout.read_split("train")) // max(1, dctx.world_size)
    batch_size = args.batch_size
    if batch_size > n_train:
        log.warning("batch_size %d > %d train samples per rank; clamping to %d.",
                    batch_size, n_train, n_train)
        batch_size = max(1, n_train)

    config = Config()
    config.seed = args.seed
    if args.lr is not None:
        config.training.optim.lr = args.lr
    if args.early_stop_patience is not None:
        if args.early_stop_patience < 1:
            parser.error("--early_stop_patience must be a positive integer")
        config.training.early_stop_patience = args.early_stop_patience
    _apply_stage_a_geometry(config, layout, log)

    text_dim, visual_dim = _infer_dims_from_store(layout, log)

    log.info("Building accelerator")
    accelerator = Accelerator.from_config(config, text_dim=text_dim, visual_dim=visual_dim)

    if args.checkpoint_load and args.checkpoint_load.exists():
        log.info("Loading checkpoint from %s", args.checkpoint_load)
        ckpt = torch.load(args.checkpoint_load, map_location=args.device,
                          weights_only=False)
        load_checkpoint(accelerator, ckpt, training_config=config.training)

    stage_b_config = StageBConfig(
        processed_root=args.processed_root,
        config=config,
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        device=torch.device(args.device),
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_save.parent,
        predictor_mean_steps=args.predictor_mean_steps,
        predictor_var_steps=args.predictor_var_steps,
        predictor_joint_lr_scale=args.predictor_joint_lr_scale,
        predictor_mean_objective=args.predictor_mean_objective,
        predictor_target_scale=args.predictor_target_scale,
        predictor_aux_isolation=bool(args.predictor_aux_isolation),
    )

    log.info("Starting Stage B: joint training")
    stage_b = JointTrainingStage(accelerator=accelerator, config=stage_b_config)
    accelerator = stage_b.run()

    if dctx.is_main:
        args.checkpoint_save.parent.mkdir(parents=True, exist_ok=True)
        from cocf.training.checkpoint import build_checkpoint
        ckpt = build_checkpoint(accelerator)
        if stage_b.phase_state is not None:
            ckpt["phase_state"] = stage_b.phase_state
        torch.save(ckpt, args.checkpoint_save)
        log.info("Saved checkpoint to %s", args.checkpoint_save)
    dist_shutdown()


if __name__ == "__main__":
    main()
