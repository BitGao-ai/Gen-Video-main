#!/usr/bin/env python
"""Generate Stage-A counterfactual teacher data."""

import argparse
import logging
from pathlib import Path

from cocf.common.alloc import configure_cuda_allocator

PYTORCH_CUDA_ALLOC_CONF = configure_cuda_allocator()

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OPENVID_CSV = REPO_ROOT / "datasets" / "OpenVidHD.csv"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_VIDEO_SUBDIR = "videos"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "LCOCF_OpenVid1M_Processed"
DEFAULT_MODEL_PATH = REPO_ROOT / "checkpoints" / "wan22"

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
from cocf.training.stage_a_data_gen import DataGenerationStage, StageAConfig


def main():
    """Run Stage-A data generation."""
    parser = argparse.ArgumentParser(
        description="Stage A: generate counterfactual teacher data for L-COCF (§1)"
    )
    parser.add_argument(
        "--openvid-csv", dest="openvid_csvs", type=Path, action="append", default=None,
        help="OpenVid metadata CSV (repeatable; e.g. OpenVid-1M.csv then OpenVidHD.csv). "
             f"Defaults to {DEFAULT_OPENVID_CSV} when omitted.",
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="Root that clips resolve under: {data_root}/{video_subdir}/{video}")
    parser.add_argument("--video-subdir", type=str, default=DEFAULT_VIDEO_SUBDIR,
                        help="Subfolder under --data-root holding the mp4s (e.g. 'videos')")
    parser.add_argument("--only-existing-videos", action="store_true",
                        help="Keep only CSV rows whose mp4 exists on disk (scopes a full "
                             "OpenVid CSV down to the extracted subset you actually have)")
    parser.add_argument("--use-real-video", action="store_true",
                        help="Anchor the teacher trajectory on the real mp4 pixels (VAE-encode "
                             "the clip) instead of caption-only text-to-video; implies "
                             "--only-existing-videos and needs decord or torchvision installed. "
                             "NOTE: this path samples no z_init, and Stage C can only reuse "
                             "Y_full when z_init was persisted with it — so a store built this "
                             "way makes every Stage-C batch re-denoise the baseline.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT,
                        help="Output root of the six-level processed store (§3)")
    add_backbone_args(
        parser,
        default_backbone="wan22",
        default_model_path=str(DEFAULT_MODEL_PATH),
        default_wan_variant="a14b-t2v",
        default_vae_tile=256,
    )
    add_geometry_args(parser)
    add_perception_args(parser, default_frame_chunk=DEFAULT_FRAME_CHUNK)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows read per CSV (debug / smoke)")
    parser.add_argument("--samples-per-video", type=int, default=None,
                        help="Override config.teacher.samples_per_video")
    parser.add_argument("--no-buckets", action="store_true",
                        help="Skip BOTH §3 level-3/level-4 per-video buckets (alias for "
                             "--no-baseline --no-tube-features)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the §3 level-3 full_baseline bucket (Y_full + z_t + z_init). "
                             "Saves ~72 MB/clip at 384x640x49 but disables Stage C's cached "
                             "baseline, so every Stage-C batch re-denoises two full "
                             "trajectories. Only for a store that will not feed Stage C.")
    parser.add_argument("--no-tube-features", action="store_true",
                        help="Skip the §3 level-4 tube_causal_features bucket (small)")
    parser.add_argument("--persist-step-latents", action="store_true",
                        help="Also write the representative-step latents z_t into the "
                             "baseline bucket (~3 MiB per step per clip). No stage reads "
                             "them today; for offline analysis of the teacher trajectory.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total parallel workers over the clip set. Launch N processes "
                             "with the same flags but --shard-index 0..N-1 (e.g. one per GPU). "
                             "Each shard owns a disjoint md5-hash slice and appends to one store.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This worker's 0-based shard id (0 <= shard-index < num-shards). "
                             "Shard 0 also writes the shared global metadata CSVs.")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Skip generation; scan the existing shards and (re)build the "
                             "merged manifest.json + splits + sample_index + norm_stats. Run "
                             "this ONCE after all shards finish. Forces --backbone mock (no "
                             "teacher forward runs, so real weights aren't loaded).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Abort on the first clip that raises. By default a failing "
                             "clip is logged (with traceback) to _failed.sNN.jsonl in the "
                             "store and the shard moves on — a days-long run must not be "
                             "forfeited by one corrupt mp4 or transient OOM. Failed clips "
                             "get no progress line, so re-running retries them.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not (0 <= args.shard_index < max(1, args.num_shards)):
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards "
                     f"(got shard-index={args.shard_index}, num-shards={args.num_shards})")

    setup_logging(level=logging.INFO)
    log = get_logger("cocf.stage_a")
    torch.manual_seed(args.seed)

    openvid_csvs = args.openvid_csvs if args.openvid_csvs else [DEFAULT_OPENVID_CSV]

    config = Config()
    config.backbone.name = "mock" if args.finalize_only else args.backbone
    config.backbone.model_path = args.model_path
    config.backbone.device = args.device
    config.backbone.dtype = args.backbone_dtype
    real_gpu_backbone = is_real_gpu_backbone(args)
    resolve_vram_policy(config, args, real_gpu_backbone)
    apply_wan_variant(config, args)
    apply_geometry(config, args)
    if real_gpu_backbone:
        log_vram_policy(config, log, PYTORCH_CUDA_ALLOC_CONF)
    config.data.video_subdir = args.video_subdir
    config.seed = args.seed

    perception, metric_extractor = build_perception_and_metrics(args, log)

    log.info("Building accelerator with backbone '%s' (perception=%s, metrics=%s)",
             config.backbone.name, "real" if perception else "mock",
             "real" if metric_extractor else "mock")
    accelerator = Accelerator.from_config(
        config, perception=perception, metric_extractor=metric_extractor,
    )

    stage_a_config = StageAConfig(
        openvid_csvs=list(openvid_csvs),
        processed_root=args.processed_root,
        data_root=args.data_root,
        config=config,
        device=torch.device(args.device),
        limit=args.limit,
        samples_per_video=args.samples_per_video,
        persist_buckets=not args.no_buckets,
        persist_baseline=not args.no_baseline,
        persist_tube_features=not args.no_tube_features,
        persist_step_latents=args.persist_step_latents,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        finalize_only=args.finalize_only,
        fail_fast=args.fail_fast,
        seed=args.seed,
        video_subdir=args.video_subdir,
        require_file=args.only_existing_videos or args.use_real_video,
        use_real_video=args.use_real_video,
    )

    log.info("Starting Stage A: counterfactual teacher data generation")
    stage_a = DataGenerationStage(
        config=stage_a_config,
        backbone=accelerator.backbone,
        metric_extractor=accelerator.metric_extractor,
        accelerator=accelerator,
    )
    processed_root = stage_a.run()
    log.info("Stage A complete. Processed store written to %s", processed_root)


if __name__ == "__main__":
    main()
