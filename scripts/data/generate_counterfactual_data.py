#!/usr/bin/env python
"""Entry script for Stage A: offline counterfactual teacher data generation (§1).

Reads the OpenVid-1M metadata CSV(s), runs the four-level quality filter (§2), the
frozen-backbone teacher forward (§1.3–§1.4) and the single-hop counterfactual label
generation (§1.5), and writes the six-level ``LCOCF_OpenVid1M_Processed`` store (§3)
that Stages B/C consume.

Because the teacher generates ``Y_full`` from the *caption* (text-to-video, §1.3),
the whole pass runs end-to-end with only the metadata CSV present — no mp4 files are
required — which is what makes it CPU/mock-testable.

Usage:
    python scripts/data/generate_counterfactual_data.py \
        --openvid-csv data/train/OpenVid-1M.csv \
        --openvid-csv data/train/OpenVidHD.csv \
        --data-root /datasets/OpenVid-1M \
        --processed-root ./LCOCF_OpenVid1M_Processed \
        --backbone hunyuanvideo
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
        description="Stage A: generate counterfactual teacher data for L-COCF (§1)"
    )
    parser.add_argument(
        "--openvid-csv", dest="openvid_csvs", type=Path, action="append", required=True,
        help="OpenVid metadata CSV (repeatable; e.g. OpenVid-1M.csv then OpenVidHD.csv)",
    )
    parser.add_argument("--data-root", type=str, default="",
                        help="Root that clips resolve under: {data_root}/{video_subdir}/{video}")
    parser.add_argument("--processed-root", type=Path, default=Path("./LCOCF_OpenVid1M_Processed"),
                        help="Output root of the six-level processed store (§3)")
    parser.add_argument("--backbone", type=str, default="mock",
                        help="Backbone registry key: hunyuanvideo | wan21 | mock")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows read per CSV (debug / smoke)")
    parser.add_argument("--samples-per-video", type=int, default=None,
                        help="Override config.teacher.samples_per_video")
    parser.add_argument("--no-buckets", action="store_true",
                        help="Skip writing the §3 level-3/level-4 per-video buckets")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    log = logging.getLogger(__name__)
    torch.manual_seed(args.seed)

    # Module hyper-parameters (teacher knobs, filter thresholds, data layout) come
    # from Config(); only the backbone choice is overridden from the CLI.
    config = Config()
    config.backbone.name = args.backbone
    config.seed = args.seed

    log.info("Building accelerator with backbone '%s'", args.backbone)
    accelerator = Accelerator.from_config(config)

    stage_a_config = StageAConfig(
        openvid_csvs=list(args.openvid_csvs),
        processed_root=args.processed_root,
        data_root=args.data_root,
        config=config,
        device=torch.device(args.device),
        limit=args.limit,
        samples_per_video=args.samples_per_video,
        persist_buckets=not args.no_buckets,
        seed=args.seed,
    )

    log.info("Starting Stage A: counterfactual teacher data generation")
    stage_a = DataGenerationStage(
        config=stage_a_config,
        backbone=accelerator.backbone,
        metric_extractor=accelerator.metric_extractor,  # injected (mock by default)
        accelerator=accelerator,
    )
    processed_root = stage_a.run()
    log.info("Stage A complete. Processed store written to %s", processed_root)


if __name__ == "__main__":
    main()
