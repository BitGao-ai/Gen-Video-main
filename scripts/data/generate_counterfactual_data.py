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
        --backbone wan22

To process a locally-extracted subset (a full CSV but only some mp4s on disk),
scope the run to the clips you actually have and — optionally — encode their real
pixels into the teacher trajectory:

    python scripts/data/generate_counterfactual_data.py \
        --openvid-csv datasets/OpenVidHD.csv \
        --data-root datasets --video-subdir videos \
        --only-existing-videos --use-real-video \
        --backbone mock --device cpu

For the real server run — Wan2.2 backbone plus the real SAM/DINOv2/CLIP/RAFT
perception and DINOv2/CLIP/RAFT damage metrics (the data that is actually valid for
training the plugins) — add ``--real-models`` (and the Wan variant on the server):

    python scripts/data/generate_counterfactual_data.py \
        --openvid-csv datasets/OpenVidHD.csv --data-root datasets --video-subdir videos \
        --only-existing-videos \
        --backbone wan22 --wan-variant a14b-t2v --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
        --num-frames 49 --height 384 --width 640 \
        --real-models --sam-model facebook/sam-vit-base \
        --device cuda --limit 8

``--use-real-video`` is deliberately absent above. It anchors the teacher trajectory
on the clip's own pixels, which means there is no sampled ``z_init`` to persist — and
Stage C's cached-baseline path needs ``z_init`` and ``Y_full`` *together* (a ``Y_full``
is only a valid reference for a run starting from the same noise). Without it every
Stage-C batch re-denoises two full trajectories and two decodes. Use it only for a
store that will not feed Stage C.

**Geometry is a cross-stage contract.** ``--num-frames/--height/--width`` must match
whatever Stage C renders at, because Stage C's quality loss compares its render
against the ``Y_full`` written here. The values used are recorded in
``metadata/stage_a_env.json`` and Stages B/C read them back, so the agreement is a
property of the store rather than of the command line that ran last.

VRAM, 40 GB card, ``--wan-variant a14b-t2v`` (2×14B MoE): exactly one expert is
resident at ~26.1 GiB, leaving ~12 GiB. That budget only closes because the text
encoder is *exclusive* — umT5 (~10 GiB) is swapped in only after the resident expert
is parked, never on top of it. At 384×640×49 (12,480 tokens) the DiT forward peaks
around 4 GiB and a 256 px VAE tile around 1.4 GiB, for a ~32.5 GiB high-water mark.
At 480×832×49 (20,280 tokens) the same run peaks near 34 GiB and is not recommended
below 48 GB.
If you OOM: lower ``--vae-tile`` to 128, then ``--metric-frame-chunk 2``, then the
render height/width. See §9.1.
"""

import argparse
import logging
from pathlib import Path

# Must run before ``import torch``: the CUDA caching allocator reads
# PYTORCH_CUDA_ALLOC_CONF once, at first CUDA use. See cocf.common.alloc.
from cocf.common.alloc import configure_cuda_allocator

PYTORCH_CUDA_ALLOC_CONF = configure_cuda_allocator()

import torch

# Repository root (…/pro_011), used to anchor the default data/weight paths so the
# script runs from anywhere without any CLI flags.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- default data & weight paths (all resolve under the repo root) ---------- #
# OpenVid metadata CSV consumed by Stage A (§1). Ships in-repo under datasets/.
DEFAULT_OPENVID_CSV = REPO_ROOT / "datasets" / "OpenVidHD.csv"
# Root the mp4 clips resolve under: {data_root}/{video_subdir}/{video}.
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_VIDEO_SUBDIR = "videos"
# Six-level processed store the run writes (§3).
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "LCOCF_OpenVid1M_Processed"
# Frozen-backbone weights (§9.1). Placeholder dir shipped in-repo; only a real
# checkpoint here is used by a non-mock backbone (see PLACEHOLDER_WEIGHTS.md).
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
    # Backbone selection + §9.1 residency, geometry, and the real perception/metric
    # stack all come from cocf.common.vram so Stage A and Stage C cannot drift: Stage
    # C's target is the Y_full rendered here, and a flag that exists on only one side
    # produces a comparison between differently-shaped videos.
    add_backbone_args(
        parser,
        default_backbone="wan22",
        default_model_path=str(DEFAULT_MODEL_PATH),
        # Stage A's decode is label-only, so it takes the largest tile its headroom
        # allows: peak scales with tile², tile *count* with 1/tile². At 384x640 that
        # is 8 tiles at 256 px against 28 at 128 px, for ~1.4 GiB of transient.
        default_vae_tile=256,
    )
    add_geometry_args(parser)
    # By default perception (tube segmentation) and the damage/CMSC metric extractor
    # are MOCK even when --backbone is real, so the labels are only structurally valid.
    # --real-models swaps in the real SAM/DINOv2/CLIP/RAFT stack for data that is
    # actually usable to train the plugins. Ignored under --finalize-only.
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
    # -- shard-parallel + resume (§1 embarrassingly parallel over clips) ------- #
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
    # setup_logging attaches the stdout handler to the "cocf" logger and sets
    # propagate=False, so a bare getLogger("__main__") would emit nothing at
    # INFO — this script's own progress lines included.
    log = get_logger("cocf.stage_a")
    torch.manual_seed(args.seed)

    # ``action="append"`` starts from ``None`` so an explicit --openvid-csv replaces
    # (rather than extends) the default instead of appending to it.
    openvid_csvs = args.openvid_csvs if args.openvid_csvs else [DEFAULT_OPENVID_CSV]

    # Module hyper-parameters (teacher knobs, filter thresholds, data layout) come
    # from Config(); the backbone choice and weight path are overridden from the CLI.
    config = Config()
    # --finalize-only only rebuilds the index from existing shards (no teacher forward),
    # so skip loading the large real backbone weights — the mock backbone still satisfies
    # the quality filter's perception provider used to recompute the split map.
    config.backbone.name = "mock" if args.finalize_only else args.backbone
    config.backbone.model_path = args.model_path
    # Keep the frozen backbone resident on the same device the teacher runs on, so
    # its weights and the noise latents/conditioning built on --device never split
    # across cpu/cuda (else patch_embed/denoise raise a device-mismatch error).
    config.backbone.device = args.device
    config.backbone.dtype = args.backbone_dtype
    # VRAM residency (§9.1) + variant geometry + render geometry, all via the shared
    # helpers so Stage C resolves them identically (cocf/common/vram.py).
    real_gpu_backbone = is_real_gpu_backbone(args)
    resolve_vram_policy(config, args, real_gpu_backbone)
    apply_wan_variant(config, args)
    apply_geometry(config, args)
    if real_gpu_backbone:
        log_vram_policy(config, log, PYTORCH_CUDA_ALLOC_CONF)
    config.data.video_subdir = args.video_subdir
    config.seed = args.seed

    # Build the real perception / metric backends when requested (§1.4/§7.1.1). Left
    # as None otherwise, so Accelerator.from_config falls back to the mocks — the
    # historical, fully-backward-compatible path. --finalize-only skips both (it forces
    # the mock backbone and runs no teacher forward, so loading SAM/DINO/CLIP is waste).
    perception, metric_extractor = build_perception_and_metrics(args, log)

    log.info("Building accelerator with backbone '%s' (perception=%s, metrics=%s)",
             args.backbone, "real" if perception else "mock",
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
        metric_extractor=accelerator.metric_extractor,  # injected (mock by default)
        accelerator=accelerator,
    )
    processed_root = stage_a.run()
    log.info("Stage A complete. Processed store written to %s", processed_root)


if __name__ == "__main__":
    main()
