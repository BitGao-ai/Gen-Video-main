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
        --only-existing-videos --use-real-video \
        --backbone wan22 --wan-variant a14b-t2v --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
        --real-models --sam-model facebook/sam-vit-base \
        --device cuda --limit 8

VRAM (40 GB card, ti2v-5b default): the single 5B expert + hi-comp VAE is ~10 GB
resident, leaving ~30 GB for activations — comfortable on a 40 GB card. The text
encoder is parked on CPU and VAE tiling (128 px) is on by default.
For an 80 GB card with the full A14B dual-expert stack, pass ``--wan-variant a14b-t2v``;
the frozen stack is ~28 GB with idle-expert offload (or ~56 GB without), and the
default 128 px VAE tile keeps the transient under ~0.4 GB.
If you still OOM: lower ``--vae-tile`` to 96, or add ``--metric-frame-chunk 2``.
See §9.1.
"""

import argparse
import logging
import os
from pathlib import Path

# The CUDA caching allocator reads this once, at first CUDA use — so it must be set
# before ``import torch`` initialises anything. Stage A allocates large, *variably
# sized* transients (VAE decode tiles, RAFT correlation volumes) against a small
# residual after the frozen backbone's weights, which fragments the default
# fixed-segment allocator badly: an OOM here typically reports several GB "reserved
# but unallocated". Expandable segments let those blocks be reused across sizes.
#
# Merged rather than ``setdefault``-ed: a launcher that exports the variable for an
# unrelated key (``max_split_size_mb``, ``garbage_collection_threshold``) would
# otherwise silently drop expandable segments and reintroduce exactly the
# fragmentation this guards against. An explicit ``expandable_segments`` in the
# environment still wins.
def _merge_alloc_conf(existing: str, key: str = "expandable_segments", value: str = "True") -> str:
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if any(p.split(":", 1)[0].strip() == key for p in parts):
        return ",".join(parts)
    return ",".join(parts + [f"{key}:{value}"])


PYTORCH_CUDA_ALLOC_CONF = _merge_alloc_conf(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

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
from cocf.core.accelerator import Accelerator
from cocf.data.metrics import DEFAULT_FRAME_CHUNK
from cocf.training.stage_a_data_gen import DataGenerationStage, StageAConfig

# Wan2.2 variant → BackboneConfig.extra (§9.1; mirrors cocf/backbones/wan22.py:24-31).
# a14b-t2v is the documented primary (dual-expert MoE + Wan2.1 VAE, nothing to set);
# ti2v-5b is single-expert with the high-compression VAE. Selected by --wan-variant.
_WAN_VARIANT_EXTRA = {
    "a14b-t2v": {},
    "a14b-i2v": {"boundary_ratio": 0.900},
    "ti2v-5b": {"boundary_ratio": None, "vae_compress": [4, 16, 16], "latent_channels": 48},
}


def resolve_vram_policy(config, args, real_gpu_backbone: bool) -> None:
    """Apply the §9.1 VRAM residency flags to ``config.backbone`` in place.

    Split out of :func:`main` so the CLI→config contract is unit-testable without
    building an accelerator. The contract that matters: **text-encoder offload and
    VAE tiling are independent**. They were once coupled behind a single ``--offload``,
    which meant asking to keep the text encoder resident (a ~11 GB steady-state
    choice) silently also unbounded the VAE decode (a 7.71 GiB *transient*, taken
    ~90x per clip) — the more dangerous of the two, because it OOMs mid-run rather
    than at load. See :meth:`cocf.backbones.diffusers_base.
    DiffusersVideoBackbone._configure_vae_memory`.

    Config defaults are now ON (40 GB-card safe); the CLI --no-* flags explicitly
    disable them for larger cards that want the speed trade-off.
    """
    if not real_gpu_backbone:
        return
    # Explicitly set from CLI so --no-offload / --no-vae-tiling override the
    # config defaults (which are now True for 40 GB safety).
    config.backbone.offload_text_encoder = args.offload
    config.backbone.vae_tiling = args.vae_tiling
    if args.vae_tiling:
        config.backbone.vae_tile_size = args.vae_tile
    config.backbone.offload_idle_expert = args.offload_idle_expert


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
                             "--only-existing-videos and needs decord or torchvision installed")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT,
                        help="Output root of the six-level processed store (§3)")
    parser.add_argument("--backbone", type=str, default="wan22",
                        help="Backbone registry key: wan22 (primary) | wan21 | hunyuanvideo | mock")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Frozen-backbone weights dir (§9.1); only used by a non-mock "
                             f"backbone. Defaults to {DEFAULT_MODEL_PATH}.")
    # -- real perception / metric backends (§1.4/§7.1.1) ---------------------- #
    # By default perception (tube segmentation) and the damage/CMSC metric extractor
    # are MOCK even when --backbone is real, so the labels are only structurally valid.
    # These flags swap in the real SAM/DINOv2/CLIP/RAFT stack for data that is actually
    # usable to train the plugins. Weights auto-download from HF (or point each --*-model
    # at a local dir on an air-gapped server). Ignored under --finalize-only.
    parser.add_argument("--real-models", action="store_true",
                        help="Shorthand for --real-perception AND --real-metrics.")
    parser.add_argument("--real-perception", action="store_true",
                        help="Use the real SAM+DINOv2+CLIP+RAFT tube segmentation "
                             "(cocf.tubes.ModelPerception) instead of the mock blobs.")
    parser.add_argument("--real-metrics", action="store_true",
                        help="Use the real DINOv2+CLIP+RAFT damage/CMSC metric extractor "
                             "(cocf.data.ModelMetricExtractor) instead of the mock.")
    parser.add_argument("--sam-model", type=str, default="facebook/sam-vit-base",
                        help="SAM checkpoint for mask generation (or a local dir). "
                             "Override with facebook/sam-vit-large|huge for finer masks.")
    parser.add_argument("--dino-model", type=str, default="facebook/dinov2-base",
                        help="DINOv2 checkpoint for identity/damage features (or a local dir).")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP checkpoint for text-alignment/appearance features (or a local dir).")
    parser.add_argument("--enable-ocr", action="store_true",
                        help="Add the easyocr OCR-fidelity term to the real metric extractor.")
    parser.add_argument("--metric-frame-chunk", type=int, default=DEFAULT_FRAME_CHUNK,
                        help="Frames (and frame pairs) per forward in the real metric "
                             "extractor. Bounds the RAFT correlation volume, which at "
                             "the default chunk is ~0.6 GB — safe on 40 GB cards. "
                             f"Default {DEFAULT_FRAME_CHUNK}; lower to 2 if the damage "
                             "pass still OOMs.")
    # -- VRAM residency policy for the frozen backbone (§9.1) ----------------- #
    # A real Wan2.2-A14B stack is ~67 GB resident (2×14B experts + 5.5B umT5),
    # which is the whole budget of an 80 GB card before a single activation.
    # These two switches are independent on purpose: --no-offload trades ~11 GB of
    # *steady-state* residency for prompt-encode speed, while VAE tiling bounds the
    # largest *transient*. Conflating them (as this script once did) meant
    # --no-offload silently unbounded the decode, which is the more dangerous of the
    # two — it OOMs mid-run rather than at load.
    parser.add_argument("--no-offload", dest="offload", action="store_false",
                        help="Keep the frozen text encoder resident. By default a real "
                             "backbone on CUDA parks it on CPU between prompts "
                             "(~11 GB on Wan2.2). Does NOT affect VAE tiling — see "
                             "--no-vae-tiling.")
    parser.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false",
                        help="Decode/encode the VAE in one un-tiled call. NOT "
                             "recommended: an untiled 480x832x49 Wan decode needs a "
                             "single ~7.7 GiB block and Stage A runs ~24 decodes per "
                             "clip. Only safe with >10 GB free after the weights load.")
    parser.add_argument("--vae-tile", type=int, default=128,
                        help="Tile edge in output pixels when VAE tiling is on. Peak "
                             "decode memory scales with the square of this, so 128 "
                             "bounds a 480x832 decode at ~0.4 GB (safe on 40 GB cards). "
                             "Raise to 256 (~1.4 GB) only if you have >16 GB free.")
    parser.add_argument("--offload-idle-expert", action="store_true",
                        help="Keep only the active Wan2.2 MoE expert resident (~28 GB "
                             "saved). Costs two ~28 GB transfers per noise-boundary "
                             "crossing — with the default representative steps every "
                             "rollout crosses, so this is throughput-expensive. Prefer "
                             "--wan-variant ti2v-5b when the run is time-bound.")
    parser.add_argument("--no-offload-idle-expert", dest="offload_idle_expert",
                        action="store_false",
                        help="Keep BOTH MoE experts resident (needs 56+ GB free). "
                             "Faster on 80 GB cards — no expert-swap transfers.")
    parser.set_defaults(offload=True, vae_tiling=True, offload_idle_expert=True)
    parser.add_argument("--wan-variant", type=str, default="ti2v-5b",
                        choices=sorted(_WAN_VARIANT_EXTRA),
                        help="Wan2.2 variant → backbone geometry/MoE (§9.1). ti2v-5b (default) "
                             "is single-expert + hi-comp VAE (~10 GB, 40GB-card safe); "
                             "a14b-t2v and a14b-i2v are dual-expert (need 40GB+ with offload).")
    parser.add_argument("--backbone-dtype", type=str, default="bfloat16",
                        help="Compute dtype for the frozen backbone (bfloat16|float16|float32).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows read per CSV (debug / smoke)")
    parser.add_argument("--samples-per-video", type=int, default=None,
                        help="Override config.teacher.samples_per_video")
    parser.add_argument("--no-buckets", action="store_true",
                        help="Skip BOTH §3 level-3/level-4 per-video buckets (alias for "
                             "--no-baseline --no-tube-features)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the §3 level-3 full_baseline bucket — the ~1TB-at-scale "
                             "Y_full/z_t store that Stages B/C do NOT read. Recommended for "
                             "full-scale runs to save disk without affecting training.")
    parser.add_argument("--no-tube-features", action="store_true",
                        help="Skip the §3 level-4 tube_causal_features bucket (small)")
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
    # VRAM residency (§9.1). Only meaningful for a real backbone actually on a GPU:
    # the mock holds no weights, and on CPU an "offload" would be a no-op move that
    # still pays the transfer bookkeeping. --finalize-only forces the mock backbone.
    real_gpu_backbone = (
        not args.finalize_only
        and args.backbone != "mock"
        and str(args.device).startswith("cuda")
    )
    resolve_vram_policy(config, args, real_gpu_backbone)
    if real_gpu_backbone:
        # State the *resolved* policy, not the requested one, so a run's actual VRAM
        # behaviour is readable from the first lines of its log. The backbone logs
        # its measured resident footprint against this once the weights land.
        log.info(
            "VRAM policy: text_encoder=%s, vae_tiling=%s, idle_expert=%s; "
            "PYTORCH_CUDA_ALLOC_CONF=%s",
            "cpu between prompts" if config.backbone.offload_text_encoder else "RESIDENT",
            f"on (tile {config.backbone.vae_tile_size}px)"
            if config.backbone.vae_tiling else "OFF (unbounded decode)",
            "cpu when idle" if config.backbone.offload_idle_expert else "both resident",
            PYTORCH_CUDA_ALLOC_CONF,
        )
        if not config.backbone.vae_tiling:
            log.warning(
                "--no-vae-tiling: a 480x832x49 Wan decode will request a single "
                "~7.7 GiB block, once per rollout seed (~24x per clip). Expect an "
                "OOM unless this card has >10 GB free after the frozen weights load."
            )
    # Wan2.2 variant geometry (§9.1). Only a wan* backbone reads these keys; mock/other
    # adapters ignore ``extra``, so this is a no-op for a mock smoke run. Skipped under
    # --finalize-only (backbone is forced to mock and no teacher forward runs).
    if not args.finalize_only and args.backbone.startswith("wan"):
        config.backbone.extra = dict(_WAN_VARIANT_EXTRA[args.wan_variant])
    config.data.video_subdir = args.video_subdir
    config.seed = args.seed

    # Build the real perception / metric backends when requested (§1.4/§7.1.1). Left
    # as None otherwise, so Accelerator.from_config falls back to the mocks — the
    # historical, fully-backward-compatible path. --finalize-only skips both (it forces
    # the mock backbone and runs no teacher forward, so loading SAM/DINO/CLIP is waste).
    perception = None
    metric_extractor = None
    want_perception = (args.real_perception or args.real_models) and not args.finalize_only
    want_metrics = (args.real_metrics or args.real_models) and not args.finalize_only
    if want_perception:
        from cocf.tubes import ModelPerception
        log.info("Loading real perception (SAM=%s, DINOv2=%s, CLIP=%s) on %s",
                 args.sam_model, args.dino_model, args.clip_model, args.device)
        perception = ModelPerception.from_pretrained(
            device=args.device, sam_model=args.sam_model,
            dino_name=args.dino_model, clip_name=args.clip_model,
        )
    if want_metrics:
        from cocf.data import ModelMetricExtractor
        log.info("Loading real metric extractor (DINOv2+CLIP+RAFT%s) on %s, frame-chunk %d",
                 " +OCR" if args.enable_ocr else "", args.device, args.metric_frame_chunk)
        metric_extractor = ModelMetricExtractor.from_pretrained(
            device=args.device, dino_name=args.dino_model,
            clip_name=args.clip_model, enable_ocr=args.enable_ocr,
            frame_chunk=args.metric_frame_chunk,
            # Reuse the perception backend's DINOv2/CLIP rather than loading a second
            # ~1 GB copy of the same frozen weights (§P2-7).
            share_from=perception,
        )

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
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        finalize_only=args.finalize_only,
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
