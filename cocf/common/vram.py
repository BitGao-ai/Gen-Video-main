"""CLI to VRAM-policy wiring for Stage-A/B/C entry points."""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import torch

from cocf.backbones.wan22 import WAN22_VARIANTS
from cocf.common.config import Config
from cocf.common.memory import resolve_dtype
from cocf.data.metrics import DEFAULT_FLOW_MAX_EDGE, DEFAULT_VIT_CHUNK

_log = logging.getLogger(__name__)


def add_backbone_args(
    parser: argparse.ArgumentParser,
    *,
    default_backbone: str = "wan22",
    default_model_path: str = "",
    default_wan_variant: str = "a14b-t2v",
    default_vae_tile: int = 256,
) -> None:
    """Add backbone and VRAM residency arguments."""
    g = parser.add_argument_group("backbone")
    g.add_argument("--backbone", type=str, default=default_backbone,
                   help="Registry key: wan22 (primary) | wan21 | hunyuanvideo | mock")
    g.add_argument("--model-path", type=str, default=default_model_path,
                   help="Frozen-backbone weights dir (§9.1); unused by the mock backbone")
    g.add_argument("--wan-variant", type=str, default=default_wan_variant,
                   choices=sorted(WAN22_VARIANTS),
                   help="Wan2.2 variant → geometry/MoE (§9.1). ti2v-5b is single-expert "
                        "+ high-compression VAE; a14b-* are dual-expert. Must match the "
                        "checkpoint — the adapter raises at load if it does not.")
    g.add_argument("--backbone-dtype", type=str, default="bfloat16",
                   help="Compute dtype for the frozen backbone (bfloat16|float16|float32)")
    g.add_argument("--flow-shift", type=float, default=None,
                   help="Override the variant's rectified-flow schedule shift. The "
                        "shift decides how the step budget is spread over noise levels "
                        "and (on a Wan2.2 MoE) where the expert boundary falls, so it "
                        "must match between Stage A and Stage C; 1.0 is the uniform "
                        "schedule.")

    v = parser.add_argument_group("VRAM residency (§9.1)")
    v.add_argument("--no-offload", dest="offload", action="store_false",
                   help="Keep the frozen text encoder resident. By default a real "
                        "backbone on CUDA parks it on CPU between prompts (~10 GiB on "
                        "Wan2.2). Does NOT affect VAE tiling — see --no-vae-tiling.")
    v.add_argument("--no-text-encoder-exclusive", dest="text_encoder_exclusive",
                   action="store_false",
                   help="Allow the text encoder to share the card with the resident DiT "
                        "expert. NOT recommended below 64 GB: 26 GiB (A14B expert) + "
                        "10 GiB (umT5) + activations does not fit on a 40 GB device, and "
                        "the failure lands on the very first prompt.")
    v.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false",
                   help="Decode/encode the VAE in one un-tiled call. NOT recommended: an "
                        "untiled 480x832x49 Wan decode needs a single ~7.7 GiB block.")
    v.add_argument("--vae-tile", type=int, default=default_vae_tile,
                   help=f"Tile edge in output pixels when VAE tiling is on. Peak decode "
                        f"memory scales with the square of this, tile count with its "
                        f"inverse square. Default {default_vae_tile}.")
    v.add_argument("--no-offload-idle-expert", dest="offload_idle_expert",
                   action="store_false",
                   help="Keep BOTH Wan2.2 MoE experts resident (needs 56+ GB free). "
                        "Impossible on a 40 GB card.")
    v.add_argument("--offload-idle-expert", dest="offload_idle_expert",
                   action="store_true",
                   help="Keep only the current noise band's Wan2.2 expert resident "
                        "(~28 GB saved). This is the default; pass it for an explicit "
                        "record in a launch script.")
    v.add_argument("--offload-device", type=str, default="cpu",
                   help="Where offloaded components (the text encoder, the idle "
                        "Wan2.2 expert) park while idle. 'cpu' (default) is always "
                        "safe; a peer GPU such as 'cuda:1' turns the ~28 GB expert "
                        "swap into a P2P copy instead of a host round trip and frees "
                        "~40 GB of host RAM per process, at the cost of that card "
                        "holding the parked weights for the whole run. Ignored (with "
                        "a warning) when it names the compute device or a device "
                        "this host does not have.")
    parser.set_defaults(offload=True, vae_tiling=True, offload_idle_expert=True,
                        text_encoder_exclusive=True)


def add_geometry_args(
    parser: argparse.ArgumentParser,
    *,
    default_num_frames: Optional[int] = None,
    default_height: Optional[int] = None,
    default_width: Optional[int] = None,
) -> None:
    """Add geometry arguments."""
    g = parser.add_argument_group("geometry (must match across Stage A and Stage C)")
    g.add_argument("--num-frames", type=int, default=default_num_frames,
                   help="Frames per clip; must be 4k+1 for the 4x causal-temporal VAE")
    g.add_argument("--height", type=int, default=default_height,
                   help="Render height; must be divisible by 16 (VAE 8x x patch 2)")
    g.add_argument("--width", type=int, default=default_width,
                   help="Render width; must be divisible by 16 (VAE 8x x patch 2)")


def add_perception_args(
    parser: argparse.ArgumentParser, *, default_frame_chunk: int
) -> None:
    """Add real perception and metric arguments."""
    g = parser.add_argument_group("real perception / metrics")
    g.add_argument("--real-models", action="store_true",
                   help="Shorthand for --real-perception AND --real-metrics.")
    g.add_argument("--real-perception", action="store_true",
                   help="Use the real SAM+DINOv2+CLIP+RAFT tube segmentation "
                        "(cocf.tubes.ModelPerception) instead of the mock blobs.")
    g.add_argument("--real-metrics", action="store_true",
                   help="Use the real DINOv2+CLIP+RAFT damage/CMSC metric extractor "
                        "(cocf.data.ModelMetricExtractor) instead of the mock.")
    g.add_argument("--sam-model", type=str, default="facebook/sam-vit-base",
                   help="SAM checkpoint for mask generation (or a local dir).")
    g.add_argument("--dino-model", type=str, default="facebook/dinov2-base",
                   help="DINOv2 checkpoint for identity/damage features (or a local dir).")
    g.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32",
                   help="CLIP checkpoint for text-alignment features (or a local dir).")
    g.add_argument("--sam-points-per-crop", type=int, default=16,
                   help="SAM point-prompt grid edge; the grid is this squared. 16 => 256 "
                        "prompts. SAM runs once per decoded frame and is the dominant "
                        "non-DiT cost of a real pass.")
    g.add_argument("--perception-dtype", type=str, default="float32",
                   help="Weight dtype for DINOv2/CLIP/SAM (float32|bfloat16|float16). "
                        "RAFT stays fp32 regardless — its all-pairs correlation volume "
                        "is numerically fragile at half precision.")
    g.add_argument("--raft-weights", type=str, default=None,
                   help="Local RAFT checkpoint (.pth). Without it torchvision fetches "
                        "its DEFAULT weights over the network, which an offline host "
                        "cannot do — and a missing RAFT zeroes motion_phase, hence the "
                        "causal action strength s_A. Under --real-models an "
                        "unavailable RAFT is a hard failure, not a silent fallback.")
    g.add_argument("--enable-ocr", action="store_true",
                   help="Add the easyocr OCR-fidelity term to the real metric extractor.")
    g.add_argument("--metric-frame-chunk", type=int, default=default_frame_chunk,
                   help="Frame PAIRS per RAFT forward in the real metric extractor. "
                        "Bounds the correlation volume, the largest allocation in the "
                        f"pass. Default {default_frame_chunk}; lower to 2 or 1 if the "
                        "damage pass OOMs.")
    g.add_argument("--metric-vit-chunk", type=int, default=DEFAULT_VIT_CHUNK,
                   help="Frames per DINOv2/CLIP forward. These resize to 224 and hold "
                        "no correlation volume, so they run far wider than RAFT; "
                        f"default {DEFAULT_VIT_CHUNK}. Lower only on an OOM traced to "
                        "the identity/appearance towers.")
    g.add_argument("--metric-flow-max-edge", type=int, default=DEFAULT_FLOW_MAX_EDGE,
                   help="Longest edge RAFT runs at; frames above it are downscaled. "
                        "The flow damage axes are ratios against the reference's own "
                        "scale, so a resolution both sides share cancels out. "
                        f"Default {DEFAULT_FLOW_MAX_EDGE}; 0 disables the cap.")


def is_real_gpu_backbone(args) -> bool:
    """Check if residency switches apply."""
    return (
        not getattr(args, "finalize_only", False)
        and args.backbone != "mock"
        and str(args.device).startswith("cuda")
    )


def resolve_vram_policy(config: Config, args, real_gpu_backbone: bool) -> None:
    """Apply VRAM residency flags to config."""
    if not real_gpu_backbone:
        return
    config.backbone.offload_text_encoder = args.offload
    config.backbone.text_encoder_exclusive = args.text_encoder_exclusive
    config.backbone.vae_tiling = args.vae_tiling
    if args.vae_tiling:
        config.backbone.vae_tile_size = args.vae_tile
    config.backbone.offload_idle_expert = args.offload_idle_expert
    config.backbone.offload_device = getattr(args, "offload_device", "cpu") or "cpu"


def apply_geometry(config: Config, args) -> Tuple[int, int, int]:
    """Write geometry flags into config."""
    d = config.data
    if getattr(args, "num_frames", None):
        d.num_frames = int(args.num_frames)
    if getattr(args, "height", None):
        d.height = int(args.height)
    if getattr(args, "width", None):
        d.width = int(args.width)
    problems = []
    if (d.num_frames - 1) % 4 != 0:
        problems.append(
            f"--num-frames {d.num_frames} is not 4k+1; the 4x causal-temporal VAE needs "
            f"one (try {((d.num_frames - 1) // 4) * 4 + 1})"
        )
    for name, value in (("--height", d.height), ("--width", d.width)):
        if value % 16 != 0:
            problems.append(
                f"{name} {value} is not divisible by 16 (VAE 8x spatial x patch 2); "
                f"try {value - value % 16}"
            )
    if problems:
        raise SystemExit("Invalid render geometry:\n  - " + "\n  - ".join(problems))
    return d.num_frames, d.height, d.width


def apply_wan_variant(config: Config, args) -> None:
    """Copy variant keys into backbone extra."""
    if getattr(args, "finalize_only", False):
        return
    if str(args.backbone).startswith("wan"):
        config.backbone.extra = {**(config.backbone.extra or {}), **WAN22_VARIANTS[args.wan_variant]}
    shift = getattr(args, "flow_shift", None)
    if shift is not None:
        config.backbone.extra = {**(config.backbone.extra or {}),
                                 "flow_shift": float(shift)}


def build_perception_and_metrics(args, log):
    """Build real perception and metric backends."""
    perception = None
    metric_extractor = None
    finalize_only = getattr(args, "finalize_only", False)
    want_perception = (args.real_perception or args.real_models) and not finalize_only
    want_metrics = (args.real_metrics or args.real_models) and not finalize_only
    raft_weights = getattr(args, "raft_weights", None)
    require_flow = bool(args.real_models or getattr(args, "require_flow", False)) and not finalize_only
    dtype = resolve_dtype(args.perception_dtype)
    if dtype is torch.float32:
        dtype = None

    if want_perception:
        from cocf.tubes import ModelPerception
        log.info("Loading real perception (SAM=%s, DINOv2=%s, CLIP=%s) on %s, "
                 "dtype=%s, %d SAM point prompts",
                 args.sam_model, args.dino_model, args.clip_model, args.device,
                 args.perception_dtype, args.sam_points_per_crop ** 2)
        perception = ModelPerception.from_pretrained(
            device=args.device, sam_model=args.sam_model,
            dino_name=args.dino_model, clip_name=args.clip_model,
            points_per_crop=args.sam_points_per_crop, dtype=dtype,
            raft_weights=raft_weights, require_flow=require_flow,
        )
    if want_metrics:
        from cocf.data import ModelMetricExtractor
        log.info("Loading real metric extractor (DINOv2+CLIP+RAFT%s) on %s, "
                 "vit-chunk %d, raft pair-chunk %d @ max edge %d",
                 " +OCR" if args.enable_ocr else "", args.device,
                 args.metric_vit_chunk, args.metric_frame_chunk,
                 args.metric_flow_max_edge)
        metric_extractor = ModelMetricExtractor.from_pretrained(
            device=args.device, dino_name=args.dino_model,
            clip_name=args.clip_model, enable_ocr=args.enable_ocr,
            frame_chunk=args.metric_frame_chunk,
            vit_chunk=args.metric_vit_chunk,
            flow_max_edge=args.metric_flow_max_edge,
            share_from=perception, dtype=dtype,
            raft_weights=raft_weights, require_flow=require_flow,
        )
    return perception, metric_extractor


def log_vram_policy(config: Config, log, alloc_conf: str) -> None:
    """Log resolved VRAM policy."""
    b = config.backbone
    log.info(
        "VRAM policy: text_encoder=%s, te_exclusive=%s, vae_tiling=%s, idle_expert=%s; "
        "geometry=%dx%dx%d; PYTORCH_CUDA_ALLOC_CONF=%s",
        f"{b.offload_device} between prompts" if b.offload_text_encoder else "RESIDENT",
        "on" if b.text_encoder_exclusive else "OFF",
        f"on (tile {b.vae_tile_size}px)" if b.vae_tiling else "OFF (unbounded decode)",
        f"{b.offload_device} when idle" if b.offload_idle_expert else "both resident",
        config.data.num_frames, config.data.height, config.data.width, alloc_conf,
    )
    if not b.vae_tiling:
        log.warning(
            "--no-vae-tiling: a Wan decode will request a single multi-GiB block, once "
            "per rollout seed. Expect an OOM unless this card has >10 GB free after the "
            "frozen weights load."
        )
    if b.offload_text_encoder and not b.text_encoder_exclusive:
        log.warning(
            "--no-text-encoder-exclusive: umT5 (~10 GiB) will be swapped in on top of "
            "the resident DiT expert (~26 GiB on A14B). Below 64 GB this OOMs at the "
            "first prompt."
        )
