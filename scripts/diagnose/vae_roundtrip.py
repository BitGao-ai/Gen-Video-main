#!/usr/bin/env python
"""VAE round-trip probe: is the project's encode/decode path faithful on this box?

Every "mud" video so far came out of the project's ``decode_latent`` (bf16 VAE +
tiled decode); every good video came out of diffusers' fp32 untiled decode. This
script removes denoising from the question entirely: it encodes a *real* mp4 and
decodes it straight back, twice —

    project : Wan22Backbone.encode_video -> decode_to_unit   (bf16 VAE, tiling as given)
    control : a second, fp32, untiled AutoencoderKLWan directly (official-equivalent)

A faithful VAE round-trips with only mild blur. If the *project* leg is mud while
the *control* leg is clean, the bug lives in the adapter's VAE handling on this
diffusers version — not in the denoiser, the schedule, or the checkpoint.

    python scripts/diagnose/vae_roundtrip.py \
        --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
        --video datasets/videos/some_clip.mp4 \
        --vae-tile 256 --output-dir outputs/vae_roundtrip
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def _read_clip(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """mp4 to [1, 3, F, H, W] in [-1, 1], evenly spaced frames (decord)."""
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    idx = torch.linspace(0, len(vr) - 1, num_frames).round().long().tolist()
    frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).float() / 255.0  # [F,H,W,3]
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,F,H,W]
    frames = F.interpolate(frames, size=(height, width), mode="bilinear",
                           align_corners=False)
    return frames * 2.0 - 1.0


def _stats(name: str, ref: torch.Tensor, out: torch.Tensor) -> None:
    mae = (ref - out).abs().mean().item()
    psnr = -10.0 * torch.log10(((ref - out) ** 2).mean()).item()
    print(f"[{name}] round-trip MAE={mae:.4f} PSNR={psnr:.2f} dB "
          f"(out: mean={out.mean().item():.3f} std={out.std().item():.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-path", required=True)
    p.add_argument("--video", required=True, help="any real mp4 (e.g. an OpenVid clip)")
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--vae-tile", type=int, default=256, help="0 disables tiling")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="outputs/vae_roundtrip")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = _read_clip(args.video, args.num_frames, args.height, args.width)
    ref01 = (x + 1.0) * 0.5  # [0,1] reference for stats

    # -- project leg: the exact adapter methods Stage A / inference use ---------- #
    from cocf.common.config import BackboneConfig
    from cocf.backbones.wan22 import Wan22Backbone
    from cocf.data.video_writer import save_video

    cfg = BackboneConfig(
        name="wan22", model_path=args.model_path, dtype="bfloat16",
        device=args.device, extra={"flow_shift": 5.0},
        vae_tiling=args.vae_tile > 0, vae_tile_size=args.vae_tile or 128,
    )
    bb = Wan22Backbone(cfg)
    lat = bb.encode_video(x.to(args.device))
    print(f"[project] latent: shape={tuple(lat.shape)} "
          f"mean={lat.mean().item():.4f} std={lat.std().item():.4f}")
    y = bb.decode_to_unit(lat).float().cpu().clamp(0, 1)
    _stats("project", ref01[0], y[0])
    save_video(y, out_dir / "roundtrip_project.mp4", fps=16)
    print(f"[project] saved {out_dir/'roundtrip_project.mp4'}")
    del bb
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # -- control leg: fp32 untiled VAE, the official pipeline's configuration ---- #
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float32
    ).to(args.device).eval()
    with torch.no_grad():
        z = vae.encode(x.to(args.device, torch.float32)).latent_dist.sample()
        y2 = vae.decode(z).sample.float().cpu()
    y2 = ((y2 + 1.0) * 0.5).clamp(0, 1)
    _stats("control", ref01[0], y2[0])
    save_video(y2, out_dir / "roundtrip_control.mp4", fps=16)
    print(f"[control] saved {out_dir/'roundtrip_control.mp4'}")

    print("\nverdict: control clean + project mud  → the adapter's bf16/tiled VAE "
          "handling is the bug;\n         both clean                   → VAE "
          "exonerated, the mud is born in the denoise loop itself.")


if __name__ == "__main__":
    main()
