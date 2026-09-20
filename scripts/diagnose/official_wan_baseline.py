#!/usr/bin/env python
"""Official diffusers WanPipeline baseline — the control for the project's backbone path.

Renders one clip with the *stock* ``WanPipeline`` (its own scheduler, MoE routing,
CFG and negative prompt) from the same weights and prompt as
``scripts/inference/run_stage_b_probe.sh``. Interpretation:

* this video good + ``MODE=full`` mud  -> the bug is in the project's adapter path
  (sigma schedule, expert routing, dtype policy, text conditioning), not in the
  weights or the server environment;
* this video mud as well               -> the checkpoint/environment is broken and
  nothing downstream of it (Stage A store included) can be trusted.

Useful variations for bisecting *settings* vs *code* once the stock run is known-good:

    # project-like settings through the stock pipeline:
    python scripts/diagnose/official_wan_baseline.py ... --steps 20 --guidance 1.0 --guidance-2 1.0
"""

import argparse
import inspect

import torch

# The official Wan default negative prompt (Chinese; shipped with the model card).
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompt", default="A person walks slowly across a park, with trees "
                                       "in the background, steady camera.")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=3.0,
                   help="CFG scale (high-noise expert on MoE checkpoints)")
    p.add_argument("--guidance-2", type=float, default=4.0,
                   help="CFG scale of the low-noise expert; ignored when the installed "
                        "diffusers has no guidance_scale_2")
    p.add_argument("--cast-experts-bf16", action="store_true",
                   help="Mimic the project adapter: cast the WHOLE transformer(s) to "
                        "bf16 with .to(), overriding diffusers' keep-in-fp32 modules "
                        "(time_embedder/scale_shift_table/norms). If the stock pipeline "
                        "turns to mud under this flag, that cast is the project's bug.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--output", default="official_baseline.mp4")
    args = p.parse_args()

    import diffusers
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    print(f"diffusers {diffusers.__version__}")

    # VAE in fp32 (official recommendation); the DiT experts in bf16 via torch_dtype,
    # which keeps diffusers' keep-in-fp32 modules (time_embedder, norms) intact —
    # the project's adapter casts those down with a bare .to(), one of the suspects
    # this baseline controls for.
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        args.model_path, vae=vae, torch_dtype=torch.bfloat16
    )
    # Two A14B experts do not fit on one 40 GB card; offload between phases.
    pipe.enable_model_cpu_offload()

    if args.cast_experts_bf16:
        # The project adapter's ``m.to(device, dtype)`` in _ensure_loaded lands here:
        # everything, including the modules from_pretrained deliberately kept fp32.
        for name in ("transformer", "transformer_2"):
            module = getattr(pipe, name, None)
            if module is not None:
                module.to(torch.bfloat16)
        print("[warn] experts force-cast to bf16 (project-adapter emulation)")

    call = inspect.signature(pipe.__call__).parameters
    extra = {}
    if "guidance_scale_2" in call:
        extra["guidance_scale_2"] = args.guidance_2
    elif args.guidance_2 != args.guidance:
        print("[warn] installed diffusers has no guidance_scale_2; using one CFG scale")

    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator().manual_seed(args.seed),
        output_type="np",
        **extra,
    ).frames[0]
    export_to_video(frames, args.output, fps=args.fps)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
