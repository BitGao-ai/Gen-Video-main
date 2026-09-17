#!/usr/bin/env python3
"""Compare stock Wan and the project adapter, without loading both concurrently.

Run with --model-path PATH --output-dir NEW_DIRECTORY. This is an A14B T2V,
batch-one, no-CFG diagnostic, not a training or production inference entrypoint.
Trace tensors are trusted local artifacts; never load traces from unknown sources.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch


def difference(actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    result = {"actual_shape": list(a.shape), "expected_shape": list(b.shape),
              "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
    if a.shape == b.shape and result["finite"]:
        delta = a - b
        result.update(mae=delta.abs().mean().item(), max_abs=delta.abs().max().item(),
                      relative_l2=(delta.norm() / b.norm().clamp_min(1e-12)).item())
    return result


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contact_sheet(video, path):
    """B,C,T,H,W unit-range video; PNG avoids player/codec ambiguity."""
    from PIL import Image
    x = video.detach().float().cpu()[0]
    if not torch.isfinite(x).all():
        raise ValueError("Decoded video contains NaN/Inf")
    indices = torch.linspace(0, x.shape[1] - 1, 5).long()
    frames = []
    for i in indices:
        a = (x[:, i].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")
        frame = Image.fromarray(a)
        frame.thumbnail((384, 384))
        frames.append(frame)
    sheet = Image.new("RGB", (sum(f.width for f in frames), frames[0].height))
    left = 0
    for frame in frames:
        sheet.paste(frame, (left, 0))
        left += frame.width
    sheet.save(path)


def official(args, root):
    import diffusers
    from diffusers import AutoencoderKLWan, WanPipeline
    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    prepare_latents = pipe.prepare_latents

    def capture_initial(*positional, **kw):
        latents = prepare_latents(*positional, **kw)
        torch.save(latents.detach().cpu(), root / "initial.pt")
        return latents

    pipe.prepare_latents = capture_initial
    handles, count = [], [0]

    def capture(name):
        def hook(module, positional, kw, output):
            if positional or not {"hidden_states", "timestep", "encoder_hidden_states"} <= kw.keys():
                raise RuntimeError("Unsupported transformer call signature; trace would be incomplete")
            eps = output.sample if hasattr(output, "sample") else output[0]
            record = {k: kw[k].detach().cpu() for k in
                      ("hidden_states", "timestep", "encoder_hidden_states")}
            record.update(eps=eps.detach().cpu(), expert=name)
            torch.save(record, root / f"official_step_{count[0]:03d}.pt")
            print(f"official step={count[0]} expert={name} t={record['timestep'].tolist()}", flush=True)
            count[0] += 1
        return hook

    for name in ("transformer", "transformer_2"):
        module = getattr(pipe, name, None)
        if module is not None:
            handles.append(module.register_forward_hook(capture(name), with_kwargs=True))
    extra = {}
    if "guidance_scale_2" in inspect.signature(pipe.__call__).parameters:
        extra["guidance_scale_2"] = 1.0
    with torch.no_grad():
        result = pipe(prompt=args.prompt, height=args.height, width=args.width,
                      num_frames=args.num_frames, num_inference_steps=args.steps,
                      guidance_scale=1.0, generator=torch.Generator().manual_seed(args.seed),
                      output_type="latent", **extra).frames
    for handle in handles:
        handle.remove()
    if count[0] != args.steps:
        raise RuntimeError(f"Expected {args.steps} no-CFG forwards; captured {count[0]}")
    torch.save(result.cpu(), root / "official_final.pt")
    save_json(root / "official_config.json", {
        "diffusers": diffusers.__version__, "torch": torch.__version__,
        "scheduler": dict(pipe.scheduler.config), "timesteps": pipe.scheduler.timesteps.tolist(),
        "sigmas": pipe.scheduler.sigmas.tolist(), "pipeline": dict(pipe.config),
    })
    # The pipeline's VAE receives denormalized latents, not scheduler latents.
    c = result.shape[1]
    mean = torch.tensor(vae.config.latents_mean, device=result.device).view(1, c, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=result.device).view(1, c, 1, 1, 1)
    with torch.no_grad():
        decoded = vae.decode(result.float() * std + mean, return_dict=False)[0]
    contact_sheet((decoded + 1) / 2, root / "official.png")
    pipe.maybe_free_model_hooks()


def project(args, root):
    from cocf.backbones.wan22 import Wan22Backbone
    from cocf.backbones.base import TextConditioning
    from cocf.common.config import BackboneConfig
    from cocf.common.types import TokenGrid
    bb = Wan22Backbone(BackboneConfig(name="wan22", model_path=args.model_path,
        dtype="bfloat16", device="cuda", vae_tile_size=args.vae_tile,
        extra={"flow_shift": args.flow_shift}))
    cond = bb.encode_text([args.prompt])
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    first = load(root / "official_step_000.pt")
    initial = load(root / "initial.pt").to("cuda")
    pt, ph, pw = bb.patch
    grid = TokenGrid(initial.shape[2] // pt, initial.shape[3] // ph, initial.shape[4] // pw)
    z = bb.to_tokens(initial).float()
    report = {"settings": vars(args), "layout_roundtrip": difference(bb.to_grid(z, grid), initial),
              "text_shapes": {"project_raw": list(cond.embeds.shape),
                              "project_transformer_input": list(bb._text_kwargs(cond, bb.transformer)["encoder_hidden_states"].shape),
                              "official": list(first["encoder_hidden_states"].shape)}, "steps": []}
    # Same input, timestep and text: isolates model/dtype policy from text and schedule.
    sigma = first["timestep"].float().to("cuda") / 1000
    t = sigma / (args.flow_shift - (args.flow_shift - 1) * sigma)
    official_cond = TextConditioning(embeds=first["encoder_hidden_states"].to("cuda"))
    with torch.no_grad():
        model_input = first["hidden_states"].to("cuda", dtype=bb.dtype)
        native, _ = bb._run_transformer(model_input, t, cond, False)
        matched, _ = bb._run_transformer(model_input, t, official_cond, False)
    report["first_step_native_text"] = difference(native, first["eps"])
    report["first_step_official_text"] = difference(matched, first["eps"])
    report["text_embedding_error"] = difference(
        bb._text_kwargs(cond, bb.transformer)["encoder_hidden_states"],
        first["encoder_hidden_states"])
    cache = None
    with torch.no_grad():
        for i in range(args.steps):
            ref = load(root / f"official_step_{i:03d}.pt")
            now = torch.tensor([1 - i / args.steps], device="cuda")
            nxt = torch.tensor([1 - (i + 1) / args.steps], device="cuda")
            row = {"step": i, "official_expert": ref["expert"],
                   "project_timestep": (bb.model_sigma(now) * 1000).item(),
                   "official_timestep": ref["timestep"].flatten()[0].item(),
                   "input_error": difference(bb.to_grid(z, grid), ref["hidden_states"])}
            out = bb.full_transition(z, now, nxt, cond, grid=grid, cache=cache)
            row["velocity_error"] = difference(bb.to_grid(out.cache.model_output, grid), ref["eps"])
            z, cache = out.model_output, out.cache
            report["steps"].append(row)
            print(json.dumps(row), flush=True)
            save_json(root / "comparison.json", report)
        final = bb.to_grid(z, grid)
        report["final_latent_error"] = difference(final, load(root / "official_final.pt"))
        torch.save(final.cpu(), root / "project_final.pt")
        contact_sheet(bb.decode_to_unit(final), root / "project.png")
        contact_sheet(bb.decode_to_unit(load(root / "official_final.pt").to("cuda")),
                      root / "official_latent_project_vae.png")
    save_json(root / "comparison.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt", default="A person walks slowly across a park, with trees in the background, steady camera.")
    for name, default in (("steps", 20), ("height", 384), ("width", 640),
                          ("num-frames", 49), ("seed", 1234), ("vae-tile", 256)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--flow-shift", type=float, default=5.0)
    p.add_argument("--worker", choices=("official", "project"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.steps < 1 or args.flow_shift <= 0 or args.num_frames < 1 or args.num_frames % 4 != 1 or min(args.height, args.width) < 16 or args.height % 16 or args.width % 16:
        p.error("Require positive steps/shift, frames=4k+1, and positive 16-aligned dimensions")
    root = Path(args.output_dir)
    if args.worker:
        (official if args.worker == "official" else project)(args, root)
        return
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "settings.json", vars(args))
    for worker in ("official", "project"):
        with (root / f"{worker}.log").open("w") as log:
            print(f"Running {worker}; log: {root / (worker + '.log')}", flush=True)
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            *sys.argv[1:], "--worker", worker], stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Done: {root}/comparison.json and three PNG contact sheets")


if __name__ == "__main__":
    main()
