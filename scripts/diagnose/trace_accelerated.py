#!/usr/bin/env python
"""Trace accelerated trajectory step by step."""

import argparse
import json
import logging
from pathlib import Path

import torch

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
    resolve_vram_policy,
)
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine

GROUPS = ("full", "lowfreq_lattice", "lowfreq_hole", "interp", "anchor", "none")

GROUP_COLORS = {
    "none": (128, 128, 128),
    "full": (0, 200, 0),
    "lowfreq_lattice": (220, 0, 0),
    "lowfreq_hole": (200, 200, 0),
    "interp": (0, 0, 220),
    "anchor": (0, 140, 255),
}


def _group_labels(state, actions, executor, device):
    """Label each token by action group."""
    import numpy as np

    grid = state.grid
    labels = np.array(["none"] * grid.num_tokens, dtype=object)
    for tube in state.tubes:
        action = actions.get(tube.tube_id, "FULL")
        idx = tube.all_token_indices().cpu().numpy()
        if action == "FULL":
            labels[idx] = "full"
        elif action == "LOWFREQ":
            labels[idx] = "lowfreq_hole"
            lattice = executor._strided_indices(tube, grid).cpu().numpy()
            labels[lattice] = "lowfreq_lattice"
        elif action == "INTERP":
            labels[idx] = "interp"
        elif action == "ANCHOR":
            labels[idx] = "anchor"
        else:
            labels[idx] = "full"
    return labels


def _record_step(state, step_idx, trace, executor, prev_z, decode_steps, out_dir,
                 backbone, log):
    """Record latent stats for one step."""
    z = state.z.detach().float()
    actions = trace.actions or {}
    labels = _group_labels(state, actions, executor, z.device)
    rec = {"step": step_idx + 1, "groups": {}}
    for name in GROUPS:
        sel = torch.from_numpy(labels == name)
        n = int(sel.sum())
        if n == 0:
            continue
        vals = z[0, sel.to(z.device)]
        g = {
            "n": n,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "absmax": float(vals.abs().max()),
        }
        if prev_z is not None:
            g["dz"] = float((vals - prev_z[0, sel.to(prev_z.device)]).abs().mean())
        rec["groups"][name] = g
    log.info("trace step %d: %s", step_idx + 1, json.dumps(rec["groups"]))

    if (step_idx + 1) in decode_steps:
        with torch.no_grad():
            video = backbone.decode_to_unit(backbone.to_grid(state.z.detach(), state.grid))
        _save_mid_frame(video, out_dir / f"decode_step{step_idx + 1:02d}.png")

    return rec


def _save_mid_frame(video, path):
    """Save middle frame as PNG."""
    import cv2
    import numpy as np

    mid = video.shape[2] // 2
    frame = video[0, :, mid].permute(1, 2, 0).cpu().numpy()
    frame = (frame.clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(str(path), frame)


def _save_action_map(state, actions, executor, out_dir, log):
    """Save action map for middle frame."""
    import cv2
    import numpy as np

    grid = state.grid
    labels = _group_labels(state, actions, executor, "cpu")
    mid = grid.t // 2
    tpf = grid.tokens_per_frame
    frame_labels = labels[mid * tpf:(mid + 1) * tpf].reshape(grid.h, grid.w)
    img = np.zeros((grid.h, grid.w, 3), dtype=np.uint8)
    for name, color in GROUP_COLORS.items():
        img[frame_labels == name] = color
    scale = 16
    img = cv2.resize(img, (grid.w * scale, grid.h * scale),
                     interpolation=cv2.INTER_NEAREST)
    path = out_dir / "action_map.png"
    cv2.imwrite(str(path), img)
    log.info("action map (latent frame %d) saved to %s", mid, path)


def main():
    """Run accelerated trajectory trace."""
    p = argparse.ArgumentParser(description="Trace the accelerated trajectory")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--quality", choices=["fast", "balanced", "quality"],
                   default="quality")
    p.add_argument("--steps", type=int)
    p.add_argument("--decode-steps", type=str, default="1,2,3,5,10,15,20",
                   help="Comma-separated 1-based step numbers to VAE-decode")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda")
    add_backbone_args(p, default_backbone="wan22", default_wan_variant="a14b-t2v",
                      default_vae_tile=128)
    add_geometry_args(p)
    add_perception_args(p, default_frame_chunk=4)
    args = p.parse_args()

    setup_logging(level=logging.INFO)
    log = get_logger("cocf.trace")

    torch.manual_seed(args.seed)
    config = Config()
    config.seed = args.seed
    config.backbone.name = args.backbone
    config.backbone.device = args.device
    config.backbone.dtype = args.backbone_dtype
    if args.model_path:
        config.backbone.model_path = args.model_path
    real_gpu_backbone = is_real_gpu_backbone(args)
    resolve_vram_policy(config, args, real_gpu_backbone)
    apply_wan_variant(config, args)
    frames, height, width = apply_geometry(config, args)

    quality_b_min = {"fast": 0.30, "balanced": 0.50, "quality": 0.80}
    config.budget.b_min = quality_b_min[args.quality]
    config.budget.b_max = max(config.budget.b_max, config.budget.b_min)
    if args.steps:
        config.engine.num_inference_steps = args.steps

    if args.backbone != "mock":
        args.real_perception = True
        args.require_flow = True
    perception, metric_extractor = build_perception_and_metrics(args, log)
    accelerator = Accelerator.from_config(config, perception=perception,
                                          metric_extractor=metric_extractor)
    backbone = accelerator.backbone
    device = torch.device(backbone.device)

    if args.checkpoint and args.checkpoint.exists():
        from cocf.training.checkpoint import load_checkpoint
        log.info("Loading checkpoint from %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=str(device), weights_only=False)
        load_checkpoint(accelerator, ckpt, training_config=config.training)

    accelerator.to(device)
    accelerator.eval()
    engine = InferenceEngine(accelerator, config.engine, config.trigger)
    engine.to(device)

    backbone.ensure_loaded()
    grid = backbone.token_grid(frames, height, width)
    cond = backbone.encode_text([args.prompt]).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    z_init = backbone.initial_latent(grid, batch=1, generator=generator, device=device)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    decode_steps = {int(s) for s in args.decode_steps.split(",") if s.strip()}
    executor = accelerator.transition

    records = []
    state_holder = {}
    prev_z = None
    orig_step = engine._step

    def wrapped_step(state, step_idx, t, backbone, record_sink=None):
        """Wrap engine step with tracing."""
        nonlocal prev_z
        trace = orig_step(state=state, step_idx=step_idx, t=t, backbone=backbone,
                          record_sink=record_sink)
        rec = _record_step(state, step_idx, trace, executor, prev_z,
                           decode_steps, out_dir, backbone, log)
        records.append(rec)
        if trace.num_tubes and "action_map" not in state_holder:
            _save_action_map(state, trace.actions or {}, executor, out_dir, log)
            state_holder["action_map"] = True
        prev_z = state.z.detach().float().clone()
        return trace

    engine._step = wrapped_step

    log.info("Tracing generation: '%s'", args.prompt)
    with torch.no_grad():
        engine.generate(
            prompts=[args.prompt], z_init=z_init, grid=grid, cond=cond, backbone=backbone,
        )

    trace_path = out_dir / "trace.json"
    trace_path.write_text(json.dumps(records, indent=2))
    log.info("Trace written to %s", trace_path)


if __name__ == "__main__":
    main()
