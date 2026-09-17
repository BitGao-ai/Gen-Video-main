#!/usr/bin/env python
"""Dump a frame grid from a stored Stage-A ``Y_full.npy`` for eyeballing.

The store's ``full_baseline/<video_id>/Y_full.npy`` is the reference video the
damage labels were measured against. If the backbone path that generated it was
broken, these frames are the cheapest place to see it — no GPU, no model load.

    python scripts/diagnose/dump_store_baseline.py \
        --processed-root ./LCOCF_OpenVid1M_Processed \
        --output /tmp/store_yfull.png

Picks the first bucket unless --video-id (or a higher --index) says otherwise.
"""

import argparse
import glob
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed-root", default="./LCOCF_OpenVid1M_Processed")
    p.add_argument("--video-id", help="exact bucket name; overrides --index")
    p.add_argument("--index", type=int, default=0,
                   help="which bucket (sorted order) to dump when --video-id is absent")
    p.add_argument("--grid", type=int, default=3, help="tile the dump as GRID x GRID frames")
    p.add_argument("--output", default="store_yfull.png")
    args = p.parse_args()

    root = Path(args.processed_root) / "full_baseline"
    buckets = sorted(d for d in glob.glob(str(root / "*/")) if Path(d).is_dir())
    if not buckets:
        raise SystemExit(f"no buckets under {root} — was Stage A run with baselines enabled?")

    if args.video_id:
        bucket = root / args.video_id
        if not bucket.is_dir():
            raise SystemExit(f"no such bucket: {bucket}")
    else:
        bucket = Path(buckets[min(args.index, len(buckets) - 1)])
    path = bucket / "Y_full.npy"
    if not path.exists():
        raise SystemExit(f"{path} missing; bucket holds: {sorted(x.name for x in bucket.iterdir())}")

    y = np.load(path).astype(np.float32)
    print(f"bucket : {bucket}")
    print(f"Y_full : shape={y.shape} dtype=fp16→f32 "
          f"min={y.min():.4f} max={y.max():.4f} mean={y.mean():.4f} std={y.std():.4f}")
    if y.ndim != 4 or y.shape[1] != 3:
        raise SystemExit(f"unexpected layout {y.shape}; expected [F, 3, H, W]")

    n = args.grid * args.grid
    f_total = y.shape[0]
    picks = np.linspace(0, f_total - 1, n).round().astype(int)
    frames = (y[picks].transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)
    fh, fw = frames.shape[1:3]
    grid = (frames.reshape(args.grid, args.grid, fh, fw, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(args.grid * fh, args.grid * fw, 3))

    from PIL import Image
    Image.fromarray(grid).save(args.output)
    print(f"frames : {n} evenly spaced of {f_total}, tiled {args.grid}x{args.grid} → {args.output}")
    print("verdict: recognisable scene content = store reference intact; "
          "structureless mud = the teacher path that built the store was broken")


if __name__ == "__main__":
    main()
