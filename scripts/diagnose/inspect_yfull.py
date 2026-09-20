#!/usr/bin/env python
"""Inspect Stage-A Y_full reference video."""

import argparse
import struct
import sys
import zlib
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

_CLAMP_ZERO_FRAC = 0.20
_CLAMP_MEAN = 0.30
_OK_ZERO_FRAC = 0.01
_OK_MEAN = (0.25, 0.65)
_MOCK_MEAN_TOL = 0.03
_STATIC_DELTA = 1e-4


class Stats(NamedTuple):
    """Stats for verdict decision."""

    shape: Tuple[int, ...]
    dtype: str
    vmin: float
    vmax: float
    mean: float
    zero_frac: float
    sat_frac: float
    frame_delta: float


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write uint8 RGB image as PNG."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected [H,W,3] uint8, got {rgb.shape} {rgb.dtype}")
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        """Build PNG chunk."""
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def compute_stats(v: np.ndarray) -> Stats:
    """Summarize clip into verdict stats."""
    f32 = v.astype(np.float32, copy=False)
    if f32.ndim == 5:
        f32 = f32[0]
    delta = 0.0
    if f32.shape[0] >= 2:
        delta = float(np.abs(np.diff(f32, axis=0)).mean())
    return Stats(
        shape=tuple(v.shape),
        dtype=str(v.dtype),
        vmin=float(f32.min()),
        vmax=float(f32.max()),
        mean=float(f32.mean()),
        zero_frac=float((f32 <= 1e-6).mean()),
        sat_frac=float((f32 >= 1.0 - 1e-6).mean()),
        frame_delta=delta,
    )


def verdict(st: Stats) -> Tuple[str, str]:
    """Map stats to failure mode tag."""
    if st.zero_frac > _CLAMP_ZERO_FRAC and st.mean < _CLAMP_MEAN:
        return ("BAD_CLAMP",
                f"{st.zero_frac:.1%} 的像素恰好为 0 且均值仅 {st.mean:.3f} —— "
                "P0-1 修复前的数据:[-1,1] 的解码被直接 clamp(0,1),负半轴全被压成纯黑。"
                "必须重跑 Stage A。")
    if (st.vmax >= 0.999 and abs(st.mean - 0.5) < _MOCK_MEAN_TOL
            and st.zero_frac < 1e-4):
        return ("MOCK",
                f"均值 {st.mean:.4f} ≈ 0.5、max 打满 1.0 且无饱和黑 —— "
                "疑似 mock backbone(decode 只是 latent 最近邻上采样,不是真实 VAE)。"
                "加 --backbone wan22 与真实权重重跑。")
    if st.frame_delta < _STATIC_DELTA:
        return ("STATIC",
                f"相邻帧平均差异仅 {st.frame_delta:.2e} —— 画面几乎不动,"
                "去噪可能没有真正执行或已塌缩到常数。")
    if st.zero_frac < _OK_ZERO_FRAC and _OK_MEAN[0] <= st.mean <= _OK_MEAN[1]:
        return ("OK",
                f"均值 {st.mean:.3f}、精确零占比 {st.zero_frac:.2%}、"
                f"帧间变化 {st.frame_delta:.4f} —— 值域与动态都正常。")
    return ("CHECK",
            f"均值 {st.mean:.3f}、精确零占比 {st.zero_frac:.2%} —— "
            "不匹配任何已知模式,请看图人工判断。")


def montage(v: np.ndarray, n: int, cols: int, gap: int = 4) -> Tuple[np.ndarray, List[int]]:
    """Tile sampled frames into one image."""
    f32 = v.astype(np.float32, copy=False)
    if f32.ndim == 5:
        f32 = f32[0]
    total = f32.shape[0]
    n = max(1, min(n, total))
    idx = np.unique(np.linspace(0, total - 1, n).round().astype(int)).tolist()
    frames = [np.clip(f32[i].transpose(1, 2, 0), 0.0, 1.0) for i in idx]

    h, w, _ = frames[0].shape
    cols = max(1, min(cols, len(frames)))
    rows = (len(frames) + cols - 1) // cols
    canvas = np.full(
        (rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 0.5, np.float32
    )
    for k, fr in enumerate(frames):
        r, c = divmod(k, cols)
        canvas[r * (h + gap): r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = fr
    return (canvas * 255.0).round().astype(np.uint8), idx


def find_clips(root: Path, video_id: Optional[str], limit: int) -> List[Tuple[str, Path]]:
    """Resolve target to Y_full file pairs."""
    if root.is_file() and root.suffix == ".npy":
        return [(root.parent.name, root)]
    baseline = root / "full_baseline"
    if not baseline.is_dir():
        if (root / "Y_full.npy").is_file():
            return [(root.name, root / "Y_full.npy")]
        raise SystemExit(f"找不到 {baseline} —— 请把 --processed-root 指向 Stage A 的输出根目录")
    if video_id:
        p = baseline / video_id / "Y_full.npy"
        if not p.is_file():
            raise SystemExit(f"找不到 {p}")
        return [(video_id, p)]
    found = [(d.name, d / "Y_full.npy") for d in sorted(baseline.iterdir())
             if (d / "Y_full.npy").is_file()]
    if not found:
        raise SystemExit(f"{baseline} 下没有任何 Y_full.npy —— Stage A 可能跑了 --no-baseline")
    return found[:limit]


def main() -> int:
    """Inspect Y_full clips and report verdicts."""
    ap = argparse.ArgumentParser(
        description="抽帧可视化 Stage A 的 Y_full 并判定它属于哪一档失败模式",
    )
    ap.add_argument("root", type=Path,
                    help="Stage A 输出根目录(或直接给一个 Y_full.npy)")
    ap.add_argument("--video-id", type=str, default=None,
                    help="只看这一个 clip;省略则扫描前 --limit 个")
    ap.add_argument("--limit", type=int, default=4, help="扫描的 clip 数上限")
    ap.add_argument("--frames", type=int, default=6, help="抽帧数(默认 6)")
    ap.add_argument("--cols", type=int, default=3, help="拼图列数(默认 3)")
    ap.add_argument("--out-dir", type=Path, default=Path("vis_out"))
    args = ap.parse_args()

    clips = find_clips(args.root, args.video_id, max(1, args.limit))
    print(f"检查 {len(clips)} 个 clip → PNG 写入 {args.out_dir}/\n")

    tally: dict = {}
    for vid, path in clips:
        v = np.load(path, allow_pickle=False, mmap_mode="r")
        st = compute_stats(np.asarray(v))
        tag, why = verdict(st)
        tally[tag] = tally.get(tag, 0) + 1

        img, idx = montage(np.asarray(v), args.frames, args.cols)
        out = args.out_dir / f"{vid}_yfull.png"
        write_png(out, img)

        print(f"── {vid}")
        print(f"   shape={st.shape} {st.dtype}")
        print(f"   min={st.vmin:.4f}  max={st.vmax:.4f}  mean={st.mean:.4f}")
        print(f"   精确零占比={st.zero_frac:.4%}   饱和占比={st.sat_frac:.4%}   "
              f"帧间变化={st.frame_delta:.5f}")
        print(f"   抽帧={idx}  →  {out}")
        print(f"   [{tag}] {why}\n")

    print("汇总: " + "  ".join(f"{k}×{v}" for k, v in sorted(tally.items())))
    return 1 if (tally.get("BAD_CLAMP") or tally.get("MOCK") or tally.get("STATIC")) else 0


if __name__ == "__main__":
    sys.exit(main())
