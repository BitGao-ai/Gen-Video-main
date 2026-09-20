"""Write a decoded video tensor to disk.

Turns a ``[B, 3, F, H, W]`` (or ``[3, F, H, W]``) float tensor into a file, encoding
best-effort over whatever the environment provides: imageio, torchvision.io, cv2, then
a PNG frame directory, then a raw ``.npy`` dump. Every path returns the file it
actually wrote so the caller can report the truth instead of the requested name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from cocf.common.logging import get_logger

Tensor = torch.Tensor
_log = get_logger(__name__)


def to_uint8_frames(video: Tensor) -> np.ndarray:
    """``[B,3,F,H,W]`` / ``[3,F,H,W]`` float → ``[F, H, W, 3]`` uint8.

    Accepts both the ``[-1, 1]`` VAE convention and an already-``[0, 1]`` tensor: the
    range is detected rather than assumed.
    """
    v = video[0] if video.dim() == 5 else video
    if v.dim() != 4:
        raise ValueError(f"expected [B,3,F,H,W] or [3,F,H,W], got {tuple(video.shape)}")
    v = v.detach().float().cpu()
    if float(v.min()) < -0.01:  # [-1, 1] → [0, 1]
        v = (v + 1.0) / 2.0
    v = v.clamp(0.0, 1.0)
    frames = (v.permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8)  # [F, H, W, 3]
    return frames.numpy()


def save_video(video: Tensor, path, fps: int = 16) -> Tuple[Path, str]:
    """Write ``video`` to ``path``. Returns ``(actual_path, backend_name)``.

    ``actual_path`` may differ from ``path`` when no encoder is installed and the
    function falls back to a frame directory or a ``.npy`` dump.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = to_uint8_frames(video)

    for backend in (_save_imageio, _save_torchvision, _save_cv2):
        try:
            written = backend(frames, path, fps)
        except ImportError:
            continue
        except Exception as exc:  # codec present but unusable → try the next one
            _log.warning("%s failed to encode (%s: %s); trying next backend",
                         backend.__name__, type(exc).__name__, exc)
            continue
        if written is not None:
            return written, backend.__name__.removeprefix("_save_")

    _log.warning(
        "No video encoder available (tried imageio, torchvision.io, cv2). "
        "Install one with:  pip install 'imageio[ffmpeg]'"
    )
    try:
        return _save_png_frames(frames, path), "png_frames"
    except ImportError:
        return _save_npy(frames, path), "npy"


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #


def _save_imageio(frames: np.ndarray, path: Path, fps: int):
    import imageio.v2 as imageio  # raises ImportError when absent

    with imageio.get_writer(str(path), fps=fps, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    return path


def _save_torchvision(frames: np.ndarray, path: Path, fps: int):
    from torchvision.io import write_video

    write_video(str(path), torch.from_numpy(frames), fps=fps)
    return path


def _save_cv2(frames: np.ndarray, path: Path, fps: int):
    import cv2

    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    try:
        for f in frames:
            # ``ascontiguousarray``: the OpenCV bindings reject a negative-stride view,
            # so the RGB->BGR flip must be made contiguous.
            writer.write(np.ascontiguousarray(f[:, :, ::-1]))
    finally:
        writer.release()
    return path


def _save_png_frames(frames: np.ndarray, path: Path) -> Path:
    from PIL import Image

    out_dir = path.with_suffix("").with_name(path.stem + "_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(out_dir / f"frame_{i:05d}.png")
    return out_dir


def _save_npy(frames: np.ndarray, path: Path) -> Path:
    out = path.with_suffix(".npy")
    np.save(out, frames)
    return out
