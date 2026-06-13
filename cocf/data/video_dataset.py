"""Video+caption dataset — frame sampling & bucketing (§7.1, user requirement #4).

The reading/sampling logic deliberately follows the conventions of the
HunyuanVideo and Wan2.1 open-source data pipelines so that latents produced here
are byte-compatible with what those backbones expect:

    * **frame count ``4k+1``** — the 3D *causal* VAE compresses time by 4× with a
      leading key-frame, so a clip must have ``F = 4k+1`` frames (49, 81, 121…).
      (HunyuanVideo ``video_dataset``; Wan2.1 ``T2V`` data spec.)
    * **uniform temporal sampling** with a stride (``frame_interval``) and a random
      start, the standard clip sampler in both repos.
    * **resolution bucketing** — clips are snapped to the nearest configured
      ``(F, H, W)`` bucket by aspect ratio, so a batch is shape-homogeneous (the
      bucket sampler used by both pipelines for variable-aspect training data).
    * **``[-1, 1]`` normalisation** — the pixel convention both VAEs encode from.

Everything heavy (actual decoding) is funnelled through a small
:class:`VideoReader` protocol, so the dataset is dependency-light and unit-testable
with the synthetic reader bundled here; a production run injects a ``decord``/
``torchvision`` reader without touching the sampling logic. This keeps the data
code decoupled from any particular codec backend (user requirement #3).

This module yields *pixels*; turning them into cached latents+embeds (the actual
training memory saving) is :mod:`cocf.data.latent_cache`'s job.
"""

from __future__ import annotations

import abc
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from cocf.common.config import DataConfig
from cocf.common.logging import get_logger

Tensor = torch.Tensor
_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Manifest record
# --------------------------------------------------------------------------- #


@dataclass
class VideoSample:
    """One decoded training clip + its caption (and optional scene tag)."""

    video: Tensor  # [F, 3, H, W] in [-1, 1] (or [0, 1] if normalize off)
    caption: str
    scene: str = "generic"  # static/dynamic/text/face/multi/occlusion (§7.1.1)
    index: int = -1

    @property
    def num_frames(self) -> int:
        return int(self.video.shape[0])


@dataclass
class VideoMeta:
    """A manifest row, before decoding."""

    path: str
    caption: str
    scene: str = "generic"


# --------------------------------------------------------------------------- #
# Pluggable video reader (decoupled from any codec backend)
# --------------------------------------------------------------------------- #


class VideoReader(abc.ABC):
    """Reads frames from a clip. Injected so the dataset is codec-agnostic."""

    @abc.abstractmethod
    def num_frames(self, path: str) -> int:
        """Total frame count of the source clip."""

    @abc.abstractmethod
    def read(self, path: str, frame_indices: Sequence[int]) -> Tensor:
        """Return frames at ``frame_indices`` as ``[len, 3, H, W]`` in ``[0, 1]``."""


class DecordVideoReader(VideoReader):  # pragma: no cover - needs decord + files
    """Production reader backed by ``decord`` (the reader both upstream repos use)."""

    def __init__(self) -> None:
        import decord  # type: ignore

        decord.bridge.set_bridge("torch")
        self._decord = decord

    def num_frames(self, path: str) -> int:
        return len(self._decord.VideoReader(path))

    def read(self, path: str, frame_indices: Sequence[int]) -> Tensor:
        vr = self._decord.VideoReader(path)
        frames = vr.get_batch(list(frame_indices))  # [len, H, W, 3] uint8
        return frames.permute(0, 3, 1, 2).float() / 255.0


class SyntheticVideoReader(VideoReader):
    """Deterministic procedural clips — makes the pipeline runnable with no files.

    Produces a smoothly drifting gradient so that frame sampling, normalisation
    and (later) the mock VAE all exercise real, content-dependent tensors on CPU.
    """

    def __init__(self, length: int = 120, height: int = 64, width: int = 64,
                 seed: int = 0) -> None:
        self._len = length
        self._h = height
        self._w = width
        self._seed = seed

    def num_frames(self, path: str) -> int:
        return self._len

    def read(self, path: str, frame_indices: Sequence[int]) -> Tensor:
        g = torch.Generator().manual_seed(self._seed + (abs(hash(path)) % (2 ** 20)))
        base = torch.rand(3, self._h, self._w, generator=g)
        yy = torch.linspace(0, 1, self._h).reshape(1, self._h, 1)
        out = []
        for fi in frame_indices:
            phase = (fi % self._len) / max(1, self._len)
            frame = (base + phase * yy).clamp(0.0, 1.0)
            out.append(frame)
        return torch.stack(out)


# --------------------------------------------------------------------------- #
# Frame sampling & bucketing (the upstream-compatible logic)
# --------------------------------------------------------------------------- #


def nearest_frame_count(available: int, target: int, interval: int) -> int:
    """Largest ``4k+1`` ≤ ``target`` that fits in ``available`` frames at ``interval``.

    Both VAEs need a ``4k+1`` temporal length; we never up-sample, so a short clip
    falls back to the largest admissible ``4k+1`` it can supply.
    """
    span_target = min(target, (available - 1) // max(1, interval) + 1)
    k = max(0, (span_target - 1) // 4)
    return 4 * k + 1


def sample_frame_indices(
    available: int, num_frames: int, interval: int, *, generator: Optional[torch.Generator] = None
) -> List[int]:
    """Uniform clip sampling with a random start (HunyuanVideo/Wan2.1 sampler).

    Picks ``num_frames`` indices spaced by ``interval`` with a random in-range
    offset; clamps to the clip end so short sources still yield a valid clip.
    """
    span = (num_frames - 1) * interval + 1
    max_start = max(0, available - span)
    if generator is not None and max_start > 0:
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
    else:
        start = 0
    idx = [min(start + i * interval, available - 1) for i in range(num_frames)]
    return idx


def pick_bucket(
    height: int, width: int, buckets: Sequence[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    """Snap a clip to the bucket whose aspect ratio is closest (bucket sampler)."""
    if not buckets:
        raise ValueError("resolution_buckets must be non-empty")
    ar = width / max(1, height)
    return min(buckets, key=lambda b: abs((b[2] / max(1, b[1])) - ar))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class VideoTextDataset(Dataset):
    """Map-style dataset of (clip, caption) pairs with upstream-compatible sampling.

    Parameters
    ----------
    config
        :class:`DataConfig` slice (frame count, interval, buckets, normalisation).
    reader
        Injected :class:`VideoReader`; defaults to the synthetic one so the
        pipeline runs with no data files (tests/CPU demos).
    metas
        Optional explicit manifest; otherwise read from ``config.meta_file``.
    """

    def __init__(
        self,
        config: DataConfig,
        reader: Optional[VideoReader] = None,
        metas: Optional[List[VideoMeta]] = None,
    ) -> None:
        self.cfg = config
        self.reader = reader or SyntheticVideoReader(
            height=config.height // 4, width=config.width // 4
        )
        self.metas = metas if metas is not None else self._load_manifest(config)
        if not self.metas:
            _log.warning("VideoTextDataset is empty (no manifest rows / metas)")

    # -- manifest -------------------------------------------------------- #

    @staticmethod
    def _load_manifest(config: DataConfig) -> List[VideoMeta]:
        """Read a ``.jsonl`` / ``.json`` / ``.csv`` manifest of {video, caption}."""
        path = config.meta_file
        if not path or not os.path.exists(path):
            return []
        rows: List[VideoMeta] = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(VideoTextDataset._row(json.loads(line), config))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                for r in json.load(fh):
                    rows.append(VideoTextDataset._row(r, config))
        elif path.endswith(".csv"):
            with open(path, "r", encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    rows.append(VideoTextDataset._row(r, config))
        else:
            raise ValueError(f"unsupported manifest format: {path}")
        return rows

    @staticmethod
    def _row(r: Dict[str, str], config: DataConfig) -> VideoMeta:
        video = r.get("video") or r.get("path") or r.get("file", "")
        if config.data_root and not os.path.isabs(video):
            video = os.path.join(config.data_root, video)
        caption = r.get("caption") or r.get("text") or r.get("prompt", "")
        return VideoMeta(path=video, caption=caption, scene=r.get("scene", "generic"))

    # -- dataset protocol ------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.metas)

    def __getitem__(self, index: int) -> VideoSample:
        meta = self.metas[index]
        cfg = self.cfg
        g = torch.Generator().manual_seed(cfg.seed + index)

        available = self.reader.num_frames(meta.path)
        n_frames = nearest_frame_count(available, cfg.num_frames, cfg.frame_interval)
        idx = sample_frame_indices(available, n_frames, cfg.frame_interval, generator=g)
        frames = self.reader.read(meta.path, idx)  # [F, 3, h, w] in [0,1]

        f, _, h, w = frames.shape
        _, bh, bw = pick_bucket(h, w, cfg.resolution_buckets)
        frames = _resize_clip(frames, bh, bw)
        if cfg.normalize_to_unit:
            frames = frames * 2.0 - 1.0  # [0,1] → [-1,1] (VAE input convention)
        return VideoSample(video=frames, caption=meta.caption, scene=meta.scene, index=index)


def _resize_clip(frames: Tensor, height: int, width: int) -> Tensor:
    """Bilinear-resize a clip ``[F,3,h,w]`` to ``[F,3,H,W]`` (no aspect cropping)."""
    if frames.shape[-2:] == (height, width):
        return frames
    return torch.nn.functional.interpolate(
        frames, size=(height, width), mode="bilinear", align_corners=False
    )


def collate_video_samples(batch: Sequence[VideoSample]) -> Dict[str, object]:
    """Collate to ``{video:[B,F,3,H,W], captions:[str], scenes:[str]}``.

    Assumes a bucket sampler has made the batch shape-homogeneous; falls back to a
    list when shapes differ so a misconfigured loader fails loud, not silently.
    """
    shapes = {tuple(s.video.shape) for s in batch}
    captions = [s.caption for s in batch]
    scenes = [s.scene for s in batch]
    if len(shapes) == 1:
        video = torch.stack([s.video for s in batch])
        return {"video": video, "captions": captions, "scenes": scenes}
    return {"video": [s.video for s in batch], "captions": captions, "scenes": scenes}
