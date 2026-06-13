"""Offline latent + text-embedding cache — the dominant training memory saving (#1).

Plugin training (Stages B/C) operates on a *frozen* backbone, so the VAE and the
text encoder produce the **same** latent ``z_0`` and conditioning ``c`` on every
epoch. Running them in the training loop would pin their (large) weights and
activations in VRAM for no benefit. Instead we run them **once, offline** and store
the results on disk; the trainer then reads small tensors and the heavy encoders
never occupy GPU memory during training. This is the single biggest VRAM lever in
the framework (the design doc, §7.1 / ``MemoryConfig.cache_latents``).

The cache is intentionally backbone-tagged: a record stores which backbone wrote
it, so Hunyuan and Wan2.1 caches never get silently mixed (their latent spaces
differ). Reading is a plain ``torch.load`` of a per-sample ``.pt`` file, keyed by a
stable hash of ``(backbone, prompt, index)`` so re-runs are idempotent and resumable.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from cocf.backbones.base import BackboneAdapter, TextConditioning
from cocf.common.config import DataConfig
from cocf.common.logging import get_logger
from cocf.common.memory import free_memory, on_device, teacher_forward
from cocf.common.types import TokenGrid
from cocf.data.video_dataset import VideoSample, VideoTextDataset

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class LatentRecord:
    """One cached training example (small tensors only — no raw frames)."""

    latent: Tensor  # [C, T, H, W] VAE latent of the clip (backbone-native grid)
    text_embeds: Tensor  # [L, d_c] prompt token sequence (conditioning c)
    text_mask: Optional[Tensor]  # [L] padding mask
    pooled: Optional[Tensor]  # [d_c] pooled text embedding (optional)
    grid: TokenGrid  # post-patchify token grid (t, h, w)
    prompt: str
    scene: str
    backbone: str

    def conditioning(self) -> TextConditioning:
        return TextConditioning(
            embeds=self.text_embeds.unsqueeze(0),
            mask=self.text_mask.unsqueeze(0) if self.text_mask is not None else None,
            pooled=self.pooled.unsqueeze(0) if self.pooled is not None else None,
            prompts=(self.prompt,),
        )


def cache_key(backbone: str, prompt: str, index: int) -> str:
    """Stable filename stem for a sample — idempotent across runs/processes."""
    h = hashlib.sha1(f"{backbone}|{index}|{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"{backbone}_{index:07d}_{h}"


class LatentCacheWriter:
    """Encodes a :class:`VideoTextDataset` into on-disk :class:`LatentRecord` files.

    The backbone is moved to the GPU only for the duration of encoding (via
    :func:`cocf.common.memory.on_device`) and offloaded afterwards, so even the
    *caching* pass respects the memory budget. Encoding is gradient-free
    (:func:`teacher_forward`), so no activation graph is ever built.
    """

    def __init__(self, backbone: BackboneAdapter, config: DataConfig) -> None:
        self.backbone = backbone
        self.cfg = config
        os.makedirs(config.cache_dir, exist_ok=True)

    @property
    def _name(self) -> str:
        return getattr(self.backbone.config, "name", "backbone")

    def path_for(self, index: int, prompt: str) -> str:
        return os.path.join(self.cfg.cache_dir, cache_key(self._name, prompt, index) + ".pt")

    def write_sample(self, sample: VideoSample, *, overwrite: bool = False) -> str:
        """Encode one clip+caption and persist it; returns the written path."""
        path = self.path_for(sample.index, sample.caption)
        if os.path.exists(path) and not overwrite:
            return path

        device = self.backbone.device
        module = self.backbone.module
        with teacher_forward():
            ctx = on_device(module, device) if module is not None else _null_ctx()
            with ctx:
                video = sample.video.unsqueeze(0)  # [1, F, 3, H, W]
                # encode_video expects [B, C_pix, F, H, W]
                latent = self.backbone.encode_video(video.permute(0, 2, 1, 3, 4))
                cond = self.backbone.encode_text([sample.caption])
        grid = TokenGrid(*self._infer_grid(latent))

        record = LatentRecord(
            latent=latent[0].to("cpu").float(),
            text_embeds=cond.embeds[0].to("cpu").float(),
            text_mask=cond.mask[0].to("cpu") if cond.mask is not None else None,
            pooled=cond.pooled[0].to("cpu").float() if cond.pooled is not None else None,
            grid=grid,
            prompt=sample.caption,
            scene=sample.scene,
            backbone=self._name,
        )
        _atomic_save(record, path)
        return path

    def write_dataset(
        self, dataset: VideoTextDataset, *, overwrite: bool = False, limit: Optional[int] = None
    ) -> List[str]:
        """Encode an entire dataset, returning the list of written paths."""
        paths: List[str] = []
        n = len(dataset) if limit is None else min(limit, len(dataset))
        for i in range(n):
            paths.append(self.write_sample(dataset[i], overwrite=overwrite))
            if (i + 1) % 50 == 0:
                _log.info("cached %d/%d latents", i + 1, n)
                free_memory()
        return paths

    def _infer_grid(self, latent: Tensor) -> Sequence[int]:
        """Map a ``[B,C,T,H,W]`` latent to its post-patchify token grid (t, h, w)."""
        _, _, t, h, w = latent.shape
        # Reuse the adapter's own pixel→token mapping where possible; fall back to
        # the patchify factor so the mock (no real patch) still produces a grid.
        patch = getattr(self.backbone, "patch", (1, 1, 1))
        return t // patch[0], h // patch[1], w // patch[2]


class LatentCacheDataset(Dataset):
    """Reads back :class:`LatentRecord` ``.pt`` files — no VAE/text encoder needed.

    This is what the Stage-B/C trainers consume. Because it returns only small
    cached tensors, the heavy encoders are entirely absent from the training-time
    memory footprint (user requirement #1).
    """

    def __init__(self, cache_dir: str, paths: Optional[List[str]] = None) -> None:
        if paths is not None:
            self.paths = list(paths)
        else:
            self.paths = sorted(
                os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")
            )
        if not self.paths:
            _log.warning("LatentCacheDataset found no .pt records in %s", cache_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> LatentRecord:
        obj = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        return obj


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _atomic_save(obj: object, path: str) -> None:
    """Write to a temp file then rename — a crashed run never leaves half files."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class _null_ctx:
    """A no-op context manager for backbones without a movable ``module``."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
