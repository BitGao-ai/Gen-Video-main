"""Deterministic mock perception backend for tests and CPU demos."""

from __future__ import annotations

from typing import List, Optional

import torch

from cocf.tubes.regions import PerceptionProvider

Tensor = torch.Tensor


class MockPerception(PerceptionProvider):
    """Synthetic blobs with stable identities and drift."""

    def __init__(self, num_objects: int = 3, drift: float = 2.0, d_id: int = 64,
                 d_clip: int = 64, seed: int = 0, drift_x: Optional[float] = None) -> None:
        """Store mock object count and drift params."""
        self.num_objects = num_objects
        self.drift = drift
        self.drift_x = float(drift * 2.0 if drift_x is None else drift_x)
        self.d_id = d_id
        self.d_clip = d_clip
        g = torch.Generator().manual_seed(seed)
        self._id = torch.randn(num_objects, d_id, generator=g)
        self._txt = torch.randn(num_objects, d_clip, generator=g)
        self._centers0 = torch.rand(num_objects, 2, generator=g)
        self._radius = 0.12 + 0.06 * torch.rand(num_objects, generator=g)
        self._device = torch.device("cpu")
        self._frame_counter = 0

    def _align_to(self, ref: Tensor) -> None:
        """Move buffers onto ref device."""
        if ref.device != self._device:
            self._id = self._id.to(ref.device)
            self._txt = self._txt.to(ref.device)
            self._centers0 = self._centers0.to(ref.device)
            self._radius = self._radius.to(ref.device)
            self._device = ref.device

    def segment(self, frame: Tensor) -> Tensor:
        """Segment drifting blob masks."""
        _, hp, wp = frame.shape
        self._align_to(frame)
        fi = int(round(float(frame[0].mean().item()) * 10)) if frame.numel() else self._frame_counter
        self._frame_counter += 1
        yy = torch.linspace(0, 1, hp, device=frame.device).reshape(hp, 1)
        xx = torch.linspace(0, 1, wp, device=frame.device).reshape(1, wp)
        masks = []
        for o in range(self.num_objects):
            cy = (self._centers0[o, 0] + self.drift / hp * fi) % 1.0
            cx = (self._centers0[o, 1] + self.drift_x / wp * fi) % 1.0
            d = (yy - cy) ** 2 + (xx - cx) ** 2
            masks.append(d < self._radius[o] ** 2)
        return torch.stack(masks)

    def identity_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """Return object identity embedding."""
        self._align_to(mask)
        return self._id[self._object_of(mask, frame)]

    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        """Return fixed high semantic score."""
        return 0.9

    def clip_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """Return object text embedding."""
        self._align_to(mask)
        return self._txt[self._object_of(mask, frame)]

    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """Return constant drift flow field."""
        _, hp, wp = frame_a.shape
        flow = torch.zeros(2, hp, wp, device=frame_a.device)
        flow[0] = self.drift
        flow[1] = self.drift_x
        return flow

    def _object_of(self, mask: Tensor, frame: Tensor) -> int:
        """Match mask centroid to nearest object index."""
        self._align_to(mask)
        hp, wp = mask.shape
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            return 0
        cy, cx = ys.float().mean() / hp, xs.float().mean() / wp
        d = (self._centers0[:, 0] - cy) ** 2 + (self._centers0[:, 1] - cx) ** 2
        return int(d.argmin())
