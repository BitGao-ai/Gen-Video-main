"""Deterministic mock perception backend for tests and CPU demos.

Implements :class:`PerceptionProvider` without SAM/CLIP/DINOv2/RAFT: it paints a
fixed number of identity-stable blobs that drift across frames, so the STA
matcher links them into clean tubes and every downstream module can be exercised
end-to-end on CPU. Real providers (SAM-2, SigLIP, DINOv2, RAFT) implement the
same four methods and are dropped in via config — no algorithm code changes.
"""

from __future__ import annotations

from typing import List

import torch

from cocf.tubes.regions import PerceptionProvider

Tensor = torch.Tensor


class MockPerception(PerceptionProvider):
    """Synthetic blobs with stable identities and a small constant drift."""

    def __init__(self, num_objects: int = 3, drift: float = 2.0, d_id: int = 64,
                 d_clip: int = 64, seed: int = 0) -> None:
        self.num_objects = num_objects
        self.drift = drift
        self.d_id = d_id
        self.d_clip = d_clip
        g = torch.Generator().manual_seed(seed)
        # fixed per-object identity / text embeddings and base centres / radii
        self._id = torch.randn(num_objects, d_id, generator=g)
        self._txt = torch.randn(num_objects, d_clip, generator=g)
        self._centers0 = torch.rand(num_objects, 2, generator=g)
        self._radius = 0.12 + 0.06 * torch.rand(num_objects, generator=g)
        self._frame_counter = 0

    def segment(self, frame: Tensor) -> Tensor:
        _, hp, wp = frame.shape
        # infer the frame index from the (drift-encoded) mean of the red channel,
        # falling back to an internal counter for robustness.
        fi = int(round(float(frame[0].mean().item()) * 10)) if frame.numel() else self._frame_counter
        self._frame_counter += 1
        yy = torch.linspace(0, 1, hp).reshape(hp, 1)
        xx = torch.linspace(0, 1, wp).reshape(1, wp)
        masks = []
        for o in range(self.num_objects):
            cy = (self._centers0[o, 0] + self.drift / hp * fi) % 1.0
            cx = (self._centers0[o, 1] + self.drift / wp * fi) % 1.0
            d = (yy - cy) ** 2 + (xx - cx) ** 2
            masks.append(d < self._radius[o] ** 2)
        return torch.stack(masks)

    def identity_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        return self._id[self._object_of(mask, frame)]

    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        return 0.9  # all mock objects are "semantic" enough to pass the filter

    def clip_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        return self._txt[self._object_of(mask, frame)]

    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        _, hp, wp = frame_a.shape
        flow = torch.zeros(2, hp, wp)
        flow[0] = self.drift  # dy
        flow[1] = self.drift  # dx
        return flow

    def _object_of(self, mask: Tensor, frame: Tensor) -> int:
        """Match a mask centroid back to the nearest object index (stable identity)."""
        hp, wp = mask.shape
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            return 0
        cy, cx = ys.float().mean() / hp, xs.float().mean() / wp
        d = (self._centers0[:, 0] - cy) ** 2 + (self._centers0[:, 1] - cx) ** 2
        return int(d.argmin())
