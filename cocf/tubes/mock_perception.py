"""Deterministic mock perception backend for tests and CPU demos.

Implements :class:`PerceptionProvider` without SAM/CLIP/DINOv2/RAFT: it paints a
fixed number of identity-stable blobs that drift across frames, so the STA
matcher links them into clean tubes and every downstream module can be exercised
end-to-end on CPU. Real providers (SAM-2, SigLIP, DINOv2, RAFT) implement the
same four methods and are dropped in via config — no algorithm code changes.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from cocf.tubes.regions import PerceptionProvider

Tensor = torch.Tensor


class MockPerception(PerceptionProvider):
    """Synthetic blobs with stable identities and a small constant drift."""

    def __init__(self, num_objects: int = 3, drift: float = 2.0, d_id: int = 64,
                 d_clip: int = 64, seed: int = 0, drift_x: Optional[float] = None) -> None:
        self.num_objects = num_objects
        self.drift = drift
        # Horizontal drift differs from vertical on purpose. With one isotropic value
        # the (dy, dx) flow contract is unfalsifiable — swapping the two channels
        # changes nothing — which is exactly why the real provider could return RAFT's
        # (dx, dy) unnoticed (§P1-11). An anisotropic mock makes a swap observable.
        self.drift_x = float(drift * 2.0 if drift_x is None else drift_x)
        self.d_id = d_id
        self.d_clip = d_clip
        g = torch.Generator().manual_seed(seed)
        # fixed per-object identity / text embeddings and base centres / radii. The
        # CPU Generator forces these onto CPU at construction; ``_align_to`` migrates
        # them to the frame's device on first use so a GPU render yields GPU masks and
        # features (keeping the whole tube path on one device instead of a CPU island).
        self._id = torch.randn(num_objects, d_id, generator=g)
        self._txt = torch.randn(num_objects, d_clip, generator=g)
        self._centers0 = torch.rand(num_objects, 2, generator=g)
        self._radius = 0.12 + 0.06 * torch.rand(num_objects, generator=g)
        self._device = torch.device("cpu")
        self._frame_counter = 0

    def _align_to(self, ref: Tensor) -> None:
        """Move the fixed buffers onto ``ref``'s device on first use.

        A real provider's masks/features follow the (GPU) frame; the mock must too, or
        every tensor it derives forms a CPU island that later collides with GPU tensors.
        """
        if ref.device != self._device:
            self._id = self._id.to(ref.device)
            self._txt = self._txt.to(ref.device)
            self._centers0 = self._centers0.to(ref.device)
            self._radius = self._radius.to(ref.device)
            self._device = ref.device

    def segment(self, frame: Tensor) -> Tensor:
        _, hp, wp = frame.shape
        self._align_to(frame)
        # infer the frame index from the (drift-encoded) mean of the red channel,
        # falling back to an internal counter for robustness.
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
        self._align_to(mask)
        return self._id[self._object_of(mask, frame)]

    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        return 0.9  # all mock objects are "semantic" enough to pass the filter

    def clip_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        self._align_to(mask)
        return self._txt[self._object_of(mask, frame)]

    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        _, hp, wp = frame_a.shape
        flow = torch.zeros(2, hp, wp, device=frame_a.device)
        flow[0] = self.drift    # dy — vertical first, per the provider contract
        flow[1] = self.drift_x  # dx
        return flow

    def _object_of(self, mask: Tensor, frame: Tensor) -> int:
        """Match a mask centroid back to the nearest object index (stable identity)."""
        self._align_to(mask)
        hp, wp = mask.shape
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            return 0
        cy, cx = ys.float().mean() / hp, xs.float().mean() / wp
        d = (self._centers0[:, 0] - cy) ** 2 + (self._centers0[:, 1] - cx) ** 2
        return int(d.argmin())
