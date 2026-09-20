"""Tube-level action smoothing loss."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cocf.common.config import TubeConfig

Tensor = torch.Tensor


class TubeSmoothingLoss:
    """Action-consistency penalty over tubes."""

    def __init__(self, config: TubeConfig) -> None:
        """Store tube config."""
        self.cfg = config

    def __call__(
        self,
        probs_t: Dict[int, Tensor],
        probs_prev: Optional[Dict[int, Tensor]],
        latent_change: Optional[Dict[int, float]] = None,
        neighbours: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Tensor:
        """Compute smoothing loss for one step."""
        device = next(iter(probs_t.values())).device if probs_t else torch.device("cpu")
        temporal = torch.zeros((), device=device)
        if probs_prev:
            for k, p in probs_t.items():
                if k not in probs_prev:
                    continue
                dz = (latent_change or {}).get(k, 0.0)
                w = torch.exp(-torch.as_tensor(dz, dtype=p.dtype, device=device))
                temporal = temporal + w * (p - probs_prev[k]).pow(2).sum()

        boundary = torch.zeros((), device=device)
        if neighbours:
            for i, j, iou in neighbours:
                if i in probs_t and j in probs_t:
                    boundary = boundary + iou * (probs_t[i] - probs_t[j]).pow(2).sum()

        return self.cfg.lambda_temporal * temporal + self.cfg.lambda_boundary * boundary
