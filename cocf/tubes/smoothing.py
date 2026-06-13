"""Tube-level action smoothing loss (§4.3.2).

Token-level independent decisions cause temporal flicker and local tearing
(§4.1). STA fixes this structurally by deciding per *tube*; this loss adds the
remaining temporal regulariser, suppressing action *jitter across denoising
steps* for the same tube, modulated by how much the tube's latent actually moved:

    L_smooth = λ_temporal · Σ_{k,t} w_{k,t} · ‖p(a_{k,t}) − p(a_{k,t−1})‖²
             + λ_boundary · Σ_{(i,j)∈neigh} IoU_ij · ‖p(a_i) − p(a_j)‖²

with ``w_{k,t} = exp(−‖Δz_{k,t}‖)`` so a tube whose latent barely changed is
*expected* to keep the same action (cheap, stable), while a fast-moving tube is
free to switch. The boundary term couples interacting tubes so neighbouring
objects don't tear apart by taking divergent actions.

The loss operates on the action *probability distributions* the allocator/
predictor produce, so it is differentiable w.r.t. the learned components.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from cocf.common.config import TubeConfig

Tensor = torch.Tensor


class TubeSmoothingLoss:
    """Temporal + boundary action-consistency penalty over tubes."""

    def __init__(self, config: TubeConfig) -> None:
        self.cfg = config

    def __call__(
        self,
        probs_t: Dict[int, Tensor],
        probs_prev: Optional[Dict[int, Tensor]],
        latent_change: Optional[Dict[int, float]] = None,
        neighbours: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Tensor:
        """Compute the scalar smoothing loss for one denoising step.

        Parameters
        ----------
        probs_t, probs_prev
            ``{tube_id: action prob vector [num_actions]}`` at this and the previous
            step. ``probs_prev=None`` (first step) ⇒ no temporal term.
        latent_change
            ``{tube_id: ‖Δz_k‖}`` used for the modulation weight; missing ⇒ weight 1.
        neighbours
            ``[(tube_i, tube_j, iou), …]`` interacting tube pairs for the boundary term.
        """
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
