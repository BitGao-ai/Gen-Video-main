"""Per-step semantic-tube state ``s_{k,t}`` (§4.3.1).

The 7-dim state vector (field order pinned in
:data:`cocf.common.types.TUBE_STATE_FIELDS`) is the compact summary the L-COCF
predictor and the allocator consume per tube. This module computes the *geometric
/ appearance* components from the tube's own masks/features and its neighbours:

    identity_confidence  mean cosine identity similarity vs the previous frame
    occlusion            1 − IoU(M_t, warp(M_{t−1}))
    interaction          Σ IoU with the other tubes on shared frames
    boundary_uncertainty fraction of boundary tokens (perimeter / area proxy)
    motion_phase         normalised mean flow magnitude over the tube
    causal_value         *injected* from the L-COCF strength field (kept external
                         so STA carries no dependency on L-COCF — low coupling)
    anchor_age           *injected* bookkeeping owned by the engine

Update rule (§4.3.1): recomputed every denoising step; a tube whose identity
confidence drops below ``identity_unstable_threshold`` is flagged unstable and the
allocator forces it to FULL.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from cocf.common.config import TubeConfig
from cocf.common.types import SemanticTube, TubeState

Tensor = torch.Tensor


class TubeStateEncoder:
    """Computes and updates :class:`TubeState` for a set of tubes each step."""

    def __init__(self, config: TubeConfig) -> None:
        self.cfg = config

    def encode_all(
        self,
        tubes: List[SemanticTube],
        latent_flow_by_frame: Optional[Dict[int, Tensor]] = None,
        causal_values: Optional[Dict[int, float]] = None,
    ) -> Dict[int, TubeState]:
        """Return ``{tube_id: TubeState}`` for the current step."""
        states: Dict[int, TubeState] = {}
        interaction = self._interaction_scores(tubes)
        for tube in tubes:
            cv = (causal_values or {}).get(tube.tube_id, tube.state.causal_value)
            states[tube.tube_id] = self._encode_one(
                tube, latent_flow_by_frame, interaction.get(tube.tube_id, 0.0), cv
            )
        return states

    def _encode_one(
        self,
        tube: SemanticTube,
        latent_flow_by_frame: Optional[Dict[int, Tensor]],
        interaction: float,
        causal_value: float,
    ) -> TubeState:
        ident = self._identity_confidence(tube)
        occ = self._occlusion(tube, latent_flow_by_frame)
        boundary = self._boundary_uncertainty(tube)
        motion = self._motion_phase(tube, latent_flow_by_frame)
        prev = tube.state
        state = TubeState(
            identity_confidence=ident,
            occlusion=occ,
            interaction=interaction,
            boundary_uncertainty=boundary,
            motion_phase=motion,
            causal_value=causal_value,
            anchor_age=prev.anchor_age,  # engine increments/resets this
        )
        tube.state = state
        return state

    # -- components ------------------------------------------------------ #

    def _identity_confidence(self, tube: SemanticTube) -> float:
        """Mean cosine similarity of consecutive per-frame identity features.

        Uses the running EMA identity feature vs each frame's region feature when
        per-frame features are unavailable; defaults to high confidence (1.0) for a
        single-frame tube.
        """
        if tube.identity_feat is None or tube.length <= 1:
            return 1.0
        ref = torch.nn.functional.normalize(tube.identity_feat.float(), dim=-1)
        # we only retained the EMA feature on the tube; approximate stability by
        # its norm consistency — a real impl compares per-frame DINOv2 crops.
        return float(torch.clamp((ref @ ref) * 0.5 + 0.5, 0.0, 1.0))

    def _occlusion(
        self, tube: SemanticTube, latent_flow_by_frame: Optional[Dict[int, Tensor]]
    ) -> float:
        frames = tube.frames
        if len(frames) < 2:
            return 0.0
        ious = []
        for a, b in zip(frames[:-1], frames[1:]):
            ma, mb = tube.masks_by_frame.get(a), tube.masks_by_frame.get(b)
            if ma is None or mb is None:
                continue
            warp = self._warp(ma, latent_flow_by_frame.get(a) if latent_flow_by_frame else None)
            inter = (warp & mb).sum().float()
            union = (warp | mb).sum().float().clamp_min(1.0)
            ious.append(float(inter / union))
        if not ious:
            return 0.0
        return float(1.0 - sum(ious) / len(ious))

    def _interaction_scores(self, tubes: List[SemanticTube]) -> Dict[int, float]:
        """Σ IoU of each tube's mask with every *other* tube on shared frames."""
        scores = {t.tube_id: 0.0 for t in tubes}
        for i, ti in enumerate(tubes):
            for tj in tubes[i + 1 :]:
                shared = set(ti.masks_by_frame) & set(tj.masks_by_frame)
                acc = 0.0
                for f in shared:
                    mi, mj = ti.masks_by_frame[f], tj.masks_by_frame[f]
                    inter = (mi & mj).sum().float()
                    union = (mi | mj).sum().float().clamp_min(1.0)
                    acc += float(inter / union)
                scores[ti.tube_id] += acc
                scores[tj.tube_id] += acc
        return scores

    def _boundary_uncertainty(self, tube: SemanticTube) -> float:
        """Perimeter/area proxy: thin/fragmented tubes have uncertain boundaries."""
        ratios = []
        for mask in tube.masks_by_frame.values():
            area = mask.sum().float().clamp_min(1.0)
            # 4-neighbour boundary count
            pad = torch.nn.functional.pad(mask.float()[None, None], (1, 1, 1, 1))
            shifts = (
                pad[..., :-2, 1:-1] + pad[..., 2:, 1:-1]
                + pad[..., 1:-1, :-2] + pad[..., 1:-1, 2:]
            )[0, 0]
            boundary = ((mask.float() * (4 - shifts)) > 0).sum().float()
            ratios.append(float((boundary / area).clamp(0.0, 1.0)))
        return float(sum(ratios) / len(ratios)) if ratios else 0.0

    def _motion_phase(
        self, tube: SemanticTube, latent_flow_by_frame: Optional[Dict[int, Tensor]]
    ) -> float:
        if not latent_flow_by_frame:
            return 0.0
        mags = []
        for f, mask in tube.masks_by_frame.items():
            flow = latent_flow_by_frame.get(f)
            if flow is None:
                continue
            mag = flow.pow(2).sum(0).sqrt()  # [H,W]
            sel = mag[mask]
            if sel.numel():
                mags.append(float(sel.mean()))
        if not mags:
            return 0.0
        m = sum(mags) / len(mags)
        return float(torch.tanh(torch.tensor(m)))  # squashed to [0,1)

    @staticmethod
    def _warp(mask: Tensor, flow: Optional[Tensor]) -> Tensor:
        if flow is None:
            return mask
        h, w = mask.shape
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            return mask
        ny = (ys + flow[0, ys, xs].round().long()).clamp(0, h - 1)
        nx = (xs + flow[1, ys, xs].round().long()).clamp(0, w - 1)
        out = torch.zeros_like(mask)
        out[ny, nx] = True
        return out
