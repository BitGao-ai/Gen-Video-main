"""Per-step semantic-tube state s_{k,t}."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from cocf.common.config import TubeConfig
from cocf.common.logging import get_logger
from cocf.common.types import SemanticTube, TubeState

Tensor = torch.Tensor
_log = get_logger(__name__)


class TubeStateEncoder:
    """Computes per-step tube states."""

    def __init__(self, config: TubeConfig) -> None:
        """Store tube config."""
        self.cfg = config

    def encode_all(
        self,
        tubes: List[SemanticTube],
        latent_flow_by_frame: Optional[Dict[int, Tensor]] = None,
        causal_values: Optional[Dict[int, float]] = None,
    ) -> Dict[int, TubeState]:
        """Encode states for all tubes."""
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
        """Encode state for one tube."""
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
            anchor_age=prev.anchor_age,
        )
        tube.state = state
        return state

    def _identity_confidence(self, tube: SemanticTube) -> float:
        """Mean cosine identity similarity across consecutive frames."""
        feats = tube.identity_feat_by_frame
        frames = [f for f in tube.frames if f in feats]
        if len(frames) < 2:
            return 1.0
        vectors = [feats[f].detach().float().reshape(-1) for f in frames]
        width = vectors[0].numel()
        if any(v.numel() != width for v in vectors):
            _log.warning(
                "tube %d: identity features have inconsistent widths %s; "
                "identity_confidence falls back to 1.0",
                tube.tube_id, sorted({int(v.numel()) for v in vectors}),
            )
            return 1.0
        normed = torch.nn.functional.normalize(torch.stack(vectors), dim=-1)
        cos = (normed[:-1] * normed[1:]).sum(-1)
        return float(torch.clamp(cos.mean() * 0.5 + 0.5, 0.0, 1.0))

    def _occlusion(
        self, tube: SemanticTube, latent_flow_by_frame: Optional[Dict[int, Tensor]]
    ) -> float:
        """Warped mask overlap deficit across frames."""
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
        """Mean mask IoU of each tube with other tubes."""
        scores = {t.tube_id: 0.0 for t in tubes}
        if len(tubes) < 2:
            return scores
        for i, ti in enumerate(tubes):
            for tj in tubes[i + 1:]:
                shared = set(ti.masks_by_frame) & set(tj.masks_by_frame)
                if not shared:
                    continue
                acc = 0.0
                for f in shared:
                    mi, mj = ti.masks_by_frame[f], tj.masks_by_frame[f]
                    inter = (mi & mj).sum().float()
                    union = (mi | mj).sum().float().clamp_min(1.0)
                    acc += float(inter / union)
                acc /= len(shared)
                scores[ti.tube_id] += acc
                scores[tj.tube_id] += acc
        peers = len(tubes) - 1
        return {tid: min(v / peers, 1.0) for tid, v in scores.items()}

    def _boundary_uncertainty(self, tube: SemanticTube) -> float:
        """Boundary ratio proxy for edge uncertainty."""
        ratios = []
        for mask in tube.masks_by_frame.values():
            area = mask.sum().float().clamp_min(1.0)
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
        """Normalized mean flow magnitude over the tube."""
        if not latent_flow_by_frame:
            return 0.0
        mags = []
        for f, mask in tube.masks_by_frame.items():
            flow = latent_flow_by_frame.get(f)
            if flow is None:
                continue
            mag = flow.pow(2).sum(0).sqrt()
            sel = mag[mask]
            if sel.numel():
                mags.append(float(sel.mean()))
        if not mags:
            return 0.0
        m = sum(mags) / len(mags)
        return float(torch.tanh(torch.tensor(m)))

    @staticmethod
    def _warp(mask: Tensor, flow: Optional[Tensor]) -> Tensor:
        """Forward-warp mask by flow."""
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
