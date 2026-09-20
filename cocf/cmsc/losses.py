"""Cross-modal semantic conservation loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.cmsc.alignment import TextTubeAlignment, alignment_risk
from cocf.common.config import CMSCConfig
from cocf.lcocf.damage import VideoFeatures

Tensor = torch.Tensor


def _stack_on(feats: Dict[int, Tensor], ids: List[int], dev=None) -> Tensor:
    """Stack tube features onto device."""
    out = torch.stack([feats[i].float() for i in ids])
    return out if dev is None else out.to(dev)


@dataclass
class CMSCObservation:
    """Features from one rendered video for conservation loss."""

    video: VideoFeatures
    text_embeds: Tensor
    tube_embeds: Dict[int, Tensor] = field(default_factory=dict)
    tube_identity: Dict[int, Tensor] = field(default_factory=dict)
    tube_centroid: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    tube_boundary: Dict[int, Tensor] = field(default_factory=dict)


class CMSCLoss(nn.Module):
    """Computes conservation loss and local certificate term."""

    def __init__(self, cfg: CMSCConfig, alignment: TextTubeAlignment) -> None:
        """Create loss with config and alignment."""
        super().__init__()
        self.cfg = cfg
        self.alignment = alignment

    def forward(
        self, full: CMSCObservation, accel: CMSCObservation
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compare full vs accel observations."""
        c = self.cfg
        ids = sorted(set(full.tube_embeds) & set(accel.tube_embeds))
        dev = full.text_embeds.device

        l_align = self._align_term(full, accel, ids, dev)
        l_id = self._id_term(full.tube_identity, accel.tube_identity, ids, dev)
        l_motion = self._motion_term(full.video, accel.video, dev)
        l_spatial = self._spatial_term(full.tube_centroid, accel.tube_centroid, ids, dev)
        l_ocr = self._ocr_term(full.video, accel.video, dev)
        l_bnd = self._id_term(full.tube_boundary, accel.tube_boundary, ids, dev)

        total = (
            c.lambda_align * l_align
            + c.lambda_id * l_id
            + c.lambda_motion * l_motion
            + c.lambda_spatial * l_spatial
            + c.lambda_ocr * l_ocr
            + c.lambda_bnd * l_bnd
        )
        components = {
            "align": float(l_align.detach()), "id": float(l_id.detach()),
            "motion": float(l_motion.detach()), "spatial": float(l_spatial.detach()),
            "ocr": float(l_ocr.detach()), "bnd": float(l_bnd.detach()),
        }
        return total, components

    # ------------------------------------------------------------------ #
    # batched offline conservation (Stage-B counterfactual samples)
    # ------------------------------------------------------------------ #

    def alignment_conservation(
        self,
        text_embeds: Tensor,                 # [B, L, d_c] padded prompt tokens
        tube_full: Tensor,                   # [B, d_v] full-render tube visual embed
        tube_cf: Tensor,                     # [B, d_v] cf-render tube visual embed
        text_mask: Optional[Tensor] = None,  # [B, L] 1=keep
    ) -> Tensor:
        """Compute per-sample alignment conservation."""
        al = self.alignment
        t = F.normalize(al.text_proj(text_embeds.float()), dim=-1)
        vf = F.normalize(al.vis_proj(tube_full.float()), dim=-1)
        vc = F.normalize(al.vis_proj(tube_cf.float()), dim=-1)
        sim_f = torch.einsum("bla,ba->bl", t, vf)
        sim_c = torch.einsum("bla,ba->bl", t, vc)
        if text_mask is not None:
            fill = torch.finfo(sim_f.dtype).min
            keep = text_mask > 0.5
            sim_f = sim_f.masked_fill(~keep, fill)
            sim_c = sim_c.masked_fill(~keep, fill)
        score_f = sim_f.max(dim=-1).values.clamp(-1, 1) * 0.5 + 0.5
        score_c = sim_c.max(dim=-1).values.clamp(-1, 1) * 0.5 + 0.5
        return (score_f - score_c).abs().mean()

    def local_conservation(
        self, text_embeds: Tensor, tube_embeds: Dict[int, Tensor], text_mask=None
    ) -> Dict[int, float]:
        """Compute per-tube alignment risk proxy."""
        ids = list(tube_embeds)
        if not ids:
            return {}
        if text_mask is not None:
            text_embeds = text_embeds[text_mask.to(text_embeds.device).bool()]
        if text_embeds.shape[0] == 0:
            return {i: 0.0 for i in ids}
        dev = self.alignment.vis_proj.weight.device
        dim = next(iter(tube_embeds.values())).shape[-1]
        v = TextTubeAlignment.stack_tube_embeds(tube_embeds, ids, dim, device=dev)
        with torch.no_grad():
            scores = self.alignment.tube_scores(text_embeds.to(dev), v)
            violation = alignment_risk(scores)
        return {i: float(violation[j]) for j, i in enumerate(ids)}

    def _align_term(
        self, full: CMSCObservation, accel: CMSCObservation, ids: List[int], dev=None
    ) -> Tensor:
        """Compute alignment deviation term."""
        if len(ids) < 1:
            return torch.zeros((), device=dev)
        dim = next(iter(full.tube_embeds.values())).shape[-1]
        dev = self.alignment.vis_proj.weight.device
        vf = TextTubeAlignment.stack_tube_embeds(full.tube_embeds, ids, dim, device=dev)
        va = TextTubeAlignment.stack_tube_embeds(accel.tube_embeds, ids, dim, device=dev)
        a_full = self.alignment.matrix(full.text_embeds.to(dev), vf)
        a_accel = self.alignment.matrix(accel.text_embeds.to(dev), va)
        return (a_full - a_accel).abs().mean()

    @staticmethod
    def _id_term(
        full: Dict[int, Tensor], accel: Dict[int, Tensor], ids: List[int], dev=None
    ) -> Tensor:
        """Compute identity deviation term."""
        common = [i for i in ids if i in full and i in accel]
        if not common:
            return torch.zeros((), device=dev)
        fa = F.normalize(_stack_on(full, common, dev), dim=-1)
        ac = F.normalize(_stack_on(accel, common, dev), dim=-1)
        cos = (fa * ac).sum(-1).clamp(-1, 1)
        return (1.0 - cos).mean()

    @staticmethod
    def _motion_term(full: VideoFeatures, accel: VideoFeatures, dev=None) -> Tensor:
        """Compute motion deviation term."""
        mf, ma = full.flow_mag_per_pair.float(), accel.flow_mag_per_pair.float()
        n = min(mf.numel(), ma.numel())
        if n == 0:
            return torch.zeros((), device=dev)
        if dev is not None:
            mf, ma = mf.to(dev), ma.to(dev)
        denom = mf[:n].abs().mean().clamp_min(1e-6)
        return ((ma[:n] - mf[:n]).abs().mean() / denom).clamp(0.0, 4.0)

    @staticmethod
    def _spatial_term(
        full: Dict[int, Tuple[float, float]],
        accel: Dict[int, Tuple[float, float]],
        ids: List[int],
        dev=None,
    ) -> Tensor:
        """Compute spatial layout deviation term."""
        common = [i for i in ids if i in full and i in accel]
        if len(common) < 2:
            return torch.zeros((), device=dev)
        cf = torch.tensor([full[i] for i in common], dtype=torch.float32, device=dev)
        ca = torch.tensor([accel[i] for i in common], dtype=torch.float32, device=dev)
        df = torch.cdist(cf, cf)
        da = torch.cdist(ca, ca)
        k = len(common)
        return (df - da).pow(2).sum().sqrt() / (k * k)

    @staticmethod
    def _ocr_term(full: VideoFeatures, accel: VideoFeatures, dev=None) -> Tensor:
        """Compute OCR fidelity deviation term."""
        return torch.tensor(max(0.0, full.ocr_accuracy - accel.ocr_accuracy), device=dev)
