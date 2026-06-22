"""Multi-dimensional cross-modal semantic-conservation loss (§6.3.2).

Pixel- or global-CLIP-only objectives let "pixels pass but semantics collapse"
(§6.1): subject–action binding errors, identity drift, wrong spatial relations,
broken OCR. CMSC instead constrains the *relations* between the prompt and the
semantic tubes to be **conserved** between the full-compute reference video
``Y_full`` and the accelerated video ``Y_accel``::

    L_CMSC = λ_align·L_align + λ_id·L_id + λ_motion·L_motion
           + λ_spatial·L_spatial + λ_ocr·L_ocr + λ_bnd·L_bnd

Each term is a *deviation* (≥0) of one semantic relation under acceleration:

    L_align    ‖A_align(full) − A_align(accel)‖²    text–tube alignment (§6.3.1)
    L_id       mean_k (1 − cos(id_full_k, id_accel_k))   subject identity
    L_motion   ‖m_full − m_accel‖₁ / |m_full|        RAFT motion field
    L_spatial  ‖D_full − D_accel‖_F / K²             pairwise tube-centroid layout
    L_ocr      relu(ocr_full − ocr_accel)            OCR fidelity (text scenes)
    L_bnd      mean_k (1 − cos(bnd_full_k, bnd_accel_k))  boundary structure (opt.)

The features come from the same injected :class:`~cocf.lcocf.damage.MetricExtractor`
the teacher pipeline uses (DINO/CLIP/RAFT/OCR), bundled in
:class:`~cocf.lcocf.damage.VideoFeatures`, plus per-tube visual embeds and
centroids. The module is differentiable only through the learnable alignment
projection; the perceptual features are detached references — which is exactly the
"conserve the reference relations" semantics, and keeps the backward graph tiny
(a memory win for Stage-C, user requirement #1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.cmsc.alignment import TextTubeAlignment
from cocf.common.config import CMSCConfig
from cocf.lcocf.damage import VideoFeatures

Tensor = torch.Tensor


@dataclass
class CMSCObservation:
    """Everything the conservation loss needs from one rendered video.

    A ``full`` and an ``accel`` observation are compared term-by-term. All tensors
    may be detached references except those that flow through the (learnable)
    alignment projection.
    """

    video: VideoFeatures                      # global DINO/CLIP/RAFT/OCR features
    text_embeds: Tensor                       # [L, d_c] prompt token sequence
    tube_embeds: Dict[int, Tensor] = field(default_factory=dict)   # tid → [d_v]
    tube_identity: Dict[int, Tensor] = field(default_factory=dict) # tid → [d_id]
    tube_centroid: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    tube_boundary: Dict[int, Tensor] = field(default_factory=dict) # tid → [d] (opt)


class CMSCLoss(nn.Module):
    """Computes ``L_CMSC`` and exposes a per-tube local term for the certificate."""

    def __init__(self, cfg: CMSCConfig, alignment: TextTubeAlignment) -> None:
        super().__init__()
        self.cfg = cfg
        self.alignment = alignment

    # ------------------------------------------------------------------ #
    # full training loss
    # ------------------------------------------------------------------ #

    def forward(
        self, full: CMSCObservation, accel: CMSCObservation
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Return ``(scalar loss, per-term components)`` comparing full vs accel."""
        c = self.cfg
        ids = sorted(set(full.tube_embeds) & set(accel.tube_embeds))

        l_align = self._align_term(full, accel, ids)
        l_id = self._id_term(full.tube_identity, accel.tube_identity, ids)
        l_motion = self._motion_term(full.video, accel.video)
        l_spatial = self._spatial_term(full.tube_centroid, accel.tube_centroid, ids)
        l_ocr = self._ocr_term(full.video, accel.video)
        l_bnd = self._id_term(full.tube_boundary, accel.tube_boundary, ids)

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
    # batched offline conservation (Stage-B counterfactual samples, §4.1)
    # ------------------------------------------------------------------ #

    def alignment_conservation(
        self,
        text_embeds: Tensor,                 # [B, L, d_c] padded prompt tokens
        tube_full: Tensor,                   # [B, d_v] full-render tube visual embed
        tube_cf: Tensor,                     # [B, d_v] cf-render tube visual embed
        text_mask: Optional[Tensor] = None,  # [B, L] 1=keep
    ) -> Tensor:
        """Per-sample text-tube alignment conservation ``mean_b (a_full − a_cf)²``.

        The Stage-B counterfactual store holds one tube's visual embed under the full
        and the counterfactual render plus the prompt tokens — not two full
        :class:`CMSCObservation`s — so this is the batched, differentiable CMSC term
        for that offline schema (§4.1 CMSC 模块). It mirrors
        :meth:`TextTubeAlignment.tube_scores` (max alignment over text tokens, →[0,1])
        and penalises the change in that alignment under acceleration; gradient flows
        through the learnable text/visual projections only.
        """
        al = self.alignment
        t = F.normalize(al.text_proj(text_embeds.float()), dim=-1)   # [B, L, a]
        vf = F.normalize(al.vis_proj(tube_full.float()), dim=-1)     # [B, a]
        vc = F.normalize(al.vis_proj(tube_cf.float()), dim=-1)       # [B, a]
        sim_f = torch.einsum("bla,ba->bl", t, vf)                    # [B, L] cosine
        sim_c = torch.einsum("bla,ba->bl", t, vc)
        if text_mask is not None:
            fill = torch.finfo(sim_f.dtype).min
            keep = text_mask > 0.5
            sim_f = sim_f.masked_fill(~keep, fill)
            sim_c = sim_c.masked_fill(~keep, fill)
        score_f = sim_f.max(dim=-1).values.clamp(-1, 1) * 0.5 + 0.5  # [B] ∈ [0,1]
        score_c = sim_c.max(dim=-1).values.clamp(-1, 1) * 0.5 + 0.5
        return (score_f - score_c).pow(2).mean()

    # ------------------------------------------------------------------ #
    # per-tube local term for the RAEC certificate (§5.3.1)
    # ------------------------------------------------------------------ #

    def local_conservation(
        self, text_embeds: Tensor, tube_embeds: Dict[int, Tensor]
    ) -> Dict[int, float]:
        """Inference-time proxy: ``1 − alignment(tube)`` per tube ∈ [0, 1].

        High when a tube is poorly aligned to the prompt — i.e. skipping it risks
        a semantic-conservation violation — so it raises that tube's certificate.
        Needs no ``Y_full`` reference, only the prompt and the tube's current embed.
        """
        ids = list(tube_embeds)
        if not ids:
            return {}
        dim = next(iter(tube_embeds.values())).shape[-1]
        v = TextTubeAlignment.stack_tube_embeds(tube_embeds, ids, dim)
        with torch.no_grad():
            scores = self.alignment.tube_scores(text_embeds, v)  # [K] ∈ [0,1]
        return {i: float(1.0 - scores[j]) for j, i in enumerate(ids)}

    # ------------------------------------------------------------------ #
    # individual terms
    # ------------------------------------------------------------------ #

    def _align_term(
        self, full: CMSCObservation, accel: CMSCObservation, ids: List[int]
    ) -> Tensor:
        if len(ids) < 1:
            return torch.zeros(())
        dim = next(iter(full.tube_embeds.values())).shape[-1]
        vf = TextTubeAlignment.stack_tube_embeds(full.tube_embeds, ids, dim)
        va = TextTubeAlignment.stack_tube_embeds(accel.tube_embeds, ids, dim)
        a_full = self.alignment.matrix(full.text_embeds, vf)   # [L, K] (grad)
        a_accel = self.alignment.matrix(accel.text_embeds, va)  # [L, K] (grad)
        return (a_full - a_accel).pow(2).mean()

    @staticmethod
    def _id_term(
        full: Dict[int, Tensor], accel: Dict[int, Tensor], ids: List[int]
    ) -> Tensor:
        common = [i for i in ids if i in full and i in accel]
        if not common:
            return torch.zeros(())
        fa = F.normalize(torch.stack([full[i].float() for i in common]), dim=-1)
        ac = F.normalize(torch.stack([accel[i].float() for i in common]), dim=-1)
        cos = (fa * ac).sum(-1).clamp(-1, 1)
        return (1.0 - cos).mean()

    @staticmethod
    def _motion_term(full: VideoFeatures, accel: VideoFeatures) -> Tensor:
        mf, ma = full.flow_mag_per_pair.float(), accel.flow_mag_per_pair.float()
        n = min(mf.numel(), ma.numel())
        if n == 0:
            return torch.zeros(())
        denom = mf[:n].abs().mean().clamp_min(1e-6)
        return ((ma[:n] - mf[:n]).abs().mean() / denom).clamp(0.0, 4.0)

    @staticmethod
    def _spatial_term(
        full: Dict[int, Tuple[float, float]],
        accel: Dict[int, Tuple[float, float]],
        ids: List[int],
    ) -> Tensor:
        common = [i for i in ids if i in full and i in accel]
        if len(common) < 2:
            return torch.zeros(())
        cf = torch.tensor([full[i] for i in common], dtype=torch.float32)
        ca = torch.tensor([accel[i] for i in common], dtype=torch.float32)
        df = torch.cdist(cf, cf)
        da = torch.cdist(ca, ca)
        k = len(common)
        return (df - da).pow(2).sum().sqrt() / (k * k)

    @staticmethod
    def _ocr_term(full: VideoFeatures, accel: VideoFeatures) -> Tensor:
        return torch.tensor(max(0.0, full.ocr_accuracy - accel.ocr_accuracy))
