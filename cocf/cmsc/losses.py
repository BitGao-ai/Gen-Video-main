"""Multi-dimensional cross-modal semantic-conservation loss (§6.3.2).

Pixel- or global-CLIP-only objectives let "pixels pass but semantics collapse"
(§6.1): subject–action binding errors, identity drift, wrong spatial relations,
broken OCR. CMSC instead constrains the *relations* between the prompt and the
semantic tubes to be **conserved** between the full-compute reference video
``Y_full`` and the accelerated video ``Y_accel``::

    L_CMSC = λ_align·L_align + λ_id·L_id + λ_motion·L_motion
           + λ_spatial·L_spatial + λ_ocr·L_ocr + λ_bnd·L_bnd

Each term is a *deviation* (≥0) of one semantic relation under acceleration:

    L_align    ‖A_align(full) − A_align(accel)‖₁    text–tube alignment (§6.3.1)
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

from cocf.cmsc.alignment import TextTubeAlignment, alignment_risk
from cocf.common.config import CMSCConfig
from cocf.lcocf.damage import VideoFeatures

Tensor = torch.Tensor


def _stack_on(feats: Dict[int, Tensor], ids: List[int], dev=None) -> Tensor:
    """Stack ``feats[i] for i in ids`` as float32 on ``dev`` (``None`` = leave alone).

    The two observations a CMSC term compares are produced by two separate extractor
    calls, and the extractor is free to park a detached, non-differentiable result on
    CPU (its ``offload`` contract). Co-locating here keeps every term device-agnostic
    without the caller having to know which branch offloaded. ``Tensor.to`` is an
    autograd op, so the accelerated branch's graph survives the move.
    """
    out = torch.stack([feats[i].float() for i in ids])
    return out if dev is None else out.to(dev)


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
        # Anchor every term's constants/empty-fallbacks to the observation device so
        # the summed loss stays on one device (the learnable align term is on the
        # module's GPU device; the other terms must not inject CPU 0-dim scalars).
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
    # batched offline conservation (Stage-B counterfactual samples, §4.1)
    # ------------------------------------------------------------------ #

    def alignment_conservation(
        self,
        text_embeds: Tensor,                 # [B, L, d_c] padded prompt tokens
        tube_full: Tensor,                   # [B, d_v] full-render tube visual embed
        tube_cf: Tensor,                     # [B, d_v] cf-render tube visual embed
        text_mask: Optional[Tensor] = None,  # [B, L] 1=keep
    ) -> Tensor:
        """Per-sample text-tube alignment conservation ``mean_b |a_full − a_cf|``.

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
        return (score_f - score_c).abs().mean()

    # ------------------------------------------------------------------ #
    # per-tube local term for the RAEC certificate (§5.3.1)
    # ------------------------------------------------------------------ #

    def local_conservation(
        self, text_embeds: Tensor, tube_embeds: Dict[int, Tensor], text_mask=None
    ) -> Dict[int, float]:
        """Inference-time risk proxy per tube ∈ [0, 1]: how far *below neutral* a tube's
        prompt alignment sits.

        High when a tube is poorly aligned to the prompt — i.e. skipping it risks a
        semantic-conservation violation — so it raises that tube's certificate. Needs no
        ``Y_full`` reference, only the prompt and the tube's current embed, which is the
        whole point: it is the one §6 signal available inside the accelerated loop, and
        the certificate's ``λ_cmsc`` term sat at a hard zero at inference for want of a
        caller (§P4-A4).

        **Why not simply ``1 − alignment``.** :meth:`TextTubeAlignment.tube_scores` maps
        cosine ``[-1, 1]`` onto ``[0, 1]``, so an *uninformative* projection — a randomly
        initialised head, or one Stage B barely moved — scores every tube at ≈0.5 and
        ``1 − score`` becomes a constant ≈0.5 on every tube. Multiplied by λ_cmsc=0.2
        that is a flat +0.1 added to every certificate: not a risk signal, a bias. It
        pushed a mean ``E_cert`` of 0.35 to 0.41, straight across ``τ_low = 0.40``, and
        turned 8 repairs into 132 — each of which pins its tube to FULL, so the
        accelerator *lost* efficiency (compute_ratio 0.25 → 0.40) for no quality reason.
        This is the same cold-start miscalibration :class:`~cocf.common.config.PredictorConfig`
        documents for the damage head.

        Measuring from the neutral point instead — ``relu(0.5 − score) · 2``, i.e.
        ``relu(−cos)`` — makes an uninformative head contribute exactly 0 and reserves
        the term for tubes whose embedding genuinely points *away* from every prompt
        token. The signal survives; the bias does not.

        Inputs are aligned to the projection's device here rather than at the call site:
        the tube embeds come from the perception provider (input-device) and the prompt
        tokens from the backbone, while the projections live wherever the accelerator was
        moved — three devices this method is the natural place to reconcile.
        """
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
            scores = self.alignment.tube_scores(text_embeds.to(dev), v)  # [K] ∈ [0,1]
            violation = alignment_risk(scores)
        return {i: float(violation[j]) for j, i in enumerate(ids)}

    # ------------------------------------------------------------------ #
    # individual terms
    # ------------------------------------------------------------------ #

    def _align_term(
        self, full: CMSCObservation, accel: CMSCObservation, ids: List[int], dev=None
    ) -> Tensor:
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
        common = [i for i in ids if i in full and i in accel]
        if not common:
            return torch.zeros((), device=dev)
        # Land both sides on ``dev`` before the product. ``dev`` used to guard only the
        # empty fallback above, which left the populated path assuming the two
        # observations were already co-located — an assumption the extractor's optional
        # CPU offload breaks (a detached reference on CPU against an on-GPU render).
        # Moving a detached reference onto the comparison device is lossless and
        # unambiguous, unlike guessing at a frame alignment, so it is done rather than
        # rejected.
        fa = F.normalize(_stack_on(full, common, dev), dim=-1)
        ac = F.normalize(_stack_on(accel, common, dev), dim=-1)
        cos = (fa * ac).sum(-1).clamp(-1, 1)
        return (1.0 - cos).mean()

    @staticmethod
    def _motion_term(full: VideoFeatures, accel: VideoFeatures, dev=None) -> Tensor:
        mf, ma = full.flow_mag_per_pair.float(), accel.flow_mag_per_pair.float()
        n = min(mf.numel(), ma.numel())
        if n == 0:
            return torch.zeros((), device=dev)
        if dev is not None:  # co-locate: see _id_term
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
        return torch.tensor(max(0.0, full.ocr_accuracy - accel.ocr_accuracy), device=dev)
