"""Multi-dimensional counterfactual damage — the L-COCF training *label* (§7.1.1, §9.4).

The L-COCF predictor regresses the **final-video** damage caused by skipping
compute on a tube — explicitly *not* a single-step latent L2 (that local/global
mismatch is the core failure mode the doc attacks, §1.1). This module defines what
that damage *is* and how to compute it as a vector over the VBench-style quality
axes used for evaluation (§9.4), from cheap proxy metrics (DINO / CLIP / RAFT /
OCR) rather than human preference (§7.1.1 cost trick).

Damage on each axis is the *degradation of that axis* in the counterfactual video
``Y_cf`` relative to the full-compute reference ``Y_full`` — clamped to ≥ 0 because
"accidentally better than full" is not a risk we need to price:

    d_axis = relu( quality_axis(Y_full) − quality_axis(Y_cf) )   (axis-normalised)

Feature extraction (DINO/CLIP/RAFT/OCR forward passes) is delegated to an injected
:class:`MetricExtractor`, so this module is pure, deterministic processing and is
unit-testable with the bundled mock. The extractor's outputs are bundled in
:class:`VideoFeatures` so the same stats serve every axis without recomputation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# The damage axes mirror the §9.4 quality metrics the paper reports. Order is
# pinned so the label vector is positionally stable everywhere it is sliced.
DAMAGE_DIMENSIONS: Tuple[str, ...] = (
    "subject_consistency",   # DINO subject feature stability across frames
    "background_consistency",# CLIP background feature stability
    "temporal_flicker",      # increase in frame-to-frame feature jitter
    "motion_smoothness",     # increase in 2nd-order motion (jerk)
    "clip_score",            # text-video alignment drop
    "dino_identity",         # DINO identity (subject) drop
    "raft_motion",           # RAFT motion-field deviation
    "ocr_accuracy",          # OCR text-fidelity drop (text scenes)
)
NUM_DAMAGE_DIMS = len(DAMAGE_DIMENSIONS)

# Default perceptual weights for reducing the vector to a scalar μ-target. They
# emphasise the failure modes the doc calls out (identity drift, flicker, OCR).
DEFAULT_DAMAGE_WEIGHTS: Dict[str, float] = {
    "subject_consistency": 0.18,
    "background_consistency": 0.08,
    "temporal_flicker": 0.16,
    "motion_smoothness": 0.12,
    "clip_score": 0.14,
    "dino_identity": 0.16,
    "raft_motion": 0.08,
    "ocr_accuracy": 0.08,
}


@dataclass
class VideoFeatures:
    """Per-video statistics needed to score every damage axis (no raw frames kept).

    All tensors are float and small, so a teacher record stays light on disk. They are
    *not* guaranteed to be on CPU: the extractor keeps them on the render's device
    whenever the caller passes ``offload=False`` (Stage C compares two observations on
    the GPU). Any binary reduction over two of these must co-locate them first — see
    :meth:`to`.
    """

    dino_per_frame: Tensor          # [F, d_dino] subject identity features
    clip_per_frame: Tensor          # [F, d_clip] global/background CLIP features
    clip_text_score: float          # CLIPScore vs prompt ∈ [0,1]
    flow_mag_per_pair: Tensor       # [F-1] mean RAFT magnitude per consecutive pair
    ocr_accuracy: float = 1.0       # OCR fidelity ∈ [0,1] (1.0 if no text)
    # optional per-tube DINO features for *localised* damage (tube_id → [F,d])
    tube_dino: Dict[int, Tensor] = field(default_factory=dict)

    def num_frames(self) -> int:
        return int(self.dino_per_frame.shape[0])

    def detached(self) -> "VideoFeatures":
        """A graph-free copy of these features.

        :class:`MultiDimDamageComputer` reduces every axis to a Python ``float``, which
        on a tensor still attached to the graph both warns and needlessly keeps that
        graph alive. Callers that extracted features with ``differentiable=True`` (the
        Stage-C accelerated branch) but want a *label* out of them — e.g. the realised
        damage the §4.2 certificate regulariser calibrates against — detach first.
        """
        return VideoFeatures(
            dino_per_frame=self.dino_per_frame.detach(),
            clip_per_frame=self.clip_per_frame.detach(),
            clip_text_score=self.clip_text_score,
            flow_mag_per_pair=self.flow_mag_per_pair.detach(),
            ocr_accuracy=self.ocr_accuracy,
            tube_dino={k: v.detach() for k, v in self.tube_dino.items()},
        )

    def to(self, device) -> "VideoFeatures":
        """These features on ``device`` (``None`` ⇒ unchanged; a no-op if already there).

        Every :class:`MultiDimDamageComputer` axis is a binary reduction over *two*
        observations, and it has no device logic of its own — so the two must already
        agree. They do not agree automatically: extraction may park a detached result on
        CPU (the ``offload`` contract on :meth:`MetricExtractor.extract`) while its
        counterpart stays on the render's device. Co-locating a detached reference is
        lossless, so the comparing caller does it rather than each axis guessing.
        """
        if device is None:
            return self
        return VideoFeatures(
            dino_per_frame=self.dino_per_frame.to(device),
            clip_per_frame=self.clip_per_frame.to(device),
            clip_text_score=self.clip_text_score,
            flow_mag_per_pair=self.flow_mag_per_pair.to(device),
            ocr_accuracy=self.ocr_accuracy,
            tube_dino={k: v.to(device) for k, v in self.tube_dino.items()},
        )


def crop_to_tube(video: Tensor, mask: Tensor) -> Tensor:
    """Restrict ``[F, 3, H, W]`` to one tube: its frames only, zeroed outside its region.

    This is what makes a damage label *about a tube* rather than about the whole clip.
    The tube-group counterfactual of §7.1.1 intervenes on one tube, so scoring the
    result on global features dilutes the signal into whatever the rest of the frame
    happened to do — the predictor then has to recover per-tube differences from a
    label that barely contains them.

    ``mask`` is ``[F, H, W]`` bool over the *same* frame axis as ``video``; frames the
    tube does not span are dropped (so the returned tensor is ``[n, 3, H, W]``, n ≤ F),
    which keeps the per-frame consistency metrics measuring the tube's own lifetime.
    Falls back to the unmasked video when the mask is empty.
    """
    if mask is None or mask.numel() == 0:
        return video
    present = mask.reshape(mask.shape[0], -1).any(dim=1).nonzero(as_tuple=True)[0]
    if present.numel() == 0:
        return video
    v = video.index_select(0, present.to(video.device))
    m = mask.index_select(0, present.to(mask.device)).to(v.device, v.dtype).unsqueeze(1)
    return v * m


class MetricExtractor(abc.ABC):
    """Extracts :class:`VideoFeatures` from a decoded video — injected dependency.

    Real impl wraps DINOv2 + CLIP + RAFT + an OCR model; the mock fabricates
    deterministic features so the damage/label pipeline is testable on CPU.
    """

    @abc.abstractmethod
    def extract(
        self, video: Tensor, prompt: str, *,
        differentiable: bool = False,
        tube_masks: Optional[Dict[int, Tensor]] = None,
        offload: bool = True,
    ) -> VideoFeatures:
        """``video`` is ``[F, 3, H, W]`` in [0,1]; returns its quality features.

        ``differentiable=False`` (default) extracts under ``no_grad``.
        ``differentiable=True`` keeps the autograd graph **and** the input device, so the
        accelerated branch of the §6.3.2 Stage-C semantic loss can back-propagate into
        the render (repair net / LoRA).

        ``offload=True`` (default) lets the implementation park the returned features on
        CPU — the cheap path for label generation, where they are detached references
        headed for disk. It is ignored when ``differentiable=True`` (moving the features
        off the render's device would sever the graph's device consistency).

        Offloading is a **separate** decision from differentiability, and conflating the
        two is a real bug rather than a hypothetical: Stage C's §6.3.2 loss builds its
        *reference* observation with ``differentiable=False`` — it genuinely wants the
        no-grad path — but then compares it element-wise, on the GPU, against the
        accelerated observation. With the offload welded to ``differentiable`` the
        reference came back on CPU and every such comparison
        (:meth:`CMSCLoss._id_term`, :meth:`CMSCLoss._motion_term`,
        :meth:`MultiDimDamageComputer.compute`) raised a cuda-vs-cpu device error. A
        caller that will *consume* the features on the render's device therefore passes
        ``offload=False``.

        ``tube_masks`` maps ``tube_id → [F, H, W]`` bool; each one adds a *localised*
        identity feature to :attr:`VideoFeatures.tube_dino`, which is what lets the
        damage computer score a tube-group counterfactual on the tube it intervened on
        (§7.1.1) instead of on the whole frame. Costs one extra identity pass per tube,
        so callers pass only the tubes they will actually score.
        """


class MultiDimDamageComputer:
    """Computes the damage vector ``d ∈ R^{NUM_DAMAGE_DIMS}`` between full & CF videos."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def compute(
        self,
        full: VideoFeatures,
        cf: VideoFeatures,
        tube_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """Per-axis degradation (≥0). If ``tube_id`` is given and per-tube DINO is
        available, ``subject_consistency``/``dino_identity`` are localised to that
        tube (the tube-group counterfactual of §7.1.1)."""
        d: Dict[str, float] = {}
        # --- appearance / identity (global or tube-localised) ---
        ref_dino = full.tube_dino.get(tube_id) if tube_id is not None else None
        cf_dino = cf.tube_dino.get(tube_id) if tube_id is not None else None
        ref_dino = ref_dino if ref_dino is not None else full.dino_per_frame
        cf_dino = cf_dino if cf_dino is not None else cf.dino_per_frame
        d["subject_consistency"] = self._consistency_drop(ref_dino, cf_dino)
        d["dino_identity"] = self._identity_drop(ref_dino, cf_dino)
        d["background_consistency"] = self._consistency_drop(full.clip_per_frame, cf.clip_per_frame)
        # --- temporal stability ---
        d["temporal_flicker"] = self._flicker_increase(full.clip_per_frame, cf.clip_per_frame)
        d["motion_smoothness"] = self._jerk_increase(full.flow_mag_per_pair, cf.flow_mag_per_pair)
        d["raft_motion"] = self._motion_deviation(full.flow_mag_per_pair, cf.flow_mag_per_pair)
        # --- semantics ---
        d["clip_score"] = max(0.0, full.clip_text_score - cf.clip_text_score)
        d["ocr_accuracy"] = max(0.0, full.ocr_accuracy - cf.ocr_accuracy)
        return d

    def as_vector(self, damage: Dict[str, float], device=None) -> Tensor:
        return torch.tensor([damage[k] for k in DAMAGE_DIMENSIONS], device=device, dtype=torch.float32)

    # -- per-axis formulas ---------------------------------------------- #

    def _consistency_drop(self, ref: Tensor, cf: Tensor) -> float:
        """Drop in across-frame feature consistency (VBench *_consistency proxy)."""
        return max(0.0, self._across_frame_consistency(ref) - self._across_frame_consistency(cf))

    @staticmethod
    def _across_frame_consistency(feats: Tensor) -> float:
        if feats.shape[0] < 2:
            return 1.0
        f = F.normalize(feats.float(), dim=-1)
        # mean cosine of each frame to the first frame (VBench subject_consistency)
        return float((f[1:] @ f[0]).mean().clamp(-1, 1) * 0.5 + 0.5)

    def _identity_drop(self, ref: Tensor, cf: Tensor) -> float:
        """Drop in mean pairwise identity similarity (DINO identity consistency)."""
        return max(0.0, self._mean_pairwise(ref) - self._mean_pairwise(cf))

    @staticmethod
    def _mean_pairwise(feats: Tensor) -> float:
        if feats.shape[0] < 2:
            return 1.0
        f = F.normalize(feats.float(), dim=-1)
        sim = f @ f.T
        n = f.shape[0]
        off = (sim.sum() - n) / (n * (n - 1))  # exclude the diagonal
        return float(off.clamp(-1, 1) * 0.5 + 0.5)

    @staticmethod
    def _flicker_increase(ref: Tensor, cf: Tensor) -> float:
        def flicker(feats: Tensor) -> float:
            if feats.shape[0] < 2:
                return 0.0
            f = F.normalize(feats.float(), dim=-1)
            return float((1.0 - (f[1:] * f[:-1]).sum(-1)).mean())
        return max(0.0, flicker(cf) - flicker(ref))

    def _jerk_increase(self, ref_mag: Tensor, cf_mag: Tensor) -> float:
        """Increase in flow-magnitude jerk, **relative to the reference's own scale**.

        The absolute difference is not comparable across metric backends: the mock's
        ``flow_mag`` is a descriptor-difference mean while the real extractor's is a
        mean RAFT magnitude, so the same motion yields values orders of magnitude apart
        and a damage label generated with one backend cannot be read with the other.
        Every other axis here is already a *ratio* or a bounded drop; this one was the
        exception. Normalising by the reference jerk makes it scale-free and puts it
        on the same [0, 1] footing as the rest.
        """
        ref_mag, cf_mag = _align(ref_mag, cf_mag)

        def jerk(mag: Tensor) -> float:
            if mag.numel() < 2:
                return 0.0
            return float((mag[1:] - mag[:-1]).abs().mean())

        ref, cf = jerk(ref_mag), jerk(cf_mag)
        if cf <= ref:
            return 0.0
        denom = max(ref, float(ref_mag.abs().mean()) if ref_mag.numel() else 0.0)
        return float(min((cf - ref) / max(denom, self.eps), 1.0))

    def _motion_deviation(self, ref_mag: Tensor, cf_mag: Tensor) -> float:
        ref_mag, cf_mag = _align(ref_mag, cf_mag)
        if ref_mag.numel() == 0:
            return 0.0
        denom = ref_mag.abs().mean().clamp_min(self.eps)
        return float(((cf_mag - ref_mag).abs().mean() / denom).clamp(0.0, 1.0))


def _align(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Trim two per-pair series to their common length, co-located on ``a``'s device.

    A tube-localised feature spans only the frames its tube covers, so the two sides of
    a comparison are not guaranteed to be the same length; an unaligned subtraction
    either raises or — worse, when one side happens to be length 1 — broadcasts and
    scores nonsense. :meth:`CMSCLoss._motion_term` already guards the same pair.
    """
    n = min(a.numel(), b.numel())
    return a[:n], b[:n].to(a.device)
