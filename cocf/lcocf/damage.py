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

    All tensors are CPU/float and small, so a teacher record stays light on disk.
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


class MetricExtractor(abc.ABC):
    """Extracts :class:`VideoFeatures` from a decoded video — injected dependency.

    Real impl wraps DINOv2 + CLIP + RAFT + an OCR model; the mock fabricates
    deterministic features so the damage/label pipeline is testable on CPU.
    """

    @abc.abstractmethod
    def extract(
        self, video: Tensor, prompt: str, *, differentiable: bool = False
    ) -> VideoFeatures:
        """``video`` is ``[F, 3, H, W]`` in [0,1]; returns its quality features.

        ``differentiable=False`` (default) extracts under ``no_grad`` and may offload
        the features to CPU — the cheap path for label generation / metric reporting,
        where the features are detached references. ``differentiable=True`` keeps the
        autograd graph **and** the input device, so the accelerated branch of the §6.3.2
        Stage-C semantic loss can back-propagate into the render (repair net / LoRA).
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

    @staticmethod
    def _jerk_increase(ref_mag: Tensor, cf_mag: Tensor) -> float:
        def jerk(mag: Tensor) -> float:
            if mag.numel() < 2:
                return 0.0
            return float((mag[1:] - mag[:-1]).abs().mean())
        return max(0.0, jerk(cf_mag) - jerk(ref_mag))

    def _motion_deviation(self, ref_mag: Tensor, cf_mag: Tensor) -> float:
        if ref_mag.numel() == 0:
            return 0.0
        denom = ref_mag.abs().mean().clamp_min(self.eps)
        return float(((cf_mag - ref_mag).abs().mean() / denom).clamp(0.0, 1.0))
