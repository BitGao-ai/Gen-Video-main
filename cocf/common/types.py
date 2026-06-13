"""Core data types shared across all COCF-SS-DCA subsystems.

This module is intentionally dependency-light (only ``torch`` + stdlib) so that
every other subsystem can import it without creating import cycles. It defines the
*vocabulary* of the framework: the four compute actions, the latent token grid,
causal triplets/sub-graphs, semantic tubes and their state vectors, damage
predictions, error certificates and allocation decisions.

Tensor-shape conventions (kept uniform everywhere to keep coupling low):

    * Latent in **token** form:   ``z`` with shape ``[B, N, d]`` where
      ``N = T_l * H_l * W_l`` enumerated in (time, height, width) row-major order.
    * Latent in **grid** form:    ``[B, d, T_l, H_l, W_l]`` (backbone-native).
    * A *tube* ``g_k`` is a set of flat token indices into the ``N`` axis, grouped
      by frame so that per-frame masks / actions remain addressable.

The mapping between the two forms is owned exclusively by the backbone adapter
(see :mod:`cocf.backbones.base`), never duplicated in the algorithm code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Actions and compute tiers (§2.1, §3.3.3)
# --------------------------------------------------------------------------- #


class Action(enum.IntEnum):
    """The four per-tube compute actions ``a_{k,t} ∈ {FULL, LOWFREQ, INTERP, ANCHOR}``.

    Ordered from most to least expensive. ``IntEnum`` so the value doubles as an
    index into cost/logit tensors.
    """

    FULL = 0  # run the full transition Φ_t on the tube's tokens
    LOWFREQ = 1  # compute only low-frequency components, reuse high-freq from cache
    INTERP = 2  # temporally interpolate the tube from neighbouring anchored frames
    ANCHOR = 3  # freeze: reuse the cached latent of the last safe anchor verbatim

    @property
    def is_skip(self) -> bool:
        """Whether the action skips (does not freshly compute) the transition."""
        return self in (Action.INTERP, Action.ANCHOR)

    @classmethod
    def cheapest(cls) -> "Action":
        return cls.ANCHOR


class StrengthLevel(enum.IntEnum):
    """Stratified causal-effect levels ``s_H ≫ s_M ≫ s_L`` (axiom §3.2.2)."""

    HIGH = 0
    MID = 1
    LOW = 2


# Default prior mapping strength-level -> action (cold-start fallback, §1.3).
# The learned allocator may override this, but it anchors behaviour before the
# L-COCF predictor has converged.
DEFAULT_LEVEL_TO_ACTION: Dict[StrengthLevel, Action] = {
    StrengthLevel.HIGH: Action.FULL,
    StrengthLevel.MID: Action.LOWFREQ,
    StrengthLevel.LOW: Action.INTERP,
}


# --------------------------------------------------------------------------- #
# Latent token grid (§2.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenGrid:
    """The (time, height, width) layout of latent tokens for one sample.

    ``N = t * h * w``. A flat token index ``n`` decomposes as
    ``n = ti * (h*w) + hi * w + wi``.
    """

    t: int
    h: int
    w: int

    @property
    def num_tokens(self) -> int:
        return self.t * self.h * self.w

    @property
    def tokens_per_frame(self) -> int:
        return self.h * self.w

    def frame_of(self, flat_index: int) -> int:
        """Frame (temporal) index that a flat token index belongs to."""
        return flat_index // self.tokens_per_frame

    def unravel(self, flat_index: int) -> Tuple[int, int, int]:
        ti, rem = divmod(flat_index, self.tokens_per_frame)
        hi, wi = divmod(rem, self.w)
        return ti, hi, wi

    def ravel(self, ti: int, hi: int, wi: int) -> int:
        return ti * self.tokens_per_frame + hi * self.w + wi

    def frame_slice(self, ti: int) -> slice:
        start = ti * self.tokens_per_frame
        return slice(start, start + self.tokens_per_frame)


# --------------------------------------------------------------------------- #
# Causal structures (§3.3.1)
# --------------------------------------------------------------------------- #


@dataclass
class CausalTriplet:
    """A causal triplet ``(E_i, A_ij, E_j)`` extracted from the prompt by a VLM."""

    subject: str
    action: str
    obj: str
    # VLM-assigned importance of the subject entity in [0, 1] (feeds s_E, §3.3.2).
    subject_importance: float = 1.0
    object_importance: float = 1.0
    # Free-form metadata (e.g. whether this triplet involves text/face/hands).
    tags: Tuple[str, ...] = ()

    def entities(self) -> Tuple[str, str]:
        return self.subject, self.obj


@dataclass
class CausalSubgraph:
    """Local causal sub-graph ``G_s`` built from triplets + spatio-temporal locality.

    We keep this deliberately lightweight (axiom §3.2.1): only the entities that
    appear in triplets, their per-entity importance, and the adjacency implied by
    the triplets. There is no global graph and no structure learning.
    """

    triplets: List[CausalTriplet] = field(default_factory=list)
    # entity name -> aggregated importance score in [0, 1]
    entity_importance: Dict[str, float] = field(default_factory=dict)
    # set of entity names tagged as text/face/hands (quality-critical, §9.3)
    critical_entities: Tuple[str, ...] = ()

    def importance_of(self, entity: str) -> float:
        return self.entity_importance.get(entity, 0.0)


# --------------------------------------------------------------------------- #
# Regions and semantic tubes (§4.3.1)
# --------------------------------------------------------------------------- #


@dataclass
class Region:
    """A frame-level region produced by SAM and filtered by CLIP (§4.3.1).

    Masks are stored at *latent* resolution (``H_l × W_l``) so they index tokens
    directly; the builder is responsible for down-sampling pixel masks.
    """

    frame: int
    region_id: int
    # boolean mask over the latent grid of this frame, shape [H_l, W_l]
    mask: Tensor
    # flat token indices covered by this region (into the full [N] axis)
    token_indices: Tensor
    # DINOv2 identity feature of the region crop, shape [d_id]
    identity_feat: Optional[Tensor] = None
    # CLIP text-alignment feature, shape [d_clip]
    text_feat: Optional[Tensor] = None
    # (cy, cx) centroid in latent coords
    center: Tuple[float, float] = (0.0, 0.0)
    # CLIP semantic-match score used for low-semantic filtering
    clip_score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.token_indices.numel())


# Order of the tube-state vector (§4.3.1). Keeping the names here makes the
# 7-dim vector self-documenting wherever it is sliced or logged.
TUBE_STATE_FIELDS: Tuple[str, ...] = (
    "identity_confidence",  # I_k   = mean cos identity similarity to previous frame
    "occlusion",  # O_k   = 1 - IoU(M_t, Warp(M_{t-1}))
    "interaction",  # I_inter = Σ IoU with other tubes
    "boundary_uncertainty",  # geometric uncertainty along the tube boundary
    "motion_phase",  # normalised motion magnitude / phase
    "causal_value",  # causal_value_k from L-COCF strength field
    "anchor_age",  # steps since this tube was last fully anchored
)
TUBE_STATE_DIM = len(TUBE_STATE_FIELDS)


@dataclass
class TubeState:
    """The 7-dimensional per-step tube state ``s_{k,t}`` (§4.3.1)."""

    identity_confidence: float = 1.0
    occlusion: float = 0.0
    interaction: float = 0.0
    boundary_uncertainty: float = 0.0
    motion_phase: float = 0.0
    causal_value: float = 0.0
    anchor_age: float = 0.0

    def as_tensor(self, device=None, dtype=torch.float32) -> Tensor:
        return torch.tensor(
            [
                self.identity_confidence,
                self.occlusion,
                self.interaction,
                self.boundary_uncertainty,
                self.motion_phase,
                self.causal_value,
                self.anchor_age,
            ],
            device=device,
            dtype=dtype,
        )

    @property
    def is_unstable(self) -> bool:
        """Identity confidence below 0.5 ⇒ unstable tube ⇒ force FULL (§4.3.1)."""
        return self.identity_confidence < 0.5


@dataclass
class SemanticTube:
    """A cross-frame semantic tube ``g_k`` — the unit of compute allocation (§4)."""

    tube_id: int
    # frame index -> flat token indices belonging to this tube on that frame
    tokens_by_frame: Dict[int, Tensor] = field(default_factory=dict)
    # frame index -> boolean latent mask [H_l, W_l]
    masks_by_frame: Dict[int, Tensor] = field(default_factory=dict)
    # running identity feature (EMA of per-frame DINOv2 features)
    identity_feat: Optional[Tensor] = None
    state: TubeState = field(default_factory=TubeState)
    # bookkeeping for RAEC: step index of the last verified-safe anchor
    last_safe_anchor_step: Optional[int] = None

    @property
    def frames(self) -> List[int]:
        return sorted(self.tokens_by_frame)

    @property
    def length(self) -> int:
        """Number of frames the tube spans."""
        return len(self.tokens_by_frame)

    @property
    def size(self) -> int:
        """Total token count ``|g_k|`` (used by the cost model, §2.2)."""
        return int(sum(t.numel() for t in self.tokens_by_frame.values()))

    def all_token_indices(self) -> Tensor:
        if not self.tokens_by_frame:
            return torch.empty(0, dtype=torch.long)
        return torch.cat([self.tokens_by_frame[f] for f in self.frames])


# --------------------------------------------------------------------------- #
# Predictions, certificates, decisions (§3.3.4, §5.3, §2.2)
# --------------------------------------------------------------------------- #


@dataclass
class DamagePrediction:
    """L-COCF prediction of the *final-video* counterfactual damage of an action.

    ``mu`` and ``sigma`` are the predicted mean and (epistemic) uncertainty of the
    marginal damage ``y_{k,t,a}`` for applying action ``a`` to tube ``g_k`` at step
    ``t`` (§3.3.4, used by the certificate §5.3.1 and the budget term §7.3).
    """

    mu: Tensor  # shape [num_actions]
    sigma: Tensor  # shape [num_actions]

    def of(self, action: Action) -> Tuple[Tensor, Tensor]:
        return self.mu[int(action)], self.sigma[int(action)]


@dataclass
class ErrorCertificate:
    """Risk certificate ``E_cert(k, t)`` for an anchoring decision (§5.3.1)."""

    value: float
    tube_id: int
    step: int
    action: Action
    # individual additive contributions, for logging / ablation
    components: Dict[str, float] = field(default_factory=dict)


class TriggerLevel(enum.IntEnum):
    """Outcome of the RAEC risk trigger (§5.3.2)."""

    KEEP = 0  # E_cert <= τ_low : keep current action
    REPAIR = 1  # τ_low < E_cert <= τ_high : boundary repair + cache refresh
    ROLLBACK = 2  # E_cert > τ_high : roll back to last safe anchor, force FULL


@dataclass
class AllocationDecision:
    """The solved per-tube action assignment for one denoising step (§2.2)."""

    step: int
    actions: Dict[int, Action]  # tube_id -> chosen action
    predicted_cost: float
    budget: float
    # tube_id -> predicted damage of the chosen action (for the cost/quality log)
    chosen_damage: Dict[int, float] = field(default_factory=dict)

    def action_for(self, tube_id: int, default: Action = Action.FULL) -> Action:
        return self.actions.get(tube_id, default)
