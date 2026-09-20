"""Core shared data types for COCF-SS-DCA subsystems."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

Tensor = torch.Tensor

class Action(enum.IntEnum):
    """Per-tube compute actions ordered by cost."""

    FULL = 0
    LOWFREQ = 1
    INTERP = 2
    ANCHOR = 3

    @property
    def is_skip(self) -> bool:
        """Check if action skips compute."""
        return self in (Action.INTERP, Action.ANCHOR)

    @classmethod
    def cheapest(cls) -> "Action":
        """Return cheapest action."""
        return cls.ANCHOR


class StrengthLevel(enum.IntEnum):
    """Stratified causal-effect levels."""

    HIGH = 0
    MID = 1
    LOW = 2


DEFAULT_LEVEL_TO_ACTION: Dict[StrengthLevel, Action] = {
    StrengthLevel.HIGH: Action.FULL,
    StrengthLevel.MID: Action.LOWFREQ,
    StrengthLevel.LOW: Action.INTERP,
}


@dataclass(frozen=True)
class TokenGrid:
    """Latent token grid layout."""

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
        """Return frame index for token."""
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


@dataclass
class CausalTriplet:
    """Causal triplet extracted from prompt."""

    subject: str
    action: str
    obj: str
    subject_importance: float = 1.0
    object_importance: float = 1.0
    tags: Tuple[str, ...] = ()

    def entities(self) -> Tuple[str, str]:
        return self.subject, self.obj


@dataclass
class CausalSubgraph:
    """Local causal sub-graph built from triplets."""

    triplets: List[CausalTriplet] = field(default_factory=list)
    entity_importance: Dict[str, float] = field(default_factory=dict)
    critical_entities: Tuple[str, ...] = ()

    def importance_of(self, entity: str) -> float:
        return self.entity_importance.get(entity, 0.0)


@dataclass
class Region:
    """Frame-level region at latent resolution."""

    frame: int
    region_id: int
    mask: Tensor
    token_indices: Tensor
    identity_feat: Optional[Tensor] = None
    text_feat: Optional[Tensor] = None
    center: Tuple[float, float] = (0.0, 0.0)
    clip_score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.token_indices.numel())


TUBE_STATE_FIELDS: Tuple[str, ...] = (
    "identity_confidence",
    "occlusion",
    "interaction",
    "boundary_uncertainty",
    "motion_phase",
    "causal_value",
    "anchor_age",
)
TUBE_STATE_DIM = len(TUBE_STATE_FIELDS)


@dataclass
class TubeState:
    """Per-step tube state vector."""

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

    def is_unstable(self, threshold: float = 0.5) -> bool:
        """Check if tube is unstable."""
        return self.identity_confidence < threshold


@dataclass
class SemanticTube:
    """Cross-frame semantic tube for compute allocation."""

    tube_id: int
    tokens_by_frame: Dict[int, Tensor] = field(default_factory=dict)
    masks_by_frame: Dict[int, Tensor] = field(default_factory=dict)
    identity_feat: Optional[Tensor] = None
    identity_feat_by_frame: Dict[int, Tensor] = field(default_factory=dict)
    state: TubeState = field(default_factory=TubeState)
    last_safe_anchor_step: Optional[int] = None

    @property
    def frames(self) -> List[int]:
        return sorted(self.tokens_by_frame)

    @property
    def length(self) -> int:
        """Return frame span length."""
        return len(self.tokens_by_frame)

    @property
    def size(self) -> int:
        """Return total token count."""
        return int(sum(t.numel() for t in self.tokens_by_frame.values()))

    def all_token_indices(self) -> Tensor:
        if not self.tokens_by_frame:
            return torch.empty(0, dtype=torch.long)
        return torch.cat([self.tokens_by_frame[f] for f in self.frames])


@dataclass
class DamagePrediction:
    """Predicted counterfactual damage for actions."""

    mu: Tensor
    sigma: Tensor

    def of(self, action: Action) -> Tuple[Tensor, Tensor]:
        return self.mu[int(action)], self.sigma[int(action)]


@dataclass
class ErrorCertificate:
    """Risk certificate for anchoring decision."""

    value: float
    tube_id: int
    step: int
    action: Action
    components: Dict[str, float] = field(default_factory=dict)


class TriggerLevel(enum.IntEnum):
    """RAEC risk trigger outcome."""

    KEEP = 0
    REPAIR = 1
    ROLLBACK = 2


@dataclass
class AllocationDecision:
    """Per-tube action assignment for one step."""

    step: int
    actions: Dict[int, Action]
    predicted_cost: float
    budget: float
    chosen_damage: Dict[int, float] = field(default_factory=dict)

    def action_for(self, tube_id: int, default: Action = Action.FULL) -> Action:
        return self.actions.get(tube_id, default)
