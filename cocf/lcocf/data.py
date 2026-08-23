"""Counterfactual teacher data generation (§1.4–§1.6 / §7.1.1).

Generates multi-dimensional training labels for L-COCF by applying tube-level
**single-hop** counterfactual interventions. The core workflow (§1.5):

    1. Run full-compute model on the reference clip → Y_full, {z_t}, tubes  (Stage A)
    2. Build semantic tubes G_t once and their 7-dim states + (s_E,s_A,s_T) features
    3. For sampled (g_k, t, a) triplets: apply action a to *only* tube g_k at step t,
       continue all-FULL denoising to z_0, decode → Y_cf  (no multi-hop propagation)
    4. Compute the multi-dimensional damage D(Y_full, Y_cf), the compute-cost label
       and the multi-seed uncertainty as the training labels (§1.5)

This module implements:
    - :class:`COCFTrainingSample`: the §4.1 per-sample label record — every field
      Stage B reads, serialised to exactly the key set
      :func:`cocf.data.cocf_batch.collate_cocf_samples` consumes (round-trips through
      the §3 LMDB store via ``to_dict``/``from_dict``).
    - :class:`TeacherTrajectory`: the per-clip teacher-forward outputs the generator
      runs counterfactuals against (assembled by Stage A).
    - Stratified sampling: by timestep / tube causal level / action type (§1.5).
    - :class:`COCFDataGenerator`: the single-hop intervention rollout that reuses the
      backbone-agnostic :class:`~cocf.backbones.transition.TransitionExecutor`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cocf.backbones.base import BackboneAdapter, TextConditioning, sigma_from_step
from cocf.backbones.transition import TransitionExecutor
from cocf.common.memory import free_memory
from cocf.common.types import (
    Action,
    AllocationDecision,
    SemanticTube,
    StrengthLevel,
    TokenGrid,
    TubeState,
)
from cocf.lcocf.damage import (
    DAMAGE_DIMENSIONS,
    DEFAULT_DAMAGE_WEIGHTS,
    NUM_DAMAGE_DIMS,
    MetricExtractor,
    MultiDimDamageComputer,
    VideoFeatures,
)
from cocf.lcocf.strength import CausalStrengthFeatureBuilder, StrengthFeatures

Tensor = torch.Tensor
_log = logging.getLogger(__name__)

# Per-action relative compute cost C(a) ∝ |g_k| — mirrors AllocatorConfig.action_cost
# and the executor's FULL/LOWFREQ/INTERP/ANCHOR tiers. FULL=1; LOWFREQ=1/stride² (the
# executor is authoritative, see ``_cost_label``); INTERP runs no denoiser but does
# gather+blend the tube's tokens in latent space, so it is small-but-nonzero (§P4-1);
# ANCHOR touches nothing, so 0.
ACTION_COST: Tuple[float, float, float, float] = (1.0, 0.25, 0.02, 0.0)


def _to_np(x: Optional[Tensor]) -> Optional[np.ndarray]:
    """Detach a tensor to a small CPU float32 numpy array for on-disk storage."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype="float32")


def _to_tensor(x: object) -> Optional[Tensor]:
    """Inverse of :func:`_to_np` — rebuild a float tensor from stored numpy/list."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


# ============================================================================= #
# Core data structures (§7.1.1)
# ============================================================================= #


@dataclass
class COCFTrainingSample:
    """One counterfactual training label — the §4.1 per-sample read record.

    Carries every field Stage B reads for the joint loss (§4.1 "单样本读取字段"):
    causal-strength features (s_E/s_A/s_T), the 7-dim tube state, timestep & action
    encodings, tube token count, the multi-dimensional degradation label, the
    compute-cost label, the multi-seed uncertainty, the tube visual embeddings of
    the full & counterfactual renders, and the prompt text embedding.

    :meth:`to_dict` serialises to exactly the key set
    :func:`cocf.data.cocf_batch.collate_cocf_samples` consumes, so a sample
    round-trips losslessly through the §3 LMDB store; :meth:`from_dict` rebuilds it.
    Tensors are stored as small CPU numpy arrays (no autograd graph) to keep the
    store compact.
    """

    # -- core predictor inputs ----------------------------------------------- #
    tube_features: Tensor                 # [7]  the tube state vector s_{k,t} (§1.4)
    timestep: int                         # t (countdown index over the schedule)
    action: int                           # Action {FULL=0,LOWFREQ=1,INTERP=2,ANCHOR=3}
    damage_label: Tensor                  # [NUM_DAMAGE_DIMS] ∈ [0,1] degradation (§1.5)

    # -- the remaining §4.1 read fields (defaulted for partial/legacy samples) - #
    strength_features: Optional[Tensor] = None   # [3]  (s_E, s_A, s_T)  (§1.4(2))
    cost_label: Optional[Tensor] = None          # [2]  (FLOPs frac, active-token frac)
    uncertainty: Optional[Tensor] = None         # [NUM_DAMAGE_DIMS] multi-seed variance
    tube_visual_embed_full: Optional[Tensor] = None  # [d_v] full-render tube embed
    tube_visual_embed_cf: Optional[Tensor] = None    # [d_v] cf-render tube embed
    text_embed: Optional[Tensor] = None          # [L, d_c] prompt token sequence

    damage_per_axis: Dict[str, float] = field(default_factory=dict)

    # -- auxiliary metadata -------------------------------------------------- #
    prompt: str = ""
    video_id: str = ""
    tube_id: int = -1
    scene_type: str = "dynamic"           # static/dynamic/multi/text/face/occlusion
    interaction_density: float = 0.0      # 0-1: how much this tube interacts (§4.1)
    strength_level: int = 1               # StrengthLevel prior (HIGH=0/MID=1/LOW=2)
    step_frac: float = 0.0                # t / T ∈ [0,1] (denoising phase)
    # δ = ‖z_full − z_action‖ on the tube's tokens at the intervened step — the same
    # quantity the inference-time certificate feeds its λ_res term. Recorded here so
    # Stage B can calibrate that coefficient against a real signal instead of the
    # hard zero it used to pass, which left λ_res (and λ_cmsc) stuck at their inits
    # for the whole run (§P1-13).
    skip_residual: float = 0.0
    tube_token_count: int = 0             # |g_k| latent tokens (cost context, §2.2)
    tube_pixels: int = 0                  # region size in pixels (diagnostic)
    tube_stability: float = 1.0           # identity stability ∈ [0,1]

    def damage_scalar(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Reduce multi-dim damage to a scalar for quick sorting/filtering."""
        if weights is None:
            weights = DEFAULT_DAMAGE_WEIGHTS
        scalar = 0.0
        for i, axis in enumerate(DAMAGE_DIMENSIONS):
            scalar += float(self.damage_label[i]) * weights.get(axis, 0.0)
        return min(1.0, scalar)  # clamp to [0,1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the §3 LMDB payload (keys ≡ collate_cocf_samples fields)."""
        return {
            # vectors (numpy, float32)
            "tube_features": _to_np(self.tube_features),
            "strength_features": _to_np(self.strength_features),
            "damage_label": _to_np(self.damage_label),
            "cost_label": _to_np(self.cost_label),
            "uncertainty": _to_np(self.uncertainty),
            "tube_visual_embed_full": _to_np(self.tube_visual_embed_full),
            "tube_visual_embed_cf": _to_np(self.tube_visual_embed_cf),
            # ``text_embed`` is deliberately absent: it is per-*clip* data and is
            # stored once under ``text_embeds/<video_id>.pt``, then joined back by
            # the reader (§P2-3). Writing it here duplicated ~8 MiB per sample.
            # long scalars
            "action": int(self.action),
            "timestep": int(self.timestep),
            "tube_token_count": int(self.tube_token_count),
            "tube_id": int(self.tube_id),
            "strength_level": int(self.strength_level),
            # float scalars
            "step_frac": float(self.step_frac),
            "skip_residual": float(self.skip_residual),
            "interaction_density": float(self.interaction_density),
            "tube_stability": float(self.tube_stability),
            # strings
            "prompt": self.prompt,
            "scene_type": self.scene_type,
            "video_id": self.video_id,
            # diagnostics (ignored by collate; kept for analysis/cleaning §1.6)
            "damage_per_axis": dict(self.damage_per_axis),
            "tube_pixels": int(self.tube_pixels),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "COCFTrainingSample":
        """Rebuild a typed sample from a stored payload dict (inverse of to_dict)."""
        return cls(
            tube_features=_to_tensor(d.get("tube_features")),
            timestep=int(d.get("timestep", 0)),
            action=int(d.get("action", 0)),
            damage_label=_to_tensor(d.get("damage_label")),
            strength_features=_to_tensor(d.get("strength_features")),
            cost_label=_to_tensor(d.get("cost_label")),
            uncertainty=_to_tensor(d.get("uncertainty")),
            tube_visual_embed_full=_to_tensor(d.get("tube_visual_embed_full")),
            tube_visual_embed_cf=_to_tensor(d.get("tube_visual_embed_cf")),
            text_embed=_to_tensor(d.get("text_embed")),
            damage_per_axis=dict(d.get("damage_per_axis", {})),
            prompt=str(d.get("prompt", "")),
            video_id=str(d.get("video_id", "")),
            tube_id=int(d.get("tube_id", -1)),
            scene_type=str(d.get("scene_type", "dynamic")),
            interaction_density=float(d.get("interaction_density", 0.0)),
            strength_level=int(d.get("strength_level", 1)),
            step_frac=float(d.get("step_frac", 0.0)),
            skip_residual=float(d.get("skip_residual", 0.0) or 0.0),
            tube_token_count=int(d.get("tube_token_count", 0)),
            tube_pixels=int(d.get("tube_pixels", 0)),
            tube_stability=float(d.get("tube_stability", 1.0)),
        )


# ============================================================================= #
# Multi-dimensional damage computation & aggregation
# ============================================================================= #


class CounterfactualDamageComputer:
    """Orchestrates multi-dimensional damage computation (§7.1.1, steps e-f).

    Encapsulates:
        - Feature extraction via :class:`MetricExtractor`
        - Damage calculation via :class:`MultiDimDamageComputer`
        - Per-axis normalization & clamping
    """

    def __init__(
        self,
        metric_extractor: MetricExtractor,
        damage_weights: Optional[Dict[str, float]] = None,
        axis_eps: float = 1e-6,
    ) -> None:
        self.metric_extractor = metric_extractor
        self.damage_computer = MultiDimDamageComputer(eps=axis_eps)
        self.damage_weights = damage_weights or DEFAULT_DAMAGE_WEIGHTS

    def reference_features(
        self, video_full: Tensor, prompt: str,
        tube_masks: Optional[Dict[int, Tensor]] = None,
    ) -> VideoFeatures:
        """Extract the reference-side features once, for reuse across rollouts.

        ``video_full`` is fixed for a whole :class:`TeacherTrajectory` while dozens
        of counterfactual rollouts are scored against it, so re-extracting it per
        sample burns a full DINOv2+CLIP+RAFT pass (and its peak VRAM) every time
        for a bit-identical result. Callers hoist this out of their loop and hand
        it to :meth:`compute_damage`.

        ``tube_masks`` requests the per-tube identity features the localised damage
        axes need (§7.1.1). Pass every tube that will be scored against this
        reference: they are computed once here rather than once per rollout.
        """
        return self.metric_extractor.extract(video_full, prompt, tube_masks=tube_masks)

    def compute_damage(
        self,
        video_full: Tensor,  # [F, 3, H, W] in [0,1]
        video_cf: Tensor,  # [F, 3, H, W] counterfactual video
        prompt: str,
        tube_mask: Optional[Tensor] = None,  # [F, H, W] binary: 1 inside tube
        *,
        tube_id: Optional[int] = None,
        feats_full: Optional[VideoFeatures] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute multi-dim damage vector for a counterfactual pair.

        Args:
            video_full: Full-compute reference video.
            video_cf: Counterfactual video (with action applied).
            prompt: Text prompt (for CLIP scoring).
            tube_mask: Per-frame ``[F, H, W]`` mask of the intervened tube. Supplying
                it (with ``tube_id``) localises the identity axes to that tube — the
                §7.1.1 tube-group counterfactual. Without it the label is global, so
                two different tubes intervened at the same step get nearly the same
                damage and the predictor cannot learn a per-tube difference.
            tube_id: Identifies which entry of ``feats_full.tube_dino`` to compare
                against; must match the key used when the reference was extracted.
            feats_full: Pre-extracted reference features from
                :meth:`reference_features`. Extracted on demand when omitted.

        Returns:
            damage_vector: [NUM_DAMAGE_DIMS] ∈ [0,1]
            per_axis_dict: {axis_name: scalar_value} for diagnostics
        """
        # Localise only when both halves are available; a tube_id whose reference
        # features were never extracted would silently fall back to global anyway.
        masks = (
            {tube_id: tube_mask}
            if (tube_mask is not None and tube_id is not None) else None
        )
        if feats_full is None:
            feats_full = self.metric_extractor.extract(video_full, prompt, tube_masks=masks)
        localise = tube_id if (masks and tube_id in feats_full.tube_dino) else None
        feats_cf = self.metric_extractor.extract(video_cf, prompt, tube_masks=masks)

        # Compute multi-dimensional damage. ``compute`` returns the per-axis dict
        # {axis_name: value}; convert it to the ordered [NUM_DAMAGE_DIMS] tensor via
        # ``as_vector`` before doing any tensor ops on it.
        per_axis = self.damage_computer.compute(feats_full, feats_cf, tube_id=localise)
        damage_vec = self.damage_computer.as_vector(per_axis, device=video_full.device)

        return damage_vec.clamp(0, 1), per_axis


# ============================================================================= #
# Stratified sampling strategies (§7.1.1 cost optimization)
# ============================================================================= #


@dataclass
class StratifiedSamplingConfig:
    """Hyperparameters for stratified counterfactual sampling."""

    # By scene type: ensure representation from static/dynamic/text/face domains
    scene_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "static": 0.15,
            "dynamic": 0.25,
            "text": 0.25,
            "face": 0.20,
            "multi": 0.10,
            "occlusion": 0.05,
        }
    )

    # By timestep: concentrate on mutation points (early structure + late detail)
    # Group timesteps into early [t ∈ 0.8T..T], mid [0.3T..0.8T], late [0..0.3T]
    timestep_strata: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: {
            "early": (0.8, 1.0, 0.35),  # (tmin_frac, tmax_frac, weight)
            "mid": (0.3, 0.8, 0.15),
            "late": (0.0, 0.3, 0.50),
        }
    )

    # By action: ensure all four actions are represented, weighted by cost
    action_weights: Dict[int, float] = field(
        default_factory=lambda: {
            Action.FULL: 0.10,
            Action.LOWFREQ: 0.30,
            Action.INTERP: 0.35,
            Action.ANCHOR: 0.25,
        }
    )

    # General parameters
    samples_per_prompt: int = 3  # sample 3 seeds per prompt (§7.1.1)


class StratifiedSampler:
    """Generates balanced (tube, timestep, action) triplets from a video.

    Implements the stratified sampling of §7.1.1 to ensure representation across:
        - Scene types (static/dynamic/text/face)
        - Temporal phases (early/mid/late)
        - Action types (FULL/LOWFREQ/INTERP/ANCHOR)
    """

    def __init__(self, config: StratifiedSamplingConfig, device: torch.device):
        self.config = config
        self.device = device

    def sample_actions(self, num_samples: int = 4) -> List[int]:
        """The four compute actions per tube (§1.5: 分别施加 FULL/LOWFREQ/INTERP/ANCHOR).

        Each tube at each representative step is probed with **all four** actions.
        Generating a FULL sample (≈zero damage, the reference) alongside the three
        skip actions is what gives the §4.1 action-balanced (1:1:1:1) training
        material the Stage-B :class:`~cocf.data.cocf_batch.StratifiedBatchSampler`
        balances over. ``num_samples`` only truncates (kept for API/back-compat).
        """
        actions = [int(Action.FULL), int(Action.LOWFREQ), int(Action.INTERP), int(Action.ANCHOR)]
        return actions if num_samples >= len(actions) else actions[:num_samples]


# ============================================================================= #
# Label interpolation (cost optimization, §7.1.1)

@dataclass
class TeacherTrajectory:
    """Per-clip teacher full-compute outputs assembled by Stage A (§1.3–§1.4).

    One reference full-denoise trajectory: the decoded reference video ``Y_full``,
    the latents ``z_t`` cached at the representative steps, the semantic tubes built
    once on the reference, their 7-dim states and ``(s_E, s_A, s_T)`` features, and
    the per-tube visual embeddings of the full render. A single-hop intervention
    (§1.5) replays one cached ``z_t`` with one tube's action changed and continues
    all-FULL to ``z_0``.
    """

    video_id: str
    prompt: str
    scene_type: str
    video_full: Tensor                              # [F, 3, H, W] reference, in [0,1]
    grid: TokenGrid
    cond: TextConditioning
    z_by_step: Dict[int, Tensor]                    # step_idx -> z_t [B, N, d] (B==1)
    tubes: List[SemanticTube]
    tube_states: Dict[int, TubeState]               # tube_id -> state
    strength_feats: Dict[int, StrengthFeatures]     # tube_id -> (s_E, s_A, s_T)
    tube_visual_embed_full: Dict[int, Tensor]       # tube_id -> [d_v]
    num_total_steps: int
    text_embed: Optional[Tensor] = None             # [L, d_c] (cond.embeds[0])
    # z_T the baseline was generated from. Persisted alongside Y_full so Stage C can
    # start its accelerated run on the *same* noise and compare against the stored
    # reference instead of re-denoising the whole trajectory itself (§P2-4).
    z_init: Optional[Tensor] = None                 # [1, N, d]


def _frames_fchw(video: Tensor) -> Tensor:
    """``[B, 3, F, H, W]`` (or ``[3, F, H, W]``) → ``[F, 3, H, W]`` in [0,1]."""
    v = video[0] if video.dim() == 5 else video
    return v.permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)


def tube_pixel_mask(video_fchw: Tensor, tube: SemanticTube) -> Tensor:
    """``[F, Hp, Wp]`` bool: the tube's latent masks upsampled to pixel resolution.

    The damage extractor works on pixels while a tube carries latent-grid masks, so
    this is the bridge that lets a label be scored on the region the counterfactual
    actually intervened on (§7.1.1).
    """
    f, _, hp, wp = video_fchw.shape
    out = torch.zeros(f, hp, wp, dtype=torch.bool, device=video_fchw.device)
    for frame, mask in tube.masks_by_frame.items():
        if not (0 <= frame < f) or mask is None:
            continue
        up = F.interpolate(
            mask[None, None].float(), size=(hp, wp), mode="nearest"
        )[0, 0] > 0.5
        out[frame] = up.to(out.device)
    return out


def tube_clip_embed(
    video_fchw: Tensor,
    tube: SemanticTube,
    grid: TokenGrid,
    perception,
    d_v: Optional[int] = None,
) -> Tensor:
    """Per-tube CLIP visual embed ``[d_v]`` from a representative frame (feeds CMSC).

    Picks the tube's middle frame, upsamples its latent mask to pixel resolution and
    calls ``perception.clip_feature`` — the same source
    :class:`~cocf.tubes.regions.RegionExtractor` uses, so train/serve embeds match.
    Returns zeros when no perception/mask is available (e.g. a pure-mock dry run).
    """
    if d_v is None:
        d_v = int(getattr(perception, "d_clip", 64)) if perception is not None else 64
    frames = tube.frames
    if perception is None or not frames:
        return torch.zeros(d_v)
    mid = frames[len(frames) // 2]
    mask_lat = tube.masks_by_frame.get(mid)
    if mask_lat is None:
        return torch.zeros(d_v)
    fi = min(mid, video_fchw.shape[0] - 1)
    frame = video_fchw[fi]                                   # [3, Hp, Wp]
    mask_pix = F.interpolate(
        mask_lat[None, None].float(), size=frame.shape[-2:], mode="nearest"
    )[0, 0] > 0.5
    return perception.clip_feature(frame, mask_pix).detach().float()


# ============================================================================= #
# Full counterfactual data generation pipeline (§1.5)
# ============================================================================= #


# ============================================================================= #


class COCFDataGenerator:
    """Single-hop counterfactual training-data generation (§1.5).

    For each representative (tube ``g_k``, step ``t``, action ``a``) it:
        1. replays the cached ``z_t`` and takes one **dense** full-compute step;
        2. applies action ``a`` to *only* ``g_k`` (FULL/LOWFREQ/INTERP/ANCHOR),
           reusing the inference :class:`~cocf.backbones.transition.TransitionExecutor`
           realisation for LOWFREQ so the label has no train/serve skew;
        3. continues all-FULL to ``z_0`` (single-hop: no multi-hop propagation),
           decodes → ``Y_cf``;
        4. scores the multi-dim damage vs ``Y_full``, the compute-cost label and the
           multi-seed uncertainty.

    FULL is the zero-damage reference by construction (no local modification), which
    is the clean anchor the §4.1 action-balanced sampler needs.
    """

    def __init__(
        self,
        metric_extractor: MetricExtractor,
        strength_feature_builder: CausalStrengthFeatureBuilder,
        damage_computer: CounterfactualDamageComputer,
        sampling_config: Optional[StratifiedSamplingConfig] = None,
        device: torch.device = torch.device("cpu"),
        *,
        perception=None,
        action_cost: Tuple[float, ...] = ACTION_COST,
        seeds_per_prompt: int = 1,
        perturb_std: float = 0.02,
        free_memory_every: int = 8,
    ):
        self.metric_extractor = metric_extractor
        self.strength_builder = strength_feature_builder
        self.damage_computer = damage_computer
        self.sampling_config = sampling_config or StratifiedSamplingConfig()
        self.device = device
        # perception backend supplies the per-tube CLIP visual embed feeding CMSC.
        self.perception = perception
        self.action_cost = tuple(action_cost)
        self.seeds_per_prompt = max(1, int(seeds_per_prompt))
        self.perturb_std = float(perturb_std)
        # Return freed allocator blocks to the driver (0 disables entirely). Acts at
        # two granularities: after every non-zeroth rollout seed in
        # :meth:`_counterfactual_labels` — the unit that ends in a full VAE decode —
        # and every N completed samples in :meth:`generate`.
        self.free_memory_every = max(0, int(free_memory_every))

        self.sampler = StratifiedSampler(self.sampling_config, device)

    def generate(
        self,
        traj: TeacherTrajectory,
        backbone: BackboneAdapter,
        transition: TransitionExecutor,
        *,
        max_tubes: int = 5,
        max_samples: int = 30,
    ) -> List[COCFTrainingSample]:
        """Generate the §1.5 counterfactual samples for one teacher trajectory.

        Covers high/mid/low causal tubes (``max_tubes``) × all four actions at every
        cached representative step, capped at ``max_samples`` labels per clip (§1.5).
        """
        steps = sorted(traj.z_by_step)
        if not traj.tubes or not steps:
            return []
        tube_idx_sel = self._select_tubes(traj, max_tubes)
        actions = self.sampler.sample_actions()
        samples: List[COCFTrainingSample] = []
        # The reference side of every damage comparison below is the same
        # ``traj.video_full``; extract its features once instead of once per
        # (tube, step, action, seed). The per-tube masks go in here too, so each
        # selected tube's localised identity reference is computed once per clip
        # rather than once per rollout (§7.1.1 / §P1-4).
        tube_masks: Dict[int, Tensor] = {
            traj.tubes[ti].tube_id: tube_pixel_mask(traj.video_full, traj.tubes[ti])
            for ti in tube_idx_sel
        }
        feats_full = self.damage_computer.reference_features(
            traj.video_full, traj.prompt, tube_masks=tube_masks
        )

        for step_idx, ti, a in self._balanced_triplets(
            steps, tube_idx_sel, actions, max_samples
        ):
            t = traj.num_total_steps - step_idx          # countdown timestep
            step_frac = t / max(1, traj.num_total_steps)
            tube = traj.tubes[ti]
            tid = tube.tube_id
            state = traj.tube_states[tid]
            feats = traj.strength_feats[tid]
            action = Action(a)
            damage, unc, cost, y_cf, per_axis = self._counterfactual_labels(
                traj, step_idx, tube, action, backbone, transition,
                feats_full=feats_full, tube_mask=tube_masks.get(tid),
            )
            samples.append(
                COCFTrainingSample(
                    tube_features=self._state_for(state, action).as_tensor(),
                    timestep=int(t),
                    action=int(action),
                    damage_label=damage,
                    damage_per_axis=per_axis,
                    strength_features=feats.as_tensor(),
                    cost_label=cost,
                    uncertainty=unc,
                    tube_visual_embed_full=traj.tube_visual_embed_full.get(tid),
                    tube_visual_embed_cf=self._tube_visual_embed(y_cf, tube, traj.grid),
                    text_embed=traj.text_embed,
                    prompt=traj.prompt,
                    video_id=traj.video_id,
                    tube_id=tid,
                    scene_type=traj.scene_type,
                    interaction_density=float(min(max(state.interaction, 0.0), 1.0)),
                    strength_level=int(self._strength_level(feats)),
                    step_frac=float(step_frac),
                    skip_residual=float(getattr(self, "_last_residual", 0.0)),
                    tube_token_count=int(tube.size),
                    tube_pixels=self._count_tube_pixels(tube),
                    tube_stability=float(state.identity_confidence),
                )
            )
            # Coarse backstop to the per-rollout reclaim in
            # :meth:`_counterfactual_labels`: catches the zeroth seed's clip
            # (held until the sample is built) and the metric backbones'
            # residue. ``empty_cache`` is a synchronising call, hence the
            # cadence rather than per-sample.
            if self.free_memory_every and len(samples) % self.free_memory_every == 0:
                free_memory()
        _log.debug("Generated %d counterfactual samples for %s", len(samples), traj.video_id)
        return samples

    @staticmethod
    def _balanced_triplets(
        steps: Sequence[int],
        tube_idx_sel: Sequence[int],
        actions: Sequence[int],
        max_samples: int,
    ) -> List[Tuple[int, int, int]]:
        """``(step_idx, tube_idx, action)`` triplets ordered so truncation stays balanced.

        The generator can only afford ``max_samples`` rollouts per clip, and the order
        in which candidates are visited therefore *is* the sampling design. Nested
        loops with an inner ``break`` visit the whole (tube × action) grid of the first
        representative step before reaching the second, so a cap of 15 against 4 tubes ×
        4 actions = 16 candidates per step drew **every** sample from the earliest step
        (§P1-5). The predictor then saw one denoising phase while inference asks it to
        decide on all of them, leaving ``step_frac`` a constant it cannot use.

        Interleaving across steps first — one triplet from each step per round, cycling
        actions before tubes — makes any prefix of the sequence balanced: steps differ
        by at most one sample, and actions are evenly spread inside each step.
        """
        per_step: Dict[int, List[Tuple[int, int, int]]] = {}
        for s in steps:
            # actions vary fastest so a short prefix still covers all four
            per_step[s] = [(s, ti, a) for ti in tube_idx_sel for a in actions]
        out: List[Tuple[int, int, int]] = []
        for round_i in range(max(len(v) for v in per_step.values()) if per_step else 0):
            for s in steps:
                bucket = per_step[s]
                if round_i < len(bucket):
                    out.append(bucket[round_i])
                    if len(out) >= max_samples:
                        return out
        return out

    # ------------------------------------------------------------------ #
    # single-hop counterfactual rollout (§1.5)
    # ------------------------------------------------------------------ #

    def _counterfactual_labels(
        self,
        traj: TeacherTrajectory,
        step_idx: int,
        tube: SemanticTube,
        action: Action,
        backbone: BackboneAdapter,
        transition: TransitionExecutor,
        *,
        feats_full: Optional[VideoFeatures] = None,
        tube_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, float]]:
        """Return ``(damage[8], uncertainty[8], cost[2], Y_cf[F,3,H,W], per_axis)``.

        FULL is the reference: zero damage by construction (no local modification),
        so its label is exact and needs no rollout. Skip actions roll out
        ``seeds_per_prompt`` times (small seeded perturbations of ``z_t``) so the
        per-axis variance is the §1.5 multi-seed uncertainty label.

        ``feats_full`` is the caller's hoisted reference-feature extraction (all
        rollouts of a trajectory compare against the same ``traj.video_full``).
        """
        cost = self._cost_label(action, transition)
        self._last_residual = 0.0
        if action == Action.FULL:
            zero = torch.zeros(NUM_DAMAGE_DIMS)
            return zero, zero.clone(), cost, traj.video_full, {ax: 0.0 for ax in DAMAGE_DIMENSIONS}

        z_t = traj.z_by_step[step_idx]
        dmgs: List[Tensor] = []
        y_cf0: Optional[Tensor] = None
        per_axis0: Dict[str, float] = {}
        for k in range(self.seeds_per_prompt):
            z0 = z_t if k == 0 else z_t + self.perturb_std * self._seeded_noise(
                z_t, traj.video_id, step_idx, tube.tube_id, int(action), k
            )
            y_cf = self._rollout(z0, step_idx, tube, action, traj, backbone, transition)
            dmg, per_axis = self.damage_computer.compute_damage(
                traj.video_full, y_cf, traj.prompt, tube_mask,
                tube_id=tube.tube_id, feats_full=feats_full,
            )
            dmgs.append(dmg)
            if k == 0:
                y_cf0, per_axis0 = y_cf, per_axis
            else:
                # Every seed but the zeroth produces a decoded clip that dies here,
                # alongside the VAE tile buffers and metric-backbone activations the
                # rollout just churned through. Reclaiming per *rollout* rather than
                # per sample matters because the rollout — not the sample — is the
                # unit that ends in a full VAE decode: at seeds_per_prompt=3 the
                # sample-level cadence below would let three decodes' worth of freed
                # blocks fragment before any of them is returned to the driver.
                del y_cf
                if self.free_memory_every:
                    free_memory()
        D = torch.stack(dmgs)                                  # [seeds, 8]
        damage = D.mean(0).clamp(0, 1)
        uncertainty = D.var(0, unbiased=False) if self.seeds_per_prompt > 1 else torch.zeros_like(damage)
        return damage, uncertainty, cost, y_cf0, per_axis0

    def _rollout(
        self,
        z_t: Tensor,
        step_idx: int,
        tube: SemanticTube,
        action: Action,
        traj: TeacherTrajectory,
        backbone: BackboneAdapter,
        transition: TransitionExecutor,
    ) -> Tensor:
        """Apply the action at step ``t`` then continue all-FULL to ``z_0`` → ``Y_cf``."""
        grid, cond, T = traj.grid, traj.cond, traj.num_total_steps
        device = z_t.device
        t = T - step_idx
        t_now = torch.full((z_t.shape[0],), sigma_from_step(t, T), device=device)
        t_next = torch.full((z_t.shape[0],), sigma_from_step(t - 1, T), device=device)
        # dense full-compute step at the intervention timestep (the FULL reference advance)
        out = backbone.denoise(z_t, t_now, cond, grid=grid, active_mask=None, cache=None)
        z_full = backbone.scheduler_step(out.cache.model_output, t_now, t_next, z_t)
        z = self._apply_action_to_tube(z_full, z_t, tube, action, grid, transition)
        # δ_k at the intervened step: exactly the quantity RAEC's certificate weights
        # with λ_res at inference. Measuring it here gives Stage B a real signal to
        # calibrate that coefficient against (§P1-13).
        idx = tube.all_token_indices().to(z.device)
        self._last_residual = float(
            (z_full.index_select(1, idx) - z.index_select(1, idx))
            .pow(2).mean().sqrt().item()
        ) if idx.numel() else 0.0
        # continue all-FULL (dense) to z_0 — single-hop: only step t was intervened
        for s in range(step_idx + 1, T):
            ts = T - s
            tn = torch.full((z.shape[0],), sigma_from_step(ts, T), device=device)
            tnn = torch.full((z.shape[0],), sigma_from_step(ts - 1, T), device=device)
            z = backbone.full_transition(z, tn, tnn, cond, grid=grid).model_output
        return _frames_fchw(backbone.decode_latent(backbone.to_grid(z, grid)))

    def _apply_action_to_tube(
        self,
        z_full: Tensor,
        z_prev: Tensor,
        tube: SemanticTube,
        action: Action,
        grid: TokenGrid,
        transition: TransitionExecutor,
    ) -> Tensor:
        """Produce ``z_t^{cf}`` by applying ``action`` to *only* this tube's tokens."""
        if action == Action.FULL:
            return z_full
        idx = tube.all_token_indices().to(z_full.device)
        if idx.numel() == 0:
            return z_full
        if action == Action.INTERP:
            # temporal interpolation via the shared inference realisation (it returns
            # its own copy, so do not pre-clone: these rollouts run ~90× per clip and
            # a full-latent clone each is pure waste). ``freeze_to=z_prev`` mirrors the
            # executor's single-frame fallback — without it a one-frame tube's INTERP
            # label would be identical to FULL (zero damage) while inference freezes it.
            return transition.interp_temporal(z_full, tube, grid, freeze_to=z_prev)
        if action == Action.LOWFREQ:
            # strided subsample + nearest upsample via the shared inference realisation
            return transition.coarsen_lowfreq(z_full, tube, grid)
        z_cf = z_full.clone()
        if action == Action.ANCHOR:
            # freeze: the tube does not advance this step (reuse the pre-step latent)
            z_cf.index_copy_(1, idx, z_prev.index_select(1, idx).to(z_cf.dtype))
        return z_cf

    @staticmethod
    def _state_for(state: TubeState, action: Action) -> TubeState:
        """The tube state to record *alongside this action's label* (§P1-3).

        ``anchor_age`` is one of the 7 predictor inputs, so it must describe the world
        the label was measured in. The ANCHOR rollout freezes the tube to the latent of
        the **immediately preceding** step (``z_prev``), i.e. an anchor exactly one step
        old — but the teacher's tube state carries the builder's default ``age = 0``,
        teaching the predictor that this damage occurs at zero staleness.

        Inference reuses an anchor that may be several steps old. That residual gap is
        priced separately and deliberately: the certificate's ``λ_age · age/(age+τ)``
        term (§5.3.1) models staleness on top of μ. Labelling the age the rollout
        actually used is what makes that decomposition sound — μ becomes "the damage of
        a one-step-old anchor" rather than "of an anchor of unknown age".
        """
        if action != Action.ANCHOR:
            return state
        return dataclass_replace(state, anchor_age=1.0)

    def _cost_label(self, action: Action, transition: TransitionExecutor) -> Tensor:
        """``[relative compute cost, active-token fraction]`` of the action (§1.5).

        The two fields are *distinct measurements* and were collapsed into one for a
        while, which is how the store came to carry ``action_cost[LOWFREQ] = 0.45``
        beside an active fraction of ``1/stride² = 0.25`` for the same sample — a cost
        head trained on that learns an average of two incompatible claims (§P1-6).
        They are kept distinct here and both sourced from the executor's own stride, so
        neither can drift from what the engine performs:

        * **relative compute cost** — what the allocator's knapsack prices the action
          at (``ACTION_COST``, LOWFREQ overridden by the real stride). INTERP is
          non-zero because it does latent-space work even though no denoiser runs
          (§P4-1), so this field is *not* identical to the active fraction.
        * **active-token fraction** — the share of the tube's tokens the denoiser
          freshly computes: 1 for FULL, ``1/stride²`` for LOWFREQ, 0 for both skips.
        """
        stride = max(1, transition.lowfreq_stride)
        lowfreq_frac = 1.0 / float(stride * stride)
        active = (1.0, lowfreq_frac, 0.0, 0.0)[int(action)]
        cost = list(self.action_cost)
        cost[int(Action.LOWFREQ)] = lowfreq_frac
        return torch.tensor([cost[int(action)], active], dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # feature extraction (real, no placeholders)
    # ------------------------------------------------------------------ #

    def _tube_visual_embed(self, video_fchw: Tensor, tube: SemanticTube, grid: TokenGrid) -> Tensor:
        """Per-tube CLIP visual embed ``[d_v]`` from a representative frame (CMSC)."""
        return tube_clip_embed(video_fchw, tube, grid, self.perception)

    def _select_tubes(self, traj: TeacherTrajectory, max_tubes: int) -> List[int]:
        """Pick ≤``max_tubes`` tube indices spanning high/mid/low causal levels (§1.5)."""
        tubes = traj.tubes
        if len(tubes) <= max_tubes:
            return list(range(len(tubes)))
        order = sorted(
            range(len(tubes)),
            key=lambda i: self._strength_scalar(traj.strength_feats[tubes[i].tube_id]),
            reverse=True,
        )
        picks = sorted({int(round(j * (len(order) - 1) / (max_tubes - 1))) for j in range(max_tubes)})
        return [order[p] for p in picks]

    @staticmethod
    def _strength_scalar(feats: StrengthFeatures) -> float:
        return (feats.s_E + feats.s_A + feats.s_T) / 3.0

    def _strength_level(self, feats: StrengthFeatures) -> StrengthLevel:
        """Map ``(s_E,s_A,s_T)`` to a HIGH/MID/LOW causal level (§3.3.3 thresholds)."""
        s = self._strength_scalar(feats)
        if s > 0.66:
            return StrengthLevel.HIGH
        if s > 0.33:
            return StrengthLevel.MID
        return StrengthLevel.LOW

    @staticmethod
    def _seeded_noise(like: Tensor, *keys) -> Tensor:
        """Deterministic Gaussian like ``like`` — reproducible per (clip,step,tube,...).

        Seeds from a *stable* hash of the keys, not Python's ``hash()`` (which is
        salted per process via ``PYTHONHASHSEED`` for string keys like ``video_id``),
        so the multi-seed perturbations — hence the §1.5 uncertainty labels — are
        byte-identical across runs, processes and shards.
        """
        digest = hashlib.sha1("|".join(map(str, keys)).encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(like.shape, generator=g).to(like.device, like.dtype)

    @staticmethod
    def _count_tube_pixels(tube: SemanticTube) -> int:
        """Total latent-mask area spanned by the tube across its frames."""
        if not tube.masks_by_frame:
            return 0
        return int(sum(int(m.sum()) for m in tube.masks_by_frame.values()))
