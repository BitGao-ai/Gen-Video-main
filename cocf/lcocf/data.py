"""Counterfactual teacher data generation.

Generates multi-dimensional training labels for L-COCF by applying tube-level
single-hop counterfactual interventions:

    1. Run the full-compute model on the reference clip to get Y_full, {z_t}, tubes
    2. Build semantic tubes once, with their states and (s_E, s_A, s_T) features
    3. For sampled (tube, step, action) triplets: apply the action to only that tube,
       continue all-FULL denoising to z_0, decode to Y_cf
    4. Compute the multi-dimensional damage, the compute-cost label and the multi-seed
       uncertainty as the training labels

This module implements :class:`COCFTrainingSample` (the per-sample label record),
:class:`TeacherTrajectory` (per-clip teacher-forward outputs), stratified sampling, and
:class:`COCFDataGenerator` (the single-hop intervention rollout).
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

# Per-action relative compute cost, mirroring AllocatorConfig.action_cost and the
# executor's FULL/LOWFREQ/INTERP/ANCHOR tiers (the executor is authoritative; see
# ``_cost_label``).
ACTION_COST: Tuple[float, float, float, float] = (1.0, 0.25, 0.02, 0.0)


def _to_np(x: Optional[Tensor]) -> Optional[np.ndarray]:
    """Detach a tensor to a small CPU float32 numpy array for on-disk storage."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype="float32")


def _to_tensor(x: object) -> Optional[Tensor]:
    """Inverse of :func:`_to_np`: rebuild a float tensor from stored numpy/list."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


# ============================================================================= #
# Core data structures
# ============================================================================= #


@dataclass
class COCFTrainingSample:
    """One counterfactual training label: the per-sample read record for Stage B.

    Carries every field Stage B reads for the joint loss: causal-strength features, the
    tube state, timestep and action encodings, tube token count, the multi-dimensional
    degradation label, the compute-cost label, the multi-seed uncertainty, the tube
    visual embeddings of the full and counterfactual renders, and the prompt embedding.
    :meth:`to_dict` / :meth:`from_dict` round-trip it through the LMDB store; tensors
    are stored as small CPU numpy arrays.
    """

    # core predictor inputs
    tube_features: Tensor                 # [7] the tube state vector
    timestep: int                         # t (countdown index over the schedule)
    action: int                           # Action {FULL=0,LOWFREQ=1,INTERP=2,ANCHOR=3}
    damage_label: Tensor                  # [NUM_DAMAGE_DIMS] in [0, 1] degradation

    # remaining read fields (defaulted for partial/legacy samples)
    strength_features: Optional[Tensor] = None   # [3] (s_E, s_A, s_T)
    cost_label: Optional[Tensor] = None          # [2] (FLOPs frac, active-token frac)
    uncertainty: Optional[Tensor] = None         # [NUM_DAMAGE_DIMS] multi-seed variance
    tube_visual_embed_full: Optional[Tensor] = None  # [d_v] full-render tube embed
    tube_visual_embed_cf: Optional[Tensor] = None    # [d_v] cf-render tube embed
    text_embed: Optional[Tensor] = None          # [L, d_c] prompt token sequence

    damage_per_axis: Dict[str, float] = field(default_factory=dict)

    # auxiliary metadata
    prompt: str = ""
    video_id: str = ""
    tube_id: int = -1
    scene_type: str = "dynamic"           # static/dynamic/multi/text/face/occlusion
    interaction_density: float = 0.0      # 0-1: how much this tube interacts
    strength_level: int = 1               # StrengthLevel prior (HIGH=0/MID=1/LOW=2)
    step_frac: float = 0.0                # t / T in [0, 1] (denoising phase)
    # Skip residual on the tube's tokens at the intervened step; the same quantity the
    # inference-time certificate feeds its residual term, recorded so Stage B can
    # calibrate that coefficient against a real signal.
    skip_residual: float = 0.0
    tube_token_count: int = 0             # latent token count (cost context)
    tube_pixels: int = 0                  # latent-token area of the region (diagnostic)
    tube_stability: float = 1.0           # identity stability in [0, 1]

    def damage_scalar(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Reduce multi-dim damage to a scalar for quick sorting/filtering."""
        if weights is None:
            weights = DEFAULT_DAMAGE_WEIGHTS
        scalar = 0.0
        for i, axis in enumerate(DAMAGE_DIMENSIONS):
            scalar += float(self.damage_label[i]) * weights.get(axis, 0.0)
        return min(1.0, scalar)  # clamp to [0,1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the LMDB payload (keys match collate_cocf_samples fields)."""
        return {
            # vectors (numpy, float32)
            "tube_features": _to_np(self.tube_features),
            "strength_features": _to_np(self.strength_features),
            "damage_label": _to_np(self.damage_label),
            "cost_label": _to_np(self.cost_label),
            "uncertainty": _to_np(self.uncertainty),
            "tube_visual_embed_full": _to_np(self.tube_visual_embed_full),
            "tube_visual_embed_cf": _to_np(self.tube_visual_embed_cf),
            # text_embed is per-clip and stored separately, then joined by the reader.
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
            # diagnostics (ignored by collate; kept for analysis/cleaning)
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
    """Orchestrates multi-dimensional damage computation.

    Encapsulates feature extraction (:class:`MetricExtractor`), damage calculation
    (:class:`MultiDimDamageComputer`) and per-axis normalisation/clamping.
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

        ``video_full`` is fixed for a whole :class:`TeacherTrajectory` while many
        counterfactual rollouts are scored against it, so callers hoist this out of
        their loop and pass it to :meth:`compute_damage`. ``tube_masks`` requests the
        per-tube identity features the localised damage axes need; pass every tube that
        will be scored against this reference.
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
            tube_mask: Per-frame ``[F, H, W]`` mask of the intervened tube. Supplying it
                (with ``tube_id``) localises the identity axes to that tube; without it
                the label is global.
            tube_id: Which entry of ``feats_full.tube_dino`` to compare against; must
                match the key used when the reference was extracted.
            feats_full: Pre-extracted reference features; extracted on demand if omitted.

        Returns:
            damage_vector: [NUM_DAMAGE_DIMS] in [0, 1]
            per_axis_dict: {axis_name: scalar_value} for diagnostics
        """
        # Localise only when both halves are available.
        masks = (
            {tube_id: tube_mask}
            if (tube_mask is not None and tube_id is not None) else None
        )
        if feats_full is None:
            feats_full = self.metric_extractor.extract(video_full, prompt, tube_masks=masks)
        localise = tube_id if (masks and tube_id in feats_full.tube_dino) else None
        feats_cf = self.metric_extractor.extract(video_cf, prompt, tube_masks=masks)

        # Compute the per-axis dict, then convert it to the ordered tensor.
        per_axis = self.damage_computer.compute(feats_full, feats_cf, tube_id=localise)
        damage_vec = self.damage_computer.as_vector(per_axis, device=video_full.device)

        return damage_vec.clamp(0, 1), per_axis


# ============================================================================= #
# Stratified sampling strategies
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
    # Group timesteps into early [0.8T..T], mid [0.3T..0.8T], late [0..0.3T]
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
    samples_per_prompt: int = 3  # sample 3 seeds per prompt


class StratifiedSampler:
    """Generates balanced (tube, timestep, action) triplets from a video.

    Stratified sampling ensures representation across scene types
    (static/dynamic/text/face), temporal phases (early/mid/late) and action types
    (FULL/LOWFREQ/INTERP/ANCHOR).
    """

    def __init__(self, config: StratifiedSamplingConfig, device: torch.device):
        self.config = config
        self.device = device

    def sample_actions(self, num_samples: int = 4) -> List[int]:
        """The four compute actions per tube (FULL/LOWFREQ/INTERP/ANCHOR).

        Each tube at each representative step is probed with all four actions, giving
        action-balanced training material. ``num_samples`` only truncates.
        """
        actions = [int(Action.FULL), int(Action.LOWFREQ), int(Action.INTERP), int(Action.ANCHOR)]
        return actions if num_samples >= len(actions) else actions[:num_samples]


# ============================================================================= #
# Label interpolation

@dataclass
class TeacherTrajectory:
    """Per-clip teacher full-compute outputs assembled by Stage A.

    One reference full-denoise trajectory: the decoded reference video ``Y_full``, the
    latents ``z_t`` cached at the representative steps, the semantic tubes built once on
    the reference, their states and ``(s_E, s_A, s_T)`` features, and the per-tube
    visual embeddings of the full render. A single-hop intervention replays one cached
    ``z_t`` with one tube's action changed and continues all-FULL to ``z_0``.
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
    # z_T the baseline was generated from, persisted so Stage C can start its
    # accelerated run on the same noise and compare against the stored reference.
    z_init: Optional[Tensor] = None                 # [1, N, d]


def _frames_fchw(video: Tensor) -> Tensor:
    """``[B, 3, F, H, W]`` (or ``[3, F, H, W]``) to ``[F, 3, H, W]``.

    Layout only; the value range is settled at the decode by
    :meth:`~cocf.backbones.base.BackboneAdapter.decode_to_unit`.
    """
    v = video[0] if video.dim() == 5 else video
    return v.permute(1, 0, 2, 3).contiguous()


def tube_pixel_mask(video_fchw: Tensor, tube: SemanticTube, grid: Optional[TokenGrid] = None,
                    *, frame_span=None, full_frame_count=None) -> Tensor:
    """``[F, Hp, Wp]`` bool: the tube's latent masks upsampled to pixel resolution.

    The damage extractor works on pixels while a tube carries latent-grid masks, so this
    bridges the two and lets a label be scored on the region the counterfactual
    intervened on. Each pixel frame uses the nearest representative latent slot; window
    offsets are in full-video pixel coordinates.
    """
    f, _, hp, wp = video_fchw.shape
    out = torch.zeros(f, hp, wp, dtype=torch.bool, device=video_fchw.device)
    start = frame_span[0] if frame_span is not None else 0
    total = full_frame_count if full_frame_count is not None else f
    slots = grid.t if grid is not None else f
    if frame_span is not None and full_frame_count is None:
        raise ValueError("Windowed tube masks require full_frame_count")
    if total < 1 or slots < 1 or start < 0 or start + f > total:
        raise ValueError("Invalid video/tube temporal geometry")
    if frame_span is not None and frame_span[1] - start != f:
        raise ValueError("frame_span length must match decoded frame count")
    # Invert the same evenly spaced representative-frame mapping used by builders.
    representatives = torch.linspace(0, total - 1, slots).round().long()
    pixels = torch.arange(start, start + f)
    owners = (pixels[:, None] - representatives[None, :]).abs().argmin(1)
    frames = [(i, tube.masks_by_frame[int(slot)]) for i, slot in enumerate(owners)
              if tube.masks_by_frame.get(int(slot)) is not None]
    if not frames:
        return out
    stacked = torch.stack([m.float() for _, m in frames]).unsqueeze(1)   # [n,1,h,w]
    up = F.interpolate(stacked, size=(hp, wp), mode="nearest")[:, 0] > 0.5
    idx = torch.tensor([frame for frame, _ in frames], device=out.device)
    out.index_copy_(0, idx, up.to(out.device))
    return out


def tube_clip_embed(
    video_fchw: Tensor,
    tube: SemanticTube,
    grid: TokenGrid,
    perception,
    d_v: Optional[int] = None,
    *, differentiable: bool = False, frame_span=None, full_frame_count=None,
) -> Tensor:
    """Per-tube CLIP visual embed ``[d_v]`` from a representative frame (feeds CMSC).

    Picks the middle visible pixel frame, maps its latent mask to pixel resolution and
    calls ``perception.clip_feature``, the same source
    :class:`~cocf.tubes.regions.RegionExtractor` uses. Returns zeros when no
    perception/mask is available.
    """
    if d_v is None:
        d_v = int(getattr(perception, "d_clip", 64)) if perception is not None else 64
    frames = tube.frames
    zeros = torch.zeros(d_v, device=video_fchw.device)
    if perception is None or not frames:
        return zeros
    masks = tube_pixel_mask(video_fchw, tube, grid, frame_span=frame_span,
                            full_frame_count=full_frame_count)
    visible = masks.flatten(1).any(1).nonzero().flatten()
    if not visible.numel():
        return zeros
    fi = int(visible[len(visible) // 2])
    frame = video_fchw[fi]                                   # [3, Hp, Wp]
    feature_fn = getattr(perception, "clip_feature_grad", perception.clip_feature) if differentiable else perception.clip_feature
    feature = feature_fn(frame, masks[fi]).float()
    return feature if differentiable else feature.detach()


# ============================================================================= #
# Full counterfactual data generation pipeline
# ============================================================================= #


def _seeded_noise(like: Tensor, *keys) -> Tensor:
    """Deterministic Gaussian like ``like``, reproducible per (clip, step, seed).

    Seeds from a stable hash of the keys (not Python's salted ``hash()``) so the
    multi-seed perturbations and uncertainty labels are identical across runs, processes
    and shards. Generated on CPU so a store built on one card reproduces on another.
    """
    digest = hashlib.sha1("|".join(map(str, keys)).encode("utf-8")).hexdigest()
    g = torch.Generator().manual_seed(int(digest[:8], 16))
    return torch.randn(like.shape, generator=g).to(like.device, like.dtype)


class _FullStepCache:
    """Memoises the dense full-compute advance shared by every rollout at a step.

    A single-hop rollout first advances the cached ``z_t`` one unmodified full-compute
    step, then applies the action to the intervened tube. That advance depends only on
    ``(step_idx, seed)``, so it is shared across the rollouts at a step instead of
    recomputed. All actions at a step then see the same perturbation (common random
    numbers), so the multi-seed variance measures the action's own sensitivity. Both
    returned tensors are read-only to callers, which clone before writing.
    """

    def __init__(self, traj: "TeacherTrajectory", backbone: BackboneAdapter,
                 perturb_std: float) -> None:
        self._traj = traj
        self._backbone = backbone
        self._perturb_std = perturb_std
        self._entries: Dict[Tuple[int, int], Tuple[Tensor, Tensor]] = {}
        self._references = {}

    def reference(self, step_idx, seed, generator, tube, transition, damage_computer, tube_mask):
        """Cache a paired full continuation for each perturbed starting state."""
        key = (step_idx, seed)
        if key not in self._references:
            before, after = self.get(step_idx, seed)
            video, _ = generator._rollout(before, after, step_idx, tube, Action.FULL,
                                          self._traj, self._backbone, transition)
            self._references[key] = (video, {})
        video, by_tube = self._references[key]
        if tube.tube_id not in by_tube:
            by_tube[tube.tube_id] = damage_computer.reference_features(
                video, self._traj.prompt, tube_masks={tube.tube_id: tube_mask})
        return video, by_tube[tube.tube_id]

    def get(self, step_idx: int, seed: int) -> Tuple[Tensor, Tensor]:
        """``(z_prev, z_full)`` at ``step_idx`` under perturbation ``seed``."""
        key = (step_idx, seed)
        cached = self._entries.get(key)
        if cached is not None:
            return cached

        traj = self._traj
        z_prev = traj.z_by_step[step_idx]
        if seed:
            z_prev = z_prev + self._perturb_std * _seeded_noise(
                z_prev, traj.video_id, step_idx, seed
            )
        T = traj.num_total_steps
        t = T - step_idx
        t_now = torch.full((z_prev.shape[0],), sigma_from_step(t, T), device=z_prev.device)
        t_next = torch.full((z_prev.shape[0],), sigma_from_step(t - 1, T), device=z_prev.device)
        out = self._backbone.denoise(
            z_prev, t_now, traj.cond, grid=traj.grid, active_mask=None, cache=None
        )
        z_full = self._backbone.scheduler_step(out.cache.model_output, t_now, t_next, z_prev)

        entry = (z_prev, z_full)
        self._entries[key] = entry
        return entry


class COCFDataGenerator:
    """Single-hop counterfactual training-data generation.

    For each representative (tube, step, action) it:
        1. replays the cached ``z_t`` and takes one dense full-compute step;
        2. applies the action to only that tube, reusing the inference
           :class:`~cocf.backbones.transition.TransitionExecutor` for LOWFREQ;
        3. continues all-FULL to ``z_0`` and decodes to ``Y_cf``;
        4. scores the multi-dim damage vs ``Y_full``, the compute-cost label and the
           multi-seed uncertainty.

    FULL is the zero-damage reference by construction.
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
        free_memory_every: int = 0,
    ):
        self.metric_extractor = metric_extractor
        self.strength_builder = strength_feature_builder
        self.damage_computer = damage_computer
        self.sampling_config = sampling_config or StratifiedSamplingConfig()
        self.device = device
        self.perception = perception
        self.action_cost = tuple(action_cost)
        self.seeds_per_prompt = max(1, int(seeds_per_prompt))
        self.perturb_std = float(perturb_std)
        # Return freed allocator blocks to the driver (0 disables).
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
        """Generate the counterfactual samples for one teacher trajectory.

        Covers high/mid/low causal tubes (``max_tubes``) across all four actions at
        every cached representative step, capped at ``max_samples`` labels per clip.
        """
        steps = sorted(traj.z_by_step)
        if not traj.tubes or not steps:
            return []
        tube_idx_sel = self._select_tubes(traj, max_tubes)
        actions = self.sampler.sample_actions()
        samples: List[COCFTrainingSample] = []
        # Extract the reference features once (with the per-tube masks) instead of once
        # per (tube, step, action, seed).
        tube_masks: Dict[int, Tensor] = {
            traj.tubes[ti].tube_id: tube_pixel_mask(traj.video_full, traj.tubes[ti], traj.grid)
            for ti in tube_idx_sel
        }
        feats_full = self.damage_computer.reference_features(
            traj.video_full, traj.prompt, tube_masks=tube_masks
        )
        step_cache = _FullStepCache(traj, backbone, self.perturb_std)

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
            damage, unc, cost, y_cf, per_axis, residual = self._counterfactual_labels(
                traj, step_idx, tube, action, backbone, transition,
                feats_full=feats_full, tube_mask=tube_masks.get(tid),
                step_cache=step_cache,
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
                    skip_residual=float(residual),
                    tube_token_count=int(tube.size),
                    tube_pixels=self._count_tube_pixels(tube),
                    tube_stability=float(state.identity_confidence),
                )
            )
            # Coarse backstop to the per-rollout reclaim in
            # :meth:`_counterfactual_labels`; ``empty_cache`` synchronises, hence the
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

        The generator can only afford ``max_samples`` rollouts per clip, so the visit
        order is the sampling design and must stay balanced across step, tube and action
        at once. Candidates are enumerated round by round: round ``r`` pairs tube ``i``
        with action ``(i + r) mod n_a``, so each round covers every tube once and walks a
        different diagonal of the (tube, action) grid, and no pair repeats for any
        ``n_t``.
        """
        n_t, n_a = len(tube_idx_sel), len(actions)
        if not n_t or not n_a:
            return []
        candidates = [
            (tube_idx_sel[i], actions[(i + r) % n_a])
            for r in range(n_a)
            for i in range(n_t)
        ]
        out: List[Tuple[int, int, int]] = []
        for ti, a in candidates:
            for s in steps:
                out.append((s, ti, a))
                if len(out) >= max_samples:
                    return out
        return out

    # ------------------------------------------------------------------ #
    # single-hop counterfactual rollout
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
        step_cache: "_FullStepCache",
        feats_full: Optional[VideoFeatures] = None,
        tube_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, float], float]:
        """Return ``(damage, uncertainty, cost, Y_cf, per_axis, residual)``.

        FULL is the reference: zero damage by construction, no rollout. Skip actions roll
        out ``seeds_per_prompt`` times (small seeded perturbations of ``z_t``) so the
        per-axis variance is the multi-seed uncertainty label; the residual is the mean
        skip residual over those seeds.

        ``feats_full`` is the caller's hoisted reference-feature extraction;
        ``step_cache`` supplies the dense advance every rollout at this step shares.
        """
        cost = self._cost_label(action, transition)
        if action == Action.FULL:
            zero = torch.zeros(NUM_DAMAGE_DIMS)
            return (zero, zero.clone(), cost, traj.video_full,
                    {ax: 0.0 for ax in DAMAGE_DIMENSIONS}, 0.0)

        dmgs: List[Tensor] = []
        residuals: List[float] = []
        y_cf0: Optional[Tensor] = None
        per_axis0: Dict[str, float] = {}
        for k in range(self.seeds_per_prompt):
            z_prev, z_full = step_cache.get(step_idx, k)
            reference_video, reference_feats = traj.video_full, feats_full
            if k:
                reference_video, reference_feats = step_cache.reference(
                    step_idx, k, self, tube, transition, self.damage_computer, tube_mask)
            y_cf, residual = self._rollout(
                z_prev, z_full, step_idx, tube, action, traj, backbone, transition
            )
            residuals.append(residual)
            dmg, per_axis = self.damage_computer.compute_damage(
                reference_video, y_cf, traj.prompt, tube_mask,
                tube_id=tube.tube_id, feats_full=reference_feats,
            )
            dmgs.append(dmg)
            if k == 0:
                y_cf0, per_axis0 = y_cf, per_axis
            else:
                # Reclaim per rollout (the unit that ends in a full VAE decode) rather
                # than per sample, so freed blocks are returned before they fragment.
                del y_cf
                if self.free_memory_every:
                    free_memory()
        D = torch.stack(dmgs)                                  # [seeds, 8]
        damage = D.mean(0).clamp(0, 1)
        uncertainty = D.var(0, unbiased=False) if self.seeds_per_prompt > 1 else torch.zeros_like(damage)
        return (damage, uncertainty, cost, y_cf0, per_axis0,
                sum(residuals) / max(1, len(residuals)))

    def _rollout(
        self,
        z_prev: Tensor,
        z_full: Tensor,
        step_idx: int,
        tube: SemanticTube,
        action: Action,
        traj: TeacherTrajectory,
        backbone: BackboneAdapter,
        transition: TransitionExecutor,
    ) -> Tuple[Tensor, float]:
        """Apply the action at step ``t`` then continue all-FULL to ``z_0``, decode to
        ``Y_cf``.

        Returns the decoded clip and the skip residual measured on the tube's tokens at
        the intervened step, the quantity RAEC's certificate weights at inference.
        ``z_prev``/``z_full`` are the step's shared pre- and post-advance latents, both
        read-only here.
        """
        grid, cond, T = traj.grid, traj.cond, traj.num_total_steps
        device = z_full.device
        z = self._apply_action_to_tube(z_full, z_prev, tube, action, grid, transition)
        idx = tube.all_token_indices().to(z.device)
        residual = float(
            (z_full.index_select(1, idx) - z.index_select(1, idx))
            .pow(2).mean().sqrt().item()
        ) if idx.numel() else 0.0
        # continue all-FULL (dense) to z_0; single-hop, only step t was intervened
        for s in range(step_idx + 1, T):
            ts = T - s
            tn = torch.full((z.shape[0],), sigma_from_step(ts, T), device=device)
            tnn = torch.full((z.shape[0],), sigma_from_step(ts - 1, T), device=device)
            z = backbone.full_transition(z, tn, tnn, cond, grid=grid).model_output
        return _frames_fchw(backbone.decode_to_unit(backbone.to_grid(z, grid))), residual

    def _apply_action_to_tube(
        self,
        z_full: Tensor,
        z_prev: Tensor,
        tube: SemanticTube,
        action: Action,
        grid: TokenGrid,
        transition: TransitionExecutor,
    ) -> Tensor:
        """Produce the counterfactual latent by applying ``action`` to only this tube."""
        if action == Action.FULL:
            return z_full
        idx = tube.all_token_indices().to(z_full.device)
        if idx.numel() == 0:
            return z_full
        if action == Action.INTERP:
            # Temporal interpolation via the shared inference realisation (returns its
            # own copy, so no pre-clone). ``freeze_to=z_prev`` keeps the single-frame
            # fallback at freeze semantics on the label side.
            return transition.interp_temporal(z_full, tube, grid, freeze_to=z_prev)
        if action == Action.LOWFREQ:
            # Strided lattice + nearest-anchor value fill via the shared inference
            # realisation (coarsens the latent itself, not the step increment).
            return transition.coarsen_lowfreq(z_full, tube, grid)
        z_cf = z_full.clone()
        if action == Action.ANCHOR:
            # Freeze: the tube does not advance this step (reuse the pre-step latent).
            z_cf.index_copy_(1, idx, z_prev.index_select(1, idx).to(z_cf.dtype))
        return z_cf

    @staticmethod
    def _state_for(state: TubeState, action: Action) -> TubeState:
        """The tube state to record alongside this action's label.

        ``anchor_age`` is a predictor input, so it must describe the world the label was
        measured in. The ANCHOR rollout freezes the tube to the immediately preceding
        step's latent (an anchor one step old), so the recorded age is set to 1.0.
        """
        if action != Action.ANCHOR:
            return state
        return dataclass_replace(state, anchor_age=1.0)

    def _cost_label(self, action: Action, transition: TransitionExecutor) -> Tensor:
        """``[relative compute cost, active-token fraction]`` of the action.

        Two distinct measurements, both sourced from the executor's own stride:

        * relative compute cost: what the allocator's knapsack prices the action at
          (``ACTION_COST``, LOWFREQ overridden by the real stride).
        * active-token fraction: the share of the tube's tokens the denoiser freshly
          computes (1 for FULL, ``1/stride^2`` for LOWFREQ, 0 for both skips).
        """
        stride = max(1, transition.lowfreq_stride)
        lowfreq_frac = 1.0 / float(stride * stride)
        active = (1.0, lowfreq_frac, 0.0, 0.0)[int(action)]
        cost = list(self.action_cost)
        cost[int(Action.LOWFREQ)] = lowfreq_frac
        return torch.tensor([cost[int(action)], active], dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # feature extraction
    # ------------------------------------------------------------------ #

    def _tube_visual_embed(self, video_fchw: Tensor, tube: SemanticTube, grid: TokenGrid) -> Tensor:
        """Per-tube CLIP visual embed ``[d_v]`` from a representative frame (CMSC)."""
        return tube_clip_embed(video_fchw, tube, grid, self.perception)

    def _select_tubes(self, traj: TeacherTrajectory, max_tubes: int) -> List[int]:
        """Pick at most ``max_tubes`` tube indices spanning high/mid/low causal levels."""
        tubes = traj.tubes
        if max_tubes < 1:
            raise ValueError("max_tubes must be positive")
        if len(tubes) <= max_tubes:
            return list(range(len(tubes)))
        order = sorted(
            range(len(tubes)),
            key=lambda i: self._strength_scalar(traj.strength_feats[tubes[i].tube_id]),
            reverse=True,
        )
        if max_tubes == 1:
            return order[:1]
        picks = sorted({int(round(j * (len(order) - 1) / (max_tubes - 1))) for j in range(max_tubes)})
        return [order[p] for p in picks]

    @staticmethod
    def _strength_scalar(feats: StrengthFeatures) -> float:
        return (feats.s_E + feats.s_A + feats.s_T) / 3.0

    def _strength_level(self, feats: StrengthFeatures) -> StrengthLevel:
        """Map ``(s_E, s_A, s_T)`` to a HIGH/MID/LOW causal level."""
        s = self._strength_scalar(feats)
        if s > 0.66:
            return StrengthLevel.HIGH
        if s > 0.33:
            return StrengthLevel.MID
        return StrengthLevel.LOW

    @staticmethod
    def _count_tube_pixels(tube: SemanticTube) -> int:
        """Total latent-mask area spanned by the tube across its frames."""
        masks = [m for m in tube.masks_by_frame.values() if m is not None]
        if not masks:
            return 0
        return int(torch.stack([m.sum() for m in masks]).sum())
