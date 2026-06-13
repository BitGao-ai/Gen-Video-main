"""Counterfactual teacher data generation (§7.1.1).

Generates multi-dimensional training labels for L-COCF by applying tube-level
counterfactual interventions. The core workflow (steps d-f of §7.1.1):

    1. Run full-compute model on reference video → Y_full, {z_t}, attention cache
    2. Detect semantic tubes G_t at representative timesteps
    3. For sampled (g_k, t, a) triplets: apply action a to tube g_k, continue
       denoising, decode → Y_cf
    4. Compute multi-dimensional damage D(Y_full, Y_cf) as the training label

This module implements:
    - :class:`COCFTrainingSample`: the label structure (§7.1.1)
    - Stratified sampling: by scene / timestep / action type
    - Label interpolation: linear interp between computed labels
    - Damage caching: avoid recomputing features between generations
    - Batch interventions: efficient tube-group counterfactuals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from cocf.common.types import Action, SemanticTube, StrengthLevel, TubeState
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


# ============================================================================= #
# Core data structures (§7.1.1)
# ============================================================================= #


@dataclass
class COCFTrainingSample:
    """Atomic counterfactual training label (§7.1.1, document storage format).

    Represents one (tube, timestep, action) triplet with its multi-dimensional
    damage label and auxiliary metadata. Small enough to fit millions on disk when
    tube features are compressed.
    """

    # Core features
    tube_features: Tensor  # [7] or [d]: tube state + optional attention
    timestep: int  # t (countdown from T to 1)
    action: int  # ActionType (FULL=0, LOWFREQ=1, INTERP=2, ANCHOR=3)

    # Multi-dimensional damage label (§7.1.1, §9.4)
    damage_label: Tensor  # [NUM_DAMAGE_DIMS=8] ∈ [0,1]
    damage_per_axis: Dict[str, float] = field(default_factory=dict)

    # Auxiliary metadata
    prompt: str = ""
    tube_id: int = -1
    scene_type: str = "generic"  # static/dynamic/text/face/multi/occlusion
    interaction_density: float = 0.0  # 0-1: how much this tube interacts with others
    strength_level: int = 1  # StrengthLevel prior (HIGH/MID/LOW)

    # Optional tube-specific metadata
    tube_pixels: int = 0  # region size in pixels (for context)
    tube_stability: float = 1.0  # identity stability ∈ [0,1]

    def damage_scalar(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Reduce multi-dim damage to a scalar for quick sorting/filtering."""
        if weights is None:
            weights = DEFAULT_DAMAGE_WEIGHTS
        scalar = 0.0
        for i, axis in enumerate(DAMAGE_DIMENSIONS):
            scalar += float(self.damage_label[i]) * weights.get(axis, 0.0)
        return min(1.0, scalar)  # clamp to [0,1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for disk storage (HDF5 / Parquet / etc.)."""
        return {
            "tube_features": self.tube_features.cpu().numpy(),
            "timestep": self.timestep,
            "action": self.action,
            "damage_label": self.damage_label.cpu().numpy(),
            "prompt": self.prompt,
            "tube_id": self.tube_id,
            "scene_type": self.scene_type,
            "interaction_density": float(self.interaction_density),
            "strength_level": self.strength_level,
            "tube_pixels": self.tube_pixels,
            "tube_stability": float(self.tube_stability),
        }


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

    def compute_damage(
        self,
        video_full: Tensor,  # [F, 3, H, W] in [0,1]
        video_cf: Tensor,  # [F, 3, H, W] counterfactual video
        prompt: str,
        tube_mask: Optional[Tensor] = None,  # [F, H, W] binary: 1 inside tube
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute multi-dim damage vector for a counterfactual pair.

        Args:
            video_full: Full-compute reference video.
            video_cf: Counterfactual video (with action applied).
            prompt: Text prompt (for CLIP scoring).
            tube_mask: Optional per-frame mask for localized damage.

        Returns:
            damage_vector: [NUM_DAMAGE_DIMS] ∈ [0,1]
            per_axis_dict: {axis_name: scalar_value} for diagnostics
        """
        # Extract features from both videos (once each)
        feats_full = self.metric_extractor.extract(video_full, prompt)
        feats_cf = self.metric_extractor.extract(video_cf, prompt)

        # Compute multi-dimensional damage. ``compute`` returns the per-axis dict
        # {axis_name: value}; convert it to the ordered [NUM_DAMAGE_DIMS] tensor via
        # ``as_vector`` before doing any tensor ops on it.
        per_axis = self.damage_computer.compute(feats_full, feats_cf)
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
    interpolation_interval: int = 5  # compute full damage every 5 steps, interpolate
    use_label_interpolation: bool = True  # linear interpolation of damage labels


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

    def sample_timesteps(self, total_steps: int) -> List[Tuple[int, str]]:
        """Sample representative timesteps across temporal strata.

        Args:
            total_steps: Total number of denoising steps T.

        Returns:
            List of (timestep, stratum_name) tuples.
        """
        sampled = []
        for stratum, (tmin_frac, tmax_frac, weight) in self.config.timestep_strata.items():
            t_min = int(tmin_frac * total_steps)
            t_max = int(tmax_frac * total_steps)

            # Sample count proportional to weight
            num_samples = max(1, int(weight * 5))  # ~5 samples total per video

            ts = torch.linspace(t_min, t_max, num_samples, dtype=torch.long).tolist()
            sampled.extend([(t, stratum) for t in ts])

        return sorted(sampled, key=lambda x: -x[0])  # descending order

    def sample_tubes(
        self, tubes: List[SemanticTube], max_per_frame: int = 3
    ) -> List[int]:
        """Sample representative tubes by scene stability.

        Prioritize stable (low-mutation) tubes so damage predictions converge faster.
        """
        if not tubes:
            return []

        # Sort by stability (proxy: tube length), take top-K
        tubes_by_stability = sorted(
            enumerate(tubes), key=lambda x: len(x[1]), reverse=True
        )
        return [idx for idx, _ in tubes_by_stability[:max_per_frame]]

    def sample_actions(self, num_samples: int = 3) -> List[int]:
        """Sample action types by cost/frequency distribution."""
        actions = []
        for action, weight in self.config.action_weights.items():
            count = max(1, int(weight * num_samples))
            actions.extend([action] * count)
        return actions[:num_samples]


# ============================================================================= #
# Label interpolation (cost optimization, §7.1.1)
# ============================================================================= #


class DamageLabelInterpolator:
    """Interpolate damage labels between computed timesteps to reduce cost.

    Key insight: damage varies smoothly across timesteps, so computing full
    damage at every 5th step and linearly interpolating neighboring steps
    reduces cost by ~5× with acceptable accuracy loss.
    """

    def __init__(self, interval: int = 5):
        self.interval = interval
        self.cache: Dict[Tuple[int, int, int], Tensor] = {}  # (step, tube_id, action) → damage

    def cache_damage(self, step: int, tube_id: int, action: int, damage: Tensor) -> None:
        """Store computed damage for later interpolation."""
        self.cache[(step, tube_id, action)] = damage.clone()

    def interpolate(
        self, step: int, tube_id: int, action: int, num_total_steps: int
    ) -> Optional[Tensor]:
        """Retrieve or interpolate damage at the given step.

        If step % interval == 0, return cached value.
        Otherwise, linearly interpolate from neighbors.
        """
        key = (step, tube_id, action)
        if key in self.cache:
            return self.cache[key]

        # Find nearest cached neighbors. The cache is keyed by the full
        # (step, tube_id, action) tuple, so neighbor lookups must rebuild that
        # tuple — an integer step alone never matches a tuple key.
        step_lo = (step // self.interval) * self.interval
        step_hi = step_lo + self.interval
        key_lo = (step_lo, tube_id, action)
        key_hi = (step_hi, tube_id, action)

        if key_lo not in self.cache or key_hi not in self.cache:
            # Can't interpolate; skip
            return None

        damage_lo = self.cache[key_lo]
        damage_hi = self.cache[key_hi]

        # Linear interpolation
        alpha = (step - step_lo) / (step_hi - step_lo)
        interpolated = (1 - alpha) * damage_lo + alpha * damage_hi

        return interpolated.clamp(0, 1)


# ============================================================================= #
# Full counterfactual data generation pipeline
# ============================================================================= #


class COCFDataGenerator:
    """End-to-end counterfactual training data generation (§7.1.1, steps d-f).

    High-level workflow:
        1. Generate reference video via full-compute backbone
        2. Extract tubes and causal sub-graphs
        3. For each (tube, timestep, action) triplet:
           - Apply action to that tube, continue denoising
           - Decode → counterfactual video
           - Compute damage via metric extractor
           - Create COCFTrainingSample with damage label
    """

    def __init__(
        self,
        metric_extractor: MetricExtractor,
        strength_feature_builder: CausalStrengthFeatureBuilder,
        damage_computer: CounterfactualDamageComputer,
        sampling_config: Optional[StratifiedSamplingConfig] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.metric_extractor = metric_extractor
        self.strength_builder = strength_feature_builder
        self.damage_computer = damage_computer
        self.sampling_config = sampling_config or StratifiedSamplingConfig()
        self.device = device

        self.sampler = StratifiedSampler(self.sampling_config, device)
        self.interpolator = DamageLabelInterpolator(self.sampling_config.interpolation_interval)

    def generate_batch(
        self,
        video_full: Tensor,  # [B, F, 3, H, W]
        prompts: List[str],
        tubes_by_step: Dict[int, List[SemanticTube]],  # timestep → tubes at that step
        z_trajectory: Dict[int, Tensor],  # timestep → [B, N, d] latent
        strength_features_by_step: Dict[int, StrengthFeatures],  # cached features
        backbone_adapter,  # For counterfactual interventions
        num_total_steps: int = 50,
        max_samples_per_video: int = 30,
    ) -> List[COCFTrainingSample]:
        """Generate counterfactual training samples from a batch of full-compute videos.

        This is the main entry point for Stage A data generation.

        Args:
            video_full: Full-compute reference videos.
            prompts: Text prompts (batch).
            tubes_by_step: Pre-extracted semantic tubes at representative steps.
            z_trajectory: Cached latent trajectory from full-compute.
            strength_features_by_step: Pre-computed strength features.
            backbone_adapter: Model for running counterfactual denoising.
            num_total_steps: Total denoising steps (T).
            max_samples_per_video: Cap on samples to generate.

        Returns:
            List of COCFTrainingSample training labels.
        """
        samples = []
        batch_size = video_full.shape[0]

        # Per-video loop (stratified sampling applied independently)
        for b in range(batch_size):
            prompt = prompts[b] if b < len(prompts) else ""
            video_b = video_full[b]  # [F, 3, H, W]

            _log.info(f"Generating counterfactual data for batch {b}/{batch_size}")

            # Sample representative (timestep, tube_id, action) triplets
            sampled_steps = self.sampler.sample_timesteps(num_total_steps)
            samples_this_video = []

            for step, stratum in sampled_steps:
                if len(samples_this_video) >= max_samples_per_video:
                    break

                tubes = tubes_by_step.get(step, [])
                if not tubes:
                    continue

                tube_indices = self.sampler.sample_tubes(tubes)
                actions = self.sampler.sample_actions(num_samples=3)

                for tube_idx in tube_indices:
                    for action in actions:
                        if len(samples_this_video) >= max_samples_per_video:
                            break

                        # Run counterfactual intervention
                        damage_label, per_axis = self._run_counterfactual_intervention(
                            video_full=video_b,
                            prompt=prompt,
                            z_init=z_trajectory.get(step, None),
                            tube=tubes[tube_idx],
                            action=Action(action),
                            backbone_adapter=backbone_adapter,
                        )

                        if damage_label is None:
                            continue

                        # Build training sample
                        strength_feats = strength_features_by_step.get(step, None)
                        tube_state = self._extract_tube_state(tubes[tube_idx], step)

                        sample = COCFTrainingSample(
                            tube_features=tube_state,
                            timestep=step,
                            action=action,
                            damage_label=damage_label,
                            damage_per_axis=per_axis,
                            prompt=prompt,
                            tube_id=tube_idx,
                            scene_type="generic",  # TODO: infer from prompt/content
                            interaction_density=self._compute_interaction_density(
                                tubes, tube_idx
                            ),
                            strength_level=int(self._estimate_strength_level(
                                tubes[tube_idx], strength_feats
                            )),
                            tube_pixels=self._count_tube_pixels(tubes[tube_idx]),
                        )

                        samples_this_video.append(sample)
                        samples.append(sample)

            _log.info(f"  Generated {len(samples_this_video)} samples from batch {b}")

        return samples

    def _run_counterfactual_intervention(
        self,
        video_full: Tensor,  # [F, 3, H, W]
        prompt: str,
        z_init: Optional[Tensor],  # [N, d] initial latent
        tube: SemanticTube,
        action: Action,
        backbone_adapter,
        num_cf_steps: int = 10,  # run CF for 10 steps
    ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        """Execute a counterfactual intervention: apply action to tube, continue denoising.

        Real implementation must:
            1. Load z_init
            2. Apply action to tube (skip/lowfreq/interp/anchor)
            3. Continue for num_cf_steps
            4. Decode & return video_cf
            5. Compute damage vs video_full

        Until that rollout exists, this MUST NOT fabricate a label: returning a
        zero/placeholder damage vector here would silently write millions of
        invalid samples to the Stage-A dataset and poison all downstream training
        with no error. Fail loudly instead.
        """
        raise NotImplementedError(
            "Counterfactual intervention rollout is not implemented yet. "
            "Implement the z_init replay + action application + decode + damage "
            "computation before running Stage-A data generation; do not return a "
            "placeholder label (it silently corrupts the training set)."
        )

    def _extract_tube_state(self, tube: SemanticTube, timestep: int) -> Tensor:
        """Extract the 7-dim tube state for the predictor input.

        Real impl: geometry (centroid, size), motion, identity score, etc. A random
        placeholder here would be written as a real feature vector and corrupt the
        dataset silently, so this fails loudly until implemented.
        """
        raise NotImplementedError(
            "Tube-state feature extraction is not implemented yet. Returning random "
            "features here would silently corrupt the Stage-A dataset."
        )

    def _compute_interaction_density(self, tubes: List[SemanticTube], tube_idx: int) -> float:
        """Compute how much this tube interacts with others (0-1)."""
        if not tubes or tube_idx >= len(tubes):
            return 0.0
        # Stub: return count of overlapping tubes
        return min(1.0, len(tubes) / 10.0)

    def _estimate_strength_level(
        self, tube: SemanticTube, strength_feats: Optional[StrengthFeatures]
    ) -> StrengthLevel:
        """Estimate causal strength level (HIGH/MID/LOW) for the tube."""
        # Stub: use strength features if available, else return MID
        return StrengthLevel.MID

    def _count_tube_pixels(self, tube: SemanticTube) -> int:
        """Count total pixels spanned by the tube across its lifespan."""
        if not tube or not tube.frames:
            return 0
        return sum(len(mask.nonzero()) for mask in tube.frames)


# ============================================================================= #
# Utilities for data caching & batch processing
# ============================================================================= #


class CounterfactualDataCache:
    """Caches full-compute outputs to amortize cost across multiple counterfactuals.

    Stores:
        - Latent trajectory z_t for each timestep
        - Attention cache KV for each step
        - Semantic tube annotations
        - Strength features
    """

    def __init__(self, max_cache_mb: int = 1024):
        self.max_cache_mb = max_cache_mb
        self.z_trajectory: Dict[int, Tensor] = {}
        self.kv_cache: Dict[int, Dict[str, Tensor]] = {}
        self.tubes_by_step: Dict[int, List[SemanticTube]] = {}
        self.strength_features: Dict[int, StrengthFeatures] = {}

    def cache_latent(self, step: int, z: Tensor) -> None:
        """Store latent for the given timestep (move to CPU if full)."""
        self.z_trajectory[step] = z.cpu()

    def get_latent(self, step: int, device: torch.device) -> Optional[Tensor]:
        """Retrieve latent, moving back to device."""
        if step in self.z_trajectory:
            return self.z_trajectory[step].to(device)
        return None

    def cache_tubes(self, step: int, tubes: List[SemanticTube]) -> None:
        """Store tube annotations."""
        self.tubes_by_step[step] = tubes

    def cache_strength_features(self, step: int, feats: StrengthFeatures) -> None:
        """Store pre-computed strength features."""
        self.strength_features[step] = feats

    def clear(self) -> None:
        """Clear all cached data."""
        self.z_trajectory.clear()
        self.kv_cache.clear()
        self.tubes_by_step.clear()
        self.strength_features.clear()

    def memory_mb(self) -> float:
        """Estimate current memory usage in MB."""
        total_bytes = 0
        for z in self.z_trajectory.values():
            total_bytes += z.element_size() * z.nelement()
        return total_bytes / (1024 * 1024)
