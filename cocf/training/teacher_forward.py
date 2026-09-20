"""Teacher full-compute forward producing a :class:`TeacherTrajectory`.

Stage A turns each kept OpenVid caption into the reference full-compute trajectory the
single-hop counterfactual generator (:class:`cocf.lcocf.data.COCFDataGenerator`) runs
interventions against: run the frozen backbone for the full denoising schedule with no
acceleration (caching ``z_t`` at representative steps and decoding ``Y_full``), then
build the semantic tubes, their states, causal-strength features and per-tube CLIP
visual embeds. This runner is the FULL-mode twin of
:class:`cocf.engine.inference.InferenceEngine`, reusing the same backbone-agnostic
operations, tube builder and strength-feature builder so the teacher features carry no
train/serve skew. The pass is label-only and runs under ``inference_mode``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from cocf.backbones.base import BackboneAdapter, TextConditioning, sigma_from_step
from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.memory import teacher_forward
from cocf.common.types import SemanticTube, TokenGrid, TubeState
from cocf.lcocf.data import TeacherTrajectory, tube_clip_embed
from cocf.lcocf.strength import StrengthFeatures

Tensor = torch.Tensor
_log = get_logger(__name__)


def _to_fchw(video: Tensor) -> Tensor:
    """``[B, 3, F, H, W]`` (or ``[3, F, H, W]``) → ``[F, 3, H, W]``.

    Layout only; the value range is settled by ``decode_to_unit``. Mirrors
    :func:`cocf.lcocf.data._frames_fchw` so the stored reference video has the same
    frame layout the counterfactual rollout produces.
    """
    v = video[0] if video.dim() == 5 else video
    return v.permute(1, 0, 2, 3).contiguous()


def _frames_per_latent_slot(video_fchw: Tensor, grid_t: int) -> Tensor:
    """Subsample a decoded clip to exactly ``grid_t`` frames (one per latent slot).

    The tube builder segments pixels and needs one RGB frame per latent-temporal slot.
    A causal-temporal VAE expands slots into more pixel frames, so we pick ``grid_t``
    evenly-spaced frames (a no-op on the mock).
    """
    f = video_fchw.shape[0]
    if f == grid_t:
        return video_fchw
    sel = torch.linspace(0, f - 1, grid_t, device=video_fchw.device).round().long()
    return video_fchw.index_select(0, sel)


def _noise_latent(z0: Tensor, sigma: float, generator: Optional[torch.Generator]) -> Tensor:
    """Forward-diffuse a clean latent ``z0`` to flow-matching level ``σ``.

    Interpolates the clean sample and Gaussian noise as ``z_σ = (1−σ)·z0 + σ·ε``. Used
    only when Stage A anchors the teacher trajectory on a real clip, where the
    representative ``z_t`` are produced by noising the encoded ``z0``.
    """
    eps = torch.randn(
        z0.shape, generator=generator,
        device=generator.device if generator is not None else z0.device,
        dtype=z0.dtype,
    )
    return (1.0 - sigma) * z0 + sigma * eps.to(z0.device)


@dataclass
class TeacherForwardConfig:
    """Knobs of the teacher full-compute pass (defaults from :class:`Config`)."""

    num_inference_steps: int = 30
    # representative ``step_frac = t/T`` values whose ``z_t`` is cached for the
    # single-hop intervention; the rest are filled by label interpolation downstream.
    representative_step_fracs: tuple = (0.9, 0.7, 0.5, 0.3, 0.1)
    num_frames: int = 49
    height: int = 480
    width: int = 832

    @classmethod
    def from_config(cls, config: Config) -> "TeacherForwardConfig":
        return cls(
            num_inference_steps=config.teacher.num_inference_steps,
            representative_step_fracs=tuple(config.teacher.representative_step_fracs),
            num_frames=config.data.num_frames,
            height=config.data.height,
            width=config.data.width,
        )


class TeacherForwardRunner:
    """Builds one :class:`TeacherTrajectory` per caption with the frozen backbone.

    Holds no per-video state, so a single instance processes the whole dataset (and
    can be sharded across devices — Stage A is embarrassingly parallel over clips).
    """

    def __init__(self, accelerator, cfg: TeacherForwardConfig,
                 device: torch.device = torch.device("cpu")) -> None:
        self.acc = accelerator
        self.backbone: BackboneAdapter = accelerator.backbone
        self.tube_builder = accelerator.tube_builder
        self.perception = accelerator.perception
        self.cfg = cfg
        self.device = device

    # ------------------------------------------------------------------ #
    # representative step selection
    # ------------------------------------------------------------------ #

    def representative_step_indices(self) -> List[int]:
        """Forward step indices (0 = t=T) whose ``z_t`` to cache (early/mid/late).

        A ``step_frac = t/T`` maps to ``step_idx = T − round(frac·T)``; deduped and
        clamped to ``[0, T−1]``.
        """
        T = self.cfg.num_inference_steps
        idxs = set()
        for frac in self.cfg.representative_step_fracs:
            t = int(round(float(frac) * T))
            idxs.add(min(max(T - t, 0), T - 1))
        return sorted(idxs)

    # ------------------------------------------------------------------ #
    # full-compute denoise (shared by Stage A teacher gen & Stage C baseline)
    # ------------------------------------------------------------------ #

    def full_denoise(
        self,
        z_init: Tensor,
        cond: TextConditioning,
        grid: TokenGrid,
        *,
        cache_steps: Sequence[int] = (),
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        """Run the full, un-accelerated schedule ``z_T → z_0`` (no per-tube actions).

        Returns ``(z0, {step_idx: z_t})`` for indices in ``cache_steps``. Always
        label-only, so it is the shared full-compute reference for Stage A and the
        Stage-C full-budget baseline.
        """
        bb = self.backbone
        T = self.cfg.num_inference_steps
        want = set(cache_steps)
        z_by_step: Dict[int, Tensor] = {}
        # ``inference_mode`` only while the backbone is label-only. Stage C calls this
        # for its Y_full reference inside ``grad_mode(True)``, where an inference tensor
        # could not be saved for backward; ``no_grad`` is graph-free and safe there.
        ctx = teacher_forward() if not bb.grad_enabled else torch.no_grad()
        with ctx:
            z = z_init
            cache = None
            for step_idx in range(T):
                t = T - step_idx
                if step_idx in want:
                    z_by_step[step_idx] = z.clone()
                t_now = torch.full((z.shape[0],), sigma_from_step(t, T), device=z.device)
                t_next = torch.full((z.shape[0],), sigma_from_step(t - 1, T), device=z.device)
                out = bb.full_transition(z, t_now, t_next, cond, grid=grid, cache=cache)
                z = out.model_output
                cache = out.cache
        return z, z_by_step

    # ------------------------------------------------------------------ #
    # the full-compute teacher forward
    # ------------------------------------------------------------------ #

    def run(
        self,
        video_id: str,
        prompt: str,
        scene_type: str = "dynamic",
        *,
        z_init: Optional[Tensor] = None,
        cond: Optional[TextConditioning] = None,
        grid: Optional[TokenGrid] = None,
        video_frames: Optional[Tensor] = None,
    ) -> Optional[TeacherTrajectory]:
        """Run the full-compute teacher forward for one caption.

        Returns the assembled :class:`TeacherTrajectory`, or ``None`` when no tube could
        be segmented on the reference video. ``z_init`` / ``cond`` / ``grid`` are
        sampled internally when omitted (Stage A's path); Stage C passes them in so the
        baseline shares the same initial noise as the accelerated run. ``video_frames``
        (``[F, 3, H, W]`` in ``[-1, 1]``) anchors the trajectory on a real clip instead
        of a text-to-video generation.
        """
        bb = self.backbone
        rep_steps = self.representative_step_indices()

        with teacher_forward():
            if cond is None:
                cond = bb.encode_text([prompt]).to(self.device)

            if video_frames is not None:
                # real-clip anchor: encode the mp4, then noise z0 to the rep steps.
                # The token grid is derived from the encoded latent (not the pixel
                # dims); ``video_frames`` is [F,3,H,W], the VAE wants [B,3,F,H,W].
                video_bcfhw = video_frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                lat = bb.encode_video(video_bcfhw)  # [1,C,T,H,W]
                p_t, p_h, p_w = getattr(bb, "patch", (1, 1, 1))
                grid = TokenGrid(
                    t=max(1, lat.shape[2] // p_t),
                    h=max(1, lat.shape[3] // p_h),
                    w=max(1, lat.shape[4] // p_w),
                )
                z0 = bb.to_tokens(lat)
                # Stable per-clip seed: sha1 on a CPU generator (device-independent).
                clip_seed = int(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:8], 16)
                gen = torch.Generator().manual_seed(clip_seed)
                T = self.cfg.num_inference_steps
                z_by_step = {
                    step_idx: _noise_latent(z0, sigma_from_step(T - step_idx, T), gen)
                    for step_idx in rep_steps
                }
            else:
                if grid is None:
                    grid = bb.token_grid(self.cfg.num_frames, self.cfg.height, self.cfg.width)
                if z_init is None:
                    z_init = bb.initial_latent(grid, batch=1, device=self.device)
                # full, un-accelerated denoise; cache z_t at the representative steps.
                z0, z_by_step = self.full_denoise(z_init, cond, grid, cache_steps=rep_steps)

            # decode the reference video Y_full (full frame layout, as the rollout uses)
            video_full = _to_fchw(bb.decode_to_unit(bb.to_grid(z0, grid)))  # [F,3,Hp,Wp]

            # build tubes + states + (s_E,s_A,s_T) + visual embeds
            frames_for_tubes = _frames_per_latent_slot(video_full, grid.t)
            tubes, states, _flows = self.tube_builder.build_with_states(
                frames_for_tubes, grid, prompt
            )
            if not tubes:
                _log.debug("teacher forward for %s yielded no tubes; skipping", video_id)
                return None

            subgraph = self.acc.lcocf.parse(prompt)
            strength_feats: Dict[int, StrengthFeatures] = self.acc.lcocf.strength_features(
                tubes, states, subgraph
            )
            tube_visual_embed_full: Dict[int, Tensor] = {
                tube.tube_id: tube_clip_embed(video_full, tube, grid, self.perception)
                for tube in tubes
            }

        return TeacherTrajectory(
            video_id=video_id,
            prompt=prompt,
            scene_type=scene_type,
            video_full=video_full,
            grid=grid,
            cond=cond,
            z_by_step=z_by_step,
            tubes=tubes,
            tube_states=states,
            strength_feats=strength_feats,
            tube_visual_embed_full=tube_visual_embed_full,
            num_total_steps=self.cfg.num_inference_steps,
            text_embed=cond.embeds[0],
            z_init=z_init,
        )
