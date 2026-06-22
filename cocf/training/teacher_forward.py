"""Teacher full-compute forward → :class:`TeacherTrajectory` (§1.3–§1.4).

Stage A turns each kept OpenVid caption into the reference *full-compute* trajectory
the single-hop counterfactual generator (:class:`cocf.lcocf.data.COCFDataGenerator`)
runs interventions against. That trajectory is exactly the design's §1.3 "全量基准
数据生成" plus the §1.4 "局部因果特征与语义管提取":

    §1.3  run the frozen backbone for the *full* denoising schedule with **no**
          acceleration, caching the latent ``z_t`` at the representative steps
          (early/mid/late) and decoding the reference video ``Y_full``;
    §1.4  build the semantic tubes once on ``Y_full`` (SAM→affinity→Hungarian via
          :meth:`TubeBuilder.build_with_states`), encode their 7-dim states, the
          ``(s_E, s_A, s_T)`` causal-strength features and the per-tube CLIP visual
          embeds that feed CMSC.

This runner is the FULL-mode twin of :class:`cocf.engine.inference.InferenceEngine`:
it reuses the *same* backbone-agnostic operations (``encode_text`` / ``initial_latent``
/ ``full_transition`` / ``decode_latent`` / ``to_grid``) and the *same* tube builder
and strength-feature builder the engine and the data pipeline use, so the teacher
features carry no train/serve skew. The whole pass is label-only, so it runs under
:func:`cocf.common.memory.teacher_forward` (``inference_mode``) — no autograd graph
is ever built, which is both correct and the cheapest path on VRAM for the full
backbone (user requirement #1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from cocf.backbones.base import BackboneAdapter, TextConditioning
from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.memory import teacher_forward
from cocf.common.types import SemanticTube, TokenGrid, TubeState
from cocf.lcocf.data import TeacherTrajectory, tube_clip_embed
from cocf.lcocf.strength import StrengthFeatures

Tensor = torch.Tensor
_log = get_logger(__name__)


def _to_fchw(video: Tensor) -> Tensor:
    """``[B, 3, F, H, W]`` (or ``[3, F, H, W]``) → ``[F, 3, H, W]`` in [0, 1].

    Mirrors :func:`cocf.lcocf.data._frames_fchw` so the reference video stored here
    has the *same* frame layout the counterfactual rollout produces when it decodes
    ``z_0`` — the damage computer compares the two frame-for-frame.
    """
    v = video[0] if video.dim() == 5 else video
    return v.permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)


def _frames_per_latent_slot(video_fchw: Tensor, grid_t: int) -> Tensor:
    """Subsample a decoded clip to exactly ``grid_t`` frames (one per latent slot).

    The tube builder segments *pixels* and needs one RGB frame per latent-temporal
    slot (``F == grid.t``). A real causal-temporal VAE expands ``grid.t`` slots into
    more pixel frames, so we pick ``grid.t`` evenly-spaced frames (no-op on the mock,
    whose decode already yields ``F == grid.t``). Identical logic to the engine's
    :meth:`InferenceEngine._decode_preview_frames` (no train/serve skew).
    """
    f = video_fchw.shape[0]
    if f == grid_t:
        return video_fchw
    sel = torch.linspace(0, f - 1, grid_t, device=video_fchw.device).round().long()
    return video_fchw.index_select(0, sel)


@dataclass
class TeacherForwardConfig:
    """Knobs of the §1.3 teacher full-compute pass (defaults from :class:`Config`)."""

    num_inference_steps: int = 30
    # representative ``step_frac = t/T`` values whose ``z_t`` is cached for the §1.5
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
        """Forward step indices (0 = t=T) whose ``z_t`` to cache (§1.3 early/mid/late).

        A ``step_frac = t/T`` maps to ``step_idx = T − round(frac·T)`` because the
        loop counts down ``t = T − step_idx``. Deduped and clamped to ``[0, T−1]``.
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

        Returns ``(z0, {step_idx: z_t})`` where the cached latents are those whose
        forward index is in ``cache_steps``. Always label-only (``teacher_forward``),
        so it is the shared full-compute reference for Stage A (§1.3) and the Stage-C
        full-budget baseline — identical maths, no train/serve skew.
        """
        bb = self.backbone
        T = self.cfg.num_inference_steps
        want = set(cache_steps)
        z_by_step: Dict[int, Tensor] = {}
        with teacher_forward():
            z = z_init
            cache = None
            for step_idx in range(T):
                t = T - step_idx
                if step_idx in want:
                    z_by_step[step_idx] = z.clone()
                t_now = torch.full((z.shape[0],), float(t), device=z.device)
                t_next = torch.full((z.shape[0],), float(t - 1), device=z.device)
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
    ) -> Optional[TeacherTrajectory]:
        """Run the full-compute teacher forward for one caption.

        Returns the assembled :class:`TeacherTrajectory`, or ``None`` when no tube
        could be segmented on the reference video (a degenerate clip with no causal
        signal — there is nothing to intervene on, so it yields no samples).

        ``z_init`` / ``cond`` / ``grid`` are sampled internally when omitted (Stage A's
        path). Stage C passes them in so the full-compute baseline shares the *same*
        initial noise as the accelerated engine run — without that, a damage compared
        against this ``Y_full`` would measure the noise difference, not the effect of
        acceleration (§4.2 主损失).
        """
        bb = self.backbone
        rep_steps = self.representative_step_indices()

        with teacher_forward():
            if grid is None:
                grid = bb.token_grid(self.cfg.num_frames, self.cfg.height, self.cfg.width)
            if cond is None:
                cond = bb.encode_text([prompt]).to(self.device)
            if z_init is None:
                z_init = bb.initial_latent(grid, batch=1, device=self.device)

            # --- §1.3: full, un-accelerated denoise; cache z_t at rep. steps --- #
            z0, z_by_step = self.full_denoise(z_init, cond, grid, cache_steps=rep_steps)

            # decode the reference video Y_full (full frame layout, as the rollout uses)
            video_full = _to_fchw(bb.decode_latent(bb.to_grid(z0, grid)))  # [F, 3, Hp, Wp]

            # --- §1.4: build tubes + states + (s_E,s_A,s_T) + visual embeds --- #
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
        )
