"""Stage-C end-to-end fine-tune losses — ``L_total`` over one accelerated render (§4.2).

Stage C runs the *full accelerated pipeline* on a real caption and tunes the
differentiable plugins (and an optional LoRA) against the full-compute baseline
``Y_full``. Its objective mirrors the design's §4.2 loss split::

    L_total = λ_pixel·L_pixel + λ_quality·L_cmsc          ← 主损失 (vs Y_full)
            + λ_sta·L_tube + λ_cert·L_cert + λ_cost·L_budget  ← 正则 (reuse Stage B)

This module is the *pure* realisation of that objective — no IO, no optimiser, no
loop — so it is unit-testable in isolation and keeps :mod:`cocf.training.stage_c_finetune`
to pure orchestration (the same split Stage B uses between ``stage_b_joint`` and
``stage_b_losses``). Two groups of helpers:

    main quality   :func:`pixel_quality_loss` (像素) and :func:`cmsc_quality_loss`
                   over a pair of :class:`~cocf.cmsc.losses.CMSCObservation`s
                   (语义 / 文本对齐), built by :func:`build_cmsc_observation` from the
                   shared :class:`MetricExtractor` + per-tube CLIP embeds. The single
                   differentiable CMSC objective serves *both* the §4.2 main-loss
                   "语义/文本对齐" role and the §4.2 CMSC-regulariser role (not double
                   counted).
    regularizes   :func:`stage_c_regularizers` recomputes the predictor μ/σ with the
                   *differentiable* causal strength (exactly as Stage B does) over the
                   per-step (tube,step) records collected during the forward, then
                   reuses the Stage-B building blocks verbatim — :func:`action_probs`,
                   :func:`tube_temporal_smoothness`, :func:`budget_penalty` and the RAEC
                   certificate hinge — so the scheduling logic is kept calibrated and
                   does not drift (§4.2 "保证调度逻辑不偏移").

Gradient targets reached (§4.2 梯度回传范围): the pixel term trains the residual-repair
net (+ optional LoRA, both on the latent→render path); the CMSC term trains the text↔tube
alignment; the regularisers train the damage predictor, the three strength weights and the
certificate coefficients. The frozen backbone is never on the optimised path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from cocf.cmsc.losses import CMSCObservation
from cocf.common.config import TrainingConfig
from cocf.common.types import TUBE_STATE_FIELDS, Action, SemanticTube, TokenGrid
from cocf.lcocf.damage import MetricExtractor
from cocf.lcocf.data import tube_clip_embed
from cocf.lcocf.predictor import build_predictor_input_batch
from cocf.training.stage_b_losses import (
    action_probs,
    budget_penalty,
    tube_temporal_smoothness,
)

Tensor = torch.Tensor

# tube-state vector indices feeding the certificate (the 7-dim s_{k,t}); same source
# Stage B uses, so the certificate sees identical inputs at both stages (no skew).
_BOUNDARY_IDX = TUBE_STATE_FIELDS.index("boundary_uncertainty")
_AGE_IDX = TUBE_STATE_FIELDS.index("anchor_age")


# --------------------------------------------------------------------------- #
# main quality loss — Y_acc vs Y_full (§4.2 主损失)
# --------------------------------------------------------------------------- #


def pixel_quality_loss(y_acc: Tensor, y_full: Tensor) -> Tensor:
    """Frame-aligned L1 between the accelerated and full-compute renders (§4.2 像素).

    Both are ``[F, 3, H, W]`` in [0,1]. Frame counts are aligned to the common
    minimum (the two decode paths share a grid, so they normally match; this guards
    a real causal-temporal VAE that expands slots differently). Differentiable
    through ``y_acc`` — the gradient path to the residual-repair net and LoRA.
    """
    f = min(y_acc.shape[0], y_full.shape[0])
    if f == 0:
        return y_acc.new_zeros(())
    return F.l1_loss(y_acc[:f].float(), y_full[:f].float().to(y_acc.device))


def build_cmsc_observation(
    metric_extractor: MetricExtractor,
    perception,
    video_fchw: Tensor,
    prompt: str,
    tubes: Sequence[SemanticTube],
    grid: TokenGrid,
    text_embed: Tensor,
) -> CMSCObservation:
    """Assemble a :class:`CMSCObservation` from a rendered clip (§6 features).

    Bundles the global DINO/CLIP/RAFT/OCR :class:`VideoFeatures` (from the injected
    :class:`MetricExtractor`) with the prompt token sequence and the per-tube CLIP
    visual embeds (via the shared :func:`~cocf.lcocf.data.tube_clip_embed`, so the
    embeds match what Stage A/serve produce). The perceptual features are detached
    references by construction; only the alignment projection is differentiable.
    """
    feats = metric_extractor.extract(video_fchw, prompt)
    tube_embeds = {
        t.tube_id: tube_clip_embed(video_fchw, t, grid, perception) for t in tubes
    }
    return CMSCObservation(video=feats, text_embeds=text_embed, tube_embeds=tube_embeds)


def cmsc_quality_loss(
    cmsc_loss, full_obs: CMSCObservation, accel_obs: CMSCObservation
) -> Tuple[Tensor, Dict[str, float]]:
    """The §6 multi-dimensional conservation loss between the full & accel renders.

    Thin wrapper over :meth:`CMSCLoss.forward` so the stage stays orchestration-only;
    returns ``(scalar, per-term components)``. Trains the text↔tube alignment.
    """
    return cmsc_loss(full_obs, accel_obs)


# --------------------------------------------------------------------------- #
# per-(tube, step) records collected during the forward → a pseudo-batch
# --------------------------------------------------------------------------- #


@dataclass
class StepRecord:
    """One (tube, step) observation from the Stage-C accelerated forward.

    Carries exactly the inputs the regularisers re-run the predictor on — no μ/σ is
    stored, because :func:`stage_c_regularizers` recomputes them with the *grad-on*
    causal strength (the detached strengths used for allocation would not train the
    strength weights).
    """

    tube_features: Tensor      # [7] tube state vector s_{k,t}
    strength_features: Tensor  # [3] (s_E, s_A, s_T)
    action: int                # executed Action {FULL,LOWFREQ,INTERP,ANCHOR}
    step_frac: float           # t / T ∈ [0,1]
    budget: float              # the step's dynamic budget B_t
    tube_id: int
    timestep: int
    video_id: str
    interaction_density: float = 0.0


def collate_step_records(records: Sequence[StepRecord]) -> Dict[str, object]:
    """Stack per-(tube,step) records into a pseudo-batch (keys ≡ Stage-B batch fields).

    The result feeds :func:`stage_c_regularizers` and the reused Stage-B helpers
    (:func:`tube_temporal_smoothness` reads ``video_id``/``tube_id``/``timestep``;
    the predictor reads the stacked feature tensors). Empty input → ``{}``.
    """
    if not records:
        return {}
    return {
        "tube_features": torch.stack([r.tube_features.float() for r in records]),       # [M,7]
        "strength_features": torch.stack([r.strength_features.float() for r in records]),  # [M,3]
        "action": torch.tensor([int(r.action) for r in records], dtype=torch.long),     # [M]
        "step_frac": torch.tensor([r.step_frac for r in records], dtype=torch.float32),  # [M]
        "budget": torch.tensor([r.budget for r in records], dtype=torch.float32),        # [M]
        "tube_id": torch.tensor([r.tube_id for r in records], dtype=torch.long),         # [M]
        "timestep": torch.tensor([r.timestep for r in records], dtype=torch.long),       # [M]
        "interaction_density": torch.tensor(
            [r.interaction_density for r in records], dtype=torch.float32
        ),
        "video_id": [r.video_id for r in records],
    }


# --------------------------------------------------------------------------- #
# regularisers — reuse the Stage-B terms (§4.2 正则项)
# --------------------------------------------------------------------------- #


def stage_c_regularizers(
    accelerator,
    batch: Mapping[str, object],
    measured_damage: Tensor,
    *,
    training_cfg: Optional[TrainingConfig] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Stage-B-style scheduling regularisers over the collected forward records (§4.2).

    Recomputes the predictor ``μ/σ`` with the differentiable causal strength
    ``s = α·s_E+β·s_A+γ·s_T`` (so the strength weights train, exactly as in
    :func:`cocf.training.stage_b_losses.compute_joint_loss`), then assembles::

        λ_sta·L_tube + λ_cert·L_cert + λ_cost·L_budget

    * ``L_tube``   action-probability temporal smoothness across a tube's adjacent
                   steps (the same tube recurs every accelerated step here).
    * ``L_cert``   certificate ``E_cert(μ,σ,boundary,age)`` calibrated to upper-bound
                   the *realised* end-to-end damage of this render (detached target).
    * ``L_budget`` expected per-action compute cost vs the dynamic budget ``B_t``.

    ``measured_damage`` is the realised scalar (or ``[M]``) damage of ``Y_acc`` vs
    ``Y_full`` — the certificate's calibration target. Returns ``(reg_total, comps)``;
    ``reg_total`` carries ``grad_fn``. Empty batch → ``(0, zeros)``.
    """
    cfg = training_cfg or accelerator.config.training
    device = next(accelerator.parameters()).device
    zero = torch.zeros((), device=device)
    comps = {"tube": 0.0, "cert": 0.0, "budget": 0.0, "reg_total": 0.0}
    if not batch:
        return zero, comps

    tube_features = batch["tube_features"].to(device).float()          # [M,7]
    strength_features = batch["strength_features"].to(device).float()  # [M,3]
    actions = batch["action"].to(device).long()                       # [M]
    step_frac = batch["step_frac"].to(device).float()                 # [M]
    budget = batch["budget"].to(device).float()                       # [M]

    # differentiable predictor forward (strength flows from the strength field)
    strength = accelerator.lcocf.strength_field(strength_features)    # [M] (grad)
    pred_input = build_predictor_input_batch(
        states=tube_features,
        strength_feats=strength_features,
        strength=strength,
        budget=budget,
        step_frac=step_frac,
        step_embed_dim=accelerator.config.lcocf.predictor.context_dim,
    )
    pred = accelerator.lcocf.predictor(pred_input)                    # μ,σ: [M,A]
    probs = action_probs(pred.mu)                                     # [M,A]

    # L_tube — STA temporal smoothness (reused verbatim from Stage B)
    l_tube = tube_temporal_smoothness(accelerator, probs, batch)

    # L_budget — expected action cost vs the dynamic budget
    action_cost = torch.tensor(
        accelerator.config.allocator.action_cost, device=device, dtype=torch.float32
    )
    l_budget = budget_penalty(probs, action_cost, budget)

    # L_cert — certificate calibrated to the realised render damage
    idx = actions.clamp(0, pred.mu.shape[-1] - 1).unsqueeze(-1)
    mu_a = pred.mu.gather(-1, idx).squeeze(-1)                        # [M]
    sigma_a = pred.sigma.gather(-1, idx).squeeze(-1)                  # [M]
    zeros = mu_a.new_zeros(mu_a.shape)
    e_cert = accelerator.raec.certificate.value(
        mu_a, sigma_a,
        residual=zeros,
        boundary=tube_features[:, _BOUNDARY_IDX],
        anchor_age=tube_features[:, _AGE_IDX],
        local_cmsc=zeros,
    )
    target = measured_damage.detach().to(device).reshape(-1).float()
    if target.numel() == 1:
        target = target.expand_as(mu_a)
    l_cert = accelerator.raec.certificate.loss(e_cert, target)

    reg_total = cfg.lambda_sta * l_tube + cfg.lambda_cert * l_cert + cfg.lambda_cost * l_budget
    comps = {
        "tube": float(l_tube.detach()),
        "cert": float(l_cert.detach()),
        "budget": float(l_budget.detach()),
        "reg_total": float(reg_total.detach()),
    }
    return reg_total, comps
