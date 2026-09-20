"""Stage-C end-to-end fine-tune losses: ``L_total`` over one accelerated render.

The pure realisation of the Stage-C objective (no IO, no optimiser, no loop), keeping
:mod:`cocf.training.stage_c_finetune` to orchestration::

    L_total = λ_pixel·L_pixel + λ_quality·L_cmsc              (main loss vs Y_full)
            + λ_sta·L_tube + λ_cert·L_cert + λ_cost·L_budget  (regularisers, reuse B)

The main quality loss is :func:`pixel_quality_loss` plus :func:`cmsc_quality_loss` over
a pair of :class:`~cocf.cmsc.losses.CMSCObservation`s built by
:func:`build_cmsc_observation`. :func:`stage_c_regularizers` recomputes the predictor
μ/σ with the differentiable causal strength and reuses the Stage-B building blocks so
the scheduling logic stays calibrated.

Gradient targets: the pixel term trains the residual-repair net (+ optional LoRA); the
CMSC term trains the text-tube alignment; the regularisers train the damage predictor,
strength weights and certificate coefficients. The frozen backbone is never optimised.
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
from cocf.lcocf.data import tube_clip_embed, tube_pixel_mask
from cocf.lcocf.predictor import build_predictor_input_batch
from cocf.training.stage_b_losses import (
    action_probs,
    batch_float,
    budget_penalty,
    tube_temporal_smoothness,
)

Tensor = torch.Tensor

# tube-state vector indices feeding the certificate; same source Stage B uses.
_BOUNDARY_IDX = TUBE_STATE_FIELDS.index("boundary_uncertainty")
_AGE_IDX = TUBE_STATE_FIELDS.index("anchor_age")


# --------------------------------------------------------------------------- #
# main quality loss — Y_acc vs Y_full
# --------------------------------------------------------------------------- #


def pixel_quality_loss(y_acc: Tensor, y_full: Tensor) -> Tensor:
    """Frame-aligned L1 between the accelerated and full-compute renders.

    Both are ``[F, 3, H, W]`` in [0,1]; frame counts are aligned to the common
    minimum. Differentiable through ``y_acc``.
    """
    f = min(y_acc.shape[0], y_full.shape[0])
    if f == 0:
        return y_acc.new_zeros(())
    return F.l1_loss(y_acc[:f].float(), y_full[:f].float().to(y_acc.device))


def _tube_centroid(tube: SemanticTube) -> Tuple[float, float]:
    """Mean ``(y, x)`` of a tube's latent masks, averaged over the frames it spans."""
    ys = xs = 0.0
    n = 0
    for mask in tube.masks_by_frame.values():
        if mask is None:
            continue
        idx = mask.nonzero()
        if idx.numel() == 0:
            continue
        ys += float(idx[:, 0].float().mean())
        xs += float(idx[:, 1].float().mean())
        n += 1
    return (ys / n, xs / n) if n else (0.0, 0.0)


def build_cmsc_observation(
    metric_extractor: MetricExtractor,
    perception,
    video_fchw: Tensor,
    prompt: str,
    tubes: Sequence[SemanticTube],
    grid: TokenGrid,
    text_embed: Tensor,
    *,
    differentiable: bool = False,
    frame_span=None,
    full_frame_count=None,
) -> CMSCObservation:
    """Assemble a :class:`CMSCObservation` from a rendered clip.

    Populates global ``video`` features, per-tube CLIP ``tube_embeds``, per-tube DINO
    ``tube_identity`` and latent-mask ``tube_centroid``. ``tube_boundary`` is left
    empty (no boundary descriptor exists, so ``L_bnd`` stays zero).

    ``differentiable`` must be set on the accelerated branch so the CMSC loss has an
    autograd graph; both branches are extracted with ``offload=False`` so they stay on
    the render's device for the element-wise comparison.
    """
    tube_masks = {t.tube_id: tube_pixel_mask(video_fchw, t, grid,
                  frame_span=frame_span, full_frame_count=full_frame_count) for t in tubes}
    feats = metric_extractor.extract(
        video_fchw, prompt, differentiable=differentiable, tube_masks=tube_masks,
        # Both observations are compared element-wise on the render's device, so
        # neither may be parked on CPU.
        offload=False,
    )
    tube_embeds = {
        t.tube_id: tube_clip_embed(video_fchw, t, grid, perception,
            differentiable=differentiable, frame_span=frame_span,
            full_frame_count=full_frame_count) for t in tubes
    }
    tube_identity = {
        tid: f.mean(0) for tid, f in feats.tube_dino.items() if f.numel()
    }
    return CMSCObservation(
        video=feats,
        text_embeds=text_embed,
        tube_embeds=tube_embeds,
        tube_identity=tube_identity,
        tube_centroid={t.tube_id: _tube_centroid(t) for t in tubes},
    )


def cmsc_quality_loss(
    cmsc_loss, full_obs: CMSCObservation, accel_obs: CMSCObservation
) -> Tuple[Tensor, Dict[str, float]]:
    """The multi-dimensional conservation loss between the full & accel renders.

    Thin wrapper over :meth:`CMSCLoss.forward`; returns ``(scalar, per-term components)``.
    """
    return cmsc_loss(full_obs, accel_obs)


# --------------------------------------------------------------------------- #
# per-(tube, step) records collected during the forward → a pseudo-batch
# --------------------------------------------------------------------------- #


@dataclass
class StepRecord:
    """One (tube, step) observation from the Stage-C accelerated forward.

    Carries the inputs the regularisers re-run the predictor and certificate on; no
    μ/σ is stored because :func:`stage_c_regularizers` recomputes them with the
    grad-on causal strength.
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
    skip_residual: float = 0.0   # δ measured by the transition (certificate λ_res)
    local_cmsc: float = 0.0      # neutral-centered alignment risk (certificate λ_cmsc)


def collate_step_records(records: Sequence[StepRecord]) -> Dict[str, object]:
    """Stack per-(tube,step) records into a pseudo-batch (keys ≡ Stage-B batch fields).

    Feeds :func:`stage_c_regularizers` and the reused Stage-B helpers. Empty input
    yields ``{}``.
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
        "skip_residual": torch.tensor(
            [r.skip_residual for r in records], dtype=torch.float32
        ),
        "local_cmsc": torch.tensor(
            [r.local_cmsc for r in records], dtype=torch.float32
        ),
        "video_id": [r.video_id for r in records],
    }


# --------------------------------------------------------------------------- #
# regularisers — reuse the Stage-B terms
# --------------------------------------------------------------------------- #


def stage_c_regularizers(
    accelerator,
    batch: Mapping[str, object],
    measured_damage: Tensor,
    *,
    training_cfg: Optional[TrainingConfig] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Stage-B-style scheduling regularisers over the collected forward records.

    Recomputes the predictor ``μ/σ`` with the differentiable causal strength, then
    assembles ``λ_sta·L_tube + λ_cert·L_cert + λ_cost·L_budget``. ``measured_damage``
    is the realised damage of ``Y_acc`` vs ``Y_full`` (the certificate's calibration
    target). Returns ``(reg_total, comps)``; empty batch yields ``(0, zeros)``.
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
        accelerator.allocator.action_cost, device=device, dtype=torch.float32
    )
    l_budget = budget_penalty(probs, action_cost, budget)

    # L_cert — certificate calibrated to the realised render damage. ``residual`` and
    # ``local_cmsc`` are the values the engine measured this step (via ``record_sink``).
    idx = actions.clamp(0, pred.mu.shape[-1] - 1).unsqueeze(-1)
    mu_a = pred.mu.gather(-1, idx).squeeze(-1)                        # [M]
    sigma_a = pred.sigma.gather(-1, idx).squeeze(-1)                  # [M]
    e_cert = accelerator.raec.certificate.value(
        mu_a, sigma_a,
        residual=batch_float(batch, "skip_residual", mu_a),
        boundary=tube_features[:, _BOUNDARY_IDX],
        anchor_age=tube_features[:, _AGE_IDX],
        local_cmsc=batch_float(batch, "local_cmsc", mu_a),
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
