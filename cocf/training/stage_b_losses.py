"""Stage-B joint loss assembly — ``L_total`` over a counterfactual batch (§4.1).

Stage B trains the four learnable plugins together on the offline counterfactual
samples, minimising the design's combined objective::

    L_total = L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc + λ_cost·L_budget

This module is the *pure* realisation of that objective from one
:func:`cocf.data.cocf_batch.collate_cocf_samples` batch — no IO, no optimiser, no
loop — so it is unit-testable in isolation and reused verbatim by Stage C's
regulariser term. Each term maps to one §4.1 module row:

    L_cocf   L-COCF predictor: Gaussian NLL of the executed action's ``(μ, σ)``
             against the realised scalar degradation label (§4.1 "高斯似然损失").
             The differentiable causal strength ``s = α·s_E+β·s_A+γ·s_T`` flows into
             the predictor input via :class:`CausalStrengthField`, so the three
             strength weights train end-to-end too.
    L_tube   STA smoothing: action-probability *temporal* consistency for the same
             tube across adjacent denoising steps present in the batch, via the
             shared :class:`~cocf.tubes.smoothing.TubeSmoothingLoss` (no math
             duplicated). The boundary term needs spatial tube adjacency, which the
             per-sample store does not carry, so Stage B exercises the temporal term
             only (the offline-data-supported half of §4.1's STA row).
    L_cert   RAEC certificate calibration: ``E_cert = μ + κσ + λ_bnd·b + λ_age·age``
             (boundary ``b`` and ``age`` read from the tube-state vector) calibrated
             to be a hinge upper bound on the true damage via
             :meth:`ErrorCertificateModule.loss`.
    L_cmsc   CMSC conservation: change in text–tube alignment between the full and
             counterfactual renders, via :meth:`CMSCLoss.alignment_conservation`.
    L_budget budget penalty: expected per-action compute cost vs the dynamic budget
             ``B_t`` (§7.3), penalising ``relu(cost − B_t)`` so the predictor prefers
             cheaper actions when the step budget is tight (§4.1 预算约束模块).

The certificate's skip-residual and per-tube local-CMSC inputs are not present in
the offline per-sample schema, so they are passed as zero here (the certificate
still calibrates on μ/σ/boundary/age, and its coefficients still receive gradient).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from cocf.common.config import TrainingConfig
from cocf.common.types import TUBE_STATE_FIELDS
from cocf.lcocf.damage import DAMAGE_DIMENSIONS, DEFAULT_DAMAGE_WEIGHTS, NUM_DAMAGE_DIMS
from cocf.lcocf.predictor import build_predictor_input_batch

Tensor = torch.Tensor

# Temperature of the soft action policy ``p(a) = softmax(−μ_a / τ)``: lower damage ⇒
# higher probability the tube takes that (cheaper) action. τ<1 sharpens the
# preference so the smoothing/budget terms see a decisive distribution rather than a
# near-uniform one when the predicted damages are all small.
ACTION_TEMP = 0.5
_EPS = 1e-6

# tube-state vector indices (the 7-dim s_{k,t}); see TUBE_STATE_FIELDS.
_BOUNDARY_IDX = TUBE_STATE_FIELDS.index("boundary_uncertainty")
_AGE_IDX = TUBE_STATE_FIELDS.index("anchor_age")


# --------------------------------------------------------------------------- #
# pure helpers (reused by Stage C)
# --------------------------------------------------------------------------- #


def damage_weight_vector(device=None) -> Tensor:
    """The ``[NUM_DAMAGE_DIMS]`` perceptual weights, in :data:`DAMAGE_DIMENSIONS` order."""
    return torch.tensor(
        [DEFAULT_DAMAGE_WEIGHTS[a] for a in DAMAGE_DIMENSIONS], device=device, dtype=torch.float32
    )


def damage_scalar_batch(damage_label: Tensor) -> Tensor:
    """Reduce a ``[B, NUM_DAMAGE_DIMS]`` damage label to a scalar target ``[B] ∈ [0,1]``."""
    w = damage_weight_vector(damage_label.device)
    return (damage_label.float() * w).sum(-1).clamp(0.0, 1.0)


def action_probs(mu: Tensor, temp: float = ACTION_TEMP) -> Tensor:
    """Soft action policy ``[B, A]`` from predicted per-action damage ``μ`` ``[B, A]``."""
    return torch.softmax(-mu / max(temp, _EPS), dim=-1)


def gaussian_nll(target: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Mean heteroscedastic Gaussian negative log-likelihood of ``target`` (all ``[B]``)."""
    var = sigma.clamp_min(_EPS).pow(2)
    return (0.5 * ((target - mu).pow(2) / var + torch.log(2 * math.pi * var))).mean()


def budget_penalty(probs: Tensor, action_cost: Tensor, budget: Tensor) -> Tensor:
    """Mean over-budget penalty ``relu(E[cost] − B_t)`` (§4.1 预算约束)."""
    expected_cost = (probs * action_cost).sum(-1)        # [B]
    return torch.relu(expected_cost - budget).mean()


# --------------------------------------------------------------------------- #
# per-sample dynamic budget B_t (§7.3) — a conditioning input, non-differentiable
# --------------------------------------------------------------------------- #


def per_sample_budget(accelerator, batch: Dict[str, object], device=None) -> Tensor:
    """Vector ``[B]`` of the dynamic step budget ``B_t`` for each sample (§7.3).

    Uses the shared :class:`~cocf.scheduler.budget.BudgetScheduler` so the budget the
    loss compares against is exactly the one the inference loop spends. Scene
    complexity is unavailable per offline sample (no parsed sub-graph), so only the
    time profile + multi-seed uncertainty + interaction-density demand signals apply.
    """
    sched = accelerator.budget_scheduler
    step_frac = batch["step_frac"]
    inter = batch.get("interaction_density")
    unc = batch.get("uncertainty")
    n = step_frac.shape[0]
    out = []
    for i in range(n):
        mean_unc = float(unc[i].mean()) if isinstance(unc, Tensor) and unc.numel() else 0.0
        idens = float(inter[i]) if isinstance(inter, Tensor) and inter.numel() else 0.0
        out.append(
            sched.budget(
                float(step_frac[i]),
                mean_uncertainty=mean_unc,
                interaction_density=idens,
            )
        )
    return torch.tensor(out, device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# individual terms
# --------------------------------------------------------------------------- #


def tube_temporal_smoothness(
    accelerator, probs: Tensor, batch: Dict[str, object]
) -> Tensor:
    """STA temporal term: mean action-prob change for a tube across adjacent steps.

    Groups the batch by ``(video_id, tube_id)``, orders each group by timestep and
    feeds consecutive-step probability pairs to the shared
    :class:`~cocf.tubes.smoothing.TubeSmoothingLoss` (single-tube dicts, so only its
    temporal term fires). Zero when no two same-tube adjacent-step samples co-occur.
    Reused by Stage C's regulariser and the Stage-B validation smoothness metric.
    """
    video_id = batch.get("video_id") or []
    tube_id = batch["tube_id"]
    timestep = batch["timestep"]
    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i in range(probs.shape[0]):
        vid = video_id[i] if i < len(video_id) else ""
        groups[(vid, int(tube_id[i]))].append(i)

    total = probs.new_zeros(())
    pairs = 0
    for (_vid, tid), idxs in groups.items():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda j: int(timestep[j]))
        for a, b in zip(idxs[:-1], idxs[1:]):
            total = total + accelerator.tube_smoothing(
                {tid: probs[b]}, {tid: probs[a]}
            )
            pairs += 1
    return total / pairs if pairs else total


def _cmsc_conservation(accelerator, batch: Dict[str, object], device) -> Optional[Tensor]:
    """CMSC alignment-conservation term, or ``None`` when the batch lacks text/embeds."""
    text = batch.get("text_embed")
    tube_full = batch.get("tube_visual_embed_full")
    tube_cf = batch.get("tube_visual_embed_cf")
    if not (isinstance(text, Tensor) and isinstance(tube_full, Tensor) and isinstance(tube_cf, Tensor)):
        return None
    if text.numel() == 0 or tube_full.numel() == 0 or tube_cf.numel() == 0:
        return None
    mask = batch.get("text_mask")
    return accelerator.cmsc_loss.alignment_conservation(
        text.to(device), tube_full.to(device), tube_cf.to(device),
        text_mask=mask.to(device) if isinstance(mask, Tensor) else None,
    )


# --------------------------------------------------------------------------- #
# the full joint loss
# --------------------------------------------------------------------------- #


def compute_joint_loss(
    accelerator,
    batch: Dict[str, object],
    *,
    training_cfg: Optional[TrainingConfig] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Assemble ``L_total`` and its (unweighted) components for one batch (§4.1).

    Parameters
    ----------
    accelerator
        The wired :class:`~cocf.core.accelerator.Accelerator` (frozen backbone +
        plugins). All learnable parameters that receive gradient here live on it.
    batch
        A :func:`cocf.data.cocf_batch.collate_cocf_samples` batch dict (tensors on
        any device; moved to the plugin device internally).
    training_cfg
        Loss weights (λ_sta/λ_cert/λ_cmsc/λ_cost); defaults to
        ``accelerator.config.training``.

    Returns
    -------
    (total, components)
        ``total`` is a scalar with ``grad_fn``; ``components`` are the *unweighted*
        per-term floats (plus the weighted total) for logging.
    """
    cfg = training_cfg or accelerator.config.training
    device = next(accelerator.parameters()).device

    tube_features = batch["tube_features"].to(device).float()        # [B, 7]
    strength_features = batch["strength_features"].to(device).float()  # [B, 3]
    actions = batch["action"].to(device).long()                      # [B]
    step_frac = batch["step_frac"].to(device).float()                # [B]
    damage_label = batch["damage_label"].to(device).float()          # [B, NUM_DAMAGE_DIMS]

    # --- L-COCF predictor forward (differentiable strength → input → μ/σ) --- #
    strength = accelerator.lcocf.strength_field(strength_features)   # [B] (grad)
    budget = per_sample_budget(accelerator, batch, device=device)    # [B]
    pred_input = build_predictor_input_batch(
        states=tube_features,
        strength_feats=strength_features,
        strength=strength,
        budget=budget,
        step_frac=step_frac,
        step_embed_dim=accelerator.config.lcocf.predictor.context_dim,
    )
    pred = accelerator.lcocf.predictor(pred_input)                   # μ,σ: [B, A]
    idx = actions.clamp(0, pred.mu.shape[-1] - 1).unsqueeze(-1)
    mu_a = pred.mu.gather(-1, idx).squeeze(-1)                       # [B]
    sigma_a = pred.sigma.gather(-1, idx).squeeze(-1)                 # [B]
    damage_true = damage_scalar_batch(damage_label)                 # [B]

    # --- L_cocf: Gaussian NLL of the executed action's damage ---------------- #
    l_cocf = gaussian_nll(damage_true, mu_a, sigma_a)

    # --- L_tube: STA temporal action-prob smoothness ------------------------- #
    probs = action_probs(pred.mu)                                   # [B, A]
    l_tube = tube_temporal_smoothness(accelerator, probs, batch)

    # --- L_cert: certificate calibrated as an upper bound on true damage ----- #
    zeros = mu_a.new_zeros(mu_a.shape)
    e_cert = accelerator.raec.certificate.value(
        mu_a, sigma_a,
        residual=zeros,
        boundary=tube_features[:, _BOUNDARY_IDX],
        anchor_age=tube_features[:, _AGE_IDX],
        local_cmsc=zeros,
    )
    l_cert = accelerator.raec.certificate.loss(e_cert, damage_true)

    # --- L_cmsc: text–tube alignment conservation (skipped if no embeds) ----- #
    l_cmsc = _cmsc_conservation(accelerator, batch, device)
    if l_cmsc is None:
        l_cmsc = mu_a.new_zeros(())

    # --- L_budget: expected action cost vs the dynamic budget ---------------- #
    action_cost = torch.tensor(
        accelerator.config.allocator.action_cost, device=device, dtype=torch.float32
    )
    l_budget = budget_penalty(probs, action_cost, budget)

    total = (
        l_cocf
        + cfg.lambda_sta * l_tube
        + cfg.lambda_cert * l_cert
        + cfg.lambda_cmsc * l_cmsc
        + cfg.lambda_cost * l_budget
    )
    components = {
        "cocf": float(l_cocf.detach()),
        "tube": float(l_tube.detach()),
        "cert": float(l_cert.detach()),
        "cmsc": float(l_cmsc.detach()),
        "budget": float(l_budget.detach()),
        "total": float(total.detach()),
    }
    return total, components
