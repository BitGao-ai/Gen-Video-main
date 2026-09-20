"""Stage-B joint loss assembly: ``L_total`` over a counterfactual batch.

The pure realisation of the combined objective from one collated batch (no IO, no
optimiser, no loop), reused verbatim by Stage C's regulariser term::

    L_total = L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc + λ_cost·L_budget

L_cocf is the L-COCF predictor's Gaussian NLL; L_tube the STA temporal smoothing;
L_cert the RAEC certificate calibration; L_cmsc the text-tube alignment conservation;
L_budget the over-budget compute penalty. The certificate's skip-residual and
local-CMSC inputs are not in the offline per-sample schema and are passed as zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from cocf.common.config import TrainingConfig
from cocf.cmsc.alignment import alignment_risk
from cocf.common.types import TUBE_STATE_FIELDS
from cocf.lcocf.damage import DAMAGE_DIMENSIONS, DEFAULT_DAMAGE_WEIGHTS, NUM_DAMAGE_DIMS
from cocf.lcocf.predictor import build_predictor_input_batch

Tensor = torch.Tensor

# Temperature of the soft action policy ``p(a) = softmax(−μ_a / τ)``: lower damage
# means a higher probability of the cheaper action.
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


def predictor_regression_loss(
    target: Tensor,
    mu: Tensor,
    actions: Optional[Tensor] = None,
    *,
    objective: str = "mse",
    scale: float = 100.0,
) -> Tensor:
    """Phase-1 mean pretraining loss: scaled MSE/Huber on ``mu``, no variance term.

    Regressing ``mu`` alone first gives the hidden layers a stable signal; ``sigma`` is
    calibrated afterwards. FULL rows (action 0, label pinned at zero) are excluded when
    ``actions`` is given. ``scale`` inflates the tiny targets into a comfortable range.
    """
    if actions is not None:
        keep = actions != 0
        if not bool(keep.any()):
            return mu.new_zeros(())
        target, mu = target[keep], mu[keep]
    if objective == "mse":
        return torch.nn.functional.mse_loss(mu * scale, target * scale)
    if objective == "huber":
        return torch.nn.functional.smooth_l1_loss(mu * scale, target * scale)
    raise ValueError(f"unknown mean-phase objective: {objective!r}")


def _nonfull_nll(target: Tensor, mu: Tensor, sigma: Tensor, actions: Tensor) -> Tensor:
    """Gaussian NLL over non-FULL rows only (differentiable zero when none).

    FULL's label and prediction are both zero, so its NLL term decreases monotonically
    with σ; including it would let the variance head shrink σ where there is nothing to
    calibrate.
    """
    keep = actions != 0
    if not bool(keep.any()):
        return sigma.sum() * 0.0
    return gaussian_nll(target[keep], mu[keep], sigma[keep])


def budget_penalty(probs: Tensor, action_cost: Tensor, budget: Tensor) -> Tensor:
    """Mean over-budget penalty ``relu(E[cost] − B_t)``."""
    expected_cost = (probs * action_cost).sum(-1)        # [B]
    return torch.relu(expected_cost - budget).mean()


# --------------------------------------------------------------------------- #
# per-sample dynamic budget B_t — a conditioning input, non-differentiable
# --------------------------------------------------------------------------- #


def per_sample_budget(accelerator, batch: Dict[str, object], device=None) -> Tensor:
    """Vector ``[B]`` of the dynamic step budget ``B_t`` for each sample.

    Uses the shared :class:`~cocf.scheduler.budget.BudgetScheduler` so the budget the
    loss compares against is the one the inference loop spends. Scene complexity is
    unavailable per offline sample, so only the time profile, uncertainty and
    interaction-density signals apply.
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
    accelerator, probs: Tensor, batch: Dict[str, object], *, return_pairs=False
):
    """STA temporal term: mean action-prob change for a tube across adjacent steps.

    Groups the batch by ``(video_id, tube_id)``, orders each group by timestep and
    feeds consecutive-step probability pairs to the shared
    :class:`~cocf.tubes.smoothing.TubeSmoothingLoss`. Zero when no two same-tube
    adjacent-step samples co-occur.
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
        by_step = defaultdict(list)
        for j in idxs:
            by_step[int(timestep[j])].append(j)
        states = [probs[by_step[step]].mean(0) for step in sorted(by_step)]
        for a, b in zip(states[:-1], states[1:]):
            total = total + accelerator.tube_smoothing(
                {tid: b}, {tid: a}
            )
            pairs += 1
    value = total / pairs if pairs else total
    return (value, pairs) if return_pairs else value


def batch_float(batch: Dict[str, object], key: str, like: Tensor) -> Tensor:
    """Per-sample float column as ``[B]`` on ``like``'s device (zeros when absent).

    Shared with Stage C so both stages feed the certificate the same way.
    """
    v = batch.get(key)
    if not isinstance(v, Tensor) or v.numel() == 0:
        return like.new_zeros(like.shape)
    return v.to(like.device).float().reshape(-1)[: like.shape[0]]


def _local_cmsc_violation(accelerator, batch: Dict[str, object], like: Tensor) -> Tensor:
    """Per-sample local CMSC violation ``1 − align(tube, prompt)`` as ``[B]``.

    Raises the certificate risk of skipping a tube poorly aligned to the prompt, scored
    on the counterfactual render's tube embed (what the skip actually produced).
    """
    text = batch.get("text_embed")
    tube_cf = batch.get("tube_visual_embed_cf")
    if not (isinstance(text, Tensor) and isinstance(tube_cf, Tensor)):
        return like.new_zeros(like.shape)
    if text.numel() == 0 or tube_cf.numel() == 0:
        return like.new_zeros(like.shape)
    dev = like.device
    txt = text.to(dev).float()
    cf = tube_cf.to(dev).float()
    mask = batch.get("text_mask")
    out = like.new_zeros(like.shape)
    with torch.no_grad():
        # ``tube_scores`` scores one prompt's [L, d_c] tokens against [K, d_v] tube
        # embeds, per sample.
        for i in range(min(like.shape[0], txt.shape[0], cf.shape[0])):
            tokens = txt[i]
            if isinstance(mask, Tensor) and mask.numel():
                keep = mask.to(dev)[i].bool()
                tokens = tokens[keep]
            if tokens.shape[0] == 0:
                continue
            score = accelerator.cmsc_alignment.tube_scores(tokens, cf[i : i + 1])
            out[i] = alignment_risk(score).reshape(-1)[0]
    return out


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
    phase: str = "joint",
    mean_objective: str = "mse",
    target_scale: float = 100.0,
    isolate_aux: bool = False,
) -> Tuple[Tensor, Dict[str, float]]:
    """Assemble ``L_total`` and its (unweighted) components for one batch.

    ``phase`` selects the objective: ``"joint"`` (default) is the classic combined
    loss; ``"mean"`` (phased mode) replaces L_cocf with a scaled regression on ``mu``;
    ``"var"`` returns the plain Gaussian NLL over non-FULL rows only. ``isolate_aux``
    feeds the certificate detached ``mu``/``sigma`` in every phase and drops the
    STA/budget terms from the mean-phase total. Returns ``(total, components)`` where
    ``total`` carries ``grad_fn`` and ``components`` are the unweighted per-term floats.
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

    # --- L_cocf: phase-dependent predictor loss ------------------------------- #
    if phase == "var":
        # Variance-calibration phase: non-FULL NLL only; other terms have no
        # trainable path (the caller froze everything but predictor.var_head).
        l_cocf = _nonfull_nll(damage_true, mu_a, sigma_a, actions)
        return l_cocf, {"cocf": float(l_cocf.detach()), "total": float(l_cocf.detach())}
    if phase == "mean":
        l_cocf = predictor_regression_loss(
            damage_true, mu_a, actions, objective=mean_objective, scale=target_scale
        )
    elif isolate_aux:
        # Keep FULL out of the NLL in the joint fine-tune too, as in the variance
        # phase. Classic training (isolate_aux=False) keeps the all-sample NLL.
        l_cocf = _nonfull_nll(damage_true, mu_a, sigma_a, actions)
    else:
        l_cocf = gaussian_nll(damage_true, mu_a, sigma_a)

    # --- L_tube: STA temporal action-prob smoothness ------------------------- #
    probs = action_probs(pred.mu)                                   # [B, A]
    l_tube = tube_temporal_smoothness(accelerator, probs, batch)

    # --- L_cert: certificate calibrated as an upper bound on true damage ----- #
    # ``residual`` and ``local_cmsc`` carry the real per-sample signal Stage A
    # measured: δ on the intervened tube, and the tube's local text-alignment
    # violation from the stored CMSC embeds.
    residual = batch_float(batch, "skip_residual", mu_a)
    local_cmsc = _local_cmsc_violation(accelerator, batch, mu_a)
    if phase == "mean" or isolate_aux:
        # The certificate's coefficients still calibrate, but against a read-only
        # (mu, sigma): during the controlled phases nothing outside the regression
        # term may push the mean or the variance.
        mu_cert, sigma_cert = mu_a.detach(), sigma_a.detach()
    else:
        mu_cert, sigma_cert = mu_a, sigma_a
    e_cert = accelerator.raec.certificate.value(
        mu_cert, sigma_cert,
        residual=residual,
        boundary=tube_features[:, _BOUNDARY_IDX],
        anchor_age=tube_features[:, _AGE_IDX],
        local_cmsc=local_cmsc,
    )
    l_cert = accelerator.raec.certificate.loss(e_cert, damage_true)

    # --- L_cmsc: text–tube alignment conservation (skipped if no embeds) ----- #
    l_cmsc = _cmsc_conservation(accelerator, batch, device)
    if l_cmsc is None:
        l_cmsc = mu_a.new_zeros(())

    # --- L_budget: expected action cost vs the dynamic budget ---------------- #
    action_cost = torch.tensor(
        accelerator.allocator.action_cost, device=device, dtype=torch.float32
    )
    l_budget = budget_penalty(probs, action_cost, budget)

    # Mean phase under auxiliary isolation: a pure regression control — the STA
    # and budget terms stay out of the total so only the regression shapes mu.
    skip_aux = phase == "mean" and isolate_aux
    total = (
        l_cocf
        + (0.0 if skip_aux else cfg.lambda_sta * l_tube)
        + cfg.lambda_cert * l_cert
        + cfg.lambda_cmsc * l_cmsc
        + (0.0 if skip_aux else cfg.lambda_cost * l_budget)
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
