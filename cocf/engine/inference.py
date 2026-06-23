"""Accelerated inference engine — the main denoising loop (§7.2).

The engine orchestrates a reverse denoising process from noisy latent z_T to clean
z_0, with dynamic per-tube compute allocation using the full COCF-SS-DCA pipeline:

    Iteration (t=T to t=1):
        1. Build/update semantic tubes G_t (STA)
        2. Extract 8D tube states s_{k,t} (STA)
        3. Compute causal strengths & damage predictions (L-COCF)
        4. Compute error certificates (RAEC)
        5. Solve action allocation under budget & risk constraints (scheduler)
        6. Execute actions: FULL/LOWFREQ/INTERP/ANCHOR (transition executor)
        7. Check error triggers & apply repairs: rollback/boundary-blend/KV-refresh
        8. Update anchor library & tube memory

The loop is fully stateless across calls (no accumulated gradients, device-agnostic
error handling). All mutable state lives in :class:`EngineState`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from cocf.backbones.base import BackboneAdapter, BackboneCache, TextConditioning
from cocf.backbones.transition import TransitionExecutor, TransitionResult
from cocf.cmsc.alignment import TextTubeAlignment
from cocf.cmsc.losses import CMSCLoss
from cocf.common.config import (
    EngineConfig,
    TriggerConfig,
)
from cocf.common.logging import get_logger
from cocf.common.types import (
    Action,
    CausalSubgraph,
    DamagePrediction,
    SemanticTube,
    StrengthLevel,
    TubeState,
    TokenGrid,
)
from cocf.core.accelerator import Accelerator
from cocf.engine.state import EngineState, GenerationResult, StepTrace
from cocf.lcocf.module import LCOCFModule
from cocf.raec.anchor_store import AnchorStore
from cocf.raec.module import RAECModule
from cocf.scheduler.allocator import ActionAllocator
from cocf.scheduler.budget import BudgetScheduler
from cocf.tubes.builder import TubeBuilder

Tensor = torch.Tensor
_log = get_logger(__name__)


class InferenceEngine(nn.Module):
    """Stateless accelerated denoising loop (§7.2).

    The engine is the **only** orchestrator of the full pipeline. It does not
    train; all trainable parameters live in the :class:`Accelerator`. The engine:

    1. Owns the :class:`BudgetScheduler` and :class:`ActionAllocator` (stateless)
    2. Calls into the accelerator's submodules (L-COCF, STA, RAEC, CMSC, etc.)
    3. Maintains transient per-generation state in :class:`EngineState`
    4. Returns :class:`GenerationResult` with efficiency traces

    The engine is device-agnostic: it works on CPU for testing and on GPU for
    production (all tensors follow the accelerator's device).

    Parameters
    ----------
    accelerator
        The wired accelerator containing all learnable components & backbone.
    engine_config
        Knobs: num_steps, tube_build_step, tube_refresh_every.
    trigger_config
        Risk thresholds & rollback params (§5.3.2).
    """

    def __init__(
        self,
        accelerator: Accelerator,
        engine_config: EngineConfig,
        trigger_config: TriggerConfig,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.engine_cfg = engine_config
        self.trigger_cfg = trigger_config

        # Stateless schedulers
        self.budget_scheduler = BudgetScheduler(accelerator.config.budget)
        self.action_allocator = ActionAllocator(accelerator.config.allocator)

        # Tube builder (only called at specific steps)
        self.tube_builder = accelerator.tube_builder

        device = next(accelerator.parameters()).device if list(accelerator.parameters()) else torch.device("cpu")
        self.device = device

    def generate(
        self,
        prompts: List[str],
        z_init: Tensor,  # [B, N, d] initial noise
        grid: TokenGrid,  # metadata about latent geometry
        cond: TextConditioning,  # text embeddings
        backbone: BackboneAdapter,  # frozen model
        return_intermediates: bool = False,
        *,
        record_sink: Optional[Callable[..., None]] = None,
        decode_grad: bool = False,
    ) -> GenerationResult:
        """Run accelerated denoising (the main entry point).

        Args:
            prompts: Text prompts (batch).
            z_init: Initial noisy latent [B, N, d].
            grid: TokenGrid metadata.
            cond: Text conditioning (embeddings, etc.).
            backbone: Frozen backbone model.
            return_intermediates: If True, return latents at all steps (memory-heavy).
            record_sink: Optional per-step callback ``(step_idx, t, budget, step_frac,
                tube_states, strength_feats, actions)`` used by Stage-C end-to-end
                fine-tuning to collect the exact per-tube features the engine allocated
                on (§4.2). ``None`` at inference (zero overhead).
            decode_grad: If True, decode the final latent **on** the autograd graph so
                ``result.video`` is differentiable (the §4.2 pixel-loss path to the
                LoRA / repair params). Inference keeps the cheaper ``no_grad`` decode.

        Returns:
            GenerationResult with final video, traces, efficiency stats.
        """
        batch_size = z_init.shape[0]

        # Initialize state for this generation
        state = EngineState(
            z=z_init,
            grid=grid,
            cond=cond,
            subgraph=None,  # Built on-demand
            # AnchorStore's only ctor arg is offload_to_cpu; RAEC owns the factory.
            anchor_store=self.accelerator.raec.new_anchor_store(),
            cache=None,
        )

        # Main denoising loop: t = T → 1
        num_steps = self.engine_cfg.num_inference_steps
        for step_idx in range(num_steps):
            # Reverse time: step 0 is t=T, step num_steps-1 is t=1
            t = num_steps - step_idx

            _log.debug(f"Denoising step {step_idx+1}/{num_steps} (t={t})")

            # Execute one denoising step with full COCF pipeline
            trace = self._step(
                state=state,
                step_idx=step_idx,
                t=t,
                backbone=backbone,
                record_sink=record_sink,
            )
            state.traces.append(trace)

        # Decode final latent to video. The adapter owns the token<->grid layout
        # (`to_grid`) and the VAE decode (`decode_latent`).
        z0_grid = self.accelerator.backbone.to_grid(state.z, state.grid)  # [B, C, T, H, W]

        # The backbone's decoder is frozen, so we call it directly. Stage-C end-to-end
        # fine-tuning needs the decode on the autograd graph (the §4.2 pixel-loss path to
        # the LoRA / repair params), so it passes ``decode_grad=True``; inference keeps the
        # cheaper no_grad decode.
        if decode_grad:
            video = backbone.decode_latent(z0_grid)  # [B, 3, F, H, W] (grad-enabled)
        else:
            with torch.no_grad():
                video = backbone.decode_latent(z0_grid)  # [B, 3, F, H, W]

        return GenerationResult(
            video=video,
            z0=state.z,
            traces=state.traces,
        )

    def _step(
        self,
        state: EngineState,
        step_idx: int,
        t: int,
        backbone: BackboneAdapter,
        record_sink: Optional[Callable[..., None]] = None,
    ) -> StepTrace:
        """Execute one complete denoising step with full COCF pipeline (§7.2).

        This is the per-timestep orchestration that implements the 8-step workflow
        described in the design document.
        """
        trace = StepTrace(
            step=step_idx,
            active_ratio=1.0,
            budget=1.0,
            predicted_cost=0.0,
            num_tubes=0,
        )

        # --- Step 1: Build/update semantic tubes G_t (STA) -----
        # The tube builder segments *RGB frames*, so decode a cheap preview of the
        # current latent first (build() expects [F, 3, Hp, Wp] with F == grid.t).
        if step_idx == self.engine_cfg.tube_build_step or (
            self.engine_cfg.tube_refresh_every > 0
            and step_idx % self.engine_cfg.tube_refresh_every == 0
            and step_idx > self.engine_cfg.tube_build_step
        ):
            frames_rgb = self._decode_preview_frames(state, backbone)
            state.tubes = self.tube_builder.build(frames_rgb, state.grid, state.prompt)
            _log.debug(f"  Built {len(state.tubes)} semantic tubes")

        trace.num_tubes = len(state.tubes)

        # No tubes yet (cold-start warm-up): advance the latent with a dense FULL
        # step and return — there is nothing to allocate or certify.
        if not state.tubes:
            state.prev_z = state.z
            state.z, state.cache = self._warmup_step(state, t, backbone)
            return trace

        # --- Step 2: Extract per-tube states s_{k,t}, keyed by tube_id -----
        # tube_builder.update returns {tube_id: TubeState}; using tube_id (not the
        # list index) is required because L-COCF indexes states[tube.tube_id] and
        # tube_ids are persistent/global (and drift from list position after splits).
        tube_states: Dict[int, TubeState] = self.tube_builder.update(state.tubes)

        # --- Step 3: Compute causal strengths & L-COCF damage predictions -----
        if state.subgraph is None:
            state.subgraph = self.accelerator.lcocf.parse(state.prompt)

        strength_feats = self.accelerator.lcocf.strength_features(
            state.tubes, tube_states, state.subgraph
        )
        strengths = self.accelerator.lcocf.strengths(strength_feats)
        priors = self.accelerator.lcocf.prior_actions(strengths, tube_states)

        # --- Step 5a: budget for this step (drives prediction & allocation) -----
        # Dynamic per-step budget B_t (§7.3): the U-shaped time profile modulated by
        # caption complexity and tube-interaction density. This is also exactly the
        # §4.2 "按字幕复杂度动态分配单步算力预算" Stage C relies on, so inference and the
        # end-to-end fine-tune size the budget identically (no train/serve skew). Gated
        # by ``use_dynamic_budget``: off ⇒ spend the full-compute ceiling every step.
        step_frac = t / self.engine_cfg.num_inference_steps
        if self.engine_cfg.use_dynamic_budget:
            complexity = self.budget_scheduler.score_complexity(state.prompt, state.subgraph)
            interaction = (
                sum(s.interaction for s in tube_states.values()) / len(tube_states)
                if tube_states else 0.0
            )
            budget_t = self.budget_scheduler.budget(
                step_frac, complexity=complexity,
                mean_uncertainty=state.prev_mean_uncertainty,
                interaction_density=interaction,
            )
        else:
            budget_t = self.accelerator.config.budget.b_max
        trace.budget = budget_t

        damage_preds = self.accelerator.lcocf.predict(
            state.tubes, tube_states, strength_feats, strengths,
            budget=budget_t, step_frac=step_frac, device=state.z.device,
        )
        trace.predicted_cost = float(
            sum(float(p.mu.detach().mean()) for p in damage_preds.values())
        )
        # Carry this step's mean damage uncertainty (mean σ over tubes) into the next
        # step's budget demand (§7.3): the more unsure the predictor, the more compute
        # the following step is allowed to spend.
        if damage_preds:
            state.prev_mean_uncertainty = float(
                sum(float(p.sigma.detach().mean()) for p in damage_preds.values())
                / len(damage_preds)
            )

        # --- Step 5b: Solve optimal action allocation -----
        # Tubes still inside a post-rollback/repair forced-FULL window (§5.3.2) are
        # pinned to FULL so a rolled-back tube is actually recomputed forward rather
        # than allowed to skip again and stay frozen at a stale latent.
        forced_full = {tid for tid, c in state.force_full_countdown.items() if c > 0}
        decision = self.action_allocator.allocate(
            state.tubes,
            damage_preds,
            budget=budget_t,
            states=tube_states,
            prior_actions=priors,
            forced_full=forced_full,
            step=step_idx,
        )
        optimal_actions = decision.actions  # {tube_id: Action}
        trace.actions = {k: Action(a).name for k, a in optimal_actions.items()}

        # Stage-C training hook: emit this step's per-tube (state, strength, action) so the
        # end-to-end fine-tune can recompute the scheduling regularisers on the exact
        # features the engine allocated on (§4.2, no train/serve skew). No-op at inference
        # (record_sink is None) and for warm-up steps (no tubes → returned above).
        if record_sink is not None:
            record_sink(
                step_idx=step_idx,
                t=t,
                budget=budget_t,
                step_frac=step_frac,
                tube_states=tube_states,
                strength_feats=strength_feats,
                actions=optimal_actions,
            )

        # Consume one step of every active forced-FULL window now that this step's
        # allocation has honoured it; drop tubes whose window has elapsed.
        for tid in list(state.force_full_countdown):
            state.force_full_countdown[tid] -= 1
            if state.force_full_countdown[tid] <= 0:
                del state.force_full_countdown[tid]

        # --- Step 6: Execute tube actions (the latent transition) -----
        result = self._execute_transition(
            state=state,
            t=t,
            decision=decision,
            backbone=backbone,
        )
        state.prev_z = state.z
        state.z = result.z_next
        state.cache = result.cache
        trace.active_ratio = result.active_ratio

        # --- Step 6b: §3.3.4 single-hop counterfactual check on skipped tubes -----
        # Fire only at temporal mutation points (s_T > θ_sT) for tubes that actually
        # skipped, capped per step (verifier.triggered_tubes). Compare the executed
        # (skip) latent against the transition's compute-everywhere reference z_full;
        # a residual above η means the skip omitted a causal effect (causal omission),
        # repaired locally by the L-COCF residual-repair net. Gated by cf_check_enabled.
        if self.engine_cfg.cf_check_enabled and result.z_full is not None:
            trace.cf_checks, trace.cf_repairs = self._counterfactual_check(
                state, result.z_full, optimal_actions, strength_feats
            )

        # --- Step 4 (post-transition): error certificates (RAEC), keyed by tube_id
        # Certify the action that was *actually executed* and feed it the real skip
        # residual δ_k = ‖z_full − z_action‖ measured by the transition (§5.3.1).
        # This must run after the transition: a pre-transition guess (prior action,
        # zero residual) decouples the risk trigger from the error it exists to catch.
        certificates: Dict[int, float] = {}
        for tube in state.tubes:
            tid = tube.tube_id
            cert = self.accelerator.raec.certify(
                tid,
                step_idx,
                optimal_actions.get(tid, Action.FULL),
                damage_preds[tid],
                residual=float(result.tube_residual.get(tid, 0.0)),
                anchor_age=float(state.anchor_store.age(tid, step_idx)),
            )
            certificates[tid] = cert.value

        # --- Step 7: Risk triggers & local repairs (RAEC) -----
        repairs_this_step = 0
        rollbacks_this_step = 0

        if self.engine_cfg.risk_control_enabled:
            for tube in state.tubes:
                cert_k = certificates.get(tube.tube_id, 0.0)

                if cert_k > self.trigger_cfg.tau_high:
                    # High risk: revoke the tube to its safe anchor, fuse the boundary
                    # against the compute-everywhere latent so the rolled-back interior
                    # joins its surroundings seam-free (§5.3.2 边界修复), and pin it to
                    # FULL for the next q steps so it is recomputed forward.
                    _log.debug(f"  Tube {tube.tube_id}: HIGH RISK ({cert_k:.3f}), rolling back")
                    if result.z_full is not None:
                        state.z = self.accelerator.raec.repair.rollback(
                            state.z, result.z_full, tube, state.grid, state.anchor_store
                        ).z
                    else:
                        state.z = state.anchor_store.rollback(state.z, tube)
                    state.force_full_countdown[tube.tube_id] = self.trigger_cfg.force_full_steps
                    rollbacks_this_step += 1

                elif self.trigger_cfg.tau_low < cert_k <= self.trigger_cfg.tau_high:
                    # Medium risk: boundary-fuse the tube's drifting rim toward the
                    # freshly computed latent (§5.3.2 边界修复/缓存刷新) and pin it to
                    # FULL for one refresh step so the region is recomputed rather than
                    # left to skip again.
                    _log.debug(f"  Tube {tube.tube_id}: MEDIUM RISK ({cert_k:.3f}), repairing")
                    if result.z_full is not None:
                        state.z = self.accelerator.raec.repair.repair(
                            state.z, result.z_full, tube, state.grid
                        ).z
                    state.force_full_countdown[tube.tube_id] = max(
                        state.force_full_countdown.get(tube.tube_id, 0), 1
                    )
                    repairs_this_step += 1

        trace.rollbacks = rollbacks_this_step
        trace.repairs = repairs_this_step

        # --- Step 8: Update anchor library (snapshot low-risk tubes) -----
        for tube in state.tubes:
            if certificates.get(tube.tube_id, 1.0) <= self.trigger_cfg.tau_low:
                state.anchor_store.update(tube, state.z, step_idx)

        return trace

    # ========================================================================= #
    # Helper methods for each step
    # ========================================================================= #

    def _decode_preview_frames(
        self, state: EngineState, backbone: BackboneAdapter
    ) -> Tensor:
        """Decode the current latent to RGB frames ``[grid.t, 3, Hp, Wp]`` for tube
        segmentation. The tube builder works on pixels and requires exactly one RGB
        frame per latent-temporal slot (``F == grid.t``).

        A real backbone's causal-temporal VAE expands the latent's ``T`` slots into
        ``(T-1)·c_t + 1`` pixel frames, so the decoded video generally has *more*
        frames than ``grid.t``. We subsample evenly back to ``grid.t`` representative
        frames (the mock keeps F == grid.t, so this is a no-op there).
        """
        with torch.no_grad():
            latent_grid = self.accelerator.backbone.to_grid(state.z, state.grid)
            video = backbone.decode_latent(latent_grid)  # [B, 3, F, Hp, Wp]
        # First batch element, reorder to [F, 3, Hp, Wp].
        frames = video[0].permute(1, 0, 2, 3).contiguous()
        f = frames.shape[0]
        if f != state.grid.t:
            # Pick grid.t evenly-spaced frames (one per latent-temporal slot).
            sel = torch.linspace(0, f - 1, state.grid.t, device=frames.device).round().long()
            frames = frames.index_select(0, sel)
        return frames

    def _warmup_step(
        self, state: EngineState, t: int, backbone: BackboneAdapter
    ) -> Tuple[Tensor, Optional[BackboneCache]]:
        """Dense FULL denoising step used before tubes exist (cold-start).

        Computes ε on every token and takes a single scheduler step, returning the
        advanced latent and the refreshed ε cache.
        """
        t_now = torch.full((state.z.shape[0],), float(t), device=state.z.device)
        t_next = torch.full((state.z.shape[0],), float(t - 1), device=state.z.device)
        out = backbone.denoise(
            state.z, t_now, state.cond, grid=state.grid,
            active_mask=None, cache=state.cache,
        )
        z_next = backbone.scheduler_step(out.model_output, t_now, t_next, state.z)
        return z_next, out.cache

    def _assemble_anchor_latent(self, state: EngineState) -> Optional[Tensor]:
        """Build the per-token "last verified-safe" latent ``[B, N, d]`` for ANCHOR
        reuse and the RAEC residual (§5.3.1).

        Each tube that has a stored safe anchor contributes its anchored tokens;
        all other tokens keep the current latent. Returns ``None`` when no tube has
        been anchored yet, in which case the transition falls back to a cheap
        ε-reuse step (and reports a zero residual, correctly — there is no safe
        reference to deviate from).
        """
        anchored = [t for t in state.tubes if state.anchor_store.has(t.tube_id)]
        if not anchored:
            return None
        z_anchor = state.z
        for tube in anchored:
            z_anchor = state.anchor_store.rollback(z_anchor, tube)
        return z_anchor

    def _execute_transition(
        self,
        state: EngineState,
        t: int,
        decision,
        backbone: BackboneAdapter,
    ) -> TransitionResult:
        """Execute one accelerated denoising transition for the allocated actions.

        Delegates to the accelerator's :class:`TransitionExecutor`, which handles
        FULL/LOWFREQ/INTERP/ANCHOR per tube and returns the advanced latent, the
        refreshed cache, the active-token mask and per-tube residuals. The safe
        anchor latent is supplied so ANCHOR tubes freeze to it and the measured
        residual ``‖z_full − z_anchor‖`` is meaningful (the RAEC trigger signal).
        """
        t_now = torch.full((state.z.shape[0],), float(t), device=state.z.device)
        t_next = torch.full((state.z.shape[0],), float(t - 1), device=state.z.device)
        return self.accelerator.transition.step(
            z_t=state.z,
            t=t_now,
            t_next=t_next,
            cond=state.cond,
            decision=decision,
            tubes=state.tubes,
            grid=state.grid,
            cache=state.cache,
            anchor_latent=self._assemble_anchor_latent(state),
            measure_residual=self.engine_cfg.measure_residual,
        )

    def _counterfactual_check(
        self,
        state: EngineState,
        z_full: Tensor,
        actions: Dict[int, Action],
        strength_feats,
    ) -> Tuple[int, int]:
        """Run the §3.3.4 single-hop counterfactual verification + local repair.

        For each tube the L-COCF verifier flags — temporal-mutation point ``s_T > θ_sT``
        *and* the tube skipped this step, capped at ``max_checks_per_step`` — the
        executed (skip) tube latent is compared against the compute-everywhere
        reference ``z_full``. When the residual exceeds ``η`` the residual-repair net
        corrects that tube's tokens (do(¬skip) causal-omission repair, §3.3.4), spliced
        back out-of-place so the autograd graph (Stage-C repair-net training) is intact.
        Returns ``(num_checks, num_repairs)``.
        """
        verifier = self.accelerator.lcocf.verifier
        skipped = {
            tube.tube_id: actions.get(tube.tube_id, Action.FULL).is_skip
            for tube in state.tubes
        }
        triggered = verifier.triggered_tubes(strength_feats, skipped)
        if not triggered:
            return 0, 0
        tubes_by_id = {t.tube_id: t for t in state.tubes}
        checks = repairs = 0
        for tid in triggered:
            tube = tubes_by_id.get(tid)
            if tube is None:
                continue
            idx = tube.all_token_indices().to(state.z.device)
            if idx.numel() == 0:
                continue
            checks += 1
            rows: List[Tensor] = []
            changed = False
            for b in range(state.z.shape[0]):
                z_skip = state.z[b].index_select(0, idx)       # [n_tok, d]
                z_ref = z_full[b].index_select(0, idx)         # [n_tok, d]
                vr = verifier.verify_and_repair(tube, z_skip, z_ref)
                if vr.repaired and vr.z_corrected is not None:
                    rows.append(vr.z_corrected)
                    changed = True
                else:
                    rows.append(z_skip)
            if changed:
                corrected = torch.stack(rows, dim=0).to(state.z.dtype)   # [B, n_tok, d]
                z_new = state.z.clone()
                z_new[:, idx] = corrected
                state.z = z_new
                repairs += 1
        return checks, repairs
