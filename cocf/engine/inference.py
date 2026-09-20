"""Accelerated inference engine: the main denoising loop.

Runs reverse denoising from noisy latent z_T to clean z_0 with dynamic per-tube
compute allocation across the COCF pipeline. Each step (t = T..1):

    1. Build/update semantic tubes (STA)
    2. Extract per-tube states (STA)
    3. Predict causal strengths and damage (L-COCF)
    4. Compute error certificates (RAEC)
    5. Allocate actions under budget and risk constraints (scheduler)
    6. Execute actions: FULL/LOWFREQ/INTERP/ANCHOR
    7. Check triggers and repair: rollback/boundary-blend/KV-refresh
    8. Update anchor library and tube memory

The loop is stateless across calls; all mutable state lives in :class:`EngineState`.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from cocf.backbones.base import BackboneAdapter, BackboneCache, TextConditioning, sigma_from_step
from cocf.backbones.transition import TransitionExecutor, TransitionResult
from cocf.cmsc.alignment import TextTubeAlignment
from cocf.cmsc.losses import CMSCLoss
from cocf.common.config import (
    EngineConfig,
    TriggerConfig,
)
from cocf.common.logging import get_logger
from cocf.common.memory import peak_memory
from cocf.common.types import (
    Action,
    TriggerLevel,
    CausalSubgraph,
    DamagePrediction,
    SemanticTube,
    StrengthLevel,
    TubeState,
    TokenGrid,
)
from cocf.core.accelerator import Accelerator
from cocf.engine.state import EngineState, GenerationResult, StepTrace
from cocf.lcocf.data import tube_clip_embed
from cocf.lcocf.module import LCOCFModule
from cocf.raec.anchor_store import AnchorStore
from cocf.raec.module import RAECModule
from cocf.scheduler.allocator import ActionAllocator
from cocf.scheduler.budget import BudgetScheduler
from cocf.tubes.builder import TubeBuilder

Tensor = torch.Tensor
_log = get_logger(__name__)


class InferenceEngine(nn.Module):
    """Stateless accelerated denoising loop.

    The engine orchestrates the pipeline but does not train; all trainable
    parameters live in the :class:`Accelerator`. It owns the (stateless) budget
    scheduler and action allocator, calls into the accelerator submodules, keeps
    transient state in :class:`EngineState`, and returns a
    :class:`GenerationResult` with efficiency traces. Device-agnostic.
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

        self.budget_scheduler = accelerator.budget_scheduler
        self.action_allocator = accelerator.allocator

        self.tube_builder = accelerator.tube_builder

        self._warned_no_pixel_span = False
        self._warned_prefix_window = False

    def generate(
        self,
        prompts: List[str],
        z_init: Tensor,  # [B, N, d] initial noise
        grid: TokenGrid,  # metadata about latent geometry
        cond: TextConditioning,  # text embeddings
        backbone: BackboneAdapter,  # frozen model
        *,
        record_sink: Optional[Callable[..., None]] = None,
        decode_grad: bool = False,
    ) -> GenerationResult:
        """Run accelerated denoising (the main entry point).

        Args:
            prompts: Text prompts (batch).
            z_init: Initial noisy latent [B, N, d].
            grid: TokenGrid metadata.
            cond: Text conditioning.
            backbone: Frozen backbone model.
            record_sink: Optional per-step callback used by Stage-C fine-tuning to
                collect the per-tube features the engine allocated on. ``None`` at
                inference.
            decode_grad: If True, run the trajectory on the autograd graph with a
                differentiable final decode; inference leaves it False.

        Returns:
            GenerationResult with final video, traces, efficiency stats.
        """
        # inference_mode is viral: a tensor produced under it can never join an autograd
        # graph, so the backbone's grad mode is opened for the whole span when a
        # differentiable decode is requested. The peak-memory probe wraps the
        # trajectory (a no-op off CUDA).
        if z_init.shape[0] != 1 or len(prompts) != 1 or cond.embeds.shape[0] != 1:
            raise ValueError("InferenceEngine requires a single video and prompt per call")
        if cond.prompts and list(cond.prompts) != list(prompts):
            raise ValueError("prompts must match the encoded conditioning prompts")
        with peak_memory("engine.generate") as mem:
            with backbone.grad_mode(decode_grad):
                result = self._generate(
                    prompts, z_init, grid, cond, backbone,
                    record_sink=record_sink, decode_grad=decode_grad,
                )
        result.peak_gib = mem["peak_gib"]  # filled on context exit
        return result

    def _generate(
        self,
        prompts: List[str],
        z_init: Tensor,
        grid: TokenGrid,
        cond: TextConditioning,
        backbone: BackboneAdapter,
        *,
        record_sink: Optional[Callable[..., None]] = None,
        decode_grad: bool = False,
    ) -> GenerationResult:
        """The denoising trajectory itself (see :meth:`generate` for the contract)."""
        batch_size = z_init.shape[0]

        # Clear RAEC's per-run trigger bookkeeping to keep the engine stateless.
        self.accelerator.raec.reset()

        state = EngineState(
            z=z_init,
            grid=grid,
            cond=cond,
            subgraph=None,
            anchor_store=self.accelerator.raec.new_anchor_store(
                self.accelerator.config.memory
            ),
            cache=None,
        )

        num_steps = self.engine_cfg.num_inference_steps
        # Truncated BPTT counted in computed steps rather than wall-clock ones; 0 keeps
        # the whole trajectory.
        window = int(getattr(self.engine_cfg, "grad_window_steps", 0))
        state.grad_window = window if decode_grad else 0
        _log.info("generation start: steps=%d batch=%d tokens=%d decode_grad=%s grad_window=%d",
                  num_steps, batch_size, grid.num_tokens, decode_grad, state.grad_window)

        for step_idx in range(num_steps):
            t = num_steps - step_idx

            _log.debug(f"Denoising step {step_idx+1}/{num_steps} (t={t})")

            trace = self._step(
                state=state,
                step_idx=step_idx,
                t=t,
                backbone=backbone,
                record_sink=record_sink,
            )
            state.traces.append(trace)

            if trace.compute_ratio > 0.0:
                state.retained_computed += 1
            log_step = _log.info if (step_idx % max(1, self.engine_cfg.log_every_steps) == 0 or step_idx == num_steps - 1) else _log.debug
            log_step(
                "step %d/%d: compute=%.4f mask=%.4f budget=%.3f tubes=%d "
                "rollbacks=%d repairs=%d cf_repairs=%d latent_grad=%s retained=%d cuts=%d",
                step_idx + 1, num_steps, trace.compute_ratio, trace.mask_ratio,
                trace.budget, trace.num_tubes, trace.rollbacks, trace.repairs,
                trace.cf_repairs, state.z.requires_grad, state.retained_computed, state.graph_cuts,
            )
            if trace.actions:
                hist = Counter(trace.actions.values())
                log_step("step %d actions: %s", step_idx + 1,
                         ", ".join(f"{name}={n}" for name, n in sorted(hist.items())))

        # Decode the final latent to video.
        z0_grid = self.accelerator.backbone.to_grid(state.z, state.grid)  # [B, C, T, H, W]

        frame_span: Optional[Tuple[int, int]] = None
        if decode_grad:
            lo, hi, frame_span = self._grad_decode_window(state.grid, backbone)
            video = backbone.decode_to_unit(z0_grid[:, :, lo:hi])  # grad-enabled
            self._warn_if_no_graph(video, state, window)
        else:
            with torch.no_grad():
                video = backbone.decode_to_unit(z0_grid)  # [B, 3, F, H, W] in [0,1]

        _log.info("generation decoded: shape=%s video_grad=%s graph_cuts=%d",
                  tuple(video.shape), video.requires_grad, state.graph_cuts)
        return GenerationResult(
            video=video,
            z0=state.z,
            traces=state.traces,
            tubes=state.tubes,
            grid=state.grid,
            frame_span=frame_span,
        )

    def _grad_decode_window(
        self, grid: TokenGrid, backbone: BackboneAdapter
    ) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """Latent slots ``[lo, hi)`` to decode on the graph, and the pixel range they cover.

        Returns the whole clip (``frame_span=None``) when windowing is off or when the
        backbone can describe no window. The offset is drawn uniformly to keep the
        subsampled loss unbiased, but it is offered to the backbone: a causal-temporal
        VAE can only reproduce a sub-range starting at slot 0, so such an adapter
        declines the random offset and this falls back to the prefix ``[0, k)``.
        """
        k = int(getattr(self.engine_cfg, "decode_grad_frames", 0))
        if k <= 0 or k >= grid.t:
            return 0, grid.t, None
        span_fn = getattr(backbone, "pixel_span", None)
        if callable(span_fn):
            lo = int(torch.randint(0, grid.t - k + 1, (1,)).item())
            for cand in ((lo, 0) if lo > 0 else (0,)):
                span = span_fn(cand, cand + k)
                if span is not None:
                    if cand == 0 and lo > 0:
                        self._warn_prefix_window(k, backbone, span)
                    return cand, cand + k, span
        if not self._warned_no_pixel_span:
            self._warned_no_pixel_span = True
            _log.warning(
                "engine.decode_grad_frames=%d requested, but %s describes no decodable "
                "window (pixel_span() is absent or declined both the random offset and "
                "the prefix), so it cannot be aligned against the full-compute "
                "reference. Decoding the whole clip instead — expect the differentiable "
                "decode to dominate peak memory.",
                k, type(backbone).__name__,
            )
        return 0, grid.t, None

    def _warn_prefix_window(
        self, k: int, backbone: BackboneAdapter, span: Tuple[int, int]
    ) -> None:
        """Say once that the windowed decode is pinned to the head of every clip.

        Not an error: a prefix window is the only one a causal decoder can align, but
        the pixel/semantic gradients then reach only frames ``[start, stop)``, so the
        clip tail trains on the schedule regularisers alone.
        """
        if self._warned_prefix_window:
            return
        self._warned_prefix_window = True
        _log.info(
            "%s decodes causally, so the differentiable window is pinned to the clip "
            "prefix: engine.decode_grad_frames=%d ⇒ pixel frames [%d, %d) carry the "
            "§4.2 pixel/semantic gradient on every step, and later frames carry none. "
            "Raise engine.decode_grad_frames to widen it (memory permitting).",
            type(backbone).__name__, k, span[0], span[1],
        )

    @staticmethod
    def _warn_if_no_graph(video: Tensor, state: EngineState, window: int) -> None:
        """Warn when a ``decode_grad`` render carries no gradient.

        Two causes: no step inside the retained window ran the denoiser, or detached
        anchors / absent LoRA-repair participation left nothing trainable in the render.
        """
        if video.requires_grad:
            return
        computed = [t.step for t in state.traces if t.compute_ratio > 0.0]
        _log.warning(
            "decode_grad=True but the render carries no autograd graph — the §4.2 "
            "pixel/semantic losses cannot reach the backbone (LoRA / repair net). "
            "Denoiser forwards ran at step(s) %s of %d; the BPTT window retains %s "
            "computed step(s). Check LoRA/repair participation and detached anchor "
            "replacement; increasing the window alone does not guarantee gradients.",
            computed or "none", len(state.traces),
            window if window > 0 else "all",
        )

    def _step(
        self,
        state: EngineState,
        step_idx: int,
        t: int,
        backbone: BackboneAdapter,
        record_sink: Optional[Callable[..., None]] = None,
    ) -> StepTrace:
        """Execute one complete denoising step with the full COCF pipeline."""
        trace = StepTrace(
            step=step_idx,
            mask_ratio=1.0,
            budget=1.0,
            predicted_cost=0.0,
            num_tubes=0,
            compute_ratio=1.0,
        )

        # Step 1: build/update semantic tubes. The builder segments RGB frames, so a
        # cheap preview of the current latent is decoded first.
        if step_idx == self.engine_cfg.tube_build_step or (
            self.engine_cfg.tube_refresh_every > 0
            and (step_idx - self.engine_cfg.tube_build_step) % self.engine_cfg.tube_refresh_every == 0
            and step_idx > self.engine_cfg.tube_build_step
        ):
            frames_rgb = self._decode_preview_frames(state, backbone)
            state.tubes, _, state.latent_flows = self.tube_builder.build_with_states(
                frames_rgb, state.grid, state.prompt)
            # Pool each tube's CLIP visual embed off the same preview frames; they feed
            # the certificate's local-CMSC term every step and only change on rebuild.
            state.tube_embeds = {
                tube.tube_id: tube_clip_embed(
                    frames_rgb, tube, state.grid, self.accelerator.perception
                )
                for tube in state.tubes
            }
            # Retire anchors and trigger bookkeeping of tubes that no longer exist.
            live_ids = {t.tube_id for t in state.tubes}
            dropped = state.anchor_store.retain(live_ids)
            self.accelerator.raec.trigger.retain(live_ids)
            _log.debug(f"  Built {len(state.tubes)} semantic tubes "
                       f"({dropped} stale anchor(s) retired)")

        trace.num_tubes = len(state.tubes)

        # No tubes yet (cold-start warm-up): advance with a dense FULL step.
        if not state.tubes:
            state.z, state.cache, warm_cost = self._warmup_step(state, t, backbone)
            trace.compute_ratio = warm_cost  # a warm-up step is a full dense forward
            return trace

        # Step 2: extract per-tube states keyed by tube_id (persistent/global ids).
        for tube in state.tubes:
            tube.state.anchor_age = float(state.anchor_store.age(tube.tube_id, step_idx))
        tube_states: Dict[int, TubeState] = self.tube_builder.update(
            state.tubes, latent_flow_by_frame=state.latent_flows)

        # Step 3: causal strengths and L-COCF damage predictions.
        if state.subgraph is None:
            state.subgraph = self.accelerator.lcocf.parse(state.prompt)

        strength_feats = self.accelerator.lcocf.strength_features(
            state.tubes, tube_states, state.subgraph
        )
        strengths = self.accelerator.lcocf.strengths(strength_feats)
        priors = self.accelerator.lcocf.prior_actions(strengths, tube_states)

        # Step 5a: per-step budget. U-shaped time profile modulated by caption
        # complexity and tube-interaction density; gated by ``use_dynamic_budget``.
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
        trace.predicted_damage = float(
            sum(float(p.mu.detach().mean()) for p in damage_preds.values())
        )
        # Carry this step's mean damage uncertainty into the next step's budget demand.
        if damage_preds:
            state.prev_mean_uncertainty = float(
                sum(float(p.sigma.detach().mean()) for p in damage_preds.values())
                / len(damage_preds)
            )

        # Step 5b: solve action allocation. Tubes inside a post-rollback/repair
        # forced-FULL window are pinned to FULL so they are recomputed forward; the pins
        # are owned by :class:`~cocf.raec.trigger.RiskTrigger`.
        trigger = self.accelerator.raec.trigger
        forced_full = (
            {t.tube_id for t in state.tubes}
            if self.engine_cfg.force_all_full
            else trigger.forced_full_tubes()
        )
        # Local conservation proxy ``1 - align(tube, prompt)``: a tube the prompt barely
        # describes risks a semantic violation on skip, raising its certificate through
        # λ_cmsc. Computed by one projection over the cached tube embeds.
        local_cmsc = self.accelerator.cmsc_loss.local_conservation(
            state.cond.embeds[0], state.tube_embeds,
            text_mask=state.cond.mask[0] if state.cond.mask is not None else None,
        ) if state.tube_embeds else {}
        # Hard risk constraint ``E_cert_k(a_k) <= τ_r``, evaluated before the action is
        # chosen.
        action_risk = {
            tube.tube_id: self.accelerator.raec.action_risk(
                damage_preds[tube.tube_id],
                boundary=float(tube_states[tube.tube_id].boundary_uncertainty)
                if tube.tube_id in tube_states else 0.0,
                anchor_age=float(state.anchor_store.age(tube.tube_id, step_idx)),
                local_cmsc=local_cmsc.get(tube.tube_id, 0.0),
            )
            for tube in state.tubes
            if tube.tube_id in damage_preds
        }
        decision = self.action_allocator.allocate(
            state.tubes,
            damage_preds,
            budget=budget_t,
            states=tube_states,
            prior_actions=priors,
            forced_full=forced_full,
            action_risk=action_risk,
            step=step_idx,
        )
        if self.engine_cfg.diagnostic_lowfreq_full:
            changed = [tid for tid, action in decision.actions.items() if action == Action.LOWFREQ]
            for tid in changed:
                decision.actions[tid] = Action.FULL
            _log.info("diagnostic: LOWFREQ -> FULL tubes=%s; allocator cost is pre-override", changed)
        # Periodic LOWFREQ refresh: promote LOWFREQ tubes to FULL every
        # ``lowfreq_refresh_every`` steps to serve fresh values and re-ground the cache.
        refresh_every = self.engine_cfg.lowfreq_refresh_every
        if refresh_every > 0 and (step_idx + 1) % refresh_every == 0:
            refreshed = [tid for tid, action in decision.actions.items()
                         if action == Action.LOWFREQ]
            for tid in refreshed:
                decision.actions[tid] = Action.FULL
            if refreshed:
                _log.info("step %d: LOWFREQ refresh — tubes %s computed FULL this step",
                          step_idx + 1, refreshed)
        optimal_actions = decision.actions  # {tube_id: Action}
        coverage = {}
        for action in Action:
            indices = [tube.all_token_indices().reshape(-1) for tube in state.tubes
                       if decision.action_for(tube.tube_id) == action]
            coverage[action.name] = int(torch.unique(torch.cat(indices)).numel()) if indices else 0
        _log.info("step %d action token coverage (unique within action; overlaps across actions): %s total=%d",
                  step_idx + 1, coverage, state.grid.num_tokens)
        trace.predicted_cost = decision.predicted_cost
        trace.actions = {k: Action(a).name for k, a in optimal_actions.items()}

        # Consume one step of every active forced-FULL window.
        trigger.step()

        # Step 6: execute tube actions (the latent transition).
        result = self._execute_transition(
            state=state,
            t=t,
            decision=decision,
            backbone=backbone,
        )
        state.z = result.z_next
        state.cache = result.cache
        trace.mask_ratio = result.mask_ratio
        trace.compute_ratio = result.compute_ratio

        # Adopt the executed actions after a promoted whole-step skip so the trace,
        # certificate and counterfactual check describe what really happened.
        if result.downgraded:
            optimal_actions = {**optimal_actions, **result.downgraded}
            trace.actions = {k: Action(a).name for k, a in optimal_actions.items()}
            _log.debug("  step promoted to a whole-step skip; %d tube(s) downgraded",
                       len(result.downgraded))

        # Step 6b: single-hop counterfactual check on skipped tubes. Fires at temporal
        # mutation points for tubes that skipped; a residual above η means the skip
        # omitted a causal effect, repaired locally by the residual-repair net.
        if self.engine_cfg.cf_check_enabled and result.z_full is not None:
            trace.cf_checks, trace.cf_repairs = self._counterfactual_check(
                state, result.z_full, optimal_actions, strength_feats
            )

        # Step 4 (post-transition): error certificates keyed by tube_id. Certify the
        # action actually executed, feeding it the measured skip residual, boundary
        # uncertainty, anchor age and local cross-modal violation. Runs after the
        # transition so the risk trigger sees the real error.
        certificates: Dict[int, float] = {}
        for tube in state.tubes:
            tid = tube.tube_id
            cert = self.accelerator.raec.certify(
                tid,
                step_idx,
                optimal_actions.get(tid, Action.FULL),
                damage_preds[tid],
                residual=float(result.tube_residual.get(tid, 0.0)),
                boundary=float(tube_states[tid].boundary_uncertainty)
                if tid in tube_states else 0.0,
                anchor_age=float(state.anchor_store.age(tid, step_idx)),
                local_cmsc=local_cmsc.get(tid, 0.0),
            )
            certificates[tid] = cert.value

        # Stage-C training hook: emit this step's per-tube (state, strength, executed
        # action) so the fine-tune recomputes the scheduling regularisers on the exact
        # features the engine allocated on. Fired after the transition and certificates
        # so ``optimal_actions`` names what ran and the residual/CMSC inputs exist.
        # No-op at inference and for warm-up steps.
        if record_sink is not None:
            record_sink(
                step_idx=step_idx,
                t=t,
                budget=budget_t,
                step_frac=step_frac,
                tube_states=tube_states,
                strength_feats=strength_feats,
                actions=optimal_actions,
                tube_residual=result.tube_residual,
                local_cmsc=local_cmsc,
            )

        # Step 7: risk triggers and local repairs.
        repairs_this_step = 0
        rollbacks_this_step = 0

        if self.engine_cfg.risk_control_enabled:
            for tube in state.tubes:
                cert_k = certificates.get(tube.tube_id, 0.0)
                level = trigger.classify_value(cert_k)

                if level == TriggerLevel.ROLLBACK:
                    # High risk: revoke the tube to its safe anchor, fuse the boundary
                    # against the compute-everywhere latent, and pin it to FULL for the
                    # next q steps so it is recomputed forward.
                    _log.debug(f"  Tube {tube.tube_id}: HIGH RISK ({cert_k:.3f}), rolling back")
                    if result.z_full is not None:
                        rr = self.accelerator.raec.repair.rollback(
                            state.z, result.z_full, tube, state.grid, state.anchor_store
                        )
                        state.z = rr.z
                        # Refresh the stale KV cache for the repaired tokens.
                        state.cache = backbone.recompute_kv(
                            state.z, state.cond, rr.refreshed, state.cache
                        )
                    else:
                        state.z = state.anchor_store.rollback(state.z, tube)
                    trigger.register_rollback(tube.tube_id)
                    rollbacks_this_step += 1

                elif level == TriggerLevel.REPAIR:
                    # Medium risk: boundary-fuse the tube's drifting rim toward the
                    # freshly computed latent and pin it to FULL for one refresh step.
                    _log.debug(f"  Tube {tube.tube_id}: MEDIUM RISK ({cert_k:.3f}), repairing")
                    if result.z_full is not None:
                        rr = self.accelerator.raec.repair.repair(
                            state.z, result.z_full, tube, state.grid
                        )
                        state.z = rr.z
                        state.cache = backbone.recompute_kv(
                            state.z, state.cond, rr.refreshed, state.cache
                        )
                    if not trigger.is_forced_full(tube.tube_id):
                        trigger.register_repair(tube.tube_id)
                    repairs_this_step += 1

        trace.rollbacks = rollbacks_this_step
        trace.repairs = repairs_this_step

        # Drop the second full-size latent now that its consumers are done.
        result.z_full = None

        # Step 8: update anchor library (snapshot low-risk tubes). The gate is
        # ``tau_anchor``; a tube computed at acceptable risk with no anchor is also
        # seeded so the library is never empty when the first high-risk step arrives.
        for tube in state.tubes:
            tid = tube.tube_id
            cert_k = certificates.get(tid, 1.0)
            computed = result.compute_ratio > 0 and not optimal_actions.get(tid, Action.FULL).is_skip
            if computed and (cert_k <= self.trigger_cfg.tau_anchor or (
                self.trigger_cfg.seed_anchor_on_first_compute
                and computed
                and cert_k <= self.trigger_cfg.tau_high
                and not state.anchor_store.has(tid)
            )):
                state.anchor_store.update(tube, state.z, step_idx)

        return trace

    # ========================================================================= #
    # Helper methods for each step
    # ========================================================================= #

    def _decode_preview_frames(
        self, state: EngineState, backbone: BackboneAdapter
    ) -> Tensor:
        """Decode the current latent to RGB frames ``[grid.t, 3, Hp, Wp]`` in ``[0, 1]``
        for tube segmentation.

        Only the first batch element is decoded (the rest is discarded) since this is the
        single largest transient in the pass. A causal-temporal VAE expands ``T`` latent
        slots into more pixel frames, so the result is subsampled evenly back to
        ``grid.t`` representative frames.
        """
        with torch.no_grad():
            latent_grid = self.accelerator.backbone.to_grid(state.z[:1], state.grid)
            video = backbone.decode_to_unit(latent_grid)  # [1, 3, F, Hp, Wp] in [0,1]
        frames = video[0].permute(1, 0, 2, 3).contiguous()
        del video
        f = frames.shape[0]
        if f != state.grid.t:
            # Pick grid.t evenly-spaced frames (one per latent-temporal slot).
            sel = torch.linspace(0, f - 1, state.grid.t, device=frames.device).round().long()
            frames = frames.index_select(0, sel)
        return frames

    def _before_compute(self, state: EngineState, t: int) -> None:
        """Release a completed BPTT segment only when another forward will replace it.

        Anchors already store detached tensors. Repairs in skipped steps remain on
        the current segment; the window bounds denoiser forwards, not repair memory.
        """
        if state.grad_window <= 0 or state.retained_computed < state.grad_window:
            return
        _log.info("BPTT cut before step %d: retained=%d latent_grad=%s",
                  self.engine_cfg.num_inference_steps - t + 1,
                  state.retained_computed, state.z.requires_grad)
        state.z = state.z.detach()
        if state.cache is not None:
            state.cache = state.cache.detach()
        state.retained_computed = 0
        state.graph_cuts += 1

    def _warmup_step(
        self, state: EngineState, t: int, backbone: BackboneAdapter
    ) -> Tuple[Tensor, Optional[BackboneCache], float]:
        """Dense FULL denoising step used before tubes exist (cold-start).

        Computes the noise estimate on every token and takes one scheduler step,
        returning the advanced latent, the refreshed cache and the adapter-reported
        compute cost.
        """
        # The backbone denoises in sigma in (0, 1]; convert the descending step index.
        T = self.engine_cfg.num_inference_steps
        t_now = torch.full((state.z.shape[0],), sigma_from_step(t, T), device=state.z.device)
        t_next = torch.full((state.z.shape[0],), sigma_from_step(t - 1, T), device=state.z.device)
        self._before_compute(state, t)
        out = backbone.denoise(
            state.z, t_now, state.cond, grid=state.grid,
            active_mask=None, cache=state.cache,
        )
        z_next = backbone.scheduler_step(out.model_output, t_now, t_next, state.z)
        return z_next, out.cache, float(getattr(out, "compute_fraction", 1.0))

    def _assemble_anchor_latent(self, state: EngineState) -> Optional[Tensor]:
        """Build the per-token last-verified-safe latent ``[B, N, d]`` for ANCHOR reuse
        and the RAEC residual.

        Each tube with a stored safe anchor contributes its anchored tokens; all other
        tokens keep the current latent. Returns ``None`` when no tube has been anchored,
        in which case the transition falls back to a cheap reuse step.
        """
        anchored = [t for t in state.tubes if state.anchor_store.has(t.tube_id)]
        if not anchored:
            return None
        z_anchor = state.z.clone()
        for tube in anchored:
            state.anchor_store.scatter_into(z_anchor, tube)
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
        FULL/LOWFREQ/INTERP/ANCHOR per tube and returns the advanced latent, refreshed
        cache, active-token mask and per-tube residuals. The safe anchor latent is
        supplied so ANCHOR tubes freeze to it and the measured residual is meaningful.
        """
        # Model-space sigma in (0, 1] from the descending step index.
        T = self.engine_cfg.num_inference_steps
        t_now = torch.full((state.z.shape[0],), sigma_from_step(t, T), device=state.z.device)
        t_next = torch.full((state.z.shape[0],), sigma_from_step(t - 1, T), device=state.z.device)
        executor = self.accelerator.transition
        if executor.adapter is not backbone:
            raise ValueError("Transition backbone must match the engine's adapter")
        transition_cache = None if self.engine_cfg.diagnostic_no_cache else state.cache
        masks = executor.prepare_masks(
            decision, state.tubes, state.grid, device=state.z.device, cache=transition_cache
        )
        will_compute = bool(masks[1].any()) or transition_cache is None or transition_cache.model_output is None
        if will_compute:
            self._before_compute(state, t)
            transition_cache = None if self.engine_cfg.diagnostic_no_cache else state.cache
        else:
            _log.debug("step %d: reuse cache; preserving current gradient segment",
                       T - t + 1)
        return executor.step(
            z_t=state.z,
            t=t_now,
            t_next=t_next,
            cond=state.cond,
            decision=decision,
            tubes=state.tubes,
            grid=state.grid,
            cache=transition_cache,
            anchor_latent=self._assemble_anchor_latent(state),
            measure_residual=self.engine_cfg.measure_residual,
            prepared_masks=masks,
            fill_lowfreq=self.engine_cfg.lowfreq_fill,
        )

    def _counterfactual_check(
        self,
        state: EngineState,
        z_full: Tensor,
        actions: Dict[int, Action],
        strength_feats,
    ) -> Tuple[int, int]:
        """Run the single-hop counterfactual verification and local repair.

        For each tube the verifier flags (temporal-mutation point and skipped this
        step, capped per step), the executed skip latent is compared against the
        compute-everywhere reference ``z_full``; when the residual exceeds the
        threshold the residual-repair net corrects that tube's tokens.

        Corrections are written into a single working copy cloned lazily on the first
        repair, in place to keep the autograd graph intact for Stage-C training.

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
        z_work: Optional[Tensor] = None
        for tid in triggered:
            tube = tubes_by_id.get(tid)
            if tube is None:
                continue
            idx = tube.all_token_indices().to(state.z.device)
            if idx.numel() == 0:
                continue
            checks += 1
            src = z_work if z_work is not None else state.z
            rows: List[Tensor] = []
            changed = False
            for b in range(src.shape[0]):
                z_skip = src[b].index_select(0, idx)           # [n_tok, d]
                z_ref = z_full[b].index_select(0, idx)         # [n_tok, d]
                vr = verifier.verify_and_repair(tube, z_skip, z_ref)
                if vr.repaired and vr.z_corrected is not None:
                    rows.append(vr.z_corrected)
                    changed = True
                else:
                    rows.append(z_skip)
            if changed:
                if z_work is None:
                    z_work = state.z.clone()
                z_work[:, idx] = torch.stack(rows, dim=0).to(z_work.dtype)  # [B, n_tok, d]
                repairs += 1
        if z_work is not None:
            state.z = z_work
        return checks, repairs
