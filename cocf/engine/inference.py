"""Accelerated inference engine — the main denoising loop (§7.2).

The engine orchestrates a reverse denoising process from noisy latent z_T to clean
z_0, with dynamic per-tube compute allocation using the full COCF-SS-DCA pipeline:

    Iteration (t=T to t=1):
        1. Build/update semantic tubes G_t (STA)
        2. Extract 7-dim tube states s_{k,t} (STA)
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

        # Both are stateless and already built (with the executor's stride) by the
        # accelerator; constructing a second pair here meant a config edit could land
        # on one copy and not the other.
        self.budget_scheduler = accelerator.budget_scheduler
        self.action_allocator = accelerator.allocator

        # Tube builder (only called at specific steps)
        self.tube_builder = accelerator.tube_builder

        # One-shot flags: an adapter with no temporal-layout description disables the
        # windowed differentiable decode, and one that only describes prefix windows
        # pins the loss to the head of the clip. Both are worth saying once, not once
        # per batch.
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
            cond: Text conditioning (embeddings, etc.).
            backbone: Frozen backbone model.
            record_sink: Optional per-step callback ``(step_idx, t, budget, step_frac,
                tube_states, strength_feats, actions)`` used by Stage-C end-to-end
                fine-tuning to collect the exact per-tube features the engine allocated
                on (§4.2). ``None`` at inference (zero overhead).
            decode_grad: If True, run the whole trajectory **on** the autograd graph —
                the backbone is put in :meth:`~cocf.backbones.base.BackboneAdapter.grad_mode`
                so its forwards stop using ``inference_mode``, and the final decode is
                differentiable. This is the §4.2 path from the pixel/semantic loss back
                to the residual-repair net and any LoRA adapters. Inference leaves it
                False and keeps the cheaper ``inference_mode``/``no_grad`` passes.

        Returns:
            GenerationResult with final video, traces, efficiency stats.
        """
        # ``inference_mode`` is viral — a tensor produced under it can never join an
        # autograd graph, which is why Stage C used to train nothing despite asking for
        # a differentiable decode. Opening the backbone's grad mode for the whole span
        # is what makes ``decode_grad`` mean anything (§4.2).
        #
        # The peak-memory probe wraps the whole trajectory because §9.4 asks for 峰值显存
        # alongside the latency/FLOPs figures, and a saving that is really a
        # memory-for-time trade should be visible as one. It is a no-op off CUDA.
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

        # RAEC's trigger carries per-run bookkeeping (the force-FULL pins). Clearing it
        # here is what keeps the engine stateless across calls now that the pins live
        # in the module rather than in EngineState; ``reset()`` had no caller at all
        # before (§P1-7).
        self.accelerator.raec.reset()

        # Initialize state for this generation
        state = EngineState(
            z=z_init,
            grid=grid,
            cond=cond,
            subgraph=None,  # Built on-demand
            # AnchorStore's only ctor arg is offload_to_cpu; RAEC owns the factory.
            # The memory policy has to be *handed* to it — the engine used to call the
            # factory bare, so ``MemoryConfig.offload_backbone_to_cpu`` could never
            # reach the store and its CPU-offload path was unreachable config.
            anchor_store=self.accelerator.raec.new_anchor_store(
                self.accelerator.config.memory
            ),
            cache=None,
        )

        # Main denoising loop: t = T → 1
        num_steps = self.engine_cfg.num_inference_steps
        # Truncated BPTT (§4.2), counted in *computed* steps rather than wall-clock
        # ones. Activations are retained only by steps whose denoiser actually ran, so
        # that is what bounds memory — and it is also what carries gradient to the LoRA
        # adapters. A wall-clock window ("keep the last 4 steps") cuts the graph on an
        # accelerated trajectory that skipped its final steps, which is the *normal*
        # case here: a run computing 2 of 30 steps would train nothing at all while
        # every log line still said use_lora. 0 keeps the whole trajectory.
        window = int(getattr(self.engine_cfg, "grad_window_steps", 0))
        truncating = decode_grad and window > 0
        retained_computed = 0

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

            # Cut the graph once it holds ``window`` denoiser forwards (no-op at
            # inference). Skipped steps add no activations, so they never trigger it.
            # Never on the final step: truncation exists to bound what *subsequent*
            # steps accumulate, and cutting here would only discard the graph the
            # decode is about to need.
            if truncating and step_idx < num_steps - 1:
                if trace.compute_ratio > 0.0:
                    retained_computed += 1
                if retained_computed >= window:
                    state.z = state.z.detach()
                    if state.cache is not None:
                        state.cache = state.cache.detach()
                    retained_computed = 0

        # Decode final latent to video. The adapter owns the token<->grid layout
        # (`to_grid`) and the VAE decode (`decode_latent`).
        z0_grid = self.accelerator.backbone.to_grid(state.z, state.grid)  # [B, C, T, H, W]

        # The backbone's decoder is frozen, so we call it directly. Stage-C end-to-end
        # fine-tuning needs the decode on the autograd graph (the §4.2 pixel-loss path to
        # the LoRA / repair params), so it passes ``decode_grad=True``; inference keeps the
        # cheaper no_grad decode.
        frame_span: Optional[Tuple[int, int]] = None
        if decode_grad:
            lo, hi, frame_span = self._grad_decode_window(state.grid, backbone)
            video = backbone.decode_latent(z0_grid[:, :, lo:hi])  # grad-enabled
            self._warn_if_no_graph(video, state, window)
        else:
            with torch.no_grad():
                video = backbone.decode_latent(z0_grid)  # [B, 3, F, H, W]

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

        Returns the whole clip (and ``frame_span=None``) when windowing is off, when the
        window would cover everything anyway, or when the backbone can describe no window
        at all — see :meth:`BackboneAdapter.pixel_span` for why guessing that layout is
        not an option.

        The offset is drawn uniformly so the subsampled loss stays an unbiased estimator
        of the full-clip one across steps (it follows the global RNG, so a seeded run is
        reproducible) — but the draw is *offered* to the backbone rather than imposed on
        it. A causal-temporal VAE can only reproduce a sub-range of its own full decode
        when the slice starts at slot 0: given any later offset it has no feature cache
        of the preceding slots and re-anchors, returning a differently-sized window of
        different pixels. Such an adapter declines the random offset (``None``), and this
        falls back to the prefix ``[0, k)``, which it can honour exactly. Adapters with a
        uniform temporal layout (the mock) accept the random offset and keep the
        unbiasedness.
        """
        k = int(getattr(self.engine_cfg, "decode_grad_frames", 0))
        if k <= 0 or k >= grid.t:
            return 0, grid.t, None
        span_fn = getattr(backbone, "pixel_span", None)
        if callable(span_fn):
            lo = int(torch.randint(0, grid.t - k + 1, (1,)).item())
            # The random draw first, then the prefix — never the prefix twice.
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

        Not an error — a prefix window is the only one a causal decoder can align — but
        it does change the training signal: the §4.2 pixel / §6.3.2 semantic gradients
        then only ever reach frames ``[start, stop)``, so the tail of the clip trains on
        the schedule regularisers alone.
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
        """Say so loudly when a ``decode_grad`` render carries no gradient.

        Silently training nothing is the failure this whole path exists to prevent, and
        it has two innocent-looking causes worth telling apart:

        * no step inside the retained window ran the denoiser (a fully-cached
          trajectory touches no backbone weight, so there is nothing to differentiate);
        * the truncation cut the graph after the last computed step.

        Both leave Stage C training only its schedule regulariser while the logs look
        perfectly healthy.
        """
        if video.requires_grad:
            return
        computed = [t.step for t in state.traces if t.compute_ratio > 0.0]
        _log.warning(
            "decode_grad=True but the render carries no autograd graph — the §4.2 "
            "pixel/semantic losses cannot reach the backbone (LoRA / repair net). "
            "Denoiser forwards ran at step(s) %s of %d; the BPTT window retains %s "
            "computed step(s). Raise engine.grad_window_steps (0 = full trajectory) "
            "or lower the skip pressure (budget.b_min, engine.dense_step_skip_below).",
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
        """Execute one complete denoising step with full COCF pipeline (§7.2).

        This is the per-timestep orchestration that implements the 8-step workflow
        described in the design document.
        """
        trace = StepTrace(
            step=step_idx,
            mask_ratio=1.0,
            budget=1.0,
            predicted_cost=0.0,
            num_tubes=0,
            compute_ratio=1.0,
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
            # Pool each tube's CLIP visual embed off the *same* preview frames while
            # they are still in hand (§P4-4). These feed the certificate's local-CMSC
            # term every step; recomputing them per step would cost a perception
            # forward per tube per step, and they only change when tubes are rebuilt.
            state.tube_embeds = {
                tube.tube_id: tube_clip_embed(
                    frames_rgb, tube, state.grid, self.accelerator.perception
                )
                for tube in state.tubes
            }
            # Re-segmentation mints fresh tube ids (they are monotonic, so nothing
            # inherits a previous tube's anchor — §P1-10). Retire the anchors of tubes
            # that no longer exist so the store does not grow for the whole run.
            dropped = state.anchor_store.retain({t.tube_id for t in state.tubes})
            _log.debug(f"  Built {len(state.tubes)} semantic tubes "
                       f"({dropped} stale anchor(s) retired)")

        trace.num_tubes = len(state.tubes)

        # No tubes yet (cold-start warm-up): advance the latent with a dense FULL
        # step and return — there is nothing to allocate or certify.
        if not state.tubes:
            state.z, state.cache, warm_cost = self._warmup_step(state, t, backbone)
            trace.compute_ratio = warm_cost  # a warm-up step is a full dense forward
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
        # than allowed to skip again and stay frozen at a stale latent. The pins are
        # owned by :class:`~cocf.raec.trigger.RiskTrigger` — the engine used to keep a
        # second, inlined copy of the same policy in ``EngineState`` while that class
        # sat unused, so the two could (and did) drift apart (§P1-7).
        trigger = self.accelerator.raec.trigger
        forced_full = trigger.forced_full_tubes()
        # §6.3.1's local conservation proxy ``1 − align(tube, prompt)``: a tube the
        # prompt barely describes is one whose skip risks a semantic violation, so it
        # raises that tube's certificate through λ_cmsc (§5.3.1). One projection over
        # the cached tube embeds — no perception forward — so it is affordable every
        # step. This term used to be a hard zero at inference: the engine never passed
        # it and ``CMSCLoss.local_conservation``, written for exactly this, had no
        # caller anywhere (§P4-4).
        local_cmsc = self.accelerator.cmsc_loss.local_conservation(
            state.cond.embeds[0], state.tube_embeds
        ) if state.tube_embeds else {}
        # §2.2's hard risk constraint ``E_cert_k(a_k) ≤ τ_r``, evaluated *before* the
        # action is chosen. The allocator has always accepted this argument; nobody
        # ever passed it, so the constraint existed only in the docstring.
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
        # allocation has honoured it; the trigger drops windows that have elapsed.
        trigger.step()

        # --- Step 6: Execute tube actions (the latent transition) -----
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

        # A promoted whole-step skip executes something cheaper than the allocation
        # asked for. Adopt the executed actions from here on so the trace, the
        # certificate and the counterfactual check all describe what really happened
        # (§P0-1); an empty ``downgraded`` — the ordinary case — changes nothing.
        if result.downgraded:
            optimal_actions = {**optimal_actions, **result.downgraded}
            trace.actions = {k: Action(a).name for k, a in optimal_actions.items()}
            _log.debug("  step promoted to a whole-step skip; %d tube(s) downgraded",
                       len(result.downgraded))

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
        # Certify the action that was *actually executed* and feed it every §5.3.1 term
        # the loop can supply: the real skip residual δ_k = ‖z_full − z_action‖ measured
        # by the transition, the tube's boundary uncertainty (from the tube state), its
        # anchor age, and the local cross-modal violation computed above. This must run
        # after the transition: a pre-transition guess (prior action, zero residual)
        # decouples the risk trigger from the error it exists to catch.
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

        # --- Step 7: Risk triggers & local repairs (RAEC) -----
        repairs_this_step = 0
        rollbacks_this_step = 0

        if self.engine_cfg.risk_control_enabled:
            for tube in state.tubes:
                cert_k = certificates.get(tube.tube_id, 0.0)
                # One owner for the threshold policy: the engine used to inline the
                # same two comparisons that RiskTrigger implements, so editing either
                # left the other stale (§P1-7).
                level = trigger.classify_value(cert_k)

                if level == TriggerLevel.ROLLBACK:
                    # High risk: revoke the tube to its safe anchor, fuse the boundary
                    # against the compute-everywhere latent so the rolled-back interior
                    # joins its surroundings seam-free (§5.3.2 边界修复), and pin it to
                    # FULL for the next q steps so it is recomputed forward.
                    _log.debug(f"  Tube {tube.tube_id}: HIGH RISK ({cert_k:.3f}), rolling back")
                    if result.z_full is not None:
                        rr = self.accelerator.raec.repair.rollback(
                            state.z, result.z_full, tube, state.grid, state.anchor_store
                        )
                        state.z = rr.z
                        # Refresh the now-stale ε/KV cache for the repaired tokens
                        # (§5.3.2 缓存刷新; a no-op for backbones without a KV cache).
                        state.cache = backbone.recompute_kv(
                            state.z, state.cond, rr.refreshed, state.cache
                        )
                    else:
                        state.z = state.anchor_store.rollback(state.z, tube)
                    trigger.register_rollback(tube.tube_id)
                    rollbacks_this_step += 1

                elif level == TriggerLevel.REPAIR:
                    # Medium risk: boundary-fuse the tube's drifting rim toward the
                    # freshly computed latent (§5.3.2 边界修复/缓存刷新) and pin it to
                    # FULL for one refresh step so the region is recomputed rather than
                    # left to skip again.
                    _log.debug(f"  Tube {tube.tube_id}: MEDIUM RISK ({cert_k:.3f}), repairing")
                    if result.z_full is not None:
                        rr = self.accelerator.raec.repair.repair(
                            state.z, result.z_full, tube, state.grid
                        )
                        state.z = rr.z
                        state.cache = backbone.recompute_kv(
                            state.z, state.cond, rr.refreshed, state.cache
                        )
                    # One refresh step so the region is recomputed rather than left to
                    # skip again (a rollback's longer window takes precedence).
                    if not trigger.is_forced_full(tube.tube_id):
                        trigger.register_repair(tube.tube_id)
                    repairs_this_step += 1

        trace.rollbacks = rollbacks_this_step
        trace.repairs = repairs_this_step

        # ``z_full`` was the last consumer of a second full-size latent: it is needed by
        # the residual measurement, the §3.3.4 check and RAEC's boundary fusion, all of
        # which are done by here. Dropping the reference now means the anchor update and
        # the next step's forward do not run with it still resident (§P4-B4).
        result.z_full = None

        # --- Step 8: Update anchor library (snapshot low-risk tubes) -----
        # The gate is ``tau_anchor``, *not* ``tau_low``: sharing the trigger's lower
        # bound made the anchor library a hostage of certificate calibration — with a
        # cold-start certificate above τ_low nothing was ever anchored, so every
        # rollback silently did nothing while still pinning the tube to FULL (§5.3.2).
        # A tube computed at acceptable risk with no anchor at all is also seeded, so
        # the library is never empty when the first high-risk step arrives.
        for tube in state.tubes:
            tid = tube.tube_id
            cert_k = certificates.get(tid, 1.0)
            computed = not optimal_actions.get(tid, Action.FULL).is_skip
            if cert_k <= self.trigger_cfg.tau_anchor or (
                self.trigger_cfg.seed_anchor_on_first_compute
                and computed
                and cert_k <= self.trigger_cfg.tau_high
                and not state.anchor_store.has(tid)
            ):
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

        Only the **first batch element** is decoded. The result is indexed as
        ``video[0]`` and the rest discarded, so decoding the whole batch bought
        nothing — and this is the single largest transient in the pass: an untiled
        49×480×832 decode needs one ~7.7 GiB block (see
        :meth:`DiffusersVideoBackbone._configure_vae_memory`), which is a strange
        thing to allocate inside a loop whose purpose is saving memory (§P2-1).

        A real backbone's causal-temporal VAE expands the latent's ``T`` slots into
        ``(T-1)·c_t + 1`` pixel frames, so the decoded video generally has *more*
        frames than ``grid.t``. We subsample evenly back to ``grid.t`` representative
        frames (the mock keeps F == grid.t, so this is a no-op there).
        """
        with torch.no_grad():
            latent_grid = self.accelerator.backbone.to_grid(state.z[:1], state.grid)
            video = backbone.decode_latent(latent_grid)  # [1, 3, F, Hp, Wp]
        frames = video[0].permute(1, 0, 2, 3).contiguous()
        del video
        f = frames.shape[0]
        if f != state.grid.t:
            # Pick grid.t evenly-spaced frames (one per latent-temporal slot).
            sel = torch.linspace(0, f - 1, state.grid.t, device=frames.device).round().long()
            frames = frames.index_select(0, sel)
        return frames

    def _warmup_step(
        self, state: EngineState, t: int, backbone: BackboneAdapter
    ) -> Tuple[Tensor, Optional[BackboneCache], float]:
        """Dense FULL denoising step used before tubes exist (cold-start).

        Computes ε on every token and takes a single scheduler step, returning the
        advanced latent, the refreshed ε cache and the adapter-reported compute cost
        of the forward (always a full dense pass, hence ~1.0 — reported rather than
        assumed so the trace stays sourced from the adapter).
        """
        # The backbone denoises in σ∈(0,1] (see sigma_from_step); the loop counts the
        # step index down, so convert before handing t to the model.
        T = self.engine_cfg.num_inference_steps
        t_now = torch.full((state.z.shape[0],), sigma_from_step(t, T), device=state.z.device)
        t_next = torch.full((state.z.shape[0],), sigma_from_step(t - 1, T), device=state.z.device)
        out = backbone.denoise(
            state.z, t_now, state.cond, grid=state.grid,
            active_mask=None, cache=state.cache,
        )
        z_next = backbone.scheduler_step(out.model_output, t_now, t_next, state.z)
        return z_next, out.cache, float(getattr(out, "compute_fraction", 1.0))

    def _assemble_anchor_latent(self, state: EngineState) -> Optional[Tensor]:
        """Build the per-token "last verified-safe" latent ``[B, N, d]`` for ANCHOR
        reuse and the RAEC residual (§5.3.1).

        Each tube that has a stored safe anchor contributes its anchored tokens;
        all other tokens keep the current latent. Returns ``None`` when no tube has
        been anchored yet, in which case the transition falls back to a cheap
        ε-reuse step (and reports a zero residual, correctly — there is no safe
        reference to deviate from).

        One clone, then a scatter per tube: chaining ``rollback`` allocated a fresh
        full latent for *every* anchored tube and dropped the previous one, K times a
        step (§P2-2).
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
        FULL/LOWFREQ/INTERP/ANCHOR per tube and returns the advanced latent, the
        refreshed cache, the active-token mask and per-tube residuals. The safe
        anchor latent is supplied so ANCHOR tubes freeze to it and the measured
        residual ``‖z_full − z_anchor‖`` is meaningful (the RAEC trigger signal).
        """
        # Model-space σ∈(0,1] from the descending step index (see sigma_from_step).
        T = self.engine_cfg.num_inference_steps
        t_now = torch.full((state.z.shape[0],), sigma_from_step(t, T), device=state.z.device)
        t_next = torch.full((state.z.shape[0],), sigma_from_step(t - 1, T), device=state.z.device)
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
        corrects that tube's tokens (do(¬skip) causal-omission repair, §3.3.4).

        The corrections are written into a **single** working copy of the latent, cloned
        lazily on the first actual repair: cloning per repaired tube allocated and threw
        away a full latent up to ``max_checks_per_step`` times a step, the same churn
        :meth:`_assemble_anchor_latent` was fixed for (§P2-2). Each tube still *reads*
        the running copy, so a later tube overlapping an earlier one sees the earlier
        correction exactly as before. The writes are in-place on the clone, which keeps
        the autograd graph intact (Stage-C repair-net training) because nothing in the
        chain — ``clone`` / ``index_select`` / ``index_put_`` — saves the mutated values
        for backward.

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
        z_work: Optional[Tensor] = None  # cloned on the first repair, not before
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
