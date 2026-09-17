"""Action-aware transition executor — where the FLOPs/VRAM saving is realised.

The accelerator decides *what* to do per tube (the :class:`AllocationDecision`);
this module turns that decision into an actual latent update ``z_t -> z_{t-1}``
while computing the denoiser ε_θ on as few tokens as possible.

It is written entirely against the backbone-agnostic :class:`BackboneAdapter`
contract, so the four compute actions are defined *once* here and work for every
backbone (user requirement #2). The mapping from action → which tokens are
"active" (freshly computed) is the single lever that produces the speed-up
(user requirement #1):

    FULL     all of the tube's tokens are active            cost ∝ 1.00·|g_k|
    LOWFREQ  a strided spatial subset is active, the rest    cost ∝ 1/stride²·|g_k|
             are upsampled from the computed neighbours      (0.25 at the default 2)
    INTERP   no token is active; the tube's tokens are       cost ∝ 0.02·|g_k|
             temporally interpolated from the frames         (no denoiser, but a
             around them (``interp_rows``)                    latent gather+blend)
    ANCHOR   no token is active; the latent is frozen to     cost ∝ 0.00
             the last verified-safe anchor verbatim (or to
             the pre-step latent when none exists yet)

Only FULL/LOWFREQ tokens enter ``adapter.denoise(active_mask=…)``. What that buys
depends entirely on the adapter: the mock (and any future sparse-attention kernel)
genuinely gathers the active tokens and computes only those; today's real video-DiTs
have no arbitrary-token-sparse attention, so they compute densely and only *splice*
the inactive outputs from cache. The saving that is real on every backbone is the
**whole-step skip** — when no token at all is active the transformer never runs.
Each adapter reports what it truly spent (``DenoiseOutput.compute_fraction``), and
:class:`TransitionResult` carries that through as ``compute_ratio``; the mask
occupancy travels separately as ``mask_ratio`` and must never be quoted as a saving.

Design note — axis conventions
------------------------------
``ANCHOR`` reuses across the *denoising-step* axis (skip this step, keep the last
latent), matching cache methods like DeepCache/TeaCache. ``INTERP`` reuses across
the *frame* axis: the tube's tokens are rebuilt from the frames on either side of
them, so the tube keeps moving with the global trajectory without being denoised.
``LOWFREQ`` keeps only a strided spatial lattice fresh; the rest of the tube
rides the spliced-ε Euler step (TeaCache-style) rather than being overwritten
with a reconstruction — reconstruction fills empirically corrupt the trajectory
(see :meth:`TransitionExecutor.coarsen_lowfreq`). These choices are local to
this file —
the rest of the framework only sees "an action was executed" — but both cheap
reconstructions (:meth:`TransitionExecutor.coarsen_lowfreq`,
:meth:`TransitionExecutor.interp_temporal`) are *shared verbatim* with Stage-A
teacher generation, so the damage labels describe the operation inference performs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from cocf.backbones.base import BackboneAdapter, BackboneCache, TextConditioning
from cocf.common.logging import get_logger
from cocf.common.types import Action, AllocationDecision, SemanticTube, TokenGrid

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class TransitionResult:
    """Output of one accelerated step (§7.2 step 6)."""

    z_next: Tensor  # [B, N, d] latent at t_next
    cache: BackboneCache  # refreshed ε_θ cache (full-resolution, for reuse)
    active_mask: Tensor  # [N] bool — tokens the allocation *planned* to compute
    # per-tube residual δ_k = ‖z_full − z_action‖ on the tube, for RAEC (§5.3.1).
    # Only populated for tubes that skipped (INTERP/ANCHOR) when ``measure_residual``.
    tube_residual: Dict[int, float]
    # |active| / N — the *plan*: what fraction of tokens the allocator marked fresh.
    # This is a mask-occupancy statistic and NOT an efficiency metric: an adapter
    # without a token-sparse attention kernel computes densely regardless of it.
    mask_ratio: float
    # What the denoiser actually spent this step, as a fraction of a dense forward,
    # self-reported by the adapter (:attr:`DenoiseOutput.compute_fraction`). This is
    # the number to quote as a saving — 0.0 on a whole-step skip, 1.0 on a dense
    # forward, |active|/N only where the adapter is genuinely sparse.
    compute_ratio: float
    # the "compute-everywhere" candidate latent z_full (cached-ε scheduler step).
    # Returned so RAEC boundary fusion (§5.3.2) and the single-hop counterfactual
    # check (§3.3.4) have the full-compute reference without recomputing it.
    z_full: Optional[Tensor] = None
    # Tubes whose *executed* action differs from the allocated one because the step
    # was promoted to a whole-step skip (``dense_step_skip_below``). Empty on every
    # ordinary step. Reported so the trace and the certificate describe what actually
    # happened rather than what was planned.
    downgraded: Dict[int, Action] = field(default_factory=dict)

    @property
    def active_ratio(self) -> float:
        """Deprecated alias of :attr:`mask_ratio`.

        Kept so external callers do not break, but renamed at the source: it was
        being reported as a FLOPs saving while measuring only how much of the token
        grid the allocator *wanted* recomputed. Use :attr:`compute_ratio` for cost.
        """
        return self.mask_ratio


class TransitionExecutor:
    """Executes a per-tube :class:`AllocationDecision` into a latent update.

    Stateless w.r.t. the diffusion trajectory: all evolving state (the anchor
    latent, the ε cache) is passed in and returned, so the engine owns memory and
    this class stays trivially testable and reusable.
    """

    def __init__(
        self,
        adapter: BackboneAdapter,
        lowfreq_stride: int = 2,
        dense_step_skip_below: float = 0.0,
        background_refresh_every: int = 0,
        max_unmeasured_steps: int = 0,
    ) -> None:
        self.adapter = adapter
        # Recompute the un-tubed background every N steps (0 = never, the old
        # behaviour: it stayed at the warm-up step's ε for the whole trajectory).
        self.background_refresh_every = max(0, int(background_refresh_every))
        # spatial stride for LOWFREQ active subsampling (2 ⇒ ~1/4 tokens computed)
        self.lowfreq_stride = max(1, int(lowfreq_stride))
        # On an adapter without token-sparse attention, a mask occupancy at or below
        # this fraction buys nothing: the dense forward runs at full price either way.
        # Promoting such a step to a *whole-step skip* converts the plan into an
        # actual saving. 0 disables the promotion (always pay for the dense forward).
        self.dense_step_skip_below = max(0.0, float(dense_step_skip_below))
        # Ceiling on how many consecutive steps a tube may go without a measured skip
        # residual. This is what makes the promotion safe enough to enable by default
        # (§P4-A2): see :meth:`_maybe_promote_to_step_skip`. 0 = no bound.
        self.max_unmeasured_steps = max(0, int(max_unmeasured_steps))
        # Set by the engine so the executor can consult (and update) the per-tube
        # certificate-coverage counters. ``None`` ⇒ the invariant is inactive.
        self.risk_trigger = None

    # ------------------------------------------------------------------ #
    # Active-mask construction (the FLOPs lever)
    # ------------------------------------------------------------------ #

    def build_active_mask(
        self,
        decision: AllocationDecision,
        tubes: List[SemanticTube],
        grid: TokenGrid,
        *,
        device,
    ) -> Tensor:
        """Bool ``[N]`` mask of tokens to freshly compute this step.

        FULL marks every tube token active; LOWFREQ marks a strided spatial subset;
        INTERP/ANCHOR mark none.

        Tokens covered by *no* tube — the background, typically ~3/4 of the grid — are
        the cheapest to reuse and the axioms (§3.2.2) say their causal effect is
        approximately constant. "Approximately constant" is not "reusable for thirty
        steps", though: starting from an all-zero mask meant the background's ε was
        taken from the single warm-up step for the entire trajectory, with no refresh
        cadence, no certificate and no rollback covering it — every RAEC safety
        mechanism is scoped to tubes (§P1-8). ``background_refresh_every`` recomputes
        it periodically; on a dense-only adapter this is free (the forward runs at full
        price regardless), and on a token-sparse one it is the cost of not letting the
        largest region of the frame drift unmonitored.
        """
        mask = torch.zeros(grid.num_tokens, dtype=torch.bool, device=device)
        for tube in tubes:
            action = decision.action_for(tube.tube_id, default=Action.FULL)
            if action == Action.FULL:
                mask[tube.all_token_indices().to(device)] = True
            elif action == Action.LOWFREQ:
                mask[self._strided_indices(tube, grid).to(device)] = True
            # INTERP / ANCHOR contribute no active tokens
        if self._refresh_background_at(decision.step):
            covered = torch.zeros_like(mask)
            for tube in tubes:
                covered[tube.all_token_indices().to(device)] = True
            mask |= ~covered
        return mask

    def _refresh_background_at(self, step: int) -> bool:
        """Whether this step recomputes the un-tubed background (§P1-8)."""
        n = self.background_refresh_every
        return n > 0 and step > 0 and step % n == 0

    def _maybe_promote_to_step_skip(
        self,
        active_mask: Tensor,
        cache: Optional[BackboneCache],
        decision: AllocationDecision,
        tubes: List[SemanticTube],
    ) -> Tensor:
        """Turn a near-empty allocation into a genuine whole-step skip.

        On an adapter that cannot compute a token subset (``supports_token_sparsity
        is False`` — every real video-DiT today) a mask with 3% of the tokens set
        costs exactly as much as a mask with 100% set: the transformer runs densely
        and the mask only decides which outputs are kept. Paying that full price to
        refresh a handful of tokens is the worst of both worlds, so when the plan
        falls at or below ``dense_step_skip_below`` we clear the mask entirely, which
        :meth:`BackboneAdapter.denoise` then answers from cache without touching the
        transformer — the one saving that is real on every backbone (§5.1).

        **A FULL allocation vetoes the promotion, however small the tube.** FULL is
        how the system expresses "this region must be recomputed": the allocator
        assigns it to unstable tubes (§4.3.1) and RAEC pins it there for ``q`` steps
        after a rollback (§5.3.2). Dropping the forward because such a tube happens to
        be tiny would quietly cancel the safety mechanism that asked for it — the
        opposite of what a cheap throughput heuristic is allowed to do.

        **Certificate coverage bounds it.** A promotion downgrades every LOWFREQ tube
        to plain cache reuse, and since nothing was computed there is no reference to
        measure those tubes' skip residual against: the certificate sees δ=0 and cannot
        price the extra error. That used to make the whole heuristic opt-in. It is now
        bounded instead: :class:`~cocf.raec.trigger.RiskTrigger` counts how many
        consecutive steps each tube has gone without a measured δ, and once any tube
        reaches ``max_unmeasured_steps`` the promotion is vetoed so the forward runs and
        every residual is re-grounded. The downgrade is still reported through
        :attr:`TransitionResult.downgraded` so traces never claim LOWFREQ ran.

        No-op when the adapter *is* sparse (the mask is already proportional to cost),
        when the threshold is 0, when the mask is empty or full, or when there is no
        cache to answer from.
        """
        if self.adapter.supports_token_sparsity or self.dense_step_skip_below <= 0.0:
            return active_mask
        if cache is None or cache.model_output is None:
            return active_mask  # nothing to reuse — the forward must run
        if any(
            decision.action_for(t.tube_id, default=Action.FULL) == Action.FULL
            for t in tubes
        ):
            return active_mask  # a tube demanded full compute; honour it
        if self.risk_trigger is not None and self.risk_trigger.coverage_exhausted(
            self.max_unmeasured_steps
        ):
            _log.debug(
                "transition: a tube has gone %d steps without a measured skip residual "
                "— running the forward so RAEC's certificate is re-grounded",
                self.max_unmeasured_steps,
            )
            return active_mask
        occupancy = float(active_mask.float().mean().item())
        if 0.0 < occupancy <= self.dense_step_skip_below:
            _log.debug(
                "transition: mask occupancy %.4f ≤ %.4f on a dense-only adapter with no "
                "FULL tube — promoting to a whole-step skip (a partial dense forward "
                "saves nothing)", occupancy, self.dense_step_skip_below,
            )
            return torch.zeros_like(active_mask)
        return active_mask

    def _strided_indices(self, tube: SemanticTube, grid: TokenGrid) -> Tensor:
        """Spatially strided subset of a tube's tokens for LOWFREQ compute.

        We keep every ``stride``-th column/row *within each frame* so the kept
        tokens form a coarse grid that bilinear upsampling can reconstruct from.
        """
        stride = self.lowfreq_stride
        kept: List[Tensor] = []
        for frame, idx in tube.tokens_by_frame.items():
            if stride == 1:
                kept.append(idx)
                continue
            # decode (hi, wi) of each flat index, keep those on the coarse lattice
            local = idx - frame * grid.tokens_per_frame
            hi = torch.div(local, grid.w, rounding_mode="floor")
            wi = local - hi * grid.w
            keep = (hi % stride == 0) & (wi % stride == 0)
            kept.append(idx[keep])
        if not kept:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(kept)

    # ------------------------------------------------------------------ #
    # The step
    # ------------------------------------------------------------------ #

    def prepare_masks(self, decision, tubes, grid, *, device, cache=None):
        """Resolve promotion once so gradient management sees the executed mask."""
        planned = self.build_active_mask(decision, tubes, grid, device=device)
        active = self._maybe_promote_to_step_skip(planned, cache, decision, tubes)
        return planned, active

    def step(
        self,
        z_t: Tensor,
        t: Tensor,
        t_next: Tensor,
        cond: TextConditioning,
        decision: AllocationDecision,
        tubes: List[SemanticTube],
        grid: TokenGrid,
        *,
        cache: Optional[BackboneCache] = None,
        anchor_latent: Optional[Tensor] = None,
        want_attention: bool = False,
        measure_residual: bool = False,
        prepared_masks: Optional[Tuple[Tensor, Tensor]] = None,
        fill_lowfreq: bool = True,
    ) -> TransitionResult:
        """Advance ``z_t`` to ``z_{t_next}`` honouring the per-tube allocation.

        Parameters
        ----------
        anchor_latent
            ``[B, N, d]`` last verified-safe latent per token (for ANCHOR reuse and
            the RAEC residual). If ``None``, ANCHOR falls back to a cheap ε-reuse
            step (still correct, just not frozen).
        measure_residual
            When True, also compute ‖z_full − z_action‖ per skipped tube against a
            *cache-reused full step* reference, feeding the error certificate.
        fill_lowfreq
            Ablation switch (``EngineConfig.lowfreq_fill``, default off). When
            False — the served default — LOWFREQ holes keep their ``z_full``
            value (riding the spliced-ε Euler step, TeaCache-style). When True
            they are overwritten with the lattice reconstruction, which is
            retained only for comparison runs.
        """
        device = z_t.device
        planned_mask, active_mask = prepared_masks if prepared_masks is not None else self.prepare_masks(
            decision, tubes, grid, device=device, cache=cache
        )
        # A promotion cleared a non-empty plan: every tube that was going to be
        # refreshed is now reusing cache. Record it so the caller certifies and logs
        # the action that ran, not the one that was allocated.
        downgraded: Dict[int, Action] = {}
        if active_mask is not planned_mask:
            downgraded = {
                t.tube_id: Action.ANCHOR
                for t in tubes
                if decision.action_for(t.tube_id, default=Action.FULL) == Action.LOWFREQ
            }

        # 1) Denoise only the active tokens; the adapter splices inactive ε from cache.
        out = self.adapter.denoise(
            z_t, t, cond, grid=grid, active_mask=active_mask,
            cache=cache, want_attention=want_attention,
        )
        # ``out.model_output``, *not* ``out.cache.model_output``: they carry identical
        # values, but adapters detach the copy they hand back as cache (it is reuse
        # state, and keeping a graph on it would chain every step's activations
        # together forever). Stepping on the detached copy severs Stage C's §4.2 loss
        # from the denoiser on exactly the backbones that detach — i.e. the real ones.
        eps = out.model_output  # [B, N, d_out] full-resolution ε (active fresh)

        # 2) A single scheduler step gives the "compute everywhere" candidate.
        z_full = self.adapter.scheduler_step(eps, t, t_next, z_t)

        # Did the denoiser actually run? LOWFREQ's reconstruction upsamples *from the
        # freshly computed lattice*; when the transformer was skipped (an empty mask,
        # or a promoted step skip) there is no such lattice — every token in z_full
        # came from the same cached ε — so smearing lattice values over the holes
        # would only destroy detail in exchange for nothing. INTERP/ANCHOR are pure
        # latent-space operations and stay meaningful either way.
        computed = float(getattr(out, "compute_fraction", 1.0)) > 0.0

        # 3) Per-action latent reconstruction for the skipped tubes.
        skips = sorted(
            (t for t in tubes
             if decision.action_for(t.tube_id, default=Action.FULL).is_skip),
            key=lambda t: -int(decision.action_for(t.tube_id, default=Action.FULL)),
        )
        lowfreq = [
            t for t in tubes
            if decision.action_for(t.tube_id, default=Action.FULL) == Action.LOWFREQ
        ]
        # Nothing to write back ⇒ ``z_next`` *is* the compute-everywhere latent, and
        # cloning it would allocate a second full latent per step for an exact copy
        # (§P4-B4). The clone is only needed once a skip or a LOWFREQ fill is about to
        # mutate it — z_full has to survive intact for the certificate's residual, the
        # §3.3.4 check and RAEC's boundary fusion.
        do_fill = fill_lowfreq and computed and self.lowfreq_stride > 1
        needs_write = bool(skips) or bool(lowfreq and do_fill)
        z_next = z_full.clone() if needs_write else z_full
        tube_residual: Dict[int, float] = {}
        if do_fill:
            for tube in lowfreq:
                reconstructed = self.coarsen_lowfreq(
                    z_full, tube, grid, protected=active_mask
                )
                idx = tube.all_token_indices().to(device)
                keep = ~active_mask.to(device).index_select(0, idx)
                z_next.index_copy_(1, idx[keep], reconstructed.index_select(1, idx[keep]))

        # Skip write-backs are applied *after* every computed tube and in a fixed
        # severity order, so an overlap does not resolve by list position (§P1-6).
        # Two rules make the result order-independent: a token that was freshly
        # computed for some other tube is never overwritten by a skip, and where two
        # skips overlap the less destructive one (INTERP, which still moves with the
        # trajectory) is written last and wins over ANCHOR's freeze.
        active_now = active_mask.to(device)
        rode_tokens = 0
        for tube in skips:
            action = decision.action_for(tube.tube_id, default=Action.FULL)
            idx = tube.all_token_indices().to(device)
            if action == Action.ANCHOR and anchor_latent is not None:
                z_skip = anchor_latent.index_select(1, idx)
            elif action == Action.INTERP:
                # Real temporal interpolation (see :meth:`interp_rows`). When the tube
                # spans a single frame there is nothing to interpolate between — and
                # freezing at ``z_t`` pins the tokens at this step's noise level for
                # the rest of the trajectory (the plan never revisits them), which
                # decodes as solid noise blocks. Ride the spliced-ε Euler step
                # instead: ``z_full`` already advanced these tokens with the cached
                # velocity, which is the TeaCache-style skip and always keeps σ moving.
                rows = self.interp_rows(z_full, tube, grid)
                if rows is None:
                    rode_tokens += int(idx.numel())
                    continue
                z_skip = rows
            else:
                # ANCHOR with no stored anchor: same reasoning as the single-frame
                # INTERP fallback above — never freeze, ride the stepped latent.
                rode_tokens += int(idx.numel())
                continue
            if measure_residual:
                ref = z_full.index_select(1, idx)
                tube_residual[tube.tube_id] = float(
                    (ref - z_skip).pow(2).mean().sqrt().item()
                )
            # Never clobber a token another tube had freshly computed: it holds a real
            # denoiser output, which is strictly better than any skip reconstruction.
            keep = ~active_now.index_select(0, idx)
            if bool(keep.all()):
                z_next.index_copy_(1, idx, z_skip.to(z_next.dtype))
            elif bool(keep.any()):
                sel = keep.nonzero(as_tuple=True)[0]
                z_next.index_copy_(
                    1, idx.index_select(0, sel),
                    z_skip.index_select(1, sel).to(z_next.dtype),
                )

        if rode_tokens:
            _log.info(
                "transition: %d token(s) rode the spliced-ε step (freeze disabled)",
                rode_tokens,
            )

        active_ratio = float(active_mask.float().mean().item())
        # Certificate coverage (§P4-A2). A tube counts as *certified* this step when its
        # error can actually be priced: either the denoiser recomputed it (FULL, or a
        # LOWFREQ whose lattice really ran) or the transition measured its skip residual
        # δ. Everything else went a step unpriced.
        #
        # The executed action is what matters, not the allocated one — a promoted
        # whole-step skip turns LOWFREQ tubes into plain cache reuse, and those are
        # precisely the tubes the invariant exists to bound. Reading ``decision`` alone
        # would have missed every one of them.
        if self.risk_trigger is not None:
            certified, blind = [], []
            for tube in tubes:
                tid = tube.tube_id
                executed = downgraded.get(tid, decision.action_for(tid, default=Action.FULL))
                if executed == Action.FULL or (executed == Action.LOWFREQ and computed):
                    certified.append(tid)
                elif tid in tube_residual:
                    certified.append(tid)
                else:
                    blind.append(tid)
            self.risk_trigger.note_measured(certified)
            self.risk_trigger.note_unmeasured(blind)
        return TransitionResult(
            z_next=z_next,
            cache=out.cache,
            active_mask=active_mask,
            tube_residual=tube_residual,
            mask_ratio=active_ratio,
            # What the adapter says it spent — not what the mask implies (§P0-1).
            compute_ratio=float(getattr(out, "compute_fraction", 1.0)),
            z_full=z_full,
            downgraded=downgraded,
        )

    def coarsen_lowfreq(
        self, z: Tensor, tube: SemanticTube, grid: TokenGrid,
        *, protected: Optional[Tensor] = None,
    ) -> Tensor:
        """Coarsen the latent itself: holes take their lattice anchor's value.

        .. deprecated:: served-path semantics

        Inference now defaults to ride-through for LOWFREQ holes (they keep their
        ``z_full`` value; see ``EngineConfig.lowfreq_fill``), because overwriting
        holes with reconstructed values — in *any* flavour — shifts the joint
        latent distribution that the next dense forward attends over and the
        corruption compounds across steps (empirically: saturated noise on Wan2.2,
        PSNR 8 dB vs 19.8 dB for ride-through against the all-FULL reference).
        This reconstruction survives for two uses: the ``--lowfreq-fill``
        ablation, and Stage-A LOWFREQ damage labels in `cocf.lcocf.data` — note
        those labels therefore describe the fill, not the served ride-through,
        until the store is regenerated.

        An earlier revision coarsened the per-step *increment* instead
        (``z_prev + anchor delta``), which preserves each hole's own noise
        realisation. That is exactly wrong over a multi-step trajectory: a hole
        never receives its own denoiser output, so its step-1 noise offset
        against the anchor is frozen into the latent for the rest of the
        trajectory (telescoping gives ``hole_T = hole_1 + anchor_T − anchor_1``,
        and at σ≈0.95 that offset is near-full-strength noise) and decodes as
        saturated noise blocks. Copying the anchor's value lets the hole's
        noise die together with the anchor's; the price is stride²-token flat
        patches, which is precisely the quality cost LOWFREQ is meant to pay.

        With ``lowfreq_stride == 1`` LOWFREQ computes every token, so coarsening
        is a no-op.
        """
        if self.lowfreq_stride <= 1:
            return z
        out = z.clone()
        self._fill_lowfreq(out, z, tube, grid, protected=protected)
        # Positions the fill skipped (protected, or snapped outside an irregular
        # tube) keep their input value exactly.
        return out

    def interp_temporal(
        self, z: Tensor, tube: SemanticTube, grid: TokenGrid,
        freeze_to: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a copy of ``z`` with ``tube``'s tokens temporally interpolated.

        The INTERP counterpart of :meth:`coarsen_lowfreq`, and shared with Stage-A
        teacher generation for the same reason: the offline damage label must be
        produced by the *exact* operation inference performs, or the predictor learns
        the wrong μ for the action (§7.1.1 no train/serve skew).

        ``freeze_to`` supplies the pre-step latent used for the single-frame fallback
        (a tube spanning one frame has nothing to interpolate *between*). Inference
        no longer freezes in that case — :meth:`TransitionExecutor.step` lets the
        tube ride the spliced-ε step, because a freeze pins σ in place for the rest
        of the trajectory and decodes as noise blocks. The label side keeps the
        freeze semantics when ``freeze_to`` is passed, so for single-frame tubes the
        predictor's INTERP μ is a conservative overestimate of the served damage.
        """
        rows = self.interp_rows(z, tube, grid)
        idx = tube.all_token_indices().to(z.device)
        if rows is None:
            if freeze_to is None or idx.numel() == 0:
                return z
            rows = freeze_to.index_select(1, idx)
        out = z.clone()
        out.index_copy_(1, idx, rows.to(out.dtype))
        return out

    def interp_rows(
        self, z: Tensor, tube: SemanticTube, grid: TokenGrid
    ) -> Optional[Tensor]:
        """INTERP's actual latent operation: ``[B, |g_k|, d]`` interpolated tube rows.

        For every frame the tube spans, its tokens are replaced by a *temporal* blend
        of the tube's content at the neighbouring frames, taken at the **same spatial
        position** — i.e. "do not denoise this frame, infer it from the frames around
        it". Endpoint frames have one neighbour and copy it.

        This is what makes INTERP an action at all. It used to read its own tokens back
        out of the freshly-stepped latent (``z_skip = z_full[idx]``), which is an
        identity: the latent was unchanged, the skip residual δ_k was *identically
        zero*, and with it the certificate's λ_res term, the §3.3.4 counterfactual
        check and every repair that depends on them (§P1-1). Blending across time
        preserves within-frame structure while introducing the temporal lag that is the
        genuine cost of skipping — the damage the predictor is supposed to learn.

        Returns ``None`` when the tube spans fewer than two frames (nothing to
        interpolate from), leaving the caller to fall back.
        """
        frames = tube.frames
        if len(frames) < 2:
            return None
        device = z.device
        per_frame = grid.tokens_per_frame
        rows: List[Tensor] = []
        for i, f in enumerate(frames):
            idx = tube.tokens_by_frame[f].to(device)
            local = idx - f * per_frame  # spatial offset, identical across frames
            lo = frames[i - 1] if i > 0 else None
            hi = frames[i + 1] if i + 1 < len(frames) else None
            if lo is None:  # first frame: hold the only neighbour it has
                rows.append(z.index_select(1, hi * per_frame + local))
            elif hi is None:  # last frame: likewise
                rows.append(z.index_select(1, lo * per_frame + local))
            else:
                w = (f - lo) / float(hi - lo)
                a = z.index_select(1, lo * per_frame + local)
                b = z.index_select(1, hi * per_frame + local)
                rows.append(a * (1.0 - w) + b * w)
        return torch.cat(rows, dim=1)

    def _fill_lowfreq(
        self, z_next: Tensor, z_full: Tensor, tube: SemanticTube, grid: TokenGrid,
        *, protected: Optional[Tensor] = None,
    ) -> None:
        """Fill the uncomputed LOWFREQ positions of ``z_next`` from ``z_full``.

        The holes of ``z_next`` are overwritten by nearest-neighbour copy of the
        kept lattice values from ``z_full`` within the same frame. Both tensors
        carry absolute latent values (see :meth:`coarsen_lowfreq` for why
        coarsening increments instead is wrong).
        """
        stride = self.lowfreq_stride
        device = z_next.device
        for frame, idx in tube.tokens_by_frame.items():
            local = idx.to(device) - frame * grid.tokens_per_frame
            hi = torch.div(local, grid.w, rounding_mode="floor")
            wi = local - hi * grid.w
            kept = (hi % stride == 0) & (wi % stride == 0)
            if kept.all():
                continue
            # snap each hole to its kept lattice anchor (floor to multiple of stride)
            hi_a = (torch.div(hi, stride, rounding_mode="floor") * stride).clamp_max(grid.h - 1)
            wi_a = (torch.div(wi, stride, rounding_mode="floor") * stride).clamp_max(grid.w - 1)
            src_flat = frame * grid.tokens_per_frame + hi_a * grid.w + wi_a
            holes = ~kept
            # A snapped source outside an irregular tube was not necessarily computed.
            holes &= torch.isin(src_flat, idx.to(device))
            if protected is not None:
                holes &= ~protected.to(device=device, dtype=torch.bool).index_select(0, idx.to(device))
            z_next.index_copy_(
                1, idx[holes].to(device),
                z_full.index_select(1, src_flat[holes].to(device)),
            )
