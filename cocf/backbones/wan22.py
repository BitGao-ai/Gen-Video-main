"""Wan2.2 backbone adapter (§9.1 — primary backbone; MoE dual-expert DiT).

Wraps Alibaba's Wan2.2 text-to-video family (T2V-A14B, I2V-A14B, TI2V-5B). Wan2.2
keeps Wan2.1's cross-attention-to-umT5 topology, so this adapter reuses the whole
:class:`~cocf.backbones.wan21.Wan21Backbone` code path (text encode, VAE wrap,
layout maths, flow-matching step) and changes only the *two* things Wan2.2 actually
introduces — both hidden behind the same :class:`DiffusersVideoBackbone` contract so
L-COCF / STA / RAEC / CMSC stay backbone-agnostic:

1.  **MoE dual-expert denoiser.** The A14B models ship *two*
    ``WanTransformer3DModel`` experts — a high-noise expert (``transformer``) and a
    low-noise expert (``transformer_2``) — switched at a noise boundary
    ``boundary_timestep = boundary_ratio · num_train_timesteps``: the high-noise
    expert runs while ``timestep ≥ boundary``, the low-noise expert below it (exactly
    🤗 ``WanPipeline`` semantics). A single-expert checkpoint (e.g. TI2V-5B) has no
    ``transformer_2`` / ``boundary_ratio``, and the adapter degrades to the Wan2.1
    single-denoiser path with zero special-casing at the call sites.

2.  **High-compression VAE (TI2V-5B).** The 5B variant uses the new Wan2.2 VAE
    (4×16×16 compression, 48 latent channels) in place of the A14B/2.1 VAE
    (4×8×8, 16 channels). Both are ``AutoencoderKLWan``; the geometry is read from
    ``config.extra`` so one adapter serves every Wan2.2 checkpoint.

Configuring a variant (all via ``BackboneConfig.extra``)::

    # Wan2.2-T2V-A14B (default): MoE, Wan2.1 VAE — nothing to set
    extra = {}
    # Wan2.2-I2V-A14B: MoE with the I2V boundary
    extra = {"boundary_ratio": 0.900}
    # Wan2.2-TI2V-5B: single expert + high-compression VAE
    extra = {"boundary_ratio": None, "vae_compress": [4, 16, 16], "latent_channels": 48}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cocf.backbones.base import BackboneCache, DenoiseOutput, TextConditioning
from cocf.backbones.wan21 import Wan21Backbone
from cocf.common.config import BackboneConfig
from cocf.common.logging import get_logger
from cocf.common.memory import free_memory, normal_mode
from cocf.common.registry import register_backbone
from cocf.common.types import TokenGrid

Tensor = torch.Tensor
_log = get_logger(__name__)

# Emit the offload-thrash warning once, after this many expert swaps. Chosen so a
# well-behaved run (one crossing per trajectory) never trips it, while Stage A's
# per-rollout re-entry above the boundary does so within the first clip.
_SWAP_WARN_AFTER = 32

#: Wan2.2 variant → :attr:`BackboneConfig.extra` (§9.1).
#:
#: Lives here rather than in an entry script because *every* stage must map the same
#: ``--wan-variant`` string to the same geometry: Stage C's target is the ``Y_full``
#: Stage A rendered, so a variant that drifts between the two silently compares videos
#: of different token geometry. ``a14b-t2v`` is the documented primary (dual-expert MoE
#: + the Wan2.1 VAE, nothing to set); ``ti2v-5b`` is single-expert with the
#: high-compression VAE.
#:
#: ``flow_shift`` is the upstream rectified-flow schedule shift (see
#: :meth:`DiffusersVideoBackbone.model_sigma`). It is part of the *variant*, not of the
#: run, because it decides both the sampled trajectory and where the MoE noise boundary
#: falls; ``--flow-shift`` overrides it for a deliberate experiment.
WAN22_VARIANTS: Dict[str, Dict[str, Any]] = {
    "a14b-t2v": {"flow_shift": 5.0},
    "a14b-i2v": {"boundary_ratio": 0.900, "flow_shift": 5.0},
    "ti2v-5b": {"boundary_ratio": None, "vae_compress": [4, 16, 16],
                "latent_channels": 48, "flow_shift": 5.0},
}


@register_backbone("wan22")
@register_backbone("wan2.2")
class Wan22Backbone(Wan21Backbone):
    """Alibaba Wan2.2 adapter — cross-attention DiT with a Mixture-of-Experts denoiser."""

    # A14B defaults (reuse the Wan2.1 VAE geometry). TI2V-5B overrides via extra.
    patch = (1, 2, 2)
    vae_compress = (4, 8, 8)
    _latent_channels = 16
    # A14B T2V ships boundary_ratio 0.875 (I2V 0.900). Set extra["boundary_ratio"]
    # to ``None`` for a single-expert checkpoint (TI2V-5B).
    _default_boundary_ratio = 0.875

    def __init__(self, config: BackboneConfig) -> None:
        extra = config.extra or {}
        # Variant geometry (TI2V-5B): let extra override the class defaults *before*
        # the base ctor derives the patch-token width from ``_latent_channels``/``patch``.
        if "patch" in extra:
            self.patch = tuple(extra["patch"])                     # e.g. (1, 2, 2)
        # Whether the geometry came from an explicit variant/extra entry (vs the
        # class default): an explicit declaration must win over config detection —
        # a conflict there means the detection is unreliable, not the declaration.
        self._vae_compress_explicit = "vae_compress" in extra
        if "vae_compress" in extra:
            self.vae_compress = tuple(extra["vae_compress"])       # e.g. (4, 16, 16)
        if "latent_channels" in extra:
            self._latent_channels = int(extra["latent_channels"])  # e.g. 48
        super().__init__(config)
        self.transformer_2: Optional[nn.Module] = None
        # Under ``offload_idle_expert`` exactly one expert is resident at a time;
        # this tracks which, so :meth:`_expert_for` only pays a transfer on a real
        # switch. ``None`` until the first routed call.
        self._resident_expert: Optional[nn.Module] = None
        self._expert_swaps = 0
        self._swap_warned = False
        # ``None`` ⇒ single-expert (no switching). ``extra["boundary_ratio"]`` may be
        # explicitly ``None`` to force the TI2V-5B single-denoiser path; any other value
        # is coerced to ``float`` so a stray string fails here, not deep in ``_expert_for``.
        br = extra.get("boundary_ratio", self._default_boundary_ratio)
        self.boundary_ratio: Optional[float] = None if br is None else float(br)
        self.num_train_timesteps = int(extra.get("num_train_timesteps", 1000))
        # The expert whose forward produced the velocity the engine's cache currently
        # holds; ``None`` until the first forward runs. Read by :meth:`denoise` to
        # refuse cross-expert cache reuse.
        self._eps_expert: Optional[nn.Module] = None

    # -- VAE geometry, read from the checkpoint's own config -------------- #

    @staticmethod
    def _vae_latent_channels(vae: nn.Module) -> Optional[int]:
        """Latent channel count from the VAE config (``z_dim`` on Wan, else the
        generic ``latent_channels``). ``None`` when neither is present."""
        cfg = getattr(vae, "config", None)
        for key in ("z_dim", "latent_channels"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    @staticmethod
    def _detect_vae_compress(vae: nn.Module) -> Optional[Tuple[int, int, int]]:
        """``(t, h, w)`` compression from the VAE config, or ``None`` if unstated.

        Only *explicitly stated* scale factors are trusted. Deriving the spatial
        factor from ``temperal_downsample`` (``2**len(...)``) is wrong for TI2V-5B:
        its VAE adds a ``patch_size=2``/``is_residual`` stage, so the real spatial
        compression is 16 while the list length yields 8 — and that wrong guess
        overwrote the correct declared ``(4, 16, 16)``, quadrupling the token grid.

        Returning ``None`` when the config states nothing is the important part:
        the adapter's declared ``vae_compress`` is then left alone. A geometry
        guess that can be wrong is worse than no guess — it silently reshapes
        every latent.
        """
        cfg = getattr(vae, "config", None)
        if cfg is None:
            return None

        # Current ``AutoencoderKLWan`` registers these two; ``WanPipeline`` reads
        # the same keys.
        temporal = getattr(cfg, "scale_factor_temporal", None)
        spatial = getattr(cfg, "scale_factor_spatial", None)
        if all(isinstance(v, int) and v > 0 for v in (temporal, spatial)):
            return temporal, spatial, spatial

        # Generic names used by other VAE families (never present on Wan).
        temporal = getattr(cfg, "temporal_compression_ratio", None)
        spatial = getattr(cfg, "spatial_compression_ratio", None)
        if all(isinstance(v, int) and v > 0 for v in (temporal, spatial)):
            return temporal, spatial, spatial
        return None

    # -- component construction (adds the low-noise expert) ------------- #

    def _load(self) -> None:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel

        path = self.config.model_path
        extra = self.config.extra or {}
        self._check_variant_against_checkpoint(path)
        # ``torch_dtype`` + ``low_cpu_mem_usage`` on *every* component: without them
        # ``from_pretrained`` materialises fp32 on the CPU and only the subsequent
        # ``.to(device, dtype)`` narrows it, so an A14B expert costs 56 GB of host RAM
        # (and umT5 22 GB) during load. Stage A runs one process per GPU, so that
        # transient is multiplied by the shard count and is what actually OOMs an
        # 8-way node — see the load-time budget in the §9.1 notes.
        hf = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}
        self.vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", **hf)
        # Adapt to the *actual* checkpoint's geometry where its config states it, so
        # A14B / TI2V-5B need no per-variant hardcoding. Each half is independent and
        # only applied when genuinely found — an unstated value leaves the declared
        # default (from ``extra`` or the class) in place.
        vae_lc = self._vae_latent_channels(self.vae)
        if vae_lc is not None and vae_lc != self._latent_channels:
            _log.info(
                "%s: overriding latent_channels %d → %d to match the VAE config",
                type(self).__name__, self._latent_channels, vae_lc,
            )
            self._latent_channels = vae_lc
            pt, ph, pw = self.patch
            self._token_dim = self._latent_channels * pt * ph * pw
        detected = self._detect_vae_compress(self.vae)
        if detected is not None and detected != self.vae_compress:
            if self._vae_compress_explicit:
                raise ValueError(
                    f"{type(self).__name__}: the VAE config states compression "
                    f"{detected} but the variant explicitly declares "
                    f"{self.vae_compress} — refusing to override an explicit "
                    f"geometry with a detected one; check --wan-variant against "
                    f"the checkpoint."
                )
            _log.info(
                "%s: overriding vae_compress %s → %s to match the VAE config",
                type(self).__name__, self.vae_compress, detected,
            )
            self.vae_compress = detected
        # High-noise expert (always present).
        self.transformer = WanTransformer3DModel.from_pretrained(path, subfolder="transformer", **hf)
        # Cross-check transformer in_channels — should already match from the VAE
        # detection above, but guard against mismatched checkpoints.
        ckpt_channels = getattr(self.transformer.config, "in_channels", None)
        if ckpt_channels is not None and ckpt_channels != self._latent_channels:
            _log.warning(
                "%s: transformer.config.in_channels=%d differs from "
                "_latent_channels=%d (vae.config.latent_channels=%s); overriding "
                "to match transformer",
                type(self).__name__, ckpt_channels, self._latent_channels, vae_lc,
            )
            self._latent_channels = ckpt_channels
            pt, ph, pw = self.patch
            self._token_dim = self._latent_channels * pt * ph * pw
        # Low-noise expert (A14B MoE). Absent on single-expert checkpoints ⇒ fall back
        # to single-denoiser denoising rather than failing the load.
        if self.boundary_ratio is not None:
            sub = extra.get("transformer_2_subfolder", "transformer_2")
            try:
                self.transformer_2 = WanTransformer3DModel.from_pretrained(path, subfolder=sub, **hf)
            except (OSError, ValueError) as e:  # subfolder absent ⇒ single-expert ckpt
                # Only a genuine "not found" (HF raises OSError / a ValueError subclass)
                # degrades to single-expert; OOM / CUDA / other RuntimeErrors propagate
                # rather than being silently misread as "no second expert".
                _log.warning(
                    "Wan2.2: could not load '%s' expert (%s: %s); falling back to "
                    "single-expert denoising. Pass extra['boundary_ratio']=None to "
                    "silence this on a known single-expert checkpoint.",
                    sub, type(e).__name__, e,
                )
                self.transformer_2 = None
                self.boundary_ratio = None
        self.text_encoder = UMT5EncoderModel.from_pretrained(path, subfolder="text_encoder", **hf)
        self.tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        self._max_len = int(extra.get("max_text_len", 512))

    # -- variant ⇄ checkpoint agreement --------------------------------- #

    def _check_variant_against_checkpoint(self, path: Optional[str]) -> None:
        """Fail *at load* when ``--wan-variant`` disagrees with the weights on disk.

        The dangerous direction is a dual-expert A14B checkpoint opened under a
        single-expert variant (``ti2v-5b``): ``boundary_ratio`` is then ``None``, so
        :meth:`_load` never reads ``transformer_2`` and :meth:`_expert_for` routes
        *every* noise level to the high-noise expert. Nothing downstream notices — the
        run proceeds at full speed and produces counterfactual labels drawn from the
        wrong denoiser below σ = ``boundary_ratio``. Failing here, with the fix in the
        message, is the only cheap place to catch it.

        Skipped for a non-local ``path`` (an HF repo id has no directory to inspect);
        the ``transformer_2`` load in :meth:`_load` still degrades gracefully there.
        """
        if not path:
            return
        root = Path(path)
        if not root.is_dir():
            return
        sub = (self.config.extra or {}).get("transformer_2_subfolder", "transformer_2")
        has_second = (root / str(sub)).is_dir()
        if has_second and self.boundary_ratio is None:
            raise RuntimeError(
                f"Wan2.2 variant mismatch: '{root}' ships a '{sub}' expert (a "
                f"dual-expert A14B checkpoint), but this run configured a "
                f"single-expert variant (boundary_ratio=None), so the low-noise "
                f"expert would never load and every σ would be routed to the "
                f"high-noise one.\n"
                f"  Fix:  --wan-variant a14b-t2v   (or a14b-i2v for image-to-video)"
            )
        if not has_second and self.boundary_ratio is not None:
            raise RuntimeError(
                f"Wan2.2 variant mismatch: this run configured the dual-expert MoE "
                f"(boundary_ratio={self.boundary_ratio}), but '{root}' has no '{sub}' "
                f"subfolder — it is a single-expert checkpoint.\n"
                f"  Fix:  --wan-variant ti2v-5b"
            )

    def _place_auxiliary_modules(self) -> None:
        """Freeze/device-place the *second* expert (:class:`DiffusersVideoBackbone` hook).

        The base placement loop only knows about ``vae``/``text_encoder``/
        ``transformer``; the MoE low-noise expert is frozen and device-placed here so
        it never leaks trainable params into the optimiser. Running as a hook (rather
        than by overriding ``_ensure_loaded``) keeps it *inside* the base's load
        sequence, so the VRAM report that follows counts this expert's ~28 GB.
        """
        if self.transformer_2 is None:
            return
        self.transformer_2.to(self._home_device(self.transformer_2), self.dtype).eval()
        for p in self.transformer_2.parameters():
            p.requires_grad_(False)
        if self.config.offload_idle_expert:
            # The high-noise expert is the one resident at load; the schedule
            # starts at the noisiest step, so this is the cheaper initial guess.
            self._resident_expert = self.transformer

    # -- VRAM residency: keep at most one expert on the compute device --- #

    def _home_device(self, module: Optional[nn.Module]) -> str:
        """Park the *idle* expert on CPU when ``offload_idle_expert`` is set.

        Each A14B expert is ~28 GB in bf16, so holding both resident costs 56 GB
        before a single activation is allocated. Only one ever runs at a given
        noise level, so the idle one is pure ballast — but see the thrash warning
        in :meth:`_expert_for`: this pays off only when the workload does not
        alternate across the boundary.
        """
        if module is not None and self.config.offload_idle_expert:
            for expert in (self.transformer, self.transformer_2):
                if module is expert:
                    # ``_resident_expert`` is still None during the initial load
                    # (``_load`` only just populated the attributes), so fall back
                    # to the primary transformer — the expert the noisiest first
                    # step routes to, and the one
                    # :meth:`_place_auxiliary_modules` records afterwards.
                    resident = self._resident_expert or self.transformer
                    return self.device if module is resident else self.offload_device
        return super()._home_device(module)

    def _resident_denoisers(self) -> List[nn.Module]:
        """The expert(s) currently occupying the compute device (§9.1).

        Overridden because the base class only knows about ``transformer``: under
        ``offload_idle_expert`` the resident denoiser is whichever expert
        :meth:`_make_resident` last swapped in, and parking the *wrong* one during a
        text encode would leave 26 GB on the card and evict weights that are about to
        be used. Consumed by :meth:`DiffusersVideoBackbone._module_active`.
        """
        if not self.config.offload_idle_expert:
            return [m for m in (self.transformer, self.transformer_2) if isinstance(m, nn.Module)]
        resident = self._resident_expert or self.transformer
        return [resident] if isinstance(resident, nn.Module) else []

    def _make_resident(self, want: nn.Module) -> None:
        """Swap ``want`` onto the compute device, evicting the other expert.

        Transfers ~28 GB each way, so it is only worth doing when switches are
        rare. They are rare along a *single* trajectory (σ decreases monotonically,
        so the boundary is crossed at most once) but Stage A restarts a trajectory
        per counterfactual rollout, and a representative step just above the
        boundary makes every rollout cross it. The warning below surfaces that
        rather than letting the run silently become transfer-bound.
        """
        other = self.transformer_2 if want is self.transformer else self.transformer
        # Outside any ambient ``inference_mode``: these moves reallocate the experts'
        # parameters, and an inference tensor can never carry a gradient afterwards
        # (:func:`cocf.common.memory.normal_mode`). Routing is called from label-only
        # passes too, so the guard belongs here rather than at the call sites.
        with normal_mode():
            if isinstance(other, nn.Module):
                other.to(self.offload_device)
            want.to(self.device)
        self._resident_expert = want
        self._expert_swaps += 1
        free_memory()
        if self._expert_swaps == _SWAP_WARN_AFTER and not self._swap_warned:
            self._swap_warned = True
            _log.warning(
                "Wan2.2: %d MoE expert swaps so far — the workload keeps crossing the "
                "noise boundary (%.3f), so each rollout pays two ~28 GB transfers. "
                "offload_idle_expert is saving VRAM at a large throughput cost; "
                "prefer offload_text_encoder + a single-expert variant (ti2v-5b) if "
                "this run is time-bound.",
                _SWAP_WARN_AFTER, self.boundary_ratio or 0.0,
            )

    # -- MoE expert selection ------------------------------------------- #

    def _expert_for(self, timestep: Tensor) -> nn.Module:
        """Pick the denoiser for this step's noise level (🤗 ``WanPipeline`` rule).

        ``timestep`` is the model-space timestep (``t·1000``). The high-noise expert
        (``transformer``) runs while ``timestep ≥ boundary_timestep``, the low-noise
        expert (``transformer_2``) below it. Single-expert models always use the
        primary ``transformer``. The engine broadcasts one scalar ``t`` across the
        batch, so the mean gives a single unambiguous choice per step.

        Under ``offload_idle_expert`` the chosen expert is also *made resident*
        here, which is why routing owns the placement: it is the only point that
        knows which expert the next forward will touch.
        """
        if self.transformer_2 is None or self.boundary_ratio is None:
            return self.transformer  # type: ignore[return-value]
        boundary = self.boundary_ratio * self.num_train_timesteps
        high = float(timestep.float().mean()) >= boundary
        want = self.transformer if high else self.transformer_2
        if (
            self.config.offload_idle_expert
            and want is not self._resident_expert
            and isinstance(want, nn.Module)
        ):
            self._make_resident(want)
        return want  # type: ignore[return-value]

    # -- the cross-attention DiT call (expert-routed) ------------------- #

    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        timestep = (self.model_sigma(t.to(self.device)) * 1000.0).flatten()
        expert = self._expert_for(timestep)
        out = expert(  # type: ignore[misc]
            hidden_states=latent_grid,
            timestep=timestep,
            **self._text_kwargs(cond, expert),
            return_dict=True,
        )
        # The velocity this call emits — and therefore any cache the caller builds
        # from it — belongs to *this* expert's vector field.
        self._eps_expert = expert
        eps = out.sample if hasattr(out, "sample") else out[0]
        return eps, {}

    # -- cache reuse is only valid inside one expert's noise regime ------ #

    def denoise(
        self,
        tokens: Tensor,
        t: Tensor,
        cond: TextConditioning,
        *,
        grid: TokenGrid,
        active_mask: Optional[Tensor] = None,
        cache: Optional[BackboneCache] = None,
        want_attention: bool = False,
    ) -> DenoiseOutput:
        """MoE-aware guard around the base splice: never reuse the other expert's ε.

        The base implementation splices ``cache.model_output`` into every inactive
        token unconditionally (``DiffusersVideoBackbone.denoise``). That is a
        TeaCache-style approximation *within* one denoiser, but the A14B MoE routes
        σ ≥ boundary to ``transformer`` and σ < boundary to ``transformer_2`` — two
        independently trained vector fields that are not interchangeable. Carrying a
        high-noise-expert velocity into the low-noise regime integrates the wrong
        field for every skipped token, and because the cache row is copied forward
        verbatim each step, a token skipped at the boundary keeps the wrong expert's
        velocity to the end of the trajectory (the "mosaic" failure).

        Dropping the cache on the boundary step is nearly free on this adapter: the
        forward is dense anyway, so the inactive tokens simply keep their *fresh*
        outputs instead of stale ones — strictly closer to the un-accelerated
        trajectory. Within an expert's regime, reuse is untouched.
        """
        if (
            cache is not None
            and cache.model_output is not None
            and self._eps_expert is not None
        ):
            timestep = (self.model_sigma(t.to(self.device)) * 1000.0).flatten()
            want = self._expert_for(timestep)
            if want is not self._eps_expert:
                _log.info(
                    "Wan22: dropping ε cache at the MoE boundary — it was produced by "
                    "the %s-noise expert and this step routes to the other one",
                    "high" if self._eps_expert is self.transformer else "low",
                )
                cache = None
        return super().denoise(
            tokens, t, cond, grid=grid, active_mask=active_mask,
            cache=cache, want_attention=want_attention,
        )

    # -- Stage-C LoRA: expose *both* experts' blocks -------------------- #

    def dit_blocks(self) -> List[nn.Module]:
        """Transformer blocks for Stage-C LoRA (§7.1.3) — from *both* experts.

        By default diffusers loads Wan2.2 LoRAs only into the first denoiser; the
        design's Stage-C fine-tune targets "the last few DiT blocks", so we expose
        the low-noise expert's blocks too. Consumers that want the *last-n* tail must
        go through :meth:`lora_target_blocks` — a blind ``dit_blocks()[-n:]`` here
        would land entirely on the second expert (see that method).
        """
        blocks = list(self._blocks_of(self.transformer))
        blocks.extend(self._blocks_of(self.transformer_2))
        return blocks

    def lora_target_blocks(self, last_n: int) -> List[nn.Module]:
        """Last ``last_n`` blocks of *each* expert (§7.1.3).

        A flat ``dit_blocks()[-last_n:]`` would wrap only the low-noise expert (its
        blocks are appended last), starving the high-noise expert — the very one that
        runs at the noisiest, most structure-defining steps. Stage-C must fine-tune
        the tail of *both* denoisers, so we take each expert's own tail.
        """
        high = self._blocks_of(self.transformer)
        targets: List[nn.Module] = list(high[-last_n:] if last_n > 0 else high)
        if self.transformer_2 is not None:
            low = self._blocks_of(self.transformer_2)
            targets.extend(low[-last_n:] if last_n > 0 else low)
        return targets

    def lora_roots(self) -> List[Tuple[str, nn.Module]]:
        """Both experts, separately named, so their LoRA checkpoint keys never collide.

        The two experts are structurally identical, so a single root would give
        ``blocks.39.attn1.to_q`` for *both* — the low-noise adapter would silently
        overwrite the high-noise one on save and be loaded into the wrong expert on
        restore. Naming them apart keeps a Stage-C checkpoint round-trippable.
        """
        roots: List[Tuple[str, nn.Module]] = []
        if self.transformer is not None:
            roots.append(("transformer", self.transformer))
        if self.transformer_2 is not None:
            roots.append(("transformer_2", self.transformer_2))
        return roots
