"""Shared base for the real diffusers-backed video backbones (Hunyuan, Wan2.1).

HunyuanVideo and Wan2.1 differ in their text encoders and attention topology
(Hunyuan: MMDiT joint text+image attention over an LLM+CLIP condition; Wan2.1:
DiT with cross-attention to a (um)T5 condition) — but they are *structurally
identical* at the boundary this framework cares about:

    * 3D causal VAE, 8× spatial / 4× temporal compression, 16 latent channels
    * patchify factor ``p = (1, 2, 2)`` (t, h, w) before the transformer
    * rectified-flow / flow-matching velocity prediction + Euler-style step

So all of the layout maths (latent grid ⇄ patch-tokens), the VAE wrap and the
scheduler step live here *once*, and each concrete adapter only supplies (a) how
to build its components and (b) how to invoke its transformer. That convergence
is exactly the evidence that the four innovations can be written backbone-agnostically.

This module imports diffusers/transformers **lazily** inside ``_load`` so the
file (and the unit tests, which use the mock) import with no heavy dependency.
"""

from __future__ import annotations

import abc
import contextlib
import inspect
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.backbones.base import (
    BackboneAdapter,
    BackboneCache,
    DenoiseOutput,
    TextConditioning,
)
from cocf.common.config import BackboneConfig
from cocf.common.logging import get_logger
from cocf.common.memory import free_memory, normal_mode, resolve_dtype
from cocf.common.types import TokenGrid

Tensor = torch.Tensor
_log = get_logger(__name__)


def _accepts(module: Optional[nn.Module], name: str) -> bool:
    """Whether ``module.forward`` takes a keyword called ``name``.

    Upstream transformer signatures differ (HunyuanVideo takes
    ``encoder_attention_mask``, some Wan variants do not), and passing an unknown
    keyword is a hard TypeError — so the mask is offered only where it is accepted.
    """
    if module is None:
        return False
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - unintrospectable forward
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _diffusers_version() -> str:
    """Installed diffusers version, for the tiling-unavailable error message."""
    try:
        import diffusers  # type: ignore

        return str(getattr(diffusers, "__version__", "unknown"))
    except ImportError:
        return "not installed"


class DiffusersVideoBackbone(BackboneAdapter):
    """Common implementation for diffusers ``*Transformer3DModel`` backbones.

    Concrete subclasses set the class attributes below and implement
    :meth:`_load` (build VAE/text-encoder/transformer) and
    :meth:`_run_transformer` (the one call that differs per model).
    """

    # patchify factor (t, h, w) applied before the transformer
    patch: Tuple[int, int, int] = (1, 2, 2)
    # VAE compression (t, h, w)
    vae_compress: Tuple[int, int, int] = (4, 8, 8)
    _latent_channels: int = 16

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        self.dtype = resolve_dtype(config.dtype) or torch.bfloat16
        self._loaded = False
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer: Any = None
        self.transformer: Optional[nn.Module] = None
        self._token_dim = self._latent_channels * self.patch[0] * self.patch[1] * self.patch[2]

    # -- lazy component construction ------------------------------------ #

    @abc.abstractmethod
    def _load(self) -> None:
        """Populate ``self.vae/text_encoder/tokenizer/transformer`` from weights."""

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            if not self.config.model_path:
                raise RuntimeError(
                    f"{type(self).__name__} needs BackboneConfig.model_path to load weights"
                )
            # ``normal_mode``: the load is lazy, so the *first forward wins* — and the
            # first forward is often a label-only one under ``inference_mode`` (the
            # Accelerator's text-dim probe, a Stage-A teacher pass). Building the
            # weights there would make every parameter an inference tensor and break
            # Stage C's grad-enabled forwards. Weights must never be inference tensors.
            with normal_mode():
                self._load()
                for m in (self.vae, self.text_encoder, self.transformer):
                    if m is not None:
                        m.to(self._home_device(m), self.dtype).eval()
                        for p in m.parameters():
                            p.requires_grad_(False)
                self._place_auxiliary_modules()
                self._configure_vae_memory()
            self._loaded = True
            # After *every* component is placed, so the report reflects the real
            # resident footprint (a subclass's extra experts included).
            self._log_vram_report()

    def ensure_loaded(self) -> None:
        """Public :class:`BackboneAdapter` hook → force the lazy ``_load``.

        Stage-C LoRA injection calls this before reading :meth:`dit_blocks`, which
        would otherwise return ``[]`` on a not-yet-loaded adapter and make the whole
        ``--use_lora`` path a silent no-op (§4.2).
        """
        self._ensure_loaded()

    def _place_auxiliary_modules(self) -> None:
        """Freeze/device-place components beyond vae/text_encoder/transformer.

        Hook for subclasses that hold extra weights the base loop does not know
        about — see :meth:`cocf.backbones.wan22.Wan22Backbone` and its MoE low-noise
        expert. Called inside :meth:`_ensure_loaded` *before* the VRAM report, so
        that report accounts for them. No-op by default.
        """

    # -- VRAM residency policy (BackboneConfig.offload_* / vae_tiling) --- #

    #: Fallback park device, used when the config names none (and by subclasses /
    #: tests that construct an adapter without going through ``BackboneConfig``).
    _DEFAULT_OFFLOAD_DEVICE: str = "cpu"

    @property
    def offload_device(self) -> str:
        """Where an offloaded component parks while it is idle.

        ``BackboneConfig.offload_device`` ("cpu" by default, so nothing changes unless
        it is set). A peer GPU ("cuda:1") makes the Wan2.2 expert swap an intra-node
        P2P copy instead of a round trip through host RAM. Validated once, in
        :meth:`_resolve_offload_device`, because a bad value here would surface as a
        device-mismatch error inside a forward hours into a run.
        """
        cached = getattr(self, "_offload_device", None)
        if cached is None:
            cached = self._resolve_offload_device()
            self._offload_device = cached
        return cached

    def _resolve_offload_device(self) -> str:
        """Validate the configured park device, falling back to CPU with a warning.

        Rejected (→ CPU): a device this build cannot address, and the *compute* device
        itself — parking a module where it already lives would make every ``_module_active``
        swap a no-op and silently defeat the residency policy the caller asked for.
        """
        want = str(getattr(self.config, "offload_device", "") or
                   self._DEFAULT_OFFLOAD_DEVICE)
        if want == self._DEFAULT_OFFLOAD_DEVICE:
            return want
        try:
            dev = torch.device(want)
        except (RuntimeError, TypeError, ValueError):
            _log.warning("offload_device=%r is not a valid torch device; parking on CPU.",
                         want)
            return self._DEFAULT_OFFLOAD_DEVICE
        if dev.type == "cuda":
            if not torch.cuda.is_available() or (
                dev.index is not None and dev.index >= torch.cuda.device_count()
            ):
                _log.warning("offload_device=%s is not available on this host "
                             "(%d CUDA device(s)); parking on CPU.",
                             want, torch.cuda.device_count() if torch.cuda.is_available() else 0)
                return self._DEFAULT_OFFLOAD_DEVICE
            compute = torch.device(self.device)
            same_index = (dev.index or 0) == (compute.index or 0)
            if compute.type == "cuda" and same_index:
                _log.warning(
                    "offload_device=%s is the compute device — an offloaded module "
                    "would not actually leave the card. Parking on CPU instead.", want,
                )
                return self._DEFAULT_OFFLOAD_DEVICE
        return want

    def _home_device(self, module: Optional[nn.Module]) -> str:
        """Resident device for ``module`` under the config's offload policy.

        Only the text encoder is parked here: it runs once per prompt and is dead
        weight for the whole denoise + counterfactual rollout that follows (on
        Wan2.2 that is ~11 GB of umT5 idling through every rollout). Subclasses
        extend this for components they know are intermittently used (see
        :meth:`Wan22Backbone._home_device` for the idle MoE expert).
        """
        if module is None:
            return self.device
        if module is self.text_encoder and self.config.offload_text_encoder:
            return self.offload_device
        return self.device

    def _resident_denoisers(self) -> List[nn.Module]:
        """Denoiser module(s) currently occupying the compute device.

        The set :meth:`_module_active` parks while an offloaded component runs. Only
        the transformer stack qualifies: the VAE is small and its calls interleave with
        nothing, whereas a denoiser is the one component large enough that co-residency
        with the text encoder decides whether the run fits. MoE subclasses override
        this — see :meth:`cocf.backbones.wan22.Wan22Backbone._resident_denoisers`.
        """
        t = self.transformer
        if isinstance(t, nn.Module) and self._home_device(t) == self.device:
            return [t]
        return []

    @contextlib.contextmanager
    def _module_active(self, module: Optional[nn.Module]) -> Iterator[None]:
        """Bring an offloaded ``module`` onto the compute device for one call.

        A no-op (and free) when the module already lives on the compute device, so
        call sites stay policy-agnostic — they simply declare "I need this now" and
        the configured policy decides whether a transfer actually happens.

        Under ``BackboneConfig.text_encoder_exclusive`` the resident denoiser is parked
        for the duration, so the two never share the card. This is not an optimisation
        but a feasibility requirement on a 40 GB device: a Wan2.2-A14B expert is
        ~26 GiB and umT5-XXL ~10 GiB, and the naive "swap in on top" costs 36 GiB
        before a single activation — it OOMs at the *first* prompt of the run. The
        round trip is ~26 GiB each way over PCIe once per prompt, which against Stage
        A's minutes-per-clip teacher forward is noise.
        """
        if module is None or self._home_device(module) == self.device:
            yield
            return
        parked: List[nn.Module] = []
        # Every ``.to()`` below reallocates parameter storage, so it must run outside
        # any ambient ``inference_mode`` — see :func:`cocf.common.memory.normal_mode`.
        # This context is entered from label-only passes all the time (encode_text
        # under Stage-A teacher forward), which is exactly when the taint would be
        # applied to the *denoisers* Stage C later needs gradients through.
        with normal_mode():
            if self.config.text_encoder_exclusive:
                parked = [m for m in self._resident_denoisers() if m is not module]
                for m in parked:
                    m.to(self.offload_device)
                if parked:
                    free_memory()
            module.to(self.device)
        try:
            yield
        finally:
            with normal_mode():
                module.to(self.offload_device)
                free_memory()
                # Restore before returning: ``_home_device`` still reports these as
                # resident, so leaving them on the CPU would make the next denoise run
                # against weights on the wrong device.
                for m in parked:
                    m.to(self.device)
                if parked:
                    free_memory()

    def _configure_vae_memory(self) -> None:
        """Enable tiled/sliced VAE encode+decode when ``config.vae_tiling`` is set.

        An un-tiled decode of a 49×480×832 clip materialises full-resolution
        decoder feature maps in one allocation — on ``AutoencoderKLWan`` that is a
        single ``192 × 54 × 480 × 832`` bf16 block (**7.71 GiB**), the largest
        transient in Stage A by an order of magnitude. Tiling caps it at one tile's
        worth (~1.4 GB at the default 256 px edge) regardless of resolution. Both
        videos of a counterfactual pair go through the *same* decode path, so the
        damage comparison stays apples-to-apples.

        This is a hard requirement, not a nicety: Stage A calls ``decode_latent``
        once per rollout seed — up to ~90 times per clip
        (:meth:`cocf.lcocf.data.COCFDataGenerator._rollout`) — so an unbounded decode
        does not merely risk OOM, it guarantees one on any card whose free VRAM after
        the frozen weights is under ~8 GB. Hence we **raise** rather than warn when
        the installed ``diffusers`` cannot bound it: failing at load with an
        actionable message beats OOMing hours into a days-long run.
        """
        if not self.config.vae_tiling or self.vae is None:
            return

        enable_tiling = getattr(self.vae, "enable_tiling", None)
        if not callable(enable_tiling):
            raise RuntimeError(
                f"{type(self).__name__}: VAE {type(self.vae).__name__} has no "
                f"enable_tiling(), so decode memory cannot be bounded (diffusers "
                f"{_diffusers_version()}). An untiled decode needs a single multi-GiB "
                f"block and Stage A runs ~90 decodes per clip.\n"
                f"  Fix:  pip install -U 'diffusers>=0.34'\n"
                f"  Or:   re-run without VAE tiling (--no-vae-tiling) only if the card "
                f"has >10 GB free after the frozen weights load."
            )

        # The tiling API is not uniform across the VAEs this base serves:
        # AutoencoderKLWan takes tile_sample_min_/tile_sample_stride_{height,width};
        # AutoencoderKLHunyuanVideo adds the *_num_frames pair; plain AutoencoderKL
        # takes a single ``use_tiling`` bool. Pass only what this one accepts.
        edge = max(64, int(self.config.vae_tile_size))
        stride = max(32, (edge * 3) // 4)
        wanted = {
            "tile_sample_min_height": edge,
            "tile_sample_min_width": edge,
            "tile_sample_min_num_frames": 16,
            "tile_sample_stride_height": stride,
            "tile_sample_stride_width": stride,
            "tile_sample_stride_num_frames": 12,
        }
        try:
            accepted = set(inspect.signature(enable_tiling).parameters)
        except (TypeError, ValueError):  # C-implemented / unintrospectable
            accepted = set()
        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        enable_tiling(**kwargs)

        # ``enable_tiling`` is a plain setter on every diffusers VAE, so a silent
        # no-op here means the class shape changed under us — surface it now rather
        # than at the first decode.
        if not getattr(self.vae, "use_tiling", True):
            raise RuntimeError(
                f"{type(self).__name__}: {type(self.vae).__name__}.enable_tiling() "
                f"left use_tiling False — decode memory is still unbounded."
            )
        _log.info(
            "%s: VAE tiling enabled on %s (%s)", type(self).__name__,
            type(self.vae).__name__,
            ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "no tile kwargs accepted",
        )

        # Batch slicing is orthogonal and harmless — Stage A runs batch 1, so it is
        # a no-op there, but it bounds any batched Stage-C encode. Best-effort.
        enable_slicing = getattr(self.vae, "enable_slicing", None)
        if callable(enable_slicing):
            enable_slicing()
            _log.info("%s: VAE slicing enabled", type(self).__name__)

    def _log_vram_report(self) -> None:
        """One-shot report of what the frozen stack actually costs on the device.

        The residency switches are easy to *think* are on while a flag path leaves
        them off — and the symptom (an OOM tens of minutes into the run) says nothing
        about which. Logging resident bytes against device capacity right after
        placement makes the policy verifiable from the first lines of the log: on
        Wan2.2-A14B, ~26 GiB means one expert is parked, ~52 GiB means neither is.

        Both *allocated* and *reserved* are reported. They answer different questions
        and are routinely confused: allocated is the live-tensor total this report is
        about, while reserved is the caching allocator's high-water mark — which is
        what ``nvidia-smi`` shows, and which never falls on its own. A run whose
        nvidia-smi figure sits far above the resident figure here has a *transient*
        problem, not a weights problem.
        """
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return
        # Bare "cuda" means *the current* device, which is this rank's card under
        # torchrun — not card 0, whose numbers would be someone else's.
        idx = torch.device(self.device).index
        idx = torch.cuda.current_device() if idx is None else idx
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        resident = torch.cuda.memory_allocated(idx) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(idx) / 1024 ** 3
        policy = [
            f"text_encoder={'cpu' if self.config.offload_text_encoder else 'resident'}",
            f"te_exclusive={'on' if self.config.text_encoder_exclusive else 'OFF'}",
            f"idle_expert={'cpu' if self.config.offload_idle_expert else 'resident'}",
            f"vae_tiling={'on' if self.config.vae_tiling else 'OFF'}",
        ]
        _log.info(
            "%s: frozen stack resident %.1f GiB (reserved %.1f) / %.1f GiB "
            "(%.1f GiB free for activations); policy: %s",
            type(self).__name__, resident, reserved, total, total - resident,
            ", ".join(policy),
        )

    # -- static description --------------------------------------------- #

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def hidden_dim(self) -> int:
        # the framework's "token" is the patchified latent (C·p), not the DiT width
        return self._token_dim

    def token_grid(self, num_frames: int, height: int, width: int) -> TokenGrid:
        ct, ch, cw = self.vae_compress
        pt, ph, pw = self.patch
        t_lat = (num_frames - 1) // ct + 1  # causal VAE: (F-1)/ct + 1
        h_lat, w_lat = height // ch, width // cw
        return TokenGrid(t=t_lat // pt, h=h_lat // ph, w=w_lat // pw)

    def timesteps(self, num_inference_steps: int) -> Tensor:
        return self.model_sigma(
            torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        )

    @property
    def flow_shift(self) -> float:
        """Rectified-flow schedule shift ``s`` (1.0 = the uniform schedule).

        Read from ``BackboneConfig.extra['flow_shift']`` so it travels with the
        variant table rather than being hard-coded per adapter.
        """
        return float((self.config.extra or {}).get("flow_shift", 1.0))

    def model_sigma(self, sigma: Tensor) -> Tensor:
        """Schedule position ``∈ (0,1]`` → the noise level the model is trained on.

        The engine, the teacher and the counterfactual generator all count steps down
        and hand this the uniform ``t/T`` (:func:`sigma_from_step`); the *adapter* owns
        what that position means to its own model, exactly as it already owns the
        ``σ → σ·1000`` timestep convention. Rectified-flow video models are trained on
        a **shifted** schedule ``s·σ / (1 + (s-1)·σ)``, which spends far more of the
        budget in the high-noise structure phase.

        Applying it here rather than at the call sites keeps the shift consistent
        across the three places it has to agree — the timestep fed to the transformer,
        the ``dt`` of the Euler step, and (for a Wan2.2 MoE) the expert routing
        boundary. A uniform schedule left the high-noise expert covering 3 of 20 steps
        where the real one covers ~9, so the teacher trajectory was drawn from the
        wrong denoiser for most of the structure phase.
        """
        s = self.flow_shift
        return sigma if s == 1.0 else s * sigma / (1.0 + (s - 1.0) * sigma)

    # -- layout: latent grid ⇄ patch-tokens (the shared maths) ---------- #

    def to_tokens(self, latent_grid: Tensor) -> Tensor:
        """``[B, C, T, H, W] -> [B, N, C·pt·ph·pw]`` by folding (pt, ph, pw) patches."""
        b, c, t, h, w = latent_grid.shape
        pt, ph, pw = self.patch
        x = latent_grid.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
        # → [B, T', H', W', C, pt, ph, pw] → flatten patch+channel
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)

    def to_grid(self, tokens: Tensor, grid: TokenGrid) -> Tensor:
        b, n, dim = tokens.shape
        pt, ph, pw = self.patch
        c = dim // (pt * ph * pw)
        x = tokens.reshape(b, grid.t, grid.h, grid.w, c, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(b, c, grid.t * pt, grid.h * ph, grid.w * pw)

    # -- VAE ------------------------------------------------------------- #

    def _latent_norm(self, ref: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
        """Per-channel ``(mean, std)`` of this VAE's latent space, or ``None``.

        Wan's VAE is **not** normalised by a single ``scaling_factor`` the way SD-era
        VAEs are: ``AutoencoderKLWan`` never registers that key at all (so the
        ``getattr(..., 1.0)`` fallback below is a no-op on it), and publishes 16
        per-channel ``latents_mean``/``latents_std`` instead. The official
        ``WanPipeline`` applies them around every VAE call — ``(raw - mean) / std``
        going in, ``latent * std + mean`` coming out. The VAE does *not* apply them
        itself, so omitting them is silent rather than fatal.

        It is load-bearing here because the transformer is a **pretrained** Wan DiT: it
        denoises in the standardised space (:meth:`initial_latent` seeds ``N(0, I)``,
        and ``z0`` leaves the loop with per-channel mean ≈ 0, std ≈ 1). Handing that
        straight to a decoder that expects the raw space — mean ∈ [-0.95, 1.55], std ∈
        [1.13, 3.27] — is not a uniform dimming that some later gamma could undo: the
        stds differ by 2.9× *across channels*, so the decoder sees the 16 channels
        mis-weighted against each other and returns a desaturated, colour-shifted wash.
        Nothing downstream can detect it, because the counterfactual side is distorted
        by exactly the same transform as the reference side.

        Returns ``None`` for VAEs publishing no such statistics — HunyuanVideo's
        ``AutoencoderKLHunyuanVideo`` is a plain ``scaling_factor=0.476986`` VAE — which
        leaves the scale-only path exactly as it was.

        ``ref`` supplies device/dtype: the VAE may be off-device between calls
        (:meth:`_reclaim_before_vae`), so the statistics are matched to the tensor they
        operate on rather than to ``self.device``.
        """
        cfg = getattr(self.vae, "config", None)
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if mean is None or std is None:
            return None
        c = ref.shape[1]
        if len(mean) != c or len(std) != c:
            # A wrong-length statistic would either raise deep inside the VAE or — for
            # c == 1 — broadcast channel 0 over everything and corrupt silently, which
            # is the whole failure class this method exists to close. Refuse instead.
            raise RuntimeError(
                f"{type(self).__name__}: vae.config latents_mean/latents_std have "
                f"{len(mean)}/{len(std)} entries but the latent has {c} channels"
            )
        kw = {"device": ref.device, "dtype": ref.dtype}
        return (torch.tensor(mean, **kw).view(1, c, 1, 1, 1),
                torch.tensor(std, **kw).view(1, c, 1, 1, 1))

    def encode_video(self, video: Tensor) -> Tensor:
        """``[B, C_pix, F, H, W] -> [B, C, T, H', W']`` in the **model's** latent space.

        Standardised, i.e. the space :meth:`denoise` operates in and the exact inverse
        of :meth:`decode_latent` — see :meth:`_latent_norm` for why that is two
        transforms on a Wan VAE and one on a Hunyuan one.
        """
        self._ensure_loaded()
        self._reclaim_before_vae()
        with self._forward_ctx():
            x = video.to(self.device, self.dtype)
            lat = self.vae.encode(x).latent_dist.sample()  # type: ignore[union-attr]
            norm = self._latent_norm(lat)
            if norm is not None:
                mean, std = norm
                lat = (lat - mean) / std
            scale = getattr(self.vae.config, "scaling_factor", 1.0)  # type: ignore[union-attr]
            return lat * scale
        return None

    def decode_latent(self, latent_grid: Tensor) -> Tensor:
        """``[B, C, T, H, W] -> [B, C_pix, F, H_pix, W_pix]``.

        Runs under :meth:`_forward_ctx`, so a Stage-C caller inside
        ``backbone.grad_mode()`` gets a decode that is **on** the autograd graph (the
        §4.2 pixel/semantic loss path); everyone else keeps the ``inference_mode``
        decode and its memory saving.

        ``latent_grid`` is in the standardised space the transformer denoises in, so it
        is de-standardised back to the VAE's own raw space before the decode — the
        inverse of :meth:`encode_video`, and what the official ``WanPipeline`` does at
        the same point. :meth:`_latent_norm` explains why skipping it is silent.
        """
        self._ensure_loaded()
        self._reclaim_before_vae()
        with self._forward_ctx():
            scale = getattr(self.vae.config, "scaling_factor", 1.0)  # type: ignore[union-attr]
            x = latent_grid.to(self.device, self.dtype) / scale
            norm = self._latent_norm(x)
            if norm is not None:
                mean, std = norm
                x = x * std + mean
            return self.vae.decode(x).sample  # type: ignore[union-attr]
        return None

    def pixel_span(self, lo: int, hi: int):
        """Causal-temporal VAE layout: slot 0 → 1 frame, every later slot → ``c_t``.

        ``F = (T-1)·c_t + 1``, so the **prefix** ``[0, hi)`` covers pixel frames
        ``[0, (hi-1)·c_t + 1)``. Verified against :meth:`decode_latent` by the contract
        test rather than trusted.

        Any window with ``lo > 0`` is **declined** (``None``). The decoder is causal in
        time: handed a slice that does not begin at slot 0 it holds no feature cache of
        the preceding slots, so it re-anchors — treating slot ``lo`` as the clip's
        leading frame and emitting ``(hi-lo-1)·c_t + 1`` frames, not the ``(hi-lo)·c_t``
        that sit at ``[(lo-1)·c_t + 1, (hi-1)·c_t + 1)`` of the full decode. Returning
        that range anyway made the base contract

            decode_latent(z[:, :, lo:hi]) == decode_latent(z)[:, :, start:stop]

        false in both length and content: Stage C's §4.2 pixel loss got a 13-frame
        render against a 16-frame reference (a broadcast error at ``c_t=4, k=4``), and
        the ``c_t == 1`` case that *did* line up would have compared the wrong frames
        silently. It holds for ``lo == 0`` and for nothing else, so that is all this
        advertises; :meth:`~cocf.engine.inference.InferenceEngine._grad_decode_window`
        falls back to the prefix window when a random offset is refused.
        """
        if lo > 0:
            return None
        ct = self.vae_compress[0]
        return (0, 0 if hi <= 0 else (hi - 1) * ct + 1)

    def _reclaim_before_vae(self) -> None:
        """Return cached-but-free allocator blocks to the driver before a VAE call.

        Even tiled, the VAE's tile buffers are the largest *contiguous* requests in
        the pass, and they land on a heap the DiT rollout just churned through — the
        OOM that motivated this had 8.10 GiB sitting in reserved-but-unallocated
        blocks against 3.33 GiB actually free. ``empty_cache`` synchronises, so this
        is gated on ``vae_tiling``: it costs nothing on the mock/CPU paths and is
        amortised on the real path, where a single decode dwarfs it.
        """
        if self.config.vae_tiling and torch.cuda.is_available():
            free_memory()

    # -- the denoiser ε_θ ------------------------------------------------ #

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
        """Dense transformer forward + output splice.

        Real video-DiTs do not expose arbitrary-token-sparse attention, so when
        *some* tokens are active we compute the dense forward and splice the
        inactive token outputs from ``cache`` (correct; the saving on real
        backbones comes from the *whole-step* skip below and any sparse-attention
        kernel a subclass chooses to wire in via :meth:`_run_transformer_sparse`).
        When **no** token is active we skip the transformer entirely and reuse the
        cache verbatim — the dominant cache-acceleration saving (§5.1).

        The returned ``compute_fraction`` states what was really spent, so the
        engine never reports the mask occupancy as if it were a FLOPs saving:

        ============================  ==================
        path taken                    compute_fraction
        ============================  ==================
        whole-step skip (no active)   ``0.0``
        sparse kernel (subclass)      ``|active| / N``
        dense forward (default)       ``1.0``
        ============================  ==================
        """
        self._ensure_loaded()
        b, n, dim = tokens.shape
        active: Optional[Tensor] = None
        if active_mask is not None:
            am = active_mask.to(tokens.device)
            active = am.reshape(-1) if am.dim() == 1 else am[0]
            if active.sum() == 0 and cache is not None and cache.model_output is not None:
                step = cache.step + 1
                return DenoiseOutput(
                    model_output=cache.model_output,
                    cache=BackboneCache(model_output=cache.model_output, step=step),
                    attention={},
                    compute_fraction=0.0,  # the transformer never ran
                )

        # Optional token-sparse path (a subclass that wires in a sparse-attention
        # kernel). ``None`` ⇒ no such kernel ⇒ dense forward below.
        sparse = (
            self._run_transformer_sparse(tokens, t, cond, grid, active, want_attention)
            if active is not None else None
        )
        if sparse is not None:
            eps, attn = sparse
            compute_fraction = float(active.float().mean()) if n else 0.0
        else:
            grid_in = self.to_grid(tokens, grid).to(self.device, self.dtype)
            with self._forward_ctx():
                eps_grid, attn = self._run_transformer(grid_in, t, cond, want_attention)
            eps = self.to_tokens(eps_grid).to(tokens.dtype)
            compute_fraction = 1.0  # dense: every token was computed, mask or not

        if active is not None and cache is not None and cache.model_output is not None:
            # ``active`` is already the [N] mask (decided before flattening so a
            # [B, N] mask keeps its rank — the per-token splice indexes axis 1, and a
            # blind ``reshape(-1)`` on a [B, N] mask would yield [B*N] and mis-index).
            inactive = ~active
            eps[:, inactive] = cache.model_output[:, inactive]

        step = (cache.step + 1) if cache is not None else 0
        out_cache = BackboneCache(model_output=eps.detach(), step=step)
        return DenoiseOutput(
            model_output=eps, cache=out_cache, attention=attn,
            compute_fraction=compute_fraction,
        )

    def _run_transformer_sparse(
        self,
        tokens: Tensor,
        t: Tensor,
        cond: TextConditioning,
        grid: TokenGrid,
        active: Tensor,
        want_attention: bool,
    ) -> Optional[Tuple[Tensor, Dict[str, Tensor]]]:
        """Optional token-sparse ε_θ over ``active`` tokens only (§9.4 future work).

        Return ``(eps_tokens [B, N, d], attention)`` — with the inactive rows left
        for the caller to splice from cache — or ``None`` (the default) to fall back
        to the dense forward. This is the seam a real sparse-attention kernel plugs
        into; until one exists, :meth:`denoise` reports ``compute_fraction = 1.0``
        so the efficiency numbers stay truthful rather than optimistic.
        """
        return None

    @abc.abstractmethod
    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run the model-specific transformer, returning (ε grid, attention readouts)."""

    # -- text conditioning handed to the transformer --------------------- #

    @staticmethod
    def _trim_text(cond: TextConditioning) -> Tuple[Tensor, Optional[Tensor]]:
        """Drop trailing positions that are padding in *every* batch row.

        The tokenizers here pad to a fixed ``max_length`` (512 for Wan's umT5), and a
        real caption occupies a few dozen of those. Cross-attention over the untrimmed
        sequence costs ``O(N · 512)`` per step no matter how short the prompt is — and,
        without a mask, the model attends to hundreds of pad embeddings as if they were
        text (§P1-15). Trimming is exact: the dropped positions are padding for the
        whole batch, so no row loses a token.
        """
        embeds = cond.embeds
        mask = cond.mask
        if mask is None or mask.numel() == 0:
            return embeds, None
        m = mask.to(embeds.device)
        if m.dim() == 1:
            m = m[None]
        used = m.bool().any(dim=0).nonzero()
        keep = int(used.max()) + 1 if used.numel() else embeds.shape[1]
        return embeds[:, :keep], m[:, :keep]

    def _text_kwargs(self, cond: TextConditioning, module: Optional[nn.Module]) -> Dict[str, Any]:
        """``encoder_hidden_states`` (+ mask when the model accepts one), trimmed."""
        embeds, mask = self._trim_text(cond)
        kwargs: Dict[str, Any] = {
            "encoder_hidden_states": embeds.to(self.device, self.dtype)
        }
        if mask is not None and _accepts(module, "encoder_attention_mask"):
            kwargs["encoder_attention_mask"] = mask.to(self.device)
        return kwargs

    # -- scheduler ------------------------------------------------------- #

    def scheduler_step(
        self, model_output: Tensor, t: Tensor, t_next: Tensor, tokens: Tensor
    ) -> Tensor:
        """Flow-matching Euler step ``z_{t_next} = z_t + (σ_{next}-σ_t)·v``.

        Both Hunyuan and Wan2.1 are rectified-flow models predicting velocity ``v``.
        ``t``/``t_next`` are *schedule positions*; the step is taken in the model's own
        noise space, so the same :meth:`model_sigma` that decides the transformer's
        timestep also sets ``dt`` — otherwise the shift would move the conditioning
        without moving the integrator with it.
        """
        sigma, sigma_next = self.model_sigma(t), self.model_sigma(t_next)
        dt = (sigma_next - sigma).reshape(-1, *([1] * (tokens.dim() - 1))).to(tokens.dtype)
        return tokens + dt * model_output

    def dit_blocks(self) -> List[nn.Module]:
        """Transformer blocks for Stage-C LoRA (§7.1.3)."""
        return self._blocks_of(self.transformer)

    @staticmethod
    def _blocks_of(module: Optional[nn.Module]) -> List[nn.Module]:
        """The transformer-block stack of a diffusers ``*Transformer3DModel``.

        Shared by :meth:`dit_blocks` and MoE subclasses that expose more than one
        denoiser; returns ``[]`` for a ``None`` module (weights not loaded yet).
        """
        if module is None:
            return []
        for attr in ("transformer_blocks", "blocks", "single_transformer_blocks"):
            blocks = getattr(module, attr, None)
            if blocks is not None:
                return list(blocks)
        return []

    @property
    def module(self) -> Optional[nn.Module]:
        return self.transformer
