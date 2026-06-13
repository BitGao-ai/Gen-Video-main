"""Backbone abstraction — the multi-model compatibility layer (user requirement #2).

The COCF algorithm never talks to HunyuanVideo or Wan2.1 directly. It talks to a
:class:`BackboneAdapter`, a thin, uniform interface that exposes exactly the
operations the accelerator needs and nothing else:

    * latent <-> token layout       (owned here so the algorithm stays layout-free)
    * text encoding                 (prompt -> conditioning ``c``)
    * VAE encode / decode           (video <-> latent)
    * the **denoiser** ε_θ          (the one op where token-sparsity saves FLOPs)
    * the **scheduler step**        (cheap latent update)
    * a composed full transition Φ_t and a *partial* transition Φ̃_t

Concrete adapters (``hunyuan.py``, ``wan21.py``) implement these by delegating to
the upstream model code; a :class:`cocf.backbones.mock.MockBackbone` implements
them with a tiny network and *real* token gather/scatter so the whole pipeline is
testable without multi-billion-parameter weights.

Why this is the compatibility seam
-----------------------------------
HunyuanVideo uses MMDiT joint text+image attention; Wan2.1 uses cross-attention to
a (um)T5 text encoder. Both, however, reduce to "predict ε_θ(z_t, t, c) over a
grid of latent tokens, then take a scheduler step". By contracting on *that*, every
upstream architectural difference is hidden behind the adapter and the four
innovations (L-COCF / STA / RAEC / CMSC) are written once, backbone-agnostically.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from cocf.common.config import BackboneConfig
from cocf.common.types import TokenGrid

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Data carried across the denoise interface
# --------------------------------------------------------------------------- #


@dataclass
class TextConditioning:
    """Encoded text condition ``c`` (§2.1), backbone-agnostic.

    ``embeds`` is the per-token text sequence used by cross/joint attention and by
    the CMSC text-tube alignment (§6.3.1); ``prompts`` is kept for the VLM causal
    parser (§3.3.1) and OCR target (§6.3.2).
    """

    embeds: Tensor  # [B, L, d_c]
    mask: Optional[Tensor] = None  # [B, L] padding mask (1 = keep)
    pooled: Optional[Tensor] = None  # [B, d_c] pooled embedding (optional)
    prompts: Sequence[str] = ()

    def to(self, device, dtype=None) -> "TextConditioning":
        embeds = self.embeds.to(device, dtype) if dtype else self.embeds.to(device)
        return TextConditioning(
            embeds=embeds,
            mask=self.mask.to(device) if self.mask is not None else None,
            pooled=self.pooled.to(device) if self.pooled is not None else None,
            prompts=self.prompts,
        )


@dataclass
class BackboneCache:
    """Reusable state carried between denoise calls.

    The accelerator only relies on ``model_output`` (the last ε_θ over *all*
    tokens), which feeds ANCHOR/skip reuse and the residual δ in RAEC. ``kv`` and
    ``hidden`` are opaque slots an adapter may use for true attention-KV reuse.
    """

    model_output: Optional[Tensor] = None  # [B, N, d_out] last full ε_θ
    step: int = -1
    kv: Dict[int, Any] = field(default_factory=dict)  # layer_idx -> (k, v) or similar
    hidden: Dict[int, Tensor] = field(default_factory=dict)

    def detach(self) -> "BackboneCache":
        mo = self.model_output.detach() if self.model_output is not None else None
        return BackboneCache(model_output=mo, step=self.step, kv=self.kv, hidden=self.hidden)


@dataclass
class DenoiseOutput:
    """Result of one denoiser forward pass."""

    model_output: Tensor  # [B, N, d_out] predicted noise / velocity over tokens
    cache: BackboneCache
    # optional attention readouts, e.g. {"text": [B, heads, N, L]} for CMSC/affinity
    attention: Dict[str, Tensor] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The adapter contract
# --------------------------------------------------------------------------- #


class BackboneAdapter(abc.ABC):
    """Uniform interface over a frozen video-diffusion backbone.

    Subclasses must implement the small set of abstract methods below. Everything
    the accelerator needs is composed from them, so adding a backbone is a single,
    self-contained file.
    """

    def __init__(self, config: BackboneConfig) -> None:
        from cocf.common.memory import resolve_device

        self.config = config
        # Fall back to CPU when the requested backend is unavailable, so a config
        # defaulting to "cuda" still runs on a CPU-only box (tests/CI).
        self.device = resolve_device(config.device)

    # -- static space description ----------------------------------------- #

    @property
    @abc.abstractmethod
    def latent_channels(self) -> int:
        """Channel count ``C`` of the VAE latent."""

    @property
    @abc.abstractmethod
    def hidden_dim(self) -> int:
        """DiT token hidden width ``d`` (== model_output last dim)."""

    @abc.abstractmethod
    def token_grid(self, num_frames: int, height: int, width: int) -> TokenGrid:
        """Map *pixel* (frames, H, W) to the post-patchify latent token grid."""

    @abc.abstractmethod
    def timesteps(self, num_inference_steps: int) -> Tensor:
        """Return the (descending) denoising timestep schedule ``[t_T, …, t_1]``."""

    # -- layout: latent grid <-> flat tokens (owned here) ----------------- #

    @abc.abstractmethod
    def to_tokens(self, latent_grid: Tensor) -> Tensor:
        """``[B, C, T, H, W] -> [B, N, d]`` in (t, h, w) row-major token order."""

    @abc.abstractmethod
    def to_grid(self, tokens: Tensor, grid: TokenGrid) -> Tensor:
        """Inverse of :meth:`to_tokens`."""

    # -- initial noise (z_T) ---------------------------------------------- #

    def initial_latent(
        self,
        grid: TokenGrid,
        *,
        batch: int = 1,
        generator: Optional[torch.Generator] = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample a fresh ``z_T ~ N(0, I)`` in **token** form ``[batch, N, d]``.

        Latent-space layout is owned by the adapter, so seeding a generation lives
        here (shared verbatim by the inference engine and the Stage-A teacher
        generator — train/serve parity). Noise is drawn in the VAE *channel* latent
        grid (the space the model actually denoises) and mapped through
        :meth:`to_tokens`, which patchifies it down to exactly ``grid.num_tokens``
        tokens. The patch factor is read from a ``patch`` attribute when present
        (real diffusers backbones expose ``(p_t, p_h, p_w)``); the mock uses
        ``(1, 1, 1)`` so the latent grid *is* the token grid.
        """
        device = device or self.device
        p_t, p_h, p_w = getattr(self, "patch", (1, 1, 1))
        latent = torch.randn(
            batch, self.latent_channels,
            grid.t * p_t, grid.h * p_h, grid.w * p_w,
            generator=generator, device=device, dtype=dtype,
        )
        return self.to_tokens(latent)

    # -- VAE -------------------------------------------------------------- #

    @abc.abstractmethod
    def encode_video(self, video: Tensor) -> Tensor:
        """``[B, C_pix, F, H_pix, W_pix] -> [B, C, T, H, W]`` latent (no grad)."""

    @abc.abstractmethod
    def decode_latent(self, latent_grid: Tensor) -> Tensor:
        """``[B, C, T, H, W] -> [B, C_pix, F, H_pix, W_pix]`` video."""

    # -- text ------------------------------------------------------------- #

    @abc.abstractmethod
    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        """Encode prompts into conditioning ``c``."""

    # -- the denoiser ε_θ (sparsity lives here) --------------------------- #

    @abc.abstractmethod
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
        """Predict ε_θ over tokens.

        If ``active_mask`` (bool ``[N]`` or ``[B, N]``) is given, the adapter is
        free to compute only the active tokens and splice the rest from
        ``cache.model_output`` — this is the FLOPs saving that powers the
        accelerator. Adapters that cannot do sparse attention may compute densely
        and still splice the output (correct, but no saving).
        """

    # -- scheduler update ------------------------------------------------- #

    @abc.abstractmethod
    def scheduler_step(
        self, model_output: Tensor, t: Tensor, t_next: Tensor, tokens: Tensor
    ) -> Tensor:
        """One ODE/SDE solver step: ``z_t -> z_{t_next}`` given ε_θ."""

    # -- composed transitions Φ_t / Φ̃_t (concrete, reused everywhere) ----- #

    def full_transition(
        self,
        tokens: Tensor,
        t: Tensor,
        t_next: Tensor,
        cond: TextConditioning,
        *,
        grid: TokenGrid,
        cache: Optional[BackboneCache] = None,
        want_attention: bool = False,
    ) -> DenoiseOutput:
        """Φ_t: dense denoise + scheduler step. ``out.model_output`` reused as cache.

        Returns a :class:`DenoiseOutput` whose ``model_output`` field has been
        replaced by the *next latent* ``z_{t_next}`` for convenience, while
        ``cache.model_output`` retains the raw ε_θ for reuse.
        """
        out = self.denoise(
            tokens, t, cond, grid=grid, cache=cache, want_attention=want_attention
        )
        z_next = self.scheduler_step(out.cache.model_output, t, t_next, tokens)
        return DenoiseOutput(model_output=z_next, cache=out.cache, attention=out.attention)

    # -- optional hooks --------------------------------------------------- #

    def dit_blocks(self) -> List[nn.Module]:
        """Transformer blocks exposed for Stage-C LoRA (§7.1.3). Empty if N/A."""
        return []

    def recompute_kv(
        self, tokens: Tensor, cond: TextConditioning, token_indices: Tensor, cache: BackboneCache
    ) -> BackboneCache:
        """Refresh the KV cache for ``token_indices`` after a RAEC repair (§5.3.2).

        Default no-op for adapters without an explicit KV cache.
        """
        return cache

    @property
    def module(self) -> Optional[nn.Module]:
        """The underlying ``nn.Module`` (for freezing / device moves). May be None."""
        return None
