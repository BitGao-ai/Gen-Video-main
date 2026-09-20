"""Backbone abstraction: the uniform interface over video-diffusion backbones."""

from __future__ import annotations

import abc
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cocf.common.config import BackboneConfig
from cocf.common.types import TokenGrid

Tensor = torch.Tensor


def sigma_from_step(t: int, num_steps: int) -> float:
    """Map a reverse step index ``t`` to a flow-matching sigma in [0, 1]."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return t / num_steps


@dataclass
class TextConditioning:
    """Encoded text condition, backbone-agnostic."""

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
    """Reusable state carried between denoise calls."""

    model_output: Optional[Tensor] = None  # [B, N, d_out] last full eps
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
    attention: Dict[str, Tensor] = field(default_factory=dict)  # optional attention readouts
    compute_fraction: float = 1.0  # fraction of a dense forward actually executed


class BackboneAdapter(abc.ABC):
    """Uniform interface over a frozen video-diffusion backbone."""

    #: Whether :meth:`denoise` turns a partial ``active_mask`` into proportionally less compute.
    supports_token_sparsity: bool = False

    def __init__(self, config: BackboneConfig) -> None:
        from cocf.common.memory import resolve_device

        self.config = config
        self.device = resolve_device(config.device)  # falls back to CPU when unavailable
        self._grad_enabled = False  # whether forwards may build an autograd graph

    @contextlib.contextmanager
    def grad_mode(self, enabled: bool = True) -> Iterator["BackboneAdapter"]:
        """Allow (or forbid) backbone forwards to build an autograd graph (re-entrant)."""
        prev = self._grad_enabled
        self._grad_enabled = bool(enabled)
        try:
            yield self
        finally:
            self._grad_enabled = prev

    @property
    def grad_enabled(self) -> bool:
        """True while backbone forwards are allowed to build an autograd graph."""
        return self._grad_enabled

    def _forward_ctx(self):
        """Context for model forwards: inference_mode unless gradients are enabled."""
        return contextlib.nullcontext() if self._grad_enabled else torch.inference_mode()

    def ensure_loaded(self) -> None:
        """Materialise lazily-built weights (no-op for eagerly-built adapters)."""

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
        """Map pixel (frames, H, W) to the post-patchify latent token grid."""

    @abc.abstractmethod
    def timesteps(self, num_inference_steps: int) -> Tensor:
        """Return the descending denoising timestep schedule."""

    @abc.abstractmethod
    def to_tokens(self, latent_grid: Tensor) -> Tensor:
        """``[B, C, T, H, W] -> [B, N, d]`` in (t, h, w) row-major token order."""

    @abc.abstractmethod
    def to_grid(self, tokens: Tensor, grid: TokenGrid) -> Tensor:
        """Inverse of :meth:`to_tokens`."""

    def initial_latent(
        self,
        grid: TokenGrid,
        *,
        batch: int = 1,
        generator: Optional[torch.Generator] = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample a fresh ``z_T ~ N(0, I)`` in token form ``[batch, N, d]``."""
        device = device or self.device
        p_t, p_h, p_w = getattr(self, "patch", (1, 1, 1))
        latent = torch.randn(
            batch, self.latent_channels,
            grid.t * p_t, grid.h * p_h, grid.w * p_w,
            generator=generator, device=device, dtype=dtype,
        )
        return self.to_tokens(latent)

    @abc.abstractmethod
    def encode_video(self, video: Tensor) -> Tensor:
        """``[B, C_pix, F, H_pix, W_pix] -> [B, C, T, H, W]`` latent (no grad)."""

    @abc.abstractmethod
    def decode_latent(self, latent_grid: Tensor) -> Tensor:
        """``[B, C, T, H, W] -> [B, C_pix, F, H_pix, W_pix]`` video in the adapter's own range."""

    #: Whether :meth:`decode_latent` emits the symmetric ``[-1, 1]`` VAE range.
    decode_is_signed: bool = True

    def decode_to_unit(self, latent_grid: Tensor) -> Tensor:
        """Decode to ``[0, 1]`` pixels; the only entry point for image consumers."""
        video = self.decode_latent(latent_grid)
        if self.decode_is_signed:
            video = (video + 1.0) * 0.5
        return video.clamp(0.0, 1.0)

    def pixel_span(self, lo: int, hi: int) -> Optional[Tuple[int, int]]:
        """Pixel-frame range ``[start, stop)`` that latent slots ``[lo, hi)`` decode to.

        ``None`` means the adapter cannot describe this window and callers must
        fall back to a supported window or a full decode.
        """
        return None

    @abc.abstractmethod
    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        """Encode prompts into conditioning ``c``."""

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
        """Predict eps over tokens, computing only ``active_mask`` where possible.

        The adapter must report what it actually spent in
        ``DenoiseOutput.compute_fraction``.
        """

    @abc.abstractmethod
    def scheduler_step(
        self, model_output: Tensor, t: Tensor, t_next: Tensor, tokens: Tensor
    ) -> Tensor:
        """One ODE/SDE solver step: ``z_t -> z_{t_next}`` given eps."""

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
        """Dense denoise + scheduler step; ``model_output`` holds ``z_{t_next}``."""
        out = self.denoise(
            tokens, t, cond, grid=grid, cache=cache, want_attention=want_attention
        )
        z_next = self.scheduler_step(out.cache.model_output, t, t_next, tokens)
        return DenoiseOutput(
            model_output=z_next, cache=out.cache, attention=out.attention,
            compute_fraction=out.compute_fraction,
        )

    def dit_blocks(self) -> List[nn.Module]:
        """Transformer blocks exposed for Stage-C LoRA. Empty if N/A."""
        return []

    def lora_target_blocks(self, last_n: int) -> List[nn.Module]:
        """The DiT blocks Stage-C LoRA should wrap (default: the last ``last_n``)."""
        blocks = list(self.dit_blocks())
        return blocks[-last_n:] if last_n > 0 else blocks

    def lora_roots(self) -> List[Tuple[str, nn.Module]]:
        """Named module roots under which injected LoRA adapters are addressed."""
        module = self.module
        return [("module", module)] if module is not None else []

    def recompute_kv(
        self, tokens: Tensor, cond: TextConditioning, token_indices: Tensor, cache: BackboneCache
    ) -> BackboneCache:
        """Refresh the KV cache for ``token_indices`` after a repair (default no-op)."""
        return cache

    @property
    def module(self) -> Optional[nn.Module]:
        """The underlying ``nn.Module`` (for freezing / device moves). May be None."""
        return None
