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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from cocf.common.memory import resolve_dtype
from cocf.common.types import TokenGrid

Tensor = torch.Tensor
_log = get_logger(__name__)


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
            self._load()
            for m in (self.vae, self.text_encoder, self.transformer):
                if m is not None:
                    m.to(self.device, self.dtype).eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
            self._loaded = True

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
        # rectified-flow sigmas in (1, 0]; subclasses may override with the
        # upstream scheduler's shifted schedule.
        return torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

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

    def encode_video(self, video: Tensor) -> Tensor:
        self._ensure_loaded()
        with torch.inference_mode():
            x = video.to(self.device, self.dtype)
            lat = self.vae.encode(x).latent_dist.sample()  # type: ignore[union-attr]
            scale = getattr(self.vae.config, "scaling_factor", 1.0)  # type: ignore[union-attr]
            return lat * scale

    def decode_latent(self, latent_grid: Tensor) -> Tensor:
        self._ensure_loaded()
        with torch.inference_mode():
            scale = getattr(self.vae.config, "scaling_factor", 1.0)  # type: ignore[union-attr]
            x = latent_grid.to(self.device, self.dtype) / scale
            return self.vae.decode(x).sample  # type: ignore[union-attr]

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
        kernel a subclass chooses to wire in). When **no** token is active we skip
        the transformer entirely and reuse the cache verbatim — the dominant
        cache-acceleration saving (§5.1).
        """
        self._ensure_loaded()
        b, n, dim = tokens.shape
        if active_mask is not None:
            active = active_mask.reshape(-1) if active_mask.dim() == 1 else active_mask[0]
            if active.sum() == 0 and cache is not None and cache.model_output is not None:
                step = cache.step + 1
                return DenoiseOutput(
                    model_output=cache.model_output,
                    cache=BackboneCache(model_output=cache.model_output, step=step),
                    attention={},
                )

        grid_in = self.to_grid(tokens, grid).to(self.device, self.dtype)
        with torch.inference_mode():
            eps_grid, attn = self._run_transformer(grid_in, t, cond, want_attention)
        eps = self.to_tokens(eps_grid).to(tokens.dtype)

        if active_mask is not None and cache is not None and cache.model_output is not None:
            active = active_mask.reshape(-1).to(tokens.device)
            inactive = ~active if active.dim() == 1 else ~active[0]
            eps[:, inactive] = cache.model_output[:, inactive]

        step = (cache.step + 1) if cache is not None else 0
        out_cache = BackboneCache(model_output=eps.detach(), step=step)
        return DenoiseOutput(model_output=eps, cache=out_cache, attention=attn)

    @abc.abstractmethod
    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run the model-specific transformer, returning (ε grid, attention readouts)."""

    # -- scheduler ------------------------------------------------------- #

    def scheduler_step(
        self, model_output: Tensor, t: Tensor, t_next: Tensor, tokens: Tensor
    ) -> Tensor:
        """Flow-matching Euler step ``z_{t_next} = z_t + (σ_{next}-σ_t)·v``.

        Both Hunyuan and Wan2.1 are rectified-flow models predicting velocity ``v``;
        this is the upstream scheduler's update in closed form. A production
        integration may instead delegate to the model's own
        ``FlowMatchEulerDiscreteScheduler`` for its exact sigma shift.
        """
        dt = (t_next - t).reshape(-1, *([1] * (tokens.dim() - 1))).to(tokens.dtype)
        return tokens + dt * model_output

    def dit_blocks(self) -> List[nn.Module]:
        """Transformer blocks for Stage-C LoRA (§7.1.3)."""
        if self.transformer is None:
            return []
        for attr in ("transformer_blocks", "blocks", "single_transformer_blocks"):
            blocks = getattr(self.transformer, attr, None)
            if blocks is not None:
                return list(blocks)
        return []

    @property
    def module(self) -> Optional[nn.Module]:
        return self.transformer
