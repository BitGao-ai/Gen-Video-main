"""A tiny, fully-functional mock backbone for testing the whole pipeline on CPU.

It implements every :class:`BackboneAdapter` method with a small network and —
crucially — *real* token gather/scatter in :meth:`denoise`, so that passing an
``active_mask`` genuinely computes fewer tokens. That lets the unit tests assert
the accelerator's FLOPs/active-ratio behaviour (user requirement #1) without any
multi-billion-parameter weights.

It is registered as ``"mock"`` so a config can select it exactly like a real
backbone, demonstrating that the algorithm code is backbone-agnostic.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from cocf.backbones.base import (
    BackboneAdapter,
    BackboneCache,
    DenoiseOutput,
    TextConditioning,
)
from cocf.common.config import BackboneConfig
from cocf.common.registry import register_backbone
from cocf.common.types import TokenGrid

Tensor = torch.Tensor


@register_backbone("mock")
class MockBackbone(BackboneAdapter):
    """Small deterministic stand-in for HunyuanVideo/Wan2.1.

    Patchify factors and dims are configurable through ``BackboneConfig.extra`` so
    tests can exercise different token-grid shapes.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        extra = config.extra or {}
        self._c = int(extra.get("latent_channels", 4))
        self._d = int(extra.get("hidden_dim", 32))
        self._patch_t = int(extra.get("patch_t", 1))
        self._patch_s = int(extra.get("patch_s", 8))  # pixel→latent spatial factor
        self._vae_s = int(extra.get("vae_spatial", 8))
        self._d_text = int(extra.get("text_dim", 16))
        self._text_len = int(extra.get("text_len", 8))
        torch.manual_seed(extra.get("seed", 0))

        d = self._d
        # patch embed: latent channels -> token dim, and back for to_grid
        self.patch_embed = nn.Linear(self._c, d)
        self.unpatch = nn.Linear(d, self._c)
        # a single attention+MLP block standing in for the DiT stack
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.cross = nn.MultiheadAttention(d, num_heads=4, batch_first=True, kdim=d, vdim=d)
        self.text_proj = nn.Linear(self._d_text, d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.t_embed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self._net = nn.ModuleList(
            [self.patch_embed, self.unpatch, self.norm, self.attn, self.cross,
             self.text_proj, self.mlp, self.t_embed]
        )
        for p in self._net.parameters():
            p.requires_grad_(False)  # frozen, like a real pretrained backbone

    # -- static description --------------------------------------------- #

    @property
    def latent_channels(self) -> int:
        return self._c

    @property
    def hidden_dim(self) -> int:
        return self._d

    def token_grid(self, num_frames: int, height: int, width: int) -> TokenGrid:
        t = max(1, num_frames // self._patch_t)
        h = max(1, height // (self._vae_s * self._patch_s) * self._patch_s) or 1
        # simplest sane mapping for tests: latent grid = pixels / vae / patch
        h = max(1, height // self._vae_s // self._patch_s)
        w = max(1, width // self._vae_s // self._patch_s)
        return TokenGrid(t=t, h=h, w=w)

    def timesteps(self, num_inference_steps: int) -> Tensor:
        return torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

    # -- layout --------------------------------------------------------- #

    def to_tokens(self, latent_grid: Tensor) -> Tensor:
        b, c, t, h, w = latent_grid.shape
        x = latent_grid.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        return self.patch_embed(x)

    def to_grid(self, tokens: Tensor, grid: TokenGrid) -> Tensor:
        b, n, d = tokens.shape
        x = self.unpatch(tokens)
        c = x.shape[-1]
        return x.reshape(b, grid.t, grid.h, grid.w, c).permute(0, 4, 1, 2, 3).contiguous()

    # -- VAE (identity-ish, channel-replicating mock) ------------------- #

    def encode_video(self, video: Tensor) -> Tensor:
        b, cpix, f, hp, wp = video.shape
        t = max(1, f // self._patch_t)
        h = max(1, hp // self._vae_s)
        w = max(1, wp // self._vae_s)
        x = torch.nn.functional.adaptive_avg_pool3d(video, (t, h, w))
        # map pixel channels -> latent channels by tiling/trunc
        if cpix >= self._c:
            x = x[:, : self._c]
        else:
            x = x.repeat(1, (self._c + cpix - 1) // cpix, 1, 1, 1)[:, : self._c]
        return x

    def decode_latent(self, latent_grid: Tensor) -> Tensor:
        b, c, t, h, w = latent_grid.shape
        up = torch.nn.functional.interpolate(
            latent_grid, scale_factor=(self._patch_t, self._vae_s, self._vae_s),
            mode="nearest",
        )
        if c >= 3:
            return up[:, :3]
        return up.repeat(1, 3, 1, 1, 1)[:, :3]

    # -- text ----------------------------------------------------------- #

    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        b = len(prompts)
        # deterministic pseudo-embedding from the hash of each prompt
        embeds = torch.zeros(b, self._text_len, self._d_text)
        for i, p in enumerate(prompts):
            g = torch.Generator().manual_seed(abs(hash(p)) % (2 ** 31))
            embeds[i] = torch.randn(self._text_len, self._d_text, generator=g)
        mask = torch.ones(b, self._text_len)
        return TextConditioning(embeds=embeds, mask=mask, prompts=tuple(prompts))

    # -- denoiser (real gather/scatter sparsity) ------------------------ #

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
        b, n, d = tokens.shape
        text = self.text_proj(cond.embeds.to(tokens.dtype))
        t_emb = self.t_embed(t.reshape(-1, 1).float().to(tokens.dtype)).reshape(b, 1, d)

        if active_mask is None:
            active = torch.ones(n, dtype=torch.bool, device=tokens.device)
        else:
            active = active_mask.reshape(-1)[:n] if active_mask.dim() == 1 else active_mask[0]

        idx = active.nonzero(as_tuple=False).squeeze(-1)
        attn_out = {}
        if idx.numel() == 0:
            # nothing to compute: reuse the whole cached output
            eps = self._cached_or_zero(cache, tokens)
            return DenoiseOutput(model_output=eps, cache=self._mk_cache(eps, cache), attention=attn_out)

        x = tokens.index_select(1, idx)  # [B, n_act, d] — the ONLY tokens we compute
        h = self.norm(x + t_emb)
        sa, _ = self.attn(h, h, h)
        x = x + sa
        ca, w = self.cross(self.norm(x), text, text, need_weights=want_attention)
        x = x + ca
        x = x + self.mlp(self.norm(x))
        eps_active = self.unpatch_noise(x)

        # scatter computed tokens back; inactive tokens come from cache (or zero).
        eps = self._cached_or_zero(cache, tokens).clone()
        eps.index_copy_(1, idx, eps_active.to(eps.dtype))
        if want_attention and w is not None:
            full_attn = torch.zeros(b, idx.numel(), text.shape[1], device=tokens.device)
            attn_out["text"] = full_attn  # placeholder layout [B, n_act, L]
        return DenoiseOutput(model_output=eps, cache=self._mk_cache(eps, cache), attention=attn_out)

    def unpatch_noise(self, x: Tensor) -> Tensor:
        # predict ε in *token* space (same dim as tokens) for a clean scheduler step
        return self.mlp(self.norm(x))

    def _cached_or_zero(self, cache: Optional[BackboneCache], tokens: Tensor) -> Tensor:
        if cache is not None and cache.model_output is not None:
            return cache.model_output
        return torch.zeros_like(tokens)

    def _mk_cache(self, eps: Tensor, prev: Optional[BackboneCache]) -> BackboneCache:
        step = (prev.step + 1) if prev is not None else 0
        return BackboneCache(model_output=eps, step=step)

    # -- scheduler (simple Euler / flow-matching style update) ---------- #

    def scheduler_step(
        self, model_output: Tensor, t: Tensor, t_next: Tensor, tokens: Tensor
    ) -> Tensor:
        dt = (t_next - t).reshape(-1, *([1] * (tokens.dim() - 1))).to(tokens.dtype)
        return tokens + dt * model_output

    @property
    def module(self) -> nn.Module:
        return self._net
