"""Tiny mock backbone for CPU testing."""

from __future__ import annotations

import hashlib
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
    """Small deterministic mock backbone."""

    supports_token_sparsity = True

    def __init__(self, config: BackboneConfig) -> None:
        """Create mock backbone from config."""
        super().__init__(config)
        extra = config.extra or {}
        self._c = int(extra.get("latent_channels", 4))
        self._d = int(extra.get("hidden_dim", 32))
        self._patch_t = int(extra.get("patch_t", 1))
        self._patch_s = int(extra.get("patch_s", 8))  # pixel→latent spatial factor
        self._vae_s = int(extra.get("vae_spatial", 8))
        self._d_text = int(extra.get("text_dim", 16))
        self._text_len = int(extra.get("text_len", 8))
        # Deterministic init without resetting the caller's global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(extra.get("seed", 0)))

            d = self._d
            self.patch_embed = nn.Linear(self._c, d)
            self.unpatch = nn.Linear(d, self._c)
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
            p.requires_grad_(False)
        self._net.to(self.device)

    @property
    def latent_channels(self) -> int:
        """Return latent channel count."""
        return self._c

    @property
    def hidden_dim(self) -> int:
        """Return token hidden width."""
        return self._d

    def token_grid(self, num_frames: int, height: int, width: int) -> TokenGrid:
        """Map pixels to token grid."""
        t = max(1, num_frames // self._patch_t)
        h = max(1, height // self._vae_s // self._patch_s)
        w = max(1, width // self._vae_s // self._patch_s)
        return TokenGrid(t=t, h=h, w=w)

    def timesteps(self, num_inference_steps: int) -> Tensor:
        """Return denoising schedule."""
        return torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

    def to_tokens(self, latent_grid: Tensor) -> Tensor:
        """Convert latent grid to tokens."""
        b, c, t, h, w = latent_grid.shape
        x = latent_grid.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        return self.patch_embed(x)

    def to_grid(self, tokens: Tensor, grid: TokenGrid) -> Tensor:
        """Convert tokens to latent grid."""
        b, n, d = tokens.shape
        x = self.unpatch(tokens)
        c = x.shape[-1]
        return x.reshape(b, grid.t, grid.h, grid.w, c).permute(0, 4, 1, 2, 3).contiguous()

    def encode_video(self, video: Tensor) -> Tensor:
        """Encode video to latent grid."""
        b, cpix, f, hp, wp = video.shape
        t = max(1, f // self._patch_t)
        h = max(1, hp // self._vae_s)
        w = max(1, wp // self._vae_s)
        x = torch.nn.functional.adaptive_avg_pool3d(video, (t, h, w))
        if cpix >= self._c:
            x = x[:, : self._c]
        else:
            x = x.repeat(1, (self._c + cpix - 1) // cpix, 1, 1, 1)[:, : self._c]
        return x

    def decode_latent(self, latent_grid: Tensor) -> Tensor:
        """Decode latent grid to video."""
        b, c, t, h, w = latent_grid.shape
        up = torch.nn.functional.interpolate(
            latent_grid, scale_factor=(self._patch_t, self._vae_s, self._vae_s),
            mode="nearest",
        )
        if c >= 3:
            return up[:, :3]
        return up.repeat(1, 3, 1, 1, 1)[:, :3]

    def pixel_span(self, lo: int, hi: int):
        """Return pixel span for latent slots."""
        return (lo * self._patch_t, hi * self._patch_t)

    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        """Encode prompts into conditioning."""
        b = len(prompts)
        embeds = torch.zeros(b, self._text_len, self._d_text)
        for i, p in enumerate(prompts):
            digest = hashlib.md5(p.encode("utf-8")).hexdigest()
            g = torch.Generator().manual_seed(int(digest, 16) % (2 ** 31))
            embeds[i] = torch.randn(self._text_len, self._d_text, generator=g)
        mask = torch.ones(b, self._text_len)
        return TextConditioning(
            embeds=embeds.to(self.device), mask=mask.to(self.device),
            prompts=tuple(prompts),
        )

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
            eps = self._cached_or_zero(cache, tokens)
            return DenoiseOutput(
                model_output=eps, cache=self._mk_cache(eps, cache), attention=attn_out,
                compute_fraction=0.0,
            )

        x = tokens.index_select(1, idx)  # [B, n_act, d] active tokens only
        h = self.norm(x + t_emb)
        sa, _ = self.attn(h, h, h)
        x = x + sa
        ca, w = self.cross(self.norm(x), text, text, need_weights=want_attention)
        x = x + ca
        x = x + self.mlp(self.norm(x))
        eps_active = self.unpatch_noise(x)

        eps = self._cached_or_zero(cache, tokens).clone()
        eps.index_copy_(1, idx, eps_active.to(eps.dtype))
        if want_attention and w is not None:
            full_attn = torch.zeros(b, idx.numel(), text.shape[1], device=tokens.device)
            attn_out["text"] = full_attn  # placeholder layout [B, n_act, L]
        return DenoiseOutput(
            model_output=eps, cache=self._mk_cache(eps, cache), attention=attn_out,
            compute_fraction=(float(idx.numel()) / max(1, n)),  # the mock is genuinely token-sparse
        )

    def unpatch_noise(self, x: Tensor) -> Tensor:
        return self.mlp(self.norm(x))

    def _cached_or_zero(self, cache: Optional[BackboneCache], tokens: Tensor) -> Tensor:
        if cache is not None and cache.model_output is not None:
            return cache.model_output
        return torch.zeros_like(tokens)

    def _mk_cache(self, eps: Tensor, prev: Optional[BackboneCache]) -> BackboneCache:
        step = (prev.step + 1) if prev is not None else 0
        return BackboneCache(model_output=eps.detach(), step=step)

    def scheduler_step(
        self, model_output: Tensor, t: Tensor, t_next: Tensor, tokens: Tensor
    ) -> Tensor:
        dt = (t_next - t).reshape(-1, *([1] * (tokens.dim() - 1))).to(tokens.dtype)
        return tokens + dt * model_output

    @property
    def module(self) -> nn.Module:
        return self._net

    def dit_blocks(self):
        """Sub-blocks on the denoiser's gradient path, exposed for Stage-C LoRA."""
        return [self.mlp, self.t_embed]
