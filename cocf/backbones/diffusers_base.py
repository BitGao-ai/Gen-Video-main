"""Shared base for the real diffusers-backed video backbones (Hunyuan, Wan2.1)."""

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
    """Whether ``module.forward`` takes a keyword called ``name``."""
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

    Subclasses implement :meth:`_load` and :meth:`_run_transformer`.
    """

    patch: Tuple[int, int, int] = (1, 2, 2)  # patchify factor (t, h, w)
    vae_compress: Tuple[int, int, int] = (4, 8, 8)  # VAE compression (t, h, w)
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

    @abc.abstractmethod
    def _load(self) -> None:
        """Populate ``self.vae/text_encoder/tokenizer/transformer`` from weights."""

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            if not self.config.model_path:
                raise RuntimeError(
                    f"{type(self).__name__} needs BackboneConfig.model_path to load weights"
                )
            with normal_mode():  # weights must never become inference tensors
                self._load()
                for m in (self.vae, self.text_encoder, self.transformer):
                    if m is not None:
                        m.to(self._home_device(m), self.dtype).eval()
                        for p in m.parameters():
                            p.requires_grad_(False)
                self._place_auxiliary_modules()
                self._configure_vae_memory()
            self._loaded = True
            self._log_vram_report()

    def ensure_loaded(self) -> None:
        """Public hook to force the lazy ``_load``."""
        self._ensure_loaded()

    def _place_auxiliary_modules(self) -> None:
        """Hook for subclasses to freeze/place extra components. No-op by default."""

    #: Fallback park device when the config names none.
    _DEFAULT_OFFLOAD_DEVICE: str = "cpu"

    @property
    def offload_device(self) -> str:
        """Device where an offloaded component parks while idle (validated once)."""
        cached = getattr(self, "_offload_device", None)
        if cached is None:
            cached = self._resolve_offload_device()
            self._offload_device = cached
        return cached

    def _resolve_offload_device(self) -> str:
        """Validate the configured park device, falling back to CPU with a warning."""
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
        """Resident device for ``module`` under the config's offload policy."""
        if module is None:
            return self.device
        if module is self.text_encoder and self.config.offload_text_encoder:
            return self.offload_device
        return self.device

    def _resident_denoisers(self) -> List[nn.Module]:
        """Denoiser module(s) currently occupying the compute device."""
        t = self.transformer
        if isinstance(t, nn.Module) and self._home_device(t) == self.device:
            return [t]
        return []

    @contextlib.contextmanager
    def _module_active(self, module: Optional[nn.Module]) -> Iterator[None]:
        """Bring an offloaded ``module`` onto the compute device for one call.

        Under ``text_encoder_exclusive`` the resident denoiser is parked for the
        duration so the two never share the card.
        """
        if module is None or self._home_device(module) == self.device:
            yield
            return
        parked: List[nn.Module] = []
        with normal_mode():  # .to() reallocates storage; must run outside inference_mode
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
                for m in parked:
                    m.to(self.device)
                if parked:
                    free_memory()

    def _configure_vae_memory(self) -> None:
        """Enable tiled/sliced VAE encode+decode when ``config.vae_tiling`` is set.

        Raises when the installed diffusers cannot bound decode memory.
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

        enable_slicing = getattr(self.vae, "enable_slicing", None)
        if callable(enable_slicing):
            enable_slicing()
            _log.info("%s: VAE slicing enabled", type(self).__name__)

    def _log_vram_report(self) -> None:
        """One-shot log of resident vs. reserved VRAM after component placement."""
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return
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

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def hidden_dim(self) -> int:
        return self._token_dim  # patchified latent width (C·p), not the DiT width

    def token_grid(self, num_frames: int, height: int, width: int) -> TokenGrid:
        ct, ch, cw = self.vae_compress
        pt, ph, pw = self.patch
        t_lat = (num_frames - 1) // ct + 1
        h_lat, w_lat = height // ch, width // cw
        return TokenGrid(t=t_lat // pt, h=h_lat // ph, w=w_lat // pw)

    def timesteps(self, num_inference_steps: int) -> Tensor:
        return self.model_sigma(
            torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        )

    @property
    def flow_shift(self) -> float:
        """Rectified-flow schedule shift ``s`` (1.0 = uniform), from config extra."""
        return float((self.config.extra or {}).get("flow_shift", 1.0))

    def model_sigma(self, sigma: Tensor) -> Tensor:
        """Map a schedule position in (0,1] to the model's shifted noise level."""
        s = self.flow_shift
        return sigma if s == 1.0 else s * sigma / (1.0 + (s - 1.0) * sigma)

    def to_tokens(self, latent_grid: Tensor) -> Tensor:
        """``[B, C, T, H, W] -> [B, N, C·pt·ph·pw]`` by folding (pt, ph, pw) patches."""
        b, c, t, h, w = latent_grid.shape
        pt, ph, pw = self.patch
        x = latent_grid.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)

    def to_grid(self, tokens: Tensor, grid: TokenGrid) -> Tensor:
        b, n, dim = tokens.shape
        pt, ph, pw = self.patch
        c = dim // (pt * ph * pw)
        x = tokens.reshape(b, grid.t, grid.h, grid.w, c, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(b, c, grid.t * pt, grid.h * ph, grid.w * pw)

    def _latent_norm(self, ref: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
        """Per-channel ``(mean, std)`` of this VAE's latent space, or ``None``.

        ``ref`` supplies device/dtype for the returned statistics.
        """
        cfg = getattr(self.vae, "config", None)
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if mean is None or std is None:
            return None
        c = ref.shape[1]
        if len(mean) != c or len(std) != c:
            raise RuntimeError(
                f"{type(self).__name__}: vae.config latents_mean/latents_std have "
                f"{len(mean)}/{len(std)} entries but the latent has {c} channels"
            )
        kw = {"device": ref.device, "dtype": ref.dtype}
        return (torch.tensor(mean, **kw).view(1, c, 1, 1, 1),
                torch.tensor(std, **kw).view(1, c, 1, 1, 1))

    def encode_video(self, video: Tensor) -> Tensor:
        """``[B, C_pix, F, H, W] -> [B, C, T, H', W']`` in the model's standardised latent space."""
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
        """``[B, C, T, H, W] -> [B, C_pix, F, H_pix, W_pix]``, inverse of :meth:`encode_video`."""
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
        """Causal-temporal VAE layout: only prefix windows (``lo == 0``) are described."""
        if lo > 0:
            return None
        ct = self.vae_compress[0]
        return (0, 0 if hi <= 0 else (hi - 1) * ct + 1)

    def _reclaim_before_vae(self) -> None:
        """Return cached-but-free allocator blocks to the driver before a VAE call."""
        if self.config.vae_tiling and torch.cuda.is_available():
            free_memory()

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

        With no active token the transformer is skipped entirely and the cache is
        reused; ``compute_fraction`` reports what was actually spent.
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
                    compute_fraction=0.0,
                )

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
            compute_fraction = 1.0

        if active is not None and cache is not None and cache.model_output is not None:
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
        """Optional token-sparse eps over ``active`` tokens; ``None`` falls back to dense."""
        return None

    @abc.abstractmethod
    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run the model-specific transformer, returning (eps grid, attention readouts)."""

    @staticmethod
    def _trim_text(cond: TextConditioning) -> Tuple[Tensor, Optional[Tensor]]:
        """Drop trailing positions that are padding in every batch row."""
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

    def scheduler_step(
        self, model_output: Tensor, t: Tensor, t_next: Tensor, tokens: Tensor
    ) -> Tensor:
        """Flow-matching Euler step ``z_{t_next} = z_t + (sigma_next - sigma_t)·v``."""
        sigma, sigma_next = self.model_sigma(t), self.model_sigma(t_next)
        dt = (sigma_next - sigma).reshape(-1, *([1] * (tokens.dim() - 1))).to(tokens.dtype)
        return tokens + dt * model_output

    def dit_blocks(self) -> List[nn.Module]:
        """Transformer blocks for Stage-C LoRA."""
        return self._blocks_of(self.transformer)

    @staticmethod
    def _blocks_of(module: Optional[nn.Module]) -> List[nn.Module]:
        """The transformer-block stack of a diffusers ``*Transformer3DModel`` (or [])."""
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
