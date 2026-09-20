"""Wan2.2 backbone adapter (MoE dual-expert DiT, high-compression VAE variants)."""

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

# Emit the offload-thrash warning once, after this many expert swaps.
_SWAP_WARN_AFTER = 32

#: Wan2.2 variant -> :attr:`BackboneConfig.extra` geometry/schedule overrides.
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

    patch = (1, 2, 2)  # A14B defaults; TI2V-5B overrides via extra
    vae_compress = (4, 8, 8)
    _latent_channels = 16
    _default_boundary_ratio = 0.875  # None in extra forces the single-expert path

    def __init__(self, config: BackboneConfig) -> None:
        extra = config.extra or {}
        if "patch" in extra:
            self.patch = tuple(extra["patch"])
        self._vae_compress_explicit = "vae_compress" in extra  # explicit geometry wins over detection
        if "vae_compress" in extra:
            self.vae_compress = tuple(extra["vae_compress"])
        if "latent_channels" in extra:
            self._latent_channels = int(extra["latent_channels"])
        super().__init__(config)
        self.transformer_2: Optional[nn.Module] = None
        self._resident_expert: Optional[nn.Module] = None  # which expert is on the compute device
        self._expert_swaps = 0
        self._swap_warned = False
        br = extra.get("boundary_ratio", self._default_boundary_ratio)
        self.boundary_ratio: Optional[float] = None if br is None else float(br)
        self.num_train_timesteps = int(extra.get("num_train_timesteps", 1000))
        self._eps_expert: Optional[nn.Module] = None  # expert that produced the cached eps

    @staticmethod
    def _vae_latent_channels(vae: nn.Module) -> Optional[int]:
        """Latent channel count from the VAE config, or ``None`` when unstated."""
        cfg = getattr(vae, "config", None)
        for key in ("z_dim", "latent_channels"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    @staticmethod
    def _detect_vae_compress(vae: nn.Module) -> Optional[Tuple[int, int, int]]:
        """``(t, h, w)`` compression from the VAE config; only explicitly stated values."""
        cfg = getattr(vae, "config", None)
        if cfg is None:
            return None

        temporal = getattr(cfg, "scale_factor_temporal", None)
        spatial = getattr(cfg, "scale_factor_spatial", None)
        if all(isinstance(v, int) and v > 0 for v in (temporal, spatial)):
            return temporal, spatial, spatial

        temporal = getattr(cfg, "temporal_compression_ratio", None)
        spatial = getattr(cfg, "spatial_compression_ratio", None)
        if all(isinstance(v, int) and v > 0 for v in (temporal, spatial)):
            return temporal, spatial, spatial
        return None

    def _load(self) -> None:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel

        path = self.config.model_path
        extra = self.config.extra or {}
        self._check_variant_against_checkpoint(path)
        hf = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}  # avoid the fp32 host-RAM transient
        self.vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", **hf)
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
        self.transformer = WanTransformer3DModel.from_pretrained(path, subfolder="transformer", **hf)
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
        if self.boundary_ratio is not None:
            sub = extra.get("transformer_2_subfolder", "transformer_2")
            try:
                self.transformer_2 = WanTransformer3DModel.from_pretrained(path, subfolder=sub, **hf)
            except (OSError, ValueError) as e:  # subfolder absent => single-expert ckpt
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

    def _check_variant_against_checkpoint(self, path: Optional[str]) -> None:
        """Fail at load when ``--wan-variant`` disagrees with the weights on disk."""
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
        """Freeze and device-place the low-noise expert (base-class hook)."""
        if self.transformer_2 is None:
            return
        self.transformer_2.to(self._home_device(self.transformer_2), self.dtype).eval()
        for p in self.transformer_2.parameters():
            p.requires_grad_(False)
        if self.config.offload_idle_expert:
            self._resident_expert = self.transformer  # schedule starts at the noisiest step

    def _home_device(self, module: Optional[nn.Module]) -> str:
        """Park the idle expert on the offload device when ``offload_idle_expert`` is set."""
        if module is not None and self.config.offload_idle_expert:
            for expert in (self.transformer, self.transformer_2):
                if module is expert:
                    resident = self._resident_expert or self.transformer
                    return self.device if module is resident else self.offload_device
        return super()._home_device(module)

    def _resident_denoisers(self) -> List[nn.Module]:
        """The expert(s) currently occupying the compute device."""
        if not self.config.offload_idle_expert:
            return [m for m in (self.transformer, self.transformer_2) if isinstance(m, nn.Module)]
        resident = self._resident_expert or self.transformer
        return [resident] if isinstance(resident, nn.Module) else []

    def _make_resident(self, want: nn.Module) -> None:
        """Swap ``want`` onto the compute device, evicting the other expert."""
        other = self.transformer_2 if want is self.transformer else self.transformer
        with normal_mode():  # device moves reallocate parameters; avoid inference tensors
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

    def _expert_for(self, timestep: Tensor) -> nn.Module:
        """Pick (and make resident) the denoiser for this step's noise level."""
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
        self._eps_expert = expert
        eps = out.sample if hasattr(out, "sample") else out[0]
        return eps, {}

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
        """Base denoise with an MoE guard: drop the eps cache across the expert boundary."""
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

    def dit_blocks(self) -> List[nn.Module]:
        """Transformer blocks for Stage-C LoRA, from both experts."""
        blocks = list(self._blocks_of(self.transformer))
        blocks.extend(self._blocks_of(self.transformer_2))
        return blocks

    def lora_target_blocks(self, last_n: int) -> List[nn.Module]:
        """Last ``last_n`` blocks of each expert."""
        high = self._blocks_of(self.transformer)
        targets: List[nn.Module] = list(high[-last_n:] if last_n > 0 else high)
        if self.transformer_2 is not None:
            low = self._blocks_of(self.transformer_2)
            targets.extend(low[-last_n:] if last_n > 0 else low)
        return targets

    def lora_roots(self) -> List[Tuple[str, nn.Module]]:
        """Both experts as separately named roots so LoRA checkpoint keys never collide."""
        roots: List[Tuple[str, nn.Module]] = []
        if self.transformer is not None:
            roots.append(("transformer", self.transformer))
        if self.transformer_2 is not None:
            roots.append(("transformer_2", self.transformer_2))
        return roots
