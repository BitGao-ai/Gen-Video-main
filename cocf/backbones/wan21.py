"""Wan2.1 backbone adapter (cross-attention DiT via diffusers)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from cocf.backbones.base import TextConditioning
from cocf.backbones.diffusers_base import DiffusersVideoBackbone
from cocf.common.config import BackboneConfig
from cocf.common.registry import register_backbone

Tensor = torch.Tensor


@register_backbone("wan21")
@register_backbone("wan2.1")
@register_backbone("wan")
class Wan21Backbone(DiffusersVideoBackbone):
    """Alibaba Wan2.1 adapter: cross-attention DiT with a umT5 text encoder."""

    patch = (1, 2, 2)
    vae_compress = (4, 8, 8)
    _latent_channels = 16

    def _load(self) -> None:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel

        path = self.config.model_path
        extra = self.config.extra or {}
        hf = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}  # avoid the fp32 host-RAM transient
        self.vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", **hf)
        self.transformer = WanTransformer3DModel.from_pretrained(path, subfolder="transformer", **hf)
        self.text_encoder = UMT5EncoderModel.from_pretrained(path, subfolder="text_encoder", **hf)
        self.tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        self._max_len = int(extra.get("max_text_len", 512))

    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        self._ensure_loaded()
        # no_grad, not inference_mode: the conditioning feeds grad-enabled forwards in Stage C.
        with torch.no_grad(), self._module_active(self.text_encoder):
            tok = self.tokenizer(
                [prompt_clean(prompt) for prompt in prompts],
                return_tensors="pt", padding="max_length",
                truncation=True, max_length=self._max_len,
                add_special_tokens=True, return_attention_mask=True,
            ).to(self.device)
            seq = self.text_encoder(**tok).last_hidden_state  # [B, L, d_t5]
            seq = seq.masked_fill(~tok["attention_mask"].bool().unsqueeze(-1), 0)
        return TextConditioning(
            embeds=seq, mask=tok["attention_mask"], pooled=None, prompts=tuple(prompts)
        )

    def _text_kwargs(
        self, cond: TextConditioning, module: Optional[torch.nn.Module]
    ) -> Dict[str, Any]:
        # Wan attends to the full zero-padded sequence; trimming would change its attention.
        embeds = cond.embeds.to(self.device, self.dtype)
        if cond.mask is not None:
            mask = cond.mask.to(device=embeds.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != embeds.shape[:2]:
                raise ValueError("Wan text mask must match embedding batch and sequence dimensions")
            embeds = embeds.masked_fill(~mask.unsqueeze(-1), 0)
        return {"encoder_hidden_states": embeds}

    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        timestep = (self.model_sigma(t.to(self.device)) * 1000.0).flatten()
        out = self.transformer(  # type: ignore[union-attr]
            hidden_states=latent_grid,
            timestep=timestep,
            **self._text_kwargs(cond, self.transformer),
            return_dict=True,
        )
        eps = out.sample if hasattr(out, "sample") else out[0]
        return eps, {}
