"""HunyuanVideo backbone adapter (MMDiT via diffusers)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from cocf.backbones.base import TextConditioning
from cocf.backbones.diffusers_base import DiffusersVideoBackbone
from cocf.common.config import BackboneConfig
from cocf.common.registry import register_backbone

Tensor = torch.Tensor


@register_backbone("hunyuanvideo")
@register_backbone("hunyuan")
class HunyuanVideoBackbone(DiffusersVideoBackbone):
    """Tencent HunyuanVideo adapter: MMDiT with LLM + CLIP text conditioning."""

    patch = (1, 2, 2)
    vae_compress = (4, 8, 8)
    _latent_channels = 16

    def _load(self) -> None:
        from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
        from transformers import (
            AutoTokenizer,
            CLIPTextModel,
            CLIPTokenizer,
            LlamaModel,
        )

        path = self.config.model_path
        extra = self.config.extra or {}
        hf = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}  # avoid the fp32 host-RAM transient
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(path, subfolder="vae", **hf)
        self.transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            path, subfolder="transformer", **hf
        )
        self.text_encoder = LlamaModel.from_pretrained(path, subfolder="text_encoder", **hf)
        self.tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        self._clip = CLIPTextModel.from_pretrained(path, subfolder="text_encoder_2", **hf)  # pooled global condition
        self._clip_tok = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2")
        self._clip.requires_grad_(False)
        self._clip.to(self.device, self.dtype).eval()
        self._max_len = int(extra.get("max_text_len", 256))

    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        self._ensure_loaded()
        # no_grad, not inference_mode: the conditioning feeds grad-enabled forwards in Stage C.
        with torch.no_grad(), self._module_active(self.text_encoder):
            tok = self.tokenizer(
                list(prompts), return_tensors="pt", padding="max_length",
                truncation=True, max_length=self._max_len,
            ).to(self.device)
            seq = self.text_encoder(**tok).last_hidden_state  # [B, L, d_llm]
            clip_tok = self._clip_tok(
                list(prompts), return_tensors="pt", padding="max_length",
                truncation=True, max_length=77,
            ).to(self.device)
            pooled = self._clip(**clip_tok).pooler_output  # [B, d_clip]
        return TextConditioning(
            embeds=seq, mask=tok["attention_mask"], pooled=pooled, prompts=tuple(prompts)
        )

    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        timestep = (t.to(self.device) * 1000.0).flatten()
        out = self.transformer(  # type: ignore[union-attr]
            hidden_states=latent_grid,
            timestep=timestep,
            **self._text_kwargs(cond, self.transformer),
            pooled_projections=cond.pooled.to(self.device, self.dtype)
            if cond.pooled is not None else None,
            return_dict=True,
        )
        eps = out.sample if hasattr(out, "sample") else out[0]
        attn: Dict[str, Tensor] = {}
        return eps, attn
