"""HunyuanVideo backbone adapter (§9.1 — main experiment backbone).

Wraps Tencent's HunyuanVideo (the design doc targets "HunyuanVideo-1.5 8B"). The
model is an **MMDiT**: text (an LLM, e.g. LLaVA, optionally + CLIP pooled) and the
video latent attend *jointly* in "double-stream" blocks, then the text is dropped
for "single-stream" blocks. We integrate via 🤗 ``diffusers``:

    AutoencoderKLHunyuanVideo        3D causal VAE (8× spatial, 4× temporal, C=16)
    HunyuanVideoTransformer3DModel   the MMDiT denoiser (patch (1,2,2))
    LLM + CLIP text encoders         conditioning ``c``

Only the parts of the upstream API the framework needs are touched; everything
else (layout, VAE, scheduler) is inherited from :class:`DiffusersVideoBackbone`.
"""

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
    patch = (1, 2, 2)
    vae_compress = (4, 8, 8)
    _latent_channels = 16

    def _load(self) -> None:
        # Lazy import keeps the dependency optional (mock tests don't need it).
        from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
        from transformers import (
            AutoTokenizer,
            CLIPTextModel,
            CLIPTokenizer,
            LlamaModel,
        )

        path = self.config.model_path
        extra = self.config.extra or {}
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(path, subfolder="vae")
        self.transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            path, subfolder="transformer"
        )
        # primary LLM text encoder (token-level sequence used for joint attention)
        self.text_encoder = LlamaModel.from_pretrained(path, subfolder="text_encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        # secondary CLIP encoder for the pooled global condition
        self._clip = CLIPTextModel.from_pretrained(path, subfolder="text_encoder_2")
        self._clip_tok = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2")
        self._clip.to(self.device, self.dtype).eval()
        self._max_len = int(extra.get("max_text_len", 256))

    # -- text ----------------------------------------------------------- #

    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        self._ensure_loaded()
        with torch.inference_mode():
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

    # -- the MMDiT call -------------------------------------------------- #

    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # HunyuanVideoTransformer3DModel returns velocity prediction over the grid.
        # ``encoder_hidden_states`` = LLM sequence, ``pooled_projections`` = CLIP pooled.
        timestep = (t.to(self.device) * 1000.0).flatten()
        out = self.transformer(  # type: ignore[union-attr]
            hidden_states=latent_grid,
            timestep=timestep,
            encoder_hidden_states=cond.embeds.to(self.device, self.dtype),
            encoder_attention_mask=cond.mask.to(self.device) if cond.mask is not None else None,
            pooled_projections=cond.pooled.to(self.device, self.dtype)
            if cond.pooled is not None else None,
            return_dict=True,
        )
        eps = out.sample if hasattr(out, "sample") else out[0]
        attn: Dict[str, Tensor] = {}
        # The text→video attention map (for CMSC/affinity) requires a forward hook on
        # the joint-attention blocks; left as an opt-in to avoid perturbing the graph.
        return eps, attn
