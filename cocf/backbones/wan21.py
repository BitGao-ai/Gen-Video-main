"""Wan2.1 backbone adapter (§9.1 — secondary backbone for cross-model validation).

Wraps Alibaba's Wan2.1 text-to-video model (the doc targets the 1.3B and 14B
variants). Unlike Hunyuan's joint attention, Wan2.1 is a **cross-attention DiT**:
the video latent self-attends and *cross*-attends to a frozen umT5 text encoding.
We integrate via 🤗 ``diffusers``:

    AutoencoderKLWan          3D causal VAE (8× spatial, 4× temporal, C=16)
    WanTransformer3DModel     the cross-attention DiT (patch (1,2,2))
    UMT5EncoderModel          the umT5 text encoder → conditioning ``c``

Wiring a *second*, architecturally-different backbone through the *same*
:class:`DiffusersVideoBackbone` base — changing only ``_load`` and the transformer
call — is the concrete demonstration of requirement #2 (multi-model support).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

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
    patch = (1, 2, 2)
    vae_compress = (4, 8, 8)
    _latent_channels = 16

    def _load(self) -> None:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel

        path = self.config.model_path
        extra = self.config.extra or {}
        # ``torch_dtype`` + ``low_cpu_mem_usage``: without them ``from_pretrained``
        # materialises fp32 on the CPU and only the subsequent ``.to(device, dtype)``
        # narrows it, so a 14B denoiser costs 56 GB of host RAM during load alone.
        hf = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}
        self.vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", **hf)
        self.transformer = WanTransformer3DModel.from_pretrained(path, subfolder="transformer", **hf)
        self.text_encoder = UMT5EncoderModel.from_pretrained(path, subfolder="text_encoder", **hf)
        self.tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        self._max_len = int(extra.get("max_text_len", 512))

    # -- text ----------------------------------------------------------- #

    def encode_text(self, prompts: Sequence[str]) -> TextConditioning:
        self._ensure_loaded()
        # umT5-XXL is ~11 GB in bf16 and runs once per prompt, so under
        # ``offload_text_encoder`` it only occupies VRAM for this one forward
        # (:meth:`DiffusersVideoBackbone._module_active` is a no-op otherwise).
        # ``no_grad`` — not ``inference_mode``: the frozen encoder never needs a
        # graph, but its output *is* consumed by the (possibly grad-enabled) DiT
        # forward in Stage C, and an inference tensor can never be saved for
        # backward — it would poison the whole §4.2 loss path.
        with torch.no_grad(), self._module_active(self.text_encoder):
            tok = self.tokenizer(
                list(prompts), return_tensors="pt", padding="max_length",
                truncation=True, max_length=self._max_len,
            ).to(self.device)
            seq = self.text_encoder(**tok).last_hidden_state  # [B, L, d_t5]
        return TextConditioning(
            embeds=seq, mask=tok["attention_mask"], pooled=None, prompts=tuple(prompts)
        )

    # -- the cross-attention DiT call ----------------------------------- #

    def _run_transformer(
        self, latent_grid: Tensor, t: Tensor, cond: TextConditioning, want_attention: bool
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        timestep = (self.model_sigma(t.to(self.device)) * 1000.0).flatten()
        out = self.transformer(  # type: ignore[union-attr]
            hidden_states=latent_grid,
            timestep=timestep,
            # Padding-trimmed conditioning (+ the attention mask when this model
            # accepts one): umT5 pads every prompt to 512 tokens and cross-attention
            # was being charged for all of them, as real text (§P1-15).
            **self._text_kwargs(cond, self.transformer),
            return_dict=True,
        )
        eps = out.sample if hasattr(out, "sample") else out[0]
        return eps, {}
