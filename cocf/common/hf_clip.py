"""Pooled CLIP embeddings with explicit projection."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

Tensor = torch.Tensor

__all__ = ["clip_image_embed", "clip_text_embed", "clip_text_inputs",
           "clip_context_length"]

_CLIP_CTX_DEFAULT = 77
_CTX_SENTINEL = 10 ** 6


def _pooler_output(out) -> Tensor:
    """Return pooled output from encoder result."""
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(out, (tuple, list)) and len(out) > 1:
        return out[1]
    raise ValueError(
        f"CLIP tower output {type(out).__name__} exposes no ``pooler_output``; "
        "cannot take the pooled embedding the projection expects."
    )


def _checked(emb: Tensor, clip, what: str) -> Tensor:
    """Validate joint-space embedding shape."""
    want = getattr(getattr(clip, "config", None), "projection_dim", None)
    if emb.ndim != 2 or (want is not None and emb.shape[-1] != int(want)):
        raise ValueError(
            f"{what} must be a [B, {want if want is not None else 'd_clip'}] pooled "
            f"CLIP embedding, got {tuple(emb.shape)}. A 3-D shape means a token "
            f"sequence came back instead of the projected vector — check the "
            f"transformers version / the CLIP checkpoint."
        )
    return emb


def clip_image_embed(clip, pixel_values: Tensor) -> Tensor:
    """Return CLIP joint-space image embedding."""
    vision = getattr(clip, "vision_model", None)
    proj = getattr(clip, "visual_projection", None)
    if vision is None or proj is None:
        return _checked(clip.get_image_features(pixel_values=pixel_values),
                        clip, "get_image_features()")
    return _checked(proj(_pooler_output(vision(pixel_values=pixel_values))),
                    clip, "visual_projection(vision_model(...))")


def clip_text_embed(clip, **text_inputs: Tensor) -> Tensor:
    """Return CLIP joint-space text embedding."""
    text = getattr(clip, "text_model", None)
    proj = getattr(clip, "text_projection", None)
    if text is None or proj is None:
        return _checked(clip.get_text_features(**text_inputs),
                        clip, "get_text_features()")
    return _checked(proj(_pooler_output(text(**text_inputs))),
                    clip, "text_projection(text_model(...))")


def clip_context_length(clip, processor=None,
                        default: int = _CLIP_CTX_DEFAULT) -> int:
    """Return text tower context window."""
    cfg = getattr(clip, "config", None)
    for owner in (getattr(cfg, "text_config", None), cfg):
        n = getattr(owner, "max_position_embeddings", None)
        if isinstance(n, int) and 0 < n < _CTX_SENTINEL:
            return n
    tok = getattr(processor, "tokenizer", processor)
    n = getattr(tok, "model_max_length", None)
    if isinstance(n, int) and 0 < n < _CTX_SENTINEL:
        return n
    return default


def clip_text_inputs(
    clip,
    processor,
    prompts: Union[str, Sequence[str]],
    *,
    device: Optional[Union[str, torch.device]] = None,
):
    """Tokenise prompts truncated to context window."""
    if isinstance(prompts, str):
        prompts = [prompts]
    texts = [p if (p and p.strip()) else " " for p in prompts]
    tok = getattr(processor, "tokenizer", processor)
    inputs = tok(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=clip_context_length(clip, processor),
    )
    return inputs.to(device) if device is not None else inputs
