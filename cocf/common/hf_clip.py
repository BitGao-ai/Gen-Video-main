"""HuggingFace CLIP adapters — pooled, *projected* embeddings on any ``transformers``.

``CLIPModel.get_image_features`` / ``get_text_features`` historically returned the
pooled, projected joint-space embedding ``[B, projection_dim]`` — the vector CLIP
similarity is actually defined on. Newer ``transformers`` releases unified the
``get_*_features`` names across all multimodal models so they return the encoder's
**token sequence** instead: ``[1, 50, 768]`` for a ViT-B/32 vision tower (1 CLS + 7×7
patches, ``hidden_size`` 768) and ``[1, 77, 512]`` for its text tower (77 context
positions, ``hidden_size`` 512). ``requirements.txt`` pins no ``transformers``
version, so a server rebuild silently switched behaviour underneath the callers.

Nothing here asked for a sequence, so the swap raised nothing where it happened. It
flowed on as an over-long "feature vector" and surfaced far downstream as

    RuntimeError: inconsistent tensor size, expected tensor [38400] and src [39424]

out of the ``img @ txt`` in :meth:`cocf.tubes.model_perception.ModelPerception.\
clip_score` — 50·768 vs 77·512. The same two calls also back the §7.1.1 CLIPScore
metric and the ``tube_clip_embed`` written into every Stage-A sample, so the crash
was the *lucky* outcome: an un-projected 768-wide "CLIP embed" is silently wrong
rather than loudly broken.

The two helpers below do the pooling and the projection explicitly —
``vision_model``/``text_model`` + ``visual_projection``/``text_projection``, which is
exactly the old ``get_*_features`` body and is stable across releases. They are
duck-typed (``transformers`` is never imported here) so this module needs only
``torch``, and they keep the autograd graph intact: the Stage-C §6.3.2 semantic loss
differentiates through :func:`clip_image_embed`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

Tensor = torch.Tensor

__all__ = ["clip_image_embed", "clip_text_embed", "clip_text_inputs",
           "clip_context_length"]

# CLIP's text tower is a *fixed-context* transformer: its learned position-embedding
# table has exactly ``max_position_embeddings`` rows (77 on every OpenAI CLIP), and
# ``CLIPTextEmbeddings.forward`` raises on a longer sequence rather than clipping it:
#
#     ValueError: Sequence length must be less than max_position_embeddings
#                 (got `sequence length`: 84 and max_position_embeddings: 77)
#
# So tokenising a prompt without ``truncation`` is a crash waiting for the first long
# caption — and OpenVid-1M's captions are routinely 80–150 tokens, i.e. it fires on
# clip #1 of a real Stage-A run. This lived in two places (the §7.1.1 CLIPScore metric
# and the tube CLIP score) and was fixed in only one; both now go through
# :func:`clip_text_inputs` so it cannot be half-applied again.
_CLIP_CTX_DEFAULT = 77
# HF marks "no limit" with model_max_length = VERY_LARGE_INTEGER (~1e30), which is a
# length, not a limit — treat anything implausible as unset.
_CTX_SENTINEL = 10 ** 6


def _pooler_output(out) -> Tensor:
    """``pooler_output`` of a HF encoder output, dataclass-style or tuple-style.

    Vision: ``post_layernorm(CLS)``. Text: the EOS position's hidden state. Both are
    what the corresponding projection expects as input. A tower that exposes neither
    is reported rather than index-guessed — picking the wrong field would reintroduce
    exactly the silent-wrong-features failure this module exists to prevent.
    """
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(out, (tuple, list)) and len(out) > 1:  # return_dict=False
        return out[1]
    raise ValueError(
        f"CLIP tower output {type(out).__name__} exposes no ``pooler_output``; "
        "cannot take the pooled embedding the projection expects."
    )


def _checked(emb: Tensor, clip, what: str) -> Tensor:
    """Reject anything that is not a ``[B, projection_dim]`` joint-space embedding.

    Cheap, and it keeps the *next* upstream API drift from turning into another
    unreadable matmul-size error several call frames away.
    """
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
    """CLIP joint-space image embedding ``[B, projection_dim]`` (differentiable).

    Parameters
    ----------
    clip
        A ``transformers.CLIPModel`` (or anything exposing ``vision_model`` +
        ``visual_projection``; otherwise its own ``get_image_features`` is trusted).
    pixel_values
        Processor-normalised pixels ``[B, 3, H, W]``, already on the model's device.
    """
    vision = getattr(clip, "vision_model", None)
    proj = getattr(clip, "visual_projection", None)
    if vision is None or proj is None:  # not a CLIPModel — trust the model's own API
        return _checked(clip.get_image_features(pixel_values=pixel_values),
                        clip, "get_image_features()")
    return _checked(proj(_pooler_output(vision(pixel_values=pixel_values))),
                    clip, "visual_projection(vision_model(...))")


def clip_text_embed(clip, **text_inputs: Tensor) -> Tensor:
    """CLIP joint-space text embedding ``[B, projection_dim]``.

    ``text_inputs`` is a tokenizer/processor call's output (``input_ids``,
    ``attention_mask``, …) already moved to the model's device — forwarded verbatim,
    exactly as ``get_text_features(**inputs)`` used to forward it.
    """
    text = getattr(clip, "text_model", None)
    proj = getattr(clip, "text_projection", None)
    if text is None or proj is None:
        return _checked(clip.get_text_features(**text_inputs),
                        clip, "get_text_features()")
    return _checked(proj(_pooler_output(text(**text_inputs))),
                    clip, "text_projection(text_model(...))")


def clip_context_length(clip, processor=None,
                        default: int = _CLIP_CTX_DEFAULT) -> int:
    """The text tower's hard context window, in tokens.

    Read from the model's own config first (``text_config.max_position_embeddings``
    — the table whose size the error above quotes), then from the tokenizer, then
    the 77 every CLIP release has shipped. Asking the model rather than trusting the
    tokenizer's ``model_max_length`` matters because a checkpoint whose tokenizer
    config omits it is exactly the case where a bare ``truncation=True`` silently
    does nothing ("Asking to truncate ... but no maximum length is provided").
    """
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
    """Tokenise ``prompts`` for :func:`clip_text_embed`, truncated to fit the tower.

    Parameters
    ----------
    clip
        The ``CLIPModel`` whose text tower will consume these ids — it defines the
        context window (see :func:`clip_context_length`).
    processor
        A ``CLIPProcessor`` or a bare ``CLIPTokenizer``; the processor's ``.tokenizer``
        is used when present so the image branch is never involved.
    prompts
        One prompt or a batch. Empty/whitespace prompts become ``" "``: CLIP's
        pooling indexes the EOS position, and an empty batch entry has no content
        token to pool between BOS and EOS.
    device
        Optional device for the returned ``BatchEncoding``.

    Truncation keeps the EOS token — HF reserves room for the special tokens before
    cutting — which is what the pooled text embedding is read from, so a truncated
    caption still yields a valid joint-space vector (just of its first ~75 tokens,
    which is all CLIP has ever seen of any prompt).
    """
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
