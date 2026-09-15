"""Stage checkpoint layout — plugins *plus* the adapters that live outside them.

The accelerator's ``state_dict()`` covers the four learnable plugins and nothing
else, by design: the frozen backbone is a plain attribute so its billions of weights
never enter an optimiser or a checkpoint (§7.1). Stage-C LoRA, however, is injected
*into* that backbone — so it is invisible to ``state_dict()`` and was being discarded
at the end of every ``--use_lora`` run.

This module owns the resulting two-part layout and the tolerant reader for it::

    {"accelerator": {...plugin tensors...},
     "lora":        {"<root>.<path>.lora_A": tensor, ...},   # optional
     "lora_config": {"rank": 8, "alpha": 16.0, "last_n_blocks": 3}}

Both are kept in *one* place because the failure mode is silent: a writer that forgets
the LoRA half, or a reader that assumes the old bare-``state_dict`` layout, produces a
checkpoint that loads without error and simply lacks the fine-tune. :func:`load_checkpoint`
therefore accepts **either** layout, so Stage-B checkpoints (bare state dicts) and
Stage-C checkpoints stay interchangeable at every entry point.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from cocf.common.logging import get_logger
from cocf.training.lora import attach_lora, lora_state_dict, LoRALinear

_log = get_logger(__name__)

ACCELERATOR_KEY = "accelerator"
LORA_KEY = "lora"
LORA_CONFIG_KEY = "lora_config"


def build_checkpoint(
    accelerator,
    *,
    rank: Optional[int] = None,
    alpha: Optional[float] = None,
    last_n_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the two-part checkpoint from a (possibly LoRA-injected) accelerator.

    The LoRA geometry is stored with the tensors because the adapters must be
    re-injected with the *same* rank / target-block count before they can be loaded
    back (see :func:`cocf.training.lora.attach_lora`).
    """
    ckpt: Dict[str, Any] = {ACCELERATOR_KEY: accelerator.state_dict()}
    from cocf.lcocf.damage import DEFAULT_DAMAGE_WEIGHTS
    ckpt["damage_weights"] = dict(DEFAULT_DAMAGE_WEIGHTS)
    ckpt["risk_definition"] = "neutral_centered_v1"
    ckpt["model_metadata"] = {
        "backbone": (accelerator.config.backbone.name
                     if accelerator.config.backbone.name != "mock" else None),
        "text_dim": accelerator.text_dim,
        "visual_dim": accelerator.visual_dim,
        "token_dim": accelerator.token_dim,
    }
    lora = lora_state_dict(accelerator.backbone)
    if lora:
        modules = {id(m): m for _, root in accelerator.backbone.lora_roots()
                   if root is not None for m in root.modules() if isinstance(m, LoRALinear)}
        geometries = {(m.lora_A.shape[0], m.scaling * m.lora_A.shape[0]) for m in modules.values()}
        if len(geometries) != 1:
            raise ValueError("Mixed LoRA ranks/scales cannot be represented by this checkpoint format")
        rank, alpha = next(iter(geometries))
        for count in range(1, len(list(accelerator.backbone.dit_blocks())) + 1):
            targets = {id(m) for block in accelerator.backbone.lora_target_blocks(count)
                       for m in block.modules() if isinstance(m, LoRALinear)}
            # Target blocks must not include unwrapped linear layers on reload.
            blocks = list(accelerator.backbone.lora_target_blocks(count))
            if targets == set(modules) and all(any(isinstance(m, LoRALinear) for m in b.modules()) for b in blocks):
                last_n_blocks = count
                break
        else:
            raise ValueError("Cannot infer the actual LoRA target block geometry")
        ckpt[LORA_KEY] = lora
        ckpt[LORA_CONFIG_KEY] = {
            "rank": rank, "alpha": alpha, "last_n_blocks": last_n_blocks,
        }
    return ckpt


def is_two_part(ckpt: Any) -> bool:
    """True when ``ckpt`` uses the ``{"accelerator": ...}`` layout."""
    return isinstance(ckpt, Mapping) and ACCELERATOR_KEY in ckpt


def _filter_shape_mismatch(accelerator, state: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop checkpoint tensors whose shape disagrees with the live module.

    The case this exists for is ``cmsc_alignment.vis_proj`` (and its twin inside
    ``cmsc_loss.alignment``): Stage B sizes that projection from the *store's*
    ``tube_visual_embed_full`` width, while Stage C sizes it from the *live* perception
    provider's ``d_clip``. A store written before the :mod:`cocf.common.hf_clip` fix
    holds un-projected CLIP features (ViT-B/32 ⇒ 768) where the fixed code now yields
    the projected joint-space vector (512), so the two disagree and
    ``load_state_dict`` aborts the whole load over one layer.

    Dropping the offender is the right resolution rather than reshaping it: a
    projection trained on a different feature space carries no usable signal into the
    new one, so Stage C re-learns that single 256×d layer from scratch and keeps every
    other plugin weight Stage B produced. Loudly logged — a silent skip here would be
    indistinguishable from a successful resume.
    """
    live = accelerator.state_dict()
    kept, dropped = {}, []
    for k, v in state.items():
        ref = live.get(k)
        if ref is not None and hasattr(v, "shape") and tuple(v.shape) != tuple(ref.shape):
            dropped.append(f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(ref.shape)}")
            continue
        kept[k] = v
    if dropped:
        _log.warning(
            "checkpoint: %d tensor(s) skipped on shape mismatch — these keep their "
            "freshly-initialised values and are trained from scratch:\n  %s\n"
            "If this names cmsc_*alignment.vis_proj, the processed store's tube visual "
            "embeds are a different width than this run's CLIP produces (d_clip = "
            "projection_dim: ViT-B/32 => 512, ViT-L/14 => 768). Point --clip-model at "
            "the encoder Stage A used, or regenerate the store, to reuse Stage B's "
            "alignment head instead of relearning it.",
            len(dropped), "\n  ".join(dropped),
        )
    return kept


def load_checkpoint(
    accelerator,
    ckpt: Any,
    *,
    training_config: Any = None,
    attach: bool = True,
    strict_lora: bool = True,
    allow_shape_mismatch: bool = False,
) -> int:
    """Restore plugin weights from **either** checkpoint layout; return LoRA count.

    Parameters
    ----------
    ckpt
        An already-``torch.load``ed object: the two-part mapping above, or a bare
        ``state_dict`` as Stage B writes.
    training_config
        ``Config.training``, used only for LoRA geometry defaults when a checkpoint
        predates ``lora_config``.
    attach
        Set False to load the plugins alone (e.g. when the caller injects LoRA itself
        with a different geometry, as Stage C does before training).
    allow_shape_mismatch
        Drop (rather than crash on) checkpoint tensors whose shape disagrees with the
        model's — see :func:`_filter_shape_mismatch`. Set False to demand an exact
        match.
    """
    from cocf.lcocf.damage import DEFAULT_DAMAGE_WEIGHTS
    if ckpt.get("damage_weights") != DEFAULT_DAMAGE_WEIGHTS:
        raise ValueError("Checkpoint damage scoring policy is missing or differs from current weights. "
                         "Retrain Stage B with OCR disabled; do not reuse the old smoke checkpoint.")
    state = ckpt[ACCELERATOR_KEY] if is_two_part(ckpt) else ckpt
    metadata = ckpt.get("model_metadata", {}) if is_two_part(ckpt) else {}
    expected = {"backbone": accelerator.config.backbone.name,
                "text_dim": accelerator.text_dim, "visual_dim": accelerator.visual_dim,
                "token_dim": accelerator.token_dim}
    if not allow_shape_mismatch:
        for key, value in metadata.items():
            if key in expected and value is not None and value != expected[key]:
                raise ValueError(f"Checkpoint {key}={value!r}, current model={expected[key]!r}")
    if not is_two_part(ckpt) or ckpt.get("risk_definition") != "neutral_centered_v1":
        _log.warning("Legacy checkpoint: certificate calibration must be validated or retrained "
                     "for neutral-centered CMSC risk inputs")
    if allow_shape_mismatch:
        state = _filter_shape_mismatch(accelerator, state)
    accelerator.load_state_dict(state, strict=not allow_shape_mismatch)
    missing = [k for k in accelerator.state_dict() if k not in state]
    if missing:
        # Includes anything _filter_shape_mismatch just dropped (already detailed
        # above); a key that appears only here was never in the checkpoint at all.
        _log.warning("checkpoint: %d plugin tensor(s) not restored: %s",
                     len(missing), ", ".join(missing[:8]))

    lora = ckpt.get(LORA_KEY) if isinstance(ckpt, Mapping) else None
    if not lora or not attach:
        return 0

    cfg = (ckpt.get(LORA_CONFIG_KEY) if isinstance(ckpt, Mapping) else None) or {}

    def _pick(key: str, attr: str, fallback):
        value = cfg.get(key)
        if value is None:
            value = getattr(training_config, attr, fallback) if training_config else fallback
        return value

    n = attach_lora(
        accelerator.backbone, lora,
        rank=int(_pick("rank", "lora_rank", 16)),
        alpha=float(_pick("alpha", "lora_alpha", 16.0)),
        last_n_blocks=int(_pick("last_n_blocks", "lora_target_last_n_blocks", 4)),
        strict=strict_lora,
    )
    _log.info("checkpoint: re-attached %d LoRA adapter(s)", n)
    return n
