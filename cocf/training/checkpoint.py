"""Stage checkpoint layout: plugins plus the adapters that live outside them.

The accelerator's ``state_dict()`` covers only the four learnable plugins (the frozen
backbone is a plain attribute). Stage-C LoRA is injected into that backbone, so it is
stored alongside in a two-part layout handled here::

    {"accelerator":    {...plugin tensors...},
     "damage_weights": {...scoring policy at train time...},
     "lora":           {"<root>.<path>.lora_A": tensor, ...},   # optional
     "lora_config":    {"rank": 8, "alpha": 16.0, "last_n_blocks": 3}}

Every checkpoint also records the damage scoring policy it was trained against:
:func:`load_checkpoint` rejects anything whose ``damage_weights`` differ from the
current defaults.
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
    re-injected with the same rank / target-block count before they can be loaded.
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

    Handles ``cmsc_alignment.vis_proj``, which Stage B sizes from the store's visual
    embed width and Stage C from the live perception provider's ``d_clip``. A mismatched
    projection is dropped (and logged) so Stage C re-learns just that layer and keeps
    every other plugin weight.
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
    allow_incomplete: bool = False,
) -> int:
    """Restore plugin weights from a :func:`build_checkpoint` payload; return LoRA count.

    Rejects bare ``state_dict`` / legacy payloads that carry no ``damage_weights``.
    ``attach=False`` loads the plugins alone; ``allow_shape_mismatch`` drops mismatched
    tensors instead of crashing; ``allow_incomplete`` loads a checkpoint from an
    unfinished phased Stage-B run (rejected by default).
    """
    from cocf.lcocf.damage import DEFAULT_DAMAGE_WEIGHTS
    if ckpt.get("damage_weights") != DEFAULT_DAMAGE_WEIGHTS:
        raise ValueError("Checkpoint damage scoring policy is missing (legacy or bare state_dict "
                         "layout) or differs from the current weights. Retrain Stage B under the "
                         "current scoring policy instead of reusing this checkpoint.")
    phase_state = ckpt.get("phase_state") if isinstance(ckpt, Mapping) else None
    if (
        isinstance(phase_state, Mapping)
        and phase_state.get("phased")
        and not phase_state.get("calibration_complete", False)
    ):
        if not allow_incomplete:
            raise ValueError(
                "Checkpoint is from an incomplete phased Stage-B run "
                f"(phase={phase_state.get('phase')!r}, updates={phase_state.get('updates')}): "
                "variance calibration did not finish, so this is not a deployable model. "
                "Pass allow_incomplete=True to load it for diagnostics anyway."
            )
        _log.warning(
            "loading an incomplete phased Stage-B checkpoint for diagnostics "
            "(phase=%r, updates=%s): variance calibration did not finish — do not "
            "treat its metrics as a deployable model's",
            phase_state.get("phase"), phase_state.get("updates"),
        )
    state = ckpt[ACCELERATOR_KEY]
    metadata = ckpt.get("model_metadata", {})
    expected = {"backbone": accelerator.config.backbone.name,
                "text_dim": accelerator.text_dim, "visual_dim": accelerator.visual_dim,
                "token_dim": accelerator.token_dim}
    if not allow_shape_mismatch:
        for key, value in metadata.items():
            if key in expected and value is not None and value != expected[key]:
                raise ValueError(f"Checkpoint {key}={value!r}, current model={expected[key]!r}")
    if ckpt.get("risk_definition") != "neutral_centered_v1":
        _log.warning("Legacy checkpoint: certificate calibration must be validated or retrained "
                     "for neutral-centered CMSC risk inputs")
    if allow_shape_mismatch:
        state = _filter_shape_mismatch(accelerator, state)
    accelerator.load_state_dict(state, strict=not allow_shape_mismatch)
    missing = [k for k in accelerator.state_dict() if k not in state]
    if missing:
        # Includes anything _filter_shape_mismatch just dropped (detailed above).
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
