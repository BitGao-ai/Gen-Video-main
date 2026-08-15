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
from cocf.training.lora import attach_lora, lora_state_dict

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
    lora = lora_state_dict(accelerator.backbone)
    if lora:
        ckpt[LORA_KEY] = lora
        ckpt[LORA_CONFIG_KEY] = {
            "rank": rank, "alpha": alpha, "last_n_blocks": last_n_blocks,
        }
    return ckpt


def is_two_part(ckpt: Any) -> bool:
    """True when ``ckpt`` uses the ``{"accelerator": ...}`` layout."""
    return isinstance(ckpt, Mapping) and ACCELERATOR_KEY in ckpt


def load_checkpoint(
    accelerator,
    ckpt: Any,
    *,
    training_config: Any = None,
    attach: bool = True,
    strict_lora: bool = False,
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
    """
    state = ckpt[ACCELERATOR_KEY] if is_two_part(ckpt) else ckpt
    accelerator.load_state_dict(state)

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
