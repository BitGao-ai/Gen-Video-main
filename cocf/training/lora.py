"""Dependency-free lightweight LoRA for the Stage-C optional backbone fine-tune (§4.2).

§4.2's gradient scope allows an *optional* LoRA adapter on "最后若干层 DiT 的 LoRA 适配器"
(the last few DiT blocks) while the backbone bulk stays frozen. This module provides
that without pulling in ``peft``:

    * :class:`LoRALinear` wraps a frozen ``nn.Linear`` with a low-rank update
      ``y = W·x + (α/r)·(x·Aᵀ)·Bᵀ``. ``B`` is zero-initialised so the adapter is an
      exact identity at the start of fine-tuning (no quality regression on step 0).
    * :func:`inject_lora` swaps the ``nn.Linear`` children of the backbone's last-N
      :meth:`~cocf.backbones.base.BackboneAdapter.dit_blocks` for ``LoRALinear`` in
      place, returning the new trainable parameters (and the wrapper modules for
      checkpointing). It is a **logged no-op** when the backbone exposes no blocks
      (e.g. a real adapter whose weights are absent in a dry run, ``dit_blocks() == []``).

Replacement is done by rebinding the child in its *immediate parent* module's
``_modules`` dict, so a block whose ``forward`` calls ``self.mlp(x)`` transparently
runs through the wrapped layer (no forward-code changes needed).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from cocf.backbones.base import BackboneAdapter
from cocf.common.logging import get_logger

Tensor = torch.Tensor
_log = get_logger(__name__)


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` + a trainable low-rank update (identity at init)."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # the pretrained weight stays frozen
        in_f, out_f = base.in_features, base.out_features
        # A ~ small normal, B = 0  ⇒  Δ = 0 at init (exact identity, no warm-up shock)
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scaling = float(alpha) / float(rank)

    def forward(self, x: Tensor) -> Tensor:
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B
        )
        return self.base(x) + self.scaling * update


def _wrap_linears_in(module: nn.Module, rank: int, alpha: float) -> List[LoRALinear]:
    """Recursively replace every ``nn.Linear`` under ``module`` with ``LoRALinear``.

    Rebinds each child in its parent's ``_modules`` so the parent's ``forward`` (which
    references the child by name/index) runs through the wrapper. Skips already-wrapped
    layers so re-injection is idempotent.
    """
    wrapped: List[LoRALinear] = []
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            lora = LoRALinear(child, rank=rank, alpha=alpha)
            lora.to(child.weight.device, child.weight.dtype)
            module._modules[name] = lora
            wrapped.append(lora)
        else:
            wrapped.extend(_wrap_linears_in(child, rank, alpha))
    return wrapped


def inject_lora(
    backbone: BackboneAdapter,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    last_n_blocks: int = 4,
) -> Tuple[List[nn.Parameter], List[LoRALinear]]:
    """Inject LoRA into the backbone's last ``last_n_blocks`` DiT blocks (§4.2).

    Returns ``(trainable_params, lora_modules)``. ``lora_modules`` is returned so the
    stage can checkpoint the adapters separately (the frozen backbone is not part of
    ``Accelerator.state_dict``). A backbone with no ``dit_blocks()`` yields ``([], [])``
    with a warning — the rest of Stage C (plugin fine-tune) proceeds unchanged.
    """
    blocks = list(backbone.dit_blocks())
    if not blocks:
        _log.warning(
            "LoRA requested but backbone '%s' exposes no dit_blocks(); skipping LoRA "
            "(plugin fine-tune still runs).", type(backbone).__name__,
        )
        return [], []

    targets = blocks[-last_n_blocks:] if last_n_blocks > 0 else blocks
    modules: List[LoRALinear] = []
    for block in targets:
        modules.extend(_wrap_linears_in(block, rank, alpha))
    params: List[nn.Parameter] = [p for m in modules for p in (m.lora_A, m.lora_B)]
    _log.info(
        "LoRA: wrapped %d Linear layers across %d block(s) (rank=%d, alpha=%.1f) → %s params",
        len(modules), len(targets), rank, alpha, f"{sum(p.numel() for p in params):,}",
    )
    return params, modules
