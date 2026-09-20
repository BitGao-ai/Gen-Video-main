"""Dependency-free lightweight LoRA for the Stage-C optional backbone fine-tune.

Adds an optional low-rank adapter to the last few DiT blocks while the backbone bulk
stays frozen, without pulling in ``peft``:

    * :class:`LoRALinear` wraps a frozen ``nn.Linear`` with a low-rank update; ``B`` is
      zero-initialised so the adapter is an exact identity at the start.
    * :func:`inject_lora` swaps the ``nn.Linear`` children of the target blocks for
      ``LoRALinear`` in place and returns the trainable params and wrapper modules.
    * :func:`lora_state_dict` / :func:`load_lora_state_dict` / :func:`attach_lora`
      persist and restore the adapters, which need their own checkpoint because the
      frozen backbone is not an ``nn.Module`` child of the accelerator.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        # A ~ small normal, B = 0, so the update is zero at init (exact identity).
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

    Rebinds each child in its parent's ``_modules`` so the parent's ``forward`` runs
    through the wrapper. Already-wrapped layers are returned as-is so a resumed run
    still hands the optimiser their parameters.
    """
    wrapped: List[LoRALinear] = []
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            wrapped.append(child)
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
    """Inject LoRA into the backbone's last ``last_n_blocks`` DiT blocks.

    Returns ``(trainable_params, lora_modules)``; a backbone with no ``dit_blocks()``
    yields ``([], [])`` with a warning. The backbone's weights are materialised first
    because real adapters build their components lazily on the first forward.
    """
    backbone.ensure_loaded()
    blocks = list(backbone.dit_blocks())
    if not blocks:
        _log.warning(
            "LoRA requested but backbone '%s' exposes no dit_blocks(); skipping LoRA "
            "(plugin fine-tune still runs).", type(backbone).__name__,
        )
        return [], []

    # A MoE backbone returns each expert's own tail so LoRA is not starved onto one.
    targets = list(backbone.lora_target_blocks(last_n_blocks))
    modules: List[LoRALinear] = []
    for block in targets:
        modules.extend(_wrap_linears_in(block, rank, alpha))

    # A resumed run re-injects over adapters restored from a checkpoint; those keep
    # the rank they were trained at, so a different requested rank is ignored.
    mismatched = {m.lora_A.shape[0] for m in modules if m.lora_A.shape[0] != rank}
    if mismatched:
        _log.warning(
            "LoRA: %d existing adapter(s) have rank %s, not the requested %d — keeping "
            "the existing geometry (a checkpoint was attached before injection).",
            sum(1 for m in modules if m.lora_A.shape[0] != rank),
            sorted(mismatched), rank,
        )

    params: List[nn.Parameter] = [p for m in modules for p in (m.lora_A, m.lora_B)]
    # Adapters restored by ``attach_lora`` were frozen for inference; re-arm them here.
    for p in params:
        p.requires_grad_(True)
    _log.info(
        "LoRA: wrapped %d Linear layers across %d block(s) (rank=%d, alpha=%.1f) → %s params",
        len(modules), len(targets), rank, alpha, f"{sum(p.numel() for p in params):,}",
    )
    return params, modules


# --------------------------------------------------------------------------- #
# Checkpointing (the adapters live inside the frozen backbone)
# --------------------------------------------------------------------------- #


def lora_state_dict(backbone: BackboneAdapter) -> Dict[str, Tensor]:
    """Collect every injected adapter's weights, keyed by module path.

    Keyed ``"<root>.<module path>.lora_{A,B}"`` so the mapping survives a save /
    re-inject / load round-trip and keeps a MoE backbone's experts apart.
    """
    out: Dict[str, Tensor] = {}
    for root_name, root in backbone.lora_roots():
        if root is None:
            continue
        for name, m in root.named_modules():
            if isinstance(m, LoRALinear):
                prefix = f"{root_name}.{name}"
                out[f"{prefix}.lora_A"] = m.lora_A.detach().cpu().clone()
                out[f"{prefix}.lora_B"] = m.lora_B.detach().cpu().clone()
    return out


def load_lora_state_dict(
    backbone: BackboneAdapter, state: Dict[str, Tensor], *, strict: bool = False
) -> Tuple[int, List[str]]:
    """Load adapter weights into an **already-injected** backbone.

    Returns ``(num_loaded, missing_keys)``. Call :func:`inject_lora` (or
    :func:`attach_lora`) with the *same* rank / ``last_n_blocks`` first, so the module
    paths line up; ``strict=True`` raises when they do not.
    """
    present: Dict[str, LoRALinear] = {}
    for root_name, root in backbone.lora_roots():
        if root is None:
            continue
        for name, m in root.named_modules():
            if isinstance(m, LoRALinear):
                present[f"{root_name}.{name}"] = m

    loaded, missing = 0, []
    for prefix, module in present.items():
        a, b = state.get(f"{prefix}.lora_A"), state.get(f"{prefix}.lora_B")
        if a is None or b is None:
            missing.append(prefix)
            continue
        with torch.no_grad():
            module.lora_A.copy_(a.to(module.lora_A.device, module.lora_A.dtype))
            module.lora_B.copy_(b.to(module.lora_B.device, module.lora_B.dtype))
        loaded += 1

    unexpected = sorted({k.rsplit(".", 1)[0] for k in state} - set(present))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA state mismatch: {len(missing)} adapter(s) had no saved weights, "
            f"{len(unexpected)} saved key(s) matched no adapter. Re-inject with the "
            f"same rank / last_n_blocks the checkpoint was trained with."
        )
    if missing or unexpected:
        _log.warning(
            "LoRA load: %d/%d adapters restored (%d without saved weights, %d "
            "unmatched saved keys)", loaded, len(present), len(missing), len(unexpected),
        )
    else:
        _log.info("LoRA load: restored %d adapter(s)", loaded)
    return loaded, missing


def attach_lora(
    backbone: BackboneAdapter,
    state: Dict[str, Tensor],
    *,
    rank: int = 16,
    alpha: float = 16.0,
    last_n_blocks: int = 4,
    strict: bool = False,
) -> int:
    """Inject LoRA and immediately restore trained weights — the inference entry.

    Returns the number of adapters restored (0 when the backbone exposes no blocks).
    """
    params, modules = inject_lora(
        backbone, rank=rank, alpha=alpha, last_n_blocks=last_n_blocks
    )
    if not modules:
        return 0
    for p in params:
        p.requires_grad_(False)  # inference: the adapters are frozen too
    loaded, _ = load_lora_state_dict(backbone, state, strict=strict)
    return loaded
