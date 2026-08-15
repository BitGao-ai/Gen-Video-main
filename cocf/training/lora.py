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
    * :func:`lora_state_dict` / :func:`load_lora_state_dict` / :func:`attach_lora`
      persist and restore those adapters. They need their own checkpoint because the
      backbone is intentionally *not* an ``nn.Module`` child of the accelerator, so
      ``Accelerator.state_dict()`` cannot see them (§4.2).

Replacement is done by rebinding the child in its *immediate parent* module's
``_modules`` dict, so a block whose ``forward`` calls ``self.mlp(x)`` transparently
runs through the wrapped layer (no forward-code changes needed).
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
    references the child by name/index) runs through the wrapper.

    An already-wrapped layer is **returned as-is rather than skipped**. Re-injection
    happens on any resumed run — the checkpoint's adapters are attached first, then the
    stage injects — and silently dropping those layers from the result would hand the
    optimiser an empty parameter list, freezing the fine-tune while every log line
    still said ``use_lora``.
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
    """Inject LoRA into the backbone's last ``last_n_blocks`` DiT blocks (§4.2).

    Returns ``(trainable_params, lora_modules)``. ``lora_modules`` is returned so the
    stage can checkpoint the adapters separately (the frozen backbone is not part of
    ``Accelerator.state_dict``). A backbone with no ``dit_blocks()`` yields ``([], [])``
    with a warning — the rest of Stage C (plugin fine-tune) proceeds unchanged.

    The backbone's weights are **materialised first**: real adapters build their
    components lazily on the first forward, so injecting before that saw
    ``transformer is None``, ``dit_blocks() == []`` and turned the whole ``--use_lora``
    flag into a silent no-op (the caller was constructing the stage — hence injecting —
    long before any forward ran).
    """
    backbone.ensure_loaded()
    blocks = list(backbone.dit_blocks())
    if not blocks:
        _log.warning(
            "LoRA requested but backbone '%s' exposes no dit_blocks(); skipping LoRA "
            "(plugin fine-tune still runs).", type(backbone).__name__,
        )
        return [], []

    # Ask the backbone which blocks to wrap: the default is the last ``last_n`` of
    # dit_blocks(), but a Mixture-of-Experts backbone returns each expert's own tail
    # so LoRA is not starved onto a single expert (§7.1.3).
    targets = list(backbone.lora_target_blocks(last_n_blocks))
    modules: List[LoRALinear] = []
    for block in targets:
        modules.extend(_wrap_linears_in(block, rank, alpha))

    # A resumed run re-injects over adapters restored from a checkpoint. Those keep
    # the rank they were trained at, so an injection asking for a different one is
    # silently ignored — say so rather than let the mismatch surface as a confusing
    # shape error at load time.
    mismatched = {m.lora_A.shape[0] for m in modules if m.lora_A.shape[0] != rank}
    if mismatched:
        _log.warning(
            "LoRA: %d existing adapter(s) have rank %s, not the requested %d — keeping "
            "the existing geometry (a checkpoint was attached before injection).",
            sum(1 for m in modules if m.lora_A.shape[0] != rank),
            sorted(mismatched), rank,
        )

    params: List[nn.Parameter] = [p for m in modules for p in (m.lora_A, m.lora_B)]
    # inject_lora is the *training* entry point: adapters restored by ``attach_lora``
    # were frozen for inference, so re-arm them or the optimiser would filter them out.
    for p in params:
        p.requires_grad_(True)
    _log.info(
        "LoRA: wrapped %d Linear layers across %d block(s) (rank=%d, alpha=%.1f) → %s params",
        len(modules), len(targets), rank, alpha, f"{sum(p.numel() for p in params):,}",
    )
    return params, modules


# --------------------------------------------------------------------------- #
# Checkpointing (§4.2 — the adapters live inside the *frozen* backbone)
# --------------------------------------------------------------------------- #


def lora_state_dict(backbone: BackboneAdapter) -> Dict[str, Tensor]:
    """Collect every injected adapter's weights, keyed by module path.

    The backbone is deliberately a plain attribute of the accelerator so its frozen
    billions never enter ``Accelerator.state_dict()`` — which also means the LoRA
    tensors living *inside* it are invisible to ``torch.save(accelerator.state_dict())``
    and were being discarded the moment Stage C finished. This gives them their own
    serialisable view, keyed ``"<root>.<module path>.lora_{A,B}"`` so the mapping
    survives a save → re-inject → load round-trip in a fresh process (and keeps a
    Mixture-of-Experts backbone's two experts apart, see ``BackboneAdapter.lora_roots``).
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
    """Inject LoRA and immediately restore trained weights — the *inference* entry.

    Without this there was no path at all from a Stage-C checkpoint back into a
    generating model: the adapters were trained, then dropped on the floor. Returns
    the number of adapters restored (0 when the backbone exposes no blocks).
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
