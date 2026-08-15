"""Anchor library — the revocable state behind RAEC rollback (§5.3.2).

Cache methods are *irreversible*: a bad cache entry propagates down the denoising
chain and compounds (§5.1). RAEC makes anchoring *revocable* by keeping, per tube,
the latent of its **last verified-safe step** so a later high-risk step can roll
that tube back instead of inheriting corrupted state.

The store is deliberately token-scoped (only a tube's own tokens are kept, not the
whole latent) and supports CPU offload, so retaining anchors for many tubes over a
long denoising trajectory stays within the VRAM budget (user requirement #1):

    update(tube, z, step)      snapshot a tube's tokens as its safe anchor
    rollback(z, tube, …)       scatter the stored safe tokens back into ``z``
    age(tube_id, step)         steps since the tube was last anchored (→ certificate)

Anchors are stored *detached* so they never extend the autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from cocf.common.types import SemanticTube

Tensor = torch.Tensor


@dataclass
class _Anchor:
    step: int
    tokens: Tensor          # [n_tok, d] the tube's latent at the safe step
    indices: Tensor         # [n_tok] flat token indices into the [N] axis


class AnchorStore:
    """Per-tube last-verified-safe latent snapshots for revocable anchoring."""

    def __init__(self, offload_to_cpu: bool = False) -> None:
        self.offload = offload_to_cpu
        self._anchors: Dict[int, _Anchor] = {}

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #

    def update(self, tube: SemanticTube, z: Tensor, step: int) -> None:
        """Snapshot ``tube``'s tokens from latent ``z`` ``[B, N, d]`` as its anchor.

        Call this when a tube has been FULLY computed *and* certified low-risk —
        i.e. it is a trustworthy point to return to.
        """
        idx = tube.all_token_indices().to(z.device)
        if idx.numel() == 0:
            return
        tokens = z[:, idx].detach()  # [B, n_tok, d]
        store_dev = "cpu" if self.offload else z.device
        self._anchors[tube.tube_id] = _Anchor(
            step=step, tokens=tokens.to(store_dev), indices=idx.to(store_dev)
        )
        tube.last_safe_anchor_step = step

    # ------------------------------------------------------------------ #
    # query / rollback
    # ------------------------------------------------------------------ #

    def has(self, tube_id: int) -> bool:
        return tube_id in self._anchors

    def anchor_step(self, tube_id: int) -> Optional[int]:
        a = self._anchors.get(tube_id)
        return a.step if a is not None else None

    def age(self, tube_id: int, step: int) -> int:
        """Steps since this tube was last anchored.

        A tube that has **never** been anchored is the *stalest* case, not the
        freshest: nothing about it has ever been verified safe, so the certificate's
        ``λ_age · age/(age+τ)`` term should charge it the most. Returning 0 (the old
        "treat as fresh") inverted that — the least-trustworthy tubes were priced as
        the most trustworthy, which is the wrong direction for a risk bound (§P1-14).
        Their age is the whole trajectory so far, i.e. ``step``, which is 0 at the
        start (nothing has drifted yet) and grows as the omission persists.
        """
        a = self._anchors.get(tube_id)
        return max(0, int(step)) if a is None else max(0, step - a.step)

    def rollback(self, z: Tensor, tube: SemanticTube) -> Tensor:
        """Return a copy of ``z`` with ``tube``'s tokens restored to its safe anchor.

        No-op (returns ``z`` unchanged) if the tube has no anchor yet — the engine
        then falls back to forcing FULL compute on it.
        """
        a = self._anchors.get(tube.tube_id)
        if a is None:
            return z
        out = z.clone()
        self.scatter_into(out, tube)
        return out

    def scatter_into(self, z: Tensor, tube: SemanticTube) -> bool:
        """Write ``tube``'s anchor into ``z`` **in place**; True if it had one.

        The in-place half of :meth:`rollback`, for callers assembling several tubes
        into one buffer. Restoring K tubes by calling ``rollback`` K times allocates
        and discards K full latents — on Wan2.2 at 480×832×49 that is ~7.8 MB copied
        and thrown away per tube per step, exactly the churn the memory budget is
        trying to avoid. One clone plus K scatters costs a single allocation.
        """
        a = self._anchors.get(tube.tube_id)
        if a is None:
            return False
        idx = a.indices.to(z.device)
        z.index_copy_(1, idx, a.tokens.to(z.device, z.dtype))
        return True

    def get_tokens(self, tube_id: int, device=None, dtype=None) -> Optional[Tensor]:
        """The stored safe tokens ``[B, n_tok, d]`` for a tube (for boundary fusion)."""
        a = self._anchors.get(tube_id)
        if a is None:
            return None
        return a.tokens.to(device, dtype) if device is not None else a.tokens

    def clear(self) -> None:
        self._anchors.clear()

    def retain(self, tube_ids) -> int:
        """Drop anchors whose tube no longer exists; return how many were dropped.

        Called after a re-segmentation. Tube ids are monotonic, so a stale anchor can
        never be *mistaken* for a live tube's — but it still pins the tokens it
        snapshotted in memory for the rest of the generation, and on a long video with
        frequent refreshes that accumulates. Dropping them keeps the store proportional
        to the tubes actually in play.
        """
        keep = set(tube_ids)
        stale = [tid for tid in self._anchors if tid not in keep]
        for tid in stale:
            del self._anchors[tid]
        return len(stale)
