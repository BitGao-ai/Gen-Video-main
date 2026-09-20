"""Per-tube anchor library for revocable rollback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from cocf.common.types import SemanticTube

Tensor = torch.Tensor


@dataclass
class _Anchor:
    """Cached safe snapshot for one tube."""
    step: int
    tokens: Tensor
    indices: Tensor


class AnchorStore:
    """Per-tube safe latent snapshots for rollback."""

    def __init__(self, offload_to_cpu: bool = False) -> None:
        """Create store with optional CPU offload."""
        self.offload = offload_to_cpu
        self._anchors: Dict[int, _Anchor] = {}

    def update(self, tube: SemanticTube, z: Tensor, step: int) -> None:
        """Snapshot tube tokens as its safe anchor."""
        idx = tube.all_token_indices().to(z.device)
        if idx.numel() == 0:
            return
        tokens = z[:, idx].detach()
        store_dev = "cpu" if self.offload else z.device
        self._anchors[tube.tube_id] = _Anchor(
            step=step, tokens=tokens.to(store_dev), indices=idx.to(store_dev)
        )
        tube.last_safe_anchor_step = step

    def has(self, tube_id: int) -> bool:
        """Check if tube has an anchor."""
        return tube_id in self._anchors

    def anchor_step(self, tube_id: int) -> Optional[int]:
        """Return anchor step for a tube."""
        a = self._anchors.get(tube_id)
        return a.step if a is not None else None

    def age(self, tube_id: int, step: int) -> int:
        """Return steps since tube was last anchored."""
        a = self._anchors.get(tube_id)
        return max(0, int(step)) if a is None else max(0, step - a.step)

    def rollback(self, z: Tensor, tube: SemanticTube) -> Tensor:
        """Restore tube tokens from its safe anchor."""
        a = self._anchors.get(tube.tube_id)
        if a is None:
            return z
        out = z.clone()
        self.scatter_into(out, tube)
        return out

    def scatter_into(self, z: Tensor, tube: SemanticTube) -> bool:
        """Write anchor into latent in place."""
        a = self._anchors.get(tube.tube_id)
        if a is None:
            return False
        idx = a.indices.to(z.device)
        z.index_copy_(1, idx, a.tokens.to(z.device, z.dtype))
        return True

    def get_tokens(self, tube_id: int, device=None, dtype=None) -> Optional[Tensor]:
        """Return stored safe tokens for a tube."""
        a = self._anchors.get(tube_id)
        if a is None:
            return None
        return a.tokens.to(device, dtype) if device is not None else a.tokens

    def clear(self) -> None:
        """Clear all stored anchors."""
        self._anchors.clear()

    def retain(self, tube_ids) -> int:
        """Drop anchors for retired tubes."""
        keep = set(tube_ids)
        stale = [tid for tid in self._anchors if tid not in keep]
        for tid in stale:
            del self._anchors[tid]
        return len(stale)
