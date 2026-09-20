"""Text-tube alignment matrix and scores."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.common.config import CMSCConfig

Tensor = torch.Tensor


def alignment_risk(scores: Tensor) -> Tensor:
    """Convert alignment scores to risk values."""
    return ((0.5 - scores) * 2.0).clamp(0.0, 1.0)


class TextTubeAlignment(nn.Module):
    """Projects text and tube embeds into shared space."""

    def __init__(self, cfg: CMSCConfig, text_dim: int, visual_dim: int) -> None:
        """Create alignment with text and visual dims."""
        super().__init__()
        self.cfg = cfg
        self.text_proj = nn.Linear(text_dim, cfg.align_dim, bias=False)
        self.vis_proj = nn.Linear(visual_dim, cfg.align_dim, bias=False)

    def _project(self, text_embeds: Tensor, tube_embeds: Tensor) -> tuple:
        """Return normalized projected text and tube embeds."""
        t = F.normalize(self.text_proj(text_embeds.float()), dim=-1)
        v = F.normalize(self.vis_proj(tube_embeds.float()), dim=-1)
        return t, v

    def matrix(self, text_embeds: Tensor, tube_embeds: Tensor) -> Tensor:
        """Compute alignment matrix over tubes."""
        t, v = self._project(text_embeds, tube_embeds)
        logits = (t @ v.T) / max(self.cfg.temperature, 1e-6)
        return torch.softmax(logits, dim=-1)

    def tube_scores(self, text_embeds: Tensor, tube_embeds: Tensor) -> Tensor:
        """Compute per-tube alignment strengths."""
        t, v = self._project(text_embeds, tube_embeds)
        sim = v @ t.T  # [K, L] cosine in [-1, 1]
        return sim.max(dim=-1).values.clamp(-1.0, 1.0) * 0.5 + 0.5  # → [0, 1]

    @staticmethod
    def stack_tube_embeds(
        tube_embeds: Dict[int, Tensor], ids: List[int], dim: int, device=None
    ) -> Tensor:
        """Stack tube embeds into matrix in id order."""
        rows = [tube_embeds.get(i) for i in ids]
        target = device
        if target is None:
            present = next((r for r in rows if r is not None), None)
            target = present.device if present is not None else None
        rows = [
            r.to(target) if r is not None else torch.zeros(dim, device=target)
            for r in rows
        ]
        return torch.stack(rows) if rows else torch.zeros(0, dim, device=target)
