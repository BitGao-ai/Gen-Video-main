"""Text–tube alignment ``A_align`` (§6.3.1).

Builds the fine-grained alignment matrix between text tokens and semantic tubes,
the object the cross-modal conservation loss (§6.3.2) and the certificate's local
CMSC term (§5.3.1) are computed from::

    A_align[l, k] = softmax_k( ⟨ φ_txt(c_l), φ_vis(v_k) ⟩ / τ )

where ``φ_txt`` and ``φ_vis`` project the text-token embedding ``c_l`` and the
tube visual embedding ``v_k`` into a shared ``align_dim`` space (the learnable
projection ``W`` of §6.3.1, generalised to two encoders so text-dim ≠ visual-dim
is handled), and ``τ`` is the temperature.

The same projections are reused for the *full* and *accelerated* videos so the
conservation loss compares like with like; and a cheap, inference-time
``tube_scores`` (max alignment per tube) doubles as the proxy the RAEC certificate
needs when the full-compute reference is unavailable (the whole point of
accelerating). This is the only learnable piece of CMSC.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.common.config import CMSCConfig

Tensor = torch.Tensor


class TextTubeAlignment(nn.Module):
    """Projects text tokens and tube visual embeds into a shared space and aligns.

    Parameters
    ----------
    cfg
        CMSC config (temperature, projection ``align_dim``).
    text_dim, visual_dim
        Input widths of the text-token embedding and the tube visual embedding.
        Supplied by the core facade from the backbone text encoder and the
        perception provider, so this module never hard-codes a backbone.
    """

    def __init__(self, cfg: CMSCConfig, text_dim: int, visual_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_proj = nn.Linear(text_dim, cfg.align_dim, bias=False)
        self.vis_proj = nn.Linear(visual_dim, cfg.align_dim, bias=False)

    # ------------------------------------------------------------------ #

    def _project(self, text_embeds: Tensor, tube_embeds: Tensor) -> tuple:
        """Return L2-normalised projected ``(text [L, a], tube [K, a])``."""
        t = F.normalize(self.text_proj(text_embeds.float()), dim=-1)
        v = F.normalize(self.vis_proj(tube_embeds.float()), dim=-1)
        return t, v

    def matrix(self, text_embeds: Tensor, tube_embeds: Tensor) -> Tensor:
        """Alignment matrix ``[L, K]`` — softmax over tubes for each text token.

        ``text_embeds`` is ``[L, d_c]`` (a single prompt's token sequence) and
        ``tube_embeds`` is ``[K, d_v]`` (one pooled visual embedding per tube).
        """
        t, v = self._project(text_embeds, tube_embeds)
        logits = (t @ v.T) / max(self.cfg.temperature, 1e-6)  # [L, K]
        return torch.softmax(logits, dim=-1)

    def tube_scores(self, text_embeds: Tensor, tube_embeds: Tensor) -> Tensor:
        """Per-tube alignment strength ``[K]`` = max over text tokens of cos-sim.

        Available at *inference* (needs only the prompt and the tube's current
        visual embed), so it feeds the certificate's local CMSC proxy without the
        full-compute reference.
        """
        t, v = self._project(text_embeds, tube_embeds)
        sim = v @ t.T  # [K, L] cosine in [-1, 1]
        return sim.max(dim=-1).values.clamp(-1.0, 1.0) * 0.5 + 0.5  # → [0, 1]

    @staticmethod
    def stack_tube_embeds(
        tube_embeds: Dict[int, Tensor], ids: List[int], dim: int, device=None
    ) -> Tensor:
        """Stack a ``{tube_id: [d_v]}`` map into ``[K, d_v]`` in ``ids`` order.

        Missing tubes get a zero row so the matrix stays well-defined when a tube
        lacks a visual embedding on some frame.
        """
        rows = [
            tube_embeds[i] if i in tube_embeds and tube_embeds[i] is not None
            else torch.zeros(dim, device=device)
            for i in ids
        ]
        return torch.stack(rows).to(device) if rows else torch.zeros(0, dim, device=device)
