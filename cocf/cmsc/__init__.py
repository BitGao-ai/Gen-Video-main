"""CMSC — Cross-Modal Semantic Conservation (§6).

Constrains the prompt↔tube semantic relations to be conserved under acceleration,
turning "accelerated semantics don't collapse" from an empirical hope into a
quantifiable objective with a metric-degradation upper bound (§6.4).

    alignment   §6.3.1  TextTubeAlignment — the learnable text↔tube projection
    losses      §6.3.2  CMSCLoss — the 6-term conservation loss + local proxy
"""

from __future__ import annotations

from cocf.cmsc.alignment import TextTubeAlignment
from cocf.cmsc.losses import CMSCLoss, CMSCObservation

__all__ = ["TextTubeAlignment", "CMSCLoss", "CMSCObservation"]
