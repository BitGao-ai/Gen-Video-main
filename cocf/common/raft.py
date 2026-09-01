"""torchvision RAFT loading — shared by the STA perception backend and the damage
metric extractor.

Both subsystems need dense optical flow and both used to build it inline behind a
bare ``except Exception``, so a RAFT that failed to load degraded to a zero (or
proxy) flow field with no log line at all. That failure is silent all the way down:
``motion_phase`` — and therefore the causal action strength ``s_A`` — becomes
identically zero, the affinity flow/IoU terms lose their warp, and two damage axes
fall back to a descriptor difference, while the run reports nothing unusual.

``DEFAULT`` weights are also a *download*, which is exactly what an air-gapped
training box cannot do, so the failure is the expected case there rather than an
exotic one. Hence one loader, with a local-checkpoint path, a load-time probe and a
``required`` mode that refuses to produce unusable labels.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cocf.common.logging import get_logger
from cocf.common.memory import freeze

_log = get_logger(__name__)

__all__ = ["RaftUnavailable", "load_raft"]


class RaftUnavailable(RuntimeError):
    """RAFT could not be built and the caller declared flow mandatory."""


def _build(variant: str, weights_path: Optional[str]) -> nn.Module:
    from torchvision.models.optical_flow import (
        Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
    )

    small = variant == "small"
    ctor = raft_small if small else raft_large
    if weights_path:
        model = ctor(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return model
    return ctor(weights=(Raft_Small_Weights if small else Raft_Large_Weights).DEFAULT)


def _probe(raft: nn.Module, device) -> None:
    """One forward over a synthetic shifted pattern; raises unless the flow is real.

    A checkpoint that loads but produces nothing (a mismatched state dict silently
    accepted, a build without the correlation kernel) is indistinguishable from a
    healthy one until the labels are already written, so the flow is required to be
    non-zero *here* rather than assumed downstream.
    """
    a = torch.zeros(1, 3, 64, 64, device=device)
    a[:, :, 16:48, 16:48] = 1.0
    b = torch.roll(a, shifts=4, dims=-1)
    with torch.no_grad():
        flow = raft(a * 2 - 1, b * 2 - 1)[-1]
    if not torch.isfinite(flow).all() or float(flow.abs().max()) <= 1e-3:
        raise RaftUnavailable(
            f"RAFT produced a zero/non-finite flow field for the load-time probe "
            f"(max |flow| = {float(flow.abs().max()):.3e})"
        )


def load_raft(
    device,
    *,
    variant: str = "large",
    weights_path: Optional[str] = None,
    required: bool = False,
) -> Optional[nn.Module]:
    """A frozen, probed RAFT on ``device`` — or ``None`` when unavailable.

    ``required`` is set by the ``--real-models`` path, whose whole contract is that
    every perception model is the real one: degrading to a zero flow there produces
    labels that cannot train the plugins, so it raises instead. Without it the caller
    keeps its own fallback and this only guarantees the failure is *reported*.
    """
    try:
        raft = freeze(_build(variant, weights_path).to(device))
        _probe(raft, device)
    except Exception as exc:
        detail = (f"local checkpoint {weights_path}" if weights_path
                  else "the torchvision DEFAULT weights (needs network access)")
        message = (
            f"RAFT ({variant}) unavailable from {detail}: {type(exc).__name__}: {exc}. "
            "Optical flow drives motion_phase (hence the causal action strength s_A), "
            "the affinity flow/IoU terms and two damage axes — without it they are "
            "constant and the labels are not trainable. Pass --raft-weights with a "
            "local checkpoint on an offline host."
        )
        if required:
            raise RaftUnavailable(message) from exc
        _log.error(message)
        return None
    _log.info("RAFT (%s) loaded and verified at load%s", variant,
              f" from {weights_path}" if weights_path else "")
    return raft
