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

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.common.logging import get_logger
from cocf.common.memory import freeze

_log = get_logger(__name__)

__all__ = ["RaftUnavailable", "RAFT_MIN_EDGE", "load_raft", "raft_pad"]

#: torchvision's correlation pyramid down-samples the input by 8 and then needs a
#: feature map of at least 16x16, so *both* input edges must be >= 128 or the
#: forward pass raises ``ValueError: Feature maps are too small…``. Every call site
#: that sizes RAFT's input (the load-time probe, the per-frame pad in the
#: perception backend, the resolution cap in the metric extractor) has to respect
#: this floor, so it lives here rather than being re-derived three times.
RAFT_MIN_EDGE = 128


class RaftUnavailable(RuntimeError):
    """RAFT could not be built and the caller declared flow mandatory."""


def raft_pad(x: torch.Tensor) -> torch.Tensor:
    """Pad ``[N,C,H,W]`` up to RAFT's input lattice: multiples of 8, edges >= 128.

    Bottom/right padding only, so the caller recovers its own field with a plain
    ``flow[..., :h, :w]`` slice. ``reflect`` needs the pad to be smaller than the
    dimension it mirrors, which is exactly false for the tiny inputs this floor
    exists for, so those fall back to ``replicate``.
    """
    h, w = x.shape[-2:]
    th = max(RAFT_MIN_EDGE, -(-h // 8) * 8)
    tw = max(RAFT_MIN_EDGE, -(-w // 8) * 8)
    ph, pw = th - h, tw - w
    if not (ph or pw):
        return x
    mode = "reflect" if ph < h and pw < w else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode)


def _resolve_weights(variant: str, weights_path: Optional[str]) -> Optional[Path]:
    """``--raft-weights`` → the checkpoint file for *this* variant.

    The two consumers ask for different nets — the perception backend for
    ``large``, the metric extractor for ``small`` — off the same CLI flag, so a
    single file can only ever satisfy one of them. A *directory* is therefore the
    useful form on an offline host: each caller picks the checkpoint whose name
    carries its own variant.
    """
    if not weights_path:
        return None
    p = Path(weights_path).expanduser()
    if p.is_dir():
        found = sorted(q for q in p.rglob("*") if q.suffix in {".pth", ".pt"})
        matched = [q for q in found if variant in q.name.lower()]
        pool = matched or found
        if len(pool) != 1:
            names = ", ".join(q.name for q in found) or "no .pth/.pt files"
            raise FileNotFoundError(
                f"{p} is a directory and does not identify one RAFT '{variant}' "
                f"checkpoint (contains: {names}). Point --raft-weights at the file "
                f"itself, or at a directory holding one checkpoint per variant with "
                f"'{variant}' in its name."
            )
        return pool[0]
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Drop --raft-weights to use the torchvision "
            f"DEFAULT weights (already downloaded if ~/.cache/torch/hub/checkpoints "
            f"holds a raft_{variant}_*.pth), or pass the checkpoint's real path."
        )
    return p


def _state_dict(obj) -> dict:
    """Unwrap the usual checkpoint envelopes down to a bare state dict."""
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k[7:] if k.startswith("module.") else k: v for k, v in obj.items()}


def _build(variant: str, weights_path: Optional[str]) -> nn.Module:
    from torchvision.models.optical_flow import (
        Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
    )

    small = variant == "small"
    ctor = raft_small if small else raft_large
    resolved = _resolve_weights(variant, weights_path)
    if resolved is not None:
        model = ctor(weights=None)
        model.load_state_dict(_state_dict(torch.load(resolved, map_location="cpu")))
        return model
    return ctor(weights=(Raft_Small_Weights if small else Raft_Large_Weights).DEFAULT)


def _probe(raft: nn.Module, device) -> None:
    """One forward over a synthetic shifted pattern; raises unless the flow is real.

    A checkpoint that loads but produces nothing (a mismatched state dict silently
    accepted, a build without the correlation kernel) is indistinguishable from a
    healthy one until the labels are already written, so the flow is required to be
    non-zero *here* rather than assumed downstream.

    The pattern is textured noise at :data:`RAFT_MIN_EDGE` — a flat shape on a flat
    ground leaves the interior of the shift ambiguous, and anything below the floor
    makes RAFT itself raise, which used to surface as "RAFT unavailable, needs
    network access" for weights that had in fact loaded.
    """
    gen = torch.Generator().manual_seed(0)  # never touch the caller's global RNG
    a = torch.rand(1, 3, RAFT_MIN_EDGE, RAFT_MIN_EDGE, generator=gen).to(
        device=device, dtype=next(raft.parameters()).dtype)
    b = torch.roll(a, shifts=8, dims=-1)
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
    why = "Optical flow drives motion_phase (hence the causal action strength s_A), " \
          "the affinity flow/IoU terms and two damage axes — without it they are " \
          "constant and the labels are not trainable."

    def _fail(message: str, exc: Exception) -> None:
        if required:
            raise RaftUnavailable(f"{message} {why}") from exc
        _log.error("%s %s", message, why)

    # Load and probe are reported apart: a probe failure means the weights were
    # found and read, so pointing at the network (or at --raft-weights) there sends
    # the reader after a problem they do not have.
    try:
        model = _build(variant, weights_path)
    except Exception as exc:
        source = (f"local checkpoint {weights_path}" if weights_path
                  else "the torchvision DEFAULT weights (a download; offline hosts "
                       "need --raft-weights)")
        _fail(f"RAFT ({variant}) could not be loaded from {source}: "
              f"{type(exc).__name__}: {exc}.", exc)
        return None
    try:
        raft = freeze(model.to(device))
        _probe(raft, device)
    except Exception as exc:
        _fail(f"RAFT ({variant}) loaded but failed its load-time probe on {device}: "
              f"{type(exc).__name__}: {exc}.", exc)
        return None
    _log.info("RAFT (%s) loaded and verified at load%s", variant,
              f" from {weights_path}" if weights_path else " from torchvision DEFAULT")
    return raft
