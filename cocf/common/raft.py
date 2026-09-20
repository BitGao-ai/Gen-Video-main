"""Load and verify torchvision RAFT optical flow models."""

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

RAFT_MIN_EDGE = 128


class RaftUnavailable(RuntimeError):
    """Raised when RAFT is required but unavailable."""


def raft_pad(x: torch.Tensor) -> torch.Tensor:
    """Pad input to RAFT size requirements."""
    h, w = x.shape[-2:]
    th = max(RAFT_MIN_EDGE, -(-h // 8) * 8)
    tw = max(RAFT_MIN_EDGE, -(-w // 8) * 8)
    ph, pw = th - h, tw - w
    if not (ph or pw):
        return x
    mode = "reflect" if ph < h and pw < w else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode)


def _resolve_weights(variant: str, weights_path: Optional[str]) -> Optional[Path]:
    """Resolve RAFT checkpoint path for variant."""
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
    """Unwrap checkpoint envelope to state dict."""
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k[7:] if k.startswith("module.") else k: v for k, v in obj.items()}


def _build(variant: str, weights_path: Optional[str]) -> nn.Module:
    """Build RAFT model for variant."""
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
    """Probe RAFT with synthetic flow."""
    gen = torch.Generator().manual_seed(0)
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
    """Return probed RAFT model or None."""
    why = "Optical flow drives motion_phase (hence the causal action strength s_A), " \
          "the affinity flow/IoU terms and two damage axes — without it they are " \
          "constant and the labels are not trainable."

    def _fail(message: str, exc: Exception) -> None:
        if required:
            raise RaftUnavailable(f"{message} {why}") from exc
        _log.error("%s %s", message, why)

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
