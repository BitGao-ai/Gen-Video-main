"""Backbone subsystem: the multi-model compatibility layer (user requirement #2).

Importing this package registers every adapter, after which a backbone is built
purely from a config string::

    from cocf.backbones import build_backbone
    adapter = build_backbone(config.backbone)   # name="hunyuanvideo" | "wan21" | "mock"

The heavy ``diffusers``/``transformers`` dependencies are imported lazily inside
each adapter's ``_load``; importing the package itself only needs ``torch``.
"""

from __future__ import annotations

from cocf.backbones.base import (
    BackboneAdapter,
    BackboneCache,
    DenoiseOutput,
    TextConditioning,
)
from cocf.backbones.transition import TransitionExecutor, TransitionResult
from cocf.common.config import BackboneConfig
from cocf.common.registry import BACKBONES

# Importing the adapter modules runs their @register_backbone decorators.
from cocf.backbones import mock as _mock  # noqa: F401  (registers "mock")
from cocf.backbones import hunyuan as _hunyuan  # noqa: F401  (registers "hunyuanvideo")
from cocf.backbones import wan21 as _wan21  # noqa: F401  (registers "wan21")


def build_backbone(config: BackboneConfig) -> BackboneAdapter:
    """Construct the adapter named by ``config.name`` from the registry."""
    return BACKBONES.build(config.name, config)


__all__ = [
    "BackboneAdapter",
    "BackboneCache",
    "DenoiseOutput",
    "TextConditioning",
    "TransitionExecutor",
    "TransitionResult",
    "build_backbone",
]
