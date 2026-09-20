"""Backbone adapters and construction helper."""

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
from cocf.backbones import wan22 as _wan22  # noqa: F401  (registers "wan22" — primary)


def build_backbone(config: BackboneConfig) -> BackboneAdapter:
    """Build backbone adapter from config."""
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
