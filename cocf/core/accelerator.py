"""The top-level accelerator — wires every subsystem into one object (§2, §7).

:class:`Accelerator` is the single hub that the inference engine, the Stage-A
teacher generator and the trainer all share. It owns:

    * the frozen **backbone** adapter (plain attribute → its weights never enter
      ``Accelerator.parameters()``, so optimisers only ever see the plugins);
    * the **STA** tube builder (with an injected perception provider);
    * the **L-COCF** module (the only sizeable learnable component);
    * the **RAEC** module (certificate + trigger + repair);
    * the **CMSC** text-tube alignment + conservation loss;
    * the **scheduler** (budget) and **allocator**;
    * the action-aware **transition executor** (the FLOPs/VRAM lever).

Why a single hub
----------------
The four innovations are deliberately decoupled (each in its own package), but
*something* has to instantiate them with mutually-consistent dimensions — the
predictor's ``token_dim``, the alignment's text/visual widths — all of which are
dictated by the chosen backbone. Centralising that wiring here keeps every other
entry point (engine / trainer / teacher) trivial and guarantees they operate on
the *same* module instances (so a checkpoint trained by the trainer is exactly
what the engine runs). Construction stays backbone-agnostic: dims are *probed*
from the adapter, never hard-coded.

The default :meth:`from_config` injects the mock perception / rule-based parser /
mock metric extractor, so the whole stack is constructible and runnable on CPU
with no model weights (tests & demos); a production run injects the real
SAM/CLIP/DINOv2/RAFT providers and a real backbone via the same call.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from cocf.backbones import build_backbone
from cocf.backbones.base import BackboneAdapter
from cocf.backbones.transition import TransitionExecutor
from cocf.cmsc.alignment import TextTubeAlignment
from cocf.cmsc.losses import CMSCLoss
from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.memory import count_parameters, freeze
from cocf.lcocf.damage import MetricExtractor
from cocf.lcocf.module import LCOCFModule
from cocf.lcocf.triplets import CausalParser, build_parser
from cocf.raec.module import RAECModule
from cocf.scheduler.allocator import ActionAllocator
from cocf.scheduler.budget import BudgetScheduler
from cocf.tubes.builder import TubeBuilder
from cocf.tubes.regions import PerceptionProvider
from cocf.tubes.smoothing import TubeSmoothingLoss

_log = get_logger(__name__)

# Fallback text/visual widths used only when a dim cannot be probed (e.g. a real
# backbone whose weights are not present in a dry run). Real runs probe the true
# dims; the mock probes instantly.
_DEFAULT_TEXT_DIM = 4096
_DEFAULT_VISUAL_DIM = 64


class Accelerator(nn.Module):
    """COCF-SS-DCA accelerator: a frozen backbone + the four learnable plugins.

    Parameters
    ----------
    config
        The root :class:`~cocf.common.config.Config`.
    backbone
        A constructed :class:`BackboneAdapter` (frozen). Built from
        ``config.backbone`` by :meth:`from_config` when not supplied.
    perception
        STA perception provider (SAM/CLIP/DINOv2/RAFT). Defaults to the mock.
    parser
        L-COCF causal-triplet parser. Defaults to the config-selected one.
    metric_extractor
        DINO/CLIP/RAFT/OCR backend for damage labels (§7.1.1) and the CMSC loss
        (§6). Defaults to the deterministic mock. Held here so every consumer
        shares one extractor instance.
    text_dim, visual_dim
        Widths of the text-token embedding and the tube visual embedding feeding
        CMSC. Probed from the backbone / perception when ``None``.
    """

    def __init__(
        self,
        config: Config,
        backbone: BackboneAdapter,
        *,
        perception: PerceptionProvider,
        parser: Optional[CausalParser] = None,
        metric_extractor: Optional[MetricExtractor] = None,
        text_dim: Optional[int] = None,
        visual_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        # Plain attribute (not an nn.Module) so the frozen backbone's weights are
        # never tracked by Accelerator.parameters()/state_dict — only plugins are.
        self.backbone = backbone
        self.perception = perception
        self.metric_extractor = metric_extractor

        token_dim = backbone.hidden_dim

        # -- STA: tube construction + the smoothing loss --------------------- #
        self.tube_builder = TubeBuilder(config.tube, perception)
        self.tube_smoothing = TubeSmoothingLoss(config.tube)

        # -- L-COCF: the only sizeable learnable component (§3.4) ------------ #
        self.lcocf = LCOCFModule(
            config.lcocf, config.tube, token_dim=token_dim,
            parser=parser or build_parser(config.lcocf),
        )

        # -- RAEC: certificate (learnable coeffs) + trigger + repair --------- #
        self.raec = RAECModule(config.certificate, config.trigger)

        # -- CMSC: learnable text↔tube alignment + the conservation loss ----- #
        text_dim = text_dim if text_dim is not None else self._probe_text_dim(backbone)
        visual_dim = visual_dim if visual_dim is not None else _probe_visual_dim(perception)
        self.cmsc_alignment = TextTubeAlignment(config.cmsc, text_dim=text_dim, visual_dim=visual_dim)
        self.cmsc_loss = CMSCLoss(config.cmsc, self.cmsc_alignment)
        self._text_dim = text_dim
        self._visual_dim = visual_dim

        # -- decision layer: budget + allocator + action executor ------------ #
        self.budget_scheduler = BudgetScheduler(config.budget)
        self.allocator = ActionAllocator(config.allocator)
        self.transition = TransitionExecutor(backbone, lowfreq_stride=config.engine.lowfreq_stride)

        self.freeze_backbone()
        trainable, total = count_parameters(self)
        _log.info(
            "Accelerator built on '%s' backbone: %.2fM trainable / %.2fM total plugin params "
            "(token_dim=%d, text_dim=%d, visual_dim=%d)",
            config.backbone.name, trainable / 1e6, total / 1e6, token_dim, text_dim, visual_dim,
        )

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        perception: Optional[PerceptionProvider] = None,
        parser: Optional[CausalParser] = None,
        metric_extractor: Optional[MetricExtractor] = None,
        text_dim: Optional[int] = None,
        visual_dim: Optional[int] = None,
    ) -> "Accelerator":
        """Build an accelerator from a config, defaulting all injectables to mocks.

        With no overrides this yields a fully-functional CPU stack (mock backbone
        if ``config.backbone.name == 'mock'``, mock perception, rule parser, mock
        metric extractor) — exactly what the tests and the smoke scripts use.
        """
        backbone = build_backbone(config.backbone)
        if perception is None:
            from cocf.tubes.mock_perception import MockPerception

            perception = MockPerception(seed=config.seed)
        if metric_extractor is None:
            from cocf.data.metrics import MockMetricExtractor

            metric_extractor = MockMetricExtractor(seed=config.seed)
        return cls(
            config, backbone,
            perception=perception, parser=parser, metric_extractor=metric_extractor,
            text_dim=text_dim, visual_dim=visual_dim,
        )

    # ------------------------------------------------------------------ #
    # backbone freezing & dim probing
    # ------------------------------------------------------------------ #

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters (the architectural memory saving, #1)."""
        module = self.backbone.module
        if module is not None:
            freeze(module)

    @staticmethod
    def _probe_text_dim(backbone: BackboneAdapter) -> int:
        """Discover the text-embedding width by a tiny encode (mock: instant).

        Falls back to a default when weights are unavailable, so construction
        never fails in a dry run; a real backbone (weights present) yields the
        true width.
        """
        try:
            with torch.inference_mode():
                return int(backbone.encode_text(["probe"]).embeds.shape[-1])
        except Exception:  # pragma: no cover - real backbone without weights
            extra = getattr(backbone.config, "extra", {}) or {}
            return int(extra.get("text_dim", _DEFAULT_TEXT_DIM))

    # ------------------------------------------------------------------ #
    # parser convenience
    # ------------------------------------------------------------------ #

    def parse(self, prompt: str):
        """Parse a prompt into its local causal sub-graph (§3.3.1)."""
        return self.lcocf.parse(prompt)

    # ------------------------------------------------------------------ #
    # trainable-parameter accounting (§3.4 — 千万级 plugins)
    # ------------------------------------------------------------------ #

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Named trainable parameter groups for the optimiser / logging.

        Mirrors the design's plugin decomposition: L-COCF (strength weights +
        damage predictor + residual-repair net), RAEC (certificate coefficients)
        and CMSC (the text↔tube alignment projection). The frozen backbone never
        appears here.
        """
        groups = dict(self.lcocf.parameter_groups())
        groups["certificate"] = list(self.raec.certificate.parameters())
        groups["cmsc_alignment"] = list(self.cmsc_alignment.parameters())
        return groups

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for g in self.parameter_groups().values() for p in g]

    @property
    def token_dim(self) -> int:
        return self.backbone.hidden_dim

    @property
    def text_dim(self) -> int:
        return self._text_dim

    @property
    def visual_dim(self) -> int:
        return self._visual_dim


def _probe_visual_dim(perception: PerceptionProvider) -> int:
    """Tube visual-embed width = the perception provider's CLIP feature dim.

    Read from a ``d_clip`` attribute when present (the mock and the standard
    providers expose it); otherwise the conventional default.
    """
    return int(getattr(perception, "d_clip", _DEFAULT_VISUAL_DIM))
