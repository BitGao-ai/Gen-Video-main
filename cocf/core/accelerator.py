"""Top-level accelerator wiring the frozen backbone and all learnable plugins."""

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

# Fallback text/visual widths used only when a dim cannot be probed.
_DEFAULT_TEXT_DIM = 4096
_DEFAULT_VISUAL_DIM = 64


class Accelerator(nn.Module):
    """COCF-SS-DCA accelerator: a frozen backbone plus the learnable plugins."""

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
        # Plain attribute so the frozen backbone is excluded from parameters()/state_dict.
        self.backbone = backbone
        self.perception = perception
        self.metric_extractor = metric_extractor

        token_dim = backbone.hidden_dim

        self.tube_builder = TubeBuilder(config.tube, perception)
        self.tube_smoothing = TubeSmoothingLoss(config.tube)

        self.lcocf = LCOCFModule(
            config.lcocf, config.tube, token_dim=token_dim,
            parser=parser or build_parser(config.lcocf),
        )

        self.raec = RAECModule(config.certificate, config.trigger)

        text_dim = text_dim if text_dim is not None else self._probe_text_dim(backbone)
        visual_dim = visual_dim if visual_dim is not None else _probe_visual_dim(perception)
        self.cmsc_alignment = TextTubeAlignment(config.cmsc, text_dim=text_dim, visual_dim=visual_dim)
        self.cmsc_loss = CMSCLoss(config.cmsc, self.cmsc_alignment)
        self._text_dim = text_dim
        self._visual_dim = visual_dim

        self.budget_scheduler = BudgetScheduler(config.budget)
        self.allocator = ActionAllocator(
            config.allocator, lowfreq_stride=config.engine.lowfreq_stride,
            identity_unstable_threshold=config.tube.identity_unstable_threshold,
        )
        self.transition = TransitionExecutor(
            backbone,
            lowfreq_stride=config.engine.lowfreq_stride,
            dense_step_skip_below=config.engine.dense_step_skip_below,
            background_refresh_every=config.engine.background_refresh_every,
            max_unmeasured_steps=config.engine.max_unmeasured_steps,
        )
        self.transition.risk_trigger = self.raec.trigger

        self.freeze_backbone()
        trainable, total = count_parameters(self)
        _log.info(
            "Accelerator built on '%s' backbone: %.2fM trainable / %.2fM total plugin params "
            "(token_dim=%d, text_dim=%d, visual_dim=%d)",
            config.backbone.name, trainable / 1e6, total / 1e6, token_dim, text_dim, visual_dim,
        )

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
        """Build an accelerator from a config, defaulting all injectables to mocks."""
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

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters."""
        module = self.backbone.module
        if module is not None:
            freeze(module)

    @staticmethod
    def _probe_text_dim(backbone: BackboneAdapter) -> int:
        """Discover the text-embedding width by a tiny encode, with a default fallback."""
        try:
            with torch.no_grad():
                return int(backbone.encode_text(["probe"]).embeds.shape[-1])
        except Exception as exc:  # pragma: no cover - real backbone without weights
            extra = getattr(backbone.config, "extra", {}) or {}
            dim = int(extra.get("text_dim", _DEFAULT_TEXT_DIM))
            _log.warning(
                "_probe_text_dim: encode_text probe failed (%s: %s); falling back to "
                "text_dim=%d. If Stage-A data was generated with a different backbone "
                "(e.g. mock with text_dim=16), pass text_dim explicitly to "
                "Accelerator.from_config to avoid a shape mismatch in CMSC.",
                type(exc).__name__, exc, dim,
            )
            return dim

    def parse(self, prompt: str):
        """Parse a prompt into its local causal sub-graph."""
        return self.lcocf.parse(prompt)

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Named trainable parameter groups for the optimiser / logging."""
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
    """Return the perception provider's CLIP feature dim, or the default."""
    return int(getattr(perception, "d_clip", _DEFAULT_VISUAL_DIM))
