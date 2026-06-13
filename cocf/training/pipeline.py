"""Unified training pipeline that orchestrates all three stages (§7.1).

The :class:`TrainingPipeline` provides a high-level API to run the full training
process from data generation through fine-tuning. Each stage is optional (can be
skipped if data/checkpoint exists), and intermediate results are cached.

Typical usage:
    pipeline = TrainingPipeline.from_config(config_path)
    pipeline.run(stages=["A", "B", "C"])
    accelerator = pipeline.accelerator  # Trained model ready for inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import yaml

from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine
from cocf.training.stage_a_data_gen import DataGenerationStage, StageAConfig
from cocf.training.stage_b_joint import JointTrainingStage, StageBConfig
from cocf.training.stage_c_finetune import FinettuneStage, StageCConfig

_log = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Top-level configuration for the three-stage training pipeline."""

    # Shared
    experiment_dir: Path
    checkpoint_load_path: Optional[Path] = None  # Resume from checkpoint
    seed: int = 42

    # Individual stage configs
    stage_a: Optional[StageAConfig] = None
    stage_b: Optional[StageBConfig] = None
    stage_c: Optional[StageCConfig] = None

    def to_dict(self):
        return {
            "experiment_dir": str(self.experiment_dir),
            "checkpoint_load_path": str(self.checkpoint_load_path) if self.checkpoint_load_path else None,
            "seed": self.seed,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        # Parse into individual stage configs
        return cls(
            experiment_dir=Path(data.get("experiment_dir", "./experiments")),
            checkpoint_load_path=Path(data.get("checkpoint_load_path")) if data.get("checkpoint_load_path") else None,
            seed=data.get("seed", 42),
        )


class TrainingPipeline:
    """Orchestrates the three-stage training workflow (§7.1).

    Responsibilities:
        1. Manage experiment directory & checkpoints
        2. Load/save accelerator state
        3. Run stages in sequence or independently
        4. Log progress & efficiency metrics
    """

    def __init__(
        self,
        config: Config,  # Full COCF config (backbones, modules, etc.)
        pipeline_cfg: PipelineConfig,
    ) -> None:
        self.config = config
        self.pipeline_cfg = pipeline_cfg
        # Device lives on the backbone sub-config (Config has no top-level `device`).
        self.device = torch.device(config.backbone.device)

        # Create experiment directory
        self.pipeline_cfg.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Build accelerator
        self.accelerator = Accelerator.from_config(config)

        # Load checkpoint if specified
        if pipeline_cfg.checkpoint_load_path:
            _log.info(f"Loading checkpoint from {pipeline_cfg.checkpoint_load_path}")
            state_dict = torch.load(pipeline_cfg.checkpoint_load_path, map_location=self.device)
            self.accelerator.load_state_dict(state_dict)

        # Build engine. The trigger config is a top-level node on Config
        # (`config.trigger`), not `config.raec.trigger`.
        self.engine = InferenceEngine(
            self.accelerator,
            config.engine,
            config.trigger,
        )

        # Stage instances (lazy-created)
        self._stage_a: Optional[DataGenerationStage] = None
        self._stage_b: Optional[JointTrainingStage] = None
        self._stage_c: Optional[FinettuneStage] = None

    def run(self, stages: List[str] = ["A", "B", "C"]) -> Accelerator:
        """Run the training pipeline for specified stages.

        Args:
            stages: List of stage names ("A", "B", "C") to run in order.

        Returns:
            The trained accelerator, ready for inference.
        """
        _log.info("=== COCF-SS-DCA Training Pipeline ===")
        _log.info(f"Running stages: {', '.join(stages)}")

        for stage in stages:
            if stage.upper() == "A":
                self._run_stage_a()
            elif stage.upper() == "B":
                self._run_stage_b()
            elif stage.upper() == "C":
                self._run_stage_c()
            else:
                _log.warning(f"Unknown stage: {stage}")

            # Save checkpoint after each stage
            self._save_checkpoint(stage)

        _log.info("=== Training Pipeline Complete ===")
        return self.accelerator

    def _run_stage_a(self) -> None:
        """Run Stage A: Data generation."""
        _log.info("\n--- Stage A: Data Generation ---")

        # Create stage config if not present
        if self.pipeline_cfg.stage_a is None:
            self.pipeline_cfg.stage_a = StageAConfig(
                manifest_path=self.pipeline_cfg.experiment_dir / "data" / "manifest.csv",
                output_dir=self.pipeline_cfg.experiment_dir / "stage_a_data",
            )

        # Run stage. The metric extractor is owned by the accelerator (defaulted to
        # a mock by Accelerator.from_config); Config carries no such runtime object.
        self._stage_a = DataGenerationStage(
            config=self.pipeline_cfg.stage_a,
            backbone=self.accelerator.backbone,
            metric_extractor=self.accelerator.metric_extractor,  # Injected
            accelerator=self.accelerator,
        )
        self._stage_a.run()

    def _run_stage_b(self) -> None:
        """Run Stage B: Joint training."""
        _log.info("\n--- Stage B: Joint Training ---")

        # Create stage config if not present
        if self.pipeline_cfg.stage_b is None:
            self.pipeline_cfg.stage_b = StageBConfig(
                cache_dir=self.pipeline_cfg.experiment_dir / "stage_a_data" / "cache",
                device=self.device,
            )

        # Run stage
        self._stage_b = JointTrainingStage(
            accelerator=self.accelerator,
            config=self.pipeline_cfg.stage_b,
        )
        self.accelerator = self._stage_b.run()

    def _run_stage_c(self) -> None:
        """Run Stage C: Fine-tuning."""
        _log.info("\n--- Stage C: Fine-tuning ---")

        # Create stage config if not present
        if self.pipeline_cfg.stage_c is None:
            self.pipeline_cfg.stage_c = StageCConfig(
                manifest_path=self.pipeline_cfg.experiment_dir / "data" / "manifest.csv",
                device=self.device,
            )

        # Run stage
        self._stage_c = FinettuneStage(
            accelerator=self.accelerator,
            engine=self.engine,
            config=self.pipeline_cfg.stage_c,
        )
        self.accelerator = self._stage_c.run()

    def _save_checkpoint(self, stage: str) -> None:
        """Save accelerator checkpoint after a stage."""
        ckpt_path = self.pipeline_cfg.experiment_dir / f"checkpoint_after_stage_{stage}.pt"
        torch.save(self.accelerator.state_dict(), ckpt_path)
        _log.info(f"Saved checkpoint to {ckpt_path}")

    @classmethod
    def from_config(cls, config_path: Path, pipeline_cfg_path: Optional[Path] = None) -> TrainingPipeline:
        """Create pipeline from config files."""
        config = Config.load(config_path)
        pipeline_cfg = (
            PipelineConfig.from_yaml(pipeline_cfg_path)
            if pipeline_cfg_path
            else PipelineConfig(experiment_dir=Path("./experiments"))
        )
        return cls(config, pipeline_cfg)
