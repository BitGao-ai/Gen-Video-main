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
from dataclasses import dataclass, field
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
    """Top-level configuration for the three-stage training pipeline.

    Stage A reads the OpenVid CSV(s) and writes the six-level processed store; Stages
    B and C read that same store. One ``processed_root`` is therefore threaded through
    all three stages (defaulting to ``experiment_dir / 'LCOCF_OpenVid1M_Processed'``)
    so the data written by A is exactly what B/C consume.
    """

    # Shared
    experiment_dir: Path
    checkpoint_load_path: Optional[Path] = None  # Resume from checkpoint
    seed: int = 42

    # Stage-A data inputs (§1.1): OpenVid metadata CSV(s) and where the clips live.
    openvid_csvs: List[Path] = field(default_factory=list)
    data_root: str = ""
    # Shared processed-store root (§3); defaults under experiment_dir when unset.
    processed_root: Optional[Path] = None

    # Individual stage configs (optional pre-built overrides; built lazily otherwise)
    stage_a: Optional[StageAConfig] = None
    stage_b: Optional[StageBConfig] = None
    stage_c: Optional[StageCConfig] = None

    def to_dict(self):
        return {
            "experiment_dir": str(self.experiment_dir),
            "checkpoint_load_path": str(self.checkpoint_load_path) if self.checkpoint_load_path else None,
            "seed": self.seed,
            "openvid_csvs": [str(p) for p in self.openvid_csvs],
            "data_root": self.data_root,
            "processed_root": str(self.processed_root) if self.processed_root else None,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            experiment_dir=Path(data.get("experiment_dir", "./experiments")),
            checkpoint_load_path=Path(data.get("checkpoint_load_path")) if data.get("checkpoint_load_path") else None,
            seed=data.get("seed", 42),
            openvid_csvs=[Path(p) for p in data.get("openvid_csvs", [])],
            data_root=data.get("data_root", ""),
            processed_root=Path(data["processed_root"]) if data.get("processed_root") else None,
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

    def _processed_root(self) -> Path:
        """The shared six-level store root (§3): A writes it, B/C read it."""
        return (
            self.pipeline_cfg.processed_root
            or self.pipeline_cfg.experiment_dir / "LCOCF_OpenVid1M_Processed"
        )

    def _run_stage_a(self) -> None:
        """Run Stage A: counterfactual teacher data generation (§1)."""
        _log.info("\n--- Stage A: Data Generation ---")

        if self.pipeline_cfg.stage_a is None:
            if not self.pipeline_cfg.openvid_csvs:
                raise RuntimeError(
                    "Stage A needs OpenVid CSV(s): set PipelineConfig.openvid_csvs "
                    "(or pass a pre-built PipelineConfig.stage_a)."
                )
            self.pipeline_cfg.stage_a = StageAConfig(
                openvid_csvs=list(self.pipeline_cfg.openvid_csvs),
                processed_root=self._processed_root(),
                data_root=self.pipeline_cfg.data_root,
                config=self.config,
                device=self.device,
                seed=self.pipeline_cfg.seed,
            )

        # The metric extractor is owned by the accelerator (mock by default); Config
        # carries no such runtime object.
        self._stage_a = DataGenerationStage(
            config=self.pipeline_cfg.stage_a,
            backbone=self.accelerator.backbone,
            metric_extractor=self.accelerator.metric_extractor,  # injected
            accelerator=self.accelerator,
        )
        self._stage_a.run()

    def _run_stage_b(self) -> None:
        """Run Stage B: joint module training (§4.1)."""
        _log.info("\n--- Stage B: Joint Training ---")

        if self.pipeline_cfg.stage_b is None:
            self.pipeline_cfg.stage_b = StageBConfig(
                processed_root=self._processed_root(),
                config=self.config,
                device=self.device,
            )

        self._stage_b = JointTrainingStage(
            accelerator=self.accelerator,
            config=self.pipeline_cfg.stage_b,
        )
        self.accelerator = self._stage_b.run()

    def _run_stage_c(self) -> None:
        """Run Stage C: end-to-end lightweight fine-tuning (§4.2)."""
        _log.info("\n--- Stage C: Fine-tuning ---")

        if self.pipeline_cfg.stage_c is None:
            self.pipeline_cfg.stage_c = StageCConfig(
                processed_root=self._processed_root(),
                config=self.config,
                device=self.device,
            )

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
